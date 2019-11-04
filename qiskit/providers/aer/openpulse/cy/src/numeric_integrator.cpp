#include <csignal>
#include <vector>
#include <complex>
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "numeric_integrator.hpp"
#include "helpers.hpp"

class Unregister {
  public:
    ~Unregister(){
        spdlog::drop_all();
    }
};

complex_t chan_value(
    double t,
    unsigned int chan_num,
    const double freq_ch,
    const NpArray<double>& chan_pulse_times,
    const NpArray<complex_t>& pulse_array,
    const NpArray<long>& pulse_indexes,
    const NpArray<double>& fc_array,
    const NpArray<uint8_t>& reg){

    static const auto get_arr_idx = [](double t, double start, double stop, size_t len_array) -> int {
        return static_cast<int>(std::floor((t - start) / (stop - start) * len_array));
    };

    complex_t out = {0., 0.};

    //1. cdef unsigned int num_times = chan_pulse_times.shape[0] // 4
    auto num_times = static_cast<int>(chan_pulse_times.shape[0]) / 4;
    spdlog::debug("num_times: {}", num_times);

    for(auto i=0; i < num_times; ++i){
        auto start_time = chan_pulse_times[4 * i];
        auto stop_time = chan_pulse_times[4 * i + 1];
        if(start_time <= t && t < stop_time){
            auto cond = static_cast<int>(chan_pulse_times[4 * i + 3]);
            if(cond < 0 || reg[cond]) {
                auto temp_idx = static_cast<int>(chan_pulse_times[4 * i + 2]);
                auto start_idx = pulse_indexes[temp_idx];
                auto stop_idx = pulse_indexes[temp_idx+1];
                auto offset_idx = get_arr_idx(t, start_time, stop_time, stop_idx - start_idx);
                out = pulse_array[start_idx + offset_idx];
            }
        }
    }

    // TODO floating point comparsion with complex<double> ?!
    // Seems like this is equivalent to: out != complex_t(0., 0.)
    if(out != 0.){
        double phase = 0.;
        num_times = fc_array.shape[0];
        for(auto i = 0; i < num_times; ++i){
            // TODO floating point comparison
            if(t >= fc_array[3 * i]){
                bool do_fc = true;
                if(fc_array[3 * i + 2] >= 0){
                    if(!reg[static_cast<int>(fc_array[3 * i +2])]){
                       do_fc = false;
                    }
                }
                if(do_fc){
                    phase += fc_array[3 * i + 1];
                }
            }else{
                break;
            }
        }
        if(phase != 0.){
            out *= std::exp(complex_t(0.,1.) * phase);
        }
        out *= std::exp(complex_t(0., -1.) * 2. * M_PI * freq_ch * t);
    }
    return out;
}


PyArrayObject * create_py_array_from_vector(
    std::vector<complex_t>& out,
    int num_rows){

    // complex_t * new_array = static_cast<complex_t *>(
    //     PyDataMem_NEW_ZEROED(num_rows,sizeof(complex_t))
    // );
    // std::copy(out.begin(), out.end(), new_array);
    npy_intp dims = num_rows;
    PyArrayObject * array = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNewFromData(1, &dims, NPY_COMPLEX128, &out[0]));
    PyArray_ENABLEFLAGS(array, NPY_OWNDATA);
    return array;
}

PyArrayObject * td_ode_rhs(
    double t,
    PyArrayObject * py_vec,
    PyObject * py_global_data,
    PyObject * py_exp,
    PyObject * py_system,
    PyObject * py_channels,
    PyObject * py_register){

    const static auto numpy_initialized = init_numpy();

    const Unregister unregister;

    auto file_logger = spdlog::basic_logger_mt("basic_logger", "logs/td_ode_rhs.txt");
    spdlog::set_default_logger(file_logger);
    spdlog::set_level(spdlog::level::debug); // Set global log level to debug
    spdlog::flush_on(spdlog::level::debug);

    spdlog::debug("td_ode_rhs!");

    if(py_vec == nullptr ||
       py_global_data == nullptr ||
       py_exp == nullptr ||
       py_system == nullptr ||
       py_register == nullptr){
           std::string msg = "These arguments cannot be null: ";
           msg += (py_vec == nullptr ? "py_vec " : "" );
           msg += (py_global_data == nullptr ? "py_global_data " : "" );
           msg += (py_exp == nullptr ? "py_exp " : "" );
           msg += (py_system == nullptr ? "py_system " : "" );
           msg += (py_register == nullptr ? "py_register " : "" );
           throw std::invalid_argument(msg);
    }

    spdlog::debug("Printing vec...");
    // 1. Get vec
    auto vec = get_value<NpArray<complex_t>>(py_vec);
    jlog("vec", vec);



    // deal with non 1D arrays as well (through PyArrayObject)
    // unsigned int num_rows = vec.shape[0]
    auto num_rows = vec.shape[0];
    spdlog::debug("num_rows: {}", num_rows);


    // 2. double complex * out = <complex *>PyDataMem_NEW_ZEROED(num_rows,sizeof(complex))
    // auto out = std::make_unique<complex_t>(
    //     PyDataMem_NEW_ZEROED(num_rows, sizeof(complex_t))
    // );
    std::vector<complex_t> out;
    out.reserve(num_rows);

    // 3. Compute complex channel values at time `t`
    // D0 = chan_value(t, 0, (double)D0_freq, ([doubles])D0_pulses,  pulse_array, pulse_indices, D0_fc, )
    // U0 = chan_value(t, 1, U0_freq, U0_pulses,  pulse_array, pulse_indices, U0_fc, )
    // D1 = chan_value(t, 2, D1_freq, D1_pulses,  pulse_array, pulse_indices, D1_fc, )
    // U1 = chan_value(t, 3, U1_freq, U1_pulses,  pulse_array, pulse_indices, U1_fc, )
    ////
    // for chan, idx in self.op_system.channels.items():
    // chan_str = "%s = chan_value(t, %s, %s_freq, " % (chan, idx, chan) + \
    //            "%s_pulses,  pulse_array, pulse_indices, " % chan + \
    //            "%s_fc, )" % (chan)

    spdlog::debug("Getting pulses...");
    // TODO: Pass const & as keys to avoid copying
    //auto pulses = get_map_from_dict_item<std::string, std::vector<NpArray<double>>>(py_exp, "channels");
    auto pulses = get_map_from_dict_item<std::string, std::vector<NpArray<double>>>(py_exp, "channels");
    spdlog::debug("Getting freqs...");
    auto freqs = get_vec_from_dict_item<double>(py_global_data, "freqs");
    spdlog::debug("Getting pulse_array...");
    auto pulse_array = get_value_from_dict_item<NpArray<complex_t>>(py_global_data, "pulse_array");
    spdlog::debug("Getting pulse_indices...");
    auto pulse_indices = get_value_from_dict_item<NpArray<long>>(py_global_data, "pulse_indices");
    spdlog::debug("Getting reg...");
    auto reg = get_value<NpArray<uint8_t>>(py_register);

    spdlog::debug("Printing pulses...");
    jlog("pulses: ", pulses);
    spdlog::debug("Printing freqs... ");
    jlog("freqs:", freqs);
    spdlog::debug("Printing pulse_array... ");
    jlog("pulse_array: ",  pulse_array);
    spdlog::debug("Printing reg...");
    jlog("reg: {}", reg);


    // auto channels = get_value<std::map<long, std::string>>(py_channels);
    std::unordered_map<std::string, complex_t> chan_values;
    chan_values.reserve(pulses.size());
    for(const auto& elem : enumerate(pulses)){
        /**
         * eleme is map of string as key type, and vector of vectors of doubles.
         * elem["D0"] = [[0.,1.,2.][0.,1.,2.]]
         **/
        auto i = elem.first;
        auto channel = elem.second.first;
        auto pulse = elem.second.second;

        auto val = chan_value(t, i, freqs[i], pulse[0], pulse_array,
                              pulse_indices, pulse[1], reg);
        chan_values.emplace(channel, val);
    }

    // 4. Eval the time-dependent terms and do SPMV.
    auto systems = get_value<std::vector<std::pair<QuantumObj, std::string>>>(py_system);
    auto vars = get_vec_from_dict_item<double>(py_global_data, "vars");
    auto vars_names = get_vec_from_dict_item<std::string>(py_global_data, "vars_names");
    auto num_h_terms = get_value_from_dict_item<long>(py_global_data, "num_h_terms");
    auto datas = get_vec_from_dict_item<NpArray<complex_t>>(py_global_data, "h_ops_data");
    auto idxs = get_vec_from_dict_item<NpArray<long>>(py_global_data, "h_ops_ind");
    auto ptrs = get_vec_from_dict_item<NpArray<long>>(py_global_data, "h_ops_ptr");
    auto energy = get_value_from_dict_item<NpArray<double>>(py_global_data, "h_diag_elems");
    for(const auto& idx_sys : enumerate(systems)){
        auto sys_index = idx_sys.first;
        auto sys = idx_sys.second;
        // 4.1
        // if (idx == len(self.op_system.system) and
        //    (len(self.op_system.system) < self.num_ham_terms)):
        //     # this is the noise term
        //         term = [1.0, 1.0]
        //     elif idx < len(self.op_system.system):
        //         term = self.op_system.system[idx]
        //     else:
        //         continue
        // TODO: Refactor
        std::string term;
        if(sys_index == systems.size() && num_h_terms > systems.size()){
            term = "1.0";
        }else if(sys_index < systems.size()){
            term = sys.second;
        }else{
            continue;
        }

        // 4.2
        // td0 = np.pi*(2*v0-alpha0)
        // if abs(td0) > 1e-15:
        //     for row in range(num_rows):
        //         dot = 0;
        //         row_start = ptr0[row];
        //         row_end = ptr0[row+1];
        //         for jj in range(row_start,row_end):
        //             osc_term = exp(1j * (energ[row] - energ[idx0[jj]]) * t)
        //             if row<idx0[jj]:
        //                 coef = conj(td0)
        //             else:
        //                 coef = td0
        //             dot += coef*osc_term*data0[jj]*vec[idx0[jj]];
        //         out[row] += dot;
        // td1 = np.pi*alpha0
        auto td = evaluate_hamiltonian_expression(term, vars, vars_names, chan_values);
        if(std::abs(td) > 1e-15){
            for(auto i=0; i<num_rows; i++){
                complex_t dot = {0., 0.};
                auto row_start = ptrs[sys_index][i];
                auto row_end = ptrs[sys_index][i+1];
                for(auto j = row_start; j<row_end; ++j){
                    auto tmp_idx = idxs[sys_index][j];
                    auto osc_term =
                        std::exp(
                            complex_t(0.,1.) * (energy[i] - energy[tmp_idx] * t)
                        );
                    complex_t coef = (i < tmp_idx ? std::conj(td) : td);
                    dot += coef * osc_term * datas[sys_index][j] * vec[tmp_idx];
                }
                out[i] += dot;
            }
        }
    } /* End of systems */

    // TODO: Pass the out vector to Pyhton memory, and return it
    return create_py_array_from_vector(out, num_rows);
}
