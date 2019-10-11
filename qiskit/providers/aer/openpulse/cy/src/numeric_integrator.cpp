
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

static bool init_numpy(){
    import_array();
};

complex_t chan_value(
    double t,
    unsigned int chan_num,
    const double freq_ch,
    const std::vector<double>& chan_pulse_times,
    const std::vector<complex_t>& pulse_array,
    const std::vector<unsigned int>& pulse_indices,
    const std::vector<double>& fc_array,
    const std::string& reg){

    return complex_t(0.0, 0.0);
}

PyObject * td_ode_rhs(
    double t,
    PyArrayObject * py_vec,
    PyObject * py_global_data,
    PyObject * py_exp,
    PyObject * py_system,
    PyObject * py_register){

    const static auto numpy_initialized = init_numpy();

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

    // Generate an interrupt
    std::raise(SIGINT);

    spdlog::debug("Printing vec...");
    // 1. Get vec
    auto vec = get_vec_from_np_array<complex_t>(py_vec);
    jlog("vec: ", vec);


    // TODO: Not quite sure about vec.size()= shape?
    // unsigned int num_rows = vec.shape[0]
    unsigned int num_rows = vec.size();
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
    auto pulses = get_map_from_dict_item<std::string, std::vector<std::vector<double>>>(py_exp, "channels");
    spdlog::debug("Getting freqs...");
    auto freqs = get_map_from_dict_item<std::string, double>(py_global_data, "freqs");
    spdlog::debug("Getting pulse_array...");
    auto pulse_array = get_vec_from_dict_item<complex_t>(py_global_data, "pulse_array");
    spdlog::debug("Getting pulse_indices...");
    auto pulse_indices = get_vec_from_dict_item<unsigned int>(py_global_data, "pulse_indices");
    spdlog::debug("Getting reg...");
    std::string reg = get_value<std::string>(py_register);

    spdlog::debug("Printing pulses...");
    jlog("pulses: ", pulses);
    spdlog::debug("Printing freqs... ");
    jlog("freqs:", freqs);
    spdlog::debug("Printing pulse_array... ");
    jlog("pulse_array: ",  pulse_array);
    spdlog::debug("Printing reg...");
    spdlog::debug("reg: {}", reg);


    std::vector<complex_t> chan_values;
    chan_values.reserve(pulses.size());
    for(const auto& elem : enumerate(pulses)){
        /**
         * eleme is map of string as key type, and vector of vectors of doubles.
         * elem["D0"] = [[0.,1.,2.][0.,1.,2.]]
         **/
        auto index = elem.first;
        auto pulse = elem.second;

        auto val = chan_value(t, index, freqs[pulse.first], pulse.second[0], pulse_array,
                              pulse_indices, pulse.second[1], reg);
        chan_values.emplace_back(val);
    }

    // 4. Eval the time-dependent terms and do SPMV.

    auto systems = get_vec_from_py_list<std::string>(py_system);
    auto vars = get_vec_from_dict_item<complex_t>(py_global_data, "vars");
    auto vars_names = get_vec_from_dict_item<std::string>(py_global_data, "vars_names");
    auto num_h_terms = get_value_from_dict_item<long>(py_global_data, "num_h_terms");
    auto datas = get_vec_from_dict_item<std::vector<complex_t>>(py_global_data, "h_ops_data");
    auto idxs = get_vec_from_dict_item<std::vector<long>>(py_global_data, "h_ops_ind");
    auto ptrs = get_vec_from_dict_item<std::vector<long>>(py_global_data, "h_ops_ptr");
    auto energy = get_vec_from_dict_item<complex_t>(py_global_data, "h_diag_elems");
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
            term = sys;
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
        auto td = evaluate_hamiltonian_expression(term, vars, vars_names);
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
    return nullptr;
}
