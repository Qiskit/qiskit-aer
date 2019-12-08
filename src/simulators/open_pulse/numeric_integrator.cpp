#include <csignal>
#include <vector>
#include <complex>
#include <iostream>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#ifdef DEBUG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <callgrind/callgrind.h>
#endif
#include "numeric_integrator.hpp"
#include "python_to_cpp.hpp"

#ifdef DEBUG
class Unregister {
  public:
    ~Unregister(){
        spdlog::drop_all();
    }
};
#endif

/**
 * Python // operator-like division
 */
int32_t floor_div(int32_t a, int32_t b) {
    int32_t q = a / b;
    int32_t r = a - q*b;
    q -= ((r != 0) & ((r ^ b) < 0));
    return q;
}

complex_t chan_value(
    double t,
    unsigned int chan_num,
    const double freq_ch,
    const NpArray<double>& chan_pulse_times,
    const NpArray<complex_t>& pulse_array,
    const NpArray<int>& pulse_indexes,
    const NpArray<double>& fc_array,
    const NpArray<uint8_t>& reg){

    static const auto get_arr_idx = [](double t, double start, double stop, size_t len_array) -> int {
        return static_cast<int>(std::floor((t - start) / (stop - start) * len_array));
    };

    complex_t out = {0., 0.};
    auto num_times = floor_div(static_cast<int>(chan_pulse_times.shape[0]), 4);

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
        num_times = floor_div(fc_array.shape[0], 3);
        for(auto i = 0; i < num_times; ++i){
            // TODO floating point comparison
            if(t >= fc_array[3 * i]){
                bool do_fc = true;
                if(fc_array[3 * i + 2] >= 0){
                    if(!reg[static_cast<int>(fc_array[3 * i + 2])]){
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
    complex_t * out,
    int num_rows){

    npy_intp dims = num_rows;
    PyArrayObject * array = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNewFromData(1, &dims, NPY_COMPLEX128, out));
    PyArray_ENABLEFLAGS(array, NPY_OWNDATA);
    #ifdef DEBUG
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
    #endif
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

    #ifdef DEBUG
    CALLGRIND_START_INSTRUMENTATION;
    #endif

    const static auto numpy_initialized = init_numpy();

    // I left this commented on porpose so we can use logging eventually
    // This is just a RAII for the logger
    //const Unregister unregister;
    //auto file_logger = spdlog::basic_logger_mt("basic_logger", "logs/td_ode_rhs.txt");
    //spdlog::set_default_logger(file_logger);
    //spdlog::set_level(spdlog::level::debug); // Set global log level to debug
    //spdlog::flush_on(spdlog::level::debug);

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

    auto vec = get_value<NpArray<complex_t>>(py_vec);
    auto num_rows = vec.shape[0];
    auto out = static_cast<complex_t *>(PyDataMem_NEW_ZEROED(num_rows, sizeof(complex_t)));

    auto pulses = get_ordered_map_from_dict_item<std::string, std::vector<NpArray<double>>>(py_exp, "channels");
    auto freqs = get_vec_from_dict_item<double>(py_global_data, "freqs");
    auto pulse_array = get_value_from_dict_item<NpArray<complex_t>>(py_global_data, "pulse_array");
    auto pulse_indices = get_value_from_dict_item<NpArray<int>>(py_global_data, "pulse_indices");
    auto reg = get_value<NpArray<uint8_t>>(py_register);

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
    auto systems = get_value<std::vector<TermExpression>>(py_system);
    auto vars = get_vec_from_dict_item<double>(py_global_data, "vars");
    auto vars_names = get_vec_from_dict_item<std::string>(py_global_data, "vars_names");
    auto num_h_terms = get_value_from_dict_item<long>(py_global_data, "num_h_terms");
    auto datas = get_vec_from_dict_item<NpArray<complex_t>>(py_global_data, "h_ops_data");
    auto idxs = get_vec_from_dict_item<NpArray<int>>(py_global_data, "h_ops_ind");
    auto ptrs = get_vec_from_dict_item<NpArray<int>>(py_global_data, "h_ops_ptr");
    auto energy = get_value_from_dict_item<NpArray<double>>(py_global_data, "h_diag_elems");
    for(const auto& idx_sys : enumerate(systems)){
        auto sys_index = idx_sys.first;
        auto sys = idx_sys.second;

        // TODO: Refactor
        std::string term;
        if(sys_index == systems.size() && num_h_terms > systems.size()){
            term = "1.0";
        }else if(sys_index < systems.size()){
            //term = sys.second;
            term = sys.term;
        }else{
            continue;
        }

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
                            complex_t(0.,1.) * (energy[i] - energy[tmp_idx]) * t
                        );
                    complex_t coef = (i < tmp_idx ? std::conj(td) : td);
                    dot += coef * osc_term * datas[sys_index][j] * vec[tmp_idx];

                }
                out[i] += dot;
            }
        }
    } /* End of systems */

    for(auto i=0; i < num_rows; ++i){
        out[i] += complex_t(0.,1.) * energy[i] * vec[i];
    }

    return create_py_array_from_vector(out, num_rows);
}
