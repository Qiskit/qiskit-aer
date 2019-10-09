#include <vector>
#include <complex>
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "numeric_integrator.hpp"
#include "helpers.hpp"

complex_t chan_value(
    double t,
    unsigned int chan_num,
    double freq_ch,
    const std::vector<double>& chan_pulse_times,
    const std::vector<complex_t>& pulse_array,
    const std::vector<unsigned int>& pulse_indices,
    const std::vector<double>& fc_array,
    const std::string& reg){

    return complex_t(0.0, 0.0);
}

PyObject * td_ode_rhs(
    PyObject * py_t,
    PyObject * py_vec,
    PyObject * py_global_data,
    PyObject * py_exp,
    PyObject * py_system,
    PyObject * py_register){

    if(py_t == nullptr ||
       py_vec == nullptr ||
       py_global_data == nullptr ||
       py_exp == nullptr ||
       py_system == nullptr){
           std::string msg = "These arguments cannot be null: ";
           msg += (py_t == nullptr ? "py_t " : "" );
           msg += (py_vec == nullptr ? "py_vec " : "" );
           msg += (py_global_data == nullptr ? "py_global_data " : "" );
           msg += (py_exp == nullptr ? "py_exp " : "" );
           msg += (py_system == nullptr ? "py_system " : "" );
           throw std::invalid_argument(msg);
    }

    // 1. Get t and vec
    auto t = get_value<double>(py_t);
    auto vec = get_vec_from_py_list<complex_t>(py_vec);


    // TODO: Not quite sure about vec.size()= shape?
    // unsigned int num_rows = vec.shape[0]
    unsigned int num_rows = vec.size();

    // 2. double complex * out = <complex *>PyDataMem_NEW_ZEROED(num_rows,sizeof(complex))
    // auto out = std::make_unique<complex_t>(
    //     PyDataMem_NEW_ZEROED(num_rows, sizeof(complex_t))
    // );
    std::vector<complex_t> out;
    out.reserve(num_rows);

    // 3. Compute complex channel values at time `t`
    // D0 = chan_value(t, 0, (double)D0_freq, ([doubles])D0_pulses,  pulse_array, pulse_indices, D0_fc, register)
    // U0 = chan_value(t, 1, U0_freq, U0_pulses,  pulse_array, pulse_indices, U0_fc, register)
    // D1 = chan_value(t, 2, D1_freq, D1_pulses,  pulse_array, pulse_indices, D1_fc, register)
    // U1 = chan_value(t, 3, U1_freq, U1_pulses,  pulse_array, pulse_indices, U1_fc, register)
    ////
    // for chan, idx in self.op_system.channels.items():
    // chan_str = "%s = chan_value(t, %s, %s_freq, " % (chan, idx, chan) + \
    //            "%s_pulses,  pulse_array, pulse_indices, " % chan + \
    //            "%s_fc, register)" % (chan)

    const auto pulses = get_map_from_dict_item<std::string, std::vector<std::vector<double>>>(py_exp, "channels");
    auto t = get_value_from_dict_item<double>(py_global_data, "h_diag_elems");
    auto freqs = get_map_from_dict_item<std::string, double>(py_global_data, "freqs");
    const auto pulse_array = get_vec_from_dict_item<complex_t>(py_global_data, "pulse_array");
    const auto pulse_indices = get_vec_from_dict_item<unsigned int>(py_global_data, "pulse_indices");
    std::string reg = get_value<std::string>(py_register);

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

    auto systems = get_vec_from_py_list<std::string>(py_system);
    auto vars = get_vec_from_dict_item<complex_t>(py_global_data, "vars");
    auto vars_names = get_vec_from_dict_item<std::string>(py_global_data, "vars_names");
    auto num_h_terms = get_value_from_dict_item<long>(py_global_data, "num_h_terms");
    auto datas = get_vec_from_dict_item<std::vector<long>>(py_global_data, "h_ops_data");
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

        for(const auto& idx_varname: enumerate(vars_names)){
            auto var_name_index = idx_varname.first;
            auto var_name = idx_varname.second;
            auto td = evaluate_hamiltonian_expression<double>(term, vars, vars_names);
            if(std::abs(td) > 1e-15){
                for(auto i=0; i<num_rows; i++){
                    complex_t dot = {0., 0.};
                    auto row_start = ptrs[sys_index][i];
                    auto row_end = ptrs[sys_index][i+1];
                    for(auto j = row_start; j<row_start; ++j){
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
        }
    } /* End of systems */

    // TODO: Pass the out vector to Pyhton memory, and return it
    return nullptr;
}
