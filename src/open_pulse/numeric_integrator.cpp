#include <csignal>
#include <vector>
#include <complex>
#include <iostream>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <Python.h>
DISABLE_WARNING_POP

#ifdef DEBUG
#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <callgrind/callgrind.h>
DISABLE_WARNING_POP
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
        num_times = floor_div(fc_array.shape[0], 3);

        // get the index of the phase change
        // this loop will result in finding the index of the phase to use +1
        auto phase_idx = 0;
        while(phase_idx < num_times){
            if(t < fc_array[3 * phase_idx]) break;
            phase_idx++;
        }

        double phase = 0.;
        if(phase_idx > 0){
            phase = fc_array[3 * (phase_idx - 1) + 1];
        }

        if(phase != 0.){
            out *= std::exp(complex_t(0., 1.) * phase);
        }
        out *= std::exp(complex_t(0., 1.) * 2. * M_PI * freq_ch * t);
    }
    return out.real();
}

struct RhsData {
  RhsData(py::object the_global_data,
          py::object the_exp,
          py::object the_system,
          py::object the_channels,
          py::object the_reg) {

      PyObject *py_global_data = the_global_data.ptr();
      PyObject *py_exp = the_exp.ptr();
      PyObject *py_system = the_system.ptr();
      PyObject *py_register = the_reg.ptr();

      if (py_global_data == nullptr ||
          py_exp == nullptr ||
          py_system == nullptr ||
          py_register == nullptr) {
          std::string msg = "These arguments cannot be null: ";
          msg += (py_global_data == nullptr ? "py_global_data " : "");
          msg += (py_exp == nullptr ? "py_exp " : "");
          msg += (py_system == nullptr ? "py_system " : "");
          msg += (py_register == nullptr ? "py_register " : "");
          throw std::invalid_argument(msg);
      }

      pulses = get_ordered_map_from_dict_item<std::string, std::vector<NpArray<double>>>(py_exp, "channels");
      freqs = get_vec_from_dict_item<double>(py_global_data, "freqs");
      pulse_array = get_value_from_dict_item<NpArray<complex_t>>(py_global_data, "pulse_array");
      pulse_indices = get_value_from_dict_item<NpArray<int>>(py_global_data, "pulse_indices");
      reg = get_value<NpArray<uint8_t>>(py_register);

      systems = get_value<std::vector<TermExpression>>(py_system);
      vars = get_vec_from_dict_item<double>(py_global_data, "vars");
      vars_names = get_vec_from_dict_item<std::string>(py_global_data, "vars_names");
      num_h_terms = get_value_from_dict_item<long>(py_global_data, "num_h_terms");
      auto tmp_datas = get_vec_from_dict_item<NpArray<complex_t>>(py_global_data, "h_ops_data");
      for (const auto& data: tmp_datas){
          auto datas_back = datas.emplace(datas.end());
          auto idxs_back = idxs.emplace(idxs.end());
          auto ptrs_back = ptrs.emplace(ptrs.end());
          ptrs_back->push_back(0);
          auto first_j = 0;
          auto last_j = 0;
          for (auto i = 0; i < data.shape[0]; i++) {
              for (auto j = 0; j < data.shape[1]; j++) {
                  if (std::abs(data(i, j)) > 1e-15) {
                      datas_back->push_back(data(i,j));
                      idxs_back->push_back(j);
                      last_j++;
                  }
              }
              ptrs_back->push_back(last_j);
          }
      }
      energy = get_value_from_dict_item<NpArray<double>>(py_global_data, "h_diag_elems");
  }

  ordered_map<std::string, std::vector<NpArray<double>>> pulses;
  std::vector<double> freqs;
  NpArray<complex_t> pulse_array;
  NpArray<int> pulse_indices;
  NpArray<uint8_t> reg;

  std::vector<TermExpression> systems;
  std::vector<double> vars;
  std::vector<std::string> vars_names;
  long num_h_terms;
  std::vector<std::vector<complex_t>> datas;
  std::vector<std::vector<int>> idxs;
  std::vector<std::vector<int>> ptrs;
  NpArray<double> energy;

  std::vector<complex_t> osc_terms_no_t;
};

py::array_t <complex_t> inner_ode_rhs(double t,
                                      py::array_t <complex_t> the_vec,
                                      const RhsData &rhs_data) {
    if (the_vec.ptr() == nullptr) {
        throw std::invalid_argument("py_vec cannot be null");
    }

    auto vec = static_cast<complex_t *>(the_vec.request().ptr);
    auto num_rows = the_vec.size();
    py::array_t <complex_t> out_arr(num_rows);
    auto out = static_cast<complex_t *>(out_arr.request().ptr);
    memset(&out[0], 0, num_rows * sizeof(complex_t));

    std::unordered_map<std::string, complex_t> chan_values;
    chan_values.reserve(rhs_data.pulses.size());
    for (const auto &elem : enumerate(rhs_data.pulses)) {
        /**
         * eleme is map of string as key type, and vector of vectors of doubles.
         * elem["D0"] = [[0.,1.,2.][0.,1.,2.]]
         **/
        auto i = elem.first;
        auto channel = elem.second.first;
        auto pulse = elem.second.second;

        auto val = chan_value(t, i, rhs_data.freqs[i], pulse[0], rhs_data.pulse_array,
                              rhs_data.pulse_indices, pulse[1], rhs_data.reg);
        chan_values.emplace(channel, val);
    }

    // 4. Eval the time-dependent terms and do SPMV.
    for (int h_idx = 0; h_idx < rhs_data.num_h_terms; h_idx++) {
        // TODO: Refactor
        std::string term;
        if (h_idx == rhs_data.systems.size() && rhs_data.num_h_terms > rhs_data.systems.size()) {
            term = "1.0";
        } else if (h_idx < rhs_data.systems.size()) {
            term = rhs_data.systems[h_idx].term;
        } else {
            continue;
        }

        auto td = evaluate_hamiltonian_expression(term, rhs_data.vars, rhs_data.vars_names, chan_values);
        if (std::abs(td) > 1e-15) {
            for (auto i = 0; i < num_rows; i++) {
                complex_t dot = {0., 0.};
                auto row_start = rhs_data.ptrs[h_idx][i];
                auto row_end = rhs_data.ptrs[h_idx][i + 1];
                for (auto j = row_start; j < row_end; ++j) {
                    auto tmp_idx = rhs_data.idxs[h_idx][j];
                    auto osc_term = std::exp(
                            complex_t(0., 1.) * (rhs_data.energy[i] - rhs_data.energy[tmp_idx]) * t
                        );
                    complex_t coef = (i < tmp_idx ? std::conj(td) : td);
                    dot += coef * osc_term * rhs_data.datas[h_idx][j] * vec[tmp_idx];
                }
                out[i] += dot;
            }
        }
    } /* End of systems */
    for (auto i = 0; i < num_rows; ++i) {
        out[i] += complex_t(0., 1.) * rhs_data.energy[i] * vec[i];
    }

    return out_arr;
}

RhsFunctor::RhsFunctor(py::object the_global_data, py::object the_exp, py::object the_system,
                       py::object the_channels, py::object the_reg)
    : rhs_data_(
    std::make_shared<RhsData>(the_global_data, the_exp, the_system, the_channels, the_reg)) {}

py::array_t <complex_t> RhsFunctor::operator()(double t, py::array_t <complex_t> the_vec) {
    return inner_ode_rhs(t, the_vec, *rhs_data_);
}

py::array_t <complex_t> td_ode_rhs(double t,
                                   py::array_t <complex_t> the_vec,
                                   py::object the_global_data,
                                   py::object the_exp,
                                   py::object the_system,
                                   py::object the_channels,
                                   py::object the_reg) {
#ifdef DEBUG
    CALLGRIND_START_INSTRUMENTATION;
#endif

    // I left this commented on porpose so we can use logging eventually
    // This is just a RAII for the logger
    // const Unregister unregister;
    // auto file_logger = spdlog::basic_logger_mt("basic_logger", "logs/td_ode_rhs.txt");
    // spdlog::set_default_logger(file_logger);
    // spdlog::set_level(spdlog::level::debug); // Set global log level to debug
    // spdlog::flush_on(spdlog::level::debug);

    auto rhs_data = RhsData(the_global_data, the_exp, the_system, the_channels, the_reg);
    return inner_ode_rhs(t, the_vec, rhs_data);
}
