/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_state_binding_hpp_
#define _aer_state_binding_hpp_

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/pybind11.h>
DISABLE_WARNING_POP
#if defined(_MSC_VER)
#undef snprintf
#endif

#include <complex>
#include <vector>

#include "framework/matrix.hpp"
#include "framework/pybind_casts.hpp"
#include "framework/python_parser.hpp"
#include "framework/results/pybind_result.hpp"
#include "framework/types.hpp"

#include "controllers/state_controller.hpp"

namespace py = pybind11;
using namespace AER;

template <typename MODULE>
void bind_aer_state(MODULE m) {
  py::class_<AerState> aer_state(m, "AerStateWrapper");

  aer_state.def(py::init<>(), "constructor");

  aer_state.def("__repr__", [](const AerState &state) {
    std::stringstream ss;
    ss << "AerStateWrapper("
       << "initialized=" << state.is_initialized()
       << ", num_of_qubits=" << state.num_of_qubits();
    ss << ")";
    return ss.str();
  });

  aer_state.def("configure", &AerState::configure);
  aer_state.def("allocate_qubits", &AerState::allocate_qubits);
  aer_state.def("reallocate_qubits", &AerState::reallocate_qubits);
  aer_state.def("set_random_seed", &AerState::set_random_seed);
  aer_state.def("set_seed", &AerState::set_seed);
  aer_state.def("clear", &AerState::clear);
  aer_state.def("num_of_qubits", &AerState::num_of_qubits);

  aer_state.def("initialize", &AerState::initialize);
  aer_state.def(
      "initialize_statevector",
      [aer_state](AerState &state, int num_of_qubits,
                  py::array_t<std::complex<double>> &values, bool copy) {
        auto c_contiguous =
            values.attr("flags").attr("c_contiguous").template cast<bool>();
        auto f_contiguous =
            values.attr("flags").attr("f_contiguous").template cast<bool>();
        if (!c_contiguous && !f_contiguous)
          return false;
        std::complex<double> *data_ptr =
            reinterpret_cast<std::complex<double> *>(values.mutable_data(0));
        state.configure("method", "statevector");
        state.initialize_statevector(num_of_qubits, data_ptr, copy);
        return true;
      });

  aer_state.def(
      "initialize_density_matrix",
      [aer_state](AER::AerState &state, int num_of_qubits,
                  py::array_t<std::complex<double>> &values, bool copy) {
        auto c_contiguous =
            values.attr("flags").attr("c_contiguous").template cast<bool>();
        auto f_contiguous =
            values.attr("flags").attr("f_contiguous").template cast<bool>();
        if (!c_contiguous && !f_contiguous)
          return false;
        std::complex<double> *data_ptr =
            reinterpret_cast<std::complex<double> *>(values.mutable_data(0));
        state.configure("method", "density_matrix");
        state.initialize_density_matrix(num_of_qubits, data_ptr, f_contiguous,
                                        copy);
        return true;
      });

  aer_state.def("move_to_buffer", [aer_state](AER::AerState &state) {
    return state.move_to_vector().move_to_buffer();
  });

  aer_state.def("move_to_ndarray", [aer_state](AerState &state) {
    auto vec = state.move_to_vector();

    auto ret = AerToPy::to_numpy(std::move(vec));
    return ret;
  });

  aer_state.def("move_to_matrix", [aer_state](AER::AerState &state) {
    auto mat = state.move_to_matrix();
    auto ret = AerToPy::to_numpy(std::move(mat));
    return ret;
  });

  aer_state.def("flush", &AerState::flush_ops);

  aer_state.def("last_result", [aer_state](AerState &state) {
    return AerToPy::to_python(state.last_result().to_json());
  });

  aer_state.def("apply_initialize", &AerState::apply_initialize);
  aer_state.def("set_statevector", &AER::AerState::set_statevector);
  aer_state.def("set_density_matrix", &AER::AerState::set_density_matrix);

  aer_state.def("apply_global_phase", &AerState::apply_global_phase);
  aer_state.def("apply_unitary",
                [aer_state](AerState &state, const reg_t &qubits,
                            const py::array_t<std::complex<double>> &values) {
                  size_t mat_len = (1UL << qubits.size());
                  auto ptr = values.unchecked<2>();
                  cmatrix_t mat(mat_len, mat_len);
                  for (uint_t i = 0; i < mat_len; ++i)
                    for (uint_t j = 0; j < mat_len; ++j)
                      mat(i, j) = ptr(i, j);
                  state.apply_unitary(qubits, mat);
                });

  aer_state.def("apply_multiplexer",
                [aer_state](AerState &state, const reg_t &control_qubits,
                            const reg_t &target_qubits,
                            const py::array_t<std::complex<double>> &values) {
                  size_t mat_len = (1UL << target_qubits.size());
                  size_t mat_size = (1UL << control_qubits.size());
                  auto ptr = values.unchecked<3>();
                  std::vector<cmatrix_t> mats;
                  for (uint_t i = 0; i < mat_size; ++i) {
                    cmatrix_t mat(mat_len, mat_len);
                    for (uint_t j = 0; j < mat_len; ++j)
                      for (uint_t k = 0; k < mat_len; ++k)
                        mat(j, k) = ptr(i, j, k);
                    mats.push_back(mat);
                  }
                  state.apply_multiplexer(control_qubits, target_qubits, mats);
                });

  aer_state.def("apply_diagonal", &AerState::apply_diagonal_matrix);
  aer_state.def("apply_x", &AerState::apply_x);
  aer_state.def("apply_cx", &AerState::apply_cx);
  aer_state.def("apply_mcx", &AerState::apply_mcx);
  aer_state.def("apply_y", &AerState::apply_y);
  aer_state.def("apply_cy", &AerState::apply_cy);
  aer_state.def("apply_mcy", &AerState::apply_mcy);
  aer_state.def("apply_z", &AerState::apply_z);
  aer_state.def("apply_cz", &AerState::apply_cz);
  aer_state.def("apply_mcz", &AerState::apply_mcz);
  aer_state.def("apply_mcphase", &AerState::apply_mcphase);
  aer_state.def("apply_h", &AerState::apply_h);
  aer_state.def("apply_u", &AerState::apply_u);
  aer_state.def("apply_cu", &AerState::apply_cu);
  aer_state.def("apply_mcu", &AerState::apply_mcu);
  aer_state.def("apply_mcswap", &AerState::apply_mcswap);
  aer_state.def("apply_measure", &AerState::apply_measure);
  aer_state.def("apply_reset", &AerState::apply_reset);
  aer_state.def("apply_kraus", &AER::AerState::apply_kraus);
  aer_state.def("probability", &AerState::probability);
  aer_state.def(
      "probabilities",
      [aer_state](AerState &state, const reg_t qubits) {
        if (qubits.empty())
          return state.probabilities();
        else
          return state.probabilities(qubits);
      },
      py::arg("qubits") = reg_t());
  aer_state.def("sample_memory", &AerState::sample_memory);
  aer_state.def("sample_counts", &AerState::sample_counts);
}

#endif