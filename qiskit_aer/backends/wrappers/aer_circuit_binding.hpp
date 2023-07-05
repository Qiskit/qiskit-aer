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

#ifndef _aer_circuit_binding_hpp_
#define _aer_circuit_binding_hpp_

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/pybind11.h>
DISABLE_WARNING_POP
#if defined(_MSC_VER)
#undef snprintf
#endif

#include <vector>

#include "framework/matrix.hpp"
#include "framework/pybind_casts.hpp"
#include "framework/pybind_json.hpp"
#include "framework/python_parser.hpp"

#include "framework/results/pybind_result.hpp"
#include "framework/types.hpp"

#include "framework/circuit.hpp"

namespace py = pybind11;
using namespace AER;

template <typename MODULE>
void bind_aer_circuit(MODULE m) {
  py::class_<Circuit, std::shared_ptr<Circuit>> aer_circuit(m, "AerCircuit");
  aer_circuit.def(py::init());
  aer_circuit.def("__repr__", [](const Circuit &circ) {
    std::stringstream ss;
    ss << "Circuit("
       << "qubit=" << circ.num_qubits << ", num_memory=" << circ.num_memory
       << ", num_registers=" << circ.num_registers;

    ss << ", ops={";
    for (auto i = 0; i < circ.ops.size(); ++i)
      if (i == 0)
        ss << circ.ops[i];
      else
        ss << "," << circ.ops[i];

    ss << "}"
       << ", shots=" << circ.shots << ", seed=" << circ.seed
       << ", global_phase_angle=" << circ.global_phase_angle;
    ss << ")";
    return ss.str();
  });

  aer_circuit.def_readwrite("shots", &Circuit::shots);
  aer_circuit.def_readwrite("num_qubits", &Circuit::num_qubits);
  aer_circuit.def_readwrite("num_memory", &Circuit::num_memory);
  aer_circuit.def_readwrite("seed", &Circuit::seed);
  aer_circuit.def_readwrite("ops", &Circuit::ops);
  aer_circuit.def_readwrite("global_phase_angle", &Circuit::global_phase_angle);
  aer_circuit.def("set_header",
                  [aer_circuit](Circuit &circ, const py::handle &header) {
                    circ.header = header;
                  });
  aer_circuit.def("bfunc", &Circuit::bfunc);
  aer_circuit.def("gate", &Circuit::gate);
  aer_circuit.def("diagonal", &Circuit::diagonal);
  aer_circuit.def("unitary", &Circuit::unitary);
  aer_circuit.def("roerror", &Circuit::roerror);
  aer_circuit.def("multiplexer", &Circuit::multiplexer);
  aer_circuit.def("kraus", &Circuit::kraus);
  aer_circuit.def("superop", &Circuit::superop);
  aer_circuit.def("save_state", &Circuit::save_state);
  aer_circuit.def("save_amplitudes", &Circuit::save_amplitudes);
  aer_circuit.def("save_expval", &Circuit::save_expval);
  aer_circuit.def("initialize", &Circuit::initialize);
  aer_circuit.def("set_statevector", &Circuit::set_statevector<py::handle>);
  aer_circuit.def("set_density_matrix",
                  &Circuit::set_density_matrix<py::handle>);
  aer_circuit.def("set_unitary", &Circuit::set_unitary<py::handle>);
  aer_circuit.def("set_superop", &Circuit::set_superop<py::handle>);
  aer_circuit.def("set_matrix_product_state",
                  &Circuit::set_matrix_product_state<py::handle>);
  aer_circuit.def("set_clifford", &Circuit::set_clifford<py::handle>);
  aer_circuit.def("jump", &Circuit::jump);
  aer_circuit.def("mark", &Circuit::mark);
  aer_circuit.def("barrier", &Circuit::barrier);
  aer_circuit.def("measure", &Circuit::measure);
  aer_circuit.def("reset", &Circuit::reset);
  aer_circuit.def("set_qerror_loc", &Circuit::set_qerror_loc);
}

#endif