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
#include "framework/python_parser.hpp"
#include "framework/pybind_json.hpp"
#include "framework/pybind_casts.hpp"

#include "framework/types.hpp"
#include "framework/results/pybind_result.hpp"

#include "framework/circuit.hpp"

namespace py = pybind11;
using namespace AER;

template<typename MODULE>
void bind_aer_optype_(MODULE m) {
  py::enum_<Operations::OpType> optype(m, "OpType_") ;
  optype.value("gate", Operations::OpType::gate);
  optype.value("measure", Operations::OpType::measure);
  optype.value("reset", Operations::OpType::reset);
  optype.value("bfunc", Operations::OpType::bfunc);
  optype.value("barrier", Operations::OpType::barrier);
  optype.value("matrix", Operations::OpType::matrix);
  optype.value("diagonal_matrix", Operations::OpType::diagonal_matrix);
  optype.value("multiplexer", Operations::OpType::multiplexer);
  optype.value("initialize", Operations::OpType::initialize);
  optype.value("sim_op", Operations::OpType::sim_op);
  optype.value("nop", Operations::OpType::nop);
  optype.value("kraus", Operations::OpType::kraus);
  optype.value("superop", Operations::OpType::superop);
  optype.value("roerror", Operations::OpType::roerror);
  optype.value("noise_switch", Operations::OpType::noise_switch);
  optype.value("save_state", Operations::OpType::save_state);
  optype.value("save_expval", Operations::OpType::save_expval);
  optype.value("save_expval_var", Operations::OpType::save_expval_var);
  optype.value("save_statevec", Operations::OpType::save_statevec);
  optype.value("save_statevec_dict", Operations::OpType::save_statevec_dict);
  optype.value("save_densmat", Operations::OpType::save_densmat);
  optype.value("save_probs", Operations::OpType::save_probs);
  optype.value("save_probs_ket", Operations::OpType::save_probs_ket);
  optype.value("save_amps", Operations::OpType::save_amps);
  optype.value("save_amps_sq", Operations::OpType::save_amps_sq);
  optype.value("save_stabilizer", Operations::OpType::save_stabilizer);
  optype.value("save_mps", Operations::OpType::save_mps);
  optype.value("save_superop", Operations::OpType::save_superop);
  optype.value("save_unitary", Operations::OpType::save_unitary);
  optype.value("set_statevec", Operations::OpType::set_statevec);
  optype.value("set_unitary", Operations::OpType::set_unitary);
  optype.value("set_densmat", Operations::OpType::set_densmat);
  optype.value("set_superop", Operations::OpType::set_superop);
  optype.value("set_stabilizer", Operations::OpType::set_stabilizer);
  optype.value("set_mps", Operations::OpType::set_mps);
  optype.export_values();
}

template<typename MODULE>
void bind_aer_reg_comparison_(MODULE m) {
  py::enum_<Operations::RegComparison> reg_comparison(m, "RegComparison_");
  reg_comparison.value("equal", Operations::RegComparison::Equal);
  reg_comparison.value("NotEqual", Operations::RegComparison::NotEqual);
  reg_comparison.value("Less", Operations::RegComparison::Less);
  reg_comparison.value("LessEqual", Operations::RegComparison::LessEqual);
  reg_comparison.value("Greater", Operations::RegComparison::Greater);
  reg_comparison.value("GreaterEqual", Operations::RegComparison::GreaterEqual);
  reg_comparison.export_values();
}

template<typename MODULE>
void bind_aer_data_sub_type_(MODULE m) {
  py::enum_<Operations::DataSubType> data_sub_type(m, "DataSubType_");
  data_sub_type.value("single", Operations::DataSubType::single);
  data_sub_type.value("c_single", Operations::DataSubType::c_single);
  data_sub_type.value("list", Operations::DataSubType::list);
  data_sub_type.value("c_list", Operations::DataSubType::c_list);
  data_sub_type.value("accum", Operations::DataSubType::accum);
  data_sub_type.value("c_accum", Operations::DataSubType::c_accum);
  data_sub_type.value("average", Operations::DataSubType::average);
  data_sub_type.value("c_average", Operations::DataSubType::c_average);
  data_sub_type.export_values();
}

template<typename MODULE>
void bind_aer_op_(MODULE m) {
  py::class_<Operations::Op> aer_op(m, "AerOp_");
  aer_op.def(py::init(), "constructor");
  aer_op.def("__repr__", [](const Operations::Op &op) { std::stringstream ss; ss << op; return ss.str(); });
  aer_op.def_readwrite("type", &Operations::Op::type);
  aer_op.def_readwrite("name", &Operations::Op::name);
  aer_op.def_readwrite("qubits", &Operations::Op::qubits);
  aer_op.def_readwrite("regs", &Operations::Op::regs);
  aer_op.def_readwrite("params", &Operations::Op::params);
  aer_op.def_readwrite("int_params", &Operations::Op::int_params);
  aer_op.def_readwrite("string_params", &Operations::Op::string_params);
  aer_op.def_readwrite("conditional", &Operations::Op::conditional);
  aer_op.def_readwrite("conditional_reg", &Operations::Op::conditional_reg);
  aer_op.def_readwrite("bfunc", &Operations::Op::bfunc);
  aer_op.def_readwrite("memory", &Operations::Op::memory);
  aer_op.def_readwrite("registers", &Operations::Op::registers);
  aer_op.def_readwrite("mats", &Operations::Op::mats);
  aer_op.def_readwrite("probs", &Operations::Op::probs);
  aer_op.def_readwrite("expval_params", &Operations::Op::expval_params);
  aer_op.def_readwrite("save_type", &Operations::Op::save_type);
  aer_op.def_readwrite("mps", &Operations::Op::mps);
}

template<typename MODULE>
void bind_aer_circuit_(MODULE m) {
  py::class_<Circuit> aer_circuit(m, "AerCircuit_");
  aer_circuit.def(py::init());
  aer_circuit.def("__repr__", [](const Circuit &circ) {
    std::stringstream ss;
    ss << "Circuit("
        << "qubit=" << circ.num_qubits
        << ", num_memory=" << circ.num_memory
        << ", num_registers=" << circ.num_registers;

    ss << ", ops={";
    for (auto i = 0; i < circ.ops.size(); ++i)
      if (i == 0)
        ss << circ.ops[i];
      else
        ss << "," << circ.ops[i];

    ss << "}"
        << ", shots=" << circ.shots
        << ", seed=" << circ.seed
        << ", global_phase_angle=" << circ.global_phase_angle
        ;
    ss << ")";
    return ss.str();
  });
  aer_circuit.def_readwrite("shots", &Circuit::shots);
  aer_circuit.def_readwrite("num_qubits", &Circuit::num_qubits);
  aer_circuit.def_readwrite("num_memory", &Circuit::num_memory);
  aer_circuit.def_readwrite("seed", &Circuit::seed);
  aer_circuit.def_readwrite("ops", &Circuit::ops);
  aer_circuit.def_readwrite("global_phase_angle", &Circuit::global_phase_angle);
  aer_circuit.def_readwrite("header", &Circuit::header);
  aer_circuit.def("set_header", [aer_circuit](Circuit &circ,
                                              const py::handle& header) {
      circ.header = header;
  });

  aer_circuit.def("append", [aer_circuit](Circuit &circ, const Operations::Op &op) {
      circ.ops.push_back(op);
  });
}

template<typename MODULE>
void bind_aer_circuit(MODULE m) {

  bind_aer_optype_(m);
  bind_aer_reg_comparison_(m);
  bind_aer_data_sub_type_(m);
  bind_aer_op_(m);
  bind_aer_circuit_(m);

}

#endif