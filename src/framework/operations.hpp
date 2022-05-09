/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_operations_hpp_
#define _aer_framework_operations_hpp_

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <tuple>

#include "framework/types.hpp"
#include "framework/json_parser.hpp"
#include "framework/utils.hpp"
#include "framework/linalg/almost_equal.hpp"
#include "simulators/stabilizer/clifford.hpp"

namespace AER {
namespace Operations {

// Comparisons enum class used for Boolean function operation.
// these are used to compare two hexadecimal strings and return a bool
// for now we only have one comparison Equal, but others will be added
enum class RegComparison {Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual};

// Enum class for operation types
enum class OpType {
  gate, measure, reset, bfunc, barrier, qerror_loc, snapshot,
  matrix, diagonal_matrix, multiplexer, initialize, sim_op, nop,
  // Noise instructions
  kraus, superop, roerror, noise_switch,
  // Save instructions
  save_state, save_expval, save_expval_var, save_statevec, save_statevec_dict,
  save_densmat, save_probs, save_probs_ket, save_amps, save_amps_sq,
  save_stabilizer, save_clifford, save_unitary, save_mps, save_superop,
  // Set instructions
  set_statevec, set_densmat, set_unitary, set_superop,
  set_stabilizer, set_mps,
  // Control Flow
  jump, mark
};

enum class DataSubType {
  single, c_single, list, c_list, accum, c_accum, average, c_average
};

static const std::unordered_set<OpType> SAVE_TYPES = {
  OpType::save_state, OpType::save_expval, OpType::save_expval_var,
  OpType::save_statevec, OpType::save_statevec_dict,
  OpType::save_densmat, OpType::save_probs, OpType::save_probs_ket,
  OpType::save_amps, OpType::save_amps_sq, OpType::save_stabilizer,
  OpType::save_clifford,
  OpType::save_unitary, OpType::save_mps, OpType::save_superop
};

inline std::ostream& operator<<(std::ostream& stream, const OpType& type) {
  switch (type) {
  case OpType::gate:
    stream << "gate";
    break;
  case OpType::measure:
    stream << "measure";
    break;
  case OpType::reset:
    stream << "reset";
    break;
  case OpType::bfunc:
    stream << "bfunc";
    break;
  case OpType::barrier:
    stream << "barrier";
    break;
  case OpType::save_state:
    stream << "save_state";
    break;
  case OpType::save_expval:
    stream << "save_expval";
    break;
  case OpType::save_expval_var:
    stream << "save_expval_var";
  case OpType::save_statevec:
    stream << "save_statevector";
    break;
  case OpType::save_statevec_dict:
    stream << "save_statevector_dict";
    break;
  case OpType::save_mps:
    stream << "save_matrix_product_state";
    break;
  case OpType::save_densmat:
    stream << "save_density_matrix";
    break;
  case OpType::save_probs:
    stream << "save_probabilities";
    break;
  case OpType::save_probs_ket:
    stream << "save_probabilities_dict";
    break;
  case OpType::save_amps:
    stream << "save_amplitudes";
    break;
  case OpType::save_amps_sq:
    stream << "save_amplitudes_sq";
    break;
  case OpType::save_stabilizer:
    stream << "save_stabilizer";
    break;
  case OpType::save_clifford:
    stream << "save_clifford";
    break;
  case OpType::save_unitary:
    stream << "save_unitary";
    break;
  case OpType::save_superop:
    stream << "save_superop";
    break;
  case OpType::set_statevec:
    stream << "set_statevector";
    break;
  case OpType::set_densmat:
    stream << "set_density_matrix";
    break;
  case OpType::set_unitary:
    stream << "set_unitary";
    break;
  case OpType::set_superop:
    stream << "set_superop";
    break;
  case OpType::set_stabilizer:
    stream << "set_stabilizer";
    break;
  case OpType::set_mps:
    stream << "set_matrix_product_state";
    break;
  case OpType::snapshot:
    stream << "snapshot";
    break;
  case OpType::matrix:
    stream << "unitary";
    break;
  case OpType::diagonal_matrix:
    stream << "diagonal";
    break;
  case OpType::multiplexer:
    stream << "multiplexer";
    break;
  case OpType::kraus:
    stream << "kraus";
    break;
  case OpType::superop:
    stream << "superop";
    break;
  case OpType::roerror:
    stream << "roerror";
    break;
  case OpType::qerror_loc:
    stream << "qerror_loc";
    break;
  case OpType::noise_switch:
    stream << "noise_switch";
    break;
  case OpType::initialize:
    stream << "initialize";
    break;
  case OpType::sim_op:
    stream << "sim_op";
    break;
  case OpType::nop:
    stream << "nop";
    break;
  case OpType::mark:
    stream << "mark";
    break;
  case OpType::jump:
    stream << "jump";
    break;
  default:
    stream << "unknown";
  }
  return stream;
}


inline std::ostream& operator<<(std::ostream& stream, const DataSubType& subtype) {
  switch (subtype) {
    case DataSubType::single:
      stream << "single";
      break;
    case DataSubType::c_single:
      stream << "c_single";
      break;
    case DataSubType::list:
      stream << "list";
      break;
    case DataSubType::c_list:
      stream << "c_list";
      break;
    case DataSubType::accum:
      stream << "accum";
      break;
    case DataSubType::c_accum:
      stream << "c_accum";
      break;
    case DataSubType::average:
      stream << "average";
      break;
    case DataSubType::c_average:
      stream << "c_average";
      break;
    default:
      stream << "unknown";
  }
  return stream;
}


//------------------------------------------------------------------------------
// Op Class
//------------------------------------------------------------------------------

struct Op {
  // General Operations
  OpType type;                    // operation type identifier
  std::string name;               // operation name
  reg_t qubits;                   //  qubits operation acts on
  std::vector<reg_t> regs;        //  list of qubits for matrixes
  std::vector<complex_t> params;  // real or complex params for gates
  std::vector<uint_t> int_params;  // integer parameters 
  std::vector<std::string> string_params; // used for snapshot label, and boolean functions

  // Conditional Operations
  bool conditional = false; // is gate conditional gate
  uint_t conditional_reg;   // (opt) the (single) register location to look up for conditional
  RegComparison bfunc;      // (opt) boolean function relation

  // Measurement
  reg_t memory;             // (opt) register operation it acts on (measure)
  reg_t registers;          // (opt) register locations it acts on (measure, conditional)

  // Mat and Kraus
  std::vector<cmatrix_t> mats;

  // Readout error
  std::vector<rvector_t> probs;

  // Expvals
  std::vector<std::tuple<std::string, double, double>> expval_params;

  // Set states
  Clifford::Clifford clifford;
  mps_container_t mps;

  // Legacy Snapshots
  DataSubType save_type = DataSubType::single;
  using pauli_component_t = std::pair<complex_t, std::string>; // Pair (coeff, label_string)
  using matrix_component_t = std::pair<complex_t, std::vector<std::pair<reg_t, cmatrix_t>>>; // vector of Pair(qubits, matrix), combined with coefficient
  std::vector<pauli_component_t> params_expval_pauli;
  std::vector<matrix_component_t> params_expval_matrix; // note that diagonal matrices are stored as
                                                        // 1 x M row-matrices
                                                        // Projector vectors are stored as
                                                        // M x 1 column-matrices
};

inline std::ostream& operator<<(std::ostream& s, const Op& op) {
  s << op.name << "[";
  bool first = true;
  for (size_t qubit: op.qubits) {
    if (!first) s << ",";
    s << qubit;
    first = false;
  }
  s << "],[";
  first = true;
  for (reg_t reg: op.regs) {
    if (!first) s << ",";
    s << "[";
    bool first0 = true;
    for (size_t qubit: reg) {
      if (!first0) s << ",";
      s << qubit;
      first0 = false;
    }
    s << "]";
    first = false;
  }
  s << "]";
  return s;
}

//------------------------------------------------------------------------------
// Error Checking
//------------------------------------------------------------------------------

// Raise an exception if name string is empty
inline void check_empty_name(const Op &op) {
  if (op.name.empty())
    throw std::invalid_argument(R"(Invalid qobj instruction ("name" is empty).)");
}

// Raise an exception if qubits list is empty
inline void check_empty_qubits(const Op &op) {
  if (op.qubits.empty())
    throw std::invalid_argument(R"(Invalid qobj ")" + op.name +
                                R"(" instruction ("qubits" is empty).)");
}

// Raise an exception if params is empty
inline void check_empty_params(const Op &op) {
  if (op.params.empty())
    throw std::invalid_argument(R"(Invalid qobj ")" + op.name +
                                R"(" instruction ("params" is empty).)");
}

// Raise an exception if params is empty
inline void check_length_params(const Op &op, const size_t size) {
  if (op.params.size() != size)
    throw std::invalid_argument(R"(Invalid qobj ")" + op.name +
                                R"(" instruction ("params" is incorrect length).)");
}

// Raise an exception if qubits list contains duplications
inline void check_duplicate_qubits(const Op &op) {
  auto cpy = op.qubits;
  std::unique(cpy.begin(), cpy.end());
  if (cpy != op.qubits)
    throw std::invalid_argument(R"(Invalid qobj ")" + op.name +
                                R"(" instruction ("qubits" are not unique).)");
}

//------------------------------------------------------------------------------
// Generator functions
//------------------------------------------------------------------------------

inline Op make_unitary(const reg_t &qubits, const cmatrix_t &mat, std::string label = "") {
  Op op;
  op.type = OpType::matrix;
  op.name = "unitary";
  op.qubits = qubits;
  op.mats = {mat};
  if (label != "")
    op.string_params = {label};
  return op;
}

inline Op make_unitary(const reg_t &qubits, cmatrix_t &&mat, std::string label = "") {
  Op op;
  op.type = OpType::matrix;
  op.name = "unitary";
  op.qubits = qubits;
  op.mats.resize(1);
  op.mats[0] = std::move(mat);
  if (label != "")
    op.string_params = {label};
  return op;
}

inline Op make_diagonal(const reg_t &qubits, cvector_t &&vec, std::string label = "") {
  Op op;
  op.type = OpType::diagonal_matrix;
  op.name = "diagonal";
  op.qubits = qubits;
  op.params = std::move(vec);

  if (label != "")
    op.string_params = {label};

  return op;
}

inline Op make_superop(const reg_t &qubits, const cmatrix_t &mat) {
  Op op;
  op.type = OpType::superop;
  op.name = "superop";
  op.qubits = qubits;
  op.mats = {mat};
  return op;
}

inline Op make_superop(const reg_t &qubits, cmatrix_t &&mat) {
  Op op;
  op.type = OpType::superop;
  op.name = "superop";
  op.qubits = qubits;
  op.mats.resize(1);
  op.mats[0] = std::move(mat);
  return op;
}

inline Op make_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &mats) {
  Op op;
  op.type = OpType::kraus;
  op.name = "kraus";
  op.qubits = qubits;
  op.mats = mats;
  return op;
}

inline Op make_kraus(const reg_t &qubits, std::vector<cmatrix_t> &&mats) {
  Op op;
  op.type = OpType::kraus;
  op.name = "kraus";
  op.qubits = qubits;
  op.mats = std::move(mats);
  return op;
}

inline Op make_roerror(const reg_t &memory, const std::vector<rvector_t> &probs) {
  Op op;
  op.type = OpType::roerror;
  op.name = "roerror";
  op.memory = memory;
  op.probs = probs;
  return op;
}

inline Op make_roerror(const reg_t &memory, std::vector<rvector_t> &&probs) {
  Op op;
  op.type = OpType::roerror;
  op.name = "roerror";
  op.memory = memory;
  op.probs = std::move(probs);
  return op;
}

template <typename T> // real or complex numeric type
inline Op make_u1(uint_t qubit, T lam) {
  Op op;
  op.type = OpType::gate;
  op.name = "u1";
  op.qubits = {qubit};
  op.params = {lam};
  op.string_params = {op.name};
  return op;
}

template <typename T> // real or complex numeric type
inline Op make_u2(uint_t qubit, T phi, T lam) {
  Op op;
  op.type = OpType::gate;
  op.name = "u2";
  op.qubits = {qubit};
  op.params = {phi, lam};
  op.string_params = {op.name};
  return op;
}

template <typename T> // real or complex numeric type
inline Op make_u3(uint_t qubit, T theta, T phi, T lam) {
  Op op;
  op.type = OpType::gate;
  op.name = "u3";
  op.qubits = {qubit};
  op.params = {theta, phi, lam};
  op.string_params = {op.name};
  return op;
}

inline Op make_reset(const reg_t & qubits, uint_t state = 0) {
  Op op;
  op.type = OpType::reset;
  op.name = "reset";
  op.qubits = qubits;
  return op;
}

inline Op make_multiplexer(const reg_t &qubits,
                           const std::vector<cmatrix_t> &mats,
                           std::string label = "") {

  // Check matrices are N-qubit
  auto dim = mats[0].GetRows();
  auto num_targets = static_cast<uint_t>(std::log2(dim));
  if (1ULL << num_targets != dim) {
    throw std::invalid_argument("invalid multiplexer matrix dimension.");
  }
  // Check number of matrix compents is power of 2.
  size_t num_mats = mats.size();
  auto num_controls = static_cast<uint_t>(std::log2(num_mats));
  if (1ULL << num_controls != num_mats) {
    throw std::invalid_argument("invalid number of multiplexer matrices.");
  }
  if (num_controls == 0) { // mats.size() must be 1
    return make_unitary(qubits, mats[0]);
  }
  // Check number of targets and controls matches qubits
  if (num_controls + num_targets != qubits.size()) {
    throw std::invalid_argument("multiplexer qubits don't match parameters.");
  }
  // Check each matrix component is unitary and same size
  for (const auto &mat : mats) {
    if (!Utils::is_unitary(mat, 1e-7))
      throw std::invalid_argument("multiplexer matrix is not unitary.");
    if (mat.GetRows() != dim) {
      throw std::invalid_argument("multiplexer matrices are different size.");
    }
  }
  // Get lists of controls and targets
  reg_t controls(num_controls), targets(num_targets);
  std::copy_n(qubits.begin(), num_targets, targets.begin());
  std::copy_n(qubits.begin() + num_targets, num_controls, controls.begin());

  // Construct the Op
  Op op;
  op.type = OpType::multiplexer;
  op.name = "multiplexer";
  op.qubits = qubits;
  op.mats = mats;
  op.regs = std::vector<reg_t>({controls, targets});
  if (label != "")
    op.string_params = {label};

  // Validate qubits are unique.
  check_empty_qubits(op);
  check_duplicate_qubits(op);

  return op;
}

//------------------------------------------------------------------------------
// JSON conversion
//------------------------------------------------------------------------------

// Main deserialization functions
template<typename inputdata_t>
Op input_to_op(const inputdata_t& input); // Partial TODO
json_t op_to_json(const Op &op); // Partial TODO

inline void from_json(const json_t &js, Op &op) {op = input_to_op(js);}

inline void to_json(json_t &js, const Op &op) { js = op_to_json(op);}

void to_json(json_t &js, const DataSubType& type);

// Standard operations
template<typename inputdata_t>
Op input_to_op_gate(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_barrier(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_measure(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_reset(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_bfunc(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_initialize(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_pauli(const inputdata_t& input);

// Set state
template<typename inputdata_t>
Op input_to_op_set_vector(const inputdata_t& input, OpType op_type);

template<typename inputdata_t>
Op input_to_op_set_matrix(const inputdata_t& input, OpType op_type);

template<typename inputdata_t>
Op input_to_op_set_clifford(const inputdata_t& input, OpType op_type);

template<typename inputdata_t>
Op input_to_op_set_mps(const inputdata_t& input, OpType op_type);

// Save data
template<typename inputdata_t>
Op input_to_op_save_default(const inputdata_t& input, OpType op_type);
template<typename inputdata_t>
Op input_to_op_save_expval(const inputdata_t& input, bool variance);
template<typename inputdata_t>
Op input_to_op_save_amps(const inputdata_t& input, bool squared);

// Snapshots
template<typename inputdata_t>
Op input_to_op_snapshot(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_snapshot_default(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_snapshot_matrix(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_snapshot_pauli(const inputdata_t& input);

// Control-Flow
template<typename inputdata_t>
Op input_to_op_jump(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_mark(const inputdata_t& input);

// Matrices
template<typename inputdata_t>
Op input_to_op_unitary(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_diagonal(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_superop(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_multiplexer(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_kraus(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_noise_switch(const inputdata_t& input);
template<typename inputdata_t>
Op input_to_op_qerror_loc(const inputdata_t& input);

// Classical bits
template<typename inputdata_t>
Op input_to_op_roerror(const inputdata_t& input);

// Optional instruction parameters
enum class Allowed {Yes, No};

template<typename inputdata_t>
void add_conditional(const Allowed val, Op& op, const inputdata_t& input);


//------------------------------------------------------------------------------
// Implementation: JSON deserialization
//------------------------------------------------------------------------------

// TODO: convert if-else to switch
template<typename inputdata_t>
Op input_to_op(const inputdata_t& input) {
  // load operation identifier
  std::string name;
  Parser<inputdata_t>::get_value(name, "name", input);
  // Barrier
  if (name == "barrier")
    return input_to_op_barrier(input);
  // Measure & Reset
  if (name == "measure")
    return input_to_op_measure(input);
  if (name == "reset")
    return input_to_op_reset(input);
  if (name == "initialize")
    return input_to_op_initialize(input);
  // Arbitrary matrix gates
  if (name == "unitary")
    return input_to_op_unitary(input);
  if (name == "diagonal" || name == "diag")
    return input_to_op_diagonal(input);
  if (name == "superop")
    return input_to_op_superop(input);
  // Save
  if (name == "save_state")
    return input_to_op_save_default(input, OpType::save_state);
  if (name == "save_expval")
    return input_to_op_save_expval(input, false);
  if (name == "save_expval_var")
    return input_to_op_save_expval(input, true);
  if (name == "save_statevector")
    return input_to_op_save_default(input, OpType::save_statevec);
  if (name == "save_statevector_dict")
    return input_to_op_save_default(input, OpType::save_statevec_dict);
  if (name == "save_stabilizer")
    return input_to_op_save_default(input, OpType::save_stabilizer);
  if (name == "save_clifford")
    return input_to_op_save_default(input, OpType::save_clifford);
  if (name == "save_unitary")
    return input_to_op_save_default(input, OpType::save_unitary);
  if (name == "save_superop")
    return input_to_op_save_default(input, OpType::save_superop);
  if (name == "save_density_matrix")
    return input_to_op_save_default(input, OpType::save_densmat);
  if (name == "save_probabilities")
    return input_to_op_save_default(input, OpType::save_probs);
  if (name == "save_matrix_product_state")
    return input_to_op_save_default(input, OpType::save_mps);
  if (name == "save_probabilities_dict")
    return input_to_op_save_default(input, OpType::save_probs_ket);
  if (name == "save_amplitudes")
    return input_to_op_save_amps(input, false);
  if (name == "save_amplitudes_sq")
    return input_to_op_save_amps(input, true);
  // Set
  if (name == "set_statevector")
    return input_to_op_set_vector(input, OpType::set_statevec);
  if (name == "set_density_matrix")
    return input_to_op_set_matrix(input, OpType::set_densmat);
  if (name == "set_unitary")
    return input_to_op_set_matrix(input, OpType::set_unitary);
  if (name == "set_superop")
    return input_to_op_set_matrix(input, OpType::set_superop);
  if (name == "set_stabilizer")
    return input_to_op_set_clifford(input, OpType::set_stabilizer);
  if (name == "set_matrix_product_state")
    return input_to_op_set_mps(input, OpType::set_mps);

  // Snapshot
  if (name == "snapshot")
    return input_to_op_snapshot(input);
  // Bit functions
  if (name == "bfunc")
    return input_to_op_bfunc(input);
  // Noise functions
  if (name == "noise_switch")
    return input_to_op_noise_switch(input);
  if (name == "qerror_loc")
    return input_to_op_qerror_loc(input);
  if (name == "multiplexer")
    return input_to_op_multiplexer(input);
  if (name == "kraus")
    return input_to_op_kraus(input);
  if (name == "roerror")
    return input_to_op_roerror(input);
  if (name == "pauli")
    return input_to_op_pauli(input);

  //Control-flow
  if (name == "jump")
    return input_to_op_jump(input);
  if (name == "mark")
    return input_to_op_mark(input);
  // Default assume gate
  return input_to_op_gate(input);
}

json_t op_to_json(const Op &op) {
  json_t ret;
  ret["name"] = op.name;
  if (!op.qubits.empty())
    ret["qubits"] = op.qubits;
  if (!op.regs.empty())
    ret["regs"] = op.regs;
  if (!op.params.empty())
    ret["params"] = op.params;
  else if (!op.int_params.empty())
    ret["params"] = op.int_params;
  if (op.conditional)
    ret["conditional"] = op.conditional_reg;
  if (!op.memory.empty())
    ret["memory"] = op.memory;
  if (!op.registers.empty())
    ret["register"] = op.registers;
  if (!op.mats.empty())
    ret["mats"] = op.mats;
  return ret;
}


void to_json(json_t &js, const OpType& type) {
  std::stringstream ss;
  ss << type;
  js = ss.str();
}


void to_json(json_t &js, const DataSubType& subtype) {
  std::stringstream ss;
  ss << subtype;
  js = ss.str();
}


//------------------------------------------------------------------------------
// Implementation: Gates, measure, reset deserialization
//------------------------------------------------------------------------------

template<typename inputdata_t>
void add_conditional(const Allowed allowed, Op& op, const inputdata_t& input) {
  // Check conditional
  if (Parser<inputdata_t>::check_key("conditional", input)) {
    // If instruction isn't allow to be conditional throw an exception
    if (allowed == Allowed::No) {
      throw std::invalid_argument("Invalid instruction: \"" + op.name + "\" cannot be conditional.");
    }
    // If instruction is allowed to be conditional add parameters
    Parser<inputdata_t>::get_value(op.conditional_reg, "conditional", input);
    op.conditional = true;
  }
}

template<typename inputdata_t>
Op input_to_op_gate(const inputdata_t& input) {
  Op op;
  op.type = OpType::gate;
  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.params, "params", input);

  // Check for optional label
  // If label is not specified record the gate name as the label
  std::string label;
  Parser<inputdata_t>::get_value(label, "label", input);
  if  (label != "") 
    op.string_params = {label};
  else
    op.string_params = {op.name};

  // Conditional
  add_conditional(Allowed::Yes, op, input);

  // Validation
  check_empty_name(op);
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.name == "u1")
    check_length_params(op, 1);
  else if (op.name == "u2")
    check_length_params(op, 2);
  else if (op.name == "u3")
    check_length_params(op, 3);
  return op;
}

template<typename inputdata_t>
Op input_to_op_qerror_loc(const inputdata_t& input) {
  Op op;
  op.type = OpType::qerror_loc;
  Parser<inputdata_t>::get_value(op.name, "label", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::Yes, op, input);
  return op;
}

template<typename inputdata_t>
Op input_to_op_barrier(const inputdata_t &input) {
  Op op;
  op.type = OpType::barrier;
  op.name = "barrier";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  // Check conditional
  add_conditional(Allowed::No, op, input);
  return op;
}

template<typename inputdata_t>
Op input_to_op_measure(const inputdata_t& input) {
  Op op;
  op.type = OpType::measure;
  op.name = "measure";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.memory, "memory", input);
  Parser<inputdata_t>::get_value(op.registers, "register", input);

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.memory.empty() == false && op.memory.size() != op.qubits.size()) {
    throw std::invalid_argument(R"(Invalid measure operation: "memory" and "qubits" are different lengths.)");
  }
  if (op.registers.empty() == false && op.registers.size() != op.qubits.size()) {
    throw std::invalid_argument(R"(Invalid measure operation: "register" and "qubits" are different lengths.)");
  }
  return op;
}

template<typename inputdata_t>
Op input_to_op_reset(const inputdata_t& input) {
  Op op;
  op.type = OpType::reset;
  op.name = "reset";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  return op;
}

template<typename inputdata_t>
Op input_to_op_initialize(const inputdata_t& input) {
  Op op;
  op.type = OpType::initialize;
  op.name = "initialize";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.params, "params", input);

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  check_length_params(op, 1ULL << op.qubits.size());
  return op;
}
template<typename inputdata_t>
Op input_to_op_pauli(const inputdata_t& input){
  Op op;
  op.type = OpType::gate;
  op.name = "pauli";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.string_params, "params", input);

  // Check for optional label
  // If label is not specified record the gate name as the label
  std::string label;
  Parser<inputdata_t>::get_value(label, "label", input);
  if  (label != "")
    op.string_params.push_back(label);
  else
    op.string_params.push_back(op.name);

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);

  return op;
}

//------------------------------------------------------------------------------
// Implementation: Boolean Functions
//------------------------------------------------------------------------------
template<typename inputdata_t>
Op input_to_op_bfunc(const inputdata_t& input) {
  Op op;
  op.type = OpType::bfunc;
  op.name = "bfunc";
  op.string_params.resize(2);
  std::string relation;
  Parser<inputdata_t>::get_value(op.string_params[0], "mask", input); // mask hexadecimal string
  Parser<inputdata_t>::get_value(op.string_params[1], "val", input);  // value hexadecimal string
  Parser<inputdata_t>::get_value(relation, "relation", input); // relation string
  // Load single register / memory bit for storing result
  uint_t tmp;
  if (Parser<inputdata_t>::get_value(tmp, "register", input)) {
    op.registers.push_back(tmp);
  }
  if (Parser<inputdata_t>::get_value(tmp, "memory", input)) {
    op.memory.push_back(tmp);
  }
  
  // Format hex strings
  Utils::format_hex_inplace(op.string_params[0]);
  Utils::format_hex_inplace(op.string_params[1]);

  const stringmap_t<RegComparison> comp_table({
    {"==", RegComparison::Equal},
    {"!=", RegComparison::NotEqual},
    {"<", RegComparison::Less},
    {"<=", RegComparison::LessEqual},
    {">", RegComparison::Greater},
    {">=", RegComparison::GreaterEqual},
  });

  auto it = comp_table.find(relation);
  if (it == comp_table.end()) {
    std::stringstream msg;
    msg << "Invalid bfunc relation string :\"" << it->first << "\"." << std::endl;
    throw std::invalid_argument(msg.str());
  } else {
    op.bfunc = it->second;
  }

  // Conditional
  add_conditional(Allowed::No, op, input);

  // Validation
  if (op.registers.empty()) {
    throw std::invalid_argument("Invalid measure operation: \"register\" is empty.");
  }
  return op;
}

template<typename inputdata_t>
Op input_to_op_roerror(const inputdata_t& input) {
  Op op;
  op.type = OpType::roerror;
  op.name = "roerror";
  Parser<inputdata_t>::get_value(op.memory, "memory", input);
  Parser<inputdata_t>::get_value(op.registers, "register", input);
  Parser<inputdata_t>::get_value(op.probs, "params", input);
  // Conditional
  add_conditional(Allowed::No, op, input);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Matrix and Kraus deserialization
//------------------------------------------------------------------------------
template<typename inputdata_t>
Op input_to_op_unitary(const inputdata_t& input) {
  Op op;
  op.type = OpType::matrix;
  op.name = "unitary";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.mats, "params", input);
  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.mats.size() != 1) {
    throw std::invalid_argument("\"unitary\" params must be a single matrix.");
  }
  for (const auto &mat : op.mats) {
    if (!Utils::is_unitary(mat, 1e-7)) {
      throw std::invalid_argument("\"unitary\" matrix is not unitary.");
    }
  }
  // Check for a label
  std::string label;
  Parser<inputdata_t>::get_value(label, "label", input);
  op.string_params.push_back(label);

  // Conditional
  add_conditional(Allowed::Yes, op, input);
  return op;
}
template<typename inputdata_t>
Op input_to_op_diagonal(const inputdata_t& input) {
  Op op;
  op.type = OpType::diagonal_matrix;
  op.name = "diagonal";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.params, "params", input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.params.size() != 1ULL << op.qubits.size()) {
    throw std::invalid_argument("\"diagonal\" matrix is wrong size.");
  }
  for (const auto &val : op.params) {
    if (!Linalg::almost_equal(std::abs(val), 1.0, 1e-7)) {
      throw std::invalid_argument("\"diagonal\" matrix is not unitary.");
    }
  }

  // Check for a label
  std::string label;
  Parser<inputdata_t>::get_value(label, "label", input);
  op.string_params.push_back(label);

  // Conditional
  add_conditional(Allowed::Yes, op, input);
  return op;
}
template<typename inputdata_t>
Op input_to_op_superop(const inputdata_t& input) {
  // Warning: we don't check superoperator is valid!
  Op op;
  op.type = OpType::superop;
  op.name = "superop";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.mats, "params", input);
  // Check conditional
  add_conditional(Allowed::Yes, op, input);
  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.mats.size() != 1) {
    throw std::invalid_argument("\"superop\" params must be a single matrix.");
  }
  return op;
}
template<typename inputdata_t>
Op input_to_op_multiplexer(const inputdata_t& input) {
  // Parse parameters
  reg_t qubits;
  std::vector<cmatrix_t> mats;
  std::string label;
  Parser<inputdata_t>::get_value(qubits, "qubits", input);
  Parser<inputdata_t>::get_value(mats, "params", input);
  Parser<inputdata_t>::get_value(label, "label", input);
  // Construct op
  auto op = make_multiplexer(qubits, mats, label);
  // Conditional
  add_conditional(Allowed::Yes, op, input);
  return op;
}
template<typename inputdata_t>
Op input_to_op_kraus(const inputdata_t& input) {
  Op op;
  op.type = OpType::kraus;
  op.name = "kraus";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.mats, "params", input);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  // Conditional
  add_conditional(Allowed::Yes, op, input);
  return op;
}

template<typename inputdata_t>
Op input_to_op_noise_switch(const inputdata_t& input) {
  Op op;
  op.type = OpType::noise_switch;
  op.name = "noise_switch";
  Parser<inputdata_t>::get_value(op.params, "params", input);
  // Conditional
  add_conditional(Allowed::No, op, input);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Set state
//------------------------------------------------------------------------------
template<typename inputdata_t>
Op input_to_op_set_vector(const inputdata_t &input, OpType op_type) {
  Op op;
  op.type = op_type;
  const inputdata_t& params = Parser<inputdata_t>::get_value("params", input);
  op.params = Parser<inputdata_t>::template get_list_elem<std::vector<complex_t>>(params, 0);
  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::No, op, input);
  return op;
}

template<typename inputdata_t>
Op input_to_op_set_matrix(const inputdata_t &input, OpType op_type) {
  Op op;
  op.type = op_type;
  const inputdata_t& params = Parser<inputdata_t>::get_value("params", input);
  op.mats.push_back(Parser<inputdata_t>::template get_list_elem<cmatrix_t>(params, 0));
  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::No, op, input);
  return op;
}

template<typename inputdata_t>
Op input_to_op_set_clifford(const inputdata_t &input, OpType op_type) {
  Op op;
  op.type = op_type;
  const inputdata_t& params = Parser<inputdata_t>::get_value("params", input);
  op.clifford = Parser<inputdata_t>::template get_list_elem<Clifford::Clifford>(params, 0);
  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::No, op, input);
  return op;
}

template<typename inputdata_t>
Op input_to_op_set_mps(const inputdata_t &input, OpType op_type) {
  Op op;
  op.type = op_type;
  const inputdata_t& params = Parser<inputdata_t>::get_value("params", input);
  op.mps = Parser<inputdata_t>::template get_list_elem<mps_container_t>(params, 0);

  Parser<inputdata_t>::get_value(op.name, "name", input);
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  add_conditional(Allowed::No, op, input);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Save data deserialization
//------------------------------------------------------------------------------
template<typename inputdata_t>
Op input_to_op_save_default(const inputdata_t& input, OpType op_type) {
  Op op;
  op.type = op_type;
  Parser<inputdata_t>::get_value(op.name, "name", input);

  // Get subtype
  static const std::unordered_map<std::string, DataSubType> subtypes {
    {"single", DataSubType::single},
    {"c_single", DataSubType::c_single},
    {"average", DataSubType::average},
    {"c_average", DataSubType::c_average},
    {"list", DataSubType::list},
    {"c_list", DataSubType::c_list},
    {"accum", DataSubType::accum},
    {"c_accum", DataSubType::c_accum},
  };
  std::string subtype;
  Parser<inputdata_t>::get_value(subtype, "snapshot_type", input);
  auto subtype_it = subtypes.find(subtype);
  if (subtype_it == subtypes.end()) {
    throw std::runtime_error("Invalid data subtype \"" + subtype +
                             "\" in save data instruction.");
  }
  op.save_type = subtype_it->second;
 
  // Get data key
  op.string_params.emplace_back("");
  Parser<inputdata_t>::get_value(op.string_params[0], "label", input);

  // Add optional qubits field
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  return op;
}
template<typename inputdata_t>
Op input_to_op_save_expval(const inputdata_t& input, bool variance) {
  // Initialized default save instruction params
  auto op_type = (variance) ? OpType::save_expval_var
                            : OpType::save_expval;
  Op op = input_to_op_save_default(input, op_type);

  // Parse Pauli operator components
  const auto threshold = 1e-12; // drop small components
  // Get components
  if (Parser<inputdata_t>::check_key("params", input) && Parser<inputdata_t>::is_array("params", input)) {
    for (const auto &comp_ : Parser<inputdata_t>::get_value("params", input)) {
      const auto& comp = Parser<inputdata_t>::get_as_list(comp_);
      // Get complex coefficient
      std::vector<double> coeffs = Parser<inputdata_t>::template get_list_elem<std::vector<double>>(comp, 1);
      if (std::abs(coeffs[0]) > threshold || std::abs(coeffs[1]) > threshold) {
        std::string pauli = Parser<inputdata_t>::template get_list_elem<std::string>(comp, 0);
        if (pauli.size() != op.qubits.size()) {
          throw std::invalid_argument(std::string("Invalid expectation value save instruction ") +
                                      "(Pauli label does not match qubit number.).");
        }
        op.expval_params.emplace_back(pauli, coeffs[0], coeffs[1]);
      }
    }
  } else {
    throw std::invalid_argument("Invalid save expectation value \"params\".");
  }

  // Check edge case of all coefficients being empty
  // In this case the operator had all coefficients zero, or sufficiently close
  // to zero that they were all truncated.
  if (op.expval_params.empty()) {
    std::string pauli(op.qubits.size(), 'I');
    op.expval_params.emplace_back(pauli, 0., 0.);
  }

  return op;
}
template<typename inputdata_t>
Op input_to_op_save_amps(const inputdata_t& input, bool squared) {
  // Initialized default save instruction params
  auto op_type = (squared) ? OpType::save_amps_sq
                           : OpType::save_amps;
  Op op = input_to_op_save_default(input, op_type);
  Parser<inputdata_t>::get_value(op.int_params, "params", input);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Snapshot deserialization
//------------------------------------------------------------------------------
template<typename inputdata_t>
Op input_to_op_snapshot(const inputdata_t& input) {
  std::string snapshot_type;
  Parser<inputdata_t>::get_value(snapshot_type, "snapshot_type", input); // LEGACY: to remove in 0.3
  Parser<inputdata_t>::get_value(snapshot_type, "type", input);
  if (snapshot_type.find("expectation_value_pauli") != std::string::npos)
    return input_to_op_snapshot_pauli(input);
  if (snapshot_type.find("expectation_value_matrix") != std::string::npos)
    return input_to_op_snapshot_matrix(input);
  // Default snapshot: has "type", "label", "qubits"
  auto op = input_to_op_snapshot_default(input);
  // Conditional
  add_conditional(Allowed::No, op, input);
  return op;
}

template<typename inputdata_t>
Op input_to_op_snapshot_default(const inputdata_t& input) {
  Op op;
  op.type = OpType::snapshot;
  Parser<inputdata_t>::get_value(op.name, "type", input); // LEGACY: to remove in 0.3
  Parser<inputdata_t>::get_value(op.name, "snapshot_type", input);
  // If missing use "default" for label
  op.string_params.emplace_back("default");
  Parser<inputdata_t>::get_value(op.string_params[0], "label", input);
  // Add optional qubits field
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  // If qubits is not empty, check for duplicates
  check_duplicate_qubits(op);
  return op;
}

template<typename inputdata_t>
Op input_to_op_snapshot_pauli(const inputdata_t& input) {
  Op op = input_to_op_snapshot_default(input);

  const auto threshold = 1e-15; // drop small components
  // Get components
  if (Parser<inputdata_t>::check_key("params", input) && Parser<inputdata_t>::is_array("params", input)) {
    for (const auto &comp_ : Parser<inputdata_t>::get_value("params", input)) {
      // Check component is length-2 array
      const auto& comp = Parser<inputdata_t>::get_as_list(comp_);
      if (comp.size() != 2)
        throw std::invalid_argument("Invalid Pauli expval params (param component " + 
                                    Parser<inputdata_t>::dump(comp) + " invalid).");
      // Get complex coefficient
      complex_t coeff = Parser<inputdata_t>::template get_list_elem<complex_t>(comp, 0);
      // If coefficient is above threshold, get the Pauli operator string
      // This string may contain I, X, Y, Z
      // qubits are stored as a list where position is qubit number:
      // eq op.qubits = [a, b, c], a is qubit-0, b is qubit-1, c is qubit-2
      // Pauli string labels are stored in little-endian ordering:
      // eg label = "CBA", A is the Pauli for qubit-0, B for qubit-1, C for qubit-2
      if (std::abs(coeff) > threshold) {
        std::string pauli = Parser<inputdata_t>::template get_list_elem<std::string>(comp, 1);
        if (pauli.size() != op.qubits.size()) {
          throw std::invalid_argument(std::string("Invalid Pauli expectation value snapshot ") +
                                      "(Pauli label does not match qubit number.).");
        }
        // make tuple and add to components
        op.params_expval_pauli.emplace_back(coeff, pauli);
      } // end if > threshold
    } // end component loop
  } else {
    throw std::invalid_argument("Invalid Pauli expectation value value snapshot \"params\".");
  }
  // Check edge case of all coefficients being empty
  // In this case the operator had all coefficients zero, or sufficiently close
  // to zero that they were all truncated.
  if (op.params_expval_pauli.empty()) {
    // Add a single identity op with zero coefficient
    std::string pauli(op.qubits.size(), 'I');
    complex_t coeff(0);
    op.params_expval_pauli.emplace_back(coeff, pauli);
  }

  return op;
}

template<typename inputdata_t>
Op input_to_op_snapshot_matrix(const inputdata_t& input) {
  // Load default snapshot parameters
  Op op = input_to_op_snapshot_default(input);

  const auto threshold = 1e-10; // drop small components
  // Get matrix operator components
  // TODO: fix repeated throw string
  if (Parser<inputdata_t>::check_key("params", input) && Parser<inputdata_t>::is_array("params", input)) {
    for (const auto &comp_ : Parser<inputdata_t>::get_value("params", input)) {
      const auto& comp = Parser<inputdata_t>::get_as_list(comp_);
      // Check component is length-2 array
      if (comp.size() != 2) {
        throw std::invalid_argument("Invalid matrix expval snapshot (param component " +
                                        Parser<inputdata_t>::dump(comp) + " invalid).");
      }
      // Get complex coefficient
      complex_t coeff = Parser<inputdata_t>::template get_list_elem<complex_t>(comp, 0);
      std::vector<std::pair<reg_t, cmatrix_t>> mats;
      if (std::abs(coeff) > threshold) {
        const inputdata_t& comp_list = comp[1];
        if (!Parser<inputdata_t>::is_array(comp_list)) {
          throw std::invalid_argument("Invalid matrix expval snapshot (param component " +
                                          Parser<inputdata_t>::dump(comp) + " invalid).");
        }
        for (const auto &subcomp_ : comp_list) {
          const auto& subcomp = Parser<inputdata_t>::get_as_list(subcomp_);
          if (subcomp.size() != 2) {
            throw std::invalid_argument("Invalid matrix expval snapshot (param component " +
                                            Parser<inputdata_t>::dump(comp) + " invalid).");
          }
          reg_t comp_qubits = Parser<inputdata_t>::template get_list_elem<reg_t>(subcomp, 0);
          cmatrix_t comp_matrix = Parser<inputdata_t>::template get_list_elem<cmatrix_t>(subcomp, 1);
          // Check qubits are ok
          // TODO: check that qubits are in range from 0 to Num of Qubits - 1 for instr
          std::unordered_set<uint_t> unique = {comp_qubits.begin(), comp_qubits.end()};
          if (unique.size() != comp_qubits.size()) {
            throw std::invalid_argument("Invalid matrix expval snapshot (param component " +
                                            Parser<inputdata_t>::dump(comp) + " invalid).");
          }
          mats.emplace_back(comp_qubits, comp_matrix);
        }
        op.params_expval_matrix.emplace_back(coeff, mats);
      }
    } // end component loop
  } else {
    throw std::invalid_argument(std::string("Invalid matrix expectation value snapshot ") +
                                "(\"params\" field missing).");
  }
  return op;
}

template<typename inputdata_t>
Op input_to_op_jump(const inputdata_t &input) {
  Op op;
  op.type = OpType::jump;
  op.name = "jump";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.string_params, "params", input);
  if (op.string_params.empty())
    throw std::invalid_argument(std::string("Invalid jump (\"params\" field missing)."));

  // Conditional
  add_conditional(Allowed::Yes, op, input);

  return op;
}

template<typename inputdata_t>
Op input_to_op_mark(const inputdata_t &input) {
  Op op;
  op.type = OpType::mark;
  op.name = "mark";
  Parser<inputdata_t>::get_value(op.qubits, "qubits", input);
  Parser<inputdata_t>::get_value(op.string_params, "params", input);
  if (op.string_params.empty())
    throw std::invalid_argument(std::string("Invalid mark (\"params\" field missing)."));

  // Conditional
  add_conditional(Allowed::No, op, input);

  return op;
}


//------------------------------------------------------------------------------
} // end namespace Operations
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
