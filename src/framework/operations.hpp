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
#include "framework/json.hpp"
#include "framework/utils.hpp"
#include "framework/linalg/almost_equal.hpp"

namespace AER {
namespace Operations {

// Comparisons enum class used for Boolean function operation.
// these are used to compare two hexadecimal strings and return a bool
// for now we only have one comparison Equal, but others will be added
enum class RegComparison {Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual};

// Enum class for operation types
enum class OpType {
  gate, measure, reset, bfunc, barrier, snapshot,
  matrix, diagonal_matrix, multiplexer, kraus, superop, roerror,
  noise_switch, initialize, nop
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
  case OpType::noise_switch:
    stream << "noise_switch";
    break;
  case OpType::initialize:
    stream << "initialize";
    break;
  case OpType::nop:
    stream << "nop";
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
  std::vector<std::string> string_params; // used or snapshot label, and boolean functions

  // Conditional Operations
  bool conditional = false; // is gate conditional gate
  uint_t conditional_reg;   // (opt) the (single) register location to look up for conditional
  RegComparison bfunc;      // (opt) boolean function relation

  // DEPRECATED: Old style conditionals (remove in 0.3)
  bool old_conditional = false;     // is gate old style conditional gate
  std::string old_conditional_mask; // hex string for conditional mask
  std::string old_conditional_val;  // hex string for conditional value

  // Measurement
  reg_t memory;             // (opt) register operation it acts on (measure)
  reg_t registers;          // (opt) register locations it acts on (measure, conditional)

  // Mat and Kraus
  std::vector<cmatrix_t> mats;

  // Readout error
  std::vector<rvector_t> probs;

  // Snapshots
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

inline Op make_superop(const reg_t &qubits, const cmatrix_t &mat) {
  Op op;
  op.type = OpType::superop;
  op.name = "superop";
  op.qubits = qubits;
  op.mats = {mat};
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
  std::copy_n(qubits.begin(), num_controls, controls.begin());
  std::copy_n(qubits.begin() + num_controls, num_targets, targets.begin());

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

inline Op make_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &mats) {
  Op op;
  op.type = OpType::kraus;
  op.name = "kraus";
  op.qubits = qubits;
  op.mats = mats;
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

//------------------------------------------------------------------------------
// JSON conversion
//------------------------------------------------------------------------------

// Main JSON deserialization functions
Op json_to_op(const json_t &js); // Patial TODO
json_t op_to_json(const Op &op); // Patial TODO
inline void from_json(const json_t &js, Op &op) {op = json_to_op(js);}
inline void to_json(json_t &js, const Op &op) { js = op_to_json(op);}

// Standard operations
Op json_to_op_gate(const json_t &js);
Op json_to_op_barrier(const json_t &js);
Op json_to_op_measure(const json_t &js);
Op json_to_op_reset(const json_t &js);
Op json_to_op_bfunc(const json_t &js);
Op json_to_op_initialize(const json_t &js);

// Snapshots
Op json_to_op_snapshot(const json_t &js);
Op json_to_op_snapshot_default(const json_t &js);
Op json_to_op_snapshot_matrix(const json_t &js);
Op json_to_op_snapshot_pauli(const json_t &js);

// Matrices
Op json_to_op_unitary(const json_t &js);
Op json_to_op_diagonal(const json_t &js);
Op json_to_op_superop(const json_t &js);
Op json_to_op_multiplexer(const json_t &js);
Op json_to_op_kraus(const json_t &js);
Op json_to_op_noise_switch(const json_t &js);

// Classical bits
Op json_to_op_roerror(const json_t &js);

// Optional instruction parameters
enum class Allowed {Yes, No};
void add_condtional(const Allowed val, Op& op, const json_t &js);


//------------------------------------------------------------------------------
// Implementation: JSON deserialization
//------------------------------------------------------------------------------

// TODO: convert if-else to switch
Op json_to_op(const json_t &js) {
  // load operation identifier
  std::string name;
  JSON::get_value(name, "name", js);
  // Barrier
  if (name == "barrier")
    return json_to_op_barrier(js);
  // Measure & Reset
  if (name == "measure")
    return json_to_op_measure(js);
  if (name == "reset")
    return json_to_op_reset(js);
  if (name == "initialize")
    return json_to_op_initialize(js);
  // Arbitrary matrix gates
  if (name == "unitary")
    return json_to_op_unitary(js);
  if (name == "diagonal" || name == "diag")
    return json_to_op_diagonal(js);
  if (name == "superop")
    return json_to_op_superop(js);
  // Snapshot
  if (name == "snapshot")
    return json_to_op_snapshot(js);
  // Bit functions
  if (name == "bfunc")
    return json_to_op_bfunc(js);
  // Noise functions
  if (name == "noise_switch")
    return json_to_op_noise_switch(js);
  if (name == "multiplexer")
    return json_to_op_multiplexer(js);
  if (name == "kraus")
    return json_to_op_kraus(js);
  if (name == "roerror")
    return json_to_op_roerror(js);
  // Default assume gate
  return json_to_op_gate(js);
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


//------------------------------------------------------------------------------
// Implementation: Gates, measure, reset deserialization
//------------------------------------------------------------------------------


void add_condtional(const Allowed allowed, Op& op, const json_t &js) {
  // Check conditional
  if (JSON::check_key("conditional", js)) {
    // If instruction isn't allow to be conditional throw an exception
    if (allowed == Allowed::No) {
      throw std::invalid_argument("Invalid instruction: \"" + op.name + "\" cannot be conditional.");
    }
    // If instruction is allowed to be conditional add parameters
    if (js["conditional"].is_number()) {
      // New style conditional
      op.conditional_reg = js["conditional"];
      op.conditional = true;
    } else {
      // DEPRECATED: old style conditional (remove in 0.3)
      JSON::get_value(op.old_conditional_mask, "mask", js["conditional"]);
      JSON::get_value(op.old_conditional_val, "val", js["conditional"]);
      op.old_conditional = true;
    }
  }
}


Op json_to_op_gate(const json_t &js) {
  Op op;
  op.type = OpType::gate;
  JSON::get_value(op.name, "name", js);
  JSON::get_value(op.qubits, "qubits", js);
  JSON::get_value(op.params, "params", js);

  // Check for optional label
  // If label is not specified record the gate name as the label
  std::string label;
  JSON::get_value(label, "label", js);
  if  (label != "") 
    op.string_params = {label};
  else
    op.string_params = {op.name};

  // Conditional
  add_condtional(Allowed::Yes, op, js);

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


Op json_to_op_barrier(const json_t &js) {
  Op op;
  op.type = OpType::barrier;
  op.name = "barrier";
  JSON::get_value(op.qubits, "qubits", js);
  // Check conditional
  add_condtional(Allowed::No, op, js);
  return op;
}


Op json_to_op_measure(const json_t &js) {
  Op op;
  op.type = OpType::measure;
  op.name = "measure";
  JSON::get_value(op.qubits, "qubits", js);
  JSON::get_value(op.memory, "memory", js);
  JSON::get_value(op.registers, "register", js);

  // Conditional
  add_condtional(Allowed::No, op, js);

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


Op json_to_op_reset(const json_t &js) {
  Op op;
  op.type = OpType::reset;
  op.name = "reset";
  JSON::get_value(op.qubits, "qubits", js);

  // Conditional
  add_condtional(Allowed::No, op, js);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  return op;
}


Op json_to_op_initialize(const json_t &js) {
  Op op;
  op.type = OpType::initialize;
  op.name = "initialize";
  JSON::get_value(op.qubits, "qubits", js);
  JSON::get_value(op.params, "params", js);

  // Conditional
  add_condtional(Allowed::No, op, js);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  check_length_params(op, 1ULL << op.qubits.size());
  return op;
}


//------------------------------------------------------------------------------
// Implementation: Boolean Functions
//------------------------------------------------------------------------------

Op json_to_op_bfunc(const json_t &js) {
  Op op;
  op.type = OpType::bfunc;
  op.name = "bfunc";
  op.string_params.resize(2);
  std::string relation;
  JSON::get_value(op.string_params[0], "mask", js); // mask hexadecimal string
  JSON::get_value(op.string_params[1], "val", js);  // value hexadecimal string
  JSON::get_value(relation, "relation", js); // relation string
  // Load single register / memory bit for storing result
  uint_t tmp;
  if (JSON::get_value(tmp, "register", js)) {
    op.registers.push_back(tmp);
  }
  if (JSON::get_value(tmp, "memory", js)) {
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
  add_condtional(Allowed::No, op, js);

  // Validation
  if (op.registers.empty()) {
    throw std::invalid_argument("Invalid measure operation: \"register\" is empty.");
  }
  return op;
}


Op json_to_op_roerror(const json_t &js) {
  Op op;
  op.type = OpType::roerror;
  op.name = "roerror";
  JSON::get_value(op.memory, "memory", js);
  JSON::get_value(op.registers, "register", js);
  JSON::get_value(op.probs, "probabilities", js); // DEPRECATED: Remove in 0.4
  JSON::get_value(op.probs, "params", js);
  // Conditional
  add_condtional(Allowed::No, op, js);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Matrix and Kraus deserialization
//------------------------------------------------------------------------------

Op json_to_op_unitary(const json_t &js) {
  Op op;
  op.type = OpType::matrix;
  op.name = "unitary";
  JSON::get_value(op.qubits, "qubits", js);
  JSON::get_value(op.mats, "params", js);
  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.mats.size() != 1) {
    throw std::invalid_argument("\"unitary\" params must be a single matrix.");
  }
  for (const auto mat : op.mats) {
    if (!Utils::is_unitary(mat, 1e-7)) {
      throw std::invalid_argument("\"unitary\" matrix is not unitary.");
    }
  }
  // Check for a label
  std::string label;
  JSON::get_value(label, "label", js);
  op.string_params.push_back(label);

  // Conditional
  add_condtional(Allowed::Yes, op, js);
  return op;
}

Op json_to_op_diagonal(const json_t &js) {
  Op op;
  op.type = OpType::diagonal_matrix;
  op.name = "diagonal";
  JSON::get_value(op.qubits, "qubits", js);
  JSON::get_value(op.params, "params", js);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.params.size() != 1ULL << op.qubits.size()) {
    throw std::invalid_argument("\"diagonal\" matrix is wrong size.");
  }
  for (const auto val : op.params) {
    if (!Linalg::almost_equal(std::abs(val), 1.0, 1e-7)) {
      throw std::invalid_argument("\"diagonal\" matrix is not unitary.");
    }
  }

  // Check for a label
  std::string label;
  JSON::get_value(label, "label", js);
  op.string_params.push_back(label);

  // Conditional
  add_condtional(Allowed::Yes, op, js);
  return op;
}

Op json_to_op_superop(const json_t &js) {
  // Warning: we don't check superoperator is valid!
  Op op;
  op.type = OpType::superop;
  op.name = "superop";
  JSON::get_value(op.qubits, "qubits", js);
  JSON::get_value(op.mats, "params", js);
  // Check conditional
  add_condtional(Allowed::Yes, op, js);
  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.mats.size() != 1) {
    throw std::invalid_argument("\"superop\" params must be a single matrix.");
  }
  return op;
}

Op json_to_op_multiplexer(const json_t &js) {
  // Parse parameters
  reg_t qubits;
  std::vector<cmatrix_t> mats;
  std::string label;
  JSON::get_value(qubits, "qubits", js);
  JSON::get_value(mats, "params", js);
  JSON::get_value(label, "label", js);
  // Construct op
  auto op = make_multiplexer(qubits, mats, label);
  // Conditional
  add_condtional(Allowed::Yes, op, js);
  return op;
}

Op json_to_op_kraus(const json_t &js) {
  Op op;
  op.type = OpType::kraus;
  op.name = "kraus";
  JSON::get_value(op.qubits, "qubits", js);
  JSON::get_value(op.mats, "params", js);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  // Conditional
  add_condtional(Allowed::Yes, op, js);
  return op;
}


Op json_to_op_noise_switch(const json_t &js) {
  Op op;
  op.type = OpType::noise_switch;
  op.name = "noise_switch";
  JSON::get_value(op.params, "params", js);
  // Conditional
  add_condtional(Allowed::No, op, js);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Snapshot deserialization
//------------------------------------------------------------------------------

Op json_to_op_snapshot(const json_t &js) {
  std::string snapshot_type;
  JSON::get_value(snapshot_type, "snapshot_type", js); // LEGACY: to remove in 0.3
  JSON::get_value(snapshot_type, "type", js);
  if (snapshot_type.find("expectation_value_pauli") != std::string::npos)
    return json_to_op_snapshot_pauli(js);
  if (snapshot_type.find("expectation_value_matrix") != std::string::npos)
    return json_to_op_snapshot_matrix(js);
  // Default snapshot: has "type", "label", "qubits"
  auto op = json_to_op_snapshot_default(js);
  // Conditional
  add_condtional(Allowed::No, op, js);
  return op;
}


Op json_to_op_snapshot_default(const json_t &js) {
  Op op;
  op.type = OpType::snapshot;
  JSON::get_value(op.name, "type", js); // LEGACY: to remove in 0.3
  JSON::get_value(op.name, "snapshot_type", js);
  // If missing use "default" for label
  op.string_params.emplace_back("default");
  JSON::get_value(op.string_params[0], "label", js);
  // Add optional qubits field
  JSON::get_value(op.qubits, "qubits", js);
  // If qubits is not empty, check for duplicates
  check_duplicate_qubits(op);
  return op;
}


Op json_to_op_snapshot_pauli(const json_t &js) {
  // Load default snapshot parameters
  Op op = json_to_op_snapshot_default(js);

  // Check qubits are valid
  check_empty_qubits(op);
  check_duplicate_qubits(op);

  // Parse Pauli operator components
  const auto threshold = 1e-10; // drop small components
  // Get components
  if (JSON::check_key("params", js) && js["params"].is_array()) {
    for (const auto &comp : js["params"]) {
      // Check component is length-2 array
      if (!comp.is_array() || comp.size() != 2)
        throw std::invalid_argument("Invalid Pauli expval snapshot (param component " + 
                                    comp.dump() + " invalid).");
      // Get complex coefficient
      complex_t coeff = comp[0];
      // If coefficient is above threshold, get the Pauli operator string
      // This string may contain I, X, Y, Z
      // qubits are stored as a list where position is qubit number:
      // eq op.qubits = [a, b, c], a is qubit-0, b is qubit-1, c is qubit-2
      // Pauli string labels are stored in little-endian ordering:
      // eg label = "CBA", A is the Pauli for qubit-0, B for qubit-1, C for qubit-2
      if (std::abs(coeff) > threshold) {
        std::string pauli = comp[1];
        if (pauli.size() != op.qubits.size()) {
          throw std::invalid_argument(std::string("Invalid Pauli expectation value snapshot ") +
                                      "(Pauli label does not match qubit number.).");
        }
        // make tuple and add to components
        op.params_expval_pauli.emplace_back(coeff, pauli);
      } // end if > threshold
    } // end component loop
  } else {
    throw std::invalid_argument("Invalid Pauli snapshot \"params\".");
  }
  return op;
}


Op json_to_op_snapshot_matrix(const json_t &js) {
  // Load default snapshot parameters
  Op op = json_to_op_snapshot_default(js);

  const auto threshold = 1e-10; // drop small components
  // Get matrix operator components
  // TODO: fix repeated throw string
  if (JSON::check_key("params", js) && js["params"].is_array()) {
    for (const auto &comp : js["params"]) {
      // Check component is length-2 array
      if (!comp.is_array() || comp.size() != 2) {
        throw std::invalid_argument("Invalid matrix expval snapshot (param component " + 
                                    comp.dump() + " invalid).");
      }
      // Get complex coefficient
      complex_t coeff = comp[0];
      std::vector<std::pair<reg_t, cmatrix_t>> mats;
      if (std::abs(coeff) > threshold) {
        if (!comp[1].is_array()) {
          throw std::invalid_argument("Invalid matrix expval snapshot (param component " + 
                                      comp.dump() + " invalid).");
        }
        for (const auto &subcomp : comp[1]) {
          if (!subcomp.is_array() || subcomp.size() != 2) {
            throw std::invalid_argument("Invalid matrix expval snapshot (param component " + 
                                        comp.dump() + " invalid).");
          }
          reg_t comp_qubits = subcomp[0];
          cmatrix_t comp_matrix = subcomp[1];
          // Check qubits are ok
          // TODO: check that qubits are in range from 0 to Num of Qubits - 1 for instr
          std::unordered_set<uint_t> unique = {comp_qubits.begin(), comp_qubits.end()};
          if (unique.size() != comp_qubits.size()) {
            throw std::invalid_argument("Invalid matrix expval snapshot (param component " + 
                                        comp.dump() + " invalid).");
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

//------------------------------------------------------------------------------
} // end namespace Operations
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
