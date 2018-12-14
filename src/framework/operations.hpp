/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_framework_operations_hpp_
#define _aer_framework_operations_hpp_

#include <algorithm> 
#include <stdexcept>
#include <sstream>
#include <tuple>

#include "framework/types.hpp"
#include "framework/json.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace Operations {

// Comparisons enum class used for Boolean function operation.
// these are used to compare two hexadecimal strings and return a bool
// for now we only have one comparison Equal, but others will be added
enum class RegComparison {Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual};

// Enum class for operation types
enum class OpType {
  gate, measure, reset, bfunc, barrier, snapshot,
  matrix, kraus, roerror, noise_switch
};

//------------------------------------------------------------------------------
// Op Class
//------------------------------------------------------------------------------


struct Op {
  // General Operations
  OpType type;                    // operation type identifier
  std::string name;               // operation name
  reg_t qubits;                   //  qubits operation acts on
  std::vector<complex_t> params;  // real or complex params for gates
  std::vector<std::string> string_params; // used or snapshot label, and boolean functions

  // Conditional Operations
  bool conditional = false; // is gate conditional gate
  uint_t conditional_reg;   // (opt) the (single) register location to look up for conditional
  RegComparison bfunc;      // (opt) boolean function relation

  // DEPRECIATED: old style conditionals (will be removed when Terra supports new style)
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

//------------------------------------------------------------------------------
// Error Checking
//------------------------------------------------------------------------------

// Raise an exception if name string is empty
inline void check_empty_name(const Op &op) {
  if (op.name.empty())
    throw std::invalid_argument("Invalid qobj instruction (\"name\" is empty.");
}

// Raise an exception if qubits list is empty
inline void check_empty_qubits(const Op &op) {
  if (op.qubits.empty())
    throw std::invalid_argument("Invalid qobj \"" + op.name + 
                                "\" instruction (\"qubits\" are empty");
}

// Raise an exception if qubits list contains duplications
inline void check_duplicate_qubits(const Op &op) {
  auto cpy = op.qubits;
  std::unique(cpy.begin(), cpy.end());
  if (cpy != op.qubits)
    throw std::invalid_argument("Invalid qobj \"" + op.name + 
                                "\" instruction (\"qubits\" are not unique)");
}

//------------------------------------------------------------------------------
// Generator functions
//------------------------------------------------------------------------------

inline Op make_mat(const reg_t &qubits, const cmatrix_t &mat, std::string label = "") {
  Op op;
  op.type = OpType::matrix;
  op.name = "mat";
  op.qubits = qubits;
  op.mats = {mat};
  if (label != "")
    op.string_params = {label};
  return op;
}

template <typename T> // real or complex numeric type
inline Op make_u1(uint_t qubit, T lam) {
  Op op;
  op.type = OpType::gate;
  op.name = "u1";
  op.qubits = {qubit};
  op.params = {lam};
  return op;
}

template <typename T> // real or complex numeric type
inline Op make_u2(uint_t qubit, T phi, T lam) {
  Op op;
  op.type = OpType::gate;
  op.name = "u2";
  op.qubits = {qubit};
  op.params = {phi, lam};
  return op;
}

template <typename T> // real or complex numeric type
inline Op make_u3(uint_t qubit, T theta, T phi, T lam) {
  Op op;
  op.type = OpType::gate;
  op.name = "u3";
  op.qubits = {qubit};
  op.params = {theta, phi, lam};
  return op;
}

inline Op make_reset(const reg_t & qubits, uint_t state = 0) {
  Op op;
  op.type = OpType::reset;
  op.name = "reset";
  op.qubits = qubits;
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
inline void from_json(const json_t &js, Op &op) {op = json_to_op(js);}

// Standard operations
Op json_to_op_gate(const json_t &js);
Op json_to_op_barrier(const json_t &js);
Op json_to_op_measure(const json_t &js);
Op json_to_op_reset(const json_t &js);
Op json_to_op_bfunc(const json_t &js);

// Snapshots
Op json_to_op_snapshot(const json_t &js);
Op json_to_op_snapshot_default(const json_t &js);
Op json_to_op_snapshot_matrix(const json_t &js);
Op json_to_op_snapshot_pauli(const json_t &js);

// Matrices
Op json_to_op_unitary(const json_t &js);
Op json_to_op_kraus(const json_t &js);
Op json_to_op_noise_switch(const json_t &js);

// Classical bits
Op json_to_op_roerror(const json_t &js);



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
  // Arbitrary matrix gates
  if (name == "unitary")
    return json_to_op_unitary(js);
  // Snapshot
  if (name == "snapshot")
    return json_to_op_snapshot(js);
  // Bit functions
  if (name == "bfunc")
    return json_to_op_bfunc(js);
  // Noise functions
  if (name == "noise_switch")
    return json_to_op_noise_switch(js);
  if (name == "kraus")
    return json_to_op_kraus(js);
  if (name == "roerror")
    return json_to_op_roerror(js);
  // Default assume gate
  return json_to_op_gate(js);
}


//------------------------------------------------------------------------------
// Implementation: Gates, measure, reset deserialization
//------------------------------------------------------------------------------

Op json_to_op_gate(const json_t &js) {
  Op op;
  op.type = OpType::gate;
  JSON::get_value(op.name, "name", js);
  JSON::get_value(op.qubits, "qubits", js);
  JSON::get_value(op.params, "params", js);

  // Check conditional
  if (JSON::check_key("conditional", js)) {
    if (js["conditional"].is_number()) {
      // New style conditional
      op.conditional_reg = js["conditional"];
      op.conditional = true;
    } else {
      // DEPRECIATED: old style conditional
      JSON::get_value(op.old_conditional_mask, "mask", js["conditional"]);
      JSON::get_value(op.old_conditional_val, "val", js["conditional"]);
      op.old_conditional = true;
    }
  }

  // Validation
  check_empty_name(op);
  check_empty_qubits(op);
  check_duplicate_qubits(op);

  return op;
}


Op json_to_op_barrier(const json_t &js) {
  Op op;
  op.type = OpType::barrier;
  op.name = "barrier";
  JSON::get_value(op.qubits, "qubits", js);
  return op;
}


Op json_to_op_measure(const json_t &js) {
  Op op;
  op.type = OpType::measure;
  op.name = "measure";
  JSON::get_value(op.qubits, "qubits", js);
  JSON::get_value(op.memory, "memory", js);
  JSON::get_value(op.registers, "register", js);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (op.memory.empty() == false && op.memory.size() != op.qubits.size()) {
    throw std::invalid_argument("Invalid measure operation: \"memory\" and \"qubits\" are different lengths.");
  }
  if (op.registers.empty() == false && op.registers.size() != op.qubits.size()) {
    throw std::invalid_argument("Invalid measure operation: \"register\" and \"qubits\" are different lengths.");
  }
  return op;
}


Op json_to_op_reset(const json_t &js) {
  Op op;
  op.type = OpType::reset;
  op.name = "reset";
  JSON::get_value(op.qubits, "qubits", js);

  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
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
  JSON::get_value(op.memory, "memory", js);
  JSON::get_value(op.registers, "register", js);
  
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
  JSON::get_value(op.probs, "probabilities", js);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Matrix and Kraus deserialization
//------------------------------------------------------------------------------

Op json_to_op_unitary(const json_t &js) {
  Op op;
  op.type = OpType::matrix;
  op.name = "mat";
  JSON::get_value(op.qubits, "qubits", js);
  cmatrix_t mat;
  JSON::get_value(mat, "params", js);
  // Validation
  check_empty_qubits(op);
  check_duplicate_qubits(op);
  if (!Utils::is_unitary(mat, 1e-10)) {
    throw std::invalid_argument("\"mat\" matrix is not unitary.");
  }
  op.mats.push_back(mat);
  // Check for a label
  std::string label;
  JSON::get_value(label, "label", js);
  op.string_params.push_back(label);
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
  return op;
}


Op json_to_op_noise_switch(const json_t &js) {
  Op op;
  op.type = OpType::noise_switch;
  op.name = "noise_switch";
  JSON::get_value(op.params, "params", js);
  return op;
}

//------------------------------------------------------------------------------
// Implementation: Snapshot deserialization
//------------------------------------------------------------------------------

Op json_to_op_snapshot(const json_t &js) {
  std::string type;
  JSON::get_value(type, "type", js);
  if (type == "expectation_value_pauli" ||
      type == "expectation_value_pauli_with_variance")
    return json_to_op_snapshot_pauli(js);
  if (type == "expectation_value_matrix" ||
      type == "expectation_value_matrix_with_variance")
    return json_to_op_snapshot_matrix(js);
  // Default snapshot: has "type", "label", "qubits"
  return json_to_op_snapshot_default(js);
}


Op json_to_op_snapshot_default(const json_t &js) {
  Op op;
  op.type = OpType::snapshot;
  JSON::get_value(op.name, "type", js);
  // If missing use "default" for label
  op.string_params.push_back("default");
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
        op.params_expval_pauli.push_back(std::make_pair(coeff, pauli));
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
        Op::matrix_component_t param;
        for (const auto &subcomp : comp[1]) {
          if (!subcomp.is_array() || subcomp.size() != 2) {
            throw std::invalid_argument("Invalid matrix expval snapshot (param component " + 
                                        comp.dump() + " invalid).");
          }
          reg_t comp_qubits = subcomp[0];
          cmatrix_t comp_matrix = subcomp[1];
          // Check qubits are ok
          std::unordered_set<uint_t> unique = {comp_qubits.begin(), comp_qubits.end()};
          if (unique.size() != comp_qubits.size()) {
            throw std::invalid_argument("Invalid matrix expval snapshot (param component " + 
                                        comp.dump() + " invalid).");
          }
          mats.push_back(std::make_pair(comp_qubits, comp_matrix));
        }
        op.params_expval_matrix.push_back(std::make_pair(coeff, mats));
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
