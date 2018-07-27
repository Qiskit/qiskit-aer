/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    operations.hpp
 * @brief   Simualator operations
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_framework_operations_hpp_
#define _aer_framework_operations_hpp_

#include <algorithm>  // for std::copy
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/types.hpp"
#include "framework/json.hpp"

namespace AER {

//------------------------------------------------------------------------------
// Op Class
//------------------------------------------------------------------------------

struct Op {
  std::string name;         // operation name
  bool conditional = false; // is gate conditional gate
  reg_t qubits;             // (opt) qubits operation acts on

  reg_t memory;             // (opt) register operation it acts on (measure)
  reg_t registers;          // (opt) register locations it acts on (measure, conditional)
  uint_t conditional_reg;   // (opt) the (single) register location to look up for conditional
  
  // Data structures for parameters
  std::vector<std::string> params_s;  // string params
  std::vector<double> params_d;       // double params
  std::vector<complex_t> params_z;    // complex double params
  std::vector<cvector_t> params_v;    // complex vector params
  std::vector<cmatrix_t> params_m;    // complex matrix params

};


//------------------------------------------------------------------------------
// Op utility functions
//------------------------------------------------------------------------------

std::pair<uint_t, uint_t> minmax_qubits(const std::vector<Op> &ops);
std::pair<uint_t, uint_t> minmax_memory(const std::vector<Op> &ops);
std::pair<uint_t, uint_t> minmax_registers(const std::vector<Op> &ops);

bool has_conditional(const std::vector<Op> &ops);
bool has_specific_op(const std::vector<Op> &ops, std::string name);

//------------------------------------------------------------------------------
// Op Generators
//------------------------------------------------------------------------------

Op make_op_gate(std::string name, reg_t &qubits, const std::vector<double> &params);
Op make_op_cond(std::string name, const reg_t &qubits, const std::vector<double> &params,
                uint_t conditional_reg);
Op make_op_measure(const reg_t &qubits, const reg_t &memory, const reg_t &registers);
Op make_op_reset(const reg_t &qubits, const std::vector<int_t> &state);
Op make_op_bfunc(std::string rel, std::string mask, std::string val,
                 const reg_t &registers, const reg_t &memory);
Op make_op_snapshot(std::string slot, const std::vector<std::string> &types);
Op make_op_matrix(const reg_t &qubits, const cmatrix_t &mat);
Op make_op_obs_mat(const reg_t &qubits, const std::vector<cmatrix_t> &mats);
Op make_op_obs_dmat(const reg_t &qubits, const std::vector<cvector_t> &diags);
Op make_op_obs_vec(const reg_t &qubits, const std::vector<cvector_t> &vecs);
Op make_op_obs_pauli(const reg_t &qubits, const std::vector<std::string> &paulis,
                     const cvector_t &coeffs);
Op make_op_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &mats);

//------------------------------------------------------------------------------
// JSON conversion
//------------------------------------------------------------------------------

// Main JSON deserialization functions
Op json_to_op(const json_t &js); // Patial TODO
inline void from_json(const json_t &js, Op &op) {op = json_to_op(js);};

// Helper deserialization functions
Op json_to_op_gate(const json_t &js);
Op json_to_op_measure(const json_t &js);
Op json_to_op_reset(const json_t &js); // TODO
Op json_to_op_bfunc(const json_t &js); // TODO
Op json_to_op_snapshot(const json_t &js); // TODO
Op json_to_op_matrix(const json_t &js); // TODO
Op json_to_op_obs_mat(const json_t &js); // TODO
Op json_to_op_obs_dmat(const json_t &js); // TODO
Op json_to_op_obs_vec(const json_t &js); // TODO
Op json_to_op_obs_pauli(const json_t &js); // TODO

// Main JSON serialization functions
json_t json_from_op(const Op &op); // Partial TODO
inline void to_json(json_t &js, const Op &op) {js = json_from_op(op);};

// Helper serialization functions
json_t json_from_op_gate(const Op& op); 
json_t json_from_op_measure(const Op& op);
json_t json_from_op_reset(const Op& op); // TODO
json_t json_from_op_bfunc(const Op& op); // TODO
json_t json_from_op_snapshot(const Op& op); // TODO
json_t json_from_op_matrix(const Op& op); // TODO
json_t json_from_op_obs_mat(const Op& op); // TODO
json_t json_from_op_obs_dmat(const Op& op); // TODO
json_t json_from_op_obs_vec(const Op& op); // TODO
json_t json_from_op_obs_pauli(const Op& op); // TODO


//------------------------------------------------------------------------------
// Implementation: utility functions
//------------------------------------------------------------------------------

bool has_conditional(const std::vector<Op> &ops) {
  for (const auto &op: ops) {
    if (op.conditional)
      return true;
  }
  return false;
}


bool has_specific_op(const std::vector<Op> &ops, std::string name) {
  for (const auto &op: ops) {
    if (op.name == name)
      return true;
  }
  return false;
}


std::pair<uint_t, uint_t> 
minmax_qubits(const std::vector<Op> &ops) {
  uint_t min = 0;
  uint_t max = 0;
  for (const auto &op: ops) {
    if (op.qubits.empty() == false) {
      auto minmax = std::minmax_element(std::begin(op.qubits),
                                        std::end(op.qubits));
      min = std::min(min, *minmax.first);
      max = std::max(max, *minmax.second);
    }
  }
  return std::make_pair(min, max);
}


std::pair<uint_t, uint_t> 
minmax_memory(const std::vector<Op> &ops) {
  uint_t min = 0;
  uint_t max = 0;
  for (const auto &op: ops) {
    if (op.memory.empty() == false) {
      auto minmax = std::minmax_element(std::begin(op.memory),
                                        std::end(op.memory));
      min = std::min(min, *minmax.first);
      max = std::max(max, *minmax.second);
    }
  }
  return std::make_pair(min, max);
}


std::pair<uint_t, uint_t> 
minmax_registers(const std::vector<Op> &ops) {
  uint_t min = 0;
  uint_t max = 0;
  for (const auto &op: ops) {
    if (op.registers.empty() == false) {
      auto minmax = std::minmax_element(std::begin(op.registers),
                                        std::end(op.registers));
      min = std::min(min, *minmax.first);
      max = std::max(max, *minmax.second);
    }
  }
  return std::make_pair(min, max);
}


//------------------------------------------------------------------------------
// Implementation: Op Generators
//------------------------------------------------------------------------------

Op make_op_gate(std::string name, 
                const reg_t &qubits,
                const std::vector<double> &params) {
  Op op;
  op.name = name;
  op.qubits = qubits;
  op.params_d = params;
  return op;
}


Op make_op_cond(std::string name, 
                const reg_t &qubits,
                const std::vector<double> &params,
                uint_t conditional_reg) {
  Op op = make_op_gate(name, qubits, params);
  op.conditional = true;
  op.conditional_reg = conditional_reg;
  return op;
}


Op make_op_measure(const reg_t &qubits,
                   const reg_t &memory,
                   const reg_t &registers) {
  Op op;
  op.name = "measure";
  op.qubits = qubits;
  op.memory = memory;
  op.registers = registers;
  // Check memory and registers are correct length
  if (memory.empty() == false && memory.size() != qubits.size()) {
    throw std::invalid_argument("memory bits are incorrect length");
  }
  if (registers.empty() == false && registers.size() != qubits.size()) {
    throw std::invalid_argument("register bits are incorrect length");
  }
  return op;
}


Op make_op_reset(const reg_t &qubits,
                 const std::vector<uint_t> &state) {
  Op op;
  op.name = "reset";
  op.qubits = qubits;
  std::copy(state.cbegin(), state.cend(), op.params_d.begin());
  // Check state is correct length
  if (state.size() != qubits.size()) {
    throw std::invalid_argument("reset state is incorrect length");
  }
  return op;
}


Op make_op_bfunc(std::string rel, std::string mask, std::string val,
                 const reg_t &registers,
                 const reg_t &memory) {
  Op op;
  op.name = "bfunc";
  op.params_s = {rel, mask, val};
  op.registers = registers;
  op.memory = memory;
  return op;
}


Op make_op_snapshot(std::string slot, const std::vector<std::string> &types) {
  Op op;
  op.name = "#snapshot";
  op.params_s = {slot};
  if (types.empty())
    op.params_s.push_back("default");
  else
    std::copy(types.cbegin(), types.cend(), back_inserter(op.params_s));
  return op;
}


Op make_op_matrix(const reg_t &qubits, const cmatrix_t &mat) {
  Op op;
  op.name = "matrix";
  op.qubits = qubits;
  op.params_m = {mat};
  return op;
}


Op make_op_obs_mat(const reg_t &qubits, const std::vector<cmatrix_t> &mats) {
  Op op;
  op.name = "obs_mat";
  op.qubits = qubits;
  op.params_m = mats;
  return op;
}


Op make_op_obs_dmat(const reg_t &qubits, const std::vector<cvector_t> &diags) {
  Op op;
  op.name = "obs_dmat";
  op.qubits = qubits;
  op.params_v = diags;
  return op;
}


Op make_op_obs_vec(const reg_t &qubits, const std::vector<cvector_t> &vecs) {
  Op op;
  op.name = "obs_vec";
  op.qubits = qubits;
  op.params_v = vecs;
  return op;
}


Op make_op_obs_pauli(const reg_t &qubits, const std::vector<std::string> &paulis, 
                     const cvector_t &coeffs) {
  Op op;
  op.name = "obs_pauli";
  op.qubits = qubits;
  op.params_s = paulis;

  // Check paulis are correct length
  for (const auto &p : paulis) {
    if (p.size() != qubits.size()) {
      throw std::invalid_argument("Pauli observable is incorrect length");
    }
  }
  if (coeffs.empty() == false) {
    op.params_z = coeffs;
    // Check coeffs are correct length
    if (coeffs.size() != paulis.size()) {
      throw std::invalid_argument("Pauli coefficient is incorrect length");
    }
  }
  return op;
}


Op make_op_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &mats) {
  Op op;
  op.name = "matrix";
  op.qubits = qubits;
  op.params_m = mats;
  return op;
}


//------------------------------------------------------------------------------
// Implementation: JSON deserialization
//------------------------------------------------------------------------------

Op json_to_op(const json_t &js) {
  // load operation identifier
  std::string name;
  JSON::get_value(name, "name", js);
  if (name.empty()) {
    throw std::invalid_argument("Invalid gate operation: \"name\" is empty.");
  }
  if (name == "measure")
    return json_to_op_measure(js);
  if (name == "reset")
    return json_to_op_reset(js);
  /* TODO: the following aren't implemented yet!
  if (name == "bfunc")
    return json_to_op_bfunc(js);
  if (name == "#snapshot")
    return json_to_op_snapshot(js);
  if (name == "matrix")
    return json_to_op_matrix(js);
  if (name == "obs_mat")
    return json_to_op_obs_mat(js);
  if (name == "obs_dmat")
    return json_to_op_obs_mat(js);
  if (name == "obs_vec")
    return json_to_op_obs_mat(js);
  if (name == "obs_pauli")
    return json_to_op_obs_mat(js);
  */
  // Default parse as gate
  return json_to_op_gate(js);
}


Op json_to_op_gate(const json_t &js) {
  Op op;
  // Load name identifier
  JSON::get_value(op.name, "name", js);
  if (op.name.empty()) {
    throw std::invalid_argument("Invalid gate operation: \"name\" are empty.");
  }
  // Load qubits
  JSON::get_value(op.qubits, "qubits", js);
  if (op.qubits.empty()) {
    throw std::invalid_argument("Invalid gate operation: \"qubits\" are empty.");
  }
  // Load double params (if present)
  JSON::get_value(op.params_d, "params", js);
  return op;
}


Op json_to_op_measure(const json_t &js) {
  Op op;
  op.name = "measure";
  // Load qubits
  JSON::get_value(op.qubits, "qubits", js);
  if (op.qubits.empty()) {
    throw std::invalid_argument("Invalid measure operation: \"qubits\" are empty.");
  }
  // Load memory (if present)
  JSON::get_value(op.memory, "memory", js);
  if (op.memory.empty() == false && op.memory.size() != op.qubits.size()) {
    throw std::invalid_argument("Invalid measure operation: \"memory\" and \"qubits\" are different lengths.");
  }
  // Load memory (if present)
  JSON::get_value(op.registers, "register", js);
  if (op.registers.empty() == false && op.registers.size() != op.registers.size()) {
    throw std::invalid_argument("Invalid measure operation: \"register\" and \"qubits\" are different lengths.");
  }
  return op;
}

Op json_to_op_reset(const json_t &js) {
  Op op;
  op.name = "reset";
  // Load qubits
  JSON::get_value(op.qubits, "qubits", js);
  if (op.qubits.empty()) {
    throw std::invalid_argument("Invalid reset operation: \"qubits\" are empty.");
  }
  // Load double params for reset state (if present)
  JSON::get_value(op.params_d, "params", js);
  if (op.params_d.empty()) {
    // If not present default reset to all zero state
    op.params_d = rvector_t(op.qubits.size(), 0.);
  }
  if (op.params_d.size() != op.qubits.size()) {
    throw std::invalid_argument("Invalid reset operation: \"params\" and \"qubits\" are different lengths.");
  }
  return op;
}

//------------------------------------------------------------------------------
// Implementation: JSON serialization
//------------------------------------------------------------------------------

json_t json_from_op(const Op &op) {
  if (op.name == "measure")
    return json_from_op_measure(op);
  if (op.name == "reset")
    return json_from_op_reset(op);
  /* TODO: the following aren't implemented yet!
  if (op.name == "bfunc")
    return json_from_op_bfunc(op);
  if (op.name == "#snapshot")
    return json_from_op_snapshot(op);
  if (op.name == "matrix")
    return json_from_op_matrix(op);
  if (op.name == "obs_mat")
    return json_from_op_obs_mat(op);
  if (op.name == "obs_dmat")
    return json_from_op_obs_mat(op);
  if (op.name == "obs_vec")
    return json_from_op_obs_mat(op);
  if (op.name == "obs_pauli")
    return json_from_op_obs_mat(op);
  */
  // Default parse as gate
  return json_from_op_gate(op);
}


json_t json_from_op_gate(const Op &op) {
  json_t js;
  js["name"] = op.name;
  js["qubits"] = op.qubits;
  if (op.params_d.empty() == false) {
    js["params"] = op.params_d;
  }
  if (op.conditional == true) {
    js["conditional"] = op.conditional_reg;
  }
  return js;
}


json_t json_from_op_measure(const Op &op) {
  json_t js;
  js["name"] = "measure";
  js["qubits"] = op.qubits;
  if (op.memory.empty() == false) {
    js["memory"] = op.memory;
  }
  if (op.registers.empty() == false) {
    js["register"] = op.registers;
  }
  return js;
}


json_t json_from_op_reset(const Op &op) {
  json_t js;
  js["name"] = "reset";
  js["qubits"] = op.qubits;
  if (op.params_d.empty() == false) {
    for (const auto p: op.params_d)
    js["params"].push_back(uint_t(p));
  }
  return js;
}

// TODO: the rest

//------------------------------------------------------------------------------
} // end namespace QISKIT
#endif
