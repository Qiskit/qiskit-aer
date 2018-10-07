/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_framework_circuit_hpp_
#define _aer_framework_circuit_hpp_

#include <algorithm>  // for std::copy
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/operations.hpp"
#include "framework/json.hpp"

namespace AER {

//============================================================================
// Circuit class for Qiskit-Aer
//============================================================================

// A circuit is a list of Ops allong with a specification of maximum needed
// qubits, memory bits, and register bits for the input operators.
class Circuit {
public:
  using Op = Operations::Op;
  std::vector<Op> ops;      // circuit operations
  uint_t num_qubits = 0;    // maximum number of qubits needed for ops
  uint_t num_memory = 0;    // maxmimum number of memory clbits needed for ops
  uint_t num_registers = 0; // maxmimum number of registers clbits needed for ops

  uint_t shots = 1;
  uint_t seed;

  // Measurement sampling
  bool measure_sampling_flag = false;

  // Optional data members from QOBJ
  json_t header;
  json_t config;

  // Constructor
  // The constructor automatically calculates the num_qubits, num_memory, num_registers
  // parameters by scaning the input list of ops.
  Circuit() {seed = std::random_device()();};
  inline Circuit(const std::vector<Op> &_ops) : Circuit() {ops = _ops; set_sizes();};

  // Construct a circuit from JSON
  Circuit(const json_t &circ);
  Circuit(const json_t &circ, const json_t &qobj_config);

  // Automatically set the number of qubits, memory, registers based on ops
  void set_sizes();

  // Check if all circuit ops are in an allowed op set
  bool check_ops(const std::set<std::string> &allowed_ops) const;

  // Return a set of all invalid circuit op names
  std::set<std::string>
  invalid_ops(const std::set<std::string> &allowed_ops) const;

  // Check if any circuit ops are conditional ops
  bool has_conditional() const;

  // Check if circuit containts a specific op
  bool has_op(std::string name) const;

  // return minimum and maximum op.qubit arguments as pair (min, max)
  std::pair<uint_t, uint_t> minmax_qubits() const;

  // return minimum and maximum op.memory arguments as pair (min, max)
  std::pair<uint_t, uint_t> minmax_memory() const;

  // return minimum and maximum op.registers arguments as pair (min, max)
  std::pair<uint_t, uint_t> minmax_registers() const;
};

// Json conversion function
inline void from_json(const json_t &js, Circuit &circ) {circ = Circuit(js);}


//============================================================================
// Implementation: Circuit methods
//============================================================================

void Circuit::set_sizes() {
  // Check maximum qubit, and register size
  // Memory size is loaded from qobj config
  for (const auto &op: ops) {
    if (!op.qubits.empty()) {
      auto max = std::max_element(std::begin(op.qubits), std::end(op.qubits));
      num_qubits = std::max(num_qubits, 1UL + *max);
    }
    if (!op.registers.empty()) {
      auto max = std::max_element(std::begin(op.registers), std::end(op.registers));
      num_registers = std::max(num_registers, 1UL + *max);
    }
  }
}


Circuit::Circuit(const json_t &circ) : Circuit(circ, json_t()) {}

Circuit::Circuit(const json_t &circ, const json_t &qobj_config) : Circuit() {

  // Get config
  config = qobj_config;
  if (JSON::check_key("config", circ)) {
    for (auto it = circ["config"].cbegin(); it != circ["config"].cend();
         ++it) {
      config[it.key()] = it.value(); // overwrite circuit level config values
    }
  }
  JSON::get_value(header, "header", circ);
  JSON::get_value(shots, "shots", config);
  JSON::get_value(seed, "seed", config);
  JSON::get_value(num_memory, "memory_slots", config);

  // Get operations
  if (JSON::check_key("instructions", circ) == false) {
    throw std::invalid_argument("Invalid Qobj experiment: no \"instructions\" field.");
  }
  ops.clear(); // remove any current operations
  const json_t &jops = circ["instructions"];
  for (auto it = jops.cbegin(); it != jops.cend(); ++it) {
    ops.emplace_back(Operations::json_to_op(*it));
  }
  set_sizes();
}


std::set<std::string>
Circuit::invalid_ops(const std::set<std::string> &allowed_ops) const {
  std::set<std::string> invalid;
  for (const auto &op : ops) {
    if (allowed_ops.find(op.name) == allowed_ops.end())
      invalid.insert(op.name);
  }
  return invalid;
}


bool Circuit::check_ops(const std::set<std::string> &allowed_ops) const {
  for (const auto &op : ops) {
    if (allowed_ops.find(op.name) == allowed_ops.end())
      return false;
  }
  return true;
}


bool Circuit::has_conditional() const {
   for (const auto &op: ops) {
    if (op.conditional)
      return true;
  }
  return false;
}


bool Circuit::has_op(std::string name) const {
  for (const auto &op: ops) {
    if (op.name == name)
      return true;
  }
  return false;
}


std::pair<uint_t, uint_t> Circuit::minmax_qubits() const{
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


std::pair<uint_t, uint_t> Circuit::minmax_memory() const {
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


std::pair<uint_t, uint_t> Circuit::minmax_registers() const {
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
} // end namespace AER
//------------------------------------------------------------------------------
#endif
