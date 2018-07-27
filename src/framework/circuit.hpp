/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    circuit.hpp
 * @brief   Qiskit-Aer Circuit Class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_framework_circuit_hpp_
#define _aer_framework_circuit_hpp_

#include <algorithm>  // for std::copy
#include <random>
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

  std::vector<Op> ops;      // circuit operations
  uint_t num_qubits = 0;    // maximum number of qubits needed for ops
  uint_t num_memory = 0;    // maxmimum number of memory clbits needed for ops
  uint_t num_registers = 0; // maxmimum number of registers clbits needed for ops

  uint_t shots = 1;
  uint_t seed;

  // Optional data members from QOBJ
  json_t header;
  json_t config;

  // Constructor
  // The constructor automatically calculates the num_qubits, num_memory, num_registers
  // parameters by scaning the input list of ops.
  Circuit() {seed = std::random_device()();};
  inline Circuit(const std::vector<Op> &_ops) : Circuit() {ops = _ops; set_sizes();};

  // Construct a circuit from JSON
  Circuit(const json_t &js);

  // Automatically set the number of qubits, memory, registers based on ops
  void set_sizes();
};

// Json conversion function
inline void from_json(const json_t &js, Circuit &circ) {circ = Circuit(js);};


//============================================================================
// Implementation: Circuit methods
//============================================================================

void Circuit::set_sizes() {
  // Check maximum qubit, memory, register size
  for (const auto &op: ops) {
    if (!op.qubits.empty()) {
      auto max = std::max_element(std::begin(op.qubits), std::end(op.qubits));
      num_qubits = std::max(num_qubits, 1UL + *max);
    }
    if (!op.memory.empty()) {
      auto max = std::max_element(std::begin(op.memory), std::end(op.memory));
      num_memory = std::max(num_memory, 1UL + *max);
    }
    if (!op.registers.empty()) {
      auto max = std::max_element(std::begin(op.registers), std::end(op.registers));
      num_registers = std::max(num_registers, 1UL + *max);
    }
  }
}


Circuit::Circuit(const json_t &js) : Circuit() {

  // Get header and config
  JSON::get_value(header, "header", js);
  JSON::get_value(config, "config", js);
  JSON::get_value(shots, "shots", config);
  JSON::get_value(seed, "seed", config);

  // Get operations
  if (JSON::check_key("instructions", js) == false) {
    throw std::invalid_argument("Invalid Qobj experiment: no \"instructions\" field.");
  }
  ops.clear(); // remove any current operations
  const json_t &jops = js["instructions"];
  for (auto it = jops.cbegin(); it != jops.cend(); ++it) {
    ops.emplace_back(json_to_op(*it));
  }
  set_sizes();
}

//------------------------------------------------------------------------------
} // end namespace QISKIT
#endif
