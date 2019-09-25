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

#ifndef _aer_framework_circuit_hpp_
#define _aer_framework_circuit_hpp_

#include <random>

#include "framework/operations.hpp"
#include "framework/json.hpp"

namespace AER {

//============================================================================
// Circuit class for Qiskit-Aer
//============================================================================

// A circuit is a list of Ops along with a specification of maximum needed
// qubits, memory bits, and register bits for the input operators.
class Circuit {
public:
  using Op = Operations::Op;
  using OpType = Operations::OpType;
  std::vector<Op> ops;      // circuit operations
  uint_t num_qubits = 0;    // maximum number of qubits needed for ops
  uint_t num_memory = 0;    // maximum number of memory clbits needed for ops
  uint_t num_registers = 0; // maximum number of registers clbits needed for ops
  uint_t shots = 1;
  uint_t seed;

  // Measurement sampling
  bool measure_sampling_flag = false;

  // Optional data members from QOBJ
  json_t header;

  // Constructor
  // The constructor automatically calculates the num_qubits, num_memory, num_registers
  // parameters by scanning the input list of ops.
  Circuit() {set_random_seed();}
  Circuit(const std::vector<Op> &_ops);

  // Construct a circuit from JSON
  Circuit(const json_t &circ);
  Circuit(const json_t &circ, const json_t &qobj_config);

  // Automatically set the number of qubits, memory, registers based on ops
  void set_sizes();

  // Set the circuit rng seed to a fixed value
  inline void set_seed(uint_t s) {seed = s;}

  // Set the circuit rng seed to random value
  inline void set_random_seed() {seed = std::random_device()();}

  // Return the opset for the circuit
  inline const Operations::OpSet& opset() const {return opset_;}

  // Check if any circuit ops are conditional ops
  bool has_conditional() const;

  // Check if circuit contains a specific op
  bool has_op(std::string name) const;

  // return minimum and maximum op.qubit arguments as pair (min, max)
  std::pair<uint_t, uint_t> minmax_qubits() const;

  // return minimum and maximum op.memory arguments as pair (min, max)
  std::pair<uint_t, uint_t> minmax_memory() const;

  // return minimum and maximum op.registers arguments as pair (min, max)
  std::pair<uint_t, uint_t> minmax_registers() const;

private:
  Operations::OpSet opset_;  // Set of operation types contained in circuit
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
    if (!op.memory.empty()) {
      auto max = std::max_element(std::begin(op.memory), std::end(op.memory));
      num_memory = std::max(num_memory, 1UL + *max);
    }
    
  }
}

Circuit::Circuit(const std::vector<Op> &_ops) : Circuit() {
  ops = _ops;
  set_sizes();
  opset_ = Operations::OpSet(ops);
}

Circuit::Circuit(const json_t &circ) : Circuit(circ, json_t()) {}

Circuit::Circuit(const json_t &circ, const json_t &qobj_config) : Circuit() {

  // Get config
  json_t config = qobj_config;
  if (JSON::check_key("config", circ)) {
    for (auto it = circ["config"].cbegin(); it != circ["config"].cend();
         ++it) {
      config[it.key()] = it.value(); // overwrite circuit level config values
    }
  }
  // Load instructions
  if (JSON::check_key("instructions", circ) == false) {
    throw std::invalid_argument("Invalid Qobj experiment: no \"instructions\" field.");
  }
  ops.clear(); // remove any current operations
  const json_t &jops = circ["instructions"];
  for(auto jop: jops){
    ops.emplace_back(Operations::json_to_op(jop));
  }

  // Set optype information
  opset_ = Operations::OpSet(ops);

  // Set minimum sizes from operations
  set_sizes();

  // Load metadata
  JSON::get_value(header, "header", circ);
  JSON::get_value(shots, "shots", config);

  // Check for specified memory slots
  uint_t memory_slots = 0;
  JSON::get_value(memory_slots, "memory_slots", config);
  if (memory_slots < num_memory) {
    throw std::invalid_argument("Invalid Qobj experiment: not enough memory slots.");
  }
  // override memory slot number
  num_memory = memory_slots;

  // Check for specified n_qubits
  if (JSON::check_key("n_qubits", config)) {
    uint_t n_qubits = config["n_qubits"];
    if (n_qubits < num_qubits) {
      throw std::invalid_argument("Invalid Qobj experiment: n_qubits < instruction qubits.");
    }
    // override qubit number
    num_qubits = n_qubits;
  }
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
