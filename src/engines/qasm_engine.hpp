/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    engine.hpp
 * @brief   Engine base class for qiskit-aer simulator
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

// TODO: Add support for bfunc operation
// TODO: Add support for readout error operation
// TODO: Add total_shots_ variable for recording shots

#ifndef _aer_engines_qasm_engine_hpp_
#define _aer_engines_qasm_engine_hpp_


#include "engines/snapshot_engine.hpp"
#include "framework/utils.hpp"
#include "framework/snapshot.hpp"

namespace AER {
namespace Engines {


//============================================================================
// QASM Engine class for Qiskit-Aer
//============================================================================

// This engine returns counts for state classes that support measurement.
// It is designed to be used to simulate the output of real hardware.
// Note that it also supports the snapshot operation for debugging purposes
// But the snapshots will return a list of snapshots for each shot of the simulation.

template <class state_t>
class QasmEngine : public virtual SnapshotEngine<state_t> {

public:

  using State = Base::State<state_t>;
  
  //----------------------------------------------------------------
  // Base class overrides
  //----------------------------------------------------------------

  virtual void initialize(State *state,
                          const Circuit &circ) override;

  virtual void compute_result(State *state) override;

  // Apply a sequence of operations to the state
  // TODO: modify this to handle conditional operations, and measurement
  virtual void apply_op(State *state, const Operations::Op &op) override;

  // Erase output data from engine
  virtual void clear() override;

  // Serialize engine data to JSON
  virtual json_t json() const override;

  // Load Engine config settings
  // Config settings for this engine are of the form:
  // { "counts": true, "hex_output": true, "memory": false, "register": false};
  // with default values true, false, false
  virtual void load_config(const json_t &config) override;

  // Combine engines for accumulating data
  // After combining argument engine should no longer be used
  void combine(QasmEngine<state_t> &eng);

  
  virtual std::set<std::string>
  validate_circuit(State *state, const Circuit &circ) override;

protected:

  // Checks if an Op should be applied, returns true if either: 
  // - op is not a conditional Op,
  // - op is a conditional Op and it passes the conditional check
  bool check_conditional(const Operations::Op &op) const;

  // Record the outcome of a measurement in the engine memory and registers
  void store_measure(const reg_t &outcome, const reg_t &memory, const reg_t &registers);

  // initialize creg strings
  void initialize_creg(const Circuit &circ);

  // Record current qubit probabilities for a measurement snapshot
  void snapshot_probabilities(State *state, const Operations::Op &op);

  // Shots
  uint_t total_shots_;

  // Classical registers
  std::string creg_memory_;
  std::string creg_registers_;

  // Output Data
  std::map<std::string, uint_t> counts_; // histogram of memory counts over shots
  std::vector<std::string> memory_;      // memory state for each shot as hex string
  std::vector<std::string> registers_;   // register state for each shot as hex string

  // Flags
  bool return_hex_strings_ = true;       // Set to false for bit-string output
  bool return_counts_ = true;            // add counts_ to output data
  bool return_memory_ = false;           // add memory_ to output data
  bool return_registers_ = false;        // add registers_ to output data

  // Probability snapshot
  using SnapshotQubits = std::set<uint_t, std::greater<uint_t>>;
  using SnapshotKey = std::pair<SnapshotQubits, std::string>; // qubits and memory value pair
  using SnapshotVal = std::map<std::string, double>;
  AveragedSnapshot<SnapshotKey, SnapshotVal> snapshots_probs_;
};


//============================================================================
// Implementation: Base engine overrides
//============================================================================

template <class state_t>
void QasmEngine<state_t>::initialize(State *state, const Circuit &circ) {
  initialize_creg(circ);
  SnapshotEngine<state_t>::initialize(state, circ);
}

template <class state_t>
std::set<std::string>
QasmEngine<state_t>::validate_circuit(State *state, const Circuit &circ) {
  auto allowed_ops = state->allowed_ops();
  allowed_ops.insert({"snapshot_state", "snapshot_probs", "measure"});
  return circ.invalid_ops(allowed_ops);
};

template <class state_t>
void QasmEngine<state_t>::apply_op(State *state, const Operations::Op &op) {
  if (op.name == "snapshot_probs") {
    snapshot_probabilities(state, op);
  } else if (check_conditional(op)) { // check if op passes conditional
    if (op.name == "measure") { // check if op is measurement
      reg_t outcome = state->apply_measure(op.qubits);
      store_measure(outcome, op.memory, op.registers);
    } else {
      SnapshotEngine<state_t>::apply_op(state, op);  // Apply operation as usual
    }
  }
}


template <class state_t>
void QasmEngine<state_t>::compute_result(State *state) {
  // Parent class
  SnapshotEngine<state_t>::compute_result(state);
  // Memory bits value
  if (!creg_memory_.empty()) {
    std::string memory_hex = Utils::bin2hex(creg_memory_);
    if (return_counts_)
      counts_[memory_hex] += 1;
    if (return_memory_)
      memory_.push_back(memory_hex);
  }
  // Register bits value
  if (!creg_registers_.empty() && return_registers_) {
      registers_.push_back(Utils::bin2hex(creg_registers_));
  }
}


template <class state_t>
void QasmEngine<state_t>::clear() {
  SnapshotEngine<state_t>::clear(); // parent class
  counts_.clear();
  memory_.clear();
  registers_.clear();
  creg_memory_.clear();
  creg_registers_.clear();
}


template <class state_t>
void QasmEngine<state_t>::combine(QasmEngine<state_t> &eng) {
  // Move memory_
  std::move(eng.memory_.begin(), eng.memory_.end(),
            std::back_inserter(memory_));
  // Move registers_
  std::move(eng.registers_.begin(), eng.registers_.end(),
            std::back_inserter(registers_));

  // Combine counts
  for (auto pair : eng.counts_) {
    counts_[pair.first] += pair.second;
  }
  eng.counts_.clear(); // delete copied count data

  SnapshotEngine<state_t>::combine(eng); // parent class
}


template <class state_t>
json_t QasmEngine<state_t>::json() const {
  json_t tmp = SnapshotEngine<state_t>::json(); // parent class;
  if (return_counts_ && counts_.empty() == false)
    tmp["counts"] = counts_;
  if (return_memory_ && memory_.empty() == false)
    tmp["memory"] = memory_;
  if (return_registers_ && registers_.empty() == false)
    tmp["register"] = registers_;
  // Add snapshot data 
  auto slots = snapshots_probs_.slots();
  for (const auto &slot : slots) {
    json_t probs_js;
    std::set<SnapshotKey> keys = snapshots_probs_.slot_data_keys(slot);
    for (const auto &key : keys) {
      json_t datum;
      datum["qubits"] = key.first;
      datum["memory"] = key.second;
      datum["values"] = snapshots_probs_.averaged_data(slot, key);
      probs_js.push_back(datum);
    }
    tmp["snapshots"][slot]["probabilities"] = probs_js;
  }
  return tmp;
}


template <class state_t>
void QasmEngine<state_t>::load_config(const json_t &js) {
  SnapshotEngine<state_t>::load_config(js); // parent class
  JSON::get_value(return_counts_, "counts", js);
  JSON::get_value(return_memory_, "memory", js);
  JSON::get_value(return_registers_, "register", js);
  JSON::get_value(return_hex_strings_, "hex_output", js);
}


//============================================================================
// Implementation: Measurement and conditionals
//============================================================================

template <class state_t>
void QasmEngine<state_t>::initialize_creg(const Circuit &circ) {
  // Clear and resize registers
  creg_memory_ = std::string(circ.num_memory, '0');
  creg_registers_ = std::string(circ.num_registers, '0');
}


template <class state_t>
bool QasmEngine<state_t>::check_conditional(const Operations::Op &op) const {
  // Check if op is not conditional
  if (op.conditional == false)
    return true;
  // Check if specified register bit is 1
  return (creg_registers_[creg_registers_.size() - op.conditional_reg] == '1');
}


template <class state_t>
void QasmEngine<state_t>::store_measure(const reg_t &outcome,
                                        const reg_t &memory,
                                        const reg_t &registers) {
  // Assumes memory and registers are either empty or same size as outcome!
  bool use_mem = !memory.empty();
  bool use_reg = !registers.empty();
  for (size_t j=0; j < outcome.size(); j++) {
    if (use_mem) {
      const size_t pos = creg_memory_.size() - memory[j] - 1; // least significant bit first ordering
      creg_memory_[pos] = std::to_string(outcome[j])[0]; // int->string->char
    }
    if (use_reg) {
      const size_t pos = creg_registers_.size() - memory[j] - 1; // least significant bit first ordering
      creg_registers_[pos] = std::to_string(outcome[j])[0];  // int->string->char
    }
  }
}


template <class state_t>
void QasmEngine<state_t>::snapshot_probabilities(State *state, const Operations::Op &op) {
  SnapshotQubits qubits(op.qubits.begin(), op.qubits.end()); // convert qubits to set
  std::string memory_hex = Utils::bin2hex(creg_memory_); // convert memory to hex string
  SnapshotVal probs = Utils::vec2ket(state->measure_probs(qubits), 1e-15, 2); // get probabilities
  snapshots_probs_.add_data(op.slot, std::make_pair(qubits, memory_hex), probs);
}


//------------------------------------------------------------------------------
} // end namespace Engines
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
