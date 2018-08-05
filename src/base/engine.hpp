/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    engine.hpp
 * @brief   Engine class for qiskit-aer simulator
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_base_engine_hpp_
#define _aer_base_engine_hpp_

#include <stdexcept>
#include <vector>

#include "framework/circuit.hpp"
#include "framework/json.hpp"
#include "framework/snapshot.hpp"
#include "framework/utils.hpp"
#include "base/state.hpp"

namespace AER {
namespace Base {

//============================================================================
// Engine base class for Qiskit-Aer
//============================================================================

template <class state_t>
class Engine {

public:
  Engine() = default;
  virtual ~Engine() = default;

  // Execute a circuit of operations for multiple shots
  // By default this loops over the set number of shots where
  // for each shot the follow is applied:
  //   1. initialize the state based on max qubit number in ops
  //   2. call apply_operations on the state for list of ops
  //   3. call compute_results after ops have been applied
  virtual void execute(State<state_t> *state,
                       const Circuit &circ,
                       uint_t shots);
  
  // Empty engine of stored data
  virtual void clear();

  // Serialize engine data to JSON
  virtual json_t json() const;

  // Load any settings for the State class from a config JSON
  virtual void load_config(const json_t &config);

  // Combine engines for accumulating data
  // Second engine should no longer be used after combining
  // as this function should use move semantics to minimize copying
  // Note: this is not a true virtual method as the argument will
  // be the derived engine type, but it must be implemented by all
  // derived classes.
  void combine(Engine<state_t> &eng);

  // validates a circuit
  // returns true if all ops in circuit are supported by the state and engine
  // otherwise it returns false
  virtual std::set<std::string>
  validate_circuit(State<state_t> *state, const Circuit &circ);

protected:

  //----------------------------------------------------------------
  // Execution helper functions
  //----------------------------------------------------------------

  // Operations that are implemented by the engine instead of the State
  // these are: {"measure", "snapshot_state", "snapshot_probs",
  // "snapshot_pauli", "snapshot_matrix"}
  // TODO: add "bfunc" and "roerror"
  static const std::set<std::string> engine_ops_;

  // Apply an operation to the state
  virtual void apply_op(State<state_t> *state, const Operations::Op &op);

  // Initialize an engine and circuit
  virtual void initialize(State<state_t> *state, const Circuit &circ);

  //----------------------------------------------------------------
  // Measurement
  //----------------------------------------------------------------

  // Add the classical bit memory to the counts dictionary
  virtual void update_counts();

  // Checks if an Op should be applied, returns true if either: 
  // - op is not a conditional Op,
  // - op is a conditional Op and it passes the conditional check
  bool check_conditional(const Operations::Op &op) const;

  // Record the outcome of a measurement in the engine memory and registers
  void store_measure(const reg_t &outcome, const reg_t &memory, const reg_t &registers);

  // Classical registers
  std::string creg_memory_;
  std::string creg_registers_;

  // Output Data
  std::map<std::string, uint_t> counts_; // histogram of memory counts over shots
  std::vector<std::string> memory_;      // memory state for each shot as hex string
  std::vector<std::string> registers_;   // register state for each shot as hex string

  // Measurement config settings
  bool return_hex_strings_ = true;       // Set to false for bit-string output
  bool return_counts_ = true;            // add counts_ to output data
  bool return_memory_ = false;           // add memory_ to output data
  bool return_registers_ = false;        // add registers_ to output data

  //----------------------------------------------------------------
  // Snapshots
  //----------------------------------------------------------------

  // Record current state as a snapshot
  void snapshot_state(State<state_t> *state, const Operations::Op &op);

  // Record current qubit probabilities for a measurement snapshot
  void snapshot_probabilities(State<state_t> *state, const Operations::Op &op);

  // Compute and store snapshot of the Pauli observable op
  void snapshot_observables_pauli(State<state_t> *state, const Operations::Op &op);

  // Compute and store snapshot of the matrix observable op
  void snapshot_observables_matrix(State<state_t> *state, const Operations::Op &op);

  // Snapshot Types
  using SnapshotQubits = std::set<uint_t, std::greater<uint_t>>;
  using SnapshotKey = std::pair<SnapshotQubits, std::string>; // qubits and memory value pair
  using SnapshotStates = Snapshots::Snapshot<std::string, state_t, Snapshots::ShotData>;
  using SnapshotProbs = Snapshots::Snapshot<SnapshotKey, std::map<std::string, double>, Snapshots::AverageData>;
  using SnapshotObs = Snapshots::Snapshot<SnapshotKey, complex_t, Snapshots::AverageData>;

  // Helper function to make the snapshot key
  // inlined because the output type aliasing is a pain
  inline SnapshotKey make_snapshot_key(const Operations::Op &op) {
    SnapshotQubits qubits(op.qubits.begin(), op.qubits.end()); // convert qubits to set
    std::string memory_hex = Utils::bin2hex(creg_memory_); // convert memory to hex string
    return std::make_pair(qubits, memory_hex);
  };

  // Snapshot data structures
  SnapshotStates snapshot_states_;
  SnapshotProbs snapshot_probs_;
  SnapshotObs snapshot_obs_;
  
  // Snapshot config settings
  std::string snapshot_state_label_ = "default"; // label and key for snapshots
  double snapshot_chop_threshold_ = 1e-15;
  bool show_snapshots_ = true;

  // Cache of saved pauli values for given qubits and memory bit values
  // to prevent recomputing for re-used Pauli components in different operators
  // Note: there must be a better way to do this cacheing than map of map of map...
  // But to do so would require sorting the vector of qubits in the obs_pauli ops
  // and also sorting the Pauli string labels so that they match the sorted qubit positions.
  std::map<SnapshotKey, std::map<std::string, double>> pauli_cache_;
  
};

//============================================================================
// Implementations: Execution
//============================================================================

template <class state_t>
const std::set<std::string> Engine<state_t>::engine_ops_ = {
  "measure", "snapshot_state", "snapshot_probs",
  "snapshot_pauli", "snapshot_matrix"
};

template <class state_t>
void Engine<state_t>::execute(State<state_t> *state,
                              const Circuit &circ,
                              uint_t shots) {
  for (size_t ishot = 0; ishot < shots; ++ishot) {
    initialize(state, circ);
    for (const auto &op: circ.ops) {
      apply_op(state, op);
    }
    update_counts();
  }
}


template <class state_t>
void Engine<state_t>::apply_op(State<state_t> *state, const Operations::Op &op) {
  auto it = engine_ops_.find(op.name);
  if (it == engine_ops_.end()) {
    if (check_conditional(op)) // check if op passes conditional
      state->apply_op(op);
  } else {
    if (op.name == "measure") {
      reg_t outcome = state->apply_measure(op.qubits);
      store_measure(outcome, op.memory, op.registers);
    } else if (op.name == "snapshot_probs") {
      snapshot_probabilities(state, op);
    } else if (op.name == "snapshot_pauli") {
      snapshot_observables_pauli(state, op);
    } else if (op.name == "snapshot_matrix") {
      snapshot_observables_matrix(state, op);
    } else if (op.name == "snapshot_state") {
      snapshot_states_.add_data(op.slot, snapshot_state_label_, state->data());
    }
  }
}


template <class state_t>
void Engine<state_t>::initialize(State<state_t> *state, const Circuit &circ) {
  // Initialize state
  state->initialize(circ.num_qubits);
  // Clear and resize registers
  creg_memory_ = std::string(circ.num_memory, '0');
  creg_registers_ = std::string(circ.num_registers, '0');
  // clear pauli snapshot cache at start of each shot
  pauli_cache_.clear(); 
}


//============================================================================
// Implementation: Measurement and conditionals
//============================================================================

template <class state_t>
void Engine<state_t>::update_counts() {
  // State is actually unused for this function?
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
void Engine<state_t>::store_measure(const reg_t &outcome,
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
bool Engine<state_t>::check_conditional(const Operations::Op &op) const {
  // Check if op is not conditional
  if (op.conditional == false)
    return true;
  // Check if specified register bit is 1
  return (creg_registers_[creg_registers_.size() - op.conditional_reg] == '1');
}


//============================================================================
// Implementations: Utilities methods
//============================================================================

template <class state_t>
std::set<std::string>
Engine<state_t>:: validate_circuit(State<state_t> *state, const Circuit &circ) {
  std::clog << "ALLOWED OPS = " << state->allowed_ops() << std::endl; // REMOVE
  return circ.invalid_ops(state->allowed_ops());
}


template <class state_t>
void Engine<state_t>::clear() {
  // Clear snapshots
  snapshot_states_.clear();
  snapshot_probs_.clear();
  snapshot_obs_.clear();
  pauli_cache_.clear();
  // Clear measure and counts
  counts_.clear();
  memory_.clear();
  registers_.clear();
  creg_memory_.clear();
  creg_registers_.clear();
}


template <class state_t>
void Engine<state_t>::combine(Engine<state_t> &eng) {
  // Combine measure
  std::move(eng.memory_.begin(), eng.memory_.end(),
            std::back_inserter(memory_));
  std::move(eng.registers_.begin(), eng.registers_.end(),
            std::back_inserter(registers_));
  // Combine counts
  for (auto pair : eng.counts_) {
    counts_[pair.first] += pair.second;
  }
  eng.counts_.clear(); // delete copied count data

  // Combine snapshots
  snapshot_states_.combine(eng.snapshot_states_);
  snapshot_probs_.combine(eng.snapshot_probs_);
  snapshot_obs_.combine(eng.snapshot_obs_); 
  // Clear other eng in case
  eng.clear();
}


template <class state_t>
void Engine<state_t>::load_config(const json_t &js) {
  JSON::get_value(snapshot_state_label_, "snapshot_label", js);
  JSON::get_value(show_snapshots_, "show_snapshots", js);
  JSON::get_value(return_counts_, "counts", js);
  JSON::get_value(return_memory_, "memory", js);
  JSON::get_value(return_registers_, "register", js);
  JSON::get_value(snapshot_chop_threshold_, "chop_threshold", js);
}


template <class state_t>
json_t Engine<state_t>::json() const {
  json_t tmp;
  // Add Counts and memory
  if (return_counts_ && counts_.empty() == false)
    tmp["counts"] = counts_;
  if (return_memory_ && memory_.empty() == false)
    tmp["memory"] = memory_;
  if (return_registers_ && registers_.empty() == false)
    tmp["register"] = registers_;
  
  // Add probability snapshots
  for (const auto &slot : snapshot_probs_.slots()) {
    json_t probs_js;
    std::set<SnapshotKey> keys = snapshot_probs_.slot_keys(slot);
    for (const auto &key : keys) {
      json_t datum;
      datum["qubits"] = key.first;
      datum["memory"] = key.second;
      datum["values"] = snapshot_probs_.get_data(slot, key).data();
      probs_js.push_back(datum);
    }
    tmp["snapshots"][slot]["probabilities"] = probs_js;
  }

  // Add observables snapshots
  for (const auto &slot : snapshot_obs_.slots()) {
    json_t probs_js;
    auto keys = snapshot_obs_.slot_keys(slot);
    for (const auto &key : keys) {
      json_t datum;
      datum["qubits"] = key.first;
      datum["memory"] = key.second;
      datum["value"] = snapshot_obs_.get_data(slot, key).data();
      probs_js.push_back(datum);
    }
    tmp["snapshots"][slot]["observables"] = probs_js;
  }

  // Add state snapshots
  for (const auto &slot : snapshot_states_.slots())
    for (const auto &key : snapshot_states_.slot_keys(slot)) {
      try {
        tmp["snapshots"][slot][key] = snapshot_states_.get_data(slot, key).data();
      } catch (std::exception &e) {
        // Leave message in output that type conversion failed
        tmp["snapshots"][slot][key] = "Error: Failed to convert state type to JSON";
      }
    }

  // return json
  return tmp;
}

//============================================================================
// Implementations: Snapshots
//============================================================================

template <class state_t>
void Engine<state_t>::snapshot_probabilities(State<state_t> *state, const Operations::Op &op) {
  auto key = make_snapshot_key(op);
  auto probs = Utils::vec2ket(state->measure_probs(key.first),
                              snapshot_chop_threshold_, 2); // get probabilities
  snapshot_probs_.add_data(op.slot, key, probs);
}


template <class state_t>
void Engine<state_t>::snapshot_observables_pauli(State<state_t> *state, const Operations::Op &op) {
  auto key = make_snapshot_key(op);
  auto &cache = pauli_cache_[key]; // pauli cache for current key
  complex_t expval(0., 0.); // complex value
  for (const auto &param : op.params_pauli_obs) {
    auto it = cache.find(param.second); // Check cache for value
    if (it != cache.end()) {
      expval += param.first * (it->second); // found cached result
    } else {
      // compute result and add to cache
      double tmp = state->pauli_observable_value(op.qubits, param.second);
      cache[param.second] = tmp;
      expval += param.first * tmp;
    }
  }
  // add to snapshot
  snapshot_obs_.add_data(op.slot, key, expval);
}


template <class state_t>
void Engine<state_t>::snapshot_observables_matrix(State<state_t> *state,
                                                  const Operations::Op &op) {
  auto key = make_snapshot_key(op);
  snapshot_obs_.add_data(op.slot, key, state->matrix_observable_value(op));
}

//------------------------------------------------------------------------------
} // end namespace Base
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
