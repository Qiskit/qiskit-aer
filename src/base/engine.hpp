/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_base_engine_hpp_
#define _aer_base_engine_hpp_

#include <unordered_map>

#include "framework/circuit.hpp"
#include "framework/snapshot.hpp"
#include "framework/utils.hpp"
#include "base/state.hpp"

namespace AER {
namespace Base {

//============================================================================
// Engine base class for Qiskit-Aer
//============================================================================

// Engine class
class Engine {

public:

  // Execute a circuit of operations for multiple shots
  // By default this loops over the set number of shots where
  // for each shot the follow is applied:
  //   1. initialize the state based on max qubit number in ops
  //   2. call apply_operations on the state for list of ops
  //   3. call compute_results after ops have been applied
  template <class State_t>
  void execute(const Circuit &circ,
               uint_t shots,
               State_t &state);

  // This method performs the same function as 'execute', except
  // that it only simulates a single shot and then generates samples
  // of outcomes at the end. It is only valid if a measure operations
  // in a circuit are at the end, and there are no reset operations.
  template <class State_t>
  void execute_with_measure_sampling(const Circuit &circ,
                                     uint_t shots,
                                     State_t &state);

  // Empty engine of stored data
  void clear();

  // Serialize engine data to JSON
  json_t json() const;

  // Load any settings for the State class from a config JSON
  void load_config(const json_t &config);

  // Combine engines for accumulating data
  // Second engine should no longer be used after combining
  // as this function should use move semantics to minimize copying
  void combine(Engine &eng);

  // validates a circuit
  // returns true if all ops in circuit are supported by the state and engine
  // otherwise it returns false
  template <class State_t>
  std::set<std::string> validate_circuit(const Circuit &circ, State_t &state);

  
protected:

  //----------------------------------------------------------------
  // Execution helper functions
  //----------------------------------------------------------------

  // Apply an operation to the state
  template <class State_t>
  void apply_op(const Operations::Op &op, State_t &state);

  // Initialize an engine and circuit
  template <class State_t>
  void initialize(const Circuit &circ, State_t &state);

  //----------------------------------------------------------------
  // Measurement
  //----------------------------------------------------------------

  // Add the classical bit memory to the counts dictionary
  void update_counts();

  // Checks if an Op should be applied, returns true if either: 
  // - op is not a conditional Op,
  // - op is a conditional Op and it passes the conditional check
  bool check_conditional(const Operations::Op &op) const;

  // Record the outcome of a measurement in the engine memory and registers
  void store_measure(const reg_t &outcome, const reg_t &memory, const reg_t &registers);

  // Apply a boolean function Op
  bool apply_bfunc(const Operations::Op &op);

  // Apply readout error instruction to classical registers
  void apply_roerror(const Operations::Op &op, RngEngine &rng);

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
  template <class State_t>
  void snapshot_state(const Operations::Op &op, State_t &state);

  // Record current qubit probabilities for a measurement snapshot
  template <class State_t>
  void snapshot_probabilities(const Operations::Op &op, State_t &state);

  // Compute and store snapshot of the Pauli observable op
  template <class State_t>
  void snapshot_observables_pauli(const Operations::Op &op, State_t &state);

  // Compute and store snapshot of the matrix observable op
  template <class State_t>
  void snapshot_observables_matrix(const Operations::Op &op, State_t &state);

  // Snapshot Types
  using qubit_set_t = Operations::Op::qubit_set_t;
  using SnapshotLabel = std::string;
  using MemoryVal = std::string;
  using SnapshotStates = Snapshots::Snapshot<SnapshotLabel, json_t, Snapshots::ShotData>;
  using ProbsKey = std::pair<SnapshotLabel, MemoryVal>;
  using SnapshotProbs = Snapshots::Snapshot<ProbsKey, std::map<std::string, double>, Snapshots::AverageData>;
  using ObsKey = std::pair<SnapshotLabel, MemoryVal>;
  using SnapshotObs = Snapshots::Snapshot<ObsKey, complex_t, Snapshots::AverageData>;

  // Snapshot data structures
  SnapshotStates snapshot_states_;
  SnapshotProbs snapshot_probs_;
  SnapshotObs snapshot_obs_;
  
  // Snapshot config settings
  std::string snapshot_state_label_ = "state"; // label and key for snapshots
  double snapshot_chop_threshold_ = 1e-15;
  bool show_snapshots_ = true;

  // Cache of saved pauli values for given qubits and memory bit values
  // to prevent recomputing for re-used Pauli components in different operators
  // Note: this cache should be cleared everytime a non-snapshot operation is applied.
  using CacheKey = std::pair<qubit_set_t, MemoryVal>;
  std::map<CacheKey, std::map<std::string, double>> pauli_cache_;
  
};

//============================================================================
// Implementations: Execution
//============================================================================

template <class State_t>
void Engine::execute(const Circuit &circ, uint_t shots, State_t &state) {
  // Check if ideal simulation check if sampling is possible
  if (circ.measure_sampling_flag && shots > 1) {
    execute_with_measure_sampling(circ, shots, state);
  } else {
    // Ideal execution without sampling
    while (shots-- > 0) {
      initialize(circ, state);
      for (const auto &op: circ.ops) {
        apply_op(op, state);
      }
      update_counts();
    }
  }
}


template <class State_t>
void Engine::apply_op(const Operations::Op &op, State_t &state) {
  switch (op.type) {
    case Operations::OpType::measure: {
      reg_t outcome = state.apply_measure(op.qubits);
      store_measure(outcome, op.memory, op.registers);
      pauli_cache_.clear(); // clear Pauli cache if we apply measure
    } break;
    case Operations::OpType::snapshot_probs:
      snapshot_probabilities(op, state);
      break;
    case Operations::OpType::snapshot_pauli:
      snapshot_observables_pauli(op, state);
      break;
    case Operations::OpType::snapshot_matrix:
      snapshot_observables_matrix(op, state);
      break;
    case Operations::OpType::snapshot_state:
      snapshot_states_.add_data(op.string_params[0], state.data());
      break;
    case Operations::OpType::bfunc:
      apply_bfunc(op);
      break;
    case Operations::OpType::roerror:
      // We use the state classes RNG for this operation
      apply_roerror(op, state.access_rng());
      break;
    default:
      // Send op to the State class for evaluation
      // check if op passes conditional
      if (check_conditional(op)) {
        state.apply_op(op);
        pauli_cache_.clear(); // clear Pauli cache since the state has changed
      }
  }
}


template <class State_t>
void Engine::initialize(const Circuit &circ, State_t &state) {
  // Initialize state
  state.initialize(circ.num_qubits);
  // Clear and resize registers
  creg_memory_ = std::string(circ.num_memory, '0');
  creg_registers_ = std::string(circ.num_registers, '0');
  // clear pauli snapshot cache at start of each shot
  pauli_cache_.clear(); 
}


//============================================================================
// Implementation: Measurement and conditionals
//============================================================================

void Engine::update_counts() {
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


void Engine::store_measure(const reg_t &outcome,
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


bool Engine::check_conditional(const Operations::Op &op) const {
  // Check if op is not conditional
  if (op.conditional == false)
    return true;
  // Check if specified register bit is 1
  return (creg_registers_[creg_registers_.size() - op.conditional_reg] == '1');
}


bool Engine::apply_bfunc(const Operations::Op &op) {
  const std::string &mask = op.string_params[0];
  const std::string &target_val = op.string_params[1];
  int_t compared; // if equal this should be 0, if less than -1, if greater than +1

  // Check if register size fits into a 64-bit integer
  if (creg_registers_.size() <= 64) {
    uint_t reg_int = std::stoull(creg_registers_, nullptr, 2); // stored as bitstring
    uint_t mask_int = std::stoull(mask, nullptr, 16); // stored as hexstring
    uint_t target_int = std::stoull(target_val, nullptr, 16); // stored as hexstring
    compared = (reg_int & mask_int) - target_int;
  } else {
    // We need to use big ints so we implement the bit-mask via the binary string
    // representation rather than using a big integer class
    std::string mask_bin = Utils::hex2bin(mask); // has 0b prefix while creg_registers_ doesn't
    size_t length = std::min(mask_bin.size() - 2, creg_registers_.size()); // -2 to remove 0b
    std::string masked_val = std::string(length, '0');
    for (size_t rev_pos = 0; rev_pos < length; rev_pos++) {
      masked_val[length - 1 - rev_pos] = (mask_bin[mask_bin.size() - 1 - rev_pos] 
                                          & creg_registers_[creg_registers_.size() - 1 - rev_pos]);
    }
    masked_val = Utils::bin2hex(masked_val); // convert to hex string
    // Using string comparison to compare to target value
    compared = masked_val.compare(target_val);
  }
  // check value of compared integer for different comparison operations
  switch (op.bfunc) {
    case Operations::RegComparison::Equal:
      return (compared == 0);
    case Operations::RegComparison::NotEqual:
      return (compared != 0);
    case Operations::RegComparison::Less:
      return (compared < 0);
    case Operations::RegComparison::LessEqual:
      return (compared <= 0);
    case Operations::RegComparison::Greater:
      return (compared > 0);
    case Operations::RegComparison::GreaterEqual:
      return (compared >= 0);
    default:
      // we shouldn't ever get here
      throw std::invalid_argument("Invalid boolean function relation.");
  }
}

// Apply readout error instruction to classical registers
void Engine::apply_roerror(const Operations::Op &op, RngEngine &rng) {
  // Get current classical bit (and optionally register bit) values
  std::string mem_str;
  // Get values of bits as binary string
  // We iterate from the end of the list of memory bits
  for (auto it = op.memory.rbegin(); it < op.memory.rend(); ++it) {
    mem_str.push_back(creg_memory_[*it]);
  }
  auto mem_val = std::stoull(mem_str, nullptr, 2);
  auto noise_str = Utils::int2string(rng.rand_int(op.probs[mem_val]), 2, op.memory.size());
  for (size_t ii = 0; ii < op.memory.size(); ++ii) {
    auto bit = op.memory[ii];
    creg_memory_[bit] = noise_str[noise_str.size() - 1 - ii];
  }
  // and the same error to register classical bits if they are used
  for (size_t ii = 0; ii < op.registers.size(); ++ii) {
    auto bit = op.registers[ii];
    creg_registers_[bit] = noise_str[noise_str.size() - 1 - ii];
  }
}

//============================================================================
// Implementations: Measurement sampling optimization
//============================================================================

template <class State_t>
void Engine::execute_with_measure_sampling(const Circuit &circ,
                                           uint_t shots,
                                           State_t &state) {                                    
  initialize(circ, state);
  // Apply operations until first measurement
  const size_t number_of_ops = circ.ops.size();
  uint_t pos = 0;
  while (pos < number_of_ops) {
    const auto &op = circ.ops[pos];
    if (op.name == "measure")
      break;
    apply_op(op, state);
    pos++;
  }
  // Check if we have reached the end of the operations
  // If so there are no measurements, so update counts by looping over shots.
  if (pos == number_of_ops) {
    while (shots-- > 0) {
      update_counts(); // add trivial counts
    }
    return;
  }

  // If not we still have remaining measure opts
  // Get measurement operations and set of measured qubits
  std::vector<Operations::Op> meas(circ.ops.begin() + pos, circ.ops.end());
  std::vector<uint_t> meas_qubits; // measured qubits
  // Now we need to map the samples to the correct memory and register locations
  std::map<uint_t, uint_t> memory_map; // map of memory locations to qubit measured
  std::map<uint_t, uint_t> registers_map;// map of register locations to qubit measured
  for (const auto &op : meas) {
    for (size_t j=0; j < op.qubits.size(); ++j) {
      meas_qubits.push_back(op.qubits[j]);
      if (!op.memory.empty())
        memory_map[op.qubits[j]] = op.memory[j];
      if (!op.registers.empty())
        registers_map[op.qubits[j]] = op.registers[j];
    }
  }
  // Sort the qubits and delete duplicates
  sort(meas_qubits.begin(), meas_qubits.end());
  meas_qubits.erase(unique(meas_qubits.begin(), meas_qubits.end()), meas_qubits.end());
  // Convert memory and register maps to ordered lists
  reg_t memory;
  if (!memory_map.empty())
    for (const auto &q: meas_qubits)
      memory.push_back(memory_map[q]);
  reg_t registers;
  if (!registers_map.empty())
    for (const auto &q: meas_qubits)
      registers.push_back(registers_map[q]);

  // Generate the samples
  auto samples = state.sample_measure(meas_qubits, shots);
  while (!samples.empty()) {
    store_measure(samples.back(), memory, registers);
    update_counts(); // add sample to counts
    samples.pop_back(); // pop off processed sample
  }
}

//============================================================================
// Implementations: Utilities methods
//============================================================================

template <class State_t>
std::set<std::string> Engine::validate_circuit(const Circuit &circ,
                                               State_t &state) {
  auto state_ops = state.allowed_ops();
  state_ops.insert("bfunc"); // handled by engine alone
  return circ.invalid_ops(state_ops);
}


void Engine::clear() {
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


void Engine::combine(Engine &eng) {
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


void Engine::load_config(const json_t &js) {
  JSON::get_value(snapshot_state_label_, "snapshot_label", js);
  JSON::get_value(show_snapshots_, "show_snapshots", js);
  JSON::get_value(return_counts_, "counts", js);
  JSON::get_value(return_memory_, "memory", js);
  JSON::get_value(return_registers_, "register", js);
  JSON::get_value(snapshot_chop_threshold_, "chop_threshold", js);
}


json_t Engine::json() const {
  json_t tmp;
  // Add Counts and memory
  if (return_counts_ && counts_.empty() == false)
    tmp["counts"] = counts_;
  if (return_memory_ && memory_.empty() == false)
    tmp["memory"] = memory_;
  if (return_registers_ && registers_.empty() == false)
    tmp["register"] = registers_;
  
  // Add probability snapshots
  for (const auto &key : snapshot_probs_.keys()) {
    std::string label = key.first;
    json_t datum;
    datum["values"] = snapshot_probs_.get_data(key).data();
    if (key.second.empty() == false)
      datum["memory"] = key.second;
    tmp["snapshots"]["probabilities"][key.first].push_back(datum);
  }

  // Add observables snapshots
  for (const auto &key : snapshot_obs_.keys()) {
    json_t datum;
    datum["value"] = snapshot_obs_.get_data(key).data();
    if (key.second.empty() == false)
      datum["memory"] = key.second;
    tmp["snapshots"]["observables"][key.first].push_back(datum);
  }

  // Add state snapshots
  for (const auto &key : snapshot_states_.keys()) {
    try {
      tmp["snapshots"][snapshot_state_label_][key] = snapshot_states_.get_data(key).data();
    } catch (std::exception &e) {
      // Leave message in output that type conversion failed
      tmp["snapshots"][snapshot_state_label_][key] = "Error: Failed to convert state type to JSON";
    }
  }

  // return json
  return tmp;
}

//============================================================================
// Implementations: Snapshots
//============================================================================

template <class State_t>
void Engine::snapshot_probabilities(const Operations::Op &op,
                                    State_t &state) {
  std::string memory_hex = Utils::bin2hex(creg_memory_); // convert memory to hex string
  ProbsKey key = std::make_pair(op.string_params[0], memory_hex);
  auto probs = Utils::vec2ket(state.measure_probs(op.qubits),
                              snapshot_chop_threshold_, 16); // get probs as hexadecimal
  snapshot_probs_.add_data(key, probs);
}


template <class State_t>
void Engine::snapshot_observables_pauli(const Operations::Op &op,
                                        State_t &state) {
  std::string memory_hex = Utils::bin2hex(creg_memory_); // convert memory to hex string
  auto key = std::make_pair(op.string_params[0], memory_hex);
  complex_t expval(0., 0.); // complex value
  for (const auto &param : op.params_pauli_obs) {
    const auto& coeff = std::get<0>(param);
    const auto& qubits = std::get<1>(param);
    const auto& pauli = std::get<2>(param);
    auto &cache = pauli_cache_[std::make_pair(qubits, memory_hex)];
    auto it = cache.find(pauli); // Check cache for value
    if (it != cache.end()) {
      expval += coeff * (it->second); // found cached result
    } else {
      // compute result and add to cache
      double tmp = state.pauli_observable_value(reg_t(qubits.begin(), qubits.end()), pauli);
      cache[pauli] = tmp;
      expval += coeff * tmp;
    }
  }
  // add to snapshot
  snapshot_obs_.add_data(key, expval);
}


template <class State_t>
void Engine::snapshot_observables_matrix(const Operations::Op &op,
                                         State_t &state) {
  auto key = std::make_pair(op.string_params[0], Utils::bin2hex(creg_memory_));
  snapshot_obs_.add_data(key, state.matrix_observable_value(op));
}

//------------------------------------------------------------------------------
} // end namespace Base
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
