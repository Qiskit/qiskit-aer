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

#ifndef _aer_engines_qasm_engine_hpp_
#define _aer_engines_qasm_engine_hpp_

#include <algorithm>

#include "base/engine.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace Engines {


  template <class state_t>
  using State = Base::State<state_t>;

  template <class state_t>
  using BaseEngine = Base::Engine<state_t>;

//============================================================================
// QASM Engine class for Qiskit-Aer
//============================================================================

// This engine returns counts for state classes that support measurement
template <class state_t>
class QasmEngine : public virtual BaseEngine<state_t> {

public:

  //----------------------------------------------------------------
  // Base class overrides
  //----------------------------------------------------------------

  virtual void initialize(State<state_t> *state,
                       const Circuit &circ) override;

  virtual void compute_result(State<state_t> *state) override;

  // Apply a sequence of operations to the state
  // TODO: modify this to handle conditional operations, and measurement
  virtual void apply_operations(State<state_t> *state,
                                const std::vector<Op> &ops) override;

  // Empty engine of stored data
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

protected:
  // Checks if operation is measure and if so records measurement outcome
  void apply_single_op(State<state_t> *state, const Op &op);

  // Checks if an Op should be applied, returns true if either: 
  // - op is not a conditional Op,
  // - op is a conditional Op and it passes the conditional check
  bool check_conditional(const Op &op) const;

  // Record the outcome of a measurement in the engine memory and registers
  void store_measure(const reg_t &outcome, const reg_t &memory, const reg_t &registers);

  // initialize creg strings
  void initialize_creg(const Circuit &circ);

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
};

//============================================================================
// Implementation: Base engine overrides
//============================================================================

template <class state_t>
void QasmEngine<state_t>::initialize(State<state_t> *state, const Circuit &circ) {
  initialize_creg(circ);
  BaseEngine<state_t>::initialize(state, circ);
}


template <class state_t>
void QasmEngine<state_t>::apply_operations(State<state_t> *state,
                                           const std::vector<Op> &ops) {
  for (const auto &op: ops) {
    if (check_conditional(op)) { // check if op passes conditional
      if (op.name == "measure") { // check if op is measurement
        reg_t outcome = state->apply_measure(op.qubits);
        store_measure(outcome, op.memory, op.registers);
      } else {
      state->apply_operation(op);  // Apply operation as usual
      }
    }
  }
}


template <class state_t>
void QasmEngine<state_t>::compute_result(State<state_t> *state) {
  (void)state; // avoid unused variable warning
  // Get memory value string
  if ((return_counts_ || return_memory_) && !creg_memory_.empty()) {
    if (return_hex_strings_) // Convert to a hex string
      creg_memory_ = "0x" + Utils::bin2hex(creg_memory_);
    if (return_counts_)
      counts_[creg_memory_] += 1;
    if (return_memory_)
      memory_.emplace_back(std::move(creg_memory_));
  }
  // Get registers value-string
  if (return_registers_ && !creg_registers_.empty()) {
    if (return_hex_strings_)
      creg_registers_ = "0x" + Utils::bin2hex(creg_registers_);
    memory_.emplace_back(std::move(creg_registers_));
  }
}


template <class state_t>
void QasmEngine<state_t>::clear() {
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
}


template <class state_t>
json_t QasmEngine<state_t>::json() const {
  json_t tmp;
  if (return_counts_ && counts_.empty() == false)
    tmp["counts"] = counts_;
  if (return_memory_ && memory_.empty() == false)
    tmp["memory"] = memory_;
  if (return_registers_ && registers_.empty() == false)
    tmp["register"] = registers_;
  return tmp;
}


template <class state_t>
void QasmEngine<state_t>::load_config(const json_t &js) {
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
bool QasmEngine<state_t>::check_conditional(const Op &op) const {
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

//------------------------------------------------------------------------------
} // end namespace Engines
} // end namespace AER
//------------------------------------------------------------------------------
#endif
