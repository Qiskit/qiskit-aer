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

#ifndef _aer_base_engine_hpp_
#define _aer_base_engine_hpp_

#include <stdexcept>
#include <vector>

#include "framework/circuit.hpp"
#include "framework/json.hpp"
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

  //----------------------------------------------------------------
  // Abstract methods that must be defined by derived classes
  //----------------------------------------------------------------

  // Compute results from the current value of the State
  virtual void compute_result(State<state_t> *state) = 0;

  // Empty engine of stored data
  virtual void clear() = 0;

  // Serialize engine data to JSON
  virtual json_t json() const = 0;

  //----------------------------------------------------------------
  // Optional methods that may be overriden
  //----------------------------------------------------------------

  // Combine engines for accumulating data
  // Second engine should no longer be used after combining
  // as this function should use move semantics to minimize copying
  void combine(Engine<state_t> &eng) {(void)eng;};

  // Load any settings for the State class from a config JSON
  inline virtual void load_config(const json_t &config) {(void)config;};

  // Apply an operation to the state
  inline virtual void apply_op(State<state_t> *state, const Op &op) {
    state->apply_op(op);
  };
  // Initialize an engine and circuit
  inline virtual void initialize(State<state_t> *state, const Circuit &circ) {
    state->initialize(circ.num_qubits);
  };

  // Execute a circuit of operations for multiple shots
  // By default this loops over the set number of shots where
  // for each shot the follow is applied:
  //   1. initialize the state based on max qubit number in ops
  //   2. call apply_operations on the state for list of ops
  //   3. call compute_results after ops have been applied
  inline virtual void execute(State<state_t> *state,
                       const Circuit &circ,
                       uint_t shots);
};

//============================================================================
// Implementations
//============================================================================

template <class state_t>
void Engine<state_t>::execute(State<state_t> *state,
                              const Circuit &circ,
                              uint_t shots) {
  for (size_t ishot = 0; ishot < shots; ++ishot) {
    initialize(state, circ);
    for (const auto &op: circ.ops) {
      apply_op(state, op);
    }
    compute_result(state);
  }
}

//------------------------------------------------------------------------------
} // end namespace Base
} // end namespace AER
//------------------------------------------------------------------------------
#endif
