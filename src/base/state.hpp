/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    state.hpp
 * @brief   State interface base class for qiskit-aer simulator engines
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_base_state_hpp_
#define _aer_base_state_hpp_

#include <set>
#include <string>

#include "framework/json.hpp"
#include "framework/operations.hpp"
#include "framework/rng.hpp"
#include "framework/types.hpp"

namespace AER {
namespace Base {

//============================================================================
// State interface base class for Qiskit-Aer
//============================================================================

template <class state_t>
class State {

public:
  State() = default;
  virtual ~State() = default;

  //----------------------------------------------------------------
  // Abstract methods that must be defined by derived classes
  //----------------------------------------------------------------
  
  // Return the set of allowed operations for the state class.
  // Example: {"u1", "u2", "u3", "cx", "measure", "reset"};
  inline virtual std::set<std::string> allowed_ops() const = 0;

  // Applies an operation to the state class.
  // This should support all and only the operations defined in
  // allowed_operations.
  virtual void apply_op(const Op &op) = 0;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize(uint_t num_qubits) = 0;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // TODO: Is this enough? Some State representaitons might also depend on 
  // the circuit (eg. tensor slicing, clifford+t simulator)
  virtual uint_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Op> &ops) = 0;

  //----------------------------------------------------------------
  // Optional methods: Measurement 
  //----------------------------------------------------------------
  
  // If the state supports measurement this should be set to true
  // and the apply_measure method overriden
  bool has_measure = false;

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  inline virtual reg_t apply_measure(const reg_t& qubits) {
    return reg_t(qubits.size(), 0);
  };

  //----------------------------------------------------------------
  // Optional methods: Config
  //----------------------------------------------------------------
  
  // Load any settings for the State class from a config JSON
  inline virtual void load_config(const json_t &config) {(void)config;};

  //----------------------------------------------------------------
  // Members that should not be modified by derrived classes
  //----------------------------------------------------------------

  // Returns a reference to the states data structure
  inline state_t &data() { return data_; };

  // Set the state data to a fixed value 
  inline void set_state(const state_t &state) {data_ = state;};
  inline void set_state(state_t &&state) {data_ = std::move(state);};

  // Sets the number of threads available to the State implementation
  // If negative there is no restriction on the backend
  inline void set_available_threads(int n) {threads_ = n;};
  inline int get_available_threads() {return threads_;};
  
  // Access the RngEngine for random number generation
  inline RngEngine &access_rng() { return rng_; };

  // Set the RngEngine seed to a fixed value
  // Otherwise it is initialized with a random value
  inline void set_rng_seed(uint_t seed) { rng_ = RngEngine(seed); };

  
protected:
  // Allowed ops
  const std::set<std::string> allowed_ops_ = {};

  // The quantum state data structure
  state_t data_;

  // Maximum threads which may be used by the backend for OpenMP multithreading
  // Default value is single-threaded unless overridden
  int threads_ = 1;

  // The RngEngine for the state
  RngEngine rng_;
};

//------------------------------------------------------------------------------
} // end namespace Base
} // end namespace AER
//------------------------------------------------------------------------------
#endif