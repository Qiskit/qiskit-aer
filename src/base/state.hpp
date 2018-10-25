/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_base_state_hpp_
#define _aer_base_state_hpp_

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
  using ignore_argument = void;
  State() = default;
  virtual ~State() = default;
  
  //----------------------------------------------------------------
  // Abstract methods that must be defined by derived classes
  //----------------------------------------------------------------
  
  // Return the set of qobj instruction types supported by the State
  // by the Operations::OpType enum class. 
  // Standard OpTypes that can be included here are:
  // - `OpType::gate` if gates are supported
  // - `OpType::measure` if measure is supported
  // - `OpType::reset` if reset is supported
  // - `OpType::barrier` if barrier is supported
  // - `OpType::matrix` if arbitrary unitary matrices are supported
  // - `OpType::snapshot_state` if state snapshots are supported
  // - `OpType::snapshot_probs` if probabilities snapshots are supported
  // - `OpType::snapshot_pauli` if pauli expectation value snapshots are supported
  // - `OpType::snapshot_matrix` if matrix expectation value snapshots are supported
  // - `OpType::kraus` if general Kraus noise channels are supported
  // For the case of gates the specific allowed gates are checked
  // with the `allowed_gates` function.
  virtual std::unordered_set<Operations::OpType> allowed_ops() const = 0;
  
  // Return the set of qobj gate instruction names supported by the state class
  // For example this could include {"u1", "u2", "u3", "U", "cx", "CX"}
  virtual stringset_t allowed_gates() const = 0;

  // Applies an operation to the state class.
  // This should support all and only the operations defined in
  // allowed_operations.
  virtual void apply_op(const Operations::Op &op) = 0;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize(uint_t num_qubits) = 0;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // TODO: Is this enough? Some State representaitons might also depend on 
  // the circuit (eg. tensor slicing, clifford+t simulator)
  virtual uint_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops) = 0;

  //----------------------------------------------------------------
  // Optional methods: Config
  //----------------------------------------------------------------
  
  // Load any settings for the State class from a config JSON
  inline virtual void set_config(const json_t &config) {
    (ignore_argument)config;
  };

  //----------------------------------------------------------------
  // Optional methods: Measurement 
  //----------------------------------------------------------------

  // Note that if measurement is supported by the derrived class
  // it should contain a "measure" and "snapshot_probs" in the
  // return of 'allowed_ops'.
  
  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure" 
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  inline virtual reg_t apply_measure(const reg_t& qubits) {
    return reg_t(qubits.size(), 0); // dummy return for base class
  };

  // Return vector of measure probabilities for specified qubits
  // If a state subclass supports this function it then "measure" 
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  inline virtual rvector_t measure_probs(const reg_t &qubits) const {
    return rvector_t(1ULL << qubits.size(), 0);
  };

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state. This leaves the system unchanged. The return is
  // a vector containing n outcomes, where each outcome is the same as
  // one that would be returned from the 'apply_measure' operation
  inline virtual std::vector<reg_t>
  sample_measure(const reg_t& qubits, uint_t shots = 1) {
    const rvector_t probs = measure_probs(qubits);
    std::vector<reg_t> samples;
    samples.reserve(shots);
    while (shots-- > 0) { // loop over shots
      samples.push_back(Utils::int2reg(rng_.rand_int(probs), 2, qubits.size()));
    }
    return samples;
  };

  //----------------------------------------------------------------
  // Optional methods: Operator observables
  //----------------------------------------------------------------

  // Return the complex expectation value for a single Pauli string
  // If a state subclass supports this function it then 
  // "snapshot_pauli" should be contained in the set returned by the
  // 'allowed_ops' method.
  inline virtual
  double pauli_observable_value(const reg_t& qubits,
                                const std::string &pauli) const {
    (ignore_argument)qubits;
    (ignore_argument)pauli;
    return 0.; // dummy return for base class
  };

  // Return the complex expectation value for an observable operator.
  // If a state subclass supports this function it then 
  // "snapshot_matrix" should be contained in the set returned by the
  // 'allowed_ops' method.
  inline virtual 
  complex_t matrix_observable_value(const Operations::Op &op) const {
    (ignore_argument)op;
    return complex_t(); // dummy return for base class
  };

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