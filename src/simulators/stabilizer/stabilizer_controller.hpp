/**
 * Copyright 2019, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_stabilizer_controller_hpp_
#define _aer_stabilizer_controller_hpp_

#include "base/controller.hpp"
#include "simulators/stabilizer/stabilizer_state.hpp"

// TODO: Measure sampling

namespace AER {
namespace Simulator {

//=========================================================================
// StabilizerController class
//=========================================================================

/**************************************************************************
 * Config settings:
 * 
 * From Stabilizer::State class
 * 
 * - TODO
 * 
 * From BaseController Class
 *
 * - "noise_model" (json): A noise model to use for simulation [Default: null]
 * - "max_parallel_threads" (int): Set the maximum OpenMP threads that may
 *      be used across all levels of parallelization. Set to 0 for maximum
 *      available. [Default : 0]
 * - "max_parallel_experiments" (int): Set number of circuits that may be
 *      executed in parallel. Set to 0 to use the number of max parallel
 *      threads [Default: 1]
 * - "max_parallel_shots" (int): Set number of shots that maybe be executed
 *      in parallel for each circuit. Sset to 0 to use the number of max
 *      parallel threads [Default: 1].
 * - "counts" (bool): Return counts objecy in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 * - "register" (bool): Return register array in circuit data [Default: False]
 * 
 **************************************************************************/

class StabilizerController : public Base::Controller {
public:
  //-----------------------------------------------------------------------
  // Base class config override
  //-----------------------------------------------------------------------
  
  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  virtual void set_config(const json_t &config) override;

  // Clear the current config
  void virtual clear_config() override;

private:

  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  virtual OutputData run_circuit(const Circuit &circ,
                                 uint_t shots,
                                 uint_t rng_seed,
                                 int num_threads_state) const override;

  //----------------------------------------------------------------
  // Run circuit without optimization
  //----------------------------------------------------------------

  // Execute n-shots of a circuit on the input state
  template <class State_t>
  void run_circuit_default(const Circuit &circ,
                           uint_t shots,
                           State_t &state,
                           OutputData &data,
                           RngEngine &rng) const;

  //----------------------------------------------------------------
  // Measure sampling optimizations
  //----------------------------------------------------------------
  
  // Execute n-shots of a circuit performing measurement sampling
  // if the input circuit supports it
  template <class State_t>
  void run_circuit_measure_sampler(const Circuit &circ,
                                   uint_t shots,
                                   State_t &state,
                                   OutputData &data,
                                   RngEngine &rng) const;


  // Sample measurement outcomes for the input measure ops from the
  // current state of the input State_t
  template <class State_t>
  void measure_sampler(const std::vector<Operations::Op> &meas_ops,
                       uint_t shots,
                       State_t &state,
                       OutputData &data,
                       RngEngine &rng) const;

  // Check if measure sampling optimization if valid for the input circuit
  // if so return a pair {true, pos} where pos is the position of the
  // first measurement operation in the input circuit
  std::pair<bool, size_t> check_measure_sampling_opt(const Circuit &circ) const;
  
  //-----------------------------------------------------------------------
  // Custom initial state
  //-----------------------------------------------------------------------        
  Clifford::Clifford initial_state_;
};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Config
//-------------------------------------------------------------------------

void StabilizerController::set_config(const json_t &config) {
  // Set base controller config
  Base::Controller::set_config(config);
  // Add custom initial state
  JSON::get_value(initial_state_, "initial_stabilizer", config);
}

void StabilizerController::clear_config() {
  Base::Controller::clear_config();
  initial_state_ = Clifford::Clifford();
}

//-------------------------------------------------------------------------
// Base class override
//-------------------------------------------------------------------------

OutputData StabilizerController::run_circuit(const Circuit &circ,
                                             uint_t shots,
                                             uint_t rng_seed,
                                             int num_threads_state) const {  
  // Initialize state
  Stabilizer::State state;

  // Validate circuit and noise model
  validate_state_except(state, circ);

  // TODO: add custom initial state
  // Check for custom initial state, and if so check it matches num qubits
  if (!initial_state_.empty()) {
    if (initial_state_.num_qubits() != circ.num_qubits) {
      std::stringstream msg;
      msg << "StabilizerController: " << initial_state_.num_qubits() << "-qubit initial state ";
      msg << "cannot be used for a " << circ.num_qubits << "-qubit circuit.";
      throw std::runtime_error(msg.str());
    }
  }

  // Set config
  state.set_config(Base::Controller::config_);
  state.set_available_threads(num_threads_state);
  
  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  OutputData data;
  data.set_config(Base::Controller::config_);
  
  // Check if there is noise for the implementation
  if (noise_model_.ideal()) {
    // Implement without noise
    run_circuit_default(circ, shots, state, data, rng);
  } else {
    // Sample noise for each shot
    while (shots-- > 0) {
      Circuit noise_circ = noise_model_.sample_noise(circ, rng);
      run_circuit_default(noise_circ, 1, state, data, rng);
    }
  }
  return data;
}


//-------------------------------------------------------------------------
// Run circuit helpers
//-------------------------------------------------------------------------

template <class State_t>
void StabilizerController::run_circuit_default(const Circuit &circ,
                                               uint_t shots,
                                               State_t &state,
                                               OutputData &data,
                                               RngEngine &rng) const {  
  while (shots-- > 0) {
    if (initial_state_.empty())
      state.initialize_qreg(circ.num_qubits);
    else
      state.initialize_qreg(circ.num_qubits, initial_state_);
    state.initialize_creg(circ.num_memory, circ.num_registers);
    state.apply_ops(circ.ops, data, rng);
    state.add_creg_to_data(data);
  }
}

//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
