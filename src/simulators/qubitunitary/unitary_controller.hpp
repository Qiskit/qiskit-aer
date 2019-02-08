/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_unitary_controller_hpp_
#define _aer_unitary_controller_hpp_

#include "base/controller.hpp"
#include "unitary_state.hpp"

namespace AER {
namespace Simulator {

//=========================================================================
// UnitaryController class
//=========================================================================


/**************************************************************************
 * Config settings:
 * 
 * From QubitUnitary::State class
 * 
 * - "initial_unitary" (json complex matrix): Use a custom initial unitary
 *      matrix for the simulation [Default: null].
 * - "chop_threshold" (double): Threshold for truncating small values to
 *      zero in result data [Default: 1e-15]
 * - "unitary_parallel_threshold" (int): Threshold that number of qubits
 *      must be greater than to enable OpenMP parallelization at State
 *      level [Default: 6]
 * 
 * From BaseController Class
 *
 * - "max_parallel_threads" (int): Set the maximum OpenMP threads that may
 *      be used across all levels of parallelization. Set to 0 for maximum
 *      available. [Default : 0]
 * - "max_parallel_experiments" (int): Set number of circuits that may be
 *      executed in parallel. Set to 0 to use the number of max parallel
 *      threads [Default: 1]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * 
 **************************************************************************/

class UnitaryController : public Base::Controller {
public:
  //-----------------------------------------------------------------------
  // Base class config override
  //-----------------------------------------------------------------------
  
  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  // Allowed config options:
  // - "initial_unitary: complex_matrix"
  // Plus Base Controller config options
  virtual void set_config(const json_t &config) override;

  // Clear the current config
  void virtual clear_config() override;

private:

  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // This simulator will only return a single shot, regardless of the
  // input shot number
  virtual OutputData run_circuit(const Circuit &circ,
                                 uint_t shots,
                                 uint_t rng_seed) const override;
  
  //-----------------------------------------------------------------------
  // Custom initial state
  //-----------------------------------------------------------------------        
  cmatrix_t initial_unitary_;
};

//=========================================================================
// Implementation
//=========================================================================

//-------------------------------------------------------------------------
// Config
//-------------------------------------------------------------------------

void UnitaryController::set_config(const json_t &config) {
  // Set base controller config
  Base::Controller::set_config(config);

  //Add custom initial unitary
  if (JSON::get_value(initial_unitary_, "initial_unitary", config) ) {
    // Check initial state is unitary
    if (!Utils::is_unitary(initial_unitary_, 1e-10))
      throw std::runtime_error("UnitaryController: initial_unitary is not unitary");
  }
}

void UnitaryController::clear_config() {
  Base::Controller::clear_config();
  initial_unitary_ = cmatrix_t();
}

//-------------------------------------------------------------------------
// Run circuit
//-------------------------------------------------------------------------

OutputData UnitaryController::run_circuit(const Circuit &circ,
                                          uint_t shots,
                                          uint_t rng_seed) const {
  // Initialize state
  QubitUnitary::State<> state;
  
  // Validate circuit and throw exception if invalid operations exist
  validate_state(state, circ, noise_model_, true);

  // Check for custom initial state, and if so check it matches num qubits
  if (!initial_unitary_.empty()) {
    auto nrows = initial_unitary_.GetRows();
    auto ncols = initial_unitary_.GetColumns();
    if (nrows != ncols ) {
      throw std::runtime_error("UnitaryController: initial unitary is not square.");
    }
    auto nstates = 1ULL << circ.num_qubits;
    if (nrows != nstates) {
      uint_t num_qubits(std::log2(nrows));
      std::stringstream msg;
      msg << "UnitaryController: " << num_qubits << "-qubit initial unitary ";
      msg << "cannot be used for a " << circ.num_qubits << "-qubit circuit.";
      throw std::runtime_error(msg.str());
    }
  }

  // Set state config
  state.set_config(Base::Controller::config_);
  state.set_available_threads(parallel_gates_);

  // Rng engine (not actually needed for unitary controller)
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  OutputData data;
  data.set_config(Base::Controller::config_);

  // Run single shot collecting measure data or snapshots
  if (initial_unitary_.empty())
    state.initialize_qreg(circ.num_qubits);
  else
    state.initialize_qreg(circ.num_qubits, initial_unitary_);
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(circ.ops, data, rng);
  state.add_creg_to_data(data);

  // Add final state unitary to the data
  data.add_additional_data("unitary", state.qreg());

  return data;
}


//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
