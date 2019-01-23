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
 
 * - "identity_checker" (bool): Return a pair (bool, double) where bool is true if
 *      the final unitary is equivalent to the identity matrix (up to
 *      global phase) and the double is phase angle theta exp(I*theta).
 *      If set to `true` the final unitary matrix will not be returned
 *      [Default: False].
 * - "validation_threshold" (double): Threshold for truncating small values to
 *      zero in result data [Default: 1e-8]
 * - "initial_unitary" (json complex matrix): Use a custom initial unitary
 *      matrix for the simulation [Default: null].
 * - "target_unitary" (json complex matrix): Use a custom target unitary
 *      for identity_checker mode [Default: null].
 *
 * From QubitUnitary::State class
 *
 * - "zero_threshold" (double): Threshold for truncating small values to
 *      zero in result data [Default: 1e-10]
 * - "identity_checker_threshold" (double): Threshold for truncating small values to
 *      zero in result data [Default: 1e-8]
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

protected:

  size_t required_memory_mb(const Circuit& circ) const override;

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

  //-----------------------------------------------------------------------
  // Method
  //-----------------------------------------------------------------------      
  bool identity_checker_ = false;

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

  // Check if we are in identity checker mode
  JSON::get_value(identity_checker_, "identity_checker", config);

  //Add custom initial unitary
  if (!identity_checker_ && JSON::get_value(initial_unitary_, "initial_unitary", config) ) {
    // Check initial state is unitary
    if (!Utils::is_unitary(initial_unitary_, validation_threshold_))
      throw std::runtime_error("UnitaryController: initial_unitary is not unitary");
  }

  //Add custom target unitary
  if (identity_checker_ && JSON::get_value(initial_unitary_, "target_unitary", config) ) {
    // Check target state is unitary
    if (!Utils::is_unitary(initial_unitary_, validation_threshold_))
      throw std::runtime_error("UnitaryController: target_unitary is not unitary");
    AER::Utils::dagger_inplace(initial_unitary_);
  }
}

void UnitaryController::clear_config() {
  Base::Controller::clear_config();
  initial_unitary_ = cmatrix_t();
}

size_t UnitaryController::required_memory_mb(const Circuit& circ) const {
  QubitUnitary::State<> state;
  return state.required_memory_mb(circ.num_qubits, circ.ops);
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
  state.set_parallalization(parallel_state_update_);

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

  // If in identity checker mode add identity check to data
  if (identity_checker_) {
    data.add_additional_data("identity", state.qreg().check_identity());
  } else {
    // Otherwise add the final state unitary to the data
    data.add_additional_data("unitary", state.qreg());
  }
  return data;
}


//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
