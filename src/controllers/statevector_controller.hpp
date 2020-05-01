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

#ifndef _aer_statevector_controller_hpp_
#define _aer_statevector_controller_hpp_

#include "controller.hpp"
#include "simulators/statevector/statevector_state.hpp"

namespace AER {
namespace Simulator {

//=========================================================================
// StatevectorController class
//=========================================================================

/**************************************************************************
 * Config settings:
 *
 * From Statevector::State class
 *
 * - "initial_statevector" (json complex vector): Use a custom initial
 *      statevector for the simulation [Default: null].
 * - "zero_threshold" (double): Threshold for truncating small values to
 *      zero in result data [Default: 1e-10]
 * - "statevector_parallel_threshold" (int): Threshold that number of qubits
 *      must be greater than to enable OpenMP parallelization at State
 *      level [Default: 13]
 * - "statevector_sample_measure_opt" (int): Threshold that number of qubits
 *      must be greater than to enable indexing optimization during
 *      measure sampling [Default: 10]
 * - "statevector_hpc_gate_opt" (bool): Enable large qubit gate optimizations.
 *      [Default: False]
 *
 * From BaseController Class
 *
 * - "max_parallel_threads" (int): Set the maximum OpenMP threads that may
 *      be used across all levels of parallelization. Set to 0 for maximum
 *      available. [Default : 0]
 * - "max_parallel_experiments" (int): Set number of circuits that may be
 *      executed in parallel. Set to 0 to use the number of max parallel
 *      threads [Default: 1]
 * - "counts" (bool): Return counts object in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 * - "register" (bool): Return register array in circuit data [Default: False]
 *
 **************************************************************************/

class StatevectorController : public Base::Controller {
 public:
  //-----------------------------------------------------------------------
  // Base class config override
  //-----------------------------------------------------------------------
  StatevectorController();

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  // Allowed config options:
  // - "initial_statevector: complex_vector"
  // Plus Base Controller config options
  virtual void set_config(const json_t& config) override;

  // Clear the current config
  void virtual clear_config() override;

 protected:
  virtual size_t required_memory_mb(
      const Circuit& circuit, const Noise::NoiseModel& noise) const override;

  // Simulation methods for the Statevector Controller
  enum class Method {
    automatic,
    statevector_cpu,
    statevector_thrust_gpu,
    statevector_thrust_cpu
  };

  // Simulation precision
  enum class Precision { double_precision, single_precision };

 private:
  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // This simulator will only return a single shot, regardless of the
  // input shot number
  virtual ExperimentData run_circuit(const Circuit& circ,
                                     const Noise::NoiseModel& noise,
                                     const json_t& config, uint_t shots,
                                     uint_t rng_seed) const override;

  // Execute n-shots of a circuit on the input state
  template <class State_t>
  ExperimentData run_circuit_helper(const Circuit& circ,
                                    const Noise::NoiseModel& noise,
                                    const json_t& config, uint_t shots,
                                    uint_t rng_seed) const;
  //-----------------------------------------------------------------------
  // Custom initial state
  //-----------------------------------------------------------------------
  cvector_t initial_state_;

  // Method for storing statevector
  Method method_ = Method::automatic;

  // Precision of statevector
  Precision precision_ = Precision::double_precision;
};

//=========================================================================
// Implementations
//=========================================================================

StatevectorController::StatevectorController() : Base::Controller() {
  // Disable qubit truncation by default
  Base::Controller::truncate_qubits_ = false;
}

//-------------------------------------------------------------------------
// Config
//-------------------------------------------------------------------------

void StatevectorController::set_config(const json_t& config) {
  // Set base controller config
  Base::Controller::set_config(config);

  // Override max parallel shots to be 1 since this should only be used
  // for single shot simulations
  Base::Controller::max_parallel_shots_ = 1;

  // Add custom initial state
  if (JSON::get_value(initial_state_, "initial_statevector", config)) {
    // Check initial state is normalized
    if (!Utils::is_unit_vector(initial_state_, validation_threshold_))
      throw std::runtime_error(
          "StatevectorController: initial_statevector is not a unit vector");
  }

  // Add method
  std::string method;
  if (JSON::get_value(method, "method", config)) {
    if (method == "statevector" || method == "statevector_cpu") {
      method_ = Method::statevector_cpu;
    } else if (method == "statevector_gpu") {
      method_ = Method::statevector_thrust_gpu;
    } else if (method == "statevector_thrust") {
      method_ = Method::statevector_thrust_cpu;
    } else if (method != "automatic") {
      throw std::runtime_error(
          std::string("UnitaryController: Invalid simulation method (") +
          method + std::string(")."));
    }
  }

  std::string precision;
  if (JSON::get_value(precision, "precision", config)) {
    if (precision == "double") {
      precision_ = Precision::double_precision;
    } else if (precision == "single") {
      precision_ = Precision::single_precision;
    }
  }
}

void StatevectorController::clear_config() {
  Base::Controller::clear_config();
  initial_state_ = cvector_t();
}

size_t StatevectorController::required_memory_mb(
    const Circuit& circ, const Noise::NoiseModel& noise) const {
  if (precision_ == Precision::single_precision) {
    Statevector::State<QV::QubitVector<float>> state;
    return state.required_memory_mb(circ.num_qubits, circ.ops);
  } else {
    Statevector::State<> state;
    return state.required_memory_mb(circ.num_qubits, circ.ops);
  }
}

//-------------------------------------------------------------------------
// Run circuit
//-------------------------------------------------------------------------

ExperimentData StatevectorController::run_circuit(
    const Circuit& circ, const Noise::NoiseModel& noise, const json_t& config,
    uint_t shots, uint_t rng_seed) const {
  switch (method_) {
    case Method::automatic:
    case Method::statevector_cpu: {
      if (precision_ == Precision::double_precision) {
        // Double-precision Statevector simulation
        return run_circuit_helper<Statevector::State<QV::QubitVector<double>>>(
            circ, noise, config, shots, rng_seed);
      } else {
        // Single-precision Statevector simulation
        return run_circuit_helper<Statevector::State<QV::QubitVector<float>>>(
            circ, noise, config, shots, rng_seed);
      }
    }
    case Method::statevector_thrust_gpu: {
#ifdef AER_THRUST_CUDA
      if (precision_ == Precision::double_precision) {
        // Double-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<double>>>(
            circ, noise, config, shots, rng_seed);
      } else {
        // Single-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<float>>>(
            circ, noise, config, shots, rng_seed);
      }
#else
      throw std::runtime_error(
          "StatevectorController: method statevector_gpu is not supported on "
          "this "
          "system");
#endif
    }
    case Method::statevector_thrust_cpu: {
#ifdef AER_THRUST_CPU
      if (precision_ == Precision::double_precision) {
        // Double-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<double>>>(
            circ, noise, config, shots, rng_seed);
      } else {
        // Single-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<float>>>(
            circ, noise, config, shots, rng_seed);
      }
#else
      throw std::runtime_error(
          "StatevectorController: method statevector_thrust is not supported "
          "on this "
          "system");
#endif
    }
    default:
      throw std::runtime_error(
          "StatevectorController:Invalid simulation method");
  }
}

template <class State_t>
ExperimentData StatevectorController::run_circuit_helper(
    const Circuit& circ, const Noise::NoiseModel& noise, const json_t& config,
    uint_t shots, uint_t rng_seed) const {
  // Initialize  state
  Statevector::State<> state;

  // Validate circuit and throw exception if invalid operations exist
  validate_state(state, circ, noise, true);

  // Check for custom initial state, and if so check it matches num qubits
  if (!initial_state_.empty()) {
    if (initial_state_.size() != 1ULL << circ.num_qubits) {
      uint_t num_qubits(std::log2(initial_state_.size()));
      std::stringstream msg;
      msg << "StatevectorController: " << num_qubits << "-qubit initial state ";
      msg << "cannot be used for a " << circ.num_qubits << "-qubit circuit.";
      throw std::runtime_error(msg.str());
    }
  }

  // Set config
  state.set_config(config);
  state.set_parallalization(parallel_state_update_);

  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  ExperimentData data;
  data.set_config(config);

  // Run single shot collecting measure data or snapshots
  if (initial_state_.empty())
    state.initialize_qreg(circ.num_qubits);
  else
    state.initialize_qreg(circ.num_qubits, initial_state_);
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(circ.ops, data, rng);
  state.add_creg_to_data(data);

  // Add final state to the data
  data.add_additional_data("statevector", state.qreg().vector());

  return data;
}

//-------------------------------------------------------------------------
}  // end namespace Simulator
//-------------------------------------------------------------------------
}  // end namespace AER
//-------------------------------------------------------------------------
#endif
