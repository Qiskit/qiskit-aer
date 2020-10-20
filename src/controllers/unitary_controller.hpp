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

#include "controller.hpp"
#include "simulators/unitary/unitary_state.hpp"
#include "transpile/fusion.hpp"

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
 * - "zero_threshold" (double): Threshold for truncating small values to
 *      zero in result data [Default: 1e-10]
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
  UnitaryController();

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  // Allowed config options:
  // - "initial_unitary: complex_matrix"
  // Plus Base Controller config options
  virtual void set_config(const json_t &config) override;

  // Clear the current config
  void virtual clear_config() override;

 protected:
  size_t required_memory_mb(const Circuit &circ,
                            const Noise::NoiseModel &noise) const override;

  // Simulation methods for the Unitary Controller
  enum class Method {
    automatic,
    unitary_cpu,
    unitary_thrust_gpu,
    unitary_thrust_cpu
  };

  // Simulation precision
  enum class Precision { double_precision, single_precision };

 private:
  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // This simulator will only return a single shot, regardless of the
  // input shot number
  virtual void run_circuit(const Circuit &circ,
                           const Noise::NoiseModel &noise,
                           const json_t &config, uint_t shots,
                           uint_t rng_seed, ExperimentResult &result) const override;

  template <class State_t>
  void run_circuit_helper(const Circuit &circ,
                          const Noise::NoiseModel &noise,
                          const json_t &config, uint_t shots,
                          uint_t rng_seed, ExperimentResult &result) const;

  //-----------------------------------------------------------------------
  // Custom initial state
  //-----------------------------------------------------------------------
  cmatrix_t initial_unitary_;

  // Method to construct a unitary matrix
  Method method_ = Method::automatic;

  // Precision of a unitary matrix
  Precision precision_ = Precision::double_precision;
};

//=========================================================================
// Implementation
//=========================================================================

UnitaryController::UnitaryController() : Base::Controller() {
  // Disable qubit truncation by default
  Base::Controller::truncate_qubits_ = false;
}

//-------------------------------------------------------------------------
// Config
//-------------------------------------------------------------------------

void UnitaryController::set_config(const json_t &config) {
  // Set base controller config
  Base::Controller::set_config(config);

  // Override max parallel shots to be 1 since this should only be used
  // for single shot simulations
  Base::Controller::max_parallel_shots_ = 1;

  // Add custom initial unitary
  if (JSON::get_value(initial_unitary_, "initial_unitary", config)) {
    // Check initial state is unitary
    if (!Utils::is_unitary(initial_unitary_, validation_threshold_))
      throw std::runtime_error(
          "UnitaryController: initial_unitary is not unitary");
  }

  // Add method
  std::string method;
  if (JSON::get_value(method, "method", config)) {
    if (method == "unitary" || method == "unitary_cpu") {
      method_ = Method::unitary_cpu;
    } else if (method == "unitary_gpu") {
      method_ = Method::unitary_thrust_gpu;
    } else if (method == "unitary_thrust") {
      method_ = Method::unitary_thrust_cpu;
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

void UnitaryController::clear_config() {
  Base::Controller::clear_config();
  initial_unitary_ = cmatrix_t();
}

size_t UnitaryController::required_memory_mb(
    const Circuit &circ, const Noise::NoiseModel &noise) const {
  if (precision_ == Precision::single_precision) {
    QubitUnitary::State<QV::UnitaryMatrix<float>> state;
    return state.required_memory_mb(circ.num_qubits, circ.ops);
  } else {
    QubitUnitary::State<> state;
    return state.required_memory_mb(circ.num_qubits, circ.ops);
  }
}

//-------------------------------------------------------------------------
// Run circuit
//-------------------------------------------------------------------------

void UnitaryController::run_circuit(const Circuit &circ,
                                    const Noise::NoiseModel &noise,
                                    const json_t &config,
                                    uint_t shots,
                                    uint_t rng_seed, ExperimentResult &result) const {
  switch (method_) {
    case Method::automatic:
    case Method::unitary_cpu: {
      if (precision_ == Precision::double_precision) {
        // Double-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrix<double>>>(circ, noise, config,
                                                            shots, rng_seed, result);
      } else {
        // Single-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrix<float>>>(circ, noise, config,
                                                           shots, rng_seed, result);
      }
    }
    case Method::unitary_thrust_gpu: {
#ifdef AER_THRUST_CUDA
      if (precision_ == Precision::double_precision) {
        // Double-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrixThrust<double>>>(
            circ, noise, config, shots, rng_seed, result);
      } else {
        // Single-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrixThrust<float>>>(
            circ, noise, config, shots, rng_seed, result);
      }
#else
      throw std::runtime_error(
          "UnitaryController: method unitary_gpu is not supported on this "
          "system");
#endif
    }
    case Method::unitary_thrust_cpu: {
#ifdef AER_THRUST_CPU
      if (precision_ == Precision::double_precision) {
        // Double-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrixThrust<double>>>(
            circ, noise, config, shots, rng_seed, result);
      } else {
        // Single-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrixThrust<float>>>(
            circ, noise, config, shots, rng_seed, result);
      }
#else
      throw std::runtime_error(
          "UnitaryController: method unitary_thrust is not supported on this "
          "system");
#endif
    }
    default:
      throw std::runtime_error("UnitaryController:Invalid simulation method");
  }
}

template <class State_t>
void UnitaryController::run_circuit_helper(
    const Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
    uint_t shots, uint_t rng_seed, ExperimentResult &result) const {
  // Initialize state
  State_t state;

  // Validate circuit and throw exception if invalid operations exist
  validate_state(state, circ, noise, true);

  // Validate memory requirements and throw exception if not enough memory
  validate_memory_requirements(state, circ, true);

  // Check for custom initial state, and if so check it matches num qubits
  if (!initial_unitary_.empty()) {
    auto nrows = initial_unitary_.GetRows();
    auto ncols = initial_unitary_.GetColumns();
    if (nrows != ncols) {
      throw std::runtime_error(
          "UnitaryController: initial unitary is not square.");
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
  state.set_config(config);
  state.set_parallalization(parallel_state_update_);
  state.set_global_phase(circ.global_phase_angle);

  // Rng engine (not actually needed for unitary controller)
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  result.set_config(config);
  result.add_metadata("method", state.name());

  // Optimize circuit
  const std::vector<Operations::Op>* op_ptr = &circ.ops;
  Transpile::Fusion fusion_pass;
  fusion_pass.threshold /= 2;  // Halve default threshold for unitary simulator
  fusion_pass.set_config(config);
  Circuit opt_circ;
  if (fusion_pass.active && circ.num_qubits >= fusion_pass.threshold) {
    opt_circ = circ; // copy circuit
    Noise::NoiseModel dummy_noise; // dummy object for transpile pass
    fusion_pass.optimize_circuit(opt_circ, dummy_noise, state.opset(), result);
    op_ptr = &opt_circ.ops;
  }

  // Run single shot collecting measure data or snapshots
  if (initial_unitary_.empty()) {
    state.initialize_qreg(circ.num_qubits);
  } else {
    state.initialize_qreg(circ.num_qubits, initial_unitary_);
  }
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(*op_ptr, result, rng);
  state.add_creg_to_data(result);

  // Add final state unitary to the data
  result.data.add_additional_data("unitary", state.qreg().move_to_matrix());
}

//-------------------------------------------------------------------------
}  // end namespace Simulator
//-------------------------------------------------------------------------
}  // end namespace AER
//-------------------------------------------------------------------------
#endif
