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
#include "simulators/statevector/statevector_state_chunk.hpp"
#include "transpile/fusion.hpp"
#include "transpile/cacheblocking.hpp"

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
 *      level [Default: 14]
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
  virtual void run_circuit(const Circuit& circ,
                           const Noise::NoiseModel& noise,
                           const json_t& config, uint_t shots,
                           uint_t rng_seed,
                           ExperimentResult &result) const override;

  // Execute n-shots of a circuit on the input state
  template <class State_t>
  void run_circuit_helper(const Circuit& circ,
                          const Noise::NoiseModel& noise,
                          const json_t& config, uint_t shots,
                          uint_t rng_seed,
                          ExperimentResult &result) const;
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

void StatevectorController::run_circuit(
    const Circuit& circ, const Noise::NoiseModel& noise, const json_t& config,
    uint_t shots, uint_t rng_seed, ExperimentResult &result) const {
  switch (method_) {
    case Method::automatic:
    case Method::statevector_cpu: {
      if(Base::Controller::multiple_chunk_required(circ,noise)){
        if (precision_ == Precision::double_precision) {
          // Double-precision Statevector simulation
          return run_circuit_helper<StatevectorChunk::State<QV::QubitVector<double>>>(
              circ, noise, config, shots, rng_seed, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<StatevectorChunk::State<QV::QubitVector<float>>>(
              circ, noise, config, shots, rng_seed, result);
        }
      }
      else{
        if (precision_ == Precision::double_precision) {
          // Double-precision Statevector simulation
          return run_circuit_helper<Statevector::State<QV::QubitVector<double>>>(
              circ, noise, config, shots, rng_seed, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<Statevector::State<QV::QubitVector<float>>>(
              circ, noise, config, shots, rng_seed, result);
        }
      }
    }
    case Method::statevector_thrust_gpu: {
#ifdef AER_THRUST_CUDA
      if(Base::Controller::multiple_chunk_required(circ,noise)){
        if (precision_ == Precision::double_precision) {
          // Double-precision Statevector simulation
          return run_circuit_helper<
              StatevectorChunk::State<QV::QubitVectorThrust<double>>>(
              circ, noise, config, shots, rng_seed, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<
              StatevectorChunk::State<QV::QubitVectorThrust<float>>>(
              circ, noise, config, shots, rng_seed, result);
        }
      }
      else{
        if (precision_ == Precision::double_precision) {
          // Double-precision Statevector simulation
          return run_circuit_helper<
              Statevector::State<QV::QubitVectorThrust<double>>>(
              circ, noise, config, shots, rng_seed, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<
              Statevector::State<QV::QubitVectorThrust<float>>>(
              circ, noise, config, shots, rng_seed, result);
        }
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
      if(Base::Controller::multiple_chunk_required(circ,noise)){
        if (precision_ == Precision::double_precision) {
          // Double-precision Statevector simulation
          return run_circuit_helper<
              StatevectorChunk::State<QV::QubitVectorThrust<double>>>(
              circ, noise, config, shots, rng_seed, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<
              StatevectorChunk::State<QV::QubitVectorThrust<float>>>(
              circ, noise, config, shots, rng_seed, result);
        }
      }
      else{
        if (precision_ == Precision::double_precision) {
          // Double-precision Statevector simulation
          return run_circuit_helper<
              Statevector::State<QV::QubitVectorThrust<double>>>(
              circ, noise, config, shots, rng_seed, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<
              Statevector::State<QV::QubitVectorThrust<float>>>(
              circ, noise, config, shots, rng_seed, result);
        }
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
void StatevectorController::run_circuit_helper(
    const Circuit& circ, const Noise::NoiseModel& noise, const json_t& config,
    uint_t shots, uint_t rng_seed, ExperimentResult &result) const 
{
  // Initialize  state
  State_t state;

  // Validate circuit and throw exception if invalid operations exist
  validate_state(state, circ, noise, true);

  // Validate memory requirements and throw exception if not enough memory
  validate_memory_requirements(state, circ, true);

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
  state.set_distribution(Base::Controller::num_process_per_experiment_);
  state.set_global_phase(circ.global_phase_angle);

  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  result.set_config(config);

  // Optimize circuit
  Transpile::Fusion fusion_pass;
  fusion_pass.set_config(config);
  fusion_pass.set_parallelization(parallel_state_update_);

  Circuit opt_circ = circ; // copy circuit
  Noise::NoiseModel dummy_noise; // dummy object for transpile pass
  if (fusion_pass.active && circ.num_qubits >= fusion_pass.threshold) {
    fusion_pass.optimize_circuit(opt_circ, dummy_noise, state.opset(), result);
  }

  Transpile::CacheBlocking cache_block_pass = transpile_cache_blocking(opt_circ,dummy_noise,config,(precision_ == Precision::single_precision) ? sizeof(std::complex<float>) : sizeof(std::complex<double>),false);
  cache_block_pass.set_save_state(true);
  cache_block_pass.optimize_circuit(opt_circ, dummy_noise, state.opset(), result);

  uint_t block_bits = 0;
  if(cache_block_pass.enabled())
    block_bits = cache_block_pass.block_bits();
  state.allocate(Base::Controller::max_qubits_,block_bits);

  // Run single shot collecting measure data or snapshots
  if (initial_state_.empty()) {
    state.initialize_qreg(circ.num_qubits);
  } else {
    state.initialize_qreg(circ.num_qubits, initial_state_);
  }
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(opt_circ.ops, result, rng);
  Base::Controller::save_count_data(result, state.creg());

  // Add final state to the data
  state.save_data_single(result, "statevector", state.move_to_vector());
}

//-------------------------------------------------------------------------
}  // end namespace Simulator
//-------------------------------------------------------------------------
}  // end namespace AER
//-------------------------------------------------------------------------
#endif
