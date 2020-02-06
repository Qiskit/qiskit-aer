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

#ifndef _aer_qasm_controller_hpp_
#define _aer_qasm_controller_hpp_

#include "controller.hpp"
#include "simulators/density_matrix/densitymatrix_state.hpp"
#include "simulators/extended_stabilizer/extended_stabilizer_state.hpp"
#include "simulators/matrix_product_state/matrix_product_state.hpp"
#include "simulators/stabilizer/stabilizer_state.hpp"
#include "simulators/statevector/statevector_state.hpp"
#include "simulators/superoperator/superoperator_state.hpp"
#include "transpile/basic_opts.hpp"
#include "transpile/delay_measure.hpp"
#include "transpile/fusion.hpp"

namespace AER {
namespace Simulator {

//=========================================================================
// QasmController class
//=========================================================================

/**************************************************************************
 * Config settings:
 * - "optimize_ideal_threshold" (int): Qubit threshold for running circuit
 *   optimizations passes for an ideal circuit [Default: 0].
 * - "optimize_noise_threshold" (int): Qubit threshold for running circuit
 *   optimizations passes for a noisy circuit [Default: 12].
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
 * From ExtendedStabilizer::State class
 * - "extended_stabilizer_approximation_error" (double): Set the error in the
 *     approximation for the ch method. A smaller error needs more
 *     memory and computational time. [Default: 0.05]
 *
 * - "extended_stabilizer_disable_measurement_opt" (bool): Force the simulator
 *to re-run the monte-carlo step for every measurement. Enabling this will
 *improve the sampling accuracy if the output distribution is strongly peaked,
 *but requires more computational time. [Default: True]
 *
 * - "extended_stabilizer_mixing_time" (int): Set how long the monte-carlo
 *method runs before performing measurements. If the output distribution is
 *strongly peaked, this can be decreased alongside setting
 *extended_stabilizer_disable_measurement_opt to True. [Default: 5000]
 *
 * - "extended_stabilizer_norm_estimation_samples" (int): Number of samples used
 *to compute the correct normalisation for a statevector snapshot. [Default:
 *100]
 *
 * - "extended_stabilizer_parallel_threshold" (int): Set the minimum size of the
 *ch decomposition before we enable OpenMP parallelisation. If parallel circuit
 *or shot execution is enabled this will only use unallocated CPU cores up to
 *max_parallel_threads. [Default: 100]
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
 * - "max_memory_mb" (int): Memory in MB available to the state class.
 *      If specified, is divided by the number of parallel shots/experiments.
 *      [Default: 0]
 *
 * From Transpile:Fision Class
 * - fusion_enable (bool): Enable fusion optimization in circuit optimization
 *       passes [Default: True]
 * - fusion_verbose (bool): Output gates generated in fusion optimization
 *       into metadata [Default: False]
 * - fusion_max_qubit (int): Maximum number of qubits for a operation generated
 *       in a fusion optimization [Default: 5]
 * - fusion_threshold (int): Threshold that number of qubits must be greater
 *       than or equal to enable fusion optimization [Default: 20]
 *
 **************************************************************************/

class QasmController : public Base::Controller {
 public:
  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  QasmController();

  //-----------------------------------------------------------------------
  // Base class config override
  //-----------------------------------------------------------------------

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  // Allowed config options:
  // - "initial_statevector: complex_vector"
  // Plus Base Controller config options
  virtual void set_config(const json_t &config) override;

  // Clear the current config
  void virtual clear_config() override;

 protected:
  //-----------------------------------------------------------------------
  // Simulation types
  //-----------------------------------------------------------------------

  // Simulation methods for the Qasm Controller
  enum class Method {
    automatic,
    statevector,
    statevector_thrust_gpu,
    statevector_thrust_cpu,
    density_matrix,
    density_matrix_thrust_gpu,
    density_matrix_thrust_cpu,
    stabilizer,
    extended_stabilizer,
    matrix_product_state
  };

  // Simulation precision
  enum class Precision { double_precision, single_precision };

  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  virtual ExperimentData run_circuit(const Circuit &circ,
                                     const Noise::NoiseModel &noise,
                                     const json_t &config, uint_t shots,
                                     uint_t rng_seed) const override;

  //----------------------------------------------------------------
  // Utility functions
  //----------------------------------------------------------------

  // Return the simulation method to use for the input circuit
  // If a custom method is specified in the config this will be
  // used. If the default automatic method is set this will choose
  // the appropriate method based on the input circuit.
  Method simulation_method(const Circuit &circ, const Noise::NoiseModel &noise,
                           bool validate = false) const;

  // Initialize a State subclass to a given initial state
  template <class State_t, class Initstate_t>
  void initialize_state(const Circuit &circ, State_t &state,
                        const Initstate_t &initial_state) const;

  // Set parallelization for qasm simulator
  virtual void set_parallelization_circuit(
      const Circuit &circ, const Noise::NoiseModel &noise) override;

  //----------------------------------------------------------------
  // Run circuit helpers
  //----------------------------------------------------------------

  // Execute n-shots of a circuit on the input state
  template <class State_t, class Initstate_t>
  ExperimentData run_circuit_helper(const Circuit &circ,
                                    const Noise::NoiseModel &noise,
                                    const json_t &config, uint_t shots,
                                    uint_t rng_seed,
                                    const Initstate_t &initial_state,
                                    const Method method) const;

  // Execute a single shot a circuit by initializing the state vector
  // to initial_state, running all ops in circ, and updating data with
  // simulation output.
  template <class State_t, class Initstate_t>
  void run_single_shot(const Circuit &circ, State_t &state,
                       const Initstate_t &initial_state, ExperimentData &data,
                       RngEngine &rng) const;

  // Execute a n-shots of a circuit without noise.
  // If possible this is done using measure sampling to only simulate
  // a single shot up to the first measurement, then sampling measure
  // outcomes for each shot.
  template <class State_t, class Initstate_t>
  void run_circuit_without_noise(const Circuit &circ, uint_t shots,
                                 State_t &state,
                                 const Initstate_t &initial_state,
                                 const Method method, ExperimentData &data,
                                 RngEngine &rng) const;

  // Execute n-shots of a circuit with noise by sampling a new noisy
  // instance of the circuit for each shot.
  template <class State_t, class Initstate_t>
  void run_circuit_with_noise(const Circuit &circ,
                              const Noise::NoiseModel &noise, uint_t shots,
                              State_t &state, const Initstate_t &initial_state,
                              ExperimentData &data, RngEngine &rng) const;

  //----------------------------------------------------------------
  // Measure sampling optimization
  //----------------------------------------------------------------

  // Sample measurement outcomes for the input measure ops from the
  // current state of the input State_t
  template <class State_t>
  void measure_sampler(const std::vector<Operations::Op> &meas_ops,
                       uint_t shots, State_t &state, ExperimentData &data,
                       RngEngine &rng) const;

  // Check if measure sampling optimization is valid for the input circuit
  // if so return a pair {true, pos} where pos is the position of the
  // first measurement operation in the input circuit
  std::pair<bool, size_t> check_measure_sampling_opt(const Circuit &circ,
                                                     const Method method) const;

  //-----------------------------------------------------------------------
  // Config
  //-----------------------------------------------------------------------
  size_t required_memory_mb(const Circuit &circ,
                            const Noise::NoiseModel &noise) const override;

  // Simulation method
  Method simulation_method_ = Method::automatic;

  // Simulation precision
  Precision simulation_precision_ = Precision::double_precision;

  // Qubit threshold for running circuit optimizations
  uint_t circuit_opt_ideal_threshold_ = 0;
  uint_t circuit_opt_noise_threshold_ = 12;

  // Initial statevector for Statevector simulation method
  cvector_t initial_statevector_;

  // TODO: initial stabilizer state

  // Controller-level parameter for CH method
  bool extended_stabilizer_measure_sampling_ = false;
};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Constructor
//-------------------------------------------------------------------------
QasmController::QasmController() {
  add_circuit_optimization(Transpile::ReduceBarrier());
  add_circuit_optimization(Transpile::DelayMeasure());
  add_circuit_optimization(Transpile::Fusion());
}

//-------------------------------------------------------------------------
// Config
//-------------------------------------------------------------------------

void QasmController::set_config(const json_t &config) {
  // Set base controller config
  Base::Controller::set_config(config);

  // Override automatic simulation method with a fixed method
  std::string method;
  if (JSON::get_value(method, "method", config)) {
    if (method == "statevector" || method == "statevector_cpu") {
      simulation_method_ = Method::statevector;
    } else if (method == "statevector_gpu") {
      simulation_method_ = Method::statevector_thrust_gpu;
    } else if (method == "statevector_thrust") {
      simulation_method_ = Method::statevector_thrust_cpu;
    } else if (method == "density_matrix" || method == "density_matrix_cpu") {
      simulation_method_ = Method::density_matrix;
    } else if (method == "density_matrix_gpu") {
      simulation_method_ = Method::density_matrix_thrust_gpu;
    } else if (method == "density_matrix_thrust") {
      simulation_method_ = Method::density_matrix_thrust_cpu;
    } else if (method == "stabilizer") {
      simulation_method_ = Method::stabilizer;
    } else if (method == "extended_stabilizer") {
      simulation_method_ = Method::extended_stabilizer;
    } else if (method == "matrix_product_state") {
      simulation_method_ = Method::matrix_product_state;
    } else if (method != "automatic") {
      throw std::runtime_error(
          std::string("QasmController: Invalid simulation method (") + method +
          std::string(")."));
    }
  }

  std::string precision;
  if (JSON::get_value(precision, "precision", config)) {
    if (precision == "double") {
      simulation_precision_ = Precision::double_precision;
    } else if (precision == "single") {
      simulation_precision_ = Precision::single_precision;
    }
  }

  // Check for circuit optimization threshold
  JSON::get_value(circuit_opt_ideal_threshold_, "optimize_ideal_threshold",
                  config);
  JSON::get_value(circuit_opt_noise_threshold_, "optimize_noise_threshold",
                  config);

  // Check for extended stabilizer measure sampling
  JSON::get_value(extended_stabilizer_measure_sampling_,
                  "extended_stabilizer_measure_sampling", config);

  // DEPRECATED: Add custom initial state
  if (JSON::get_value(initial_statevector_, "initial_statevector", config)) {
    // Raise error if method is set to stabilizer or ch
    if (simulation_method_ == Method::stabilizer) {
      throw std::runtime_error(
          std::string("QasmController: Using an initial statevector") +
          std::string(" is not valid with stabilizer simulation method.") +
          method);
    } else if (simulation_method_ == Method::extended_stabilizer) {
      throw std::runtime_error(
          std::string("QasmController: Using an initial statevector") +
          std::string(" is not valid with the CH simulation method.") + method);
    }
    // Override simulator method to statevector
    simulation_method_ = Method::statevector;
    // Check initial state is normalized
    if (!Utils::is_unit_vector(initial_statevector_, validation_threshold_)) {
      throw std::runtime_error(
          "QasmController: initial_statevector is not a unit vector");
    }
  }
}

void QasmController::clear_config() {
  Base::Controller::clear_config();
  simulation_method_ = Method::automatic;
  initial_statevector_ = cvector_t();
}

//-------------------------------------------------------------------------
// Base class override
//-------------------------------------------------------------------------

ExperimentData QasmController::run_circuit(const Circuit &circ,
                                           const Noise::NoiseModel &noise,
                                           const json_t &config, uint_t shots,
                                           uint_t rng_seed) const {
  // Validate circuit for simulation method
  switch (simulation_method(circ, noise, true)) {
    case Method::statevector:
      if (simulation_precision_ == Precision::double_precision) {
        // Double-precision Statevector simulation
        return run_circuit_helper<Statevector::State<QV::QubitVector<double>>>(
            circ, noise, config, shots, rng_seed, initial_statevector_,
            Method::statevector);
      } else {
        // Single-precision Statevector simulation
        return run_circuit_helper<Statevector::State<QV::QubitVector<float>>>(
            circ, noise, config, shots, rng_seed, initial_statevector_,
            Method::statevector);
      }
    case Method::statevector_thrust_gpu:
#ifndef AER_THRUST_CUDA
      throw std::runtime_error(
          "QasmController: method statevector_gpu is not supported on this "
          "system");
#else
      if (simulation_precision_ == Precision::double_precision) {
        // Double-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<double>>>(
            circ, noise, config, shots, rng_seed, initial_statevector_,
            Method::statevector_thrust_gpu);
      } else {
        // Single-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<float>>>(
            circ, noise, config, shots, rng_seed, initial_statevector_,
            Method::statevector_thrust_gpu);
      }
#endif
    case Method::statevector_thrust_cpu:
#ifndef AER_THRUST_CPU
      throw std::runtime_error(
          "QasmController: method statevector_thrust is not supported on this "
          "system");
#else
      if (simulation_precision_ == Precision::double_precision) {
        // Double-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<double>>>(
            circ, noise, config, shots, rng_seed, initial_statevector_,
            Method::statevector_thrust_cpu);
      } else {
        // Single-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<float>>>(
            circ, noise, config, shots, rng_seed, initial_statevector_,
            Method::statevector_thrust_cpu);
      }
#endif
    case Method::density_matrix:
      if (simulation_precision_ == Precision::double_precision) {
        // Double-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrix<double>>>(
            circ, noise, config, shots, rng_seed, cvector_t(),
            Method::density_matrix);
      } else {
        // Single-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrix<float>>>(
            circ, noise, config, shots, rng_seed, cvector_t(),
            Method::density_matrix);
      }
    case Method::density_matrix_thrust_gpu:
#ifndef AER_THRUST_CUDA
      throw std::runtime_error(
          "QasmController: method density_matrix_gpu is not supported on this "
          "system");
#else
      if (simulation_precision_ == Precision::double_precision) {
        // Double-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrixThrust<double>>>(
            circ, noise, config, shots, rng_seed, cvector_t(),
            Method::density_matrix_thrust_gpu);
      } else {
        // Single-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrixThrust<float>>>(
            circ, noise, config, shots, rng_seed, cvector_t(),
            Method::density_matrix_thrust_gpu);
      }
#endif
    case Method::density_matrix_thrust_cpu:
#ifndef AER_THRUST_CPU
      throw std::runtime_error(
          "QasmController: method density_matrix_thrust is not supported on this "
          "system");
#else
      if (simulation_precision_ == Precision::double_precision) {
        // Double-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrixThrust<double>>>(
            circ, noise, config, shots, rng_seed, cvector_t(),
            Method::density_matrix_thrust_cpu);
      } else {
        // Single-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrixThrust<float>>>(
            circ, noise, config, shots, rng_seed, cvector_t(),
            Method::density_matrix_thrust_cpu);
      }
#endif
    case Method::stabilizer:
      // Stabilizer simulation
      // TODO: Stabilizer doesn't yet support custom state initialization
      return run_circuit_helper<Stabilizer::State>(
          circ, noise, config, shots, rng_seed, Clifford::Clifford(),
          Method::stabilizer);
    case Method::extended_stabilizer:
      return run_circuit_helper<ExtendedStabilizer::State>(
          circ, noise, config, shots, rng_seed, CHSimulator::Runner(),
          Method::extended_stabilizer);

    case Method::matrix_product_state:
      return run_circuit_helper<MatrixProductState::State>(
          circ, noise, config, shots, rng_seed, MatrixProductState::MPS(),
          Method::matrix_product_state);

    default:
      throw std::runtime_error("QasmController:Invalid simulation method");
  }
}

//-------------------------------------------------------------------------
// Utility methods
//-------------------------------------------------------------------------

QasmController::Method QasmController::simulation_method(
    const Circuit &circ, const Noise::NoiseModel &noise_model,
    bool validate) const {
  // Check simulation method and validate state
  switch (simulation_method_) {
    case Method::statevector: {
      if (validate) {
        if (simulation_precision_ == Precision::single_precision) {
          Statevector::State<QV::QubitVector<float>> state;
          validate_state(state, circ, noise_model, true);
        } else {
          Statevector::State<QV::QubitVector<>> state;
          validate_state(state, circ, noise_model, true);
        }
      }
      return Method::statevector;
    }
    case Method::statevector_thrust_gpu: {
#ifndef AER_THRUST_CUDA
      throw std::runtime_error(
          "QasmController: method statevector_gpu is not supported on this "
          "system");
#else
      if (validate) {
        if (simulation_precision_ == Precision::single_precision) {
          Statevector::State<QV::QubitVectorThrust<float>> state;
          validate_state(state, circ, noise_model, true);
        } else {
          Statevector::State<QV::QubitVectorThrust<>> state;
          validate_state(state, circ, noise_model, true);
        }
      }
      return Method::statevector_thrust_gpu;
#endif
    }
    case Method::statevector_thrust_cpu: {
#ifndef AER_THRUST_CPU
      throw std::runtime_error(
          "QasmController: method statevector_thrust is not supported on this "
          "system");
#else
      if (validate) {
        if (simulation_precision_ == Precision::single_precision) {
          Statevector::State<QV::QubitVectorThrust<float>> state;
          validate_state(state, circ, noise_model, true);
        } else {
          Statevector::State<QV::QubitVectorThrust<>> state;
          validate_state(state, circ, noise_model, true);
        }
      }
      return Method::statevector_thrust_cpu;
#endif
    }
    case Method::density_matrix: {
      if (validate) {
        if (simulation_precision_ == Precision::single_precision) {
          validate_state(DensityMatrix::State<QV::DensityMatrix<float>>(), circ,
                         noise_model, true);
        } else {
          validate_state(DensityMatrix::State<QV::DensityMatrix<double>>(),
                         circ, noise_model, true);
        }
      }
      return Method::density_matrix;
    }
    case Method::density_matrix_thrust_gpu: {
#ifndef AER_THRUST_SUPPORTED
      throw std::runtime_error(
          "QasmController: method density_matrix_gpu is not supported on this "
          "system");
#else
      if (validate) {
        if (simulation_precision_ == Precision::single_precision) {
          validate_state(DensityMatrix::State<QV::DensityMatrixThrust<float>>(),
                         circ, noise_model, true);
        } else {
          validate_state(
              DensityMatrix::State<QV::DensityMatrixThrust<double>>(), circ,
              noise_model, true);
        }
      }
      return Method::density_matrix_thrust_gpu;
#endif
    }
    case Method::density_matrix_thrust_cpu: {
#ifndef AER_THRUST_SUPPORTED
      throw std::runtime_error(
          "QasmController: method density_matrix_thrust is not supported on this "
          "system");
#else
      if (validate) {
        if (simulation_precision_ == Precision::single_precision) {
          validate_state(DensityMatrix::State<QV::DensityMatrixThrust<float>>(),
                         circ, noise_model, true);
        } else {
          validate_state(
              DensityMatrix::State<QV::DensityMatrixThrust<double>>(), circ,
              noise_model, true);
        }
      }
      return Method::density_matrix_thrust_cpu;
#endif
    }
    case Method::stabilizer: {
      if (validate)
        validate_state(Stabilizer::State(), circ, noise_model, true);
      return Method::stabilizer;
    }
    case Method::extended_stabilizer: {
      if (validate)
        validate_state(ExtendedStabilizer::State(), circ, noise_model, true);
      return Method::extended_stabilizer;
    }
    case Method::matrix_product_state: {
      if (validate)
        validate_state(MatrixProductState::State(), circ, noise_model, true);
      return Method::matrix_product_state;
    }
    case Method::automatic: {
      // If circuit and noise model are Clifford run on Stabilizer simulator
      if (validate_state(Stabilizer::State(), circ, noise_model, false)) {
        return Method::stabilizer;
      }
      // For noisy simulations we enable the density matrix method if
      // shots > 2 ** num_qubits. This is based on a rough estimate that
      // a single shot of the density matrix simulator is approx 2 ** nq
      // times slow than a single shot of statevector due the increased
      // dimension
      if (noise_model.has_quantum_errors() &&
          circ.shots > (1 << circ.num_qubits) &&
          validate_memory_requirements(DensityMatrix::State<>(), circ, false) &&
          validate_state(DensityMatrix::State<>(), circ, noise_model, false) &&
          check_measure_sampling_opt(circ, Method::density_matrix).first) {
        return Method::density_matrix;
      }
      // Finally we check the statevector memory requirement for the
      // current number of qubits. If it fits in available memory we
      // default to the Statevector method. Otherwise we attempt to use
      // the extended stabilizer simulator.
      bool enough_memory = true;
      if (simulation_precision_ == Precision::single_precision) {
        Statevector::State<QV::QubitVector<float>> sv_state;
        enough_memory = validate_memory_requirements(sv_state, circ, false);
      } else {
        Statevector::State<> sv_state;
        enough_memory = validate_memory_requirements(sv_state, circ, false);
      }
      if (!enough_memory) {
        if (validate_state(ExtendedStabilizer::State(), circ, noise_model,
                           false)) {
          return Method::extended_stabilizer;
        } else {
          throw std::runtime_error(
              "QasmSimulator: Circuit cannot be run using available methods.");
        }
      }
    }
    // If we didn't select extended stabilizer above proceed to the default
    // switch clause
    default: {
      // For default we use statevector followed by density matrix (for the case
      // when the circuit contains invalid instructions for statevector)
      if (validate_state(Statevector::State<>(), circ, noise_model, false)) {
        return Method::statevector;
      }
      // If circuit contains invalid instructions for statevector throw a hail
      // mary and try for density matrix.
      if (validate)
        validate_state(DensityMatrix::State<>(), circ, noise_model, true);
      return Method::density_matrix;
    }
  }
}

template <class State_t, class Initstate_t>
void QasmController::initialize_state(const Circuit &circ, State_t &state,
                                      const Initstate_t &initial_state) const {
  if (initial_state.empty()) {
    state.initialize_qreg(circ.num_qubits);
  } else {
    state.initialize_qreg(circ.num_qubits, initial_state);
  }
  state.initialize_creg(circ.num_memory, circ.num_registers);
}

size_t QasmController::required_memory_mb(
    const Circuit &circ, const Noise::NoiseModel &noise) const {
  switch (simulation_method(circ, noise, false)) {
    case Method::statevector:
    case Method::statevector_thrust_cpu:
    case Method::statevector_thrust_gpu: {
      if (simulation_precision_ == Precision::single_precision) {
        Statevector::State<QV::QubitVector<float>> state;
        return state.required_memory_mb(circ.num_qubits, circ.ops);
      } else {
        Statevector::State<> state;
        return state.required_memory_mb(circ.num_qubits, circ.ops);
      }
    }
    case Method::density_matrix:
    case Method::density_matrix_thrust_cpu:
    case Method::density_matrix_thrust_gpu: {
      if (simulation_precision_ == Precision::single_precision) {
        DensityMatrix::State<QV::DensityMatrix<float>> state;
        return state.required_memory_mb(circ.num_qubits, circ.ops);
      } else {
        DensityMatrix::State<> state;
        return state.required_memory_mb(circ.num_qubits, circ.ops);
      }
    }
    case Method::stabilizer: {
      Stabilizer::State state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
    case Method::extended_stabilizer: {
      ExtendedStabilizer::State state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
    case Method::matrix_product_state: {
      MatrixProductState::State state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
    default:
      // We shouldn't get here, so throw an exception if we do
      throw std::runtime_error("QasmController: Invalid simulation method");
  }
}

void QasmController::set_parallelization_circuit(
    const Circuit &circ, const Noise::NoiseModel &noise_model) {
  const auto method = simulation_method(circ, noise_model, false);
  switch (method) {
    case Method::statevector:
    case Method::statevector_thrust_gpu:
    case Method::statevector_thrust_cpu:
    case Method::stabilizer:
    case Method::matrix_product_state: {
      if ((noise_model.is_ideal() || !noise_model.has_quantum_errors()) &&
          check_measure_sampling_opt(circ, Method::statevector).first) {
        parallel_shots_ = 1;
        parallel_state_update_ =
            std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
        return;
      }
      Base::Controller::set_parallelization_circuit(circ, noise_model);
      break;
    }
    case Method::density_matrix:
    case Method::density_matrix_thrust_gpu:
    case Method::density_matrix_thrust_cpu:{
      if (check_measure_sampling_opt(circ, Method::density_matrix).first) {
        parallel_shots_ = 1;
        parallel_state_update_ =
            std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
        return;
      }
      Base::Controller::set_parallelization_circuit(circ, noise_model);
      break;
    }
    default: {
      Base::Controller::set_parallelization_circuit(circ, noise_model);
    }
  }
}

//-------------------------------------------------------------------------
// Run circuit helpers
//-------------------------------------------------------------------------

template <class State_t, class Initstate_t>
ExperimentData QasmController::run_circuit_helper(
    const Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
    uint_t shots, uint_t rng_seed, const Initstate_t &initial_state,
    const Method method) const {
  // Initialize new state object
  State_t state;

  // Check memory requirements, raise exception if they're exceeded
  validate_memory_requirements(state, circ, true);

  // Set state config
  state.set_config(config);
  state.set_parallalization(parallel_state_update_);

  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  ExperimentData data;
  data.set_config(config);
  data.add_metadata("method", state.name());
  // Add measure sampling to metadata
  // Note: this will set to `true` if sampling is enabled for the circuit
  data.add_metadata("measure_sampling", false);

  // Choose execution method based on noise and method
  if (noise.is_ideal()) {
    run_circuit_without_noise(circ, shots, state, initial_state, method, data,
                              rng);
  } else if ((method == Method::density_matrix ||
              method == Method::density_matrix_thrust_gpu ||
              method == Method::density_matrix_thrust_cpu) &&
             noise.has_quantum_errors()) {
    // We can sample the noise model using superoperator method
    // and then execute the resulting circuit containing superoperators
    Noise::NoiseModel noise_cpy = noise;
    noise_cpy.activate_superop_method();
    Circuit noise_circ = noise_cpy.sample_noise(circ, rng);
    run_circuit_without_noise(noise_circ, shots, state, initial_state, method,
                              data, rng);
  } else if (noise.has_quantum_errors() == false) {
    // We can insert the readout errors from the noise model and then
    // execute the resulting circuit
    Circuit noise_circ = noise.sample_noise(circ, rng);
    run_circuit_without_noise(noise_circ, shots, state, initial_state, method,
                              data, rng);
  } else {
    // Run sampling a noisy instance of the circuit for each shot
    run_circuit_with_noise(circ, noise, shots, state, initial_state, data, rng);
  }
  return data;
}

template <class State_t, class Initstate_t>
void QasmController::run_single_shot(const Circuit &circ, State_t &state,
                                     const Initstate_t &initial_state,
                                     ExperimentData &data,
                                     RngEngine &rng) const {
  initialize_state(circ, state, initial_state);
  state.apply_ops(circ.ops, data, rng);
  state.add_creg_to_data(data);
}

template <class State_t, class Initstate_t>
void QasmController::run_circuit_with_noise(const Circuit &circ,
                                            const Noise::NoiseModel &noise,
                                            uint_t shots, State_t &state,
                                            const Initstate_t &initial_state,
                                            ExperimentData &data,
                                            RngEngine &rng) const {
  // Sample a new noise circuit and optimize for each shot
  while (shots-- > 0) {
    Circuit noise_circ = noise.sample_noise(circ, rng);
    noise_circ.shots = 1;
    if (noise_circ.num_qubits > circuit_opt_noise_threshold_) {
      Noise::NoiseModel dummy;
      optimize_circuit(noise_circ, dummy, state, data);
    }
    run_single_shot(noise_circ, state, initial_state, data, rng);
  }
}

template <class State_t, class Initstate_t>
void QasmController::run_circuit_without_noise(const Circuit &circ,
                                               uint_t shots, State_t &state,
                                               const Initstate_t &initial_state,
                                               const Method method,
                                               ExperimentData &data,
                                               RngEngine &rng) const {
  // Optimize circuit for state type
  Circuit opt_circ = circ;
  if (opt_circ.num_qubits > circuit_opt_ideal_threshold_) {
    Noise::NoiseModel dummy;
    optimize_circuit(opt_circ, dummy, state, data);
  }
  // Check if measure sampler and optimization are valid
  auto check = check_measure_sampling_opt(opt_circ, method);
  if (check.first == false) {
    // Perform standard execution if we cannot apply the
    // measurement sampling optimization
    while (shots-- > 0) {
      run_single_shot(opt_circ, state, initial_state, data, rng);
    }
  } else {
    // Implement measure sampler
    auto pos = check.second;  // Position of first measurement op

    // Run circuit instructions before first measure
    std::vector<Operations::Op> ops(opt_circ.ops.begin(),
                                    opt_circ.ops.begin() + pos);
    initialize_state(opt_circ, state, initial_state);
    state.apply_ops(ops, data, rng);

    // Get measurement operations and set of measured qubits
    ops = std::vector<Operations::Op>(opt_circ.ops.begin() + pos,
                                      opt_circ.ops.end());
    measure_sampler(ops, shots, state, data, rng);
    // Add measure sampling metadata
    data.add_metadata("measure_sampling", true);
  }
}

//-------------------------------------------------------------------------
// Measure sampling optimization
//-------------------------------------------------------------------------

std::pair<bool, size_t> QasmController::check_measure_sampling_opt(
    const Circuit &circ, const Method method) const {
  // Find first instance of a measurement and check there
  // are no reset or initialize operations before the measurement
  if (method == Method::extended_stabilizer &&
      !extended_stabilizer_measure_sampling_) {
    return std::make_pair(false, 0);
  }
  auto start = circ.ops.begin();
  while (start != circ.ops.end()) {
    const auto type = start->type;
    if (method != Method::density_matrix &&
        method != Method::density_matrix_thrust_gpu &&
        method != Method::density_matrix_thrust_cpu) {
      if (type == Operations::OpType::reset ||
          type == Operations::OpType::initialize ||
          type == Operations::OpType::kraus ||
          type == Operations::OpType::superop) {
        return std::make_pair(false, 0);
      }
    }
    if (type == Operations::OpType::measure ||
        type == Operations::OpType::roerror)
      break;
    ++start;
  }
  // Record position for if optimization passes
  auto start_meas = start;
  // Check all remaining operations are measurements
  while (start != circ.ops.end()) {
    if ((start->type != Operations::OpType::measure &&
         start->type != Operations::OpType::roerror) ||
        start->conditional) {
      return std::make_pair(false, 0);
    }
    ++start;
  }
  // If we made it this far we can apply the optimization
  // size_t meas_pos = start_meas - circ.ops.begin();
  size_t meas_pos = std::distance(circ.ops.begin(), start_meas);
  return std::make_pair(true, meas_pos);
}

template <class State_t>
void QasmController::measure_sampler(
    const std::vector<Operations::Op> &meas_roerror_ops, uint_t shots,
    State_t &state, ExperimentData &data, RngEngine &rng) const {
  // Check if meas_circ is empty, and if so return initial creg
  if (meas_roerror_ops.empty()) {
    while (shots-- > 0) {
      state.add_creg_to_data(data);
    }
    return;
  }

  std::vector<Operations::Op> meas_ops;
  std::vector<Operations::Op> roerror_ops;
  for (const Operations::Op &op : meas_roerror_ops)
    if (op.type == Operations::OpType::roerror)
      roerror_ops.push_back(op);
    else /*(op.type == Operations::OpType::measure) */
      meas_ops.push_back(op);

  // Get measured qubits from circuit sort and delete duplicates
  std::vector<uint_t> meas_qubits;  // measured qubits
  for (const auto &op : meas_ops) {
    for (size_t j = 0; j < op.qubits.size(); ++j)
      meas_qubits.push_back(op.qubits[j]);
  }
  sort(meas_qubits.begin(), meas_qubits.end());
  meas_qubits.erase(unique(meas_qubits.begin(), meas_qubits.end()),
                    meas_qubits.end());
  // Generate the samples
  auto all_samples = state.sample_measure(meas_qubits, shots, rng);

  // Make qubit map of position in vector of measured qubits
  std::unordered_map<uint_t, uint_t> qubit_map;
  for (uint_t j = 0; j < meas_qubits.size(); ++j) {
    qubit_map[meas_qubits[j]] = j;
  }

  // Maps of memory and register to qubit position
  std::unordered_map<uint_t, uint_t> memory_map;
  std::unordered_map<uint_t, uint_t> register_map;
  for (const auto &op : meas_ops) {
    for (size_t j = 0; j < op.qubits.size(); ++j) {
      auto pos = qubit_map[op.qubits[j]];
      if (!op.memory.empty()) memory_map[op.memory[j]] = pos;
      if (!op.registers.empty()) register_map[op.registers[j]] = pos;
    }
  }

  // Process samples
  // Convert opts to circuit so we can get the needed creg sizes
  // NB: this function could probably be moved somewhere else like Utils or Ops
  Circuit meas_circ(meas_roerror_ops);
  ClassicalRegister creg;
  while (!all_samples.empty()) {
    auto sample = all_samples.back();
    creg.initialize(meas_circ.num_memory, meas_circ.num_registers);

    // process memory bit measurements
    for (const auto &pair : memory_map) {
      creg.store_measure(reg_t({sample[pair.second]}), reg_t({pair.first}),
                         reg_t());
    }
    // process register bit measurements
    for (const auto &pair : register_map) {
      creg.store_measure(reg_t({sample[pair.second]}), reg_t(),
                         reg_t({pair.first}));
    }

    // process read out errors for memory and registers
    for (const Operations::Op &roerror : roerror_ops) {
      creg.apply_roerror(roerror, rng);
    }

    auto memory = creg.memory_hex();
    data.add_memory_count(memory);
    data.add_pershot_memory(memory);

    data.add_pershot_register(creg.register_hex());

    // pop off processed sample
    all_samples.pop_back();
  }
}

//-------------------------------------------------------------------------
}  // end namespace Simulator
//-------------------------------------------------------------------------
}  // end namespace AER
//-------------------------------------------------------------------------
#endif
