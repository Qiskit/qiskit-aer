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

#ifndef _aer_controller_hpp_
#define _aer_controller_hpp_

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#elif defined(_WIN64) || defined(_WIN32)
// This is needed because windows.h redefine min()/max() so interferes with
// std::min/max
#define NOMINMAX
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

#include "framework/config.hpp"
#include "framework/creg.hpp"
#include "framework/qobj.hpp"
#include "framework/results/experiment_result.hpp"
#include "framework/results/result.hpp"
#include "framework/rng.hpp"
#include "noise/noise_model.hpp"

#include "transpile/cacheblocking.hpp"
#include "transpile/fusion.hpp"

#include "simulators/circuit_executor.hpp"
#include "simulators/density_matrix/densitymatrix_executor.hpp"
#include "simulators/multi_state_executor.hpp"
#include "simulators/simulators.hpp"
#include "simulators/statevector/statevector_executor.hpp"
#include "simulators/tensor_network/tensor_net_executor.hpp"
#include "simulators/unitary/unitary_executor.hpp"

namespace AER {

//=========================================================================
// AER::Controller class
//=========================================================================

// This is the top level controller for the Qiskit-Aer simulator
// It manages execution of all the circuits in a QOBJ, parallelization,
// noise sampling from a noise model, and circuit optimizations.

class Controller {
public:
  Controller() { clear_parallelization(); }

  //-----------------------------------------------------------------------
  // Execute qobj
  //-----------------------------------------------------------------------

  // Load a QOBJ from a JSON file and execute on the State type
  // class.
  template <typename inputdata_t>
  Result execute(const inputdata_t &qobj);

  Result execute(std::vector<Circuit> &circuits, Noise::NoiseModel &noise_model,
                 const Config &config);

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  void set_config(const Config &config);

  // Clear the current config
  void clear_config();

protected:
  //-----------------------------------------------------------------------
  // Simulation types
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  // Config
  //-----------------------------------------------------------------------

  // Timer type
  using myclock_t = std::chrono::high_resolution_clock;

  // Validation threshold for validating states and operators
  double validation_threshold_ = 1e-8;

  // Save counts as memory list
  bool save_creg_memory_ = false;

  // Simulation method
  Method method_ = Method::automatic;

  // Simulation device
  Device sim_device_ = Device::CPU;
  std::string sim_device_name_ = "CPU";

  // Simulation precision
  Precision sim_precision_ = Precision::Double;

  // Controller-level parameter for CH method
  bool extended_stabilizer_measure_sampling_ = false;

  //-----------------------------------------------------------------------
  // Circuit Execution
  //-----------------------------------------------------------------------

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  void run_circuit(Circuit &circ, const Noise::NoiseModel &noise,
                   const Method method, const Config &config,
                   ExperimentResult &result);

  //-------------------------------------------------------------------------
  // State validation
  //-------------------------------------------------------------------------

  // Return True if the operations in the circuit and noise model are valid
  // for execution on the given method, and that the required memory is less
  // than the maximum allowed memory, otherwise return false.
  // If `throw_except` is true an exception will be thrown on the return false
  // case listing the invalid instructions in the circuit or noise model, or
  // the required memory.
  bool validate_method(Method method, const Circuit &circ,
                       const Noise::NoiseModel &noise,
                       bool throw_except = false) const;

  template <class state_t>
  bool validate_state(const state_t &state, const Circuit &circ,
                      const Noise::NoiseModel &noise,
                      bool throw_except = false) const;

  // Return an estimate of the required memory for a circuit.
  size_t required_memory_mb(const Circuit &circuit,
                            const Noise::NoiseModel &noise,
                            const Method method) const;

  //----------------------------------------------------------------
  // Utility functions
  //----------------------------------------------------------------

  // Return a vector of simulation methods for each circuit.
  // If the default method is automatic this will be computed based on the
  // circuit and noise model.
  // The noise model will be modified to enable superop or kraus sampling
  // methods if required by the chosen methods.
  std::vector<Method> simulation_methods(std::vector<Circuit> &circuits,
                                         Noise::NoiseModel &noise_model) const;

  // Return the simulation method to use based on the input circuit
  // and noise model
  Method
  automatic_simulation_method(const Circuit &circ,
                              const Noise::NoiseModel &noise_model) const;

  bool has_statevector_ops(const Circuit &circuit) const;
  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Set OpenMP thread settings to default values
  void clear_parallelization();

  // Set parallelization for experiments
  void set_parallelization_experiments(const std::vector<Circuit> &circuits,
                                       const Noise::NoiseModel &noise,
                                       const std::vector<Method> &methods);

  void save_exception_to_results(Result &result, const std::exception &e) const;

  // Get system memory size
  size_t get_system_memory_mb();
  size_t get_gpu_memory_mb();

  // The maximum number of threads to use for various levels of parallelization
  int max_parallel_threads_;

  // Parameters for parallelization management in configuration
  int max_parallel_experiments_;
  int max_parallel_shots_;
  size_t max_memory_mb_;
  size_t max_gpu_memory_mb_;

  // use explicit parallelization
  bool explicit_parallelization_;

  // Parameters for parallelization management for experiments
  int parallel_experiments_;
  int parallel_shots_;
  int parallel_state_update_;

  bool parallel_nested_ = false;

  // results are stored independently in each process if true
  bool accept_distributed_results_ = true;

  // process information (MPI)
  int myrank_ = 0;
  int num_processes_ = 1;
  int num_process_per_experiment_ = 1;
};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Config settings
//-------------------------------------------------------------------------

void Controller::set_config(const Config &config) {

  // Load validation threshold
  validation_threshold_ = config.validation_threshold;

  // Load config for memory (creg list data)
  if (config.memory.has_value())
    save_creg_memory_ = config.memory.value();

#ifdef _OPENMP
  // Load OpenMP maximum thread settings
  if (config.max_parallel_threads.has_value())
    max_parallel_threads_ = config.max_parallel_threads.value();
  if (config.max_parallel_experiments.has_value())
    max_parallel_experiments_ = config.max_parallel_experiments.value();
  if (config.max_parallel_shots.has_value())
    max_parallel_shots_ = config.max_parallel_shots.value();
  // Limit max threads based on number of available OpenMP threads
  auto omp_threads = omp_get_max_threads();
  max_parallel_threads_ = (max_parallel_threads_ > 0)
                              ? std::min(max_parallel_threads_, omp_threads)
                              : std::max(1, omp_threads);
#else
  // No OpenMP so we disable parallelization
  max_parallel_threads_ = 1;
  max_parallel_shots_ = 1;
  max_parallel_experiments_ = 1;
  parallel_nested_ = false;
#endif

  // Load configurations for parallelization

  if (config.max_memory_mb.has_value())
    max_memory_mb_ = config.max_memory_mb.value();

  // for debugging
  if (config._parallel_experiments.has_value()) {
    parallel_experiments_ = config._parallel_experiments.value();
    explicit_parallelization_ = true;
  }

  // for debugging
  if (config._parallel_shots.has_value()) {
    parallel_shots_ = config._parallel_shots.value();
    explicit_parallelization_ = true;
  }

  // for debugging
  if (config._parallel_state_update.has_value()) {
    parallel_state_update_ = config._parallel_state_update.value();
    explicit_parallelization_ = true;
  }

  if (explicit_parallelization_) {
    parallel_experiments_ = std::max<int>({parallel_experiments_, 1});
    parallel_shots_ = std::max<int>({parallel_shots_, 1});
    parallel_state_update_ = std::max<int>({parallel_state_update_, 1});
  }

  if (config.accept_distributed_results.has_value())
    accept_distributed_results_ = config.accept_distributed_results.value();

  // Override automatic simulation method with a fixed method
  std::string method = config.method;
  if (config.method == "statevector") {
    method_ = Method::statevector;
  } else if (config.method == "density_matrix") {
    method_ = Method::density_matrix;
  } else if (config.method == "stabilizer") {
    method_ = Method::stabilizer;
  } else if (config.method == "extended_stabilizer") {
    method_ = Method::extended_stabilizer;
  } else if (config.method == "matrix_product_state") {
    method_ = Method::matrix_product_state;
  } else if (config.method == "unitary") {
    method_ = Method::unitary;
  } else if (config.method == "superop") {
    method_ = Method::superop;
  } else if (config.method == "tensor_network") {
    method_ = Method::tensor_network;
  } else if (config.method != "automatic") {
    throw std::runtime_error(std::string("Invalid simulation method (") +
                             method + std::string(")."));
  }

  // Override automatic simulation method with a fixed method
  sim_device_name_ = config.device;
  if (sim_device_name_ == "CPU") {
    sim_device_ = Device::CPU;
  } else if (sim_device_name_ == "Thrust") {
#ifndef AER_THRUST_CPU
    throw std::runtime_error(
        "Simulation device \"Thrust\" is not supported on this system");
#else
    sim_device_ = Device::ThrustCPU;
#endif
  } else if (sim_device_name_ == "GPU") {
#ifndef AER_THRUST_CUDA
    throw std::runtime_error(
        "Simulation device \"GPU\" is not supported on this system");
#else

#ifndef AER_CUSTATEVEC
    // cuStateVec configs
    if (config.cuStateVec_enable.has_value()) {
      if (config.cuStateVec_enable.value()) {
        // Aer is not built for cuStateVec
        throw std::runtime_error("Simulation device \"GPU\" does not support "
                                 "cuStateVec on this system");
      }
    }
#endif
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      throw std::runtime_error("No CUDA device available!");
    }
    sim_device_ = Device::GPU;
#endif
  } else {
    throw std::runtime_error(std::string("Invalid simulation device (\"") +
                             sim_device_name_ + std::string("\")."));
  }

  if (method_ == Method::tensor_network) {
#if defined(AER_THRUST_CUDA) && defined(AER_CUTENSORNET)
    if (sim_device_ != Device::GPU)
#endif
      throw std::runtime_error(
          "Invalid combination of simulation method and device, "
          "\"tensor_network\" only supports \"device=GPU\"");
  }

  std::string precision = config.precision;
  if (precision == "double") {
    sim_precision_ = Precision::Double;
  } else if (precision == "single") {
    sim_precision_ = Precision::Single;
  } else {
    throw std::runtime_error(std::string("Invalid simulation precision (") +
                             precision + std::string(")."));
  }
}

void Controller::clear_config() {
  clear_parallelization();
  validation_threshold_ = 1e-8;
  method_ = Method::automatic;
  sim_device_ = Device::CPU;
  sim_precision_ = Precision::Double;
}

void Controller::clear_parallelization() {
  max_parallel_threads_ = 0;
  max_parallel_experiments_ = 1;
  max_parallel_shots_ = 0;

  parallel_experiments_ = 1;
  parallel_shots_ = 1;
  parallel_state_update_ = 1;
  parallel_nested_ = false;

  num_process_per_experiment_ = 1;

  explicit_parallelization_ = false;
  max_memory_mb_ = get_system_memory_mb();
  max_gpu_memory_mb_ = get_gpu_memory_mb();
}

void Controller::set_parallelization_experiments(
    const std::vector<Circuit> &circuits, const Noise::NoiseModel &noise,
    const std::vector<Method> &methods) {
  std::vector<size_t> required_memory_mb_list(circuits.size());
  for (size_t j = 0; j < circuits.size(); j++) {
    required_memory_mb_list[j] =
        required_memory_mb(circuits[j], noise, methods[j]);
  }
  std::sort(required_memory_mb_list.begin(), required_memory_mb_list.end(),
            std::greater<>());

  if (explicit_parallelization_)
    return;

  if (circuits.size() == 1) {
    parallel_experiments_ = 1;
    return;
  }

  // Use a local variable to not override stored maximum based
  // on currently executed circuits
  const auto max_experiments =
      (max_parallel_experiments_ > 0)
          ? std::min({max_parallel_experiments_, max_parallel_threads_})
          : max_parallel_threads_;

  if (max_experiments == 1) {
    // No parallel experiment execution
    parallel_experiments_ = 1;
    return;
  }

  // If memory allows, execute experiments in parallel
  size_t total_memory = 0;
  int parallel_experiments = 0;
  for (size_t required_memory_mb : required_memory_mb_list) {
    total_memory += required_memory_mb;
    if (total_memory > max_memory_mb_)
      break;
    ++parallel_experiments;
  }

  if (parallel_experiments <= 0)
    throw std::runtime_error(
        "a circuit requires more memory than max_memory_mb.");
  parallel_experiments_ =
      std::min<int>({parallel_experiments, max_experiments,
                     max_parallel_threads_, static_cast<int>(circuits.size())});
}

size_t Controller::get_system_memory_mb() {
  size_t total_physical_memory = Utils::get_system_memory_mb();
#ifdef AER_MPI
  // get minimum memory size per process
  uint64_t locMem, minMem;
  locMem = total_physical_memory;
  MPI_Allreduce(&locMem, &minMem, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  total_physical_memory = minMem;
#endif

  return total_physical_memory;
}

size_t Controller::get_gpu_memory_mb() {
  size_t total_physical_memory = 0;
#ifdef AER_THRUST_CUDA
  int iDev, nDev, j;
  if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
    cudaGetLastError();
    nDev = 0;
  }
  for (iDev = 0; iDev < nDev; iDev++) {
    size_t freeMem, totalMem;
    cudaSetDevice(iDev);
    cudaMemGetInfo(&freeMem, &totalMem);
    total_physical_memory += totalMem;
  }
#endif

#ifdef AER_MPI
  // get minimum memory size per process
  uint64_t locMem, minMem;
  locMem = total_physical_memory;
  MPI_Allreduce(&locMem, &minMem, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  total_physical_memory = minMem;
#endif

  return total_physical_memory >> 20;
}

//-------------------------------------------------------------------------
// Qobj execution
//-------------------------------------------------------------------------
template <typename inputdata_t>
Result Controller::execute(const inputdata_t &input_qobj) {
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
#endif

  // Load QOBJ in a try block so we can catch parsing errors and still return
  // a valid JSON output containing the error message.
  try {
    // Start QOBJ timer
    auto timer_start = myclock_t::now();

    // Initialize QOBJ
    Qobj qobj(input_qobj);
    auto qobj_time_taken =
        std::chrono::duration<double>(myclock_t::now() - timer_start).count();

    // Set config
    set_config(qobj.config);

    // Run qobj circuits
    auto result = execute(qobj.circuits, qobj.noise_model, qobj.config);

    // Add QOBJ loading time
    result.metadata.add(qobj_time_taken, "time_taken_load_qobj");

    // Get QOBJ id and pass through header to result
    result.qobj_id = qobj.id;
    if (!qobj.header.empty()) {
      result.header = qobj.header;
    }

    // Stop the timer and add total timing data including qobj parsing
    auto time_taken =
        std::chrono::duration<double>(myclock_t::now() - timer_start).count();
    result.metadata.add(time_taken, "time_taken");
    return result;
  } catch (std::exception &e) {
    // qobj was invalid, return valid output containing error message
    Result result;

    result.status = Result::Status::error;
    result.message = std::string("Failed to load qobj: ") + e.what();
    return result;
  }
}

//-------------------------------------------------------------------------
// Experiment execution
//-------------------------------------------------------------------------

Result Controller::execute(std::vector<Circuit> &circuits,
                           Noise::NoiseModel &noise_model,
                           const Config &config) {
  // Start QOBJ timer
  auto timer_start = myclock_t::now();

  // Determine simulation method for each circuit
  // and enable required noise sampling methods
  auto methods = simulation_methods(circuits, noise_model);

  // Initialize Result object for the given number of experiments
  Result result(circuits.size());

  // Execute each circuit in a try block
  try {
    num_process_per_experiment_ = num_processes_;

    // set parallelization for experiments
    try {
      // catch exception raised by required_memory_mb because of invalid
      // simulation method
      set_parallelization_experiments(circuits, noise_model, methods);
    } catch (std::exception &e) {
      save_exception_to_results(result, e);
    }

#ifdef _OPENMP
    result.metadata.add(true, "omp_enabled");
#else
    result.metadata.add(false, "omp_enabled");
#endif
    result.metadata.add(parallel_experiments_, "parallel_experiments");
    result.metadata.add(max_memory_mb_, "max_memory_mb");
    result.metadata.add(max_gpu_memory_mb_, "max_gpu_memory_mb");

    // store rank and number of processes, if no distribution rank=0 procs=1 is
    // set
    result.metadata.add(num_process_per_experiment_,
                        "num_processes_per_experiments");
    result.metadata.add(num_processes_, "num_mpi_processes");
    result.metadata.add(myrank_, "mpi_rank");

#ifdef _OPENMP
    // Check if circuit parallelism is nested with one of the others
    if (parallel_experiments_ > 1 &&
        parallel_experiments_ < max_parallel_threads_) {
      // Nested parallel experiments
      parallel_nested_ = true;

      // nested should be set to zero if num_threads clause will be used
#if _OPENMP >= 200805
      omp_set_max_active_levels(0);
#else
      omp_set_nested(0);
#endif

      result.metadata.add(parallel_nested_, "omp_nested");
    } else {
      parallel_nested_ = false;
    }
#endif

#ifdef AER_MPI
    // average random seed to set the same seed to each process (when
    // seed_simulator is not set)
    if (num_processes_ > 1) {
      reg_t seeds(circuits.size());
      reg_t avg_seeds(circuits.size());
      for (int_t i = 0; i < circuits.size(); i++)
        seeds[i] = circuits[i].seed;
      MPI_Allreduce(seeds.data(), avg_seeds.data(), circuits.size(),
                    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
      for (int_t i = 0; i < circuits.size(); i++)
        circuits[i].seed = avg_seeds[i] / num_processes_;
    }
#endif

    const int NUM_RESULTS = result.results.size();
    // following looks very similar but we have to separate them to avoid omp
    // nested loops that causes performance degradation (DO NOT use if statement
    // in #pragma omp)
    if (parallel_experiments_ == 1) {
      for (int j = 0; j < NUM_RESULTS; ++j) {
        run_circuit(circuits[j], noise_model, methods[j], config,
                    result.results[j]);
      }
    } else {
#pragma omp parallel for num_threads(parallel_experiments_)
      for (int j = 0; j < NUM_RESULTS; ++j) {
        run_circuit(circuits[j], noise_model, methods[j], config,
                    result.results[j]);
      }
    }

    // Check each experiment result for completed status.
    // If only some experiments completed return partial completed status.

    bool all_failed = true;
    result.status = Result::Status::completed;
    for (int i = 0; i < NUM_RESULTS; ++i) {
      auto &experiment = result.results[i];
      if (experiment.status == ExperimentResult::Status::completed) {
        all_failed = false;
      } else {
        result.status = Result::Status::partial_completed;
        result.message += std::string(" [Experiment ") + std::to_string(i) +
                          std::string("] ") + experiment.message;
      }
    }
    if (all_failed) {
      result.status = Result::Status::error;
    }

    // Stop the timer and add total timing data
    auto timer_stop = myclock_t::now();
    auto time_taken =
        std::chrono::duration<double>(timer_stop - timer_start).count();
    result.metadata.add(time_taken, "time_taken_execute");
  }
  // If execution failed return valid output reporting error
  catch (std::exception &e) {
    result.status = Result::Status::error;
    result.message = e.what();
  }
  return result;
}

//-------------------------------------------------------------------------
// Base class override
//-------------------------------------------------------------------------
void Controller::run_circuit(Circuit &circ, const Noise::NoiseModel &noise,
                             const Method method, const Config &config,
                             ExperimentResult &result) {
  // Run the circuit
  switch (method) {
  case Method::statevector:
    if (sim_device_ == Device::CPU) {
      if (sim_precision_ == Precision::Double) {
        // Double-precision Statevector simulation
        Statevector::Executor<Statevector::State<QV::QubitVector<double>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      } else {
        // Single-precision Statevector simulation
        Statevector::Executor<Statevector::State<QV::QubitVector<float>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      // Chunk based simulation
      if (sim_precision_ == Precision::Double) {
        // Double-precision Statevector simulation
        Statevector::Executor<Statevector::State<QV::QubitVectorThrust<double>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      } else {
        // Single-precision Statevector simulation
        Statevector::Executor<Statevector::State<QV::QubitVectorThrust<float>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      }
#endif
    }
    break;
  case Method::density_matrix:
    if (sim_device_ == Device::CPU) {
      if (sim_precision_ == Precision::Double) {
        // Double-precision DensityMatrix simulation
        DensityMatrix::Executor<DensityMatrix::State<QV::DensityMatrix<double>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      } else {
        // Single-precision DensityMatrix simulation
        DensityMatrix::Executor<DensityMatrix::State<QV::DensityMatrix<float>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      // Chunk based simulation
      if (sim_precision_ == Precision::Double) {
        // Double-precision DensityMatrix simulation
        DensityMatrix::Executor<
            DensityMatrix::State<QV::DensityMatrixThrust<double>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      } else {
        // Single-precision DensityMatrix simulation
        DensityMatrix::Executor<
            DensityMatrix::State<QV::DensityMatrixThrust<float>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      }
#endif
    }
    break;
  case Method::unitary:
    if (sim_device_ == Device::CPU) {
      if (sim_precision_ == Precision::Double) {
        // Double-precision unitary simulation
        QubitUnitary::Executor<QubitUnitary::State<QV::UnitaryMatrix<double>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      } else {
        // Single-precision unitary simulation
        QubitUnitary::Executor<QubitUnitary::State<QV::UnitaryMatrix<float>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      // Chunk based simulation
      if (sim_precision_ == Precision::Double) {
        // Double-precision unitary simulation
        QubitUnitary::Executor<
            QubitUnitary::State<QV::UnitaryMatrixThrust<double>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      } else {
        // Single-precision unitary simulation
        QubitUnitary::Executor<
            QubitUnitary::State<QV::UnitaryMatrixThrust<float>>>
            executor;
        executor.run_circuit(circ, noise, config, method, sim_device_, result);
      }
#endif
    }
    break;
  case Method::superop:
    if (sim_precision_ == Precision::Double) {
      CircuitExecutor::Executor<
          QubitSuperoperator::State<QV::Superoperator<double>>>
          executor;
      executor.run_circuit(circ, noise, config, method, sim_device_, result);
    } else {
      CircuitExecutor::Executor<
          QubitSuperoperator::State<QV::Superoperator<float>>>
          executor;
      executor.run_circuit(circ, noise, config, method, sim_device_, result);
    }
    break;
  case Method::stabilizer:
    // Stabilizer simulation
    // TODO: Stabilizer doesn't yet support custom state initialization
    {
      CircuitExecutor::Executor<Stabilizer::State> executor;
      executor.run_circuit(circ, noise, config, method, sim_device_, result);
    }
    break;
  case Method::extended_stabilizer: {
    CircuitExecutor::Executor<ExtendedStabilizer::State> executor;
    executor.run_circuit(circ, noise, config, method, sim_device_, result);
  } break;
  case Method::matrix_product_state: {
    CircuitExecutor::Executor<MatrixProductState::State> executor;
    executor.run_circuit(circ, noise, config, method, sim_device_, result);
  } break;
  case Method::tensor_network: {
    if (sim_precision_ == Precision::Double) {
      TensorNetwork::Executor<
          TensorNetwork::State<TensorNetwork::TensorNet<double>>>
          executor;
      executor.run_circuit(circ, noise, config, method, sim_device_, result);
    } else {
      TensorNetwork::Executor<
          TensorNetwork::State<TensorNetwork::TensorNet<float>>>
          executor;
      executor.run_circuit(circ, noise, config, method, sim_device_, result);
    }
  } break;
  default:
    throw std::runtime_error("Controller:Invalid simulation method");
  }
}

//-------------------------------------------------------------------------
// Utility methods
//-------------------------------------------------------------------------

size_t Controller::required_memory_mb(const Circuit &circ,
                                      const Noise::NoiseModel &noise,
                                      const Method method) const {
  switch (method) {
  case Method::statevector: {
    if (sim_precision_ == Precision::Single) {
      Statevector::State<QV::QubitVector<float>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    } else {
      Statevector::State<QV::QubitVector<double>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
  }
  case Method::density_matrix: {
    if (sim_precision_ == Precision::Single) {
      DensityMatrix::State<QV::DensityMatrix<float>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    } else {
      DensityMatrix::State<QV::DensityMatrix<double>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
  }
  case Method::unitary: {
    if (sim_precision_ == Precision::Single) {
      QubitUnitary::State<QV::UnitaryMatrix<float>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    } else {
      QubitUnitary::State<QV::UnitaryMatrix<double>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
  }
  case Method::superop: {
    if (sim_precision_ == Precision::Single) {
      QubitSuperoperator::State<QV::Superoperator<float>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    } else {
      QubitSuperoperator::State<QV::Superoperator<double>> state;
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
  case Method::tensor_network: {
    if (sim_precision_ == Precision::Single) {
      TensorNetwork::State<TensorNetwork::TensorNet<float>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    } else {
      TensorNetwork::State<TensorNetwork::TensorNet<double>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
  }
  default:
    // We shouldn't get here, so throw an exception if we do
    throw std::runtime_error("Controller: Invalid simulation method");
  }
}

//-------------------------------------------------------------------------
// Validation
//-------------------------------------------------------------------------

std::vector<Method>
Controller::simulation_methods(std::vector<Circuit> &circuits,
                               Noise::NoiseModel &noise_model) const {
  // Does noise model contain kraus noise
  bool kraus_noise =
      (noise_model.opset().contains(Operations::OpType::kraus) ||
       noise_model.opset().contains(Operations::OpType::superop));

  if (method_ == Method::automatic) {
    // Determine simulation methods for each circuit and noise model
    std::vector<Method> sim_methods;
    bool superop_enabled = false;
    bool kraus_enabled = false;
    for (const auto &circ : circuits) {
      auto method = automatic_simulation_method(circ, noise_model);
      sim_methods.push_back(method);
      if (!superop_enabled &&
          (method == Method::density_matrix || method == Method::superop ||
           (method == Method::tensor_network && !has_statevector_ops(circ)))) {
        noise_model.enable_superop_method(max_parallel_threads_);
        superop_enabled = true;
      } else if (kraus_noise && !kraus_enabled &&
                 (method == Method::statevector ||
                  method == Method::matrix_product_state ||
                  (method == Method::tensor_network &&
                   has_statevector_ops(circ)))) {
        noise_model.enable_kraus_method(max_parallel_threads_);
        kraus_enabled = true;
      }
    }
    return sim_methods;
  }

  // Use non-automatic default method for all circuits
  std::vector<Method> sim_methods(circuits.size(), method_);
  if (method_ == Method::density_matrix || method_ == Method::superop) {
    noise_model.enable_superop_method(max_parallel_threads_);
  } else if (kraus_noise && (method_ == Method::statevector ||
                             method_ == Method::matrix_product_state)) {
    noise_model.enable_kraus_method(max_parallel_threads_);
  } else if (method_ == Method::tensor_network) {
    bool has_save_statevec = false;
    for (const auto &circ : circuits) {
      has_save_statevec |= has_statevector_ops(circ);
      if (has_save_statevec)
        break;
    }
    if (!has_save_statevec)
      noise_model.enable_superop_method(max_parallel_threads_);
    else if (kraus_noise)
      noise_model.enable_kraus_method(max_parallel_threads_);
  }
  return sim_methods;
}

Method Controller::automatic_simulation_method(
    const Circuit &circ, const Noise::NoiseModel &noise_model) const {
  // If circuit and noise model are Clifford run on Stabilizer simulator
  if (validate_method(Method::stabilizer, circ, noise_model, false)) {
    return Method::stabilizer;
  }
  // For noisy simulations we enable the density matrix method if
  // shots > 2 ** num_qubits. This is based on a rough estimate that
  // a single shot of the density matrix simulator is approx 2 ** nq
  // times slower than a single shot of statevector due the increased
  // dimension
  if (noise_model.has_quantum_errors() && circ.num_qubits < 64 &&
      circ.shots > (1ULL << circ.num_qubits) &&
      validate_method(Method::density_matrix, circ, noise_model, false) &&
      circ.can_sample) {
    return Method::density_matrix;
  }

  // If the special conditions for stabilizer or density matrix are
  // not satisfied we choose simulation method based on supported
  // operations only with preference given by memory requirements
  // statevector > density matrix > matrix product state > unitary > superop
  // typically any save state instructions will decide the method.
  const std::vector<Method> methods(
      {Method::statevector, Method::density_matrix,
       Method::matrix_product_state, Method::unitary, Method::superop});
  for (const auto &method : methods) {
    if (validate_method(method, circ, noise_model, false))
      return method;
  }

  // If we got here, circuit isn't compatible with any of the simulation
  // method so fallback to a default method of statevector. The execution will
  // fail but we will get partial result generation and generate a user facing
  // error message
  return Method::statevector;
}

bool Controller::validate_method(Method method, const Circuit &circ,
                                 const Noise::NoiseModel &noise_model,
                                 bool throw_except) const {
  // Switch wrapper for templated function validate_state
  switch (method) {
  case Method::stabilizer:
    return validate_state(Stabilizer::State(), circ, noise_model, throw_except);
  case Method::extended_stabilizer:
    return validate_state(ExtendedStabilizer::State(), circ, noise_model,
                          throw_except);
  case Method::matrix_product_state:
    return validate_state(MatrixProductState::State(), circ, noise_model,
                          throw_except);
  case Method::statevector:
    return validate_state(Statevector::State<>(), circ, noise_model,
                          throw_except);
  case Method::density_matrix:
    return validate_state(DensityMatrix::State<>(), circ, noise_model,
                          throw_except);
  case Method::unitary:
    return validate_state(QubitUnitary::State<>(), circ, noise_model,
                          throw_except);
  case Method::superop:
    return validate_state(QubitSuperoperator::State<>(), circ, noise_model,
                          throw_except);
  case Method::tensor_network:
    return validate_state(TensorNetwork::State<>(), circ, noise_model,
                          throw_except);
  case Method::automatic:
    throw std::runtime_error(
        "Cannot validate circuit for unresolved simulation method.");
  }
}

template <class state_t>
bool Controller::validate_state(const state_t &state, const Circuit &circ,
                                const Noise::NoiseModel &noise,
                                bool throw_except) const {
  std::stringstream error_msg;
  std::string circ_name;
  JSON::get_value(circ_name, "name", circ.header);

  // Check if a circuit is valid for state ops
  bool circ_valid = state.opset().contains(circ.opset());
  if (throw_except && !circ_valid) {
    error_msg << "Circuit " << circ_name << " contains invalid instructions ";
    error_msg << state.opset().difference(circ.opset());
    error_msg << " for \"" << state.name() << "\" method.";
  }

  // Check if a noise model valid for state ops
  bool noise_valid = noise.is_ideal() || state.opset().contains(noise.opset());
  if (throw_except && !noise_valid) {
    error_msg << "Noise model contains invalid instructions ";
    error_msg << state.opset().difference(noise.opset());
    error_msg << " for \"" << state.name() << "\" method.";
  }

  // Validate memory requirements
  bool memory_valid = true;
  if (max_memory_mb_ > 0) {
    size_t required_mb = state.required_memory_mb(circ.num_qubits, circ.ops) /
                         num_process_per_experiment_;
    size_t mem_size = (sim_device_ == Device::GPU)
                          ? max_memory_mb_ + max_gpu_memory_mb_
                          : max_memory_mb_;
    memory_valid = (required_mb <= mem_size);
    if (throw_except && !memory_valid) {
      error_msg << "Insufficient memory to run circuit " << circ_name;
      error_msg << " using the " << state.name() << " simulator.";
      error_msg << " Required memory: " << required_mb
                << "M, max memory: " << max_memory_mb_ << "M";
      if (sim_device_ == Device::GPU) {
        error_msg << " (Host) + " << max_gpu_memory_mb_ << "M (GPU)";
      }
    }
  }

  if (noise_valid && circ_valid && memory_valid) {
    return true;
  }

  // One of the validation checks failed for the current state
  if (throw_except) {
    throw std::runtime_error(error_msg.str());
  }
  return false;
}

void Controller::save_exception_to_results(Result &result,
                                           const std::exception &e) const {
  result.status = Result::Status::error;
  result.message = e.what();
  for (auto &res : result.results) {
    res.status = ExperimentResult::Status::error;
    res.message = e.what();
  }
}

bool Controller::has_statevector_ops(const Circuit &circ) const {
  return circ.opset().contains(Operations::OpType::save_statevec) ||
         circ.opset().contains(Operations::OpType::save_statevec_dict) ||
         circ.opset().contains(Operations::OpType::save_amps);
}

//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
