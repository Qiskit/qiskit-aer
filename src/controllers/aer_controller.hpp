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

#include "simulators/simulators.hpp"

#include "simulators/circuit_executor.hpp"
#include "simulators/multi_state_executor.hpp"

#include "simulators/density_matrix/densitymatrix_executor.hpp"
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
  Controller() {}

  //-----------------------------------------------------------------------
  // Execute qobj
  //-----------------------------------------------------------------------

  // Load a QOBJ from a JSON file and execute on the State type
  // class.
  template <typename inputdata_t>
  Result execute(const inputdata_t &qobj);

  Result execute(std::vector<std::shared_ptr<Circuit>> &circuits,
                 Noise::NoiseModel &noise_model, const Config &config);

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  void set_config(const Config &config);

  // return available devicess
  std::vector<std::string> available_devices();

protected:
  //-----------------------------------------------------------------------
  // Simulation types
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  // Config
  //-----------------------------------------------------------------------

  // Timer type
  using myclock_t = std::chrono::high_resolution_clock;

  // Simulation method
  Method method_ = Method::automatic;

  // Simulation device
  Device sim_device_ = Device::CPU;
  std::string sim_device_name_ = "CPU";

  // Simulation precision
  Precision sim_precision_ = Precision::Double;

  //-------------------------------------------------------------------------
  // State validation
  //-------------------------------------------------------------------------

  // Return True if the operations in the circuit and noise model are valid
  // for execution on the given method, and that the required memory is less
  // than the maximum allowed memory, otherwise return false.
  // If `throw_except` is true an exception will be thrown on the return false
  // case listing the invalid instructions in the circuit or noise model, or
  // the required memory.
  bool validate_method(Method method, const Config &config, const Circuit &circ,
                       const Noise::NoiseModel &noise,
                       bool throw_except = false) const;

  //----------------------------------------------------------------
  // Utility functions
  //----------------------------------------------------------------
  std::shared_ptr<CircuitExecutor::Base>
  make_circuit_executor(const Method method) const;

  // Return a vector of simulation methods for each circuit.
  // If the default method is automatic this will be computed based on the
  // circuit and noise model.
  // The noise model will be modified to enable superop or kraus sampling
  // methods if required by the chosen methods.
  std::vector<Method>
  simulation_methods(const Config &config,
                     std::vector<std::shared_ptr<Circuit>> &circuits,
                     Noise::NoiseModel &noise_model) const;

  // Return the simulation method to use based on the input circuit
  // and noise model
  Method
  automatic_simulation_method(const Config &config, const Circuit &circ,
                              const Noise::NoiseModel &noise_model) const;

  bool has_statevector_ops(const Circuit &circuit) const;
  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Set parallelization for experiments
  void set_parallelization_experiments(const reg_t &required_memory_list);

  void save_exception_to_results(Result &result, const std::exception &e) const;

  // Get system memory size
  size_t get_system_memory_mb();
  size_t get_gpu_memory_mb();

  // The maximum number of threads to use for various levels of parallelization
  int max_parallel_threads_ = 0;

  // Parameters for parallelization management in configuration
  int max_parallel_experiments_ = 1;
  size_t max_memory_mb_ = 0;
  size_t max_gpu_memory_mb_ = 0;

  // use explicit parallelization
  bool explicit_parallelization_ = false;

  // Parameters for parallelization management for experiments
  int parallel_experiments_ = 1;

  bool parallel_nested_ = false;

  // process information (MPI)
  int myrank_ = 0;
  int num_processes_ = 1;
  int num_process_per_experiment_ = 1;

  // runtime parameter binding
  bool runtime_parameter_bind_ = false;

  reg_t target_gpus_; // GPUs to be used
};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Config settings
//-------------------------------------------------------------------------

void Controller::set_config(const Config &config) {

#ifdef _OPENMP
  // Load OpenMP maximum thread settings
  if (config.max_parallel_threads.has_value())
    max_parallel_threads_ = config.max_parallel_threads.value();
  if (config.max_parallel_experiments.has_value())
    max_parallel_experiments_ = config.max_parallel_experiments.value();
  // Limit max threads based on number of available OpenMP threads
  auto omp_threads = omp_get_max_threads();
  max_parallel_threads_ = (max_parallel_threads_ > 0)
                              ? std::min(max_parallel_threads_, omp_threads)
                              : std::max(1, omp_threads);
#else
  // No OpenMP so we disable parallelization
  max_parallel_threads_ = 1;
  max_parallel_experiments_ = 1;
  parallel_nested_ = false;
#endif

  // Load configurations for parallelization

  if (config.max_memory_mb.has_value())
    max_memory_mb_ = config.max_memory_mb.value();
  else
    max_memory_mb_ = get_system_memory_mb();

  // for debugging
  if (config._parallel_experiments.has_value()) {
    parallel_experiments_ = config._parallel_experiments.value();
    explicit_parallelization_ = true;
  }

  // for debugging
  if (config._parallel_shots.has_value()) {
    explicit_parallelization_ = true;
  }

  // for debugging
  if (config._parallel_state_update.has_value()) {
    explicit_parallelization_ = true;
  }

  if (explicit_parallelization_) {
    parallel_experiments_ = std::max<int>({parallel_experiments_, 1});
  }

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
#ifndef AER_THRUST_GPU
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
    if (config.target_gpus.has_value()) {
      target_gpus_ = config.target_gpus.value();

      if (nDev < target_gpus_.size()) {
        throw std::invalid_argument(
            "target_gpus has more GPUs than available.");
      }
    } else {
      target_gpus_.resize(nDev);
      for (int_t i = 0; i < nDev; i++)
        target_gpus_[i] = i;
    }
    sim_device_ = Device::GPU;

    max_gpu_memory_mb_ = get_gpu_memory_mb();
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

  // check if runtime binding is enable
  if (config.runtime_parameter_bind_enable.has_value())
    runtime_parameter_bind_ = config.runtime_parameter_bind_enable.value();
}

void Controller::set_parallelization_experiments(
    const reg_t &required_memory_mb_list) {

  if (explicit_parallelization_)
    return;

  if (required_memory_mb_list.size() == 1) {
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
  reg_t required_sorted = required_memory_mb_list;
  std::sort(required_sorted.begin(), required_sorted.end(), std::greater<>());

  size_t total_memory = 0;
  int parallel_experiments = 0;
  for (size_t required_memory_mb : required_sorted) {
    total_memory += required_memory_mb;
    if (total_memory > max_memory_mb_)
      break;
    ++parallel_experiments;
  }

  if (parallel_experiments <= 0)
    throw std::runtime_error(
        "a circuit requires more memory than max_memory_mb.");
  parallel_experiments_ = std::min<int>(
      {parallel_experiments, max_experiments, max_parallel_threads_,
       static_cast<int>(required_memory_mb_list.size())});
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
#ifdef AER_THRUST_GPU
  for (uint_t iDev = 0; iDev < target_gpus_.size(); iDev++) {
    size_t freeMem, totalMem;
    cudaSetDevice(target_gpus_[iDev]);
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

std::vector<std::string> Controller::available_devices() {
  std::vector<std::string> ret;

  ret.push_back(std::string("CPU"));
#ifdef AER_THRUST_GPU
  ret.push_back(std::string("GPU"));
#else
#ifdef AER_THRUST_CPU
  ret.push_back(std::string("Thrust"));
#endif
#endif
  return ret;
}

//-------------------------------------------------------------------------
// Qobj execution
//-------------------------------------------------------------------------
template <typename inputdata_t>
Result Controller::execute(const inputdata_t &input_qobj) {
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

Result Controller::execute(std::vector<std::shared_ptr<Circuit>> &circuits,
                           Noise::NoiseModel &noise_model,
                           const Config &config) {
  // Start QOBJ timer
  auto timer_start = myclock_t::now();

#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
#endif
  // Determine simulation method for each circuit
  // and enable required noise sampling methods
  auto methods = simulation_methods(config, circuits, noise_model);

  // Initialize Result object for the given number of experiments
  uint_t result_size;
  reg_t result_offset(circuits.size());
  result_size = 0;
  for (uint_t i = 0; i < circuits.size(); i++) {
    result_offset[i] = result_size;
    result_size += circuits[i]->num_bind_params;
  }
  Result result(result_size);
  // Initialize circuit executors for each circuit
  std::vector<std::shared_ptr<CircuitExecutor::Base>> executors(
      circuits.size());
  reg_t required_memory_mb_list(circuits.size());

  // Execute each circuit in a try block
  try {
    num_process_per_experiment_ = num_processes_;

    // set parallelization for experiments
    try {
      uint_t res_pos = 0;
      for (uint_t i = 0; i < circuits.size(); i++) {
        executors[i] = make_circuit_executor(methods[i]);
        required_memory_mb_list[i] =
            executors[i]->required_memory_mb(config, *circuits[i], noise_model);
        for (uint_t j = 0; j < circuits[i]->num_bind_params; j++) {
          result.results[res_pos++].metadata.add(required_memory_mb_list[i],
                                                 "required_memory_mb");
        }
      }
      set_parallelization_experiments(required_memory_mb_list);
    } catch (std::exception &e) {
      save_exception_to_results(result, e);
    }

    result.metadata.add(parallel_experiments_, "parallel_experiments");
    result.metadata.add(max_memory_mb_, "max_memory_mb");
    result.metadata.add(max_gpu_memory_mb_, "max_gpu_memory_mb");

#ifdef _OPENMP
    result.metadata.add(true, "omp_enabled");

    // Check if circuit parallelism is nested with one of the others
    if (parallel_experiments_ > 1 &&
        parallel_experiments_ < max_parallel_threads_) {
      // Nested parallel experiments
      parallel_nested_ = true;

      // nested should be set to zero if num_threads clause will be used
#if _OPENMP >= 200805
      omp_set_max_active_levels(1);
#else
      omp_set_nested(1);
#endif

      result.metadata.add(parallel_nested_, "omp_nested");
    } else {
      parallel_nested_ = false;
    }
#else
    result.metadata.add(false, "omp_enabled");
#endif

#ifdef AER_MPI
    // store rank and number of processes, if no distribution rank=0 procs=1 is
    // set
    result.metadata.add(num_process_per_experiment_,
                        "num_processes_per_experiments");
    result.metadata.add(num_processes_, "num_mpi_processes");
    result.metadata.add(myrank_, "mpi_rank");

    // average random seed to set the same seed to each process (when
    // seed_simulator is not set)
    if (num_processes_ > 1) {
      reg_t seeds(result_size);
      reg_t avg_seeds(result_size);
      int_t iseed = 0;
      for (uint_t i = 0; i < circuits.size(); i++) {
        if (circuits[i]->num_bind_params > 1) {
          for (uint_t j = 0; i < circuits[i]->num_bind_params; i++)
            seeds[iseed++] = circuits[i]->seed_for_params[j];
        } else
          seeds[iseed++] = circuits[i]->seed;
      }
      MPI_Allreduce(seeds.data(), avg_seeds.data(), result_size, MPI_UINT64_T,
                    MPI_SUM, MPI_COMM_WORLD);
      iseed = 0;
      for (uint_t i = 0; i < circuits.size(); i++) {
        if (circuits[i]->num_bind_params > 1) {
          for (uint_t j = 0; i < circuits[i]->num_bind_params; i++)
            circuits[i]->seed_for_params[j] =
                avg_seeds[iseed++] / num_processes_;
        } else
          circuits[i]->seed = avg_seeds[iseed++] / num_processes_;
      }
    }
#endif

    auto run_circuits = [this, &executors, &circuits, &noise_model, &config,
                         &methods, &result, &result_offset](int_t i) {
      executors[i]->run_circuit(*circuits[i], noise_model, config, methods[i],
                                sim_device_,
                                result.results.begin() + result_offset[i]);
    };
    Utils::apply_omp_parallel_for((parallel_experiments_ > 1), 0,
                                  circuits.size(), run_circuits,
                                  parallel_experiments_);

    executors.clear();

    // Check each experiment result for completed status.
    // If only some experiments completed return partial completed status.

    bool all_failed = true;
    result.status = Result::Status::completed;
    for (uint_t i = 0; i < result.results.size(); ++i) {
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
// Utility methods
//-------------------------------------------------------------------------
std::shared_ptr<CircuitExecutor::Base>
Controller::make_circuit_executor(const Method method) const {
  // Run the circuit
  switch (method) {
  case Method::statevector:
    if (sim_device_ == Device::CPU) {
      if (sim_precision_ == Precision::Double) {
        // Double-precision Statevector simulation
        return std::make_shared<Statevector::Executor<
            Statevector::State<QV::QubitVector<double>>>>();
      } else {
        // Single-precision Statevector simulation
        return std::make_shared<Statevector::Executor<
            Statevector::State<QV::QubitVector<float>>>>();
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      // Chunk based simulation
      if (sim_precision_ == Precision::Double) {
        // Double-precision Statevector simulation
        return std::make_shared<Statevector::Executor<
            Statevector::State<QV::QubitVectorThrust<double>>>>();
      } else {
        // Single-precision Statevector simulation
        return std::make_shared<Statevector::Executor<
            Statevector::State<QV::QubitVectorThrust<float>>>>();
      }
#endif
    }
    break;
  case Method::density_matrix:
    if (sim_device_ == Device::CPU) {
      if (sim_precision_ == Precision::Double) {
        // Double-precision DensityMatrix simulation
        return std::make_shared<DensityMatrix::Executor<
            DensityMatrix::State<QV::DensityMatrix<double>>>>();
      } else {
        // Single-precision DensityMatrix simulation
        return std::make_shared<DensityMatrix::Executor<
            DensityMatrix::State<QV::DensityMatrix<float>>>>();
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      // Chunk based simulation
      if (sim_precision_ == Precision::Double) {
        // Double-precision DensityMatrix simulation
        return std::make_shared<DensityMatrix::Executor<
            DensityMatrix::State<QV::DensityMatrixThrust<double>>>>();
      } else {
        // Single-precision DensityMatrix simulation
        return std::make_shared<DensityMatrix::Executor<
            DensityMatrix::State<QV::DensityMatrixThrust<float>>>>();
      }
#endif
    }
    break;
  case Method::unitary:
    if (sim_device_ == Device::CPU) {
      if (sim_precision_ == Precision::Double) {
        // Double-precision unitary simulation
        return std::make_shared<QubitUnitary::Executor<
            QubitUnitary::State<QV::UnitaryMatrix<double>>>>();
      } else {
        // Single-precision unitary simulation
        return std::make_shared<QubitUnitary::Executor<
            QubitUnitary::State<QV::UnitaryMatrix<float>>>>();
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      // Chunk based simulation
      if (sim_precision_ == Precision::Double) {
        // Double-precision unitary simulation
        return std::make_shared<QubitUnitary::Executor<
            QubitUnitary::State<QV::UnitaryMatrixThrust<double>>>>();
      } else {
        // Single-precision unitary simulation
        return std::make_shared<QubitUnitary::Executor<
            QubitUnitary::State<QV::UnitaryMatrixThrust<float>>>>();
      }
#endif
    }
    break;
  case Method::superop:
    if (sim_precision_ == Precision::Double) {
      return std::make_shared<CircuitExecutor::Executor<
          QubitSuperoperator::State<QV::Superoperator<double>>>>();
    } else {
      return std::make_shared<CircuitExecutor::Executor<
          QubitSuperoperator::State<QV::Superoperator<float>>>>();
    }
    break;
  case Method::stabilizer: {
    return std::make_shared<CircuitExecutor::Executor<Stabilizer::State>>();
  } break;
  case Method::extended_stabilizer: {
    return std::make_shared<
        CircuitExecutor::Executor<ExtendedStabilizer::State>>();
  } break;
  case Method::matrix_product_state: {
    return std::make_shared<
        CircuitExecutor::Executor<MatrixProductState::State>>();
  } break;
  case Method::tensor_network: {
    if (sim_precision_ == Precision::Double) {
      return std::make_shared<TensorNetwork::Executor<
          TensorNetwork::State<TensorNetwork::TensorNet<double>>>>();
    } else {
      return std::make_shared<TensorNetwork::Executor<
          TensorNetwork::State<TensorNetwork::TensorNet<float>>>>();
    }
  } break;
  case Method::automatic:
    throw std::runtime_error(
        "Cannot make circuit executor for automatic simulation method.");
  default:
    throw std::runtime_error("Controller:Invalid simulation method");
  }
}

std::vector<Method>
Controller::simulation_methods(const Config &config,
                               std::vector<std::shared_ptr<Circuit>> &circuits,
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
    for (const auto &_circ : circuits) {
      const auto circ = *_circ;
      auto method = automatic_simulation_method(config, circ, noise_model);
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
      has_save_statevec |= has_statevector_ops(*circ);
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
    const Config &config, const Circuit &circ,
    const Noise::NoiseModel &noise_model) const {
  // For noisy simulations we enable the density matrix method if
  // shots > 2 ** num_qubits. This is based on a rough estimate that
  // a single shot of the density matrix simulator is approx 2 ** nq
  // times slower than a single shot of statevector due the increased
  // dimension
  if (noise_model.has_quantum_errors() && circ.num_qubits < 30 &&
      circ.shots > (1ULL << circ.num_qubits) &&
      validate_method(Method::density_matrix, config, circ, noise_model,
                      false) &&
      circ.can_sample) {
    return Method::density_matrix;
  }
  // If circuit and noise model are Clifford run on Stabilizer simulator
  if (validate_method(Method::stabilizer, config, circ, noise_model, false)) {
    return Method::stabilizer;
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
    if (validate_method(method, config, circ, noise_model, false))
      return method;
  }

  // If we got here, circuit isn't compatible with any of the simulation
  // method so fallback to a default method of statevector. The execution will
  // fail but we will get partial result generation and generate a user facing
  // error message
  return Method::statevector;
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
// Validation
//-------------------------------------------------------------------------
bool Controller::validate_method(Method method, const Config &config,
                                 const Circuit &circ,
                                 const Noise::NoiseModel &noise_model,
                                 bool throw_except) const {
  std::shared_ptr<CircuitExecutor::Base> executor =
      make_circuit_executor(method);
  bool ret = executor->validate_state(config, circ, noise_model, throw_except);
  executor.reset();
  return ret;
}

//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
