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

#ifndef _aer_base_controller_hpp_
#define _aer_base_controller_hpp_

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

// Base Controller
#include "framework/creg.hpp"
#include "framework/qobj.hpp"
#include "framework/results/experiment_result.hpp"
#include "framework/results/result.hpp"
#include "framework/rng.hpp"
#include "noise/noise_model.hpp"
#include "transpile/basic_opts.hpp"
#include "transpile/truncate_qubits.hpp"

namespace AER {
namespace Base {

//=========================================================================
// Controller base class
//=========================================================================

// This is the top level controller for the Qiskit-Aer simulator
// It manages execution of all the circuits in a QOBJ, parallelization,
// noise sampling from a noise model, and circuit optimizations.

/**************************************************************************
 * ---------------
 * Parallelization
 * ---------------
 * Parallel execution uses the OpenMP library. It may happen at three levels:
 *
 *  1. Parallel execution of circuits in a QOBJ
 *  2. Parallel execution of shots in a Circuit
 *  3. Parallelization used by the State class for performing gates.
 *
 * Options 1 and 2 are mutually exclusive: enabling circuit parallelization
 * disables shot parallelization. Option 3 is available for both cases but
 * conservatively limits the number of threads since these are subthreads
 * spawned by the higher level threads. If no parallelization is used for
 * 1 and 2, all available threads will be used for 3.
 *
 * -------------------------
 * Config settings:
 *
 * - "noise_model" (json): A noise model to use for simulation [Default: null]
 * - "max_parallel_threads" (int): Set the maximum OpenMP threads that may
 *      be used across all levels of parallelization. Set to 0 for maximum
 *      available. [Default : 0]
 * - "max_parallel_experiments" (int): Set number of circuits that may be
 *      executed in parallel. Set to 0 to automatically select a number of
 *      parallel threads. [Default: 1]
 * - "max_parallel_shots" (int): Set number of shots that maybe be executed
 *      in parallel for each circuit. Set to 0 to automatically select a
 *      number of parallel threads. [Default: 0].
 * - "max_memory_mb" (int): Sets the maximum size of memory for a store.
 *      If a state needs more, an error is thrown. If set to 0, the maximum
 *      will be automatically set to the system memory size [Default: 0].
 *
 * Config settings from Data class:
 *
 * - "counts" (bool): Return counts object in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 * - "register" (bool): Return register array in circuit data [Default: False]
 **************************************************************************/

class Controller {
public:
  Controller() { clear_parallelization(); }

  //-----------------------------------------------------------------------
  // Execute qobj
  //-----------------------------------------------------------------------

  // Load a QOBJ from a JSON file and execute on the State type
  // class.
  virtual Result execute(const json_t &qobj);

  virtual Result execute(std::vector<Circuit> &circuits,
                         const Noise::NoiseModel &noise_model,
                         const json_t &config);

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  virtual void set_config(const json_t &config);

  // Clear the current config
  void virtual clear_config();

protected:
  //-----------------------------------------------------------------------
  // Circuit Execution
  //-----------------------------------------------------------------------

  // Parallel execution of a circuit
  // This function manages parallel shot configuration and internally calls
  // the `run_circuit` method for each shot thread
  virtual void execute_circuit(Circuit &circ,
                               Noise::NoiseModel &noise,
                               const json_t &config,
                               ExperimentResult &result);

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  virtual void run_circuit(const Circuit &circ, const Noise::NoiseModel &noise,
                           const json_t &config, uint_t shots, uint_t rng_seed,
                           ExperimentResult &result) const = 0;

  //-------------------------------------------------------------------------
  // State validation
  //-------------------------------------------------------------------------

  // Return True if a given circuit (and internal noise model) are valid for
  // execution on the given state. Otherwise return false.
  // If throw_except is true an exception will be thrown on the return false
  // case listing the invalid instructions in the circuit or noise model.
  template <class state_t>
  static bool validate_state(const state_t &state, const Circuit &circ,
                             const Noise::NoiseModel &noise,
                             bool throw_except = false);

  // Return True if a given circuit are valid for execution on the given state.
  // Otherwise return false.
  // If throw_except is true an exception will be thrown directly.
  template <class state_t>
  bool validate_memory_requirements(const state_t &state, const Circuit &circ,
                                    bool throw_except = false) const;

  //-----------------------------------------------------------------------
  // Config
  //-----------------------------------------------------------------------

  // Timer type
  using myclock_t = std::chrono::high_resolution_clock;

  // Transpile pass override flags
  bool truncate_qubits_ = true;

  // Validation threshold for validating states and operators
  double validation_threshold_ = 1e-8;

  // Save counts as memory list
  bool save_creg_memory_ = false;

  // Save count data
  void save_count_data(ExperimentResult &result,
                       const ClassicalRegister &creg) const;

  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Set OpenMP thread settings to default values
  void clear_parallelization();

  // Set parallelization for experiments
  virtual void
  set_parallelization_experiments(const std::vector<Circuit> &circuits,
                                  const Noise::NoiseModel &noise);

  // Set parallelization for a circuit
  virtual void set_parallelization_circuit(const Circuit &circuit,
                                           const Noise::NoiseModel &noise);

  // Return an estimate of the required memory for a circuit.
  virtual size_t required_memory_mb(const Circuit &circuit,
                                    const Noise::NoiseModel &noise) const = 0;

  // Get system memory size
  size_t get_system_memory_mb();

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
};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Config settings
//-------------------------------------------------------------------------

void Controller::set_config(const json_t &config) {

  // Load validation threshold
  JSON::get_value(validation_threshold_, "validation_threshold", config);

  // Load config for memory (creg list data)
  JSON::get_value(save_creg_memory_, "memory", config);

#ifdef _OPENMP
  // Load OpenMP maximum thread settings
  if (JSON::check_key("max_parallel_threads", config))
    JSON::get_value(max_parallel_threads_, "max_parallel_threads", config);
  if (JSON::check_key("max_parallel_experiments", config))
    JSON::get_value(max_parallel_experiments_, "max_parallel_experiments",
                    config);
  if (JSON::check_key("max_parallel_shots", config))
    JSON::get_value(max_parallel_shots_, "max_parallel_shots", config);
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

  if (JSON::check_key("max_memory_mb", config)) {
    JSON::get_value(max_memory_mb_, "max_memory_mb", config);
  }

  // for debugging
  if (JSON::check_key("_parallel_experiments", config)) {
    JSON::get_value(parallel_experiments_, "_parallel_experiments", config);
    explicit_parallelization_ = true;
  }

  // for debugging
  if (JSON::check_key("_parallel_shots", config)) {
    JSON::get_value(parallel_shots_, "_parallel_shots", config);
    explicit_parallelization_ = true;
  }

  // for debugging
  if (JSON::check_key("_parallel_state_update", config)) {
    JSON::get_value(parallel_state_update_, "_parallel_state_update", config);
    explicit_parallelization_ = true;
  }

  if (explicit_parallelization_) {
    parallel_experiments_ = std::max<int>({parallel_experiments_, 1});
    parallel_shots_ = std::max<int>({parallel_shots_, 1});
    parallel_state_update_ = std::max<int>({parallel_state_update_, 1});
  }
}

void Controller::clear_config() {
  clear_parallelization();
  validation_threshold_ = 1e-8;
}

void Controller::clear_parallelization() {
  max_parallel_threads_ = 0;
  max_parallel_experiments_ = 1;
  max_parallel_shots_ = 0;

  parallel_experiments_ = 1;
  parallel_shots_ = 1;
  parallel_state_update_ = 1;
  parallel_nested_ = false;

  explicit_parallelization_ = false;
  max_memory_mb_ = get_system_memory_mb() / 2;
}

void Controller::set_parallelization_experiments(
    const std::vector<Circuit> &circuits, const Noise::NoiseModel &noise) {
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
  std::vector<size_t> required_memory_mb_list(circuits.size());
  for (size_t j = 0; j < circuits.size(); j++) {
    required_memory_mb_list[j] = required_memory_mb(circuits[j], noise);
  }
  std::sort(required_memory_mb_list.begin(), required_memory_mb_list.end(),
            std::greater<>());
  size_t total_memory = 0;
  parallel_experiments_ = 0;
  for (size_t required_memory_mb : required_memory_mb_list) {
    total_memory += required_memory_mb;
    if (total_memory > max_memory_mb_)
      break;
    ++parallel_experiments_;
  }

  if (parallel_experiments_ <= 0)
    throw std::runtime_error(
        "a circuit requires more memory than max_memory_mb.");
  parallel_experiments_ =
      std::min<int>({parallel_experiments_, max_experiments,
                     max_parallel_threads_, static_cast<int>(circuits.size())});
}

void Controller::set_parallelization_circuit(const Circuit &circ,
                                             const Noise::NoiseModel &noise) {

  // Use a local variable to not override stored maximum based
  // on currently executed circuits
  const auto max_shots =
      (max_parallel_shots_ > 0)
          ? std::min({max_parallel_shots_, max_parallel_threads_})
          : max_parallel_threads_;

  // If we are executing circuits in parallel we disable
  // parallel shots
  if (max_shots == 1 || parallel_experiments_ > 1) {
    parallel_shots_ = 1;
  } else {
    // Parallel shots is > 1
    // Limit parallel shots by available memory and number of shots
    // And assign the remaining threads to state update
    int circ_memory_mb = required_memory_mb(circ, noise);
    if (max_memory_mb_ < circ_memory_mb)
      throw std::runtime_error(
          "a circuit requires more memory than max_memory_mb.");
    // If circ memory is 0, set it to 1 so that we don't divide by zero
    circ_memory_mb = std::max<int>({1, circ_memory_mb});

    parallel_shots_ =
        std::min<int>({static_cast<int>(max_memory_mb_ / circ_memory_mb),
                       max_shots, static_cast<int>(circ.shots)});
  }
  parallel_state_update_ =
      (parallel_shots_ > 1)
          ? std::max<int>({1, max_parallel_threads_ / parallel_shots_})
          : std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
}

size_t Controller::get_system_memory_mb() {
  size_t total_physical_memory = 0;
#if defined(__linux__) || defined(__APPLE__)
  auto pages = sysconf(_SC_PHYS_PAGES);
  auto page_size = sysconf(_SC_PAGE_SIZE);
  total_physical_memory = pages * page_size;
#elif defined(_WIN64)  || defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  total_physical_memory = status.ullTotalPhys;
#endif
#ifdef AER_THRUST_CUDA
  int iDev,nDev,j;
  if(cudaGetDeviceCount(&nDev) != cudaSuccess) nDev = 0;
  for(iDev=0;iDev<nDev;iDev++){
    size_t freeMem,totalMem;
    cudaSetDevice(iDev);
    cudaMemGetInfo(&freeMem,&totalMem);
    max_gpu_memory_mb_ += totalMem;

    for(j=0;j<nDev;j++){
      if(iDev != j){
        int ip;
        cudaDeviceCanAccessPeer(&ip,iDev,j);
        if(ip){
          if(cudaDeviceEnablePeerAccess(j,0) != cudaSuccess)
            cudaGetLastError();
        }
      }
    }
  }
  total_physical_memory += max_gpu_memory_mb_;
  max_gpu_memory_mb_ >>= 20;
#endif

  return total_physical_memory >> 20;
}

//-------------------------------------------------------------------------
// State validation
//-------------------------------------------------------------------------

template <class state_t>
bool Controller::validate_state(const state_t &state, const Circuit &circ,
                                const Noise::NoiseModel &noise,
                                bool throw_except) {
  // First check if a noise model is valid for a given state
  bool noise_valid = noise.is_ideal() || state.opset().contains(noise.opset());
  bool circ_valid = state.opset().contains(circ.opset());
  if (noise_valid && circ_valid) {
    return true;
  }

  // If we didn't return true then either noise model or circ has
  // invalid instructions.
  if (throw_except == false)
    return false;

  // If we are throwing an exception we include information
  // about the invalid operations
  std::stringstream msg;
  if (!noise_valid) {
    msg << "Noise model contains invalid instructions ";
    msg << state.opset().difference(noise.opset());
    msg << " for \"" << state.name() << "\" method";
  }
  if (!circ_valid) {
    msg << "Circuit contains invalid instructions ";
    msg << state.opset().difference(circ.opset());
    msg << " for \"" << state.name() << "\" method";
  }
  throw std::runtime_error(msg.str());
}

template <class state_t>
bool Controller::validate_memory_requirements(const state_t &state,
                                              const Circuit &circ,
                                              bool throw_except) const {
  if (max_memory_mb_ == 0)
    return true;

  size_t required_mb = state.required_memory_mb(circ.num_qubits, circ.ops);
  if (max_memory_mb_ < required_mb) {
    if (throw_except) {
      std::string name = "";
      JSON::get_value(name, "name", circ.header);
      throw std::runtime_error("Insufficient memory to run circuit \"" + name +
                               "\" using the " + state.name() + " simulator.");
    }
    return false;
  }
  return true;
}

//-------------------------------------------------------------------------
// Qobj execution
//-------------------------------------------------------------------------
Result Controller::execute(const json_t &qobj_js) {
  // Load QOBJ in a try block so we can catch parsing errors and still return
  // a valid JSON output containing the error message.
  try {
    // Start QOBJ timer
    auto timer_start = myclock_t::now();

    Qobj qobj(qobj_js);
    Noise::NoiseModel noise_model;
    json_t config;
    // Check for config
    if (JSON::get_value(config, "config", qobj_js)) {
      // Set config
      set_config(config);
      // Load noise model
      JSON::get_value(noise_model, "noise_model", config);
    }
    auto result = execute(qobj.circuits, noise_model, config);
    // Get QOBJ id and pass through header to result
    result.qobj_id = qobj.id;
    if (!qobj.header.empty()) {
      result.header = qobj.header;
    }
    // Stop the timer and add total timing data including qobj parsing
    auto timer_stop = myclock_t::now();
    auto time_taken = std::chrono::duration<double>(timer_stop - timer_start).count();
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
                           const Noise::NoiseModel &noise_model,
                           const json_t &config) {
  // Start QOBJ timer
  auto timer_start = myclock_t::now();

  // Initialize Result object for the given number of experiments
  const auto num_circuits = circuits.size();
  Result result(num_circuits);

  // Execute each circuit in a try block
  try {
    if (!explicit_parallelization_) {
      // set parallelization for experiments
      set_parallelization_experiments(circuits, noise_model);
    }

#ifdef _OPENMP
    result.metadata.add(true, "omp_enabled");
#else
    result.metadata.add(false, "omp_enabled");
#endif
    result.metadata.add(parallel_experiments_, "parallel_experiments");
    result.metadata.add(max_memory_mb_, "max_memory_mb");

#ifdef _OPENMP
    // Check if circuit parallelism is nested with one of the others
    if (parallel_experiments_ > 1 && parallel_experiments_ < max_parallel_threads_) {
      // Nested parallel experiments
      parallel_nested_ = true;
      #ifdef _WIN32
      omp_set_nested(1);
      #else
      omp_set_max_active_levels(3);
      #endif
      result.metadata.add(parallel_nested_, "omp_nested");
    } else {
      parallel_nested_ = false;
      #ifdef _WIN32
      omp_set_nested(0);
      #else
      omp_set_max_active_levels(1);
      #endif
    }
#endif
    // then- and else-blocks have intentionally duplication.
    // Nested omp has significant overheads even though a guard condition exists.
    const int NUM_RESULTS = result.results.size();
    if (parallel_experiments_ > 1) {
      #pragma omp parallel for num_threads(parallel_experiments_)
      for (int j = 0; j < NUM_RESULTS; ++j) {
        // Make a copy of the noise model for each circuit execution
        // so that it can be modified if required
        auto circ_noise_model = noise_model;
        execute_circuit(circuits[j], circ_noise_model, config, result.results[j]);
      }
    } else {
      for (int j = 0; j < NUM_RESULTS; ++j) {
        // Make a copy of the noise model for each circuit execution
        // so that it can be modified if required
        auto circ_noise_model = noise_model;
        execute_circuit(circuits[j], circ_noise_model, config, result.results[j]);
      }
    }

    // Check each experiment result for completed status.
    // If only some experiments completed return partial completed status.

    bool all_failed = true;
    result.status = Result::Status::completed;
    for (int i = 0; i < NUM_RESULTS; ++i) {
      auto& experiment = result.results[i];
      if (experiment.status == ExperimentResult::Status::completed) {
        all_failed = false;
      } else {
        result.status = Result::Status::partial_completed;
        result.message += std::string(" [Experiment ") + std::to_string(i)
                          + std::string("] ") + experiment.message;
      }
    }
    if (all_failed) {
      result.status = Result::Status::error;
    }

    // Stop the timer and add total timing data
    auto timer_stop = myclock_t::now();
    auto time_taken = std::chrono::duration<double>(timer_stop - timer_start).count();
    result.metadata.add(time_taken, "time_taken");
  }
  // If execution failed return valid output reporting error
  catch (std::exception &e) {
    result.status = Result::Status::error;
    result.message = e.what();
  }
  return result;
}

void Controller::execute_circuit(Circuit &circ,
                                 Noise::NoiseModel &noise,
                                 const json_t &config,
                                 ExperimentResult &result) {

  // Start individual circuit timer
  auto timer_start = myclock_t::now(); // state circuit timer

  // Initialize circuit json return
  result.legacy_data.set_config(config);

  // Execute in try block so we can catch errors and return the error message
  // for individual circuit failures.
  try {
    // Remove barriers from circuit
    Transpile::ReduceBarrier barrier_pass;
    barrier_pass.optimize_circuit(circ, noise, circ.opset(), result);

    // Truncate unused qubits from circuit and noise model
    if (truncate_qubits_) {
      Transpile::TruncateQubits truncate_pass;
      truncate_pass.set_config(config);
      truncate_pass.optimize_circuit(circ, noise, circ.opset(),
                                     result);
    }

    // set parallelization for this circuit
    if (!explicit_parallelization_) {
      set_parallelization_circuit(circ, noise);
    }

    // Single shot thread execution
    if (parallel_shots_ <= 1) {
      run_circuit(circ, noise, config, circ.shots, circ.seed, result);
      // Parallel shot thread execution
    } else {
      // Calculate shots per thread
      std::vector<unsigned int> subshots;
      for (int j = 0; j < parallel_shots_; ++j) {
        subshots.push_back(circ.shots / parallel_shots_);
      }
      // If shots is not perfectly divisible by threads, assign the remainder
      for (int j = 0; j < int(circ.shots % parallel_shots_); ++j) {
        subshots[j] += 1;
      }

      // Vector to store parallel thread output data
      std::vector<ExperimentResult> par_results(parallel_shots_);
      std::vector<std::string> error_msgs(parallel_shots_);

    #ifdef _OPENMP
    if (!parallel_nested_) {
      if (parallel_shots_ > 1 && parallel_state_update_ > 1) {
        // Nested parallel shots + state update
        #ifdef _WIN32
        omp_set_nested(1);
        #else
        omp_set_max_active_levels(2);
        #endif
        result.metadata.add(true, "omp_nested");
      } else {
        #ifdef _WIN32
        omp_set_nested(0);
        #else
        omp_set_max_active_levels(1);
        #endif
      }
    }
    #endif

#pragma omp parallel for if (parallel_shots_ > 1) num_threads(parallel_shots_)
      for (int i = 0; i < parallel_shots_; i++) {
        try {
          run_circuit(circ, noise, config, subshots[i], circ.seed + i,
                      par_results[i]);
        } catch (std::runtime_error &error) {
          error_msgs[i] = error.what();
        }
      }

      for (std::string error_msg : error_msgs)
        if (error_msg != "")
          throw std::runtime_error(error_msg);

      // Accumulate results across shots
      // Use move semantics to avoid copying data
      for (auto &res : par_results) {
        result.combine(std::move(res));
      }
    }
    // Report success
    result.status = ExperimentResult::Status::completed;

    // Pass through circuit header and add metadata
    result.header = circ.header;
    result.shots = circ.shots;
    result.seed = circ.seed;
    result.metadata.add(parallel_shots_, "parallel_shots");
    result.metadata.add(parallel_state_update_, "parallel_state_update");
    // Add timer data
    auto timer_stop = myclock_t::now(); // stop timer
    double time_taken =
        std::chrono::duration<double>(timer_stop - timer_start).count();
    result.time_taken = time_taken;
  }
  // If an exception occurs during execution, catch it and pass it to the output
  catch (std::exception &e) {
    result.status = ExperimentResult::Status::error;
    result.message = e.what();
  }
}


void Controller::save_count_data(ExperimentResult &result,
                                 const ClassicalRegister &creg) const {
  if (creg.memory_size() > 0) {
    std::string memory_hex = creg.memory_hex();
    result.data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
    if (save_creg_memory_) {
      result.data.add_list(std::move(memory_hex), "memory");
    }
  }
}

//-------------------------------------------------------------------------
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
