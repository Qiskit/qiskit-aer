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
#elif defined(_WIN64)
   // This is needed because windows.h redefine min()/max() so interferes with std::min/max
   #define NOMINMAX
   #include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Base Controller
#include "framework/qobj.hpp"
#include "framework/rng.hpp"
#include "framework/creg.hpp"
#include "framework/results/result.hpp"
#include "framework/results/experiment_data.hpp"
#include "noise/noise_model.hpp"
#include "transpile/circuitopt.hpp"
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
 * - "counts" (bool): Return counts objecy in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 * - "register" (bool): Return register array in circuit data [Default: False]
 * - "noise_model" (json): A noise model JSON dictionary for the simulator.
 *                         [Default: null]
 **************************************************************************/

class Controller {
public:

  Controller() {clear_parallelization();}

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

  // Add circuit optimization
  template <typename Type>
  inline auto add_circuit_optimization(Type&& opt)-> typename std::enable_if_t<std::is_base_of<Transpile::CircuitOptimization, std::remove_const_t<std::remove_reference_t<Type>>>::value >
  {
      optimizations_.push_back(std::make_shared<std::remove_const_t<std::remove_reference_t<Type>>>(std::forward<Type>(opt)));
  }

protected:

  //-----------------------------------------------------------------------
  // Circuit Execution
  //-----------------------------------------------------------------------

  // Parallel execution of a circuit
  // This function manages parallel shot configuration and internally calls
  // the `run_circuit` method for each shot thread
  virtual ExperimentResult execute_circuit(Circuit &circ,
                                           Noise::NoiseModel &noise,
                                           const json_t &config);

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  virtual ExperimentData run_circuit(const Circuit &circ,
                                     const Noise::NoiseModel &noise,
                                     const json_t &config,
                                     uint_t shots,
                                     uint_t rng_seed) const = 0;

  //-------------------------------------------------------------------------
  // State validation
  //-------------------------------------------------------------------------

  // Return True if a given circuit (and internal noise model) are valid for
  // execution on the given state. Otherwise return false.
  // If throw_except is true an exception will be thrown on the return false
  // case listing the invalid instructions in the circuit or noise model.
  template <class state_t>
  static bool validate_state(const state_t &state,
                             const Circuit &circ,
                             const Noise::NoiseModel &noise,
                             bool throw_except = false);

  // Return True if a given circuit are valid for execution on the given state.
  // Otherwise return false. 
  // If throw_except is true an exception will be thrown directly.
  template <class state_t>
  bool validate_memory_requirements(const state_t &state,
                                    const Circuit &circ,
                                    bool throw_except = false) const;

  //-------------------------------------------------------------------------
  // Circuit optimization
  //-------------------------------------------------------------------------

  // Generate an equivalent circuit with input_circ as output_circ.
  template <class state_t>
  void optimize_circuit(Circuit &circ,
                        Noise::NoiseModel& noise,
                        state_t& state,
                        ExperimentData &data) const;

  //-----------------------------------------------------------------------
  // Config
  //-----------------------------------------------------------------------

  // Timer type
  using myclock_t = std::chrono::high_resolution_clock;

  // Circuit optimization
  std::vector<std::shared_ptr<Transpile::CircuitOptimization>> optimizations_;

  // Validation threshold for validating states and operators
  double validation_threshold_ = 1e-8;

  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Set OpenMP thread settings to default values
  void clear_parallelization();

  // Set parallelization for experiments
  virtual void set_parallelization_experiments(const std::vector<Circuit>& circuits,
                                               const Noise::NoiseModel& noise);

  // Set parallelization for a circuit
  virtual void set_parallelization_circuit(const Circuit& circuit,
                                           const Noise::NoiseModel& noise);

  // Return an estimate of the required memory for a circuit.
  virtual size_t required_memory_mb(const Circuit& circuit,
                                    const Noise::NoiseModel& noise) const = 0;

  // Get system memory size
  size_t get_system_memory_mb();

  // The maximum number of threads to use for various levels of parallelization
  int max_parallel_threads_;

  // Parameters for parallelization management in configuration
  int max_parallel_experiments_;
  int max_parallel_shots_;
  size_t max_memory_mb_;

  // use explicit parallelization
  bool explicit_parallelization_;

  // Parameters for parallelization management for experiments
  int parallel_experiments_;
  int parallel_shots_;
  int parallel_state_update_;

  // Truncate qubits
  bool truncate_qubits_ = true;
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

  // Load qubit truncation
  JSON::get_value(truncate_qubits_, "truncate_enable", config);

  #ifdef _OPENMP
  // Load OpenMP maximum thread settings
  if (JSON::check_key("max_parallel_threads", config))
    JSON::get_value(max_parallel_threads_, "max_parallel_threads", config);
  if (JSON::check_key("max_parallel_experiments", config))
    JSON::get_value(max_parallel_experiments_, "max_parallel_experiments", config);
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
  #endif

  // Load configurations for parallelization
  
  if (JSON::check_key("max_memory_mb", config)) {
    JSON::get_value(max_memory_mb_, "max_memory_mb", config);
  }

  for (std::shared_ptr<Transpile::CircuitOptimization> opt: optimizations_)
    opt->set_config(config);

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
    parallel_experiments_ = std::max<int>( { parallel_experiments_, 1 });
    parallel_shots_ = std::max<int>( { parallel_shots_, 1 });
    parallel_state_update_ = std::max<int>( { parallel_state_update_, 1 });
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

  explicit_parallelization_ = false;
  max_memory_mb_ = get_system_memory_mb() / 2;
}

void Controller::set_parallelization_experiments(const std::vector<Circuit>& circuits,
                                                 const Noise::NoiseModel& noise) {
  // Use a local variable to not override stored maximum based
  // on currently executed circuits
  const auto max_experiments = (max_parallel_experiments_ > 0)
    ? std::min({max_parallel_experiments_, max_parallel_threads_})
    : max_parallel_threads_;
  
  if (max_experiments == 1) {
    // No parallel experiment execution
    parallel_experiments_ = 1;
    return;
  }

  // If memory allows, execute experiments in parallel
  std::vector<size_t> required_memory_mb_list(circuits.size());
  for (size_t j=0; j<circuits.size(); j++) {
    required_memory_mb_list[j] = required_memory_mb(circuits[j], noise);
  }
  std::sort(required_memory_mb_list.begin(), required_memory_mb_list.end(), std::greater<>());
  size_t total_memory = 0;
  parallel_experiments_ = 0;
  for (size_t required_memory_mb : required_memory_mb_list) {
    total_memory += required_memory_mb;
    if (total_memory > max_memory_mb_)
      break;
    ++parallel_experiments_;
  }

  if (parallel_experiments_ <= 0)
    throw std::runtime_error("a circuit requires more memory than max_memory_mb.");
  parallel_experiments_ = std::min<int>({parallel_experiments_,
                                         max_experiments,
                                         max_parallel_threads_,
                                         static_cast<int>(circuits.size())});
}

void Controller::set_parallelization_circuit(const Circuit& circ,
                                             const Noise::NoiseModel& noise) {

  // Use a local variable to not override stored maximum based
  // on currently executed circuits
  const auto max_shots = (max_parallel_shots_ > 0)
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
      throw std::runtime_error("a circuit requires more memory than max_memory_mb.");
    // If circ memory is 0, set it to 1 so that we don't divide by zero
    circ_memory_mb = std::max<int>({1, circ_memory_mb});

    parallel_shots_ = std::min<int>({static_cast<int>(max_memory_mb_ / circ_memory_mb),
                                     max_shots,
                                     static_cast<int>(circ.shots)});
  }
  parallel_state_update_ = (parallel_shots_ > 1)
    ? std::max<int>({1, max_parallel_threads_ / parallel_shots_})
    : std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
}


size_t Controller::get_system_memory_mb(){
  size_t total_physical_memory = 0;
#if defined(__linux__) || defined(__APPLE__)
   auto pages = sysconf(_SC_PHYS_PAGES);
   auto page_size = sysconf(_SC_PAGE_SIZE);
   total_physical_memory = pages * page_size;
#elif defined(_WIN64)
   MEMORYSTATUSEX status;
   status.dwLength = sizeof(status);
   GlobalMemoryStatusEx(&status);
   total_physical_memory = status.ullTotalPhys;
#endif
   return total_physical_memory >> 20;
}

//-------------------------------------------------------------------------
// State validation
//-------------------------------------------------------------------------

template <class state_t>
bool Controller::validate_state(const state_t &state,
                                const Circuit &circ,
                                const Noise::NoiseModel &noise,
                                bool throw_except) {
  // First check if a noise model is valid a given state
  bool noise_valid = noise.is_ideal() || state.validate_opset(noise.opset());
  bool circ_valid = state.validate_opset(circ.opset());
  if (noise_valid && circ_valid)
  {
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
    msg << "Noise model contains invalid instructions (";
    msg << state.invalid_opset_message(noise.opset()) << ")";
  }
  if (!circ_valid) {
    msg << "Circuit contains invalid instructions (";
    msg << state.invalid_opset_message(circ.opset()) << ")";
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
  if(max_memory_mb_ < required_mb) {
    if(throw_except) {
      std::string name = "";
      JSON::get_value(name, "name", circ.header);
      throw std::runtime_error("AER::Base::Controller: State " + state.name() +
                               " has insufficient memory to run the circuit " +
                               name);
    }
    return false;
  }
  return true;
}

//-------------------------------------------------------------------------
// Circuit optimization
//-------------------------------------------------------------------------
template <class state_t>
void Controller::optimize_circuit(Circuit &circ,
                                  Noise::NoiseModel& noise,
                                  state_t& state,
                                  ExperimentData &data) const {

  Operations::OpSet allowed_opset;
  allowed_opset.optypes = state.allowed_ops();
  allowed_opset.gates = state.allowed_gates();
  allowed_opset.snapshots = state.allowed_snapshots();

  for (std::shared_ptr<Transpile::CircuitOptimization> opt: optimizations_) {
    opt->optimize_circuit(circ, noise, allowed_opset, data);
  }
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
    result.metadata["time_taken"] = std::chrono::duration<double>(timer_stop - timer_start).count();
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
    result.metadata["omp_enabled"] = true;
  #else
    result.metadata["omp_enabled"] = false;
  #endif
    result.metadata["parallel_experiments"] = parallel_experiments_;
    result.metadata["max_memory_mb"] = max_memory_mb_;
    

  #ifdef _OPENMP
    if (parallel_shots_ > 1 || parallel_state_update_ > 1)
      omp_set_nested(1);
  #endif
    if (parallel_experiments_ > 1) {
      // Parallel circuit execution
      #pragma omp parallel for num_threads(parallel_experiments_)
      for (int j = 0; j < result.results.size(); ++j) {
        // Make a copy of the noise model for each circuit execution
        // so that it can be modified if required
        auto circ_noise_model = noise_model;
        result.results[j] = execute_circuit(circuits[j],
                                            circ_noise_model,
                                            config);
      }
    } else {
      // Serial circuit execution
      for (int j = 0; j < num_circuits; ++j) {
        // Make a copy of the noise model for each circuit execution
        auto circ_noise_model = noise_model;
        result.results[j] = execute_circuit(circuits[j],
                                            circ_noise_model,
                                            config);
      }
    }

    // Check each experiment result for completed status.
    // If only some experiments completed return partial completed status.
    result.status = Result::Status::completed;
    for (const auto& experiment: result.results) {
      if (experiment.status != ExperimentResult::Status::completed) {
        result.status = Result::Status::partial_completed;
        break;
      }
    }
    // Stop the timer and add total timing data
    auto timer_stop = myclock_t::now();
    result.metadata["time_taken"] = std::chrono::duration<double>(timer_stop - timer_start).count();
  }
  // If execution failed return valid output reporting error
  catch (std::exception &e) {
    result.status = Result::Status::error;
    result.message = e.what();
  }
  return result;
}


ExperimentResult Controller::execute_circuit(Circuit &circ,
                                             Noise::NoiseModel& noise,
                                             const json_t &config) {

  // Start individual circuit timer
  auto timer_start = myclock_t::now(); // state circuit timer

  // Initialize circuit json return
  ExperimentResult exp_result;
  ExperimentData data;
  data.set_config(config);

  // Execute in try block so we can catch errors and return the error message
  // for individual circuit failures.
  try {
  
    // Remove barriers from circuit
    Transpile::ReduceBarrier barrier_pass;
    barrier_pass.optimize_circuit(circ, noise, circ.opset(), data);

    // Truncate unused qubits from circuit and noise model
    if (truncate_qubits_) {
      Transpile::TruncateQubits truncate_pass;
      truncate_pass.set_config(config);
      truncate_pass.optimize_circuit(circ, noise, circ.opset(), data);
    }

    // set parallelization for this circuit
    if (!explicit_parallelization_) {
      set_parallelization_circuit(circ, noise);
    }

    // Single shot thread execution
    if (parallel_shots_ <= 1) {
      auto tmp_data = run_circuit(circ, noise, config, circ.shots, circ.seed);
      data.combine(std::move(tmp_data));
    // Parallel shot thread execution
    } else {
      // Calculate shots per thread
      std::vector<unsigned int> subshots;
      for (int j = 0; j < parallel_shots_; ++j) {
        subshots.push_back(circ.shots / parallel_shots_);
      }
      // If shots is not perfectly divisible by threads, assign the remainder
      for (int j=0; j < int(circ.shots % parallel_shots_); ++j) {
        subshots[j] += 1;
      }

      // Vector to store parallel thread output data
      std::vector<ExperimentData> par_data(parallel_shots_);
      std::vector<std::string> error_msgs(parallel_shots_);
      #pragma omp parallel for if (parallel_shots_ > 1) num_threads(parallel_shots_)
      for (int i = 0; i < parallel_shots_; i++) {
        try {
          par_data[i] = run_circuit(circ, noise, config, subshots[i], circ.seed + i);
        } catch (std::runtime_error &error) {
          error_msgs[i] = error.what();
        }
      }

      for (std::string error_msg: error_msgs)
        if (error_msg != "")
          throw std::runtime_error(error_msg);

      // Accumulate results across shots
      // Use move semantics to avoid copying data
      for (auto &datum : par_data) {
        data.combine(std::move(datum));
      }
    }
    // Report success
    exp_result.data = data;
    exp_result.status = ExperimentResult::Status::completed;

    // Pass through circuit header and add metadata
    exp_result.header = circ.header;
    exp_result.shots = circ.shots;
    exp_result.seed = circ.seed;
    // Move any metadata from the subclass run_circuit data
    // to the experiment resultmetadata field
    for(const auto& pair: exp_result.data.metadata()) {
      exp_result.add_metadata(pair.first, pair.second);
    }
    // Remove the metatdata field from data
    exp_result.data.metadata().clear();
    exp_result.metadata["parallel_shots"] = parallel_shots_;
    exp_result.metadata["parallel_state_update"] = parallel_state_update_;
    // Add timer data
    auto timer_stop = myclock_t::now(); // stop timer
    double time_taken = std::chrono::duration<double>(timer_stop - timer_start).count();
    exp_result.time_taken = time_taken;
  }
  // If an exception occurs during execution, catch it and pass it to the output
  catch (std::exception &e) {
    exp_result.status = ExperimentResult::Status::error;
    exp_result.message = e.what();
  }
  return exp_result;
}

//-------------------------------------------------------------------------
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
