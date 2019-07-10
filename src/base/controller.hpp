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

// Base Controller
#include "framework/qobj.hpp"
#include "framework/data.hpp"
#include "framework/rng.hpp"
#include "framework/creg.hpp"
#include "transpile/circuitopt.hpp"
#include "noise/noise_model.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#include "misc/hacks.hpp"

namespace AER {

//=========================================================================
// Controller Execute interface
//=========================================================================

// This is used to make wrapping Controller classes in Cython easier
// by handling the parsing of std::string input into JSON objects.
template <class controller_t>
std::string controller_execute(const std::string &qobj_str) {
  controller_t controller;
  auto qobj_js = json_t::parse(qobj_str);
  // Check for config
  if (JSON::check_key("config", qobj_js)) {
    controller.set_config(qobj_js["config"]);
  }
  return controller.execute(qobj_js).dump(-1);
}

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
 *      parallel threads. [Default: 0]
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
  virtual json_t execute(const json_t &qobj);

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
  virtual json_t execute_circuit(Circuit &circ);

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  virtual OutputData run_circuit(const Circuit &circ,
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
  bool validate_memory_requirements(state_t &state,
                                    const Circuit &circ,
                                    bool throw_except = false) const;

  //-------------------------------------------------------------------------
  // Circuit optimization
  //-------------------------------------------------------------------------

  // Generate an equivalent circuit with input_circ as output_circ.
  template <class state_t>
  void optimize_circuit(Circuit &input_circ,
                           state_t& state,
                           OutputData &data) const;

  //-----------------------------------------------------------------------
  // Config
  //-----------------------------------------------------------------------

  // Timer type
  using myclock_t = std::chrono::high_resolution_clock;

  // Controller config settings
  json_t config_;

  // Noise model
  Noise::NoiseModel noise_model_;

  // Circuit optimization
  std::vector<std::shared_ptr<Transpile::CircuitOptimization>> optimizations_;

  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Set OpenMP thread settings to default values
  void clear_parallelization();

  // Set parallelization for experiments
  virtual void set_parallelization_experiments(const std::vector<Circuit>& circuits);

  // Set parallelization for a circuit
  virtual void set_parallelization_circuit(const Circuit& circuit);

  // Return an estimate of the required memory for a circuit.
  virtual size_t required_memory_mb(const Circuit& circuit) const = 0;

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

};


//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Config settings
//-------------------------------------------------------------------------

void Controller::set_config(const json_t &config) {
  // Save config for passing to State and Data classes
  config_ = config;

  // Load noise model
  if (JSON::check_key("noise_model", config))
    noise_model_ = Noise::NoiseModel(config["noise_model"]);

  // Load OpenMP maximum thread settings
  if (JSON::check_key("max_parallel_threads", config))
    JSON::get_value(max_parallel_threads_, "max_parallel_threads", config);

  // Load configurations for parallelization
  if (JSON::check_key("max_parallel_experiments", config))
    JSON::get_value(max_parallel_experiments_, "max_parallel_experiments", config);
  if (JSON::check_key("max_parallel_shots", config))
    JSON::get_value(max_parallel_shots_, "max_parallel_shots", config);
  if (JSON::check_key("max_memory_mb", config)) {
    JSON::get_value(max_memory_mb_, "max_memory_mb", config);
  } else {
    auto system_memory_mb = get_system_memory_mb();
    max_memory_mb_ = system_memory_mb / 2;
  }

  for (std::shared_ptr<Transpile::CircuitOptimization> opt: optimizations_)
    opt->set_config(config_);

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

  std::string path;
  JSON::get_value(path, "library_dir", config);
  // Fix for MacOS and OpenMP library double initialization crash.
  // Issue: https://github.com/Qiskit/qiskit-aer/issues/1
  Hacks::maybe_load_openmp(path);
}

void Controller::clear_config() {
  config_ = json_t();
  noise_model_ = Noise::NoiseModel();
  clear_parallelization();
}

void Controller::clear_parallelization() {
  max_parallel_threads_ = 0;
  max_parallel_experiments_ = 0;
  max_parallel_shots_ = 0;

  parallel_experiments_ = 1;
  parallel_shots_ = 1;
  parallel_state_update_ = 1;

  explicit_parallelization_ = false;
}

void Controller::set_parallelization_experiments(const std::vector<Circuit>& circuits) {

  if (max_parallel_experiments_ <= 0)
    return;

  // if memory allows, execute experiments in parallel
  std::vector<size_t> required_memory_mb_list;
  for (const Circuit &circ : circuits)
    required_memory_mb_list.push_back(required_memory_mb(circ));
  std::sort(required_memory_mb_list.begin(), required_memory_mb_list.end(), std::greater<size_t>());

  int total_memory = 0;
  parallel_experiments_ = 0;
  for (int required_memory_mb : required_memory_mb_list) {
    total_memory += required_memory_mb;
    if (total_memory > max_memory_mb_)
      break;
    ++parallel_experiments_;
  }

  if (parallel_experiments_ == 0) {
    throw std::runtime_error("a circuit requires more memory than max_memory_mb.");
  } else if (parallel_experiments_ != 1) {
    parallel_experiments_ = std::min<int> ({ parallel_experiments_,
                                             max_parallel_experiments_,
                                             max_parallel_threads_,
                                             static_cast<int>(circuits.size()) });
    max_parallel_shots_ = 1;
  }
}

void Controller::set_parallelization_circuit(const Circuit& circ) {

  if (max_parallel_threads_ < max_parallel_shots_)
    max_parallel_shots_ = max_parallel_threads_;

  int circ_memory_mb = required_memory_mb(circ);

  if (max_memory_mb_ < circ_memory_mb)
    throw std::runtime_error("a circuit requires more memory than max_memory_mb.");

  if (circ_memory_mb == 0) {
    parallel_shots_ = max_parallel_threads_;
    parallel_state_update_ = 1;
  } else if (max_parallel_shots_ > 0) {
    parallel_shots_ = std::min<int> ({ static_cast<int>(max_memory_mb_ / circ_memory_mb),
                                       max_parallel_shots_,
                                       static_cast<int>(circ.shots) });
    parallel_state_update_ = max_parallel_threads_ / parallel_shots_;
  } else {
    // try to use all the threads for shot-level parallelization
    // no nested parallelization if max_parallel_shots is not configured
    parallel_shots_ = std::min<int> ({ static_cast<int>(max_memory_mb_ / circ_memory_mb),
                                       max_parallel_threads_,
                                       static_cast<int>(circ.shots) });
    if (parallel_shots_ == max_parallel_threads_) {
      parallel_state_update_ = 1;
    } else {
      parallel_shots_ = 1;
      parallel_state_update_ = max_parallel_threads_;
    }
  }
}


size_t Controller::get_system_memory_mb(void){
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
  bool noise_valid = noise.ideal() || state.validate_opset(noise.opset());
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
bool Controller::validate_memory_requirements(state_t &state,
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
void Controller::optimize_circuit(Circuit &input_circ,
                                     state_t& state,
                                     OutputData &data) const {

  Operations::OpSet allowed_opset;
  allowed_opset.optypes = state.allowed_ops();
  allowed_opset.gates = state.allowed_gates();
  allowed_opset.snapshots = state.allowed_snapshots();

  for (std::shared_ptr<Transpile::CircuitOptimization> opt: optimizations_) {
    opt->optimize_circuit(input_circ, allowed_opset, data);
  }
}

//-------------------------------------------------------------------------
// Qobj and Circuit Execution to JSON output
//-------------------------------------------------------------------------

json_t Controller::execute(const json_t &qobj_js) {

  // Start QOBJ timer
  auto timer_start = myclock_t::now();

  // Generate empty return JSON that matches Result spec
  json_t result;
  result["qobj_id"] = nullptr;
  result["success"] = true;
  result["status"] = nullptr;
  result["backend_name"] = nullptr;
  result["backend_version"] = nullptr;
  result["date"] = nullptr;
  result["job_id"] = nullptr;

  // Load QOBJ in a try block so we can catch parsing errors and still return
  // a valid JSON output containing the error message.
  Qobj qobj;
  try {
    qobj.load_qobj_from_json(qobj_js);
  }
  catch (std::exception &e) {
    // qobj was invalid, return valid output containing error message
    result["success"] = false;
    result["status"] = std::string("ERROR: Failed to load qobj: ") + e.what();
    return result;
  }

  // Get QOBJ id and pass through header to result
  result["qobj_id"] = qobj.id;
  if (!qobj.header.empty())
      result["header"] = qobj.header;

  // Qobj was loaded successfully, now we proceed
  try {

    // Set max_parallel_threads_
    if (max_parallel_threads_ < 1)
    #ifdef _OPENMP
      max_parallel_threads_ = std::max(1, omp_get_max_threads());
    #else
      max_parallel_threads_ = 1;
    #endif

    if (!explicit_parallelization_) {
      // set parallelization for experiments
      set_parallelization_experiments(qobj.circuits);
    }

  #ifdef _OPENMP
    result["metadata"]["omp_enabled"] = true;
  #else
    result["metadata"]["omp_enabled"] = false;
  #endif
    result["metadata"]["parallel_experiments"] = parallel_experiments_;
    result["metadata"]["max_memory_mb"] = max_memory_mb_;

    const int num_circuits = qobj.circuits.size();

  #ifdef _OPENMP
    if (parallel_shots_ > 1 || parallel_state_update_ > 1)
      omp_set_nested(1);
  #endif

    // Initialize container to store parallel circuit output
    result["results"] = std::vector<json_t>(num_circuits);
    if (parallel_experiments_ > 1) {
      // Parallel circuit execution
      #pragma omp parallel for num_threads(parallel_experiments_)
      for (int j = 0; j < num_circuits; ++j) {
        result["results"][j] = execute_circuit(qobj.circuits[j]);
      }
    } else {
      // Serial circuit execution
      for (int j = 0; j < num_circuits; ++j) {
        result["results"][j] = execute_circuit(qobj.circuits[j]);
      }
    }

    // check success
    for (const auto& experiment: result["results"]) {
      if (experiment["success"].get<bool>() == false) {
        result["success"] = false;
        break;
      }
    }
    // Set status to completed
    result["status"] = std::string("COMPLETED");

    // Stop the timer and add total timing data
    auto timer_stop = myclock_t::now();
    result["metadata"]["time_taken"] = std::chrono::duration<double>(timer_stop - timer_start).count();
  }
  // If execution failed return valid output reporting error
  catch (std::exception &e) {
    result["success"] = false;
    result["status"] = std::string("ERROR: ") + e.what();
  }
  return result;
}


json_t Controller::execute_circuit(Circuit &circ) {

  // Start individual circuit timer
  auto timer_start = myclock_t::now(); // state circuit timer

  // Initialize circuit json return
  json_t result;

  // Execute in try block so we can catch errors and return the error message
  // for individual circuit failures.
  try {
    // set parallelization for this circuit
    if (!explicit_parallelization_ && parallel_experiments_ == 1) {
      set_parallelization_circuit(circ);
    }
    // Single shot thread execution
    if (parallel_shots_ <= 1) {
      result["data"] = run_circuit(circ, circ.shots, circ.seed);
    // Parallel shot thread execution
    } else {
      // Calculate shots per thread
      std::vector<unsigned int> subshots;
      for (int j = 0; j < parallel_shots_; ++j) {
        subshots.push_back(circ.shots / parallel_shots_);
      }
      // If shots is not perfectly divisible by threads, assign the remaineder
      for (int j=0; j < int(circ.shots % parallel_shots_); ++j) {
        subshots[j] += 1;
      }

      // Vector to store parallel thread output data
      std::vector<OutputData> data(parallel_shots_);
      std::vector<std::string> error_msgs(parallel_shots_);
      #pragma omp parallel for if (parallel_shots_ > 1) num_threads(parallel_shots_)
      for (int i = 0; i < parallel_shots_; i++) {
        try {
          data[i] = run_circuit(circ, subshots[i], circ.seed + i);
        } catch (std::runtime_error &error) {
          error_msgs[i] = error.what();
        }
      }

      for (std::string error_msg: error_msgs)
        if (error_msg != "")
          throw std::runtime_error(error_msg);

      // Accumulate results across shots
      for (uint_t j=1; j<data.size(); j++) {
        data[0].combine(data[j]);
      }
      // Update output
      result["data"] = data[0];
    }
    // Report success
    result["success"] = true;
    result["status"] = std::string("DONE");

    // Pass through circuit header and add metadata
    result["header"] = circ.header;
    result["shots"] = circ.shots;
    result["seed_simulator"] = circ.seed;
    // Move any metadata from the subclass run_circuit data
    // to the experiment resultmetadata field
    if (JSON::check_key("metadata", result["data"])) {

      for(auto& metadata: result["data"]["metadata"].items()) {
        result["metadata"][metadata.key()] = metadata.value();
      }
      // Remove the metatdata field from data
      result["data"].erase("metadata");
    }
    result["metadata"]["parallel_shots"] = parallel_shots_;
    result["metadata"]["parallel_state_update"] = parallel_state_update_;
    // Add timer data
    auto timer_stop = myclock_t::now(); // stop timer
    double time_taken = std::chrono::duration<double>(timer_stop - timer_start).count();
    result["time_taken"] = time_taken;
  }
  // If an exception occurs during execution, catch it and pass it to the output
  catch (std::exception &e) {
    result["success"] = false;
    result["status"] = std::string("ERROR: ") + e.what();
  }
  return result;
}

//-------------------------------------------------------------------------
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
