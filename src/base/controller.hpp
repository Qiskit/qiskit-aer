/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
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

// Base Controller
#include "framework/qobj.hpp"
#include "framework/data.hpp"
#include "framework/rng.hpp"
#include "framework/creg.hpp"
#include "framework/circuitopt.hpp"
#include "noise/noise_model.hpp"

#ifdef _OPENMP
#include <omp.h>
#include "misc/hacks.hpp"
#endif

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
 *      executed in parallel. Set to 0 to use the number of max parallel
 *      threads [Default: 1]
 * - "max_parallel_shots" (int): Set number of shots that maybe be executed
 *      in parallel for each circuit. Sset to 0 to use the number of max
 *      parallel threads [Default: 1].
 * - "available_memory" (int): Set the amount of memory available to the
 *      state in MB. If specified, this is divided by the number of parallel
 *      shots/experiments. [Default: 0].
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
  inline auto add_circuit_optimization(Type&& opt)-> typename std::enable_if_t<std::is_base_of<CircuitOptimization, std::remove_const_t<std::remove_reference_t<Type>>>::value >
  {
      optimizations_.push_back(std::make_shared<std::remove_const_t<std::remove_reference_t<Type> > >(std::forward<Type>(opt)));
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
  Circuit optimize_circuit(const Circuit &input_circ) const;

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
  std::vector<std::shared_ptr<CircuitOptimization>> optimizations_;

  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Set OpenMP thread settings to default values
  void clear_parallelization();

  // Set OpenMP thread settings for experiments
  virtual void set_parallelization(Qobj& qobj);

  // The maximum number of threads to use for various levels of parallelization
  int max_parallel_threads_;

  // Parameters for parallelization management in configuration
  int max_parallel_experiments_;
  int max_parallel_shots_;

  // Parameters for parallelization management for experiments
  int parallel_experiments_;
  int parallel_shots_;
  int parallel_state_update_;

  uint_t state_available_memory_mb_ = 0;
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
  JSON::get_value(max_parallel_threads_, "max_parallel_threads", config);
  JSON::get_value(max_parallel_shots_, "max_parallel_shots", config);
  JSON::get_value(max_parallel_experiments_, "max_parallel_experiments", config);

  // Prevent using both parallel circuits and parallel shots
  // with preference given to parallel circuit execution
  if (max_parallel_experiments_ > 1)
    max_parallel_shots_ = 1;
 
  JSON::get_value(state_available_memory_mb_, "available_memory", config);
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
  max_parallel_experiments_ = 1;
  max_parallel_shots_ = 1;

  parallel_experiments_ = 1;
  parallel_shots_ = 1;
  parallel_state_update_ = 1;
}

void Controller::set_parallelization(Qobj& qobj) {

  // Set max_parallel_threads_
  if (max_parallel_threads_ < 1)
  #ifdef _OPENMP
    max_parallel_threads_ = std::max(1, omp_get_max_threads());
  #else
    max_parallel_threads_ = 1;
  #endif

  // Set max_parallel_experiments_
  parallel_experiments_ = (max_parallel_experiments_ < 1)?
      std::min<int>({ (int) qobj.circuits.size(), max_parallel_threads_}):
      std::min<int>({ (int) qobj.circuits.size(), max_parallel_threads_, max_parallel_experiments_});

  int max_num_shots = 0;
  for (Circuit &circ: qobj.circuits)
    max_num_shots = std::max<int>({ max_num_shots, (int) circ.shots});

  parallel_shots_ = (max_parallel_shots_ < 1)?
      std::min<int>({max_num_shots, max_parallel_threads_/parallel_experiments_}):
      std::min<int>({max_num_shots, max_parallel_threads_/parallel_experiments_, max_parallel_shots_});
  parallel_shots_ = std::max<int>({1, parallel_shots_});

  parallel_state_update_ = std::max<int>({1, max_parallel_threads_/(parallel_experiments_*parallel_shots_)});

  if(state_available_memory_mb_ > 0)
  {
    state_available_memory_mb_ /= parallel_shots_;
    state_available_memory_mb_ /= parallel_experiments_;
  }
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
                                  bool throw_except) const
{
  if (state_available_memory_mb_ == 0)
  {
    return true;
  }
  auto required_mb = state.required_memory_mb(circ.num_qubits, circ.ops);
  if(state_available_memory_mb_ < required_mb)
  {
    if(throw_except)
    {
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

Circuit Controller::optimize_circuit(const Circuit &input_circ) const {

  Circuit working_circ = input_circ;
  for (std::shared_ptr<CircuitOptimization> opt: optimizations_)
    opt->optimize_circuit(working_circ);

  return working_circ;
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

    // set parallelization
    set_parallelization(qobj);

  #ifdef _OPENMP
    result["metadata"]["omp_enabled"] = true;
  #else
    result["metadata"]["omp_enabled"] = false;
  #endif
    result["metadata"]["parallel_experiments"] = parallel_experiments_;
    result["metadata"]["parallel_shots"] = parallel_shots_;
    result["metadata"]["parallel_state_update"] = parallel_state_update_;

    const int num_circuits = qobj.circuits.size();

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
      #pragma omp parallel for if (parallel_shots_ > 1) num_threads(parallel_shots_)
        for (int j = 0; j < parallel_shots_; j++) {
          data[j] = run_circuit(circ, subshots[j], circ.seed + j);
        }
      // Accumulate results across shots
      for (size_t j=1; j<data.size(); j++) {
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
    result["seed"] = circ.seed;
    // Move any metadata from the subclass run_circuit data
    // to the experiment resultmetadata field
    if (JSON::check_key("metadata", result["data"])) {

      for(auto& metadata: result["data"]["metadata"].items()) {
        result["metadata"][metadata.key()] = metadata.value();
      }
      // Remove the metatdata field from data
      result["data"].erase("metadata");
    }
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
