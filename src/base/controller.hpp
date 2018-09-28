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

#ifdef _OPENMP
#include <omp.h>
#endif

// Base Controller
#include "framework/qobj.hpp"
#include "base/engine.hpp"
#include "noise/noise_model.hpp"


namespace AER {
namespace Base {

//============================================================================
// Controller base class
//============================================================================

// This is the top level controller for the Qiskit-Aer simulator
// It manages execution of all the circuits in a QOBJ, parallelization,
// noise sampling from a noise model, and circuit optimizations.

// Parallelization:
//    Parallel execution uses the OpenMP library. It may happen at three
//    levels:
//      1. Parallel execution of circuits in a QOBJ
//      2. Parallel execution of shots in a Circuit
//      3. Parallelization used by the State class for performing gates.
//    Options 1 and 2 are mutually exclusive: enabling circuit parallelization
//    disables shot parallelization and vice versa. Option 3 is available for 
//    both cases but conservatively limits the number of threads since these
//    are subthreads spawned by the higher level threads.

class Controller {
public:

  //-----------------------------------------------------------------------
  // Executing qobj
  //-----------------------------------------------------------------------

  // Load a QOBJ from a JSON file and execute on the State type
  // class.
  template <class State_t>
  json_t execute(const json_t &qobj);

  // Execute a circuit object on the State type
  template <class State_t>
  json_t execute(Circuit &circ);

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  // Load a noise model from a noise_model JSON file
  void load_noise_model(const json_t &config);

  // Load a default Engine config file from a JSON
  void load_engine_config(const json_t &config);

  // Load a default State config file from a JSON
  void load_state_config(const json_t &config);

  // Clear the current noise model
  inline void clear_noise_model() {noise_model_ = Noise::NoiseModel();}

  // Clear the current engine config
  inline void clear_engine_config() {engine_config_ = json_t();}

  // Clear the current state config
  inline void clear_state_config() {state_config_ = json_t();}

  //-----------------------------------------------------------------------
  // OpenMP Parallelization settings
  //-----------------------------------------------------------------------

  // Set the maximum OpenMP threads that may be used across all levels
  // of parallelization. Set to -1 for maximum available.
  void set_max_threads(int max_threads = -1);

  // Return the current value for maximum threads
  inline int get_max_threads() const {return max_threads_total_;}

  // Set the maximum OpenMP threads that may be used for parallel
  // circuit evaluation. Set to -1 for maximum available.
  // Setting this to any number than 1 automatically sets the maximum
  // shot threads to 1.
  void set_max_threads_circuit(int max_threads = -1);

  // Return the current value for maximum circuit threads
  inline int get_max_threads_circuit() const {return max_threads_circuit_;}

  // Set the maximum OpenMP threads that may be used for parallel
  // shot evaluation. Set to -1 for maximum available.
  // Setting this to any number than 1 automatically sets the maximum
  // circuit threads to 1.
  void set_max_threads_shot(int max_threads = -1);

  // Return the current value for maximum shot threads
  inline int get_max_threads_shot() const {return max_threads_shot_;}

  // Set the maximum OpenMP threads that may be by the state class
  // for parallelization of operations. Set to -1 for maximum available.
  void set_max_threads_state(int max_threads = -1);

  // Return the current value for maximum state threads
  inline int get_max_threads_state() const {return max_threads_state_;}

private:

  using myclock_t = std::chrono::high_resolution_clock;

  //----------------------------------------------------------------
  // Circuit Execution
  //----------------------------------------------------------------

  template <class State_t>
  Engine execute_circuit(Circuit &circ,
                         uint_t shots,
                         uint_t state_seed,
                         uint_t noise_seed,
                         int state_threads);
  
  template <class State_t>
  Engine parallel_execute_circuit(Circuit &circ,
                                  uint_t shots,
                                  uint_t state_seed,
                                  uint_t noise_seed,
                                  int num_threads_shot,
                                  int num_threads_state);

  //----------------------------------------------------------------
  // Optimizations
  //----------------------------------------------------------------

  void optimize_circuit(Circuit &circ) const;

  // Check if measurement sampling can be performed for a circuit
  // and set circuit flag if it is compatible                               
  void measure_sampling_optimization(Circuit &circ) const;
  
  //----------------------------------------------------------------
  // Engine and State config
  //----------------------------------------------------------------

  json_t engine_config_;
  json_t state_config_;

  //----------------------------------------------------------------
  // Noise Model
  //----------------------------------------------------------------
  
  Noise::NoiseModel noise_model_;

  //----------------------------------------------------------------
  // Parallelization Config
  //----------------------------------------------------------------

  // Internal counter of number of threads still available for subthreads
  int available_threads_ = 1;

  // The maximum number of threads to use for various levels of parallelization
  int max_threads_total_ = -1; 

  int max_threads_circuit_ = 1; // -1 for maximum available

  int max_threads_shot_ = 1;   // -1 for maximum available
  
  int max_threads_state_ = -1;   // -1 for maximum available

  void add_backend_info(json_t &result) {
    result["backend_name"] = "qiskit_aer_simulator";
    result["backend_version"] = "alpha 0.1";
    result["date"] = "TODO";
  }

};


//============================================================================
// Implementations
//============================================================================

//-------------------------------------------------------------------------
// Config settings
//-------------------------------------------------------------------------

void Controller::set_max_threads(int max_threads) {
  max_threads_total_ = max_threads;
}

void Controller::set_max_threads_circuit(int max_threads) {
  max_threads_circuit_ = max_threads;
  if (max_threads != 1)
    max_threads_shot_ = 1;
}

void Controller::set_max_threads_shot(int max_threads) {
  max_threads_shot_ = max_threads;
  if (max_threads != 1)
    max_threads_circuit_ = 1;
}

void Controller::set_max_threads_state(int max_threads) {
  max_threads_state_ = max_threads;
}

void Controller::load_noise_model(const json_t &config) {
  noise_model_.load_from_json(config);
}

void Controller::load_engine_config(const json_t &config) {
  engine_config_ = config;
}

void Controller::load_state_config(const json_t &config) {
  state_config_ = config;
}


//-------------------------------------------------------------------------
// Circuit Execution
//-------------------------------------------------------------------------

template <class State_t>
json_t Controller::execute(const json_t &qobj_js) {
  
  // Start QOBJ timer
  auto timer_start = myclock_t::now();

  // Load QOBJ in a try block so we can catch parsing errors and still return
  // a valid JSON output containing the error message.
  Qobj qobj;
  try {
    qobj.load_qobj_from_json(qobj_js);
  } 
  catch (std::exception &e) {
    json_t ret;
    ret["id"] = "ERROR";
    ret["success"] = false;
    ret["status"] = std::string("ERROR: Failed to load qobj: ") + e.what();
    add_backend_info(ret);
    return ret; // qobj was invalid, return valid output containing error message
  }

  // Qobj was loaded successfully, now we proceed
  json_t ret;
  bool all_success = true;

  try {
    int num_circuits = qobj.circuits.size();

  // Check for OpenMP and number of available CPUs
  #ifdef _OPENMP
    int omp_ncpus = std::max(1, omp_get_num_procs());
    omp_set_nested(1); // allow nested parallel threads for states
    available_threads_ = omp_ncpus;
    if (max_threads_total_ < 1)
      max_threads_total_ = available_threads_;

    // Calculate threads for parallel circuit execution
    // TODO: add memory checking for limiting thread number
    int num_threads_circuit = (max_threads_circuit_ < 1) 
      ? std::min<int>({num_circuits, available_threads_ , max_threads_total_})
      : std::min<int>({num_circuits, available_threads_ , max_threads_total_, max_threads_circuit_});
    
    // Since threads can spawn subthreads, divide available threads by circuit threads to
    // get the number of sub threads each can spawn
    available_threads_ /= num_threads_circuit;
    
    // Add thread metatdata to output
    ret["metadata"]["omp_enabled"] = true;
    ret["metadata"]["omp_available_threads"] = omp_ncpus;
    ret["metadata"]["omp_circuit_threads"] = num_threads_circuit;
  #else
    ret["metadata"]["omp_enabled"] = false;
  #endif

    // Initialize container to store parallel circuit output
    ret["results"] = std::vector<json_t>(num_circuits);
    
    // Begin parallel circuit execution
  #pragma omp parallel for if (num_threads_circuit > 1) num_threads(num_threads_circuit)
    for (int j = 0; j < num_circuits; ++j) {
      ret["results"][j] = execute<State_t>(qobj.circuits[j]);
    }

    // check success
    for (const auto& res: ret["results"]) {
      all_success &= res["success"].get<bool>();
    }

    // Add success data
    ret["success"] = all_success;
    ret["status"] = std::string("COMPLETED");
    ret["id"] = qobj.id;
    ret["qobj_id"] = "TODO";
    if (!qobj.header.empty())
      ret["header"] = qobj.header;
    add_backend_info(ret);

    // Stop the timer and add total timing data
    auto timer_stop = myclock_t::now();
    ret["metadata"]["time_taken"] = std::chrono::duration<double>(timer_stop - timer_start).count();
  } 
  // If execution failed return valid output reporting error
  catch (std::exception &e) {
    ret["success"] = false;
    ret["status"] = std::string("ERROR: ") + e.what();
  }
  return ret;
}


template <class State_t>
json_t Controller::execute(Circuit &circ) {
  
  // Start individual circuit timer
  auto timer_start = myclock_t::now(); // state circuit timer
  
  // Initialize circuit json return
  json_t ret;

  // Execute in try block so we can catch errors and return the error message
  // for individual circuit failures.
  try {

    // Calculate threads for parallel shot execution
    // We do this rather than in the excute_circuit function so we can add the
    // number of shot threads to the JSON circuit output.
    int num_threads_shot = 1;
    int num_threads_state = 1;
    #ifdef _OPENMP
      int num_shots = circ.shots;
      // Calculate threads for parallel circuit execution
      // TODO: add memory checking for limiting thread number
      num_threads_shot = (max_threads_shot_ < 1) 
        ? std::min<int>({num_shots, available_threads_ , max_threads_total_})
        : std::min<int>({num_shots, available_threads_ , max_threads_total_, max_threads_shot_});
      available_threads_ /= num_threads_shot;

      // Calculate remaining threads for the State class to use
      num_threads_state = (max_threads_state_ < 1) 
        ? std::min<int>({available_threads_ , max_threads_total_,})
        : std::min<int>({available_threads_ , max_threads_total_, max_threads_state_});

      // Add thread information to result metadata
      ret["metadata"]["omp_shot_threads"] = num_threads_shot;
      ret["metadata"]["omp_state_threads"] = num_threads_state;
    #endif

    // Execute circuit
    ret["data"] = parallel_execute_circuit<State_t>(
                    circ, circ.shots, circ.seed, 3*circ.seed,
                    num_threads_shot, num_threads_state).json();
    
    // Report success
    ret["success"] = true;
    ret["status"] = std::string("DONE");

    // Pass through circuit header and add metadata
    ret["header"] = circ.header;
    ret["shots"] = circ.shots;
    ret["seed"] = circ.seed;
    // Add timer data
    auto timer_stop = myclock_t::now(); // stop timer
    double time_taken = std::chrono::duration<double>(timer_stop - timer_start).count();
    ret["time_taken"] = time_taken;
  } 
  // If an exception occurs during execution, catch it and pass it to the output
  catch (std::exception &e) {
    ret["success"] = false;
    ret["status"] = std::string("ERROR: ") + e.what();
  }
  return ret;
}


//-------------------------------------------------------------------------
// Circuit execution engine
//-------------------------------------------------------------------------

template <class State_t>
Engine Controller::execute_circuit(Circuit &circ,
                                   uint_t shots,
                                   uint_t state_seed,
                                   uint_t noise_seed,
                                   int num_threads_state) {
  // Initialize Engine and load config
  Engine engine;
  engine.load_config(engine_config_); // load stored config
  engine.load_config(circ.config); // override with circuit config

  // Initialize state class and load config
  State_t state; 
  state.load_config(state_config_); // load stored config
  state.load_config(circ.config); // override with circuit config
  state.set_rng_seed(state_seed); 
  state.set_available_threads(num_threads_state);

  // Check operations are allowed
  const auto invalid = engine.validate_circuit(circ, state);
  if (!invalid.empty()) {
    std::stringstream ss;
    ss << "Circuit contains invalid operations: " << invalid;
    throw std::invalid_argument(ss.str());
  }

  // Check if there is noise for the implementation
  if (noise_model_.ideal()) {
    // Implement without noise
    optimize_circuit(circ);
    engine.execute(circ, shots, state);
  } else {
    // Sample noise for each shot
    RngEngine noise_rng;
    noise_rng.set_seed(noise_seed);
    while (shots-- > 0) {
      Circuit noise_circ = noise_model_.sample_noise(circ, noise_rng);
      engine.execute(noise_circ, 1, state);
    }
  }
  return engine;
}


template <class State_t>
Engine Controller::parallel_execute_circuit(Circuit &circ,
                                            uint_t shots,
                                            uint_t state_seed,
                                            uint_t noise_seed,
                                            int num_threads_shot,
                                            int num_threads_state) {

  // Calculate shots per thread
  if (num_threads_shot > 1) {
    // Calculate shots per thread
    std::vector<unsigned int> subshots;
    for (int j = 0; j < num_threads_shot; ++j) {
      subshots.push_back(shots / num_threads_shot);
    }
    subshots[0] += (shots % num_threads_shot);

    // Vector to store parallel thread engines
    std::vector<Engine> data(num_threads_shot);
    #pragma omp parallel for if (num_threads_shot > 1) num_threads(num_threads_shot)
      for (int j = 0; j < num_threads_shot; j++) {
        data[j] = execute_circuit<State_t>(circ, subshots[j], state_seed + j,
                                                         noise_seed + j, num_threads_state);
      }
    // Accumulate results across shots 
    for (size_t ii=1; ii<data.size(); ii++) {
      data[0].combine(data[ii]);
    }
    return data[0];
  } else {
    return execute_circuit<State_t>(circ, shots, state_seed,
                                                  noise_seed, num_threads_state);
  }
}


//-------------------------------------------------------------------------
// Circuit optimization
//-------------------------------------------------------------------------

void Controller::optimize_circuit(Circuit &circ) const {
  // Check for measurement optimization
  measure_sampling_optimization(circ);
}


void Controller::measure_sampling_optimization(Circuit &circ) const {
  // Find first instance of a measurement and check there
  // are no reset operations before the measurement
  auto start = circ.ops.begin();
  while (start != circ.ops.end()) {
    const auto name = start->name;
    if (name == "reset" || name == "kraus" || name == "roerr") {
      circ.measure_sampling_flag = false;
      return;
    }
    if (name == "measure")
      break;
    ++start;
  }
  // Check all remaining operations are measurements
  while (start != circ.ops.end()) {
    if (start->name != "measure") {
      circ.measure_sampling_flag = false;
      return;
    }
    ++start;
  }
  // If we made it this far we can apply the optimization
  circ.measure_sampling_flag = true;
  circ.header["memory_sampling_opt"] = true;
  // Now we delete the measure operations:?
}


//------------------------------------------------------------------------------
} // end namespace Base
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
