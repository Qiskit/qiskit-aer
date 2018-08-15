/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    Controller.hpp
 * @brief   Controller base class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
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
#include "base/noise.hpp"

namespace AER {
namespace Base {

using myclock_t = std::chrono::high_resolution_clock;

//============================================================================
// Controller base class
//============================================================================

template < class Engine_t, class State_t>
class Controller {
public:
  
  // Constructor
  Controller();

  Controller(State_t state, Engine_t engine);
  
  //----------------------------------------------------------------
  // Loading state and engine config
  //----------------------------------------------------------------

  void load_config(const json_t &config);

  inline void load_engine_config(const json_t &config) {engine_.load_config(config);}

  inline void load_state_config(const json_t &config) {state_.load_config(config);}
  
  inline int get_num_threads() {return thread_limit_;}

  inline void set_num_threads(int threads) {
    thread_limit_ = std::min(std::max(1, threads), omp_ncpus_);
  }
  
  //----------------------------------------------------------------
  // Executing qobj
  //----------------------------------------------------------------
  
  // Load and execute a qobj
  inline json_t execute(const json_t &qobj) {return execute(qobj, nullptr);}

  json_t execute(const json_t &qobj, Noise::Model *noise_ptr);

  // Execute a single circuit
  json_t execute_circuit(Circuit &circ, 
                         int max_shot_threads,
                         Noise::Model *noise_ptr);

  // Check if measurement sampling can be performed for a circuit
  // and set circuit flag if it is compatible                               
  void check_measure_sampling_opt(Circuit &circ) const;

protected:

  State_t state_;       // Reference State interface for the controller
  Engine_t engine_;     // Reference Engine for the controller
  
  int omp_ncpus_ = 1;   // The number of available threads determined by OpenMP
  int available_threads_ = 1;

  int thread_limit_ = 1; // The maximum number of threads to use for parallelization
  int thread_limit_circuit_ = -1;
  int thread_limit_shots_ = 1;
  int thread_limit_state_ = -1;
};

/*******************************************************************************
 *
 * Controller Methods
 *
 ******************************************************************************/

template <class Engine_t, class State_t>
Controller<Engine_t, State_t>::Controller() {
  // OpenMP Setup
  #ifdef _OPENMP
    omp_ncpus_ = std::max(1, omp_get_num_procs());
    thread_limit_ = omp_ncpus_;
    omp_set_nested(1); // allow nested parallel threads for states
  #endif
}

template < class Engine_t, class State_t>
Controller<Engine_t, State_t>::Controller(State_t state, Engine_t engine)
  : Controller<Engine_t, State_t>() {
    state_ = state;
    engine_ = engine;
}

template < class Engine_t, class State_t>
void Controller<Engine_t, State_t>::load_config(const json_t &config) {
  JSON::get_value(thread_limit_, "parallel_thread_limit", config);
  JSON::get_value(thread_limit_circuit_, "parallel_circuit_thread_limit", config);
  JSON::get_value(thread_limit_shots_, "parallel_shots_thread_limit", config);
  JSON::get_value(thread_limit_state_, "parallel_state_thread_limit", config);
}

//============================================================================
// Execution
//============================================================================

template < class Engine_t, class State_t>
json_t Controller<Engine_t, State_t>::execute(const json_t &qobj_js,
                                              Noise::Model *noise_ptr) {
  
  auto timer_start = myclock_t::now(); // start timer
  Qobj qobj; // Load QOBJ from json
  try {
    qobj.load_qobj_from_json(qobj_js);
  } catch (std::exception &e) {
    json_t ret;
    ret["id"] = "ERROR";
    ret["success"] = false;
    ret["status"] = std::string("ERROR: Failed to load qobj: ") + e.what();
    return ret; // qobj was invalid, return valid output containing error message
  }
  // Check for config in qobj
  if (JSON::check_key("config", qobj_js)) {
    load_config(qobj_js["config"]);
  }
  // Qobj was loaded successfully, now we proceed
  json_t ret;
  bool all_success = true;
  try {
    ret["id"] = qobj.id;
    // Check for header in qobj
    if (JSON::check_key("header", qobj_js) && qobj_js["header"].is_object()) {
      ret["header"] = qobj_js["header"];
    }
    int num_circuits = qobj.circuits.size();
    
    // Parallelization preference circuits > shots > backend
    // Calculate threads assigned to parallel circuit execution
    available_threads_ = thread_limit_; // reset available threads to thread limit
    int circ_threads = (thread_limit_circuit_ < 1) 
      ? std::min<int>(available_threads_ , num_circuits)
      : std::min<int>({available_threads_ , num_circuits, thread_limit_circuit_});
    available_threads_ /= circ_threads; // reduce available threads for each subthread   
    // If we are using circuit threads, set shot threads to 1
    int shot_threads = (circ_threads > 1) ? 1 : thread_limit_circuit_;
    ret["result"] = std::vector<json_t>(num_circuits); // initialize correct size for results

    // Parallel circuit execution
    #pragma omp parallel for if (circ_threads > 1) num_threads(circ_threads)
    for (int j = 0; j < num_circuits; ++j) {
      ret["result"][j] = execute_circuit(qobj.circuits[j],
                                         shot_threads,
                                         noise_ptr);
    }
    // check success
    for (const auto& res: ret["result"]) {
      all_success &= res["success"].get<bool>();
    }
    // Add success data
    ret["success"] = all_success;
    ret["status"] = std::string("COMPLETED");

    // Add metadata (change to header?)
    #ifdef _OPENMP
    if (omp_ncpus_ > 1)
      ret["metadata"]["num_openmp_threads"] = omp_ncpus_;
    #endif
    ret["metadata"]["num_circuit_threads"] = circ_threads;
    auto timer_stop = myclock_t::now(); // stop timer
    ret["metadata"]["time_taken"] = std::chrono::duration<double>(timer_stop - timer_start).count();
  } 
  // If execution failed return valid output reporting error
  catch (std::exception &e) {
    ret["success"] = false;
    ret["status"] = std::string("ERROR: ") + e.what();
  }
  return ret;
}


template <class Engine_t, class State_t>
json_t Controller<Engine_t, State_t>::execute_circuit(Circuit &circ,
                                                      int max_shot_threads,
                                                      Noise::Model *noise_ptr) {
  
  // Initialize Return
  auto timer_start = myclock_t::now(); // state circuit timer
  json_t ret;
  try {
    // Check measurement sampling optimization and set flag
    check_measure_sampling_opt(circ);

    // Initialize new copy of reference engine and state
    Engine_t engine = engine_;
    engine.load_config(circ.config); // load config
    State_t state = state_;
    state.load_config(circ.config); // load config
    state.set_rng_seed(circ.seed); // set rng seed

    // Check operations are allowed
    const auto invalid = engine.validate_circuit(&state, circ);
    if (!invalid.empty()) {
      std::stringstream ss;
      ss << "Circuit contains invalid operations: " << invalid;
      throw std::invalid_argument(ss.str());
    }

    // Set number of shot parallel threads
    int num_shots = circ.shots;
    int shot_threads = (max_shot_threads < 1) 
      ? std::min<int>({available_threads_ , num_shots, thread_limit_shots_})
      : std::min<int>({available_threads_ , num_shots, max_shot_threads, thread_limit_shots_});
    // reduce available threads left for state subthreads
    available_threads_ /= shot_threads; 
    int state_threads = (thread_limit_state_ < 1)
      ? std::max<int>(1, available_threads_)
      : std::max<int>(1, std::min<int>(available_threads_, thread_limit_state_));
    state.set_available_threads(state_threads);

    // OpenMP Parallelization
  #ifdef _OPENMP
    if (shot_threads < 2)
      engine.execute(circ, circ.shots, &state, noise_ptr);
    else {
      
      // Set shots for each thread
      std::vector<unsigned int> subshots;
      for (int j = 0; j < shot_threads; ++j) {
        subshots.push_back(circ.shots / shot_threads);
      }
      subshots[0] += (circ.shots % shot_threads);

      // Vector to store parallel thread engines
      std::vector<Engine_t> data(shot_threads);
    #pragma omp parallel for if (shot_threads > 1) num_threads(shot_threads)
      for (int j = 0; j < shot_threads; j++) {
        const auto &ssj = subshots[j];
        State_t thread_state(state);
        thread_state.set_rng_seed(circ.seed + j); // shift rng seed for each thread
        Engine_t thread_engine(engine);
        thread_engine.execute(circ, ssj, &thread_state, noise_ptr);
        data[j] = std::move(thread_engine);
      }
      // Accumulate results across shots
      for (auto &d : data)
        engine.combine(d);
    } // end parallel shots

    // Add multi-threading information to output
    if (shot_threads > 1)
      ret["metadata"]["num_shot_threads"] = shot_threads;
    if (state_threads > 1)
      ret["metadata"]["num_state_threads"] = state_threads;
  #else
    // Non-parallel execution
    engine.execute(circ, circ.shots, &state, noise_ptr);
  #endif
    
    // Add output data and metadata
    ret["data"] = engine.json();
    
    // Report success
    ret["success"] = true;
    ret["status"] = std::string("DONE");

    // Pass through circuit header and add metadata
    ret["header"] = circ.header;
    ret["metadata"]["shots"] = circ.shots;
    ret["metadata"]["seed"] = circ.seed;
    // Add timer data
    auto timer_stop = myclock_t::now(); // stop timer
    double time_taken = std::chrono::duration<double>(timer_stop - timer_start).count();
    ret["metadata"]["time_taken"] = time_taken;
  } 
  // If an exception occurs during execution, catch it and pass it to the output
  catch (std::exception &e) {
    ret["success"] = false;
    ret["status"] = std::string("ERROR: ") + e.what();
  }
  return ret;
}


template < class Engine_t, class State_t>
void Controller<Engine_t, State_t>::check_measure_sampling_opt(Circuit &circ) const {
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
} // end namespace AER
//------------------------------------------------------------------------------
#endif