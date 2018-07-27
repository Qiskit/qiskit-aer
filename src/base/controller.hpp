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
  virtual ~Controller() = default;
  
  //----------------------------------------------------------------
  // Loading state and engine config
  //----------------------------------------------------------------

  inline void load_engine_config(const json_t &config) {engine_.load_config(config);};
  inline void load_state_config(const json_t &config) {state_.load_config(config);};
  
  inline int get_num_threads() {return num_threads_;};
  inline void set_num_threads(int threads) {num_threads_ = std::min(std::max(1, threads), omp_ncpus_);};
  
  //----------------------------------------------------------------
  // Executing qobj
  //----------------------------------------------------------------

  // Execute a qobj
  virtual json_t execute(const Qobj &qobj, int threads = 1) const;
  
  // Load and execute a qobj
  // These wrap the Qobj loading methods to catch and report errors in
  // loading
  virtual json_t execute(const json_t &qobj_js, int threads = 1) const;

  // Execute a single circuit
  virtual json_t execute_circuit(const Circuit &circ, int threads) const;

protected:

  State_t state_;       // Reference State interface for the controller
  Engine_t engine_;     // Reference Engine for the controller
  
  int omp_ncpus_ = 1;   // The number of available threads determined by OpenMP
  int num_threads_ = 1; // The maximum number of threads to use for parallelization
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
    num_threads_ = omp_ncpus_;
    omp_set_nested(1); // allow nested parallel threads for states
  #endif
}

template < class Engine_t, class State_t>
Controller<Engine_t, State_t>::Controller(State_t state, Engine_t engine)
  : Controller<Engine_t, State_t>() {
    state_ = state;
    engine_ = engine;
}

//============================================================================
// Execution
//============================================================================

template < class Engine_t, class State_t>
json_t Controller<Engine_t, State_t>::execute(const Qobj &qobj, int threads) const{

  // Start Simulation timer
  std::chrono::time_point<myclock_t> start = myclock_t::now(); // start timer

  // Initialize output JSON
  json_t ret;
  bool qobj_success = true;

  // Choose simulator and execute circuits
  try {
    ret["id"] = qobj.id;
    // Execute each circuit in qobj
    for (const auto &circ : qobj.circuits) {
      json_t result = execute_circuit(circ, threads);
      qobj_success &= result["success"].get<bool>();
      ret["result"].push_back(result);
    }
    // Add timing metadata
    ret["time_taken"] =
        std::chrono::duration<double>(myclock_t::now() - start).count();
    // Add success data
    ret["status"] = std::string("COMPLETED");
    ret["success"] = qobj_success;
    #ifdef _OPENMP
    if (omp_ncpus_ > 1)
      ret["omp_num_cpus"] = omp_ncpus_;
    #endif
  } catch (std::exception &e) {
    // If execution failed report error
    ret["success"] = false;
    ret["status"] = std::string("ERROR: ") + e.what();
  }
  return ret;
}

//------------------------------------------------------------------------------

template < class Engine_t, class State_t>
json_t Controller<Engine_t, State_t>::execute(const json_t &qobj_js,
                                              int threads) const {

  // Load QOBJ from json
  Qobj qobj;
  try {
    qobj.load_qobj_from_json(qobj_js);
  } catch (std::exception &e) {
    json_t ret;
    ret["id"] = nullptr;
    ret["success"] = false;
    ret["status"] = std::string("ERROR: Failed to load qobj: ") + e.what();
    return ret;
  }
  // Execute QOBJ
  return execute(qobj, threads);
}

//------------------------------------------------------------------------------

template < class Engine_t, class State_t>
json_t Controller<Engine_t, State_t>::execute_circuit(const Circuit &circ, int threads) const {
  
  // Initialize Return
  json_t ret;
  try {
    auto timer_start = myclock_t::now(); // start timer

    // Initialize new copy of reference engine and state
    Engine_t engine = engine_;
    engine.load_config(circ.config); // load config
    State_t state = state_;
    state.load_config(circ.config); // load config
    state.set_rng_seed(circ.seed); // set rng seed

    // Check operations are allowed
    const auto &allowed = state.allowed_ops;
    for (const auto &op: circ.ops) {
      if (allowed.find(op.name) == allowed.end()) {
        throw std::invalid_argument("Operation \"" + op.name 
                                    + "\" is not allowed by State subclass.");
      }
    }

    // Bound max threads by shots and number of cpu cores
    threads = std::min({std::max(1, threads), num_threads_, static_cast<int>(circ.shots)});
    int num_state_threads = std::max<int>(1, num_threads_ / threads);
    state.set_available_threads(num_state_threads);

    // OpenMP Parallelization
  #ifdef _OPENMP
    if (threads < 2)
      engine.execute(&state, circ, circ.shots);
    else {
      
      // Set shots for each thread
      std::vector<unsigned int> thread_shots;
      for (int j = 0; j < threads; ++j) {
        thread_shots.push_back(circ.shots / threads);
      }
      thread_shots[0] += (circ.shots % threads);

      // Vector to store parallel thread engines
      std::vector<Engine_t> futures(threads);
    #pragma omp parallel for if (threads > 1) num_threads(threads)
      for (int j = 0; j < threads; j++) {
        const auto &tshots = thread_shots[j];
        State_t thread_state(state);
        thread_state.set_rng_seed(circ.seed + j); // shift rng seed for each thread
        Engine_t thread_engine(engine);
        std::cout << circ.config << std::endl;
        thread_engine.execute(&thread_state, circ, tshots);
        futures[j] = std::move(thread_engine);
      }
      // Accumulate results across shots
      for (auto &f : futures)
        engine.combine(f);
    } // end parallel shots

    // Add multi-threading information to output
    if (threads > 1)
      ret["num_threads_shot"] = threads;
    if (num_state_threads > 1)
      ret["num_threads_state"] = num_state_threads;
  #else
    // Non-parallel execution
    engine.execute(&state, circ, circ.shots);
  #endif

   
    
    // Add output data and metadata
    ret["data"] = engine.json();
    // Add metadata
    ret["shots"] = circ.shots;
    ret["seed"] = circ.seed;
    // Report success
    ret["success"] = true;
    ret["status"] = std::string("DONE");

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

//------------------------------------------------------------------------------
} // end namespace Base
} // end namespace AER
//------------------------------------------------------------------------------
#endif