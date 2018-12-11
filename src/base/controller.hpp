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
#include "framework/data.hpp"
#include "framework/rng.hpp"
#include "framework/creg.hpp"
#include "noise/noise_model.hpp"


namespace AER {
namespace Base {

//=========================================================================
// Controller base class
//=========================================================================

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
  // Execute qobj
  //-----------------------------------------------------------------------

  // Load a QOBJ from a JSON file and execute on the State type
  // class.
  virtual json_t execute(const json_t &qobj);

  // Execute from string to string
  inline virtual std::string execute_string(const std::string &qobj_str) {
    return execute(json_t::parse(qobj_str)).dump(-1);
  }

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  // Load a noise model from a noise_model JSON file
  virtual void set_noise_model(const json_t &config);

  // Load a default OutputData config file from a JSON
  virtual void set_config(const json_t &config);

  // Load a noise model from string
  inline virtual void set_noise_model_string(const std::string &config) {
    set_noise_model(json_t::parse(config));
  }

  // Load state config from string
  inline virtual void set_config_string(const std::string &config) {
    set_config(json_t::parse(config));
  }

  // Clear the current noise model
  inline virtual void clear_noise_model() {noise_model_ = Noise::NoiseModel();}

  // Clear the current config
  inline virtual void clear_config() {config_ = json_t();}

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
                                 uint_t rng_seed,
                                 int num_threads_state) const = 0;

  //-----------------------------------------------------------------------
  // Config
  //-----------------------------------------------------------------------

  // Timer type
  using myclock_t = std::chrono::high_resolution_clock;

  // Config settings
  json_t config_;

  // Noise model
  Noise::NoiseModel noise_model_;

  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Internal counter of number of threads still available for subthreads
  int available_threads_ = 1;

  // The maximum number of threads to use for various levels of parallelization
  int max_threads_total_ = -1; 

  int max_threads_circuit_ = 1; // -1 for maximum available

  int max_threads_shot_ = 1;   // -1 for maximum available
  
  int max_threads_state_ = -1;   // -1 for maximum available

};


//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Config settings: parallelization
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

void Controller::set_noise_model(const json_t &config) {
  noise_model_ = Noise::NoiseModel(config);
}

void Controller::set_config(const json_t &config) {
  config_ = config;
}

//-------------------------------------------------------------------------
// Qobj and Circuit Execution to JSON output
//-------------------------------------------------------------------------

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
    ret["qobj_id"] = "ERROR";
    ret["success"] = false;
    ret["status"] = std::string("ERROR: Failed to load qobj: ") + e.what();
    ret["backend_name"] = nullptr;
    ret["backend_version"] = nullptr;
    ret["date"] = nullptr;
    ret["job_id"] = nullptr;
    return ret; // qobj was invalid, return valid output containing error message
  }

  // Qobj was loaded successfully, now we proceed
  json_t ret;
  bool all_success = true;

  try {
    int num_circuits = qobj.circuits.size();

  // Check for OpenMP and number of available CPUs
  #ifdef _OPENMP
    int omp_nthreads = std::max(1, omp_get_max_threads());
    omp_set_nested(1); // allow nested parallel threads for states
    available_threads_ = omp_nthreads;
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
    ret["metadata"]["omp_available_threads"] = omp_nthreads;
    ret["metadata"]["omp_circuit_threads"] = num_threads_circuit;
  #else
    ret["metadata"]["omp_enabled"] = false;
  #endif

    // Initialize container to store parallel circuit output
    ret["results"] = std::vector<json_t>(num_circuits);
    
    if (num_threads_circuit > 1) {
      // Parallel circuit execution
      #pragma omp parallel for if (num_threads_circuit > 1) num_threads(num_threads_circuit)
      for (int j = 0; j < num_circuits; ++j) {
        ret["results"][j] = execute_circuit(qobj.circuits[j]);
      }
    } else {
      // Serial circuit execution
      for (int j = 0; j < num_circuits; ++j) {
        ret["results"][j] = execute_circuit(qobj.circuits[j]);
      }
    }

    // check success
    for (const auto& res: ret["results"]) {
      all_success &= res["success"].get<bool>();
    }

    // Add success data
    ret["success"] = all_success;
    ret["status"] = std::string("COMPLETED");
    ret["qobj_id"] = qobj.id;
    if (!qobj.header.empty())
      ret["header"] = qobj.header;
    ret["backend_name"] = nullptr;
    ret["backend_version"] = nullptr;
    ret["date"] = nullptr;
    ret["job_id"] = nullptr;

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


json_t Controller::execute_circuit(Circuit &circ) {
  
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

    // Single shot thread execution
    if (num_threads_shot <= 1) {
      ret["data"] = run_circuit(circ, circ.shots, circ.seed, num_threads_state);
    // Parallel shot thread execution
    } else {
      // Calculate shots per thread
      std::vector<unsigned int> subshots;
      for (int j = 0; j < num_threads_shot; ++j) {
        subshots.push_back(circ.shots / num_threads_shot);
      }
      subshots[0] += (circ.shots % num_threads_shot);

      // Vector to store parallel thread output data
      std::vector<OutputData> data(num_threads_shot);
      #pragma omp parallel for if (num_threads_shot > 1) num_threads(num_threads_shot)
        for (int j = 0; j < num_threads_shot; j++) {
          data[j] = run_circuit(circ, subshots[j], circ.seed + j, num_threads_state);
        }
      // Accumulate results across shots 
      for (size_t j=1; j<data.size(); j++) {
        data[0].combine(data[j]);
      }
      // Update output
      ret["data"] = data[0];
    }
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
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
