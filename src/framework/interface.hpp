/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_interface_hpp_
#define _aer_interface_hpp_

#include "base/controller.hpp"

namespace AER {

//============================================================================
// Interface class for wrapping Controllers for Python
//============================================================================

class Interface {
public:
  
  //-----------------------------------------------------------------------
  // Execution
  //-----------------------------------------------------------------------

  // Execute from string to string
  template <class State_t>
  inline std::string execute(const std::string &qobj_str) {
    return controller_.execute<State_t>(json_t::parse(qobj_str)).dump(-1);
  }

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  // Load controller config from string
  inline void load_noise_model(const std::string &config) {
    controller_.load_noise_model(json_t::parse(config));
  }

  // Load controller config from string
  inline void load_state_config(const std::string &config) {
    controller_.load_state_config(json_t::parse(config));
  }

  // Load engine config from string
  inline void load_engine_config(const std::string &config) {
    controller_.load_engine_config(json_t::parse(config));
  }

  // Load controller config from string
  inline void clear_noise_model() {
    controller_.clear_noise_model();
  }

  // Load controller config from string
  inline void clear_state_config() {
    controller_.clear_state_config();
  }

  // Load engine config from string
  inline void clear_engine_config() {
    controller_.clear_engine_config();
  }

  //-----------------------------------------------------------------------
  // OpenMP Parallelization settings
  //-----------------------------------------------------------------------

  // Set the maximum OpenMP threads that may be used across all levels
  // of parallelization. Set to -1 for maximum available.
  inline void set_max_threads(int max_threads = -1) {
    controller_.set_max_threads(max_threads);
  }

  // Return the current value for maximum threads
  inline int get_max_threads() const {
    return controller_.get_max_threads();
  }

  // Set the maximum OpenMP threads that may be used for parallel
  // circuit evaluation. Set to -1 for maximum available.
  // Setting this to any number than 1 automatically sets the maximum
  // shot threads to 1.
  void set_max_threads_circuit(int max_threads = -1) {
    controller_.set_max_threads_circuit(max_threads);
  }
  // Return the current value for maximum circuit threads
  inline int get_max_threads_circuit() const {
    return controller_.get_max_threads_circuit();
  }

  // Set the maximum OpenMP threads that may be used for parallel
  // shot evaluation. Set to -1 for maximum available.
  // Setting this to any number than 1 automatically sets the maximum
  // circuit threads to 1.
  void set_max_threads_shot(int max_threads = -1) {
    controller_.set_max_threads_shot(max_threads);
  }

  // Return the current value for maximum shot threads
  inline int get_max_threads_shot() const {
    return controller_.get_max_threads_shot();
  }

  // Set the maximum OpenMP threads that may be by the state class
  // for parallelization of operations. Set to -1 for maximum available.
  void set_max_threads_state(int max_threads = -1) {
    controller_.set_max_threads_state(max_threads);
  }

  // Return the current value for maximum state threads
  inline int get_max_threads_state() const {
    return controller_.get_max_threads_state();
  }

private:
  // The controller being interfaced with python
  Base::Controller controller_;
};

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif