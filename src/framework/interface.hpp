/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    interface.hpp
 * @brief   Python Interface Class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_interface_hpp_
#define _aer_interface_hpp_

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/json.hpp"

namespace AER {

//============================================================================
// Interface class for wrapping Controllers for Python
//============================================================================

template <class controller_t>
class Interface {
public:

  // Constructors
  Interface() = default;
  Interface(const controller_t &ctrlr) {controller_ = ctrlr;};
  Interface(controller_t &&ctrlr) {controller_ = std::move(ctrlr);};
  virtual ~Interface() = default;
  
  // Execute from string to string
  inline std::string execute(const std::string &qobj_str) {
    return controller_.execute(json_t::parse(qobj_str)).dump(-1);
  };

  // Load controller config from string
  inline void load_controller_config(std::string config) {
    controller_.load_config(json_t::parse(config));
  };

  // Load engine config from string
  inline void load_engine_config(std::string config) {
    controller_.load_engine_config(json_t::parse(config));
  };

  // Load state config from string
  inline void load_state_config(std::string config) {
    controller_.load_state_config(json_t::parse(config));
  };

  // Get number of threads for the controller
  inline int get_num_threads() {
    return controller_.get_num_threads();
  };

  // Set number of parallelization threads for controller
  inline void set_num_threads(int threads) {
    controller_.set_num_threads(threads);
  };

private:
  controller_t controller_;
};

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif