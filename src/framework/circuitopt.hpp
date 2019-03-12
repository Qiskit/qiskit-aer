/**
 * Copyright 2019, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_circuit_optimization_hpp_
#define _aer_circuit_optimization_hpp_

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/controller.hpp"
#include "base/state.hpp"

namespace AER {

class CircuitOptimization {
public:
  virtual void optimize_circuit(Circuit& circ) const = 0;
  void set_config(const json_t &config);

protected:
  json_t config_;
};

void CircuitOptimization::set_config(const json_t& config) {
  for ( auto it = config.begin(); it != config.end(); ++it )
    config_[it.key()] = it.value();
}

//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
