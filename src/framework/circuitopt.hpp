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

#include "framework/operations.hpp"

namespace AER {

class CircuitOptimization {
public:
  virtual void optimize_circuit(Circuit& circ,
                                const Operations::OpSet &opset,
                                OutputData &data) const = 0;

  virtual void set_config(const json_t &config);

protected:
  json_t config_;
};

void CircuitOptimization::set_config(const json_t& config) {
  config_ = config;
}

//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
