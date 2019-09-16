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
#include "noise/noise_model.hpp"


namespace AER {
namespace Transpile {

class CircuitOptimization {
public:

  CircuitOptimization() = default;
  virtual ~CircuitOptimization() = default;

  virtual void optimize_circuit(Circuit& circ,
                                Noise::NoiseModel& noise,
                                const Operations::OpSet &opset,
                                ExperimentData &data) const = 0;

  virtual void set_config(const json_t &config);

protected:
  json_t config_;
};

void CircuitOptimization::set_config(const json_t& config) {
  config_ = config;
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
