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

#include <vector>
#include <algorithm>

#include "framework/opset.hpp"
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
                                ExperimentResult &result) const;

  virtual void optimize_circuit(Circuit& circ,
                                Noise::NoiseModel& noise,
                                const Operations::OpSet &opset,
                                uint_t ops_start,
                                uint_t ops_end,
                                ExperimentResult &result) const { };

  virtual void reduce_results(Circuit& circ,
                              ExperimentResult &result,
                              std::vector<ExperimentResult> &results) const;

  virtual void set_config(const json_t &config);

  virtual void set_parallelization(uint_t num) { parallelization_ = num; };

  virtual void set_parallelization_threshold(uint_t num) { parallel_threshold_ = num; };

protected:
  json_t config_;
  uint_t parallelization_ = 1;
  uint_t parallel_threshold_ = 10000;
};

void CircuitOptimization::set_config(const json_t &config) {
  config_ = config;
}

void CircuitOptimization::optimize_circuit(Circuit& circ,
                                           Noise::NoiseModel& noise,
                                           const Operations::OpSet &opset,
                                           ExperimentResult &result) const {

  if (circ.ops.size() < parallel_threshold_ || parallelization_ <= 1) {
    std::vector<ExperimentResult> results(1);
    optimize_circuit(circ, noise, opset, 0, circ.ops.size(), results[0]);
    reduce_results(circ, result, results);
  } else {
    // determine unit for each OMP thread
    int_t unit = circ.ops.size() / parallelization_;
    if (circ.ops.size() % parallelization_)
      ++unit;

    // Vector to store parallel thread output data
    std::vector<ExperimentResult> par_results(parallelization_);

#pragma omp parallel for if (parallelization_ > 1) num_threads(parallelization_)
    for (int_t i = 0; i < parallelization_; i++) {
      int_t start = unit * i;
      int_t end = std::min(start + unit, (int_t) circ.ops.size());
      optimize_circuit(circ, noise, opset, start, end, par_results[i]);
    }

    reduce_results(circ, result, par_results);
  }
}

void CircuitOptimization::reduce_results(Circuit& circ,
                                         ExperimentResult &result,
                                         std::vector<ExperimentResult> &results) const {
  for (auto &res : results) {
    result.combine(std::move(res));
  }
}


//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
