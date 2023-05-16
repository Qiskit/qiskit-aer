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

#include <algorithm>
#include <vector>

#include "framework/config.hpp"
#include "framework/opset.hpp"
#include "noise/noise_model.hpp"

namespace AER {
namespace Transpile {

using op_t = Operations::Op;
using optype_t = Operations::OpType;
using oplist_t = std::vector<op_t>;
using opset_t = Operations::OpSet;
using reg_t = std::vector<uint_t>;

class CircuitOptimization {
public:
  CircuitOptimization() = default;
  virtual ~CircuitOptimization() = default;

  virtual void optimize_circuit(Circuit &circ, Noise::NoiseModel &noise,
                                const Operations::OpSet &opset,
                                ExperimentResult &result) const = 0;

  virtual void set_config(const Config &config){};

protected:
};

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
