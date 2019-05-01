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

#ifndef _aer_transpile_basic_opt_hpp_
#define _aer_transpile_basic_opt_hpp_

#include "transpile/circuitopt.hpp"

namespace AER {
namespace Transpile {

using uint_t = uint_t;
using op_t = Operations::Op;
using optype_t = Operations::OpType;
using oplist_t = std::vector<op_t>;
using opset_t = Operations::OpSet;
using reg_t = std::vector<uint_t>;

class ReduceNop : public CircuitOptimization {
public:
  void optimize_circuit(Circuit& circ,
                        const opset_t &opset,
                        OutputData &data) const override;
};

void ReduceNop::optimize_circuit(Circuit& circ,
                                 const opset_t &allowed_opset,
                                 OutputData &data) const {

  oplist_t::iterator it = circ.ops.begin();
  while (it != circ.ops.end()) {
    if (it->type == optype_t::barrier)
      it = circ.ops.erase(it);
    else
      ++it;
  }
}

class Debug : public CircuitOptimization {
public:
  void optimize_circuit(Circuit& circ,
                        const opset_t &opset,
                        OutputData &data) const override;
};

void Debug::optimize_circuit(Circuit& circ,
                             const opset_t &allowed_opset,
                             OutputData &data) const {

  oplist_t::iterator it = circ.ops.begin();
  while (it != circ.ops.end()) {
    std::clog << it->name << std::endl;
    ++it;
  }
}


//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
