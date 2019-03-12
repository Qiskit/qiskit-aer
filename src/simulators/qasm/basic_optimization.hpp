/**
 * Copyright 2019, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_basic_opt_hpp_
#define _aer_basic_opt_hpp_

#include "framework/circuitopt.hpp"

namespace AER {

class ReduceNop : public CircuitOptimization {
public:
  void optimize_circuit(Circuit& circ) const;
};

void ReduceNop::optimize_circuit(Circuit& circ) const {

  std::vector<Operations::Op>::iterator it = circ.ops.begin();
  while (it != circ.ops.end()) {
    if (it->type == Operations::OpType::barrier)
      it = circ.ops.erase(it);
    else
      ++it;
  }
}

class Debug : public CircuitOptimization {
public:
  void optimize_circuit(Circuit& circ) const;
};

void Debug::optimize_circuit(Circuit& circ) const {

  std::vector<Operations::Op>::iterator it = circ.ops.begin();
  while (it != circ.ops.end()) {
    std::cerr << it->name << std::endl;
    ++it;
  }
}


//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------


#endif
