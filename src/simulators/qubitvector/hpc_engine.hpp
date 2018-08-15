/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    hpc_engine.hpp
 * @brief   QubitVector Simulator State
 * @authors Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _hpc_engine_hpp_
#define _hpc_engine_hpp_

#include <algorithm>
#include <array>
#include <complex>
#include <unordered_map>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "base/engine.hpp"
#include "base/state.hpp"

#include "optimization.hpp"

namespace AER {
namespace QubitVector {

/*******************************************************************************
 *
 * Engine Class
 *
 ******************************************************************************/
template <class state_t>
class Engine : public Base::Engine<state_t> {
public:
  Engine();
  ~Engine();

  virtual void execute(State<state_t> *state,
                       const Circuit &circ,
                       uint_t shots);

protected:
  std::vector<Optimization*> optimizations;

};

template <class state_t>
Engine<state_t>::Engine() {
  optimizations = { new Fusion(3) };

}

template <class state_t>
Engine<state_t>::~Engine() {

  for (Optimization* opt : optimizations)
    free(opt);

  optimizations.clear();
}

template<class state_t>
void Engine<state_t>::execute(State<state_t> *state, const Circuit &circ, uint_t shots) {

  std::vector<Circuit> current;
  current.push_back(circ);

//  for (Circuit &c : current) {
//    std::cout << "original: " << std::endl;
//    for (Op& op: c.ops)
//      std::cout << "   " << op.name << op.qubits << std::endl;
//  }

  std::vector <Circuit> optimized;

  while (true) {
    bool applied = false;
    for (Optimization* opt : optimizations) {
      if (opt->optimize(current, optimized)) {
        applied = true;
        break;
      }
    }
    if (!applied)
      break;

    current = optimized;
    optimized.clear();
  }

  for (Circuit &c : current) {

    // Check for sampling measurement optimization
    if (c.measure_sampling_flag) {
      Base::Engine<state_t>::execute_with_sampling(state, c, shots);
    } else {
      // execute without sampling
      while (shots-- > 0) {
        Base::Engine<state_t>::initialize(state, c);
        for (const auto &op : c.ops) {
          Base::Engine<state_t>::apply_op(state, op);
        }
        Base::Engine<state_t>::update_counts();
      }
    }
  }
}

}
}
#endif
