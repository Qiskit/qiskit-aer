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
#include "base/noise.hpp"

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
  virtual ~Engine();

  void execute(const Circuit &circ,
               uint_t shots,
               State<state_t> *state_ptr,
               Noise::Model *noise_ptr = nullptr);

protected:
  std::vector<std::shared_ptr<Optimization>> optimizations;

};

template <class state_t>
Engine<state_t>::Engine() {
  optimizations.push_back(std::shared_ptr<Optimization>(new Fusion(3)));
}

template <class state_t>
Engine<state_t>::~Engine() {

  optimizations.clear();
}

template<class state_t>
void Engine<state_t>::execute(const Circuit &circ,
             uint_t shots,
             State<state_t> *state_ptr,
             Noise::Model *noise_ptr)  {

  std::vector<Circuit> current;
  current.push_back(circ);

  std::vector <Circuit> optimized;

  while (true) {
    bool applied = false;
    for (std::shared_ptr<Optimization> opt : optimizations) {
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

//  for (Circuit &c : current) {
//    std::cout << "optimized: " << std::endl;
//    for (Op& op: c.ops) {
//      if (op.mats.empty())
//        std::cout << "   " << op.name << op.qubits << std::endl;
//      else
//        std::cout << "   " << op.name << op.qubits << ": " << op.mats[0].GetColumns() << "x" << op.mats[0].GetRows() << std::endl;
//    }
//  }

  for (Circuit &c : current) {
    // Check if ideal simulation check if sampling is possible
    if (noise_ptr == nullptr && c.measure_sampling_flag) {
      Base::Engine<state_t>::execute_with_measure_sampling(c, shots, state_ptr);
    } else {
      // Ideal execution without sampling
      while (shots-- > 0) {
        Base::Engine<state_t>::initialize(state_ptr, c);
        for (const auto &op: c.ops) {
          Base::Engine<state_t>::apply_op(op, state_ptr, noise_ptr);
        }
        Base::Engine<state_t>::update_counts();
      }
    }
  }
}

}
}
#endif
