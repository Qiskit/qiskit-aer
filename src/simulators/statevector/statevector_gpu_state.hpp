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

#ifndef _statevector_gpu_state_hpp
#define _statevector_gpu_state_hpp

#include "statevector_state.hpp"
#include "qubitvector_thrust.hpp"


namespace AER {
namespace StatevectorThrust {

//=========================================================================
// QubitVectorThrust State subclass
//=========================================================================

template <class statevec_t = QV::QubitVectorThrust<double>>
class State : public Statevector::State<statevec_t> {
public:

  State() = default;
  virtual ~State() = default;
  using BaseState = Statevector::State<statevec_t>;

  //-----------------------------------------------------------------------
  // Statevector::State class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  #ifdef AER_THRUST_CUDA
  virtual std::string name() const override {return "statevector_gpu";}
  #else
  virtual std::string name() const override {return "statevector_fake_gpu";}
  #endif
};


//-------------------------------------------------------------------------
} // end namespace StatevectorThrust
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
