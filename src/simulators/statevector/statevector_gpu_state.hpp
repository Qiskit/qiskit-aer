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

  

  // Return the set of qobj instruction types supported by the State
  // This state doesn't support the multiplexer operation
  virtual Operations::OpSet::optypeset_t allowed_ops() const override {
    return Operations::OpSet::optypeset_t({
      Operations::OpType::gate,
      Operations::OpType::measure,
      Operations::OpType::reset,
      Operations::OpType::initialize,
      Operations::OpType::snapshot,
      Operations::OpType::barrier,
      Operations::OpType::bfunc,
      Operations::OpType::roerror,
      Operations::OpType::matrix,
      Operations::OpType::kraus
    });
  }

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exeption will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentData &data,
                         RngEngine &rng) override;
};

//=========================================================================
// Implementation: apply operations
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_ops(const std::vector<Operations::Op> &ops,
                                 ExperimentData &data,
                                 RngEngine &rng) {

  // Simple loop over vector of input operations
  for (const auto & op: ops) {
    if(BaseState::creg_.check_conditional(op)) {
      switch (op.type) {
        case Operations::OpType::barrier:
          break;
        case Operations::OpType::reset:
          BaseState::apply_reset(op.qubits, rng);
          break;
        case Operations::OpType::initialize:
          BaseState::apply_initialize(op.qubits, op.params, rng);
          break;
        case Operations::OpType::measure:
          BaseState::apply_measure(op.qubits, op.memory, op.registers, rng);
          break;
        case Operations::OpType::bfunc:
          BaseState::creg_.apply_bfunc(op);
          break;
        case Operations::OpType::roerror:
          BaseState::creg_.apply_roerror(op, rng);
          break;
        case Operations::OpType::gate:
          BaseState::apply_gate(op);
          break;
        case Operations::OpType::snapshot:
          BaseState::apply_snapshot(op, data);
          break;
        case Operations::OpType::matrix:
          BaseState::apply_matrix(op);
          break;
        case Operations::OpType::kraus:
          BaseState::apply_kraus(op.qubits, op.mats, rng);
          break;
        default:
          throw std::invalid_argument("QubitVector::State::invalid instruction \'" +
                                      op.name + "\'.");
      }
    }
  }
}

//-------------------------------------------------------------------------
} // end namespace StatevectorThrust
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
