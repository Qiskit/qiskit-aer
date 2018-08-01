/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    engine.hpp
 * @brief   Engine base class for qiskit-aer simulator
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_engines_finalstate_engine_hpp_
#define _aer_engines_finalstate_engine_hpp_



#include "engines/snapshot_engine.hpp"

namespace AER {
namespace Engines {


  template <class state_t>
  using State = Base::State<state_t>;

//============================================================================
// Final state engine base class for Qiskit-Aer
//============================================================================

// This engine returns the final state of the simulator at the end of a 
// single circuit shot. It is designed to be used for statevector simulators
// and Unitary simulators.
// This state also supports the snapshot operation.
template <class state_t>
class FinalStateEngine : public virtual SnapshotEngine<state_t> {

public:

  //----------------------------------------------------------------
  // Base class abstract method overrides
  //----------------------------------------------------------------
  
  // Empty engine of final state data
  inline virtual void clear() override;

  // Serialize final state data to JSON
  virtual json_t json() const override;

  // Move snapshot data from another SnapshotEngine to this one
  void combine(FinalStateEngine<state_t> &eng);

  // Load Engine config settings
  // Config settings for this engine are of the form:
  // { "finalstate_label": "finalstate"};
  // with default values "finalstate",
  virtual void load_config(const json_t &config) override;

  // Unused
  virtual void compute_result(State<state_t> *state) override;

  //----------------------------------------------------------------
  // Base class overrides
  //----------------------------------------------------------------

  // Override execute to only perform a single shot regardless of input
  // shot number
  virtual void execute(State<state_t> *state,
                       const Circuit &circ,
                       uint_t shots) override;

protected:
  std::string output_label_ = "final_state";
  std::vector<state_t> final_states_; // final state after each shot
};

//============================================================================
// Implementations
//============================================================================

template <class state_t>
void FinalStateEngine<state_t>::execute(State<state_t> *state,
                                        const Circuit &circ,
                                        uint_t shots) {
  shots = 1;
  SnapshotEngine<state_t>::execute(state, circ, 1); // Parent class execute
}


template <class state_t>
void FinalStateEngine<state_t>::compute_result(State<state_t> *state) {
  // Move final state data rather than copy
  // Note that no more operations can be applied to state after this move
  // unless state is re-initialized first!
  final_states_.emplace_back(std::move(state->data()));
}


template <class state_t>
void FinalStateEngine<state_t>::load_config(const json_t &js) {
  JSON::get_value(output_label_, "finalstate_label", js);
  SnapshotEngine<state_t>::load_config(js); // load parent class config
}


template <class state_t>
void FinalStateEngine<state_t>::combine(FinalStateEngine<state_t> &eng) {
    std::move(eng.final_states_.begin(),
              eng.final_states_.end(),
              std::back_inserter(final_states_));
    SnapshotEngine<state_t>::combine(eng); // parent class combine
  };


template <class state_t>
void FinalStateEngine<state_t>::clear() {
  final_states_.clear();
  SnapshotEngine<state_t>::clear(); // parent class clear
}


template <class state_t>
json_t FinalStateEngine<state_t>::json() const {
  json_t tmp = SnapshotEngine<state_t>::json(); // parent class JSON
  try {
    if (final_states_.empty() == false)
      tmp[output_label_] = final_states_[0];
  } catch (std::exception &e) {
      // Leave message in output that type conversion failed
    tmp["status"] = "Error: Failed to convert state type to JSON";
  }
  return tmp;
}

//------------------------------------------------------------------------------
} // end namespace Engines
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
