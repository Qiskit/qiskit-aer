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

#include <algorithm>

#include "base/engine.hpp"

namespace AER {
namespace Engines {


  template <class state_t>
  using State = Base::State<state_t>;

  template <class state_t>
  using BaseEngine = Base::Engine<state_t>;

//============================================================================
// Engine base class for Qiskit-Aer
//============================================================================

template <class state_t>
class FinalStateEngine : public virtual BaseEngine<state_t> {

public:


  //----------------------------------------------------------------
  // Base class overrides
  //----------------------------------------------------------------

  virtual void execute(State<state_t> *state,
                       const Circuit &circ,
                       uint_t shots) override;
  
  // Copy the final state data of the State class.
  virtual void compute_result(State<state_t> *state) override;

  // Empty engine of stored data
  inline virtual void clear() override {final_states_.clear();};

  // Serialize engine data to JSON
  virtual json_t json() const override;

  // Load Engine config settings
  // Config settings for this engine are of the form:
  // { "label": str, "single_shot": true};
  // with default values "final_state", false
  virtual void load_config(const json_t &config) override;

  // Combine engines for accumulating data
  // After combining argument engine should no longer be used
  void combine(FinalStateEngine<state_t> &eng);

protected:
  bool single_shot_ = true;
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
  // Check for single shot option
  uint_t shot_override = (single_shot_) ? 1 : shots;
  BaseEngine<state_t>::execute(state, circ, shot_override);
}


template <class state_t>
void FinalStateEngine<state_t>::compute_result(State<state_t> *state) {
  // Move final state data rather than copy
  // Note that no more operations can be applied to state after this move
  // unless state is re-initialized first!
  if (single_shot_) {
    // delete any current data so it will contain only one shot
    final_states_.clear();
  }
  final_states_.emplace_back(std::move(state->data()));
}


template <class state_t>
void FinalStateEngine<state_t>::load_config(const json_t &js) {
  JSON::get_value(single_shot_, "single_shot", js);
  JSON::get_value(output_label_, "label", js);
}


template <class state_t>
void FinalStateEngine<state_t>::combine(FinalStateEngine<state_t> &eng) {
    std::move(eng.final_states_.begin(),
              eng.final_states_.end(),
              std::back_inserter(final_states_));
  };


template <class state_t>
json_t FinalStateEngine<state_t>::json() const {
  json_t tmp;
  try {
    if (single_shot_ && final_states_.empty() == false)
      tmp[output_label_] = final_states_[0];
    else
      tmp[output_label_] = final_states_;
  } catch (std::exception &e) {
      // Leave message in output that type conversion failed
    tmp["status"] = "Error: Failed to convert state type to JSON";
  }
  return tmp;
}

//------------------------------------------------------------------------------
} // end namespace Engines
} // end namespace AER
//------------------------------------------------------------------------------
#endif
