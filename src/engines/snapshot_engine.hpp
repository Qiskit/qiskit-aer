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

#ifndef _aer_engines_snapshot_engine_hpp_
#define _aer_engines_snapshot_engine_hpp_

#include "base/engine.hpp"
#include "framework/snapshot.hpp"


namespace AER {
namespace Engines {


//============================================================================
// Engine base class for Qiskit-Aer
//============================================================================

template <class state_t>
class SnapshotEngine : public virtual Base::Engine<state_t> {

public:
  using State = Base::State<state_t>;

  //----------------------------------------------------------------
  // Base class abstract method overrides
  //----------------------------------------------------------------
  
  // Empty engine of Snapshot data
  inline virtual void clear() override {snapshot_states_.clear();};

  // Serialize Snapshot data to JSON
  virtual json_t json() const override;

  // Move snapshot data from another SnapshotEngine to this one
  inline void combine(SnapshotEngine<state_t> &eng) {
    snapshot_states_.combine(eng.snapshot_states_);
  };

  // Load Engine config settings
  // Config settings for this engine are of the form:
  // { "snapshot_label": "default", "hide_snapshots": false};
  // with default values "default", false
  virtual void load_config(const json_t &config) override;

  // Unused
  inline virtual void
  compute_result(State *state) override {(void)state;};

  //----------------------------------------------------------------
  // Base class additional overrides
  //----------------------------------------------------------------

  // Implement snapshot op on engine, and pass remaining ops to State
  virtual void apply_op(State *state,
                        const Operations::Op &op) override;
  
  // Add snapshot op as valid circuit op
  virtual std::set<std::string>
  validate_circuit(State *state, const Circuit &circ) override;

protected:
  using SnapshotStates = Snapshots::Snapshot<std::string, state_t, Snapshots::ShotData>;
  SnapshotStates snapshot_states_;
  bool show_snapshots_ = true;
  std::string snapshot_label_ = "default"; // label and key for snapshots
  //std::map<std::string, std::vector<state_t>> snapshots_; 
};

//============================================================================
// Implementations
//============================================================================

template <class state_t>
void SnapshotEngine<state_t>::load_config(const json_t &js) {
  JSON::get_value(snapshot_label_, "snapshot_label", js);
  JSON::get_value(show_snapshots_, "show_snapshots", js);
}


template <class state_t>
std::set<std::string>
SnapshotEngine<state_t>::validate_circuit(State *state, const Circuit &circ) {
  auto allowed_ops = state->allowed_ops();
  allowed_ops.insert("snapshot_state");
  return circ.invalid_ops(allowed_ops);
};


template <class state_t>
void SnapshotEngine<state_t>::apply_op(State *state, const Operations::Op &op) {
  if (op.name == "snapshot_state") { 
      // copy state data at snapshot point
      snapshot_states_.add_data(op.slot, snapshot_label_, state->data());
  } else {
    Base::Engine<state_t>::apply_op(state, op);  // Apply operation as usual
  }
}


template <class state_t>
json_t SnapshotEngine<state_t>::json() const {
  // add snapshots
  json_t tmp;
  auto slots = snapshot_states_.slots();
  if (slots.empty() == false) {
    try {
      for (const auto &slot : slots)
        for (const auto &key : snapshot_states_.slot_keys(slot)) {
          tmp["snapshots"][slot][key] = snapshot_states_.get_data(slot, key).data();
        }
    } catch (std::exception &e) {
      // Leave message in output that type conversion failed
      tmp["snapshots"] = "Error: Failed to convert state type to JSON";
    }
  }
  return tmp;
}


//------------------------------------------------------------------------------
} // end namespace Engines
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
