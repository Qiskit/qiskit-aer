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

#include <algorithm> // used for std::move
#include "base/engine.hpp"


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
  inline virtual void clear() override {snapshots_.clear();};

  // Serialize Snapshot data to JSON
  virtual json_t json() const override;

  // Move snapshot data from another SnapshotEngine to this one
  void combine(SnapshotEngine<state_t> &eng);

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
                        const Op &op) override;
  
  // Add snapshot op as valid circuit op
  virtual std::set<std::string>
  validate_circuit(State *state, const Circuit &circ) override;

protected:
   bool show_snapshots_ = true;
   std::string snapshot_label_ = "default";
   std::map<std::string, std::vector<state_t>> snapshots_; 
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
SnapshotEngine<state_t>::validate_circuit(State *state,
                                          const Circuit &circ) {
  auto allowed_ops = state->allowed_ops();
  allowed_ops.insert("snapshot");
  return circ.invalid_ops(allowed_ops);
};


template <class state_t>
void SnapshotEngine<state_t>::apply_op(State *state, const Op &op) {
  if (op.name == "snapshot") { 
      // copy state data at snapshot point
      snapshots_[op.params_s[0]].push_back(state->data());
  } else {
    Base::Engine<state_t>::apply_op(state, op);  // Apply operation as usual
  }
}


template <class state_t>
void SnapshotEngine<state_t>::combine(SnapshotEngine<state_t> &eng) {
  for (auto &s: eng.snapshots_) {
    std::move(s.second.begin(), s.second.end(), back_inserter(snapshots_[s.first]));
  }
  eng.snapshots_.clear();
}


template <class state_t>
json_t SnapshotEngine<state_t>::json() const {
  // add snapshots
  json_t tmp;
  if (snapshots_.empty() == false) {
    try {
      // use try incase state class doesn't have json conversion method
      for (const auto& pair: snapshots_) {
        if (!pair.second.empty()) {
          tmp["snapshots"][pair.first][snapshot_label_] = pair.second;
        }
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
