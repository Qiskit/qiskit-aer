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

#include "engines/finalstate_engine.hpp"

namespace AER {
namespace Engines {


  template <class state_t>
  using State = Base::State<state_t>;

//============================================================================
// Engine base class for Qiskit-Aer
//============================================================================

template <class state_t>
class SnapshotEngine : public FinalStateEngine<state_t> {

public:

  //----------------------------------------------------------------
  // Base class overrides
  //----------------------------------------------------------------

  virtual void apply_op(State<state_t> *state,
                        const Op &op) override;
  
  // Copy the final state data of the State class.
  virtual void compute_result(State<state_t> *state) override;

  // Empty engine of stored data
  inline virtual void clear() override;

  // Serialize engine data to JSON
  virtual json_t json() const override; // TODO

  // Add #snapshot to valid circuit operations
  virtual bool validate_circuit(State<state_t> *state,
                                const Circuit &circ) override;

  // Combine engines for accumulating data
  // After combining argument engine should no longer be used
  void combine(SnapshotEngine<state_t> &eng);

protected:
   std::map<std::string, std::vector<state_t>> snapshots_; 
};

//============================================================================
// Implementations
//============================================================================

template <class state_t>
bool SnapshotEngine<state_t>::validate_circuit(State<state_t> *state,
                                               const Circuit &circ) {
  auto allowed_ops = state->allowed_ops();
  allowed_ops.insert("#snapshot");
  return circ.check_ops(allowed_ops);
};


template <class state_t>
void SnapshotEngine<state_t>::apply_op(State<state_t> *state, const Op &op) {
  if (op.name == "#snapshot") { 
      // copy state data at snapshot point
      snapshots_[op.params_s[0]].push_back(state->data());
  } else {
    FinalStateEngine<state_t>::apply_op(state, op);  // Apply operation as usual
  }
}


template <class state_t>
void SnapshotEngine<state_t>::compute_result(State<state_t> *state) {
  FinalStateEngine<state_t>::compute_result(state);
}


template <class state_t>
void SnapshotEngine<state_t>::combine(SnapshotEngine<state_t> &eng) {
  FinalStateEngine<state_t>::combine(eng);
  for (auto &s: eng.snapshots_) {
    std::move(s.second.begin(), s.second.end(),
              back_inserter(snapshots_[s.first]));
  }
  eng.snapshots_.clear();
}


template <class state_t>
void SnapshotEngine<state_t>::clear() {
  FinalStateEngine<state_t>::clear();
  snapshots_.clear();
}


template <class state_t>
json_t SnapshotEngine<state_t>::json() const {
  json_t tmp = FinalStateEngine<state_t>::json();
  // add snapshots
  if (snapshots_.empty() == false) {
    try {
      std::string output_label = FinalStateEngine<state_t>::output_label_;
      bool single_shot = FinalStateEngine<state_t>::single_shot_;
      // use try incase state class doesn't have json conversion method
      for (const auto& pair: snapshots_) {
        if (!pair.second.empty()) {
          if (single_shot)
            tmp["snapshots"][pair.first][output_label] = pair.second[0];
          else
             tmp["snapshots"][pair.first][output_label] = pair.second;
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
} // end namespace AER
//------------------------------------------------------------------------------
#endif
