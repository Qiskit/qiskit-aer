/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    observables_engine.hpp
 * @brief   Observables expectation value engine class for qiskit-aer simulator
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_engines_observables_engine_hpp_
#define _aer_engines_observables_engine_hpp_

#include <tuple>
#include "engines/qasm_engine.hpp"


namespace AER {
namespace Engines {



//============================================================================
// QASM Engine class for Qiskit-Aer
//============================================================================

// This engine returns counts for state classes that support measurement
template <class state_t>
class ObservablesEngine : public QasmEngine<state_t> {

public:

  // Internal type aliasing
  using State = Base::State<state_t>;

  //----------------------------------------------------------------
  // Base class abstract method overrides
  //----------------------------------------------------------------
  
  // Apply a sequence of operations to the state
  virtual void apply_op(State *state, const Operations::Op &op) override;

  // Erase output data from engine
  virtual void clear() override;

  // Serialize engine data to JSON
  virtual json_t json() const override;

  // Combine results from two observables engines
  void combine(ObservablesEngine<state_t> &eng);

  // Initialize Pauli cache
  virtual void initialize(State *state, const Circuit &circ) override;

  //----------------------------------------------------------------
  // Base class additional overrides
  //----------------------------------------------------------------

  // Add snapshot op as valid circuit op
  virtual std::set<std::string>
  validate_circuit(State *state, const Circuit &circ) override;

protected:

  // Compute and store snapshot of the Pauli observable op
  void snapshot_observables_pauli(State *state, const Operations::Op &op);

  // Compute and store snapshot of the matrix observable op
  void snapshot_observables_matrix(State *state, const Operations::Op &op);

  // Observables Snapshots
  using SnapshotQubits = std::set<uint_t, std::greater<uint_t>>;
  using SnapshotKey = std::pair<SnapshotQubits, std::string>; // qubits and memory value pair
  using SnapshotObs = Snapshots::Snapshot<SnapshotKey, complex_t, Snapshots::AverageData>;
  SnapshotObs snapshot_obs_;
  
  // Cache of saved pauli values for given qubits and memory bit values
  // to prevent recomputing for re-used Pauli components in different operators
  // Note: there must be a better way to do this cacheing than map of map of map...
  // But to do so would require sorting the vector of qubits in the obs_pauli ops
  // and also sorting the Pauli string labels so that they match the sorted qubit positions.
  std::map<SnapshotKey, std::map<std::string, double>> pauli_cache_;

};

//============================================================================
// Implementation: Base engine overrides
//============================================================================

template <class state_t>
void ObservablesEngine<state_t>::apply_op(State *state, const Operations::Op &op) {
  if (op.name == "snapshot_pauli") {
    snapshot_observables_pauli(state, op);
  } else if (op.name == "snapshot_matrix") {
    snapshot_observables_matrix(state, op);
  } 
  else {
    // Use parent engine apply op
    QasmEngine<state_t>::apply_op(state, op);
  }
}


template <class state_t>
void ObservablesEngine<state_t>::initialize(State *state, const Circuit &circ) {
  QasmEngine<state_t>::initialize(state, circ);
  pauli_cache_.clear(); // clear pauli cache at start of each shot
}



template <class state_t>
std::set<std::string>
ObservablesEngine<state_t>::validate_circuit(State *state,
                                          const Circuit &circ) {
  auto allowed_ops = state->allowed_ops();
  allowed_ops.insert({"measure", "snapshot_state", "snapshot_probs", // from parents
                      "snapshot_pauli", "snapshot_matrix"});
  return circ.invalid_ops(allowed_ops);
};


template <class state_t>
void ObservablesEngine<state_t>::clear() {
  QasmEngine<state_t>::clear(); // parent class
  snapshot_obs_.clear();
  pauli_cache_.clear();
}


template <class state_t>
void ObservablesEngine<state_t>::combine(ObservablesEngine<state_t> &eng) {
  QasmEngine<state_t>::combine(eng); // parent class
  snapshot_obs_.combine(eng.snapshot_obs_); // combine snapshots
}


template <class state_t>
json_t ObservablesEngine<state_t>::json() const {
  json_t tmp = QasmEngine<state_t>::json();

  // Add snapshot data 
  auto slots = snapshot_obs_.slots();
  for (const auto &slot : slots) {
    json_t probs_js;
    auto keys = snapshot_obs_.slot_keys(slot);
    for (const auto &key : keys) {
      json_t datum;
      datum["qubits"] = key.first;
      datum["memory"] = key.second;
      datum["value"] = snapshot_obs_.get_data(slot, key).data();
      probs_js.push_back(datum);
    }
    tmp["snapshots"][slot]["observables"] = probs_js;
  }
  return tmp;
}


//============================================================================
// Implementation: Snapshot functions
//============================================================================

template <class state_t>
void ObservablesEngine<state_t>::snapshot_observables_pauli(State *state, const Operations::Op &op) {
  auto key = QasmEngine<state_t>::make_snapshot_key(op);
  auto &cache = pauli_cache_[key]; // pauli cache for current key
  complex_t expval(0., 0.); // complex value
  for (const auto &param : op.params_pauli_obs) {
    auto it = cache.find(param.second); // Check cache for value
    if (it != cache.end()) {
      expval += param.first * (it->second); // found cached result
    } else {
      // compute result and add to cache
      double tmp = state->pauli_observable_value(op.qubits, param.second);
      cache[param.second] = tmp;
      expval += param.first * tmp;
    }
  }
  // add to snapshot
  snapshot_obs_.add_data(op.slot, key, expval);
}


template <class state_t>
void ObservablesEngine<state_t>::snapshot_observables_matrix(State *state, const Operations::Op &op) {
  auto key = QasmEngine<state_t>::make_snapshot_key(op);
  snapshot_obs_.add_data(op.slot, key, state->matrix_observable_value(op));
}


//------------------------------------------------------------------------------
} // end namespace Engines
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
