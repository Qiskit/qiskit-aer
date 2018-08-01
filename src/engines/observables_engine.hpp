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

#include "engines/qasm_engine.hpp"


// TODO: Matrix observables

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
  using set_t = std::set<uint_t>;
  using key_t = std::pair<set_t, std::string>;
  using State = Base::State<state_t>;

  //----------------------------------------------------------------
  // Base class abstract method overrides
  //----------------------------------------------------------------
  
  // Erase output data from engine
  virtual void clear() override;

  // Serialize engine data to JSON
  virtual json_t json() const override;

  // Move snapshot data from another QasmEngine to this one
  // TODO: shots for normalization
  void combine(ObservablesEngine<state_t> &eng);

  // Load Engine config settings
  // Config settings for this engine are of the form:
  // {"probabilities": [probs ops],
  //  "observables": [obs ops],
  //  "observables_chop_threshold": 1e-16}
  // where obs ops can be either "ops_pauli", "ops_mat", "ops_dmat", "ops_vec"
  virtual void load_config(const json_t &config) override;

  // Compute observables conditioned on memory bit values
  // Note that we add the probabilities (or complex amplitudes) for each shot
  // conditioned on the QasmEngine memory bit value. These quantities must be
  // renormalized at the end based on the number of shots
  virtual void compute_result(State *state) override;

  // Initialize Pauli cache
  virtual void initialize(State *state, const Circuit &circ) override;

  //----------------------------------------------------------------
  // Base class additional overrides
  //----------------------------------------------------------------

  // Add snapshot op as valid circuit op
  virtual std::set<std::string>
  validate_circuit(State *state, const Circuit &circ) override;

protected:

  void compute_observables_probs(State *state);
  void compute_observables_ops(State *state);

  complex_t pauli_expval(State *state, const key_t &key, const Op &op);

  std::map<std::string, double>
  probs_ket(const std::vector<double> &vec, double scale, double epsilon) const;

  // Measure observables
  std::set<set_t> obs_meas_;

  // Observables to measure indexed by qubit
  // TODO: matrix observables
  std::map<set_t, std::vector<Op>> obs_ops_;
  
  // Measurement observables data
  // indexed by qubits and memory string (if any)
  std::map<key_t, rvector_t> data_meas_;
  std::map<key_t, uint_t> data_meas_counts_;

  // Operator observables data
  // indexed by qubits and memory string (if any)
  std::map<key_t, cvector_t> data_ops_;
  std::map<key_t, uint_t> data_ops_counts_;

  // Cache of saved pauli values for given qubits and memory bit values
  // to prevent recomputing for re-used Pauli components in different operators
  // Note: there must be a better way to do this cacheing than map of map of map...
  // But to do so would require sorting the vector of qubits in the obs_pauli ops
  // and also sorting the Pauli string labels so that they match the sorted qubit positions.
  std::map<key_t, std::map<std::string, double>> pauli_cache_;

  // Chop value for small observables
  double chop_epsilon_ = 1e-16;
};

//============================================================================
// Implementation: Base engine overrides
//============================================================================


template <class state_t>
void ObservablesEngine<state_t>::initialize(State *state, const Circuit &circ) {
  QasmEngine<state_t>::initialize(state, circ);
  pauli_cache_.clear(); // clear pauli cache at start of each shot
}


template <class state_t>
void ObservablesEngine<state_t>::compute_result(State *state) {
  // Compute qasm engine results first since we need to make observables
  // conditional on the value of the memory register (if present)
  QasmEngine<state_t>::compute_result(state); // parent class
  compute_observables_probs(state);
  compute_observables_ops(state);
}


template <class state_t>
std::set<std::string>
ObservablesEngine<state_t>::validate_circuit(State *state,
                                          const Circuit &circ) {
  auto allowed_ops = state->allowed_ops();
  allowed_ops.insert({"snapshot", "measure"}); // from parents
  return circ.invalid_ops(allowed_ops);
};


template <class state_t>
void ObservablesEngine<state_t>::clear() {
  QasmEngine<state_t>::clear(); // parent class
  data_ops_.clear();
  data_ops_counts_.clear();
  data_meas_.clear();
  data_meas_counts_.clear();
  pauli_cache_.clear();
}


template <class state_t>
void ObservablesEngine<state_t>::combine(ObservablesEngine<state_t> &eng) {
  QasmEngine<state_t>::combine(eng); // parent class
  // combine meas data
  for (auto &pair : eng.data_meas_) {
    Utils::combine(data_meas_[pair.first], pair.second);
  }
  for (auto &pair : eng.data_meas_counts_) {
    data_meas_counts_[pair.first] += pair.second;
  }
  for (auto &pair : eng.data_ops_) {
    Utils::combine(data_ops_[pair.first], pair.second);
  }
  for (auto &pair : eng.data_ops_counts_) {
    data_ops_counts_[pair.first] += pair.second;
  }
  eng.clear(); // clear added engine
}


template <class state_t>
json_t ObservablesEngine<state_t>::json() const {
  json_t tmp = QasmEngine<state_t>::json();

  // Convert measurement probabilities
  for (const auto &pair : data_meas_) {
    double counts = data_meas_counts_.find(pair.first) -> second;
    auto &memory = pair.first.second;
    json_t datum;
    datum["qubits"] = pair.first.first;
    datum["value"] = probs_ket(pair.second, 1. / counts, chop_epsilon_);
    if (!memory.empty())
      datum["memory"] = memory; 
    tmp["probabilities"].push_back(datum);
  }

  // Convert operator observables
  for (const auto &pair : data_ops_) {
    double counts = data_ops_counts_.find(pair.first) -> second;
    auto &memory = pair.first.second;
    auto val = Utils::multiply<complex_t>(pair.second, 1. / counts);
    Utils::chop(val, chop_epsilon_);
    json_t datum;
    datum["qubits"] = pair.first.first;  // key_t.first
    datum["value"] = val;
    if (!memory.empty())
      datum["memory"] = memory; 
    tmp["observables"].push_back(datum);
  }

  return tmp;
}


template <class state_t>
void ObservablesEngine<state_t>::load_config(const json_t &js) {
  QasmEngine<state_t>::load_config(js); // parent class
  // Load chop value
  JSON::get_value(chop_epsilon_, "observables_chop_threshold", js);
  // Load measurement observables
  if (JSON::check_key("probabilities", js)) {
    if (js["probabilities"].is_array()) {
      for (auto& elt : js["probabilities"]) {
        auto tmp = json_to_op_probs(elt).qubits; // qubit vector
        set_t qubits(tmp.begin(), tmp.end()); // convert to set
        obs_meas_.insert(qubits);
      }
    } else {
      throw std::invalid_argument("ObservablesEngine::load_config (config \"probabilities\" field is not an array");
    }
  }

  // Load operator observables
  if (JSON::check_key("observables", js)) {
    if (js["observables"].is_array()) {
      for (auto &elt : js["observables"]) {
        const Op op = json_to_op_obs(elt);
        set_t qubits(op.qubits.begin(), op.qubits.end());
        obs_ops_[qubits].push_back(op);
      }
    } else {
      throw std::invalid_argument("ObservablesEngine::load_config (config \"observables\" field is not an array");
    }
  }
}


//============================================================================
// Implementation: Helper functions
//============================================================================

template <class state_t>
void ObservablesEngine<state_t>::compute_observables_probs(State *state) {
  for (const auto &qubits : obs_meas_) {
    key_t key({qubits, QasmEngine<state_t>::creg_memory_});
    rvector_t probs = state->measure_probs(qubits);
    Utils::combine(data_meas_[key], probs); // accumulate with other probs
    data_meas_counts_[key] += 1; // increment counts
  }
}


// TODO: add matrix observable operators
template <class state_t>
void ObservablesEngine<state_t>::compute_observables_ops(State *state) {
  // Compute operator observables
  for (const auto &pair : obs_ops_) { // pair is (qubits, vector<Op>)
    key_t key({pair.first, QasmEngine<state_t>::creg_memory_});
    cvector_t expvals;
    for (const auto &op : pair.second) {
      if (op.name == "obs_pauli")
        expvals.push_back(pauli_expval(state, key, op));
      else
        std::invalid_argument("TODO: only pauli ops currently implemented!");
    }
    Utils::combine(data_ops_[key], expvals); // accumulate with other expvals
    data_ops_counts_[key] += 1; // increment counts
  }
}


template <class state_t>
complex_t ObservablesEngine<state_t>::pauli_expval(State *state, const key_t &key, const Op &op) {
  // Compute operator observables
  auto &cache = pauli_cache_[key];
  complex_t expval(0., 0.);
  for (size_t j=0; j < op.params_s.size(); j++) {
    std::string pauli = op.params_s[j];
    auto it = cache.find(pauli); // Check cache for value
    if (it != cache.end()) {
      // found cached result
      expval += op.params_z[j] * (it->second);
    } else {
      // compute result and add to cache
      double tmp = state->pauli_observable_value(op.qubits, pauli);
      cache[pauli] = tmp;
      expval += op.params_z[j] * tmp;
    }
  }
  return expval;
}


template <class state_t>
std::map<std::string, double>
ObservablesEngine<state_t>::probs_ket(const rvector_t &vec, double scale, double epsilon) const {
  // check vector length
  double n = std::log2(vec.size());
  if (std::abs(trunc(n) - n) > 1e-5) {
    throw std::invalid_argument("ObservablesEngine::probs_ket (probability vector is not of size 2^N)");
  }
  std::map<std::string, double> ketmap;
  for (size_t k = 0; k < vec.size(); ++k) {
    double tmp = vec[k] * scale;
    if (std::abs(tmp) > epsilon) { 
      ketmap.insert({Utils::int2bin(k, trunc(n)), tmp});
    }
  }
  return ketmap;
}


//------------------------------------------------------------------------------
} // end namespace Engines
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
