/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    reset_error.hpp
 * @brief   Reset Error class for Qiskit-Aer simulator
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_noise_reset_error_hpp_
#define _aer_noise_reset_error_hpp_

#include "noise/unitary_error.hpp"


namespace AER {
namespace Noise {

//=========================================================================
// Reset Error class
//=========================================================================

// Single qubit reset error:
// E(rho) = (1 - P) rho + \sum_j p_j |j><j|
// where: P = sum_j p_j
//        p_j are the probabilities of reset to each Z-basis state
// If this error is applied to multiple qubits (eg for a multi-qubit gate)
// The probabilities can either be independent of which qubit is being 
// reset, or be specified for each qubit being reset.

class ResetError : public AbstractError {
public:

  //-----------------------------------------------------------------------
  // Error base required methods
  //-----------------------------------------------------------------------

  // Sample a noisy implementation of op
  NoiseOps sample_noise(const reg_t &qubits,
                        RngEngine &rng) override;
  
  // TODO
  // Check that the reset error probabilities are valid
  std::pair<bool, std::string> validate() const override;

  // Load a ResetError object from a JSON Error object
  void load_from_json(const json_t &js) override;

  //-----------------------------------------------------------------------
  // Additional class methods
  //-----------------------------------------------------------------------

  // Set the default probabilities for reset error for a qubit.
  // The input vector should be length-2 with 
  // probs[0] being the probability of reseting to |0> state
  // probs[1] the probability of resetting to the |1> state
  // The probability of no-reset error is given by 1 - p[0] - p[1]
  inline void set_default_probabilities(const rvector_t &probs) {
    default_probabilities_ = format_probabilities(probs);
  }

  // Set the reset error probabilities for a specific qubit.
  // This will override the default probabilities for that qubit.
  inline void set_qubit_probabilities(uint_t qubit, const rvector_t &probs) {
    multi_probabilities_[qubit] = format_probabilities(probs);
  }

protected:
  using probs_t = std::discrete_distribution<uint_t>;
  
  // Reset error probabilities for each qubit
  // 0 -> no reset
  // 1 -> reset to |0>
  // 2 -> reset to |1>
  probs_t default_probabilities_; 

  // Map for getting reset parameters for each qubit
  std::unordered_map<uint_t, probs_t> multi_probabilities_; 

  // Helper function to convert input to internal probability format
  probs_t format_probabilities(const rvector_t &probs);
};

//-------------------------------------------------------------------------
// Implementation: Mixed unitary error subclass
//-------------------------------------------------------------------------

ResetError::NoiseOps ResetError::sample_noise(const reg_t &qubits,
                                              RngEngine &rng) {
  // Initialize return ops,
  NoiseOps ret;
  // Sample reset error for each qubit
  for (const auto &q: qubits) {
    auto it = multi_probabilities_.find(q);
    auto r = (it == multi_probabilities_.end())
      ? rng.rand_int(default_probabilities_)
      : rng.rand_int(it->second);
    if (r > 0)
      ret.push_back(Operations::make_reset(q, r));
  }
  return ret;
}


ResetError::probs_t ResetError::format_probabilities(const rvector_t &probs) {
  // error probability vector with [0] = p_identity
  rvector_t error_probs = {1.}; 
  for (const auto &p : probs) {
    if (p < 0 || p > 1) {
      throw std::invalid_argument("ResetError probability is not valid (p=" + std::to_string(p) +").");
    }
    error_probs[0] -= p;
    error_probs.push_back(p);
  }
  // Check the p_identity value is valid
  if (error_probs[0] < 0) {
    throw std::invalid_argument("ResetError probabilities > 1.");
  }
  return probs_t(error_probs.begin(), error_probs.end());
}


std::pair<bool, std::string> ResetError::validate() const {
  // TODO
  return std::make_pair(true, std::string());
}


void ResetError::load_from_json(const json_t &js) {
  rvector_t default_probs;
  JSON::get_value(default_probs, "default_probabilities", js);
  if (!default_probs.empty()) {
    set_default_probabilities(default_probs);
  }
  std::vector<std::pair<double, rvector_t>> qubit_probs;
  JSON::get_value(qubit_probs, "qubit_probabilities", js);
  if (!qubit_probs.empty()) {
    for (const auto pair : qubit_probs)
      set_qubit_probabilities(pair.first, pair.second);
  }
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif