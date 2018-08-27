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

  // Return a character labeling the error type
  inline char error_type() const override {return 'R';}

  // Sample a noisy implementation of op
  NoiseOps sample_noise(const reg_t &qubits,
                        RngEngine &rng) const override;
  
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
  void set_probabilities(const rvector_t &probs);

protected:

  // Reset error probabilities for n-qubit system
  // 0 -> no reset
  // i -> reset to state |i-1>
  // For single-qubit
  // 1, 2 -> reset to |0>, |1>
  // For two-qubit:
  // 1, 2, 3, 4 -> reset to |00>, |01>, |10>, |11>
  rvector_t probabilities_; 
};

//-------------------------------------------------------------------------
// Implementation: Mixed unitary error subclass
//-------------------------------------------------------------------------

ResetError::NoiseOps ResetError::sample_noise(const reg_t &qubits,
                                              RngEngine &rng) const {
  // Initialize return ops,
  NoiseOps ret;
  // Sample reset error for each qubit
  auto r = rng.rand_int(probabilities_);
  if (r > 0)
    ret.push_back(Operations::make_reset(qubits, r - 1));
  return ret;
}


void ResetError::set_probabilities(const rvector_t &probs) {
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
  probabilities_ = std::move(error_probs);
}


std::pair<bool, std::string> ResetError::validate() const {
  // TODO
  return std::make_pair(true, std::string());
}


void ResetError::load_from_json(const json_t &js) {
  rvector_t default_probs;
  JSON::get_value(default_probs, "probabilities", js);
  if (!default_probs.empty()) {
    set_probabilities(default_probs);
  }
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif