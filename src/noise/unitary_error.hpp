/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    unitary_error.hpp
 * @brief   Unitary Error class for Qiskit-Aer simulator
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_noise_unitary_error_hpp_
#define _aer_noise_unitary_error_hpp_

#include "noise/abstract_error.hpp"

// TODO optimization for gate fusion with original operator matrix

namespace AER {
namespace Noise {

//=========================================================================
// Unitary Error class
//=========================================================================

// Unitary error class that can model mixed unitary error channels, and
// coherent unitary errors.

class UnitaryError : public AbstractError {
public:
  //-----------------------------------------------------------------------
  // Error base required methods
  //-----------------------------------------------------------------------

  // Sample a noisy implementation of op
  NoiseOps sample_noise(const reg_t &qubits,
                        RngEngine &rng) override;

  // Check that Error matrices are unitary, and that the correct number
  // of probabilities are specified.
  std::pair<bool, std::string> validate() const override;

  // Load a UnitaryError object from a JSON Error object
  void load_from_json(const json_t &js) override;

  //-----------------------------------------------------------------------
  // Additional class methods
  //-----------------------------------------------------------------------

  // Sets the probabilities for the given error matrices. The length
  // of the probability vector should be less than or equal to the number
  // of matrices. If the total of the vector is less than 1, the remaining
  // probability 1 - total is the probability of no error (identity matrix)
  void set_probabilities(const rvector_t &probs);

  // Sets the unitary error matrices for the error map
  void set_unitaries(const std::vector<cmatrix_t> &mats);

protected:
  // Probabilities, first entry is no-error (identity)
  std::discrete_distribution<uint_t> probabilities_;

  // List of unitary error matrices
  std::vector<cmatrix_t> unitaries_;

  // threshold for validating if matrices are unitary
  double unitary_threshold_ = 1e-10;
};

//-------------------------------------------------------------------------
// Implementation: Mixed unitary error subclass
//-------------------------------------------------------------------------

UnitaryError::NoiseOps UnitaryError::sample_noise(const reg_t &qubits,
                                                  RngEngine &rng) {
  auto r = rng.rand_int(probabilities_);
  // Check for no error case
  if (r == 0) {
    // no error
    return {};
  }
  // Check for invalid arguments
  if (r > unitaries_.size()) {
    throw std::invalid_argument("Unitary error probability vector does not match number of unitaries.");
  }
  if (unitaries_.empty()) {
    throw std::invalid_argument("Unitary error matrices are not set.");
  }
  // Get the error operation for the unitary
  Operations::Op error;
  error.name = "mat";
  error.mats.push_back(unitaries_[r - 1]);
  error.qubits = qubits;
  return {error};
}


void UnitaryError::set_probabilities(const rvector_t &probs) {
  rvector_t probs_with_identity({1.});
  bool probs_valid = true;
  for (const auto &p : probs) {
    probs_valid &= !(p < 0 || p > 1);
    probs_with_identity[0] -= p;
    probs_with_identity.push_back(p);
  }
  if (probs_with_identity[0] > 1 || probs_with_identity[0] < 0 || !probs_valid) {
    throw std::invalid_argument("UnitaryError: invalid probability vector.");
  }
  probabilities_ = std::discrete_distribution<uint_t>(
                      probs_with_identity.begin(),
                      probs_with_identity.end());
}


void UnitaryError::set_unitaries(const std::vector<cmatrix_t> &mats) {
  unitaries_ = mats;
}

std::pair<bool, std::string> UnitaryError::validate() const {
  bool valid = true;
  std::string msg = "UnitaryError not valid:";
  for (const auto &mat : unitaries_) {
    if (!Utils::is_unitary(mat, unitary_threshold_)) {
      valid = false;
      msg += " Matrix is not unitary.";
      break;
    }
  }
  if (probabilities_.probabilities().size() > unitaries_.size() + 1) {
    valid = false;
    msg += " Number of probabilities and unitaries do no match.";
  }
  if (valid)
    return std::make_pair(true, std::string());
  else
    return std::make_pair(false, msg);
}


void UnitaryError::load_from_json(const json_t &js) {
  rvector_t probs;
  JSON::get_value(probs, "probabilities", js);
  if (!probs.empty())
    set_probabilities(probs);
  std::vector<cmatrix_t> mats;
  JSON::get_value(mats, "matrices", js);
  if (!mats.empty())
    set_unitaries(mats);

  // Check input is valid unitary error
  auto valid = validate();
  if (valid.first == false)
    throw std::invalid_argument(valid.second);
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif