/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    kraus_error.hpp
 * @brief   Kraus Error class for Qiskit-Aer simulator
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_noise_kraus_error_hpp_
#define _aer_noise_kraus_error_hpp_

#include "noise/abstract_error.hpp"

// TODO: optimization for gate fusion with original operator matrix

namespace AER {
namespace Noise {

//=========================================================================
// Kraus Error class
//=========================================================================

// This class is used to model a general CPTP error map in Kraus form
// It contains a vector of matrices for the Kraus operators implementing the
// error map E(rho) = sum_j K_j rho K_j^\dagger (1)
// It may optionally contain as single probability p which is the probability
// of the Kraus error applying. If this is set the error map is
// Error(rho) = (1-p) rho + p E(rho), with E defined as in (1).

class KrausError : public AbstractError {
public:

  //-----------------------------------------------------------------------
  // Error base required methods
  //-----------------------------------------------------------------------

  // Return the operator followed by a Kraus operation
  NoiseOps sample_noise(const reg_t &qubits,
                        RngEngine &rng) override;

  //-----------------------------------------------------------------------
  // Additional class methods
  //-----------------------------------------------------------------------

  // Set the Kraus matrices for the error
  void set_kraus(const std::vector<cmatrix_t> &ops);

  // Set the probability of the Kraus error channel being applied
  // the probability of no error (identity) will be 1 - p_Kraus
  void set_probability(double p_kraus);

  // Set the sampled errors to be applied after the original operation
  inline void set_errors_after() {errors_after_op_ = true;}

  // Set the sampled errors to be applied before the original operation
  inline void set_errors_before() {errors_after_op_ = false;}

  // Set whether the input operator should be combined into the error
  // term (if compatible). The default is True
  inline void combine_error(bool val) {combine_error_ = val;}

protected:
  // Probabilities of Kraus error being applied
  // When sampled the outcomes correspond to:
  // 0 -> Kraus error map applied
  // 1 -> no error
  std::discrete_distribution<uint_t> probabilities_;
  
  // The Kraus operation for the error channel
  // This still needs to have the qubits it is applied to added to the Op
  // which is done during the sample_noise method
  Operations::Op kraus_;

  // Flag for applying errors before or after the operation
  bool errors_after_op_ = true;

  // Combine error with input matrix to return a single operation
  // if it acts on the same qubits
  bool combine_error_ = true;
};


//=========================================================================
// Implementation
//=========================================================================

// TODO add gate fusion optimization
KrausError::NoiseOps KrausError::sample_noise(const reg_t &qubits,
                                              RngEngine &rng) {

  // check we have the correct number of qubits in the op for the error
  //if (qubits.size() != num_qubits_) {
  //  throw std::invalid_argument("Incorrect number of qubits for the error.");
  //}
  auto r = rng.rand_int(probabilities_);
  if (r == 1) {
    return {}; // no error
  }
  Operations::Op error = kraus_;
  error.qubits = qubits;
  return {error};
}


void KrausError::set_kraus(const std::vector<cmatrix_t> &ops) {
  kraus_ = Operations::Op();
  kraus_.name = "kraus";
  kraus_.mats = ops;
}


void KrausError::set_probability(double p_kraus) {
  if (p_kraus < 0 || p_kraus > 1) {
    throw std::invalid_argument("Invalid KrausError probability.");
  }
  probabilities_ = std::discrete_distribution<uint_t>({p_kraus, 1 - p_kraus});
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------

#endif