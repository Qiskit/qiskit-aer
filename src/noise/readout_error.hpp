/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_noise_readout_error_hpp_
#define _aer_noise_readout_error_hpp_

namespace AER {
namespace Noise {

//=========================================================================
// Readout Error class
//=========================================================================



class ReadoutError {
public:

  // Alias for return type
  using NoiseOps = std::vector<Operations::Op>;

  //-----------------------------------------------------------------------
  // Error sampling
  //-----------------------------------------------------------------------

  // Sample a noisy implementation of op
  NoiseOps sample_noise(const reg_t &memory,
                        RngEngine &rng) const;

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Load a ReadoutError object from a JSON Error object
  void load_from_json(const json_t &js);

  // Set the default assignment probabilities for measurement of each qubit
  // Each vector is a list of probabilities {P(0|q), P(1|q), ... P(n-1|q)}
  // For the no-error case this list viewed as a matrix should be an
  // identity matrix
  void set_probabilities(const std::vector<rvector_t> &probs);

  //-----------------------------------------------------------------------
  // Utility
  //-----------------------------------------------------------------------

  // Set number of qubits or memory bits for error
  inline void set_num_qubits(uint_t num_qubits) {num_qubits_ = num_qubits;}

  // Get number of qubits or memory bits for error
  inline uint_t get_num_qubits() const {return num_qubits_;}

protected:

  // Number of qubits/memory bits the error applies to
  uint_t num_qubits_ = 0;

  // Vector of assignment probability vectors
  std::vector<rvector_t> assignment_probabilities_; 

  // threshold for checking probabilities
  double threshold_ = 1e-10;
};

//-------------------------------------------------------------------------
// Implementation: Mixed unitary error subclass
//-------------------------------------------------------------------------

ReadoutError::NoiseOps ReadoutError::sample_noise(const reg_t &memory,
                                                  RngEngine &rng) const {
  (void)rng; // RNG is unused for readout error since it is handled by engine
  // Check assignment fidelity matrix is correct size
  if (memory.size() > get_num_qubits())
    throw std::invalid_argument("ReadoutError: number of qubits don't match assignment probability matrix.");
  // Initialize return ops,
  return {Operations::make_roerror(memory, assignment_probabilities_)};
}


void ReadoutError::set_probabilities(const std::vector<rvector_t> &probs) {
  assignment_probabilities_ = probs;
  set_num_qubits(assignment_probabilities_.size());
  for (const auto  &ps : assignment_probabilities_) {
    double total = 0.0;
    for (const auto &p : ps) {
      if (p < 0 || p > 1) {
        throw std::invalid_argument("ReadoutError probability is not valid (p=" + std::to_string(p) +").");
      }
      total += p;
    }
    if (std::abs(total - 1) > threshold_)
      throw std::invalid_argument("ReadoutError probability vector is not normalized.");
  }
}


void ReadoutError::load_from_json(const json_t &js) {
  std::vector<rvector_t> probs;
  JSON::get_value(probs, "probabilities", js);
  if (!probs.empty()) {
    set_probabilities(probs);
  }
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif