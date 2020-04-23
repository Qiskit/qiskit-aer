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

#ifndef _aer_noise_quantum_error_hpp_
#define _aer_noise_quantum_error_hpp_

#include "framework/opset.hpp"
#include "simulators/superoperator/superoperator_state.hpp"

namespace AER {
namespace Noise {

//=========================================================================
// Quantum Error class
//=========================================================================

// Quantum error class that can model any error that is expressed as a
// qobj instruction acting on qubits.

class QuantumError {
public:

  // Methods for sampling
  enum class Method {standard, superop};

  // Alias for return type
  using NoiseOps = std::vector<Operations::Op>;

  //-----------------------------------------------------------------------
  // Sampling method
  //-----------------------------------------------------------------------

  // Sample a noisy implementation of op
  NoiseOps sample_noise(const reg_t &qubits,
                        RngEngine &rng,
                        Method method = Method::standard) const;

  // Return the opset for the quantum error
  const Operations::OpSet& opset() const {return opset_;}

  // Return the superoperator matrix representation of the error.
  // If the error cannot be converted to a superoperator and error
  // will be raised.
  const cmatrix_t& superoperator() const;

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Load a QuantumError object from a JSON Error object
  void load_from_json(const json_t &js);

  // Sets the sub-circuits and probabilities to be sampled from.
  // The length of the circuits vector and probability vector must be equal.
  void set_circuits(const std::vector<NoiseOps> &circuits,
                    const rvector_t &probs);

  // Construct a quantum error from a set of Kraus matrices
  // This will factor out any identity or unitary Kraus operators into
  // non-kraus subcircuits.
  void set_from_kraus(const std::vector<cmatrix_t> &mats);

  // Compute the superoperator representation of the quantum error
  void compute_superoperator();
  
  //-----------------------------------------------------------------------
  // Utility
  //-----------------------------------------------------------------------

  // Set number of qubits or memory bits for error
  inline void set_num_qubits(uint_t num_qubits) {num_qubits_ = num_qubits;}

  // Get number of qubits or memory bits for error
  inline uint_t get_num_qubits() const {return num_qubits_;}

  // Set the sampled errors to be applied after the original operation
  inline void set_errors_after() {errors_after_op_ = true;}

  // Set the sampled errors to be applied before the original operation
  inline void set_errors_before() {errors_after_op_ = false;}

  // Returns true if the errors are to be applied after the operation
  inline bool errors_after() const {return errors_after_op_;}

  // Set threshold for checking probabilities and matrices
  void set_threshold(double);

protected:
  // Number of qubits sthe error applies to
  uint_t num_qubits_ = 0;

  // Probabilities, first entry is no-error (identity)
  rvector_t probabilities_;

  // List of unitary error matrices
  std::vector<NoiseOps> circuits_;

  // List of OpTypes contained in error circuits
  Operations::OpSet opset_;

  // threshold for validating if matrices are unitary
  double threshold_ = 1e-10;

  // Superoperator matrix representation of the error
  cmatrix_t superoperator_;

  // flag for where errors should be applied relative to the sampled op
  bool errors_after_op_ = true;  
};

//-------------------------------------------------------------------------
// Implementation: Mixed unitary error subclass
//-------------------------------------------------------------------------


QuantumError::NoiseOps QuantumError::sample_noise(const reg_t &qubits,
                                                  RngEngine &rng,
                                                  Method method) const {
  if (qubits.size() < get_num_qubits()) {
    std::stringstream msg;
    msg << "QuantumError: qubits size (" << qubits.size() << ")";
    msg << " < error qubits (" << get_num_qubits() << ").";
    throw std::invalid_argument(msg.str());
  }
  switch (method) {
    case Method::superop: {
      // Truncate qubits to size of the actual error
      reg_t op_qubits = qubits;
      op_qubits.resize(get_num_qubits());
      auto op = Operations::make_superop(op_qubits, superoperator());
      return NoiseOps({op});
    }
    default: {
      auto r = rng.rand_int(probabilities_);
      // Check for invalid arguments
      if (r + 1 > circuits_.size()) {
        throw std::invalid_argument(
          "QuantumError: probability outcome (" + std::to_string(r) + ")"
          " is greater than number of circuits (" + std::to_string(circuits_.size()) + ")."
        );
      }
      NoiseOps noise_ops = circuits_[r];
      // Add qubits to noise op commands;
      for (auto &op : noise_ops) {
        // Update qubits based on position in qubits list
        for (auto &qubit: op.qubits) {
          qubit = qubits[qubit];
        }
      }
      return noise_ops;
    }
  }
}

void QuantumError::set_threshold(double threshold) {
  threshold_ = std::abs(threshold);
}

void QuantumError::set_circuits(const std::vector<NoiseOps> &circuits,
                                const rvector_t &probs) {
  if (probs.size() != circuits.size()) {
    throw std::invalid_argument(
      "QuantumError: invalid input, number of circuits (" +
      std::to_string(circuits.size()) + ") and number of probabilities (" +
      std::to_string(probs.size()) + ") are not equal."
    );
  }
  // Check probability vector
  double total = 0.;
  bool probs_valid = true;
  uint_t num_qubits = 0;
  for (const auto &p : probs) {
    probs_valid &= !(p < 0 || p > 1);
    total += p;
  }
  if (!probs_valid || std::abs(total - 1.0) > threshold_) {
    throw std::invalid_argument("QuantumError: invalid probability vector total (" +
                                std::to_string(total) + "!= 1)");
  }
  // Reset OpSet
  opset_ = Operations::OpSet();
  // Add elements with non-zero probability
  for (size_t j=0; j < probs.size(); j++ ) {
    if (probs[j] > threshold_) {
      probabilities_.push_back(probs[j]);
      circuits_.push_back(circuits[j]);
      for (const auto &op: circuits[j]) {
        // Check max qubit size
        for (const auto &qubit : op.qubits) {
          num_qubits = std::max(num_qubits, qubit + 1);
        }
        // Record op in opset
        opset_.insert(op);
      }
    }
  }
  set_num_qubits(num_qubits);
}


void QuantumError::set_from_kraus(const std::vector<cmatrix_t> &mats) {
  // Check input isn't empty
  if (mats.empty())
    throw std::invalid_argument("QuantumError: Kraus channel input is empty.");

  // Check input is a CPTP map
  if (Utils::is_cptp_kraus(mats, threshold_) == false)
    throw std::invalid_argument("QuantumError: Kraus channel input is not a CPTP map.");

  // Get number of qubits from first Kraus operator
  size_t mat_dim = mats[0].GetRows();
  auto num_qubits = static_cast<unsigned>(std::log2(mat_dim));
  set_num_qubits(num_qubits);
  if (mat_dim != 1UL << num_qubits)
    throw std::invalid_argument("QuantumError: Kraus channel input is a multi-qubit channel.");

  // Check if each matrix is a:
  // - scaled identity matrix
  // - scaled non-identity unitary matrix
  // - a non-unitary Kraus operator

  // Probabilities
  double p_unitary = 0.;  // total probability of all unitary ops
  rvector_t probs = {0.}; // initialize with probability of Identity

  // Matrices
  std::vector<cmatrix_t> unitaries; // non-identity unitaries
  std::vector<cmatrix_t> kraus; // non-unitary Kraus matrices

  for (const auto &mat : mats) {
    if (!Utils::is_square(mat) && !Utils::is_diagonal(mat)) {
      throw std::invalid_argument("Error matrix is not square or diagonal.");
    }
    // Get the value of the first non-zero diagonal element of mat * dagger(mat) for rescaling
    double p = 0.;
    for (size_t i=0; i < mat.GetColumns(); i ++) {
      for (size_t j=0; j < mat.GetRows(); j ++) {
        p += std::real(std::abs(mat(j, i) * std::conj(mat(j, i)) ));
      }
      if (p > threshold_)
        break;
    }
    if (p > 0) {
      // Rescale mat by probability
      cmatrix_t tmp = (1 / std::sqrt(p)) * mat;
      // Check if rescaled matrix is an identity
      if (Utils::is_identity(tmp, threshold_) || Utils::is_diagonal_identity(tmp, threshold_)) {
        // Add to identity probability
        probs[0] += p;
        p_unitary += p;
      }
      // Check if rescaled matrix is a unitary
      else if (Utils::is_unitary(tmp, threshold_)) {
        unitaries.push_back(tmp);
        probs.push_back(p); // add probability for unitary
        p_unitary += p;
      } else {
        // Original matrix is non-unitary so add original matrix to Kraus ops
        kraus.push_back(mat);
      }
    }
  }

  // Create noise circuits:
  std::vector<NoiseOps> circuits;
  
  // Add identity
  Operations::Op iden;
  iden.name = "id";
  iden.qubits = reg_t({0});
  iden.type = Operations::OpType::gate;
  circuits.push_back({iden});

  // Create n-qubit argument {0, ... , n-1}
  // for indexing remaining errors operators
  reg_t error_qubits(num_qubits);
  std::iota(error_qubits.begin(), error_qubits.end(), 0);
  for (size_t j=1; j < probs.size(); j++) {
    auto op = Operations::make_unitary(error_qubits, unitaries[j - 1]);
    circuits.push_back({op});
  }

  // Add Kraus
  // Probability of non-unitary Kraus error is 1 - p_total
  double p_kraus = 1.0 - p_unitary;
  if (std::abs(p_kraus) > threshold_) {
    // Rescale Kraus operators by probability
    Utils::scalar_multiply_inplace(kraus, complex_t(1. / std::sqrt(p_kraus), 0.0));
    // Check rescaled Kraus map is still CPTP
    if (Utils::is_cptp_kraus(kraus, threshold_) == false) {
      throw std::invalid_argument("QuantumError: Rescaled non-unitary Kraus channel is not a CPTP map.");
    }
    // Add Kraus error subcircuit
    auto op = Operations::make_kraus(error_qubits, kraus);
    circuits.push_back({op});
    // Add kraus error prob
    probs.push_back(p_kraus);
  }
  
  // Add the circuits
  set_circuits(circuits, probs);
}


const cmatrix_t& QuantumError::superoperator() const {
  // Check the superoperator is actually computed
  // If not raise an exception
  if (superoperator_.empty()) {
    throw std::runtime_error("QuantumError: superoperator is empty.");
  }
  return superoperator_;
}


void QuantumError::compute_superoperator() {
  // Initialize superoperator matrix to correct size
  size_t dim = 1 << (2 * get_num_qubits());
  superoperator_.initialize(dim, dim);
  // We use the superoperator simulator state to do this
  QubitSuperoperator::State<> superop;
  for (size_t j=0; j<circuits_.size(); j++ ){
    // Initialize identity superoperator
    superop.initialize_qreg(get_num_qubits());
    // Apply each gate in the circuit
    // We don't need output data or RNG for this
    ExperimentData data;
    RngEngine rng;
    superop.apply_ops(circuits_[j], data, rng);
    superoperator_ += probabilities_[j] * superop.qreg().matrix();
  }
}


void QuantumError::load_from_json(const json_t &js) {
  rvector_t probs;
  JSON::get_value(probs, "probabilities", js);
  std::vector<NoiseOps> circuits;
  JSON::get_value(circuits, "instructions", js);
  set_circuits(circuits, probs);
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif