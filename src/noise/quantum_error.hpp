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

#include "framework/noise_utils.hpp"
#include "framework/opset.hpp"
#include "simulators/stabilizer/pauli.hpp"
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
  enum class Method { circuit, superop, kraus };

  // Alias for return type
  using NoiseOps = std::vector<Operations::Op>;

  //-----------------------------------------------------------------------
  // Sampling method
  //-----------------------------------------------------------------------

  // Sample a noisy implementation of op
  NoiseOps sample_noise(const reg_t &qubits, RngEngine &rng,
                        Method method = Method::circuit) const;

  // Return the opset for the quantum error
  const Operations::OpSet &opset() const { return opset_; }

  // Return the superoperator matrix representation of the error.
  // If the error cannot be converted to a superoperator an error
  // will be raised.
  const cmatrix_t &superoperator() const;

  // Return the canonical Kraus representation of the error.
  // If the error cannot be converted to a Kraus an error
  // will be raised.
  const std::vector<cmatrix_t> &kraus() const;

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Load a QuantumError object from a JSON Error object
  void load_from_json(const json_t &js);

  // Sets the sub-circuits and probabilities to be sampled from.
  // The length of the circuits vector and probability vector must be equal.
  void set_circuits(const std::vector<NoiseOps> &circuits,
                    const rvector_t &probs);

  // Sets the generator Pauli sub-circuits and rates to be sampled from.
  // The length of the circuits vector and probability vector must be equal.
  void set_generators(const std::vector<NoiseOps> &circuits,
                      const rvector_t &rates);

  // Construct a quantum error from a set of Kraus matrices
  // This will factor out any identity or unitary Kraus operators into
  // non-kraus subcircuits.
  void set_from_kraus(const std::vector<cmatrix_t> &mats);

  // Compute the superoperator representation of the quantum error
  void compute_superoperator();

  // Compute canonical Kraus representation of the quantum error
  void compute_kraus();

  //-----------------------------------------------------------------------
  // Utility
  //-----------------------------------------------------------------------

  // Set number of qubits or memory bits for error
  inline void set_num_qubits(uint_t num_qubits) { num_qubits_ = num_qubits; }

  // Get number of qubits or memory bits for error
  inline uint_t get_num_qubits() const { return num_qubits_; }

  // Set the sampled errors to be applied after the original operation
  inline void set_errors_after() { errors_after_op_ = true; }

  // Set the sampled errors to be applied before the original operation
  inline void set_errors_before() { errors_after_op_ = false; }

  // Returns true if the errors are to be applied after the operation
  inline bool errors_after() const { return errors_after_op_; }

  // Set threshold for checking probabilities and matrices
  void set_threshold(double);

protected:
  // Number of qubits sthe error applies to
  uint_t num_qubits_ = 0;

  // Probabilities, first entry is no-error (identity)
  rvector_t probabilities_;

  // Generator rates if Pauli lindblad generated error
  rvector_t rates_;

  // List of unitary error matrices or Pauli generators
  std::vector<NoiseOps> circuits_;

  // List of OpTypes contained in error circuits
  Operations::OpSet opset_;

  // threshold for validating if matrices are unitary
  double threshold_ = 1e-10;

  // Superoperator matrix representation of the error
  cmatrix_t superoperator_;

  std::vector<cmatrix_t> canonical_kraus_;

  // flag for where errors should be applied relative to the sampled op
  bool errors_after_op_ = true;

  // flag for whether circuits_ represent channel generators or terms
  bool generator_circuits_ = false;

  // Sample gate level noise from circuits
  NoiseOps sample_noise_circuits(const reg_t &qubits, RngEngine &rng) const;

  // Sample gate level noise from generators
  NoiseOps sample_noise_generators(const reg_t &qubits, RngEngine &rng) const;

  // Helper method to convert a Pauli Op to a Pauli::Pauli
  Pauli::Pauli<BV::BinaryVector> get_op_pauli(const Operations::Op &op) const;

  // Helper method to convert a I,X,Y,Z or Pauli Op to a Pauli op
  Operations::Op get_pauli_op(const Operations::Op &op) const;

  // Compute the superoperator representation of the quantum error from channel
  // circuits
  void compute_circuits_superoperator();

  // Compute the superoperator representation of the quantum error from
  // generator circuits
  void compute_generators_superoperator();
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
  case Method::kraus: {
    // Truncate qubits to size of the actual error
    reg_t op_qubits = qubits;
    op_qubits.resize(get_num_qubits());
    auto op = Operations::make_kraus(op_qubits, kraus());
    return NoiseOps({op});
  }
  default: {
    if (generator_circuits_) {
      return sample_noise_generators(qubits, rng);
    } else {
      return sample_noise_circuits(qubits, rng);
    }
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
        std::to_string(probs.size()) + ") are not equal.");
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
    throw std::invalid_argument(
        "QuantumError: invalid probability vector total (" +
        std::to_string(total) + "!= 1)");
  }
  // Reset OpSet
  opset_ = Operations::OpSet();
  // Add elements with non-zero probability
  for (size_t j = 0; j < probs.size(); j++) {
    if (probs[j] > threshold_) {
      probabilities_.push_back(probs[j]);
      circuits_.push_back(circuits[j]);
      for (const auto &op : circuits[j]) {
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

void QuantumError::set_generators(const std::vector<NoiseOps> &circuits,
                                  const rvector_t &rates) {
  if (rates.size() != circuits.size()) {
    throw std::invalid_argument(
        "QuantumError: invalid input, number of generator circuits (" +
        std::to_string(circuits.size()) + ") and number of rates (" +
        std::to_string(rates.size()) + ") are not equal.");
  }
  // Reset OpSet
  generator_circuits_ = true;
  opset_ = Operations::OpSet();
  uint_t num_qubits = 0;

  // Add elements with non-zero probability
  for (size_t j = 0; j < rates.size(); j++) {
    if (abs(rates[j]) > threshold_) {
      if (rates[j] < 0) {
        throw std::invalid_argument(
            "QuantumError: cannot contain negative rates");
      }
      NoiseOps circuit;
      for (const auto &op : circuits[j]) {
        if (op.name == "i") {
          // Ignore identities
          continue;
        }
        auto pauli_op = get_pauli_op(op);
        circuit.push_back(pauli_op);
        opset_.insert(pauli_op);
        // Check max qubit size
        for (const auto &qubit : op.qubits) {
          num_qubits = std::max(num_qubits, qubit + 1);
        }
      }
      rates_.push_back(rates[j]);
      circuits_.push_back(circuit);
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
    throw std::invalid_argument(
        "QuantumError: Kraus channel input is not a CPTP map.");

  // Get number of qubits from first Kraus operator
  size_t mat_dim = mats[0].GetRows();
  auto num_qubits = static_cast<unsigned>(std::log2(mat_dim));
  set_num_qubits(num_qubits);
  if (mat_dim != 1ULL << num_qubits)
    throw std::invalid_argument(
        "QuantumError: Kraus channel input is a multi-qubit channel.");

  // Check if each matrix is a:
  // - scaled identity matrix
  // - scaled non-identity unitary matrix
  // - a non-unitary Kraus operator

  // Probabilities
  double p_unitary = 0.;  // total probability of all unitary ops
  rvector_t probs = {0.}; // initialize with probability of Identity

  // Matrices
  std::vector<cmatrix_t> unitaries; // non-identity unitaries
  std::vector<cmatrix_t> kraus;     // non-unitary Kraus matrices

  for (const auto &mat : mats) {
    if (!Utils::is_square(mat) && !Utils::is_diagonal(mat)) {
      throw std::invalid_argument("Error matrix is not square or diagonal.");
    }
    // Get the value of the first non-zero diagonal element of mat * dagger(mat)
    // for rescaling
    double p = 0.;
    for (size_t i = 0; i < mat.GetColumns(); i++) {
      for (size_t j = 0; j < mat.GetRows(); j++) {
        p += std::real(std::abs(mat(j, i) * std::conj(mat(j, i))));
      }
      if (p > threshold_)
        break;
    }
    if (p > 0) {
      // Rescale mat by probability
      cmatrix_t tmp = (1 / std::sqrt(p)) * mat;
      // Check if rescaled matrix is an identity
      if (Utils::is_identity(tmp, threshold_) ||
          Utils::is_diagonal_identity(tmp, threshold_)) {
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
  for (size_t j = 1; j < probs.size(); j++) {
    auto op = Operations::make_unitary(error_qubits, unitaries[j - 1]);
    circuits.push_back({op});
  }

  // Add Kraus
  // Probability of non-unitary Kraus error is 1 - p_total
  double p_kraus = 1.0 - p_unitary;
  if (std::abs(p_kraus) > threshold_) {
    // Rescale Kraus operators by probability
    Utils::scalar_multiply_inplace(kraus,
                                   complex_t(1. / std::sqrt(p_kraus), 0.0));
    // Check rescaled Kraus map is still CPTP
    if (Utils::is_cptp_kraus(kraus, threshold_) == false) {
      throw std::invalid_argument("QuantumError: Rescaled non-unitary Kraus "
                                  "channel is not a CPTP map.");
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

const cmatrix_t &QuantumError::superoperator() const {
  // Check the superoperator is actually computed
  // If not raise an exception
  if (superoperator_.empty()) {
    throw std::runtime_error("QuantumError: superoperator is empty.");
  }
  return superoperator_;
}

const std::vector<cmatrix_t> &QuantumError::kraus() const {
  // Check the canonical Kraus method is actually computed
  // If not raise an exception
  if (canonical_kraus_.empty()) {
    throw std::runtime_error("QuantumError: Kraus is empty.");
  }
  return canonical_kraus_;
}

void QuantumError::compute_superoperator() {
  if (generator_circuits_) {
    compute_generators_superoperator();
  } else {
    compute_circuits_superoperator();
  }
}

void QuantumError::compute_kraus() {
  // Check superoperator representation is computed
  if (superoperator_.empty()) {
    compute_superoperator();
  }
  // Conver to Kraus
  size_t dim = 1 << get_num_qubits();
  canonical_kraus_ = Utils::superop2kraus(superoperator_, dim);
}

void QuantumError::load_from_json(const json_t &js) {
  std::vector<NoiseOps> circuits;
  JSON::get_value(circuits, "instructions", js);
  if (JSON::check_key("rates", js)) {
    rvector_t rates;
    JSON::get_value(rates, "rates", js);
    set_generators(circuits, rates);
  } else {
    rvector_t probs;
    JSON::get_value(probs, "probabilities", js);
    set_circuits(circuits, probs);
  }
}

Pauli::Pauli<BV::BinaryVector>
QuantumError::get_op_pauli(const Operations::Op &op) const {
  // Initialize an empty Pauli
  Pauli::Pauli<BV::BinaryVector> pauli(num_qubits_);
  const auto size = op.qubits.size();
  const auto pauli_str = op.string_params[0];
  for (size_t i = 0; i < op.qubits.size(); ++i) {
    const auto qubit = op.qubits[size - 1 - i];
    switch (pauli_str[i]) {
    case 'I':
      break;
    case 'X':
      pauli.X.set1(qubit);
      break;
    case 'Y':
      pauli.X.set1(qubit);
      pauli.Z.set1(qubit);
      break;
    case 'Z':
      pauli.Z.set1(qubit);
      break;
    default:
      throw std::invalid_argument("invalid Pauli \'" +
                                  std::to_string(pauli_str[i]) + "\'.");
    }
  }
  return pauli;
}

Operations::Op QuantumError::get_pauli_op(const Operations::Op &op) const {
  const std::string allowed_gates = "ixyz";
  if (op.name == "pauli") {
    return op;
  }
  if (op.name.size() == 1 &&
      allowed_gates.find(op.name[0]) != std::string::npos) {
    // Convert to a Pauli op
    Operations::Op pauli_op;
    pauli_op.type = Operations::OpType::gate;
    pauli_op.name = "pauli";
    pauli_op.qubits = op.qubits;
    return pauli_op;
  }
  throw std::invalid_argument(
      "QuantumError: generator errors can only contain Pauli ops");
}

void QuantumError::compute_circuits_superoperator() {
  // Initialize superoperator matrix to correct size
  size_t dim = 1ULL << (2 * get_num_qubits());
  superoperator_.initialize(dim, dim);
  // We use the superoperator simulator state to do this
  QubitSuperoperator::State<> superop;
  for (size_t j = 0; j < circuits_.size(); j++) {
    // Initialize identity superoperator
    superop.initialize_qreg(get_num_qubits());
    // Apply each gate in the circuit
    // We don't need output data or RNG for this
    ExperimentResult data;
    RngEngine rng;
    superop.apply_ops(circuits_[j].cbegin(), circuits_[j].cend(), data, rng);
    superoperator_ += probabilities_[j] * superop.move_to_matrix();
  }
}

void QuantumError::compute_generators_superoperator() {
  // Initialize superoperator matrix to correct size
  size_t dim = 1ULL << (2 * get_num_qubits());
  const cmatrix_t iden = Linalg::Matrix::identity(dim);
  superoperator_ = iden;

  // We use the superoperator simulator state to do this
  QubitSuperoperator::State<> superop;

  for (size_t j = 0; j < circuits_.size(); j++) {
    // Initialize identity superoperator
    superop.initialize_qreg(get_num_qubits());
    // Apply each gate in the circuit
    // We don't need output data or RNG for this
    ExperimentResult data;
    RngEngine rng;
    superop.apply_ops(circuits_[j].cbegin(), circuits_[j].cend(), data, rng);

    // Now compute the superoperator for the generator term
    auto p_iden = 0.5 + 0.5 * std::exp(-2 * rates_[j]);
    const auto non_iden = superoperator_ * superop.move_to_matrix();
    superoperator_ = p_iden * superoperator_ + (1 - p_iden) * non_iden;
  }
}

QuantumError::NoiseOps
QuantumError::sample_noise_circuits(const reg_t &qubits, RngEngine &rng) const {
  auto r = rng.rand_int(probabilities_);
  // Check for invalid arguments
  if (r + 1 > circuits_.size()) {
    throw std::invalid_argument("QuantumError: probability outcome (" +
                                std::to_string(r) +
                                ")"
                                " is greater than number of circuits (" +
                                std::to_string(circuits_.size()) + ").");
  }
  NoiseOps noise_ops = circuits_[r];
  // Add qubits to noise op commands;
  for (auto &op : noise_ops) {
    // Update qubits based on position in qubits list
    for (auto &qubit : op.qubits) {
      qubit = qubits[qubit];
    }
  }
  return noise_ops;
}

QuantumError::NoiseOps
QuantumError::sample_noise_generators(const reg_t &qubits,
                                      RngEngine &rng) const {
  Pauli::Pauli<BV::BinaryVector> sampled_pauli(num_qubits_);
  for (uint_t i = 0; i < rates_.size(); ++i) {
    // Use the generator rate to calculate the probability of sampling the
    // generator Pauli for the single term channel exp(r D[P]) = p S[I] + (1-p)
    // * S[P] where p = 0.5 + 0.5 * exp(-2r)
    auto p_iden = 0.5 + 0.5 * std::exp(-2 * rates_[i]);
    if (rng.rand() > p_iden) {
      // Convert each circuit Op to Pauli so we can more efficiently compose
      // terms
      for (auto op : circuits_[i]) {
        auto term_pauli = get_op_pauli(op);
        sampled_pauli.X += term_pauli.X;
        sampled_pauli.Z += term_pauli.Z;
      }
    }
  }

  // If sampled Pauli is an identity we don't return error terms
  if (sampled_pauli == Pauli::Pauli<BV::BinaryVector>(num_qubits_)) {
    return NoiseOps();
  }

  // Convert back to an pauli Op for noise insertion
  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "pauli";
  op.qubits = qubits;
  op.string_params = {sampled_pauli.str()};
  return NoiseOps({op});
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif