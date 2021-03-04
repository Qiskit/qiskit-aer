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

#ifndef _qv_density_matrix_hpp_
#define _qv_density_matrix_hpp_


#include "framework/utils.hpp"
#include "simulators/unitary/unitarymatrix.hpp"

namespace AER {
namespace QV {

//============================================================================
// DensityMatrix class
//============================================================================

// This class is derived from the UnitaryMatrix class and stores an N-qubit
// matrix as a 2*N-qubit vector.
// The vector is formed using column-stacking vectorization as under this
// convention left-matrix multiplication on qubit-n is equal to multiplication
// of the vectorized 2*N qubit vector also on qubit-n.

template <typename data_t = double>
class DensityMatrix : public UnitaryMatrix<data_t> {

public:
  // Parent class aliases
  using BaseVector = QubitVector<data_t>;
  using BaseMatrix = UnitaryMatrix<data_t>;

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  DensityMatrix() : DensityMatrix(0) {};
  explicit DensityMatrix(size_t num_qubits);
  DensityMatrix(const DensityMatrix& obj) {};
  DensityMatrix &operator=(const DensityMatrix& obj) {};

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Return the string name of the DensityMatrix class
  static std::string name() {return "density_matrix";}

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // The vector can be either a statevector or a vectorized density matrix
  // If the length of the data vector does not match either case for the
  // number of qubits an exception is raised.
  void initialize_from_vector(const cvector_t<double> &data);

  // Returns the number of qubits for the superoperator
  virtual uint_t num_qubits() const override {return BaseMatrix::num_qubits_;}

  // Convert qubit indicies to vectorized-density matrix qubitvector indices
  // For the QubitVector apply matrix function
  virtual reg_t superop_qubits(const reg_t &qubits) const;

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit unitary matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_unitary_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a N-qubit superoperator matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit superop.
  void apply_superop_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a N-qubit diagonal unitary matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_unitary_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a N-qubit diagonal superoperator matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_superop_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  //-----------------------------------------------------------------------
  // Apply Specialized Gates
  //-----------------------------------------------------------------------

  // Apply a 2-qubit Controlled-NOT gate to the state vector
  void apply_cnot(const uint_t qctrl, const uint_t qtrgt);

  // Apply 2-qubit controlled-phase gate
  void apply_cphase(const uint_t q0, const uint_t q1, const complex_t &phase);

  // Apply a 2-qubit SWAP gate to the state vector
  void apply_swap(const uint_t q0, const uint_t q1);

  // Apply a single-qubit Pauli-X gate to the state vector
  void apply_x(const uint_t qubit);

  // Apply a single-qubit Pauli-Y gate to the state vector
  void apply_y(const uint_t qubit);

  // Apply 1-qubit phase gate
  void apply_phase(const uint_t q, const complex_t &phase);

  // Apply a 3-qubit toffoli gate
  void apply_toffoli(const uint_t qctrl0, const uint_t qctrl1, const uint_t qtrgt);

  //-----------------------------------------------------------------------
  // Z-measurement outcome probabilities
  //-----------------------------------------------------------------------

  // Return the Z-basis measurement outcome probability P(outcome) for
  // outcome in [0, 2^num_qubits - 1]
  virtual double probability(const uint_t outcome) const override;

  //-----------------------------------------------------------------------
  // Expectation Values
  //-----------------------------------------------------------------------

  // Return Pauli expectation value
  double expval_pauli(const reg_t &qubits, const std::string &pauli,const complex_t initial_phase=1.0) const;

protected:

  // Construct a vectorized superoperator from a vectorized matrix
  // This is equivalent to vec(tensor(conj(A), A))
  cvector_t<double> vmat2vsuperop(const cvector_t<double> &vmat) const;

  // Qubit threshold for when apply unitary will apply as two matrix multiplications
  // rather than as a 2n-qubit superoperator matrix.
  size_t apply_unitary_threshold_ = 4;
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/


//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t>
DensityMatrix<data_t>::DensityMatrix(size_t num_qubits)
  : UnitaryMatrix<data_t>(num_qubits) {}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
void DensityMatrix<data_t>::initialize() {
  // Zero the underlying vector
  BaseVector::zero();
  // Set to be all |0> sate
  BaseVector::data_[0] = 1.0;
}

template <typename data_t>
void DensityMatrix<data_t>::initialize_from_vector(const cvector_t<double> &statevec) {
  if (BaseVector::data_size_ == statevec.size()) {
    // Use base class initialize for already vectorized matrix
    BaseVector::initialize_from_vector(statevec);
  } else if (BaseVector::data_size_ == statevec.size() * statevec.size()) {
    // Convert statevector into density matrix
    cvector_t<double> densitymat = AER::Utils::tensor_product(AER::Utils::conjugate(statevec),
                                                      statevec);
    std::move(densitymat.begin(), densitymat.end(), BaseVector::data_);
  } else {
    throw std::runtime_error("DensityMatrix::initialize input vector is incorrect length. Expected: " +
                             std::to_string(BaseVector::data_size_) + " Received: " +
                             std::to_string(statevec.size()));
  }
}

//------------------------------------------------------------------------------
// Apply matrix functions
//------------------------------------------------------------------------------

template <typename data_t>
reg_t DensityMatrix<data_t>::superop_qubits(const reg_t &qubits) const {
  reg_t superop_qubits = qubits;
  // Number of qubits
  const auto nq = num_qubits();
  for (const auto &q: qubits) {
    superop_qubits.push_back(q + nq);
  }
  return superop_qubits;
}

template <typename data_t>
cvector_t<double> DensityMatrix<data_t>::vmat2vsuperop(const cvector_t<double> &vmat) const {
  // Get dimension of unvectorized matrix
  size_t dim = size_t(std::sqrt(vmat.size()));
  cvector_t<double> ret(dim * dim * dim * dim, 0.);
  for (size_t i=0; i < dim; i++)
    for (size_t j=0; j < dim; j++)
      for (size_t k=0; k < dim; k++)
        for (size_t l=0; l < dim; l++)
          ret[dim*i+k+(dim*dim)*(dim*j+l)] = std::conj(vmat[i+dim*j])*vmat[k+dim*l];
  return ret;
}

template <typename data_t>
void DensityMatrix<data_t>::apply_superop_matrix(const reg_t &qubits,
                                                 const cvector_t<double> &mat) {
  BaseVector::apply_matrix(superop_qubits(qubits), mat);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_diagonal_superop_matrix(const reg_t &qubits,
                                                          const cvector_t<double> &diag) {
  BaseVector::apply_diagonal_matrix(superop_qubits(qubits), diag);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_unitary_matrix(const reg_t &qubits,
                                                 const cvector_t<double> &mat) {
  // Check if we apply as two N-qubit matrix multiplications or a single 2N-qubit matrix mult.
  if (qubits.size() > apply_unitary_threshold_) {
    // Apply as two N-qubit matrix mults
    auto nq = num_qubits();
    reg_t conj_qubits;
    for (const auto &q: qubits) {
      conj_qubits.push_back(q + nq);
    }
    // Apply id \otimes U
    BaseVector::apply_matrix(qubits, mat);
    // Apply conj(U) \otimes id
    BaseVector::apply_matrix(conj_qubits, AER::Utils::conjugate(mat));
  } else {
    // Apply as single 2N-qubit matrix mult.
    apply_superop_matrix(qubits, vmat2vsuperop(mat));
  }
}

template <typename data_t>
void DensityMatrix<data_t>::apply_diagonal_unitary_matrix(const reg_t &qubits,
                                                          const cvector_t<double> &diag) {
  // Apply as single 2N-qubit matrix mult.
  apply_diagonal_superop_matrix(qubits, AER::Utils::tensor_product(AER::Utils::conjugate(diag), diag));
}

//-----------------------------------------------------------------------
// Apply Specialized Gates
//-----------------------------------------------------------------------

template <typename data_t>
void DensityMatrix<data_t>::apply_cnot(const uint_t qctrl, const uint_t qtrgt) {
  std::vector<std::pair<uint_t, uint_t>> pairs = {
    {{1, 3}, {4, 12}, {5, 15}, {6, 14}, {7, 13}, {9, 11}}
  };
  const size_t nq = num_qubits();
  const reg_t qubits = {{qctrl, qtrgt, qctrl + nq, qtrgt + nq}};
  BaseVector::apply_permutation_matrix(qubits, pairs);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_phase(const uint_t q, const complex_t &phase) {
  const complex_t iphase = std::conj(phase);
  // Lambda function for CZ gate
  auto lambda = [&](const areg_t<1ULL << 2> &inds)->void {
    BaseVector::data_[inds[1]] *= phase;
    BaseVector::data_[inds[2]] *= iphase;
  };
  const auto nq = num_qubits();
  const areg_t<2> qubits = {{q, q + nq}};
  BaseVector::apply_lambda(lambda, qubits);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_cphase(const uint_t q0, const uint_t q1,
                                         const complex_t &phase) {
  const complex_t iphase = std::conj(phase);
  // Lambda function for CZ gate
  auto lambda = [&](const areg_t<1ULL << 4> &inds)->void {
    BaseVector::data_[inds[3]] *= phase;
    BaseVector::data_[inds[7]] *= phase;
    BaseVector::data_[inds[11]] *= phase;
    BaseVector::data_[inds[12]] *= iphase;
    BaseVector::data_[inds[13]] *= iphase;
    BaseVector::data_[inds[14]] *= iphase;
  };
  const auto nq = num_qubits();
  const areg_t<4> qubits = {{q0, q1, q0 + nq, q1 + nq}};
  BaseVector::apply_lambda(lambda, qubits);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_swap(const uint_t q0, const uint_t q1) {
  std::vector<std::pair<uint_t, uint_t>> pairs = {
   {{1, 2}, {4, 8}, {5, 10}, {6, 9}, {7, 11}, {13, 14}}
  };
  const size_t nq = num_qubits();
  const reg_t qubits = {{q0, q1, q0 + nq, q1 + nq}};
  BaseVector::apply_permutation_matrix(qubits, pairs);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_x(const uint_t qubit) {
  // Lambda function for X gate superoperator
  auto lambda = [&](const areg_t<1ULL << 2> &inds)->void {
    std::swap(BaseVector::data_[inds[0]], BaseVector::data_[inds[3]]);
    std::swap(BaseVector::data_[inds[1]], BaseVector::data_[inds[2]]);
  };
  // Use the lambda function
  const areg_t<2> qubits = {{qubit, qubit + num_qubits()}};
  BaseVector::apply_lambda(lambda, qubits);

}

template <typename data_t>
void DensityMatrix<data_t>::apply_y(const uint_t qubit) {
  // Lambda function for Y gate superoperator
  auto lambda = [&](const areg_t<1ULL << 2> &inds)->void {
    std::swap(BaseVector::data_[inds[0]], BaseVector::data_[inds[3]]);
    const std::complex<data_t> cache = std::complex<data_t>(-1) * BaseVector::data_[inds[1]];
    BaseVector::data_[inds[1]] = std::complex<data_t>(-1) * BaseVector::data_[inds[2]];
    BaseVector::data_[inds[2]] = cache;
  };
  // Use the lambda function
  const areg_t<2> qubits = {{qubit, qubit + num_qubits()}};
  BaseVector::apply_lambda(lambda, qubits);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_toffoli(const uint_t qctrl0,
                                          const uint_t qctrl1,
                                          const uint_t qtrgt) {
  std::vector<std::pair<uint_t, uint_t>> pairs = {
    {{3, 7}, {11, 15}, {19, 23}, {24, 56}, {25, 57}, {26, 58}, {27, 63},
    {28, 60}, {29, 61}, {30, 62}, {31, 59}, {35, 39}, {43,47}, {51, 55}}
  };
  const size_t nq = num_qubits();
  const reg_t qubits = {{qctrl0, qctrl1, qtrgt,
                         qctrl0 + nq, qctrl1 + nq, qtrgt + nq}};
  BaseVector::apply_permutation_matrix(qubits, pairs);
}

template <typename data_t>
double DensityMatrix<data_t>::expval_pauli(const reg_t &qubits,
                                           const std::string &pauli,const complex_t initial_phase) const {

  uint_t x_mask, z_mask, num_y, x_max;
  std::tie(x_mask, z_mask, num_y, x_max) = QV::pauli_masks_and_phase(qubits, pauli);

  // Special case for only I Paulis
  if (x_mask + z_mask == 0) {
    return std::real(BaseMatrix::trace());
  }

  // Size of density matrix 
  const size_t nrows = BaseMatrix::rows_;
  const size_t diag_stride = 1 + nrows;

  // specialize x_max == 0
  if (!x_mask) {
    auto lambda = [&](const int_t i, double &val_re, double &val_im)->void {
      (void)val_im; // unused
      auto val = std::real(BaseVector::data_[i * diag_stride]);
      if (z_mask && (AER::Utils::popcount(i & z_mask) & 1)) {
        val = -val;
      }
      val_re += val;
    };
    return std::real(BaseVector::apply_reduction_lambda(std::move(lambda), size_t(0), nrows));
  }

  auto phase = std::complex<data_t>(initial_phase);
  QV::add_y_phase(num_y, phase);

  const uint_t mask_u = ~MASKS[x_max + 1];
  const uint_t mask_l = MASKS[x_max];
  auto lambda = [&](const int_t i, double &val_re, double &val_im)->void {
    (void)val_im; // unused
    auto idx_vec = ((i << 1) & mask_u) | (i & mask_l);
    auto idx_mat = idx_vec ^ x_mask + nrows * idx_vec;
    // Since rho is hermitian rho[i, j] + rho[j, i] = 2 real(rho[i, j])
    auto val = 2 * std::real(phase * BaseVector::data_[idx_mat]);
    if (z_mask && (AER::Utils::popcount(idx_vec & z_mask) & 1)) {
      val = - val;
    }
    val_re += val;
  };
  return std::real(BaseVector::apply_reduction_lambda(
    std::move(lambda), size_t(0), nrows >> 1));
}

//-----------------------------------------------------------------------
// Z-measurement outcome probabilities
//-----------------------------------------------------------------------

template <typename data_t>
double DensityMatrix<data_t>::probability(const uint_t outcome) const {
  const auto shift = BaseMatrix::num_rows() + 1;
  return std::real(BaseVector::data_[outcome * shift]);
}

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const AER::QV::DensityMatrix<data_t>&m) {
  out << m.copy_to_matrix();
  return out;
}

//------------------------------------------------------------------------------
#endif // end module

