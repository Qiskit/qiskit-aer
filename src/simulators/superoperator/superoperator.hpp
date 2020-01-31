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

#ifndef _qv_superoperator_hpp_
#define _qv_superoperator_hpp_


#include "framework/utils.hpp"
#include "simulators/density_matrix/densitymatrix.hpp"

namespace QV {

//============================================================================
// Superoperator class
//============================================================================

// This class is derived from the DensityMatrix class and stores an N-qubit 
// superoperator as a 2 * N-qubit vector.
// The vector is formed using column-stacking vectorization of the
// superoperator (itself with respect to column-stacking vectorization).

template <typename data_t = double>
class Superoperator : public DensityMatrix<data_t> {

public:
  // Parent class aliases
  using BaseVector = QubitVector<data_t>;
  using BaseDensity = DensityMatrix<data_t>;
  using BaseUnitary = UnitaryMatrix<data_t>;

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  Superoperator() : Superoperator(0) {};
  explicit Superoperator(size_t num_qubits);
  Superoperator(const Superoperator& obj) = delete;
  Superoperator &operator=(const Superoperator& obj) = delete;

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------
  
  // Set the size of the vector in terms of qubit number
  void set_num_qubits(size_t num_qubits);

  // Returns the number of qubits for the superoperator
  virtual uint_t num_qubits() const override {return num_qubits_;}

  // Initialize to the identity superoperator
  void initialize();

  // Initializes the vector to a custom initial state.
  // The matrix can either be superoperator matrix or unitary matrix.
  // The type is inferred by the dimensions of the input matrix.
  void initialize_from_matrix(const AER::cmatrix_t &data);

protected:
  // Number of qubits for the superoperator
  size_t num_qubits_;
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
Superoperator<data_t>::Superoperator(size_t num_qubits) {
  set_num_qubits(num_qubits);
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <class data_t>
void Superoperator<data_t>::set_num_qubits(size_t num_qubits) {
  num_qubits_ = num_qubits;
  // Superoperator is same size matrix as a unitary matrix
  // of twice as many qubits
  BaseDensity::set_num_qubits(2 * num_qubits);
}


template <typename data_t>
void Superoperator<data_t>::initialize() {
  // Set underlying unitary matrix to identity
  BaseUnitary::initialize();
}


template <class data_t>
void Superoperator<data_t>::initialize_from_matrix(const AER::cmatrix_t &mat) {
  if (AER::Utils::is_square(mat)) {
    const size_t nrows = mat.GetRows();
    if (nrows == BaseUnitary::rows_) {
      // The matrix is the same size as the superoperator matrix so we
      // initialze as the matrix.
      BaseUnitary::initialize_from_matrix(mat);
      return;
    } else if (nrows * nrows == BaseUnitary::rows_) {
      // If the input matrix has half the number of rows we assume it is
      // A unitary matrix input so we convert to a superoperator
      BaseUnitary::initialize_from_matrix(
        AER::Utils::tensor_product(AER::Utils::conjugate(mat), mat)
      );
      return;
    }
  }
  // Throw an exception if the input matrix is the wrong size for
  // unitary or superoperator input
  throw std::runtime_error(
    "Superoperator::initial matrix is wrong size (" +
    std::to_string(BaseUnitary::rows_) + "," +
    std::to_string(BaseUnitary::rows_) + ")!=(" +
    std::to_string(mat.GetRows()) + "," + std::to_string(mat.GetColumns()) + ")."
  );
};


//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const QV::Superoperator<data_t>&m) {
  out << m.matrix();
  return out;
}

//------------------------------------------------------------------------------
#endif // end module

