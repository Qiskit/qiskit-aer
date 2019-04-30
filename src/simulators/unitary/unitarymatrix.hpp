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

#ifndef _qv_unitary_matrix_hpp_
#define _qv_unitary_matrix_hpp_


#include "framework/utils.hpp"
#include "simulators/statevector/qubitvector.hpp"

namespace QV {

//============================================================================
// UnitaryMatrix class
//============================================================================

// This class is derived from the QubitVector class and stores an N-qubit 
// matrix as a 2*N-qubit vector.
// The vector is formed using column-stacking vectorization as under this
// convention left-matrix multiplication on qubit-n is equal to multiplication
// of the vectorized 2*N qubit vector also on qubit-n.

template <class data_t = complex_t*>
class UnitaryMatrix : public QubitVector<data_t> {

public:
  // Type aliases
  using BaseVector = QubitVector<data_t>;

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  UnitaryMatrix() : UnitaryMatrix(0) {};
  explicit UnitaryMatrix(size_t num_qubits);
  UnitaryMatrix(const UnitaryMatrix& obj) = delete;
  UnitaryMatrix &operator=(const UnitaryMatrix& obj) = delete;

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Set the size of the vector in terms of qubit number
  inline void set_num_qubits(size_t num_qubits);

  // Return the number of rows in the matrix
  size_t num_rows() const {return rows_;}

  // Returns the number of qubits for the current vector
  inline uint_t num_qubits() const { return num_qubits_;}

  // Returns a copy of the underlying data_t data as a complex vector
  AER::cmatrix_t matrix() const;

  // Return JSON serialization of UnitaryMatrix;
  json_t json() const;

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // If the length of the statevector does not match the number of qubits
  // an exception is raised.
  void initialize_from_matrix(const AER::cmatrix_t &mat);

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  size_t num_qubits_;
  size_t rows_;
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

template <class data_t>
inline void to_json(json_t &js, const UnitaryMatrix<data_t> &qmat) {
  js = qmat.json();
}

template <class data_t>
json_t UnitaryMatrix<data_t>::json() const {
  const int_t nrows = rows_;
  // Initialize empty matrix
  const json_t ZERO = complex_t(0.0, 0.0);
  json_t js = json_t(nrows, json_t(nrows, ZERO));
  
  if (BaseVector::json_chop_threshold_ > 0) {
    #pragma omp parallel if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
    {
    #ifdef _WIN32
      #pragma omp for
    #else
      #pragma omp for collapse(2)
    #endif
    for (int_t i=0; i < nrows; i++)
      for (int_t j=0; j < nrows; j++) {
        const auto val = BaseVector::data_[i + nrows * j];
        if (std::abs(val.real()) > BaseVector::json_chop_threshold_)
          js[i][j][0] = val.real();
        if (std::abs(val.imag()) > BaseVector::json_chop_threshold_)
          js[i][j][1] = val.imag();
      }
    }
  } else {
    #pragma omp parallel if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
    {
    #ifdef _WIN32
      #pragma omp for
    #else
      #pragma omp for collapse(2)
    #endif
    for (int_t i=0; i < nrows; i++)
      for (int_t j=0; j < nrows; j++) {
        const auto val = BaseVector::data_[i + nrows * j];
        js[i][j][0] = val.real();
        js[i][j][1] = val.imag();
      }
    }
  }
  return js;
}


//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <class data_t>
UnitaryMatrix<data_t>::UnitaryMatrix(size_t num_qubits) {
  set_num_qubits(num_qubits);
}

//------------------------------------------------------------------------------
// Convert data vector to matrix
//------------------------------------------------------------------------------

template <class data_t>
AER::cmatrix_t UnitaryMatrix<data_t>::matrix() const {

  AER::cmatrix_t ret(rows_, rows_);
  const int_t nrows = rows_;

  #pragma omp parallel if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
  {
  #ifdef _WIN32
    #pragma omp for
  #else
    #pragma omp for collapse(2)
  #endif
    for (int_t i=0; i < nrows; i++)
      for (int_t j=0; j < nrows; j++) {
        ret(i, j) = BaseVector::data_[i + nrows * j];
      }
  } // end omp parallel
  return ret;
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <class data_t>
void UnitaryMatrix<data_t>::initialize() {
  // Zero the underlying vector
  BaseVector::zero();
  // Set to be identity matrix
  const int_t nrows = rows_;    // end for k loop
 #pragma omp parallel if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
  for (int_t k = 0; k < nrows; ++k) {
    BaseVector::data_[k * (nrows + 1)] = 1.0;
  }
}

template <class data_t>
void UnitaryMatrix<data_t>::initialize_from_matrix(const AER::cmatrix_t &mat) {
  const int_t nrows = rows_;    // end for k loop
  if (nrows != static_cast<int_t>(mat.GetRows()) ||
      nrows != static_cast<int_t>(mat.GetColumns())) {
    throw std::runtime_error(
      "UnitaryMatrix::initialize input matrix is incorrect shape (" +
      std::to_string(nrows) + "," + std::to_string(nrows) + ")!=(" +
      std::to_string(mat.GetRows()) + "," + std::to_string(mat.GetColumns()) + ")."
    );
  }
  if (AER::Utils::is_unitary(mat, 1e-10) == false) {
    throw std::runtime_error(
      "UnitaryMatrix::initialize input matrix is not unitary."
    );
  }
#pragma omp parallel if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
  for (int_t row = 0; row < nrows; ++row)
    for  (int_t col = 0; col < nrows; ++col) {
      BaseVector::data_[row + nrows * col] = mat(row, col);
    }
}

template <class data_t>
void UnitaryMatrix<data_t>::set_num_qubits(size_t num_qubits) {
  // Set the number of rows for the matrix
  num_qubits_ = num_qubits;
  rows_ = 1ULL << num_qubits;
  // Set the underlying vectorized matrix to be 2 * number of qubits
  BaseVector::set_num_qubits(2 * num_qubits);
}

//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <class data_t>
inline std::ostream &operator<<(std::ostream &out, const QV::UnitaryMatrix<data_t>&m) {
  out << m.matrix();
  return out;
}

//------------------------------------------------------------------------------
#endif // end module

