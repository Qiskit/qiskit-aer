/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _qubit_matrix_hpp_
#define _qubit_matrix_hpp_

//#define DEBUG // error checking

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "simulators/qubitvector/indexing.hpp" // multipartite qubit indexing

namespace QM {

// Indexing Types
using Indexing::uint_t;
using Indexing::int_t;
using Indexing::Qubit::indexes;
using Indexing::Qubit::indexes_dynamic;

// Data types
using complex_t = std::complex<double>;
using cmatrix_t = matrix<complex_t>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;

//============================================================================
// QubitMatrix class
//============================================================================

// Template class for qubit vector.
// The arguement of the template must have an operator[] access method.
// The following methods may also need to be template specialized:
//   * set_num_qubits(size_t)
//   * initialize()
//   * initialize(cvector_t)
// If the template argument does not have these methods then template
// specialization must be used to override the default implementations.

template <class statematrix_t = cmatrix_t>
class QubitMatrix {

public:

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  QubitMatrix();
  explicit QubitMatrix(size_t num_qubits);
  ~QubitMatrix();

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Set the size of the vector in terms of qubit number
  inline void set_num_qubits(size_t num_qubits);

  // Returns the size of the underlying n-qubit vector
  inline uint_t size() const { return num_states_;}

  // Returns the number of qubits for the current vector
  inline uint_t num_qubits() const { return num_qubits_;}

  // Returns a reference to the underlying statematrix_t data class
  inline statematrix_t &data() { return statematrix_;}

  // Returns a copy of the underlying statematrix_t data class
  inline statematrix_t data() const { return statematrix_;}

  // Returns a copy of the underlying statematrix_t data as a complex vector
  cmatrix_t matrix() const;

  // Return JSON serialization of QubitMatrix;
  json_t json() const;

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // If the length of the statevector does not match the number of qubits
  // an exception is raised.
  void initialize(const cmatrix_t &statemat);

  //-----------------------------------------------------------------------
  // Configuration settings
  //-----------------------------------------------------------------------

  // Set the threshold for chopping values to 0 in JSON
  void set_json_chop_threshold(double threshold);

  // Set the threshold for chopping values to 0 in JSON
  double get_json_chop_threshold() {return json_chop_threshold_;}

  // Set the maximum number of OpenMP thread for operations.
  void set_omp_threads(int n);

  // Get the maximum number of OpenMP thread for operations.
  uint_t get_omp_threads() {return omp_threads_;}

  // Set the qubit threshold for activating OpenMP.
  // If self.qubits() > threshold OpenMP will be activated.
  void set_omp_threshold(int n);

  // Get the qubit threshold for activating OpenMP.
  uint_t get_omp_threshold() {return omp_threshold_;}

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const std::vector<uint_t> &qubits, const cmatrix_t &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const std::vector<uint_t> &qubits,
                             const cmatrix_t &mat);

  //-----------------------------------------------------------------------
  // Apply Specialized Gates
  //-----------------------------------------------------------------------

  // Apply a 2-qubit Controlled-NOT gate to the state vector
  void apply_cnot(const uint_t qctrl, const uint_t qtrgt);

  // Apply a 2-qubit Controlled-Z gate to the state vector
  void apply_cz(const uint_t q0, const uint_t q1);

  // Apply a 2-qubit SWAP gate to the state vector
  void apply_swap(const uint_t q0, const uint_t q1);

  // Apply a single-qubit Pauli-X gate to the state vector
  void apply_x(const uint_t qubit);

  // Apply a single-qubit Pauli-Y gate to the state vector
  void apply_y(const uint_t qubit);

  // Apply a single-qubit Pauli-Z gate to the state vector
  void apply_z(const uint_t qubit);

  // Apply a 3-qubit toffoli gate
  void apply_toffoli(const uint_t qctrl0, const uint_t qctrl1, const uint_t qtrgt);

  //-----------------------------------------------------------------------
  // Matrix Operators
  //-----------------------------------------------------------------------

  // Element access
  complex_t &operator()(uint_t row, uint_t col);
  complex_t operator()(uint_t row, uint_t col) const;
  complex_t &operator[](uint_t element);
  complex_t operator[](uint_t element) const;

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  size_t num_qubits_;
  size_t num_states_;
  statematrix_t statematrix_;

 //-----------------------------------------------------------------------
  // Config settings
  //----------------------------------------------------------------------- 
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 6;   // Qubit threshold for multithreading when enabled
  double json_chop_threshold_ = 0;  // Threshold for choping small values
                                    // in JSON serialization

  //-----------------------------------------------------------------------
  // State update functions with Lambda function bodies
  //-----------------------------------------------------------------------
  
  template <typename Lambda>
  void apply_matrix_lambda(const uint_t qubit,
                           const cmatrix_t &mat,
                           Lambda&& func);
  
  template <size_t N, typename Lambda>
  void apply_matrix_lambda(const std::array<uint_t, N> &qubits,
                           const cmatrix_t &mat,
                           Lambda&& func);

  template <typename Lambda>
  void apply_matrix_lambda(const std::vector<uint_t> &qubits,
                           const cmatrix_t &mat,
                           Lambda&& func);

  //-----------------------------------------------------------------------
  // Matrix helper functions
  //-----------------------------------------------------------------------
  
  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  template <size_t N>
  void apply_matrix(const std::array<uint_t, N> &qubits, const cmatrix_t &mat);
  void apply_matrix(const std::array<uint_t, 1> &qubits, const cmatrix_t &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  template <size_t N>
  void apply_diagonal_matrix(const std::array<uint_t, N> &qubits,
                             const cmatrix_t &mat);
  void apply_diagonal_matrix(const std::array<uint_t, 1> &qubits,
                             const cmatrix_t &mat);

  //-----------------------------------------------------------------------
  // Error Messages
  //-----------------------------------------------------------------------
  void check_qubit(const uint_t qubit) const;

};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

template <class statematrix_t>
inline void to_json(json_t &js, const QubitMatrix<statematrix_t> &qmat) {
  js = qmat.json();
}

template <class statematrix_t>
json_t QubitMatrix<statematrix_t>::json() const {
  const int_t end = num_states_;
  // Initialize empty matrix
  const json_t zero = complex_t(0.0, 0.0);
  json_t js = json_t(num_states_, json_t(num_states_, zero));
  
  if (json_chop_threshold_ > 0) {
    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
    #ifdef _WIN32
      #pragma omp for
    #else
      #pragma omp for collapse(2)
    #endif
    for (int_t i=0; i < end; i++)
      for (int_t j=0; j < end; j++) {
        const auto val = statematrix_(i, j);
        if (std::abs(val.real()) > json_chop_threshold_)
          js[i][j][0] = val.real();
        if (std::abs(val.imag()) > json_chop_threshold_)
          js[i][j][1] = val.imag();
      }
    }
  } else {
    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
    #ifdef _WIN32
      #pragma omp for
    #else
      #pragma omp for collapse(2)
    #endif
    for (int_t i=0; i < end; i++)
      for (int_t j=0; j < end; j++) {
        const auto val = statematrix_(i, j);
        js[i][j][0] = val.real();
        js[i][j][1] = val.imag();
      }
    }
  }
  return js;
}

//------------------------------------------------------------------------------
// Error Handling
//------------------------------------------------------------------------------

template <class statematrix_t>
void QubitMatrix<statematrix_t>::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits_) {
    throw std::runtime_error(
      "QubitMatrix: qubit index " + std::to_string(qubit) + " > " +
      std::to_string(num_qubits_)
    );
  }
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <class statematrix_t>
QubitMatrix<statematrix_t>::QubitMatrix(size_t num_qubits) {
  set_num_qubits(num_qubits);
}

template <class statematrix_t>
QubitMatrix<statematrix_t>::QubitMatrix() : QubitMatrix(0) {}

template <class statematrix_t>
QubitMatrix<statematrix_t>::~QubitMatrix() = default;

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <class statematrix_t>
complex_t &QubitMatrix<statematrix_t>::operator()(uint_t row, uint_t col) {
  // TODO: Juan: No error checking out of DEBUG mode?
  // Error checking
  #ifdef DEBUG
  if (row > num_states_) {
    throw std::runtime_error("QubitMatrix: row index " + std::to_string(row) +
                             " > " + std::to_string(num_states_));
  }
  if (col > num_states_) {
    throw std::runtime_error("QubitMatrix: row index " + std::to_string(col) +
                             " > " + std::to_string(num_states_));
  }
  #endif
  return statematrix_(row, col);
}

template <class statematrix_t>
complex_t QubitMatrix<statematrix_t>::operator()(uint_t row, uint_t col) const {
  // Error checking
  #ifdef DEBUG
  if (row > num_states_) {
    throw std::runtime_error("QubitMatrix: row index " + std::to_string(row) +
                             " > " + std::to_string(num_states_));
  }
  if (col > num_states_) {
    throw std::runtime_error("QubitMatrix: row index " + std::to_string(col) +
                             " > " + std::to_string(num_states_));
  }
  #endif
  return statematrix_(row, col);
}

template <class statematrix_t>
complex_t &QubitMatrix<statematrix_t>::operator[](uint_t element) {
  // Error checking
  #ifdef DEBUG
  if (element > num_states_ * num_states_) {
    throw std::runtime_error("QubitMatrix: vector index " + std::to_string(element) +
                             " > " + std::to_string(num_states_));
  }
  #endif
  return statematrix_[element];
}

template <class statematrix_t>
complex_t QubitMatrix<statematrix_t>::operator[](uint_t element) const {
  // Error checking
  #ifdef DEBUG
  if (element > num_states_ * num_states_) {
    throw std::runtime_error("QubitMatrix: vector index " + std::to_string(element) +
                             " > " + std::to_string(num_states_));
  }
  #endif
  return statematrix_[element];
}

template <class statematrix_t>
cmatrix_t QubitMatrix<statematrix_t>::matrix() const {

  cmatrix_t ret(num_states_, num_states_);
  const int_t end = num_states_;

  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
  #ifdef _WIN32
    #pragma omp for
  #else
    #pragma omp for collapse(2)
  #endif
    for (int_t i=0; i < end; i++)
      for (int_t j=0; j < end; j++)
        ret(i, j) = statematrix_(i, j);
  } // end omp parallel
  return ret;
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <class statematrix_t>
void QubitMatrix<statematrix_t>::initialize() {
  // Set to n-qubit identity matrix
  statematrix_ = AER::Utils::Matrix::Identity(num_states_);
}

template <class statematrix_t>
void QubitMatrix<statematrix_t>::initialize(const cmatrix_t &mat) {
  if (num_states_ != mat.GetRows() || num_states_ != mat.GetColumns()) {
    throw std::runtime_error(
      "QubitMatrix<statematrix_t>::initialize input matrix is incorrect shape (" +
      std::to_string(num_states_) + "," + std::to_string(num_states_) + ")!=(" +
      std::to_string(mat.GetRows()) + "," + std::to_string(mat.GetColumns()) + ")."
    );
  }
  if (AER::Utils::is_unitary(mat, 1e-10) == false) {
    throw std::runtime_error(
      "QubitMatrix<statematrix_t>::initialize input matrix is not unitary."
    );
  }
  statematrix_ = mat;
}

template <class statematrix_t>
void QubitMatrix<statematrix_t>::set_num_qubits(size_t num_qubits) {
  num_qubits_ = num_qubits;
  num_states_ = 1ULL << num_qubits;
  statematrix_ = AER::Utils::Matrix::Identity(num_states_);
}

/*******************************************************************************
 *
 * CONFIG SETTINGS
 *
 ******************************************************************************/

template <class statematrix_t>
void QubitMatrix<statematrix_t>::set_json_chop_threshold(double threshold) {
  json_chop_threshold_ = threshold;
}

template <class statematrix_t>
void QubitMatrix<statematrix_t>::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

template <class statematrix_t>
void QubitMatrix<statematrix_t>::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

/*******************************************************************************
 *
 * LAMBDA FUNCTION TEMPLATES
 *
 ******************************************************************************/

// Single qubit
template <class statematrix_t>
template<typename Lambda>
void QubitMatrix<statematrix_t>::apply_matrix_lambda(const uint_t qubit,
                                                     const cmatrix_t &mat,
                                                     Lambda&& func) {

  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t ncols = num_states_; // number of columns
  const int_t end1 = num_states_;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(3)
#endif
    for (int_t col=0; col < ncols; col++)
      for (int_t k1 = 0; k1 < end1; k1 += step1)
        for (int_t k2 = 0; k2 < end2; k2++) {
          std::forward<Lambda>(func)(mat, col, k1, k2, end2);
        }
  }
}

// Static N-qubit
template <class statematrix_t>
template<size_t N, typename Lambda>
void QubitMatrix<statematrix_t>::apply_matrix_lambda(const std::array<uint_t, N> &qs,
                                                     const cmatrix_t &mat,
                                                     Lambda&& func) {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t ncols = num_states_; // number of columns
  const int_t end = num_states_ >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t col = 0; col < ncols; col++)
      for (int_t k = 0; k < end; k++) {
        // store entries touched by U
        const auto inds = indexes(qs, qubits_sorted, k);
        std::forward<Lambda>(func)(mat, col, inds);
      }
  }
}

// Dynamic N-qubit
template <class statematrix_t>
template<typename Lambda>
void QubitMatrix<statematrix_t>::apply_matrix_lambda(const std::vector<uint_t> &qubits,
                                                     const cmatrix_t &mat,
                                                     Lambda&& func) {

  const auto N = qubits.size();
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const int_t ncols = num_states_; // number of columns
  const int_t end = num_states_ >> N;
  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t col = 0; col < ncols; col++)
      for (int_t k = 0; k < end; k++) {
        // store entries touched by U
        const auto inds = indexes_dynamic(qubits, qubits_sorted, N, k);
        std::forward<Lambda>(func)(mat, col, inds);
      }
  }
}


/*******************************************************************************
 *
 * GENERAL MATRIX MULTIPLICATION
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Static N
//------------------------------------------------------------------------------


template <class statematrix_t>
template <size_t N>
void QubitMatrix<statematrix_t>::apply_matrix(const std::array<uint_t, N> &qs,
                                              const cmatrix_t &mat) {
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Lambda function for N-qubit matrix multiplication
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col,
                    const std::array<uint_t, 1ULL << N> &inds)->void {
    const uint_t dim = 1ULL << N;
    std::array<complex_t, dim> cache;
    for (size_t i = 0; i < dim; i++) {
      const auto ii = inds[i];
      cache[i] = statematrix_(ii, col);
      statematrix_(ii, col) = 0.;
    }
    // update state vector
    for (size_t i = 0; i < dim; i++)
      for (size_t j = 0; j < dim; j++)
        statematrix_(inds[i], col) += _mat(i, j) * cache[j];
  };
  // Use the lambda function
  apply_matrix_lambda(qs, mat, lambda);
}

template <class statematrix_t>
template <size_t N>
void QubitMatrix<statematrix_t>::apply_diagonal_matrix(const std::array<uint_t, N> &qs,
                                                       const cmatrix_t &diag) {

  // Error checking
  #ifdef DEBUG
  check_vector(diag, N);
  #endif

  // Lambda function for N-qubit matrix multiplication
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col,
                    const std::array<uint_t, 1ULL << N> &inds)->void {
    const uint_t dim = 1ULL << N;
    for (size_t i = 0; i < dim; i++) {
      statematrix_(inds[i], col) *= _mat(0, i);
    }
  };

  // Use the lambda function
  apply_matrix_lambda(qs, diag, lambda);
}


//------------------------------------------------------------------------------
// Single-qubit
//------------------------------------------------------------------------------

template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_matrix(const std::array<uint_t, 1> &qubits,
                                              const cmatrix_t &mat) {
  // Lambda function for single-qubit matrix multiplication
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col,
                    const int_t &k1, const int_t &k2, const int_t &end2)->void {
    const auto k = k1 | k2;
    const auto cache0 = statematrix_(k, col);
    const auto cache1 = statematrix_(k | end2, col);
    statematrix_(k, col) = _mat(0, 0) * cache0 + _mat(0, 1) * cache1;
    statematrix_(k | end2, col) = _mat(1, 0) * cache0 + _mat(1, 1) * cache1;
  };
  apply_matrix_lambda(qubits[0], mat, lambda);
}

template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_diagonal_matrix(const std::array<uint_t, 1> &qubits,
                                                       const cmatrix_t &diag) {
  // Lambda function for diagonal matrix multiplication
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col,
                    const int_t &k1, const int_t &k2, const int_t &end2)->void {
    const auto k = k1 | k2;
    statematrix_(k, col) *= _mat(0, 0);
    statematrix_(k | end2, col) *= _mat(0, 1);
  };
  apply_matrix_lambda(qubits[0], diag, lambda);
}


//------------------------------------------------------------------------------
// Dynamic N
//------------------------------------------------------------------------------

/* Generate the repeatative switch cases using the following python code:

```python
code = ""
for j in range(3, 15 + 1):
    code += "  case {0}: {{\n".format(j)
    code += "    std::array<uint_t, {0}> qubits_arr;\n".format(j)
    code += "    std::copy_n(qubits.begin(), {0}, qubits_arr.begin());\n".format(j)
    code += "    apply_matrix<{0}>(qubits_arr, mat);\n".format(j)
    code += "  }} break;\n".format(j)
print(code)
```
*/
template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_matrix(const std::vector<uint_t> &qubits,
                                              const cmatrix_t &mat) {
  // Special low N cases using faster static indexing
  switch (qubits.size()) {
    case 1:
      apply_matrix<1>(std::array<uint_t, 1>({{qubits[0]}}), mat);
      break;
    case 2:
      apply_matrix<2>(std::array<uint_t, 2>({{qubits[0], qubits[1]}}), mat);
      break;
    case 3: {
      std::array<uint_t, 3> qubits_arr;
      std::copy_n(qubits.begin(), 3, qubits_arr.begin());
      apply_matrix<3>(qubits_arr, mat);
    } break;
    case 4: {
      std::array<uint_t, 4> qubits_arr;
      std::copy_n(qubits.begin(), 4, qubits_arr.begin());
      apply_matrix<4>(qubits_arr, mat);
    } break;
    case 5: {
      std::array<uint_t, 5> qubits_arr;
      std::copy_n(qubits.begin(), 5, qubits_arr.begin());
      apply_matrix<5>(qubits_arr, mat);
    } break;
    case 6: {
      std::array<uint_t, 6> qubits_arr;
      std::copy_n(qubits.begin(), 6, qubits_arr.begin());
      apply_matrix<6>(qubits_arr, mat);
    } break;
    case 7: {
      std::array<uint_t, 7> qubits_arr;
      std::copy_n(qubits.begin(), 7, qubits_arr.begin());
      apply_matrix<7>(qubits_arr, mat);
    } break;
    case 8: {
      std::array<uint_t, 8> qubits_arr;
      std::copy_n(qubits.begin(), 8, qubits_arr.begin());
      apply_matrix<8>(qubits_arr, mat);
    } break;
    case 9: {
      std::array<uint_t, 9> qubits_arr;
      std::copy_n(qubits.begin(), 9, qubits_arr.begin());
      apply_matrix<9>(qubits_arr, mat);
    } break;
    case 10: {
      std::array<uint_t, 10> qubits_arr;
      std::copy_n(qubits.begin(), 10, qubits_arr.begin());
      apply_matrix<10>(qubits_arr, mat);
    } break;
    case 11: {
      std::array<uint_t, 11> qubits_arr;
      std::copy_n(qubits.begin(), 11, qubits_arr.begin());
      apply_matrix<11>(qubits_arr, mat);
    } break;
    case 12: {
      std::array<uint_t, 12> qubits_arr;
      std::copy_n(qubits.begin(), 12, qubits_arr.begin());
      apply_matrix<12>(qubits_arr, mat);
    } break;
    case 13: {
      std::array<uint_t, 13> qubits_arr;
      std::copy_n(qubits.begin(), 13, qubits_arr.begin());
      apply_matrix<13>(qubits_arr, mat);
    } break;
    case 14: {
      std::array<uint_t, 14> qubits_arr;
      std::copy_n(qubits.begin(), 14, qubits_arr.begin());
      apply_matrix<14>(qubits_arr, mat);
    } break;
    case 15: {
      std::array<uint_t, 15> qubits_arr;
      std::copy_n(qubits.begin(), 15, qubits_arr.begin());
      apply_matrix<15>(qubits_arr, mat);
    } break;
    default: {
      throw std::runtime_error("QubitMatrix::apply_matrix: " +
        std::to_string(qubits.size()) + "-qubit matrix is too large to apply.");
      break;
    }
  } // end switch
}

/* Generate the repeatative switch cases using the following python code:

```python
code = ""
for j in range(3, 15 + 1):
    code += "  case {0}: {{\n".format(j)
    code += "    std::array<uint_t, {0}> qubits_arr;\n".format(j)
    code += "    std::copy_n(qubits.begin(), {0}, qubits_arr.begin());\n".format(j)
    code += "    apply_diagonal_matrix<{0}>(qubits_arr, mat);\n".format(j)
    code += "  }} break;\n".format(j)
print(code)
```
*/
template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_diagonal_matrix(const std::vector<uint_t> &qubits,
                                                       const cmatrix_t &mat) {
  // Special low N cases using faster static indexing
  //TODO: Juan: Let's see if vector with an special allocator can replace this
  switch (qubits.size()) {
    case 1:
      apply_diagonal_matrix<1>(std::array<uint_t, 1>({{qubits[0]}}), mat);
      break;
    case 2:
      apply_diagonal_matrix<2>(std::array<uint_t, 2>({{qubits[0], qubits[1]}}), mat);
      break;
      case 3: {
      std::array<uint_t, 3> qubits_arr;
      std::copy_n(qubits.begin(), 3, qubits_arr.begin());
      apply_diagonal_matrix<3>(qubits_arr, mat);
    } break;
    case 4: {
      std::array<uint_t, 4> qubits_arr;
      std::copy_n(qubits.begin(), 4, qubits_arr.begin());
      apply_diagonal_matrix<4>(qubits_arr, mat);
    } break;
    case 5: {
      std::array<uint_t, 5> qubits_arr;
      std::copy_n(qubits.begin(), 5, qubits_arr.begin());
      apply_diagonal_matrix<5>(qubits_arr, mat);
    } break;
    case 6: {
      std::array<uint_t, 6> qubits_arr;
      std::copy_n(qubits.begin(), 6, qubits_arr.begin());
      apply_diagonal_matrix<6>(qubits_arr, mat);
    } break;
    case 7: {
      std::array<uint_t, 7> qubits_arr;
      std::copy_n(qubits.begin(), 7, qubits_arr.begin());
      apply_diagonal_matrix<7>(qubits_arr, mat);
    } break;
    case 8: {
      std::array<uint_t, 8> qubits_arr;
      std::copy_n(qubits.begin(), 8, qubits_arr.begin());
      apply_diagonal_matrix<8>(qubits_arr, mat);
    } break;
    case 9: {
      std::array<uint_t, 9> qubits_arr;
      std::copy_n(qubits.begin(), 9, qubits_arr.begin());
      apply_diagonal_matrix<9>(qubits_arr, mat);
    } break;
    case 10: {
      std::array<uint_t, 10> qubits_arr;
      std::copy_n(qubits.begin(), 10, qubits_arr.begin());
      apply_diagonal_matrix<10>(qubits_arr, mat);
    } break;
    case 11: {
      std::array<uint_t, 11> qubits_arr;
      std::copy_n(qubits.begin(), 11, qubits_arr.begin());
      apply_diagonal_matrix<11>(qubits_arr, mat);
    } break;
    case 12: {
      std::array<uint_t, 12> qubits_arr;
      std::copy_n(qubits.begin(), 12, qubits_arr.begin());
      apply_diagonal_matrix<12>(qubits_arr, mat);
    } break;
    case 13: {
      std::array<uint_t, 13> qubits_arr;
      std::copy_n(qubits.begin(), 13, qubits_arr.begin());
      apply_diagonal_matrix<13>(qubits_arr, mat);
    } break;
    case 14: {
      std::array<uint_t, 14> qubits_arr;
      std::copy_n(qubits.begin(), 14, qubits_arr.begin());
      apply_diagonal_matrix<14>(qubits_arr, mat);
    } break;
    case 15: {
      std::array<uint_t, 15> qubits_arr;
      std::copy_n(qubits.begin(), 15, qubits_arr.begin());
      apply_diagonal_matrix<15>(qubits_arr, mat);
    } break;
    default: {
      std::stringstream ss;
      ss << "QubitMatrix::apply_matrix: " << qubits.size();
      ss << "-qubit matrix is too large to apply.";
      throw std::runtime_error(ss.str());
      break;
    }
  } // end switch
}


/*******************************************************************************
 *
 * APPLY OPTIMIZED GATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Single-qubit gates
//------------------------------------------------------------------------------

template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_x(const uint_t qubit) {
  // Lambda function for optimized Pauli-X gate
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col, 
                    const int_t &k1, const int_t &k2, const int_t &end2)->void {
    (void)_mat; // unused
    const auto i0 = k1 | k2;
    const auto i1 = i0 | end2;
    const complex_t cache = statematrix_(i0, col);
    statematrix_(i0, col) = statematrix_(i1, col); // mat(0,1)
    statematrix_(i1, col) = cache;    // mat(1,0)
  };
  apply_matrix_lambda(qubit, {}, lambda);
}

template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_y(const uint_t qubit) {
  // Lambda function for optimized Pauli-Y gate
  const complex_t I(0., 1.);
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col, 
                    const int_t &k1, const int_t &k2, const int_t &end2)->void {
    (void)_mat; // unused
    const auto i0 = k1 | k2;
    const auto i1 = i0 | end2;
    const complex_t cache = statematrix_(i0, col);
    statematrix_(i0, col) = -I * statematrix_(i1, col); // mat(0,1)
    statematrix_(i1, col) = I * cache;     // mat(1,0)
  };
  apply_matrix_lambda(qubit, {}, lambda);
}

template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_z(const uint_t qubit) {
  // Lambda function for optimized Pauli-Z gate
  const complex_t minus_one(-1.0, 0.0);
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col, 
                    const int_t &k1, const int_t &k2, const int_t &end2)->void {
    (void)_mat; // unused
    statematrix_(k1 | k2 | end2, col) *= minus_one;
  };
  apply_matrix_lambda(qubit, {}, lambda);
}

//------------------------------------------------------------------------------
// Two-qubit gates
//------------------------------------------------------------------------------
template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_cnot(const uint_t qubit_ctrl, const uint_t qubit_trgt) {
  // Lambda function for CNOT gate
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col,
                    const std::array<uint_t, 1ULL << 2> &inds)->void {
    (void)_mat; //unused
    const complex_t cache = statematrix_(inds[3], col);
    statematrix_(inds[3], col) = statematrix_(inds[1], col);
    statematrix_(inds[1], col) = cache;
  };
  // Use the lambda function
  apply_matrix_lambda(std::array<uint_t, 2>({{qubit_ctrl, qubit_trgt}}), {}, lambda);
}

template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_swap(const uint_t qubit0, const uint_t qubit1) {
  // Lambda function for SWAP gate
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col,
                    const std::array<uint_t, 1ULL << 2> &inds)->void {
    (void)_mat; //unused
    const complex_t cache = statematrix_(inds[2], col);
      statematrix_(inds[2], col) = statematrix_(inds[1], col);
      statematrix_(inds[1], col) = cache;
  };
  // Use the lambda function
  apply_matrix_lambda(std::array<uint_t, 2>({{qubit0, qubit1}}), {}, lambda);
}

template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_cz(const uint_t qubit_ctrl, const uint_t qubit_trgt) {

  // Lambda function for CZ gate
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col,
                    const std::array<uint_t, 1ULL << 2> &inds)->void {
    (void)_mat; //unused
    statematrix_(inds[3], col) *= -1.;
  };
  // Use the lambda function
  apply_matrix_lambda(std::array<uint_t, 2>({{qubit_ctrl, qubit_trgt}}), {}, lambda);
}

//------------------------------------------------------------------------------
// Three-qubit gates
//------------------------------------------------------------------------------
template <class statematrix_t>
void QubitMatrix<statematrix_t>::apply_toffoli(const uint_t qubit_ctrl0,
                                const uint_t qubit_ctrl1,
                                const uint_t qubit_trgt) {
  // Lambda function for Toffoli gate
  auto lambda = [&](const cmatrix_t &_mat, const int_t &col,
                    const std::array<uint_t, 1ULL << 3> &inds)->void {
    (void)_mat; //unused
    const complex_t cache = statematrix_(inds[7], col);
    statematrix_(inds[7], col) = statematrix_(inds[3], col);
    statematrix_(inds[3], col) = cache;
  };
  // Use the lambda function
  std::array<uint_t, 3> qubits = {{qubit_ctrl0, qubit_ctrl1, qubit_trgt}};
  apply_matrix_lambda(qubits, {}, lambda);
}

//------------------------------------------------------------------------------
} // end namespace QM
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <class statematrix_t>
inline std::ostream &operator<<(std::ostream &out, const QM::QubitMatrix<statematrix_t>&qmat) {
  out << qmat.matrix();
  return out;
}

//------------------------------------------------------------------------------
#endif // end module

