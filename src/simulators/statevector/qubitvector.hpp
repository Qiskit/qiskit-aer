/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _qv_qubit_vector_hpp_
#define _qv_qubit_vector_hpp_

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "indexing.hpp"
#include "framework/json.hpp"

namespace QV {

// Type aliases
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;

//============================================================================
// QubitVector class
//============================================================================

// Template class for qubit vector.
// The arguement of the template must have an operator[] access method.
// The following methods may also need to be template specialized:
//   * set_num_qubits(size_t)
//   * initialize()
//   * initialize_from_vector(cvector_t)
// If the template argument does not have these methods then template
// specialization must be used to override the default implementations.

template <class data_t = complex_t*>
class QubitVector {

public:

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  QubitVector();
  explicit QubitVector(size_t num_qubits);
  virtual ~QubitVector();
  QubitVector(const QubitVector& obj) = delete;
  QubitVector &operator=(const QubitVector& obj) = delete;

  //-----------------------------------------------------------------------
  // Data access
  //-----------------------------------------------------------------------

  // Element access
  complex_t &operator[](uint_t element);
  complex_t operator[](uint_t element) const;

  // Returns a reference to the underlying data_t data class
  data_t &data() {return data_;}

  // Returns a copy of the underlying data_t data class
  data_t data() const {return data_;}

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Set the size of the vector in terms of qubit number
  virtual void set_num_qubits(size_t num_qubits);

  // Returns the number of qubits for the current vector
  uint_t num_qubits() const {return num_qubits_;}

  // Returns the size of the underlying n-qubit vector
  uint_t size() const {return size_;}

  // Returns a copy of the underlying data_t data as a complex vector
  cvector_t vector() const;

  // Return JSON serialization of QubitVector;
  json_t json() const;

  // Set all entries in the vector to 0.
  void zero();

  //-----------------------------------------------------------------------
  // Check point operations
  //-----------------------------------------------------------------------

  // Create a checkpoint of the current state
  void checkpoint();

  // Revert to the checkpoint
  void revert(bool keep);

  // Compute the inner product of current state with checkpoint state
  complex_t inner_product() const;

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // If the length of the data vector does not match the number of qubits
  // an exception is raised.
  void initialize_from_vector(const cvector_t &data);

  // Initializes the vector to a custom initial state.
  // If num_states does not match the number of qubits an exception is raised.
  void initialize_from_data(const data_t &data, const size_t num_states);

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const reg_t &qubits, const cvector_t &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const reg_t &qubits, const cvector_t &mat);
  
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
  // Z-measurement outcome probabilities
  //-----------------------------------------------------------------------

  // Return the Z-basis measurement outcome probability P(outcome) for
  // outcome in [0, 2^num_qubits - 1]
  double probability(const uint_t outcome) const;

  // Return the probabilities for all measurement outcomes in the current vector
  // This is equivalent to returning a new vector with  new[i]=|orig[i]|^2.
  // Eg. For 2-qubits this is [P(00), P(01), P(010), P(11)]
  rvector_t probabilities() const;

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  rvector_t probabilities(const std::vector<uint_t> &qubits) const;

  // Return M sampled outcomes for Z-basis measurement of all qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  std::vector<uint_t> sample_measure(const std::vector<double> &rnds) const;

  //-----------------------------------------------------------------------
  // Norms
  //-----------------------------------------------------------------------
  
  // Returns the norm of the current vector
  double norm() const;

  // These functions return the norm <psi|A^dagger.A|psi> obtained by
  // applying a matrix A to the vector. It is equivalent to returning the
  // expectation value of A^\dagger A, and could probably be removed because
  // of this.

  // Return the norm for of the vector obtained after apply the N-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  double norm(const std::vector<uint_t> &qubits, const cvector_t &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const std::vector<uint_t> &qubits, const cvector_t &mat) const;

  //-----------------------------------------------------------------------
  // JSON configuration settings
  //-----------------------------------------------------------------------

  // Set the threshold for chopping values to 0 in JSON
  void set_json_chop_threshold(double threshold);

  // Set the threshold for chopping values to 0 in JSON
  double get_json_chop_threshold() {return json_chop_threshold_;}

  //-----------------------------------------------------------------------
  // OpenMP configuration settings
  //-----------------------------------------------------------------------

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
  // Optimization configuration settings
  //-----------------------------------------------------------------------

  // Enable sorted qubit matrix gate optimization (Default disabled)
  inline void enable_gate_opt() {gate_opt_ = true;}

  // Disable sorted qubit matrix gate optimization
  inline void disable_gate_opt() {gate_opt_ = false;}

  // Set the sample_measure index size
  inline void set_sample_measure_index_size(int n) {sample_measure_index_size_ = n;}

  // Get the sample_measure index size
  inline int get_sample_measure_index_size() {return sample_measure_index_size_;}

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  size_t num_qubits_;
  size_t size_;
  data_t data_;
  data_t checkpoint_;

  //-----------------------------------------------------------------------
  // Config settings
  //----------------------------------------------------------------------- 
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 13;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 10; // Sample measure indexing qubit size
  bool gate_opt_ = false;      // enable large-qubit optimized gates
  double json_chop_threshold_ = 0;  // Threshold for choping small values
                                    // in JSON serialization

  //-----------------------------------------------------------------------
  // Error Messages
  //-----------------------------------------------------------------------

  void check_qubit(const uint_t qubit) const;
  void check_vector(const cvector_t &diag, uint_t nqubits) const;
  void check_matrix(const cvector_t &mat, uint_t nqubits) const;
  void check_dimension(const QubitVector &qv) const;
  void check_checkpoint() const;

  //-----------------------------------------------------------------------
  // State update with Lambda functions
  //-----------------------------------------------------------------------
  // These functions loop through the indexes of the qubitvector data and
  // apply a lambda function to each entry.

  // Apply a lambda function to all entries of the statevector.
  // The function signature should be:
  // [&](const int_t &k)->void
  // where k is the index of the vector
  template <typename Lambda>
  void apply_lambda(Lambda&& func);

  // Apply a single-qubit lambda function to all blocks of the statevector
  // for the given qubit. The function signature should be:
  // [&](const int_t &k1, const int_t &k2)->void
  // where (k1, k2) are the 0 and 1 indexes for each qubit block
  template <typename Lambda>
  void apply_lambda(Lambda&& func, const uint_t qubit);

  // Apply a N-qubit lambda function to all blocks of the statevector
  // for the given qubits. The function signature should be:
  // [&](const areg_t<1ULL<<N> &inds)->void
  // where inds are the 2 ** N indexes for each N-qubit block returned by
  // the static_indexes function
  template <size_t N, typename Lambda>
  void apply_lambda(Lambda&& func, const areg_t<N> &qubits);

  // Apply a N-qubit lambda function to all blocks of the statevector
  // for the given qubits. The function signature should be:
  // [&](const reg_t &inds)->void
  // where inds are the 2 ** N indexes for each N-qubit block returned by
  // the dynamic_indexes function
  template <typename Lambda>
  void apply_lambda(Lambda&& func, const reg_t &qubits);

  //-----------------------------------------------------------------------
  // State matrix update with Lambda functions
  //-----------------------------------------------------------------------
  // These functions loop through the indexes of the qubitvector data and
  // apply a lambda function taking a vector matrix argument to each entry.

  // Apply a single-qubit matrix lambda function to all blocks of the
  // statevector for the given qubit. The function signature should be:
  // [&](const int_t &k1, const int_t &k2, const cvector_t &m)->void
  // where (k1, k2) are the 0 and 1 indexes for each qubit block and
  // m is a vectorized complex matrix.
  template <typename Lambda>
  void apply_matrix_lambda(Lambda&& func,
                           const uint_t qubit,
                           const cvector_t &mat);

  // Apply a N-qubit matrix lambda function to all blocks of the statevector
  // for the given qubits. The function signature should be:
  // [&](const areg_t<1ULL<<N> &inds, const cvector_t &m)->void
  // where inds are the 2 ** N indexes for each N-qubit block returned by
  // the static_indexes function and m is a vectorized complex matrix.
  template <size_t N, typename Lambda>
  void apply_matrix_lambda(Lambda&& func,
                           const areg_t<N> &qubits,
                           const cvector_t &mat);

  // Apply a N-qubit matrix lambda function to all blocks of the statevector
  // for the given qubits. The function signature should be:
  // [&](const reg_t &inds, const cvector_t &m)->void
  // where inds are the 2 ** N indexes for each N-qubit block returned by
  // the dynamic_indexes function and m is a vectorized complex matrix.
  template <typename Lambda>
  void apply_matrix_lambda(Lambda&& func,
                           const reg_t &qubits,
                           const cvector_t &mat);

  //-----------------------------------------------------------------------
  // State reduction functions with Lambda functions
  //-----------------------------------------------------------------------
  // These functions loop through the indexes of the qubitvector data and
  // apply a lambda function to each entry that appplies a reduction on
  // two doubles (val_re, val_im) and returns the complex result
  // complex_t(val_re, val_im)

  // Apply a complex reduction lambda function to all entries of the
  // statevector and return the complex result.
  // The function signature should be:
  // [&](const int_t &k, double &val_re, double &val_im)->void
  // where k is the index of the vector, val_re and val_im are the doubles
  // to store the reduction.
  // Returns complex_t(val_re, val_im)
  template <typename Lambda>
  complex_t apply_reduction_lambda(Lambda&& func) const;

  // Apply a 1-qubit complex reduction  lambda function to all blocks of the
  // statevector for the given qubit. The function signature should be:
  // [&](const int_t &k1, const int_t &k2, double &val_re, double &val_im)->void
  // where (k1, k2) are the 0 and 1 indexes for each qubit block
  // val_re and val_im are the doubles to store the reduction.
  // Returns complex_t(val_re, val_im)
  template <typename Lambda>
  complex_t apply_reduction_lambda(Lambda&& func,
                                   const uint_t qubit) const;

  // Apply a N-qubit complex reduction lambda function to all blocks of the
  // statevector for the given qubits. The function signature should be:
  // [&](const areg_t<1ULL<<N> &inds, double &val_re, double &val_im)->void
  // where inds are the 2 ** N indexes for each N-qubit block returned by
  // the static_indexes function, val_re and val_im are the doubles to store
  // the reduction.
  // Returns complex_t(val_re, val_im)
  template <size_t N, typename Lambda>
  complex_t apply_reduction_lambda(Lambda&& func,
                                   const areg_t<N> &qubits) const;

  // Apply a N-qubit complex reduction lambda function to all blocks of the
  // statevector for the given qubits. The function signature should be:
  // [&](const reg_t &inds, double &val_re, double &val_im)->void
  // where inds are the 2 ** N indexes for each N-qubit block returned by
  // the static_indexes function, val_re and val_im are the doubles to store
  // the reduction.
  // Returns complex_t(val_re, val_im)
  template <typename Lambda>
  complex_t apply_reduction_lambda(Lambda&& func,
                                   const reg_t &qubits) const;

  //-----------------------------------------------------------------------
  // State matrix reduction functions with Lambda functions
  //-----------------------------------------------------------------------
  // These functions loop through the indexes of the qubitvector data and
  // apply a lambda function to each entry that also applies a reduction
  // and also applies a matrix. Unlike apply_matrix_lambda these should not
  // update the state.

  // Apply a 1-qubit complex matrix reduction lambda function to all blocks of the
  // statevector for the given qubit. The function signature should be:
  // [&](const int_t &k1, const int_t &k2, const cvector_t &m,
  //     double &val_re, double &val_im)->void
  // where (k1, k2) are the 0 and 1 indexes for each qubit block, m is a
  // vectorized complex matrix, val_re and val_im are the doubles to store
  // the reduction.
  // Returns complex_t(val_re, val_im)
  template <typename Lambda>
  complex_t apply_matrix_reduction_lambda(Lambda&& func,
                                          const uint_t qubit,
                                          const cvector_t &mat) const;

  // Apply a N-qubit complex matrix reduction lambda function to all blocks of the
  // statevector for the given qubits. The function signature should be:
  // [&](const areg_t<1ULL<<N> &inds, const cvector_t &m,
  //     double &val_re, double &val_im)->void
  // where inds are the 2 ** N indexes for each N-qubit block returned by
  // the static_indexes function, m is a vectorized complex matrix,
  // val_re and val_im are the doubles to store the reduction.
  // Returns complex_t(val_re, val_im)
  template <size_t N, typename Lambda>
  complex_t apply_matrix_reduction_lambda(Lambda&& func,
                                          const areg_t<N> &qubits,
                                          const cvector_t &mat) const;

  // Apply a N-qubit complex matrix reduction lambda function to all blocks of the
  // statevector for the given qubits. The function signature should be:
  // [&](const reg_t &inds, const cvector_t &m,
  //     double &val_re, double &val_im)->void
  // where inds are the 2 ** N indexes for each N-qubit block returned by
  // the static_indexes function, m is a vectorized complex matrix,
  // val_re and val_im are the doubles to store the reduction.
  // Returns complex_t(val_re, val_im)
  template <typename Lambda>
  complex_t apply_matrix_reduction_lambda(Lambda&& func,
                                          const reg_t &qubits,
                                          const cvector_t &mat) const;

  //-----------------------------------------------------------------------
  // Static size matrix multiplication
  //-----------------------------------------------------------------------
  
  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  template <size_t N>
  void apply_matrix(const areg_t<N> &qubits, const cvector_t &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  template <size_t N>
  void apply_diagonal_matrix(const areg_t<N> &qubits,
                             const cvector_t &mat);

  // Swap pairs of indicies in the underlying vector
  template <size_t N, size_t M>
  void apply_permutation_matrix(const areg_t<N> &qubits,
                                const std::array<std::pair<uint_t, uint_t>, M> &pairs);
  
  template <size_t N>
  void apply_permutation_matrix(const areg_t<N> &qubits,
                                const std::vector<std::pair<uint_t, uint_t>> &pairs);

  //-----------------------------------------------------------------------
  // Optimized matrix multiplication
  //-----------------------------------------------------------------------
  
  // Optimized implementations
  void apply_matrix(const areg_t<1> &qubits, const cvector_t &mat);
  void apply_matrix(const areg_t<2> &qubits, const cvector_t &mat);
  void apply_matrix(const areg_t<3> &qubits, const cvector_t &mat);
  void apply_matrix(const areg_t<4> &qubits, const cvector_t &mat);
  void apply_matrix(const areg_t<5> &qubits, const cvector_t &mat);
  void apply_matrix(const areg_t<6> &qubits, const cvector_t &mat);

  // Permute an N-qubit vectorized matrix to match a reordering of qubits
  template <size_t N>
  cvector_t sort_matrix(const areg_t<N> &src,
                        const areg_t<N> &sorted,
                        const cvector_t &mat) const;

  // Swap cols and rows of vectorized matrix
  void swap_cols_and_rows(const uint_t idx1, const uint_t idx2,
                          cvector_t &mat, uint_t dim) const;

  //-----------------------------------------------------------------------
  // Optimized diagonal matrix multiplication
  //-----------------------------------------------------------------------
  
  void apply_diagonal_matrix(const areg_t<1> &qubits,
                             const cvector_t &mat);

  //-----------------------------------------------------------------------
  // Probabilities helper functions
  //-----------------------------------------------------------------------

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  template <size_t N>
  rvector_t probabilities(const areg_t<N> &qubits) const;
  rvector_t probabilities(const areg_t<1> &qubits) const;

  //-----------------------------------------------------------------------
  // Norm helper functions
  //-----------------------------------------------------------------------

  // Return the norm for of the vector obtained after apply the N-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  template <size_t N>
  double norm(const areg_t<N> &qubits, const cvector_t &mat) const;
  double norm(const areg_t<1> &qubits, const cvector_t &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  template <size_t N>
  double norm_diagonal(const areg_t<N> &qubits, const cvector_t &mat) const;
  double norm_diagonal(const areg_t<1> &qubits, const cvector_t &mat) const;

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
inline void to_json(json_t &js, const QubitVector<data_t> &qv) {
  js = qv.json();
}

template <class data_t>
json_t QubitVector<data_t>::json() const {
  const int_t end = size_;
  const json_t ZERO = complex_t(0.0, 0.0);
  json_t js = json_t(size_, ZERO);
  
  if (json_chop_threshold_ > 0) {
    #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    for (int_t j=0; j < end; j++) {
      if (std::abs(data_[j].real()) > json_chop_threshold_)
        js[j][0] = data_[j].real();
      if (std::abs(data_[j].imag()) > json_chop_threshold_)
        js[j][1] = data_[j].imag();
    }
  } else {
    #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    for (int_t j=0; j < end; j++) {
      js[j][0] = data_[j].real();
      js[j][1] = data_[j].imag();
    }
  }
  return js;
}

//------------------------------------------------------------------------------
// Error Handling
//------------------------------------------------------------------------------

template <class data_t>
void QubitVector<data_t>::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits_) {
    std::stringstream ss;
    ss << "QubitVector: qubit index " << qubit << " > " << num_qubits_;
    throw std::runtime_error(ss.str());
  }
}

template <class data_t>
void QubitVector<data_t>::check_matrix(const cvector_t &vec, uint_t nqubits) const {
  const size_t dim = 1ULL << nqubits;
  const auto sz = vec.size();
  if (sz != dim * dim) {
    std::stringstream ss;
    ss << "QubitVector: vector size is " << sz << " != " << (dim * dim);
    throw std::runtime_error(ss.str());
  }
}

template <class data_t>
void QubitVector<data_t>::check_vector(const cvector_t &vec, uint_t nqubits) const {
  const size_t dim = 1ULL << nqubits;
  const auto sz = vec.size();
  if (sz != dim) {
    std::stringstream ss;
    ss << "QubitVector: vector size is " << sz << " != " << dim;
    throw std::runtime_error(ss.str());
  }
}

template <class data_t>
void QubitVector<data_t>::check_dimension(const QubitVector &qv) const {
  if (size_ != qv.size_) {
    std::stringstream ss;
    ss << "QubitVector: vectors are different shape ";
    ss << size_ << " != " << qv.size_;
    throw std::runtime_error(ss.str());
  }
}

template <class data_t>
void QubitVector<data_t>::check_checkpoint() const {
  if (!checkpoint_) {
    throw std::runtime_error("QubitVector: checkpoint must exist for inner_product() or revert()");
  }
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <class data_t>
QubitVector<data_t>::QubitVector(size_t num_qubits) : num_qubits_(0), data_(0), checkpoint_(0){
  set_num_qubits(num_qubits);
}

template <class data_t>
QubitVector<data_t>::QubitVector() : QubitVector(0) {}

template <class data_t>
QubitVector<data_t>::~QubitVector() {
  if (data_)
    free(data_);

  if (checkpoint_)
    free(checkpoint_);
}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <class data_t>
complex_t &QubitVector<data_t>::operator[](uint_t element) {
  // Error checking
  #ifdef DEBUG
  if (element > size_) {
    std::stringstream ss;
    ss << "QubitVector: vector index " << element << " > " << size_;
    throw std::runtime_error(ss.str());
  }
  #endif
  return data_[element];
}

template <class data_t>
complex_t QubitVector<data_t>::operator[](uint_t element) const {
  // Error checking
  #ifdef DEBUG
  if (element > size_) {
    std::stringstream ss;
    ss << "QubitVector: vector index " << element << " > " << size_;
    throw std::runtime_error(ss.str());
  }
  #endif
  return data_[element];
}

template <class data_t>
cvector_t QubitVector<data_t>::vector() const {
  cvector_t ret(size_, 0.);
  const int_t end = size_;
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < end; j++) {
    ret[j] = data_[j];
  }
  return ret;
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <class data_t>
void QubitVector<data_t>::zero() {
  const int_t end = size_;    // end for k loop

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < end; ++k) {
    data_[k] = 0.0;
  }
}

template <class data_t>
void QubitVector<data_t>::set_num_qubits(size_t num_qubits) {
  num_qubits_ = num_qubits;
  size_ = 1ULL << num_qubits;

  // Free any currently assigned memory
  if (data_)
    free(data_);

  if (checkpoint_) {
    free(checkpoint_);
    checkpoint_ = 0;
  }

  // Allocate memory for new vector
  data_ = reinterpret_cast<complex_t*>(malloc(sizeof(complex_t) * size_));
}


template <class data_t>
void QubitVector<data_t>::checkpoint() {
  if (!checkpoint_)
    checkpoint_ = reinterpret_cast<complex_t*>(malloc(sizeof(complex_t) * size_));

  const int_t end = size_;    // end for k loop
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < end; ++k)
    checkpoint_[k] = data_[k];
}


template <class data_t>
void QubitVector<data_t>::revert(bool keep) {

  #ifdef DEBUG
  check_checkpoint();
  #endif

  const int_t end = size_;    // end for k loop
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < end; ++k)
    data_[k] = checkpoint_[k];

  if (!keep) {
    free(checkpoint_);
    checkpoint_ = 0;
  }
}

template <class data_t>
complex_t QubitVector<data_t>::inner_product() const {

  #ifdef DEBUG
  check_checkpoint();
  #endif
  // Lambda function for inner product with checkpoint state
  auto lambda = [&](int_t k, double &val_re, double &val_im)->void {
    const complex_t z = data_[k] * std::conj(checkpoint_[k]);
    val_re += std::real(z);
    val_im += std::imag(z);
  };
  return apply_reduction_lambda(lambda);
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

template <class data_t>
void QubitVector<data_t>::initialize() {
  zero();
  data_[0] = 1.;
}

template <class data_t>
void QubitVector<data_t>::initialize_from_vector(const cvector_t &statevec) {
  if (size_ != statevec.size()) {
    std::stringstream ss;
    ss << "QubitVector::initialize input vector is incorrect length (";
    ss << size_ << "!=" << statevec.size() << ")";
    throw std::runtime_error(ss.str());
  }

  const int_t end = size_;    // end for k loop

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < end; ++k)
    data_[k] = statevec[k];
}

template <class data_t>
void QubitVector<data_t>::initialize_from_data(const data_t &statevec, const size_t num_states) {
  if (size_ != num_states) {
    std::stringstream ss;
    ss << "QubitVector::initialize input vector is incorrect length (";
    ss << size_ << "!=" << num_states << ")";
    throw std::runtime_error(ss.str());
  }

  const int_t end = size_;    // end for k loop

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < end; ++k)
    data_[k] = statevec[k];
}


/*******************************************************************************
 *
 * CONFIG SETTINGS
 *
 ******************************************************************************/

template <class data_t>
void QubitVector<data_t>::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

template <class data_t>
void QubitVector<data_t>::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

template <class data_t>
void QubitVector<data_t>::set_json_chop_threshold(double threshold) {
  json_chop_threshold_ = threshold;
}

/*******************************************************************************
 *
 * LAMBDA FUNCTION TEMPLATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// General lambda
//------------------------------------------------------------------------------

// Static N-qubit
template <class data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_lambda(Lambda&& func) {
  const int_t end = size_;
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      std::forward<Lambda>(func)(k);
    }
  }
}

// Single qubit
template <class data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_lambda(Lambda&& func,
                                       const uint_t qubit) {
  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t end1 = size_;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        std::forward<Lambda>(func)(k1, k2);
      }
  }
}

// Static N-qubit
template <class data_t>
template<size_t N, typename Lambda>
void QubitVector<data_t>::apply_lambda(Lambda&& func,
                                       const areg_t<N> &qs) {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = size_ >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_static(qs, qubits_sorted, k);
      std::forward<Lambda>(func)(inds);
    }
  }
}

// Dynamic N-qubit
template <class data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_lambda(Lambda&& func,
                                       const reg_t &qubits) {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const auto N = qubits.size();
  const int_t end = size_ >> N;
  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qubits, qubits_sorted, N, k);
      std::forward<Lambda>(func)(inds);
    }
  }
}

//------------------------------------------------------------------------------
// Matrix Lambda
//------------------------------------------------------------------------------

// Single qubit
template <class data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_matrix_lambda(Lambda&& func,
                                              const uint_t qubit,
                                              const cvector_t &mat) {
  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t end1 = size_;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        std::forward<Lambda>(func)(k1, k2, mat);
      }
  }
}

// Static N-qubit
template <class data_t>
template<size_t N, typename Lambda>
void QubitVector<data_t>::apply_matrix_lambda(Lambda&& func,
                                              const areg_t<N> &qubits,
                                              const cvector_t &mat) {
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const int_t end = size_ >> N;
  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_static(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, mat);
    }
  }
}

// Dynamic N-qubit
template <class data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_matrix_lambda(Lambda&& func,
                                              const reg_t &qubits,
                                              const cvector_t &mat) {
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const auto N = qubits.size();
  const int_t end = size_ >> N;
  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qubits, qubits_sorted, N, k);
      std::forward<Lambda>(func)(inds, mat);
    }
  }
}

//------------------------------------------------------------------------------
// Reduction Lambda
//------------------------------------------------------------------------------

template <class data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_reduction_lambda(Lambda &&func) const {
  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
  const int_t end = size_;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
        std::forward<Lambda>(func)(k, val_re, val_im);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

// Single-qubit
template <class data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_reduction_lambda(Lambda &&func,
                                                      const uint_t qubit) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t end1 = size_;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        std::forward<Lambda>(func)(k1, k2, val_re, val_im);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

// Static N-qubit
template <class data_t>
template<size_t N, typename Lambda>
complex_t QubitVector<data_t>::apply_reduction_lambda(Lambda&& func,
                                                      const areg_t<N> &qs) const {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = size_ >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_static(qs, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, val_re, val_im);
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

// Dynamic N-qubit
template <class data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_reduction_lambda(Lambda&& func,
                                                      const reg_t &qs) const {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const size_t N =  qs.size();
  const int_t end = size_ >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qs, qubits_sorted, N, k);
      std::forward<Lambda>(func)(inds, val_re, val_im);
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

//------------------------------------------------------------------------------
// Matrix and Reduction Lambda
//------------------------------------------------------------------------------

template <class data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_matrix_reduction_lambda(Lambda &&func,
                                                             const uint_t qubit,
                                                             const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t end1 = size_;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        std::forward<Lambda>(func)(k1, k2, mat, val_re, val_im);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

// Static N-qubit
template <class data_t>
template<size_t N, typename Lambda>
complex_t QubitVector<data_t>::apply_matrix_reduction_lambda(Lambda&& func,
                                                             const areg_t<N> &qs,
                                                             const cvector_t &mat) const {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = size_ >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_static(qs, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, mat, val_re, val_im);
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

// Dynamic N-qubit
template <class data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_matrix_reduction_lambda(Lambda&& func,
                                                             const reg_t &qubits,
                                                             const cvector_t &mat) const {

  const auto N = qubits.size();
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const int_t end = size_ >> N;
  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qubits, qubits_sorted, N, k);
      std::forward<Lambda>(func)(inds, mat, val_re, val_im);
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Static N
//------------------------------------------------------------------------------

template <class data_t>
template <size_t N>
void QubitVector<data_t>::apply_matrix(const areg_t<N> &qs,
                                       const cvector_t &mat) {
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Lambda function for N-qubit matrix multiplication
  auto lambda = [&](const areg_t<1ULL << N> &inds,
                    const cvector_t &_mat)->void {
    const uint_t dim = 1ULL << N;
    std::array<complex_t, dim> cache;
    for (size_t i = 0; i < dim; i++) {
      const auto ii = inds[i];
      cache[i] = data_[ii];
      data_[ii] = 0.;
    }
    // update state vector
    for (size_t i = 0; i < dim; i++)
      for (size_t j = 0; j < dim; j++)
        data_[inds[i]] += _mat[i + dim * j] * cache[j];
  };
  // Use the lambda function
  apply_matrix_lambda(lambda, qs, mat);
}

template <class data_t>
template <size_t N>
void QubitVector<data_t>::apply_diagonal_matrix(const areg_t<N> &qs,
                                                const cvector_t &diag) {

  // Error checking
  #ifdef DEBUG
  check_vector(diag, N);
  #endif

  // Lambda function for N-qubit matrix multiplication
  auto lambda = [&](const areg_t<1ULL << N> &inds,
                    const cvector_t &_mat)->void {
    const uint_t dim = 1ULL << N;
    for (size_t i = 0; i < dim; i++) {
      data_[inds[i]] *= _mat[i];
    }
  };

  // Use the lambda function
  apply_matrix_lambda(lambda, qs, diag);
}

// Static number of pairs
template <class data_t>
template <size_t N, size_t M>
void QubitVector<data_t>::apply_permutation_matrix(const areg_t<N> &qubits,
                                                   const std::array<std::pair<uint_t, uint_t>, M> &pairs) {
  // Lambda function for permutation matrix
  auto lambda = [&](const areg_t<1ULL << N> &inds)->void {
    complex_t cache;
    for (const auto& p : pairs) {
      cache = data_[inds[p.first]];
      data_[inds[p.first]] = data_[inds[p.second]];
      data_[inds[p.second]] = cache;
    }
  };
  // Use the lambda function
  apply_lambda(lambda, qubits);
}

// Dynamic number of pairs
template <class data_t>
template <size_t N>
void QubitVector<data_t>::apply_permutation_matrix(const areg_t<N> &qubits,
                                                   const std::vector<std::pair<uint_t, uint_t>> &pairs) {
  // Lambda function for permutation matrix
  auto lambda = [&](const areg_t<1ULL << N> &inds)->void {
    complex_t cache;
    for (const auto& p : pairs) {
      cache = data_[inds[p.first]];
      data_[inds[p.first]] = data_[inds[p.second]];
      data_[inds[p.second]] = cache;
    }
  };
  // Use the lambda function
  apply_lambda(lambda, qubits);
}

//------------------------------------------------------------------------------
// Dynamic N
//------------------------------------------------------------------------------

/* Generate the repeatative switch cases using the following python code:

```python
code = ""
for j in range(3, 16 + 1):
    code += "  case {0}: {{\n".format(j)
    code += "    areg_t<{0}> qubits_arr;\n".format(j)
    code += "    std::copy_n(qubits.begin(), {0}, qubits_arr.begin());\n".format(j)
    code += "    apply_matrix<{0}>(qubits_arr, mat);\n".format(j)
    code += "  }} break;\n".format(j)
print(code)
```
*/
template <class data_t>
void QubitVector<data_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t &mat) {
  // Special low N cases using faster static indexing
  switch (qubits.size()) {
  case 1:
    apply_matrix(areg_t<1>({{qubits[0]}}), mat);
    break;
  case 2:
    apply_matrix(areg_t<2>({{qubits[0], qubits[1]}}), mat);
    break;
  case 3: {
    areg_t<3> qubits_arr;
    std::copy_n(qubits.begin(), 3, qubits_arr.begin());
    apply_matrix(qubits_arr, mat);
  } break;
  case 4: {
    areg_t<4> qubits_arr;
    std::copy_n(qubits.begin(), 4, qubits_arr.begin());
    apply_matrix(qubits_arr, mat);
  } break;
  case 5: {
    areg_t<5> qubits_arr;
    std::copy_n(qubits.begin(), 5, qubits_arr.begin());
    apply_matrix(qubits_arr, mat);
  } break;
  case 6: {
    areg_t<6> qubits_arr;
    std::copy_n(qubits.begin(), 6, qubits_arr.begin());
    apply_matrix<6>(qubits_arr, mat);
  } break;
  case 7: {
    areg_t<7> qubits_arr;
    std::copy_n(qubits.begin(), 7, qubits_arr.begin());
    apply_matrix<7>(qubits_arr, mat);
  } break;
  case 8: {
    areg_t<8> qubits_arr;
    std::copy_n(qubits.begin(), 8, qubits_arr.begin());
    apply_matrix<8>(qubits_arr, mat);
  } break;
  case 9: {
    areg_t<9> qubits_arr;
    std::copy_n(qubits.begin(), 9, qubits_arr.begin());
    apply_matrix<9>(qubits_arr, mat);
  } break;
  case 10: {
    areg_t<10> qubits_arr;
    std::copy_n(qubits.begin(), 10, qubits_arr.begin());
    apply_matrix<10>(qubits_arr, mat);
  } break;
  case 11: {
    areg_t<11> qubits_arr;
    std::copy_n(qubits.begin(), 11, qubits_arr.begin());
    apply_matrix<11>(qubits_arr, mat);
  } break;
  case 12: {
    areg_t<12> qubits_arr;
    std::copy_n(qubits.begin(), 12, qubits_arr.begin());
    apply_matrix<12>(qubits_arr, mat);
  } break;
  case 13: {
    areg_t<13> qubits_arr;
    std::copy_n(qubits.begin(), 13, qubits_arr.begin());
    apply_matrix<13>(qubits_arr, mat);
  } break;
  case 14: {
    areg_t<14> qubits_arr;
    std::copy_n(qubits.begin(), 14, qubits_arr.begin());
    apply_matrix<14>(qubits_arr, mat);
  } break;
  case 15: {
    areg_t<15> qubits_arr;
    std::copy_n(qubits.begin(), 15, qubits_arr.begin());
    apply_matrix<15>(qubits_arr, mat);
  } break;
  case 16: {
    areg_t<16> qubits_arr;
    std::copy_n(qubits.begin(), 16, qubits_arr.begin());
    apply_matrix<16>(qubits_arr, mat);
  } break;
  default: {
    // Default case using dynamic indexing
    // Error checking
    #ifdef DEBUG
    check_vector(mat, 2 * N);
    #endif

    // Lambda function for N-qubit matrix multiplication
    auto lambda = [&](const std::vector<uint_t> &inds,
                      const cvector_t &_mat)->void {
      const uint_t dim = 1ULL << qubits.size();
      std::vector<complex_t> cache(dim);
      for (size_t i = 0; i < dim; i++) {
        const auto ii = inds[i];
        cache[i] = data_[ii];
        data_[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
          data_[inds[i]] += _mat[i + dim * j] * cache[j];
    };
    // Use the lambda function
    apply_matrix_lambda(lambda, qubits, mat);
  } // end default
    break;
  } // end switch
}

/* Generate the repeatative switch cases using the following python code:

```python
code = ""
for j in range(3, 16 + 1):
    code += "  case {0}: {{\n".format(j)
    code += "    areg_t<{0}> qubits_arr;\n".format(j)
    code += "    std::copy_n(qubits.begin(), {0}, qubits_arr.begin());\n".format(j)
    code += "    apply_diagonal_matrix<{0}>(qubits_arr, mat);\n".format(j)
    code += "  }} break;\n".format(j)
print(code)
```
*/
template <class data_t>
void QubitVector<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t &mat) {
  // Special low N cases using faster static indexing
  switch (qubits.size()) {
  case 1:
    apply_diagonal_matrix(areg_t<1>({{qubits[0]}}), mat);
    break;
  case 2:
    apply_diagonal_matrix<2>(areg_t<2>({{qubits[0], qubits[1]}}), mat);
    break;
    case 3: {
    areg_t<3> qubits_arr;
    std::copy_n(qubits.begin(), 3, qubits_arr.begin());
    apply_diagonal_matrix<3>(qubits_arr, mat);
  } break;
  case 4: {
    areg_t<4> qubits_arr;
    std::copy_n(qubits.begin(), 4, qubits_arr.begin());
    apply_diagonal_matrix<4>(qubits_arr, mat);
  } break;
  case 5: {
    areg_t<5> qubits_arr;
    std::copy_n(qubits.begin(), 5, qubits_arr.begin());
    apply_diagonal_matrix<5>(qubits_arr, mat);
  } break;
  case 6: {
    areg_t<6> qubits_arr;
    std::copy_n(qubits.begin(), 6, qubits_arr.begin());
    apply_diagonal_matrix<6>(qubits_arr, mat);
  } break;
  case 7: {
    areg_t<7> qubits_arr;
    std::copy_n(qubits.begin(), 7, qubits_arr.begin());
    apply_diagonal_matrix<7>(qubits_arr, mat);
  } break;
  case 8: {
    areg_t<8> qubits_arr;
    std::copy_n(qubits.begin(), 8, qubits_arr.begin());
    apply_diagonal_matrix<8>(qubits_arr, mat);
  } break;
  case 9: {
    areg_t<9> qubits_arr;
    std::copy_n(qubits.begin(), 9, qubits_arr.begin());
    apply_diagonal_matrix<9>(qubits_arr, mat);
  } break;
  case 10: {
    areg_t<10> qubits_arr;
    std::copy_n(qubits.begin(), 10, qubits_arr.begin());
    apply_diagonal_matrix<10>(qubits_arr, mat);
  } break;
  case 11: {
    areg_t<11> qubits_arr;
    std::copy_n(qubits.begin(), 11, qubits_arr.begin());
    apply_diagonal_matrix<11>(qubits_arr, mat);
  } break;
  case 12: {
    areg_t<12> qubits_arr;
    std::copy_n(qubits.begin(), 12, qubits_arr.begin());
    apply_diagonal_matrix<12>(qubits_arr, mat);
  } break;
  case 13: {
    areg_t<13> qubits_arr;
    std::copy_n(qubits.begin(), 13, qubits_arr.begin());
    apply_diagonal_matrix<13>(qubits_arr, mat);
  } break;
  case 14: {
    areg_t<14> qubits_arr;
    std::copy_n(qubits.begin(), 14, qubits_arr.begin());
    apply_diagonal_matrix<14>(qubits_arr, mat);
  } break;
  case 15: {
    areg_t<15> qubits_arr;
    std::copy_n(qubits.begin(), 15, qubits_arr.begin());
    apply_diagonal_matrix<15>(qubits_arr, mat);
  } break;
  case 16: {
    areg_t<16> qubits_arr;
    std::copy_n(qubits.begin(), 16, qubits_arr.begin());
    apply_diagonal_matrix<16>(qubits_arr, mat);
  } break;
  default: {
    // Default case using dynamic indexing
    // Error checking
    #ifdef DEBUG
    check_vector(mat, N);
    #endif

    // Lambda function for N-qubit matrix multiplication
    auto lambda = [&](const std::vector<uint_t> &inds,
                      const cvector_t &_mat)->void {
      const uint_t dim = 1ULL << qubits.size();
      for (size_t i = 0; i < dim; i++)
            data_[inds[i]] *= _mat[i];
    };
    // Use the lambda function
    apply_matrix_lambda(lambda, qubits, mat);
  } // end default
    break;
  } // end switch
}

/*******************************************************************************
 *
 * OPTIMIZED MATRIX MULTIPLICATION
 *
 ******************************************************************************/

/*******************************************************************************
 *
 * APPLY OPTIMIZED GATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Single-qubit gates
//------------------------------------------------------------------------------
template <class data_t>
void QubitVector<data_t>::apply_x(const uint_t qubit) {
  // Lambda function for optimized Pauli-X gate
  auto lambda = [&](const int_t &k1, const int_t &k2)->void {
    const auto i0 = k1 | k2;
    const auto i1 = i0 | (1LL << qubit);
    const complex_t cache = data_[i0];
    data_[i0] = data_[i1]; // mat(0,1)
    data_[i1] = cache;    // mat(1,0)
  };
  apply_lambda(lambda, qubit);
}

template <class data_t>
void QubitVector<data_t>::apply_y(const uint_t qubit) {
  // Lambda function for optimized Pauli-Y gate
  const complex_t I(0., 1.);
  auto lambda = [&](const int_t &k1, const int_t &k2)->void {
    const auto i0 = k1 | k2;
    const auto i1 = i0 | (1LL << qubit);
    const complex_t cache = data_[i0];
    data_[i0] = -I * data_[i1]; // mat(0,1)
    data_[i1] = I * cache;     // mat(1,0)
  };
  apply_lambda(lambda, qubit);
}

template <class data_t>
void QubitVector<data_t>::apply_z(const uint_t qubit) {
  // Lambda function for optimized Pauli-Z gate
  auto lambda = [&](const int_t &k1, const int_t &k2)->void {
    data_[k1 | k2 | (1LL << qubit)] *= complex_t(-1.0, 0.0);
  };
  apply_lambda(lambda, qubit);
}

//------------------------------------------------------------------------------
// Two-qubit gates
//------------------------------------------------------------------------------
template <class data_t>
void QubitVector<data_t>::apply_cnot(const uint_t qubit_ctrl, const uint_t qubit_trgt) {
  // Lambda function for CNOT gate
  auto lambda = [&](const areg_t<1ULL << 2> &inds)->void {
    const complex_t cache = data_[inds[3]];
    data_[inds[3]] = data_[inds[1]];
    data_[inds[1]] = cache;
  };
  // Use the lambda function
  const areg_t<2> qubits = {{qubit_ctrl, qubit_trgt}};
  apply_lambda(lambda, qubits);
}

template <class data_t>
void QubitVector<data_t>::apply_swap(const uint_t qubit0, const uint_t qubit1) {
  // Lambda function for SWAP gate
  auto lambda = [&](const areg_t<1ULL << 2> &inds)->void {
    const complex_t cache = data_[inds[2]];
      data_[inds[2]] = data_[inds[1]];
      data_[inds[1]] = cache;
  };
  // Use the lambda function
  const areg_t<2> qubits = {{qubit0, qubit1}};
  apply_lambda(lambda, qubits);
}

template <class data_t>
void QubitVector<data_t>::apply_cz(const uint_t qubit_ctrl, const uint_t qubit_trgt) {

  // Lambda function for CZ gate
  auto lambda = [&](const areg_t<1ULL << 2> &inds)->void {
    data_[inds[3]] *= -1.;
  };
  // Use the lambda function
  const areg_t<2> qubits = {{qubit_ctrl, qubit_trgt}};
  apply_lambda(lambda, qubits);
}

//------------------------------------------------------------------------------
// Three-qubit gates
//------------------------------------------------------------------------------
template <class data_t>
void QubitVector<data_t>::apply_toffoli(const uint_t qubit_ctrl0,
                                const uint_t qubit_ctrl1,
                                const uint_t qubit_trgt) {
  // Lambda function for Toffoli gate
  auto lambda = [&](const areg_t<1ULL << 3> &inds)->void {
    const complex_t cache = data_[inds[7]];
    data_[inds[7]] = data_[inds[3]];
    data_[inds[3]] = cache;
  };
  // Use the lambda function
  const areg_t<3> qubits = {{qubit_ctrl0, qubit_ctrl1, qubit_trgt}};
  apply_lambda(lambda, qubits);
}

//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <class data_t>
void QubitVector<data_t>::apply_matrix(const areg_t<1> &qubits,
                                       const cvector_t &mat) {
  const int_t bit = 1LL << qubits[0];
  // Lambda function for single-qubit matrix multiplication
  auto lambda = [&](const int_t &k1, const int_t &k2,
                    const cvector_t &_mat)->void {
    const auto k = k1 | k2;
    const auto cache0 = data_[k];
    const auto cache1 = data_[k | bit];
    data_[k] = _mat[0] * cache0 + _mat[2] * cache1;
    data_[k | bit] = _mat[1] * cache0 + _mat[3] * cache1;
  };
  apply_matrix_lambda(lambda, qubits[0], mat);
}

template <class data_t>
void QubitVector<data_t>::apply_diagonal_matrix(const areg_t<1> &qubits,
                                                const cvector_t &diag) {
  // TODO: This should be changed so it isn't checking doubles with ==
  const int_t bit = 1LL << qubits[0];
  if (diag[0] == 1.0) {
    // [[1, 0], [0, z]] matrix
    if (diag[1] == 1.0) {
      // Identity
      return;
    } else if (diag[1] == complex_t(0., -1.)) {
      // [[1, 0], [0, -i]]
      auto lambda = [&](const int_t &k1, const int_t &k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2 | bit;
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real() * -1.);
        data_[k].real(cache);
      };
      apply_matrix_lambda(lambda, qubits[0], diag);
    } else if (diag[1] == complex_t(0., 1.)) {
      // [[1, 0], [0, i]]
      auto lambda = [&](const int_t &k1, const int_t &k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2 | bit;
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real());
        data_[k].real(cache * -1.);
      };
      apply_matrix_lambda(lambda, qubits[0], diag);
    } else if (diag[0] == 0.0) {
      // [[1, 0], [0, 0]]
      auto lambda = [&](const int_t &k1, const int_t &k2,
                        const cvector_t &_mat)->void {
        data_[k1 | k2 | bit] = 0.0;
      };
      apply_matrix_lambda(lambda, qubits[0], diag);
    } else {
      // general [[1, 0], [0, z]]
      auto lambda = [&](const int_t &k1, const int_t &k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2 | bit;
        data_[k] *= _mat[1];
      };
      apply_matrix_lambda(lambda, qubits[0], diag);
    }
  } else if (diag[1] == 1.0) {
    // [[z, 0], [0, 1]] matrix
    if (diag[0] == complex_t(0., -1.)) {
      // [[-i, 0], [0, 1]]
      auto lambda = [&](const int_t &k1, const int_t &k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2;
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real() * -1.);
        data_[k].real(cache);
      };
      apply_matrix_lambda(lambda, qubits[0], diag);
    } else if (diag[0] == complex_t(0., 1.)) {
      // [[i, 0], [0, 1]]
      auto lambda = [&](const int_t &k1, const int_t &k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2;
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real());
        data_[k].real(cache * -1.);
      };
      apply_matrix_lambda(lambda, qubits[0], diag);
    } else if (diag[0] == 0.0) {
      // [[0, 0], [0, 1]]
      auto lambda = [&](const int_t &k1, const int_t &k2,
                        const cvector_t &_mat)->void {
        data_[k1 | k2] = 0.0;
      };
      apply_matrix_lambda(lambda, qubits[0], diag);
    } else {
      // general [[z, 0], [0, 1]]
      auto lambda = [&](const int_t &k1, const int_t &k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2;
        data_[k] *= _mat[0];
      };
      apply_matrix_lambda(lambda, qubits[0], diag);
    }
  } else {
    // Lambda function for diagonal matrix multiplication
    auto lambda = [&](const int_t &k1, const int_t &k2,
                      const cvector_t &_mat)->void {
      const auto k = k1 | k2;
      data_[k] *= _mat[0];
      data_[k | bit] *= _mat[1];
    };
    apply_matrix_lambda(lambda, qubits[0], diag);
  }
}

//------------------------------------------------------------------------------
// 2-6 qubit matrices
//------------------------------------------------------------------------------

template <class data_t>
void QubitVector<data_t>::apply_matrix(const areg_t<2> &qubits,
                                       const cvector_t &vmat) {
  if (gate_opt_ == false) {
    apply_matrix<2>(qubits, vmat);
  } else {
  // Optimized implementation
    auto sorted_qs = qubits;
    std::sort(sorted_qs.begin(), sorted_qs.end());
    auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);

    int_t end = size_;
    int_t step1 = (1ULL << sorted_qs[0]);
    int_t step2 = (1ULL << sorted_qs[1]);
  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
  #ifdef _WIN32
  #pragma omp for
  #else
  #pragma omp for collapse(3)
  #endif
      for (int_t k1 = 0; k1 < end; k1 += (step2 * 2UL)) {
        for (int_t k2 = 0; k2 < step2; k2 += (step1 * 2UL)) {
          for (int_t k3 = 0; k3 < step1; k3++) {
            int_t t0 = k1 | k2 | k3;
            int_t t1 = t0 | step1;
            int_t t2 = t0 | step2;
            int_t t3 = t2 | step1;

            const complex_t psi0 = data_[t0];
            const complex_t psi1 = data_[t1];
            const complex_t psi2 = data_[t2];
            const complex_t psi3 = data_[t3];

            data_[t0] = psi0 * sorted_vmat[0] + psi1 * sorted_vmat[1] + psi2 * sorted_vmat[2] + psi3 * sorted_vmat[3];
            data_[t1] = psi0 * sorted_vmat[4] + psi1 * sorted_vmat[5] + psi2 * sorted_vmat[6] + psi3 * sorted_vmat[7];
            data_[t2] = psi0 * sorted_vmat[8] + psi1 * sorted_vmat[9] + psi2 * sorted_vmat[10] + psi3 * sorted_vmat[11];
            data_[t3] = psi0 * sorted_vmat[12] + psi1 * sorted_vmat[13] + psi2 * sorted_vmat[14] + psi3 * sorted_vmat[15];
          }
        }
      }
    }
  }
}

template <class data_t>
void QubitVector<data_t>::apply_matrix(const areg_t<3> &qubits,
                               const cvector_t &vmat) {
  if (gate_opt_ == false) {
    apply_matrix<3>(qubits, vmat);
  } else {
    // Optimized implementation
    auto sorted_qs = qubits;
    std::sort(sorted_qs.begin(), sorted_qs.end());
    auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
    const uint_t dim = 1ULL << 3;

    int_t end = size_;
    int_t step1 = (1ULL << sorted_qs[0]);
    int_t step2 = (1ULL << sorted_qs[1]);
    int_t step3 = (1ULL << sorted_qs[2]);

    int_t masks[] = {//
        0, //
        step1, //
        step2, //
        step2 | step1, //
        step3, //
        step3 | step1, //
        step3 | step2, //
        step3 | step2 | step1 //
    };

  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
  #ifdef _WIN32
  #pragma omp for
  #else
  #pragma omp for collapse(4)
  #endif
      for (int_t k1 = 0; k1 < end; k1 += (step3 * 2UL)) {
        for (int_t k2 = 0; k2 < step3; k2 += (step2 * 2UL)) {
          for (int_t k3 = 0; k3 < step2; k3 += (step1 * 2UL)) {
            for (int_t k4 = 0; k4 < step1; k4++) {
              int_t base = k1 | k2 | k3 | k4;
              complex_t psi[8];
              for (int_t i = 0; i < 8; ++i) {
                psi[i] = data_[base | masks[i]];
                data_[base | masks[i]] = 0.;
              }
              for (size_t i = 0; i < 8; ++i)
                for (size_t j = 0; j < 8; ++j)
                  data_[base | masks[i]] += psi[j] * sorted_vmat[j * dim + i];
            }
          }
        }
      }
    }
  }
}

template <class data_t>
void QubitVector<data_t>::apply_matrix(const areg_t<4> &qubits,
                               const cvector_t &vmat) {
  if (gate_opt_ == false) {
    apply_matrix<4>(qubits, vmat);
  } else {
    // Optimized implementation
    auto sorted_qs = qubits;
    std::sort(sorted_qs.begin(), sorted_qs.end());
    auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
    const uint_t dim = 1ULL << 4;

    int_t end = size_;
    int_t step1 = (1ULL << sorted_qs[0]);
    int_t step2 = (1ULL << sorted_qs[1]);
    int_t step3 = (1ULL << sorted_qs[2]);
    int_t step4 = (1ULL << sorted_qs[3]);

    int_t masks[] = {//
        0, //
        step1, //
        step2, //
        step2 | step1, //
        step3, //
        step3 | step1, //
        step3 | step2, //
        step3 | step2 | step1, //
        step4, //
        step4 | step1, //
        step4 | step2, //
        step4 | step2 | step1, //
        step4 | step3, //
        step4 | step3 | step1, //
        step4 | step3 | step2, //
        step4 | step3 | step2 | step1 //
    };

  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
  #ifdef _WIN32
  #pragma omp for
  #else
  #pragma omp for collapse(5)
  #endif
      for (int_t k1 = 0; k1 < end; k1 += (step4 * 2UL)) {
        for (int_t k2 = 0; k2 < step4; k2 += (step3 * 2UL)) {
          for (int_t k3 = 0; k3 < step3; k3 += (step2 * 2UL)) {
            for (int_t k4 = 0; k4 < step2; k4 += (step1 * 2UL)) {
              for (int_t k5 = 0; k5 < step1; k5++) {
                int_t base = k1 | k2 | k3 | k4 | k5;
                complex_t psi[16];
                for (int_t i = 0; i < 16; ++i) {
                  psi[i] = data_[base | masks[i]];
                  data_[base | masks[i]] = 0.;
                }
                for (size_t i = 0; i < 16; ++i)
                  for (size_t j = 0; j < 16; ++j)
                    data_[base | masks[i]] += psi[j] * sorted_vmat[j * dim + i];
              }
            }
          }
        }
      }
    }
  }
}

template <class data_t>
void QubitVector<data_t>::apply_matrix(const areg_t<5> &qubits,
                                       const cvector_t &vmat) {
  if (gate_opt_ == false) {
    apply_matrix<5>(qubits, vmat);
  } else {
    // Optimized implementation
    auto sorted_qs = qubits;
    std::sort(sorted_qs.begin(), sorted_qs.end());
    auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
    const uint_t dim = 1ULL << 5;

    int_t end = size_;
    int_t step1 = (1ULL << sorted_qs[0]);
    int_t step2 = (1ULL << sorted_qs[1]);
    int_t step3 = (1ULL << sorted_qs[2]);
    int_t step4 = (1ULL << sorted_qs[3]);
    int_t step5 = (1ULL << sorted_qs[4]);

    int_t masks[] = {//
        0, //
        step1, //
        step2, //
        step2 | step1, //
        step3, //
        step3 | step1, //
        step3 | step2, //
        step3 | step2 | step1, //
        step4, //
        step4 | step1, //
        step4 | step2, //
        step4 | step2 | step1, //
        step4 | step3, //
        step4 | step3 | step1, //
        step4 | step3 | step2, //
        step4 | step3 | step2 | step1, //
        step5, //
        step5 | step1, //
        step5 | step2, //
        step5 | step2 | step1, //
        step5 | step3, //
        step5 | step3 | step1, //
        step5 | step3 | step2, //
        step5 | step3 | step2 | step1, //
        step5 | step4, //
        step5 | step4 | step1, //
        step5 | step4 | step2, //
        step5 | step4 | step2 | step1, //
        step5 | step4 | step3, //
        step5 | step4 | step3 | step1, //
        step5 | step4 | step3 | step2, //
        step5 | step4 | step3 | step2 | step1 //
    };

  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
  #ifdef _WIN32
  #pragma omp for
  #else
  #pragma omp for collapse(6)
  #endif
      for (int_t k1 = 0; k1 < end; k1 += (step5 * 2UL)) {
        for (int_t k2 = 0; k2 < step5; k2 += (step4 * 2UL)) {
          for (int_t k3 = 0; k3 < step4; k3 += (step3 * 2UL)) {
            for (int_t k4 = 0; k4 < step3; k4 += (step2 * 2UL)) {
              for (int_t k5 = 0; k5 < step2; k5 += (step1 * 2UL)) {
                for (int_t k6 = 0; k6 < step1; k6++) {
                  int_t base = k1 | k2 | k3 | k4 | k5 | k6;
                  complex_t psi[32];
                  for (int_t i = 0; i < 32; ++i) {
                    psi[i] = data_[base | masks[i]];
                    data_[base | masks[i]] = 0.;
                  }
                  for (size_t i = 0; i < 32; ++i)
                    for (size_t j = 0; j < 32; ++j)
                      data_[base | masks[i]] += psi[j] * sorted_vmat[j * dim + i];
                }
              }
            }
          }
        }
      }
    }
  }
}

template <class data_t>
void QubitVector<data_t>::apply_matrix(const areg_t<6> &qubits,
                                       const cvector_t &vmat) {
  if (gate_opt_ == false) {
    apply_matrix<6>(qubits, vmat);
  } else {
    // Optimized implementation
    auto sorted_qs = qubits;
    std::sort(sorted_qs.begin(), sorted_qs.end());
    auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
    const uint_t dim = 1ULL << 6;

    int_t end = size_;
    int_t step1 = (1ULL << sorted_qs[0]);
    int_t step2 = (1ULL << sorted_qs[1]);
    int_t step3 = (1ULL << sorted_qs[2]);
    int_t step4 = (1ULL << sorted_qs[3]);
    int_t step5 = (1ULL << sorted_qs[4]);
    int_t step6 = (1ULL << sorted_qs[5]);

    int_t masks[] = {//
        0, //
        step1, //
        step2, //
        step2 | step1, //
        step3, //
        step3 | step1, //
        step3 | step2, //
        step3 | step2 | step1, //
        step4, //
        step4 | step1, //
        step4 | step2, //
        step4 | step2 | step1, //
        step4 | step3, //
        step4 | step3 | step1, //
        step4 | step3 | step2, //
        step4 | step3 | step2 | step1, //
        step5, //
        step5 | step1, //
        step5 | step2, //
        step5 | step2 | step1, //
        step5 | step3, //
        step5 | step3 | step1, //
        step5 | step3 | step2, //
        step5 | step3 | step2 | step1, //
        step5 | step4, //
        step5 | step4 | step1, //
        step5 | step4 | step2, //
        step5 | step4 | step2 | step1, //
        step5 | step4 | step3, //
        step5 | step4 | step3 | step1, //
        step5 | step4 | step3 | step2, //
        step5 | step4 | step3 | step2 | step1, //
        step6, //
        step6 | step1, //
        step6 | step2, //
        step6 | step2 | step1, //
        step6 | step3, //
        step6 | step3 | step1, //
        step6 | step3 | step2, //
        step6 | step3 | step2 | step1, //
        step6 | step4, //
        step6 | step4 | step1, //
        step6 | step4 | step2, //
        step6 | step4 | step2 | step1, //
        step6 | step4 | step3, //
        step6 | step4 | step3 | step1, //
        step6 | step4 | step3 | step2, //
        step6 | step4 | step3 | step2 | step1, //
        step6 | step5, //
        step6 | step5 | step1, //
        step6 | step5 | step2, //
        step6 | step5 | step2 | step1, //
        step6 | step5 | step3, //
        step6 | step5 | step3 | step1, //
        step6 | step5 | step3 | step2, //
        step6 | step5 | step3 | step2 | step1, //
        step6 | step5 | step4, //
        step6 | step5 | step4 | step1, //
        step6 | step5 | step4 | step2, //
        step6 | step5 | step4 | step2 | step1, //
        step6 | step5 | step4 | step3, //
        step6 | step5 | step4 | step3 | step1, //
        step6 | step5 | step4 | step3 | step2, //
        step6 | step5 | step4 | step3 | step2 | step1 //
    };

  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
  #ifdef _WIN32
  #pragma omp for
  #else
  #pragma omp for collapse(7)
  #endif
      for (int_t k1 = 0; k1 < end; k1 += (step6 * 2UL)) {
        for (int_t k2 = 0; k2 < step6; k2 += (step5 * 2UL)) {
          for (int_t k3 = 0; k3 < step5; k3 += (step4 * 2UL)) {
            for (int_t k4 = 0; k4 < step4; k4 += (step3 * 2UL)) {
              for (int_t k5 = 0; k5 < step3; k5 += (step2 * 2UL)) {
                for (int_t k6 = 0; k6 < step2; k6 += (step1 * 2UL)) {
                  for (int_t k7 = 0; k7 < step1; k7++) {
                    int_t base = k1 | k2 | k3 | k4 | k5 | k6 | k7;
                    complex_t psi[64];
                    for (int_t i = 0; i < 64; ++i) {
                      psi[i] = data_[base | masks[i]];
                      data_[base | masks[i]] = 0.;
                    }
                    for (size_t i = 0; i < 64; ++i)
                      for (size_t j = 0; j < 64; ++j)
                        data_[base | masks[i]] += psi[j] * sorted_vmat[j * dim + i];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Gate-swap optimized helper functions
//------------------------------------------------------------------------------
template <class data_t>
void QubitVector<data_t>::swap_cols_and_rows(const uint_t idx1,
                                             const uint_t idx2,
                                             cvector_t &mat,
                                             uint_t dim) const {

  uint_t mask1 = (1UL << idx1);
  uint_t mask2 = (1UL << idx2);

  for (uint_t first = 0; first < dim; ++first) {
    if ((first & mask1) && !(first & mask2)) {
      uint_t second = (first ^ mask1) | mask2;

      for (uint_t i = 0; i < dim; ++i) {
        complex_t cache = mat[first * dim + i];
        mat[first * dim + i] = mat[second * dim +  i];
        mat[second * dim +  i] = cache;
      }
      for (uint_t i = 0; i < dim; ++i) {
        complex_t cache = mat[i * dim + first];
        mat[i * dim + first] = mat[i * dim + second];
        mat[i * dim + second] = cache;
      }
    }
  }
}

template <class data_t>
template <size_t N>
cvector_t QubitVector<data_t>::sort_matrix(const areg_t<N> &src,
                                           const areg_t<N> &sorted,
                                           const cvector_t &mat) const {

  const uint_t dim = 1ULL << N;
  auto ret = mat;
  auto current = src;

  while (current != sorted) {
    uint_t from;
    uint_t to;
    for (from = 0; from < current.size(); ++from)
      if (current[from] != sorted[from])
        break;
    if (from == current.size())
      break;
    for (to = from + 1; to < current.size(); ++to)
      if (current[from] == sorted[to])
        break;
    if (to == current.size()) {
      std::stringstream ss;
      ss << "QubitVector<data_t>::sort_matrix we should not reach here";
      throw std::runtime_error(ss.str());
    }
    swap_cols_and_rows(from, to, ret, dim);

    uint_t cache = current[from];
    current[from] = current[to];
    current[to] = cache;
  }

  return ret;
}

/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <class data_t>
double QubitVector<data_t>::norm() const {
  // Lambda function for norm
  auto lambda = [&](int_t k, double &val_re, double &val_im)->void {
    (void)val_im; // unused
    val_re += std::real(data_[k] * std::conj(data_[k]));
  };
  return std::real(apply_reduction_lambda(lambda));
}


//------------------------------------------------------------------------------
// Static N
//------------------------------------------------------------------------------
template <class data_t>
template <size_t N>
double QubitVector<data_t>::norm(const areg_t<N> &qs,
                         const cvector_t &mat) const {
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Lambda function for N-qubit matrix norm
  auto lambda = [&](const areg_t<1ULL << N> &inds, const cvector_t &_mat, 
                    double &val_re, double &val_im)->void {
    (void)val_im; // unused
    const uint_t dim = 1ULL << N;
    for (size_t i = 0; i < dim; i++) {
      complex_t vi = 0;
      for (size_t j = 0; j < dim; j++)
        vi += _mat[i + dim * j] * data_[inds[j]];
      val_re += std::real(vi * std::conj(vi));
    }
  };
  // Use the lambda function
  return std::real(apply_matrix_reduction_lambda(lambda, qs, mat));
}

template <class data_t>
template <size_t N>
double QubitVector<data_t>::norm_diagonal(const areg_t<N> &qs,
                                  const cvector_t &mat) const {

  // Error checking
  #ifdef DEBUG
  check_vector(mat, N);
  #endif

  // Lambda function for N-qubit matrix norm
  auto lambda = [&](const areg_t<1ULL << N> &inds, const cvector_t &_mat,
                    double &val_re, double &val_im)->void {
    (void)val_im; // unused
    const uint_t dim = 1ULL << N;
    for (size_t i = 0; i < dim; i++) {
      const auto vi = _mat[i] * data_[inds[i]];
      val_re += std::real(vi * std::conj(vi));
    }
  };
  // Use the lambda function
  return std::real(apply_matrix_reduction_lambda(lambda, qs, mat));
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <class data_t>
double QubitVector<data_t>::norm(const areg_t<1> &qubits,
                                        const cvector_t &mat) const {
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 2);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const int_t &k1, const int_t &k2,const cvector_t &_mat,
                    double &val_re, double &val_im)->void {
    (void)val_im; // unused;
    const auto k = k1 | k2;
    const auto cache0 = data_[k];
    const auto cache1 = data_[k | (1LL << qubits[0])];
    const auto v0 = _mat[0] * cache0 + _mat[2] * cache1;
    const auto v1 = _mat[1] * cache0 + _mat[3] * cache1;
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_matrix_reduction_lambda(lambda, qubits[0], mat));
}

template <class data_t>
double QubitVector<data_t>::norm_diagonal(const areg_t<1> &qubits,
                                  const cvector_t &mat) const {
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 1);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const int_t &k1, const int_t &k2,const cvector_t &_mat,
                    double &val_re, double &val_im)->void {
    (void)val_im; // unused;
    const auto k = k1 | k2;
    const auto v0 = _mat[0] * data_[k];
    const auto v1 = _mat[1] * data_[k | (1LL << qubits[0])];
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_matrix_reduction_lambda(lambda, qubits[0], mat));
}

//------------------------------------------------------------------------------
// Dynamic N
//------------------------------------------------------------------------------

/* Generate the repeatative switch cases using the following python code:

```python
code = ""
for j in range(3, 16 + 1):
    code += "  case {0}: {{\n".format(j)
    code += "    areg_t<{0}> qubits_arr;\n".format(j)
    code += "    std::copy_n(qubits.begin(), {0}, qubits_arr.begin());\n".format(j)
    code += "    return norm<{0}>(qubits_arr, mat);\n".format(j)
    code += "  }\n"
print(code)
```
*/

template <class data_t>
double QubitVector<data_t>::norm(const reg_t &qubits,
                         const cvector_t &mat) const {

  // Special low N cases using faster static indexing
  switch (qubits.size()) {
  case 1:
    return norm<1>(areg_t<1>({{qubits[0]}}), mat);
  case 2:
    return norm<2>(areg_t<2>({{qubits[0], qubits[1]}}), mat);
  case 3: {
    areg_t<3> qubits_arr;
    std::copy_n(qubits.begin(), 3, qubits_arr.begin());
    return norm<3>(qubits_arr, mat);
  }
  case 4: {
    areg_t<4> qubits_arr;
    std::copy_n(qubits.begin(), 4, qubits_arr.begin());
    return norm<4>(qubits_arr, mat);
  }
  case 5: {
    areg_t<5> qubits_arr;
    std::copy_n(qubits.begin(), 5, qubits_arr.begin());
    return norm<5>(qubits_arr, mat);
  }
  case 6: {
    areg_t<6> qubits_arr;
    std::copy_n(qubits.begin(), 6, qubits_arr.begin());
    return norm<6>(qubits_arr, mat);
  }
  case 7: {
    areg_t<7> qubits_arr;
    std::copy_n(qubits.begin(), 7, qubits_arr.begin());
    return norm<7>(qubits_arr, mat);
  }
  case 8: {
    areg_t<8> qubits_arr;
    std::copy_n(qubits.begin(), 8, qubits_arr.begin());
    return norm<8>(qubits_arr, mat);
  }
  case 9: {
    areg_t<9> qubits_arr;
    std::copy_n(qubits.begin(), 9, qubits_arr.begin());
    return norm<9>(qubits_arr, mat);
  }
  case 10: {
    areg_t<10> qubits_arr;
    std::copy_n(qubits.begin(), 10, qubits_arr.begin());
    return norm<10>(qubits_arr, mat);
  }
  case 11: {
    areg_t<11> qubits_arr;
    std::copy_n(qubits.begin(), 11, qubits_arr.begin());
    return norm<11>(qubits_arr, mat);
  }
  case 12: {
    areg_t<12> qubits_arr;
    std::copy_n(qubits.begin(), 12, qubits_arr.begin());
    return norm<12>(qubits_arr, mat);
  }
  case 13: {
    areg_t<13> qubits_arr;
    std::copy_n(qubits.begin(), 13, qubits_arr.begin());
    return norm<13>(qubits_arr, mat);
  }
  case 14: {
    areg_t<14> qubits_arr;
    std::copy_n(qubits.begin(), 14, qubits_arr.begin());
    return norm<14>(qubits_arr, mat);
  }
  case 15: {
    areg_t<15> qubits_arr;
    std::copy_n(qubits.begin(), 15, qubits_arr.begin());
    return norm<15>(qubits_arr, mat);
  }
  case 16: {
    areg_t<16> qubits_arr;
    std::copy_n(qubits.begin(), 16, qubits_arr.begin());
    return norm<16>(qubits_arr, mat);
  }
  default: {

    // Error checking
    const uint_t N = qubits.size();
    const uint_t dim = 1ULL << N;
    #ifdef DEBUG
    check_vector(mat, 2 * N);
    #endif

    // Lambda function for N-qubit matrix norm
    auto lambda = [&](const reg_t &inds, const cvector_t &_mat,
                      double &val_re, double &val_im)->void {
      (void)val_im; // unused
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++)
          vi += _mat[i + dim * j] * data_[inds[j]];
        val_re += std::real(vi * std::conj(vi));
      }
    };
    // Use the lambda function
    return std::real(apply_matrix_reduction_lambda(lambda, qubits, mat));
  } // end default
  } // end switch
}

/* Generate the repeatative switch cases using the following python code:

```python
code = ""
for j in range(3, 16 + 1):
    code += "  case {0}: {{\n".format(j)
    code += "    areg_t<{0}> qubits_arr;\n".format(j)
    code += "    std::copy_n(qubits.begin(), {0}, qubits_arr.begin());\n".format(j)
    code += "    return norm_diagonal<{0}>(qubits_arr, mat);\n".format(j)
    code += "  }\n"
print(code)
```
*/

template <class data_t>
double QubitVector<data_t>::norm_diagonal(const reg_t &qubits, const cvector_t &mat) const {

  // Special low N cases using faster static indexing
  switch (qubits.size()) {
  case 1:
    return norm_diagonal<1>(areg_t<1>({{qubits[0]}}), mat);
  case 2:
    return norm_diagonal<2>(areg_t<2>({{qubits[0], qubits[1]}}), mat);
    case 3: {
    areg_t<3> qubits_arr;
    std::copy_n(qubits.begin(), 3, qubits_arr.begin());
    return norm_diagonal<3>(qubits_arr, mat);
  }
  case 4: {
    areg_t<4> qubits_arr;
    std::copy_n(qubits.begin(), 4, qubits_arr.begin());
    return norm_diagonal<4>(qubits_arr, mat);
  }
  case 5: {
    areg_t<5> qubits_arr;
    std::copy_n(qubits.begin(), 5, qubits_arr.begin());
    return norm_diagonal<5>(qubits_arr, mat);
  }
  case 6: {
    areg_t<6> qubits_arr;
    std::copy_n(qubits.begin(), 6, qubits_arr.begin());
    return norm_diagonal<6>(qubits_arr, mat);
  }
  case 7: {
    areg_t<7> qubits_arr;
    std::copy_n(qubits.begin(), 7, qubits_arr.begin());
    return norm_diagonal<7>(qubits_arr, mat);
  }
  case 8: {
    areg_t<8> qubits_arr;
    std::copy_n(qubits.begin(), 8, qubits_arr.begin());
    return norm_diagonal<8>(qubits_arr, mat);
  }
  case 9: {
    areg_t<9> qubits_arr;
    std::copy_n(qubits.begin(), 9, qubits_arr.begin());
    return norm_diagonal<9>(qubits_arr, mat);
  }
  case 10: {
    areg_t<10> qubits_arr;
    std::copy_n(qubits.begin(), 10, qubits_arr.begin());
    return norm_diagonal<10>(qubits_arr, mat);
  }
  case 11: {
    areg_t<11> qubits_arr;
    std::copy_n(qubits.begin(), 11, qubits_arr.begin());
    return norm_diagonal<11>(qubits_arr, mat);
  }
  case 12: {
    areg_t<12> qubits_arr;
    std::copy_n(qubits.begin(), 12, qubits_arr.begin());
    return norm_diagonal<12>(qubits_arr, mat);
  }
  case 13: {
    areg_t<13> qubits_arr;
    std::copy_n(qubits.begin(), 13, qubits_arr.begin());
    return norm_diagonal<13>(qubits_arr, mat);
  }
  case 14: {
    areg_t<14> qubits_arr;
    std::copy_n(qubits.begin(), 14, qubits_arr.begin());
    return norm_diagonal<14>(qubits_arr, mat);
  }
  case 15: {
    areg_t<15> qubits_arr;
    std::copy_n(qubits.begin(), 15, qubits_arr.begin());
    return norm_diagonal<15>(qubits_arr, mat);
  }
  case 16: {
    areg_t<16> qubits_arr;
    std::copy_n(qubits.begin(), 16, qubits_arr.begin());
    return norm_diagonal<16>(qubits_arr, mat);
  }
  default: {
    // Default dynamic index case
    // Error checking
    const uint_t N = qubits.size();
    const uint_t dim = 1ULL << N;
    #ifdef DEBUG
    check_vector(mat, N);
    #endif

    // Lambda function for N-qubit matrix norm
    auto lambda = [&](const reg_t &inds, const cvector_t &_mat,
                      double &val_re, double &val_im)->void {
      (void)val_im; // unused
      for (size_t i = 0; i < dim; i++) {
        const auto vi = _mat[i] * data_[inds[i]];
        val_re += std::real(vi * std::conj(vi));
      }
    };
    // Use the lambda function
    return std::real(apply_matrix_reduction_lambda(lambda, qubits, mat));
  } // end default
  } // end switch
}

/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/

template <class data_t>
double QubitVector<data_t>::probability(const uint_t outcome) const {
  const auto v = data_[outcome];
  return std::real(v * std::conj(v));
}

template <class data_t>
rvector_t QubitVector<data_t>::probabilities() const {
  rvector_t probs(size_);
  const int_t end = size_;
  probs.assign(size_, 0.);

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < end; j++) {
    probs[j] = probability(j);
  }
  return probs;
}

//------------------------------------------------------------------------------
// Static N-qubit
//------------------------------------------------------------------------------
template <class data_t>
template <size_t N>
rvector_t QubitVector<data_t>::probabilities(const areg_t<N> &qs) const {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  if (N == 0)
    return rvector_t({norm()});

  const uint_t dim = 1ULL << N;
  const uint_t end = (1ULL << num_qubits_) >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  if ((N == num_qubits_) && (qs == qss))
    return probabilities();

  rvector_t probs(dim, 0.);
  for (size_t k = 0; k < end; k++) {
    const auto idx = indexes_static<N>(qs, qubits_sorted, k);
    for (size_t m = 0; m < dim; ++m) {
      probs[m] += probability(idx[m]);
    }
  }
  return probs;
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------

template <class data_t>
rvector_t QubitVector<data_t>::probabilities(const areg_t<1> &qubits) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  // Lambda function for single qubit probs as reduction
  // p(0) stored as real part p(1) as imag part
  auto lambda = [&](const int_t &k1, const int_t &k2,
                    double &val_p0, double &val_p1)->void {
    const auto k = k1 | k2;
    val_p0 += probability(k);
    val_p1 += probability(k | (1LL << qubits[0]));
  };
  auto p0p1 = apply_reduction_lambda(lambda, qubits[0]);
  return rvector_t({std::real(p0p1), std::imag(p0p1)});
}

//------------------------------------------------------------------------------
// Dynamic N-qubit
//------------------------------------------------------------------------------

/* Generate the repeatative switch cases using the following python code:

```python
code = ""
for j in range(3, 16 + 1):
    code += "  case {0}: {{\n".format(j)
    code += "    areg_t<{0}> qubits_arr;\n".format(j)
    code += "    std::copy_n(qubits.begin(), {0}, qubits_arr.begin());\n".format(j)
    code += "    return probabilities<{0}>(qubits_arr);\n".format(j)
    code += "  }\n"
print(code)
```
*/
template <class data_t>
rvector_t QubitVector<data_t>::probabilities(const reg_t &qubits) const {

  // Special cases using faster static indexing
  const uint_t N = qubits.size();
  switch (N) {
  case 0:
    return rvector_t({norm()});
  case 1:
    return probabilities<1>(areg_t<1>({{qubits[0]}}));
  case 2:
    return probabilities<2>(areg_t<2>({{qubits[0], qubits[1]}}));
  case 3: {
    areg_t<3> qubits_arr;
    std::copy_n(qubits.begin(), 3, qubits_arr.begin());
    return probabilities<3>(qubits_arr);
  }
  case 4: {
    areg_t<4> qubits_arr;
    std::copy_n(qubits.begin(), 4, qubits_arr.begin());
    return probabilities<4>(qubits_arr);
  }
  case 5: {
    areg_t<5> qubits_arr;
    std::copy_n(qubits.begin(), 5, qubits_arr.begin());
    return probabilities<5>(qubits_arr);
  }
  case 6: {
    areg_t<6> qubits_arr;
    std::copy_n(qubits.begin(), 6, qubits_arr.begin());
    return probabilities<6>(qubits_arr);
  }
  case 7: {
    areg_t<7> qubits_arr;
    std::copy_n(qubits.begin(), 7, qubits_arr.begin());
    return probabilities<7>(qubits_arr);
  }
  case 8: {
    areg_t<8> qubits_arr;
    std::copy_n(qubits.begin(), 8, qubits_arr.begin());
    return probabilities<8>(qubits_arr);
  }
  case 9: {
    areg_t<9> qubits_arr;
    std::copy_n(qubits.begin(), 9, qubits_arr.begin());
    return probabilities<9>(qubits_arr);
  }
  case 10: {
    areg_t<10> qubits_arr;
    std::copy_n(qubits.begin(), 10, qubits_arr.begin());
    return probabilities<10>(qubits_arr);
  }
  case 11: {
    areg_t<11> qubits_arr;
    std::copy_n(qubits.begin(), 11, qubits_arr.begin());
    return probabilities<11>(qubits_arr);
  }
  case 12: {
    areg_t<12> qubits_arr;
    std::copy_n(qubits.begin(), 12, qubits_arr.begin());
    return probabilities<12>(qubits_arr);
  }
  case 13: {
    areg_t<13> qubits_arr;
    std::copy_n(qubits.begin(), 13, qubits_arr.begin());
    return probabilities<13>(qubits_arr);
  }
  case 14: {
    areg_t<14> qubits_arr;
    std::copy_n(qubits.begin(), 14, qubits_arr.begin());
    return probabilities<14>(qubits_arr);
  }
  case 15: {
    areg_t<15> qubits_arr;
    std::copy_n(qubits.begin(), 15, qubits_arr.begin());
    return probabilities<15>(qubits_arr);
  }
  case 16: {
    areg_t<16> qubits_arr;
    std::copy_n(qubits.begin(), 16, qubits_arr.begin());
    return probabilities<16>(qubits_arr);
  }
  default: {
    // else
    // Error checking
    #ifdef DEBUG
    for (const auto &qubit : qubits)
      check_qubit(qubit);
    #endif

    const uint_t dim = 1ULL << N;
    const uint_t end = (1ULL << num_qubits_) >> N;
    auto qss = qubits;
    std::sort(qss.begin(), qss.end());
    if ((N == num_qubits_) && (qss == qubits))
      return probabilities();
    const auto &qubits_sorted = qss;
    rvector_t probs(dim, 0.);
  
    for (size_t k = 0; k < end; k++) {
      const auto idx = indexes_dynamic(qubits, qubits_sorted, N, k);
      for (size_t m = 0; m < dim; ++m)
        probs[m] += probability(idx[m]);
    }
    return probs;
  } // end default
  } // end switch
}


//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <class data_t>
reg_t QubitVector<data_t>::sample_measure(const std::vector<double> &rnds) const {

  const int_t end = size_;
  const int_t shots = rnds.size();
  reg_t samples;
  samples.assign(shots, 0);

  const int index_size = sample_measure_index_size_;
  const int_t index_end = 1LL << index_size;
  // Qubit number is below index size, loop over shots
  if (end < index_end) {
    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
      #pragma omp for
      for (int_t i = 0; i < shots; ++i) {
        double rnd = rnds[i];
        double p = .0;
        int_t sample;
        for (sample = 0; sample < end - 1; ++sample) {
          p += std::real(std::conj(data_[sample]) * data_[sample]);
          if (rnd < p)
            break;
        }
        samples[i] = sample;
      }
    } // end omp parallel
  }
  // Qubit number is above index size, loop over index blocks
  else {
    // Initialize indexes
    std::vector<double> idxs;
    idxs.assign((1<<index_size), .0);
    uint_t loop = (end >> index_size);

    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
      #pragma omp for
      for (int_t i = 0; i < (1 << index_size); ++i) {
        uint_t base = loop * i;
        double total = .0;
        double p = .0;
        for (uint_t j = 0; j < loop; ++j) {
          uint_t k = base | j;
          p = std::real(std::conj(data_[k]) * data_[k]);
          total += p;
        }
        idxs[i] = total;
      }
    } // end omp parallel

    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
      #pragma omp for
      for (int_t i = 0; i < shots; ++i) {
        double rnd = rnds[i];
        double p = .0;
        int_t sample = 0;
        for (uint_t j = 0; j < idxs.size(); ++j) {
          if (rnd < (p + idxs[j])) {
            break;
          }
          p += idxs[j];
          sample += loop;
        }

        for (; sample < end - 1; ++sample) {
          p += std::real(std::conj(data_[sample]) * data_[sample]);
          if (rnd < p){
            break;
          }
        }
        samples[i] = sample;
      }
    } // end omp parallel
  }
  return samples;
}

//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <class data_t>
inline std::ostream &operator<<(std::ostream &out, const QV::QubitVector<data_t>&qv) {

  out << "[";
  size_t last = qv.size() - 1;
  for (size_t i = 0; i < qv.size(); ++i) {
    out << qv[i];
    if (i != last)
      out << ", ";
  }
  out << "]";
  return out;
}

//------------------------------------------------------------------------------
#endif // end module
