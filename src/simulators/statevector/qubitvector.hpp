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
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "framework/json.hpp"

namespace QV {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using reg_t = std::vector<uint_t>;
using indexes_t = std::unique_ptr<uint_t[]>;
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;

//============================================================================
// BIT MASKS and indexing
//============================================================================

/*
# Auto generate these values with following python snippet
import json

def cpp_init_list(int_lst):
    ret = json.dumps([str(i) + 'ULL' for i in int_lst])
    return ret.replace('"', '').replace('[','{{').replace(']','}};')

print('const std::array<uint_t, 64> BITS = ' + cpp_init_list([(1 << i) for i in range(64)]) + '\n')
print('const std::array<uint_t, 64> MASKS = ' + cpp_init_list([(1 << i) - 1 for i in range(64)]))
*/

const std::array<uint_t, 64> BITS {{
  1ULL, 2ULL, 4ULL, 8ULL,
  16ULL, 32ULL, 64ULL, 128ULL,
  256ULL, 512ULL, 1024ULL, 2048ULL,
  4096ULL, 8192ULL, 16384ULL, 32768ULL,
  65536ULL, 131072ULL, 262144ULL, 524288ULL,
  1048576ULL, 2097152ULL, 4194304ULL, 8388608ULL,
  16777216ULL, 33554432ULL, 67108864ULL, 134217728ULL,
  268435456ULL, 536870912ULL, 1073741824ULL, 2147483648ULL,
  4294967296ULL, 8589934592ULL, 17179869184ULL, 34359738368ULL, 
  68719476736ULL, 137438953472ULL, 274877906944ULL, 549755813888ULL,
  1099511627776ULL, 2199023255552ULL, 4398046511104ULL, 8796093022208ULL,
  17592186044416ULL, 35184372088832ULL, 70368744177664ULL, 140737488355328ULL, 
  281474976710656ULL, 562949953421312ULL, 1125899906842624ULL, 2251799813685248ULL,
  4503599627370496ULL, 9007199254740992ULL, 18014398509481984ULL, 36028797018963968ULL,
  72057594037927936ULL, 144115188075855872ULL, 288230376151711744ULL, 576460752303423488ULL,
  1152921504606846976ULL, 2305843009213693952ULL, 4611686018427387904ULL, 9223372036854775808ULL
}};


const std::array<uint_t, 64> MASKS {{
  0ULL, 1ULL, 3ULL, 7ULL,
  15ULL, 31ULL, 63ULL, 127ULL,
  255ULL, 511ULL, 1023ULL, 2047ULL,
  4095ULL, 8191ULL, 16383ULL, 32767ULL,
  65535ULL, 131071ULL, 262143ULL, 524287ULL,
  1048575ULL, 2097151ULL, 4194303ULL, 8388607ULL,
  16777215ULL, 33554431ULL, 67108863ULL, 134217727ULL,
  268435455ULL, 536870911ULL, 1073741823ULL, 2147483647ULL,
  4294967295ULL, 8589934591ULL, 17179869183ULL, 34359738367ULL,
  68719476735ULL, 137438953471ULL, 274877906943ULL, 549755813887ULL,
  1099511627775ULL, 2199023255551ULL, 4398046511103ULL, 8796093022207ULL,
  17592186044415ULL, 35184372088831ULL, 70368744177663ULL, 140737488355327ULL,
  281474976710655ULL, 562949953421311ULL, 1125899906842623ULL, 2251799813685247ULL,
  4503599627370495ULL, 9007199254740991ULL, 18014398509481983ULL, 36028797018963967ULL,
  72057594037927935ULL, 144115188075855871ULL, 288230376151711743ULL, 576460752303423487ULL,
  1152921504606846975ULL, 2305843009213693951ULL, 4611686018427387903ULL, 9223372036854775807ULL
}};


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

template <typename data_t = complex_t*>
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
  uint_t size() const {return data_size_;}

  // Returns a copy of the underlying data_t data as a complex vector
  cvector_t vector() const;

  // Return JSON serialization of QubitVector;
  json_t json() const;

  // Set all entries in the vector to 0.
  void zero();

  // Return as an int an N qubit bitstring for M-N qubit bit k with 0s inserted
  // for N qubits at the locations specified by qubits_sorted.
  // qubits_sorted must be sorted lowest to highest. Eg. {0, 1}.
  uint_t index0(const reg_t &qubits_sorted, const uint_t k) const;

  // Return a std::unique_ptr to an array of of 2^N in ints
  // each int corresponds to an N qubit bitstring for M-N qubit bits in state k,
  // and the specified N qubits in states [0, ..., 2^N - 1]
  // qubits_sorted must be sorted lowest to highest. Eg. {0, 1}.
  // qubits specifies the location of the qubits in the retured strings.
  // NOTE: since the return is a unique_ptr it cannot be copied.
  indexes_t indexes(const reg_t & qubits, const reg_t &qubits_sorted, const uint_t k) const;

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

  // Apply a 1-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  void apply_matrix(const uint_t qubit, const cvector_t &mat);

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const reg_t &qubits, const cvector_t &mat);

  // Apply a 1-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const uint_t qubit, const cvector_t &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const reg_t &qubits, const cvector_t &mat);
  
  // Swap pairs of indicies in the underlying vector
  void apply_permutation_matrix(const reg_t &qubits,
                                const std::vector<std::pair<uint_t, uint_t>> &pairs);

  //-----------------------------------------------------------------------
  // Apply Specialized Gates
  //-----------------------------------------------------------------------

  // Apply a single-qubit Pauli-X gate to the state vector
  void apply_x(const uint_t qubit);

  // Apply a single-qubit Pauli-Y gate to the state vector
  void apply_y(const uint_t qubit);

  // Apply a single-qubit Pauli-Z gate to the state vector
  void apply_z(const uint_t qubit);

  // Apply a 2-qubit SWAP gate to the state vector
  void apply_swap(const uint_t q0, const uint_t q1);

  // Apply multi-controlled X-gate
  void apply_mcx(const reg_t &qubits);

  // Apply multi-controlled Y-gate
  void apply_mcy(const reg_t &qubits);

  // Apply multi-controlled Z-gate
  void apply_mcz(const reg_t &qubits);
  
  // Apply multi-controlled single-qubit unitary gate
  void apply_mcu(const reg_t &qubits, const cvector_t &mat);

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

  // Return the Z-basis measurement outcome probabilities [P(0), P(1)]
  // for measurement of specified qubit
  rvector_t probabilities(const uint_t qubit) const;

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  rvector_t probabilities(const reg_t &qubits) const;

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

  // Return the norm for of the vector obtained after apply the 1-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  double norm(const uint_t qubit, const cvector_t &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  double norm(const reg_t &qubits, const cvector_t &mat) const;

  // Return the norm for of the vector obtained after apply the 1-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const uint_t qubit, const cvector_t &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const reg_t &qubits, const cvector_t &mat) const;

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
  void enable_gate_opt() {gate_opt_ = true;}

  // Disable sorted qubit matrix gate optimization
  void disable_gate_opt() {gate_opt_ = false;}

  // Set the sample_measure index size
  void set_sample_measure_index_size(int n) {sample_measure_index_size_ = n;}

  // Get the sample_measure index size
  int get_sample_measure_index_size() {return sample_measure_index_size_;}

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  size_t num_qubits_;
  size_t data_size_;
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
  // [&](const int_t k)->void
  // where k is the index of the vector
  template <typename Lambda>
  void apply_lambda(Lambda&& func);

  // Apply a single-qubit lambda function to all blocks of the statevector
  // for the given qubit. The function signature should be:
  // [&](const int_t k1, const int_t k2)->void
  // where (k1, k2) are the 0 and 1 indexes for each qubit block
  template <typename Lambda>
  void apply_lambda(Lambda&& func, const uint_t qubit);

  // Apply a N-qubit lambda function to all blocks of the statevector
  // for the given qubits. The function signature should be:
  // [&](const indexes_t &inds)->void
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
  // [&](const int_t k1, const int_t k2, const cvector_t &m)->void
  // where (k1, k2) are the 0 and 1 indexes for each qubit block and
  // m is a vectorized complex matrix.
  template <typename Lambda>
  void apply_matrix_lambda(Lambda&& func,
                           const uint_t qubit,
                           const cvector_t &mat);

  // Apply a N-qubit matrix lambda function to all blocks of the statevector
  // for the given qubits. The function signature should be:
  // [&](const indexes_t &inds, const cvector_t &m)->void
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
  // [&](const int_t k, double &val_re, double &val_im)->void
  // where k is the index of the vector, val_re and val_im are the doubles
  // to store the reduction.
  // Returns complex_t(val_re, val_im)
  template <typename Lambda>
  complex_t apply_reduction_lambda(Lambda&& func) const;

  // Apply a 1-qubit complex reduction  lambda function to all blocks of the
  // statevector for the given qubit. The function signature should be:
  // [&](const int_t k1, const int_t k2, double &val_re, double &val_im)->void
  // where (k1, k2) are the 0 and 1 indexes for each qubit block
  // val_re and val_im are the doubles to store the reduction.
  // Returns complex_t(val_re, val_im)
  template <typename Lambda>
  complex_t apply_reduction_lambda(Lambda&& func,
                                   const uint_t qubit) const;

  // Apply a N-qubit complex reduction lambda function to all blocks of the
  // statevector for the given qubits. The function signature should be:
  // [&](const indexes_t &inds, double &val_re, double &val_im)->void
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
  // [&](const int_t k1, const int_t k2, const cvector_t &m,
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
  // [&](const indexes_t &inds, const cvector_t &m,
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
  // Optimized matrix multiplication
  //-----------------------------------------------------------------------
  
  // Optimized implementations
  void apply_matrix2(const reg_t &qubits, const cvector_t &mat);
  void apply_matrix3(const reg_t &qubits, const cvector_t &mat);
  void apply_matrix4(const reg_t &qubits, const cvector_t &mat);
  void apply_matrix5(const reg_t &qubits, const cvector_t &mat);
  void apply_matrix6(const reg_t &qubits, const cvector_t &mat);

  // Permute an N-qubit vectorized matrix to match a reordering of qubits
  cvector_t sort_matrix(const reg_t &src,
                        const reg_t &sorted,
                        const cvector_t &mat) const;

  // Swap cols and rows of vectorized matrix
  void swap_cols_and_rows(const uint_t idx1, const uint_t idx2,
                          cvector_t &mat, uint_t dim) const;
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

template <typename data_t>
inline void to_json(json_t &js, const QubitVector<data_t> &qv) {
  js = qv.json();
}

template <typename data_t>
json_t QubitVector<data_t>::json() const {
  const int_t END = data_size_;
  const json_t ZERO = complex_t(0.0, 0.0);
  json_t js = json_t(data_size_, ZERO);
  
  if (json_chop_threshold_ > 0) {
    #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    for (int_t j=0; j < END; j++) {
      if (std::abs(data_[j].real()) > json_chop_threshold_)
        js[j][0] = data_[j].real();
      if (std::abs(data_[j].imag()) > json_chop_threshold_)
        js[j][1] = data_[j].imag();
    }
  } else {
    #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    for (int_t j=0; j < END; j++) {
      js[j][0] = data_[j].real();
      js[j][1] = data_[j].imag();
    }
  }
  return js;
}

//------------------------------------------------------------------------------
// Error Handling
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits_) {
    std::string error = "QubitVector: qubit index " + std::to_string(qubit) +
                        " > " + std::to_string(num_qubits_);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVector<data_t>::check_matrix(const cvector_t &vec, uint_t nqubits) const {
  const size_t DIM = BITS[nqubits];
  const auto SIZE = vec.size();
  if (SIZE != DIM * DIM) {
    std::string error = "QubitVector: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM * DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVector<data_t>::check_vector(const cvector_t &vec, uint_t nqubits) const {
  const size_t DIM = BITS[nqubits];
  const auto SIZE = vec.size();
  if (SIZE != DIM) {
    std::string error = "QubitVector: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVector<data_t>::check_dimension(const QubitVector &qv) const {
  if (data_size_ != qv.size_) {
    std::string error = "QubitVector: vectors are different shape " +
                         std::to_string(data_size_) + " != " +
                         std::to_string(qv.num_states_);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVector<data_t>::check_checkpoint() const {
  if (!checkpoint_) {
    throw std::runtime_error("QubitVector: checkpoint must exist for inner_product() or revert()");
  }
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t>
QubitVector<data_t>::QubitVector(size_t num_qubits) : num_qubits_(0), data_(0), checkpoint_(0){
  set_num_qubits(num_qubits);
}

template <typename data_t>
QubitVector<data_t>::QubitVector() : QubitVector(0) {}

template <typename data_t>
QubitVector<data_t>::~QubitVector() {
  if (data_)
    free(data_);

  if (checkpoint_)
    free(checkpoint_);
}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <typename data_t>
complex_t &QubitVector<data_t>::operator[](uint_t element) {
  // Error checking
  #ifdef DEBUG
  if (element > data_size_) {
    std::string error = "QubitVector: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }
  #endif
  return data_[element];
}

template <typename data_t>
complex_t QubitVector<data_t>::operator[](uint_t element) const {
  // Error checking
  #ifdef DEBUG
  if (element > data_size_) {
    std::string error = "QubitVector: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }
  #endif
  return data_[element];
}

template <typename data_t>
cvector_t QubitVector<data_t>::vector() const {
  cvector_t ret(data_size_, 0.);
  const int_t END = data_size_;
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < END; j++) {
    ret[j] = data_[j];
  }
  return ret;
}

//------------------------------------------------------------------------------
// Indexing
//------------------------------------------------------------------------------

template <typename data_t>
uint_t QubitVector<data_t>::index0(const reg_t& qubits_sorted, const uint_t k) const {
  uint_t lowbits, retval = k;
  for (const auto& qubit : qubits_sorted) {
    lowbits = retval & MASKS[qubit];
    retval >>= qubit;
    retval <<= qubit + 1;
    retval |= lowbits;
  }
  return retval;
}

template <typename data_t>
indexes_t QubitVector<data_t>::indexes(const reg_t& qubits,
                                       const reg_t& qubits_sorted,
                                       const uint_t k) const {
  const auto N = qubits_sorted.size();
  indexes_t ret(new uint_t[BITS[N]]);
  // Get index0
  ret[0] = index0(qubits_sorted, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = BITS[i];
    const auto bit = BITS[qubits[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::zero() {
  const int_t END = data_size_;    // end for k loop

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < END; ++k) {
    data_[k] = 0.0;
  }
}

template <typename data_t>
void QubitVector<data_t>::set_num_qubits(size_t num_qubits) {
  num_qubits_ = num_qubits;
  data_size_ = BITS[num_qubits];

  // Free any currently assigned memory
  if (data_)
    free(data_);

  if (checkpoint_) {
    free(checkpoint_);
    checkpoint_ = nullptr;
  }

  // Allocate memory for new vector
  data_ = reinterpret_cast<complex_t*>(malloc(sizeof(complex_t) * data_size_));
}


template <typename data_t>
void QubitVector<data_t>::checkpoint() {
  if (!checkpoint_)
    checkpoint_ = reinterpret_cast<complex_t*>(malloc(sizeof(complex_t) * data_size_));

  const int_t END = data_size_;    // end for k loop
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < END; ++k)
    checkpoint_[k] = data_[k];
}


template <typename data_t>
void QubitVector<data_t>::revert(bool keep) {

  #ifdef DEBUG
  check_checkpoint();
  #endif

  const int_t END = data_size_;    // end for k loop
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < END; ++k)
    data_[k] = checkpoint_[k];

  if (!keep) {
    free(checkpoint_);
    checkpoint_ = nullptr;
  }
}

template <typename data_t>
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

template <typename data_t>
void QubitVector<data_t>::initialize() {
  zero();
  data_[0] = 1.;
}

template <typename data_t>
void QubitVector<data_t>::initialize_from_vector(const cvector_t &statevec) {
  if (data_size_ != statevec.size()) {
    std::string error = "QubitVector::initialize input vector is incorrect length (" + 
                        std::to_string(data_size_) + "!=" +
                        std::to_string(statevec.size()) + ")";
    throw std::runtime_error(error);
  }

  const int_t END = data_size_;    // end for k loop

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < END; ++k)
    data_[k] = statevec[k];
}

template <typename data_t>
void QubitVector<data_t>::initialize_from_data(const data_t &statevec, const size_t num_states) {
  if (data_size_ != num_states) {
    std::string error = "QubitVector::initialize input vector is incorrect length (" +
                        std::to_string(data_size_) + "!=" + std::to_string(num_states) + ")";
    throw std::runtime_error(error);
  }

  const int_t END = data_size_;    // end for k loop

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < END; ++k)
    data_[k] = statevec[k];
}


/*******************************************************************************
 *
 * CONFIG SETTINGS
 *
 ******************************************************************************/

template <typename data_t>
void QubitVector<data_t>::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

template <typename data_t>
void QubitVector<data_t>::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

template <typename data_t>
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
template <typename data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_lambda(Lambda&& func) {
  const int_t END = data_size_;
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      std::forward<Lambda>(func)(k);
    }
  }
}

// Single qubit
template <typename data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_lambda(Lambda&& func,
                                       const uint_t qubit) {
  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t END1 = data_size_;       // end for k1 loop
  const int_t END2 = BITS[qubit];      // end for k2 loop
  const int_t STEP1 = BITS[qubit + 1]; // step for k1 loop
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < END1; k1 += STEP1)
      for (int_t k2 = 0; k2 < END2; k2++) {
        std::forward<Lambda>(func)(k1, k2);
      }
  }
}

// Dynamic N-qubit
template <typename data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_lambda(Lambda&& func,
                                       const reg_t &qubits) {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const auto NUM_QUBITS = qubits.size();
  const int_t END = data_size_ >> NUM_QUBITS;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      // store entries touched by U
      auto inds = indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds);
    }
  }
}

//------------------------------------------------------------------------------
// Matrix Lambda
//------------------------------------------------------------------------------

// Single qubit
template <typename data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_matrix_lambda(Lambda&& func,
                                              const uint_t qubit,
                                              const cvector_t &mat) {
  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t END1 = data_size_;       // end for k1 loop
  const int_t END2 = BITS[qubit];      // end for k2 loop
  const int_t STEP1 = BITS[qubit + 1]; // step for k1 loop
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < END1; k1 += STEP1)
      for (int_t k2 = 0; k2 < END2; k2++) {
        std::forward<Lambda>(func)(k1, k2, mat);
      }
  }
}

// Dynamic N-qubit
template <typename data_t>
template<typename Lambda>
void QubitVector<data_t>::apply_matrix_lambda(Lambda&& func, const reg_t &qubits, const cvector_t &mat) {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const auto NUM_QUBITS = qubits.size();
  const int_t END = data_size_ >> NUM_QUBITS;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds = indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, mat);
    }
  }
}


//------------------------------------------------------------------------------
// Reduction Lambda
//------------------------------------------------------------------------------

template <typename data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_reduction_lambda(Lambda &&func) const {
  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
  const int_t END = data_size_;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
        std::forward<Lambda>(func)(k, val_re, val_im);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

// Single-qubit
template <typename data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_reduction_lambda(Lambda &&func,
                                                      const uint_t qubit) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t END1 = data_size_;       // end for k1 loop
  const int_t END2 = BITS[qubit];      // end for k2 loop
  const int_t STEP1 = BITS[qubit + 1]; // step for k1 loop

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
    for (int_t k1 = 0; k1 < END1; k1 += STEP1)
      for (int_t k2 = 0; k2 < END2; k2++) {
        std::forward<Lambda>(func)(k1, k2, val_re, val_im);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

// Dynamic N-qubit
template <typename data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_reduction_lambda(Lambda&& func,
                                                      const reg_t &qubits) const {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const size_t NUM_QUBITS =  qubits.size();
  const int_t END = data_size_ >> NUM_QUBITS;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds = indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, val_re, val_im);
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

//------------------------------------------------------------------------------
// Matrix and Reduction Lambda
//------------------------------------------------------------------------------

template <typename data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_matrix_reduction_lambda(Lambda &&func,
                                                             const uint_t qubit,
                                                             const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t END1 = data_size_;       // end for k1 loop
  const int_t END2 = BITS[qubit];      // end for k2 loop
  const int_t STEP1 = BITS[qubit + 1]; // step for k1 loop

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
    for (int_t k1 = 0; k1 < END1; k1 += STEP1)
      for (int_t k2 = 0; k2 < END2; k2++) {
        std::forward<Lambda>(func)(k1, k2, mat, val_re, val_im);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


template <typename data_t>
template<typename Lambda>
complex_t QubitVector<data_t>::apply_matrix_reduction_lambda(Lambda&& func,
                                                             const reg_t &qubits,
                                                             const cvector_t &mat) const {

  const auto NUM_QUBITS = qubits.size();
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const int_t END = data_size_ >> NUM_QUBITS;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds = indexes(qubits, qubits_sorted, k);
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
template <typename data_t>
void QubitVector<data_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t &mat) {
  
  const size_t N = qubits.size();
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Optimized 1-qubit apply matrix.
  if (N==1) {
    apply_matrix(qubits[0], mat);
    return;
  }

  // Matrix-swap based optimized 2-6 qubit implementation
  if (gate_opt_ && N <= 6) {
    switch (N) {
      case 2:
        apply_matrix2(qubits, mat);
        return;
      case 3:
        apply_matrix3(qubits, mat);
        return;
      case 4:
        apply_matrix4(qubits, mat);
        return;
      case 5:
        apply_matrix5(qubits, mat);
        return;
      case 6:
        apply_matrix6(qubits, mat);
        return;
    default:
      break;
    } // end switch
  }
  
  // General implementation
  const uint_t DIM = BITS[N];
  // Lambda function for N-qubit matrix multiplication
  auto lambda = [&](const indexes_t &inds, const cvector_t &_mat)->void {
    auto cache = std::make_unique<complex_t[]>(DIM);
    for (size_t i = 0; i < DIM; i++) {
      const auto ii = inds[i];
      cache[i] = data_[ii];
      data_[ii] = 0.;
    }
    // update state vector
    for (size_t i = 0; i < DIM; i++)
      for (size_t j = 0; j < DIM; j++)
        data_[inds[i]] += _mat[i + DIM * j] * cache[j];
  };

  // Use the lambda function
  apply_matrix_lambda(lambda, qubits, mat);
}

template <typename data_t>
void QubitVector<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t &diag) {

  // Optimized 1-qubit apply matrix.                                                
  if (qubits.size() == 1) {
    apply_diagonal_matrix(qubits[0], diag);
    return;
  }

  const size_t N = qubits.size();
  const uint_t DIM = BITS[N];
  // Error checking
  #ifdef DEBUG
  check_vector(diag, N);
  #endif

  // Lambda function for N-qubit matrix multiplication
  auto lambda = [&](const indexes_t &inds,
                    const cvector_t &_mat)->void {
    for (size_t i = 0; i < DIM; i++) {
      data_[inds[i]] *= _mat[i];
    }
  };

  // Use the lambda function
  apply_matrix_lambda(lambda, qubits, diag);
}

template <typename data_t>
void QubitVector<data_t>::apply_permutation_matrix(const reg_t& qubits,
                                                   const std::vector<std::pair<uint_t, uint_t>> &pairs) {
  // Lambda function for permutation matrix
  auto lambda = [&](const indexes_t &inds)->void {
    for (const auto& p : pairs) {
      std::swap(data_[inds[p.first]], data_[inds[p.second]]);
    }
  };
  // Use the lambda function
  apply_lambda(lambda, qubits);
}

/*******************************************************************************
 *
 * APPLY OPTIMIZED GATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Single-qubit gates
//------------------------------------------------------------------------------
template <typename data_t>
void QubitVector<data_t>::apply_x(const uint_t qubit) {
  // Lambda function for optimized Pauli-X gate
  auto lambda = [&](const int_t k1, const int_t k2)->void {
    const auto i0 = k1 | k2;
    const auto i1 = i0 | BITS[qubit];
    std::swap(data_[i0], data_[i1]);
  };
  apply_lambda(lambda, qubit);
}

template <typename data_t>
void QubitVector<data_t>::apply_y(const uint_t qubit) {
  // Lambda function for optimized Pauli-Y gate
  const complex_t I(0., 1.);
  auto lambda = [&](const int_t k1, const int_t k2)->void {
    const auto i0 = k1 | k2;
    const auto i1 = i0 | BITS[qubit];
    const complex_t cache = data_[i0];
    data_[i0] = -I * data_[i1]; // mat(0,1)
    data_[i1] = I * cache;     // mat(1,0)
  };
  apply_lambda(lambda, qubit);
}

template <typename data_t>
void QubitVector<data_t>::apply_z(const uint_t qubit) {
  // Lambda function for optimized Pauli-Z gate
  auto lambda = [&](const int_t k1, const int_t k2)->void {
    data_[k1 | k2 | BITS[qubit]] *= complex_t(-1.0, 0.0);
  };
  apply_lambda(lambda, qubit);
}

//------------------------------------------------------------------------------
// Multi-controlled gates
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::apply_mcx(const reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = MASKS[N];
  // Lambda function for multi-controlled X gate
  auto lambda = [&](indexes_t inds)->void {
    std::swap(data_[inds[pos0]], data_[inds[pos1]]);
  };
  apply_lambda(lambda, qubits);
}

template <typename data_t>
void QubitVector<data_t>::apply_mcy(const reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = MASKS[N];
  const complex_t I(0., 1.);
  // Lambda function for multi-controlled Y gate
  auto lambda = [&](indexes_t inds)->void {
    const complex_t cache = data_[inds[pos0]];
    data_[inds[pos0]] = -I * data_[inds[pos1]];
    data_[inds[pos1]] = I * cache;
  };
  apply_lambda(lambda, qubits);
}

template <typename data_t>
void QubitVector<data_t>::apply_mcz(const reg_t &qubits) {
  // Lambda function for multi-controlled Z gate
  auto lambda = [&](indexes_t inds)->void {
    // Multiply last block index by -1
    data_[inds[MASKS[qubits.size()]]] *= -1.;
  };
  apply_lambda(lambda, qubits);
}

template <typename data_t>
void QubitVector<data_t>::apply_mcu(const reg_t &qubits,
                                    const cvector_t &mat) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = MASKS[N];
  // Lambda function for multi-controlled single-qubit gate
  auto lambda = [&](const indexes_t &inds,
                    const cvector_t &_mat)->void {
    const auto cache = data_[pos0];
    data_[pos0] = _mat[0] * data_[pos0] + _mat[2] * data_[pos1];
    data_[pos1] = _mat[1] * cache + _mat[3] * data_[pos1];
  };
  apply_matrix_lambda(lambda, qubits, mat);
}

//------------------------------------------------------------------------------
// Swap gates
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::apply_swap(const uint_t qubit0, const uint_t qubit1) {
  // Lambda function for SWAP gate
  auto lambda = [&](const indexes_t &inds)->void {
    std::swap(data_[inds[1]], data_[inds[2]]);
  };
  // Use the lambda function
  const reg_t qubits = {qubit0, qubit1};
  apply_lambda(lambda, qubits);
}

//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::apply_matrix(const uint_t qubit,
                                       const cvector_t& mat) {
  // Check if matrix is actually diagonal and if so use 
  // apply_diagonal_matrix
  // TODO: this should be changed to not check doubles with ==
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const cvector_t diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix(qubit, diag);
    return;
  }

  // Lambda function for single-qubit matrix multiplication
  const int_t BIT = BITS[qubit];
  auto lambda = [&](const int_t k1, const int_t k2,
                    const cvector_t &_mat)->void {
    const auto pos0 = k1 | k2;
    const auto pos1 = pos0 | BIT;
    const auto cache = data_[pos0];
    data_[pos0] = _mat[0] * data_[pos0] + _mat[2] * data_[pos1];
    data_[pos1] = _mat[1] * cache + _mat[3] * data_[pos1];
  };
  apply_matrix_lambda(lambda, qubit, mat);
}

template <typename data_t>
void QubitVector<data_t>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t& diag) {
  // TODO: This should be changed so it isn't checking doubles with ==
  const int_t BIT = BITS[qubit];
  if (diag[0] == 1.0) {  // [[1, 0], [0, z]] matrix
    if (diag[1] == 1.0)
      return; // Identity

    if (diag[1] == complex_t(0., -1.)) { // [[1, 0], [0, -i]]
      auto lambda = [&](const int_t k1, const int_t k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2 | BIT;
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real() * -1.);
        data_[k].real(cache);
      };
      apply_matrix_lambda(lambda, qubit, diag);
    } else if (diag[1] == complex_t(0., 1.)) {
      // [[1, 0], [0, i]]
      auto lambda = [&](const int_t k1, const int_t k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2 | BIT;
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real());
        data_[k].real(cache * -1.);
      };
      apply_matrix_lambda(lambda, qubit, diag);
    } else if (diag[0] == 0.0) {
      // [[1, 0], [0, 0]]
      auto lambda = [&](const int_t k1, const int_t k2,
                        const cvector_t &_mat)->void {
        data_[k1 | k2 | BIT] = 0.0;
      };
      apply_matrix_lambda(lambda, qubit, diag);
    } else {
      // general [[1, 0], [0, z]]
      auto lambda = [&](const int_t k1, const int_t k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2 | BIT;
        data_[k] *= _mat[1];
      };
      apply_matrix_lambda(lambda, qubit, diag);
    }
  } else if (diag[1] == 1.0) {
    // [[z, 0], [0, 1]] matrix
    if (diag[0] == complex_t(0., -1.)) {
      // [[-i, 0], [0, 1]]
      auto lambda = [&](const int_t k1, const int_t k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2;
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real() * -1.);
        data_[k].real(cache);
      };
      apply_matrix_lambda(lambda, qubit, diag);
    } else if (diag[0] == complex_t(0., 1.)) {
      // [[i, 0], [0, 1]]
      auto lambda = [&](const int_t k1, const int_t k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2;
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real());
        data_[k].real(cache * -1.);
      };
      apply_matrix_lambda(lambda, qubit, diag);
    } else if (diag[0] == 0.0) {
      // [[0, 0], [0, 1]]
      auto lambda = [&](const int_t k1, const int_t k2,
                        const cvector_t &_mat)->void {
        data_[k1 | k2] = 0.0;
      };
      apply_matrix_lambda(lambda, qubit, diag);
    } else {
      // general [[z, 0], [0, 1]]
      auto lambda = [&](const int_t k1, const int_t k2,
                        const cvector_t &_mat)->void {
        const auto k = k1 | k2;
        data_[k] *= _mat[0];
      };
      apply_matrix_lambda(lambda, qubit, diag);
    }
  } else {
    // Lambda function for diagonal matrix multiplication
    auto lambda = [&](const int_t k1, const int_t k2,
                      const cvector_t &_mat)->void {
      const auto k = k1 | k2;
      data_[k] *= _mat[0];
      data_[k | BIT] *= _mat[1];
    };
    apply_matrix_lambda(lambda, qubit, diag);
  }
}

//------------------------------------------------------------------------------
// 2-6 qubit optimized matrices
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::apply_matrix2(const reg_t& qubits, const cvector_t &vmat) {
  // Check qubits is size.
  if (qubits.size() != 2) {
    throw std::runtime_error("QubitVector::apply_matrix2 called for wrong number of qubits");
  }
  // Optimized implementation
  auto sorted_qs = qubits;
  std::sort(sorted_qs.begin(), sorted_qs.end());
  auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);

  int_t END = data_size_;
  int_t step1 = BITS[sorted_qs[0]];
  int_t step2 = BITS[sorted_qs[1]];
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#ifdef _WIN32
#pragma omp for
#else
#pragma omp for collapse(3)
#endif
    for (int_t k1 = 0; k1 < END; k1 += (step2 * 2UL)) {
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
          // data_[t_i] = sum_j mat[i + 4 * j] * psi[j]
          data_[t0] = psi0 * sorted_vmat[0] + psi1 * sorted_vmat[4] + psi2 * sorted_vmat[8] + psi3 * sorted_vmat[12];
          data_[t1] = psi0 * sorted_vmat[1] + psi1 * sorted_vmat[5] + psi2 * sorted_vmat[9] + psi3 * sorted_vmat[13];
          data_[t2] = psi0 * sorted_vmat[2] + psi1 * sorted_vmat[6] + psi2 * sorted_vmat[10] + psi3 * sorted_vmat[14];
          data_[t3] = psi0 * sorted_vmat[3] + psi1 * sorted_vmat[7] + psi2 * sorted_vmat[11] + psi3 * sorted_vmat[15];
        }
      }
    }
  }
}

template <typename data_t>
void QubitVector<data_t>::apply_matrix3(const reg_t& qubits, const cvector_t &vmat) {
  // Check qubits is size.
  if (qubits.size() != 3) {
    throw std::runtime_error("QubitVector::apply_matrix3 called for wrong number of qubits");
  }
  // Optimized implementation
  auto sorted_qs = qubits;
  std::sort(sorted_qs.begin(), sorted_qs.end());
  auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
  const uint_t dim = BITS[3];

  int_t END = data_size_;
  int_t step1 = BITS[sorted_qs[0]];
  int_t step2 = BITS[sorted_qs[1]];
  int_t step3 = BITS[sorted_qs[2]];

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
    for (int_t k1 = 0; k1 < END; k1 += (step3 * 2UL)) {
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

template <typename data_t>
void QubitVector<data_t>::apply_matrix4(const reg_t& qubits, const cvector_t &vmat) {
  // Check qubits is size.
  if (qubits.size() != 4) {
    throw std::runtime_error("QubitVector::apply_matrix4 called for wrong number of qubits");
  }
  // Optimized implementation
  auto sorted_qs = qubits;
  std::sort(sorted_qs.begin(), sorted_qs.end());
  auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
  const uint_t dim = BITS[4];

  int_t END = data_size_;
  int_t step1 = BITS[sorted_qs[0]];
  int_t step2 = BITS[sorted_qs[1]];
  int_t step3 = BITS[sorted_qs[2]];
  int_t step4 = BITS[sorted_qs[3]];

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
    for (int_t k1 = 0; k1 < END; k1 += (step4 * 2UL)) {
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

template <typename data_t>
void QubitVector<data_t>::apply_matrix5(const reg_t &qubits, const cvector_t &vmat) {
  // Check qubits is size.
  if (qubits.size() != 5) {
    throw std::runtime_error("QubitVector::apply_matrix5 called for wrong number of qubits");
  }
  // Optimized implementation
  auto sorted_qs = qubits;
  std::sort(sorted_qs.begin(), sorted_qs.end());
  auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
  const uint_t dim = BITS[5];

  int_t END = data_size_;
  int_t step1 = BITS[sorted_qs[0]];
  int_t step2 = BITS[sorted_qs[1]];
  int_t step3 = BITS[sorted_qs[2]];
  int_t step4 = BITS[sorted_qs[3]];
  int_t step5 = BITS[sorted_qs[4]];

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
    for (int_t k1 = 0; k1 < END; k1 += (step5 * 2UL)) {
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

template <typename data_t>
void QubitVector<data_t>::apply_matrix6(const reg_t &qubits, const cvector_t &vmat) {
  // Check qubits is size.
  if (qubits.size() != 6) {
    throw std::runtime_error("QubitVector::apply_matrix6 called for wrong number of qubits");
  }
  // Optimized implementation
  auto sorted_qs = qubits;
  std::sort(sorted_qs.begin(), sorted_qs.end());
  auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
  const uint_t dim = BITS[6];

  int_t END = data_size_;
  int_t step1 = BITS[sorted_qs[0]];
  int_t step2 = BITS[sorted_qs[1]];
  int_t step3 = BITS[sorted_qs[2]];
  int_t step4 = BITS[sorted_qs[3]];
  int_t step5 = BITS[sorted_qs[4]];
  int_t step6 = BITS[sorted_qs[5]];

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
    for (int_t k1 = 0; k1 < END; k1 += (step6 * 2UL)) {
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

//------------------------------------------------------------------------------
// Gate-swap optimized helper functions
//------------------------------------------------------------------------------
template <typename data_t>
void QubitVector<data_t>::swap_cols_and_rows(const uint_t idx1,
                                             const uint_t idx2,
                                             cvector_t &mat,
                                             uint_t dim) const {

  uint_t mask1 = BITS[idx1];
  uint_t mask2 = BITS[idx2];

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

template <typename data_t>
cvector_t QubitVector<data_t>::sort_matrix(const reg_t& src,
                                           const reg_t& sorted,
                                           const cvector_t &mat) const {

  const uint_t N = src.size();
  const uint_t DIM = BITS[N];
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
      throw std::runtime_error("QubitVector<data_t>::sort_matrix we should not reach here");
    }
    swap_cols_and_rows(from, to, ret, DIM);
    std::swap(current[from], current[to]);
  }

  return ret;
}

/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <typename data_t>
double QubitVector<data_t>::norm() const {
  // Lambda function for norm
  auto lambda = [&](int_t k, double &val_re, double &val_im)->void {
    (void)val_im; // unused
    val_re += std::real(data_[k] * std::conj(data_[k]));
  };
  return std::real(apply_reduction_lambda(lambda));
}

template <typename data_t>
double QubitVector<data_t>::norm(const reg_t &qubits, const cvector_t &mat) const {

  const uint_t N = qubits.size();
  const uint_t DIM = BITS[N];
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Lambda function for N-qubit matrix norm
  auto lambda = [&](const indexes_t &inds, const cvector_t &_mat, 
                    double &val_re, double &val_im)->void {
    (void)val_im; // unused
    for (size_t i = 0; i < DIM; i++) {
      complex_t vi = 0;
      for (size_t j = 0; j < DIM; j++)
        vi += _mat[i + DIM * j] * data_[inds[j]];
      val_re += std::real(vi * std::conj(vi));
    }
  };
  // Use the lambda function
  return std::real(apply_matrix_reduction_lambda(lambda, qubits, mat));
}

template <typename data_t>
double QubitVector<data_t>::norm_diagonal(const reg_t &qubits, const cvector_t &mat) const {

  const uint_t N = qubits.size();
  const uint_t DIM = BITS[N];

  // Error checking
  #ifdef DEBUG
  check_vector(mat, N);
  #endif

  // Lambda function for N-qubit matrix norm
  auto lambda = [&](const indexes_t &inds, const cvector_t &_mat,
                    double &val_re, double &val_im)->void {
    (void)val_im; // unused
    for (size_t i = 0; i < DIM; i++) {
      const auto vi = _mat[i] * data_[inds[i]];
      val_re += std::real(vi * std::conj(vi));
    }
  };
  // Use the lambda function
  return std::real(apply_matrix_reduction_lambda(lambda, qubits, mat));
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
double QubitVector<data_t>::norm(const uint_t qubit, const cvector_t &mat) const {
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 2);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const int_t k1, const int_t k2,const cvector_t &_mat,
                    double &val_re, double &val_im)->void {
    (void)val_im; // unused;
    const auto k = k1 | k2;
    const auto cache0 = data_[k];
    const auto cache1 = data_[k | BITS[qubit]];
    const auto v0 = _mat[0] * cache0 + _mat[2] * cache1;
    const auto v1 = _mat[1] * cache0 + _mat[3] * cache1;
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_matrix_reduction_lambda(lambda, qubit, mat));
}

template <typename data_t>
double QubitVector<data_t>::norm_diagonal(const uint_t qubit, const cvector_t &mat) const {
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 1);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const int_t k1, const int_t k2,const cvector_t &_mat,
                    double &val_re, double &val_im)->void {
    (void)val_im; // unused;
    const auto k = k1 | k2;
    const auto v0 = _mat[0] * data_[k];
    const auto v1 = _mat[1] * data_[k | BITS[qubit]];
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_matrix_reduction_lambda(lambda, qubit, mat));
}


/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/

template <typename data_t>
double QubitVector<data_t>::probability(const uint_t outcome) const {
  const auto v = data_[outcome];
  return std::real(v * std::conj(v));
}

template <typename data_t>
rvector_t QubitVector<data_t>::probabilities() const {
  rvector_t probs(data_size_);
  const int_t END = data_size_;
  probs.assign(data_size_, 0.);

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < END; j++) {
    probs[j] = probability(j);
  }
  return probs;
}

template <typename data_t>
rvector_t QubitVector<data_t>::probabilities(const reg_t &qubits) const {

  const size_t N = qubits.size();
  const uint_t DIM = BITS[N];
  const uint_t END = BITS[num_qubits_ - N];

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  if (N == 0)
    return rvector_t({norm()});

  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
  if ((N == num_qubits_) && (qubits == qubits_sorted))
    return probabilities();

  rvector_t probs(DIM, 0.);
  for (size_t k = 0; k < END; k++) {
    auto idx = indexes(qubits, qubits_sorted, k);
    for (size_t m = 0; m < DIM; ++m) {
      probs[m] += probability(idx[m]);
    }
  }
  return probs;
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------

template <typename data_t>
rvector_t QubitVector<data_t>::probabilities(const uint_t qubit) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  // Lambda function for single qubit probs as reduction
  // p(0) stored as real part p(1) as imag part
  auto lambda = [&](const int_t k1, const int_t k2,
                    double &val_p0, double &val_p1)->void {
    const auto k = k1 | k2;
    val_p0 += probability(k);
    val_p1 += probability(k | BITS[qubit]);
  };
  auto p0p1 = apply_reduction_lambda(lambda, qubit);
  return rvector_t({std::real(p0p1), std::imag(p0p1)});
}


//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
reg_t QubitVector<data_t>::sample_measure(const std::vector<double> &rnds) const {

  const int_t END = data_size_;
  const int_t SHOTS = rnds.size();
  reg_t samples;
  samples.assign(SHOTS, 0);

  const int INDEX_SIZE = sample_measure_index_size_;
  const int_t INDEX_END = BITS[INDEX_SIZE];
  // Qubit number is below index size, loop over shots
  if (END < INDEX_END) {
    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
      #pragma omp for
      for (int_t i = 0; i < SHOTS; ++i) {
        double rnd = rnds[i];
        double p = .0;
        int_t sample;
        for (sample = 0; sample < END - 1; ++sample) {
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
    idxs.assign(INDEX_END, 0.0);
    uint_t loop = (END >> INDEX_SIZE);
    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
      #pragma omp for
      for (int_t i = 0; i < INDEX_END; ++i) {
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
      for (int_t i = 0; i < SHOTS; ++i) {
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

        for (; sample < END - 1; ++sample) {
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
template <typename data_t>
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
