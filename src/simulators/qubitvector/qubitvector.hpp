/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _qubit_vector_hpp_
#define _qubit_vector_hpp_

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

#include "framework/json.hpp"
#include "indexing.hpp" // multipartite qubit indexing

namespace QV {

// Indexing Types
using Indexing::uint_t;
using Indexing::int_t;
using Indexing::Qubit::indexes;
using Indexing::Qubit::indexes_dynamic;

// Data types
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;

//============================================================================
// QubitVector class
//============================================================================

class QubitVector {

public:

  //-----------------------------------------------------------------------
  // Constructors
  //-----------------------------------------------------------------------

  explicit QubitVector(size_t num_qubits_ = 0);
  QubitVector(const cvector_t &vec);
  QubitVector(const rvector_t &vec);

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Returns the size of the underlying n-qubit vector
  inline uint_t size() const { return num_states_;};

  // Returns the number of qubits for the current vector
  inline uint_t qubits() const { return num_qubits_;};

  // Returns a reference to the underlying complex vector
  inline cvector_t &vector() { return state_vector_;};

  // Returns a copy of the underlying complex vector
  inline cvector_t vector() const { return state_vector_;};

  // Compute the inner product with another complex vector and returns the value
  // This is equivalent to: self.dot(sconj(qv));
  complex_t inner_product(const QubitVector &qv) const;

  // Returns the norm of the current vector
  double norm() const;

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  //-----------------------------------------------------------------------
  // Configuration settings
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

  // Enable sorted qubit matrix gate optimization (Default disabled)
  inline void enable_gate_opt() {gate_opt_ = true;}

  // Disable sorted qubit matrix gate optimization
  inline void disable_gate_opt() {gate_opt_ = true;}

  // Set the sample_measure index size
  inline void set_sample_measure_index_size(int n) {sample_measure_index_size_ = n;}

  // Get the sample_measure index size
  inline int get_sample_measure_index_size() {return sample_measure_index_size_;}

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

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  template <size_t N>
  rvector_t probabilities(const std::array<uint_t, N> &qubits) const;

  // Return M sampled outcomes for Z-basis measurement of all qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  std::vector<uint_t> sample_measure(const std::vector<double> &rnds) const;

  //-----------------------------------------------------------------------
  // Norms
  //-----------------------------------------------------------------------
  // These functions return the norm <psi|A^dagger.A|psi> obtained by
  // applying a matrix A to the vector. It is equivalent to returning the
  // expectation value of A^\dagger A, and could probably be removed because
  // of this.

  // Return the norm for of the vector obtained after apply the N-qubit
  // matrix mat to the vector.
  double norm(const std::vector<uint_t> &qubits, const cvector_t &mat) const;

  double norm_diagonal(const std::vector<uint_t> &qubits, const cvector_t &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // matrix mat to the vector.
  template <size_t N>
  double norm(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;

  template <size_t N>
  double norm_diagonal(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector. The input matrix can
  // either be the diagonal of a diagonal matrix, or a column-major
  // vectorized N-qubit matrix.
  void apply_matrix(const std::vector<uint_t> &qubits, const cvector_t &mat);

  //
  void apply_diagonal_matrix(const std::vector<uint_t> &qubits,
                             const cvector_t &mat);

  // Apply a N-qubit matrix to the state vector. The input matrix can
  // either be the diagonal of a diagonal matrix, or a column-major
  // vectorized N-qubit matrix.
  template <size_t N>
  void apply_matrix(const std::array<uint_t, N> &qubits, const cvector_t &mat);

  //
  template <size_t N>
  void apply_diagonal_matrix(const std::array<uint_t, N> &qubits,
                             const cvector_t &mat);

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
  // Vector Operators
  //-----------------------------------------------------------------------

  // Assignment operator
  QubitVector &operator=(const cvector_t &vec);
  QubitVector &operator=(const rvector_t &vec);

  // Element access
  complex_t &operator[](uint_t element);
  complex_t operator[](uint_t element) const;

  // Scalar multiplication
  QubitVector &operator*=(const complex_t &lambda);
  QubitVector &operator*=(const double &lambda);
  friend QubitVector operator*(const complex_t &lambda, const QubitVector &qv);
  friend QubitVector operator*(const double &lambda, const QubitVector &qv);
  friend QubitVector operator*(const QubitVector &qv, const complex_t &lambda);
  friend QubitVector operator*(const QubitVector &qv, const double &lambda);

  // Vector addition
  QubitVector &operator+=(const QubitVector &qv);
  QubitVector operator+(const QubitVector &qv) const;

  // Vector subtraction
  QubitVector &operator-=(const QubitVector &qv);
  QubitVector operator-(const QubitVector &qv) const;

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  size_t num_qubits_;
  size_t num_states_;
  cvector_t state_vector_;

 //-----------------------------------------------------------------------
  // Config settings
  //----------------------------------------------------------------------- 
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 16;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 10; // Sample measure indexing qubit size
  bool gate_opt_ = false; // enable large-qubit optimized gates

  //-----------------------------------------------------------------------
  // State update functions with Lambda function bodies
  //-----------------------------------------------------------------------
  
  template <typename Lambda>
  void apply_matrix_lambda(const uint_t qubit,
                           const cvector_t &mat,
                           Lambda&& func);
  
  template <size_t N, typename Lambda>
  void apply_matrix_lambda(const std::array<uint_t, N> &qubits,
                           const cvector_t &mat,
                           Lambda&& func);

  template <typename Lambda>
  void apply_matrix_lambda(const std::vector<uint_t> &qubits,
                           const cvector_t &mat,
                           Lambda&& func);

  template<typename Lambda>
  complex_t apply_reduction_lambda(const uint_t qubit,
                                   const cvector_t &mat,
                                   Lambda&& func) const;

  template <size_t N, typename Lambda>
  complex_t  apply_reduction_lambda(const std::array<uint_t, N> &qubits,
                                    const cvector_t &mat,
                                    Lambda&& func) const;

  template <typename Lambda>
  complex_t  apply_reduction_lambda(const std::vector<uint_t> &qubits,
                                    const cvector_t &mat,
                                    Lambda&& func) const;

  //-----------------------------------------------------------------------
  // Matrix helper functions
  //-----------------------------------------------------------------------

  template <size_t N>
  void apply_matrix_std(const std::array<uint_t, N> &qubits,
                              const cvector_t &mat);
  
  void swap_cols_and_rows(const uint_t idx1, const uint_t idx2,
                          cvector_t &mat, uint_t dim) const;
  template <size_t N>
  cvector_t sort_matrix(const std::array<uint_t, N> &src,
                        const std::array<uint_t, N> &sorted,
                        const cvector_t &mat) const;

  // Error messages
  void check_qubit(const uint_t qubit) const;
  void check_vector(const cvector_t &diag, uint_t nqubits) const;
  void check_matrix(const cvector_t &mat, uint_t nqubits) const;
  void check_dimension(const QubitVector &qv) const;

};

//-----------------------------------------------------------------------
// JSON serialization for QubitVector class
//-----------------------------------------------------------------------
inline void to_json(json_t &js, const QubitVector&qv) {
  to_json(js, qv.vector());
}

inline void from_json(const json_t &js, QubitVector&qv) {
  cvector_t tmp;
  from_json(js, tmp);
  qv = tmp;
}

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Error Handling
//------------------------------------------------------------------------------

void QubitVector::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits_) {
    std::stringstream ss;
    ss << "QubitVector: qubit index " << qubit << " > " << num_qubits_;
    throw std::runtime_error(ss.str());
  }
}

void QubitVector::check_matrix(const cvector_t &vec, uint_t nqubits) const {
  const size_t dim = 1ULL << nqubits;
  const auto sz = vec.size();
  if (sz != dim * dim) {
    std::stringstream ss;
    ss << "QubitVector: vector size is " << sz << " != " << (dim * dim);
    throw std::runtime_error(ss.str());
  }
}

void QubitVector::check_vector(const cvector_t &vec, uint_t nqubits) const {
  const size_t dim = 1ULL << nqubits;
  const auto sz = vec.size();
  if (sz != dim) {
    std::stringstream ss;
    ss << "QubitVector: vector size is " << sz << " != " << dim;
    throw std::runtime_error(ss.str());
  }
}

void QubitVector::check_dimension(const QubitVector &qv) const {
  if (num_states_ != qv.num_states_) {
    std::stringstream ss;
    ss << "QubitVector: vectors are different size ";
    ss << num_states_ << " != " << qv.num_states_;
    throw std::runtime_error(ss.str());
  }
}

//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------

QubitVector::QubitVector(size_t num_qubits__) : num_qubits_(num_qubits__),
                                               num_states_(1ULL << num_qubits__) {
  // Set state vector
  state_vector_.assign(num_states_, 0.);
}

QubitVector::QubitVector(const cvector_t &vec) : QubitVector() {
  *this = vec;
}

QubitVector::QubitVector(const rvector_t &vec) : QubitVector() {
  *this = vec;
}


//------------------------------------------------------------------------------
// Operators
//------------------------------------------------------------------------------

// Access opertors

complex_t &QubitVector::operator[](uint_t element) {
  // Error checking
  #ifdef DEBUG
  auto size = state_vector_.size();
  if (element > size) {
    std::stringstream ss;
    ss << "QubitVector: vector index " << element << " > " << size;
    throw std::runtime_error(ss.str());
  }
  #endif
  return state_vector_[element];
}


complex_t QubitVector::operator[](uint_t element) const {
  // Error checking
  #ifdef DEBUG
  auto size = state_vector_.size();
  if (element > size) {
    std::stringstream ss;
    ss << "QubitVector: vector index " << element << " > " << size;
    throw std::runtime_error(ss.str());
  }
  #endif
  return state_vector_[element];
}

// Equal operators
QubitVector &QubitVector::operator=(const cvector_t &vec) {

  num_states_ = vec.size();
  // Get qubit number
  uint_t size = num_states_;
  num_qubits_ = 0;
  while (size >>= 1) ++num_qubits_;

  // Error handling
  #ifdef DEBUG
    if (num_states_ != 1ULL << num_qubits_) {
      std::stringstream ss;
      ss << "QubitVector: input vector is not a multi-qubit vector.";
      throw std::runtime_error(ss.str());
    }
  #endif
  // Set state_vector_
  state_vector_ = vec;
  return *this;
}

QubitVector &QubitVector::operator=(const rvector_t &vec) {

  num_states_ = vec.size();
  // Get qubit number
  uint_t size = num_states_;
  num_qubits_ = 0;
  while (size >>= 1) ++num_qubits_;

  // Error handling
  #ifdef DEBUG
    if (num_states_ != 1ULL << num_qubits_) {
      std::stringstream ss;
      ss << "QubitVector: input vector is not a multi-qubit vector.";
      throw std::runtime_error(ss.str());
    }
  #endif
  // Set state_vector_
  state_vector_.clear();
  state_vector_.reserve(size);
  for (const auto& v: vec)
    state_vector_.push_back(v);
  return *this;
}

// Scalar multiplication
QubitVector &QubitVector::operator*=(const complex_t &lambda) {
const int_t end = num_states_;    // end for k loop
#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++)
      state_vector_[k] *= lambda;
  } // end omp parallel
  return *this;
}

QubitVector &QubitVector::operator*=(const double &lambda) {
  *this *= complex_t(lambda);
  return *this;
}

QubitVector operator*(const complex_t &lambda, const QubitVector &qv) {
  QubitVector ret = qv;
  ret *= lambda;
  return ret;
}

QubitVector operator*(const QubitVector &qv, const complex_t &lambda) {
  return lambda * qv;
}

QubitVector operator*(const double &lambda, const QubitVector &qv) {
  return complex_t(lambda) * qv;
}

QubitVector operator*(const QubitVector &qv, const double &lambda) {
  return lambda * qv;
}

// Vector addition

QubitVector &QubitVector::operator+=(const QubitVector &qv) {
  // Error checking
#ifdef DEBUG
  check_dimension(qv);
#endif
  const int_t end = num_states_;    // end for k loop
  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++)
      state_vector_[k] += qv.state_vector_[k];
  } // end omp parallel
  return *this;
}

QubitVector QubitVector::operator+(const QubitVector &qv) const{
  QubitVector ret = *this;
  ret += qv;
  return ret;
}

// Vector subtraction

QubitVector &QubitVector::operator-=(const QubitVector &qv) {
  // Error checking
#ifdef DEBUG
  check_dimension(qv);
#endif
  const int_t end = num_states_;    // end for k loop
  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++)
      state_vector_[k] -= qv.state_vector_[k];
  } // end omp parallel
  return *this;
}

QubitVector QubitVector::operator-(const QubitVector &qv) const{
  QubitVector ret = *this;
  ret -= qv;
  return ret;
}


//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

void QubitVector::initialize() {
  state_vector_.assign(num_states_, 0.);
  state_vector_[0] = 1.;
}

double QubitVector::norm() const {
  double val = 0;
  const int_t end = num_states_;    // end for k loop
  #pragma omp parallel reduction(+:val) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++)
      val += std::real(state_vector_[k] * std::conj(state_vector_[k]));
  } // end omp parallel
  return val;
}

complex_t QubitVector::inner_product(const QubitVector &qv) const {
  // Error checking
#ifdef DEBUG
  check_dimension(qv);
#endif

double z_re = 0., z_im = 0.;
const int_t end = num_states_;    // end for k loop
#pragma omp parallel reduction(+:z_re, z_im) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++) {
      const complex_t z = state_vector_[k] * std::conj(qv.state_vector_[k]);
      z_re += std::real(z);
      z_im += std::imag(z);
    }
  } // end omp parallel
  return complex_t(z_re, z_im);
}

/*******************************************************************************
 *
 * CONFIG SETTINGS
 *
 ******************************************************************************/

void QubitVector::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

void QubitVector::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

/*******************************************************************************
 *
 * LAMBDA FUNCTION TEMPLATES
 *
 ******************************************************************************/

// Single qubit
template<typename Lambda>
void QubitVector::apply_matrix_lambda(const uint_t qubit,
                                      const cvector_t &mat,
                                      Lambda&& func) {

  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t end1 = num_states_;    // end for k1 loop
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
        std::forward<Lambda>(func)(mat, k1, k2, end2);
      }
  }
}

// Static N-qubit
template<size_t N, typename Lambda>
void QubitVector::apply_matrix_lambda(const std::array<uint_t, N> &qs,
                                      const cvector_t &mat,
                                      Lambda&& func) {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states_ >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes(qs, qubits_sorted, k);
      std::forward<Lambda>(func)(mat, inds);
    }
  }
}

// Dynamic N-qubit
template<typename Lambda>
void QubitVector::apply_matrix_lambda(const std::vector<uint_t> &qubits,
                                      const cvector_t &mat,
                                      Lambda&& func) {

  const auto N = qubits.size();
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const int_t end = num_states_ >> N;
  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qubits, qubits_sorted, N, k);
      std::forward<Lambda>(func)(mat, inds);
    }
  }
}

//------------------------------------------------------------------------------
// Reductions
//------------------------------------------------------------------------------

template<typename Lambda>
complex_t QubitVector::apply_reduction_lambda(const uint_t qubit,
                                              const cvector_t &mat,
                                              Lambda &&func) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t end1 = num_states_;    // end for k1 loop
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
        std::forward<Lambda>(func)(mat, val_re, val_im, k1, k2, end2);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

// Static N-qubit
template<size_t N, typename Lambda>
complex_t QubitVector::apply_reduction_lambda(const std::array<uint_t, N> &qs,
                                              const cvector_t &mat,
                                              Lambda&& func) const {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states_ >> N;
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
      const auto inds = indexes(qs, qubits_sorted, k);
      std::forward<Lambda>(func)(mat, val_re, val_im, inds);
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

// Dynamic N-qubit
template<typename Lambda>
complex_t QubitVector::apply_reduction_lambda(const std::vector<uint_t> &qubits,
                                              const cvector_t &mat,
                                              Lambda&& func) const {

  const auto N = qubits.size();
  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const int_t end = num_states_ >> N;
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
      std::forward<Lambda>(func)(mat, val_re, val_im, inds);
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


/*******************************************************************************
 *
 * GENERAL MATRIX MULTIPLICATION
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Static N
//------------------------------------------------------------------------------

template <size_t N>
void QubitVector::apply_matrix(const std::array<uint_t, N> &qs, const cvector_t &mat) {
  apply_matrix_std<N>(qs, mat);
}

template <size_t N>
void QubitVector::apply_matrix_std(const std::array<uint_t, N> &qs,
                                   const cvector_t &mat) {
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Lambda function for N-qubit matrix multiplication
  auto lambda = [&](const cvector_t &mat,
                    const std::array<uint_t, 1ULL << N> &inds)->void {
    const uint_t dim = 1ULL << N;
    std::array<complex_t, dim> cache;
    for (size_t i = 0; i < dim; i++) {
      const auto ii = inds[i];
      cache[i] = state_vector_[ii];
      state_vector_[ii] = 0.;
    }
    // update state vector
    for (size_t i = 0; i < dim; i++)
      for (size_t j = 0; j < dim; j++)
        state_vector_[inds[i]] += mat[i + dim * j] * cache[j];
  };
  // Use the lambda function
  apply_matrix_lambda(qs, mat, lambda);
}

template <size_t N>
void QubitVector::apply_diagonal_matrix(const std::array<uint_t, N> &qs,
                                        const cvector_t &diag) {

  // Error checking
  #ifdef DEBUG
  check_vector(diag, N);
  #endif

  // Lambda function for N-qubit matrix multiplication
  auto lambda = [&](const cvector_t &mat,
                    const std::array<uint_t, 1ULL << N> &inds)->void {
    const uint_t dim = 1ULL << N;
    for (size_t i = 0; i < dim; i++) {
      state_vector_[inds[i]] *= mat[i];
    }
  };

  // Use the lambda function
  apply_matrix_lambda(qs, diag, lambda);
}


//------------------------------------------------------------------------------
// Single-qubit
//------------------------------------------------------------------------------

template <>
void QubitVector::apply_matrix(const std::array<uint_t, 1> &qs,
                               const cvector_t &mat) {
  // Lambda function for single-qubit matrix multiplication
  auto lambda = [&](const cvector_t &mat, const int_t &k1, const int_t &k2,
                    const int_t &end2)->void {
    const auto k = k1 | k2;
    const auto cache0 = state_vector_[k];
    const auto cache1 = state_vector_[k | end2];
    state_vector_[k] = mat[0] * cache0 + mat[2] * cache1;
    state_vector_[k | end2] = mat[1] * cache0 + mat[3] * cache1;
  };
  apply_matrix_lambda(qs[0], mat, lambda);
}

template <>
void QubitVector::apply_diagonal_matrix(const std::array<uint_t, 1> &qs,
                                        const cvector_t &diag) {
  // Lambda function for diagonal matrix multiplication
  auto lambda = [&](const cvector_t &mat, const int_t &k1, const int_t &k2,
                    const int_t &end2)->void {
    const auto k = k1 | k2;
    state_vector_[k] *= mat[0];
    state_vector_[k | end2] *= mat[1];
  };
  apply_matrix_lambda(qs[0], diag, lambda);
}

//------------------------------------------------------------------------------
// Gate-swap optimized
//------------------------------------------------------------------------------

template <>
void QubitVector::apply_matrix(const std::array<uint_t, 2> &qubits,
                               const cvector_t &vmat) {
  if (gate_opt_ == false) {
    apply_matrix_std(qubits, vmat);
  } else {
  // Optimized implementation
    auto sorted_qs = qubits;
    std::sort(sorted_qs.begin(), sorted_qs.end());
    auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);

    int_t end = num_states_;
    int_t step1 = (1ULL << sorted_qs[0]);
    int_t step2 = (1ULL << sorted_qs[1]);
  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
  #ifdef _WIN32
  #pragma omp for
  #else
  #pragma omp for collapse(3) schedule(static)
  #endif
      for (int_t k1 = 0; k1 < end; k1 += (step2 * 2UL)) {
        for (int_t k2 = 0; k2 < step2; k2 += (step1 * 2UL)) {
          for (int_t k3 = 0; k3 < step1; k3++) {
            int_t t0 = k1 | k2 | k3;
            int_t t1 = t0 | step1;
            int_t t2 = t0 | step2;
            int_t t3 = t2 | step1;

            const complex_t psi0 = state_vector_[t0];
            const complex_t psi1 = state_vector_[t1];
            const complex_t psi2 = state_vector_[t2];
            const complex_t psi3 = state_vector_[t3];

            state_vector_[t0] = psi0 * sorted_vmat[0] + psi1 * sorted_vmat[1] + psi2 * sorted_vmat[2] + psi3 * sorted_vmat[3];
            state_vector_[t1] = psi0 * sorted_vmat[4] + psi1 * sorted_vmat[5] + psi2 * sorted_vmat[6] + psi3 * sorted_vmat[7];
            state_vector_[t2] = psi0 * sorted_vmat[8] + psi1 * sorted_vmat[9] + psi2 * sorted_vmat[10] + psi3 * sorted_vmat[11];
            state_vector_[t3] = psi0 * sorted_vmat[12] + psi1 * sorted_vmat[13] + psi2 * sorted_vmat[14] + psi3 * sorted_vmat[15];
          }
        }
      }
    }
  }
}

template <>
void QubitVector::apply_matrix(const std::array<uint_t, 3> &qubits,
                               const cvector_t &vmat) {
  if (gate_opt_ == false) {
    apply_matrix_std(qubits, vmat);
  } else {
    // Optimized implementation
    auto sorted_qs = qubits;
    std::sort(sorted_qs.begin(), sorted_qs.end());
    auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
    const uint_t dim = 1ULL << 3;

    int_t end = num_states_;
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
  #pragma omp for collapse(4) schedule(static)
  #endif
      for (int_t k1 = 0; k1 < end; k1 += (step3 * 2UL)) {
        for (int_t k2 = 0; k2 < step3; k2 += (step2 * 2UL)) {
          for (int_t k3 = 0; k3 < step2; k3 += (step1 * 2UL)) {
            for (int_t k4 = 0; k4 < step1; k4++) {
              int_t base = k1 | k2 | k3 | k4;
              complex_t psi[8];
              for (int_t i = 0; i < 8; ++i) {
                psi[i] = state_vector_[base | masks[i]];
                state_vector_[base | masks[i]] = 0.;
              }
              for (size_t i = 0; i < 8; ++i)
                for (size_t j = 0; j < 8; ++j)
                  state_vector_[base | masks[i]] += psi[j] * sorted_vmat[j * dim + i];
            }
          }
        }
      }
    }
  }
}

template <>
void QubitVector::apply_matrix(const std::array<uint_t, 4> &qubits,
                               const cvector_t &vmat) {
  if (gate_opt_ == false) {
    apply_matrix_std(qubits, vmat);
  } else {
    // Optimized implementation
    auto sorted_qs = qubits;
    std::sort(sorted_qs.begin(), sorted_qs.end());
    auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
    const uint_t dim = 1ULL << 4;

    int_t end = num_states_;
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
  #pragma omp for collapse(5) schedule(static)
  #endif
      for (int_t k1 = 0; k1 < end; k1 += (step4 * 2UL)) {
        for (int_t k2 = 0; k2 < step4; k2 += (step3 * 2UL)) {
          for (int_t k3 = 0; k3 < step3; k3 += (step2 * 2UL)) {
            for (int_t k4 = 0; k4 < step2; k4 += (step1 * 2UL)) {
              for (int_t k5 = 0; k5 < step1; k5++) {
                int_t base = k1 | k2 | k3 | k4 | k5;
                complex_t psi[16];
                for (int_t i = 0; i < 16; ++i) {
                  psi[i] = state_vector_[base | masks[i]];
                  state_vector_[base | masks[i]] = 0.;
                }
                for (size_t i = 0; i < 16; ++i)
                  for (size_t j = 0; j < 16; ++j)
                    state_vector_[base | masks[i]] += psi[j] * sorted_vmat[j * dim + i];
              }
            }
          }
        }
      }
    }
  }
}

template <>
void QubitVector::apply_matrix(const std::array<uint_t, 5> &qubits,
                               const cvector_t &vmat) {
  if (gate_opt_ == false) {
    apply_matrix_std(qubits, vmat);
  } else {
    // Optimized implementation
    auto sorted_qs = qubits;
    std::sort(sorted_qs.begin(), sorted_qs.end());
    auto sorted_vmat = sort_matrix(qubits, sorted_qs, vmat);
    const uint_t dim = 1ULL << 5;

    int_t end = num_states_;
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
  #pragma omp for collapse(6) schedule(static)
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
                    psi[i] = state_vector_[base | masks[i]];
                    state_vector_[base | masks[i]] = 0.;
                  }
                  for (size_t i = 0; i < 32; ++i)
                    for (size_t j = 0; j < 32; ++j)
                      state_vector_[base | masks[i]] += psi[j] * sorted_vmat[j * dim + i];
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
void QubitVector::swap_cols_and_rows(const uint_t idx1, const uint_t idx2,
                                     cvector_t &mat, uint_t dim) const {

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

template <size_t N>
cvector_t QubitVector::sort_matrix(const std::array<uint_t, N> &src,
                                   const std::array<uint_t, N> &sorted,
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
      ss << "QubitVector::sort_matrix we should not reach here";
      throw std::runtime_error(ss.str());
    }
    swap_cols_and_rows(from, to, ret, dim);

    uint_t cache = current[from];
    current[from] = current[to];
    current[to] = cache;
  }

  return ret;
}


//------------------------------------------------------------------------------
// Dynamic N
//------------------------------------------------------------------------------

void QubitVector::apply_matrix(const std::vector<uint_t> &qs, const cvector_t &mat) {
  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    apply_matrix<1>(std::array<uint_t, 1>({{qs[0]}}), mat);
    break;
  case 2:
    apply_matrix<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), mat);
    break;
  case 3:
    apply_matrix<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), mat);
    break;
  case 4:
    apply_matrix<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), mat);
    break;
  case 5:
    apply_matrix<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), mat);
    break;
  default: {
    // Default case using dynamic indexing
    // Error checking
    #ifdef DEBUG
    check_vector(mat, 2 * N);
    #endif

    // Lambda function for N-qubit matrix multiplication
    auto lambda = [&](const cvector_t &mat,
                      const std::vector<uint_t> &inds)->void {
      const uint_t dim = 1ULL << qs.size();
      std::vector<complex_t> cache(dim);
      for (size_t i = 0; i < dim; i++) {
        const auto ii = inds[i];
        cache[i] = state_vector_[ii];
        state_vector_[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
          state_vector_[inds[i]] += mat[i + dim * j] * cache[j];
    };
    // Use the lambda function
    apply_matrix_lambda(qs, mat, lambda);
  } // end default
    break;
  } // end switch
}


void QubitVector::apply_diagonal_matrix(const std::vector<uint_t> &qs,
                               const cvector_t &diag) {
  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    apply_diagonal_matrix<1>(std::array<uint_t, 1>({{qs[0]}}), diag);
    break;
  case 2:
    apply_diagonal_matrix<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), diag);
    break;
  case 3:
    apply_diagonal_matrix<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), diag);
    break;
  case 4:
    apply_diagonal_matrix<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), diag);
    break;
  case 5:
    apply_diagonal_matrix<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), diag);
    break;
  default: {
    // Default case using dynamic indexing
    // Error checking
    #ifdef DEBUG
    check_vector(mat, N);
    #endif

    // Lambda function for N-qubit matrix multiplication
    auto lambda = [&](const cvector_t &mat,
                      const std::vector<uint_t> &inds)->void {
      const uint_t dim = 1ULL << qs.size();
      for (size_t i = 0; i < dim; i++)
            state_vector_[inds[i]] *= mat[i];
    };
    // Use the lambda function
    apply_matrix_lambda(qs, diag, lambda);
  } // end default
    break;
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

void QubitVector::apply_x(const uint_t qubit) {
  // Lambda function for optimized Pauli-X gate
  auto lambda = [&](const cvector_t &mat, const int_t &k1, const int_t &k2,
                    const int_t &end2)->void {
    (void)mat; // unused
    const auto i0 = k1 | k2;
    const auto i1 = i0 | end2;
    const complex_t cache = state_vector_[i0];
    state_vector_[i0] = state_vector_[i1]; // mat(0,1)
    state_vector_[i1] = cache;    // mat(1,0)
  };
  apply_matrix_lambda(qubit, {}, lambda);
}


void QubitVector::apply_y(const uint_t qubit) {
  // Lambda function for optimized Pauli-Y gate
  const complex_t I(0., 1.);
  auto lambda = [&](const cvector_t &mat, const int_t &k1, const int_t &k2,
                    const int_t &end2)->void {
    (void)mat; // unused
    const auto i0 = k1 | k2;
    const auto i1 = i0 | end2;
    const complex_t cache = state_vector_[i0];
    state_vector_[i0] = -I * state_vector_[i1]; // mat(0,1)
    state_vector_[i1] = I * cache;     // mat(1,0)
  };
  apply_matrix_lambda(qubit, {}, lambda);
}


void QubitVector::apply_z(const uint_t qubit) {
  // Lambda function for optimized Pauli-Z gate
  const complex_t minus_one(-1.0, 0.0);
  auto lambda = [&](const cvector_t &mat, const int_t &k1, const int_t &k2,
                    const int_t &end2)->void {
    (void)mat; // unused
    state_vector_[k1 | k2 | end2] *= minus_one;
  };
  apply_matrix_lambda(qubit, {}, lambda);
}

//------------------------------------------------------------------------------
// Two-qubit gates
//------------------------------------------------------------------------------

void QubitVector::apply_cnot(const uint_t qubit_ctrl, const uint_t qubit_trgt) {
  // Lambda function for CNOT gate
  auto lambda = [&](const cvector_t &mat,
                    const std::array<uint_t, 1ULL << 2> &inds)->void {
    (void)mat; //unused
    const complex_t cache = state_vector_[inds[3]];
    state_vector_[inds[3]] = state_vector_[inds[1]];
    state_vector_[inds[1]] = cache;
  };
  // Use the lambda function
  apply_matrix_lambda(std::array<uint_t, 2>({{qubit_ctrl, qubit_trgt}}), {}, lambda);
}


void QubitVector::apply_swap(const uint_t qubit0, const uint_t qubit1) {
  // Lambda function for SWAP gate
  auto lambda = [&](const cvector_t &mat,
                    const std::array<uint_t, 1ULL << 2> &inds)->void {
    (void)mat; //unused
    const complex_t cache = state_vector_[inds[2]];
      state_vector_[inds[2]] = state_vector_[inds[1]];
      state_vector_[inds[1]] = cache;
  };
  // Use the lambda function
  apply_matrix_lambda(std::array<uint_t, 2>({{qubit0, qubit1}}), {}, lambda);
}


void QubitVector::apply_cz(const uint_t qubit_ctrl, const uint_t qubit_trgt) {

  // Lambda function for CZ gate
  auto lambda = [&](const cvector_t &mat,
                    const std::array<uint_t, 1ULL << 2> &inds)->void {
    (void)mat; //unused
    state_vector_[inds[3]] *= -1.;
  };
  // Use the lambda function
  apply_matrix_lambda(std::array<uint_t, 2>({{qubit_ctrl, qubit_trgt}}), {}, lambda);
}

//------------------------------------------------------------------------------
// Three-qubit gates
//------------------------------------------------------------------------------

void QubitVector::apply_toffoli(const uint_t qubit_ctrl0,
                                const uint_t qubit_ctrl1,
                                const uint_t qubit_trgt) {
  // Lambda function for Toffoli gate
  auto lambda = [&](const cvector_t &mat,
                    const std::array<uint_t, 1ULL << 3> &inds)->void {
    (void)mat; //unused
    const complex_t cache = state_vector_[inds[7]];
    state_vector_[inds[7]] = state_vector_[inds[3]];
    state_vector_[inds[3]] = cache;
  };
  // Use the lambda function
  std::array<uint_t, 3> qubits = {{qubit_ctrl0, qubit_ctrl1, qubit_trgt}};
  apply_matrix_lambda(qubits, {}, lambda);
}

/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/


//------------------------------------------------------------------------------
// Static N
//------------------------------------------------------------------------------

template <size_t N>
double QubitVector::norm(const std::array<uint_t, N> &qs,
                         const cvector_t &mat) const {
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Lambda function for N-qubit matrix norm
  auto lambda = [&](const cvector_t &mat, double &val_re, double &val_im,
                    const std::array<uint_t, 1ULL << N> &inds)->void {
    (void)val_im; // unused
    const uint_t dim = 1ULL << N;
    for (size_t i = 0; i < dim; i++) {
      complex_t vi = 0;
      for (size_t j = 0; j < dim; j++)
        vi += mat[i + dim * j] * state_vector_[inds[j]];
      val_re += std::real(vi * std::conj(vi));
    }
  };
  // Use the lambda function
  return std::real(apply_reduction_lambda(qs, mat, lambda));
}


template <size_t N>
double QubitVector::norm_diagonal(const std::array<uint_t, N> &qs,
                                  const cvector_t &mat) const {

  // Error checking
  #ifdef DEBUG
  check_vector(mat, N);
  #endif

  // Lambda function for N-qubit matrix norm
  auto lambda = [&](const cvector_t &mat, double &val_re, double &val_im,
                    const std::array<uint_t, 1ULL << N> &inds)->void {
    (void)val_im; // unused
    const uint_t dim = 1ULL << N;
    for (size_t i = 0; i < dim; i++) {
      const auto vi = mat[i] * state_vector_[inds[i]];
      val_re += std::real(vi * std::conj(vi));
    }
  };
  // Use the lambda function
  return std::real(apply_reduction_lambda(qs, mat, lambda));
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------

template <>
double QubitVector::norm(const std::array<uint_t, 1> &qubits,
                         const cvector_t &mat) const {
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 2);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const cvector_t &mat, double &val_re, double &val_im,
                    const int_t &k1, const int_t &k2, const int_t &end2)->void {
    (void)val_im; // unused;
    const auto k = k1 | k2;
    const auto cache0 = state_vector_[k];
    const auto cache1 = state_vector_[k | end2];
    const auto v0 = mat[0] * cache0 + mat[2] * cache1;
    const auto v1 = mat[1] * cache0 + mat[3] * cache1;
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_reduction_lambda(qubits[0], mat, lambda));
}

template <>
double QubitVector::norm_diagonal(const std::array<uint_t, 1> &qubits,
                                  const cvector_t &mat) const {
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 1);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const cvector_t &mat, double &val_re, double &val_im,
                    const int_t &k1, const int_t &k2, const int_t &end2)->void {
    (void)val_im; // unused;
    const auto k = k1 | k2;
    const auto v0 = mat[0] * state_vector_[k];
    const auto v1 = mat[1] * state_vector_[k | end2];
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_reduction_lambda(qubits[0], mat, lambda));
}

//------------------------------------------------------------------------------
// Dynamic N
//------------------------------------------------------------------------------

double QubitVector::norm(const std::vector<uint_t> &qs, const cvector_t &mat) const {

  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    return norm<1>(std::array<uint_t, 1>({{qs[0]}}), mat);
  case 2:
    return norm<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), mat);
  case 3:
    return norm<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), mat);
  case 4:
    return norm<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), mat);
  case 5:
    return norm<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), mat);
  default: {

    // Error checking
    const uint_t N = qs.size();
    const uint_t dim = 1ULL << N;
    #ifdef DEBUG
    check_vector(mat, 2 * N);
    #endif

    // Lambda function for N-qubit matrix norm
    auto lambda = [&](const cvector_t &mat, double &val_re, double &val_im,
                      const std::vector<uint_t> &inds)->void {
      (void)val_im; // unused
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++)
          vi += mat[i + dim * j] * state_vector_[inds[j]];
        val_re += std::real(vi * std::conj(vi));
      }
    };
    // Use the lambda function
    return std::real(apply_reduction_lambda(qs, mat, lambda));
  } // end default
  } // end switch
}

double QubitVector::norm_diagonal(const std::vector<uint_t> &qs, const cvector_t &mat) const {

  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    return norm_diagonal<1>(std::array<uint_t, 1>({{qs[0]}}), mat);
  case 2:
    return norm_diagonal<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), mat);
  case 3:
    return norm_diagonal<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), mat);
  case 4:
    return norm_diagonal<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), mat);
  case 5:
    return norm_diagonal<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), mat);
  default: {
    // Default dynamic index case
    // Error checking
    const uint_t N = qs.size();
    const uint_t dim = 1ULL << N;
    #ifdef DEBUG
    check_vector(mat, N);
    #endif

    // Lambda function for N-qubit matrix norm
    auto lambda = [&](const cvector_t &mat, double &val_re, double &val_im,
                      const std::vector<uint_t> &inds)->void {
      (void)val_im; // unused
      for (size_t i = 0; i < dim; i++) {
        const auto vi = mat[i] * state_vector_[inds[i]];
        val_re += std::real(vi * std::conj(vi));
      }
    };
    // Use the lambda function
    return std::real(apply_reduction_lambda(qs, mat, lambda));
  } // end default
  } // end switch
}

/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/

double QubitVector::probability(const uint_t outcome) const {
  const auto v = state_vector_[outcome];
  return std::real(v * std::conj(v));
}

rvector_t QubitVector::probabilities() const {
  rvector_t probs;
  probs.reserve(num_states_);
  const int_t end = state_vector_.size();
  for (int_t j=0; j < end; j++) {
    probs.push_back(probability(j));
  }
  return probs;
}

//------------------------------------------------------------------------------
// Static N-qubit
//------------------------------------------------------------------------------

template <size_t N>
rvector_t QubitVector::probabilities(const std::array<uint_t, N> &qs) const {

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
    const auto idx = indexes<N>(qs, qubits_sorted, k);
    for (size_t m = 0; m < dim; ++m) {
      probs[m] += probability(idx[m]);
    }
  }
  return probs;
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------

template <>
rvector_t QubitVector::probabilities(const std::array<uint_t, 1> &qs) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t end1 = num_states_;    // end for k1 loop
  const int_t end2 = 1LL << qs[0]; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
  double p0 = 0., p1 = 0.;
#pragma omp parallel reduction(+:p0, p1) if (num_qubits_ > omp_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        p0 += probability(k);
        p1 += probability(k | end2);
      }
  } // end omp parallel
  return rvector_t({p0, p1});
}

//------------------------------------------------------------------------------
// Dynamic N-qubit
//------------------------------------------------------------------------------

rvector_t QubitVector::probabilities(const std::vector<uint_t> &qs) const {

  // Special cases using faster static indexing
  const uint_t N = qs.size();
  switch (N) {
  case 0:
    return rvector_t({norm()});
  case 1:
    return probabilities<1>(std::array<uint_t, 1>({{qs[0]}}));
  case 2:
    return probabilities<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}));
  case 3:
    return probabilities<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}));
  case 4:
    return probabilities<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}));
  case 5:
    return probabilities<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}));
  default: {
    // else
    // Error checking
    #ifdef DEBUG
    for (const auto &qubit : qs)
      check_qubit(qubit);
    #endif

    const uint_t dim = 1ULL << N;
    const uint_t end = (1ULL << num_qubits_) >> N;
    auto qss = qs;
    std::sort(qss.begin(), qss.end());
    if ((N == num_qubits_) && (qss == qs))
      return probabilities();
    const auto &qubits_sorted = qss;
    rvector_t probs(dim, 0.);

    for (size_t k = 0; k < end; k++) {
      const auto idx = indexes_dynamic(qs, qubits_sorted, N, k);
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

std::vector<uint_t> QubitVector::sample_measure(const std::vector<double> &rnds) const {

  const int_t end = num_states_;
  const int_t shots = rnds.size();
  std::vector<uint_t> samples;
  samples.reserve(shots);

  const int index_size = sample_measure_index_size_;
  const int_t index_end = 1LL << index_size;
  #pragma omp parallel if (omp_threads_ > 1) num_threads(omp_threads_)
  {
    // Qubit number is below index size, loop over shots
    if (end < index_end){
      #pragma omp for
      for (int_t i = 0; i < shots; ++i) {
        double rnd = rnds[i];
        double p = .0;
        int_t sample;
        for (sample = 0; sample < end - 1; ++sample) {
          p += std::real(std::conj(state_vector_[sample]) * state_vector_[sample]);
          if (rnd < p)
            break;
        }
        samples.push_back(sample);
      }
    } 
    // Qubit number is above index size, loop over index blocks
    else {
      // Initialize indexes
      std::vector<double> indexes;
      indexes.assign((1<<index_size), .0);
      uint_t loop = (end >> index_size);

      #pragma omp for
      for (uint_t i = 0; i < (1 << index_size); ++i) {
        uint_t base = loop * i;
        double total = .0;
        double p = .0;
        for (uint_t j = 0; j < loop; ++j) {
          uint_t k = base | j;
          p = std::real(std::conj(state_vector_[k]) * state_vector_[k]);
          total += p;
        }
        indexes[i] = total;
      }

      #pragma omp for
      for (uint_t i = 0; i < shots; ++i) {
        double rnd = rnds[i];
        double p = .0;
        int_t sample = 0;
        for (uint_t j = 0; j < indexes.size(); ++j) {
          if (rnd < (p + indexes[j])) {
            break;
          }
          p += indexes[j];
          sample += loop;
        }

        for (; sample < end - 1; ++sample) {
          p += std::real(std::conj(state_vector_[sample]) * state_vector_[sample]);
          if (rnd < p){
            break;
          }
        }
        samples.push_back(sample);
      }
    } // end omp parallel
  }
  return samples;
}


//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// QubitVector ostream overload
//------------------------------------------------------------------------------

inline std::ostream &operator<<(std::ostream &out, const QV::QubitVector&qv) {
  out << qv.vector();
  return out;
}

//------------------------------------------------------------------------------
#endif // end module
