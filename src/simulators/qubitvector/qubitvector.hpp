/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    qubit_vector.hpp
 * @brief   QubitVector class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _qubit_vector_hpp_
#define _qubit_vector_hpp_

//#define DEBUG // error checking
#define OPT

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "framework/matrix.hpp"

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
using cmatrix_t = matrix<complex_t>;

/*******************************************************************************
 *
 * QubitVector Class
 *
 ******************************************************************************/

class QubitVector {

public:

  /************************
   * Constructors
   ************************/

  explicit QubitVector(size_t num_qubits = 0);
  QubitVector(const cvector_t &vec);
  QubitVector(const rvector_t &vec);

  /************************
   * Utility
   ************************/

  inline uint_t size() const { return num_states;};
  inline uint_t qubits() const { return num_qubits;};
  inline cvector_t &vector() { return state_vector;};
  inline cvector_t vector() const { return state_vector;};

  complex_t dot(const QubitVector &qv) const;
  complex_t inner_product(const QubitVector &qv) const;
  double norm() const;
  void conj();
  void renormalize();
  void initialize();
  void initialize_plus();

  void set_omp_threads(int n);
  void set_omp_threshold(int n);

  /**************************************
   * Z-measurement outcome probabilities
   **************************************/

  rvector_t probabilities() const;
  rvector_t probabilities(const uint_t qubit) const;
  rvector_t probabilities(const std::vector<uint_t> &qubits) const;
  template <size_t N>
  rvector_t probabilities(const std::array<uint_t, N> &qubits) const;

  /**************************************
   * Z-measurement outcome probability
   **************************************/
  double probability(const uint_t outcome) const;
  double probability(const uint_t qubit, const uint_t outcome) const;
  double probability(const std::vector<uint_t> &qubits, const uint_t outcome) const;
  template <size_t N>
  double probability(const std::array<uint_t, N> &qubits, const uint_t outcome) const;

  /************************
   * Apply Matrices
   ************************/

  // Matrices vectorized in column-major
  void apply_matrix(const uint_t qubit, const cvector_t &mat);
  void apply_matrix(const uint_t qubit0, const uint_t qubit1, const cvector_t &mat);
  void apply_matrix(const std::vector<uint_t> &qubits, const cvector_t &mat);
  template <size_t N>
  void apply_matrix(const std::array<uint_t, N> &qubits, const cvector_t &mat);

  // Specialized gates
  void apply_cnot(const uint_t qctrl, const uint_t qtrgt);
  void apply_cz(const uint_t q0, const uint_t q1);
  void apply_swap(const uint_t q0, const uint_t q1);
  void apply_x(const uint_t qubit);
  void apply_y(const uint_t qubit);
  void apply_z(const uint_t qubit);

  /************************
   * Norms
   ************************/

  double norm(const uint_t qubit, const cvector_t &mat) const;
  double norm(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  template <size_t N>
  double norm(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;

  /************************
   * Expectation Values
   ************************/

  complex_t expectation_value(const uint_t qubit, const cvector_t &mat) const;
  complex_t expectation_value(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  template <size_t N>
  complex_t expectation_value(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;


  /************************
   * Operators
   ************************/

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
  size_t num_qubits;
  size_t num_states;
  cvector_t state_vector;

  // OMP
  uint_t omp_threads = 1;     // Disable multithreading by default
  uint_t omp_threshold = 16;  // Qubit threshold for multithreading when enabled

  /************************
   * Matrix-mult Helper functions
   ************************/

  void apply_matrix_col_major(const uint_t qubit, const cvector_t &mat);
  void apply_matrix_col_major(const std::vector<uint_t> &qubits, const cvector_t &mat);
  template <size_t N>
  void apply_matrix_col_major(const std::array<uint_t, N> &qubits, const cvector_t &mat);

  void apply_matrix_col_major_2(const std::vector<uint_t> &qubits, const cvector_t &mat);
  void apply_matrix_col_major_3(const std::vector<uint_t> &qubits, const cvector_t &mat);
  void apply_matrix_col_major_4(const std::vector<uint_t> &qubits, const cvector_t &mat);
  void apply_matrix_col_major_5(const std::vector<uint_t> &qubits, const cvector_t &mat);

  cmatrix_t generate_matrix(const uint_t qubit_size, const cvector_t &mat) const;
  void swap_cols_and_rows(const uint_t idx1, const uint_t idx2, cmatrix_t &mat) const;
  cmatrix_t sort_matrix(const std::vector<uint_t> &original, const std::vector<uint_t> &sorted, const cmatrix_t &mat) const;

  void apply_matrix_diagonal(const uint_t qubit, const cvector_t &mat);
  void apply_matrix_diagonal(const std::vector<uint_t> &qubits, const cvector_t &mat);
  template <size_t N>
  void apply_matrix_diagonal(const std::array<uint_t, N> &qubits, const cvector_t &mat);

  // Norms
  // Warning: no test coverage
  double norm_matrix(const uint_t qubit, const cvector_t &mat) const;
  double norm_matrix_diagonal(const uint_t qubit, const cvector_t &mat) const;
  double norm_matrix(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  double norm_matrix_diagonal(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  template <size_t N>
  double norm_matrix(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;
  template <size_t N>
  double norm_matrix_diagonal(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;

  // Matrix Expectation Values
  // Warning: no test coverage
  complex_t expectation_value_matrix(const uint_t qubit, const cvector_t &mat) const;
  complex_t expectation_value_matrix_diagonal(const uint_t qubit, const cvector_t &mat) const;
  complex_t expectation_value_matrix(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  complex_t expectation_value_matrix_diagonal(const std::vector<uint_t> &qubits, const cvector_t &mat) const;
  template <size_t N>
  complex_t expectation_value_matrix(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;
  template <size_t N>
  complex_t expectation_value_matrix_diagonal(const std::array<uint_t, N> &qubits, const cvector_t &mat) const;

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
  if (qubit + 1 > num_qubits) {
    std::stringstream ss;
    ss << "QubitVector: qubit index " << qubit << " > " << num_qubits;
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
  if (num_states != qv.num_states) {
    std::stringstream ss;
    ss << "QubitVector: vectors are different size ";
    ss << num_states << " != " << qv.num_states;
    throw std::runtime_error(ss.str());
  }
}

//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------

QubitVector::QubitVector(size_t num_qubits_) : num_qubits(num_qubits_),
                                               num_states(1ULL << num_qubits_) {
  // Set state vector
  state_vector.assign(num_states, 0.);
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
  auto size = state_vector.size();
  if (element > size) {
    std::stringstream ss;
    ss << "QubitVector: vector index " << element << " > " << size;
    throw std::runtime_error(ss.str());
  }
  #endif
  return state_vector[element];
}


complex_t QubitVector::operator[](uint_t element) const {
  // Error checking
  #ifdef DEBUG
  auto size = state_vector.size();
  if (element > size) {
    std::stringstream ss;
    ss << "QubitVector: vector index " << element << " > " << size;
    throw std::runtime_error(ss.str());
  }
  #endif
  return state_vector[element];
}

// Equal operators
QubitVector &QubitVector::operator=(const cvector_t &vec) {

  num_states = vec.size();
  // Get qubit number
  uint_t size = num_states;
  num_qubits = 0;
  while (size >>= 1) ++num_qubits;

  // Error handling
  #ifdef DEBUG
    if (num_states != 1ULL << num_qubits) {
      std::stringstream ss;
      ss << "QubitVector: input vector is not a multi-qubit vector.";
      throw std::runtime_error(ss.str());
    }
  #endif
  // Set state_vector
  state_vector = vec;
  return *this;
}

QubitVector &QubitVector::operator=(const rvector_t &vec) {

  num_states = vec.size();
  // Get qubit number
  uint_t size = num_states;
  num_qubits = 0;
  while (size >>= 1) ++num_qubits;

  // Error handling
  #ifdef DEBUG
    if (num_states != 1ULL << num_qubits) {
      std::stringstream ss;
      ss << "QubitVector: input vector is not a multi-qubit vector.";
      throw std::runtime_error(ss.str());
    }
  #endif
  // Set state_vector
  state_vector.clear();
  state_vector.reserve(size);
  for (const auto& v: vec)
    state_vector.push_back(v);
  return *this;
}

// Scalar multiplication
QubitVector &QubitVector::operator*=(const complex_t &lambda) {
const int_t end = num_states;    // end for k loop
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++)
      state_vector[k] *= lambda;
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
  const int_t end = num_states;    // end for k loop
  #pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++)
      state_vector[k] += qv.state_vector[k];
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
  const int_t end = num_states;    // end for k loop
  #pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++)
      state_vector[k] -= qv.state_vector[k];
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
  state_vector.assign(num_states, 0.);
  state_vector[0] = 1.;
}

void QubitVector::initialize_plus() {
  complex_t val(1.0 / std::pow(2, 0.5 * num_qubits), 0.);
  state_vector.assign(num_states, val);
}

void QubitVector::conj() {
  const int_t end = num_states;    // end for k loop
  #pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++) {
      state_vector[k] = std::conj(state_vector[k]);
    }
  } // end omp parallel
}

complex_t QubitVector::dot(const QubitVector &qv) const {
  // Error checking
#ifdef DEBUG
  check_dimension(qv);
#endif

// split variable for OpenMP 2.0 compatible reduction
double z_re = 0., z_im = 0.;
const int_t end = num_states;    // end for k loop
#pragma omp parallel reduction(+:z_re, z_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++) {
      const complex_t z = state_vector[k] * qv.state_vector[k];
      z_re += std::real(z);
      z_im += std::imag(z);
    }
  } // end omp parallel
  return complex_t(z_re, z_im);
}

complex_t QubitVector::inner_product(const QubitVector &qv) const {
  // Error checking
#ifdef DEBUG
  check_dimension(qv);
#endif

double z_re = 0., z_im = 0.;
const int_t end = num_states;    // end for k loop
#pragma omp parallel reduction(+:z_re, z_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++) {
      const complex_t z = state_vector[k] * std::conj(qv.state_vector[k]);
      z_re += std::real(z);
      z_im += std::imag(z);
    }
  } // end omp parallel
  return complex_t(z_re, z_im);
}

void QubitVector::renormalize() {
  double nrm = norm();
  #ifdef DEBUG
    if ((nrm > 0.) == false) {
      std::stringstream ss;
      ss << "QubitVector: vector has norm zero.";
      throw std::runtime_error(ss.str());
    }
  #endif
  if (nrm > 0.) {
    const double scale = 1.0 / std::sqrt(nrm);
    *this *= scale;
  }
}

void QubitVector::set_omp_threads(int n) {
  if (n > 0)
    omp_threads = n;
}

void QubitVector::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold = n;
}


/*******************************************************************************
 *
 * SINGLE QUBIT OPERATIONS
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Matrix multiplication
//------------------------------------------------------------------------------

void QubitVector::apply_matrix(const uint_t qubit, const cvector_t &mat) {
  if (mat.size() == 2)
    apply_matrix_diagonal(qubit, mat);
  else
    apply_matrix_col_major(qubit, mat);
}

void QubitVector::apply_matrix_col_major(const uint_t qubit, const cvector_t &mat) {
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2);
  #endif

  const int_t end1 = num_states;   // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;   // step for k1 loop

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto cache0 = state_vector[k];
        const auto cache1 = state_vector[k | end2];
        state_vector[k] = mat[0] * cache0 + mat[2] * cache1;
        state_vector[k | end2] = mat[1] * cache0 + mat[3] * cache1;
      }
  }
}

void QubitVector::apply_matrix_diagonal(const uint_t qubit, const cvector_t &diag) {

  // Error checking
  #ifdef DEBUG
  check_vector(diag, 1);
  check_qubit(qubit);
  #endif

  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        state_vector[k] *= diag[0];
        state_vector[k | end2] *= diag[1];
      }
  }
}

void QubitVector::apply_x(const uint_t qubit) {

  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  // Optimized ideal Pauli-X gate
  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        const auto i0 = k1 | k2;
        const auto i1 = i0 | end2;
        const complex_t cache = state_vector[i0];
        state_vector[i0] = state_vector[i1]; // mat(0,1)
        state_vector[i1] = cache;    // mat(1,0)
      }
  }
}

void QubitVector::apply_y(const uint_t qubit) {
 // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  // Optimized ideal Pauli-Y gate
  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
  const complex_t I(0., 1.);
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        const auto i0 = k1 | k2;
        const auto i1 = i0 | end2;
        const complex_t cache = state_vector[i0];
        state_vector[i0] = -I * state_vector[i1]; // mat(0,1)
        state_vector[i1] = I * cache;     // mat(1,0)
      }
  }
}

void QubitVector::apply_z(const uint_t qubit) {

  // Error checking
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  // Optimized ideal Pauli-Z gate
  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
  const complex_t minus_one(-1.0, 0.0);
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        state_vector[k1 | k2 | end2] *= minus_one;
      }
  }
}


//------------------------------------------------------------------------------
// Norm
//------------------------------------------------------------------------------


double QubitVector::norm() const {
  double val = 0;
  const int_t end = num_states;    // end for k loop
  #pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++)
      val += std::real(state_vector[k] * std::conj(state_vector[k]));
  } // end omp parallel
  return val;
}

double QubitVector::norm(const uint_t qubit, const cvector_t &mat) const {
  if (mat.size() == 2)
      return norm_matrix_diagonal(qubit, mat);
  else
      return norm_matrix(qubit, mat);
}

double QubitVector::norm_matrix(const uint_t qubit, const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  check_vector(mat, 2);
  #endif

  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
  double val = 0.;
#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto cache0 = state_vector[k];
        const auto cache1 = state_vector[k | end2];
        const auto v0 = mat[0] * cache0 + mat[2] * cache1;
        const auto v1 = mat[1] * cache0 + mat[3] * cache1;
        val += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
      }
  } // end omp parallel
  return val;
}

double QubitVector::norm_matrix_diagonal(const uint_t qubit, const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  check_vector(mat, 1);
  #endif

  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
  double val = 0.;
#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto v0 = mat[0] * state_vector[k];
        const auto v1 = mat[1] * state_vector[k | end2];
        val += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
      }
  } // end omp parallel
  return val;
}


//------------------------------------------------------------------------------
// Expectation Values
//------------------------------------------------------------------------------

complex_t QubitVector::expectation_value(const uint_t qubit, const cvector_t &mat) const {
  if (mat.size() == 2)
    return expectation_value_matrix_diagonal(qubit, mat);
  else
    return expectation_value_matrix(qubit, mat);
}

complex_t QubitVector::expectation_value_matrix(const uint_t qubit, const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  check_vector(mat, 2);
  #endif

  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto cache0 = state_vector[k];
        const auto cache1 = state_vector[k | end2];
        const auto v0 = mat[0] * cache0 + mat[2] * cache1;
        const auto v1 = mat[1] * cache0 + mat[3] * cache1;
        const complex_t val = v0 * std::conj(cache0) + v1 * std::conj(cache1);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

complex_t QubitVector::expectation_value_matrix_diagonal(const uint_t qubit, const cvector_t &mat) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  check_vector(mat, 1);
  #endif

  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++) {
        const auto k = k1 | k2;
        const auto cache0 = state_vector[k];
        const auto cache1 = state_vector[k | end2];
        const complex_t val = mat[0] * cache0 * std::conj(cache0) + mat[1] * cache1 * std::conj(cache1);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


/*******************************************************************************
 *
 * STATIC N-QUBIT OPERATIONS (N known at compile time)
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Matrix multiplication
//------------------------------------------------------------------------------

void QubitVector::apply_matrix(const uint_t qubit0, const uint_t qubit1,
                               const cvector_t &mat) {
  if (mat.size() == 4)
    apply_matrix_diagonal<2>({{qubit0, qubit1}}, mat);
  else
#ifdef OPT
    apply_matrix_col_major_2( {{qubit0, qubit1}}, mat);
#else
    apply_matrix_col_major<2>( {{qubit0, qubit1}}, mat);
#endif
}

template <size_t N>
void QubitVector::apply_matrix(const std::array<uint_t, N> &qs, const cvector_t &mat) {
  if (mat.size() == (1ULL << N)) {
    apply_matrix_diagonal<N>(qs, mat);
  } else {
#ifdef OPT
    switch(N) {
    case 1:
      apply_matrix_col_major(qs[0], mat);
    case 2:
      apply_matrix_col_major_2({{qs[0], qs[1]}}, mat);
    case 3:
      apply_matrix_col_major_3({{qs[0], qs[1], qs[2]}}, mat);
    case 4:
      apply_matrix_col_major_4({{qs[0], qs[1], qs[2], qs[3]}}, mat);
    case 5:
      apply_matrix_col_major_5({{qs[0], qs[1], qs[2], qs[3], qs[4]}}, mat);
    default:
      apply_matrix_col_major<N>(qs, mat);
    }
#else
    apply_matrix_col_major<N>(qs, mat);
#endif
  }
}

void QubitVector::apply_matrix_col_major_2(const std::vector<uint_t> &qubits, const cvector_t &vmat) {
  auto sorted_qs = qubits;
  std::sort(sorted_qs.begin(), sorted_qs.end());
  cmatrix_t mat = sort_matrix(qubits, sorted_qs, generate_matrix(2, vmat));

  int_t end = num_states;
  int_t step1 = (1ULL << sorted_qs[0]);
  int_t step2 = (1ULL << sorted_qs[1]);
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
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

          const complex_t psi0 = state_vector[t0];
          const complex_t psi1 = state_vector[t1];
          const complex_t psi2 = state_vector[t2];
          const complex_t psi3 = state_vector[t3];

          state_vector[t0] = psi0 * mat(0, 0) + psi1 * mat(0, 1) + psi2 * mat(0, 2) + psi3 * mat(0, 3);
          state_vector[t1] = psi0 * mat(1, 0) + psi1 * mat(1, 1) + psi2 * mat(1, 2) + psi3 * mat(1, 3);
          state_vector[t2] = psi0 * mat(2, 0) + psi1 * mat(2, 1) + psi2 * mat(2, 2) + psi3 * mat(2, 3);
          state_vector[t3] = psi0 * mat(3, 0) + psi1 * mat(3, 1) + psi2 * mat(3, 2) + psi3 * mat(3, 3);
        }
      }
    }
  }
}

void QubitVector::apply_matrix_col_major_3(const std::vector<uint_t> &qubits, const cvector_t &vmat) {
  auto sorted_qs = qubits;
  std::sort(sorted_qs.begin(), sorted_qs.end());
  cmatrix_t mat = sort_matrix(qubits, sorted_qs, generate_matrix(3, vmat));

  int_t end = num_states;
  int_t step1 = (1ULL << sorted_qs[0]);
  int_t step2 = (1ULL << sorted_qs[1]);
  int_t step3 = (1ULL << sorted_qs[2]);
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
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
            int_t t0 = k1 | k2 | k3 | k4;
            int_t t1 = t0 | step1;
            int_t t2 = t0 | step2;
            int_t t3 = t2 | step1;
            int_t t4 = t0 | step3;
            int_t t5 = t4 | step1;
            int_t t6 = t4 | step2;
            int_t t7 = t6 | step1;

            const complex_t psi0 = state_vector[t0];
            const complex_t psi1 = state_vector[t1];
            const complex_t psi2 = state_vector[t2];
            const complex_t psi3 = state_vector[t3];
            const complex_t psi4 = state_vector[t4];
            const complex_t psi5 = state_vector[t5];
            const complex_t psi6 = state_vector[t6];
            const complex_t psi7 = state_vector[t7];

            state_vector[t0] = psi0 * mat(0, 0) + psi1 * mat(0, 1) + psi2 * mat(0, 2) + psi3 * mat(0, 3) + psi4 * mat(0, 4) + psi5 * mat(0, 5) + psi6 * mat(0, 6) + psi7 * mat(0, 7);
            state_vector[t1] = psi0 * mat(1, 0) + psi1 * mat(1, 1) + psi2 * mat(1, 2) + psi3 * mat(1, 3) + psi4 * mat(1, 4) + psi5 * mat(1, 5) + psi6 * mat(1, 6) + psi7 * mat(1, 7);
            state_vector[t2] = psi0 * mat(2, 0) + psi1 * mat(2, 1) + psi2 * mat(2, 2) + psi3 * mat(2, 3) + psi4 * mat(2, 4) + psi5 * mat(2, 5) + psi6 * mat(2, 6) + psi7 * mat(2, 7);
            state_vector[t3] = psi0 * mat(3, 0) + psi1 * mat(3, 1) + psi2 * mat(3, 2) + psi3 * mat(3, 3) + psi4 * mat(3, 4) + psi5 * mat(3, 5) + psi6 * mat(3, 6) + psi7 * mat(3, 7);
            state_vector[t4] = psi0 * mat(4, 0) + psi1 * mat(4, 1) + psi2 * mat(4, 2) + psi3 * mat(4, 3) + psi4 * mat(4, 4) + psi5 * mat(4, 5) + psi6 * mat(4, 6) + psi7 * mat(4, 7);
            state_vector[t5] = psi0 * mat(5, 0) + psi1 * mat(5, 1) + psi2 * mat(5, 2) + psi3 * mat(5, 3) + psi4 * mat(5, 4) + psi5 * mat(5, 5) + psi6 * mat(5, 6) + psi7 * mat(5, 7);
            state_vector[t6] = psi0 * mat(6, 0) + psi1 * mat(6, 1) + psi2 * mat(6, 2) + psi3 * mat(6, 3) + psi4 * mat(6, 4) + psi5 * mat(6, 5) + psi6 * mat(6, 6) + psi7 * mat(6, 7);
            state_vector[t7] = psi0 * mat(7, 0) + psi1 * mat(7, 1) + psi2 * mat(7, 2) + psi3 * mat(7, 3) + psi4 * mat(7, 4) + psi5 * mat(7, 5) + psi6 * mat(7, 6) + psi7 * mat(7, 7);
          }
        }
      }
    }
  }
}

void QubitVector::apply_matrix_col_major_4(const std::vector<uint_t> &qubits, const cvector_t &vmat) {
  auto sorted_qs = qubits;
  std::sort(sorted_qs.begin(), sorted_qs.end());
  cmatrix_t mat = sort_matrix(qubits, sorted_qs, generate_matrix(4, vmat));

  int_t end = num_states;
  int_t step1 = (1ULL << sorted_qs[0]);
  int_t step2 = (1ULL << sorted_qs[1]);
  int_t step3 = (1ULL << sorted_qs[2]);
  int_t step4 = (1ULL << sorted_qs[3]);
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
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
              int_t t0 = k1 | k2 | k3 | k4 | k5;
              int_t t1 = t0 | step1;
              int_t t2 = t0 | step2;
              int_t t3 = t0 | step2 | step1;
              int_t t4 = t0 | step3;
              int_t t5 = t0 | step3 | step1;
              int_t t6 = t0 | step3 | step2;
              int_t t7 = t0 | step3 | step2 | step1;
              int_t t8 = t0 | step4;
              int_t t9 = t0 | step4 | step1;
              int_t t10 = t0 | step4 | step2;
              int_t t11 = t0 | step4 | step2 | step1;
              int_t t12 = t0 | step4 | step3;
              int_t t13 = t0 | step4 | step3 | step1;
              int_t t14 = t0 | step4 | step3 | step2;
              int_t t15 = t0 | step4 | step3 | step2 | step1;

              //cout << t0 << "," << t1 << "," << t2 << "," << t3 << "," << t4 << "," << t5 << "," << t6 << "," << t7 << std::endl;

              const complex_t psi0 = state_vector[t0];
              const complex_t psi1 = state_vector[t1];
              const complex_t psi2 = state_vector[t2];
              const complex_t psi3 = state_vector[t3];
              const complex_t psi4 = state_vector[t4];
              const complex_t psi5 = state_vector[t5];
              const complex_t psi6 = state_vector[t6];
              const complex_t psi7 = state_vector[t7];
              const complex_t psi8 = state_vector[t8];
              const complex_t psi9 = state_vector[t9];
              const complex_t psi10 = state_vector[t10];
              const complex_t psi11 = state_vector[t11];
              const complex_t psi12 = state_vector[t12];
              const complex_t psi13 = state_vector[t13];
              const complex_t psi14 = state_vector[t14];
              const complex_t psi15 = state_vector[t15];

              state_vector[t0] = //
                  psi0 * mat(0, 0) + psi1 * mat(0, 1) + psi2 * mat(0, 2) + psi3 * mat(0, 3) + psi4 * mat(0, 4) + psi5 * mat(0, 5) + psi6 * mat(0, 6) + psi7 * mat(0, 7) + //
                      psi8 * mat(0, 8) + psi9 * mat(0, 9) + psi10 * mat(0, 10) + psi11 * mat(0, 11) + psi12 * mat(0, 12) + psi13 * mat(0, 13) + psi14 * mat(0, 14) + psi15 * mat(0, 15);
              state_vector[t1] = //
                  psi0 * mat(1, 0) + psi1 * mat(1, 1) + psi2 * mat(1, 2) + psi3 * mat(1, 3) + psi4 * mat(1, 4) + psi5 * mat(1, 5) + psi6 * mat(1, 6) + psi7 * mat(1, 7) + //
                      psi8 * mat(1, 8) + psi9 * mat(1, 9) + psi10 * mat(1, 10) + psi11 * mat(1, 11) + psi12 * mat(1, 12) + psi13 * mat(1, 13) + psi14 * mat(1, 14) + psi15 * mat(1, 15);
              state_vector[t2] = //
                  psi0 * mat(2, 0) + psi1 * mat(2, 1) + psi2 * mat(2, 2) + psi3 * mat(2, 3) + psi4 * mat(2, 4) + psi5 * mat(2, 5) + psi6 * mat(2, 6) + psi7 * mat(2, 7) + //
                      psi8 * mat(2, 8) + psi9 * mat(2, 9) + psi10 * mat(2, 10) + psi11 * mat(2, 11) + psi12 * mat(2, 12) + psi13 * mat(2, 13) + psi14 * mat(2, 14) + psi15 * mat(2, 15);
              state_vector[t3] = //
                  psi0 * mat(3, 0) + psi1 * mat(3, 1) + psi2 * mat(3, 2) + psi3 * mat(3, 3) + psi4 * mat(3, 4) + psi5 * mat(3, 5) + psi6 * mat(3, 6) + psi7 * mat(3, 7) + //
                      psi8 * mat(3, 8) + psi9 * mat(3, 9) + psi10 * mat(3, 10) + psi11 * mat(3, 11) + psi12 * mat(3, 12) + psi13 * mat(3, 13) + psi14 * mat(3, 14) + psi15 * mat(3, 15);
              state_vector[t4] = //
                  psi0 * mat(4, 0) + psi1 * mat(4, 1) + psi2 * mat(4, 2) + psi3 * mat(4, 3) + psi4 * mat(4, 4) + psi5 * mat(4, 5) + psi6 * mat(4, 6) + psi7 * mat(4, 7) + //
                      psi8 * mat(4, 8) + psi9 * mat(4, 9) + psi10 * mat(4, 10) + psi11 * mat(4, 11) + psi12 * mat(4, 12) + psi13 * mat(4, 13) + psi14 * mat(4, 14) + psi15 * mat(4, 15);
              state_vector[t5] = //
                  psi0 * mat(5, 0) + psi1 * mat(5, 1) + psi2 * mat(5, 2) + psi3 * mat(5, 3) + psi4 * mat(5, 4) + psi5 * mat(5, 5) + psi6 * mat(5, 6) + psi7 * mat(5, 7) + //
                      psi8 * mat(5, 8) + psi9 * mat(5, 9) + psi10 * mat(5, 10) + psi11 * mat(5, 11) + psi12 * mat(5, 12) + psi13 * mat(5, 13) + psi14 * mat(5, 14) + psi15 * mat(5, 15);
              state_vector[t6] = //
                  psi0 * mat(6, 0) + psi1 * mat(6, 1) + psi2 * mat(6, 2) + psi3 * mat(6, 3) + psi4 * mat(6, 4) + psi5 * mat(6, 5) + psi6 * mat(6, 6) + psi7 * mat(6, 7) + //
                      psi8 * mat(6, 8) + psi9 * mat(6, 9) + psi10 * mat(6, 10) + psi11 * mat(6, 11) + psi12 * mat(6, 12) + psi13 * mat(6, 13) + psi14 * mat(6, 14) + psi15 * mat(6, 15);
              state_vector[t7] = //
                  psi0 * mat(7, 0) + psi1 * mat(7, 1) + psi2 * mat(7, 2) + psi3 * mat(7, 3) + psi4 * mat(7, 4) + psi5 * mat(7, 5) + psi6 * mat(7, 6) + psi7 * mat(7, 7) + //
                      psi8 * mat(7, 8) + psi9 * mat(7, 9) + psi10 * mat(7, 10) + psi11 * mat(7, 11) + psi12 * mat(7, 12) + psi13 * mat(7, 13) + psi14 * mat(7, 14) + psi15 * mat(7, 15);
              state_vector[t8] = //
                  psi0 * mat(8, 0) + psi1 * mat(8, 1) + psi2 * mat(8, 2) + psi3 * mat(8, 3) + psi4 * mat(8, 4) + psi5 * mat(8, 5) + psi6 * mat(8, 6) + psi7 * mat(8, 7) + //
                      psi8 * mat(8, 8) + psi9 * mat(8, 9) + psi10 * mat(8, 10) + psi11 * mat(8, 11) + psi12 * mat(8, 12) + psi13 * mat(8, 13) + psi14 * mat(8, 14) + psi15 * mat(8, 15);
              state_vector[t9] = //
                  psi0 * mat(9, 0) + psi1 * mat(9, 1) + psi2 * mat(9, 2) + psi3 * mat(9, 3) + psi4 * mat(9, 4) + psi5 * mat(9, 5) + psi6 * mat(9, 6) + psi7 * mat(9, 7) + //
                      psi8 * mat(9, 8) + psi9 * mat(9, 9) + psi10 * mat(9, 10) + psi11 * mat(9, 11) + psi12 * mat(9, 12) + psi13 * mat(9, 13) + psi14 * mat(9, 14) + psi15 * mat(9, 15);
              state_vector[t10] = //
                  psi0 * mat(10, 0) + psi1 * mat(10, 1) + psi2 * mat(10, 2) + psi3 * mat(10, 3) + psi4 * mat(10, 4) + psi5 * mat(10, 5) + psi6 * mat(10, 6) + psi7 * mat(10, 7) + //
                      psi8 * mat(10, 8) + psi9 * mat(10, 9) + psi10 * mat(10, 10) + psi11 * mat(10, 11) + psi12 * mat(10, 12) + psi13 * mat(10, 13) + psi14 * mat(10, 14) + psi15 * mat(10, 15);
              state_vector[t11] = //
                  psi0 * mat(11, 0) + psi1 * mat(11, 1) + psi2 * mat(11, 2) + psi3 * mat(11, 3) + psi4 * mat(11, 4) + psi5 * mat(11, 5) + psi6 * mat(11, 6) + psi7 * mat(11, 7) + //
                      psi8 * mat(11, 8) + psi9 * mat(11, 9) + psi10 * mat(11, 10) + psi11 * mat(11, 11) + psi12 * mat(11, 12) + psi13 * mat(11, 13) + psi14 * mat(11, 14) + psi15 * mat(11, 15);
              state_vector[t12] = //
                  psi0 * mat(12, 0) + psi1 * mat(12, 1) + psi2 * mat(12, 2) + psi3 * mat(12, 3) + psi4 * mat(12, 4) + psi5 * mat(12, 5) + psi6 * mat(12, 6) + psi7 * mat(12, 7) + //
                      psi8 * mat(12, 8) + psi9 * mat(12, 9) + psi10 * mat(12, 10) + psi11 * mat(12, 11) + psi12 * mat(12, 12) + psi13 * mat(12, 13) + psi14 * mat(12, 14) + psi15 * mat(12, 15);
              state_vector[t13] = //
                  psi0 * mat(13, 0) + psi1 * mat(13, 1) + psi2 * mat(13, 2) + psi3 * mat(13, 3) + psi4 * mat(13, 4) + psi5 * mat(13, 5) + psi6 * mat(13, 6) + psi7 * mat(13, 7) + //
                      psi8 * mat(13, 8) + psi9 * mat(13, 9) + psi10 * mat(13, 10) + psi11 * mat(13, 11) + psi12 * mat(13, 12) + psi13 * mat(13, 13) + psi14 * mat(13, 14) + psi15 * mat(13, 15);
              state_vector[t14] = //
                  psi0 * mat(14, 0) + psi1 * mat(14, 1) + psi2 * mat(14, 2) + psi3 * mat(14, 3) + psi4 * mat(14, 4) + psi5 * mat(14, 5) + psi6 * mat(14, 6) + psi7 * mat(14, 7) + //
                      psi8 * mat(14, 8) + psi9 * mat(14, 9) + psi10 * mat(14, 10) + psi11 * mat(14, 11) + psi12 * mat(14, 12) + psi13 * mat(14, 13) + psi14 * mat(14, 14) + psi15 * mat(14, 15);
              state_vector[t15] = //
                  psi0 * mat(15, 0) + psi1 * mat(15, 1) + psi2 * mat(15, 2) + psi3 * mat(15, 3) + psi4 * mat(15, 4) + psi5 * mat(15, 5) + psi6 * mat(15, 6) + psi7 * mat(15, 7) + //
                      psi8 * mat(15, 8) + psi9 * mat(15, 9) + psi10 * mat(15, 10) + psi11 * mat(15, 11) + psi12 * mat(15, 12) + psi13 * mat(15, 13) + psi14 * mat(15, 14) + psi15 * mat(15, 15);
            }
          }
        }
      }
    }
  }
}

void QubitVector::apply_matrix_col_major_5(const std::vector<uint_t> &qubits, const cvector_t &vmat) {
  auto sorted_qs = qubits;
  std::sort(sorted_qs.begin(), sorted_qs.end());
  cmatrix_t mat = sort_matrix(qubits, sorted_qs, generate_matrix(5, vmat));

  int_t end = num_states;
  int_t step1 = (1ULL << sorted_qs[0]);
  int_t step2 = (1ULL << sorted_qs[1]);
  int_t step3 = (1ULL << sorted_qs[2]);
  int_t step4 = (1ULL << sorted_qs[3]);
  int_t step5 = (1ULL << sorted_qs[4]);
#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
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
                int_t t0 = k1 | k2 | k3 | k4 | k5 | k6;
                int_t t1 = t0 | step1;
                int_t t2 = t0 | step2;
                int_t t3 = t0 | step2 | step1;
                int_t t4 = t0 | step3;
                int_t t5 = t0 | step3 | step1;
                int_t t6 = t0 | step3 | step2;
                int_t t7 = t0 | step3 | step2 | step1;
                int_t t8 = t0 | step4;
                int_t t9 = t0 | step4 | step1;
                int_t t10 = t0 | step4 | step2;
                int_t t11 = t0 | step4 | step2 | step1;
                int_t t12 = t0 | step4 | step3;
                int_t t13 = t0 | step4 | step3 | step1;
                int_t t14 = t0 | step4 | step3 | step2;
                int_t t15 = t0 | step4 | step3 | step2 | step1;
                int_t t16 = t0 | step5;
                int_t t17 = t0 | step5 | step1;
                int_t t18 = t0 | step5 | step2;
                int_t t19 = t0 | step5 | step2 | step1;
                int_t t20 = t0 | step5 | step3;
                int_t t21 = t0 | step5 | step3 | step1;
                int_t t22 = t0 | step5 | step3 | step2;
                int_t t23 = t0 | step5 | step3 | step2 | step1;
                int_t t24 = t0 | step5 | step4;
                int_t t25 = t0 | step5 | step4 | step1;
                int_t t26 = t0 | step5 | step4 | step2;
                int_t t27 = t0 | step5 | step4 | step2 | step1;
                int_t t28 = t0 | step5 | step4 | step3;
                int_t t29 = t0 | step5 | step4 | step3 | step1;
                int_t t30 = t0 | step5 | step4 | step3 | step2;
                int_t t31 = t0 | step5 | step4 | step3 | step2 | step1;

                //cout << t0 << "," << t1 << "," << t2 << "," << t3 << "," << t4 << "," << t5 << "," << t6 << "," << t7 << std::endl;

                const complex_t psi0 = state_vector[t0];
                const complex_t psi1 = state_vector[t1];
                const complex_t psi2 = state_vector[t2];
                const complex_t psi3 = state_vector[t3];
                const complex_t psi4 = state_vector[t4];
                const complex_t psi5 = state_vector[t5];
                const complex_t psi6 = state_vector[t6];
                const complex_t psi7 = state_vector[t7];
                const complex_t psi8 = state_vector[t8];
                const complex_t psi9 = state_vector[t9];
                const complex_t psi10 = state_vector[t10];
                const complex_t psi11 = state_vector[t11];
                const complex_t psi12 = state_vector[t12];
                const complex_t psi13 = state_vector[t13];
                const complex_t psi14 = state_vector[t14];
                const complex_t psi15 = state_vector[t15];
                const complex_t psi16 = state_vector[t16];
                const complex_t psi17 = state_vector[t17];
                const complex_t psi18 = state_vector[t18];
                const complex_t psi19 = state_vector[t19];
                const complex_t psi20 = state_vector[t20];
                const complex_t psi21 = state_vector[t21];
                const complex_t psi22 = state_vector[t22];
                const complex_t psi23 = state_vector[t23];
                const complex_t psi24 = state_vector[t24];
                const complex_t psi25 = state_vector[t25];
                const complex_t psi26 = state_vector[t26];
                const complex_t psi27 = state_vector[t27];
                const complex_t psi28 = state_vector[t28];
                const complex_t psi29 = state_vector[t29];
                const complex_t psi30 = state_vector[t30];
                const complex_t psi31 = state_vector[t31];

                state_vector[t0] = psi0 * mat(0, 0) + psi1 * mat(0, 1) + psi2 * mat(0, 2) + psi3 * mat(0, 3) + psi4 * mat(0, 4) + psi5 * mat(0, 5) + psi6 * mat(0, 6) + psi7 * mat(0, 7) + //
                    psi8 * mat(0, 8) + psi9 * mat(0, 9) + psi10 * mat(0, 10) + psi11 * mat(0, 11) + psi12 * mat(0, 12) + psi13 * mat(0, 13) + psi14 * mat(0, 14) + psi15 * mat(0, 15) + //
                    psi16 * mat(0, 16) + psi17 * mat(0, 17) + psi18 * mat(0, 18) + psi19 * mat(0, 19) + psi20 * mat(0, 20) + psi21 * mat(0, 21) + psi22 * mat(0, 22) + psi23 * mat(0, 23) + //
                    psi24 * mat(0, 24) + psi25 * mat(0, 25) + psi26 * mat(0, 26) + psi27 * mat(0, 27) + psi28 * mat(0, 28) + psi29 * mat(0, 29) + psi30 * mat(0, 30) + psi31 * mat(0, 31);
                state_vector[t1] = //
                    psi0 * mat(1, 0) + psi1 * mat(1, 1) + psi2 * mat(1, 2) + psi3 * mat(1, 3) + psi4 * mat(1, 4) + psi5 * mat(1, 5) + psi6 * mat(1, 6) + psi7 * mat(1, 7) + //
                        psi8 * mat(1, 8) + psi9 * mat(1, 9) + psi10 * mat(1, 10) + psi11 * mat(1, 11) + psi12 * mat(1, 12) + psi13 * mat(1, 13) + psi14 * mat(1, 14) + psi15 * mat(1, 15) + //
                        psi16 * mat(1, 16) + psi17 * mat(1, 17) + psi18 * mat(1, 18) + psi19 * mat(1, 19) + psi20 * mat(1, 20) + psi21 * mat(1, 21) + psi22 * mat(1, 22) + psi23 * mat(1, 23) + //
                        psi24 * mat(1, 24) + psi25 * mat(1, 25) + psi26 * mat(1, 26) + psi27 * mat(1, 27) + psi28 * mat(1, 28) + psi29 * mat(1, 29) + psi30 * mat(1, 30) + psi31 * mat(1, 31);
                state_vector[t2] = //
                    psi0 * mat(2, 0) + psi1 * mat(2, 1) + psi2 * mat(2, 2) + psi3 * mat(2, 3) + psi4 * mat(2, 4) + psi5 * mat(2, 5) + psi6 * mat(2, 6) + psi7 * mat(2, 7) + //
                        psi8 * mat(2, 8) + psi9 * mat(2, 9) + psi10 * mat(2, 10) + psi11 * mat(2, 11) + psi12 * mat(2, 12) + psi13 * mat(2, 13) + psi14 * mat(2, 14) + psi15 * mat(2, 15) + //
                        psi16 * mat(2, 16) + psi17 * mat(2, 17) + psi18 * mat(2, 18) + psi19 * mat(2, 19) + psi20 * mat(2, 20) + psi21 * mat(2, 21) + psi22 * mat(2, 22) + psi23 * mat(2, 23) + //
                        psi24 * mat(2, 24) + psi25 * mat(2, 25) + psi26 * mat(2, 26) + psi27 * mat(2, 27) + psi28 * mat(2, 28) + psi29 * mat(2, 29) + psi30 * mat(2, 30) + psi31 * mat(2, 31);
                state_vector[t3] = //
                    psi0 * mat(3, 0) + psi1 * mat(3, 1) + psi2 * mat(3, 2) + psi3 * mat(3, 3) + psi4 * mat(3, 4) + psi5 * mat(3, 5) + psi6 * mat(3, 6) + psi7 * mat(3, 7) + //
                        psi8 * mat(3, 8) + psi9 * mat(3, 9) + psi10 * mat(3, 10) + psi11 * mat(3, 11) + psi12 * mat(3, 12) + psi13 * mat(3, 13) + psi14 * mat(3, 14) + psi15 * mat(3, 15) + //
                        psi16 * mat(3, 16) + psi17 * mat(3, 17) + psi18 * mat(3, 18) + psi19 * mat(3, 19) + psi20 * mat(3, 20) + psi21 * mat(3, 21) + psi22 * mat(3, 22) + psi23 * mat(3, 23) + //
                        psi24 * mat(3, 24) + psi25 * mat(3, 25) + psi26 * mat(3, 26) + psi27 * mat(3, 27) + psi28 * mat(3, 28) + psi29 * mat(3, 29) + psi30 * mat(3, 30) + psi31 * mat(3, 31);
                state_vector[t4] = //
                    psi0 * mat(4, 0) + psi1 * mat(4, 1) + psi2 * mat(4, 2) + psi3 * mat(4, 3) + psi4 * mat(4, 4) + psi5 * mat(4, 5) + psi6 * mat(4, 6) + psi7 * mat(4, 7) + //
                        psi8 * mat(4, 8) + psi9 * mat(4, 9) + psi10 * mat(4, 10) + psi11 * mat(4, 11) + psi12 * mat(4, 12) + psi13 * mat(4, 13) + psi14 * mat(4, 14) + psi15 * mat(4, 15) + //
                        psi16 * mat(4, 16) + psi17 * mat(4, 17) + psi18 * mat(4, 18) + psi19 * mat(4, 19) + psi20 * mat(4, 20) + psi21 * mat(4, 21) + psi22 * mat(4, 22) + psi23 * mat(4, 23) + //
                        psi24 * mat(4, 24) + psi25 * mat(4, 25) + psi26 * mat(4, 26) + psi27 * mat(4, 27) + psi28 * mat(4, 28) + psi29 * mat(4, 29) + psi30 * mat(4, 30) + psi31 * mat(4, 31);
                state_vector[t5] = //
                    psi0 * mat(5, 0) + psi1 * mat(5, 1) + psi2 * mat(5, 2) + psi3 * mat(5, 3) + psi4 * mat(5, 4) + psi5 * mat(5, 5) + psi6 * mat(5, 6) + psi7 * mat(5, 7) + //
                        psi8 * mat(5, 8) + psi9 * mat(5, 9) + psi10 * mat(5, 10) + psi11 * mat(5, 11) + psi12 * mat(5, 12) + psi13 * mat(5, 13) + psi14 * mat(5, 14) + psi15 * mat(5, 15) + //
                        psi16 * mat(5, 16) + psi17 * mat(5, 17) + psi18 * mat(5, 18) + psi19 * mat(5, 19) + psi20 * mat(5, 20) + psi21 * mat(5, 21) + psi22 * mat(5, 22) + psi23 * mat(5, 23) + //
                        psi24 * mat(5, 24) + psi25 * mat(5, 25) + psi26 * mat(5, 26) + psi27 * mat(5, 27) + psi28 * mat(5, 28) + psi29 * mat(5, 29) + psi30 * mat(5, 30) + psi31 * mat(5, 31);
                state_vector[t6] = //
                    psi0 * mat(6, 0) + psi1 * mat(6, 1) + psi2 * mat(6, 2) + psi3 * mat(6, 3) + psi4 * mat(6, 4) + psi5 * mat(6, 5) + psi6 * mat(6, 6) + psi7 * mat(6, 7) + //
                        psi8 * mat(6, 8) + psi9 * mat(6, 9) + psi10 * mat(6, 10) + psi11 * mat(6, 11) + psi12 * mat(6, 12) + psi13 * mat(6, 13) + psi14 * mat(6, 14) + psi15 * mat(6, 15) + //
                        psi16 * mat(6, 16) + psi17 * mat(6, 17) + psi18 * mat(6, 18) + psi19 * mat(6, 19) + psi20 * mat(6, 20) + psi21 * mat(6, 21) + psi22 * mat(6, 22) + psi23 * mat(6, 23) + //
                        psi24 * mat(6, 24) + psi25 * mat(6, 25) + psi26 * mat(6, 26) + psi27 * mat(6, 27) + psi28 * mat(6, 28) + psi29 * mat(6, 29) + psi30 * mat(6, 30) + psi31 * mat(6, 31);
                state_vector[t7] = //
                    psi0 * mat(7, 0) + psi1 * mat(7, 1) + psi2 * mat(7, 2) + psi3 * mat(7, 3) + psi4 * mat(7, 4) + psi5 * mat(7, 5) + psi6 * mat(7, 6) + psi7 * mat(7, 7) + //
                        psi8 * mat(7, 8) + psi9 * mat(7, 9) + psi10 * mat(7, 10) + psi11 * mat(7, 11) + psi12 * mat(7, 12) + psi13 * mat(7, 13) + psi14 * mat(7, 14) + psi15 * mat(7, 15) + //
                        psi16 * mat(7, 16) + psi17 * mat(7, 17) + psi18 * mat(7, 18) + psi19 * mat(7, 19) + psi20 * mat(7, 20) + psi21 * mat(7, 21) + psi22 * mat(7, 22) + psi23 * mat(7, 23) + //
                        psi24 * mat(7, 24) + psi25 * mat(7, 25) + psi26 * mat(7, 26) + psi27 * mat(7, 27) + psi28 * mat(7, 28) + psi29 * mat(7, 29) + psi30 * mat(7, 30) + psi31 * mat(7, 31);
                state_vector[t8] = //
                    psi0 * mat(8, 0) + psi1 * mat(8, 1) + psi2 * mat(8, 2) + psi3 * mat(8, 3) + psi4 * mat(8, 4) + psi5 * mat(8, 5) + psi6 * mat(8, 6) + psi7 * mat(8, 7) + //
                        psi8 * mat(8, 8) + psi9 * mat(8, 9) + psi10 * mat(8, 10) + psi11 * mat(8, 11) + psi12 * mat(8, 12) + psi13 * mat(8, 13) + psi14 * mat(8, 14) + psi15 * mat(8, 15) + //
                        psi16 * mat(8, 16) + psi17 * mat(8, 17) + psi18 * mat(8, 18) + psi19 * mat(8, 19) + psi20 * mat(8, 20) + psi21 * mat(8, 21) + psi22 * mat(8, 22) + psi23 * mat(8, 23) + //
                        psi24 * mat(8, 24) + psi25 * mat(8, 25) + psi26 * mat(8, 26) + psi27 * mat(8, 27) + psi28 * mat(8, 28) + psi29 * mat(8, 29) + psi30 * mat(8, 30) + psi31 * mat(8, 31);
                state_vector[t9] = //
                    psi0 * mat(9, 0) + psi1 * mat(9, 1) + psi2 * mat(9, 2) + psi3 * mat(9, 3) + psi4 * mat(9, 4) + psi5 * mat(9, 5) + psi6 * mat(9, 6) + psi7 * mat(9, 7) + //
                        psi8 * mat(9, 8) + psi9 * mat(9, 9) + psi10 * mat(9, 10) + psi11 * mat(9, 11) + psi12 * mat(9, 12) + psi13 * mat(9, 13) + psi14 * mat(9, 14) + psi15 * mat(9, 15) + //
                        psi16 * mat(9, 16) + psi17 * mat(9, 17) + psi18 * mat(9, 18) + psi19 * mat(9, 19) + psi20 * mat(9, 20) + psi21 * mat(9, 21) + psi22 * mat(9, 22) + psi23 * mat(9, 23) + //
                        psi24 * mat(9, 24) + psi25 * mat(9, 25) + psi26 * mat(9, 26) + psi27 * mat(9, 27) + psi28 * mat(9, 28) + psi29 * mat(9, 29) + psi30 * mat(9, 30) + psi31 * mat(9, 31);
                state_vector[t10] = //
                    psi0 * mat(10, 0) + psi1 * mat(10, 1) + psi2 * mat(10, 2) + psi3 * mat(10, 3) + psi4 * mat(10, 4) + psi5 * mat(10, 5) + psi6 * mat(10, 6) + psi7 * mat(10, 7) + //
                        psi8 * mat(10, 8) + psi9 * mat(10, 9) + psi10 * mat(10, 10) + psi11 * mat(10, 11) + psi12 * mat(10, 12) + psi13 * mat(10, 13) + psi14 * mat(10, 14) + psi15 * mat(10, 15) + //
                        psi16 * mat(10, 16) + psi17 * mat(10, 17) + psi18 * mat(10, 18) + psi19 * mat(10, 19) + psi20 * mat(10, 20) + psi21 * mat(10, 21) + psi22 * mat(10, 22) + psi23 * mat(10, 23) + //
                        psi24 * mat(10, 24) + psi25 * mat(10, 25) + psi26 * mat(10, 26) + psi27 * mat(10, 27) + psi28 * mat(10, 28) + psi29 * mat(10, 29) + psi30 * mat(10, 30) + psi31 * mat(10, 31);
                state_vector[t11] = //
                    psi0 * mat(11, 0) + psi1 * mat(11, 1) + psi2 * mat(11, 2) + psi3 * mat(11, 3) + psi4 * mat(11, 4) + psi5 * mat(11, 5) + psi6 * mat(11, 6) + psi7 * mat(11, 7) + //
                        psi8 * mat(11, 8) + psi9 * mat(11, 9) + psi10 * mat(11, 10) + psi11 * mat(11, 11) + psi12 * mat(11, 12) + psi13 * mat(11, 13) + psi14 * mat(11, 14) + psi15 * mat(11, 15) + //
                        psi16 * mat(11, 16) + psi17 * mat(11, 17) + psi18 * mat(11, 18) + psi19 * mat(11, 19) + psi20 * mat(11, 20) + psi21 * mat(11, 21) + psi22 * mat(11, 22) + psi23 * mat(11, 23) + //
                        psi24 * mat(11, 24) + psi25 * mat(11, 25) + psi26 * mat(11, 26) + psi27 * mat(11, 27) + psi28 * mat(11, 28) + psi29 * mat(11, 29) + psi30 * mat(11, 30) + psi31 * mat(11, 31);
                state_vector[t12] = //
                    psi0 * mat(12, 0) + psi1 * mat(12, 1) + psi2 * mat(12, 2) + psi3 * mat(12, 3) + psi4 * mat(12, 4) + psi5 * mat(12, 5) + psi6 * mat(12, 6) + psi7 * mat(12, 7) + //
                        psi8 * mat(12, 8) + psi9 * mat(12, 9) + psi10 * mat(12, 10) + psi11 * mat(12, 11) + psi12 * mat(12, 12) + psi13 * mat(12, 13) + psi14 * mat(12, 14) + psi15 * mat(12, 15) + //
                        psi16 * mat(12, 16) + psi17 * mat(12, 17) + psi18 * mat(12, 18) + psi19 * mat(12, 19) + psi20 * mat(12, 20) + psi21 * mat(12, 21) + psi22 * mat(12, 22) + psi23 * mat(12, 23) + //
                        psi24 * mat(12, 24) + psi25 * mat(12, 25) + psi26 * mat(12, 26) + psi27 * mat(12, 27) + psi28 * mat(12, 28) + psi29 * mat(12, 29) + psi30 * mat(12, 30) + psi31 * mat(12, 31);
                state_vector[t13] = //
                    psi0 * mat(13, 0) + psi1 * mat(13, 1) + psi2 * mat(13, 2) + psi3 * mat(13, 3) + psi4 * mat(13, 4) + psi5 * mat(13, 5) + psi6 * mat(13, 6) + psi7 * mat(13, 7) + //
                        psi8 * mat(13, 8) + psi9 * mat(13, 9) + psi10 * mat(13, 10) + psi11 * mat(13, 11) + psi12 * mat(13, 12) + psi13 * mat(13, 13) + psi14 * mat(13, 14) + psi15 * mat(13, 15) + //
                        psi16 * mat(13, 16) + psi17 * mat(13, 17) + psi18 * mat(13, 18) + psi19 * mat(13, 19) + psi20 * mat(13, 20) + psi21 * mat(13, 21) + psi22 * mat(13, 22) + psi23 * mat(13, 23) + //
                        psi24 * mat(13, 24) + psi25 * mat(13, 25) + psi26 * mat(13, 26) + psi27 * mat(13, 27) + psi28 * mat(13, 28) + psi29 * mat(13, 29) + psi30 * mat(13, 30) + psi31 * mat(13, 31);
                state_vector[t14] = //
                    psi0 * mat(14, 0) + psi1 * mat(14, 1) + psi2 * mat(14, 2) + psi3 * mat(14, 3) + psi4 * mat(14, 4) + psi5 * mat(14, 5) + psi6 * mat(14, 6) + psi7 * mat(14, 7) + //
                        psi8 * mat(14, 8) + psi9 * mat(14, 9) + psi10 * mat(14, 10) + psi11 * mat(14, 11) + psi12 * mat(14, 12) + psi13 * mat(14, 13) + psi14 * mat(14, 14) + psi15 * mat(14, 15) + //
                        psi16 * mat(14, 16) + psi17 * mat(14, 17) + psi18 * mat(14, 18) + psi19 * mat(14, 19) + psi20 * mat(14, 20) + psi21 * mat(14, 21) + psi22 * mat(14, 22) + psi23 * mat(14, 23) + //
                        psi24 * mat(14, 24) + psi25 * mat(14, 25) + psi26 * mat(14, 26) + psi27 * mat(14, 27) + psi28 * mat(14, 28) + psi29 * mat(14, 29) + psi30 * mat(14, 30) + psi31 * mat(14, 31);
                state_vector[t15] = //
                    psi0 * mat(15, 0) + psi1 * mat(15, 1) + psi2 * mat(15, 2) + psi3 * mat(15, 3) + psi4 * mat(15, 4) + psi5 * mat(15, 5) + psi6 * mat(15, 6) + psi7 * mat(15, 7) + //
                        psi8 * mat(15, 8) + psi9 * mat(15, 9) + psi10 * mat(15, 10) + psi11 * mat(15, 11) + psi12 * mat(15, 12) + psi13 * mat(15, 13) + psi14 * mat(15, 14) + psi15 * mat(15, 15) + //
                        psi16 * mat(15, 16) + psi17 * mat(15, 17) + psi18 * mat(15, 18) + psi19 * mat(15, 19) + psi20 * mat(15, 20) + psi21 * mat(15, 21) + psi22 * mat(15, 22) + psi23 * mat(15, 23) + //
                        psi24 * mat(15, 24) + psi25 * mat(15, 25) + psi26 * mat(15, 26) + psi27 * mat(15, 27) + psi28 * mat(15, 28) + psi29 * mat(15, 29) + psi30 * mat(15, 30) + psi31 * mat(15, 31);
                state_vector[t16] = //
                    psi0 * mat(16, 0) + psi1 * mat(16, 1) + psi2 * mat(16, 2) + psi3 * mat(16, 3) + psi4 * mat(16, 4) + psi5 * mat(16, 5) + psi6 * mat(16, 6) + psi7 * mat(16, 7) + //
                        psi8 * mat(16, 8) + psi9 * mat(16, 9) + psi10 * mat(16, 10) + psi11 * mat(16, 11) + psi12 * mat(16, 12) + psi13 * mat(16, 13) + psi14 * mat(16, 14) + psi15 * mat(16, 15) + //
                        psi16 * mat(16, 16) + psi17 * mat(16, 17) + psi18 * mat(16, 18) + psi19 * mat(16, 19) + psi20 * mat(16, 20) + psi21 * mat(16, 21) + psi22 * mat(16, 22) + psi23 * mat(16, 23) + //
                        psi24 * mat(16, 24) + psi25 * mat(16, 25) + psi26 * mat(16, 26) + psi27 * mat(16, 27) + psi28 * mat(16, 28) + psi29 * mat(16, 29) + psi30 * mat(16, 30) + psi31 * mat(16, 31);
                state_vector[t17] = //
                    psi0 * mat(17, 0) + psi1 * mat(17, 1) + psi2 * mat(17, 2) + psi3 * mat(17, 3) + psi4 * mat(17, 4) + psi5 * mat(17, 5) + psi6 * mat(17, 6) + psi7 * mat(17, 7) + //
                        psi8 * mat(17, 8) + psi9 * mat(17, 9) + psi10 * mat(17, 10) + psi11 * mat(17, 11) + psi12 * mat(17, 12) + psi13 * mat(17, 13) + psi14 * mat(17, 14) + psi15 * mat(17, 15) + //
                        psi16 * mat(17, 16) + psi17 * mat(17, 17) + psi18 * mat(17, 18) + psi19 * mat(17, 19) + psi20 * mat(17, 20) + psi21 * mat(17, 21) + psi22 * mat(17, 22) + psi23 * mat(17, 23) + //
                        psi24 * mat(17, 24) + psi25 * mat(17, 25) + psi26 * mat(17, 26) + psi27 * mat(17, 27) + psi28 * mat(17, 28) + psi29 * mat(17, 29) + psi30 * mat(17, 30) + psi31 * mat(17, 31);
                state_vector[t18] = //
                    psi0 * mat(18, 0) + psi1 * mat(18, 1) + psi2 * mat(18, 2) + psi3 * mat(18, 3) + psi4 * mat(18, 4) + psi5 * mat(18, 5) + psi6 * mat(18, 6) + psi7 * mat(18, 7) + //
                        psi8 * mat(18, 8) + psi9 * mat(18, 9) + psi10 * mat(18, 10) + psi11 * mat(18, 11) + psi12 * mat(18, 12) + psi13 * mat(18, 13) + psi14 * mat(18, 14) + psi15 * mat(18, 15) + //
                        psi16 * mat(18, 16) + psi17 * mat(18, 17) + psi18 * mat(18, 18) + psi19 * mat(18, 19) + psi20 * mat(18, 20) + psi21 * mat(18, 21) + psi22 * mat(18, 22) + psi23 * mat(18, 23) + //
                        psi24 * mat(18, 24) + psi25 * mat(18, 25) + psi26 * mat(18, 26) + psi27 * mat(18, 27) + psi28 * mat(18, 28) + psi29 * mat(18, 29) + psi30 * mat(18, 30) + psi31 * mat(18, 31);
                state_vector[t19] = //
                    psi0 * mat(19, 0) + psi1 * mat(19, 1) + psi2 * mat(19, 2) + psi3 * mat(19, 3) + psi4 * mat(19, 4) + psi5 * mat(19, 5) + psi6 * mat(19, 6) + psi7 * mat(19, 7) + //
                        psi8 * mat(19, 8) + psi9 * mat(19, 9) + psi10 * mat(19, 10) + psi11 * mat(19, 11) + psi12 * mat(19, 12) + psi13 * mat(19, 13) + psi14 * mat(19, 14) + psi15 * mat(19, 15) + //
                        psi16 * mat(19, 16) + psi17 * mat(19, 17) + psi18 * mat(19, 18) + psi19 * mat(19, 19) + psi20 * mat(19, 20) + psi21 * mat(19, 21) + psi22 * mat(19, 22) + psi23 * mat(19, 23) + //
                        psi24 * mat(19, 24) + psi25 * mat(19, 25) + psi26 * mat(19, 26) + psi27 * mat(19, 27) + psi28 * mat(19, 28) + psi29 * mat(19, 29) + psi30 * mat(19, 30) + psi31 * mat(19, 31);
                state_vector[t20] = //
                    psi0 * mat(20, 0) + psi1 * mat(20, 1) + psi2 * mat(20, 2) + psi3 * mat(20, 3) + psi4 * mat(20, 4) + psi5 * mat(20, 5) + psi6 * mat(20, 6) + psi7 * mat(20, 7) + //
                        psi8 * mat(20, 8) + psi9 * mat(20, 9) + psi10 * mat(20, 10) + psi11 * mat(20, 11) + psi12 * mat(20, 12) + psi13 * mat(20, 13) + psi14 * mat(20, 14) + psi15 * mat(20, 15) + //
                        psi16 * mat(20, 16) + psi17 * mat(20, 17) + psi18 * mat(20, 18) + psi19 * mat(20, 19) + psi20 * mat(20, 20) + psi21 * mat(20, 21) + psi22 * mat(20, 22) + psi23 * mat(20, 23) + //
                        psi24 * mat(20, 24) + psi25 * mat(20, 25) + psi26 * mat(20, 26) + psi27 * mat(20, 27) + psi28 * mat(20, 28) + psi29 * mat(20, 29) + psi30 * mat(20, 30) + psi31 * mat(20, 31);
                state_vector[t21] = //
                    psi0 * mat(21, 0) + psi1 * mat(21, 1) + psi2 * mat(21, 2) + psi3 * mat(21, 3) + psi4 * mat(21, 4) + psi5 * mat(21, 5) + psi6 * mat(21, 6) + psi7 * mat(21, 7) + //
                        psi8 * mat(21, 8) + psi9 * mat(21, 9) + psi10 * mat(21, 10) + psi11 * mat(21, 11) + psi12 * mat(21, 12) + psi13 * mat(21, 13) + psi14 * mat(21, 14) + psi15 * mat(21, 15) + //
                        psi16 * mat(21, 16) + psi17 * mat(21, 17) + psi18 * mat(21, 18) + psi19 * mat(21, 19) + psi20 * mat(21, 20) + psi21 * mat(21, 21) + psi22 * mat(21, 22) + psi23 * mat(21, 23) + //
                        psi24 * mat(21, 24) + psi25 * mat(21, 25) + psi26 * mat(21, 26) + psi27 * mat(21, 27) + psi28 * mat(21, 28) + psi29 * mat(21, 29) + psi30 * mat(21, 30) + psi31 * mat(21, 31);
                state_vector[t22] = //
                    psi0 * mat(22, 0) + psi1 * mat(22, 1) + psi2 * mat(22, 2) + psi3 * mat(22, 3) + psi4 * mat(22, 4) + psi5 * mat(22, 5) + psi6 * mat(22, 6) + psi7 * mat(22, 7) + //
                        psi8 * mat(22, 8) + psi9 * mat(22, 9) + psi10 * mat(22, 10) + psi11 * mat(22, 11) + psi12 * mat(22, 12) + psi13 * mat(22, 13) + psi14 * mat(22, 14) + psi15 * mat(22, 15) + //
                        psi16 * mat(22, 16) + psi17 * mat(22, 17) + psi18 * mat(22, 18) + psi19 * mat(22, 19) + psi20 * mat(22, 20) + psi21 * mat(22, 21) + psi22 * mat(22, 22) + psi23 * mat(22, 23) + //
                        psi24 * mat(22, 24) + psi25 * mat(22, 25) + psi26 * mat(22, 26) + psi27 * mat(22, 27) + psi28 * mat(22, 28) + psi29 * mat(22, 29) + psi30 * mat(22, 30) + psi31 * mat(22, 31);
                state_vector[t23] = //
                    psi0 * mat(23, 0) + psi1 * mat(23, 1) + psi2 * mat(23, 2) + psi3 * mat(23, 3) + psi4 * mat(23, 4) + psi5 * mat(23, 5) + psi6 * mat(23, 6) + psi7 * mat(23, 7) + //
                        psi8 * mat(23, 8) + psi9 * mat(23, 9) + psi10 * mat(23, 10) + psi11 * mat(23, 11) + psi12 * mat(23, 12) + psi13 * mat(23, 13) + psi14 * mat(23, 14) + psi15 * mat(23, 15) + //
                        psi16 * mat(23, 16) + psi17 * mat(23, 17) + psi18 * mat(23, 18) + psi19 * mat(23, 19) + psi20 * mat(23, 20) + psi21 * mat(23, 21) + psi22 * mat(23, 22) + psi23 * mat(23, 23) + //
                        psi24 * mat(23, 24) + psi25 * mat(23, 25) + psi26 * mat(23, 26) + psi27 * mat(23, 27) + psi28 * mat(23, 28) + psi29 * mat(23, 29) + psi30 * mat(23, 30) + psi31 * mat(23, 31);
                state_vector[t24] = //
                    psi0 * mat(24, 0) + psi1 * mat(24, 1) + psi2 * mat(24, 2) + psi3 * mat(24, 3) + psi4 * mat(24, 4) + psi5 * mat(24, 5) + psi6 * mat(24, 6) + psi7 * mat(24, 7) + //
                        psi8 * mat(24, 8) + psi9 * mat(24, 9) + psi10 * mat(24, 10) + psi11 * mat(24, 11) + psi12 * mat(24, 12) + psi13 * mat(24, 13) + psi14 * mat(24, 14) + psi15 * mat(24, 15) + //
                        psi16 * mat(24, 16) + psi17 * mat(24, 17) + psi18 * mat(24, 18) + psi19 * mat(24, 19) + psi20 * mat(24, 20) + psi21 * mat(24, 21) + psi22 * mat(24, 22) + psi23 * mat(24, 23) + //
                        psi24 * mat(24, 24) + psi25 * mat(24, 25) + psi26 * mat(24, 26) + psi27 * mat(24, 27) + psi28 * mat(24, 28) + psi29 * mat(24, 29) + psi30 * mat(24, 30) + psi31 * mat(24, 31);
                state_vector[t25] = //
                    psi0 * mat(25, 0) + psi1 * mat(25, 1) + psi2 * mat(25, 2) + psi3 * mat(25, 3) + psi4 * mat(25, 4) + psi5 * mat(25, 5) + psi6 * mat(25, 6) + psi7 * mat(25, 7) + //
                        psi8 * mat(25, 8) + psi9 * mat(25, 9) + psi10 * mat(25, 10) + psi11 * mat(25, 11) + psi12 * mat(25, 12) + psi13 * mat(25, 13) + psi14 * mat(25, 14) + psi15 * mat(25, 15) + //
                        psi16 * mat(25, 16) + psi17 * mat(25, 17) + psi18 * mat(25, 18) + psi19 * mat(25, 19) + psi20 * mat(25, 20) + psi21 * mat(25, 21) + psi22 * mat(25, 22) + psi23 * mat(25, 23) + //
                        psi24 * mat(25, 24) + psi25 * mat(25, 25) + psi26 * mat(25, 26) + psi27 * mat(25, 27) + psi28 * mat(25, 28) + psi29 * mat(25, 29) + psi30 * mat(25, 30) + psi31 * mat(25, 31);
                state_vector[t26] = //
                    psi0 * mat(26, 0) + psi1 * mat(26, 1) + psi2 * mat(26, 2) + psi3 * mat(26, 3) + psi4 * mat(26, 4) + psi5 * mat(26, 5) + psi6 * mat(26, 6) + psi7 * mat(26, 7) + //
                        psi8 * mat(26, 8) + psi9 * mat(26, 9) + psi10 * mat(26, 10) + psi11 * mat(26, 11) + psi12 * mat(26, 12) + psi13 * mat(26, 13) + psi14 * mat(26, 14) + psi15 * mat(26, 15) + //
                        psi16 * mat(26, 16) + psi17 * mat(26, 17) + psi18 * mat(26, 18) + psi19 * mat(26, 19) + psi20 * mat(26, 20) + psi21 * mat(26, 21) + psi22 * mat(26, 22) + psi23 * mat(26, 23) + //
                        psi24 * mat(26, 24) + psi25 * mat(26, 25) + psi26 * mat(26, 26) + psi27 * mat(26, 27) + psi28 * mat(26, 28) + psi29 * mat(26, 29) + psi30 * mat(26, 30) + psi31 * mat(26, 31);
                state_vector[t27] = //
                    psi0 * mat(27, 0) + psi1 * mat(27, 1) + psi2 * mat(27, 2) + psi3 * mat(27, 3) + psi4 * mat(27, 4) + psi5 * mat(27, 5) + psi6 * mat(27, 6) + psi7 * mat(27, 7) + //
                        psi8 * mat(27, 8) + psi9 * mat(27, 9) + psi10 * mat(27, 10) + psi11 * mat(27, 11) + psi12 * mat(27, 12) + psi13 * mat(27, 13) + psi14 * mat(27, 14) + psi15 * mat(27, 15) + //
                        psi16 * mat(27, 16) + psi17 * mat(27, 17) + psi18 * mat(27, 18) + psi19 * mat(27, 19) + psi20 * mat(27, 20) + psi21 * mat(27, 21) + psi22 * mat(27, 22) + psi23 * mat(27, 23) + //
                        psi24 * mat(27, 24) + psi25 * mat(27, 25) + psi26 * mat(27, 26) + psi27 * mat(27, 27) + psi28 * mat(27, 28) + psi29 * mat(27, 29) + psi30 * mat(27, 30) + psi31 * mat(27, 31);
                state_vector[t28] = //
                    psi0 * mat(28, 0) + psi1 * mat(28, 1) + psi2 * mat(28, 2) + psi3 * mat(28, 3) + psi4 * mat(28, 4) + psi5 * mat(28, 5) + psi6 * mat(28, 6) + psi7 * mat(28, 7) + //
                        psi8 * mat(28, 8) + psi9 * mat(28, 9) + psi10 * mat(28, 10) + psi11 * mat(28, 11) + psi12 * mat(28, 12) + psi13 * mat(28, 13) + psi14 * mat(28, 14) + psi15 * mat(28, 15) + //
                        psi16 * mat(28, 16) + psi17 * mat(28, 17) + psi18 * mat(28, 18) + psi19 * mat(28, 19) + psi20 * mat(28, 20) + psi21 * mat(28, 21) + psi22 * mat(28, 22) + psi23 * mat(28, 23) + //
                        psi24 * mat(28, 24) + psi25 * mat(28, 25) + psi26 * mat(28, 26) + psi27 * mat(28, 27) + psi28 * mat(28, 28) + psi29 * mat(28, 29) + psi30 * mat(28, 30) + psi31 * mat(28, 31);
                state_vector[t29] = //
                    psi0 * mat(29, 0) + psi1 * mat(29, 1) + psi2 * mat(29, 2) + psi3 * mat(29, 3) + psi4 * mat(29, 4) + psi5 * mat(29, 5) + psi6 * mat(29, 6) + psi7 * mat(29, 7) + //
                        psi8 * mat(29, 8) + psi9 * mat(29, 9) + psi10 * mat(29, 10) + psi11 * mat(29, 11) + psi12 * mat(29, 12) + psi13 * mat(29, 13) + psi14 * mat(29, 14) + psi15 * mat(29, 15) + //
                        psi16 * mat(29, 16) + psi17 * mat(29, 17) + psi18 * mat(29, 18) + psi19 * mat(29, 19) + psi20 * mat(29, 20) + psi21 * mat(29, 21) + psi22 * mat(29, 22) + psi23 * mat(29, 23) + //
                        psi24 * mat(29, 24) + psi25 * mat(29, 25) + psi26 * mat(29, 26) + psi27 * mat(29, 27) + psi28 * mat(29, 28) + psi29 * mat(29, 29) + psi30 * mat(29, 30) + psi31 * mat(29, 31);
                state_vector[t30] = //
                    psi0 * mat(30, 0) + psi1 * mat(30, 1) + psi2 * mat(30, 2) + psi3 * mat(30, 3) + psi4 * mat(30, 4) + psi5 * mat(30, 5) + psi6 * mat(30, 6) + psi7 * mat(30, 7) + //
                        psi8 * mat(30, 8) + psi9 * mat(30, 9) + psi10 * mat(30, 10) + psi11 * mat(30, 11) + psi12 * mat(30, 12) + psi13 * mat(30, 13) + psi14 * mat(30, 14) + psi15 * mat(30, 15) + //
                        psi16 * mat(30, 16) + psi17 * mat(30, 17) + psi18 * mat(30, 18) + psi19 * mat(30, 19) + psi20 * mat(30, 20) + psi21 * mat(30, 21) + psi22 * mat(30, 22) + psi23 * mat(30, 23) + //
                        psi24 * mat(30, 24) + psi25 * mat(30, 25) + psi26 * mat(30, 26) + psi27 * mat(30, 27) + psi28 * mat(30, 28) + psi29 * mat(30, 29) + psi30 * mat(30, 30) + psi31 * mat(30, 31);
                state_vector[t31] = //
                    psi0 * mat(31, 0) + psi1 * mat(31, 1) + psi2 * mat(31, 2) + psi3 * mat(31, 3) + psi4 * mat(31, 4) + psi5 * mat(31, 5) + psi6 * mat(31, 6) + psi7 * mat(31, 7) + //
                        psi8 * mat(31, 8) + psi9 * mat(31, 9) + psi10 * mat(31, 10) + psi11 * mat(31, 11) + psi12 * mat(31, 12) + psi13 * mat(31, 13) + psi14 * mat(31, 14) + psi15 * mat(31, 15) + //
                        psi16 * mat(31, 16) + psi17 * mat(31, 17) + psi18 * mat(31, 18) + psi19 * mat(31, 19) + psi20 * mat(31, 20) + psi21 * mat(31, 21) + psi22 * mat(31, 22) + psi23 * mat(31, 23) + //
                        psi24 * mat(31, 24) + psi25 * mat(31, 25) + psi26 * mat(31, 26) + psi27 * mat(31, 27) + psi28 * mat(31, 28) + psi29 * mat(31, 29) + psi30 * mat(31, 30) + psi31 * mat(31, 31);
              }
            }
          }
        }
      }
    }
  }
}

template <size_t N>
void QubitVector::apply_matrix_col_major(const std::array<uint_t, N> &qs,
                               const cvector_t &mat) {

  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes(qs, qubits_sorted, k);
      std::array<complex_t, dim> cache;
      for (size_t i = 0; i < dim; i++) {
        const auto ii = inds[i];
        cache[i] = state_vector[ii];
        state_vector[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
          state_vector[inds[i]] += mat[i + dim * j] * cache[j];
    }
  }
}

template <size_t N>
void QubitVector::apply_matrix_diagonal(const std::array<uint_t, N> &qs,
                               const cvector_t &diag) {

  // Error checking
  #ifdef DEBUG
  check_vector(diag, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      const auto inds = indexes(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++)
          state_vector[inds[i]] *= diag[i];
    }
  }
}

void QubitVector::apply_cnot(const uint_t qubit_ctrl, const uint_t qubit_trgt) {

  // Error checking
  #ifdef DEBUG
  check_qubit(qubit_ctrl);
  check_qubit(qubit_trgt);
  #endif

  const int_t end = num_states >> 2;
  const auto qubits_sorted = (qubit_ctrl < qubit_trgt)
                          ? std::array<uint_t, 2>{{qubit_ctrl, qubit_trgt}}
                          : std::array<uint_t, 2>{{qubit_trgt, qubit_ctrl}};

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {

      const auto ii = indexes<2>({{qubit_ctrl, qubit_trgt}},
                                            qubits_sorted, k);
      const complex_t cache = state_vector[ii[3]];
      state_vector[ii[3]] = state_vector[ii[1]];
      state_vector[ii[1]] = cache;
    }
  } // end omp parallel
}

cmatrix_t QubitVector::generate_matrix(const uint_t qubit_size, const cvector_t &vmat) const {
  uint_t dim = 1ULL << qubit_size;
  cmatrix_t mat(dim, dim);
  for (uint_t i = 0; i < dim; ++i)
    for (uint_t j = 0; j < dim; ++j)
      mat(i, j) = vmat[i + dim * j];
  return mat;
}

void QubitVector::swap_cols_and_rows(const uint_t idx1, const uint_t idx2, cmatrix_t &mat) const {

  uint_t size = (mat.size() >> 1);
  uint_t mask1 = (1UL << idx1);
  uint_t mask2 = (1UL << idx2);

  for (uint_t first = 0; first < size; ++first) {
    if ((first & mask1) && !(first & mask2)) {
      uint_t second = first | mask2;

      for (uint_t i = 0; i < size; ++i) {
        complex_t cache = mat(first, i);
        mat(first, i) = mat(i, second);
        mat(second, i) = cache;
      }
      for (uint_t i = 0; i < size; ++i) {
        complex_t cache = mat(i, first);
        mat(i, first) = mat(second, i);
        mat(i, second) = cache;
      }
    }
  }
}

cmatrix_t QubitVector::sort_matrix(const std::vector<uint_t> &src, const std::vector<uint_t> &sorted, const cmatrix_t &mat) const {
  cmatrix_t ret = mat;
  std::vector<uint_t> current = src;

  while (current != sorted) {
    uint_t from;
    uint_t to;
    for (from = 0; from < current.size(); ++from)
      if (current[from] != sorted[from])
        break;
    if (from == current.size())
      break;
    for (to = from + 1; to < current.size(); ++to)
      if (current[to] == sorted[to])
        break;
    if (to == current.size())
      throw std::runtime_error("should not reach here");
    swap_cols_and_rows(from, to, ret);
  }

  return ret;
}

void QubitVector::apply_swap(const uint_t qubit0, const uint_t qubit1) {

  // Error checking
  #ifdef DEBUG
  check_qubit(qubit0);
  check_qubit(qubit1);
  #endif

  const int_t end = num_states >> 2;
  const auto qubits_sorted = (qubit0 < qubit1)
                          ? std::array<uint_t, 2>{{qubit0, qubit1}}
                          : std::array<uint_t, 2>{{qubit1, qubit0}};

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {

      const auto ii = indexes<2>({{qubit0, qubit1}},
                                            qubits_sorted, k);
      const complex_t cache = state_vector[ii[2]];
      state_vector[ii[2]] = state_vector[ii[1]];
      state_vector[ii[1]] = cache;
    }
  } // end omp parallel
}

void QubitVector::apply_cz(const uint_t qubit_ctrl, const uint_t qubit_trgt) {

  // Error checking
  #ifdef DEBUG
  check_qubit(qubit_ctrl);
  check_qubit(qubit_trgt);
  #endif

  const int_t end = num_states >> 2;
  const auto qubits_sorted = (qubit_ctrl < qubit_trgt)
                          ? std::array<uint_t, 2>{{qubit_ctrl, qubit_trgt}}
                          : std::array<uint_t, 2>{{qubit_trgt, qubit_ctrl}};

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      const auto ii = indexes<2>({{qubit_ctrl, qubit_trgt}},
                                            qubits_sorted, k);
      state_vector[ii[3]] *= -1.;
    }
  }
}


//------------------------------------------------------------------------------
// Norm
//------------------------------------------------------------------------------

template <size_t N>
double QubitVector::norm(const std::array<uint_t, N> &qs, const cvector_t &mat) const {
  if (mat.size() == (1ULL << N))
    return norm_matrix_diagonal<N>(qs, mat);
  else
    return norm_matrix<N>(qs, mat);
}

template <size_t N>
double QubitVector::norm_matrix(const std::array<uint_t, N> &qs, const cvector_t &mat) const {

  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val = 0.;

#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++)
          vi += mat[i + dim * j] * state_vector[inds[j]];
        val += std::real(vi * std::conj(vi));
      }
    }
  } // end omp parallel
  return val;
}


template <size_t N>
double QubitVector::norm_matrix_diagonal(const std::array<uint_t, N> &qs, const cvector_t &mat) const {

  // Error checking
  #ifdef DEBUG
  check_vector(mat, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val = 0.;
#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++) {
        const auto vi = mat[i] * state_vector[inds[i]];
        val += std::real(vi * std::conj(vi));
      }
    }
  } // end omp parallel
  return val;
}

//------------------------------------------------------------------------------
// Expectation Values
//------------------------------------------------------------------------------

template <size_t N>
complex_t QubitVector::expectation_value(const std::array<uint_t, N> &qs, const cvector_t &mat) const {
  if (mat.size() == (1ULL << N))
    return expectation_value_matrix_diagonal<N>(qs, mat);
  else
    return expectation_value_matrix<N>(qs, mat);
}

template <size_t N>
complex_t QubitVector::expectation_value_matrix(const std::array<uint_t, N> &qs, const cvector_t &mat) const {

  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++) {
          vi += mat[i + dim * j] * state_vector[inds[j]];
        }
        const complex_t val = vi * std::conj(state_vector[inds[i]]);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


template <size_t N>
complex_t QubitVector::expectation_value_matrix_diagonal(const std::array<uint_t, N> &qs, const cvector_t &mat) const {

  // Error checking
  #ifdef DEBUG
  check_vector(mat, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes(qs, qubits_sorted, k);
      for (size_t i = 0; i < dim; i++) {
        const auto cache = state_vector[inds[i]];
        const complex_t val = mat[i] * cache * std::conj(cache);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


/*******************************************************************************
 *
 * DYNAMIC N-QUBIT OPERATIONS (N known at run time)
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Matrix multiplication
//------------------------------------------------------------------------------

void QubitVector::apply_matrix(const std::vector<uint_t> &qs, const cvector_t &mat) {

  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    apply_matrix(qs[0], mat);
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
  default:
    // General case
    if (mat.size() == (1ULL << qs.size()))
      apply_matrix_diagonal(qs, mat);
    else
      apply_matrix_col_major(qs, mat);
    break;
  }
}

void QubitVector::apply_matrix_col_major(const std::vector<uint_t> &qubits, const cvector_t &mat) {

  const auto N = qubits.size();
  const uint_t dim = 1ULL << N;
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;

  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qubits, qubits_sorted, N, k);
      std::vector<complex_t> cache(dim);
      for (size_t i = 0; i < dim; i++) {
        const auto ii = inds[i];
        cache[i] = state_vector[ii];
        state_vector[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
          state_vector[inds[i]] += mat[i + dim * j] * cache[j];
    }
  }
}

void QubitVector::apply_matrix_diagonal(const std::vector<uint_t> &qubits,
                               const cvector_t &diag) {

  const auto N = qubits.size();
  const uint_t dim = 1ULL << N;
  // Error checking
  #ifdef DEBUG
  check_vector(diag, N);
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  auto qss = qubits;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;

#pragma omp parallel if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      const auto inds = indexes_dynamic(qubits, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++)
          state_vector[inds[i]] *= diag[i];
    }
  }
}


//------------------------------------------------------------------------------
// Norm
//------------------------------------------------------------------------------

double QubitVector::norm(const std::vector<uint_t> &qs, const cvector_t &mat) const {
  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    return norm(qs[0], mat);
  case 2:
    return norm<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), mat);
  case 3:
    return norm<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), mat);
  case 4:
    return norm<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), mat);
  case 5:
    return norm<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), mat);
  default:
    // General case
    if (mat.size() == (1ULL << qs.size()))
      return norm_matrix_diagonal(qs, mat);
    else
      return norm_matrix(qs, mat);
  }
}

double QubitVector::norm_matrix(const std::vector<uint_t> &qs, const cvector_t &mat) const {

  // Error checking
  const uint_t N = qs.size();
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val = 0.;

#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qs, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++)
          vi += mat[i + dim * j] * state_vector[inds[j]];
        val += std::real(vi * std::conj(vi));
      }
    }
  } // end omp parallel
  return val;
}

double QubitVector::norm_matrix_diagonal(const std::vector<uint_t> &qs, const cvector_t &mat) const {

  // Error checking
  const uint_t N = qs.size();
  #ifdef DEBUG
  check_vector(mat, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val = 0.;
#pragma omp parallel reduction(+:val) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
#pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qs, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++) {
        const auto vi = mat[i] * state_vector[inds[i]];
        val += std::real(vi * std::conj(vi));
      }
    }
  } // end omp parallel
  return val;
}


//------------------------------------------------------------------------------
// Expectation Values
//------------------------------------------------------------------------------

complex_t QubitVector::expectation_value(const std::vector<uint_t> &qs, const cvector_t &mat) const {
  // Special low N cases using faster static indexing
  switch (qs.size()) {
  case 1:
    return expectation_value(qs[0], mat);
  case 2:
    return expectation_value<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), mat);
  case 3:
    return expectation_value<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), mat);
  case 4:
    return expectation_value<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), mat);
  case 5:
    return expectation_value<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), mat);
  default:
    // General case
    if (mat.size() == (1ULL << qs.size()))
      return expectation_value_matrix_diagonal(qs, mat);
    else
      return expectation_value_matrix(qs, mat);
  }
}

complex_t QubitVector::expectation_value_matrix(const std::vector<uint_t> &qs, const cvector_t &mat) const {

  // Error checking
  const uint_t N = qs.size();
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qs, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++) {
        complex_t vi = 0;
        for (size_t j = 0; j < dim; j++) {
          vi += mat[i + dim * j] * state_vector[inds[j]];
        }
        const complex_t val = vi * std::conj(state_vector[inds[i]]);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}

complex_t QubitVector::expectation_value_matrix_diagonal(const std::vector<uint_t> &qs, const cvector_t &mat) const {

  // Error checking
  const uint_t N = qs.size();
  #ifdef DEBUG
  check_vector(mat, N);
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = num_states >> N;
  const uint_t dim = 1ULL << N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double val_re = 0., val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++) {
      // store entries touched by U
      const auto inds = indexes_dynamic(qs, qubits_sorted, N, k);
      for (size_t i = 0; i < dim; i++) {
        const auto cache = state_vector[inds[i]];
        const complex_t val = mat[i] * cache * std::conj(cache);
        val_re += std::real(val);
        val_im += std::imag(val);
      }
    }
  } // end omp parallel
  return complex_t(val_re, val_im);
}


/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// All outcome probabilities
//------------------------------------------------------------------------------

rvector_t QubitVector::probabilities() const {
  rvector_t probs;
  probs.reserve(num_states);
  const int_t end = state_vector.size();
  for (int_t j=0; j < end; j++) {
    probs.push_back(probability(j));
  }
  return probs;
}

rvector_t QubitVector::probabilities(const uint_t qubit) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
  double p0 = 0., p1 = 0.;
#pragma omp parallel reduction(+:p0, p1) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
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
  const uint_t end = (1ULL << num_qubits) >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  if ((N == num_qubits) && (qs == qss))
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

rvector_t QubitVector::probabilities(const std::vector<uint_t> &qs) const {

  // Special cases using faster static indexing
  const uint_t N = qs.size();
  switch (N) {
  case 0:
    return rvector_t({norm()});
  case 1:
    return probabilities(qs[0]);
  case 2:
    return probabilities<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}));
  case 3:
    return probabilities<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}));
  case 4:
    return probabilities<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}));
  case 5:
    return probabilities<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}));
  default:
    // else
    // Error checking
    #ifdef DEBUG
    for (const auto &qubit : qs)
      check_qubit(qubit);
    #endif

    const uint_t dim = 1ULL << N;
    const uint_t end = (1ULL << num_qubits) >> N;
    auto qss = qs;
    std::sort(qss.begin(), qss.end());
    if ((N == num_qubits) && (qss == qs))
      return probabilities();
    const auto &qubits_sorted = qss;
    rvector_t probs(dim, 0.);

    for (size_t k = 0; k < end; k++) {
      const auto idx = indexes_dynamic(qs, qubits_sorted, N, k);
      for (size_t m = 0; m < dim; ++m)
        probs[m] += probability(idx[m]);
    }
    return probs;
  }
}

//------------------------------------------------------------------------------
// Single outcome probability
//------------------------------------------------------------------------------
double QubitVector::probability(const uint_t outcome) const {
  const auto v = state_vector[outcome];
  return std::real(v * std::conj(v));
}

double QubitVector::probability(const uint_t qubit, const uint_t outcome) const {

  // Error handling
  #ifdef DEBUG
  check_qubit(qubit);
  #endif

  const int_t end1 = num_states;    // end for k1 loop
  const int_t end2 = 1LL << qubit; // end for k2 loop
  const int_t step1 = end2 << 1;    // step for k1 loop
  double p = 0.;
#pragma omp parallel reduction(+:p) if (num_qubits > omp_threshold && omp_threads > 1)         \
                                               num_threads(omp_threads)
  {
  if (outcome == 0) {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++)
        p += probability(k1 | k2);
  } else if (outcome == 1) {
#ifdef _WIN32
  #pragma omp for
#else
  #pragma omp for collapse(2)
#endif
    for (int_t k1 = 0; k1 < end1; k1 += step1)
      for (int_t k2 = 0; k2 < end2; k2++)
        p += probability(k1 | k2 | end2);
  }
  } // end omp parallel
  return p;
}

template <size_t N>
double QubitVector::probability(const std::array<uint_t, N> &qs,
                                const uint_t outcome) const {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qs)
    check_qubit(qubit);
  #endif

  const int_t end = (1ULL << num_qubits) >> N;
  auto qss = qs;
  std::sort(qss.begin(), qss.end());
  const auto &qubits_sorted = qss;
  double p = 0.;

#pragma omp parallel reduction(+:p) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
  {
  #pragma omp for
    for (int_t k = 0; k < end; k++)
      p += probability(indexes<N>(qs, qubits_sorted, k)[outcome]);
  }
  return p;
}

double QubitVector::probability(const std::vector<uint_t> &qs,
                                const uint_t outcome) const {

  // Special cases using faster static indexing
  const uint_t N = qs.size();
  switch (N) {
  case 0:
    return norm();
  case 1:
    return probability(qs[0], outcome);
  case 2:
    return probability<2>(std::array<uint_t, 2>({{qs[0], qs[1]}}), outcome);
  case 3:
    return probability<3>(std::array<uint_t, 3>({{qs[0], qs[1], qs[2]}}), outcome);
  case 4:
    return probability<4>(std::array<uint_t, 4>({{qs[0], qs[1], qs[2], qs[3]}}), outcome);
  case 5:
    return probability<5>(std::array<uint_t, 5>({{qs[0], qs[1], qs[2], qs[3], qs[4]}}), outcome);
  default:
    // else
    // Error checking
    #ifdef DEBUG
    for (const auto &qubit : qs)
      check_qubit(qubit);
    #endif

    const int_t end = (1ULL << num_qubits) >> N;
    auto qss = qs;
    std::sort(qss.begin(), qss.end());
    const auto &qubits_sorted = qss;
    double p = 0.;

  #pragma omp parallel reduction(+:p) if (num_qubits > omp_threshold && omp_threads > 1) num_threads(omp_threads)
    {
    #pragma omp for
      for (int_t k = 0; k < end; k++)
        p += probability(indexes_dynamic(qs, qubits_sorted, N, k)[outcome]);
    }
    return p;
  }
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
