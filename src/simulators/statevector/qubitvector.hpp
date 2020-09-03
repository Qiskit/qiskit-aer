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

#include "simulators/statevector/indexes.hpp"
#include "framework/json.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace QV {

template <typename T> using cvector_t = std::vector<std::complex<T>>;

//============================================================================
// QubitVector class
//============================================================================

// Template class for qubit vector.
// The arguement of the template must have an operator[] access method.
// The following methods may also need to be template specialized:
//   * set_num_qubits(size_t)
//   * initialize()
//   * initialize_from_vector(cvector_t<data_t>)
// If the template argument does not have these methods then template
// specialization must be used to override the default implementations.

template <typename data_t = double, typename Derived = void>
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
  std::complex<data_t> &operator[](uint_t element);
  std::complex<data_t> operator[](uint_t element) const;

  // Returns a reference to the underlying data_t data class
  std::complex<data_t>* &data() {return data_;}

  // Returns a copy of the underlying data_t data class
  std::complex<data_t>* data() const {return data_;}

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Return the string name of the QUbitVector class
  static std::string name() {return "statevector";}

  // Set the size of the vector in terms of qubit number
  void set_num_qubits(size_t num_qubits);

  // Returns the number of qubits for the current vector
  virtual uint_t num_qubits() const {return num_qubits_;}

  // Returns the size of the underlying n-qubit vector
  uint_t size() const {return data_size_;}

  // Returns required memory
  size_t required_memory_mb(uint_t num_qubits) const;

  // Returns a copy of the underlying data_t data as a complex vector
  cvector_t<data_t> vector() const;

  // Return JSON serialization of QubitVector;
  json_t json() const;

  // Set all entries in the vector to 0.
  void zero();

  // convert vector type to data type of this qubit vector
  cvector_t<data_t> convert(const cvector_t<double>& v) const;

  // State initialization of a component
  // Initialize the specified qubits to a desired statevector
  // (leaving the other qubits in their current state)
  // assuming the qubits being initialized have already been reset to the zero state
  // (using apply_reset)
  void initialize_component(const reg_t &qubits, const cvector_t<double> &state);

  //-----------------------------------------------------------------------
  // Check point operations
  //-----------------------------------------------------------------------

  // Create a checkpoint of the current state
  void checkpoint();

  // Revert to the checkpoint
  void revert(bool keep);

  // Compute the inner product of current state with checkpoint state
  std::complex<double> inner_product() const;

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // If the length of the data vector does not match the number of qubits
  // an exception is raised.
  void initialize_from_vector(const cvector_t<double> &data);

  // Initializes the vector to a custom initial state.
  // If num_states does not match the number of qubits an exception is raised.
  void initialize_from_data(const std::complex<data_t>* data, const size_t num_states);

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a 1-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  void apply_matrix(const uint_t qubit, const cvector_t<double> &mat);

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a stacked set of 2^control_count target_count--qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const cvector_t<double> &mat);

  // Apply a 1-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const uint_t qubit, const cvector_t<double> &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Swap pairs of indicies in the underlying vector
  void apply_permutation_matrix(const reg_t &qubits,
                                const std::vector<std::pair<uint_t, uint_t>> &pairs);

  //-----------------------------------------------------------------------
  // Apply Specialized Gates
  //-----------------------------------------------------------------------

  // Apply a general N-qubit multi-controlled X-gate
  // If N=1 this implements an optimized X gate
  // If N=2 this implements an optimized CX gate
  // If N=3 this implements an optimized Toffoli gate
  void apply_mcx(const reg_t &qubits);

  // Apply a general multi-controlled Y-gate
  // If N=1 this implements an optimized Y gate
  // If N=2 this implements an optimized CY gate
  // If N=3 this implements an optimized CCY gate
  void apply_mcy(const reg_t &qubits);

  // Apply a general multi-controlled single-qubit phase gate
  // with diagonal [1, ..., 1, phase]
  // If N=1 this implements an optimized single-qubit phase gate
  // If N=2 this implements an optimized CPhase gate
  // If N=3 this implements an optimized CCPhase gate
  // if phase = -1 this is a Z, CZ, CCZ gate
  void apply_mcphase(const reg_t &qubits, const std::complex<double> phase);

  // Apply a general multi-controlled single-qubit unitary gate
  // If N=1 this implements an optimized single-qubit U gate
  // If N=2 this implements an optimized CU gate
  // If N=3 this implements an optimized CCU gate
  void apply_mcu(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a general multi-controlled SWAP gate
  // If N=2 this implements an optimized SWAP  gate
  // If N=3 this implements an optimized Fredkin gate
  void apply_mcswap(const reg_t &qubits);

  //-----------------------------------------------------------------------
  // Z-measurement outcome probabilities
  //-----------------------------------------------------------------------

  // Return the Z-basis measurement outcome probability P(outcome) for
  // outcome in [0, 2^num_qubits - 1]
  virtual double probability(const uint_t outcome) const;

  // Return the probabilities for all measurement outcomes in the current vector
  // This is equivalent to returning a new vector with  new[i]=|orig[i]|^2.
  // Eg. For 2-qubits this is [P(00), P(01), P(010), P(11)]
  virtual std::vector<double> probabilities() const;

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  virtual std::vector<double> probabilities(const reg_t &qubits) const;

  // Return M sampled outcomes for Z-basis measurement of all qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  virtual reg_t sample_measure(const std::vector<double> &rnds) const;

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
  double norm(const uint_t qubit, const cvector_t<double> &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  double norm(const reg_t &qubits, const cvector_t<double> &mat) const;

  // Return the norm for of the vector obtained after apply the 1-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const reg_t &qubits, const cvector_t<double> &mat) const;

  //-----------------------------------------------------------------------
  // Expectation Value
  //-----------------------------------------------------------------------

  // These functions return the expectation value <psi|A|psi> for a matrix A.
  // If A is hermitian these will return real values, if A is non-Hermitian
  // they in general will return complex values.

  // Return the expectation value of an N-qubit Pauli matrix.
  // The Pauli is input as a length N string of I,X,Y,Z characters.
  double expval_pauli(const reg_t &qubits, const std::string &pauli) const;

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
  std::complex<data_t>* data_;
  std::complex<data_t>* checkpoint_;

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 14;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 10; // Sample measure indexing qubit size
  double json_chop_threshold_ = 0;  // Threshold for choping small values
                                    // in JSON serialization
  inline uint_t omp_threads_managed() const { return (num_qubits_ > omp_threshold_ && omp_threads_ > 1) ? omp_threads_: 0; }

  //-----------------------------------------------------------------------
  // Error Messages
  //-----------------------------------------------------------------------

  void check_qubit(const uint_t qubit) const;
  void check_vector(const cvector_t<data_t> &diag, uint_t nqubits) const;
  void check_matrix(const cvector_t<data_t> &mat, uint_t nqubits) const;
  void check_dimension(const QubitVector &qv) const;
  void check_checkpoint() const;

  //-----------------------------------------------------------------------
  // Statevector update with Lambda function
  //-----------------------------------------------------------------------
  // Apply a lambda function to all entries of the statevector.
  // The function signature should be:
  //
  // [&](const int_t k)->void
  //
  // where k is the index of the vector
  template <typename Lambda>
  void apply_lambda(Lambda&& func);

  //-----------------------------------------------------------------------
  // Statevector block update with Lambda function
  //-----------------------------------------------------------------------
  // These functions loop through the indexes of the qubitvector data and
  // apply a lambda function to each block specified by the qubits argument.
  //
  // NOTE: The lambda functions can use the dynamic or static indexes
  // signature however if N is known at compile time the static case should
  // be preferred as it is significantly faster.

  // Apply a N-qubit lambda function to all blocks of the statevector
  // for the given qubits. The function signature should be either:
  //
  // (Static): [&](const areg_t<1ULL<<N> &inds)->void
  // (Dynamic): [&](const indexes_t &inds)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexes` function.
  template <typename Lambda, typename list_t>
  void apply_lambda(Lambda&& func, const list_t &qubits);

  // Apply an N-qubit parameterized lambda function to all blocks of the
  // statevector for the given qubits. The function signature should be:
  //
  // (Static): [&](const areg_t<1ULL<<N> &inds, const param_t &params)->void
  // (Dynamic): [&](const indexes_t &inds, const param_t &params)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexes` function and `param` is a templated parameter class.
  // (typically a complex vector).
  template <typename Lambda, typename list_t, typename param_t>
  void apply_lambda(Lambda&& func, const list_t &qubits, const param_t &par);

  //-----------------------------------------------------------------------
  // State reduction with Lambda functions
  //-----------------------------------------------------------------------
  // Apply a complex reduction lambda function over the specified entries
  // of the state vector given by start, stop.
  //
  // [&](const int_t k, double &val_re, double &val_im)->void
  //
  // where k is the index of the vector, val_re and val_im are the doubles
  // to store the reduction.
  // Returns std::complex<double>(val_re, val_im)
  template <typename Lambda>
  std::complex<double> apply_reduction_lambda(Lambda&& func, size_t start, size_t stop) const;

  template <typename Lambda>
  std::complex<double> apply_reduction_lambda(Lambda&& func) const;

  //-----------------------------------------------------------------------
  // Statevector block reduction with Lambda function
  //-----------------------------------------------------------------------
  // These functions loop through the indexes of the qubitvector data and
  // apply a reduction lambda function to each block specified by the qubits
  // argument. The reduction lambda stores the reduction in two doubles
  // (val_re, val_im) and returns the complex result std::complex<double>(val_re, val_im)
  //
  // NOTE: The lambda functions can use the dynamic or static indexes
  // signature however if N is known at compile time the static case should
  // be preferred as it is significantly faster.

  // Apply a N-qubit complex matrix reduction lambda function to all blocks
  // of the statevector for the given qubits.
  // The lambda function signature should be:
  //
  // (Static): [&](const areg_t<1ULL<<N> &inds, const param_t &mat,
  //               double &val_re, double &val_im)->void
  // (Dynamic): [&](const indexes_t &inds, const param_t &mat,
  //                double &val_re, double &val_im)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexes` function, `val_re` and `val_im` are the doubles to
  // store the reduction returned as std::complex<double>(val_re, val_im).
  template <typename Lambda, typename list_t>
  std::complex<double> apply_reduction_lambda(Lambda&& func,
                                              const list_t &qubits) const;

  // Apply a N-qubit complex matrix reduction lambda function to all blocks
  // of the statevector for the given qubits.
  // The lambda function signature should be:
  //
  // (Static): [&](const areg_t<1ULL<<N> &inds, const param_t &parms,
  //               double &val_re, double &val_im)->void
  // (Dynamic): [&](const indexes_t &inds, const param_t &params,
  //                double &val_re, double &val_im)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexe`s function, `params` is a templated parameter class
  // (typically a complex vector), `val_re` and `val_im` are the doubles to
  // store the reduction returned as std::complex<double>(val_re, val_im).
  template <typename Lambda, typename list_t, typename param_t>
  std::complex<double> apply_reduction_lambda(Lambda&& func,
                                              const list_t &qubits,
                                              const param_t &params) const;

  // Free allocated memory
  void free_mem();

  // Free allocated checkpoint
  void free_checkpoint();

  // Allocates memory for the underlaying quantum state
  void allocate_mem(size_t data_size);

  // Allocates memory for the checkoiunt
  void allocate_checkpoint(size_t data_size);
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

template <typename data_t, typename Derived>
inline void to_json(json_t &js, const QubitVector<data_t> &qv) {
  js = qv.json();
}

template <typename data_t, typename Derived>
json_t QubitVector<data_t, Derived>::json() const {
  const int_t END = data_size_;
  const json_t ZERO = std::complex<data_t>(0.0, 0.0);
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

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits_) {
    std::string error = "QubitVector: qubit index " + std::to_string(qubit) +
                        " > " + std::to_string(num_qubits_);
    throw std::runtime_error(error);
  }
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::check_matrix(const cvector_t<data_t> &vec, uint_t nqubits) const {
  const size_t DIM = BITS[nqubits];
  const auto SIZE = vec.size();
  if (SIZE != DIM * DIM) {
    std::string error = "QubitVector: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM * DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::check_vector(const cvector_t<data_t> &vec, uint_t nqubits) const {
  const size_t DIM = BITS[nqubits];
  const auto SIZE = vec.size();
  if (SIZE != DIM) {
    std::string error = "QubitVector: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::check_dimension(const QubitVector &qv) const {
  if (data_size_ != qv.size_) {
    std::string error = "QubitVector: vectors are different shape " +
                         std::to_string(data_size_) + " != " +
                         std::to_string(qv.num_states_);
    throw std::runtime_error(error);
  }
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::check_checkpoint() const {
  if (!checkpoint_) {
    throw std::runtime_error("QubitVector: checkpoint must exist for inner_product() or revert()");
  }
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t, typename Derived>
QubitVector<data_t, Derived>::QubitVector(size_t num_qubits) : num_qubits_(0), data_(nullptr), checkpoint_(0){
  set_num_qubits(num_qubits);
}

template <typename data_t, typename Derived>
QubitVector<data_t, Derived>::QubitVector() : QubitVector(0) {}

template <typename data_t, typename Derived>
QubitVector<data_t, Derived>::~QubitVector() {
  free_mem();
  free_checkpoint();
}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <typename data_t, typename Derived>
std::complex<data_t> &QubitVector<data_t, Derived>::operator[](uint_t element) {
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

template <typename data_t, typename Derived>
std::complex<data_t> QubitVector<data_t, Derived>::operator[](uint_t element) const {
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

template <typename data_t, typename Derived>
cvector_t<data_t> QubitVector<data_t, Derived>::vector() const {
  cvector_t<data_t> ret(data_size_, 0.);
  const int_t END = data_size_;
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < END; j++) {
    ret[j] = data_[j];
  }
  return ret;
}

//------------------------------------------------------------------------------
// State initialize component
//------------------------------------------------------------------------------
template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::initialize_component(const reg_t &qubits, const cvector_t<double> &state0) {

  cvector_t<data_t> state = convert(state0);

  // Lambda function for initializing component
  auto lambda = [&](const indexes_t &inds, const cvector_t<data_t> &_state)->void {
    const uint_t DIM = 1ULL << qubits.size();
    std::complex<data_t> cache = data_[inds[0]];  // the k-th component of non-initialized vector
    for (size_t i = 0; i < DIM; i++) {
      data_[inds[i]] = cache * _state[i];  // set component to psi[k] * state[i]
    }    // (where psi is is the post-reset state of the non-initialized qubits)
   };
  // Use the lambda function
  apply_lambda(lambda, qubits, state);
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::zero() {
  const int_t END = data_size_;    // end for k loop

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < END; ++k) {
    data_[k] = 0.0;
  }
}

template <typename data_t, typename Derived>
cvector_t<data_t> QubitVector<data_t, Derived>::convert(const cvector_t<double>& v) const {
  cvector_t<data_t> ret(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    ret[i] = v[i];
  return ret;
}


template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::set_num_qubits(size_t num_qubits) {

  free_checkpoint();
  if(num_qubits != num_qubits_){
    free_mem();
  }
  data_size_ = BITS[num_qubits];
  allocate_mem(data_size_);

  num_qubits_ = num_qubits;
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::free_mem(){
  if (data_) {
    free(data_);
    data_ = nullptr;
  }
}


template<typename data_t, typename Derived>
void QubitVector<data_t, Derived>::free_checkpoint(){
  if (checkpoint_) {
    free(checkpoint_);
    checkpoint_ = nullptr;
  }
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::allocate_mem(size_t data_size){
  // Free any currently assigned memory
  free_mem();
  // Allocate memory for new vector
  if (data_ == nullptr) {
#ifndef _WIN64
    void* data;
    posix_memalign(&data, 64, sizeof(std::complex<data_t>) * data_size);
    data_ = reinterpret_cast<std::complex<data_t>*>(data);
#else
    data_ = reinterpret_cast<std::complex<data_t>*>(malloc(sizeof(std::complex<data_t>) * data_size));
#endif
  }
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::allocate_checkpoint(size_t data_size){
  free_checkpoint();
#ifndef _WIN64
  void* data;
  posix_memalign(&data, 64, sizeof(std::complex<data_t>) * data_size);
  checkpoint_ = reinterpret_cast<std::complex<data_t>*>(data);
#else
  checkpoint_ = reinterpret_cast<std::complex<data_t>*>(malloc(sizeof(std::complex<data_t>) * data_size));
#endif

}

template <typename data_t, typename Derived>
size_t QubitVector<data_t, Derived>::required_memory_mb(uint_t num_qubits) const {

  size_t unit = std::log2(sizeof(std::complex<data_t>));
  size_t shift_mb = std::max<int_t>(0, num_qubits + unit - 20);
  size_t mem_mb = 1ULL << shift_mb;
  return mem_mb;
}


template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::checkpoint() {

  allocate_checkpoint(data_size_);
  const int_t END = data_size_;    // end for k loop
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < END; ++k)
    checkpoint_[k] = data_[k];
}


template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::revert(bool keep) {

  #ifdef DEBUG
  check_checkpoint();
  #endif

  // If we aren't keeping checkpoint we don't need to copy memory
  // we can simply swap the pointers and free discarded memory
  if (!keep) {
    free_mem();
    data_ = checkpoint_;
    checkpoint_ = nullptr;
    return;
  }
  // Otherwise we need to copy data
  const int_t END = data_size_;    // end for k loop
  #pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < END; ++k)
    data_[k] = checkpoint_[k];

}

template <typename data_t, typename Derived>
std::complex<double> QubitVector<data_t, Derived>::inner_product() const {

  #ifdef DEBUG
  check_checkpoint();
  #endif
  // Lambda function for inner product with checkpoint state
  auto lambda = [&](int_t k, double &val_re, double &val_im)->void {
    const std::complex<double> z = data_[k] * std::conj(checkpoint_[k]);
    val_re += std::real(z);
    val_im += std::imag(z);
  };
  return apply_reduction_lambda(lambda);
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::initialize() {
  zero();
  data_[0] = 1.;
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::initialize_from_vector(const cvector_t<double> &statevec) {
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

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::initialize_from_data(const std::complex<data_t>* statevec, const size_t num_states) {
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

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

template <typename data_t,typename Derived>
void QubitVector<data_t, Derived>::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::set_json_chop_threshold(double threshold) {
  json_chop_threshold_ = threshold;
}

/*******************************************************************************
 *
 * LAMBDA FUNCTION TEMPLATES
 *
 ******************************************************************************/


//------------------------------------------------------------------------------
// State update
//------------------------------------------------------------------------------

template <typename data_t, typename Derived>
template<typename Lambda>
void QubitVector<data_t, Derived>::apply_lambda(Lambda&& func) {
  QV::apply_lambda(0, data_size_, omp_threads_managed(), func);
}

template <typename data_t, typename Derived>
template<typename Lambda, typename list_t>
void QubitVector<data_t, Derived>::apply_lambda(Lambda&& func, const list_t &qubits) {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  QV::apply_lambda(0, data_size_, omp_threads_managed(), func, qubits);
}

template <typename data_t, typename Derived>
template<typename Lambda, typename list_t, typename param_t>
void QubitVector<data_t, Derived>::apply_lambda(Lambda&& func,
                                       const list_t &qubits,
                                       const param_t &params) {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  QV::apply_lambda(0, data_size_, omp_threads_managed(), func, qubits, params);
}


//------------------------------------------------------------------------------
// Reduction Lambda
//------------------------------------------------------------------------------

template <typename data_t, typename Derived>
template<typename Lambda>
std::complex<double>
QubitVector<data_t, Derived>::apply_reduction_lambda(Lambda &&func, size_t start, size_t stop) const {
  return QV::apply_reduction_lambda(start, stop, omp_threads_managed(), func);
}

template <typename data_t, typename Derived>
template<typename Lambda>
std::complex<double> QubitVector<data_t, Derived>::apply_reduction_lambda(Lambda &&func) const {
  return apply_reduction_lambda(std::move(func), size_t(0), data_size_);
}

template <typename data_t, typename Derived>
template<typename Lambda, typename list_t>
std::complex<double>
QubitVector<data_t, Derived>::apply_reduction_lambda(Lambda&& func,
                                            const list_t &qubits) const {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  return QV::apply_reduction_lambda(0, data_size_, omp_threads_managed(), func, qubits);
}


template <typename data_t, typename Derived>
template<typename Lambda, typename list_t, typename param_t>
std::complex<double>
QubitVector<data_t, Derived>::apply_reduction_lambda(Lambda&& func,
                                            const list_t &qubits,
                                            const param_t &params) const {

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  return QV::apply_reduction_lambda(0, data_size_, omp_threads_managed(), func, qubits, params);
}


/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/
template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_matrix(const reg_t &qubits,
                                       const cvector_t<double> &mat) {

  const size_t N = qubits.size();
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Static array optimized lambda functions
  switch (N) {
    case 1:
      apply_matrix(qubits[0], mat);
      return;
    case 2: {
      // Lambda function for 2-qubit matrix multiplication
      auto lambda = [&](const areg_t<4> &inds, const cvector_t<data_t> &_mat)->void {
        std::array<std::complex<data_t>, 4> cache;
        for (size_t i = 0; i < 4; i++) {
          const auto ii = inds[i];
          cache[i] = data_[ii];
          data_[ii] = 0.;
        }
        // update state vector
        for (size_t i = 0; i < 4; i++)
          for (size_t j = 0; j < 4; j++)
            data_[inds[i]] += _mat[i + 4 * j] * cache[j];
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}), convert(mat));
      return;
    }
    case 3: {
      // Lambda function for 3-qubit matrix multiplication
      auto lambda = [&](const areg_t<8> &inds, const cvector_t<data_t> &_mat)->void {
        std::array<std::complex<data_t>, 8> cache;
        for (size_t i = 0; i < 8; i++) {
          const auto ii = inds[i];
          cache[i] = data_[ii];
          data_[ii] = 0.;
        }
        // update state vector
        for (size_t i = 0; i < 8; i++) {
          for (size_t j = 0; j < 8; j++)
            data_[inds[i]] += _mat[i + 8 * j] * cache[j];
        }
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}), convert(mat));
      return;
    }
    case 4: {
      // Lambda function for 4-qubit matrix multiplication
      auto lambda = [&](const areg_t<16> &inds, const cvector_t<data_t> &_mat)->void {
        std::array<std::complex<data_t>, 16> cache;
        for (size_t i = 0; i < 16; i++) {
          const auto ii = inds[i];
          cache[i] = data_[ii];
          data_[ii] = 0.;
        }
        // update state vector
        for (size_t i = 0; i < 16; i++)
          for (size_t j = 0; j < 16; j++)
            data_[inds[i]] += _mat[i + 16 * j] * cache[j];
      };
      apply_lambda(lambda, areg_t<4>({{qubits[0], qubits[1], qubits[2], qubits[3]}}), convert(mat));
      return;
    }
    default: {
      // Lambda function for N-qubit matrix multiplication
      auto lambda = [&](const indexes_t &inds, const cvector_t<data_t> &_mat)->void {
        const uint_t DIM = BITS[N];
        auto cache = std::make_unique<std::complex<data_t>[]>(DIM);
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
      apply_lambda(lambda, qubits, convert(mat));
    }
  } // end switch
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_multiplexer(const reg_t &control_qubits,
                                            const reg_t &target_qubits,
                                            const cvector_t<double>  &mat) {

  auto lambda = [&](const indexes_t &inds, const cvector_t<data_t> &_mat)->void {
    // General implementation
    const size_t control_count = control_qubits.size();
    const size_t target_count  = target_qubits.size();
    const uint_t DIM = BITS[(target_count+control_count)];
    const uint_t columns = BITS[target_count];
    const uint_t blocks = BITS[control_count];
    // Lambda function for stacked matrix multiplication
    auto cache = std::make_unique<std::complex<data_t>[]>(DIM);
    for (uint_t i = 0; i < DIM; i++) {
      const auto ii = inds[i];
      cache[i] = data_[ii];
      data_[ii] = 0.;
    }
    // update state vector
    for (uint_t b = 0; b < blocks; b++)
      for (uint_t i = 0; i < columns; i++)
        for (uint_t j = 0; j < columns; j++)
	{
	  data_[inds[i+b*columns]] += _mat[i+b*columns + DIM * j] * cache[b*columns+j];
	}
  };
  
  // Use the lambda function
  auto qubits = target_qubits;
  for (const auto &q : control_qubits) {qubits.push_back(q);}
  apply_lambda(lambda, qubits, convert(mat));
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t<double> &diag) {

  // Error checking
  #ifdef DEBUG
  check_vector(diag, qubits.size());
  #endif

  if (qubits.size() == 1) {
    apply_diagonal_matrix(qubits[0], diag);
    return;
  }

  auto lambda = [&](const areg_t<2> &inds, const cvector_t<data_t> &_diag)->void {
    for (int_t i = 0; i < 2; ++i) {
      const int_t k = inds[i];
      int_t iv = 0;
      for (int_t j = 0; j < qubits.size(); j++)
        if ((k & (1ULL << qubits[j])) != 0)
          iv += (1ULL << j);
      if (_diag[iv] != (data_t) 1.0)
        data_[k] *= _diag[iv];
    }
  };
  apply_lambda(lambda, areg_t<1>({{qubits[0]}}), convert(diag));
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_permutation_matrix(const reg_t& qubits,
                                                   const std::vector<std::pair<uint_t, uint_t>> &pairs) {
  const size_t N = qubits.size();

  // Error checking
  #ifdef DEBUG
  check_vector(diag, N);
  #endif

  switch (N) {
    case 1: {
      // Lambda function for permutation matrix
      auto lambda = [&](const areg_t<2> &inds)->void {
        for (const auto& p : pairs) {
          std::swap(data_[inds[p.first]], data_[inds[p.second]]);
        }
      };
      apply_lambda(lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for permutation matrix
      auto lambda = [&](const areg_t<4> &inds)->void {
        for (const auto& p : pairs) {
          std::swap(data_[inds[p.first]], data_[inds[p.second]]);
        }
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for permutation matrix
      auto lambda = [&](const areg_t<8> &inds)->void {
        for (const auto& p : pairs) {
          std::swap(data_[inds[p.first]], data_[inds[p.second]]);
        }
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    case 4: {
      // Lambda function for permutation matrix
      auto lambda = [&](const areg_t<16> &inds)->void {
        for (const auto& p : pairs) {
          std::swap(data_[inds[p.first]], data_[inds[p.second]]);
        }
      };
      apply_lambda(lambda, areg_t<4>({{qubits[0], qubits[1], qubits[2], qubits[3]}}));
      return;
    }
    case 5: {
      // Lambda function for permutation matrix
      auto lambda = [&](const areg_t<32> &inds)->void {
        for (const auto& p : pairs) {
          std::swap(data_[inds[p.first]], data_[inds[p.second]]);
        }
      };
      apply_lambda(lambda, areg_t<5>({{qubits[0], qubits[1], qubits[2],
                                       qubits[3], qubits[4]}}));
      return;
    }
    case 6: {
      // Lambda function for permutation matrix
      auto lambda = [&](const areg_t<64> &inds)->void {
        for (const auto& p : pairs) {
          std::swap(data_[inds[p.first]], data_[inds[p.second]]);
        }
      };
      apply_lambda(lambda, areg_t<6>({{qubits[0], qubits[1], qubits[2],
                                       qubits[3], qubits[4], qubits[5]}}));
      return;
    }
    default: {
      // Lambda function for permutation matrix
      auto lambda = [&](const indexes_t &inds)->void {
        for (const auto& p : pairs) {
          std::swap(data_[inds[p.first]], data_[inds[p.second]]);
        }
      };
      // Use the lambda function
      apply_lambda(lambda, qubits);
    }
  } // end switch
}


/*******************************************************************************
 *
 * APPLY OPTIMIZED GATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Multi-controlled gates
//------------------------------------------------------------------------------

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_mcx(const reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = MASKS[N];

  switch (N) {
    case 1: {
      // Lambda function for X gate
      auto lambda = [&](const areg_t<2> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CX gate
      auto lambda = [&](const areg_t<4> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for Toffli gate
      auto lambda = [&](const areg_t<8> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled X gate
      auto lambda = [&](const indexes_t &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, qubits);
    }
  } // end switch
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_mcy(const reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = MASKS[N];
  const std::complex<data_t> I(0., 1.);

  switch (N) {
    case 1: {
      // Lambda function for Y gate
      auto lambda = [&](const areg_t<2> &inds)->void {
        const std::complex<data_t> cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CY gate
      auto lambda = [&](const areg_t<4> &inds)->void {
        const std::complex<data_t> cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for CCY gate
      auto lambda = [&](const areg_t<8> &inds)->void {
        const std::complex<data_t> cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled Y gate
      auto lambda = [&](const indexes_t &inds)->void {
        const std::complex<data_t> cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, qubits);
    }
  } // end switch
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_mcswap(const reg_t &qubits) {
  // Calculate the swap positions for the last two qubits.
  // If N = 2 this is just a regular SWAP gate rather than a controlled-SWAP gate.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = pos0 + BITS[N - 2];

  switch (N) {
    case 2: {
      // Lambda function for SWAP gate
      auto lambda = [&](const areg_t<4> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for C-SWAP gate
      auto lambda = [&](const areg_t<8> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled SWAP gate
      auto lambda = [&](const indexes_t &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, qubits);
    }
  } // end switch
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_mcphase(const reg_t &qubits, const std::complex<double> phase) {
  const size_t N = qubits.size();
  switch (N) {
    case 1: {
      // Lambda function for arbitrary Phase gate with diagonal [1, phase]
      auto lambda = [&](const areg_t<2> &inds)->void {
        data_[inds[1]] *= phase;
      };
      apply_lambda(lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CPhase gate with diagonal [1, 1, 1, phase]
      auto lambda = [&](const areg_t<4> &inds)->void {
        data_[inds[3]] *= phase;
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      auto lambda = [&](const areg_t<8> &inds)->void {
         data_[inds[7]] *= phase;
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled Phase gate
      // with diagonal [1, ..., 1, phase]
      auto lambda = [&](const indexes_t &inds)->void {
         data_[inds[MASKS[N]]] *= phase;
      };
      apply_lambda(lambda, qubits);
    }
  } // end switch
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_mcu(const reg_t &qubits,
                                    const cvector_t<double> &mat) {

  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = MASKS[N];

  // Check if matrix is actually diagonal and if so use
  // diagonal matrix lambda function
  // TODO: this should be changed to not check doubles with ==
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    // Check if actually a phase gate
    if (mat[0] == 1.0) {
      apply_mcphase(qubits, mat[3]);
      return;
    }
    // Otherwise apply general diagonal gate
    const cvector_t<double> diag = {{mat[0], mat[3]}};
    // Diagonal version
    switch (N) {
      case 1: {
        // If N=1 this is just a single-qubit matrix
        apply_diagonal_matrix(qubits[0], diag);
        return;
      }
      case 2: {
        // Lambda function for CU gate
        auto lambda = [&](const areg_t<4> &inds,
                          const cvector_t<data_t> &_diag)->void {
          data_[inds[pos0]] = _diag[0] * data_[inds[pos0]];
          data_[inds[pos1]] = _diag[1] * data_[inds[pos1]];
        };
        apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}), convert(diag));
        return;
      }
      case 3: {
        // Lambda function for CCU gate
        auto lambda = [&](const areg_t<8> &inds,
                          const cvector_t<data_t> &_diag)->void {
          data_[inds[pos0]] = _diag[0] * data_[inds[pos0]];
          data_[inds[pos1]] = _diag[1] * data_[inds[pos1]];
        };
        apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}), convert(diag));
        return;
      }
      default: {
        // Lambda function for general multi-controlled U gate
        auto lambda = [&](const indexes_t &inds,
                          const cvector_t<data_t> &_diag)->void {
          data_[inds[pos0]] = _diag[0] * data_[inds[pos0]];
          data_[inds[pos1]] = _diag[1] * data_[inds[pos1]];
        };
        apply_lambda(lambda, qubits, convert(diag));
        return;
      }
    } // end switch
  }

  // Non-diagonal version
  switch (N) {
    case 1: {
      // If N=1 this is just a single-qubit matrix
      apply_matrix(qubits[0], mat);
      return;
    }
    case 2: {
      // Lambda function for CU gate
      auto lambda = [&](const areg_t<4> &inds,
                        const cvector_t<data_t> &_mat)->void {
      const auto cache = data_[inds[pos0]];
      data_[inds[pos0]] = _mat[0] * data_[inds[pos0]] + _mat[2] * data_[inds[pos1]];
      data_[inds[pos1]] = _mat[1] * cache + _mat[3] * data_[inds[pos1]];
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}), convert(mat));
      return;
    }
    case 3: {
      // Lambda function for CCU gate
      auto lambda = [&](const areg_t<8> &inds,
                        const cvector_t<data_t> &_mat)->void {
      const auto cache = data_[inds[pos0]];
      data_[inds[pos0]] = _mat[0] * data_[inds[pos0]] + _mat[2] * data_[inds[pos1]];
      data_[inds[pos1]] = _mat[1] * cache + _mat[3] * data_[inds[pos1]];
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}), convert(mat));
      return;
    }
    default: {
      // Lambda function for general multi-controlled U gate
      auto lambda = [&](const indexes_t &inds,
                        const cvector_t<data_t> &_mat)->void {
      const auto cache = data_[inds[pos0]];
      data_[inds[pos0]] = _mat[0] * data_[inds[pos0]] + _mat[2] * data_[inds[pos1]];
      data_[inds[pos1]] = _mat[1] * cache + _mat[3] * data_[inds[pos1]];
      };
      apply_lambda(lambda, qubits, convert(mat));
      return;
    }
  } // end switch
}

//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_matrix(const uint_t qubit,
                                       const cvector_t<double>& mat) {

  // Check if matrix is diagonal and if so use optimized lambda
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const cvector_t<double> diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix(qubit, diag);
    return;
  }

  // Convert qubit to array register for lambda functions
  areg_t<1> qubits = {{qubit}};

  // Check if anti-diagonal matrix and if so use optimized lambda
  if(mat[0] == 0.0 && mat[3] == 0.0) {
    if (mat[1] == 1.0 && mat[2] == 1.0) {
      // X-matrix
      auto lambda = [&](const areg_t<2> &inds)->void {
        std::swap(data_[inds[0]], data_[inds[1]]);
      };
      apply_lambda(lambda, qubits);
      return;
    }
    if (mat[2] == 0.0) {
      // Non-unitary projector
      // possibly used in measure/reset/kraus update
      auto lambda = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        data_[inds[1]] = _mat[1] * data_[inds[0]];
        data_[inds[0]] = 0.0;
      };
      apply_lambda(lambda, qubits, convert(mat));
      return;
    }
    if (mat[1] == 0.0) {
      // Non-unitary projector
      // possibly used in measure/reset/kraus update
      auto lambda = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        data_[inds[0]] = _mat[2] * data_[inds[1]];
        data_[inds[1]] = 0.0;
      };
      apply_lambda(lambda, qubits, convert(mat));
      return;
    }
    // else we have a general anti-diagonal matrix
    auto lambda = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat)->void {
      const std::complex<data_t> cache = data_[inds[0]];
      data_[inds[0]] = _mat[2] * data_[inds[1]];
      data_[inds[1]] = _mat[1] * cache;
    };
    apply_lambda(lambda, qubits, convert(mat));
    return;
  }
  // Otherwise general single-qubit matrix multiplication
  auto lambda = [&](const areg_t<2> &inds, const cvector_t<data_t> &_mat)->void {
    const auto cache = data_[inds[0]];
    data_[inds[0]] = _mat[0] * cache + _mat[2] * data_[inds[1]];
    data_[inds[1]] = _mat[1] * cache + _mat[3] * data_[inds[1]];
  };
  apply_lambda(lambda, qubits, convert(mat));
}

template <typename data_t, typename Derived>
void QubitVector<data_t, Derived>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t<double>& diag) {

  // TODO: This should be changed so it isn't checking doubles with ==
  if (diag[0] == 1.0) {  // [[1, 0], [0, z]] matrix
    if (diag[1] == 1.0)
      return; // Identity

    if (diag[1] == std::complex<double>(0., -1.)) { // [[1, 0], [0, -i]]
      auto lambda = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        const auto k = inds[1];
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real() * -1.);
        data_[k].real(cache);
      };
      apply_lambda(lambda, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    if (diag[1] == std::complex<double>(0., 1.)) {
      // [[1, 0], [0, i]]
      auto lambda = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        const auto k = inds[1];
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real());
        data_[k].real(cache * -1.);
      };
      apply_lambda(lambda, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    if (diag[0] == 0.0) {
      // [[1, 0], [0, 0]]
      auto lambda = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        data_[inds[1]] = 0.0;
      };
      apply_lambda(lambda, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    // general [[1, 0], [0, z]]
    auto lambda = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat)->void {
      const auto k = inds[1];
      data_[k] *= _mat[1];
    };
    apply_lambda(lambda, areg_t<1>({{qubit}}), convert(diag));
    return;
  } else if (diag[1] == 1.0) {
    // [[z, 0], [0, 1]] matrix
    if (diag[0] == std::complex<double>(0., -1.)) {
      // [[-i, 0], [0, 1]]
      auto lambda = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        const auto k = inds[1];
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real() * -1.);
        data_[k].real(cache);
      };
      apply_lambda(lambda, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    if (diag[0] == std::complex<double>(0., 1.)) {
      // [[i, 0], [0, 1]]
      auto lambda = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        const auto k = inds[1];
        double cache = data_[k].imag();
        data_[k].imag(data_[k].real());
        data_[k].real(cache * -1.);
      };
      apply_lambda(lambda, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    if (diag[0] == 0.0) {
      // [[0, 0], [0, 1]]
      auto lambda = [&](const areg_t<2> &inds,
                        const cvector_t<data_t> &_mat)->void {
        data_[inds[0]] = 0.0;
      };
      apply_lambda(lambda, areg_t<1>({{qubit}}), convert(diag));
      return;
    }
    // general [[z, 0], [0, 1]]
    auto lambda = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat)->void {
      const auto k = inds[0];
      data_[k] *= _mat[0];
    };
    apply_lambda(lambda, areg_t<1>({{qubit}}), convert(diag));
    return;
  } else {
    // Lambda function for diagonal matrix multiplication
    auto lambda = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat)->void {
      const auto k0 = inds[0];
      const auto k1 = inds[1];
      data_[k0] *= _mat[0];
      data_[k1] *= _mat[1];
    };
    apply_lambda(lambda, areg_t<1>({{qubit}}), convert(diag));
  }
}

/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <typename data_t, typename Derived>
double QubitVector<data_t, Derived>::norm() const {
  // Lambda function for norm
  auto lambda = [&](int_t k, double &val_re, double &val_im)->void {
    (void)val_im; // unused
    val_re += std::real(data_[k] * std::conj(data_[k]));
  };
  return std::real(apply_reduction_lambda(lambda));
}

template <typename data_t, typename Derived>
double QubitVector<data_t, Derived>::norm(const reg_t &qubits, const cvector_t<double> &mat) const {

  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * qubits.size());
  #endif

  // Static array optimized lambda functions
  switch (qubits.size()) {
    case 1:
      return norm(qubits[0], mat);
    case 2: {
      // Lambda function for 2-qubit matrix norm
      auto lambda = [&](const areg_t<4> &inds, const cvector_t<data_t> &_mat,
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 4; i++) {
          std::complex<data_t> vi = 0;
          for (size_t j = 0; j < 4; j++)
            vi += _mat[i + 4 * j] * data_[inds[j]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<2> qubits_arr = {{qubits[0], qubits[1]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, convert(mat)));
    }
    case 3: {
      // Lambda function for 3-qubit matrix norm
      auto lambda = [&](const areg_t<8> &inds, const cvector_t<data_t> &_mat,
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 8; i++) {
          std::complex<data_t> vi = 0;
          for (size_t j = 0; j < 8; j++)
            vi += _mat[i + 8 * j] * data_[inds[j]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<3> qubits_arr = {{qubits[0], qubits[1], qubits[2]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, convert(mat)));
    }
    case 4: {
      // Lambda function for 4-qubit matrix norm
      auto lambda = [&](const areg_t<16> &inds, const cvector_t<data_t> &_mat,
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 16; i++) {
          std::complex<data_t> vi = 0;
          for (size_t j = 0; j < 16; j++)
            vi += _mat[i + 16 * j] * data_[inds[j]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<4> qubits_arr = {{qubits[0], qubits[1], qubits[2], qubits[3]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, convert(mat)));
    }
    default: {
      // Lambda function for N-qubit matrix norm
      auto lambda = [&](const indexes_t &inds, const cvector_t<data_t> &_mat,
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        const uint_t DIM = BITS[qubits.size()];
        for (size_t i = 0; i < DIM; i++) {
          std::complex<data_t> vi = 0;
          for (size_t j = 0; j < DIM; j++)
            vi += _mat[i + DIM * j] * data_[inds[j]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      // Use the lambda function
      return std::real(apply_reduction_lambda(lambda, qubits, convert(mat)));
    }
  } // end switch
}

template <typename data_t, typename Derived>
double QubitVector<data_t, Derived>::norm_diagonal(const reg_t &qubits, const cvector_t<double> &mat) const {

  const uint_t N = qubits.size();

  // Error checking
  #ifdef DEBUG
  check_vector(mat, N);
  #endif

  // Static array optimized lambda functions
  switch (N) {
    case 1:
      return norm_diagonal(qubits[0], mat);
    case 2: {
      // Lambda function for 2-qubit matrix norm
      auto lambda = [&](const areg_t<4> &inds, const cvector_t<data_t> &_mat,
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 4; i++) {
          const auto vi = _mat[i] * data_[inds[i]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<2> qubits_arr = {{qubits[0], qubits[1]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, convert(mat)));
    }
    case 3: {
      // Lambda function for 3-qubit matrix norm
      auto lambda = [&](const areg_t<8> &inds, const cvector_t<data_t> &_mat,
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 8; i++) {
          const auto vi = _mat[i] * data_[inds[i]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<3> qubits_arr = {{qubits[0], qubits[1], qubits[2]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, convert(mat)));
    }
    case 4: {
      // Lambda function for 4-qubit matrix norm
      auto lambda = [&](const areg_t<16> &inds, const cvector_t<data_t> &_mat,
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 16; i++) {
          const auto vi = _mat[i] * data_[inds[i]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<4> qubits_arr = {{qubits[0], qubits[1], qubits[2], qubits[3]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, convert(mat)));
    }
    default: {
      // Lambda function for N-qubit matrix norm
      const uint_t DIM = BITS[N];
      auto lambda = [&](const indexes_t &inds, const cvector_t<data_t> &_mat,
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < DIM; i++) {
          const auto vi = _mat[i] * data_[inds[i]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      // Use the lambda function
      return std::real(apply_reduction_lambda(lambda, qubits, convert(mat)));
    }
  } // end switch
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t, typename Derived>
double QubitVector<data_t, Derived>::norm(const uint_t qubit, const cvector_t<double> &mat) const {
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 2);
  #endif

  // Check if input matrix is diagonal, and if so use diagonal function.
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const cvector_t<double> diag = {{mat[0], mat[3]}};
    return norm_diagonal(qubit, diag);
  }

  // Lambda function for norm reduction to real value.
  auto lambda = [&](const areg_t<2> &inds,
                    const cvector_t<data_t> &_mat,
                    double &val_re,
                    double &val_im)->void {
    (void)val_im; // unused
    const auto v0 = _mat[0] * data_[inds[0]] + _mat[2] * data_[inds[1]];
    const auto v1 = _mat[1] * data_[inds[0]] + _mat[3] * data_[inds[1]];
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_reduction_lambda(lambda, areg_t<1>({{qubit}}), convert(mat)));
}

template <typename data_t, typename Derived>
double QubitVector<data_t, Derived>::norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const {
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 1);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const areg_t<2> &inds,
                    const cvector_t<data_t> &_mat,
                    double &val_re,
                    double &val_im)->void {
    (void)val_im; // unused
    const auto v0 = _mat[0] * data_[inds[0]];
    const auto v1 = _mat[1] * data_[inds[1]];
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_reduction_lambda(lambda, areg_t<1>({{qubit}}), convert(mat)));
}


/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/
template <typename data_t, typename Derived>
double QubitVector<data_t, Derived>::probability(const uint_t outcome) const {
  return std::real(data_[outcome] * std::conj(data_[outcome]));
}

template <typename data_t, typename Derived>
std::vector<double> QubitVector<data_t, Derived>::probabilities() const {
  const int_t END = 1LL << num_qubits();
  std::vector<double> probs(END, 0.);
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < END; j++) {
    probs[j] = probability(j);
  }
  return probs;
}

template <typename data_t, typename Derived>
std::vector<double> QubitVector<data_t, Derived>::probabilities(const reg_t &qubits) const {

  const size_t N = qubits.size();
  const int_t DIM = BITS[N];
  const int_t END = BITS[num_qubits() - N];

  // Error checking
  #ifdef DEBUG
  for (const auto &qubit : qubits)
    check_qubit(qubit);
  #endif

  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
  if ((N == num_qubits_) && (qubits == qubits_sorted))
    return probabilities();

  std::vector<double> probs(DIM, 0.);
  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
    std::vector<data_t> probs_private(DIM, 0.);
    #pragma omp for
      for (int_t k = 0; k < END; k++) {
        auto idx = indexes(qubits, qubits_sorted, k);
        for (int_t m = 0; m < DIM; ++m) {
          probs_private[m] += probability(idx[m]);
        }
      }
    #pragma omp critical
    for (int_t m = 0; m < DIM; ++m) {
      probs[m] += probs_private[m];
    }
  }
  return probs;
}

//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t, typename Derived>
reg_t QubitVector<data_t, Derived>::sample_measure(const std::vector<double> &rnds) const {

  const int_t END = 1LL << num_qubits();
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
          p += probability(sample);
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
          p = probability(k);
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
          p += probability(sample);
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

/*******************************************************************************
 *
 * EXPECTATION VALUES
 *
 ******************************************************************************/

template <typename data_t, typename Derived>
double QubitVector<data_t, Derived>::expval_pauli(const reg_t &qubits,
                                         const std::string &pauli) const {
  // Break string up into Z and X
  // With Y being both Z and X (plus a phase)
  const size_t N = qubits.size();
  uint_t x_mask = 0;
  uint_t z_mask = 0;
  uint_t num_y = 0;
  uint_t x_max = 0;
  for (size_t i = 0; i < N; ++i) {
    const auto bit = BITS[qubits[i]];
    switch (pauli[N - 1 - i]) {
      case 'I':
        break;
      case 'X': {
        x_mask += bit;
        x_max = std::max(x_max, (qubits[i]));
        break;
      }
      case 'Z': {
        z_mask += bit;
        break;
      }
      case 'Y': {
        x_mask += bit;
        x_max = std::max(x_max, (qubits[i]));
        z_mask += bit;
        num_y++;
        break;
      }
      default:
        throw std::invalid_argument("Invalid Pauli \"" + std::to_string(pauli[N - 1 - i]) + "\".");
    }
  }

  // Special case for only I Paulis
  if (x_mask + z_mask == 0) {
    return norm();
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  std::complex<data_t> phase(1, 0);
  switch (num_y & 3) {
    case 0:
      // phase = 1
      break;
    case 1:
      // phase = -1j
      phase = std::complex<data_t>(0, -1);
      break;
    case 2:
      // phase = -1
      phase = std::complex<data_t>(-1, 0);
      break;
    case 3:
      // phase = 1j
      phase = std::complex<data_t>(0, 1);
      break;
  }

  // specialize x_max == 0
  if (!x_mask) {
    auto lambda = [&](const int_t i, double &val_re, double &val_im)->void {
      (void)val_im; // unused
      auto val = std::real(phase * data_[i] * std::conj(data_[i]));
      if (z_mask && (AER::Utils::popcount(i & z_mask) & 1)) {
        val = -val;
      }
      val_re += val;
    };
    return std::real(apply_reduction_lambda(std::move(lambda)));
  }

  const uint_t mask_u = ~MASKS[x_max + 1];
  const uint_t mask_l = MASKS[x_max];
  auto lambda = [&](const int_t i, double &val_re, double &val_im)->void {
    (void)val_im; // unused
    int_t idxs[2];
    idxs[0] = ((i << 1) & mask_u) | (i & mask_l);
    idxs[1] = idxs[0] ^ x_mask;
    double vals[2];
    vals[0] = std::real(phase * data_[idxs[1]] * std::conj(data_[idxs[0]]));
    vals[1] = std::real(phase * data_[idxs[0]] * std::conj(data_[idxs[1]]));
    for (int_t j = 0; j < 2; ++j) {
      if (z_mask && (AER::Utils::popcount(idxs[j] & z_mask) & 1)) {
        val_re -= vals[j];
      } else {
        val_re += vals[j];
      }
    }
  };
  return std::real(apply_reduction_lambda(std::move(lambda), (size_t) 0, (data_size_ >> 1)));
}

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t, typename Derived>
inline std::ostream &operator<<(std::ostream &out, const AER::QV::QubitVector<data_t>&qv) {

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
