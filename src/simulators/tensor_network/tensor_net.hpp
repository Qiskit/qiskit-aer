/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2022.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _tensor_net_hpp_
#define _tensor_net_hpp_

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "simulators/tensor_network/tensor.hpp"

#include "simulators/tensor_network/tensor_net_contractor.hpp"
#if defined(AER_THRUST_CUDA) && defined(AER_CUTENSORNET)
#include "simulators/tensor_network/tensor_net_contractor_cuTensorNet.hpp"
#endif

namespace AER {
namespace TensorNetwork {

template <typename T>
using cvector_t = std::vector<std::complex<T>>;
template <typename T>
using cdict_t = std::map<std::string, std::complex<T>>;

enum class Rotation {
  x,
  y,
  z,
  xx,
  yy,
  zz,
  zx,
};

//============================================================================
// TensorNet class
//============================================================================

template <typename data_t = double>
class TensorNet {
protected:
  uint_t num_qubits_;
  int32_t mode_index_;                                   // index of modes
  std::vector<std::shared_ptr<Tensor<data_t>>> tensors_; // list of tensors
  std::vector<std::shared_ptr<Tensor<data_t>>> qubits_;  // tail tensor for
                                                         // qubits
  std::vector<std::shared_ptr<Tensor<data_t>>>
      qubits_sp_;                        // tail tensor for super qubits
  std::vector<int32_t> modes_qubits_;    // tail mode index for qubits
  std::vector<int32_t> modes_qubits_sp_; // tail mode index for super qubits

  mutable cvector_t<data_t> statevector_; // temporary statevector buffer for
                                          // get_state/save_statevector

  uint_t num_sampling_qubits_ = 10;
  bool use_cuTensorNet_autotuning_ = false;

  bool is_density_matrix_ = false;

  bool cuTensorNet_enable_ = false;

public:
  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  TensorNet();
  explicit TensorNet(size_t num_qubits);
  virtual ~TensorNet();
  TensorNet(const TensorNet &obj);
  TensorNet &operator=(const TensorNet &obj) {}

  //-----------------------------------------------------------------------
  // Data access
  //-----------------------------------------------------------------------

  // Element access
  std::complex<data_t> &operator[](uint_t element);
  std::complex<data_t> operator[](uint_t element) const;

  void set_state(uint_t pos, std::complex<data_t> &val);
  std::complex<data_t> get_state(uint_t pos) const;

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Return the string name of the QUbitVector class
  static std::string name() { return "tensor_net"; }

  // Set the size of the vector in terms of qubit number
  void set_num_qubits(size_t num_qubits);

  // Returns the number of qubits for the current vector
  virtual uint_t num_qubits() const { return num_qubits_; }

  // Returns required memory
  size_t required_memory_mb(uint_t num_qubits) const;

  // Returns a copy of the underlying data_t data as a complex vector
  AER::Vector<std::complex<data_t>> copy_to_vector() const;

  // Moves the data to a complex vector
  AER::Vector<std::complex<data_t>> move_to_vector();

  // Returns a copy of the underlying data_t data as a complex ket dictionary
  cdict_t<data_t> vector_ket(double epsilon = 0) const;

  matrix<std::complex<data_t>> reduced_density_matrix(const reg_t &qubits);

  // Return JSON serialization of TensorNet;
  json_t json() const;

  // State initialization of a component
  // Initialize the specified qubits to a desired statevector
  // (leaving the other qubits in their current state)
  // assuming the qubits being initialized have already been reset to the zero
  // state (using apply_reset)
  void initialize_component(const reg_t &qubits,
                            const cvector_t<double> &state);

  void enable_cuTensorNet(bool flg) { cuTensorNet_enable_ = flg; }

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // initialize from existing state (copy)
  void initialize(const TensorNet<data_t> &obj);

  void initialize_from_matrix(const cmatrix_t &matrix);

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit
  // matrix.
  void apply_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a N-qubit superoperator matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit
  // superop.
  void apply_superop_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a stacked set of 2^control_count target_count--qubit matrix to the
  // state vector. The matrix is input as vector of the column-major vectorized
  // N-qubit matrix.
  void apply_multiplexer(const reg_t &control_qubits,
                         const reg_t &target_qubits,
                         const cvector_t<double> &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a N-qubit diagonal superoperator matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_superop_matrix(const reg_t &qubits,
                                     const cvector_t<double> &mat);

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

  // apply multiple swap gates
  //  qubits is a list of pair of swaps
  void apply_multi_swaps(const reg_t &qubits);

  // apply rotation around axis
  void apply_rotation(const reg_t &qubits, const Rotation r,
                      const double theta);

  void apply_pauli(const reg_t &qubits, const std::string &pauli,
                   const complex_t &coeff = 1);

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
  std::vector<reg_t> sample_measure(const std::vector<double> &rnds) const;

  void apply_reset(const reg_t &qubits);

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
  // The matrix is input as vector of the column-major vectorized N-qubit
  // matrix.
  double norm(const reg_t &qubits, const cvector_t<double> &mat) const;

  //-----------------------------------------------------------------------
  // Expectation Value
  //-----------------------------------------------------------------------

  // These functions return the expectation value <psi|A|psi> for a matrix A.
  // If A is hermitian these will return real values, if A is non-Hermitian
  // they in general will return complex values.

  // Return the expectation value of an N-qubit Pauli matrix.
  // The Pauli is input as a length N string of I,X,Y,Z characters.
  double expval_pauli(const reg_t &qubits, const std::string &pauli,
                      const complex_t initial_phase = 1.0) const;

  //-----------------------------------------------------------------------
  // JSON configuration settings
  //-----------------------------------------------------------------------

  // Set the threshold for chopping values to 0 in JSON
  void set_json_chop_threshold(double threshold);

  // Set the threshold for chopping values to 0 in JSON
  double get_json_chop_threshold() { return 0; }

  //-----------------------------------------------------------------------
  // OpenMP configuration settings
  //-----------------------------------------------------------------------

  // Set the maximum number of OpenMP thread for operations.
  void set_omp_threads(int n);

  // Get the maximum number of OpenMP thread for operations.
  uint_t get_omp_threads() { return 1; }

  // Set the qubit threshold for activating OpenMP.
  // If self.qubits() > threshold OpenMP will be activated.
  void set_omp_threshold(int n);

  // Get the qubit threshold for activating OpenMP.
  uint_t get_omp_threshold() { return 1; }

  void set_num_sampling_qubits(uint_t nq) { num_sampling_qubits_ = nq; }
  void use_autotuning(bool flg) { use_cuTensorNet_autotuning_ = flg; }

protected:
  void add_tensor(const reg_t &qubits, std::vector<std::complex<data_t>> &mat);
  void add_superop_tensor(const reg_t &qubits,
                          std::vector<std::complex<data_t>> &mat);

  void buffer_statevector(void) const;

  void sample_measure_branch(std::vector<reg_t> &samples,
                             const std::vector<double> &rnds,
                             const reg_t &input_sample_index,
                             const reg_t &input_shot_index,
                             const reg_t &input_measured_probs,
                             const uint_t pos_measured) const;
};

// TODO : implement CPU version of contractor
#if defined(AER_THRUST_CUDA) && defined(AER_CUTENSORNET)
#define create_contractor(contractor)                                          \
  contractor = new TensorNetContractor_cuTensorNet<data_t>
#else
#define create_contractor(contractor)                                          \
  contractor = new TensorNetContractorDummy<data_t>
#endif

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

template <typename data_t>
inline void to_json(json_t &js, const TensorNet<data_t> &tn) {
  js = tn.json();
}

template <typename data_t>
json_t TensorNet<data_t>::json() const {
  // TODO : is this required?
  return json_t(1, 0);
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t>
TensorNet<data_t>::TensorNet(size_t num_qubits) : num_qubits_(0) {
  set_num_qubits(num_qubits);
}

template <typename data_t>
TensorNet<data_t>::TensorNet() : TensorNet(0) {}

template <typename data_t>
TensorNet<data_t>::TensorNet(const TensorNet &obj) {}
template <typename data_t>
TensorNet<data_t>::~TensorNet() {
  uint_t i;
  for (i = 0; i < tensors_.size(); i++) {
    tensors_[i].reset();
  }
  for (i = 0; i < num_qubits_; i++) {
    qubits_[i].reset();
    qubits_sp_[i].reset();
  }
}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <typename data_t>
std::complex<data_t> &TensorNet<data_t>::operator[](uint_t element) {
  return 0.0;
}

template <typename data_t>
std::complex<data_t> TensorNet<data_t>::operator[](uint_t element) const {
  return 0.0;
}

template <typename data_t>
void TensorNet<data_t>::set_state(uint_t pos, std::complex<data_t> &val) {}

template <typename data_t>
void TensorNet<data_t>::buffer_statevector(void) const {
  if (is_density_matrix_) {
    throw std::invalid_argument(
        "TensorNet save_statevec/save_statevec/save_amplitudes are not allowed "
        "to use with density matrix operations.");
  }

  TensorNetContractor<data_t> *contractor;
  create_contractor(contractor);
  contractor->set_network(tensors_, false);

  std::vector<int32_t> modes_out(num_qubits_);
  std::vector<int64_t> extents_out(num_qubits_);

  // output tensor
  for (uint_t i = 0; i < num_qubits_; i++) {
    modes_out[i] = modes_qubits_[i];
    extents_out[i] = 2;
  }

  contractor->set_output(modes_out, extents_out);
  contractor->setup_contraction(use_cuTensorNet_autotuning_);
  contractor->contract(statevector_);

  delete contractor;
}

template <typename data_t>
std::complex<data_t> TensorNet<data_t>::get_state(uint_t pos) const {
  if (statevector_.size() == 0)
    buffer_statevector();

  return statevector_[pos];
}

template <typename data_t>
AER::Vector<std::complex<data_t>> TensorNet<data_t>::copy_to_vector() const {
  buffer_statevector();

  return AER::Vector<std::complex<data_t>>::copy_from_buffer(
      statevector_.size(), statevector_.data());
}

template <typename data_t>
AER::Vector<std::complex<data_t>> TensorNet<data_t>::move_to_vector() {
  return copy_to_vector();
}

template <typename data_t>
cdict_t<data_t> TensorNet<data_t>::vector_ket(double epsilon) const {
  buffer_statevector();

  return AER::Utils::vec2ket(statevector_.data(), statevector_.size(), epsilon,
                             16);
}

template <typename data_t>
matrix<std::complex<data_t>>
TensorNet<data_t>::reduced_density_matrix(const reg_t &qubits) {
  uint_t nqubits = qubits.size();

  // connect qubits not to be reduced
  for (uint_t i = 0; i < num_qubits_; i++) {
    bool check = false;
    for (uint_t j = 0; j < qubits.size(); j++) {
      if (i == qubits[j]) {
        check = true;
        break;
      }
    }
    if (!check) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == modes_qubits_sp_[i]) {
          qubits_sp_[i]->modes()[j] = modes_qubits_[i];
          break;
        }
      }
    }
  }

  TensorNetContractor<data_t> *contractor;
  create_contractor(contractor);
  contractor->set_network(tensors_);

  std::vector<int32_t> modes_out(nqubits * 2);
  std::vector<int64_t> extents_out(nqubits * 2);
  std::vector<std::complex<data_t>> trace;

  // output tensor
  for (uint_t i = 0; i < nqubits; i++) {
    modes_out[i] = modes_qubits_[qubits[i]];
    modes_out[i + nqubits] = modes_qubits_sp_[qubits[i]];
    extents_out[i] = 2;
    extents_out[i + nqubits] = 2;
  }

  contractor->set_output(modes_out, extents_out);
  contractor->setup_contraction(use_cuTensorNet_autotuning_);
  contractor->contract(trace);

  delete contractor;

  // recover connectted qubits
  for (uint_t i = 0; i < num_qubits_; i++) {
    bool check = false;
    for (uint_t j = 0; j < qubits.size(); j++) {
      if (i == qubits[j]) {
        check = true;
        break;
      }
    }
    if (!check) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == modes_qubits_[i]) {
          qubits_sp_[i]->modes()[j] = modes_qubits_sp_[i];
          break;
        }
      }
    }
  }

  uint_t size = 1ull << qubits.size();
  return matrix<std::complex<data_t>>::copy_from_buffer(size, size,
                                                        trace.data());
}

//------------------------------------------------------------------------------
// State initialize component
//------------------------------------------------------------------------------
template <typename data_t>
void TensorNet<data_t>::initialize_component(const reg_t &qubits,
                                             const cvector_t<double> &state0) {
  if (statevector_.size() > 0)
    statevector_.clear(); // invalidate statevector buffer

  cvector_t<data_t> state(state0.size());
  for (uint_t i = 0; i < state0.size(); i++)
    state[i] = (std::complex<data_t>)state0[i];

  tensors_.push_back(std::make_shared<Tensor<data_t>>());
  uint_t last = tensors_.size() - 1;
  tensors_[last]->set(qubits, state);
  tensors_.push_back(std::make_shared<Tensor<data_t>>());
  tensors_[last + 1]->set_conj(qubits, state);

  for (uint_t i = 0; i < qubits.size(); i++) {
    modes_qubits_[qubits[i]] = mode_index_;
    tensors_[last]->modes()[i] = mode_index_++;
    qubits_[qubits[i]] = tensors_[last];

    modes_qubits_sp_[qubits[i]] = mode_index_;
    tensors_[last + 1]->modes()[i] = mode_index_++;
    qubits_sp_[qubits[i]] = tensors_[last + 1];
  }

  if (num_qubits_ == qubits.size())
    is_density_matrix_ = false;
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
void TensorNet<data_t>::set_num_qubits(size_t num_qubits) {
  num_qubits_ = num_qubits;
}

template <typename data_t>
size_t TensorNet<data_t>::required_memory_mb(uint_t num_qubits) const {
  return 0;
}

template <typename data_t>
void TensorNet<data_t>::add_tensor(const reg_t &qubits,
                                   std::vector<std::complex<data_t>> &mat) {
  if (statevector_.size() > 0)
    statevector_.clear(); // invalidate statevector buffer

  tensors_.push_back(std::make_shared<Tensor<data_t>>());
  uint_t last = tensors_.size() - 1;
  tensors_[last]->set(qubits, mat);
  for (uint_t i = 0; i < qubits.size(); i++) {
    tensors_[last]->modes()[i] = modes_qubits_[qubits[i]];
    modes_qubits_[qubits[i]] = mode_index_;
    tensors_[last]->modes()[qubits.size() + i] = mode_index_++;
    qubits_[qubits[i]] = tensors_[last];
  }

  tensors_.push_back(std::make_shared<Tensor<data_t>>());
  last++;
  tensors_[last]->set_conj(qubits, mat);
  for (uint_t i = 0; i < qubits.size(); i++) {
    tensors_[last]->modes()[i] = modes_qubits_sp_[qubits[i]];
    modes_qubits_sp_[qubits[i]] = mode_index_;
    tensors_[last]->modes()[qubits.size() + i] = mode_index_++;
    qubits_sp_[qubits[i]] = tensors_[last];
  }
}

template <typename data_t>
void TensorNet<data_t>::add_superop_tensor(
    const reg_t &qubits, std::vector<std::complex<data_t>> &mat) {
  if (statevector_.size() > 0)
    statevector_.clear(); // invalidate statevector buffer

  uint_t size = qubits.size();

  tensors_.push_back(std::make_shared<Tensor<data_t>>());
  uint_t last = tensors_.size() - 1;
  tensors_[last]->set(qubits, mat);

  for (uint_t i = 0; i < size; i++) {
    tensors_[last]->modes()[i] = modes_qubits_[qubits[i]];
    modes_qubits_[qubits[i]] = mode_index_;
    tensors_[last]->modes()[size * 2 + i] = mode_index_++;
    qubits_[qubits[i]] = tensors_[last];
  }
  for (uint_t i = 0; i < size; i++) {
    tensors_[last]->modes()[size + i] = modes_qubits_sp_[qubits[i]];
    modes_qubits_sp_[qubits[i]] = mode_index_;
    tensors_[last]->modes()[size * 3 + i] = mode_index_++;
    qubits_sp_[qubits[i]] = tensors_[last];
  }

  is_density_matrix_ = true;
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

template <typename data_t>
void TensorNet<data_t>::initialize() {
  uint_t i;

  if (statevector_.size() > 0)
    statevector_.clear(); // invalidate statevector buffer

  for (i = 0; i < tensors_.size(); i++) {
    tensors_[i].reset();
  }
  tensors_.clear();

  qubits_.resize(num_qubits_);
  qubits_sp_.resize(num_qubits_);
  modes_qubits_.resize(num_qubits_);
  modes_qubits_sp_.resize(num_qubits_);

  // set initial values for qubits
  std::vector<std::complex<data_t>> init(2);
  init[0] = 1.0;
  init[1] = 0.0;
  for (i = 0; i < num_qubits_; i++) {
    tensors_.push_back(std::make_shared<Tensor<data_t>>());
    uint_t last = tensors_.size() - 1;
    tensors_[last]->set({(int)i}, init);

    modes_qubits_[i] = mode_index_;
    tensors_[last]->modes()[0] = mode_index_++;
    qubits_[i] = tensors_[last];
  }
  for (i = 0; i < num_qubits_; i++) { // for super qubits
    tensors_.push_back(std::make_shared<Tensor<data_t>>());
    uint_t last = tensors_.size() - 1;
    tensors_[last]->set({(int)i}, init);

    modes_qubits_sp_[i] = mode_index_;
    tensors_[last]->modes()[0] = mode_index_++;
    qubits_sp_[i] = tensors_[last];
  }

  is_density_matrix_ = false;
}

template <typename data_t>
void TensorNet<data_t>::initialize(const TensorNet<data_t> &obj) {
  if (statevector_.size() > 0)
    statevector_.clear(); // invalidate statevector buffer

  num_qubits_ = obj.num_qubits_;
  mode_index_ = obj.mode_index_;
  tensors_ = obj.tensors_;
  qubits_ = obj.qubits_;
  qubits_sp_ = obj.qubits_sp_;
  modes_qubits_ = obj.modes_qubits_;
  modes_qubits_sp_ = obj.modes_qubits_sp_;

  num_sampling_qubits_ = obj.num_sampling_qubits_;

  is_density_matrix_ = obj.is_density_matrix_;

  cuTensorNet_enable_ = obj.cuTensorNet_enable_;
}

template <typename data_t>
void TensorNet<data_t>::initialize_from_matrix(const cmatrix_t &matrix0) {
  cvector_t<data_t> matrix(matrix0.size());
  for (uint_t i = 0; i < matrix0.size(); i++)
    matrix[i] = (std::complex<data_t>)matrix0[i];

  tensors_.push_back(std::make_shared<Tensor<data_t>>());
  uint_t last = tensors_.size() - 1;
  tensors_[last]->set(num_qubits_, matrix);

  for (uint_t i = 0; i < num_qubits_; i++) {
    modes_qubits_[i] = mode_index_++;
    tensors_[last]->modes()[i] = modes_qubits_[i];
    qubits_[i] = tensors_[last];
  }
  for (uint_t i = 0; i < num_qubits_; i++) {
    modes_qubits_sp_[i] = mode_index_++;
    tensors_[last]->modes()[i + num_qubits_] = modes_qubits_sp_[i];
    qubits_sp_[i] = tensors_[last];
  }

  is_density_matrix_ = true;
}

/*******************************************************************************
 *
 * CONFIG SETTINGS
 *
 ******************************************************************************/

/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/
template <typename data_t>
void TensorNet<data_t>::apply_matrix(const reg_t &qubits,
                                     const cvector_t<double> &mat) {
  // convert column major to row major
  cvector_t<data_t> matR(mat.size());
  int nr = 1 << qubits.size();
  for (int_t i = 0; i < nr; i++) {
    for (int_t j = 0; j < nr; j++)
      matR[i + j * nr] = mat[j + i * nr];
  }
  add_tensor(qubits, matR);
}

template <typename data_t>
void TensorNet<data_t>::apply_superop_matrix(const reg_t &qubits,
                                             const cvector_t<double> &mat) {
  // convert column major to row major
  cvector_t<data_t> matR(mat.size());
  int nr = 1 << (qubits.size() * 2);
  for (int_t i = 0; i < nr; i++) {
    for (int_t j = 0; j < nr; j++)
      matR[i + j * nr] = mat[j + i * nr];
  }
  add_superop_tensor(qubits, matR);
}

template <typename data_t>
void TensorNet<data_t>::apply_multiplexer(const reg_t &control_qubits,
                                          const reg_t &target_qubits,
                                          const cvector_t<double> &mat) {
  const size_t control_count = control_qubits.size();
  const size_t target_count = target_qubits.size();
  const uint_t DIM = 1ull << (target_count + control_count);
  const uint_t columns = 1ull << target_count;
  const uint_t blocks = 1ull << control_count;

  auto qubits = target_qubits;
  for (const auto &q : control_qubits) {
    qubits.push_back(q);
  }

  cvector_t<double> matMP(DIM * DIM, 0.0);
  uint_t b, i, j;

  // make DIMxDIM matrix
  for (b = 0; b < blocks; b++) {
    for (i = 0; i < columns; i++) {
      for (j = 0; j < columns; j++) {
        matMP[(i + b * columns) + DIM * (b * columns + j)] +=
            mat[i + b * columns + DIM * j];
      }
    }
  }

  apply_matrix(qubits, matMP);
}

template <typename data_t>
void TensorNet<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                              const cvector_t<double> &diag) {
  cvector_t<data_t> mat(diag.size() * diag.size(), 0.0);
  for (uint_t i = 0; i < diag.size(); i++) {
    mat[i * (diag.size() + 1)] = diag[i];
  }

  add_tensor(qubits, mat);
}

template <typename data_t>
void TensorNet<data_t>::apply_diagonal_superop_matrix(
    const reg_t &qubits, const cvector_t<double> &diag) {
  cvector_t<data_t> mat(diag.size() * diag.size(), 0.0);
  for (uint_t i = 0; i < diag.size(); i++) {
    mat[i * (diag.size() + 1)] = diag[i];
  }
  add_superop_tensor(qubits, mat);
}

/*******************************************************************************
 *
 * APPLY OPTIMIZED GATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Multi-controlled gates
//------------------------------------------------------------------------------

template <typename data_t>
void TensorNet<data_t>::apply_mcx(const reg_t &qubits) {
  int n = (1ull << qubits.size());
  cvector_t<data_t> mat(n * n, 0.0);
  for (int i = 0; i < n - 2; i++)
    mat[i * (n + 1)] = 1.0;
  mat[(n - 2) * (n + 1) + 1] = 1.0;
  mat[(n - 1) * (n + 1) - 1] = 1.0;

  reg_t qubits_t;
  qubits_t.push_back(qubits[qubits.size() - 1]);
  for (uint_t i = 0; i < qubits.size() - 1; i++)
    qubits_t.push_back(qubits[i]);

  add_tensor(qubits_t, mat);
}

template <typename data_t>
void TensorNet<data_t>::apply_mcy(const reg_t &qubits) {
  int n = (1ull << qubits.size());
  cvector_t<data_t> mat(n * n, 0.0);
  for (int i = 0; i < n - 2; i++)
    mat[i * (n + 1)] = 1.0;
  mat[(n - 2) * (n + 1) + 1] = {0.0, -1.0};
  mat[(n - 1) * (n + 1) - 1] = {0.0, 1.0};

  reg_t qubits_t;
  qubits_t.push_back(qubits[qubits.size() - 1]);
  for (uint_t i = 0; i < qubits.size() - 1; i++)
    qubits_t.push_back(qubits[i]);

  add_tensor(qubits_t, mat);
}

template <typename data_t>
void TensorNet<data_t>::apply_mcswap(const reg_t &qubits) {
  int n = (1ull << qubits.size());
  cvector_t<data_t> mat(n * n, 0.0);
  for (int i = 0; i < n - 3; i++)
    mat[i * (n + 1)] = 1.0;
  mat[(n - 3) * (n + 1) + 1] = 1.0;
  mat[(n - 2) * (n + 1) - 1] = 1.0;
  mat[(n - 1) * (n + 1)] = 1.0;

  reg_t qubits_t;
  qubits_t.push_back(qubits[qubits.size() - 2]);
  qubits_t.push_back(qubits[qubits.size() - 1]);
  for (uint_t i = 0; i < qubits.size() - 2; i++)
    qubits_t.push_back(qubits[i]);

  add_tensor(qubits_t, mat);
}

template <typename data_t>
void TensorNet<data_t>::apply_mcphase(const reg_t &qubits,
                                      const std::complex<double> phase) {
  int n = (1ull << qubits.size());
  cvector_t<data_t> mat(n * n, 0.0);
  for (int i = 0; i < n - 1; i++)
    mat[i * (n + 1)] = 1.0;
  mat[(n - 1) * (n + 1)] = phase;

  reg_t qubits_t;
  qubits_t.push_back(qubits[qubits.size() - 1]);
  for (uint_t i = 0; i < qubits.size() - 1; i++)
    qubits_t.push_back(qubits[i]);

  add_tensor(qubits_t, mat);
}

template <typename data_t>
void TensorNet<data_t>::apply_mcu(const reg_t &qubits,
                                  const cvector_t<double> &mat) {
  int n = (1ull << qubits.size());
  cvector_t<data_t> matR(n * n, 0.0);
  for (int i = 0; i < n - 2; i++)
    matR[i * (n + 1)] = 1.0;

  matR[(n - 2) * (n + 1)] = mat[0];
  matR[(n - 1) * (n + 1) - 1] = mat[1];
  matR[(n - 2) * (n + 1) + 1] = mat[2];
  matR[(n - 1) * (n + 1)] = mat[3];

  reg_t qubits_t;
  qubits_t.push_back(qubits[qubits.size() - 1]);
  for (uint_t i = 0; i < qubits.size() - 1; i++)
    qubits_t.push_back(qubits[i]);

  add_tensor(qubits_t, matR);
}

template <typename data_t>
void TensorNet<data_t>::apply_rotation(const reg_t &qubits, const Rotation r,
                                       const double theta) {
  switch (r) {
  case Rotation::x:
    apply_mcu(qubits, Linalg::VMatrix::rx(theta));
    break;
  case Rotation::y:
    apply_mcu(qubits, Linalg::VMatrix::ry(theta));
    break;
  case Rotation::z:
    apply_mcu(qubits, Linalg::VMatrix::rz(theta));
    break;
  case Rotation::xx:
    apply_matrix(qubits, Linalg::VMatrix::rxx(theta));
    break;
  case Rotation::yy:
    apply_matrix(qubits, Linalg::VMatrix::ryy(theta));
    break;
  case Rotation::zz:
    apply_diagonal_matrix(qubits, Linalg::VMatrix::rzz_diag(theta));
    break;
  case Rotation::zx:
    apply_matrix(qubits, Linalg::VMatrix::rzx(theta));
    break;
  default:
    throw std::invalid_argument("TensorNet::invalid rotation axis.");
  }
}

/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <typename data_t>
double TensorNet<data_t>::norm() const {
  // connect qubits not used for trace
  for (uint_t i = 1; i < num_qubits_; i++) {
    for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
      if (qubits_sp_[i]->modes()[j] == modes_qubits_sp_[i]) {
        qubits_sp_[i]->modes()[j] = modes_qubits_[i];
        break;
      }
    }
  }

  TensorNetContractor<data_t> *contractor;
  create_contractor(contractor);
  contractor->set_network(tensors_);

  std::vector<int32_t> modes_out(2);
  std::vector<int64_t> extents_out(2);

  // output tensor, only 0 qubit is used for contraction
  modes_out[0] = modes_qubits_[0];
  modes_out[1] = modes_qubits_sp_[0];
  extents_out[0] = 2;
  extents_out[1] = 2;

  contractor->set_output(modes_out, extents_out);
  contractor->setup_contraction(use_cuTensorNet_autotuning_);
  double val = contractor->contract_and_trace(1);

  delete contractor;

  // restore connected qubits
  for (uint_t i = 1; i < num_qubits_; i++) {
    for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
      if (qubits_sp_[i]->modes()[j] == modes_qubits_[i]) {
        qubits_sp_[i]->modes()[j] = modes_qubits_sp_[i];
        break;
      }
    }
  }

  return val;
}

template <typename data_t>
double TensorNet<data_t>::norm(const reg_t &qubits,
                               const cvector_t<double> &mat) const {
  std::vector<std::shared_ptr<Tensor<data_t>>> mat_tensors(2);
  std::vector<int32_t> tmp_modes = modes_qubits_;
  std::vector<int32_t> tmp_modes_sp = modes_qubits_sp_;
  int32_t tmp_index = mode_index_;

  // additional matrix
  std::vector<std::complex<data_t>> mat_t(mat.size());
  for (uint_t i = 0; i < mat.size(); i++)
    mat_t[i] = mat[i];

  mat_tensors[0] = std::make_shared<Tensor<data_t>>();
  mat_tensors[0]->set(qubits, mat_t);
  for (uint_t i = 0; i < qubits.size(); i++) {
    mat_tensors[0]->modes()[i] = tmp_modes[qubits[i]];
    tmp_modes[qubits[i]] = tmp_index;
    mat_tensors[0]->modes()[qubits.size() + i] = tmp_index++;
  }
  mat_tensors[1] = std::make_shared<Tensor<data_t>>();
  mat_tensors[1]->set_conj(qubits, mat_t);
  for (uint_t i = 0; i < qubits.size(); i++) {
    mat_tensors[1]->modes()[i] = tmp_modes_sp[qubits[i]];
    tmp_modes_sp[qubits[i]] = tmp_index;
    mat_tensors[1]->modes()[qubits.size() + i] = tmp_index++;
  }

  // connect qubits not used for trace
  for (uint_t i = 0; i < num_qubits_; i++) {
    if (i != qubits[0]) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == modes_qubits_sp_[i]) {
          qubits_sp_[i]->modes()[j] = tmp_modes[i];
          break;
        }
      }
    }
  }

  TensorNetContractor<data_t> *contractor;
  create_contractor(contractor);
  contractor->set_network(tensors_);
  contractor->allocate_additional_tensors(mat.size() * 2);
  contractor->set_additional_tensors(mat_tensors);

  std::vector<int32_t> modes_out(2);
  std::vector<int64_t> extents_out(2);

  // output tensor, only qubits[0] is used for contraction
  modes_out[0] = tmp_modes[qubits[0]];
  modes_out[1] = tmp_modes_sp[qubits[0]];
  extents_out[0] = 2;
  extents_out[1] = 2;

  contractor->set_output(modes_out, extents_out);
  contractor->setup_contraction(use_cuTensorNet_autotuning_);
  double val = contractor->contract_and_trace(1);

  delete contractor;

  // restore connected qubits
  for (uint_t i = 1; i < num_qubits_; i++) {
    if (i != qubits[0]) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == tmp_modes[i]) {
          qubits_sp_[i]->modes()[j] = modes_qubits_sp_[i];
          break;
        }
      }
    }
  }

  mat_tensors[0].reset();
  mat_tensors[1].reset();

  return val;
}

/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/
template <typename data_t>
double TensorNet<data_t>::probability(const uint_t outcome) const {
  std::complex<data_t> s = get_state(outcome);
  return std::real(s * std::conj(s));
}

template <typename data_t>
std::vector<double> TensorNet<data_t>::probabilities() const {
  reg_t qubits(num_qubits_);
  for (uint_t i = 0; i < num_qubits_; i++)
    qubits[i] = i;
  return probabilities(qubits);
}

template <typename data_t>
std::vector<double>
TensorNet<data_t>::probabilities(const reg_t &qubits) const {
  uint_t nqubits = qubits.size();

  std::vector<int32_t> modes_out(nqubits * 2);
  std::vector<int64_t> extents_out(nqubits * 2);
  std::vector<std::complex<data_t>> trace;
  // connect qubits not to be measured
  for (uint_t i = 0; i < num_qubits_; i++) {
    bool check = false;
    for (uint_t j = 0; j < qubits.size(); j++) {
      if (i == qubits[j]) {
        check = true;
        break;
      }
    }
    if (!check) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == modes_qubits_sp_[i]) {
          qubits_sp_[i]->modes()[j] = modes_qubits_[i];
          break;
        }
      }
    }
  }

  TensorNetContractor<data_t> *contractor;
  create_contractor(contractor);
  contractor->set_network(tensors_);

  // output tensor
  for (uint_t i = 0; i < nqubits; i++) {
    modes_out[i] = modes_qubits_[qubits[i]];
    modes_out[i + nqubits] = modes_qubits_sp_[qubits[i]];
    extents_out[i] = 2;
    extents_out[i + nqubits] = 2;
  }

  contractor->set_output(modes_out, extents_out);
  contractor->setup_contraction(use_cuTensorNet_autotuning_);
  contractor->contract(trace);

  int_t size = 1ull << qubits.size();
  std::vector<double> probs(size, 0.);

  if (omp_get_num_threads() > 1) {
    for (int_t i = 0; i < size; i++)
      probs[i] = std::real(trace[i * (size + 1)]);
  } else {
#pragma omp parallel for
    for (int_t i = 0; i < size; i++)
      probs[i] = std::real(trace[i * (size + 1)]);
  }
  delete contractor;

  // recover connected qubits
  for (uint_t i = 0; i < num_qubits_; i++) {
    bool check = false;
    for (uint_t j = 0; j < qubits.size(); j++) {
      if (i == qubits[j]) {
        check = true;
        break;
      }
    }
    if (!check) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == modes_qubits_[i]) {
          qubits_sp_[i]->modes()[j] = modes_qubits_sp_[i];
          break;
        }
      }
    }
  }
  return probs;
}

template <typename data_t>
void TensorNet<data_t>::apply_reset(const reg_t &qubits) {
  const auto reset_op = Linalg::SMatrix::reset(1ULL << qubits.size());
  apply_superop_matrix(qubits, Utils::vectorize_matrix(reset_op));
}

//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
std::vector<reg_t>
TensorNet<data_t>::sample_measure(const std::vector<double> &rnds) const {
  const int_t SHOTS = rnds.size();
  std::vector<reg_t> samples(SHOTS);
  reg_t sample_index(SHOTS);
  reg_t shot_index(SHOTS);
  reg_t probs(num_qubits_, 0);

  for (int_t i = 0; i < SHOTS; i++)
    shot_index[i] = i;

  sample_measure_branch(samples, rnds, sample_index, shot_index, probs,
                        num_qubits_);

  return samples;
}

template <typename data_t>
void TensorNet<data_t>::sample_measure_branch(std::vector<reg_t> &samples,
                                              const std::vector<double> &rnds,
                                              const reg_t &input_sample_index,
                                              const reg_t &input_shot_index,
                                              const reg_t &input_measured_probs,
                                              const uint_t pos_measured) const {
  const uint_t SHOTS = rnds.size();

  /*---------------------------------------------------------------------------
   |  cccccccccccc  |  oooooooooooooo  |  **************  |  xxxxxxxxxxxxxx  |
   0            -nqubits       pos_measured           +nqubits num_qubits_
     closed             contract here     branch by probs    fixed probs
  ----------------------------------------------------------------------------*/

  uint_t nqubits = num_sampling_qubits_;
  uint_t nqubits_branch = num_sampling_qubits_;
  if (pos_measured == num_qubits_) { // this is 1st call
    nqubits = num_qubits_ % num_sampling_qubits_;
    if (nqubits == 0)
      nqubits = num_sampling_qubits_;
    nqubits_branch = 0;
  } else {
    if (nqubits > pos_measured)
      nqubits = pos_measured;
    if (nqubits_branch + pos_measured > num_qubits_)
      nqubits_branch = num_qubits_ - pos_measured;
  }
  uint_t measured_qubits = num_qubits_ - pos_measured;

  TensorNetContractor<data_t> *contractor;
  create_contractor(contractor);
  contractor->set_network(tensors_);
  if (measured_qubits > 0)
    contractor->allocate_additional_tensors(measured_qubits * 2 * 2);

  // output tensor
  std::vector<int32_t> modes_out(nqubits * 2);
  std::vector<int64_t> extents_out(nqubits * 2);
  for (uint_t i = 0; i < nqubits; i++) {
    modes_out[i] = modes_qubits_[pos_measured - nqubits + i];
    modes_out[i + nqubits] = modes_qubits_sp_[pos_measured - nqubits + i];
    extents_out[i] = 2;
    extents_out[i + nqubits] = 2;
  }
  contractor->set_output(modes_out, extents_out);

  contractor->allocate_sampling_buffers();

  // connect qubits not to be measured
  if (pos_measured - nqubits > 0) {
    for (uint_t i = 0; i < pos_measured - nqubits; i++) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == modes_qubits_sp_[i]) {
          qubits_sp_[i]->modes()[j] = modes_qubits_[i];
          break;
        }
      }
    }
  }

  uint_t num_branches;
  num_branches = 1ull << nqubits_branch;

  // copy shots to each branch
  std::vector<std::vector<double>> shots(num_branches);
  std::vector<reg_t> shot_index(num_branches);
  std::vector<reg_t> sample_index(num_branches);
  if (pos_measured == num_qubits_) { // this is 1st call
    shots[0] = rnds;
    shot_index[0] = input_shot_index;
  } else {
    for (uint_t i = 0; i < SHOTS; i++) {
      shots[input_sample_index[i]].push_back(rnds[i]);
      shot_index[input_sample_index[i]].push_back(input_shot_index[i]);
    }
  }

  // tensors for measuredirmed probabilities
  std::vector<std::shared_ptr<Tensor<data_t>>> measured_tensors;
  if (measured_qubits > 0) {
    measured_tensors.resize(measured_qubits * 2);
    for (uint_t i = 0; i < measured_qubits; i++) {
      std::vector<std::complex<data_t>> prob(2, 0.0);
      prob[input_measured_probs[pos_measured + i]] = 1.0;
      measured_tensors[i * 2] = std::make_shared<Tensor<data_t>>();
      measured_tensors[i * 2 + 1] = std::make_shared<Tensor<data_t>>();
      measured_tensors[i * 2]->set(pos_measured + i, prob);
      measured_tensors[i * 2]->modes()[0] = modes_qubits_[pos_measured + i];
      measured_tensors[i * 2 + 1]->set(pos_measured + i, prob);
      measured_tensors[i * 2 + 1]->modes()[0] =
          modes_qubits_sp_[pos_measured + i];
    }
    contractor->set_additional_tensors(measured_tensors);
  }
  contractor->setup_contraction(use_cuTensorNet_autotuning_);

  // 1st loop, sampling each branch before traversing branches to reuse tensor
  // network
  for (uint_t ib = 0; ib < num_branches; ib++) {
    if (shots[ib].size() > 0) {
      if (nqubits_branch > 0) {
        // tensors for measuredirmed probabilities
        for (uint_t i = 0; i < nqubits_branch; i++) {
          std::vector<std::complex<data_t>> prob(2, 0.0);
          if (((ib >> i) & 1) == 0)
            prob[0] = 1.0;
          else
            prob[1] = 1.0;
          measured_tensors[i * 2]->tensor() = prob;
          measured_tensors[i * 2 + 1]->tensor() = prob;
        }
        contractor->update_additional_tensors(measured_tensors);
      }

      sample_index[ib].resize(shots[ib].size());
      contractor->contract_and_sample_measure(sample_index[ib], shots[ib],
                                              nqubits);
    }
  }

  // recover connected qubits
  if (pos_measured - nqubits > 0) {
    for (uint_t i = 0; i < pos_measured - nqubits; i++) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == modes_qubits_[i]) {
          qubits_sp_[i]->modes()[j] = modes_qubits_sp_[i];
          break;
        }
      }
    }
  }
  for (uint_t i = 0; i < measured_tensors.size(); i++)
    measured_tensors[i].reset();
  delete contractor;

  // 2nd loop traverse branches
  if (pos_measured - nqubits > 0) {
    for (uint_t ib = 0; ib < num_branches; ib++) {
      if (shots[ib].size() > 0) {
        reg_t measured_probs = input_measured_probs;
        for (uint_t i = 0; i < nqubits_branch; i++)
          measured_probs[pos_measured + i] = ((ib >> i) & 1);

        sample_measure_branch(samples, shots[ib], sample_index[ib],
                              shot_index[ib], measured_probs,
                              pos_measured - nqubits);
      }
    }
  } else {
    // save samples
    for (uint_t ib = 0; ib < num_branches; ib++) {
      if (shots[ib].size() > 0) {
        reg_t sample = input_measured_probs;
        for (uint_t i = 0; i < nqubits_branch; i++)
          sample[pos_measured + i] = ((ib >> i) & 1);
        for (uint_t i = 0; i < shots[ib].size(); i++) {
          uint_t shot_id = shot_index[ib][i];
          samples[shot_id] = sample;
          for (uint_t j = 0; j < nqubits; j++) {
            samples[shot_id][j] = ((sample_index[ib][i] >> j) & 1);
          }
        }
      }
    }
  }
}

/*******************************************************************************
 *
 * EXPECTATION VALUES
 *
 ******************************************************************************/

template <typename data_t>
double TensorNet<data_t>::expval_pauli(const reg_t &qubits,
                                       const std::string &pauli,
                                       const complex_t initial_phase) const {
  int_t iqubit = 0;
  double expval = 0.0;
  uint_t size = qubits.size();
  std::vector<std::shared_ptr<Tensor<data_t>>> pauli_tensors;
  std::vector<int32_t> tmp_modes = modes_qubits_;
  int32_t tmp_index = mode_index_;

  pauli_tensors.reserve(size * 2);
  cvector_t<data_t> mat_phase(4, 0.0);
  mat_phase[0] = 1.0;
  mat_phase[3] = initial_phase;

  // add Pauli ops to qubits
  for (uint_t i = 0; i < size; i++) {
    cvector_t<data_t> mat(4, 0.0);

    switch (pauli[size - 1 - i]) {
    case 'I':
      mat[0] = 1.0;
      mat[3] = 1.0;
      break;
    case 'X':
      mat[1] = 1.0;
      mat[2] = 1.0;
      break;
    case 'Y':
      mat[1] = {0.0, -1.0};
      mat[2] = {0.0, 1.0};
      break;
    case 'Z':
      mat[0] = 1.0;
      mat[3] = -1.0;
      break;
    default:
      throw std::invalid_argument("Invalid Pauli \"" +
                                  std::to_string(pauli[size - 1 - i]) + "\".");
      break;
    }
    std::shared_ptr<Tensor<data_t>> t = std::make_shared<Tensor<data_t>>();
    t->set(qubits[iqubit + i], mat);
    t->modes()[0] = tmp_modes[qubits[i]];
    t->modes()[1] = tmp_index++;
    tmp_modes[qubits[i]] = t->modes()[1];
    if (initial_phase != 1.0)
      t->mult_matrix(mat_phase);
    pauli_tensors.push_back(t);
  }

  // connect qubits not used for trace
  for (uint_t i = 0; i < num_qubits_; i++) {
    if (i != qubits[0]) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == modes_qubits_sp_[i]) {
          qubits_sp_[i]->modes()[j] = tmp_modes[i];
          break;
        }
      }
    }
  }

  TensorNetContractor<data_t> *contractor;
  create_contractor(contractor);
  contractor->set_network(tensors_);
  contractor->allocate_additional_tensors(pauli_tensors.size() * 4);
  contractor->set_additional_tensors(pauli_tensors);

  std::vector<int32_t> modes_out(2);
  std::vector<int64_t> extents_out(2);

  // output tensor, only qubits[0] is used for contraction
  modes_out[0] = tmp_modes[qubits[0]];
  modes_out[1] = modes_qubits_sp_[qubits[0]];
  extents_out[0] = 2;
  extents_out[1] = 2;

  contractor->set_output(modes_out, extents_out);
  contractor->setup_contraction(use_cuTensorNet_autotuning_);
  expval = contractor->contract_and_trace(1);

  delete contractor;

  // restore connected qubits
  for (uint_t i = 0; i < num_qubits_; i++) {
    if (i != qubits[0]) {
      for (int_t j = 0; j < qubits_sp_[i]->rank(); j++) {
        if (qubits_sp_[i]->modes()[j] == tmp_modes[i]) {
          qubits_sp_[i]->modes()[j] = modes_qubits_sp_[i];
          break;
        }
      }
    }
  }

  for (uint_t i = 0; i < pauli_tensors.size(); i++) {
    pauli_tensors[i].reset();
  }

  return expval;
}

/*******************************************************************************
 *
 * PAULI
 *
 ******************************************************************************/
template <typename data_t>
void TensorNet<data_t>::apply_pauli(const reg_t &qubits,
                                    const std::string &pauli,
                                    const complex_t &coeff) {
  int_t nqubits = qubits.size();
  cvector_t<data_t> mat_phase(4, 0.0);
  mat_phase[0] = 1.0;
  mat_phase[3] = coeff;

  for (int_t i = 0; i < nqubits; i++) {
    cvector_t<data_t> mat(4, 0.0);

    if (pauli[nqubits - 1 - i] == 'I')
      continue;

    switch (pauli[nqubits - 1 - i]) {
    case 'X':
      mat[1] = 1.0;
      mat[2] = 1.0;
      break;
    case 'Y':
      mat[1] = {0.0, -1.0};
      mat[2] = {0.0, 1.0};
      break;
    case 'Z':
      mat[0] = 1.0;
      mat[3] = -1.0;
      break;
    default:
      throw std::invalid_argument(
          "Invalid Pauli \"" + std::to_string(pauli[nqubits - 1 - i]) + "\".");
      break;
    }
    add_tensor({qubits[i]}, mat);
  }
}

//------------------------------------------------------------------------------
} // namespace TensorNetwork
} // end namespace AER
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &
operator<<(std::ostream &out, const AER::TensorNetwork::TensorNet<data_t> &tn) {
  out << "[";
  out << "]";
  return out;
}

//------------------------------------------------------------------------------
#endif // end module
