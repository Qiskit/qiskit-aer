/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */


#ifndef _qv_qubit_vector_thrust_hpp_
#define _qv_qubit_vector_thrust_hpp_

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
#include <spdlog/spdlog.h>

#include "framework/json.hpp"

#include "simulators/statevector/chunk/chunk_manager.hpp"

//#define AER_DEBUG


namespace AER {
namespace QV {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using reg_t = std::vector<uint_t>;
using indexes_t = std::unique_ptr<uint_t[]>;
template <size_t N> using areg_t = std::array<uint_t, N>;
template <typename T> using cvector_t = std::vector<std::complex<T>>;

//============================================================================
// QubitVectorThrust class
//============================================================================

template <typename data_t = double>
class QubitVectorThrust {

public:

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  QubitVectorThrust();
  explicit QubitVectorThrust(size_t num_qubits);
  virtual ~QubitVectorThrust();

  //-----------------------------------------------------------------------
  // Data access
  //-----------------------------------------------------------------------

  // Element access
  thrust::complex<data_t> &operator[](uint_t element);
  thrust::complex<data_t> operator[](uint_t element) const;

  void set_state(uint_t pos,std::complex<data_t>& c);
  std::complex<data_t> get_state(uint_t pos) const;

  // Returns a reference to the underlying data_t data class
  //  std::complex<data_t>* &data() {return data_;}

  // Returns a copy of the underlying data_t data class
  std::complex<data_t>* data() const {return (std::complex<data_t>*)chunk_->pointer();}

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Return the string name of the QubitVector class
#ifdef AER_THRUST_CUDA
  static std::string name() {return "statevector_gpu";}
#else
  static std::string name() {return "statevector_thrust";}
#endif

  // Set the size of the vector in terms of qubit number
  virtual void set_num_qubits(size_t num_qubits);

  // Returns the number of qubits for the current vector
  virtual uint_t num_qubits() const {return num_qubits_;}

  // Returns the size of the underlying n-qubit vector
  uint_t size() const {return data_size_;}

  // Returns required memory
  size_t required_memory_mb(uint_t num_qubits) const;

  // Returns a copy of the underlying data_t data as a complex vector
  cvector_t<data_t> vector() const;

  // Returns a copy of the underlying data_t data as a complex ket dictionary
  cdict_t<data_t> vector_ket(double epsilon = 0) const;

  // Returns a copy of the underlying data_t data as a complex vector
  AER::Vector<std::complex<data_t>> copy_to_vector() const;

  // Moves the data to a complex vector
  AER::Vector<std::complex<data_t>> move_to_vector();

  // Return JSON serialization of QubitVectorThrust;
  json_t json() const;

  // Set all entries in the vector to 0.
  void zero();


  // State initialization of a component
  // Initialize the specified qubits to a desired statevector
  // (leaving the other qubits in their current state)
  // assuming the qubits being initialized have already been reset to the zero state
  // (using apply_reset)
  void initialize_component(const reg_t &qubits, const cvector_t<double> &state);

  //chunk setup
  void chunk_setup(int chunk_bits,int num_qubits,uint_t chunk_index,uint_t num_local_chunks);

  //cache control for chunks on host
  bool fetch_chunk(void) const;
  void release_chunk(bool write_back = true) const;

  //blocking
  void enter_register_blocking(const reg_t& qubits);
  void leave_register_blocking(void);

  //prepare buffer for MPI send/recv
  thrust::complex<data_t>* send_buffer(uint_t& size_in_byte);
  thrust::complex<data_t>* recv_buffer(uint_t& size_in_byte);

  void release_send_buffer(void) const;
  void release_recv_buffer(void) const;

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
  template <typename list_t>
  void initialize_from_vector(const list_t &vec);
  void initialize_from_vector(const std::vector<std::complex<data_t>>& vec);
  void initialize_from_vector(const AER::Vector<std::complex<data_t>>& vec);

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

  //swap between chunk
  void apply_chunk_swap(const reg_t &qubits, QubitVectorThrust<data_t> &chunk, bool write_back = true);
  void apply_chunk_swap(const reg_t &qubits, uint_t remote_chunk_index);
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
  virtual reg_t sample_measure(const std::vector<double> &rnds) const;

  //-----------------------------------------------------------------------
  // Norms
  //-----------------------------------------------------------------------
  
  // Returns the norm of the current vector
  double norm() const;

  // These functions return the norm <psi|A^dagger.A|psi> obtained by
  // applying a matrix A to the vector. It is equivalent to returning the
  // expectation value of A^\dagger A, and could probably be removed because
  // of this

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
  double expval_pauli(const reg_t &qubits, const std::string &pauli,const complex_t initial_phase=1.0) const;
  //for multi-chunk inter chunk expectation
  double expval_pauli(const reg_t &qubits, const std::string &pauli,
                      const QubitVectorThrust<data_t>& pair_chunk,
                      const uint_t z_count,const uint_t z_count_pair,const complex_t initial_phase=1.0) const;


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

  mutable std::shared_ptr<Chunk<data_t>> chunk_;
  mutable std::shared_ptr<Chunk<data_t>> buffer_chunk_;
  std::shared_ptr<Chunk<data_t>> checkpoint_;
  mutable std::shared_ptr<Chunk<data_t>> send_chunk_;
  mutable std::shared_ptr<Chunk<data_t>> recv_chunk_;
  static ChunkManager<data_t> chunk_manager_;

  uint_t chunk_index_;
  bool multi_chunk_distribution_;
  bool multi_shots_;

  bool register_blocking_;

  //-----------------------------------------------------------------------
  // Config settings
  //----------------------------------------------------------------------- 
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 1;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 1; // Sample measure indexing qubit size
  double json_chop_threshold_ = 0;  // Threshold for choping small values
                                    // in JSON serialization

  //-----------------------------------------------------------------------
  // Error Messages
  //-----------------------------------------------------------------------

  void check_qubit(const uint_t qubit) const;
  void check_vector(const cvector_t<data_t> &diag, uint_t nqubits) const;
  void check_matrix(const cvector_t<data_t> &mat, uint_t nqubits) const;
  void check_dimension(const QubitVectorThrust &qv) const;
  void check_checkpoint() const;

  //-----------------------------------------------------------------------
  // Statevector update with Lambda function
  //-----------------------------------------------------------------------
  template <typename Function>
  void apply_function(Function func) const;

  template <typename Function>
  double apply_function_sum(Function func) const;

#ifdef AER_DEBUG
  //for debugging
  mutable uint_t debug_count;

  void DebugMsg(const char* str,const reg_t &qubits) const;
  void DebugMsg(const char* str,const int qubit) const;
  void DebugMsg(const char* str) const;
  void DebugMsg(const char* str,const std::complex<double> c) const;
  void DebugMsg(const char* str,const double d) const;
  void DebugMsg(const char* str,const std::vector<double>& v) const;
  void DebugDump(void) const;
#endif
};

template <typename data_t>
ChunkManager<data_t> QubitVectorThrust<data_t>::chunk_manager_;


/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

template <typename data_t>
inline void to_json(json_t &js, const QubitVectorThrust<data_t> &qv) {
  js = qv.json();
}

template <typename data_t>
json_t QubitVectorThrust<data_t>::json() const 
{
  thrust::complex<data_t> t;
  uint_t i;

  const json_t ZERO = std::complex<data_t>(0.0, 0.0);
  json_t js = json_t(data_size_, ZERO);

#ifdef AER_DEBUG
  DebugMsg("json()");
#endif

  for(i=0;i<data_size_;i++){
    t = chunk_->Get(i);
    js[i][0] = t.real();
    js[i][1] = t.imag();
  }
  return js;
}

//------------------------------------------------------------------------------
// Error Handling
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits_) {
    std::string error = "QubitVectorThrust: qubit index " + std::to_string(qubit) +
                        " > " + std::to_string(num_qubits_);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_matrix(const cvector_t<data_t> &vec, uint_t nqubits) const {
  const size_t DIM = 1ull << nqubits;
  const auto SIZE = vec.size();
  if (SIZE != DIM * DIM) {
    std::string error = "QubitVectorThrust: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM * DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_vector(const cvector_t<data_t> &vec, uint_t nqubits) const {
  const size_t DIM = 1ull << nqubits;
  const auto SIZE = vec.size();
  if (SIZE != DIM) {
    std::string error = "QubitVectorThrust: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_dimension(const QubitVectorThrust &qv) const {
  if (data_size_ != qv.size_) {
    std::string error = "QubitVectorThrust: vectors are different shape " +
                         std::to_string(data_size_) + " != " +
                         std::to_string(qv.num_states_);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_checkpoint() const {
  if (!checkpoint_) {
    throw std::runtime_error("QubitVectorThrust: checkpoint must exist for inner_product() or revert()");
  }
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t>
QubitVectorThrust<data_t>::QubitVectorThrust(size_t num_qubits) : num_qubits_(0)
{
  chunk_ = nullptr;
  chunk_index_ = 0;
  multi_chunk_distribution_ = false;
  multi_shots_ = false;
  checkpoint_ = nullptr;
  recv_chunk_ = nullptr;
  send_chunk_ = nullptr;

#ifdef AER_DEBUG
  debug_count = 0;
#endif

  if(num_qubits != 0){
    set_num_qubits(num_qubits);
  }
  register_blocking_ = false;
}

template <typename data_t>
QubitVectorThrust<data_t>::QubitVectorThrust() : QubitVectorThrust(0)
{

}

template <typename data_t>
QubitVectorThrust<data_t>::~QubitVectorThrust() 
{
  if(checkpoint_){
    chunk_manager_.UnmapCheckpoint(checkpoint_);
  }

  if(chunk_){
    chunk_manager_.UnmapChunk(chunk_);
  }
}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <typename data_t>
thrust::complex<data_t> &QubitVectorThrust<data_t>::operator[](uint_t element) {
  // Error checking
  if (element > data_size_) {
    std::string error = "QubitVectorThrust: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }

  return (*chunk_)[element];
}


template <typename data_t>
thrust::complex<data_t> QubitVectorThrust<data_t>::operator[](uint_t element) const
{
  // Error checking
  if (element > data_size_) {
    std::string error = "QubitVectorThrust: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }

#ifdef AER_DEBUG
    DebugMsg(" calling []");
#endif

  return (*chunk_)[element];
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_state(uint_t pos, std::complex<data_t>& c)
{
  if(pos < data_size_){
    thrust::complex<data_t> t = c;
    chunk_->Set(pos,t);
  }
}

template <typename data_t>
std::complex<data_t> QubitVectorThrust<data_t>::get_state(uint_t pos) const
{
  std::complex<data_t> ret = 0.0;

  if(pos < data_size_){
    ret = chunk_->Get(pos);
  }
  return ret;
}


template <typename data_t>
cvector_t<data_t> QubitVectorThrust<data_t>::vector() const 
{
  cvector_t<data_t> ret(data_size_, 0.);

  chunk_->CopyOut((thrust::complex<data_t>*)&ret[0], data_size_);

#ifdef AER_DEBUG
  DebugMsg("vector");
#endif

  return ret;
}

template <typename data_t>
cdict_t<data_t> QubitVectorThrust<data_t>::vector_ket(double epsilon) const{
    // non-optimized version; relies on creating a copy of the statevector
    return AER::Utils::vec2ket(vector(), epsilon, 16);
}

template <typename data_t>
AER::Vector<std::complex<data_t>> QubitVectorThrust<data_t>::copy_to_vector() const 
{
  cvector_t<data_t> ret(data_size_, 0.);
  chunk_->CopyOut((thrust::complex<data_t>*)&ret[0], data_size_);

#ifdef AER_DEBUG
  DebugMsg("copy_to_vector");
#endif

  return AER::Vector<std::complex<data_t>>::copy_from_buffer(data_size_, &ret[0]);
}

template <typename data_t>
AER::Vector<std::complex<data_t>> QubitVectorThrust<data_t>::move_to_vector() 
{
  cvector_t<data_t> ret(data_size_, 0.);

#ifdef AER_DEBUG
  DebugMsg("move_to_vector", ret[0]);
  DebugDump();
#endif

  chunk_->CopyOut((thrust::complex<data_t>*)&ret[0], data_size_);

  return AER::Vector<std::complex<data_t>>::copy_from_buffer(data_size_, &ret[0]);
}

//------------------------------------------------------------------------------
// State initialize component
//------------------------------------------------------------------------------
template <typename data_t>
class initialize_component_1qubit_func : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> s0,s1;
  uint_t mask;
  uint_t offset;
public:
  initialize_component_1qubit_func(int qubit,thrust::complex<double> state0,thrust::complex<double> state1)
  {
    s0 = state0;
    s1 = state1;

    mask = (1ull << qubit) - 1;
    offset = 1ull << qubit;
  }

  virtual __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;
    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec0[i0];

    vec0[i0] = s0*q0;
    vec1[i0] = s1*q0;
  }

  const char* name(void)
  {
    return "initialize_component 1 qubit";
  }
};

template <typename data_t>
class initialize_component_func : public GateFuncBase<data_t>
{
protected:
  int nqubits;
  uint_t matSize;
public:
  initialize_component_func(const cvector_t<double>& mat,const reg_t &qb)
  {
    nqubits = qb.size();
    matSize = 1ull << nqubits;
  }

  int qubits_count(void)
  {
    return nqubits;
  }
  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<double> q0;
    thrust::complex<double> q;
    thrust::complex<double>* state;
    uint_t* qubits;
    uint_t* qubits_sorted;
    uint_t j,k;
    uint_t ii,idx,t;
    uint_t mask;

    //get parameters from iterator
    vec = this->data_;
    state = this->matrix_;
    qubits = this->params_;
    qubits_sorted = qubits + nqubits;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      mask = (1ull << qubits_sorted[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    q0 = vec[idx];
    for(k=0;k<matSize;k++){
      ii = idx;
      for(j=0;j<nqubits;j++){
        if(((k >> j) & 1) != 0)
          ii += (1ull << qubits[j]);
      }
      q = q0 * state[k];
      vec[ii] = q;
    }
  }

  const char* name(void)
  {
    return "initialize_component";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_component(const reg_t &qubits, const cvector_t<double> &state0) 
{
  if(qubits.size() == 1){
    apply_function(initialize_component_1qubit_func<data_t>(qubits[0],state0[0],state0[1]) );
  }
  else{
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

    auto qubits_param = qubits;
    int i;
    for(i=0;i<qubits.size();i++)
      qubits_param.push_back(qubits_sorted[i]);

    chunk_->StoreMatrix(state0);
    chunk_->StoreUintParams(qubits_param);

    apply_function(initialize_component_func<data_t>(state0,qubits_sorted) );
  }
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
class ZeroClear : public GateFuncBase<data_t>
{
protected:
public:
  ZeroClear() {}
  bool is_diagonal(void)
  {
    return true;
  }
  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    vec = this->data_;
    vec[i] = 0.0;
  }
  const char* name(void)
  {
    return "zero";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::zero()
{
#ifdef AER_DEBUG
  DebugMsg("zero");
#endif

  apply_function(ZeroClear<data_t>());

#ifdef AER_DEBUG
  DebugMsg("zero done");
#endif
}


template <typename data_t>
void QubitVectorThrust<data_t>::chunk_setup(int chunk_bits,int num_qubits,uint_t chunk_index,uint_t num_local_chunks)
{
  //only first chunk call allocation function
  if(num_local_chunks > 0){
    chunk_manager_.Allocate(chunk_bits,num_qubits,num_local_chunks);
  }

  //set global chunk ID
  chunk_index_ = chunk_index;

  if(chunk_bits < num_qubits){
    multi_chunk_distribution_ = true;
  }

  if(omp_get_num_threads() > 1)
    multi_shots_ = true;
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_num_qubits(size_t num_qubits)
{
  int nid = omp_get_num_threads();

  num_qubits_ = num_qubits;
  data_size_ = 1ull << num_qubits;
  if(!chunk_){
    chunk_ = chunk_manager_.MapChunk(0);
    chunk_->set_num_qubits(num_qubits);
    chunk_->set_chunk_index(chunk_index_);
  }

  chunk_->enable_omp((num_qubits_ > omp_threshold_ && omp_threads_ > 1));

  register_blocking_ = false;

#ifdef AER_DEBUG
  spdlog::debug(" ==== Thrust qubit vector initialization {} qubits ====",num_qubits_);
  if(chunk_->device() >= 0)
    spdlog::debug("    TEST [id={}]: device = {}, pos = {}, place = {} / {}",chunk_index_,chunk_->device(),chunk_->pos(),chunk_->place(),chunk_manager_.num_places());
  else
    spdlog::debug("    TEST [id={}]: allocated on host (place = {})",chunk_index_,chunk_->place());
#endif

}

template <typename data_t>
size_t QubitVectorThrust<data_t>::required_memory_mb(uint_t num_qubits) const {

  size_t unit = std::log2(sizeof(std::complex<data_t>));
  size_t shift_mb = std::max<int_t>(0, num_qubits + unit - 20);
  size_t mem_mb = 1ULL << shift_mb;

  return mem_mb;
}


template <typename data_t>
void QubitVectorThrust<data_t>::checkpoint()
{
#ifdef AER_DEBUG
  DebugMsg("calling checkpoint");
  DebugDump();
#endif

  checkpoint_ = chunk_manager_.MapCheckpoint(chunk_);
  if(checkpoint_){
    chunk_->CopyOut(checkpoint_);
  }

#ifdef AER_DEBUG
  DebugMsg("checkpoint done");
#endif
}


template <typename data_t>
void QubitVectorThrust<data_t>::revert(bool keep) 
{
#ifdef AER_DEBUG
  DebugMsg("calling revert");
#endif
  if(checkpoint_){
    chunk_->CopyIn(checkpoint_);
    chunk_manager_.UnmapCheckpoint(checkpoint_);
  }

#ifdef AER_DEBUG
  DebugMsg("revert");
//  DebugDump();
#endif

}

template <typename data_t>
std::complex<double> QubitVectorThrust<data_t>::inner_product() const
{

#ifdef AER_DEBUG
  DebugMsg("calling inner_product");
#endif

  double dot;
  data_t* vec0;
  data_t* vec1;

  if(!checkpoint_){
    return std::complex<double>(0.0,0.0);
  }

  chunk_->set_device();

  vec0 = (data_t*)chunk_->pointer();
  vec1 = (data_t*)checkpoint_->pointer();
#ifdef AER_THRUST_CUDA
  cudaStream_t strm = chunk_->stream();
  if(strm)
    dot = thrust::inner_product(thrust::device,vec0,vec0 + data_size_*2,vec1,0.0);
  else
    dot = thrust::inner_product(thrust::omp::par,vec0,vec0 + data_size_*2,vec1,0.0);
#else
  if(num_qubits_ > omp_threshold_ && omp_threads_ > 1)
    dot = thrust::inner_product(thrust::device,vec0,vec0 + data_size_*2,vec1,0.0);
  else
    dot = thrust::inner_product(thrust::seq,vec0,vec0 + data_size_*2,vec1,0.0);
#endif

#ifdef AER_DEBUG
  DebugMsg("inner_product",std::complex<double>(dot,0.0));
#endif

  return std::complex<double>(dot,0.0);
}

template <typename data_t>
bool QubitVectorThrust<data_t>::fetch_chunk(void) const
{
  int tid,nid;
  int idev;

  if(chunk_->device() < 0){
    //on host
    idev = 0;
    do{
      buffer_chunk_ = chunk_manager_.MapBufferChunk(idev++ % chunk_manager_.num_devices());
    }while(!buffer_chunk_);
    chunk_->set_cache(buffer_chunk_);
    buffer_chunk_->CopyIn(chunk_);
  }
  else if(multi_chunk_distribution_){
    if(chunk_->pos() != 0){
      return false;
    }
  }

  return true;
}

template <typename data_t>
void QubitVectorThrust<data_t>::release_chunk(bool write_back) const
{
  if(chunk_->device() < 0){
    //on host
    buffer_chunk_->synchronize();
    buffer_chunk_->CopyOut(chunk_);
    chunk_manager_.UnmapBufferChunk(buffer_chunk_);
    chunk_->set_cache(nullptr);
    buffer_chunk_ = nullptr;
  }
  else if(multi_chunk_distribution_){
    if(chunk_->pos() == 0){
      chunk_->synchronize();    //synchronize stream before chunk exchange
    }
  }

}


template <typename data_t>
void QubitVectorThrust<data_t>::enter_register_blocking(const reg_t& qubits)
{
  register_blocking_ = true;
  chunk_->set_blocked_qubits(qubits);
}

template <typename data_t>
void QubitVectorThrust<data_t>::leave_register_blocking(void)
{
  chunk_->apply_blocked_gates();
  register_blocking_ = false;
}


template <typename data_t>
thrust::complex<data_t>* QubitVectorThrust<data_t>::send_buffer(uint_t& size_in_byte)
{
  thrust::complex<data_t>* pRet;

//  send_chunk_.reset();
#ifdef AER_DISABLE_GDR
  if(chunk_->device() < 0){
    pRet = chunk_->pointer();
  }
  else{   //if there is no GPUDirectRDMA support, copy chunk on CPU before using MPI
    send_chunk_ = chunk_manager_.MapBufferChunkOnHost();
    chunk_->CopyOut(send_chunk_);
    pRet = send_chunk_->pointer();
  }
#else
    pRet = chunk_->pointer();
#endif

  size_in_byte = (uint_t)sizeof(thrust::complex<data_t>) << num_qubits_;
  return pRet;
}

template <typename data_t>
thrust::complex<data_t>* QubitVectorThrust<data_t>::recv_buffer(uint_t& size_in_byte)
{

#ifdef AER_DISABLE_GDR
  if(chunk_->device() < 0){
    recv_chunk_ = chunk_manager_.MapBufferChunk(chunk_->place());
  }
  else{   //if there is no GPUDirectRDMA support, receive in CPU memory
    recv_chunk_ = chunk_manager_.MapBufferChunkOnHost();
  }
#else
    recv_chunk_ = chunk_manager_.MapBufferChunk(chunk_->place());
#endif
  if(!recv_chunk_){
    throw std::runtime_error("QubitVectorThrust: receive buffer can not be allocated");
  }

  size_in_byte = (uint_t)sizeof(thrust::complex<data_t>) << num_qubits_;
  return recv_chunk_->pointer();
}

template <typename data_t>
void QubitVectorThrust<data_t>::release_send_buffer(void) const
{
#ifdef AER_DISABLE_GDR
  if(send_chunk_){
    chunk_manager_.UnmapBufferChunk(send_chunk_);
    send_chunk_ = nullptr;
  }
#endif
}

template <typename data_t>
void QubitVectorThrust<data_t>::release_recv_buffer(void) const
{
  if(recv_chunk_){
    chunk_manager_.UnmapBufferChunk(recv_chunk_);
    recv_chunk_ = nullptr;
  }
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::initialize()
{
#ifdef AER_DEBUG
  DebugMsg("initialize");
#endif

  zero();

  thrust::complex<data_t> t;
  t = 1.0;

  if(chunk_index_ == 0){
    chunk_->Set(0,t);
  }

#ifdef AER_DEBUG
  DebugMsg("initialize done");
#endif
}

template <typename data_t>
template <typename list_t>
void QubitVectorThrust<data_t>::initialize_from_vector(const list_t &statevec) {
  if(data_size_ < statevec.size()) {
    std::string error = "QubitVectorThrust::initialize input vector is incorrect length (" + 
                        std::to_string(data_size_) + "!=" +
                        std::to_string(statevec.size()) + ")";
    throw std::runtime_error(error);
  }
#ifdef AER_DEBUG
  DebugMsg("calling initialize_from_vector");
#endif
  // Convert vector data type to complex<data_t>
  AER::Vector<std::complex<data_t>> tmp(data_size_, false);
  int_t i;
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for(i=0; i < data_size_; i++){
    tmp[i] = statevec[i];
  }
  initialize_from_vector(tmp);
}

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_from_vector(const std::vector<std::complex<data_t>>& vec) {
  initialize_from_data(&vec[0], vec.size());
}

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_from_vector(const AER::Vector<std::complex<data_t>>& vec) {
  initialize_from_data(vec.data(), vec.size());
}

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_from_data(const std::complex<data_t>* statevec, const size_t num_states) {
  if (data_size_ != num_states) {
    std::string error = "QubitVectorThrust::initialize input vector is incorrect length (" +
                        std::to_string(data_size_) + "!=" + std::to_string(num_states) + ")";
    throw std::runtime_error(error);
  }

#ifdef AER_DEBUG
  DebugMsg("calling initialize_from_data");
#endif

  chunk_->CopyIn((thrust::complex<data_t>*)(statevec), data_size_);

#ifdef AER_DEBUG
  DebugMsg("initialize_from_data");
  DebugDump();
#endif

}

//--------------------------------------------------------------------------------------
//  gate kernel execution
//--------------------------------------------------------------------------------------

template <typename data_t>
template <typename Function>
void QubitVectorThrust<data_t>::apply_function(Function func) const
{
#ifdef AER_DEBUG
  DebugMsg(func.name());
#endif


  if(func.batch_enable() && multi_chunk_distribution_ && chunk_->device() >= 0){
    if(chunk_->pos() == 0){   //only first chunk on device calculates all the chunks
      func.set_base_index(chunk_index_ << num_qubits_);
      chunk_->Execute(func,chunk_->container()->num_chunks());
    }
  }
  else{
    func.set_base_index(chunk_index_ << num_qubits_);
    chunk_->Execute(func,1);
  }

#ifdef AER_DEBUG
  DebugDump();
#endif
}

template <typename data_t>
template <typename Function>
double QubitVectorThrust<data_t>::apply_function_sum(Function func) const
{
  double ret = 0.0;

#ifdef AER_DEBUG
  DebugMsg(func.name());
#endif

  func.set_base_index(chunk_index_ << num_qubits_);
  ret = chunk_->ExecuteSum(func,1);

  return ret;
}


/*******************************************************************************
 *
 * CONFIG SETTINGS
 *
 ******************************************************************************/

template <typename data_t>
void QubitVectorThrust<data_t>::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_json_chop_threshold(double threshold) {
  json_chop_threshold_ = threshold;
}


/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/
template <typename data_t>
class MatrixMult2x2 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  int qubit;
  uint_t mask;
  uint_t offset0;

public:
  MatrixMult2x2(const cvector_t<double>& mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];

    mask = (1ull << qubit) - 1;

    offset0 = 1ull << qubit;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;
    vec1 = vec0 + offset0;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec0[i0];
    q1 = vec1[i0];

    vec0[i0] = m0 * q0 + m2 * q1;
    vec1[i0] = m1 * q0 + m3 * q1;
  }
  const char* name(void)
  {
    return "mult2x2";
  }
};


template <typename data_t>
class MatrixMult4x4 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m00,m10,m20,m30;
  thrust::complex<double> m01,m11,m21,m31;
  thrust::complex<double> m02,m12,m22,m32;
  thrust::complex<double> m03,m13,m23,m33;
  uint_t mask0;
  uint_t mask1;
  uint_t offset0;
  uint_t offset1;

public:
  MatrixMult4x4(const cvector_t<double>& mat,int qubit0,int qubit1)
  {
    m00 = mat[0];
    m01 = mat[1];
    m02 = mat[2];
    m03 = mat[3];

    m10 = mat[4];
    m11 = mat[5];
    m12 = mat[6];
    m13 = mat[7];

    m20 = mat[8];
    m21 = mat[9];
    m22 = mat[10];
    m23 = mat[11];

    m30 = mat[12];
    m31 = mat[13];
    m32 = mat[14];
    m33 = mat[15];

    offset0 = 1ull << qubit0;
    offset1 = 1ull << qubit1;
    if(qubit0 < qubit1){
      mask0 = offset0 - 1;
      mask1 = offset1 - 1;
    }
    else{
      mask0 = offset1 - 1;
      mask1 = offset0 - 1;
    }
  }

  int qubits_count(void)
  {
    return 2;
  }
  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1,i2;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;
    thrust::complex<data_t>* vec2;
    thrust::complex<data_t>* vec3;
    thrust::complex<data_t> q0,q1,q2,q3;

    vec0 = this->data_;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    vec1 = vec0 + offset0;
    vec2 = vec0 + offset1;
    vec3 = vec2 + offset0;

    q0 = vec0[i0];
    q1 = vec1[i0];
    q2 = vec2[i0];
    q3 = vec3[i0];

    vec0[i0] = m00 * q0 + m10 * q1 + m20 * q2 + m30 * q3;
    vec1[i0] = m01 * q0 + m11 * q1 + m21 * q2 + m31 * q3;
    vec2[i0] = m02 * q0 + m12 * q1 + m22 * q2 + m32 * q3;
    vec3[i0] = m03 * q0 + m13 * q1 + m23 * q2 + m33 * q3;
  }
  const char* name(void)
  {
    return "mult4x4";
  }
};

template <typename data_t>
class MatrixMult8x8 : public GateFuncBase<data_t>
{
protected:
  uint_t offset0;
  uint_t offset1;
  uint_t offset2;
  uint_t mask0;
  uint_t mask1;
  uint_t mask2;

public:
  MatrixMult8x8(const reg_t &qubit,const reg_t &qubit_ordered)
  {
    offset0 = (1ull << qubit[0]);
    offset1 = (1ull << qubit[1]);
    offset2 = (1ull << qubit[2]);

    mask0 = (1ull << qubit_ordered[0]) - 1;
    mask1 = (1ull << qubit_ordered[1]) - 1;
    mask2 = (1ull << qubit_ordered[2]) - 1;
  }

  int qubits_count(void)
  {
    return 3;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1,i2,i3;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
    thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;
    thrust::complex<double>* pMat;

    vec = this->data_;
    pMat = this->matrix_;

    i0 = i & mask0;
    i3 = (i - i0) << 1;
    i1 = i3 & mask1;
    i3 = (i3 - i1) << 1;
    i2 = i3 & mask2;
    i3 = (i3 - i2) << 1;

    i0 = i0 + i1 + i2 + i3;

    q0 = vec[i0];
    q1 = vec[i0 + offset0];
    q2 = vec[i0 + offset1];
    q3 = vec[i0 + offset1 + offset0];
    q4 = vec[i0 + offset2];
    q5 = vec[i0 + offset2 + offset0];
    q6 = vec[i0 + offset2 + offset1];
    q7 = vec[i0 + offset2 + offset1 + offset0];

    m0 = pMat[0];
    m1 = pMat[8];
    m2 = pMat[16];
    m3 = pMat[24];
    m4 = pMat[32];
    m5 = pMat[40];
    m6 = pMat[48];
    m7 = pMat[56];

    vec[i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[1];
    m1 = pMat[9];
    m2 = pMat[17];
    m3 = pMat[25];
    m4 = pMat[33];
    m5 = pMat[41];
    m6 = pMat[49];
    m7 = pMat[57];

    vec[i0 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[2];
    m1 = pMat[10];
    m2 = pMat[18];
    m3 = pMat[26];
    m4 = pMat[34];
    m5 = pMat[42];
    m6 = pMat[50];
    m7 = pMat[58];

    vec[i0 + offset1] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[3];
    m1 = pMat[11];
    m2 = pMat[19];
    m3 = pMat[27];
    m4 = pMat[35];
    m5 = pMat[43];
    m6 = pMat[51];
    m7 = pMat[59];

    vec[i0 + offset1 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[4];
    m1 = pMat[12];
    m2 = pMat[20];
    m3 = pMat[28];
    m4 = pMat[36];
    m5 = pMat[44];
    m6 = pMat[52];
    m7 = pMat[60];

    vec[i0 + offset2] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[5];
    m1 = pMat[13];
    m2 = pMat[21];
    m3 = pMat[29];
    m4 = pMat[37];
    m5 = pMat[45];
    m6 = pMat[53];
    m7 = pMat[61];

    vec[i0 + offset2 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[6];
    m1 = pMat[14];
    m2 = pMat[22];
    m3 = pMat[30];
    m4 = pMat[38];
    m5 = pMat[46];
    m6 = pMat[54];
    m7 = pMat[62];

    vec[i0 + offset2 + offset1] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[7];
    m1 = pMat[15];
    m2 = pMat[23];
    m3 = pMat[31];
    m4 = pMat[39];
    m5 = pMat[47];
    m6 = pMat[55];
    m7 = pMat[63];

    vec[i0 + offset2 + offset1 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
  }
  const char* name(void)
  {
    return "mult8x8";
  }
};

template <typename data_t>
class MatrixMult16x16 : public GateFuncBase<data_t>
{
protected:
  uint_t offset0;
  uint_t offset1;
  uint_t offset2;
  uint_t offset3;
  uint_t mask0;
  uint_t mask1;
  uint_t mask2;
  uint_t mask3;
public:
  MatrixMult16x16(const reg_t &qubit,const reg_t &qubit_ordered)
  {
    offset0 = (1ull << qubit[0]);
    offset1 = (1ull << qubit[1]);
    offset2 = (1ull << qubit[2]);
    offset3 = (1ull << qubit[3]);

    mask0 = (1ull << qubit_ordered[0]) - 1;
    mask1 = (1ull << qubit_ordered[1]) - 1;
    mask2 = (1ull << qubit_ordered[2]) - 1;
    mask3 = (1ull << qubit_ordered[3]) - 1;
  }

  int qubits_count(void)
  {
    return 4;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1,i2,i3,i4,offset,f0,f1,f2;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
    thrust::complex<data_t> q8,q9,q10,q11,q12,q13,q14,q15;
    thrust::complex<double> r;
    thrust::complex<double>* pMat;
    int j;

    vec = this->data_;
    pMat = this->matrix_;

    i0 = i & mask0;
    i4 = (i - i0) << 1;
    i1 = i4 & mask1;
    i4 = (i4 - i1) << 1;
    i2 = i4 & mask2;
    i4 = (i4 - i2) << 1;
    i3 = i4 & mask3;
    i4 = (i4 - i3) << 1;

    i0 = i0 + i1 + i2 + i3 + i4;

    q0 = vec[i0];
    q1 = vec[i0 + offset0];
    q2 = vec[i0 + offset1];
    q3 = vec[i0 + offset1 + offset0];
    q4 = vec[i0 + offset2];
    q5 = vec[i0 + offset2 + offset0];
    q6 = vec[i0 + offset2 + offset1];
    q7 = vec[i0 + offset2 + offset1 + offset0];
    q8 = vec[i0 + offset3];
    q9 = vec[i0 + offset3 + offset0];
    q10 = vec[i0 + offset3 + offset1];
    q11 = vec[i0 + offset3 + offset1 + offset0];
    q12 = vec[i0 + offset3 + offset2];
    q13 = vec[i0 + offset3 + offset2 + offset0];
    q14 = vec[i0 + offset3 + offset2 + offset1];
    q15 = vec[i0 + offset3 + offset2 + offset1 + offset0];

    offset = 0;
    f0 = 0;
    f1 = 0;
    f2 = 0;
    for(j=0;j<16;j++){
      r = pMat[0+j]*q0;
      r += pMat[16+j]*q1;
      r += pMat[32+j]*q2;
      r += pMat[48+j]*q3;
      r += pMat[64+j]*q4;
      r += pMat[80+j]*q5;
      r += pMat[96+j]*q6;
      r += pMat[112+j]*q7;
      r += pMat[128+j]*q8;
      r += pMat[144+j]*q9;
      r += pMat[160+j]*q10;
      r += pMat[176+j]*q11;
      r += pMat[192+j]*q12;
      r += pMat[208+j]*q13;
      r += pMat[224+j]*q14;
      r += pMat[240+j]*q15;

      offset = offset3 * (((uint_t)j >> 3) & 1) + 
               offset2 * (((uint_t)j >> 2) & 1) + 
               offset1 * (((uint_t)j >> 1) & 1) + 
               offset0 *  ((uint_t)j & 1);

      vec[i0 + offset] = r;
    }
  }
  const char* name(void)
  {
    return "mult16x16";
  }
};

template <typename data_t>
class MatrixMultNxN : public GateFuncWithCache<data_t>
{
protected:
public:
  MatrixMultNxN(uint_t nq) : GateFuncWithCache<data_t>(nq)
  {
    ;
  }

  __host__ __device__ void run_with_cache(uint_t _tid,uint_t _idx,thrust::complex<data_t>* _cache) const
  {
    uint_t j,threadID;
    thrust::complex<data_t> q,r;
    thrust::complex<double> m;
    uint_t mat_size,irow;
    thrust::complex<data_t>* vec;
    thrust::complex<double>* pMat;

    vec = this->data_;
    pMat = this->matrix_;

    mat_size = 1ull << this->nqubits_;
    irow = _tid & (mat_size - 1);

    r = 0.0;
    for(j=0;j<mat_size;j++){
      m = pMat[irow + mat_size*j];
      q = _cache[_tid - irow + j];

      r += m*q;
    }

    vec[_idx] = r;
  }

  const char* name(void)
  {
    return "multNxN";
  }

};

//in-place NxN matrix multiplication using LU factorization
template <typename data_t>
class MatrixMultNxN_LU : public GateFuncBase<data_t>
{
protected:
  int nqubits;
  uint_t matSize;
  int nswap;
public:
  MatrixMultNxN_LU(const cvector_t<double>& mat,const reg_t &qb,cvector_t<double>& matLU,reg_t& params)
  {
    uint_t i,j,k,imax;
    std::complex<double> c0,c1;
    double d,dmax;
    uint_t* pSwap;

    nqubits = qb.size();
    matSize = 1ull << nqubits;

    matLU = mat;
    params.resize(nqubits + matSize*2);

    for(k=0;k<nqubits;k++){
      params[k] = qb[k];
    }

    //LU factorization of input matrix
    for(i=0;i<matSize;i++){
      params[nqubits + i] = i;  //init pivot
    }
    for(i=0;i<matSize;i++){
      imax = i;
      dmax = std::abs(matLU[(i << nqubits) + params[nqubits + i]]);
      for(j=i+1;j<matSize;j++){
        d = std::abs(matLU[(i << nqubits) + params[nqubits + j]]);
        if(d > dmax){
          dmax = d;
          imax = j;
        }
      }
      if(imax != i){
        j = params[nqubits + imax];
        params[nqubits + imax] = params[nqubits + i];
        params[nqubits + i] = j;
      }

      if(dmax != 0){
        c0 = matLU[(i << nqubits) + params[nqubits + i]];

        for(j=i+1;j<matSize;j++){
          c1 = matLU[(i << nqubits) + params[nqubits + j]]/c0;

          for(k=i+1;k<matSize;k++){
            matLU[(k << nqubits) + params[nqubits + j]] -= c1*matLU[(k << nqubits) + params[nqubits + i]];
          }
          matLU[(i << nqubits) + params[nqubits + j]] = c1;
        }
      }
    }

    //making table for swapping pivotted result
    pSwap = new uint_t[matSize];
    nswap = 0;
    for(i=0;i<matSize;i++){
      pSwap[i] = params[nqubits + i];
    }
    i = 0;
    while(i<matSize){
      if(pSwap[i] != i){
        params[nqubits + matSize + nswap++] = i;
        j = pSwap[i];
        params[nqubits + matSize + nswap++] = j;
        k = pSwap[j];
        pSwap[j] = j;
        while(i != k){
          j = k;
          params[nqubits + matSize + nswap++] = k;
          k = pSwap[j];
          pSwap[j] = j;
        }
        pSwap[i] = i;
      }
      i++;
    }
    delete[] pSwap;
  }

  int qubits_count(void)
  {
    return nqubits;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q,qt;
    thrust::complex<double> m;
    thrust::complex<double> r;
    uint_t j,k,l,iq;
    uint_t ii,idx,t;
    uint_t mask,offset_j,offset_k;
    thrust::complex<data_t>* vec;
    thrust::complex<double>* pMat;
    uint_t* qubits;
    uint_t* pivot;
    uint_t* table;

    vec = this->data_;
    pMat = this->matrix_;
    qubits = this->params_;

    pivot = qubits + nqubits;
    table = pivot + matSize;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      mask = (1ull << qubits[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    //mult U
    for(j=0;j<matSize;j++){
      r = 0.0;
      for(k=j;k<matSize;k++){
        l = (pivot[j] + (k << nqubits));
        m = pMat[l];

        offset_k = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((k >> iq) & 1) != 0)
            offset_k += (1ull << qubits[iq]);
        }
        q = vec[offset_k+idx];

        r += m*q;
      }
      offset_j = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((j >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      vec[offset_j+idx] = r;
    }

    //mult L
    for(j=matSize-1;j>0;j--){
      offset_j = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((j >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      r = vec[offset_j+idx];

      for(k=0;k<j;k++){
        l = (pivot[j] + (k << nqubits));
        m = pMat[l];

        offset_k = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((k >> iq) & 1) != 0)
            offset_k += (1ull << qubits[iq]);
        }
        q = vec[offset_k+idx];

        r += m*q;
      }
      offset_j = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((j >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      vec[offset_j+idx] = r;
    }

    //swap results
    if(nswap > 0){
      offset_j = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((table[0] >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      q = vec[offset_j+idx];
      k = pivot[table[0]];
      for(j=1;j<nswap;j++){
        offset_j = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((table[j] >> iq) & 1) != 0)
            offset_j += (1ull << qubits[iq]);
        }
        qt = vec[offset_j+idx];

        offset_k = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((k >> iq) & 1) != 0)
            offset_k += (1ull << qubits[iq]);
        }
        vec[offset_k+idx] = q;
        q = qt;
        k = pivot[table[j]];
      }
      offset_k = 0;
      for(iq=0;iq<nqubits;iq++){
        if(((k >> iq) & 1) != 0)
          offset_k += (1ull << qubits[iq]);
      }
      vec[offset_k+idx] = q;
    }
  }
  const char* name(void)
  {
    return "multNxN";
  }
};



template <typename data_t>
void QubitVectorThrust<data_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t<double> &mat)
{
  const size_t N = qubits.size();
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  if(N == 1){
    if(register_blocking_){
      chunk_->queue_blocked_gate('u',qubits[0],0,&mat[0]);
    }
    else{
      apply_function(MatrixMult2x2<data_t>(mat,qubits[0]));
    }
  }
  else if(N == 2){
    apply_function(MatrixMult4x4<data_t>(mat,qubits[0],qubits[1]));
  }
  else if(N <= 10){
    int i;
    for(i=0;i<N;i++){
      qubits_sorted.push_back(qubits[i]);
    }

    chunk_->StoreMatrix(mat);
    chunk_->StoreUintParams(qubits_sorted);
    apply_function(MatrixMultNxN<data_t>(N));
  }
  else{
    cvector_t<double> matLU;
    reg_t params;
    MatrixMultNxN_LU<data_t> f(mat,qubits_sorted,matLU,params);

    chunk_->StoreMatrix(matLU);
    chunk_->StoreUintParams(params);

    apply_function(f);
  }

}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_multiplexer(const reg_t &control_qubits,
                                            const reg_t &target_qubits,
                                            const cvector_t<double>  &mat)
{
  const size_t control_count = control_qubits.size();
  const size_t target_count  = target_qubits.size();
  const uint_t DIM = 1ull << (target_count+control_count);
  const uint_t columns = 1ull << target_count;
  const uint_t blocks = 1ull << control_count;

  auto qubits = target_qubits;
  for (const auto &q : control_qubits) {qubits.push_back(q);}
  size_t N = qubits.size();

  cvector_t<double> matMP(DIM*DIM,0.0);
  uint_t b,i,j;

  //make DIMxDIM matrix
  for(b = 0; b < blocks; b++){
    for(i = 0; i < columns; i++){
      for(j = 0; j < columns; j++){
        matMP[(i+b*columns) + DIM*(b*columns+j)] += mat[i+b*columns + DIM * j];
      }
    }
  }

#ifdef AER_DEBUG
  DebugMsg("apply_multiplexer",control_qubits);
  DebugMsg("                 ",target_qubits);
#endif

  apply_matrix(qubits,matMP);
}

template <typename data_t>
class DiagonalMult2x2 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1;
  int qubit;
public:

  DiagonalMult2x2(const cvector_t<double>& mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q;
    thrust::complex<data_t>* vec;
    thrust::complex<double> m;
    uint_t gid;

    vec = this->data_;
    gid = this->base_index_;

    q = vec[i];
    if((((i + gid) >> qubit) & 1) == 0){
      m = m0;
    }
    else{
      m = m1;
    }

    vec[i] = m * q;
  }
  const char* name(void)
  {
    return "diagonal_mult2x2";
  }
};

template <typename data_t>
class DiagonalMult4x4 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  int qubit0;
  int qubit1;
public:

  DiagonalMult4x4(const cvector_t<double>& mat,int q0,int q1)
  {
    qubit0 = q0;
    qubit1 = q1;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];
  }

  bool is_diagonal(void)
  {
    return true;
  }
  int qubits_count(void)
  {
    return 2;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q;
    thrust::complex<data_t>* vec;
    thrust::complex<double> m;
    uint_t gid;

    vec = this->data_;
    gid = this->base_index_;

    q = vec[i];
    if((((i+gid) >> qubit1) & 1) == 0){
      if((((i+gid) >> qubit0) & 1) == 0){
        m = m0;
      }
      else{
        m = m1;
      }
    }
    else{
      if((((i+gid) >> qubit0) & 1) == 0){
        m = m2;
      }
      else{
        m = m3;
      }
    }

    vec[i] = m * q;
  }
  const char* name(void)
  {
    return "diagonal_mult4x4";
  }
};

template <typename data_t>
class DiagonalMultNxN : public GateFuncBase<data_t>
{
protected:
  int nqubits;
public:
  DiagonalMultNxN(const reg_t &qb)
  {
    nqubits = qb.size();
  }

  bool is_diagonal(void)
  {
    return true;
  }
  int qubits_count(void)
  {
    return nqubits;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t j,im;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q;
    thrust::complex<double> m;
    thrust::complex<double>* pMat;
    uint_t* qubits;
    uint_t gid;

    vec = this->data_;
    gid = this->base_index_;

    pMat = this->matrix_;
    qubits = this->params_;

    im = 0;
    for(j=0;j<nqubits;j++){
      if((((i + gid) >> qubits[j]) & 1) != 0){
        im += (1 << j);
      }
    }

    q = vec[i];
    m = pMat[im];

    vec[i] = m * q;
  }
  const char* name(void)
  {
    return "diagonal_multNxN";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t<double> &diag)
{
  const int_t N = qubits.size();

  if(N == 1){
    if(register_blocking_){
      chunk_->queue_blocked_gate('d',qubits[0],0,&diag[0]);
    }
    else{
      apply_function(DiagonalMult2x2<data_t>(diag,qubits[0]));
    }
  }
  else if(N == 2){
    apply_function(DiagonalMult4x4<data_t>(diag,qubits[0],qubits[1]));
  }
  else{
    chunk_->StoreMatrix(diag);
    chunk_->StoreUintParams(qubits);

    apply_function(DiagonalMultNxN<data_t>(qubits));
  }
}


template <typename data_t>
class Permutation : public GateFuncBase<data_t>
{
protected:
  int nqubits;
  int npairs;

public:
  Permutation(const reg_t& qubits_sorted,const reg_t& qubits,const std::vector<std::pair<uint_t, uint_t>> &pairs,reg_t& params)
  {
    uint_t j,k;
    uint_t offset0,offset1;

    nqubits = qubits.size();
    npairs = pairs.size();

    params.resize(nqubits + npairs*2);

    for(j=0;j<nqubits;j++){ //save masks
      params[j] = (1ull << qubits_sorted[j]) - 1;
    }
    //make offset for pairs
    for(j=0;j<npairs;j++){
      offset0 = 0;
      offset1 = 0;
      for(k=0;k<nqubits;k++){
        if(((pairs[j].first >> k) & 1) != 0){
          offset0 += (1ull << qubits[k]);
        }
        if(((pairs[j].second >> k) & 1) != 0){
          offset1 += (1ull << qubits[k]);
        }
      }
      params[nqubits + j*2  ] = offset0;
      params[nqubits + j*2+1] = offset1;
    }
  }
  int qubits_count(void)
  {
    return nqubits;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    uint_t j;
    uint_t ii,idx,t;
    uint_t* mask;
    uint_t* pairs;

    vec = this->data_;
    mask = this->params_;
    pairs = mask + nqubits;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      t = ii & mask[j];
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    for(j=0;j<npairs;j++){
      q0 = vec[idx + pairs[j*2]];
      q1 = vec[idx + pairs[j*2+1]];

      vec[idx + pairs[j*2]]   = q1;
      vec[idx + pairs[j*2+1]] = q0;
    }
  }
  const char* name(void)
  {
    return "Permutation";
  }
};


template <typename data_t>
void QubitVectorThrust<data_t>::apply_permutation_matrix(const reg_t& qubits,
             const std::vector<std::pair<uint_t, uint_t>> &pairs)
{
  const size_t N = qubits.size();
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  reg_t params;
  Permutation<data_t> f(qubits_sorted,qubits,pairs,params);
  chunk_->StoreUintParams(params);

  apply_function(f);
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
class CX_func : public GateFuncBase<data_t>
{
protected:
  uint_t offset;
  uint_t mask;
  uint_t cmask;
  int nqubits;
  int qubit_t;
public:

  CX_func(const reg_t &qubits)
  {
    int i;
    nqubits = qubits.size();

    qubit_t = qubits[nqubits-1];
    offset = 1ull << qubit_t;
    mask = offset - 1;

    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void)
  {
    return nqubits;
  }
  int num_control_bits(void)
  {
    return nqubits - 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;
    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if((i0 & cmask) == cmask){
      q0 = vec0[i0];
      q1 = vec1[i0];

      vec0[i0] = q1;
      vec1[i0] = q0;
    }
  }
  const char* name(void)
  {
    return "CX";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcx(const reg_t &qubits) 
{
  if(register_blocking_){
    int i;
    uint_t mask = 0;
    for(i=0;i<qubits.size()-1;i++){
      mask |= (1ull << qubits[i]);
    }
    chunk_->queue_blocked_gate('x',qubits[qubits.size()-1],mask);
  }
  else{
    apply_function(CX_func<data_t>(qubits));
  }
}


template <typename data_t>
class CY_func : public GateFuncBase<data_t>
{
protected:
  uint_t mask;
  uint_t cmask;
  uint_t offset;
  int nqubits;
  int qubit_t;
public:
  CY_func(const reg_t &qubits)
  {
    int i;
    nqubits = qubits.size();

    qubit_t = qubits[nqubits-1];
    offset = (1ull << qubit_t);
    mask = (1ull << qubit_t) - 1;

    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void)
  {
    return nqubits;
  }
  int num_control_bits(void)
  {
    return nqubits - 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;

    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if((i0 & cmask) == cmask){
      q0 = vec0[i0];
      q1 = vec1[i0];

      vec0[i0] = thrust::complex<data_t>(q1.imag(),-q1.real());
      vec1[i0] = thrust::complex<data_t>(-q0.imag(),q0.real());
    }
  }
  const char* name(void)
  {
    return "CY";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcy(const reg_t &qubits) 
{
  if(register_blocking_){
    int i;
    uint_t mask = 0;
    for(i=0;i<qubits.size()-1;i++){
      mask |= (1ull << qubits[i]);
    }
    chunk_->queue_blocked_gate('y',qubits[qubits.size()-1],mask);
  }
  else{
    apply_function(CY_func<data_t>(qubits));
  }
}

template <typename data_t>
class CSwap_func : public GateFuncBase<data_t>
{
protected:
  uint_t mask0;
  uint_t mask1;
  uint_t cmask;
  int nqubits;
  int qubit_t0;
  int qubit_t1;
  uint_t offset1;
  uint_t offset2;
public:

  CSwap_func(const reg_t &qubits)
  {
    int i;
    nqubits = qubits.size();

    if(qubits[nqubits-2] < qubits[nqubits-1]){
      qubit_t0 = qubits[nqubits-2];
      qubit_t1 = qubits[nqubits-1];
    }
    else{
      qubit_t1 = qubits[nqubits-2];
      qubit_t0 = qubits[nqubits-1];
    }
    mask0 = (1ull << qubit_t0) - 1;
    mask1 = (1ull << qubit_t1) - 1;

    offset1 = 1ull << qubit_t0;
    offset2 = 1ull << qubit_t1;

    cmask = 0;
    for(i=0;i<nqubits-2;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void)
  {
    return nqubits;
  }
  int num_control_bits(void)
  {
    return nqubits - 2;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1,i2;
    thrust::complex<data_t> q1,q2;
    thrust::complex<data_t>* vec1;
    thrust::complex<data_t>* vec2;

    vec1 = this->data_;

    vec2 = vec1 + offset2;
    vec1 = vec1 + offset1;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    if((i0 & cmask) == cmask){
      q1 = vec1[i0];
      q2 = vec2[i0];
      vec1[i0] = q2;
      vec2[i0] = q1;
    }
  }
  const char* name(void)
  {
    return "CSWAP";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcswap(const reg_t &qubits)
{
  apply_function(CSwap_func<data_t>(qubits));
}


//swap operator between chunks
template <typename data_t>
class CSwapChunk_func : public GateFuncBase<data_t>
{
protected:
  uint_t mask;
  int qubit_t;
  bool write_back_;
  thrust::complex<data_t>* vec0;
  thrust::complex<data_t>* vec1;
public:

  CSwapChunk_func(const reg_t &qubits,thrust::complex<data_t>* pVec0,thrust::complex<data_t>* pVec1,bool wb)
  {
    int i;
    int nqubits;
    nqubits = qubits.size();

    if(qubits[nqubits-2] < qubits[nqubits-1]){
      qubit_t = qubits[nqubits-2];
    }
    else{
      qubit_t = qubits[nqubits-1];
    }
    mask = (1ull << qubit_t) - 1;

    vec0 = pVec0;
    vec1 = pVec1;

    write_back_ = wb;
  }

  uint_t size(int num_qubits)
  {
    return (1ull << (num_qubits - 1));
  }
  int num_control_bits(void)
  {
    //return 1 to claculate "size = 1ull << (num_qubits_ -1)" in apply_function
    return 1;
  }
  bool batch_enable(void)
  {
    return false;
  }

  __host__ __device__  void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;

    i0 = i & mask;
    i1 = (i - i0) << 1;
    i0 += i1;

    q0 = vec0[i0];
    q1 = vec1[i0];
    vec0[i0] = q1;
    if(write_back_)
      vec1[i0] = q0;
  }
  const char* name(void)
  {
    return "Chunk SWAP";
  }
};


template <typename data_t>
void QubitVectorThrust<data_t>::apply_chunk_swap(const reg_t &qubits, QubitVectorThrust<data_t> &src, bool write_back)
{
  int q0,q1,t;


  q0 = qubits[qubits.size() - 2];
  q1 = qubits[qubits.size() - 1];

  if(q0 > q1){
    t = q0;
    q0 = q1;
    q1 = t;
  }


  if(q0 >= num_qubits_){  //exchange whole of chunk each other
#ifdef AER_DEBUG
    DebugMsg("SWAP chunks",qubits);
#endif
    if(write_back){
      chunk_->Swap(src.chunk_);
    }
    else{
      chunk_->CopyIn(src.chunk_);
    }
  }
  else{
    thrust::complex<data_t>* pChunk0;
    thrust::complex<data_t>* pChunk1;
    std::shared_ptr<Chunk<data_t>> pBuffer0 = nullptr;
    std::shared_ptr<Chunk<data_t>> pExec;

    if(chunk_->device() >= 0){
      pExec = chunk_;
      if(chunk_->container()->peer_access(src.chunk_->device())){
        pChunk1 = src.chunk_->pointer();
      }
      else{
        do{
          pBuffer0 = chunk_manager_.MapBufferChunk(chunk_->place());
        }while(!pBuffer0);
        pBuffer0->CopyIn(src.chunk_);
        pChunk1 = pBuffer0->pointer();
      }
      pChunk0 = chunk_->pointer();
    }
    else{
      if(src.chunk_->device() >= 0){
        do{
          pBuffer0 = chunk_manager_.MapBufferChunk(src.chunk_->place());
        }while(!pBuffer0);
        pBuffer0->CopyIn(chunk_);
        pChunk0 = pBuffer0->pointer();
        pChunk1 = src.chunk_->pointer();
        pExec = src.chunk_;
      }
      else{
        pChunk1 = src.chunk_->pointer();
        pChunk0 = chunk_->pointer();
        pExec = chunk_;
      }
    }

    if(chunk_index_ < src.chunk_index_)
      pChunk0 += (1ull << q0);
    else
      pChunk1 += (1ull << q0);

#ifdef AER_DEBUG
    DebugMsg("chunk swap",qubits);
#endif
    pExec->Execute(CSwapChunk_func<data_t>(qubits,pChunk0,pChunk1,true),1 );
    pExec->synchronize();    //should be synchronized here

    if(pBuffer0){
      if(pExec == chunk_)
        pBuffer0->CopyOut(src.chunk_);
      else
        pBuffer0->CopyOut(chunk_);

      chunk_manager_.UnmapBufferChunk(pBuffer0);
    }
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_chunk_swap(const reg_t &qubits, uint_t remote_chunk_index)
{
  int q0,q1,t;


  q0 = qubits[qubits.size() - 2];
  q1 = qubits[qubits.size() - 1];

  if(q0 > q1){
    t = q0;
    q0 = q1;
    q1 = t;
  }

  if(q0 >= num_qubits_){  //exchange whole of chunk each other
#ifdef AER_DEBUG
    DebugMsg("SWAP chunks between process",qubits);
#endif
    chunk_->CopyIn(recv_chunk_);
  }
  else{
    thrust::complex<data_t>* pLocal;
    thrust::complex<data_t>* pRemote;
    std::shared_ptr<Chunk<data_t>> pBuffer;

#ifdef AER_DISABLE_GDR
    if(chunk_->device() >= 0){    //if there is no GPUDirectRDMA support, copy chunk from CPU
      pBuffer = chunk_manager_.MapBufferChunk(chunk_->place());
      pBuffer->CopyIn(recv_chunk_);
      pRemote = pBuffer->pointer();
    }
    else{
      pRemote = recv_chunk_->pointer();
    }
#else
    pRemote = recv_chunk_->pointer();
#endif
    pLocal = chunk_->pointer();

    if(chunk_index_ < remote_chunk_index)
      pLocal += (1ull << q0);
    else
      pRemote += (1ull << q0);

#ifdef AER_DEBUG
    DebugMsg("chunk swap (process)",qubits);
#endif

    chunk_->Execute(CSwapChunk_func<data_t>(qubits,pLocal,pRemote,false),1);
    chunk_->synchronize();    //should be synchronized here

    if(pBuffer){
      chunk_manager_.UnmapBufferChunk(pBuffer);
    }
  }

  release_recv_buffer();

#ifdef AER_DISABLE_GDR
  release_send_buffer();
#endif
}

template <typename data_t>
class phase_func : public GateFuncBase<data_t> 
{
protected:
  thrust::complex<double> phase;
  uint_t mask;
  int nqubits;
public:
  phase_func(const reg_t &qubits,thrust::complex<double> p)
  {
    int i;
    nqubits = qubits.size();
    phase = p;

    mask = 0;
    for(i=0;i<nqubits;i++){
      mask |= (1ull << qubits[i]);
    }
  }
  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t gid;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;

    vec = this->data_;
    gid = this->base_index_;

    if(((i+gid) & mask) == mask){
      q0 = vec[i];
      vec[i] = q0 * phase;
    }
  }
  const char* name(void)
  {
    return "phase";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcphase(const reg_t &qubits, const std::complex<double> phase)
{
  if(register_blocking_){
    int i;
    uint_t mask = 0;
    for(i=0;i<qubits.size()-1;i++){
      mask |= (1ull << qubits[i]);
    }
    chunk_->queue_blocked_gate('p',qubits[qubits.size()-1],mask,&phase);
  }
  else{
    apply_function(phase_func<data_t>(qubits,*(thrust::complex<double>*)&phase) );
  }
}

template <typename data_t>
class DiagonalMult2x2Controlled : public GateFuncBase<data_t> 
{
protected:
  thrust::complex<double> m0,m1;
  uint_t mask;
  uint_t cmask;
  int nqubits;
public:
  DiagonalMult2x2Controlled(const cvector_t<double>& mat,const reg_t &qubits)
  {
    int i;
    nqubits = qubits.size();

    m0 = mat[0];
    m1 = mat[1];

    mask = (1ull << qubits[nqubits-1]) - 1;
    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void)
  {
    return nqubits;
  }
  int num_control_bits(void)
  {
    return nqubits - 1;
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t gid;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    thrust::complex<double> m;

    vec = this->data_;
    gid = this->base_index_;

    if(((i + gid) & cmask) == cmask){
      if((i + gid) & mask){
        m = m1;
      }
      else{
        m = m0;
      }

      q0 = vec[i];
      vec[i] = m*q0;
    }
  }
  const char* name(void)
  {
    return "diagonal_Cmult2x2";
  }
};

template <typename data_t>
class MatrixMult2x2Controlled : public GateFuncBase<data_t> 
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  uint_t mask;
  uint_t cmask;
  uint_t offset;
  int nqubits;
public:
  MatrixMult2x2Controlled(const cvector_t<double>& mat,const reg_t &qubits)
  {
    int i;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];
    nqubits = qubits.size();

    offset = 1ull << qubits[nqubits-1];
    mask = (1ull << qubits[nqubits-1]) - 1;
    cmask = 0;
    for(i=0;i<nqubits-1;i++){
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void)
  {
    return nqubits;
  }
  int num_control_bits(void)
  {
    return nqubits - 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;

    vec0 = this->data_;

    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if((i0 & cmask) == cmask){
      q0 = vec0[i0];
      q1 = vec1[i0];

      vec0[i0] = m0 * q0 + m2 * q1;
      vec1[i0] = m1 * q0 + m3 * q1;
    }
  }
  const char* name(void)
  {
    return "matrix_Cmult2x2";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcu(const reg_t &qubits,
                                    const cvector_t<double> &mat) 
{
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();

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

    if(N == 1){
      // If N=1 this is just a single-qubit matrix
      apply_diagonal_matrix(qubits[0], diag);
      return;
    }
    else{
      if(register_blocking_){
        int i;
        uint_t mask = 0;
        for(i=0;i<qubits.size()-1;i++){
          mask |= (1ull << qubits[i]);
        }
        chunk_->queue_blocked_gate('d',qubits[qubits.size()-1],mask,&diag[0]);
      }
      else{
        apply_function(DiagonalMult2x2Controlled<data_t>(diag,qubits) );
      }
    }
  }
  else{
    if(N == 1){
      // If N=1 this is just a single-qubit matrix
      apply_matrix(qubits[0], mat);
      return;
    }
    else{
      if(register_blocking_){
        int i;
        uint_t mask = 0;
        for(i=0;i<qubits.size()-1;i++){
          mask |= (1ull << qubits[i]);
        }
        chunk_->queue_blocked_gate('u',qubits[qubits.size()-1],mask,&mat[0]);
      }
      else{
        apply_function(MatrixMult2x2Controlled<data_t>(mat,qubits) );
      }
    }
  }
}


//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::apply_matrix(const uint_t qubit,
                                       const cvector_t<double>& mat)
{
  // Check if matrix is diagonal and if so use optimized lambda
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const std::vector<std::complex<double>> diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix(qubit, diag);
    return;
  }
  if(register_blocking_){
    chunk_->queue_blocked_gate('u',qubit,0,&mat[0]);
  }
  else{
    apply_function(MatrixMult2x2<data_t>(mat,qubit));
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t<double>& diag) 
{
  if(register_blocking_){
    chunk_->queue_blocked_gate('d',qubit,0,&diag[0]);
  }
  else{
    reg_t qubits = {qubit};
    apply_function(DiagonalMult2x2<data_t>(diag,qubits[0]));
  }
}
/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <typename data_t>
double QubitVectorThrust<data_t>::norm() const
{
  thrust::complex<double> ret;

  ret = chunk_->norm();

#ifdef AER_DEBUG
  DebugMsg("norm",ret.real() + ret.imag());
#endif

  return ret.real() + ret.imag();
}

template <typename data_t>
class NormMatrixMultNxN : public GateFuncBase<data_t>
{
protected:
  int nqubits;
  uint_t matSize;
public:
  NormMatrixMultNxN(const cvector_t<double>& mat,const reg_t &qb)
  {
    nqubits = qb.size();
    matSize = 1ull << nqubits;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    uint_t offset;
    thrust::complex<double>* pMat;

    thrust::complex<data_t> q;
    thrust::complex<double> m;
    thrust::complex<double> r;
    double sum = 0.0;
    uint_t j,k,l,iq;
    uint_t ii,idx,t;
    uint_t mask;
    uint_t* qubits;

    vec = this->data_;
    pMat = this->matrix_;
    qubits = this->params_;

    idx = 0;
    ii = i;
    for(j=0;j<nqubits;j++){
      mask = (1ull << qubits[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    for(j=0;j<matSize;j++){
      r = 0.0;
      for(k=0;k<matSize;k++){
        l = (j + (k << nqubits));
        m = pMat[l];

        offset = 0;
        for(iq=0;iq<nqubits;iq++){
          if(((k >> iq) & 1) != 0)
            offset += (1ull << qubits[iq]);
        }
        q = vec[offset+idx];
        r += m*q;
      }
      sum += (r.real()*r.real() + r.imag()*r.imag());
    }
    return sum;
  }
  int qubits_count(void)
  {
    return nqubits;
  }
  const char* name(void)
  {
    return "Norm_multNxN";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm(const reg_t &qubits, const cvector_t<double> &mat) const 
{
  const size_t N = qubits.size();

  if(N == 1){
    return norm(qubits[0], mat);
  }
  else{

    chunk_->StoreMatrix(mat);
    chunk_->StoreUintParams(qubits);

    double ret = apply_function_sum(NormMatrixMultNxN<data_t>(mat,qubits));
    return ret;
  }
}

template <typename data_t>
class NormDiagonalMultNxN : public GateFuncBase<data_t>
{
protected:
  int nqubits;
public:
  NormDiagonalMultNxN(const reg_t &qb)
  {
    nqubits = qb.size();
  }

  bool is_diagonal(void)
  {
    return true;
  }
  int qubits_count(void)
  {
    return nqubits;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    uint_t im,j,gid;
    thrust::complex<data_t> q;
    thrust::complex<double> m,r;
    thrust::complex<double>* pMat;
    thrust::complex<data_t>* vec;
    uint_t* qubits;

    vec = this->data_;
    pMat = this->matrix_;
    qubits = this->params_;
    gid = this->base_index_;

    im = 0;
    for(j=0;j<nqubits;j++){
      if(((i+gid) & (1ull << qubits[j])) != 0){
        im += (1 << j);
      }
    }

    q = vec[i];
    m = pMat[im];

    r = m * q;
    return (r.real()*r.real() + r.imag()*r.imag());
  }
  const char* name(void)
  {
    return "Norm_diagonal_multNxN";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const reg_t &qubits, const cvector_t<double> &mat) const {

  const uint_t N = qubits.size();

  if(N == 1){
    return norm_diagonal(qubits[0], mat);
  }
  else{
    chunk_->StoreMatrix(mat);
    chunk_->StoreUintParams(qubits);

    double ret = apply_function_sum(NormDiagonalMultNxN<data_t>(qubits) );
    return ret;
  }
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
class NormMatrixMult2x2 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  int qubit;
  uint_t mask;
  uint_t offset;
public:
  NormMatrixMult2x2(const cvector_t<double> &mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];

    offset = 1ull << qubit;
    mask = (1ull << qubit) - 1;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0,q1;
    thrust::complex<double> r0,r1;
    double sum = 0.0;

    vec = this->data_;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec[i0];
    q1 = vec[offset+i0];

    r0 = m0 * q0 + m2 * q1;
    sum += r0.real()*r0.real() + r0.imag()*r0.imag();
    r1 = m1 * q0 + m3 * q1;
    sum += r1.real()*r1.real() + r1.imag()*r1.imag();
    return sum;
  }
  const char* name(void)
  {
    return "Norm_mult2x2";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm(const uint_t qubit, const cvector_t<double> &mat) const
{
  double ret = apply_function_sum(NormMatrixMult2x2<data_t>(mat,qubit));

  return ret;
}


template <typename data_t>
class NormDiagonalMult2x2 : public GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1;
  int qubit;
public:
  NormDiagonalMult2x2(cvector_t<double> &mat,int q)
  {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    uint_t gid;
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q;
    thrust::complex<double> m,r;

    vec = this->data_;
    gid = this->base_index_;

    q = vec[i];
    if((((i+gid) >> qubit) & 1) == 0){
      m = m0;
    }
    else{
      m = m1;
    }

    r = m * q;

    return (r.real()*r.real() + r.imag()*r.imag());
  }
  const char* name(void)
  {
    return "Norm_diagonal_mult2x2";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const
{
  double ret = apply_function_sum(NormDiagonalMult2x2<data_t>(mat,qubit));

  return ret;
}



/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/
template <typename data_t>
double QubitVectorThrust<data_t>::probability(const uint_t outcome) const 
{

  std::complex<data_t> ret;
  ret = (std::complex<data_t>)chunk_->Get(outcome);

  return std::real(ret)*std::real(ret) + std::imag(ret) * std::imag(ret);
}

template <typename data_t>
std::vector<double> QubitVectorThrust<data_t>::probabilities() const {
  const int_t END = 1LL << num_qubits();
  std::vector<double> probs(END, 0.);
#ifdef AER_DEBUG
  DebugMsg("calling probabilities");
#endif

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < END; j++) {
    probs[j] = probability(j);
  }

#ifdef AER_DEBUG
  DebugMsg("probabilities",probs);
#endif
  return probs;
}


template <typename data_t>
class probability_func : public GateFuncBase<data_t>
{
protected:
  uint_t mask;
  uint_t cmask;
public:
  probability_func(const reg_t &qubits,int i)
  {
    int k;
    int nq = qubits.size();

    mask = 0;
    cmask = 0;
    for(k=0;k<nq;k++){
      mask |= (1ull << qubits[k]);

      if(((i >> k) & 1) != 0){
        cmask |= (1ull << qubits[k]);
      }
    }
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q;
    thrust::complex<data_t>* vec;
    double ret;

    vec = this->data_;

    ret = 0.0;

    if((i & mask) == cmask){
      q = vec[i];
      ret = q.real()*q.real() + q.imag()*q.imag();
    }
    return ret;
  }

  const char* name(void)
  {
    return "probabilities";
  }
};

template <typename data_t>
std::vector<double> QubitVectorThrust<data_t>::probabilities(const reg_t &qubits) const 
{
  const size_t N = qubits.size();
  const int_t DIM = 1 << N;

  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
  if ((N == num_qubits_) && (qubits == qubits_sorted))
    return probabilities();

  std::vector<double> probs(DIM, 0.);

  int i;
  for(i=0;i<DIM;i++){
    probs[i] = apply_function_sum(probability_func<data_t>(qubits,i));
  }

#ifdef AER_DEBUG
  DebugMsg("probabilities",probs);
#endif

  return probs;
}

//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
reg_t QubitVectorThrust<data_t>::sample_measure(const std::vector<double> &rnds) const
{
#ifdef AER_DEBUG
  reg_t samples;
  DebugMsg("sample_measure begin");
  samples = chunk_->sample_measure(rnds);
  DebugMsg("sample_measure",samples);
  return samples;
#else
  return chunk_->sample_measure(rnds);
#endif
}



/*******************************************************************************
 *
 * EXPECTATION VALUES
 *
 ******************************************************************************/

inline __host__ __device__ uint_t pop_count_kernel(uint_t val)
{
  uint_t count = val;
  count = (count & 0x5555555555555555) + ((count >> 1) & 0x5555555555555555);
  count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
  count = (count & 0x0f0f0f0f0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f0f0f0f0f);
  count = (count & 0x00ff00ff00ff00ff) + ((count >> 8) & 0x00ff00ff00ff00ff);
  count = (count & 0x0000ffff0000ffff) + ((count >> 16) & 0x0000ffff0000ffff);
  count = (count & 0x00000000ffffffff) + ((count >> 32) & 0x00000000ffffffff);
  return count;
}

//special case Z only
template <typename data_t>
class expval_pauli_Z_func : public GateFuncBase<data_t>
{
protected:
  uint_t z_mask_;

public:
  expval_pauli_Z_func(uint_t z)
  {
    z_mask_ = z;
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    double ret = 0.0;

    vec = this->data_;

    q0 = vec[i];
    ret = q0.real()*q0.real() + q0.imag()*q0.imag();

    if(z_mask_ != 0){
      if(pop_count_kernel(i & z_mask_) & 1)
        ret = -ret;
    }

    return ret;
  }
  const char* name(void)
  {
    return "expval_pauli_Z";
  }
};

template <typename data_t>
class expval_pauli_XYZ_func : public GateFuncBase<data_t>
{
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  uint_t mask_l_;
  uint_t mask_u_;
  thrust::complex<data_t> phase_;
public:
  expval_pauli_XYZ_func(uint_t x,uint_t z,uint_t x_max,std::complex<data_t> p)
  {
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;

    mask_u_ = ~((1ull << (x_max+1)) - 1);
    mask_l_ = (1ull << x_max) - 1;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    thrust::complex<data_t> q0p;
    thrust::complex<data_t> q1p;
    double d0,d1,ret = 0.0;
    uint_t idx0,idx1;

    vec = this->data_;

    idx0 = ((i << 1) & mask_u_) | (i & mask_l_);
    idx1 = idx0 ^ x_mask_;

    q0 = vec[idx0];
    q1 = vec[idx1];
    q0p = q1 * phase_;
    q1p = q0 * phase_;
    d0 = q0.real()*q0p.real() + q0.imag()*q0p.imag();
    d1 = q1.real()*q1p.real() + q1.imag()*q1p.imag();

    if(z_mask_ != 0){
      if(pop_count_kernel(idx0 & z_mask_) & 1)
        ret = -d0;
      else
        ret = d0;
      if(pop_count_kernel(idx1 & z_mask_) & 1)
        ret -= d1;
      else
        ret += d1;
    }
    else{
      ret = d0 + d1;
    }

    return ret;
  }
  const char* name(void)
  {
    return "expval_pauli_XYZ";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::expval_pauli(const reg_t &qubits,
                                               const std::string &pauli,const complex_t initial_phase) const 
{
  uint_t x_mask, z_mask, num_y, x_max;
  std::tie(x_mask, z_mask, num_y, x_max) = pauli_masks_and_phase(qubits, pauli);

  // Special case for only I Paulis
  if (x_mask + z_mask == 0) {
    return norm();
  }
  
  // specialize x_max == 0
  if(x_mask == 0) {
    return apply_function_sum( expval_pauli_Z_func<data_t>(z_mask) );
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  auto phase = std::complex<data_t>(initial_phase);
  add_y_phase(num_y, phase);
  return apply_function_sum( expval_pauli_XYZ_func<data_t>(x_mask, z_mask, x_max, phase) );
}

template <typename data_t>
class expval_pauli_inter_chunk_func : public GateFuncBase<data_t>
{
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  thrust::complex<data_t> phase_;
  thrust::complex<data_t>* pair_chunk_;
  uint_t z_count_;
  uint_t z_count_pair_;
public:
  expval_pauli_inter_chunk_func(uint_t x,uint_t z,std::complex<data_t> p,thrust::complex<data_t>* pair_chunk,uint_t zc,uint_t zcp)
  {
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;

    pair_chunk_ = pair_chunk;
    z_count_ = zc;
    z_count_pair_ = zcp;
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    thrust::complex<data_t> q0p;
    thrust::complex<data_t> q1p;
    double d0,d1,ret = 0.0;
    uint_t ip;

    vec = this->data_;

    ip = i ^ x_mask_;
    q0 = vec[i];
    q1 = pair_chunk_[ip];
    q0p = q1 * phase_;
    q1p = q0 * phase_;
    d0 = q0.real()*q0p.real() + q0.imag()*q0p.imag();
    d1 = q1.real()*q1p.real() + q1.imag()*q1p.imag();

    if((pop_count_kernel(i & z_mask_) + z_count_) & 1)
      ret = -d0;
    else
      ret = d0;
    if((pop_count_kernel(ip & z_mask_) + z_count_pair_) & 1)
      ret -= d1;
    else
      ret += d1;

    return ret;
  }
  const char* name(void)
  {
    return "expval_pauli_inter_chunk";
  }
};

template <typename data_t>
double QubitVectorThrust<data_t>::expval_pauli(const reg_t &qubits,
                                               const std::string &pauli,
                                               const QubitVectorThrust<data_t>& pair_chunk,
                                               const uint_t z_count,const uint_t z_count_pair,const complex_t initial_phase) const 
{
  uint_t x_mask, z_mask, num_y, x_max;
  std::tie(x_mask, z_mask, num_y, x_max) = pauli_masks_and_phase(qubits, pauli);

  //get pointer to pairing chunk (copy if needed)
  double ret;
  thrust::complex<data_t>* pair_ptr;
  std::shared_ptr<Chunk<data_t>> buffer = nullptr;

  if(pair_chunk.data() == this->data()){
#ifdef AER_DISABLE_GDR
    if(chunk_->device() >= 0){    //if there is no GPUDirectRDMA support, copy chunk from CPU
      buffer = chunk_manager_.MapBufferChunk(chunk_->place());
      buffer->CopyIn(recv_chunk_);
      pair_ptr = buffer->pointer();
    }
    else{
      pair_ptr = recv_chunk_->pointer();
    }
#else
    pair_ptr = recv_chunk_->pointer();
#endif
  }
  else{   //on other memory space, copy required
    if(chunk_->device() >= 0){
      if(chunk_->container()->peer_access(pair_chunk.chunk_->device())){
        pair_ptr = pair_chunk.chunk_->pointer();
      }
      else{
        do{
          buffer = chunk_manager_.MapBufferChunk(chunk_->place());
        }while(!buffer);
        buffer->CopyIn(pair_chunk.chunk_);
        pair_ptr = buffer->pointer();
      }
    }
    else{
      if(pair_chunk.chunk_->device() >= 0){
        do{
          buffer = chunk_manager_.MapBufferChunk(chunk_->place());
        }while(!buffer);
        buffer->CopyIn(chunk_);
        pair_ptr = buffer->pointer();
      }
      else{
        pair_ptr = pair_chunk.chunk_->pointer();
      }
    }
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  auto phase = std::complex<data_t>(initial_phase);
  add_y_phase(num_y, phase);

  ret = apply_function_sum( expval_pauli_inter_chunk_func<data_t>(x_mask, z_mask, phase, pair_ptr,z_count,z_count_pair) );

  if(buffer){
    chunk_manager_.UnmapBufferChunk(buffer);
  }

  if(pair_chunk.data() == this->data()){
    release_recv_buffer();
  }

  return ret;
}

/*******************************************************************************
 *
 * PAULI
 *
 ******************************************************************************/

template <typename data_t>
class multi_pauli_func : public GateFuncBase<data_t>
{
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  uint_t mask_l_;
  uint_t mask_u_;
  thrust::complex<data_t> phase_;
  uint_t nqubits_;
public:
  multi_pauli_func(uint_t x,uint_t z,uint_t x_max,std::complex<data_t> p)
  {
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;

    mask_u_ = ~((1ull << (x_max+1)) - 1);
    mask_l_ = (1ull << x_max) - 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    uint_t idx0,idx1;

    vec = this->data_;

    idx0 = ((i << 1) & mask_u_) | (i & mask_l_);
    idx1 = idx0 ^ x_mask_;

    q0 = vec[idx0];
    q1 = vec[idx1];

    if(z_mask_ != 0){
      if(pop_count_kernel(idx0 & z_mask_) & 1)
        q0 *= -1;

      if(pop_count_kernel(idx1 & z_mask_) & 1)
        q1 *= -1;
    }
    vec[idx0] = q1 * phase_;
    vec[idx1] = q0 * phase_;
  }
  const char* name(void)
  {
    return "multi_pauli";
  }
};

//special case Z only
template <typename data_t>
class multi_pauli_Z_func : public GateFuncBase<data_t>
{
protected:
  uint_t z_mask_;
  thrust::complex<data_t> phase_;
public:
  multi_pauli_Z_func(uint_t z,std::complex<data_t> p)
  {
    z_mask_ = z;
    phase_ = p;
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;

    vec = this->data_;

    q0 = vec[i];

    if(z_mask_ != 0){
      if(pop_count_kernel(i & z_mask_) & 1)
        q0 = -q0;
    }
    vec[i] = q0 * phase_;
  }
  const char* name(void)
  {
    return "multi_pauli_Z";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_pauli(const reg_t &qubits,
                                            const std::string &pauli,
                                            const complex_t &coeff)
{
  uint_t x_mask, z_mask, num_y, x_max;
  std::tie(x_mask, z_mask, num_y, x_max) = pauli_masks_and_phase(qubits, pauli);

  // Special case for only I Paulis
  if (x_mask + z_mask == 0) {
    return;
  }
  auto phase = std::complex<data_t>(coeff);
  add_y_phase(num_y, phase);

  if(x_mask == 0){
    apply_function(multi_pauli_Z_func<data_t>(z_mask, phase));
  }
  else{
    apply_function(multi_pauli_func<data_t>(x_mask, z_mask, x_max, phase) );
  }
}



#ifdef AER_DEBUG

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const reg_t &qubits) const
{
  std::string qstr;
  int iq;
  for(iq=0;iq<qubits.size();iq++){
    qstr += std::to_string(qubits[iq]);
    qstr += ' ';
  }

  spdlog::debug(" [{}] : {} {}, {}",debug_count++,str,qubits.size(),qstr);
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const int qubit) const
{
  spdlog::debug(" [{}] : {} {} ",debug_count++,str,qubit);
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str) const
{
  spdlog::debug(" [{}] : {} ",debug_count++,str);
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const std::complex<double> c) const
{
  spdlog::debug(" [{0}] {1} : {2:e}, {3:e} ",debug_count++,str,std::real(c),imag(c));
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const double d) const
{
  spdlog::debug(" [{0}] {1} : {2:e}",debug_count++,str,d);
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const std::vector<double>& v) const
{
  std::string vstr;
  int i,n;
  n = v.size();
  for(i=0;i<n;i++){
    vstr += std::to_string(v[i]);
    vstr += ' ';
  }

  spdlog::debug(" [{}] {} : <{}>",debug_count++,str,vstr);
}


template <typename data_t>
void QubitVectorThrust<data_t>::DebugDump(void) const
{
  thrust::complex<data_t> t;
  uint_t i,idx,n;

  chunk_->synchronize();

  n = 16;
  if(n > data_size_)
    n = data_size_;
  for(i=0;i<n;i++){
    idx = i*data_size_/n;
    t = chunk_->Get(idx);
    spdlog::debug("   {0:05b} | {1:e}, {2:e}",idx,t.real(),t.imag());
  }
  if(n < data_size_){
    idx = data_size_-1;
    t = chunk_->Get(idx);
    spdlog::debug("   {0:05b} | {1:e}, {2:e}",idx,t.real(),t.imag());
  }
}


#endif

//------------------------------------------------------------------------------
} // end namespace QV
} // namespace AER
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const AER::QV::QubitVectorThrust<data_t>&qv) {

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
