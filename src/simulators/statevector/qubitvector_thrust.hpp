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

#include "framework/operations.hpp"

#include "simulators/statevector/chunk/chunk_manager.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif


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

  //copy constructor for std::vector
  QubitVectorThrust(const QubitVectorThrust<data_t>& qv)
  {
    
  }

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
  std::complex<data_t>* data() const {return (std::complex<data_t>*)chunk_.pointer();}

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Return the string name of the QubitVector class
#ifdef AER_THRUST_CUDA
  static std::string name() {return "statevector_gpu";}
#else
  static std::string name() {return "statevector_thrust";}
#endif
  virtual bool is_density_matrix(void) {return false;}

  // Set the size of the vector in terms of qubit number
  virtual void set_num_qubits(size_t num_qubits);

  // Returns the number of qubits for the current vector
  virtual uint_t num_qubits() const {return num_qubits_;}

  // Returns the size of the underlying n-qubit vector
  uint_t size() const {return data_size_;}

  // Returns required memory
  size_t required_memory_mb(uint_t num_qubits) const;

  //check if this register is on the top of array on device
  bool top_of_group()
  {
    return (chunk_.pos() == 0);
  }

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
  bool chunk_setup(int chunk_bits,int num_qubits,uint_t chunk_index,uint_t num_local_chunks);
  bool chunk_setup(QubitVectorThrust<data_t>& base,const uint_t chunk_index);

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

  void set_max_matrix_bits(int_t bits);

  void synchronize(void)
  {
    chunk_.synchronize();
  }

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

  // Initialize classical memory and register to default value (all-0)
  virtual void initialize_creg(uint_t num_memory, uint_t num_register);

  // Initialize classical memory and register to specific values
  virtual void initialize_creg(uint_t num_memory,
                       uint_t num_register,
                       const std::string &memory_hex,
                       const std::string &register_hex);

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

  //apply multiple swap gates
  // qubits is a list of pair of swaps
  void apply_multi_swaps(const reg_t &qubits);

  //apply rotation around axis
  void apply_rotation(const reg_t &qubits, const Rotation r, const double theta);

  //swap between chunk
  void apply_chunk_swap(const reg_t &qubits, QubitVectorThrust<data_t> &chunk, bool write_back = true);
  void apply_chunk_swap(const reg_t &qubits, uint_t remote_chunk_index);

  void apply_chunk_swap(QubitVectorThrust<data_t> &chunk, uint_t dest_offset, uint_t src_offset, uint_t size);

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
  // for batched optimization
  //-----------------------------------------------------------------------
  virtual bool batched_optimization_supported(void)
  {
#ifdef AER_THRUST_CUDA
    if(multi_shots_ && enable_batch_)
      return true;
    else
      return false;
#else
    return false;
#endif
  }

  bool enable_batch(bool flg) const;

  virtual void apply_bfunc(const Operations::Op &op);
  virtual void set_conditional(int_t reg);

  virtual void apply_roerror(const Operations::Op &op, std::vector<RngEngine> &rng);

  //optimized measure (async)
  virtual void apply_batched_measure(const reg_t& qubits,std::vector<RngEngine>& rng,const reg_t& cmemory,const reg_t& cregs);
  virtual void apply_batched_reset(const reg_t& qubits,std::vector<RngEngine>& rng);

  //return measured cbit (for asynchronous measure)
  virtual int measured_cregister(uint_t qubit);
  virtual int measured_cmemory(uint_t qubit);

  virtual int_t set_batched_system_conditional(int_t src_reg, reg_t& mask);

  virtual void store_cregister(uint_t qubit,int val);
  virtual void copy_cregister(uint_t dest,uint_t src);
  virtual void store_cmemory(uint_t qubit,int val);

  //copy classical register stored on qreg 
  void read_measured_data(ClassicalRegister& creg);

  //apply Pauli ops to multiple-shots (apply sampled Pauli noises)
  virtual void apply_batched_pauli_ops(const std::vector<std::vector<Operations::Op>> &ops);

  //Apply Kraus to multiple-shots
  virtual void apply_batched_kraus(const reg_t &qubits,
                   const std::vector<cmatrix_t> &kmats,
                   std::vector<RngEngine>& rng);

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

  //set number of threads per group(GPU)
  void set_num_threads_per_group(int n)
  {
    num_threads_per_group_ = n;
  }

  //cuStateVec
  void cuStateVec_enable(bool flg)
  {
    cuStateVec_enable_ = flg;
  }

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

  mutable Chunk::Chunk<data_t> chunk_;
  mutable Chunk::Chunk<data_t> buffer_chunk_;
  mutable Chunk::Chunk<data_t> send_chunk_;
  mutable Chunk::Chunk<data_t> recv_chunk_;
  std::shared_ptr<Chunk::ChunkManager<data_t>> chunk_manager_ = nullptr;

  mutable thrust::host_vector<thrust::complex<data_t>> checkpoint_;

  uint_t chunk_index_;
  bool multi_chunk_distribution_;
  bool multi_shots_;
  mutable bool enable_batch_;
  bool cuStateVec_enable_ = false;

  bool register_blocking_;

  uint_t num_creg_bits_;
  uint_t num_cmem_bits_;

  int_t max_matrix_bits_ = 0;

  //-----------------------------------------------------------------------
  // Config settings
  //----------------------------------------------------------------------- 
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 14;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 1; // Sample measure indexing qubit size
  double json_chop_threshold_ = 0;  // Threshold for choping small values
                                    // in JSON serialization
  int num_threads_per_group_ = 1;   //number of threads per GPU for multi-chunks/shots

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
  void apply_function(Function func, uint_t count = 0) const;

  template <typename Function>
  void apply_function(Function func, const std::vector<std::complex<double>>& mat, const std::vector<uint_t>& prm, uint_t count = 0) const;

  template <typename Function>
  void apply_function_sum(double* pSum,Function func,bool async=false) const;

  template <typename Function>
  void apply_function_sum2(double* pSum,Function func,bool async=false) const;

  //get number of chunk to be applied
  uint_t get_chunk_count(void);
  
#ifdef AER_DEBUG
  //for debugging
  mutable uint_t debug_count;

  void DebugMsg(const char* str,const reg_t &qubits) const;
  void DebugMsg(const char* str,const int qubit) const;
  void DebugMsg(const char* str) const;
  void DebugMsg(const char* str,const std::complex<double> c) const;
  void DebugMsg(const char* str,const double d) const;
  void DebugMsg(const char* str,const std::vector<double>& v) const;
  virtual void DebugDump(void) const;
#endif
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
    t = chunk_.Get(i);
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
  chunk_index_ = 0;
  multi_chunk_distribution_ = false;
  multi_shots_ = false;
  enable_batch_ = false;

  max_matrix_bits_ = 0;

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
  if(chunk_manager_){
    if(chunk_.is_mapped()){
      chunk_.unmap();
      chunk_manager_->UnmapChunk(chunk_);
    }
    chunk_manager_.reset();
  }
  checkpoint_.clear();
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

  return chunk_[element];
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

  return chunk_[element];
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_state(uint_t pos, std::complex<data_t>& c)
{
  if(pos < data_size_){
    thrust::complex<data_t> t = c;
    chunk_.Set(pos,t);
  }
}

template <typename data_t>
std::complex<data_t> QubitVectorThrust<data_t>::get_state(uint_t pos) const
{
  std::complex<data_t> ret = 0.0;

  if(pos < data_size_){
    ret = chunk_.Get(pos);
  }
  return ret;
}


template <typename data_t>
cvector_t<data_t> QubitVectorThrust<data_t>::vector() const 
{
  cvector_t<data_t> ret(data_size_, 0.);

  chunk_.CopyOut((thrust::complex<data_t>*)&ret[0], data_size_);

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
  chunk_.CopyOut((thrust::complex<data_t>*)&ret[0], data_size_);

#ifdef AER_DEBUG
  DebugMsg("copy_to_vector");
#endif

  return AER::Vector<std::complex<data_t>>::copy_from_buffer(data_size_, &ret[0]);
}

template <typename data_t>
AER::Vector<std::complex<data_t>> QubitVectorThrust<data_t>::move_to_vector() 
{
  cvector_t<data_t> ret(data_size_, 0.);
  chunk_.CopyOut((thrust::complex<data_t>*)&ret[0], data_size_);

#ifdef AER_DEBUG
  DebugMsg("move_to_vector", ret[0]);
  DebugDump();
#endif
  return AER::Vector<std::complex<data_t>>::copy_from_buffer(data_size_, &ret[0]);
}


//------------------------------------------------------------------------------
// State initialize component
//------------------------------------------------------------------------------
template <typename data_t>
void QubitVectorThrust<data_t>::initialize_component(const reg_t &qubits, const cvector_t<double> &state0) 
{
  if(qubits.size() == 1){
      apply_function(Chunk::initialize_component_1qubit_func<data_t>(qubits[0],state0[0],state0[1]) );
  }
  else if(qubits.size() <= chunk_.container()->matrix_bits()){
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

    auto qubits_param = qubits;
    int i;
    for(i=0;i<qubits.size();i++)
      qubits_param.push_back(qubits_sorted[i]);

//    chunk_.StoreMatrix(state0);
//    chunk_.StoreUintParams(qubits_param);

    apply_function(Chunk::initialize_component_func<data_t>(state0,qubits_sorted), state0, qubits_param );
  }
  else{
    //if initial state is larger that matrix buffer, set one by one.
    uint_t DIM = 1ull << qubits.size();
    uint_t i;
    for(i=0;i<DIM;i++){
        apply_function(Chunk::initialize_large_component_func<data_t>(state0[i],qubits,i) );
    }
  }
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------
template <typename data_t>
void QubitVectorThrust<data_t>::zero()
{
#ifdef AER_DEBUG
  DebugMsg("zero");
#endif

  apply_function(Chunk::ZeroClear<data_t>(), cvector_t<double>(), reg_t());

#ifdef AER_DEBUG
  DebugMsg("zero done");
#endif
}

template <typename data_t>
bool QubitVectorThrust<data_t>::chunk_setup(int chunk_bits,int num_qubits,uint_t chunk_index,uint_t num_local_chunks)
{
  //set global chunk ID / shot ID
  chunk_index_ = chunk_index;

  if(chunk_manager_){
    if(chunk_.is_mapped()){
      chunk_.unmap();
      chunk_manager_->UnmapChunk(chunk_);
    }

    if(chunk_manager_->chunk_bits() == chunk_bits && chunk_manager_->num_qubits() == num_qubits){
      bool mapped = chunk_manager_->MapChunk(chunk_,0);
      chunk_.set_chunk_index(chunk_index_);
      return mapped;
    }
    chunk_manager_.reset();
  }

  //only first chunk call allocation function
  if(chunk_bits > 0 && num_qubits > 0){
    chunk_manager_ = std::make_shared<Chunk::ChunkManager<data_t>>();
    chunk_manager_->set_num_threads_per_group(num_threads_per_group_);
    chunk_manager_->Allocate(chunk_bits,num_qubits,num_local_chunks,chunk_index_,max_matrix_bits_, is_density_matrix(), cuStateVec_enable_);
  }

  multi_chunk_distribution_ = false;
  if(chunk_bits < num_qubits){
    multi_chunk_distribution_ = true;
  }

  chunk_.unmap();
  buffer_chunk_.unmap();
  send_chunk_.unmap();
  recv_chunk_.unmap();

  //mapping/setting chunk
  bool mapped = chunk_manager_->MapChunk(chunk_,0);
  chunk_.set_chunk_index(chunk_index_);

  return mapped;
}

template <typename data_t>
bool QubitVectorThrust<data_t>::chunk_setup(QubitVectorThrust<data_t>& base,const uint_t chunk_index)
{
  chunk_manager_ = base.chunk_manager_;

  multi_chunk_distribution_ = base.multi_chunk_distribution_;
  if(!multi_chunk_distribution_){
    if(chunk_manager_->chunk_bits() == chunk_manager_->num_qubits()){
      multi_shots_ = true;
      base.multi_shots_ = true;
    }
  }
  cuStateVec_enable_ = base.cuStateVec_enable_;

  //set global chunk ID / shot ID
  chunk_index_ = chunk_index;

  chunk_.unmap();
  buffer_chunk_.unmap();
  send_chunk_.unmap();
  recv_chunk_.unmap();

  //mapping/setting chunk
  bool mapped = chunk_manager_->MapChunk(chunk_,0);
  chunk_.set_chunk_index(chunk_index_);

  return mapped;
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_max_matrix_bits(int_t bits)
{
  if(bits > max_matrix_bits_){
    max_matrix_bits_ = bits;
  }
}
template <typename data_t>
void QubitVectorThrust<data_t>::set_num_qubits(size_t num_qubits)
{
  num_qubits_ = num_qubits;
  data_size_ = 1ull << num_qubits;

  chunk_.set_num_qubits(num_qubits);

  register_blocking_ = false;

  //set OpenMP threads for ThrustCPU
  if(num_qubits_ > omp_threshold_ && omp_threads_ > 1)
    chunk_.container()->set_omp_threads(omp_threads_);

#ifdef AER_DEBUG
  if(chunk_.pos() == 0){
    spdlog::debug(" ==== Thrust qubit vector initialization {} qubits ====",num_qubits_);
    if(chunk_.device() >= 0)
      spdlog::debug("    TEST [id={}]: device = {}, pos = {}, place = {} / {}",chunk_index_,chunk_.device(),chunk_.pos(),chunk_.place(),chunk_manager_->num_places());
    else
      spdlog::debug("    TEST [id={}]: allocated on host (place = {})",chunk_index_,chunk_.place());
  }
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

  checkpoint_.resize(data_size_);
  chunk_.CopyOut((thrust::complex<data_t>*)thrust::raw_pointer_cast(checkpoint_.data()),data_size_);

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

  if(checkpoint_.size() == data_size_){
    chunk_.CopyIn((thrust::complex<data_t>*)thrust::raw_pointer_cast(checkpoint_.data()),data_size_);
    checkpoint_.clear();
    checkpoint_.shrink_to_fit();
  }

#ifdef AER_DEBUG
  DebugMsg("revert");
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

  if(checkpoint_.size() != data_size_){
    return std::complex<double>(0.0,0.0);
  }

  chunk_.set_device();

  vec0 = (data_t*)chunk_.pointer();
  vec1 = (data_t*)thrust::raw_pointer_cast(checkpoint_.data());
#ifdef AER_THRUST_CUDA
  cudaStream_t strm = chunk_.stream();
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
  int idev;

  if(chunk_.device() < 0){ //on host
    idev = chunk_.place() % chunk_manager_->num_places();
    do{
      chunk_manager_->MapBufferChunk(buffer_chunk_, idev);
    }while(!buffer_chunk_.is_mapped());
    chunk_.map_cache(buffer_chunk_);
    buffer_chunk_.CopyIn(chunk_);
  }
  return true;
}

template <typename data_t>
void QubitVectorThrust<data_t>::release_chunk(bool write_back) const
{
  if(chunk_.device() < 0){    //on host
    buffer_chunk_.CopyOut(chunk_);
    buffer_chunk_.synchronize();
    chunk_manager_->UnmapBufferChunk(buffer_chunk_);
    chunk_.unmap_cache();
  }
  else{
    if(chunk_.pos() == 0){
      chunk_.synchronize();    //synchronize stream before chunk exchange
    }
  }

}


template <typename data_t>
void QubitVectorThrust<data_t>::enter_register_blocking(const reg_t& qubits)
{
  register_blocking_ = true;
  chunk_.set_blocked_qubits(qubits);
}

template <typename data_t>
void QubitVectorThrust<data_t>::leave_register_blocking(void)
{
  chunk_.apply_blocked_gates();
  register_blocking_ = false;
}


template <typename data_t>
thrust::complex<data_t>* QubitVectorThrust<data_t>::send_buffer(uint_t& size_in_byte)
{
  thrust::complex<data_t>* pRet;

#ifdef AER_DISABLE_GDR
  if(chunk_.device() < 0){
    pRet = chunk_.pointer();
  }
  else{   //if there is no GPUDirectRDMA support, copy chunk on CPU before using MPI
    pRet = nullptr;
    if(chunk_manager_->MapBufferChunkOnHost(send_chunk_)){
      chunk_.CopyOut(send_chunk_);
      pRet = send_chunk_.pointer();
    }
    else{
      throw std::runtime_error("QubitVectorThrust: send buffer can not be allocated");
    }
  }
#else
  pRet = chunk_.pointer();
#endif
  chunk_.synchronize();

  size_in_byte = (uint_t)sizeof(thrust::complex<data_t>) << num_qubits_;
  return pRet;
}

template <typename data_t>
thrust::complex<data_t>* QubitVectorThrust<data_t>::recv_buffer(uint_t& size_in_byte)
{

#ifdef AER_DISABLE_GDR
  if(chunk_.device() < 0){
    chunk_manager_->MapBufferChunk(recv_chunk_,chunk_.place());
  }
  else{   //if there is no GPUDirectRDMA support, receive in CPU memory
    chunk_manager_->MapBufferChunkOnHost(recv_chunk_);
  }
#else
  if(!recv_chunk_.is_mapped())
    chunk_manager_->MapBufferChunk(recv_chunk_,chunk_.place());
#endif
  if(!recv_chunk_.is_mapped()){
    throw std::runtime_error("QubitVectorThrust: receive buffer can not be allocated");
  }
  chunk_.synchronize();

  size_in_byte = (uint_t)sizeof(thrust::complex<data_t>) << num_qubits_;
  return recv_chunk_.pointer();
}

template <typename data_t>
void QubitVectorThrust<data_t>::release_send_buffer(void) const
{
#ifdef AER_DISABLE_GDR
  if(send_chunk_.is_mapped()){
    chunk_manager_->UnmapBufferChunk(send_chunk_);
  }
#else
  chunk_.synchronize();
#endif
}

template <typename data_t>
void QubitVectorThrust<data_t>::release_recv_buffer(void) const
{
  if(recv_chunk_.is_mapped()){
    chunk_manager_->UnmapBufferChunk(recv_chunk_);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_conditional(int_t reg)
{
  chunk_.set_conditional(reg);
}

template <typename data_t>
bool QubitVectorThrust<data_t>::enable_batch(bool flg) const
{
  bool prev = enable_batch_;

//  if(flg != prev){
//    chunk_.synchronize();
//  }
  enable_batch_ = flg;

  return prev;
}

template <typename data_t>
uint_t QubitVectorThrust<data_t>::get_chunk_count(void)
{
  if(multi_chunk_distribution_){
    if(chunk_.device() < 0 || cuStateVec_enable_)
      return 1;
    else if(chunk_.pos() != 0)
      return 0;   //first chunk execute all in batch
  }
  else{   //multi-shots
    if(enable_batch_ && chunk_.pos() != 0)
      return 0;   //first chunk execute all in batch
  }
  return chunk_.container()->num_chunks();
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------
template <typename data_t>
void QubitVectorThrust<data_t>::initialize()
{
  thrust::complex<data_t> t;
  t = 1.0;

  if(multi_chunk_distribution_){
    if(chunk_index_ == 0){
      apply_function(Chunk::initialize_kernel<data_t>(t,chunk_manager_->chunk_bits(),(1ull << chunk_manager_->num_qubits())));
    }
    else{
      zero();
    }
    chunk_.synchronize();
  }
  else{
    apply_function(Chunk::initialize_kernel<data_t>(t,chunk_manager_->chunk_bits(),(1ull << chunk_manager_->chunk_bits())));
  }

#ifdef AER_DEBUG
  if(chunk_.pos() == 0){
    DebugMsg("initialize done");
    DebugDump();
  }
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

  // Convert vector data type to complex<data_t>
  AER::Vector<std::complex<data_t>> tmp(data_size_, false);
  int_t i;
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for(i=0; i < data_size_; i++){
    tmp[i] = statevec[i];
  }
  initialize_from_data(&tmp[0], tmp.size());
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
void QubitVectorThrust<data_t>::initialize_from_data(const std::complex<data_t>* statevec, const size_t num_states) 
{
  if (data_size_ != num_states) {
    std::string error = "QubitVectorThrust::initialize input vector is incorrect length (" +
                        std::to_string(data_size_) + "!=" + std::to_string(num_states) + ")";
    throw std::runtime_error(error);
  }

#ifdef AER_DEBUG
  DebugMsg("calling initialize_from_data");
#endif

  chunk_.CopyIn((thrust::complex<data_t>*)(statevec), data_size_);

#ifdef AER_DEBUG
  DebugMsg("initialize_from_data");
  DebugDump();
#endif

}

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_creg(uint_t num_memory, uint_t num_register)
{
  if(chunk_manager_){
    num_creg_bits_ = num_register;
    num_cmem_bits_ = num_memory;
    if(chunk_.pos() == 0){
      chunk_.container()->allocate_creg(num_cmem_bits_,num_creg_bits_);
    }
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_creg(uint_t num_memory,
                       uint_t num_register,
                       const std::string &memory_hex,
                       const std::string &register_hex)
{
  if(chunk_manager_){
    num_creg_bits_ = num_register;
    num_cmem_bits_ = num_memory;
    if(chunk_.pos() == 0){
      chunk_.container()->allocate_creg(num_cmem_bits_,num_creg_bits_);

      int_t i;
      for(i=0;i<num_register;i++){
        if(register_hex[register_hex.size() - 1 - i] == '0'){
          store_cregister(i,0);
        }
        else{
          store_cregister(i,1);
        }
      }
      for(i=0;i<num_memory;i++){
        if(memory_hex[memory_hex.size() - 1 - i] == '0'){
          store_cregister(i+num_creg_bits_,0);
        }
        else{
          store_cregister(i+num_creg_bits_,1);
        }
      }
    }
  }
}
//--------------------------------------------------------------------------------------
//  gate kernel execution
//--------------------------------------------------------------------------------------

template <typename data_t>
template <typename Function>
void QubitVectorThrust<data_t>::apply_function(Function func, uint_t count) const
{
  uint_t chunk_count = count;
  if(chunk_count == 0){
    if(!cuStateVec_enable_ && func.batch_enable() && ((multi_chunk_distribution_ && chunk_.device() >= 0) || enable_batch_)){
      if(chunk_.pos() == 0)        //only first chunk on device calculates all the chunks
        chunk_count = chunk_.container()->num_chunks();
      else
        return;
    }
    else
      chunk_count = 1;
  }

  //set global state index
  func.set_base_index(chunk_index_ << num_qubits_);

  chunk_.Execute(func, chunk_count);

#ifdef AER_DEBUG
  DebugMsg(func.name(), chunk_count);
  DebugDump();
#endif
}

template <typename data_t>
template <typename Function>
void QubitVectorThrust<data_t>::apply_function(Function func, const std::vector<std::complex<double>>& mat, const std::vector<uint_t>& prm, uint_t count) const
{
  uint_t chunk_count = count;
  if(chunk_count == 0){
    if(!cuStateVec_enable_ && func.batch_enable() && ((multi_chunk_distribution_ && chunk_.device() >= 0) || enable_batch_)){
      if(chunk_.pos() == 0)        //only first chunk on device calculates all the chunks
        chunk_count = chunk_.container()->num_chunks();
      else
        return;
    }
    else
      chunk_count = 1;
  }

  if(mat.size() > 0)
    chunk_.StoreMatrix(mat);
  if(prm.size() > 0)
    chunk_.StoreUintParams(prm);
  chunk_.Execute(func, chunk_count);

#ifdef AER_DEBUG
  DebugMsg(func.name(), chunk_count);
  DebugDump();
#endif
}

template <typename data_t>
template <typename Function>
void QubitVectorThrust<data_t>::apply_function_sum(double* pSum,Function func,bool async) const
{
  uint_t count = 1;
#ifdef AER_THRUST_CUDA
  if(!cuStateVec_enable_ && func.batch_enable() && ((multi_chunk_distribution_ && chunk_.device() >= 0 && num_qubits_ == num_qubits()) || (enable_batch_))){
    if(chunk_.pos() != 0){
      //only first chunk on device calculates all the chunks
      if(pSum)
        *pSum = 0.0;
      return;
    }
    count = chunk_.container()->num_chunks();
  }
#endif

  func.set_base_index(chunk_index_ << num_qubits_);
  chunk_.ExecuteSum(pSum,func,count);
#ifdef AER_DEBUG
  DebugMsg(func.name(),(int)count);
#endif

  if(!async)
    chunk_.synchronize();
}

template <typename data_t>
template <typename Function>
void QubitVectorThrust<data_t>::apply_function_sum2(double* pSum,Function func,bool async) const
{
  uint_t count = 1;
#ifdef AER_THRUST_CUDA
  if(!cuStateVec_enable_ && func.batch_enable() && ((multi_chunk_distribution_ && chunk_.device() >= 0 && num_qubits_ == num_qubits()) || (enable_batch_))){
    if(chunk_.pos() != 0){
      //only first chunk on device calculates all the chunks
      if(pSum){
        pSum[0] = 0.0;
        pSum[1] = 0.0;
      }
      return;
    }
    count = chunk_.container()->num_chunks();
  }
#endif

  func.set_base_index(chunk_index_ << num_qubits_);
  chunk_.ExecuteSum2(pSum,func,count);
#ifdef AER_DEBUG
  DebugMsg(func.name(),(int)count);
#endif

  if(!async)
    chunk_.synchronize();
}

/*******************************************************************************
 *
 * CONFIG SETTINGS
 *
 ******************************************************************************/

template <typename data_t>
void QubitVectorThrust<data_t>::set_omp_threads(int n) 
{
  if (n > 0)
    omp_threads_ = n;

#ifdef _OPENMP
  //disable nested parallel for ThrustCPU
  if(omp_get_num_threads() > 1)
    omp_threads_ = 1;
#endif
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
void QubitVectorThrust<data_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t<double> &mat)
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  if(qubits.size() == 1 && register_blocking_)
    chunk_.queue_blocked_gate('u',qubits[0],0,&mat[0]);
  else
    chunk_.apply_matrix(qubits,0,mat,count);
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
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t<double> &diag)
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  const int_t N = qubits.size();
  if(N == 1 && register_blocking_)
    chunk_.queue_blocked_gate('d',qubits[0],0,&diag[0]);
  else
    chunk_.apply_diagonal_matrix(qubits,0,diag,count);
}


template <typename data_t>
void QubitVectorThrust<data_t>::apply_permutation_matrix(const reg_t& qubits,
             const std::vector<std::pair<uint_t, uint_t>> &pairs)
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  chunk_.apply_permutation(qubits,pairs,count);
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
void QubitVectorThrust<data_t>::apply_mcx(const reg_t &qubits) 
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  if(register_blocking_){
    int i;
    uint_t mask = 0;
    for(i=0;i<qubits.size()-1;i++){
      mask |= (1ull << qubits[i]);
    }
    chunk_.queue_blocked_gate('x',qubits[qubits.size()-1],mask);
  }
  else{
    chunk_.apply_X(qubits, count);
  }
}


template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcy(const reg_t &qubits) 
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  if(register_blocking_){
    int i;
    uint_t mask = 0;
    for(i=0;i<qubits.size()-1;i++){
      mask |= (1ull << qubits[i]);
    }
    chunk_.queue_blocked_gate('y',qubits[qubits.size()-1],mask);
  }
  else{
    chunk_.apply_Y(qubits, count);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcswap(const reg_t &qubits)
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  chunk_.apply_swap(qubits,qubits.size()-2,count);
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_multi_swaps(const reg_t &qubits)
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  chunk_.apply_multi_swaps(qubits,chunk_.container()->num_chunks());
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_chunk_swap(const reg_t &qubits, QubitVectorThrust<data_t> &src, bool write_back)
{
  int q0,q1,t;

  q0 = qubits[0];
  q1 = qubits[1];

  if(q0 > q1){
    t = q0;
    q0 = q1;
    q1 = t;
  }

  if(q0 >= num_qubits_){  //exchange whole of chunk each other
    if(chunk_.device() >= 0 || src.chunk_.device() < 0)
      chunk_.Swap(src.chunk_);
    else
      src.chunk_.Swap(chunk_);
  }
  else{
    thrust::complex<data_t>* pChunk0;
    thrust::complex<data_t>* pChunk1;
    Chunk::Chunk<data_t> bufferChunk;

    if(chunk_.device() >= 0){
      if(chunk_.container()->peer_access(src.chunk_.device())){
        pChunk1 = src.chunk_.pointer();
      }
      else{
        do{
          chunk_manager_->MapBufferChunk(bufferChunk,chunk_.place());
        }while(!bufferChunk.is_mapped());
        bufferChunk.set_chunk_index(src.chunk_index_);
        bufferChunk.CopyIn(src.chunk_);
        pChunk1 = bufferChunk.pointer();
      }
      pChunk0 = chunk_.pointer();
    }
    else{
      if(src.chunk_.device() >= 0){
        do{
          chunk_manager_->MapBufferChunk(bufferChunk,src.chunk_.place());
        }while(!bufferChunk.is_mapped());
        bufferChunk.set_chunk_index(chunk_index_);
        bufferChunk.CopyIn(chunk_);
        pChunk0 = bufferChunk.pointer();
        pChunk1 = src.chunk_.pointer();
      }
      else{
        pChunk1 = src.chunk_.pointer();
        pChunk0 = chunk_.pointer();
      }
    }

#ifdef AER_DEBUG
    DebugMsg("chunk swap",qubits);
#endif

    if(q0 < num_qubits_){
      if(chunk_index_ < src.chunk_index_)
        pChunk0 += (1ull << q0);
      else
        pChunk1 += (1ull << q0);
    }

    if(bufferChunk.is_mapped()){
      bufferChunk.Execute(Chunk::CSwapChunk_func<data_t>(qubits,num_qubits_,pChunk0,pChunk1,true), 1);
      if(pChunk1 == bufferChunk.pointer())
        bufferChunk.CopyOut(src.chunk_);
      else
        bufferChunk.CopyOut(chunk_);
      bufferChunk.synchronize();
      chunk_manager_->UnmapBufferChunk(bufferChunk);
    }
    else{
      chunk_.Execute(Chunk::CSwapChunk_func<data_t>(qubits,num_qubits_,pChunk0,pChunk1,true), 1);
      chunk_.synchronize();
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
    chunk_.CopyIn(recv_chunk_);
  }
  else{
    thrust::complex<data_t>* pLocal;
    thrust::complex<data_t>* pRemote;
    Chunk::Chunk<data_t> buffer;

#ifdef AER_DISABLE_GDR
    if(chunk_.device() >= 0){    //if there is no GPUDirectRDMA support, copy chunk from CPU
      chunk_manager_->MapBufferChunk(buffer,chunk_.place());
      buffer.CopyIn(recv_chunk_);
      pRemote = buffer.pointer();
    }
    else{
      pRemote = recv_chunk_.pointer();
    }
#else
    pRemote = recv_chunk_.pointer();
#endif
    pLocal = chunk_.pointer();

    if(chunk_index_ < remote_chunk_index)
      pLocal += (1ull << q0);
    else
      pRemote += (1ull << q0);

#ifdef AER_DEBUG
    DebugMsg("chunk swap (process)",qubits);
#endif

    chunk_.Execute(Chunk::CSwapChunk_func<data_t>(qubits,num_qubits_,pLocal,pRemote,false),1);
    chunk_.synchronize();    //should be synchronized here

    if(buffer.is_mapped()){
      chunk_manager_->UnmapBufferChunk(buffer);
    }
  }

  release_recv_buffer();

#ifdef AER_DISABLE_GDR
  release_send_buffer();
#endif
}

//swap part of chunks
template <typename data_t>
void QubitVectorThrust<data_t>::apply_chunk_swap(QubitVectorThrust<data_t> &src, uint_t dest_offset, uint_t src_offset, uint_t size)
{
  if(src.chunk_index_ == chunk_index_)   //save swapped 
    chunk_.Swap(src.recv_chunk_,dest_offset,src_offset,size, false);
  else
    chunk_.Swap(src.chunk_,dest_offset,src_offset,size);
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcphase(const reg_t &qubits, const std::complex<double> phase)
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  if(register_blocking_){
    int i;
    uint_t mask = 0;
    for(i=0;i<qubits.size()-1;i++){
      mask |= (1ull << qubits[i]);
    }
    chunk_.queue_blocked_gate('p',qubits[qubits.size()-1],mask,&phase);
  }
  else{
    chunk_.apply_phase(qubits,qubits.size()-1,phase,count);
  }
}


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
        chunk_.queue_blocked_gate('d',qubits[qubits.size()-1],mask,&diag[0]);
      }
      else{
        uint_t count = get_chunk_count();
        if(count == 0)
          return;

        chunk_.apply_diagonal_matrix(qubits,qubits.size()-1,diag,count);
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
        chunk_.queue_blocked_gate('u',qubits[qubits.size()-1],mask,&mat[0]);
      }
      else{
        uint_t count = get_chunk_count();
        if(count == 0)
          return;

        chunk_.apply_matrix(qubits,qubits.size()-1,mat,count);
      }
    }
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_rotation(const reg_t &qubits, const Rotation r, const double theta)
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  chunk_.apply_rotation(qubits,r,theta,count);
}

//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::apply_matrix(const uint_t qubit,
                                       const cvector_t<double>& mat)
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  // Check if matrix is diagonal and if so use optimized lambda
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const std::vector<std::complex<double>> diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix(qubit, diag);
    return;
  }
  if(register_blocking_){
    chunk_.queue_blocked_gate('u',qubit,0,&mat[0]);
  }
  else{
    reg_t qubits = {qubit};
    chunk_.apply_matrix(qubits,0,mat,count);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t<double>& diag) 
{
  uint_t count = get_chunk_count();
  if(count == 0)
    return;

  if(register_blocking_){
    chunk_.queue_blocked_gate('d',qubit,0,&diag[0]);
  }
  else{
    reg_t qubits = {qubit};
    chunk_.apply_diagonal_matrix(qubits,0,diag,count);
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
  double ret;
  uint_t count = 1;

#ifdef AER_THRUST_CUDA
  if(enable_batch_ && ((multi_chunk_distribution_ && chunk_.device() >= 0) || !multi_chunk_distribution_)){
    if(chunk_.pos() != 0)
      return 0.0;   //first chunk execute all in batch
    count = chunk_.container()->num_chunks();
  }
#endif

  ret = chunk_.norm(count);

#ifdef AER_DEBUG
  DebugMsg("norm",ret);
#endif

  return ret;
}


template <typename data_t>
double QubitVectorThrust<data_t>::norm(const reg_t &qubits, const cvector_t<double> &mat) const 
{
  const size_t N = qubits.size();

  if(N == 1){
    return norm(qubits[0], mat);
  }
  else{
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());
    for(int_t i=0;i<N;i++){
      qubits_sorted.push_back(qubits[i]);
    }

    chunk_.StoreMatrix(mat);
    chunk_.StoreUintParams(qubits_sorted);

    double ret;
    apply_function_sum(&ret,Chunk::NormMatrixMultNxN<data_t>(N));
    return ret;
  }
}


template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const reg_t &qubits, const cvector_t<double> &mat) const {

  const uint_t N = qubits.size();

  if(N == 1){
    return norm_diagonal(qubits[0], mat);
  }
  else{
    chunk_.StoreMatrix(mat);
    chunk_.StoreUintParams(qubits);

    double ret;
    apply_function_sum(&ret,Chunk::NormDiagonalMultNxN<data_t>(qubits) );
    return ret;
  }
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
double QubitVectorThrust<data_t>::norm(const uint_t qubit, const cvector_t<double> &mat) const
{
  double ret;
  apply_function_sum(&ret,Chunk::NormMatrixMult2x2<data_t>(mat,qubit));

  return ret;
}

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const
{
  double ret;
  apply_function_sum(&ret,Chunk::NormDiagonalMult2x2<data_t>(mat,qubit));

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
  ret = (std::complex<data_t>)chunk_.Get(outcome);

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
std::vector<double> QubitVectorThrust<data_t>::probabilities(const reg_t &qubits) const 
{
  const size_t N = qubits.size();
  const int_t DIM = 1 << N;
  std::vector<double> probs(DIM, 0.);

  chunk_.probabilities(probs, qubits);

#ifdef AER_DEBUG
  DebugMsg("probabilities",probs);
#endif

  return probs;
}

#define QV_RESET_TOTAL_PROB     0
#define QV_RESET_CURRENT_PROB   1
#define QV_RESET_SUM_PROB       2
#define QV_RESET_TARGET_PROB    3

template <typename data_t>
class reset_after_measure_func : public Chunk::GateFuncBase<data_t>
{
protected:
  int num_qubits_;
  double* probs_;
  uint_t prob_buf_size_;
  uint_t iter_;
public:

  reset_after_measure_func(int nq,double* probs,uint_t prob_size,uint_t iter)
  {
    num_qubits_ = nq;
    probs_ = probs;
    prob_buf_size_ = prob_size;
    iter_ = iter;
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q;
    thrust::complex<data_t>* vec;
    double scale;
    uint_t* qubits;
    int j;

    vec = this->data_;
    qubits = this->params_;

    uint_t iChunk = (i >> this->chunk_bits_);

    scale = 1.0/sqrt(probs_[iChunk + QV_RESET_CURRENT_PROB*prob_buf_size_]);
    uint_t my_bit = 0;
    for(j=0;j<num_qubits_;j++){
      my_bit += ( ((i >> qubits[j]) & 1) << j);
    }
    if(iter_ == my_bit)
      vec[i] = scale*vec[i];
    else
      vec[i] = 0.0;
  }
  const char* name(void)
  {
    return "reset_after_measure";
  }
};

template <typename data_t>
class set_probability_buffer_for_reset_func : public Chunk::GateFuncBase<data_t>
{
protected:
  uint_t reduce_buf_size_;
  uint_t prob_buf_size_;
  double* probs_;
  double* reduce_;
public:

  set_probability_buffer_for_reset_func(double* probs,uint_t prob_size,double* reduce,uint_t red_size)
  {
    probs_ = probs;
    reduce_ = reduce;
    prob_buf_size_ = prob_size;
    reduce_buf_size_ = red_size;
  }
  bool is_diagonal(void)
  {
    return true;
  }
  uint_t size(int num_qubits)
  {
    this->chunk_bits_ = 0;
    return 1;
  }
  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t iChunk = i;
    if(reduce_){
      probs_[iChunk] = reduce_[iChunk*reduce_buf_size_];
    }
    probs_[iChunk + QV_RESET_CURRENT_PROB*prob_buf_size_] = 0.0;
    probs_[iChunk + QV_RESET_SUM_PROB*prob_buf_size_] = 0.0;
  }
  const char* name(void)
  {
    return "set_probability_buffer_for_reset";
  }
};

template <typename data_t>
class check_measure_probability_func : public Chunk::GateFuncBase<data_t>
{
protected:
  int num_qubits_;
  uint_t reduce_buf_size_;
  uint_t prob_buf_size_;
  double* probs_;
  double* reduce_;
  uint_t iter_;
  uint_t num_mem_;
  uint_t num_reg_;
public:

  check_measure_probability_func(int nq,double* probs,uint_t prob_size,double* reduce,uint_t red_size,uint_t iter,uint_t nm,uint_t nr)
  {
    num_qubits_ = nq;
    probs_ = probs;
    reduce_ = reduce;
    prob_buf_size_ = prob_size;
    reduce_buf_size_ = red_size;
    iter_ = iter;
    num_mem_ = nm;
    num_reg_ = nr;
  }
  bool is_diagonal(void)
  {
    return true;
  }
  uint_t size(int num_qubits)
  {
    this->chunk_bits_ = 0;
    return 1;
  }
  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t iChunk = i;
    double p0,p1,p_total,rnd,p;
    uint_t bit,j;
    uint_t* qubits;
    uint_t* mem_bits;
    uint_t* reg_bits;
    bool reset = false;

    p_total = probs_[iChunk];
    p0 = reduce_[iChunk*reduce_buf_size_];
    p1 = probs_[iChunk + QV_RESET_SUM_PROB*prob_buf_size_];
    probs_[iChunk + QV_RESET_SUM_PROB*prob_buf_size_] = p0 + p1;  //update sum
    rnd = probs_[iChunk + QV_RESET_TARGET_PROB*prob_buf_size_];

    p = p0;
    if(rnd < ((p0 + p1) / p_total)){
      bit = iter_;
      reset = true;
    }
    else if(iter_ + 2 == (1ull << num_qubits_)){  //last one
      bit = iter_ + 1;
      p = p_total - p0 - p1;
      reset = true;
    }
    probs_[iChunk + QV_RESET_CURRENT_PROB*prob_buf_size_] = p;

    uint_t n64,i64,ibit,sreg;
    n64 = (this->num_creg_bits_ + 63) >> 6;
    sreg = this->num_creg_bits_ - QV_NUM_INTERNAL_REGS;
    if(reset){
      //save measure
      qubits = this->params_;
      mem_bits = qubits + num_qubits_;
      reg_bits = mem_bits + num_mem_;
      for(j=0;j<num_mem_;j++){
        i64 = mem_bits[j] >> 6;
        ibit = mem_bits[j] & 63;
        this->cregs_[iChunk*n64 + i64] = (this->cregs_[iChunk*n64 + i64] & (~(1ull << ibit))) | (((bit >> j) & 1) << ibit);
      }
      for(j=0;j<num_reg_;j++){
        i64 = reg_bits[j] >> 6;
        ibit = reg_bits[j] & 63;
        this->cregs_[iChunk*n64 + i64] = (this->cregs_[iChunk*n64 + i64] & (~(1ull << ibit))) | (((bit >> j) & 1) << ibit);
      }

      if(bit == iter_){  //applying only for iter != last, last case will be aplied out of loop
        //set system register[0] to 0 not to execute measure/reset kernels anymore
        i64 = sreg >> 6;
        ibit = sreg & 63;
        this->cregs_[iChunk*n64 + i64] &= (~(1ull << ibit));

        //set system register[1] to 1 to apply reset
        i64 = (sreg+1) >> 6;
        ibit = (sreg+1) & 63;
        this->cregs_[iChunk*n64 + i64] |= (1ull << ibit);
      }
      else{
        //set system register[1] to 0
        i64 = (sreg+1) >> 6;
        ibit = (sreg+1) & 63;
        this->cregs_[iChunk*n64 + i64] &= (~(1ull << ibit));
      }
    }
    else{
      //set system register[1] to 0
      i64 = (sreg+1) >> 6;
      ibit = (sreg+1) & 63;
      this->cregs_[iChunk*n64 + i64] &= (~(1ull << ibit));
    }
  }
  const char* name(void)
  {
    return "check_measure_probability";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_batched_measure(const reg_t& qubits,std::vector<RngEngine>& rng,const reg_t& cmemory,const reg_t& cregs)
{
  const int_t DIM = 1 << qubits.size();
  uint_t i,count = 1;
  if(enable_batch_){
    if(chunk_.pos() != 0){
      return;   //first chunk execute all in batch
    }
  }
  count = chunk_.container()->num_chunks();

  //set system register[0] to 1 used for conditional register
  uint_t system_reg = num_creg_bits_ + num_cmem_bits_;

  //handling conditional
  int_t reg_cond = chunk_.get_conditional();
  if(reg_cond >= 0){
    //copy conditional register to system register
    chunk_.set_conditional(-1);
    copy_cregister(system_reg,reg_cond);
    store_cregister(system_reg+1,0);
  }
  else{
    //can be applied to all states
    store_cregister(system_reg,1);
  }
  chunk_.set_conditional(system_reg);
  chunk_.keep_conditional(true);

  //total probability
  apply_function_sum(nullptr,Chunk::norm_func<data_t>(),true);
  apply_function(set_probability_buffer_for_reset_func<data_t>(chunk_.probability_buffer(),chunk_.container()->num_chunks(),
                                                               chunk_.reduce_buffer(),chunk_.reduce_buffer_size()) );

  reg_t params(qubits.size() + cmemory.size() + cregs.size());
  for(i=0;i<qubits.size();i++){
    params[i] = qubits[i];
  }
  for(i=0;i<cmemory.size();i++){
    params[i+qubits.size()] = cmemory[i] + num_creg_bits_;
  }
  for(i=0;i<cregs.size();i++){
    params[cmemory.size()+qubits.size()+i] = cregs[i];
  }
  chunk_.StoreUintParams(params);

  //probability
  std::vector<double> r(count);
  for(i=0;i<count;i++){
    r[i] = rng[i].rand();
  }
  chunk_.container()->copy_to_probability_buffer(r,QV_RESET_TARGET_PROB);

  //loop for probability
  for(i=0;i<DIM-1;i++){
    chunk_.set_conditional(system_reg);
    apply_function_sum(nullptr,Chunk::probability_func<data_t>(qubits,i),true);

    apply_function(check_measure_probability_func<data_t>(qubits.size(),chunk_.probability_buffer(),chunk_.container()->num_chunks(),
                                                                        chunk_.reduce_buffer(),chunk_.reduce_buffer_size(),
                                                                        i,cmemory.size(),cregs.size()) );

    chunk_.set_conditional(system_reg+1);
    apply_function(reset_after_measure_func<data_t>(qubits.size(),chunk_.probability_buffer(),chunk_.container()->num_chunks(),i ));
    store_cregister(system_reg+1,0);
  }
  //for last case
  chunk_.keep_conditional(false);
  chunk_.set_conditional(system_reg);
  apply_function(reset_after_measure_func<data_t>(qubits.size(),chunk_.probability_buffer(),chunk_.container()->num_chunks(),DIM-1 ));

  chunk_.container()->request_creg_update();
}

template <typename data_t>
class reset_func : public Chunk::GateFuncBase<data_t>
{
protected:
  int num_qubits_;
  double* probs_;
  uint_t iter_;
  uint_t prob_buf_size_;
public:

  reset_func(int nq,double* probs,uint_t prob_size,uint_t iter)
  {
    num_qubits_ = nq;
    probs_ = probs;
    iter_ = iter;
    prob_buf_size_ = prob_size;
  }

  bool is_diagonal(void)
  {
    return true;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t> q;
    thrust::complex<data_t>* vec;
    double scale;
    uint_t* qubits;
    int j;

    vec = this->data_;
    qubits = this->params_;

    uint_t iChunk = (i >> this->chunk_bits_);

    scale = 1.0/sqrt(probs_[iChunk + QV_RESET_CURRENT_PROB*prob_buf_size_]);

    uint_t my_bit = 0;
    for(j=0;j<num_qubits_;j++){
      my_bit += ( ((i >> qubits[j]) & 1) << j);
    }
    if(my_bit == 0){
      uint_t i_pair = i;
      for(j=0;j<num_qubits_;j++){
        if( ((iter_ >> j) & 1) != 0)
          i_pair += (1ull << qubits[j]);
      }
      vec[i] = scale*vec[i_pair];
      if(i != i_pair){
        vec[i_pair] = 0.0;
      }
    }
    else if(iter_ != my_bit){
      vec[i] = 0.0;
    }
  }
  const char* name(void)
  {
    return "reset";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_batched_reset(const reg_t& qubits,std::vector<RngEngine>& rng)
{
  const int_t DIM = 1 << qubits.size();
  uint_t i,count = 1;
  if(enable_batch_){
    if(chunk_.pos() != 0){
      return;   //first chunk execute all in batch
    }
  }
  count = chunk_.container()->num_chunks();

  //set system register[0] to 1 used for conditional register
  uint_t system_reg = num_creg_bits_ + num_cmem_bits_;

  //handling conditional
  int_t reg_cond = chunk_.get_conditional();
  if(reg_cond >= 0){
    //copy conditional register to system register
    chunk_.set_conditional(-1);
    copy_cregister(system_reg,reg_cond);
    store_cregister(system_reg+1,0);
  }
  else{
    //can be applied to all states
    store_cregister(system_reg,1);
  }
  chunk_.set_conditional(system_reg);
  chunk_.keep_conditional(true);

  //total probability
  apply_function_sum(nullptr,Chunk::norm_func<data_t>(),true);
  apply_function(set_probability_buffer_for_reset_func<data_t>(chunk_.probability_buffer(),chunk_.container()->num_chunks(),
                                                               chunk_.reduce_buffer(),chunk_.reduce_buffer_size()) );

  //probability
  std::vector<double> r(count);
  for(i=0;i<count;i++){
    r[i] = rng[i].rand();
  }
  chunk_.container()->copy_to_probability_buffer(r,QV_RESET_TARGET_PROB);

  chunk_.StoreUintParams(qubits);
  for(i=0;i<DIM-1;i++){
    chunk_.set_conditional(system_reg);
    apply_function_sum(nullptr,Chunk::probability_func<data_t>(qubits,i),true);

    apply_function(check_measure_probability_func<data_t>(qubits.size(),chunk_.probability_buffer(),chunk_.container()->num_chunks(),
                                                                        chunk_.reduce_buffer(),chunk_.reduce_buffer_size(),
                                                                        i,0,0) );

    chunk_.set_conditional(system_reg+1);
    apply_function(reset_func<data_t>(qubits.size(),chunk_.probability_buffer(),chunk_.container()->num_chunks(),i ) );
    store_cregister(system_reg+1,0);
  }
  chunk_.keep_conditional(false);
  chunk_.set_conditional(system_reg);
  apply_function(reset_func<data_t>(qubits.size(),chunk_.probability_buffer(),chunk_.container()->num_chunks(),DIM-1 ) );
}



template <typename data_t>
int QubitVectorThrust<data_t>::measured_cregister(uint_t qubit)
{
  //read from memory
  return chunk_.measured_cbit(qubit);
}

template <typename data_t>
int QubitVectorThrust<data_t>::measured_cmemory(uint_t qubit)
{
  //read from memory
  return chunk_.measured_cbit(qubit + num_creg_bits_);
}

template <typename data_t>
void QubitVectorThrust<data_t>::read_measured_data(ClassicalRegister& creg)
{
  uint_t i;
  reg_t pos(1);
  reg_t dummy_pos;

  for(i=0;i<creg.memory_size();i++){
    int bit = chunk_.measured_cbit(i + num_creg_bits_);
    if(bit >= 0){
      const reg_t outcome = Utils::int2reg(bit, 2, 1);
      pos[0] = i;
      creg.store_measure(outcome, pos , dummy_pos);
    }
  }
  for(i=0;i<creg.register_size();i++){
    int bit = chunk_.measured_cbit(i);
    if(bit >= 0){
      const reg_t outcome = Utils::int2reg(bit, 2, 1);
      pos[0] = i;
      creg.store_measure(outcome, dummy_pos, pos);
    }
  }
}

template <typename data_t>
class set_creg_func : public Chunk::GateFuncBase<data_t>
{
protected:
  uint_t reg_set_;
  uint_t val_;
public:
  set_creg_func(uint_t reg_set,uint_t val)
  {
    reg_set_ = reg_set;
    val_ = val;
  }
  bool is_diagonal(void)
  {
    return true;
  }
  uint_t size(int num_qubits)
  {
    this->chunk_bits_ = 0;
    return 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t iChunk = i;
    uint_t n64,i64,ibit;
    n64 = (this->num_creg_bits_ + 63) >> 6;

    i64 = reg_set_ >> 6;
    ibit = reg_set_ & 63;
    this->cregs_[iChunk*n64 + i64] = (this->cregs_[iChunk*n64 + i64] & (~(1ull << ibit))) | (val_ << ibit);
  }
  const char* name(void)
  {
    return "set_creg";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::store_cregister(uint_t qubit,int val)
{
  apply_function(set_creg_func<data_t>(qubit,(uint_t)val));
}

template <typename data_t>
void QubitVectorThrust<data_t>::store_cmemory(uint_t qubit,int val)
{
  apply_function(set_creg_func<data_t>(qubit + num_creg_bits_,(uint_t)val));
}

template <typename data_t>
class set_batched_creg_func : public Chunk::GateFuncBase<data_t>
{
protected:
  int_t reg_set_;
  int_t reg_copy_;
public:
  set_batched_creg_func(int_t reg_set,int_t reg_copy)
  {
    reg_set_ = reg_set;
    reg_copy_ = reg_copy;
  }
  bool is_diagonal(void)
  {
    return true;
  }
  uint_t size(int num_qubits)
  {
    this->chunk_bits_ = 0;
    return 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t iChunk = i;
    uint_t n64,i64,ibit;
    uint_t* mask;
    uint_t val = 1;
    n64 = (this->num_creg_bits_ + 63) >> 6;
    int j;

    mask = this->params_;

    if(reg_copy_ >= 0){   //mask conditional register
      i64 = reg_copy_ >> 6;
      ibit = reg_copy_ & 63;
      val = (this->cregs_[iChunk*n64 + i64] >> ibit) & 1;
    }
    val &= mask[iChunk];

    i64 = reg_set_ >> 6;
    ibit = reg_set_ & 63;
    this->cregs_[iChunk*n64 + i64] = (this->cregs_[iChunk*n64 + i64] & (~(1ull << ibit))) | (val << ibit);
  }
  const char* name(void)
  {
    return "set_batched_creg";
  }
};

template <typename data_t>
int_t QubitVectorThrust<data_t>::set_batched_system_conditional(int_t src_reg, reg_t& mask)
{
  int_t sys_reg = num_creg_bits_ + num_cmem_bits_ + QV_NUM_INTERNAL_REGS - 1; //use last internal reg

  //copy bit from src_reg to system reg and mask them
  chunk_.StoreUintParams(mask);
  apply_function(set_batched_creg_func<data_t>(sys_reg,src_reg));

  return sys_reg;
}

template <typename data_t>
class copy_creg_func : public Chunk::GateFuncBase<data_t>
{
protected:
  uint_t reg_dest_;
  uint_t reg_src_;
public:
  copy_creg_func(uint_t dest,uint_t src)
  {
    reg_dest_ = dest;
    reg_src_ = src;
  }
  bool is_diagonal(void)
  {
    return true;
  }
  uint_t size(int num_qubits)
  {
    this->chunk_bits_ = 0;
    return 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t iChunk = i;
    uint_t n64,i64,ibit,val;
    n64 = (this->num_creg_bits_ + 63) >> 6;

    i64 = reg_src_ >> 6;
    ibit = reg_src_ & 63;
    val = (this->cregs_[iChunk*n64 + i64] >> ibit) & 1;

    i64 = reg_dest_ >> 6;
    ibit = reg_dest_ & 63;
    this->cregs_[iChunk*n64 + i64] = (this->cregs_[iChunk*n64 + i64] & (~(1ull << ibit))) | (val << ibit);
  }
  const char* name(void)
  {
    return "copy_creg";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::copy_cregister(uint_t dest,uint_t src)
{
  apply_function(copy_creg_func<data_t>(dest,src));
}

//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
reg_t QubitVectorThrust<data_t>::sample_measure(const std::vector<double> &rnds) const
{
  uint_t count = 1;
#ifdef AER_THRUST_CUDA
  if((multi_chunk_distribution_ && chunk_.device() >= 0) || enable_batch_){
    if(chunk_.pos() != 0)
      return reg_t();   //first chunk execute all in batch
    count = chunk_.container()->num_chunks();
  }
#endif

#ifdef AER_DEBUG
  reg_t samples;
  DebugMsg("sample_measure begin",(int)count);
  samples = chunk_.sample_measure(rnds,1,true,count);
  DebugMsg("sample_measure",samples);
  return samples;
#else
  return chunk_.sample_measure(rnds,1,true,count);
#endif
}



/*******************************************************************************
 *
 * EXPECTATION VALUES
 *
 ******************************************************************************/

template <typename data_t>
double QubitVectorThrust<data_t>::expval_pauli(const reg_t &qubits,
                                               const std::string &pauli,const complex_t initial_phase) const 
{
  return chunk_.expval_pauli(qubits,pauli,initial_phase);

  uint_t x_mask, z_mask, num_y, x_max;
  std::tie(x_mask, z_mask, num_y, x_max) = pauli_masks_and_phase(qubits, pauli);

  // Special case for only I Paulis
  if (x_mask + z_mask == 0) {
    thrust::complex<double> ret = chunk_.norm(1);
    return ret.real() + ret.imag();
  }
  double ret;
  // specialize x_max == 0
  if(x_mask == 0) {
    apply_function_sum(&ret, Chunk::expval_pauli_Z_func<data_t>(z_mask) );
    return ret;
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  auto phase = std::complex<data_t>(initial_phase);
  add_y_phase(num_y, phase);
  apply_function_sum(&ret, Chunk::expval_pauli_XYZ_func<data_t>(x_mask, z_mask, x_max, phase) );
  return ret;
}


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
  Chunk::Chunk<data_t> buffer;

  if(pair_chunk.data() == this->data()){
#ifdef AER_DISABLE_GDR
    if(chunk_.device() >= 0){    //if there is no GPUDirectRDMA support, copy chunk from CPU
      chunk_manager_->MapBufferChunk(buffer,chunk_.place());
      buffer.CopyIn(recv_chunk_);
      pair_ptr = buffer.pointer();
    }
    else{
      pair_ptr = recv_chunk_.pointer();
    }
#else
    pair_ptr = recv_chunk_.pointer();
#endif
  }
  else{   //on other memory space, copy required
    if(chunk_.device() >= 0){
      if(chunk_.container()->peer_access(pair_chunk.chunk_.device())){
        pair_ptr = pair_chunk.chunk_.pointer();
      }
      else{
        do{
          chunk_manager_->MapBufferChunk(buffer,chunk_.place());
        }while(!buffer.is_mapped());
        buffer.CopyIn(pair_chunk.chunk_);
        pair_ptr = buffer.pointer();
      }
    }
    else{
      if(pair_chunk.chunk_.device() >= 0){
        do{
          chunk_manager_->MapBufferChunk(buffer,chunk_.place());
        }while(!buffer.is_mapped());
        buffer.CopyIn(chunk_);
        pair_ptr = buffer.pointer();
      }
      else{
        pair_ptr = pair_chunk.chunk_.pointer();
      }
    }
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  auto phase = std::complex<data_t>(initial_phase);
  add_y_phase(num_y, phase);

  apply_function_sum(&ret, Chunk::expval_pauli_inter_chunk_func<data_t>(x_mask, z_mask, phase, pair_ptr,z_count,z_count_pair) );

  if(buffer.is_mapped()){
    chunk_manager_->UnmapBufferChunk(buffer);
  }

  if(pair_chunk.data() == this->data()){
    release_recv_buffer();
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
    apply_function(Chunk::multi_pauli_Z_func<data_t>(z_mask, phase));
  }
  else{
    apply_function(Chunk::multi_pauli_func<data_t>(x_mask, z_mask, x_max, phase) );
  }
}

//batched Pauli operation used for Pauli noise
template <typename data_t>
class batched_pauli_func : public Chunk::GateFuncBase<data_t>
{
protected:
  thrust::complex<data_t> coeff_;
  int num_qubits_state_;
public:
  batched_pauli_func(int nqs,std::complex<data_t> c)
  {
    num_qubits_state_ = nqs;
    coeff_ = c;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    uint_t idx0,idx1;
    uint_t* param;
    thrust::complex<data_t> phase;

    uint_t iChunk = (i >> (num_qubits_state_ - 1));

    param = this->params_ + iChunk * 4;
    uint_t x_max = param[0];
    uint_t num_y = param[1];
    uint_t x_mask_ = param[2];
    uint_t z_mask_ = param[3];
    uint_t mask_l_;
    uint_t mask_u_;

    mask_u_ = ~((1ull << (x_max+1)) - 1);
    mask_l_ = (1ull << x_max) - 1;

    vec = this->data_;

    if(x_mask_ == 0){
      idx0 = i << 1;
      idx1 = idx0 + 1;
    }
    else{
      idx0 = ((i << 1) & mask_u_) | (i & mask_l_);
      idx1 = idx0 ^ x_mask_;
    }

    q0 = vec[idx0];
    q1 = vec[idx1];

    if(num_y == 0)
      phase = coeff_;
    else if(num_y == 1)
      phase = thrust::complex<data_t>(coeff_.imag(),-coeff_.real());
    else if(num_y == 2)
      phase = thrust::complex<data_t>(-coeff_.real(),-coeff_.imag());
    else
      phase = thrust::complex<data_t>(-coeff_.imag(),coeff_.real());

    if(z_mask_ != 0){
      if(Chunk::pop_count_kernel(idx0 & z_mask_) & 1)
        q0 *= -1;

      if(Chunk::pop_count_kernel(idx1 & z_mask_) & 1)
        q1 *= -1;
    }
    if(x_mask_ == 0){
      vec[idx0] = q0 * phase;
      vec[idx1] = q1 * phase;
    }
    else{
      vec[idx0] = q1 * phase;
      vec[idx1] = q0 * phase;
    }
  }
  const char* name(void)
  {
    return "batched_pauli";
  }
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_batched_pauli_ops(const std::vector<std::vector<Operations::Op>>& ops)
{
  if(enable_batch_ && chunk_.pos() != 0){
    return;   //first chunk execute all in batch
  }
  uint_t count = ops.size();
  int num_inner_threads = omp_get_max_threads() / num_threads_per_group_;
  int_t i;

  reg_t params(4*count);

  auto count_paulis = [this,&params,ops](int_t i){
    int_t j;
    uint_t x_max = 0;
    uint_t num_y = 0;
    uint_t x_mask = 0;
    uint_t z_mask = 0;

    for(j=0;j<ops[i].size();j++){
      if(ops[i][j].conditional)
        set_conditional(ops[i][j].conditional_reg);

      if(ops[i][j].name == "x"){
        x_mask ^= (1ull << ops[i][j].qubits[0]);
        x_max = std::max<uint_t>(x_max, (ops[i][j].qubits[0]));
      }
      else if(ops[i][j].name == "z"){
        z_mask ^= (1ull << ops[i][j].qubits[0]);
      }
      else if(ops[i][j].name == "y"){
        x_mask ^= (1ull << ops[i][j].qubits[0]);
        z_mask ^= (1ull << ops[i][j].qubits[0]);
        x_max = std::max<uint_t>(x_max, (ops[i][j].qubits[0]));
        num_y++;
      }
      else if(ops[i][j].name == "pauli"){
        uint_t pauli_x_mask = 0, pauli_z_mask = 0, pauli_num_y = 0, pauli_x_max = 0;
        std::tie(pauli_x_mask, pauli_z_mask, pauli_num_y, pauli_x_max) = pauli_masks_and_phase(ops[i][j].qubits, ops[i][j].string_params[0]);

        x_mask ^= pauli_x_mask;
        z_mask ^= pauli_z_mask;
        x_max = std::max<uint_t>(x_max, pauli_x_max);
        num_y += pauli_num_y;
      }
    }
    params[i*4] = x_max;
    params[i*4+1] = num_y % 4;
    params[i*4+2] = x_mask;
    params[i*4+3] = z_mask;
  };
  Utils::apply_omp_parallel_for((num_inner_threads > 1),0,count,count_paulis,num_inner_threads);

  thrust::complex<data_t> coeff(1.0,0.0);
  chunk_.StoreUintParams(params);
  apply_function(batched_pauli_func<data_t>(num_qubits_,coeff) );
}

template <typename data_t>
class MatrixMult2x2_conditional : public Chunk::GateFuncBase<data_t>
{
protected:
  thrust::complex<double> m0,m1,m2,m3;
  uint_t offset0;
  uint_t prob_buf_size_;
  double* probs_;
public:
  MatrixMult2x2_conditional(const cvector_t<double>& mat,int q,double* probs,uint_t prob_size)
  {
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];

    offset0 = 1ull << q;

    probs_ = probs;
    prob_buf_size_ = prob_size;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t i0,i1;
    thrust::complex<data_t> q0,q1;
    thrust::complex<data_t>* vec0;
    thrust::complex<data_t>* vec1;
    double p,p0,p1,rnd;

    uint_t iChunk = i >> this->chunk_bits_;
    double scale  = 1.0/sqrt(probs_[iChunk + QV_RESET_CURRENT_PROB*prob_buf_size_]);

    vec0 = this->data_;
    vec1 = vec0 + offset0;

    i1 = i & (offset0 - 1);
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec0[i0];
    q1 = vec1[i0];
    vec0[i0] = scale*(m0 * q0 + m2 * q1);
    vec1[i0] = scale*(m1 * q0 + m3 * q1);
  }
  const char* name(void)
  {
    return "MatrixMult2x2_conditional";
  }
};

template <typename data_t>
class MatrixMultNxN_conditional : public Chunk::GateFuncWithCache<data_t>
{
protected:
  uint_t prob_buf_size_;
  double* probs_;
public:
  MatrixMultNxN_conditional(uint_t nq,double* probs,uint_t prob_size) : Chunk::GateFuncWithCache<data_t>(nq)
  {
    probs_ = probs;
    prob_buf_size_ = prob_size;
  }

  __host__ __device__ void run_with_cache(uint_t _tid,uint_t _idx,thrust::complex<data_t>* _cache) const
  {
    uint_t j,threadID;
    thrust::complex<data_t> q,r;
    thrust::complex<double> m;
    uint_t mat_size,irow;
    thrust::complex<data_t>* vec;
    thrust::complex<double>* pMat;

    uint_t iChunk = _idx >> this->chunk_bits_;

    vec = this->data_;
    pMat = this->matrix_;

    double scale = 1.0/sqrt(probs_[iChunk + QV_RESET_CURRENT_PROB*prob_buf_size_]);

    mat_size = 1ull << this->nqubits_;
    irow = _tid & (mat_size - 1);

    r = 0.0;
    for(j=0;j<mat_size;j++){
      m = pMat[irow + mat_size*j];
      q = _cache[(_tid & 1023) - irow + j];
      r += m*q;
    }
    vec[_idx] = r*scale;
  }

  const char* name(void)
  {
    return "multNxN_conditional";
  }
};

template <typename data_t>
class check_kraus_probability_func : public Chunk::GateFuncBase<data_t>
{
protected:
  uint_t reduce_buf_size_;
  uint_t prob_buf_size_;
  double* probs_;
  double* reduce_;
public:
  check_kraus_probability_func(double* probs,uint_t prob_size,double* reduce,uint_t red_size)
  {
    probs_ = probs;
    reduce_ = reduce;
    prob_buf_size_ = prob_size;
    reduce_buf_size_ = red_size;
  }
  bool is_diagonal(void)
  {
    return true;
  }
  uint_t size(int num_qubits)
  {
    this->chunk_bits_ = 0;
    return 1;
  }
  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t iChunk = i;
    double p0,p1,rnd;
    bool mult = false;

    p0 = reduce_[iChunk*reduce_buf_size_];
    probs_[iChunk + QV_RESET_CURRENT_PROB*prob_buf_size_] = p0;
    p1 = probs_[iChunk + QV_RESET_SUM_PROB*prob_buf_size_];
    probs_[iChunk + QV_RESET_SUM_PROB*prob_buf_size_] = p0 + p1;  //update sum
    rnd = probs_[iChunk + QV_RESET_TARGET_PROB*prob_buf_size_];

    uint_t n64,i64,ibit,sreg;
    n64 = (this->num_creg_bits_ + 63) >> 6;
    sreg = this->num_creg_bits_ - QV_NUM_INTERNAL_REGS;
    if(rnd >= p1 && rnd < p0 + p1){
      //set system register[0] to 0 not to execute Kraus kernels anymore
      i64 = sreg >> 6;
      ibit = sreg & 63;
      this->cregs_[iChunk*n64 + i64] &= (~(1ull << ibit));

      //set system register[1] to 1 to multiply matrix
      i64 = (sreg+1) >> 6;
      ibit = (sreg+1) & 63;
      this->cregs_[iChunk*n64 + i64] |= (1ull << ibit);
    }
    else{
      //set system register[1] to 0
      i64 = (sreg+1) >> 6;
      ibit = (sreg+1) & 63;
      this->cregs_[iChunk*n64 + i64] &= (~(1ull << ibit));
    }
  }
  const char* name(void)
  {
    return "check_kraus_probability";
  }
};


template <typename data_t>
void QubitVectorThrust<data_t>::apply_batched_kraus(const reg_t &qubits,
                                            const std::vector<cmatrix_t> &kmats,
                                            std::vector<RngEngine>& rng)
{
  const size_t N = qubits.size();
  uint_t i,count;
  double ret;

  count = chunk_.container()->num_chunks();

  //set system register[0] to 1 used for conditional register
  uint_t system_reg = num_creg_bits_ + num_cmem_bits_;

  //handling conditional
  int_t reg_cond = chunk_.get_conditional();
  if(reg_cond >= 0){
    //copy conditional register to system register
    chunk_.set_conditional(-1);
    copy_cregister(system_reg,reg_cond);
    store_cregister(system_reg+1,0);
  }
  else{
    //can be applied to all states
    store_cregister(system_reg,1);
  }
  chunk_.set_conditional(system_reg);
  chunk_.keep_conditional(true);

  //total probability
  apply_function(set_probability_buffer_for_reset_func<data_t>(chunk_.probability_buffer(),chunk_.container()->num_chunks(),
                                                               nullptr,0) );

  std::vector<double> r(count);
  for(i=0;i<count;i++){
    r[i] = rng[i].rand(0., 1.);
  }
  chunk_.container()->copy_to_probability_buffer(r,QV_RESET_TARGET_PROB);

  if(N == 1){
    for(i=0;i<kmats.size();i++){
      cvector_t<double> vmat = Utils::vectorize_matrix(kmats[i]);

      chunk_.set_conditional(system_reg);
      apply_function_sum(nullptr,Chunk::NormMatrixMult2x2<data_t>(vmat,qubits[0]),true);

      apply_function(check_kraus_probability_func<data_t>(chunk_.probability_buffer(),chunk_.container()->num_chunks(),
                                                          chunk_.reduce_buffer(),chunk_.reduce_buffer_size() ) );

      //multiply only when system reg[1] is 1
      chunk_.set_conditional(system_reg+1);
      apply_function(MatrixMult2x2_conditional<data_t>(vmat,qubits[0],chunk_.probability_buffer(),chunk_.container()->num_chunks()) );
      store_cregister(system_reg+1,0);
    }
  }
  else{
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());
    for(i=0;i<N;i++)
      qubits_sorted.push_back(qubits[i]);
    chunk_.StoreUintParams(qubits_sorted);

    for(i=0;i<kmats.size();i++){
      chunk_.set_conditional(system_reg);

      chunk_.StoreMatrix(Utils::vectorize_matrix(kmats[i]));
      apply_function_sum(nullptr,Chunk::NormMatrixMultNxN<data_t>(N),true);

      apply_function(check_kraus_probability_func<data_t>(chunk_.probability_buffer(),chunk_.container()->num_chunks(),
                                                          chunk_.reduce_buffer(),chunk_.reduce_buffer_size() ) );

      //multiply only when system reg[1] is 1
      chunk_.set_conditional(system_reg+1);
      apply_function(MatrixMultNxN_conditional<data_t>(N,chunk_.probability_buffer(),chunk_.container()->num_chunks()) );
      store_cregister(system_reg+1,0);
    }
  }
  chunk_.set_conditional(-1);
  chunk_.keep_conditional(false);
}

template <typename data_t>
class bfunc_kernel : public Chunk::GateFuncBase<data_t>
{
protected:
  uint_t bfunc_num_regs_;
  Operations::RegComparison bfunc_;
public:
  bfunc_kernel(uint_t n,Operations::RegComparison bfunc)
  {
    bfunc_num_regs_ = n;    //number of registers to be updated
    bfunc_ = bfunc;
  }
  bool is_diagonal(void)
  {
    return true;
  }
  uint_t size(int num_qubits)
  {
    this->chunk_bits_ = 0;
    return 1;
  }

  //in uint param array: array of qubits results to be stored, mask registers, target registers 
  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t iChunk = i;
    uint_t n64 = (this->num_creg_bits_ + 63) >> 6;
    uint_t* regs = this->params_;
    uint_t* mask = regs + bfunc_num_regs_;
    uint_t* target = mask + n64;
    int_t comp;
    uint_t j,i64,ibit;
    bool ret = true;

    for(j=0;j<n64;j++){
      comp = (this->cregs_[iChunk*n64 + n64-j-1] & mask[n64-j-1]) - target[n64-j-1];
      if(comp < 0){
        if(bfunc_ == Operations::RegComparison::Less || bfunc_ == Operations::RegComparison::LessEqual){
          break;
        }
        else if(bfunc_ == Operations::RegComparison::Equal || bfunc_ == Operations::RegComparison::Greater || bfunc_ == Operations::RegComparison::GreaterEqual){
          ret= false;
          break;
        }
      }
      else if(comp > 0){
        if(bfunc_ == Operations::RegComparison::Greater || bfunc_ == Operations::RegComparison::GreaterEqual){
          break;
        }
        else if(bfunc_ == Operations::RegComparison::Equal || bfunc_ == Operations::RegComparison::Less || bfunc_ == Operations::RegComparison::LessEqual){
          ret= false;
          break;
        }
      }
      else if(bfunc_ == Operations::RegComparison::NotEqual && mask[n64-j-1] != 0){
        ret= false;
        break;
      }
    }
    //store result in creg
    if(ret){
      for(j=0;j<bfunc_num_regs_;j++){
        i64 = regs[j] >> 6;
        ibit = regs[j] & 63;
        this->cregs_[iChunk*n64 + i64] |= (1ull << ibit);
      }
    }
    else{
      for(j=0;j<bfunc_num_regs_;j++){
        i64 = regs[j] >> 6;
        ibit = regs[j] & 63;
        this->cregs_[iChunk*n64 + i64] &= ~(1ull << ibit);
      }
    }
  }

  const char* name(void)
  {
    return "bfunc_kernel";
  }

};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_bfunc(const Operations::Op &op)
{
  if(((multi_chunk_distribution_ && chunk_.device() >= 0) || enable_batch_) && chunk_.pos() != 0)
    return;   //first chunk execute all in batch

  reg_t params;
  int_t i,n64,n,iparam;

  //registers to be updated
  for(i=0;i<op.registers.size();i++)
    params.push_back(op.registers[i]);

  n64 = (num_creg_bits_ + num_cmem_bits_ + QV_NUM_INTERNAL_REGS + 63) >> 6;   //number of 64-bit integer

  for(iparam=0;iparam<2;iparam++){
    uint_t val,added = 0;
    n = op.string_params[iparam].size();
    for(i=0;i<n-2;i+=16){
      std::string tmp;
      tmp = "0x";
      if(i + 16 > n - 2)
        tmp += op.string_params[iparam].substr(2,n-2-i);
      else
        tmp += op.string_params[iparam].substr(n-i-16,16);
      val = std::stoull(tmp, nullptr, 16);

      params.push_back(val);
      added++;
    }
    for(i=added;i<n64;i++){  //pack 0 if there is not enough values in string_params
      val = 0;
      params.push_back(val);
    }
  }

  chunk_.StoreUintParams(params);

  apply_function(bfunc_kernel<data_t>(op.registers.size(),op.bfunc));

  chunk_.container()->request_creg_update();
}

template <typename data_t>
class roerror_kernel : public Chunk::GateFuncBase<data_t>
{
protected:
  uint_t num_regs_;
  uint_t num_mems_;
  double* rnd_buf_;
public:
  roerror_kernel(uint_t n,uint_t nr,double* pRnd)
  {
    num_regs_ = nr;
    num_mems_ = n;
    rnd_buf_ = pRnd;
  }
  bool is_diagonal(void)
  {
    return true;
  }
  uint_t size(int num_qubits)
  {
    this->chunk_bits_ = 0;
    return 1;
  }

  //in uint param array: array of qubits results to be stored, mask registers, target registers 
  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t iChunk = i;
    uint_t n64 = (this->num_creg_bits_ + 63) >> 6;
    uint_t* mems = this->params_;
    uint_t* probs_offset = mems + num_mems_;
    double* probs = (double*)this->matrix_;
    uint_t j,i64,ibit,ip,outcome;
    double total_prob,p;

    ip = 0;
    for(j=0;j<num_mems_;j++){
      i64 = (mems[j]+num_regs_) >> 6;
      ibit = (mems[j]+num_regs_) & 63;
      ip += ( ((this->cregs_[iChunk*n64 + i64] >> ibit) & 1) << j);
    }

    total_prob = 0;
    for(j=probs_offset[ip];j<probs_offset[ip+1];j++){
      total_prob += probs[j];
    }
    outcome = probs_offset[ip+1];
    p = 0;
    for(j=probs_offset[ip];j<probs_offset[ip+1];j++){
      p += probs[j];
      if(p/total_prob >= rnd_buf_[iChunk]){
        outcome = j;
        break;
      }
    }
    outcome -= probs_offset[ip];

    for(j=0;j<num_mems_;j++){
      i64 = (mems[j]+num_regs_) >> 6;
      ibit = (mems[j]+num_regs_) & 63;
      this->cregs_[iChunk*n64 + i64] = (this->cregs_[iChunk*n64 + i64] & (~(1ull << ibit)) ) | (((outcome >> j) & 1) << ibit);
    }
  }

  const char* name(void)
  {
    return "roerror_kernel";
  }

};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_roerror(const Operations::Op &op, std::vector<RngEngine> &rng)
{
  if(((multi_chunk_distribution_ && chunk_.device() >= 0) || enable_batch_) && chunk_.pos() != 0)
    return;   //first chunk execute all in batch

  reg_t params;
  std::vector<double> probs;
  int_t i,j,offset;

  for(i=0;i<op.memory.size();i++)
    params.push_back(op.memory[i]);

  offset = 0;
  for(i=0;i<op.probs.size();i++){
    params.push_back(offset);
    offset += op.probs[i].size();
    probs.insert(probs.end(),op.probs[i].begin(),op.probs[i].end());
  }
  params.push_back(offset);

  std::vector<double> r(chunk_.container()->num_chunks());
  for(i=0;i<chunk_.container()->num_chunks();i++){
    r[i] = rng[i].rand(0., 1.);
  }
  chunk_.container()->copy_to_probability_buffer(r,0);

  chunk_.StoreUintParams(params);
  if((offset & 1) == 1)
    probs.push_back(0.0);
  chunk_.StoreMatrix((std::complex<double>*)&probs[0],probs.size()/2);

  apply_function(roerror_kernel<data_t>(op.memory.size(),num_creg_bits_,chunk_.probability_buffer()));

  chunk_.container()->request_creg_update();
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

  chunk_.synchronize();

  n = 16;
  if(n > data_size_)
    n = data_size_;
  for(i=0;i<n;i++){
    idx = i*data_size_/n;
    t = chunk_.Get(idx);
    spdlog::debug("   {0:05b} | {1:e}, {2:e}",idx,t.real(),t.imag());
  }
  if(n < data_size_){
    idx = data_size_-1;
    t = chunk_.Get(idx);
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
