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



#ifndef _qv_qubit_vector_thrust_hpp_
#define _qv_qubit_vector_thrust_hpp_

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/complex.h>

#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>

#include <thrust/execution_policy.h>


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


#include <omp.h>

#include "framework/json.hpp"

#include "simulators/statevector/qubitvector.hpp"

#ifdef QASM_TIMING

#include <sys/time.h>
double mysecond()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

#define QS_NUM_GATES					5
#define QS_GATE_INIT					0
#define QS_GATE_MULT					1
#define QS_GATE_CX						2
#define QS_GATE_DIAG					3
#define QS_GATE_MEASURE					4

#endif

namespace QV {

	
	
template <typename data_t = complex_t*>
class QubitVectorThrust {

public:

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  QubitVectorThrust();
  explicit QubitVectorThrust(size_t num_qubits);
  virtual ~QubitVectorThrust();
  QubitVectorThrust(const QubitVectorThrust& obj) = delete;
  QubitVectorThrust &operator=(const QubitVectorThrust& obj) = delete;

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

  // Return JSON serialization of QubitVectorThrust;
  json_t json() const;

  // Set all entries in the vector to 0.
  void zero();

  // index0 returns the integer representation of a number of bits set
  // to zero inserted into an arbitrary bit string.
  // Eg: for qubits 0,2 in a state k = ba ( ba = 00 => k=0, etc).
  // indexes0([1], k) -> int(b0a)
  // indexes0([1,3], k) -> int(0b0a)
  // Example: k = 77  = 1001101 , qubits_sorted = [1,4]
  // ==> output = 297 = 100101001 (with 0's put into places 1 and 4).
  template<typename list_t>
  uint_t index0(const list_t &qubits_sorted, const uint_t k) const;

  // Return a std::unique_ptr to an array of of 2^N in ints
  // each int corresponds to an N qubit bitstring for M-N qubit bits in state k,
  // and the specified N qubits in states [0, ..., 2^N - 1]
  // qubits_sorted must be sorted lowest to highest. Eg. {0, 1}.
  // qubits specifies the location of the qubits in the returned strings.
  // NOTE: since the return is a unique_ptr it cannot be copied.
  // indexes returns the array of all bit values for the specified qubits
  // (Eg: for qubits 0,2 in a state k = ba:
  // indexes([1], [1], k) = [int(b0a), int(b1a)],
  // if it were two qubits inserted say at 1,3 it would be:
  // indexes([1,3], [1,3], k) -> [int(0b0a), int(0b1a), int(1b0a), (1b1a)]
  // If the qubits were passed in reverse order it would swap qubit position in the list:
  // indexes([3,1], [1,3], k) -> [int(0b0a), int(1b0a), int(0b1a), (1b1a)]
  // Example: k=77, qubits=qubits_sorted=[1,4] ==> output=[297,299,313,315]
  // input: k = 77  = 1001101
  // output[0]: 297 = 100101001 (with 0's put into places 1 and 4).
  // output[1]: 299 = 100101011 (with 0 put into place 1, and 1 put into place 4).
  // output[2]: 313 = 100111001 (with 1 put into place 1, and 0 put into place 4).
  // output[3]: 313 = 100111011 (with 1's put into places 1 and 4).
  indexes_t indexes(const reg_t &qubits, const reg_t &qubits_sorted, const uint_t k) const;

  // As above but returns a fixed sized array of of 2^N in ints
  template<size_t N>
  areg_t<1ULL << N> indexes(const areg_t<N> &qs, const areg_t<N> &qubits_sorted, const uint_t k) const;

  // State initialization of a component
  // Initialize the specified qubits to a desired statevector
  // (leaving the other qubits in their current state)
  // assuming the qubits being initialized have already been reset to the zero state
  // (using apply_reset)
  void initialize_component(const reg_t &qubits, const cvector_t &state);

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

  // Apply a stacked set of 2^control_count target_count--qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const cvector_t &mat);

  // Apply a 1-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const uint_t qubit, const cvector_t &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const reg_t &qubits, const cvector_t &mat);
  
  // Swap pairs of indicies in the underlying vector
  void apply_permutation_matrix(const reg_t &qubits,
                                const std::vector<std::pair<uint_t, uint_t>> &pairs);

  	void matMult_2x2(thrust::complex<double>* pVec,int qubit,thrust::complex<double>* pMat);
  	void matMult_4x4(thrust::complex<double>* pVec,int qubit0,int qubit1,thrust::complex<double>* pMat);
  	void matMult_8x8(thrust::complex<double>* pVec,int qubit0,int qubit1,int qubit2,thrust::complex<double>* pMat);
  	void matMult_16x16(thrust::complex<double>* pVec,int qubit0,int qubit1,int qubit2,int qubit3,thrust::complex<double>* pMat);
  	void matMult_NxN(thrust::complex<double>* pVec,uint_t* qubits,int nqubits,thrust::complex<double>* pMat);
  	void diagMult_2x2(thrust::complex<double>* pVec,int qubit,thrust::complex<double>* pMat);
  	void diagMult_NxN(thrust::complex<double>* pVec,uint_t* qubits,int nqubits,thrust::complex<double>* pMat);
  	void CX(thrust::complex<double>* pVec,int qubit_c,int qubit_t);
  	void CY(thrust::complex<double>* pVec,int qubit_c,int qubit_t);
  	void X(thrust::complex<double>* pVec,int qubit);
  	void Y(thrust::complex<double>* pVec,int qubit);
  	void phase_1(thrust::complex<double>* pVec,int qubit,thrust::complex<double> p);

  	double dot_q(thrust::complex<double>* pVec,int qubit) const;

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
  void apply_mcphase(const reg_t &qubits, const complex_t phase);

  // Apply a general multi-controlled single-qubit unitary gate
  // If N=1 this implements an optimized single-qubit U gate
  // If N=2 this implements an optimized CU gate
  // If N=3 this implements an optimized CCU gate
  void apply_mcu(const reg_t &qubits, const cvector_t &mat);

  // Apply a general multi-controlled SWAP gate
  // If N=2 this implements an optimized SWAP  gate
  // If N=3 this implements an optimized Fredkin gate
  void apply_mcswap(const reg_t &qubits);

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
  uint_t omp_threshold_ = 14;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 10; // Sample measure indexing qubit size
  double json_chop_threshold_ = 0;  // Threshold for choping small values
                                    // in JSON serialization

  //-----------------------------------------------------------------------
  // Error Messages
  //-----------------------------------------------------------------------

  void check_qubit(const uint_t qubit) const;
  void check_vector(const cvector_t &diag, uint_t nqubits) const;
  void check_matrix(const cvector_t &mat, uint_t nqubits) const;
  void check_dimension(const QubitVectorThrust &qv) const;
  void check_checkpoint() const;

	int m_iDev;
	int m_nDev;

#ifdef QASM_TIMING
	mutable uint_t m_gateCounts[QS_NUM_GATES];
	mutable double m_gateTime[QS_NUM_GATES];
	mutable double m_gateStartTime[QS_NUM_GATES];

	void TimeStart(int i) const;
	void TimeEnd(int i) const;
	void TimeReset(void);
	void TimePrint(void);
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
json_t QubitVectorThrust<data_t>::json() const {
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
void QubitVectorThrust<data_t>::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits_) {
    std::string error = "QubitVectorThrust: qubit index " + std::to_string(qubit) +
                        " > " + std::to_string(num_qubits_);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_matrix(const cvector_t &vec, uint_t nqubits) const {
  const size_t DIM = BITS[nqubits];
  const auto SIZE = vec.size();
  if (SIZE != DIM * DIM) {
    std::string error = "QubitVectorThrust: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM * DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_vector(const cvector_t &vec, uint_t nqubits) const {
  const size_t DIM = BITS[nqubits];
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
QubitVectorThrust<data_t>::QubitVectorThrust(size_t num_qubits) : num_qubits_(0), data_(nullptr), checkpoint_(0){
  set_num_qubits(num_qubits);
}

template <typename data_t>
QubitVectorThrust<data_t>::QubitVectorThrust() : QubitVectorThrust(0) {}

template <typename data_t>
QubitVectorThrust<data_t>::~QubitVectorThrust() {

#ifdef QASM_TIMING
	TimePrint();
#endif

	if (data_){
		//free(data_);
		cudaFree(data_);
	}

  if (checkpoint_)
    free(checkpoint_);
}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <typename data_t>
complex_t &QubitVectorThrust<data_t>::operator[](uint_t element) {
  // Error checking
  #ifdef DEBUG
  if (element > data_size_) {
    std::string error = "QubitVectorThrust: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }
  #endif
  return data_[element];
}

template <typename data_t>
complex_t QubitVectorThrust<data_t>::operator[](uint_t element) const {
  // Error checking
  #ifdef DEBUG
  if (element > data_size_) {
    std::string error = "QubitVectorThrust: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }
  #endif
  return data_[element];
}

template <typename data_t>
cvector_t QubitVectorThrust<data_t>::vector() const {
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
template <typename list_t>
uint_t QubitVectorThrust<data_t>::index0(const list_t &qubits_sorted, const uint_t k) const {
  uint_t lowbits, retval = k;
  for (size_t j = 0; j < qubits_sorted.size(); j++) {
    lowbits = retval & MASKS[qubits_sorted[j]];
    retval >>= qubits_sorted[j];
    retval <<= qubits_sorted[j] + 1;
    retval |= lowbits;
  }
  return retval;
}

template <typename data_t>
template <size_t N>
areg_t<1ULL << N> QubitVectorThrust<data_t>::indexes(const areg_t<N> &qs,
                                               const areg_t<N> &qubits_sorted,
                                               const uint_t k) const {
  areg_t<1ULL << N> ret;
  ret[0] = index0(qubits_sorted, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = BITS[i];
    const auto bit = BITS[qs[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}

template <typename data_t>
indexes_t QubitVectorThrust<data_t>::indexes(const reg_t& qubits,
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
// State initialize component
//------------------------------------------------------------------------------
template <typename data_t>
void QubitVectorThrust<data_t>::initialize_component(const reg_t &qubits, const cvector_t &state) {

	/*
  // Lambda function for initializing component
  const size_t N = qubits.size();
  auto lambda = [&](const indexes_t &inds, const cvector_t &_state)->void {
    const uint_t DIM = 1ULL << N;
    complex_t cache = data_[inds[0]];  // the k-th component of non-initialized vector
    for (size_t i = 0; i < DIM; i++) {
      data_[inds[i]] = cache * _state[i];  // set component to psi[k] * state[i]
    }    // (where psi is is the post-reset state of the non-initialized qubits)
   };
  // Use the lambda function
  apply_lambda(lambda, qubits, state);
	*/
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::zero()
{
	uint_t n = data_size_;
	thrust::complex<double>* pVec = (thrust::complex<double>*)&data_[0];

	thrust::fill(thrust::device, pVec, pVec+n, 0.0);
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_num_qubits(size_t num_qubits) {

  size_t prev_num_qubits = num_qubits_;
  num_qubits_ = num_qubits;
  data_size_ = BITS[num_qubits];

  if (checkpoint_) {
    free(checkpoint_);
    checkpoint_ = nullptr;
  }

  // Free any currently assigned memory
  if (data_) {
    if (prev_num_qubits != num_qubits_) {
      //free(data_);
    	cudaFree(data_);
      data_ = nullptr;
    }
  }


	int tid,nid;
	nid = omp_get_num_threads();
	tid = omp_get_thread_num();
	cudaGetDeviceCount(&m_nDev);
	m_iDev = tid % m_nDev;
	cudaSetDevice(m_iDev);

  // Allocate memory for new vector
	if (data_ == nullptr){
		void* pData;

#ifdef QASM_TIMING
		TimeReset();
		TimeStart(QS_GATE_INIT);
#endif
		//using Unified Memory for debug/develoment
		cudaMallocManaged(&pData,sizeof(complex_t) * data_size_);
		//posix_memalign(&pData,128,sizeof(complex_t) * data_size_);
		data_ = reinterpret_cast<complex_t*>(pData);

		cudaMemPrefetchAsync(pData,sizeof(complex_t) * data_size_,m_iDev);

#ifdef QASM_DEBUG
	printf(" ==== Thrust qubit vector initialization ==== \n");
#endif

#ifdef QASM_TIMING
		TimeEnd(QS_GATE_INIT);
#endif
	}

}


template <typename data_t>
void QubitVectorThrust<data_t>::checkpoint() {
  if (!checkpoint_)
    checkpoint_ = reinterpret_cast<complex_t*>(malloc(sizeof(complex_t) * data_size_));

  const int_t END = data_size_;    // end for k loop
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t k = 0; k < END; ++k)
    checkpoint_[k] = data_[k];
}


template <typename data_t>
void QubitVectorThrust<data_t>::revert(bool keep) {

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
complex_t QubitVectorThrust<data_t>::inner_product() const {

	/*
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
	*/
	return 0.0;
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::initialize() {
  zero();
  data_[0] = 1.;
}

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_from_vector(const cvector_t &statevec) {
  if (data_size_ != statevec.size()) {
    std::string error = "QubitVectorThrust::initialize input vector is incorrect length (" + 
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
void QubitVectorThrust<data_t>::initialize_from_data(const data_t &statevec, const size_t num_states) {
  if (data_size_ != num_states) {
    std::string error = "QubitVectorThrust::initialize input vector is incorrect length (" +
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
void QubitVectorThrust<data_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t &mat) 
{
	const size_t N = qubits.size();

#ifdef QASM_TIMING
	TimeStart(QS_GATE_MULT);
#endif
	if(N == 1){
		matMult_2x2((thrust::complex<double>*)&data_[0],qubits[0],(thrust::complex<double>*)&mat[0]);
	}
	else if(N == 2){
		matMult_4x4((thrust::complex<double>*)&data_[0],qubits[0],qubits[1],(thrust::complex<double>*)&mat[0]);
	}
	else if(N == 3){
		matMult_8x8((thrust::complex<double>*)&data_[0],qubits[0],qubits[1],qubits[2],(thrust::complex<double>*)&mat[0]);
	}
	else if(N == 4){
		matMult_16x16((thrust::complex<double>*)&data_[0],qubits[0],qubits[1],qubits[2],qubits[3],(thrust::complex<double>*)&mat[0]);
	}
	else{
		matMult_NxN((thrust::complex<double>*)&data_[0],(uint_t*)&qubits[0],N,(thrust::complex<double>*)&mat[0]);
	}

#ifdef QASM_TIMING
	TimeEnd(QS_GATE_MULT);
#endif
}


template <typename data_t>
void QubitVectorThrust<data_t>::apply_multiplexer(const reg_t &control_qubits,
		const reg_t &target_qubits,
		const cvector_t &mat)
{
	printf(" apply_multiplexer NOT SUPPORTED : %d, %d \n",control_qubits.size(),target_qubits.size());
			/*
  // General implementation
  const size_t control_count = control_qubits.size();
  const size_t target_count  = target_qubits.size();
  const uint_t DIM = BITS[(target_count+control_count)];
  const uint_t columns = BITS[target_count];
  const uint_t blocks = BITS[control_count];
  // Lambda function for stacked matrix multiplication
  auto lambda = [&](const indexes_t &inds, const cvector_t &_mat)->void {
    auto cache = std::make_unique<complex_t[]>(DIM);
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
  apply_lambda(lambda, qubits, mat);
	*/
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t &diag) 
{
	const int_t N = qubits.size();

#ifdef QASM_TIMING
	TimeStart(QS_GATE_DIAG);
#endif
	if(N == 1){
		diagMult_2x2((thrust::complex<double>*)&data_[0],qubits[0],(thrust::complex<double>*)&diag[0]);
	}
	else{
		diagMult_NxN((thrust::complex<double>*)&data_[0],(uint_t*)&qubits[0],N,(thrust::complex<double>*)&diag[0]);
	}

#ifdef QASM_TIMING
	TimeEnd(QS_GATE_DIAG);
#endif
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_permutation_matrix(const reg_t& qubits,
                                                   const std::vector<std::pair<uint_t, uint_t>> &pairs) {
  const size_t N = qubits.size();
	printf(" apply_permutation_matrix : %d , %d   NOT SUPPORTED\n",N,qubits[0]);
/*
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
                                                   	*/
}

	
template <typename data_t>
void QubitVectorThrust<data_t>::matMult_2x2(thrust::complex<double>* pVec,int qubit,thrust::complex<double>* pMat)
{
//	printf("   TEST : mat 2x2 : %d\n",qubit);

	uint_t n;
	uint_t add = 1ull << qubit;
	uint_t mask = add - 1;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto matMult2x2_lambda = [=] __host__ __device__ (uint_t i) 
	{
		uint_t i0,i1;
		thrust::complex<double> q0,q1,m0,m1,m2,m3;

		i1 = i & mask;
		i0 = (i - i1) << 1;
		i0 += i1;
		i1 = i0 + add;

		q0 = pVec[i0];
		q1 = pVec[i1];

		m0 = pMat[0];
		m1 = pMat[1];
		m2 = pMat[2];
		m3 = pMat[3];

		pVec[i0] = m0 * q0 + m2 * q1;
		pVec[i1] = m1 * q0 + m3 * q1;
	};

	n = data_size_ >> 1;
	thrust::for_each(thrust::device, ci, ci+n, matMult2x2_lambda);
}

template <typename data_t>
void QubitVectorThrust<data_t>::matMult_4x4(thrust::complex<double>* pVec,int qubit0,int qubit1,thrust::complex<double>* pMat)
{
//	printf("   TEST : mat 4x4 : %d, %d\n",qubit0,qubit1);

	uint_t n;
	uint_t add0 = 1ull << qubit0;
	uint_t add1 = 1ull << qubit1;
	uint_t mask0 = add0 - 1;
	uint_t mask1 = add1 - 1;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto matMult4x4_lambda = [=] __host__ __device__ (uint_t i) 
	{
		uint_t i0,i1,i2,i3;
		thrust::complex<double> q0,q1,q2,q3,m0,m1,m2,m3;

		i0 = i & mask0;
		i = (i - i0) << 1;
		i1 = i & mask1;
		i = (i - i1) << 1;
		i2 = i;

		i0 = i0 + i1 + i2;
		i1 = i0 + add0;
		i2 = i0 + add1;
		i3 = i2 + add0;

		q0 = pVec[i0];
		q1 = pVec[i1];
		q2 = pVec[i2];
		q3 = pVec[i3];

		m0 = pMat[0];
		m1 = pMat[4];
		m2 = pMat[8];
		m3 = pMat[12];

		pVec[i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3;

		m0 = pMat[1];
		m1 = pMat[5];
		m2 = pMat[9];
		m3 = pMat[13];

		pVec[i1] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3;

		m0 = pMat[2];
		m1 = pMat[6];
		m2 = pMat[10];
		m3 = pMat[14];

		pVec[i2] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3;

		m0 = pMat[3];
		m1 = pMat[7];
		m2 = pMat[11];
		m3 = pMat[15];

		pVec[i3] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3;
	};

	n = data_size_ >> 2;
	thrust::for_each(thrust::device, ci, ci+n, matMult4x4_lambda);
}

template <typename data_t>
void QubitVectorThrust<data_t>::matMult_8x8(thrust::complex<double>* pVec,int qubit0,int qubit1,int qubit2,thrust::complex<double>* pMat)
{
//	printf("   TEST : mat 8x8 : %d, %d, %d\n",qubit0,qubit1,qubit2);

	uint_t n;
	uint_t add0 = 1ull << qubit0;
	uint_t add1 = 1ull << qubit1;
	uint_t add2 = 1ull << qubit2;
	uint_t mask0 = add0 - 1;
	uint_t mask1 = add1 - 1;
	uint_t mask2 = add2 - 1;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto matMult8x8_lambda = [=] __host__ __device__ (uint_t i) 
	{
		uint_t i0,i1,i2,i3,i4,i5,i6,i7;
		thrust::complex<double> q0,q1,q2,q3,q4,q5,q6,q7;
		thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;

		i0 = i & mask0;
		i = (i - i0) << 1;
		i1 = i & mask1;
		i = (i - i1) << 1;
		i2 = i & mask2;
		i = (i - i2) << 1;
		i3 = i;

		i0 = i0 + i1 + i2 + i3;
		i1 = i0 + add0;
		i2 = i0 + add1;
		i3 = i2 + add0;
		i4 = i0 + add2;
		i5 = i4 + add0;
		i6 = i4 + add1;
		i7 = i6 + add0;

		q0 = pVec[i0];
		q1 = pVec[i1];
		q2 = pVec[i2];
		q3 = pVec[i3];
		q4 = pVec[i4];
		q5 = pVec[i5];
		q6 = pVec[i6];
		q7 = pVec[i7];

		m0 = pMat[0];
		m1 = pMat[8];
		m2 = pMat[16];
		m3 = pMat[24];
		m4 = pMat[32];
		m5 = pMat[40];
		m6 = pMat[48];
		m7 = pMat[56];

		pVec[i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[1];
		m1 = pMat[9];
		m2 = pMat[17];
		m3 = pMat[25];
		m4 = pMat[33];
		m5 = pMat[41];
		m6 = pMat[49];
		m7 = pMat[57];

		pVec[i1] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[2];
		m1 = pMat[10];
		m2 = pMat[18];
		m3 = pMat[26];
		m4 = pMat[34];
		m5 = pMat[42];
		m6 = pMat[50];
		m7 = pMat[58];

		pVec[i2] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[3];
		m1 = pMat[11];
		m2 = pMat[19];
		m3 = pMat[27];
		m4 = pMat[35];
		m5 = pMat[43];
		m6 = pMat[51];
		m7 = pMat[59];

		pVec[i3] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[4];
		m1 = pMat[12];
		m2 = pMat[20];
		m3 = pMat[28];
		m4 = pMat[36];
		m5 = pMat[44];
		m6 = pMat[52];
		m7 = pMat[60];

		pVec[i4] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[5];
		m1 = pMat[13];
		m2 = pMat[21];
		m3 = pMat[29];
		m4 = pMat[37];
		m5 = pMat[45];
		m6 = pMat[53];
		m7 = pMat[61];

		pVec[i5] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[6];
		m1 = pMat[14];
		m2 = pMat[22];
		m3 = pMat[30];
		m4 = pMat[38];
		m5 = pMat[46];
		m6 = pMat[54];
		m7 = pMat[62];

		pVec[i6] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[7];
		m1 = pMat[15];
		m2 = pMat[23];
		m3 = pMat[31];
		m4 = pMat[39];
		m5 = pMat[47];
		m6 = pMat[55];
		m7 = pMat[63];

		pVec[i7] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
	};

	n = data_size_ >> 3;
	thrust::for_each(thrust::device, ci, ci+n, matMult8x8_lambda);
}

template <typename data_t>
void QubitVectorThrust<data_t>::matMult_16x16(thrust::complex<double>* pVec,int qubit0,int qubit1,int qubit2,int qubit3,thrust::complex<double>* pMat)
{
//	printf("   TEST : mat 16x16 : %d, %d, %d\n",qubit0,qubit1,qubit2);

	uint_t n;
	uint_t add0 = 1ull << qubit0;
	uint_t add1 = 1ull << qubit1;
	uint_t add2 = 1ull << qubit2;
	uint_t add3 = 1ull << qubit3;
	uint_t mask0 = add0 - 1;
	uint_t mask1 = add1 - 1;
	uint_t mask2 = add2 - 1;
	uint_t mask3 = add3 - 1;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto matMult16x16_lambda = [=] __host__ __device__ (uint_t i) 
	{
		uint_t i0,i1,i2,i3,i4,i5,i6,i7;
		uint_t i8,i9,i10,i11,i12,i13,i14,i15;
		thrust::complex<double> q0,q1,q2,q3,q4,q5,q6,q7;
		thrust::complex<double> q8,q9,q10,q11,q12,q13,q14,q15;
		thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;
		thrust::complex<double> m8,m9,m10,m11,m12,m13,m14,m15;
		int j;

		i0 = i & mask0;
		i = (i - i0) << 1;
		i1 = i & mask1;
		i = (i - i1) << 1;
		i2 = i & mask2;
		i = (i - i2) << 1;
		i3 = i & mask3;
		i = (i - i3) << 1;
		i4 = i;

		i0 = i0 + i1 + i2 + i3 + i4;
		i1 = i0 + add0;
		i2 = i0 + add1;
		i3 = i2 + add0;
		i4 = i0 + add2;
		i5 = i4 + add0;
		i6 = i4 + add1;
		i7 = i6 + add0;
		i8 = i0 + add3;
		i9 = i8 + add0;
		i10= i8 + add1;
		i11= i10+ add0;
		i12= i8 + add2;
		i13= i12+ add0;
		i14= i12+ add1;
		i15= i14+ add0;

		q0 = pVec[i0];
		q1 = pVec[i1];
		q2 = pVec[i2];
		q3 = pVec[i3];
		q4 = pVec[i4];
		q5 = pVec[i5];
		q6 = pVec[i6];
		q7 = pVec[i7];
		q8 = pVec[i8];
		q9 = pVec[i9];
		q10 = pVec[i10];
		q11 = pVec[i11];
		q12 = pVec[i12];
		q13 = pVec[i13];
		q14 = pVec[i14];
		q15 = pVec[i15];

		j = 0;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i0] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 1;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i1] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 2;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i2] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 3;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i3] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 4;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i4] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 5;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i5] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 6;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i6] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 7;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i7] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 8;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i8] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 9;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i9] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 10;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i10] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 11;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i11] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 12;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i12] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 13;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i13] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 14;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i14] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;

		j = 15;
		m0 = pMat[0+j];
		m1 = pMat[16+j];
		m2 = pMat[32+j];
		m3 = pMat[48+j];
		m4 = pMat[64+j];
		m5 = pMat[80+j];
		m6 = pMat[96+j];
		m7 = pMat[112+j];
		m8 = pMat[128+j];
		m9 = pMat[144+j];
		m10= pMat[160+j];
		m11= pMat[176+j];
		m12= pMat[192+j];
		m13= pMat[208+j];
		m14= pMat[224+j];
		m15= pMat[240+j];

		pVec[i15] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
					m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;
	};

	n = data_size_ >> 4;
	thrust::for_each(thrust::device, ci, ci+n, matMult16x16_lambda);
}

template <typename data_t>
void QubitVectorThrust<data_t>::matMult_NxN(thrust::complex<double>* pVec,uint_t* qubits,int nqubits,thrust::complex<double>* pMat)
{

	uint_t n;
	int matSize = 1 << nqubits;
	uint_t* offset;
	int j,k;
	uint_t add;

//	printf("   TEST : mat %dx%d\n",matSize,matSize);

	auto ci = thrust::counting_iterator<uint_t>(0);

	cudaMallocManaged(&offset,sizeof(uint_t)*matSize);

	for(k=0;k<matSize;k++){
		offset[k] = 0;
	}
	for(j=0;j<nqubits;j++){
		add = (1ull << qubits[j]);
		for(k=0;k<matSize;k++){
			if((k >> j) & 1){
				offset[k] += add;
			}
		}
	}

	auto matMultNxN_lambda = [=] __host__ __device__ (uint_t i) 
	{
		thrust::complex<double> q[32];
		thrust::complex<double> m;
		thrust::complex<double> r;
		int j,k,l;
		uint_t ii,idx,t;
		uint_t mask;

		idx = 0;
		ii = i;
		for(j=0;j<nqubits;j++){
			mask = (1ull << qubits[j]) - 1;

			t = ii & mask;
			idx += t;
			ii = (ii - t) << 1;
		}
		idx += ii;

		for(k=0;k<matSize;k++){
			q[k] = pVec[offset[k] + idx];
		}

		for(j=0;j<matSize;j++){
			r = 0.0;
			for(k=0;k<matSize;k++){
				l = (j + (k << nqubits));
				m = pMat[l];

				r += m*q[k];
			}

			pVec[offset[j] + idx] = r;
		}
	};

	n = data_size_ >> nqubits;
	thrust::for_each(thrust::device, ci, ci+n, matMultNxN_lambda);

	cudaFree(offset);
}

template <typename data_t>
void QubitVectorThrust<data_t>::diagMult_2x2(thrust::complex<double>* pVec,int qubit,thrust::complex<double>* pMat)
{
//	printf("   TEST : diag 2x2 : %d\n",qubit);

	uint_t n;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto diagMult2x2_lambda = [=] __host__ __device__ (uint_t i) 
	{
		int im;
		thrust::complex<double> q,m;

		im = (i >> qubit) & 1;

		q = pVec[i];
		m = pMat[im];

		pVec[i] = m * q;
	};

	n = data_size_;
	thrust::for_each(thrust::device, ci, ci+n, diagMult2x2_lambda);
}

template <typename data_t>
void QubitVectorThrust<data_t>::diagMult_NxN(thrust::complex<double>* pVec,uint_t* qubits,int nqubits,thrust::complex<double>* pMat)
{
//	printf("   TEST : diag NxN \n");

	uint_t n;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto diagMultNxN_lambda = [=] __host__ __device__ (uint_t i) 
	{
		int im,j;
		thrust::complex<double> q,m;

		im = 0;
		for(j=0;j<nqubits;j++){
			if((i & (1ull << qubits[j])) != 0){
				im += (1 << j);
			}
		}

		q = pVec[i];
		m = pMat[im];

		pVec[i] = m * q;
	};

	n = data_size_;
	thrust::for_each(thrust::device, ci, ci+n, diagMultNxN_lambda);
}

template <typename data_t>
void QubitVectorThrust<data_t>::phase_1(thrust::complex<double>* pVec,int qubit,thrust::complex<double> p)
{
//	printf("   TEST : Phase : %d\n",qubit);

	uint_t n;
	uint_t mask = 1ull << qubit;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto phase_lambda = [=] __host__ __device__ (uint_t i) 
	{
		thrust::complex<double> q0;

		if((i & mask) != 0){
			q0 = pVec[i];
			pVec[i ] = q0 * p;
		}
	};

	n = data_size_;
	thrust::for_each(thrust::device, ci, ci+n, phase_lambda);

}

template <typename data_t>
void QubitVectorThrust<data_t>::CX(thrust::complex<double>* pVec,int qubit_c,int qubit_t)
{
//	printf("   TEST : CX : %d, %d\n",qubit_c,qubit_t);

	uint_t n;
	uint_t add = 1ull << qubit_t;
	uint_t mask = 1ull << qubit_c;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto CX_lambda = [=] __host__ __device__ (uint_t i) 
	{
		uint_t ip;
		thrust::complex<double> q0,q1;

		if((i & mask) != 0){
			ip = i ^ add;
			if(i < ip){
				q0 = pVec[i];
				q1 = pVec[ip];

				pVec[i ] = q1;
				pVec[ip] = q0;
			}
		}
	};

	n = data_size_;
	thrust::for_each(thrust::device, ci, ci+n, CX_lambda);
}

template <typename data_t>
void QubitVectorThrust<data_t>::X(thrust::complex<double>* pVec,int qubit)
{
//	printf("   TEST : X : %d\n",qubit);

	uint_t n;
	uint_t add = 1ull << qubit;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto X_lambda = [=] __host__ __device__ (uint_t i) 
	{
		uint_t ip;
		thrust::complex<double> q0,q1;

		ip = i ^ add;
		if(i < ip){
			q0 = pVec[i];
			q1 = pVec[ip];

			pVec[i ] = q1;
			pVec[ip] = q0;
		}
	};

	n = data_size_;
	thrust::for_each(thrust::device, ci, ci+n, X_lambda);
}

template <typename data_t>
void QubitVectorThrust<data_t>::Y(thrust::complex<double>* pVec,int qubit)
{
//	printf("   TEST : Y : %d\n",qubit);

	uint_t n;
	uint_t add = 1ull << qubit;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto Y_lambda = [=] __host__ __device__ (uint_t i) 
	{
		uint_t ip;
		thrust::complex<double> q0,q1;

		ip = i ^ add;
		if(i < ip){
			q0 = pVec[i];
			q1 = pVec[ip];

			pVec[i ] = thrust::complex<double>(q1.imag(),-q1.real());
			pVec[ip] = thrust::complex<double>(-q0.imag(),q0.real());
		}
	};

	n = data_size_;
	thrust::for_each(thrust::device, ci, ci+n, Y_lambda);
}

template <typename data_t>
void QubitVectorThrust<data_t>::CY(thrust::complex<double>* pVec,int qubit_c,int qubit_t)
{
//	printf("   TEST : CY : %d, %d\n",qubit_c,qubit_t);

	uint_t n;
	uint_t add = 1ull << qubit_t;
	uint_t mask = 1ull << qubit_c;

	auto ci = thrust::counting_iterator<uint_t>(0);

	auto CY_lambda = [=] __host__ __device__ (uint_t i) 
	{
		uint_t ip;
		thrust::complex<double> q0,q1;

		if((i & mask) != 0){
			ip = i ^ add;
			if(i < ip){
				q0 = pVec[i];
				q1 = pVec[ip];

				pVec[i ] = thrust::complex<double>(q1.imag(),-q1.real());
				pVec[ip] = thrust::complex<double>(-q0.imag(),q0.real());
			}
		}
	};

	n = data_size_;
	thrust::for_each(thrust::device, ci, ci+n, CY_lambda);
}

struct dot_lambda : public unary_function<uint_t,double>
{
	thrust::complex<double>* m_pVec;
	int m_qubit;

	dot_lambda(thrust::complex<double>* pV,int q)
	{
		m_pVec = pV;
		m_qubit = q;
	}

	__host__ __device__ double operator()(const uint_t &i) const
	{
		uint_t i0,i1;
		thrust::complex<double> q0;

		i1 = i & ((1ull << m_qubit) - 1);
		i0 = (i - i1) << 1;
		i0 += i1;

		q0 = m_pVec[i0];

		return q0.real()*q0.real() + q0.imag()*q0.imag();
	}
};

template <typename data_t>
double QubitVectorThrust<data_t>::dot_q(thrust::complex<double>* pVec,int qubit) const
{
//	printf("   TEST : Dot : %d\n",qubit);

	uint_t n;
	uint_t add = 1ull << qubit;
	uint_t mask = add - 1;
	double ret;

	auto ci = thrust::counting_iterator<uint_t>(0);


	n = data_size_ >> 1;
	ret = thrust::transform_reduce(thrust::device, ci, ci+n, dot_lambda(pVec,qubit),0.0,thrust::plus<double>());

	return ret;
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
void QubitVectorThrust<data_t>::apply_mcx(const reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();

	if(N == 1){
		X((thrust::complex<double>*)&data_[0],qubits[0]);
	}
	else if(N == 2){
#ifdef QASM_TIMING
		TimeStart(QS_GATE_CX);
#endif
		CX((thrust::complex<double>*)&data_[0],qubits[0],qubits[1]);
#ifdef QASM_TIMING
		TimeEnd(QS_GATE_CX);
#endif
	}

/*
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
	*/
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcy(const reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = MASKS[N];
  const complex_t I(0., 1.);

	if(N == 1){
		Y((thrust::complex<double>*)&data_[0],qubits[0]);
	}
	else if(N == 2){
		CY((thrust::complex<double>*)&data_[0],qubits[0],qubits[1]);
	}


	/*
  switch (N) {
    case 1: {
      // Lambda function for Y gate
      auto lambda = [&](const areg_t<2> &inds)->void {
        const complex_t cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CY gate
      auto lambda = [&](const areg_t<4> &inds)->void {
        const complex_t cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for CCY gate
      auto lambda = [&](const areg_t<8> &inds)->void {
        const complex_t cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled Y gate
      auto lambda = [&](const indexes_t &inds)->void {
        const complex_t cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, qubits);
    }
  } // end switch
	*/
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcswap(const reg_t &qubits) {
  // Calculate the swap positions for the last two qubits.
  // If N = 2 this is just a regular SWAP gate rather than a controlled-SWAP gate.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = pos0 + BITS[N - 2];

	printf(" apply_mcswap : %d NOT SUPPORTED\n",N);
	/*
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
	*/
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcphase(const reg_t &qubits, const complex_t phase) {
  const size_t N = qubits.size();

	if(N == 1){
		phase_1((thrust::complex<double>*)&data_[0],qubits[0],*(thrust::complex<double>*)&phase);
	}

	/*
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
	*/
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcu(const reg_t &qubits,
                                    const cvector_t &mat) 
{
	// Calculate the permutation positions for the last qubit.
	const size_t N = qubits.size();

	if(N == 1){
		if(mat[1] == 0.0 && mat[2] == 0.0){
#ifdef QASM_TIMING
			TimeStart(QS_GATE_DIAG);
#endif
			const cvector_t diag = {{mat[0], mat[3]}};

			diagMult_2x2((thrust::complex<double>*)&data_[0],qubits[0],(thrust::complex<double>*)&diag[0]);

#ifdef QASM_TIMING
			TimeEnd(QS_GATE_DIAG);
#endif
		}
		else{
#ifdef QASM_TIMING
			TimeStart(QS_GATE_MULT);
#endif
			matMult_2x2((thrust::complex<double>*)&data_[0],qubits[0],(thrust::complex<double>*)&mat[0]);

#ifdef QASM_TIMING
			TimeEnd(QS_GATE_MULT);
#endif
		}
	}

	/*
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
    const cvector_t diag = {{mat[0], mat[3]}};
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
                          const cvector_t &_diag)->void {
          data_[inds[pos0]] = _diag[0] * data_[inds[pos0]];
          data_[inds[pos1]] = _diag[1] * data_[inds[pos1]];
        };
        apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}), diag);
        return;
      }
      case 3: {
        // Lambda function for CCU gate
        auto lambda = [&](const areg_t<8> &inds,
                          const cvector_t &_diag)->void {
          data_[inds[pos0]] = _diag[0] * data_[inds[pos0]];
          data_[inds[pos1]] = _diag[1] * data_[inds[pos1]];
        };
        apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}), diag);
        return;
      }
      default: {
        // Lambda function for general multi-controlled U gate
        auto lambda = [&](const indexes_t &inds,
                          const cvector_t &_diag)->void {
          data_[inds[pos0]] = _diag[0] * data_[inds[pos0]];
          data_[inds[pos1]] = _diag[1] * data_[inds[pos1]];
        };
        apply_lambda(lambda, qubits, diag);
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
                        const cvector_t &_mat)->void {
      const auto cache = data_[inds[pos0]];
      data_[inds[pos0]] = _mat[0] * data_[inds[pos0]] + _mat[2] * data_[inds[pos1]];
      data_[inds[pos1]] = _mat[1] * cache + _mat[3] * data_[inds[pos1]];
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}), mat);
      return;
    }
    case 3: {
      // Lambda function for CCU gate
      auto lambda = [&](const areg_t<8> &inds,
                        const cvector_t &_mat)->void {
      const auto cache = data_[inds[pos0]];
      data_[inds[pos0]] = _mat[0] * data_[inds[pos0]] + _mat[2] * data_[inds[pos1]];
      data_[inds[pos1]] = _mat[1] * cache + _mat[3] * data_[inds[pos1]];
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}), mat);
      return;
    }
    default: {
      // Lambda function for general multi-controlled U gate
      auto lambda = [&](const indexes_t &inds,
                        const cvector_t &_mat)->void {
      const auto cache = data_[inds[pos0]];
      data_[inds[pos0]] = _mat[0] * data_[inds[pos0]] + _mat[2] * data_[inds[pos1]];
      data_[inds[pos1]] = _mat[1] * cache + _mat[3] * data_[inds[pos1]];
      };
      apply_lambda(lambda, qubits, mat);
      return;
    }
  } // end switch
	*/
}

//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::apply_matrix(const uint_t qubit,
                                       const cvector_t& mat)
{
  // Check if matrix is diagonal and if so use optimized lambda
  if (mat[1] == 0.0 && mat[2] == 0.0) {
#ifdef QASM_TIMING
	TimeStart(QS_GATE_DIAG);
#endif
  	const cvector_t diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix(qubit, diag);

#ifdef QASM_TIMING
	TimeEnd(QS_GATE_DIAG);
#endif
  	return;
  }
#ifdef QASM_TIMING
	TimeStart(QS_GATE_MULT);
#endif
	matMult_2x2((thrust::complex<double>*)&data_[0],qubit,(thrust::complex<double>*)&mat[0]);
#ifdef QASM_TIMING
	TimeEnd(QS_GATE_MULT);
#endif
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t& diag)
{
#ifdef QASM_TIMING
	TimeStart(QS_GATE_DIAG);
#endif
	diagMult_2x2((thrust::complex<double>*)&data_[0],qubit,(thrust::complex<double>*)&diag[0]);
#ifdef QASM_TIMING
	TimeEnd(QS_GATE_DIAG);
#endif
}

/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <typename data_t>
double QubitVectorThrust<data_t>::norm() const {
	/*
  // Lambda function for norm
  auto lambda = [&](int_t k, double &val_re, double &val_im)->void {
    (void)val_im; // unused
    val_re += std::real(data_[k] * std::conj(data_[k]));
  };
  return std::real(apply_reduction_lambda(lambda));
	*/
	return 0.0;
}

template <typename data_t>
double QubitVectorThrust<data_t>::norm(const reg_t &qubits, const cvector_t &mat) const {

	/*
  const uint_t N = qubits.size();

  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Static array optimized lambda functions
  switch (N) {
    case 1:
      return norm(qubits[0], mat);
    case 2: {
      // Lambda function for 2-qubit matrix norm
      auto lambda = [&](const areg_t<4> &inds, const cvector_t &_mat, 
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 4; i++) {
          complex_t vi = 0;
          for (size_t j = 0; j < 4; j++)
            vi += _mat[i + 4 * j] * data_[inds[j]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<2> qubits_arr = {{qubits[0], qubits[1]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, mat));
    }
    case 3: {
      // Lambda function for 3-qubit matrix norm
      auto lambda = [&](const areg_t<8> &inds, const cvector_t &_mat, 
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 8; i++) {
          complex_t vi = 0;
          for (size_t j = 0; j < 8; j++)
            vi += _mat[i + 8 * j] * data_[inds[j]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<3> qubits_arr = {{qubits[0], qubits[1], qubits[2]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, mat));
    }
    case 4: {
      // Lambda function for 4-qubit matrix norm
      auto lambda = [&](const areg_t<16> &inds, const cvector_t &_mat, 
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 16; i++) {
          complex_t vi = 0;
          for (size_t j = 0; j < 16; j++)
            vi += _mat[i + 16 * j] * data_[inds[j]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<4> qubits_arr = {{qubits[0], qubits[1], qubits[2], qubits[3]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, mat));
    }
    default: {
      // Lambda function for N-qubit matrix norm
      const uint_t DIM = BITS[N];
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
      return std::real(apply_reduction_lambda(lambda, qubits, mat));
    }
  } // end switch
	*/
	return 0.0;
}

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const reg_t &qubits, const cvector_t &mat) const {
	/*

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
      auto lambda = [&](const areg_t<4> &inds, const cvector_t &_mat, 
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 4; i++) {
          const auto vi = _mat[i] * data_[inds[i]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<2> qubits_arr = {{qubits[0], qubits[1]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, mat));
    }
    case 3: {
      // Lambda function for 3-qubit matrix norm
      auto lambda = [&](const areg_t<8> &inds, const cvector_t &_mat, 
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 8; i++) {
          const auto vi = _mat[i] * data_[inds[i]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<3> qubits_arr = {{qubits[0], qubits[1], qubits[2]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, mat));
    }
    case 4: {
      // Lambda function for 4-qubit matrix norm
      auto lambda = [&](const areg_t<16> &inds, const cvector_t &_mat, 
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < 16; i++) {
          const auto vi = _mat[i] * data_[inds[i]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      areg_t<4> qubits_arr = {{qubits[0], qubits[1], qubits[2], qubits[3]}};
      return std::real(apply_reduction_lambda(lambda, qubits_arr, mat));
    }
    default: {
      // Lambda function for N-qubit matrix norm
      const uint_t DIM = BITS[N];
      auto lambda = [&](const indexes_t &inds, const cvector_t &_mat,
                        double &val_re, double &val_im)->void {
        (void)val_im; // unused
        for (size_t i = 0; i < DIM; i++) {
          const auto vi = _mat[i] * data_[inds[i]];
          val_re += std::real(vi * std::conj(vi));
        }
      };
      // Use the lambda function
      return std::real(apply_reduction_lambda(lambda, qubits, mat));
    }
  } // end switch
	*/
	return 0.0;
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
double QubitVectorThrust<data_t>::norm(const uint_t qubit, const cvector_t &mat) const {
	/*
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 2);
  #endif

  // Check if input matrix is diagonal, and if so use diagonal function.
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const cvector_t diag = {{mat[0], mat[3]}};
    return norm_diagonal(qubit, diag);
  }

  // Lambda function for norm reduction to real value.
  auto lambda = [&](const areg_t<2> &inds,
                    const cvector_t &_mat,
                    double &val_re,
                    double &val_im)->void {
    (void)val_im; // unused
    const auto v0 = _mat[0] * data_[inds[0]] + _mat[2] * data_[inds[1]];
    const auto v1 = _mat[1] * data_[inds[0]] + _mat[3] * data_[inds[1]];
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_reduction_lambda(lambda, areg_t<1>({{qubit}}), mat));
	*/
	return 0.0;
}

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const uint_t qubit, const cvector_t &mat) const {
	/*
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 1);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const areg_t<2> &inds,
                    const cvector_t &_mat,
                    double &val_re,
                    double &val_im)->void {
    (void)val_im; // unused
    const auto v0 = _mat[0] * data_[inds[0]];
    const auto v1 = _mat[1] * data_[inds[1]];
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_reduction_lambda(lambda, areg_t<1>({{qubit}}), mat));
	*/
	return 0.0;
}


/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/

template <typename data_t>
double QubitVectorThrust<data_t>::probability(const uint_t outcome) const {
  const auto v = data_[outcome];
  return std::real(v * std::conj(v));
}

template <typename data_t>
rvector_t QubitVectorThrust<data_t>::probabilities() const {
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
rvector_t QubitVectorThrust<data_t>::probabilities(const reg_t &qubits) const
{
	const size_t N = qubits.size();
	rvector_t probs((1ull << N), 0.);

	if(N == 1){
		double d = dot_q((thrust::complex<double>*)&data_[0],qubits[0]);
		probs[0] = d;
		probs[1] = 1.0 - probs[0];

#ifdef QASM_DEBUG
//		printf(" prob[%d] : (%e, %e) \n",qubits[0],probs[0],probs[1]);
#endif
	}
	return probs;
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------

template <typename data_t>
rvector_t QubitVectorThrust<data_t>::probabilities(const uint_t qubit) const 
{
	double p0,p1;

	p0 = dot_q((thrust::complex<double>*)&data_[0],qubit);
	p1 = 1.0 - p0;

	return rvector_t({p0,p1});

}


//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
reg_t QubitVectorThrust<data_t>::sample_measure(const std::vector<double> &rnds) const 
{
	const int_t SHOTS = rnds.size();
	reg_t samples;
	double* pVec = (double*)&data_[0];
	uint_t n = data_size_*2;
	int i;
	thrust::device_vector<double> vRnd(SHOTS);
	thrust::device_vector<unsigned long> vSamp(SHOTS);
	thrust::host_vector<double> hvRnd(SHOTS);
	thrust::host_vector<unsigned long> hvSamp(SHOTS);

#ifdef QASM_TIMING
	TimeStart(QS_GATE_MEASURE);
#endif

	samples.assign(SHOTS, 0);

	thrust::transform_inclusive_scan(thrust::device,pVec,pVec+n,pVec,thrust::square<double>(),thrust::plus<double>());

#pragma omp parallel for
	for(i=0;i<SHOTS;i++){
		hvRnd[i] = rnds[i];
	}
	vRnd = hvRnd;

	thrust::lower_bound(thrust::device, pVec, pVec + n, vRnd.begin(), vRnd.end(), vSamp.begin());

	hvSamp = vSamp;

#pragma omp parallel for
	for(i=0;i<SHOTS;i++){
		samples[i] = hvSamp[i]/2;
	}

#ifdef QASM_TIMING
	TimeEnd(QS_GATE_MEASURE);
#endif
	return samples;
}

#ifdef QASM_TIMING

template <typename data_t>
void QubitVectorThrust<data_t>::TimeReset(void)
{
	int i;
	for(i=0;i<QS_NUM_GATES;i++){
		m_gateCounts[i] = 0;
		m_gateTime[i] = 0.0;
	}
}

template <typename data_t>
void QubitVectorThrust<data_t>::TimeStart(int i) const
{
	m_gateStartTime[i] = mysecond();
}

template <typename data_t>
void QubitVectorThrust<data_t>::TimeEnd(int i) const
{
	double t = mysecond();
	m_gateTime[i] += t - m_gateStartTime[i];
	m_gateCounts[i]++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::TimePrint(void)
{
	int i;
	double total;

	total = 0;
	for(i=0;i<QS_NUM_GATES;i++){
		total += m_gateTime[i];
	}

	printf("   ==================== Timing Summary =================== \n");
	if(m_gateCounts[QS_GATE_INIT] > 0)
		printf("  Initialization : %f \n",m_gateTime[QS_GATE_INIT]);
	if(m_gateCounts[QS_GATE_MULT] > 0)
		printf("    Matrix mult. : %f  (%d)\n",m_gateTime[QS_GATE_MULT],m_gateCounts[QS_GATE_MULT]);
	if(m_gateCounts[QS_GATE_CX] > 0)
		printf("    CX           : %f  (%d)\n",m_gateTime[QS_GATE_CX],m_gateCounts[QS_GATE_CX]);
	if(m_gateCounts[QS_GATE_DIAG] > 0)
		printf("    Diagonal     : %f  (%d)\n",m_gateTime[QS_GATE_DIAG],m_gateCounts[QS_GATE_DIAG]);
	if(m_gateCounts[QS_GATE_MEASURE] > 0)
		printf("    Measure      : %f  (%d)\n",m_gateTime[QS_GATE_MEASURE],m_gateCounts[QS_GATE_MEASURE]);
	printf("    Total Kernel time : %f sec\n",total);
	printf("   ======================================================= \n");

}

#endif
	
//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const QV::QubitVectorThrust<data_t>&qv) {

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
