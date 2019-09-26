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
#include <thrust/functional.h>
#include <thrust/system/cuda/pointer.h>

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

#define QASM_DEFAULT_MATRIX_BITS		8


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

// Template class for qubit vector.
// The arguement of the template must have an operator[] access method.
// The following methods may also need to be template specialized:
//   * set_num_qubits(size_t)
//   * initialize()
//   * initialize_from_vector(cvector_t<data_t>)
// If the template argument does not have these methods then template
// specialization must be used to override the default implementations.

template <typename data_t = double>
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
  std::complex<data_t> &operator[](uint_t element);
  std::complex<data_t> operator[](uint_t element) const;

  // Returns a reference to the underlying data_t data class
  std::complex<data_t>* &data() {return data_;}

  // Returns a copy of the underlying data_t data class
  std::complex<data_t>* data() const {return data_;}

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

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

  // Return JSON serialization of QubitVectorThrust;
  json_t json() const;

  // Set all entries in the vector to 0.
  void zero();

  // convert vector type to data type of this qubit vector
  cvector_t<data_t> convert(const cvector_t<double>& v) const;

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
  	template <typename UnaryFunction>
	void apply_lambda(UnaryFunction func,uint_t n);

  	template <typename UnaryFunction>
	double apply_sum_lambda(UnaryFunction func,uint_t n) const;

	void allocate_buffers(int qubit);

	int m_iDev;
	int m_nDev;
	int m_nDevParallel;
	int m_useATS;
	int m_useDevMem;

	thrust::complex<double>* m_pMatDev;
	uint_t* m_pUintBuf;
	int m_matBits;
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
  const size_t DIM = BITS[nqubits];
  const auto SIZE = vec.size();
  if (SIZE != DIM * DIM) {
    std::string error = "QubitVectorThrust: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM * DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVectorThrust<data_t>::check_vector(const cvector_t<data_t> &vec, uint_t nqubits) const {
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
	m_pMatDev = NULL;
	m_useATS = 0;
  set_num_qubits(num_qubits);
}

template <typename data_t>
QubitVectorThrust<data_t>::QubitVectorThrust() : QubitVectorThrust(0) {
	m_pMatDev = NULL;
	m_useATS = 0;
}

template <typename data_t>
QubitVectorThrust<data_t>::~QubitVectorThrust() {
#ifdef QASM_TIMING
	TimePrint();
#endif

	if (data_){
		if(m_useATS){
			free(data_);
		}
		else{
			cudaFree(data_);
		}
	}
	if(m_pMatDev){
		cudaFree(m_pMatDev);
		cudaFree(m_pUintBuf);
	}

  if (checkpoint_)
    free(checkpoint_);
}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <typename data_t>
std::complex<data_t> &QubitVectorThrust<data_t>::operator[](uint_t element) {
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
std::complex<data_t> QubitVectorThrust<data_t>::operator[](uint_t element) const {
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
cvector_t<data_t> QubitVectorThrust<data_t>::vector() const {
  cvector_t<data_t> ret(data_size_, 0.);
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
struct initialize_component_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double>* status;
	uint_t* offset;
	uint_t* qubits;
	int nqubits;
	uint_t matSize;

	initialize_component_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pS,uint_t* pBuf,uint_t* qb,int nq)
	{
		int j,k;
		uint_t add;
		pVec = pV;
		nqubits = nq;
		matSize = 1ull << nqubits;
		offset = pBuf;
		qubits = pBuf + matSize;
		status = pS;

		for(k=0;k<matSize;k++){
			qubits[k] = qb[k];
			offset[k] = 0;
		}
		for(j=0;j<nqubits;j++){
			add = (1ull << qb[j]);
			for(k=0;k<matSize;k++){
				if((k >> j) & 1){
					offset[k] += add;
				}
			}
		}
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		thrust::complex<double> q0;
		thrust::complex<double> q;
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

		q0 = pVec[offset[0] + idx];
		for(k=0;k<matSize;k++){
			q = pVec[offset[k] + idx];
			q = q0 * status[k];
			pVec[offset[k] + idx] = q;
		}
	}
};


template <typename data_t>
void QubitVectorThrust<data_t>::initialize_component(const reg_t &qubits, const cvector_t<double> &state0) 
{
	const size_t N = qubits.size();
	uint_t size;

	thrust::complex<double>* pMat;

#ifdef QASM_HAS_ATS
	pMat = (thrust::complex<double>*)&state0;
#else
	uint_t i,matSize;
	matSize = 1ull << N;

	allocate_buffers(N);

	pMat = m_pMatDev;
#pragma omp parallel for 
	for(i=0;i<matSize*matSize;i++){
		m_pMatDev[i] = state0[i];
	}
#endif

	size = data_size_ >> N;
	apply_lambda(initialize_component_lambda<data_t>((thrust::complex<data_t>*)&data_[0],pMat,m_pUintBuf,(uint_t*)&qubits[0],N), size);
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
struct fill_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<data_t> val;

	fill_lambda(thrust::complex<data_t>* pV,thrust::complex<data_t>& v)
	{
		pVec = pV;
		val = v;
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		pVec[i] = val;
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::zero()
{
	uint_t size;
	thrust::complex<data_t> z = 0.0;

	size = data_size_;
	apply_lambda(fill_lambda<data_t>((thrust::complex<data_t>*)&data_[0],z), size);
}

template <typename data_t>
cvector_t<data_t> QubitVectorThrust<data_t>::convert(const cvector_t<double>& v) const {
  cvector_t<data_t> ret(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    ret[i] = v[i];
  return ret;
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
    	if(m_useATS){
    		free(data_);
    	}
    	else{
	    	cudaFree(data_);
    	}
    	data_ = nullptr;
    }
  }

	int tid,nid;
	char* str;

	nid = omp_get_num_threads();
	tid = omp_get_thread_num();
	cudaGetDeviceCount(&m_nDev);

	m_iDev = 0;
	if(nid > 1){
		m_iDev = tid % m_nDev;
		cudaSetDevice(m_iDev);
		m_nDevParallel = 1;
	}
	else{
		m_nDevParallel = 1;
		str = getenv("QASM_MULTI_GPU");
		if(str != NULL){
			m_nDevParallel = m_nDev;
		}
	}

	// Allocate memory for new vector
	if (data_ == nullptr){
		void* pData;

#ifdef QASM_TIMING
		TimeReset();
		TimeStart(QS_GATE_INIT);
#endif
		str = getenv("QASM_USE_ATS");
		if(str != NULL){
			posix_memalign(&pData,128,sizeof(thrust::complex<data_t>) * data_size_);
			m_useATS = 1;
		}
		else{
			str = getenv("QASM_USE_DEVMEM");
			if(str != NULL){
				cudaMalloc(&pData,sizeof(thrust::complex<data_t>) * data_size_);
				m_useDevMem = 1;
			}
			else{
				cudaMallocManaged(&pData,sizeof(thrust::complex<data_t>) * data_size_);
			}
			m_useATS = 0;
		}
		data_ = reinterpret_cast<std::complex<data_t>*>(pData);

		if(m_nDevParallel > 1){
			/*
			int iDev;
#pragma omp parallel for
			for(iDev=0;iDev < m_nDev;iDev++){
				uint_t is,ie;
				cudaStream_t strm;
				is = data_size_ * iDev / m_nDev;
				ie = data_size_ * (iDev+1) / m_nDev;
				cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
				cudaMemPrefetchAsync((complex_t*)pData + is,sizeof(complex_t) * (ie - is),iDev,strm);
				cudaStreamSynchronize(strm);
				cudaStreamDestroy(strm);
			}
			*/
		}
		else{
			//cudaMemPrefetchAsync(pData,sizeof(complex_t) * data_size_,m_iDev);
		}

#ifdef QASM_DEBUG
	printf(" ==== Thrust qubit vector initialization ==== \n");
	printf("    TEST : threads %d/%d , dev %d/%d\n",tid,nid,m_iDev,m_nDev);
#endif

#ifdef QASM_TIMING
		TimeEnd(QS_GATE_INIT);
#endif
	}

	allocate_buffers(QASM_DEFAULT_MATRIX_BITS);
}

template <typename data_t>
void QubitVectorThrust<data_t>::allocate_buffers(int nq)
{
	uint_t matSize;
	if(m_pMatDev == NULL){
		m_matBits = nq;
		matSize = 1ull << m_matBits;
		cudaMallocManaged(&m_pMatDev,sizeof(thrust::complex<data_t>) * matSize*matSize);
		cudaMallocManaged(&m_pUintBuf,sizeof(uint_t) * matSize * 4);
	}
	else{
		if(nq > m_matBits){
			matSize = 1ull << nq;
			cudaFree(m_pMatDev);
			cudaFree(m_pUintBuf);
			m_matBits = nq;
			cudaMallocManaged(&m_pMatDev,sizeof(thrust::complex<data_t>) * matSize*matSize);
			cudaMallocManaged(&m_pUintBuf,sizeof(uint_t) * matSize*4);
		}
	}
}

template <typename data_t>
size_t QubitVectorThrust<data_t>::required_memory_mb(uint_t num_qubits) const {

  size_t unit = std::log2(sizeof(std::complex<data_t>));
  size_t shift_mb = std::max<int_t>(0, num_qubits + unit - 20);
  size_t mem_mb = 1ULL << shift_mb;
  return mem_mb;
}


template <typename data_t>
void QubitVectorThrust<data_t>::checkpoint() {
  if (!checkpoint_)
    checkpoint_ = reinterpret_cast<std::complex<data_t>*>(malloc(sizeof(std::complex<data_t>) * data_size_));

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
std::complex<double> QubitVectorThrust<data_t>::inner_product() const
{
	uint_t i;
	double dr=0.0,di=0.0;

#pragma omp parallel for reduction(+:dr,di)
	for(i=0;i<data_size_;i++){
		dr += std::real(data_[i]) * std::real(checkpoint_[i]);
		di += std::imag(data_[i]) * std::imag(checkpoint_[i]);
	}
	return std::complex<double>(dr,di);
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
void QubitVectorThrust<data_t>::initialize_from_vector(const cvector_t<double> &statevec) {
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
void QubitVectorThrust<data_t>::initialize_from_data(const std::complex<data_t>* statevec, const size_t num_states) {
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
struct matMult2x2_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double> m0,m1,m2,m3;
	int qubit;
	uint_t add;
	uint_t mask;

	matMult2x2_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pMat,int q)
	{
		pVec = pV;
		qubit = q;
		m0 = pMat[0];
		m1 = pMat[1];
		m2 = pMat[2];
		m3 = pMat[3];

		add = 1ull << qubit;
		mask = add - 1;
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		uint_t i0,i1;
		thrust::complex<data_t> q0,q1;

		i1 = i & mask;
		i0 = (i - i1) << 1;
		i0 += i1;
		i1 = i0 + add;

		q0 = pVec[i0];
		q1 = pVec[i1];

		pVec[i0] = m0 * q0 + m2 * q1;
		pVec[i1] = m1 * q0 + m3 * q1;
	}
};

template <typename data_t>
struct matMult4x4_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double> m00,m10,m20,m30;
	thrust::complex<double> m01,m11,m21,m31;
	thrust::complex<double> m02,m12,m22,m32;
	thrust::complex<double> m03,m13,m23,m33;
	int qubit0;
	int qubit1;
	uint_t add0;
	uint_t mask0;
	uint_t add1;
	uint_t mask1;

	matMult4x4_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pMat,int q0,int q1)
	{
		pVec = pV;
		qubit0 = q0;
		qubit1 = q1;

		m00 = pMat[0];
		m01 = pMat[1];
		m02 = pMat[2];
		m03 = pMat[3];

		m10 = pMat[4];
		m11 = pMat[5];
		m12 = pMat[6];
		m13 = pMat[7];

		m20 = pMat[8];
		m21 = pMat[9];
		m22 = pMat[10];
		m23 = pMat[11];

		m30 = pMat[12];
		m31 = pMat[13];
		m32 = pMat[14];
		m33 = pMat[15];

		add0 = 1ull << qubit0;
		add1 = 1ull << qubit1;
		mask0 = add0 - 1;
		mask1 = add1 - 1;
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		uint_t i0,i1,i2,i3;
		thrust::complex<data_t> q0,q1,q2,q3;

		i0 = i & mask0;
		i2 = (i - i0) << 1;
		i1 = i2 & mask1;
		i2 = (i2 - i1) << 1;

		i0 = i0 + i1 + i2;
		i1 = i0 + add0;
		i2 = i0 + add1;
		i3 = i2 + add0;

		q0 = pVec[i0];
		q1 = pVec[i1];
		q2 = pVec[i2];
		q3 = pVec[i3];

		pVec[i0] = m00 * q0 + m10 * q1 + m20 * q2 + m30 * q3;

		pVec[i1] = m01 * q0 + m11 * q1 + m21 * q2 + m31 * q3;

		pVec[i2] = m02 * q0 + m12 * q1 + m22 * q2 + m32 * q3;

		pVec[i3] = m03 * q0 + m13 * q1 + m23 * q2 + m33 * q3;
	}
};

template <typename data_t>
struct matMult8x8_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double>* pMat;
	int qubit0;
	int qubit1;
	int qubit2;
	uint_t add0;
	uint_t mask0;
	uint_t add1;
	uint_t mask1;
	uint_t add2;
	uint_t mask2;

	matMult8x8_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pM,int q0,int q1,int q2)
	{
		pVec = pV;
		qubit0 = q0;
		qubit1 = q1;
		qubit2 = q2;

		pMat = pM;

		add0 = 1ull << qubit0;
		add1 = 1ull << qubit1;
		add2 = 1ull << qubit2;
		mask0 = add0 - 1;
		mask1 = add1 - 1;
		mask2 = add2 - 1;
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		uint_t i0,i1,i2,i3,i4,i5,i6,i7;
		thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
		thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;

		i0 = i & mask0;
		i3 = (i - i0) << 1;
		i1 = i3 & mask1;
		i3 = (i3 - i1) << 1;
		i2 = i3 & mask2;
		i3 = (i3 - i2) << 1;

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
	}
};

template <typename data_t>
struct matMult16x16_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double>* pMat;
	int qubit0;
	int qubit1;
	int qubit2;
	int qubit3;
	uint_t add0;
	uint_t mask0;
	uint_t add1;
	uint_t mask1;
	uint_t add2;
	uint_t mask2;
	uint_t add3;
	uint_t mask3;

	matMult16x16_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pM,int q0,int q1,int q2,int q3)
	{
		pVec = pV;
		qubit0 = q0;
		qubit1 = q1;
		qubit2 = q2;
		qubit3 = q3;

		pMat = pM;

		add0 = 1ull << qubit0;
		add1 = 1ull << qubit1;
		add2 = 1ull << qubit2;
		add3 = 1ull << qubit3;
		mask0 = add0 - 1;
		mask1 = add1 - 1;
		mask2 = add2 - 1;
		mask3 = add3 - 1;
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		uint_t i0,i1,i2,i3,i4,i5,i6,i7;
		uint_t i8,i9,i10,i11,i12,i13,i14,i15;
		thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
		thrust::complex<data_t> q8,q9,q10,q11,q12,q13,q14,q15;
		thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;
		thrust::complex<double> m8,m9,m10,m11,m12,m13,m14,m15;
		int j;

		i0 = i & mask0;
		i4 = (i - i0) << 1;
		i1 = i4 & mask1;
		i4 = (i4 - i1) << 1;
		i2 = i4 & mask2;
		i4 = (i4 - i2) << 1;
		i3 = i4 & mask3;
		i4 = (i4 - i3) << 1;

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
	}
};

//in-place NxN matrix multiplication using LU factorization
template <typename data_t>
struct matMultNxN_LU_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double>* pMat;
	uint_t* offset;
	uint_t* qubits;
	uint_t* pivot;
	uint_t* table;
	int nqubits;
	uint_t matSize;
	int nswap;

	matMultNxN_LU_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pM,uint_t* pBuf,uint_t* qb,int nq)
	{
		uint_t i,j,k,imax;
		thrust::complex<double> c0,c1;
		double d,dmax;
		uint_t add;
		uint_t* pSwap;

		pVec = pV;
		nqubits = nq;
		pMat = pM;
		matSize = 1ull << nqubits;
		offset = pBuf;
		qubits = pBuf + matSize;
		pivot = pBuf + matSize*2;
		table = pBuf + matSize*3;

		for(k=0;k<matSize;k++){
			qubits[k] = qb[k];
			offset[k] = 0;
		}
		for(j=0;j<nqubits;j++){
			add = (1ull << qb[j]);
			for(k=0;k<matSize;k++){
				if((k >> j) & 1){
					offset[k] += add;
				}
			}
		}

		//LU factorization of input matrix
		for(i=0;i<matSize;i++){
			pivot[i] = i;
		}
		for(i=0;i<matSize;i++){
			imax = i;
			dmax = thrust::abs(pMat[(i << nqubits) + pivot[i]]);
			for(j=i+1;j<matSize;j++){
				d = thrust::abs(pMat[(i << nqubits) + pivot[j]]);
				if(d > dmax){
					dmax = d;
					imax = j;
				}
			}
			if(imax != i){
				j = pivot[imax];
				pivot[imax] = pivot[i];
				pivot[i] = j;
			}

			if(dmax != 0){
				c0 = pMat[(i << nqubits) + pivot[i]];

				for(j=i+1;j<matSize;j++){
					c1 = pMat[(i << nqubits) + pivot[j]]/c0;

					for(k=i+1;k<matSize;k++){
						pMat[(k << nqubits) + pivot[j]] -= c1*pMat[(k << nqubits) + pivot[i]];
					}
					pMat[(i << nqubits) + pivot[j]] = c1;
				}
			}
		}

		//making table for swapping pivotted result
		pSwap = new uint_t[matSize];
		nswap = 0;
		for(i=0;i<matSize;i++){
			pSwap[i] = pivot[i];
		}
		i = 0;
		while(i<matSize){
			if(pSwap[i] != i){
				table[nswap++] = i;
				j = pSwap[i];
				table[nswap++] = j;
				k = pSwap[j];
				pSwap[j] = j;
				while(i != k){
					j = k;
					table[nswap++] = k;
					k = pSwap[j];
					pSwap[j] = j;
				}
				pSwap[i] = i;
			}
			i++;
		}
		delete[] pSwap;
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		thrust::complex<data_t> q,qt;
		thrust::complex<double> m;
		thrust::complex<double> r;
		uint_t j,k,l,ip;
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

		//mult U
		for(j=0;j<matSize;j++){
			r = 0.0;
			for(k=j;k<matSize;k++){
				l = (pivot[j] + (k << nqubits));
				m = pMat[l];
				q = pVec[offset[k] + idx];

				r += m*q;
			}
			pVec[offset[j] + idx] = r;
		}

		//mult L
		for(j=matSize-1;j>0;j--){
			r = pVec[offset[j] + idx];

			for(k=0;k<j;k++){
				l = (pivot[j] + (k << nqubits));
				m = pMat[l];
				q = pVec[offset[k] + idx];

				r += m*q;
			}
			pVec[offset[j] + idx] = r;
		}

		//swap results
		if(nswap > 0){
			q = pVec[offset[table[0]] + idx];
			k = pivot[table[0]];
			for(j=1;j<nswap;j++){
				qt = pVec[offset[table[j]] + idx];
				pVec[offset[k] + idx] = q;
				q = qt;
				k = pivot[table[j]];
			}
			pVec[offset[k] + idx] = q;
		}
	}
};


template <typename data_t>
template <typename UnaryFunction>
void QubitVectorThrust<data_t>::apply_lambda(UnaryFunction func,uint_t n)
{
	if(m_nDevParallel == 1){
		auto ci = thrust::counting_iterator<uint_t>(0);

		thrust::for_each(thrust::device, ci, ci+n, func);
	}
	else{
		int iDev;

#pragma omp parallel for
		for(iDev=0;iDev<m_nDevParallel;iDev++){
			uint_t is,ie;
			is = n * iDev / m_nDevParallel;
			ie = n * (iDev+1) / m_nDevParallel;

			auto ci = thrust::counting_iterator<uint_t>(0);

			cudaSetDevice(iDev);
			thrust::for_each(thrust::device, ci + is, ci + ie, func);
		}
	}
}

template <typename data_t>
template <typename UnaryFunction>
double QubitVectorThrust<data_t>::apply_sum_lambda(UnaryFunction func,uint_t n) const
{
	double ret = 0.0;

	if(m_nDevParallel == 1){
		auto ci = thrust::counting_iterator<uint_t>(0);

		ret = thrust::transform_reduce(thrust::device, ci, ci+n, func,0.0,thrust::plus<double>());
	}
	else{
		int iDev;

#pragma omp parallel for reduction(+:ret)
		for(iDev=0;iDev<m_nDevParallel;iDev++){
			uint_t is,ie;
			is = n * iDev / m_nDevParallel;
			ie = n * (iDev+1) / m_nDevParallel;

			auto ci = thrust::counting_iterator<uint_t>(0);

			cudaSetDevice(iDev);
			ret += thrust::transform_reduce(thrust::device, ci + is, ci + ie, func,0.0,thrust::plus<double>());
		}
	}
	return ret;
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t<double> &mat)
{
	const size_t N = qubits.size();
	uint_t size;

//	printf(" Mat Mult : %d\n",N);

#ifdef QASM_TIMING
	TimeStart(QS_GATE_MULT);
#endif
	if(N == 1){
		size = data_size_ >> 1;
		apply_lambda(matMult2x2_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(thrust::complex<double>*)&mat[0],qubits[0]), size);
	}
	else if(N == 2){
		size = data_size_ >> 2;
		apply_lambda(matMult4x4_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(thrust::complex<double>*)&mat[0],qubits[0],qubits[1]), size);
	}
	else{
		thrust::complex<double>* pMat;

		uint_t i,matSize;
		matSize = 1ull << N;

		allocate_buffers(N);

		pMat = m_pMatDev;

#pragma omp parallel for 
		for(i=0;i<matSize*matSize;i++){
			m_pMatDev[i] = mat[i];
		}

		if(N == 3){
			size = data_size_ >> 3;
			apply_lambda(matMult8x8_lambda<data_t>((thrust::complex<data_t>*)&data_[0],pMat,qubits[0],qubits[1],qubits[2]), size);
		}
		else if(N == 4){
			size = data_size_ >> 4;
			apply_lambda(matMult16x16_lambda<data_t>((thrust::complex<data_t>*)&data_[0],pMat,qubits[0],qubits[1],qubits[2],qubits[3]), size);
		}
		else{
			size = data_size_ >> N;
			apply_lambda(matMultNxN_LU_lambda<data_t>((thrust::complex<data_t>*)&data_[0],pMat,m_pUintBuf,(uint_t*)&qubits[0],N), size);
		}
	}

#ifdef QASM_TIMING
	TimeEnd(QS_GATE_MULT);
#endif

}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_multiplexer(const reg_t &control_qubits,
                                            const reg_t &target_qubits,
                                            const cvector_t<double>  &mat) {
	printf(" apply_multiplexer NOT SUPPORTED : %d, %d \n",control_qubits.size(),target_qubits.size());
			/*
  
  // General implementation
  const size_t control_count = control_qubits.size();
  const size_t target_count  = target_qubits.size();
  const uint_t DIM = BITS[(target_count+control_count)];
  const uint_t columns = BITS[target_count];
  const uint_t blocks = BITS[control_count];
  // Lambda function for stacked matrix multiplication
  auto lambda = [&](const indexes_t &inds, const cvector_t<data_t> &_mat)->void {
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
                                            	*/
}

template <typename data_t>
struct diagMult2x2_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double> m0,m1;
	int qubit;

	diagMult2x2_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pMat,int q)
	{
		pVec = pV;
		qubit = q;
		m0 = pMat[0];
		m1 = pMat[1];
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		thrust::complex<data_t> q;
		thrust::complex<double> m;

		q = pVec[i];
		if(((i >> qubit) & 1) == 0){
			m = m0;
		}
		else{
			m = m1;
		}

		pVec[i] = m * q;
	}
};

template <typename data_t>
struct diagMultNxN_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double>* pMat;
	int nqubits;
	uint_t* qubits;

	diagMultNxN_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pM,uint_t* pBuf,uint_t* qb,int nq)
	{
		int i;
		pVec = pV;
		pMat = pM;
		qubits = pBuf;
		nqubits = nq;
		for(i=0;i<nqubits;i++){
			qubits[i] = qb[i];
		}
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		int im,j;
		thrust::complex<data_t> q;
		thrust::complex<double> m;

		im = 0;
		for(j=0;j<nqubits;j++){
			if((i & (1ull << qubits[j])) != 0){
				im += (1 << j);
			}
		}

		q = pVec[i];
		m = pMat[im];

		pVec[i] = m * q;
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t<double> &diag)
{
	const int_t N = qubits.size();

//	printf(" Diag Mult : %d",N);
	
#ifdef QASM_TIMING
	TimeStart(QS_GATE_DIAG);
#endif
	if(N == 1){
		apply_lambda(diagMult2x2_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(thrust::complex<double>*)&diag[0],qubits[0]), data_size_ );
	}
	else{
		thrust::complex<double>* pMat;

#ifdef QASM_HAS_ATS
		pMat = (thrust::complex<double>*)&diag[0];
#else

		uint_t i,matSize;
		matSize = 1ull << N;

		allocate_buffers(N);

#pragma omp parallel for 
		for(i=0;i<matSize;i++){
			m_pMatDev[i] = diag[i];
		}
		pMat = m_pMatDev;
#endif
		apply_lambda(diagMultNxN_lambda<data_t>((thrust::complex<data_t>*)&data_[0],pMat,m_pUintBuf,(uint_t*)&qubits[0], N), data_size_ );
	}

#ifdef QASM_TIMING
	TimeEnd(QS_GATE_DIAG);
#endif
}

	
template <typename data_t>
struct permutation_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	uint_t* offset;
	uint_t* qubits;
	uint_t matSize;
	int nqubits;
	int npairs;

	permutation_lambda(thrust::complex<data_t>* pV,uint_t* pBuf,uint_t* qb,int nq,uint_t* pairs,int np)
	{
		uint_t j,k,ip;
		uint_t add;

		pVec = pV;
		nqubits = nq;
		offset = pBuf;
		qubits = pBuf + matSize;
		matSize = 1ull << nqubits;
		npairs = np;

		for(k=0;k<matSize;k++){
			qubits[k] = qb[k];
			offset[k] = 0;
		}

		for(j=0;j<nqubits;j++){
			add = (1ull << qubits[j]);
			for(k=0;k<matSize;k++){
				if((k >> j) & 1){
					for(ip=0;ip<np*2;ip++){
						if(k == pairs[ip]){
							offset[k] += add;
						}
					}
				}
			}
		}
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		thrust::complex<data_t> q;
		uint_t j;
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

		for(j=0;j<npairs;j++){
			q = pVec[offset[j*2] + idx];
			pVec[offset[j*2] + idx] = pVec[offset[j*2+1] + idx];
			pVec[offset[j*2+1] + idx] = q;
		}
	}
};


template <typename data_t>
void QubitVectorThrust<data_t>::apply_permutation_matrix(const reg_t& qubits,
                                                   const std::vector<std::pair<uint_t, uint_t>> &pairs)
{
	const size_t N = qubits.size();
	uint_t size = data_size_ >> N;

	allocate_buffers(N);

	apply_lambda(permutation_lambda<data_t>((thrust::complex<data_t>*)&data_[0],m_pUintBuf,(uint_t*)&qubits[0], N, (uint_t*)&pairs[0], pairs.size()),  size);
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
struct CX_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	uint_t add;
	uint_t mask;

	CX_lambda(thrust::complex<data_t>* pV,uint_t* qubits,int nqubits)
	{
		int i;
		pVec = pV;

		add = 1ull << qubits[nqubits-1];
		mask = 0;
		for(i=0;i<nqubits-1;i++){
			mask |= (1ull << qubits[i]);
		}
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		uint_t ip;
		thrust::complex<data_t> q0,q1;

		if((i & mask) == mask){
			ip = i ^ add;
			if(i < ip){
				q0 = pVec[i];
				q1 = pVec[ip];

				pVec[i ] = q1;
				pVec[ip] = q0;
			}
		}
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcx(const reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();

#ifdef QASM_TIMING
		TimeStart(QS_GATE_CX);
#endif

	apply_lambda(CX_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(uint_t*)&qubits[0], N), data_size_);

#ifdef QASM_TIMING
		TimeEnd(QS_GATE_CX);
#endif
}


template <typename data_t>
struct CY_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	uint_t add;
	uint_t mask;

	CY_lambda(thrust::complex<data_t>* pV,uint_t* qubits,int nqubits)
	{
		int i;
		pVec = pV;

		add = 1ull << qubits[nqubits-1];
		mask = 0;
		for(i=0;i<nqubits-1;i++){
			mask |= (1ull << qubits[i]);
		}
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		uint_t ip;
		thrust::complex<data_t> q0,q1;

		if((i & mask) == mask){
			ip = i ^ add;
			if(i < ip){
				q0 = pVec[i];
				q1 = pVec[ip];

				pVec[i ] = thrust::complex<data_t>(q1.imag(),-q1.real());
				pVec[ip] = thrust::complex<data_t>(-q0.imag(),q0.real());
			}
		}
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcy(const reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
//		printf("  Y\n ");

	apply_lambda(CY_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(uint_t*)&qubits[0], N), data_size_);
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcswap(const reg_t &qubits) {
  // Calculate the swap positions for the last two qubits.
  // If N = 2 this is just a regular SWAP gate rather than a controlled-SWAP gate.
	const size_t N = qubits.size();

	uint_t size = data_size_ >> N;
	uint_t pos[2];

	allocate_buffers(N);

	pos[0] = (1ull << (N-1)) - 1;
	pos[1] = pos[0] + (1ull << (N-2));
	apply_lambda(permutation_lambda<data_t>((thrust::complex<data_t>*)&data_[0],m_pUintBuf,(uint_t*)&qubits[0], N, pos, 1),  size);
}

template <typename data_t>
struct phase_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double> phase;
	uint_t mask;

	phase_lambda(thrust::complex<data_t>* pV,uint_t* qubits,int nqubits,thrust::complex<double> p)
	{
		int i;
		pVec = pV;
		phase = p;

		mask = 0;
		for(i=0;i<nqubits;i++){
			mask |= (1ull << qubits[i]);
		}
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		thrust::complex<data_t> q0;

		if((i & mask) == mask){
			q0 = pVec[i];
			pVec[i ] = q0 * phase;
		}
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcphase(const reg_t &qubits, const std::complex<double> phase) {
	const size_t N = qubits.size();

	apply_lambda(phase_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(uint_t*)&qubits[0], N,*(thrust::complex<double>*)&phase), data_size_ );
}


template <typename data_t>
struct diagMult2x2_controlled_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double> m0,m1;
	uint_t add;
	uint_t mask;

	diagMult2x2_controlled_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pMat,uint_t* qubits,int nqubits)
	{
		int i;
		pVec = pV;
		m0 = pMat[0];
		m1 = pMat[1];

		add = 1ull << qubits[nqubits-1];
		mask = 0;
		for(i=0;i<nqubits-1;i++){
			mask |= (1ull << qubits[i]);
		}
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		uint_t ip;
		thrust::complex<data_t> q;
		thrust::complex<double> m;

		if((i & mask) == mask){
			ip = i ^ add;
			if(i < ip){
				m = m0;
			}
			else{
				m = m1;
			}
			q = pVec[i];
			pVec[i] = m * q;
		}
	}
};

template <typename data_t>
struct matMult2x2_controlled_lambda : public unary_function<uint_t,void>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double> m0,m1,m2,m3;
	uint_t add;
	uint_t mask;

	matMult2x2_controlled_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pMat,uint_t* qubits,int nqubits)
	{
		int i;
		pVec = pV;
		m0 = pMat[0];
		m1 = pMat[1];
		m2 = pMat[2];
		m3 = pMat[3];

		add = 1ull << qubits[nqubits-1];
		mask = 0;
		for(i=0;i<nqubits-1;i++){
			mask |= (1ull << qubits[i]);
		}
	}

	__host__ __device__ void operator()(const uint_t &i) const
	{
		uint_t ip;
		thrust::complex<data_t> q0,q1;

		if((i & mask) == mask){
			ip = i ^ add;
			if(i < ip){
				q0 = pVec[i];
				q1 = pVec[ip];

				pVec[i]  = m0 * q0 + m2 * q1;
				pVec[ip] = m1 * q0 + m3 * q1;
			}
		}
	}
};
	
template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcu(const reg_t &qubits,
                                    const cvector_t<double> &mat) 
{
	// Calculate the permutation positions for the last qubit.
	const size_t N = qubits.size();
//	printf("   MCU %d\n",N);

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
			apply_lambda(diagMult2x2_controlled_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(thrust::complex<double>*)&diag[0],(uint_t*)&qubits[0],N), data_size_ );
		}
	}
	else{
		if(N == 1){
			// If N=1 this is just a single-qubit matrix
			apply_matrix(qubits[0], mat);
			return;
		}
		else{
			apply_lambda(matMult2x2_controlled_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(thrust::complex<double>*)&mat[0],(uint_t*)&qubits[0],N), data_size_ );
		}
	}
}

//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::apply_matrix(const uint_t qubit,
                                       const cvector_t<double>& mat) {

//	printf(" single Mat Mult : %d\n",qubit);

  // Check if matrix is diagonal and if so use optimized lambda
  if (mat[1] == 0.0 && mat[2] == 0.0) {
#ifdef QASM_TIMING
	TimeStart(QS_GATE_DIAG);
#endif
  	const std::vector<std::complex<double>> diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix(qubit, diag);

#ifdef QASM_TIMING
	TimeEnd(QS_GATE_DIAG);
#endif
  	return;
  }
#ifdef QASM_TIMING
	TimeStart(QS_GATE_MULT);
#endif

	apply_lambda(matMult2x2_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(thrust::complex<double>*)&mat[0],qubit), data_size_ >> 1);

#ifdef QASM_TIMING
	TimeEnd(QS_GATE_MULT);
#endif
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t<double>& diag) {

#ifdef QASM_TIMING
	TimeStart(QS_GATE_DIAG);
#endif
	apply_lambda(diagMult2x2_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(thrust::complex<double>*)&diag[0],qubit), data_size_ );
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
struct norm_lambda : public unary_function<uint_t,double>
{
	thrust::complex<data_t>* pVec;

	norm_lambda(thrust::complex<data_t>* pV)
	{
		pVec = pV;
	}

	__host__ __device__ double operator()(const uint_t &i) const
	{
		thrust::complex<data_t> q0;
		double ret;

		ret = 0.0;

		q0 = pVec[i];
		ret = q0.real()*q0.real() + q0.imag()*q0.imag();
		return ret;
	}
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm() const
{
	return apply_sum_lambda(norm_lambda<data_t>((thrust::complex<data_t>*)&data_[0]), data_size_);
}

template <typename data_t>
struct norm_matMultNxN_lambda : public unary_function<uint_t,double>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double>* pMat;
	uint_t* offset;
	uint_t* qubits;
	int nqubits;
	uint_t matSize;

	norm_matMultNxN_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pM,uint_t* pBuf,uint_t* qb,int nq)
	{
		uint_t j,k;
		uint_t add;

		pVec = pV;
		nqubits = nq;
		pMat = pM;
		matSize = 1ull << nqubits;
		offset = pBuf;
		qubits = pBuf + matSize;

		for(k=0;k<matSize;k++){
			qubits[k] = qb[k];
			offset[k] = 0;
		}
		for(j=0;j<nqubits;j++){
			add = (1ull << qb[j]);
			for(k=0;k<matSize;k++){
				if((k >> j) & 1){
					offset[k] += add;
				}
			}
		}
	}

	__host__ __device__ double operator()(const uint_t &i) const
	{
		thrust::complex<data_t> q;
		thrust::complex<double> m;
		thrust::complex<double> r;
		double sum = 0.0;
		uint_t j,k,l;
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

		for(j=0;j<matSize;j++){
			r = 0.0;
			for(k=0;k<matSize;k++){
				l = (j + (k << nqubits));
				m = pMat[l];
				q = pVec[offset[k] + idx];
				r += m*q;
			}
			sum += (r.real()*r.real() + r.imag()*r.imag());
		}
		return sum;
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
		thrust::complex<double>* pMat;

		uint_t i,matSize,size;
		matSize = 1ull << N;

		//allocate_buffers(N);

		pMat = m_pMatDev;

#pragma omp parallel for 
		for(i=0;i<matSize*matSize;i++){
			m_pMatDev[i] = mat[i];
		}

		size = data_size_ >> N;
		return apply_sum_lambda(norm_matMultNxN_lambda<data_t>((thrust::complex<data_t>*)&data_[0],pMat,m_pUintBuf,(uint_t*)&qubits[0],N), size);
	}
}

template <typename data_t>
struct norm_diagMultNxN_lambda : public unary_function<uint_t,double>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double>* pMat;
	int nqubits;
	uint_t* qubits;

	norm_diagMultNxN_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pM,uint_t* pBuf,uint_t* qb,int nq)
	{
		int i;
		pVec = pV;
		pMat = pM;
		qubits = pBuf;
		nqubits = nq;
		for(i=0;i<nqubits;i++){
			qubits[i] = qb[i];
		}
	}

	__host__ __device__ double operator()(const uint_t &i) const
	{
		int im,j;
		thrust::complex<data_t> q;
		thrust::complex<double> m,r;

		im = 0;
		for(j=0;j<nqubits;j++){
			if((i & (1ull << qubits[j])) != 0){
				im += (1 << j);
			}
		}

		q = pVec[i];
		m = pMat[im];

		r = m * q;
		return (r.real()*r.real() + r.imag()*r.imag());
	}
};
	
template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const reg_t &qubits, const cvector_t<double> &mat) const {

	const uint_t N = qubits.size();

	if(N == 1){
		return norm_diagonal(qubits[0], mat);
	}
	else{
		thrust::complex<double>* pMat;

#ifdef QASM_HAS_ATS
		pMat = (thrust::complex<double>*)&mat[0];
#else

		uint_t i,matSize;
		matSize = 1ull << N;

		//allocate_buffers(N);

#pragma omp parallel for 
		for(i=0;i<matSize;i++){
			m_pMatDev[i] = mat[i];
		}
		pMat = m_pMatDev;
#endif
		return apply_sum_lambda(norm_diagMultNxN_lambda<data_t>((thrust::complex<data_t>*)&data_[0],pMat,m_pUintBuf,(uint_t*)&qubits[0], N), data_size_ );
	}
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
struct norm_matMult2x2_lambda : public unary_function<uint_t,double>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double> m0,m1,m2,m3;
	int qubit;
	uint_t add;
	uint_t mask;

	norm_matMult2x2_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pMat,int q)
	{
		pVec = pV;
		qubit = q;
		m0 = pMat[0];
		m1 = pMat[1];
		m2 = pMat[2];
		m3 = pMat[3];

		add = 1ull << qubit;
		mask = add - 1;
	}

	__host__ __device__ double operator()(const uint_t &i) const
	{
		uint_t i0,i1;
		thrust::complex<data_t> q0,q1;
		thrust::complex<double> r0,r1;

		i1 = i & mask;
		i0 = (i - i1) << 1;
		i0 += i1;
		i1 = i0 + add;

		q0 = pVec[i0];
		q1 = pVec[i1];

		r0 = m0 * q0 + m2 * q1;
		r1 = m1 * q0 + m3 * q1;
		return (r0.real()*r0.real() + r0.imag()*r0.imag() + r1.real()*r1.real() + r1.imag()*r1.imag());
	}
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm(const uint_t qubit, const cvector_t<double> &mat) const
{
	uint_t size = data_size_ >> 1;
	return apply_sum_lambda(norm_matMult2x2_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(thrust::complex<double>*)&mat[0],qubit), size);
}


template <typename data_t>
struct norm_diagMult2x2_lambda : public unary_function<uint_t,double>
{
	thrust::complex<data_t>* pVec;
	thrust::complex<double> m0,m1;
	int qubit;

	norm_diagMult2x2_lambda(thrust::complex<data_t>* pV,thrust::complex<double>* pMat,int q)
	{
		pVec = pV;
		qubit = q;
		m0 = pMat[0];
		m1 = pMat[1];
	}

	__host__ __device__ double operator()(const uint_t &i) const
	{
		thrust::complex<data_t> q;
		thrust::complex<double> m,r;

		q = pVec[i];
		if(((i >> qubit) & 1) == 0){
			m = m0;
		}
		else{
			m = m1;
		}

		r = m * q;

		return (r.real()*r.real() + r.imag()*r.imag());
	}
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const
{
	uint_t size = data_size_;
	
	return apply_sum_lambda(norm_diagMult2x2_lambda<data_t>((thrust::complex<data_t>*)&data_[0],(thrust::complex<double>*)&mat[0],qubit), size);
}


/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/
template <typename data_t>
double QubitVectorThrust<data_t>::probability(const uint_t outcome) const {
  return std::real(data_[outcome] * std::conj(data_[outcome]));
}

template <typename data_t>
std::vector<double> QubitVectorThrust<data_t>::probabilities() const {
  const int_t END = 1LL << num_qubits();
  std::vector<double> probs(END, 0.);
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < END; j++) {
    probs[j] = probability(j);
  }
  return probs;
}

template <typename data_t>
struct dot_lambda : public unary_function<uint_t,double>
{
	thrust::complex<data_t>* pVec;
	uint64_t mask;

	dot_lambda(thrust::complex<data_t>* pV,int q)
	{
		pVec = pV;
		mask = (1ull << q);
	}

	__host__ __device__ double operator()(const uint_t &i) const
	{
		thrust::complex<data_t> q0;
		double ret;

		ret = 0.0;

		if((i & mask) == 0){
			q0 = pVec[i];
			ret = q0.real()*q0.real() + q0.imag()*q0.imag();
		}
		return ret;
	}
};

template <typename data_t>
std::vector<double> QubitVectorThrust<data_t>::probabilities(const reg_t &qubits) const {

	const size_t N = qubits.size();
	std::vector<double> probs((1ull << N), 0.);

	if(N == 1){
		probs[0] = apply_sum_lambda(dot_lambda<data_t>((thrust::complex<data_t>*)&data_[0],qubits[0]), data_size_);
		probs[1] = 1.0 - probs[0];

#ifdef QASM_DEBUG
//		printf(" prob[%d] : (%e, %e) \n",qubits[0],probs[0],probs[1]);
#endif
	}
	return probs;
}

//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
reg_t QubitVectorThrust<data_t>::sample_measure(const std::vector<double> &rnds) const
{
	const int_t SHOTS = rnds.size();
	reg_t samples;
	data_t* pVec = (data_t*)&data_[0];
	uint_t n = data_size_*2;
	int i;

#ifdef QASM_TIMING
	TimeStart(QS_GATE_MEASURE);
#endif

	samples.assign(SHOTS, 0);

	if(m_nDevParallel == 1){
		thrust::device_vector<double> vRnd(SHOTS);
		thrust::device_vector<unsigned long> vSamp(SHOTS);
		thrust::host_vector<double> hvRnd(SHOTS);
		thrust::host_vector<unsigned long> hvSamp(SHOTS);

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
	}
	else{
		int iDev;
		double* pDevSum = new double[m_nDevParallel];

#pragma omp parallel for private(i)
		for(iDev=0;iDev<m_nDevParallel;iDev++){
			uint_t is,ie;
			is = n * iDev / m_nDevParallel;
			ie = n * (iDev+1) / m_nDevParallel;

			cudaSetDevice(iDev);

			thrust::transform_inclusive_scan(thrust::device,pVec + is,pVec+ie,pVec+is,thrust::square<double>(),thrust::plus<double>());

			pDevSum[iDev] = pVec[ie-1];
		}

#pragma omp parallel for private(i)
		for(iDev=0;iDev<m_nDevParallel;iDev++){
			uint_t is,ie;
			double low,high;
			is = n * iDev / m_nDevParallel;
			ie = n * (iDev+1) / m_nDevParallel;

			cudaSetDevice(iDev);

			thrust::device_vector<double> vRnd(SHOTS);
			thrust::device_vector<unsigned long> vSamp(SHOTS);
			thrust::host_vector<double> hvRnd(SHOTS);
			thrust::host_vector<unsigned long> hvSamp(SHOTS);

			low = 0.0;
			for(i=0;i<iDev;i++){
				low += pDevSum[i];
			}
			high = low + pDevSum[iDev];

			for(i=0;i<SHOTS;i++){
				if(rnds[i] < low || rnds[i] >= high){
					hvRnd[i] = 10.0;
				}
				else{
					hvRnd[i] = rnds[i] - low;
				}
			}
			vRnd = hvRnd;

			thrust::lower_bound(thrust::device, pVec + is, pVec + ie, vRnd.begin(), vRnd.end(), vSamp.begin());

			hvSamp = vSamp;
			for(i=0;i<SHOTS;i++){
				if(hvSamp[i] < ie-is){
					samples[i] = (is + hvSamp[i])/2;
				}
			}
		}

		delete[] pDevSum;
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
