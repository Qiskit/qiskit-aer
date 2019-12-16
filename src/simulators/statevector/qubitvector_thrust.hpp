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

#ifdef AER_THRUST_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <thrust/for_each.h>
#include <thrust/complex.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/adjacent_difference.h>

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

#ifdef DEBUG
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
#endif //DEBUG

#define AER_DEFAULT_MATRIX_BITS		8


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
	void apply_function(UnaryFunction func,const reg_t &qubits);

  	template <typename UnaryFunction>
	double apply_sum_function(UnaryFunction func,const reg_t &qubits) const;

	void allocate_buffers(int qubit);

	int m_iDev;
	int m_nDev;
	int m_nDevParallel;
	int m_useATS;
	int m_useDevMem;

	thrust::complex<double>* m_pMatDev;		//matrix buffer for device
	uint_t* m_pUintBuf;						//buffer to store parameters
	thrust::complex<data_t>** m_ppBuffer;	//pointer to buffers
	int m_matBits;							//max number of fusion bits
	uint_t m_matSize;

#ifdef DEBUG
	mutable uint_t m_gateCounts[QS_NUM_GATES];
	mutable double m_gateTime[QS_NUM_GATES];
	mutable double m_gateStartTime[QS_NUM_GATES];

	void TimeStart(int i) const;
	void TimeEnd(int i) const;
	void TimeReset(void);
	void TimePrint(void);

	//for debugging
	mutable FILE* debug_fp;
	mutable uint_t debug_count;

	void DebugMsg(const char* str,const reg_t &qubits) const;
	void DebugMsg(const char* str,const int qubit) const;
	void DebugMsg(const char* str) const;
	void DebugMsg(const char* str,const std::complex<double> c) const;
	void DebugMsg(const char* str,const double d) const;
	void DebugMsg(const char* str,const std::vector<double>& v) const;
	void DebugDump(void);
#endif

};


//base class of gate functions
class GateFuncBase
{
public:
	virtual bool IsDiagonal(void)
	{
		return false;
	}
	virtual int NumControlBits(void)
	{
		return 0;
	}
	virtual int ControlMask(void)
	{
		return 1;
	}
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
QubitVectorThrust<data_t>::QubitVectorThrust(size_t num_qubits)
{
	m_pMatDev = NULL;
	m_matBits = 0;
	m_useATS = 0;
	data_ = NULL;
	checkpoint_ = NULL;

	set_num_qubits(num_qubits);

#ifdef DEBUG
	debug_fp = NULL;
	debug_count = 0;
#endif
}

template <typename data_t>
QubitVectorThrust<data_t>::QubitVectorThrust()
{
	m_pMatDev = NULL;
	m_matBits = 0;
	m_useATS = 0;
	data_ = NULL;
	checkpoint_ = NULL;

#ifdef DEBUG
	debug_fp = NULL;
	debug_count = 0;
#endif
}

template <typename data_t>
QubitVectorThrust<data_t>::~QubitVectorThrust() {
#ifdef DEBUG
	TimePrint();
#endif

	if (data_){
		if(m_useATS){
			free(data_);
		}
		else{
#ifdef AER_THRUST_CUDA
			cudaFree(data_);
#else
			free(data_);
#endif
		}
	}
	if(m_matBits > 0){
#ifdef AER_THRUST_CUDA
		cudaFree(m_pMatDev);
		cudaFree(m_pUintBuf);
		cudaFree(m_ppBuffer);
#else
		free(m_pMatDev);
		free(m_pUintBuf);
		free(m_ppBuffer);
#endif
	}

  if (checkpoint_)
    free(checkpoint_);

#ifdef DEBUG
	if(debug_fp != NULL){
		fflush(debug_fp);
		fclose(debug_fp);
	}
#endif
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
#ifdef DEBUG
	DebugMsg("vector");
#endif
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
template <typename data_t>
class initialize_component_func : public GateFuncBase
{
protected:
	thrust::complex<double>* state;
	uint_t* qubits;
	int nqubits;
	uint_t matSize;
public:

	initialize_component_func(thrust::complex<double>* pS,uint_t* pBuf,const reg_t &qb)
	{
		uint_t k;
		nqubits = qb.size();
		matSize = 1ull << nqubits;
		qubits = pBuf;
		state = pS;

		for(k=0;k<matSize;k++){
			qubits[k] = qb[k];
		}
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		thrust::complex<data_t>** ppChunk;
		thrust::complex<double> q0;
		thrust::complex<double> q;
		uint_t i,j,k;
		uint_t ii,idx,t;
		uint_t mask;

		i = thrust::get<0>(iter);
		ppChunk = thrust::get<1>(iter);

		idx = 0;
		ii = i;
		for(j=0;j<nqubits;j++){
			mask = (1ull << qubits[j]) - 1;

			t = ii & mask;
			idx += t;
			ii = (ii - t) << 1;
		}
		idx += ii;

		q0 = ppChunk[0][idx];
		for(k=0;k<matSize;k++){
			q = q0 * state[k];
			ppChunk[k][idx] = q;
		}
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::initialize_component(const reg_t &qubits, const cvector_t<double> &state0)
{
	const size_t N = qubits.size();
	thrust::complex<double>* pMat;

#ifdef AER_HAS_ATS
	pMat = (thrust::complex<double>*)&state0;
#else
	int_t i,matSize;
	matSize = 1ull << N;

	allocate_buffers(N);

	pMat = m_pMatDev;
#pragma omp parallel for
	for(i=0;i<matSize;i++){
		m_pMatDev[i] = state0[i];
	}
#endif

	auto qubits_sorted = qubits;
	std::sort(qubits_sorted.begin(), qubits_sorted.end());

	apply_function(initialize_component_func<data_t>(pMat,m_pUintBuf,qubits_sorted), qubits);

#ifdef DEBUG
	DebugMsg("initialize_component",qubits);
	DebugDump();
#endif

}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
class fill_func : public GateFuncBase
{
protected:
	thrust::complex<data_t> val;
public:

	fill_func(thrust::complex<data_t>& v)
	{
		val = v;
	}

	bool IsDiagonal(void)
	{
		return true;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i;
		thrust::complex<data_t>** ppChunk;
		thrust::complex<data_t>* pV;

		i = thrust::get<0>(iter);
		ppChunk = thrust::get<1>(iter);
		pV = ppChunk[0];

		pV[i] = val;
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::zero()
{
	thrust::complex<data_t> z = 0.0;

	reg_t qubits = {0};
	apply_function(fill_func<data_t>(z), qubits);
}

template <typename data_t>
void QubitVectorThrust<data_t>::set_num_qubits(size_t num_qubits)
{
	size_t prev_num_qubits = num_qubits_;
	num_qubits_ = num_qubits;
	data_size_ = 1ull << num_qubits;

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
#ifdef AER_THRUST_CUDA
				cudaFree(data_);
#else
				free(data_);
#endif
			}
			data_ = nullptr;
		}
	}

	int tid,nid;
	char* str;

	nid = omp_get_num_threads();
	tid = omp_get_thread_num();
	m_nDev = 1;
#ifdef AER_THRUST_CUDA
	cudaGetDeviceCount(&m_nDev);
#endif

	m_iDev = 0;
	if(nid > 1){
		m_iDev = tid % m_nDev;
#ifdef AER_THRUST_CUDA
		cudaSetDevice(m_iDev);
#endif
		m_nDevParallel = 1;
	}
	else{
		m_nDevParallel = 1;
		str = getenv("AER_MULTI_GPU");
		if(str != NULL){
			m_nDevParallel = m_nDev;
		}

#ifndef AER_THRUST_CUDA
#pragma omp parallel private(nid)
		{
			nid = omp_get_num_threads();
#pragma omp master
			{
				m_nDevParallel = nid;
			}
		}
#endif
	}

	// Allocate memory for new vector
	if (data_ == nullptr){
		void* pData;

#ifdef DEBUG
		TimeReset();
		TimeStart(QS_GATE_INIT);
#endif

#ifdef AER_THRUST_CUDA
	#ifdef __linux__
			str = getenv("AER_USE_ATS");
			if(str != NULL){
				posix_memalign(&pData,128,sizeof(thrust::complex<data_t>) * data_size_);
				m_useATS = 1;
			}
			else{
	#endif
				str = getenv("AER_USE_DEVMEM");
				if(str != NULL){
					cudaMalloc(&pData,sizeof(thrust::complex<data_t>) * data_size_);
					m_useDevMem = 1;
				}
				else{
					cudaMallocManaged(&pData,sizeof(thrust::complex<data_t>) * data_size_);
				}
				m_useATS = 0;
	#ifdef __linux__
			}
	#endif
#else	//AER_THRUST_CUDA
		pData = (thrust::complex<data_t>*)malloc(sizeof(thrust::complex<data_t>) * data_size_);
#endif	//AER_THRUST_CUDA

		data_ = reinterpret_cast<std::complex<data_t>*>(pData);

#ifdef DEBUG
		TimeEnd(QS_GATE_INIT);
#endif
	}

#ifdef DEBUG
	//TODO Migrate to SpdLog
	if(debug_fp == NULL && tid == 0){
		char filename[1024];
		sprintf(filename,"logs/debug_%d.txt",getpid());
		debug_fp = fopen(filename,"a");

		fprintf(debug_fp," ==== Thrust qubit vector initialization %d qubits ==== tt\n",num_qubits_);
		fprintf(debug_fp,"    TEST : threads %d/%d , dev %d/%d, using %d devices\n",tid,nid,m_iDev,m_nDev,m_nDevParallel);
	}
#endif
	allocate_buffers(AER_DEFAULT_MATRIX_BITS);
}

template <typename data_t>
void QubitVectorThrust<data_t>::allocate_buffers(int nq)
{
	uint_t matSize;

	if(nq > m_matBits){
		matSize = 1ull << nq;
		m_matSize = matSize;
		if(m_matBits > 0){
#ifdef AER_THRUST_CUDA
			cudaFree(m_pMatDev);
			cudaFree(m_ppBuffer);
			cudaFree(m_pUintBuf);
#else
			free(m_pMatDev);
			free(m_ppBuffer);
			free(m_pUintBuf);
#endif
		}
		m_matBits = nq;
#ifdef AER_THRUST_CUDA
		cudaMallocManaged(&m_pMatDev,sizeof(thrust::complex<double>) * matSize*matSize);
		cudaMallocManaged(&m_ppBuffer,sizeof(thrust::complex<data_t>*) * matSize);
		cudaMallocManaged(&m_pUintBuf,sizeof(uint_t) * matSize * 4);
#else
		m_pMatDev = (thrust::complex<double>*)malloc(sizeof(thrust::complex<double>) * matSize*matSize);
		m_ppBuffer =(thrust::complex<data_t>**)malloc(sizeof(thrust::complex<data_t>) * matSize);
		m_pUintBuf = (uint_t*)malloc(sizeof(uint_t) * matSize * 4);
#endif
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
#ifdef DEBUG
	DebugMsg("checkpoint");
	DebugDump();
#endif

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
#ifdef DEBUG
	DebugMsg("revert");
	DebugDump();
#endif
}

template <typename data_t>
std::complex<double> QubitVectorThrust<data_t>::inner_product() const
{
	int_t i;
	double d = 0.0;

#pragma omp parallel for reduction(+:d)
	for(i=0;i<data_size_;i++){
		d += std::real(data_[i]) * std::real(checkpoint_[i]) + std::imag(data_[i]) * std::imag(checkpoint_[i]);
	}
#ifdef DEBUG
	DebugMsg("inner_product",std::complex<double>(d,0.0));
#endif

	return std::complex<double>(d,0.0);
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorThrust<data_t>::initialize()
{
#ifdef DEBUG
	DebugMsg("initialize");
#endif
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

#ifdef DEBUG
	DebugMsg("initialize_from_vector");
	DebugDump();
#endif
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


#ifdef DEBUG
	DebugMsg("initialize_from_data");
	DebugDump();
#endif
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
class MatrixMult2x2 : public GateFuncBase
{
protected:
	thrust::complex<double> m0,m1,m2,m3;
	int qubit;
	uint_t mask;

public:
	MatrixMult2x2(thrust::complex<double>* pMat,int q)
	{
		qubit = q;
		m0 = pMat[0];
		m1 = pMat[1];
		m2 = pMat[2];
		m3 = pMat[3];

		mask = (1ull << qubit) - 1;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,i0,i1;
		thrust::complex<data_t>** ppV;
		thrust::complex<data_t> q0,q1;
		thrust::complex<data_t>* pV0;
		thrust::complex<data_t>* pV1;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);
		pV0 = ppV[0];
		pV1 = ppV[1];

		i1 = i & mask;
		i0 = (i - i1) << 1;
		i0 += i1;

		q0 = pV0[i0];
		q1 = pV1[i0];

		pV0[i0] = m0 * q0 + m2 * q1;
		pV1[i0] = m1 * q0 + m3 * q1;
	}
};


template <typename data_t>
class MatrixMult4x4 : public GateFuncBase
{
protected:
	thrust::complex<double> m00,m10,m20,m30;
	thrust::complex<double> m01,m11,m21,m31;
	thrust::complex<double> m02,m12,m22,m32;
	thrust::complex<double> m03,m13,m23,m33;
	int qubit0;
	int qubit1;
	uint_t mask0;
	uint_t mask1;

public:
	MatrixMult4x4(thrust::complex<double>* pMat,int q0,int q1)
	{
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

		mask0 = (1ull << qubit0) - 1;
		mask1 = (1ull << qubit1) - 1;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,i0,i1,i2;
		thrust::complex<data_t>** ppV;
		thrust::complex<data_t> q0,q1,q2,q3;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

		i0 = i & mask0;
		i2 = (i - i0) << 1;
		i1 = i2 & mask1;
		i2 = (i2 - i1) << 1;

		i0 = i0 + i1 + i2;

		q0 = ppV[0][i0];
		q1 = ppV[1][i0];
		q2 = ppV[2][i0];
		q3 = ppV[3][i0];

		ppV[0][i0] = m00 * q0 + m10 * q1 + m20 * q2 + m30 * q3;

		ppV[1][i0] = m01 * q0 + m11 * q1 + m21 * q2 + m31 * q3;

		ppV[2][i0] = m02 * q0 + m12 * q1 + m22 * q2 + m32 * q3;

		ppV[3][i0] = m03 * q0 + m13 * q1 + m23 * q2 + m33 * q3;
	}
};

template <typename data_t>
class MatrixMult8x8 : public GateFuncBase
{
protected:
	thrust::complex<double>* pMat;
	int qubit0;
	int qubit1;
	int qubit2;
	uint_t mask0;
	uint_t mask1;
	uint_t mask2;

public:
	MatrixMult8x8(thrust::complex<double>* pM,int q0,int q1,int q2)
	{
		qubit0 = q0;
		qubit1 = q1;
		qubit2 = q2;

		pMat = pM;

		mask0 = (1ull << qubit0) - 1;
		mask1 = (1ull << qubit1) - 1;
		mask2 = (1ull << qubit2) - 1;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,i0,i1,i2,i3;
		thrust::complex<data_t>** ppV;
		thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
		thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

		i0 = i & mask0;
		i3 = (i - i0) << 1;
		i1 = i3 & mask1;
		i3 = (i3 - i1) << 1;
		i2 = i3 & mask2;
		i3 = (i3 - i2) << 1;

		i0 = i0 + i1 + i2 + i3;

		q0 = ppV[0][i0];
		q1 = ppV[1][i0];
		q2 = ppV[2][i0];
		q3 = ppV[3][i0];
		q4 = ppV[4][i0];
		q5 = ppV[5][i0];
		q6 = ppV[6][i0];
		q7 = ppV[7][i0];

		m0 = pMat[0];
		m1 = pMat[8];
		m2 = pMat[16];
		m3 = pMat[24];
		m4 = pMat[32];
		m5 = pMat[40];
		m6 = pMat[48];
		m7 = pMat[56];

		ppV[0][i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[1];
		m1 = pMat[9];
		m2 = pMat[17];
		m3 = pMat[25];
		m4 = pMat[33];
		m5 = pMat[41];
		m6 = pMat[49];
		m7 = pMat[57];

		ppV[1][i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[2];
		m1 = pMat[10];
		m2 = pMat[18];
		m3 = pMat[26];
		m4 = pMat[34];
		m5 = pMat[42];
		m6 = pMat[50];
		m7 = pMat[58];

		ppV[2][i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[3];
		m1 = pMat[11];
		m2 = pMat[19];
		m3 = pMat[27];
		m4 = pMat[35];
		m5 = pMat[43];
		m6 = pMat[51];
		m7 = pMat[59];

		ppV[3][i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[4];
		m1 = pMat[12];
		m2 = pMat[20];
		m3 = pMat[28];
		m4 = pMat[36];
		m5 = pMat[44];
		m6 = pMat[52];
		m7 = pMat[60];

		ppV[4][i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[5];
		m1 = pMat[13];
		m2 = pMat[21];
		m3 = pMat[29];
		m4 = pMat[37];
		m5 = pMat[45];
		m6 = pMat[53];
		m7 = pMat[61];

		ppV[5][i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[6];
		m1 = pMat[14];
		m2 = pMat[22];
		m3 = pMat[30];
		m4 = pMat[38];
		m5 = pMat[46];
		m6 = pMat[54];
		m7 = pMat[62];

		ppV[6][i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

		m0 = pMat[7];
		m1 = pMat[15];
		m2 = pMat[23];
		m3 = pMat[31];
		m4 = pMat[39];
		m5 = pMat[47];
		m6 = pMat[55];
		m7 = pMat[63];

		ppV[7][i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;
	}
};

template <typename data_t>
class MatrixMult16x16 : public GateFuncBase
{
protected:
	thrust::complex<double>* pMat;
	int qubit0;
	int qubit1;
	int qubit2;
	int qubit3;
	uint_t mask0;
	uint_t mask1;
	uint_t mask2;
	uint_t mask3;
public:
	MatrixMult16x16(thrust::complex<double>* pM,int q0,int q1,int q2,int q3)
	{
		qubit0 = q0;
		qubit1 = q1;
		qubit2 = q2;
		qubit3 = q3;

		pMat = pM;

		mask0 = (1ull << qubit0) - 1;
		mask1 = (1ull << qubit1) - 1;
		mask2 = (1ull << qubit2) - 1;
		mask3 = (1ull << qubit3) - 1;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,i0,i1,i2,i3,i4;
		thrust::complex<data_t>** ppV;
		thrust::complex<data_t> q0,q1,q2,q3,q4,q5,q6,q7;
		thrust::complex<data_t> q8,q9,q10,q11,q12,q13,q14,q15;
		thrust::complex<double> m0,m1,m2,m3,m4,m5,m6,m7;
		thrust::complex<double> m8,m9,m10,m11,m12,m13,m14,m15;
		int j;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

		i0 = i & mask0;
		i4 = (i - i0) << 1;
		i1 = i4 & mask1;
		i4 = (i4 - i1) << 1;
		i2 = i4 & mask2;
		i4 = (i4 - i2) << 1;
		i3 = i4 & mask3;
		i4 = (i4 - i3) << 1;

		i0 = i0 + i1 + i2 + i3 + i4;

		q0 = ppV[0][i0];
		q1 = ppV[1][i0];
		q2 = ppV[2][i0];
		q3 = ppV[3][i0];
		q4 = ppV[4][i0];
		q5 = ppV[5][i0];
		q6 = ppV[6][i0];
		q7 = ppV[7][i0];
		q8 = ppV[8][i0];
		q9 = ppV[9][i0];
		q10 = ppV[10][i0];
		q11 = ppV[11][i0];
		q12 = ppV[12][i0];
		q13 = ppV[13][i0];
		q14 = ppV[14][i0];
		q15 = ppV[15][i0];

		for(j=0;j<16;j++){
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

			ppV[j][i0] = 	m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7 +
							m8 * q8 + m9 * q9 + m10* q10+ m11* q11+ m12* q12+ m13* q13+ m14* q14+ m15* q15;
		}
	}
};


//in-place NxN matrix multiplication using LU factorization
template <typename data_t>
class MatrixMultNxN_LU : public GateFuncBase
{
protected:
	thrust::complex<double>* pMat;
	uint_t* qubits;
	uint_t* pivot;
	uint_t* table;
	int nqubits;
	uint_t matSize;
	int nswap;
public:
	MatrixMultNxN_LU(thrust::complex<double>* pM,uint_t* pBuf,const reg_t &qb)
	{
		uint_t i,j,k,imax;
		thrust::complex<double> c0,c1;
		double d,dmax;
		uint_t* pSwap;

		nqubits = qb.size();
		pMat = pM;
		matSize = 1ull << nqubits;
		qubits = pBuf;
		pivot = pBuf + matSize;
		table = pBuf + matSize*2;

		for(k=0;k<matSize;k++){
			qubits[k] = qb[k];
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

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		thrust::complex<data_t> q,qt;
		thrust::complex<double> m;
		thrust::complex<double> r;
		uint_t i,j,k,l;
		uint_t ii,idx,t;
		uint_t mask;
		thrust::complex<data_t>** ppV;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

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
				q = ppV[k][idx];

				r += m*q;
			}
			ppV[j][idx] = r;
		}

		//mult L
		for(j=matSize-1;j>0;j--){
			r = ppV[j][idx];

			for(k=0;k<j;k++){
				l = (pivot[j] + (k << nqubits));
				m = pMat[l];
				q = ppV[k][idx];

				r += m*q;
			}
			ppV[j][idx] = r;
		}

		//swap results
		if(nswap > 0){
			q = ppV[table[0]][idx];
			k = pivot[table[0]];
			for(j=1;j<nswap;j++){
				qt = ppV[table[j]][idx];
				ppV[k][idx] = q;
				q = qt;
				k = pivot[table[j]];
			}
			ppV[k][idx] = q;
		}
	}
};


template <typename data_t>
template <typename UnaryFunction>
void QubitVectorThrust<data_t>::apply_function(UnaryFunction func,const reg_t &qubits)
{
	const size_t N = qubits.size();
	uint_t size,add;
	int nBuf;
	int i,j,ncb;

	if(func.IsDiagonal()){
		size = data_size_;
		nBuf = 1;
		m_ppBuffer[0] = (thrust::complex<data_t>*)&data_[0];
	}else{
		ncb = func.NumControlBits();
		size = data_size_ >> (N - ncb);
		nBuf = 1ull << (N - ncb);

		for(i=0;i<nBuf;i++){
			m_ppBuffer[i] = (thrust::complex<data_t>*)&data_[0];
		}
		for(j=ncb;j<N;j++){
			add = (1ull << qubits[j]);
			for(i=0;i<nBuf;i++){
				if((i >> (j-ncb)) & 1){
					m_ppBuffer[i] += add;
				}
			}
		}
	}

	auto ci = thrust::counting_iterator<uint_t>(0);
	thrust::constant_iterator<thrust::complex<data_t>**> cc(m_ppBuffer);
	auto chunkTuple = thrust::make_tuple(ci,cc);
	auto chunkIter = thrust::make_zip_iterator(chunkTuple);

	if(m_nDevParallel == 1){
		thrust::for_each(thrust::device, chunkIter, chunkIter + size, func);
	}else{
		int iDev;

		#pragma omp parallel for
		for(iDev=0;iDev<m_nDevParallel;iDev++){
			uint_t is,ie;
			is = size * iDev / m_nDevParallel;
			ie = size * (iDev+1) / m_nDevParallel;

#ifdef AER_THRUST_CUDA
			cudaSetDevice(iDev);
#endif
			thrust::for_each(thrust::device, chunkIter + is, chunkIter + ie, func);
		}
	}
}

template <typename data_t>
template <typename UnaryFunction>
double QubitVectorThrust<data_t>::apply_sum_function(UnaryFunction func,const reg_t &qubits) const
{
	double ret = 0.0;

	const size_t N = qubits.size();
	uint_t size,add;
	int nBuf;
	int i,j,ncb;

	if(func.IsDiagonal()){
		size = data_size_;
		nBuf = 1;
		m_ppBuffer[0] = (thrust::complex<data_t>*)&data_[0];
	}else{
		ncb = func.NumControlBits();
		size = data_size_ >> (N - ncb);
		nBuf = 1ull << (N - ncb);

		for(i=0;i<nBuf;i++){
			m_ppBuffer[i] = (thrust::complex<data_t>*)&data_[0];
		}
		for(j=ncb;j<N;j++){
			add = (1ull << qubits[j]);
			for(i=0;i<nBuf;i++){
				if((i >> (j-ncb)) & 1){
					m_ppBuffer[i] += add;
				}
			}
		}
	}

	auto ci = thrust::counting_iterator<uint_t>(0);
	thrust::constant_iterator<thrust::complex<data_t>**> cc(m_ppBuffer);
	auto chunkTuple = thrust::make_tuple(ci,cc);
	auto chunkIter = thrust::make_zip_iterator(chunkTuple);

	if(m_nDevParallel == 1){
		ret = thrust::transform_reduce(thrust::device, chunkIter, chunkIter + size, func,0.0,thrust::plus<double>());
	}
	else{
		int iDev;

#pragma omp parallel for reduction(+:ret)
		for(iDev=0;iDev<m_nDevParallel;iDev++){
			uint_t is,ie;
			is = size * iDev / m_nDevParallel;
			ie = size * (iDev+1) / m_nDevParallel;

#ifdef AER_THRUST_CUDA
			cudaSetDevice(iDev);
#endif
			ret += thrust::transform_reduce(thrust::device, chunkIter + is, chunkIter + ie, func,0.0,thrust::plus<double>());
		}
	}

	return ret;
}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t<double> &mat)
{
	const size_t N = qubits.size();
	auto qubits_sorted = qubits;
	std::sort(qubits_sorted.begin(), qubits_sorted.end());

#ifdef DEBUG
	TimeStart(QS_GATE_MULT);
#endif
	if(N == 1){
		apply_function(MatrixMult2x2<data_t>((thrust::complex<double>*)&mat[0],qubits_sorted[0]), qubits);
	}
	else if(N == 2){
		apply_function(MatrixMult4x4<data_t>((thrust::complex<double>*)&mat[0],qubits_sorted[0],qubits_sorted[1]), qubits);
	}
	else{
		thrust::complex<double>* pMat;

		int_t i,matSize;
		matSize = 1ull << N;

		allocate_buffers(N);

		pMat = m_pMatDev;

		#pragma omp parallel for
		for(i=0;i<matSize*matSize;i++){
			m_pMatDev[i] = mat[i];
		}

		if(N == 3){
			apply_function(MatrixMult8x8<data_t>(pMat,qubits_sorted[0],qubits_sorted[1],qubits_sorted[2]), qubits);
		}
		else if(N == 4){
			apply_function(MatrixMult16x16<data_t>(pMat,qubits_sorted[0],qubits_sorted[1],qubits_sorted[2],qubits_sorted[3]), qubits);
		}
		else{
			apply_function(MatrixMultNxN_LU<data_t>(pMat,m_pUintBuf,qubits_sorted), qubits);
		}
	}

#ifdef DEBUG
	TimeEnd(QS_GATE_MULT);
#endif

#ifdef DEBUG
	DebugMsg("apply_matrix",qubits);
	DebugDump();
#endif
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


#ifdef DEBUG
	DebugMsg("apply_multiplexer",control_qubits);
	DebugMsg("                 ",target_qubits);
#endif

	apply_matrix(qubits,matMP);
}

template <typename data_t>
class DiagonalMult2x2 : public GateFuncBase
{
protected:
	thrust::complex<double> m0,m1;
	int qubit;
public:

	DiagonalMult2x2(thrust::complex<double>* pMat,int q)
	{
		qubit = q;
		m0 = pMat[0];
		m1 = pMat[1];
	}

	bool IsDiagonal(void)
	{
		return true;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i;
		thrust::complex<data_t>** ppChunk;
		thrust::complex<data_t> q;
		thrust::complex<data_t>* pV;
		thrust::complex<double> m;

		i = thrust::get<0>(iter);
		ppChunk = thrust::get<1>(iter);
		pV = ppChunk[0];

		q = pV[i];
		if(((i >> qubit) & 1) == 0){
			m = m0;
		}
		else{
			m = m1;
		}

		pV[i] = m * q;
	}
};

template <typename data_t>
class DiagonalMultNxN : public GateFuncBase
{
protected:
	thrust::complex<double>* pMat;
	int nqubits;
	uint_t* qubits;

public:
	DiagonalMultNxN(thrust::complex<double>* pM,uint_t* pBuf,const reg_t &qb)
	{
		int i;
		pMat = pM;
		qubits = pBuf;
		nqubits = qb.size();
		for(i=0;i<nqubits;i++){
			qubits[i] = qb[i];
		}
	}

	bool IsDiagonal(void)
	{
		return true;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,j,im;
		thrust::complex<data_t>** ppChunk;
		thrust::complex<data_t> q;
		thrust::complex<data_t>* pV;
		thrust::complex<double> m;

		i = thrust::get<0>(iter);
		ppChunk = thrust::get<1>(iter);
		pV = ppChunk[0];

		im = 0;
		for(j=0;j<nqubits;j++){
			if((i & (1ull << qubits[j])) != 0){
				im += (1 << j);
			}
		}

		q = pV[i];
		m = pMat[im];

		pV[i] = m * q;
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t<double> &diag)
{
	const int_t N = qubits.size();

#ifdef DEBUG
	TimeStart(QS_GATE_DIAG);
#endif
	if(N == 1){
		apply_function(DiagonalMult2x2<data_t>((thrust::complex<double>*)&diag[0],qubits[0]), qubits);
	}else{
		thrust::complex<double>* pMat;

#ifdef AER_HAS_ATS
		pMat = (thrust::complex<double>*)&diag[0];
#else

		int_t i,matSize;
		matSize = 1ull << N;

		allocate_buffers(N);

#pragma omp parallel for
		for(i=0;i<matSize;i++){
			m_pMatDev[i] = diag[i];
		}
		pMat = m_pMatDev;
#endif

		apply_function(DiagonalMultNxN<data_t>(pMat,m_pUintBuf,qubits), qubits);
	}

#ifdef DEBUG
	TimeEnd(QS_GATE_DIAG);
	DebugMsg("apply_diagonal_matrix",qubits);
	DebugDump();
#endif

}


template <typename data_t>
class Permutation : public GateFuncBase
{
protected:
	uint_t* pairs;
	uint_t* qubits;
	uint_t matSize;
	int nqubits;
	int npairs;

public:
	Permutation(uint_t* pBuf,const reg_t& qb,const std::vector<std::pair<uint_t, uint_t>> &pairs_in)
	{
		uint_t j;

		nqubits = qb.size();
		qubits = pBuf;
		pairs = pBuf + nqubits;
		matSize = 1ull << nqubits;
		npairs = pairs_in.size();

		for(j=0;j<matSize;j++){
			qubits[j] = qb[j];
		}
		for(j=0;j<npairs;j++){
			pairs[j*2  ] = pairs_in[j].first;
			pairs[j*2+1] = pairs_in[j].second;
		}
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i;
		thrust::complex<data_t>** ppV;
		thrust::complex<data_t> q;
		uint_t j,ip0,ip1;
		uint_t ii,idx,t;
		uint_t mask;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

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
			ip0 = pairs[j*2];
			ip1 = pairs[j*2+1];
			q = ppV[ip0][idx];

			ppV[ip0][idx] = ppV[ip1][idx];
			ppV[ip1][idx] = q;
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

	apply_function(Permutation<data_t>(m_pUintBuf,qubits,pairs), qubits);

#ifdef DEBUG
	DebugMsg("apply_permutation_matrix",qubits);
	DebugDump();
#endif

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
class CX_func : public GateFuncBase
{
protected:
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
		mask = (1ull << qubit_t) - 1;

		cmask = 0;
		for(i=0;i<nqubits-1;i++){
			cmask |= (1ull << qubits[i]);
		}
	}

	int NumControlBits(void)
	{
		return nqubits - 1;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,i0,i1;
		thrust::complex<data_t>** ppChunk;
		thrust::complex<data_t> q0,q1;
		thrust::complex<data_t>* pV0;
		thrust::complex<data_t>* pV1;

		i = thrust::get<0>(iter);
		ppChunk = thrust::get<1>(iter);
		pV0 = ppChunk[0];
		pV1 = ppChunk[1];

		i1 = i & mask;
		i0 = (i - i1) << 1;
		i0 += i1;

		if(((i0) & cmask) == cmask){
			q0 = pV0[i0];
			q1 = pV1[i0];

			pV0[i0] = q1;
			pV1[i0] = q0;
		}
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcx(const reg_t &qubits)
{

#ifdef DEBUG
	TimeStart(QS_GATE_CX);
#endif

	apply_function(CX_func<data_t>(qubits), qubits);

#ifdef DEBUG
	TimeEnd(QS_GATE_CX);
	DebugMsg("apply_mcx",qubits);
	DebugDump();
#endif

}


template <typename data_t>
class CY_func : public GateFuncBase
{
protected:
	uint_t mask;
	uint_t cmask;
	int nqubits;
	int qubit_t;
public:
	CY_func(const reg_t &qubits)
	{
		int i;
		nqubits = qubits.size();

		qubit_t = qubits[nqubits-1];
		mask = (1ull << qubit_t) - 1;

		cmask = 0;
		for(i=0;i<nqubits-1;i++){
			cmask |= (1ull << qubits[i]);
		}
	}

	int NumControlBits(void)
	{
		return nqubits - 1;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,i0,i1;
		thrust::complex<data_t>** ppChunk;
		thrust::complex<data_t> q0,q1;
		thrust::complex<data_t>* pV0;
		thrust::complex<data_t>* pV1;

		i = thrust::get<0>(iter);
		ppChunk = thrust::get<1>(iter);
		pV0 = ppChunk[0];
		pV1 = ppChunk[1];

		i1 = i & mask;
		i0 = (i - i1) << 1;
		i0 += i1;

		if(((i0) & cmask) == cmask){
			q0 = pV0[i0];
			q1 = pV1[i0];

			pV0[i0] = thrust::complex<data_t>(q1.imag(),-q1.real());
			pV1[i0] = thrust::complex<data_t>(-q0.imag(),q0.real());
		}
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcy(const reg_t &qubits)
{

	apply_function(CY_func<data_t>(qubits), qubits);

#ifdef DEBUG
	DebugMsg("apply_mcy",qubits);
	DebugDump();
#endif

}

template <typename data_t>
class CSwap_func : public GateFuncBase
{
protected:
	uint_t mask0;
	uint_t mask1;
	uint_t cmask;
	int nqubits;
	int qubit_t0;
	int qubit_t1;
public:

	CSwap_func(const reg_t &qubits)
	{
		int i;
		nqubits = qubits.size();

		qubit_t0 = qubits[nqubits-2];
		qubit_t1 = qubits[nqubits-1];
		mask0 = (1ull << qubit_t0) - 1;
		mask1 = (1ull << qubit_t1) - 1;

		cmask = 0;
		for(i=0;i<nqubits-2;i++){
			cmask |= (1ull << qubits[i]);
		}
	}

	int NumControlBits(void)
	{
		return nqubits - 2;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,i0,i1,i2;
		thrust::complex<data_t>** ppChunk;
		thrust::complex<data_t> q1,q2;
		thrust::complex<data_t>* pV1;
		thrust::complex<data_t>* pV2;

		i = thrust::get<0>(iter);
		ppChunk = thrust::get<1>(iter);
		pV1 = ppChunk[1];
		pV2 = ppChunk[2];

		i0 = i & mask0;
		i2 = (i - i0) << 1;
		i1 = i2 & mask1;
		i2 = (i2 - i1) << 1;

		i0 = i0 + i1 + i2;

		if(((i0) & cmask) == cmask){
			q1 = pV1[i0];
			q2 = pV2[i0];
			pV1[i0] = q2;
			pV2[i0] = q1;
		}
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcswap(const reg_t &qubits)
{

	apply_function(CSwap_func<data_t>(qubits), qubits);

#ifdef DEBUG
	DebugMsg("apply_mcswap",qubits);
	DebugDump();
#endif

}

template <typename data_t>
class phase_func : public GateFuncBase
{
protected:
	thrust::complex<double> phase;
	uint_t mask;
	int nqubits;
public:
	phase_func(const reg_t &qubits,const std::complex<double> p)
	{
		int i;
		nqubits = qubits.size();
		phase = p;

		mask = 0;
		for(i=0;i<nqubits;i++){
			mask |= (1ull << qubits[i]);
		}
	}
	int NumControlBits(void)
	{
		return nqubits - 1;
	}

	bool IsDiagonal(void)
	{
		return true;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i;
		thrust::complex<data_t>** ppV;
		thrust::complex<data_t> q0;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

		if(((i) & mask) == mask){
			q0 = ppV[0][i];
			ppV[0][i] = q0 * phase;
		}
	}
};

template <typename data_t>
void QubitVectorThrust<data_t>::apply_mcphase(const reg_t &qubits, const std::complex<double> phase)
{
	apply_function(phase_func<data_t>(qubits,phase), qubits );

#ifdef DEBUG
	DebugMsg("apply_mcphase",qubits);
	DebugDump();
#endif

}

template <typename data_t>
class DiagonalMult2x2Controlled : public GateFuncBase
{
protected:
	thrust::complex<double> m0,m1;
	uint_t mask;
	uint_t cmask;
	int nqubits;
public:
	DiagonalMult2x2Controlled(thrust::complex<double>* pMat,const reg_t &qubits)
	{
		int i;
		nqubits = qubits.size();

		m0 = pMat[0];
		m1 = pMat[1];

		mask = (1ull << qubits[nqubits-1]) - 1;
		cmask = 0;
		for(i=0;i<nqubits-1;i++){
			cmask |= (1ull << qubits[i]);
		}
	}

	int NumControlBits(void)
	{
		return nqubits - 1;
	}

	bool IsDiagonal(void)
	{
		return true;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i;
		thrust::complex<data_t>** ppV;
		thrust::complex<data_t> q0;
		thrust::complex<double> m;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

		if(((i) & cmask) == cmask){
			if((i) & mask){
				m = m1;
			}else{
				m = m0;
			}

			q0 = ppV[0][i];
			ppV[0][i] = m*q0;
		}
	}
};

template <typename data_t>
class MatrixMult2x2Controlled : public GateFuncBase
{
protected:
	thrust::complex<double> m0,m1,m2,m3;
	uint_t mask;
	uint_t cmask;
	int nqubits;
public:
	MatrixMult2x2Controlled(thrust::complex<double>* pMat,const reg_t &qubits)
	{
		int i;
		m0 = pMat[0];
		m1 = pMat[1];
		m2 = pMat[2];
		m3 = pMat[3];
		nqubits = qubits.size();

		mask = (1ull << qubits[nqubits-1]) - 1;
		cmask = 0;
		for(i=0;i<nqubits-1;i++){
			cmask |= (1ull << qubits[i]);
		}
	}

	int NumControlBits(void)
	{
		return nqubits - 1;
	}

	__host__ __device__ void operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,i0,i1;
		thrust::complex<data_t>** ppChunk;
		thrust::complex<data_t> q0,q1;
		thrust::complex<data_t>* pV0;
		thrust::complex<data_t>* pV1;

		i = thrust::get<0>(iter);
		ppChunk = thrust::get<1>(iter);
		pV0 = ppChunk[0];
		pV1 = ppChunk[1];

		i1 = i & mask;
		i0 = (i - i1) << 1;
		i0 += i1;

		if(((i0) & cmask) == cmask){
			q0 = pV0[i0];
			q1 = pV1[i0];

			pV0[i0]  = m0 * q0 + m2 * q1;
			pV1[i0] = m1 * q0 + m3 * q1;
		}
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
			apply_function(DiagonalMult2x2Controlled<data_t>((thrust::complex<double>*)&diag[0],qubits), qubits );
		}
	}
	else{
		if(N == 1){
			// If N=1 this is just a single-qubit matrix
			apply_matrix(qubits[0], mat);
			return;
		}
		else{
			apply_function(MatrixMult2x2Controlled<data_t>((thrust::complex<double>*)&mat[0],qubits), qubits );
		}
	}

#ifdef DEBUG
	DebugMsg("apply_mcu",qubits);
	DebugDump();
#endif
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
#ifdef DEBUG
	TimeStart(QS_GATE_DIAG);
#endif
  	const std::vector<std::complex<double>> diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix(qubit, diag);

#ifdef DEBUG
	TimeEnd(QS_GATE_DIAG);
#endif
  	return;
  }
#ifdef DEBUG
	TimeStart(QS_GATE_MULT);
#endif

	reg_t qubits = {qubit};
	apply_function(MatrixMult2x2<data_t>((thrust::complex<double>*)&mat[0],qubit), qubits);


#ifdef DEBUG
	TimeEnd(QS_GATE_MULT);
	DebugMsg("apply_matrix",(int)qubit);
	DebugDump();
#endif

}

template <typename data_t>
void QubitVectorThrust<data_t>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t<double>& diag)
{
#ifdef DEBUG
	TimeStart(QS_GATE_DIAG);
#endif
	reg_t qubits = {qubit};
	apply_function(DiagonalMult2x2<data_t>((thrust::complex<double>*)&diag[0],qubits[0]), qubits);

#ifdef DEBUG
	TimeEnd(QS_GATE_DIAG);
	DebugMsg("apply_diagonal_matrix",(int)qubit);
	DebugDump();
#endif

}
/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <typename data_t>
class Norm : public GateFuncBase
{
protected:

public:
	Norm()
	{}

	bool IsDiagonal(void){
		return true;
	}

	__host__ __device__ double operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i;
		thrust::complex<data_t>* pV;
		thrust::complex<data_t> q0;
		double ret;

		i = thrust::get<0>(iter);
		pV = thrust::get<1>(iter)[0];

		ret = 0.0;

		q0 = pV[i];
		ret = q0.real()*q0.real() + q0.imag()*q0.imag();

		return ret;
	}
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm() const
{
	reg_t qubits = {0};
	double ret;
	ret = apply_sum_function(Norm<data_t>(),qubits);

#ifdef DEBUG
	DebugMsg("norm",ret);
#endif
	return ret;
}

template <typename data_t>
class NormMatrixMultNxN : public GateFuncBase
{
protected:
	thrust::complex<double>* pMat;
	uint_t* offset;
	uint_t* qubits;
	int nqubits;
	uint_t matSize;
public:
	NormMatrixMultNxN(thrust::complex<double>* pM,uint_t* pBuf,const reg_t &qb)
	{
		uint_t k;

		nqubits = qb.size();
		pMat = pM;
		matSize = 1ull << nqubits;
		qubits = pBuf;

		for(k=0;k<matSize;k++){
			qubits[k] = qb[k];
		}
	}

	__host__ __device__ double operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i;
		thrust::complex<data_t>** ppV;

		thrust::complex<data_t> q;
		thrust::complex<double> m;
		thrust::complex<double> r;
		double sum = 0.0;
		uint_t j,k,l;
		uint_t ii,idx,t;
		uint_t mask;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

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
				q = ppV[k][idx];
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
	}else{
		thrust::complex<double>* pMat;
		double ret;

		int_t i,matSize;
		matSize = 1ull << N;

		pMat = m_pMatDev;

		#pragma omp parallel for
		for(i=0;i<matSize*matSize;i++){
			m_pMatDev[i] = mat[i];
		}

		ret = apply_sum_function(NormMatrixMultNxN<data_t>(pMat,m_pUintBuf,qubits), qubits);

#ifdef DEBUG
		DebugMsg("norm",qubits);
		DebugMsg("    ",ret);
#endif
		return ret;
	}
}

template <typename data_t>
class NormDiagonalMultNxN : public GateFuncBase
{
protected:
	thrust::complex<double>* pMat;
	int nqubits;
	uint_t* qubits;
public:
	NormDiagonalMultNxN(thrust::complex<double>* pM,uint_t* pBuf,const reg_t &qb)
	{
		int i;
		pMat = pM;
		qubits = pBuf;
		nqubits = qb.size();
		for(i=0;i<nqubits;i++){
			qubits[i] = qb[i];
		}
	}

	bool IsDiagonal(void)
	{
		return true;
	}

	__host__ __device__ double operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,im,j;
		thrust::complex<data_t> q;
		thrust::complex<double> m,r;

		thrust::complex<data_t>** ppV;
		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

		im = 0;
		for(j=0;j<nqubits;j++){
			if(((i) & (1ull << qubits[j])) != 0){
				im += (1 << j);
			}
		}

		q = ppV[0][i];
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
	}else{
		thrust::complex<double>* pMat;
		double ret;

#ifdef AER_HAS_ATS
		pMat = (thrust::complex<double>*)&mat[0];
#else

		uint_t i,matSize;
		matSize = 1ull << N;

		#pragma omp parallel for
		for(i=0;i<matSize;i++){
			m_pMatDev[i] = mat[i];
		}
		pMat = m_pMatDev;
#endif
		ret = apply_sum_function(NormDiagonalMultNxN<data_t>(pMat,m_pUintBuf,qubits), qubits );
#ifdef DEBUG
		DebugMsg("norm_diagonal",qubits);
		DebugMsg("             ",ret);
#endif
		return ret;
	}
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
class NormMatrixMult2x2 : public GateFuncBase
{
protected:
	thrust::complex<double> m0,m1,m2,m3;
	int qubit;
	uint_t mask;
public:
	NormMatrixMult2x2(thrust::complex<double>* pMat,int q)
	{
		qubit = q;
		m0 = pMat[0];
		m1 = pMat[1];
		m2 = pMat[2];
		m3 = pMat[3];

		mask = (1ull << qubit) - 1;
	}

	__host__ __device__ double operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i,i0,i1;
		thrust::complex<data_t>** ppV;
		thrust::complex<data_t> q0,q1;
		thrust::complex<double> r0,r1;
		double sum = 0.0;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

		i1 = i & mask;
		i0 = (i - i1) << 1;
		i0 += i1;

		q0 = ppV[0][i0];
		q1 = ppV[1][i0];

		r0 = m0 * q0 + m2 * q1;
		sum += r0.real()*r0.real() + r0.imag()*r0.imag();
		r1 = m1 * q0 + m3 * q1;
		sum += r1.real()*r1.real() + r1.imag()*r1.imag();
		return sum;
	}
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm(const uint_t qubit, const cvector_t<double> &mat) const
{
	reg_t qubits = {qubit};
	double ret;

	ret = apply_sum_function(NormMatrixMult2x2<data_t>((thrust::complex<double>*)&mat[0],qubit), qubits);

#ifdef DEBUG
		DebugMsg("norm2x2",qubits);
		DebugMsg("       ",ret);
#endif
	return ret;
}


template <typename data_t>
class NormDiagonalMult2x2 : public GateFuncBase
{
protected:
	thrust::complex<double> m0,m1;
	int qubit;
public:
	NormDiagonalMult2x2(thrust::complex<double>* pMat,int q)
	{
		qubit = q;
		m0 = pMat[0];
		m1 = pMat[1];
	}

	bool IsDiagonal(void)
	{
		return true;
	}

	__host__ __device__ double operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i;
		thrust::complex<data_t>** ppV;
		thrust::complex<data_t> q;
		thrust::complex<double> m,r;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

		q = ppV[0][i];
		if((((i) >> qubit) & 1) == 0){
			m = m0;
		}else{
			m = m1;
		}

		r = m * q;

		return (r.real()*r.real() + r.imag()*r.imag());
	}
};

template <typename data_t>
double QubitVectorThrust<data_t>::norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const
{
	reg_t qubits = {qubit};
	double ret;

	ret = apply_sum_function(NormDiagonalMult2x2<data_t>((thrust::complex<double>*)&mat[0],qubit), qubits);

#ifdef DEBUG
		DebugMsg("norm_diagonal",qubits);
		DebugMsg("             ",ret);
#endif
	return ret;
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

#ifdef DEBUG
	DebugMsg("probabilities",probs);
#endif
	return probs;
}


template <typename data_t>
class dot_func : public GateFuncBase
{
protected:
	uint64_t mask;
	uint64_t cmask;
public:
	dot_func(const reg_t &qubits,const reg_t &qubits_sorted,int i)
	{
		int k;
		int nq = qubits.size();

		mask = 0;
		cmask = 0;
		for(k=0;k<nq;k++){
			mask |= (1ull << qubits_sorted[k]);

			if(((i >> k) & 1) != 0){
				cmask |= (1ull << qubits[k]);
			}
		}
	}

	bool IsDiagonal(void)
	{
		return true;
	}

	__host__ __device__ double operator()(const thrust::tuple<uint_t,thrust::complex<data_t>**> &iter) const
	{
		uint_t i;
		thrust::complex<data_t> q;
		thrust::complex<data_t>** ppV;
		double ret;

		i = thrust::get<0>(iter);
		ppV = thrust::get<1>(iter);

		ret = 0.0;

		if((i & mask) == cmask){
			q = ppV[0][i];
			ret = q.real()*q.real() + q.imag()*q.imag();
		}
		return ret;
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
		probs[i] = apply_sum_function(dot_func<data_t>(qubits,qubits_sorted,i), qubits_sorted);
	}

#ifdef DEBUG
	DebugMsg("probabilities",qubits);
	DebugMsg("             ",probs);
#endif
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
	int i,j;
	double* pRnd;
	uint_t* pSamp;

#ifdef DEBUG
	TimeStart(QS_GATE_MEASURE);
#endif

	samples.assign(SHOTS, 0);

	if(m_nDevParallel == 1){
#ifdef AER_THRUST_CUDA
		cudaMallocManaged(&pRnd,sizeof(double)*SHOTS);
		cudaMallocManaged(&pSamp,sizeof(uint_t)*SHOTS);
#else
		pRnd = (double*)malloc(sizeof(double)*SHOTS);
		pSamp = (uint_t*)malloc(sizeof(uint_t)*SHOTS);
#endif

		thrust::transform_inclusive_scan(thrust::device,pVec,pVec+n,pVec,thrust::square<double>(),thrust::plus<double>());

		#pragma omp parallel for
		for(i=0;i<SHOTS;i++){
			pRnd[i] = rnds[i];
		}

		thrust::lower_bound(thrust::device, pVec, pVec + n, pRnd, pRnd + SHOTS, pSamp);

		#pragma omp parallel for
		for(i=0;i<SHOTS;i++){
			samples[i] = pSamp[i]/2;
		}

		//restore statevector
		thrust::adjacent_difference(thrust::device,pVec,pVec+n,pVec);
		thrust::for_each(thrust::device,pVec,pVec+n,[=] __host__ __device__ (data_t a){return sqrt(a);});

#ifdef AER_THRUST_CUDA
		cudaFree(pRnd);
		cudaFree(pSamp);
#else
		free(pRnd);
		free(pSamp);
#endif
	}
	else{
		int iDev;
		double* pDevSum = new double[m_nDevParallel];

		#pragma omp parallel for private(i)
		for(iDev=0;iDev<m_nDevParallel;iDev++){
			uint_t is,ie;
			is = n * iDev / m_nDevParallel;
			ie = n * (iDev+1) / m_nDevParallel;

#ifdef AER_THRUST_CUDA
			cudaSetDevice(iDev);
#endif
			thrust::transform_inclusive_scan(thrust::device,pVec + is,pVec+ie,pVec+is,thrust::square<double>(),thrust::plus<double>());

			pDevSum[iDev] = pVec[ie-1];
		}

		#pragma omp parallel for private(i,pRnd,pSamp)
		for(iDev=0;iDev<m_nDevParallel;iDev++){
			uint_t is,ie;
			double low,high;
			is = n * iDev / m_nDevParallel;
			ie = n * (iDev+1) / m_nDevParallel;

#ifdef AER_THRUST_CUDA
			cudaSetDevice(iDev);
			cudaMallocManaged(&pRnd,sizeof(double)*SHOTS);
			cudaMallocManaged(&pSamp,sizeof(uint_t)*SHOTS);
#else
			pRnd = (double*)malloc(sizeof(double)*SHOTS);
			pSamp = (uint_t*)malloc(sizeof(uint_t)*SHOTS);
#endif

			low = 0.0;
			for(i=0;i<iDev;i++){
				low += pDevSum[i];
			}
			high = low + pDevSum[iDev];

			for(i=0;i<SHOTS;i++){
				if(rnds[i] < low || rnds[i] >= high){
					pRnd[i] = 10.0;
				}
				else{
					pRnd[i] = rnds[i] - low;
				}
			}

			thrust::lower_bound(thrust::device, pVec + is, pVec + ie, pRnd, pRnd + SHOTS, pSamp);

			for(i=0;i<SHOTS;i++){
				if(pSamp[i] < ie-is){
					samples[i] = (is + pSamp[i])/2;
				}
			}

			//restore statevector
			thrust::adjacent_difference(thrust::device,pVec + is,pVec + ie,pVec + is);
			thrust::for_each(thrust::device,pVec + is,pVec + ie,[=] __host__ __device__ (data_t a){return sqrt(a);});

#ifdef AER_THRUST_CUDA
			cudaFree(pRnd);
			cudaFree(pSamp);
#else
			free(pRnd);
			free(pSamp);
#endif
		}

		delete[] pDevSum;
	}

#ifdef DEBUG
	TimeEnd(QS_GATE_MEASURE);
#endif

#ifdef DEBUG
	DebugMsg("sample_measure",samples);
#endif

	return samples;
}

#ifdef DEBUG
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

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const reg_t &qubits) const
{
	if(debug_fp != NULL){
		fprintf(debug_fp," [%d] %s : %d (",debug_count,str,qubits.size());
		int iq;
		for(iq=0;iq<qubits.size();iq++){
			fprintf(debug_fp," %d ",qubits[iq]);
		}
		fprintf(debug_fp," )\n");
	}
	debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const int qubit) const
{
	if(debug_fp != NULL){
		fprintf(debug_fp," [%d] %s : (%d) \n",debug_count,str,qubit);
	}
	debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str) const
{
	if(debug_fp != NULL){
		fprintf(debug_fp," [%d] %s \n",debug_count,str);
	}
	debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const std::complex<double> c) const
{
	if(debug_fp != NULL){
		fprintf(debug_fp," [%d] %s : %e, %e \n",debug_count,str,std::real(c),imag(c));
	}
	debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const double d) const
{
	if(debug_fp != NULL){
		fprintf(debug_fp," [%d] %s : %e \n",debug_count,str,d);
	}
	debug_count++;
}

template <typename data_t>
void QubitVectorThrust<data_t>::DebugMsg(const char* str,const std::vector<double>& v) const
{
	if(debug_fp != NULL){
		fprintf(debug_fp," [%d] %s : <",debug_count,str);
		int i,n;
		n = v.size();
		for(i=0;i<n;i++){
			fprintf(debug_fp," %e ",v[i]);
		}
		fprintf(debug_fp," >\n");
	}
	debug_count++;

}


template <typename data_t>
void QubitVectorThrust<data_t>::DebugDump(void)
{
	if(debug_fp != NULL){
		uint_t i,j;
		char bin[64];
		for(i=0;i<data_size_;i++){
			for(j=0;j<num_qubits_;j++){
				bin[num_qubits_-j-1] = '0' + (char)((i >> j) & 1);
			}
			bin[num_qubits_] = 0;
			fprintf(debug_fp,"   %s | %e, %e\n",bin,std::real(data_[i]),imag(data_[i]));
		}
	}
}
#endif	//DEBUG

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



