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

#ifndef _qv_density_matrix_thrust_hpp_
#define _qv_density_matrix_thrust_hpp_


#include "framework/utils.hpp"
#include "simulators/unitary/unitarymatrix_thrust.hpp"

namespace QV {

//============================================================================
// DensityMatrixThrust class
//============================================================================

// This class is derived from the UnitaryMatrix class and stores an N-qubit 
// matrix as a 2*N-qubit vector.
// The vector is formed using column-stacking vectorization as under this
// convention left-matrix multiplication on qubit-n is equal to multiplication
// of the vectorized 2*N qubit vector also on qubit-n.

template <typename data_t = double>
class DensityMatrixThrust : public UnitaryMatrixThrust<data_t> {

public:
  // Parent class aliases
  using BaseVector = QubitVectorThrust<data_t>;
  using BaseMatrix = UnitaryMatrixThrust<data_t>;

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  DensityMatrixThrust() : DensityMatrixThrust(0) {};
  explicit DensityMatrixThrust(size_t num_qubits);
  DensityMatrixThrust(const DensityMatrixThrust& obj) {}
  DensityMatrixThrust &operator=(const DensityMatrixThrust& obj) {}

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Return the string name of the class
#ifdef AER_THRUST_CUDA
  static std::string name() {return "density_matrix_gpu";}
#else
  static std::string name() {return "density_matrix_thrust";}
#endif

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // The vector can be either a statevector or a vectorized density matrix
  // If the length of the data vector does not match either case for the
  // number of qubits an exception is raised.
  void initialize_from_vector(const cvector_t<double> &data);

  // Returns the number of qubits for the superoperator
  virtual uint_t num_qubits() const override {return BaseMatrix::num_qubits_;}

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit unitary matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_unitary_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a N-qubit superoperator matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit superop.
  void apply_superop_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a N-qubit diagonal unitary matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_unitary_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  // Apply a N-qubit diagonal superoperator matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_superop_matrix(const reg_t &qubits, const cvector_t<double> &mat);

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
  // Z-measurement outcome probabilities
  //-----------------------------------------------------------------------

  // Return the Z-basis measurement outcome probability P(outcome) for
  // outcome in [0, 2^num_qubits - 1]
  virtual double probability(const uint_t outcome) const override;

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  virtual std::vector<double> probabilities(const reg_t &qubits) const;

  // Return M sampled outcomes for Z-basis measurement of all qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  virtual reg_t sample_measure(const std::vector<double> &rnds) const override;


  //-----------------------------------------------------------------------
  // Expectation Value
  //-----------------------------------------------------------------------

  // These functions return the expectation value <psi|A|psi> for a matrix A.
  // If A is hermitian these will return real values, if A is non-Hermitian
  // they in general will return complex values.

  // Return the expectation value of an N-qubit Pauli matrix.
  // The Pauli is input as a length N string of I,X,Y,Z characters.
  double expval_pauli(const reg_t &qubits, const std::string &pauli) const;

protected:

  // Convert qubit indicies to vectorized-density matrix qubitvector indices
  // For the QubitVector apply matrix function
  virtual reg_t superop_qubits(const reg_t &qubits) const;

  // Construct a vectorized superoperator from a vectorized matrix
  // This is equivalent to vec(tensor(conj(A), A))
  cvector_t<double> vmat2vsuperop(const cvector_t<double> &vmat) const;

  // Qubit threshold for when apply unitary will apply as two matrix multiplications
  // rather than as a 2n-qubit superoperator matrix.
  size_t apply_unitary_threshold_ = 4;
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/


//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t>
DensityMatrixThrust<data_t>::DensityMatrixThrust(size_t num_qubits)
  : UnitaryMatrixThrust<data_t>(num_qubits) {};

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
void DensityMatrixThrust<data_t>::initialize() {
  // Zero the underlying vector
  BaseVector::zero();
  // Set to be all |0> sate
	std::complex<double> one = 1.0;
	BaseVector::set_state(0,one);
}

template <typename data_t>
void DensityMatrixThrust<data_t>::initialize_from_vector(const cvector_t<double> &statevec) {
  if (BaseVector::data_size_ == statevec.size()) {
    // Use base class initialize for already vectorized matrix
    BaseVector::initialize_from_vector(statevec);
  } else if (BaseVector::data_size_ == statevec.size() * statevec.size()) {
    // Convert statevector into density matrix
    cvector_t<double> densitymat = AER::Utils::tensor_product(AER::Utils::conjugate(statevec),
                                                      statevec);
//    std::move(densitymat.begin(), densitymat.end(), BaseVector::data_);
    BaseVector::initialize_from_vector(densitymat);

  } else {
    throw std::runtime_error("DensityMatrixThrust::initialize input vector is incorrect length. Expected: " +
                             std::to_string(BaseVector::data_size_) + " Received: " +
                             std::to_string(statevec.size()));
  }

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::initialize_from_vector");
#endif
}

//------------------------------------------------------------------------------
// Apply matrix functions
//------------------------------------------------------------------------------

template <typename data_t>
reg_t DensityMatrixThrust<data_t>::superop_qubits(const reg_t &qubits) const {
  reg_t superop_qubits = qubits;
  // Number of qubits
  for (const auto q: qubits) {
    superop_qubits.push_back(q + num_qubits());
  }
  return superop_qubits;
}

template <typename data_t>
cvector_t<double> DensityMatrixThrust<data_t>::vmat2vsuperop(const cvector_t<double> &vmat) const {
  // Get dimension of unvectorized matrix
  size_t dim = size_t(std::sqrt(vmat.size()));
  cvector_t<double> ret(dim * dim * dim * dim, 0.);
  for (size_t i=0; i < dim; i++)
    for (size_t j=0; j < dim; j++)
      for (size_t k=0; k < dim; k++)
        for (size_t l=0; l < dim; l++)
          ret[dim*i+k+(dim*dim)*(dim*j+l)] = std::conj(vmat[i+dim*j])*vmat[k+dim*l];
  return ret;
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_superop_matrix(const reg_t &qubits,
                                                 const cvector_t<double> &mat) {
  BaseVector::apply_matrix(superop_qubits(qubits), mat);
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_superop_matrix",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_diagonal_superop_matrix(const reg_t &qubits,
                                                          const cvector_t<double> &diag) {
  BaseVector::apply_diagonal_matrix(superop_qubits(qubits), diag);
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_diagonal_superop_matrix",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_unitary_matrix(const reg_t &qubits,
                                                 const cvector_t<double> &mat) {
  // Check if we apply as two N-qubit matrix multiplications or a single 2N-qubit matrix mult.
  if (qubits.size() > apply_unitary_threshold_) {
    // Apply as two N-qubit matrix mults
    reg_t conj_qubits;
    for (const auto q: qubits) {
      conj_qubits.push_back(q + num_qubits());
    }
    // Apply id \otimes U
    BaseVector::apply_matrix(qubits, mat);
    // Apply conj(U) \otimes id
    BaseVector::apply_matrix(conj_qubits, AER::Utils::conjugate(mat));
  } else {
    // Apply as single 2N-qubit matrix mult.
    apply_superop_matrix(qubits, vmat2vsuperop(mat));
  }
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_unitary_matrix",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_diagonal_unitary_matrix(const reg_t &qubits,
                                                          const cvector_t<double> &diag) {
  // Apply as single 2N-qubit matrix mult.
  apply_diagonal_superop_matrix(qubits, AER::Utils::tensor_product(AER::Utils::conjugate(diag), diag));
}

//-----------------------------------------------------------------------
// Apply Specialized Gates
//-----------------------------------------------------------------------

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_cnot(const uint_t qctrl, const uint_t qtrgt) {
  std::vector<std::pair<uint_t, uint_t>> pairs = {
    {{1, 3}, {4, 12}, {5, 15}, {6, 14}, {7, 13}, {9, 11}}
  };
  const reg_t qubits = {{qctrl, qtrgt, qctrl + num_qubits(), qtrgt + num_qubits()}};
  BaseVector::apply_permutation_matrix(qubits, pairs);
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_cnot",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_cz(const uint_t q0, const uint_t q1) {
  cvector_t<double> vec;
  vec.resize(16, 1.0);
  vec[3] = -1.;
  vec[7] = -1.;
  vec[11] = -1.;
  vec[12] = -1.;
  vec[13] = -1.;
  vec[14] = -1.;

  const reg_t qubits = {{q0, q1, q0 + num_qubits(), q1 + num_qubits()}};
  BaseVector::apply_diagonal_matrix(qubits, vec);

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_cz",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_swap(const uint_t q0, const uint_t q1) {
  std::vector<std::pair<uint_t, uint_t>> pairs = {
   {{1, 2}, {4, 8}, {5, 10}, {6, 9}, {7, 11}, {13, 14}}
  };
  const reg_t qubits = {{q0, q1, q0 + num_qubits(), q1 + num_qubits()}};
  BaseVector::apply_permutation_matrix(qubits, pairs);
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_swap",qubits);
#endif
}

template <typename data_t>
class DensityX : public GateFuncBase<data_t>
{
protected:
  uint_t mask0;
  uint_t mask1;
  uint_t offset0;
  uint_t offset1;

public:
  DensityX(int q0,int q1)
  {
    offset0 = 1ull << q0;
    offset1 = 1ull << q1;
  	if(q0 < q1){
      mask0 = (1ull << q0) - 1;
      mask1 = (1ull << q1) - 1;
  	}
  	else{
      mask0 = (1ull << q1) - 1;
      mask1 = (1ull << q0) - 1;
  	}
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
    vec1 = vec0 + offset0;
    vec2 = vec0 + offset1;
    vec3 = vec2 + offset0;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    q0 = vec0[i0];
    q1 = vec1[i0];
    q2 = vec2[i0];
    q3 = vec3[i0];

    vec0[i0] = q3;
    vec1[i0] = q2;
    vec2[i0] = q1;
    vec3[i0] = q0;
  }
  const char* name(void)
  {
    return "DensityX";
  }
};

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_x(const uint_t qubit) {
  // Use the lambda function
  const reg_t qubits = {{qubit, qubit + num_qubits()}};

	BaseVector::apply_function(DensityX<data_t>(qubits[0], qubits[1]), qubits);

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_x",qubits);
#endif
}

template <typename data_t>
class DensityY : public GateFuncBase<data_t>
{
protected:
  uint_t mask0;
  uint_t mask1;
  uint_t offset0;
  uint_t offset1;

public:
  DensityY(int q0,int q1)
  {
    offset0 = 1ull << q0;
    offset1 = 1ull << q1;
  	if(q0 < q1){
      mask0 = (1ull << q0) - 1;
      mask1 = (1ull << q1) - 1;
  	}
  	else{
      mask0 = (1ull << q1) - 1;
      mask1 = (1ull << q0) - 1;
  	}
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
    vec1 = vec0 + offset0;
    vec2 = vec0 + offset1;
    vec3 = vec2 + offset0;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    q0 = vec0[i0];
    q1 = vec1[i0];
    q2 = vec2[i0];
    q3 = vec3[i0];

    vec0[i0] = q0;
    vec1[i0] = -q2;
    vec2[i0] = -q1;
    vec3[i0] = q3;
  }
  const char* name(void)
  {
    return "DensityY";
  }
};

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_y(const uint_t qubit) 
{
  const reg_t qubits = {{qubit, qubit + num_qubits()}};

	BaseVector::apply_function(DensityY<data_t>(qubits[0], qubits[1]), qubits);

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_y",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_z(const uint_t qubit) {
  cvector_t<double> vec;
  vec.resize(4, 1.);
  vec[1] = -1.;
  vec[2] = -1.;

  // Use the lambda function
  const reg_t qubits = {{qubit, qubit + num_qubits()}};
  BaseVector::apply_diagonal_matrix(qubits, vec);

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_z",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_toffoli(const uint_t qctrl0,
                                          const uint_t qctrl1,
                                          const uint_t qtrgt) {
  std::vector<std::pair<uint_t, uint_t>> pairs = {
    {{3, 7}, {11, 15}, {19, 23}, {24, 56}, {25, 57}, {26, 58}, {27, 63},
    {28, 60}, {29, 61}, {30, 62}, {31, 59}, {35, 39}, {43,47}, {51, 55}}
  };
  const reg_t qubits = {{qctrl0, qctrl1, qtrgt,
                         qctrl0 + num_qubits(), qctrl1 + num_qubits(), qtrgt + num_qubits()}};
  BaseVector::apply_permutation_matrix(qubits, pairs);
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_toffoli",qubits);
#endif

}


//-----------------------------------------------------------------------
// Z-measurement outcome probabilities
//-----------------------------------------------------------------------

template <typename data_t>
double DensityMatrixThrust<data_t>::probability(const uint_t outcome) const {
  const auto shift = BaseMatrix::num_rows() + 1;

	return std::real(BaseVector::get_state(outcome * shift));
}


template <typename data_t>
class density_probability_func : public GateFuncBase<data_t>
{
protected:
  uint_t rows_;
  uint_t mask_;
  uint_t cmask_;
public:
  density_probability_func(const reg_t &qubits,const reg_t &qubits_sorted,int i,uint_t stride)
  {
    int k;
    int nq = qubits.size();

    rows_ = stride;

    mask_ = 0;
    cmask_ = 0;
    for(k=0;k<nq;k++){
      mask_ |= (1ull << qubits_sorted[k]);

      if(((i >> k) & 1) != 0){
        cmask_ |= (1ull << qubits[k]);
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
    uint_t idx;

    vec = this->data_;
    idx = i * (rows_ + 1);

    ret = 0.0;
    if((i & mask_) == cmask_){
      q = vec[idx];
      ret = q.real();
    }
    return ret;
  }
  uint_t size(int num_qubits,int n)
  {
    (void)num_qubits;
    (void)n;
    return rows_;
  }

  const char* name(void)
  {
    return "density probabilities";
  }
};

template <typename data_t>
std::vector<double> DensityMatrixThrust<data_t>::probabilities(const reg_t &qubits) const 
{
  const size_t N = qubits.size();
  const int_t DIM = 1 << N;

  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
  if ((N == num_qubits()) && (qubits == qubits_sorted))
    return BaseVector::probabilities();

  std::vector<double> probs(DIM, 0.);

  int i;
  for(i=0;i<DIM;i++){
    probs[i] = BaseVector::apply_function_sum(density_probability_func<data_t>(qubits,qubits_sorted,i,BaseMatrix::num_rows()), qubits);
  }

  return probs;
}

template <typename data_t>
class DensityMatrixDiagonalReal
{
protected:
  thrust::complex<data_t>* data_;
  data_t* diag_;
  uint_t rows_;
public:
  DensityMatrixDiagonalReal(thrust::complex<data_t>* src,data_t* dest,uint_t stride)
  {
    data_ = src;
    diag_ = dest;
    rows_ = stride;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t idx;
    thrust::complex<data_t> q;

    idx = i * (rows_ + 1);

    q = data_[idx];
    diag_[i] = q.real();
  }
};

template <typename data_t>
reg_t DensityMatrixThrust<data_t>::sample_measure(const std::vector<double> &rnds) const {

  const int_t SHOTS = rnds.size();
  uint_t nrows = BaseMatrix::num_rows();
  reg_t samples;
  samples.assign(SHOTS, 0);

  if(BaseVector::chunk_->device() >= 0){
    BaseVector::chunk_->set_device();

    //buffer to store diagonal elements
    thrust::device_vector<data_t> diag_vec(nrows);

    auto ci = thrust::counting_iterator<uint_t>(0);
    thrust::for_each_n(thrust::device, ci ,nrows , 
                DensityMatrixDiagonalReal<data_t>(BaseVector::chunk_->pointer(),(data_t*)thrust::raw_pointer_cast(diag_vec.data()),nrows));

    thrust::inclusive_scan(thrust::device,diag_vec.begin(),diag_vec.end(),diag_vec.begin());

#ifdef AER_THRUST_CUDA
    thrust::device_vector<double> vRnd_dev(SHOTS);
    thrust::device_vector<uint_t> vSmp_dev(SHOTS);

    vRnd_dev = rnds;

    thrust::lower_bound(thrust::device, diag_vec.begin(), diag_vec.end(), vRnd_dev.begin(), vRnd_dev.end(), vSmp_dev.begin());

    thrust::copy_n(vSmp_dev.begin(),SHOTS,samples.begin());

    vRnd_dev.clear();
    vSmp_dev.clear();
#else
    thrust::lower_bound(thrust::device, diag_vec.begin(), diag_vec.end(), rnds.begin(), rnds.end(), samples.begin());
#endif

    diag_vec.clear();
  }
  else{
    //buffer to store diagonal elements
    thrust::host_vector<data_t> diag_vec(nrows);

    auto ci = thrust::counting_iterator<uint_t>(0);
    if((BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1 && omp_get_num_threads() == 1)){
      thrust::for_each_n(thrust::omp::par, ci ,nrows , 
                  DensityMatrixDiagonalReal<data_t>(BaseVector::chunk_->pointer(),(data_t*)thrust::raw_pointer_cast(diag_vec.data()),nrows));

      thrust::inclusive_scan(thrust::omp::par,diag_vec.begin(),diag_vec.end(),diag_vec.begin());
      thrust::lower_bound(thrust::omp::par, diag_vec.begin(), diag_vec.end(), rnds.begin(), rnds.end(), samples.begin());
    }
    else{
      thrust::for_each_n(thrust::host, ci ,nrows , 
                  DensityMatrixDiagonalReal<data_t>(BaseVector::chunk_->pointer(),(data_t*)thrust::raw_pointer_cast(diag_vec.data()),nrows));

      thrust::inclusive_scan(thrust::host,diag_vec.begin(),diag_vec.end(),diag_vec.begin());
      thrust::lower_bound(thrust::host, diag_vec.begin(), diag_vec.end(), rnds.begin(), rnds.end(), samples.begin());
    }

    diag_vec.clear();
  }

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::sample_measure",samples);
#endif

  return samples;
}

//-----------------------------------------------------------------------
// Pauli expectation value
//-----------------------------------------------------------------------

template <typename data_t>
class density_expval_pauli_func : public GateFuncBase<data_t>
{
protected:
  int num_qubits_;
  uint_t x_mask_;
  uint_t z_mask_;
  thrust::complex<data_t> phase_;
public:
  density_expval_pauli_func(int nq,uint_t x,uint_t z,thrust::complex<data_t> p)
  {
    num_qubits_ = nq;
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;
  }

  bool IsDiagonal(void)
  {
    return true;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    double ret = 0.0;

    vec = this->data_;

    q0 = vec[(i ^ x_mask_) + (i << num_qubits_)];
    q0 = q0 * phase_;
    ret = q0.real();

    if(z_mask_ != 0){
      //count bits (__builtin_popcountll can not be used on GPU)
      uint_t count = i & z_mask_;
      count = (count & 0x5555555555555555) + ((count >> 1) & 0x5555555555555555);
      count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
      count = (count & 0x0f0f0f0f0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f0f0f0f0f);
      count = (count & 0x00ff00ff00ff00ff) + ((count >> 8) & 0x00ff00ff00ff00ff);
      count = (count & 0x0000ffff0000ffff) + ((count >> 16) & 0x0000ffff0000ffff);
      count = (count & 0x00000000ffffffff) + ((count >> 32) & 0x00000000ffffffff);
      if(count & 1)
        ret = -ret;
    }

    return ret;
  }
};


template <typename data_t>
double DensityMatrixThrust<data_t>::expval_pauli(const reg_t &qubits,
                                                 const std::string &pauli) const 
{
  // Break string up into Z and X
  // With Y being both Z and X (plus a phase)
  const size_t N = qubits.size();
  uint_t x_mask = 0;
  uint_t z_mask = 0;
  uint_t num_y = 0;
  for (size_t i = 0; i < N; ++i) {
    const auto bit = BITS[qubits[i]];
    switch (pauli[N - 1 - i]) {
      case 'I':
        break;
      case 'X': {
        x_mask += bit;
        break;
      }
      case 'Z': {
        z_mask += bit;
        break;
      }
      case 'Y': {
        x_mask += bit;
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
    return std::real(BaseMatrix::trace());
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  thrust::complex<data_t> phase(1, 0);
  switch (num_y & 3) {
    case 0:
      // phase = 1
      break;
    case 1:
      // phase = -1j
      phase = thrust::complex<data_t>(0, -1);
      break;
    case 2:
      // phase = -1
      phase = thrust::complex<data_t>(-1, 0);
      break;
    case 3:
      // phase = 1j
      phase = thrust::complex<data_t>(0, 1);
      break;
  }
  return BaseVector::chunk_->ExecuteSum(density_expval_pauli_func<data_t>(num_qubits(),x_mask,z_mask,phase),BaseVector::data_size_);
}


//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const QV::DensityMatrixThrust<data_t>&m) {
  out << m.matrix();
  return out;
}

//------------------------------------------------------------------------------
#endif // end module

