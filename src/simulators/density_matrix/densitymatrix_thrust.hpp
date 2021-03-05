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

namespace AER {
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

  // Convert qubit indicies to vectorized-density matrix qubitvector indices
  // For the QubitVector apply matrix function
  virtual reg_t superop_qubits(const reg_t &qubits) const;

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

  // Apply 2-qubit controlled-phase gate
  void apply_cphase(const uint_t q0, const uint_t q1, const complex_t &phase);

  // Apply a 2-qubit SWAP gate to the state vector
  void apply_swap(const uint_t q0, const uint_t q1);

  // Apply a single-qubit Pauli-X gate to the state vector
  void apply_x(const uint_t qubit);

  // Apply a single-qubit Pauli-Y gate to the state vector
  void apply_y(const uint_t qubit);

  // Apply 1-qubit phase gate
  void apply_phase(const uint_t q, const complex_t &phase);

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
  virtual std::vector<double> probabilities(const reg_t &qubits) const override;

  // Return M sampled outcomes for Z-basis measurement of all qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  virtual reg_t sample_measure(const std::vector<double> &rnds) const override;

  //-----------------------------------------------------------------------
  // Expectation Values
  //-----------------------------------------------------------------------

  // Return the expectation value of an N-qubit Pauli matrix.
  // The Pauli is input as a length N string of I,X,Y,Z characters.
  double expval_pauli(const reg_t &qubits, const std::string &pauli,const complex_t initial_phase=1.0) const;

protected:
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
  std::complex<data_t> one = 1.0;
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
  const auto nq = num_qubits();
  for (const auto &q: qubits) {
    superop_qubits.push_back(q + nq);
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
class DensityDiagMatMult2x2 : public GateFuncBase<data_t>
{
protected:
  uint_t offset;
  uint_t offset_sp;
  thrust::complex<double> m0,m1;
public:
  DensityDiagMatMult2x2(const cvector_t<double>& mat,uint_t q,uint_t qs)
  {
    offset = 1ull << q;
    offset_sp = 1ull << (q + qs);

    m0 = mat[0];
    m1 = mat[1];
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
    thrust::complex<data_t> q0,q1,q2,q3,q0t,q1t,q2t,q3t;

    vec0 = this->data_;
    vec1 = vec0 + offset;
    vec2 = vec0 + offset_sp;
    vec3 = vec2 + offset;

    i0 = i & (offset - 1);
    i2 = (i - i0) << 1;
    i1 = i2 & (offset_sp - 1);
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    q0 = vec0[i0];
    q1 = vec1[i0];
    q2 = vec2[i0];
    q3 = vec3[i0];

    q0t = m0 * q0;
    q1t = m1 * q1;

    q2t = m0 * q2;
    q3t = m1 * q3;

    vec0[i0] = thrust::conj(m0) * q0t;
    vec2[i0] = thrust::conj(m1) * q2t;

    vec1[i0] = thrust::conj(m0) * q1t;
    vec3[i0] = thrust::conj(m1) * q3t;
  }
  const char* name(void)
  {
    return "DensityDiagMatMult2x2";
  }
};

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_unitary_matrix(const reg_t &qubits,
                                                 const cvector_t<double> &mat) 
{
  // Apply as two N-qubit matrix mults
  reg_t conj_qubits;
  for (const auto q: qubits) {
    conj_qubits.push_back(q + num_qubits());
  }
  // Apply id \otimes U
  BaseVector::apply_matrix(qubits, mat);
  // Apply conj(U) \otimes id
  BaseVector::apply_matrix(conj_qubits, AER::Utils::conjugate(mat));

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_unitary_matrix",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_diagonal_unitary_matrix(const reg_t &qubits,
                                                          const cvector_t<double> &diag) 
{
  if(qubits.size() == 1){
    const reg_t qubits_sp = {{qubits[0], qubits[0] + num_qubits()}};
    BaseVector::apply_function(DensityDiagMatMult2x2<data_t>(diag,qubits[0], num_qubits()));
  }
  else{
    // Apply as single 2N-qubit matrix mult.
    apply_diagonal_superop_matrix(qubits, AER::Utils::tensor_product(AER::Utils::conjugate(diag), diag));
  }
}

//-----------------------------------------------------------------------
// Apply Specialized Gates
//-----------------------------------------------------------------------
template <typename data_t>
class DensityCX : public GateFuncBase<data_t>
{
protected:
  uint_t offset;
  uint_t offset_sp;
  uint_t cmask;
  uint_t cmask_sp;
public:
  DensityCX(uint_t qc,uint_t qt,uint_t qs)
  {
    offset = 1ull << qt;
    offset_sp = 1ull << (qt + qs);
    cmask = 1ull << qc;
    cmask_sp = 1ull << (qc + qs);
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
    thrust::complex<data_t> q0,q1,q2,q3,t;

    vec0 = this->data_;
    vec1 = vec0 + offset;
    vec2 = vec0 + offset_sp;
    vec3 = vec2 + offset;

    i0 = i & (offset - 1);
    i2 = (i - i0) << 1;
    i1 = i2 & (offset_sp - 1);
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    q0 = vec0[i0];
    q1 = vec1[i0];
    q2 = vec2[i0];
    q3 = vec3[i0];

    if((i0 & cmask) == cmask){
      t = q0;
      q0 = q1;
      q1 = t;

      t = q2;
      q2 = q3;
      q3 = t;
    }

    if((i0 & cmask_sp) == cmask_sp){
      vec0[i0] = q2;
      vec1[i0] = q3;
      vec2[i0] = q0;
      vec3[i0] = q1;
    }
    else{
      vec0[i0] = q0;
      vec1[i0] = q1;
      vec2[i0] = q2;
      vec3[i0] = q3;
    }
  }
  const char* name(void)
  {
    return "DensityCX";
  }
};

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_cnot(const uint_t qctrl, const uint_t qtrgt) 
{
  BaseVector::apply_function(DensityCX<data_t>(qctrl, qtrgt, num_qubits()));

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_cnot");
#endif
}

template <typename data_t>
class DensityPhase : public GateFuncBase<data_t>
{
protected:
  uint_t offset;
  uint_t offset_sp;
  thrust::complex<double> phase_;
public:
  DensityPhase(uint_t qt,uint_t qs,thrust::complex<double>* phase)
  {
    offset = 1ull << qt;
    offset_sp = 1ull << (qt + qs);
    phase_ = *phase;
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
    vec1 = vec0 + offset;
    vec2 = vec0 + offset_sp;
    vec3 = vec2 + offset;

    i0 = i & (offset - 1);
    i2 = (i - i0) << 1;
    i1 = i2 & (offset_sp - 1);
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    q1 = vec1[i0];
    vec1[i0] = phase_*q1;

    q2 = vec2[i0];
    vec2[i0] = thrust::conj(phase_)*q2;
  }
  const char* name(void)
  {
    return "DensityPhase";
  }
};

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_phase(const uint_t q,const complex_t &phase) 
{
  BaseVector::apply_function(DensityPhase<data_t>(q, num_qubits(), (thrust::complex<double>*)&phase ));

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_phase");
#endif
}

template <typename data_t>
class DensityCPhase : public GateFuncBase<data_t>
{
protected:
  uint_t offset;
  uint_t offset_sp;
  uint_t cmask;
  uint_t cmask_sp;
  thrust::complex<double> phase_;
public:
  DensityCPhase(uint_t qc,uint_t qt,uint_t qs,std::complex<double> phase)
  {
    offset = 1ull << qt;
    offset_sp = 1ull << (qt + qs);
    cmask = 1ull << qc;
    cmask_sp = 1ull << (qc + qs);
    phase_ = phase;
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
    vec1 = vec0 + offset;
    vec2 = vec0 + offset_sp;
    vec3 = vec2 + offset;

    i0 = i & (offset - 1);
    i2 = (i - i0) << 1;
    i1 = i2 & (offset_sp - 1);
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    q3 = vec3[i0];
    if((i0 & cmask) == cmask){
      q1 = vec1[i0];
      vec1[i0] = phase_*q1;

      q3 = phase_*q3;
    }
    if((i0 & cmask_sp) == cmask_sp){
      q2 = vec2[i0];
      vec2[i0] = thrust::conj(phase_)*q2;

      q3 = thrust::conj(phase_)*q3;
    }
    vec3[i0] = q3;
  }
  const char* name(void)
  {
    return "DensityCPhase";
  }
};

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_cphase(const uint_t q0, const uint_t q1,
                                         const complex_t &phase) 
{
  BaseVector::apply_function(DensityCPhase<data_t>(q0, q1, num_qubits(), phase ));

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_cphase");
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

	BaseVector::apply_function(DensityX<data_t>(qubits[0], qubits[1]) );

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
    vec1[i0] = -q2;
    vec2[i0] = -q1;
    vec3[i0] = q0;
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

	BaseVector::apply_function(DensityY<data_t>(qubits[0], qubits[1]) );

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_y",qubits);
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
// Expectation Values
//-----------------------------------------------------------------------

//special case Z only
template <typename data_t>
class expval_pauli_Z_func_dm : public GateFuncBase<data_t>
{
protected:
  uint_t z_mask_;
  uint_t diag_stride_;
public:
  expval_pauli_Z_func_dm(uint_t z,uint_t stride)
  {
    z_mask_ = z;
    diag_stride_ = 1 + stride;
  }

  bool is_diagonal(void)
  {
    return true;
  }
  uint_t size(int num_qubits)
  {
    return diag_stride_ - 1;
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    double ret = 0.0;

    vec = this->data_;
    q0 = vec[i * diag_stride_];
    ret = q0.real();

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
class expval_pauli_XYZ_func_dm : public GateFuncBase<data_t>
{
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  uint_t mask_l_;
  uint_t mask_u_;
  thrust::complex<data_t> phase_;
  uint_t rows_;
public:
  expval_pauli_XYZ_func_dm(uint_t x,uint_t z,uint_t x_max,std::complex<data_t> p,uint_t stride)
  {
    rows_ = stride;
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;

    mask_u_ = ~((1ull << (x_max+1)) - 1);
    mask_l_ = (1ull << x_max) - 1;
  }

  uint_t size(int num_qubits)
  {
    return (rows_ >> 1);
  }

  __host__ __device__ double operator()(const uint_t &i) const
  {
    thrust::complex<data_t>* vec;
    thrust::complex<data_t> q0;
    double ret = 0.0;
    uint_t idx_vec, idx_mat;

    vec = this->data_;

    idx_vec = ((i << 1) & mask_u_) | (i & mask_l_);
    idx_mat = idx_vec ^ x_mask_ + rows_ * idx_vec;

    q0 = vec[idx_mat];
    q0 = 2 * phase_ * q0;
    ret = q0.real();
    if(z_mask_ != 0){
      if(pop_count_kernel(idx_vec & z_mask_) & 1)
        ret = -ret;
    }
    return ret;
  }
  const char* name(void)
  {
    return "expval_pauli_XYZ";
  }
};

template <typename data_t>
double DensityMatrixThrust<data_t>::expval_pauli(const reg_t &qubits,
                                                 const std::string &pauli,const complex_t initial_phase) const 
{
  uint_t x_mask, z_mask, num_y, x_max;
  std::tie(x_mask, z_mask, num_y, x_max) = pauli_masks_and_phase(qubits, pauli);

  // Special case for only I Paulis
  if (x_mask + z_mask == 0) {
    return BaseMatrix::trace().real();
  }

  // specialize x_max == 0
  if(x_mask == 0) {
    return BaseVector::apply_function_sum(
      expval_pauli_Z_func_dm<data_t>(z_mask, BaseMatrix::rows_) );
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  auto phase = std::complex<data_t>(initial_phase);
  add_y_phase(num_y, phase);
  return BaseVector::apply_function_sum(
    expval_pauli_XYZ_func_dm<data_t>(x_mask, z_mask, x_max, phase, BaseMatrix::rows_) );
}

//-----------------------------------------------------------------------
// Z-measurement outcome probabilities
//-----------------------------------------------------------------------

template <typename data_t>
double DensityMatrixThrust<data_t>::probability(const uint_t outcome) const 
{
  const auto shift = BaseMatrix::num_rows() + 1;
  std::complex<data_t> ret;
  ret = (std::complex<data_t>)BaseVector::chunk_->Get(outcome * shift);
  return std::real(ret);
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
  uint_t size(int num_qubits)
  {
    (void)num_qubits;
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
    probs[i] = BaseVector::apply_function_sum(density_probability_func<data_t>(qubits,qubits_sorted,i,BaseMatrix::num_rows()));
  }

  return probs;
}

template <typename data_t>
reg_t DensityMatrixThrust<data_t>::sample_measure(const std::vector<double> &rnds) const 
{
  uint_t nrows = BaseMatrix::num_rows();
#ifdef AER_DEBUG
  reg_t samples;

  BaseVector::DebugMsg("sample_measure begin");

  samples = BaseVector::chunk_->sample_measure(rnds,nrows+1,false);
  BaseVector::DebugMsg("sample_measure",samples);
  return samples;
#else
  return BaseVector::chunk_->sample_measure(rnds,nrows+1,false);
#endif
}


//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const AER::QV::DensityMatrixThrust<data_t>&m) {
  out << m.copy_to_matrix();
  return out;
}

//------------------------------------------------------------------------------
#endif // end module

