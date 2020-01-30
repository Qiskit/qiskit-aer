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
  DensityMatrixThrust(const DensityMatrixThrust& obj) = delete;
  DensityMatrixThrust &operator=(const DensityMatrixThrust& obj) = delete;

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

  // Return M sampled outcomes for Z-basis measurement of all qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  virtual reg_t sample_measure(const std::vector<double> &rnds) const override;

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
  const auto nq = num_qubits();
  for (const auto q: qubits) {
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
void DensityMatrixThrust<data_t>::apply_unitary_matrix(const reg_t &qubits,
                                                 const cvector_t<double> &mat) {
  // Check if we apply as two N-qubit matrix multiplications or a single 2N-qubit matrix mult.
  if (qubits.size() > apply_unitary_threshold_) {
    // Apply as two N-qubit matrix mults
    auto nq = num_qubits();
    reg_t conj_qubits;
    for (const auto q: qubits) {
      conj_qubits.push_back(q + nq);
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
  const size_t nq = num_qubits();
  const reg_t qubits = {{qctrl, qtrgt, qctrl + nq, qtrgt + nq}};
  BaseVector::apply_permutation_matrix(qubits, pairs);
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_cnot",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_cz(const uint_t q0, const uint_t q1) {
  cvector_t<double> vec;
  vec.resize(16, 0.);

  vec[3] = -1.;
  vec[7] = -1.;
  vec[11] = -1.;
  vec[12] = -1.;
  vec[13] = -1.;
  vec[14] = -1.;

  const auto nq =  num_qubits();
  const reg_t qubits = {{q0, q1, q0 + nq, q1 + nq}};
  BaseVector::apply_matrix(qubits, vec);
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_cz",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_swap(const uint_t q0, const uint_t q1) {
  std::vector<std::pair<uint_t, uint_t>> pairs = {
   {{1, 2}, {4, 8}, {5, 10}, {6, 9}, {7, 11}, {13, 14}}
  };
  const size_t nq = num_qubits();
  const reg_t qubits = {{q0, q1, q0 + nq, q1 + nq}};  //TODO support
  BaseVector::apply_permutation_matrix(qubits, pairs);
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_swap",qubits);
#endif
}

template <typename data_t>
class DensityX : public GateFuncBase
{
protected:
  uint_t mask0;
  uint_t mask1;

public:
  DensityX(int q0,int q1)
  {
  	if(q0 < q1){
      mask0 = (1ull << q0) - 1;
      mask1 = (1ull << q1) - 1;
  	}
  	else{
      mask0 = (1ull << q1) - 1;
      mask1 = (1ull << q0) - 1;
  	}

  }

	__host__ __device__ double operator()(const thrust::tuple<uint_t,struct GateParams<data_t>> &iter) const
  {
    uint_t i,i0,i1,i2;
	thrust::complex<data_t>* pV;
	uint_t* offsets;
    thrust::complex<data_t> q0,q1,q2,q3;
		struct GateParams<data_t> params;

  	i = ExtractIndexFromTuple(iter);
		params = ExtractParamsFromTuple(iter);
		pV = params.buf_;
		offsets = params.offsets_;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    q0 = pV[offsets[0]+i0];
    q1 = pV[offsets[1]+i0];
    q2 = pV[offsets[2]+i0];
    q3 = pV[offsets[3]+i0];

    pV[offsets[0]+i0] = q3;
    pV[offsets[1]+i0] = q2;
    pV[offsets[2]+i0] = q1;
    pV[offsets[3]+i0] = q0;
		return 0.0;
  }

};

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_x(const uint_t qubit) {
  // Use the lambda function
  const reg_t qubits = {{qubit, qubit + num_qubits()}};

	BaseVector::apply_function(DensityX<data_t>(qubits[0], qubits[1]), qubits);

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_x",qubits);
	BaseVector::DebugDump();
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_y(const uint_t qubit) {
  cvector_t<double> vec;
  vec.resize(16, 0.);
  vec[0 * 4 + 3] = 1.;
  vec[1 * 4 + 2] = -1.;
  vec[2 * 4 + 1] = -1.;
  vec[3 * 4 + 0] = 1.;
  // Use the lambda function
  const reg_t qubits = {{qubit, qubit + num_qubits()}};
  BaseVector::apply_matrix(qubits, vec);

#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::apply_y",qubits);
#endif
}

template <typename data_t>
void DensityMatrixThrust<data_t>::apply_z(const uint_t qubit) {
  cvector_t<double> vec;
  vec.resize(16, 0.);
  vec[0 * 4 + 0] = 1.;
  vec[1 * 4 + 1] = -1.;
  vec[2 * 4 + 2] = -1.;
  vec[3 * 4 + 3] = 1.;

  // Use the lambda function
  const reg_t qubits = {{qubit, qubit + num_qubits()}};
  BaseVector::apply_matrix(qubits, vec);

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
  const size_t nq = num_qubits();
  const reg_t qubits = {{qctrl0, qctrl1, qtrgt,
                         qctrl0 + nq, qctrl1 + nq, qtrgt + nq}};
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
reg_t DensityMatrixThrust<data_t>::sample_measure(const std::vector<double> &rnds) const {

  const int_t END = 1LL << num_qubits();
  const int_t SHOTS = rnds.size();
  reg_t samples;
  samples.assign(SHOTS, 0);

  const int INDEX_SIZE = BaseVector::sample_measure_index_size_;
  const int_t INDEX_END = BITS[INDEX_SIZE];
  // Qubit number is below index size, loop over shots
  if (END < INDEX_END) {
    #pragma omp parallel if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
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
    #pragma omp parallel if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
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

    #pragma omp parallel if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
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
#ifdef AER_DEBUG
	BaseVector::DebugMsg(" density::sample_measure",samples);
#endif
	
  return samples;
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

