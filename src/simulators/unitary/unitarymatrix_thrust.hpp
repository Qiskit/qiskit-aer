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

#ifndef _qv_unitary_matrix_thrust_hpp_
#define _qv_unitary_matrix_thrust_hpp_


#include "framework/utils.hpp"
#include "simulators/statevector/qubitvector_thrust.hpp"

namespace QV {

//============================================================================
// UnitaryMatrixThrust class
//============================================================================

// This class is derived from the QubitVectorThrust class and stores an N-qubit
// matrix as a 2*N-qubit vector.
// The vector is formed using column-stacking vectorization as under this
// convention left-matrix multiplication on qubit-n is equal to multiplication
// of the vectorized 2*N qubit vector also on qubit-n.

template <class data_t = double>
class UnitaryMatrixThrust : public QubitVectorThrust<data_t> {

public:
  // Type aliases
  using BaseVector = QubitVectorThrust<data_t>;

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  UnitaryMatrixThrust() : UnitaryMatrixThrust(0) {};
  explicit UnitaryMatrixThrust(size_t num_qubits);
  UnitaryMatrixThrust(const UnitaryMatrixThrust& obj) = delete;
  UnitaryMatrixThrust &operator=(const UnitaryMatrixThrust& obj) = delete;

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Return the string name of the class
#ifdef AER_THRUST_CUDA
  static std::string name() {return "unitary_gpu";}
#else
  static std::string name() {return "unitary_thrust";}
#endif

  // Set the size of the vector in terms of qubit number
  void set_num_qubits(size_t num_qubits);

  // Return the number of rows in the matrix
  size_t num_rows() const {return rows_;}

  // Returns the number of qubits for the current vector
  virtual uint_t num_qubits() const override { return num_qubits_;}

  // Copy the internal data array to a complex matrix
  matrix<std::complex<data_t>> copy_to_matrix() const;

  // Move the internal data array to a complex matrix
  // Note that this is technically still a copy as it must be copied
  // from device to host memory
  matrix<std::complex<data_t>> move_to_matrix() { return copy_to_matrix(); }

  // Return the trace of the unitary
  std::complex<double> trace() const;

  // Return JSON serialization of UnitaryMatrixThrust;
  json_t json() const;

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // If the length of the statevector does not match the number of qubits
  // an exception is raised.
  void initialize_from_matrix(const AER::cmatrix_t &mat);

  //-----------------------------------------------------------------------
  // Identity checking
  //-----------------------------------------------------------------------
  
  // Return pair (True, theta) if the current matrix is equal to
  // exp(i * theta) * identity matrix. Otherwise return (False, 0).
  // The phase is returned as a parameter between -Pi and Pi.
  std::pair<bool, double> check_identity() const;

  // Set the threshold for verify_identity
  void set_check_identity_threshold(double threshold) {
    identity_threshold_ = threshold;
  }

  // Get the threshold for verify_identity
  double get_check_identity_threshold() {return identity_threshold_;}

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  size_t num_qubits_;
  size_t rows_;

  //-----------------------------------------------------------------------
  // Additional config settings
  //-----------------------------------------------------------------------
  
  double identity_threshold_ = 1e-10; // Threshold for verifying if the
                                      // internal matrix is identity up to
                                      // global phase
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

template <class data_t>
inline void to_json(json_t &js, const UnitaryMatrixThrust<data_t> &qmat) {
  js = qmat.json();
}

template <class data_t>
json_t UnitaryMatrixThrust<data_t>::json() const 
{
  const int_t nrows = rows_;
  int iPlace;
  int_t i, irow, icol;
  uint_t csize = BaseVector::data_size_;
  cvector_t<data_t> tmp(csize);

  const json_t ZERO = std::complex < data_t > (0.0, 0.0);
  json_t js = json_t(nrows, json_t(nrows, ZERO));

#pragma omp parallel private(i,irow,icol) if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
  {
    if (BaseVector::json_chop_threshold_ > 0) {
#pragma omp for
      for (i = 0; i < csize; i++) {
        irow = (i >> num_qubits_);
        icol = i - (irow << num_qubits_);

        if (std::abs(tmp[i].real()) > BaseVector::json_chop_threshold_)
          js[icol][irow][0] = tmp[i].real();
        if (std::abs(tmp[i].imag()) > BaseVector::json_chop_threshold_)
          js[icol][irow][1] = tmp[i].imag();
      }
    } else {
#pragma omp for
      for (i = 0; i < csize; i++) {
        irow = (i >> num_qubits_);
        icol = i - (irow << num_qubits_);

        js[icol][irow][0] = tmp[i].real();
        js[icol][irow][1] = tmp[i].imag();
      }
    }
  }

  return js;
}


//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <class data_t>
UnitaryMatrixThrust<data_t>::UnitaryMatrixThrust(size_t num_qubits) {
	if(num_qubits > 0){
		set_num_qubits(num_qubits);
	}
}

//------------------------------------------------------------------------------
// Convert data vector to matrix
//------------------------------------------------------------------------------

template <class data_t>
matrix<std::complex<data_t>> UnitaryMatrixThrust<data_t>::copy_to_matrix() const 
{
  const int_t nrows = rows_;
  AER::cmatrix_t ret(nrows, nrows);
  uint_t csize = BaseVector::data_size_;

  cvector_t<data_t> qreg = BaseVector::vector();

  int_t i;
  uint_t irow, icol;
#pragma omp parallel for private(i,irow,icol) if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
  for (i = 0; i < csize; i++) {
    irow = (i >> num_qubits_);
    icol = i - (irow << num_qubits_);

    ret(icol, irow) = qreg[i];
  }
	return ret;
}
	
//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <class data_t>
void UnitaryMatrixThrust<data_t>::initialize() 
{
  const int_t nrows = rows_;
	std::complex<double> one = 1.0;
  // Zero the underlying vector
  BaseVector::zero();
  // Set to be identity matrix

  uint_t is,ie,idx;
  int_t i;

  is = BaseVector::chunk_index_ << BaseVector::num_qubits_;
  ie = is + (1ull << BaseVector::num_qubits_);
#pragma omp parallel private(idx) if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
  for(i=0;i<nrows;i++){
    idx = i * (nrows + 1);
    if(idx >= is && idx < ie){
      BaseVector::set_state(idx-is,one);
    }
  }
}

template <class data_t>
void UnitaryMatrixThrust<data_t>::initialize_from_matrix(const AER::cmatrix_t &mat) {
  const int_t nrows = rows_;    // end for k loop
  if (nrows != static_cast<int_t>(mat.GetRows()) ||
      nrows != static_cast<int_t>(mat.GetColumns())) {
    throw std::runtime_error(
      "UnitaryMatrixThrust::initialize input matrix is incorrect shape (" +
      std::to_string(nrows) + "," + std::to_string(nrows) + ")!=(" +
      std::to_string(mat.GetRows()) + "," + std::to_string(mat.GetColumns()) + ")."
    );
  }
  if (AER::Utils::is_unitary(mat, 1e-10) == false) {
    throw std::runtime_error(
      "UnitaryMatrixThrust::initialize input matrix is not unitary."
    );
  }

  cvector_t<data_t> tmp(BaseVector::data_size_);
  uint_t n_chunks_in_row = mat.GetRows() >> num_qubits_;
  uint_t base_row = (BaseVector::chunk_index_ / n_chunks_in_row) << num_qubits_;
  uint_t base_col = (BaseVector::chunk_index_ % n_chunks_in_row) << num_qubits_;

  int_t i,irow,icol;
#pragma omp parallel for private(i,irow,icol) if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
  for(i=0;i<BaseVector::data_size_;i++){
    std::complex<double> t;
    irow = (i >> num_qubits_);
    icol = i - (irow << num_qubits_);

    t = mat(base_row + irow,base_col + icol);
    tmp[i] = t;
	}

  BaseVector::chunk_->CopyIn((thrust::complex<data_t>*)&tmp[0]);
}

template <class data_t>
void UnitaryMatrixThrust<data_t>::set_num_qubits(size_t num_qubits) {
  // Set the number of rows for the matrix
  num_qubits_ = num_qubits;
  rows_ = 1ULL << num_qubits;
  // Set the underlying vectorized matrix to be 2 * number of qubits
  BaseVector::set_num_qubits(2 * num_qubits);
}

template <typename data_t>
class UnitaryTrace : public GateFuncBase<data_t>
{
protected:
  uint_t rows_;
public:
  UnitaryTrace(uint_t nr)
  {
    rows_ = nr;
  }

  bool IsDiagonal(void)
  {
    return true;
  }

  __host__ __device__ thrust::complex<double> operator()(const uint_t &i) const
  {
    uint_t idx;
    thrust::complex<data_t>* vec;
    thrust::complex<double> q;

    idx = i * (rows_ + 1);
    vec = this->data_;

    q = vec[idx];

    return q;
  }

  uint_t size(int num_qubits,int n)
  {
    (void)num_qubits;
    (void)n;
    return rows_;
  }
  const char* name(void)
  {
    return "UnitaryTrace";
  }
};

template <class data_t>
std::complex<double> UnitaryMatrixThrust<data_t>::trace() const 
{
  reg_t dummy(1,0);
  std::complex<double> sum;

  sum = BaseVector::apply_function_complex_sum(UnitaryTrace<data_t>(rows_),dummy);

#ifdef AER_DEBUG
  BaseVector::DebugMsg("trace",sum);
#endif

  return sum;
}


//------------------------------------------------------------------------------
// Check Identity
//------------------------------------------------------------------------------

template <class data_t>
std::pair<bool, double> UnitaryMatrixThrust<data_t>::check_identity() const {
  // To check if identity we first check we check that:
  // 1. U(0, 0) = exp(i * theta)
  // 2. U(i, i) = U(0, 0)
  // 3. U(i, j) = 0 for j != i 
  auto failed = std::make_pair(false, 0.0);

  // Check condition 1.
	const auto u00 = BaseVector::get_state(0);
  if (std::norm(std::abs(u00) - 1.0) > identity_threshold_) {
    return failed;
  }
  const auto theta = std::arg(u00);

  // Check conditions 2 and 3
  double delta = 0.;
	int iPlace;
	uint_t i,irow,icol,ic,nc;
	uint_t pos = 0;
	uint_t csize = BaseVector::data_size_;
	cvector_t<data_t> tmp(csize);

  BaseVector::chunk_->CopyOut((thrust::complex<data_t>*)&tmp[0]);

  uint_t offset = BaseVector::chunk_index_ << BaseVector::num_qubits_;
  uint_t err_count = 0;
#pragma omp parallel for private(i,irow,icol) reduction(+:delta,err_count) if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) num_threads(BaseVector::omp_threads_)
  for(i=0;i<csize;i++){
    irow = ((i + offset) >> num_qubits_);
    icol = (i + offset) - (irow << num_qubits_);

    auto val = (irow==icol) ? std::norm(tmp[i] - u00)
                            : std::norm(tmp[i]);
    if (val > identity_threshold_) {
      err_count++;
    }
    else{
      delta += val; // accumulate difference
    }
  }
  if(err_count > 0){
    return failed;
  }

  // Check small errors didn't accumulate
  if (delta > identity_threshold_) {
    return failed;
  }
  // Otherwise we pass
  return std::make_pair(true, theta);
}

//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <class data_t>
inline std::ostream &operator<<(std::ostream &out, const QV::UnitaryMatrixThrust<data_t>&m) {
  out << m.copy_to_matrix();
  return out;
}

//------------------------------------------------------------------------------
#endif // end module
