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

#ifndef _qv_superoperator_avx2_hpp_
#define _qv_superoperator_avx2_hpp_


#include "framework/utils.hpp"
#include "superoperator.hpp"
#include "simulators/statevector/qv_avx2.hpp"

namespace AER {
namespace QV {

//============================================================================
// Superoperator class
//============================================================================

// This class is derived from the DensityMatrix class and stores an N-qubit 
// superoperator as a 2 * N-qubit vector.
// The vector is formed using column-stacking vectorization of the
// superoperator (itself with respect to column-stacking vectorization).

template <typename data_t = double>
class SuperoperatorAvx2 : public Superoperator<data_t, SuperoperatorAvx2<data_t>> {

public:
  // Parent class aliases
  using BaseVector = QubitVector<data_t, SuperoperatorAvx2<data_t>>;

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  SuperoperatorAvx2();
  explicit SuperoperatorAvx2(size_t num_qubits);
  SuperoperatorAvx2(const SuperoperatorAvx2& obj) = delete;
  SuperoperatorAvx2 &operator=(const SuperoperatorAvx2& obj) = delete;

  // Apply a 1-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  void apply_matrix(const uint_t qubit, const cvector_t<double> &mat);

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  protected:
  size_t calculate_num_threads();
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/
// We do not define this functions in case we don't use AVX2
// so it can compile, as this class won't be used
#if defined(_MSC_VER) || defined(GNUC_AVX2)
// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const QV::SuperoperatorAvx2<data_t>&qv) {

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
// Constructors & Destructor
//------------------------------------------------------------------------------
template <typename data_t>
SuperoperatorAvx2<data_t>::SuperoperatorAvx2(size_t num_qubits) {
  BaseVector::num_qubits_ = 0;
  BaseVector::data_ = nullptr;
  BaseVector::checkpoint_ = 0;
  BaseVector::set_num_qubits(num_qubits);
}

template <typename data_t>
SuperoperatorAvx2<data_t>::SuperoperatorAvx2() : SuperoperatorAvx2(0) {}

template <typename data_t>
void SuperoperatorAvx2<data_t>::apply_matrix(const uint_t qubit,
                                           const cvector_t<double>& mat) {
  if ((mat[1] == 0.0 && mat[2] == 0.0) || (mat[0] == 0.0 && mat[3] == 0.0)) {
    BaseVector::apply_matrix(qubit, mat);
    return;
  }

  reg_t qubits = {qubit};
  if (apply_matrix_avx<data_t>(reinterpret_cast<data_t*>(BaseVector::data_),
                               BaseVector::data_size_, qubits.data(), qubits.size(),
                               reinterpret_cast<data_t *>(BaseVector::convert(mat).data()),
                               calculate_num_threads()) == Avx::NotApplied) {
    BaseVector::apply_matrix(qubit, mat);
  }
}

template <typename data_t>
void SuperoperatorAvx2<data_t>::apply_matrix(const reg_t& qubits,
                                           const cvector_t<double>& mat) {
  if (apply_matrix_avx<data_t>(reinterpret_cast<data_t*>(BaseVector::data_),
                               BaseVector::data_size_, qubits.data(), qubits.size(),
                               reinterpret_cast<data_t *>(BaseVector::convert(mat).data()),
                               calculate_num_threads()) == Avx::NotApplied) {
    BaseVector::apply_matrix(qubits, mat);
  }
}

template <typename data_t>
size_t SuperoperatorAvx2<data_t>::calculate_num_threads() {
  if (BaseVector::num_qubits_ > BaseVector::omp_threshold_ && BaseVector::omp_threads_ > 1) {
    return BaseVector::omp_threads_;
  }
  return 1;
}
#endif

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const AER::QV::SuperoperatorAvx2<data_t>&m) {
  out << m.copy_to_matrix();
  return out;
}

//------------------------------------------------------------------------------
#endif // end module

