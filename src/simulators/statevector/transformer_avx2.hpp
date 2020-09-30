/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _qv_transformer_avx2_
#define _qv_transformer_avx2_

#include "misc/common_macros.hpp"
#include "simulators/statevector/qv_avx2.hpp"
#include "simulators/statevector/transformer.hpp"

namespace AER {
namespace QV {

template <typename data_t = double>
class TransformerAVX2 : public Transformer<data_t> {

  using Base = Transformer<data_t>;

public:
  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------
  TransformerAVX2(size_t omp_threads = 1);

  virtual ~TransformerAVX2() = default;

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit
  // matrix.
  template <typename Container>
  void apply_matrix(Container &data, size_t data_size, const reg_t &qubits,
                    const cvector_t<double> &mat) const;
};

/*******************************************************************************
 *
 * Implementation
 *
 ******************************************************************************/

// We do not define this functions in case we don't use AVX2
// so it can compile, as this class won't be used
#if defined(_MSC_VER) || defined(GNUC_AVX2)

template <typename data_t>
TransformerAVX2<data_t>::TransformerAVX2(size_t omp_threads)
    : Transformer<data_t>(omp_threads) {}

template <typename data_t>
template <typename Container>
void TransformerAVX2<data_t>::apply_matrix(Container &data, size_t data_size,
                                           const reg_t &qubits,
                                           const cvector_t<double> &mat) const {

  if (qubits.size() == 1 &&
      ((mat[1] == 0.0 && mat[2] == 0.0) || (mat[0] == 0.0 && mat[3] == 0.0))) {
    return Base::apply_matrix_1(data, data_size, qubits[0], mat);
  }

  if (apply_matrix_avx<data_t>(
          reinterpret_cast<data_t *>(data), data_size, qubits.data(),
          qubits.size(), reinterpret_cast<data_t *>(Base::convert(mat).data()),
          Base::omp_threads_) == Avx::Applied) {
    return;
  }

  Base::apply_matrix(data, data_size, qubits, mat);
}

#endif // AVX2 Code

template class TransformerAVX2<double>;
template class TransformerAVX2<float>;

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
