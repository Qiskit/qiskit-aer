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
#include "simulators/statevector/transformer.hpp"
#include "simulators/statevector/qv_avx2.hpp"

namespace AER {
namespace QV {

template <typename Container, typename data_t = double>
class TransformerAVX2 : public Transformer<Container, data_t, TransformerAVX2<Container, data_t>> {

using Base = Transformer<Container, data_t, TransformerAVX2<Container, data_t>>;

public:
  
  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------
  TransformerAVX2(Container& data, size_t data_size, size_t omp_threads = 1);

  // Prevent initializing with temporary since container is stored as a reference
  TransformerAVX2(Container&& data, size_t data_size, size_t omp_threads) = delete;

  virtual ~TransformerAVX2() = default;

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const reg_t &qubits,
                    const cvector_t<double> &mat);
};

/*******************************************************************************
 *
 * Implementaiton
 *
 ******************************************************************************/

// We do not define this functions in case we don't use AVX2
// so it can compile, as this class won't be used
#if defined(_MSC_VER) || defined(GNUC_AVX2)

template <typename Container, typename data_t>
TransformerAVX2<Container, data_t>::TransformerAVX2(Container& data,
                                                    size_t data_size,
                                                    size_t omp_threads)
    : Transformer<Container, data_t, TransformerAVX2<Container, data_t>>(data, data_size, omp_threads) {}


template <typename Container, typename data_t>
void TransformerAVX2<Container, data_t>::apply_matrix(const reg_t &qubits,
                                                      const cvector_t<double> &mat) {
  
  if ((mat[1] == 0.0 && mat[2] == 0.0) || (mat[0] == 0.0 && mat[3] == 0.0)) {
    return Base::apply_matrix_1(qubits[0], mat);
  }

  if (apply_matrix_avx<data_t>(reinterpret_cast<data_t*>(Base::data_),
                               Base::data_size_, qubits.data(), qubits.size(),
                               reinterpret_cast<data_t*>(Base::convert(mat).data()),
                               Base::omp_threads_) == Avx::Applied) {
    return;
  }

  Base::apply_matrix(qubits, mat);
}

#endif // AVX2 Code

// Declare used template arguments
template class TransformerAVX2<std::complex<double>*, double>;
template class TransformerAVX2<std::complex<float>*, float>;

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
