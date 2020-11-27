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

template <typename Container, typename data_t = double>
class TransformerAVX2 : public Transformer<Container, data_t> {
  using Base = Transformer<Container, data_t>;

public:
  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit
  // matrix.
  void apply_matrix(Container &data, size_t data_size, int threads,
                           const reg_t &qubits, const cvector_t<double> &mat) const override;

  void apply_diagonal_matrix(Container &data, size_t data_size, int threads,
                            const reg_t &qubits, const cvector_t<double> &diag) const override;

  void apply_diagonal_matrices(Container &data, size_t data_size, int threads,
                               const std::vector<reg_t> &qubits_list,
                               const std::vector<cvector_t<double>> &diags) const;


};

/*******************************************************************************
 *
 * Implementation
 *
 ******************************************************************************/

// We do not define this functions in case we don't use AVX2
// so it can compile, as this class won't be used
#if defined(_MSC_VER) || defined(GNUC_AVX2)

template <typename Container, typename data_t>
void TransformerAVX2<Container, data_t>::apply_matrix(Container &data, size_t data_size,
                                           int threads, const reg_t &qubits,
                                           const cvector_t<double> &mat) const{

  if (qubits.size() == 1 &&
      ((mat[1] == 0.0 && mat[2] == 0.0) || (mat[0] == 0.0 && mat[3] == 0.0))) {
    return Base::apply_matrix_1(data, data_size, threads, qubits[0], mat);
  }

  if (apply_matrix_avx<data_t>(
          reinterpret_cast<data_t *>(data), data_size, qubits.data(),
          qubits.size(), reinterpret_cast<data_t *>(Base::convert(mat).data()),
          threads) == Avx::Applied) {
    return;
  }

  Base::apply_matrix(data, data_size, threads, qubits, mat);
}


template <typename Container, typename data_t>
void TransformerAVX2<Container, data_t>::apply_diagonal_matrix(Container &data,
                                                               size_t data_size,
                                                               int threads,
                                                               const reg_t &qubits,
                                                               const cvector_t<double> &diag) const {

  if (apply_diagonal_matrix_avx<data_t>(
          reinterpret_cast<data_t *>(data), data_size, qubits.data(),
          qubits.size(), reinterpret_cast<data_t *>(Base::convert(diag).data()),
          threads) == Avx::Applied) {
    return;
  }

  Base::apply_diagonal_matrix(data, data_size, threads, qubits, diag);
}


template <typename Container, typename data_t>
void TransformerAVX2<Container, data_t>::apply_diagonal_matrices(Container &data,
                                                                 size_t data_size,
                                                                 int threads,
                                                                 const std::vector<reg_t> &qubits_list,
                                                                 const std::vector<cvector_t<double>> &diags) const {

  std::vector<const uint64_t*> qregs_list;
  std::vector<size_t> qregs_size_list;
  std::vector<const data_t*> vec_list;

  for (auto i = 0; i < qubits_list.size(); ++i) {
    qregs_list.push_back(qubits_list[i].data());
    qregs_size_list.push_back(qubits_list[i].size());
    vec_list.push_back(reinterpret_cast<data_t *>(Base::convert(diags[i]).data()));
  }

  uint64_t mat_size_ = qubits_list.size();
  const uint64_t** qregs_list_ = qregs_list.data();
  size_t* qregs_size_list_ = qregs_size_list.data();
  const data_t** vec_list_ = vec_list.data();

  if (apply_diagonal_matrices_avx<data_t>(
          reinterpret_cast<data_t *>(data), data_size, mat_size_,
          qregs_list_, qregs_size_list_, vec_list_, threads) == Avx::Applied) {
    return;
  }

  for (int i = 0; i < qubits_list.size(); ++i)
    apply_diagonal_matrix(data, data_size, threads, qubits_list[i], diags[i]);
}

#endif // AVX2 Code

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
