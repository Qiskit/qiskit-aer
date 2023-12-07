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

#ifndef _qv_transformer_
#define _qv_transformer_

#include "framework/utils.hpp"
#include "simulators/statevector/indexes.hpp"

namespace AER {
namespace QV {

template <typename T>
using cvector_t = std::vector<std::complex<T>>;

template <typename Container, typename data_t = double>
class Transformer {

  // TODO: This class should have the indexes.hpp moved inside it

public:
  virtual ~Transformer() {}
  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit
  // matrix.

  virtual void apply_matrix(Container &data, size_t data_size, int threads,
                            const reg_t &qubits,
                            const cvector_t<double> &mat) const;

  // Apply a N-qubit diagonal matrix to a array container
  // The matrix is input as vector of the matrix diagonal.
  virtual void apply_diagonal_matrix(Container &data, size_t data_size,
                                     int threads, const reg_t &qubits,
                                     const cvector_t<double> &diag) const;

protected:
  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit
  // matrix.
  template <size_t N>
  void apply_matrix_n(Container &data, size_t data_size, int threads,
                      const reg_t &qubits, const cvector_t<double> &mat) const;

  // Specialized single qubit apply matrix function
  void apply_matrix_1(Container &data, size_t data_size, int threads,
                      const uint_t qubit, const cvector_t<double> &mat) const;

  // Specialized single qubit apply matrix function
  void apply_diagonal_matrix_1(Container &data, size_t data_size, int threads,
                               const uint_t qubit,
                               const cvector_t<double> &mat) const;

  // Convert a matrix to a different type
  // TODO: this makes an unnecessary copy when data_t = double.
  cvector_t<data_t> convert(const cvector_t<double> &v) const;
};

/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/

template <typename Container, typename data_t>
cvector_t<data_t>
Transformer<Container, data_t>::convert(const cvector_t<double> &v) const {
  cvector_t<data_t> ret(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    ret[i] = v[i];
  return ret;
}

template <typename Container, typename data_t>
void Transformer<Container, data_t>::apply_matrix(
    Container &data, size_t data_size, int threads, const reg_t &qubits,
    const cvector_t<double> &mat) const {
  // Static array optimized lambda functions
  switch (qubits.size()) {
  case 1:
    return apply_matrix_1(data, data_size, threads, qubits[0], mat);
  case 2:
    return apply_matrix_n<2>(data, data_size, threads, qubits, mat);
  case 3:
    return apply_matrix_n<3>(data, data_size, threads, qubits, mat);
  case 4:
    return apply_matrix_n<4>(data, data_size, threads, qubits, mat);
  case 5:
    return apply_matrix_n<5>(data, data_size, threads, qubits, mat);
  case 6:
    return apply_matrix_n<6>(data, data_size, threads, qubits, mat);
  case 7:
    return apply_matrix_n<7>(data, data_size, threads, qubits, mat);
  case 8:
    return apply_matrix_n<8>(data, data_size, threads, qubits, mat);
  case 9:
    return apply_matrix_n<9>(data, data_size, threads, qubits, mat);
  case 10:
    return apply_matrix_n<10>(data, data_size, threads, qubits, mat);
  case 11:
    return apply_matrix_n<11>(data, data_size, threads, qubits, mat);
  case 12:
    return apply_matrix_n<12>(data, data_size, threads, qubits, mat);
  case 13:
    return apply_matrix_n<13>(data, data_size, threads, qubits, mat);
  case 14:
    return apply_matrix_n<14>(data, data_size, threads, qubits, mat);
  case 15:
    return apply_matrix_n<15>(data, data_size, threads, qubits, mat);
  case 16:
    return apply_matrix_n<16>(data, data_size, threads, qubits, mat);
  case 17:
    return apply_matrix_n<17>(data, data_size, threads, qubits, mat);
  case 18:
    return apply_matrix_n<18>(data, data_size, threads, qubits, mat);
  case 19:
    return apply_matrix_n<19>(data, data_size, threads, qubits, mat);
  case 20:
    return apply_matrix_n<20>(data, data_size, threads, qubits, mat);
  default: {
    throw std::runtime_error(
        "Maximum size of apply matrix is a 20-qubit matrix.");
  }
  }
}

template <typename Container, typename data_t>
template <size_t N>
void Transformer<Container, data_t>::apply_matrix_n(
    Container &data, size_t data_size, int threads, const reg_t &qs,
    const cvector_t<double> &mat) const {
  const size_t DIM = 1ULL << N;
  auto func = [&](const areg_t<1UL << N> &inds,
                  const cvector_t<data_t> &_mat) -> void {
    std::array<std::complex<data_t>, 1ULL << N> cache;
    for (size_t i = 0; i < DIM; i++) {
      const auto ii = inds[i];
      cache[i] = data[ii];
      data[ii] = 0.;
    }
    // update state vector
    for (size_t i = 0; i < DIM; i++)
      for (size_t j = 0; j < DIM; j++)
        data[inds[i]] += _mat[i + DIM * j] * cache[j];
  };
  areg_t<N> qubits;
  std::copy_n(qs.begin(), N, qubits.begin());
  apply_lambda(0, data_size, threads, func, qubits, convert(mat));
}

template <typename Container, typename data_t>
void Transformer<Container, data_t>::apply_matrix_1(
    Container &data, size_t data_size, int threads, const uint_t qubit,
    const cvector_t<double> &mat) const {

  // Check if matrix is diagonal and if so use optimized lambda
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const cvector_t<double> diag = {{mat[0], mat[3]}};
    apply_diagonal_matrix_1(data, data_size, threads, qubit, diag);
    return;
  }

  // Convert qubit to array register for lambda functions
  areg_t<1> qubits = {{qubit}};

  // Check if anti-diagonal matrix and if so use optimized lambda
  if (mat[0] == 0.0 && mat[3] == 0.0) {
    if (mat[1] == 1.0 && mat[2] == 1.0) {
      // X-matrix
      auto func = [&](const areg_t<2> &inds) -> void {
        std::swap(data[inds[0]], data[inds[1]]);
      };
      apply_lambda(0, data_size, threads, func, qubits);
      return;
    }
    if (mat[2] == 0.0) {
      // Non-unitary projector
      // possibly used in measure/reset/kraus update
      auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat) -> void {
        data[inds[1]] = _mat[1] * data[inds[0]];
        data[inds[0]] = 0.0;
      };
      apply_lambda(0, data_size, threads, func, qubits, convert(mat));
      return;
    }
    if (mat[1] == 0.0) {
      // Non-unitary projector
      // possibly used in measure/reset/kraus update
      auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat) -> void {
        data[inds[0]] = _mat[2] * data[inds[1]];
        data[inds[1]] = 0.0;
      };
      apply_lambda(0, data_size, threads, func, qubits, convert(mat));
      return;
    }
    // else we have a general anti-diagonal matrix
    auto func = [&](const areg_t<2> &inds,
                    const cvector_t<data_t> &_mat) -> void {
      const std::complex<data_t> cache = data[inds[0]];
      data[inds[0]] = _mat[2] * data[inds[1]];
      data[inds[1]] = _mat[1] * cache;
    };
    apply_lambda(0, data_size, threads, func, qubits, convert(mat));
    return;
  }

  auto func = [&](const areg_t<2> &inds,
                  const cvector_t<data_t> &_mat) -> void {
    const auto cache = data[inds[0]];
    data[inds[0]] = _mat[0] * cache + _mat[2] * data[inds[1]];
    data[inds[1]] = _mat[1] * cache + _mat[3] * data[inds[1]];
  };

  apply_lambda(0, data_size, threads, func, qubits, convert(mat));
}

template <typename Container, typename data_t>
void Transformer<Container, data_t>::apply_diagonal_matrix(
    Container &data, size_t data_size, int threads, const reg_t &qubits,
    const cvector_t<double> &diag) const {
  if (qubits.size() == 1) {
    apply_diagonal_matrix_1(data, data_size, threads, qubits[0], diag);
    return;
  }

  const size_t N = qubits.size();
  auto func = [&](const areg_t<2> &inds,
                  const cvector_t<data_t> &_diag) -> void {
    for (int_t i = 0; i < 2; ++i) {
      const uint_t k = inds[i];
      int_t iv = 0;
      for (uint_t j = 0; j < N; j++)
        if ((k & (1ULL << qubits[j])) != 0)
          iv += (1ULL << j);
      if (_diag[iv] != (data_t)1.0)
        data[k] *= _diag[iv];
    }
  };
  apply_lambda(0, data_size, threads, func, areg_t<1>({{qubits[0]}}),
               convert(diag));
}

template <typename Container, typename data_t>
void Transformer<Container, data_t>::apply_diagonal_matrix_1(
    Container &data, size_t data_size, int threads, const uint_t qubit,
    const cvector_t<double> &diag) const {
  // TODO: This should be changed so it isn't checking doubles with ==
  if (diag[0] == 1.0) { // [[1, 0], [0, z]] matrix
    if (diag[1] == 1.0)
      return; // Identity

    if (diag[1] == std::complex<double>(0., -1.)) { // [[1, 0], [0, -i]]
      auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat) -> void {
        const auto k = inds[1];
        double cache = data[k].imag();
        data[k].imag(data[k].real() * -1.);
        data[k].real(cache);
      };
      apply_lambda(0, data_size, threads, func, areg_t<1>({{qubit}}),
                   convert(diag));
      return;
    }
    if (diag[1] == std::complex<double>(0., 1.)) {
      // [[1, 0], [0, i]]
      auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat) -> void {
        const auto k = inds[1];
        double cache = data[k].imag();
        data[k].imag(data[k].real());
        data[k].real(cache * -1.);
      };
      apply_lambda(0, data_size, threads, func, areg_t<1>({{qubit}}),
                   convert(diag));
      return;
    }
    if (diag[0] == 0.0) {
      // [[1, 0], [0, 0]]
      auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat) -> void {
        data[inds[1]] = 0.0;
      };
      apply_lambda(0, data_size, threads, func, areg_t<1>({{qubit}}),
                   convert(diag));
      return;
    }
    // general [[1, 0], [0, z]]
    auto func = [&](const areg_t<2> &inds,
                    const cvector_t<data_t> &_mat) -> void {
      const auto k = inds[1];
      data[k] *= _mat[1];
    };
    apply_lambda(0, data_size, threads, func, areg_t<1>({{qubit}}),
                 convert(diag));
    return;
  } else if (diag[1] == 1.0) {
    // [[z, 0], [0, 1]] matrix
    if (diag[0] == std::complex<double>(0., -1.)) {
      // [[-i, 0], [0, 1]]
      auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat) -> void {
        const auto k = inds[1];
        double cache = data[k].imag();
        data[k].imag(data[k].real() * -1.);
        data[k].real(cache);
      };
      apply_lambda(0, data_size, threads, func, areg_t<1>({{qubit}}),
                   convert(diag));
      return;
    }
    if (diag[0] == std::complex<double>(0., 1.)) {
      // [[i, 0], [0, 1]]
      auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat) -> void {
        const auto k = inds[1];
        double cache = data[k].imag();
        data[k].imag(data[k].real());
        data[k].real(cache * -1.);
      };
      apply_lambda(0, data_size, threads, func, areg_t<1>({{qubit}}),
                   convert(diag));
      return;
    }
    if (diag[0] == 0.0) {
      // [[0, 0], [0, 1]]
      auto func = [&](const areg_t<2> &inds,
                      const cvector_t<data_t> &_mat) -> void {
        data[inds[0]] = 0.0;
      };
      apply_lambda(0, data_size, threads, func, areg_t<1>({{qubit}}),
                   convert(diag));
      return;
    }
    // general [[z, 0], [0, 1]]
    auto func = [&](const areg_t<2> &inds,
                    const cvector_t<data_t> &_mat) -> void {
      const auto k = inds[0];
      data[k] *= _mat[0];
    };
    apply_lambda(0, data_size, threads, func, areg_t<1>({{qubit}}),
                 convert(diag));
    return;
  } else {
    // Lambda function for diagonal matrix multiplication
    auto func = [&](const areg_t<2> &inds,
                    const cvector_t<data_t> &_mat) -> void {
      const auto k0 = inds[0];
      const auto k1 = inds[1];
      data[k0] *= _mat[0];
      data[k1] *= _mat[1];
    };
    apply_lambda(0, data_size, threads, func, areg_t<1>({{qubit}}),
                 convert(diag));
  }
}

//------------------------------------------------------------------------------
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
