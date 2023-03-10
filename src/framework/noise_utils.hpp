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

#ifndef _aer_noise_utils_hpp_
#define _aer_noise_utils_hpp_

#include "framework/utils.hpp"

namespace AER {
namespace Utils {

//=========================================================================
// Noise Transformation Functions
//=========================================================================

// Transform a superoperator matrix to a Choi matrix
template <class T>
matrix<T> superop2choi(const matrix<T> &superop, size_t dim);

// Transform a superoperator matrix to a set of Kraus matrices
template <class T>
std::vector<matrix<T>> superop2kraus(const matrix<T> &superop, size_t dim,
                                     double threshold = 1e-10);

// Transform a Choi matrix to a superoperator matrix
template <class T>
matrix<T> choi2superop(const matrix<T> &choi, size_t dim);

// Transform a Choi matrix to a set of Kraus matrices
template <class T>
std::vector<matrix<std::complex<T>>>
choi2kraus(const matrix<std::complex<T>> &choi, size_t dim,
           double threshold = 1e-10);

// Transform a set of Kraus matrices to a Choi matrix
template <class T>
matrix<T> kraus2choi(const std::vector<matrix<T>> &kraus, size_t dim);

// Transform a set of Kraus matrices to a superoperator matrix
template <class T>
matrix<T> kraus2superop(const std::vector<matrix<T>> &kraus, size_t dim);

// Reshuffle transformation
// Transforms a matrix with dimension (d0 * d1, d2 * d3)
// into a matrix with dimension (d3 * d1, d2 * d0)
// by transposition of bipartite indices
// M[(i, j), (k, l)] -> M[(i, j), (k, i)]
template <class T>
matrix<T> reshuffle(const matrix<T> &mat, size_t d0, size_t d1, size_t d2,
                    size_t d3);

//=========================================================================
// Implementations
//=========================================================================

template <class T>
matrix<T> kraus2superop(const std::vector<matrix<T>> &kraus, size_t dim) {
  matrix<T> superop(dim * dim, dim * dim);
  for (const auto mat : kraus) {
    superop += Utils::tensor_product(Utils::conjugate(mat), mat);
  }
  return superop;
}

template <class T>
matrix<T> kraus2choi(const std::vector<matrix<T>> &kraus, size_t dim) {
  return superop2choi(kraus2superop(kraus, dim), dim);
}

template <class T>
matrix<T> superop2choi(const matrix<T> &superop, size_t dim) {
  return reshuffle(superop, dim, dim, dim, dim);
}

template <class T>
std::vector<matrix<T>> superop2kraus(const matrix<T> &superop, size_t dim,
                                     double threshold) {
  return choi2kraus(superop2choi(superop, dim), dim, threshold);
}

template <class T>
matrix<T> choi2superop(const matrix<T> &choi, size_t dim) {
  return reshuffle(choi, dim, dim, dim, dim);
}

template <class T>
std::vector<matrix<std::complex<T>>>
choi2kraus(const matrix<std::complex<T>> &choi, size_t dim, double threshold) {
  size_t dim2 = dim * dim;

  std::vector<T> evals;
  matrix<std::complex<T>> evecs;

  eigensystem_hermitian(choi, evals, evecs);

  // Convert eigensystem to Kraus operators
  std::vector<matrix<std::complex<T>>> kraus;
  for (size_t i = 0; i < dim2; i++) {
    // Eigenvalues sorted smallest to largest so we index from the back
    const size_t idx = dim2 - 1 - i;
    const T eval = evals[idx];
    if (eval > 0.0 && !Linalg::almost_equal(eval, 0.0, threshold)) {
      std::complex<T> coeff(std::sqrt(eval), 0.0);
      matrix<std::complex<T>> kmat(dim, dim);
      for (size_t col = 0; col < dim; col++)
        for (size_t row = 0; row < dim; row++) {
          kmat(row, col) = coeff * evecs(row + dim * col, idx);
        }
      kraus.push_back(kmat);
    }
  }
  return kraus;
}

template <class T>
matrix<T> reshuffle(const matrix<T> &mat, size_t d0, size_t d1, size_t d2,
                    size_t d3) {
  matrix<T> ret(d1 * d3, d0 * d2);
  for (size_t i0 = 0; i0 < d0; ++i0)
    for (size_t i1 = 0; i1 < d1; ++i1)
      for (size_t i2 = 0; i2 < d2; ++i2)
        for (size_t i3 = 0; i3 < d3; ++i3) {
          ret(d1 * i3 + i1, d0 * i2 + i0) = mat(d1 * i0 + i1, d3 * i2 + i3);
        }
  return ret;
}

//-------------------------------------------------------------------------
} // namespace Utils
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif