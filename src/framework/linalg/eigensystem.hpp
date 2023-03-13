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

#ifndef _aer_framework_linalg_eigensystem_hpp_
#define _aer_framework_linalg_eigensystem_hpp_

#include "framework/blas_protos.hpp"
#include "framework/matrix.hpp"
#include <type_traits>

/**
 * Returns the eigenvalues and eigenvectors
 * of a Hermitian matrix.
 * Uses the blas function ?heevx
 * @param hermitian_matrix: The Hermitian matrix.
 * @param eigenvalues: On output: vector with the eignevalues of the matrix
 * (input is overwritten)
 * @param eigenvectors: On output: matrix with the eigenvectors stored as
 * columns.
 *
 * @returns: void
 */
template <class T>
void eigensystem_hermitian(const matrix<std::complex<T>> &hermitian_matrix,
                           /* out */ std::vector<T> &eigenvalues,
                           /* out */ matrix<std::complex<T>> &eigenvectors);

template <typename T>
struct HeevxFuncs;

template <>
struct HeevxFuncs<double> {
  HeevxFuncs() = delete;
  static decltype(zheevx_) &heevx;
  static decltype(dlamch_) &lamch;
};

decltype(zheevx_) &HeevxFuncs<double>::heevx = zheevx_;
decltype(dlamch_) &HeevxFuncs<double>::lamch = dlamch_;

template <>
struct HeevxFuncs<float> {
  HeevxFuncs() = delete;
  static decltype(cheevx_) &heevx;
  static decltype(slamch_) &lamch;
};

decltype(cheevx_) &HeevxFuncs<float>::heevx = cheevx_;
decltype(slamch_) &HeevxFuncs<float>::lamch = slamch_;

template <typename T>
void eigensystem_hermitian(const matrix<std::complex<T>> &hermitian_matrix,
                           std::vector<T> &eigenvalues,
                           matrix<std::complex<T>> &eigenvectors) {
  if (hermitian_matrix.GetRows() != hermitian_matrix.GetColumns()) {
    throw std::runtime_error("Input matrix in eigensystem_hermitian "
                             "function is not a square matrix.");
  }

  int n = static_cast<int>(hermitian_matrix.GetLD());
  int ldz{n}, lda{n}, lwork{2 * n};
  int il{0}, iu{0};   // not referenced if range='A'
  T vl{0.0}, vu{0.0}; // not referenced if range='A'
  char cmach{'S'};
  T abstol{static_cast<T>(2.0 * HeevxFuncs<T>::lamch(&cmach))};
  int m{0}; // number of eigenvalues found
  int info{0};

  eigenvectors.resize(ldz, n);
  eigenvalues.clear();
  eigenvalues.resize(n);
  matrix<std::complex<T>> heevx_copy{hermitian_matrix};
  auto work = std::vector<std::complex<T>>(lwork, {0.0, 0.0});
  auto rwork = std::vector<T>(7 * n, 0.0);
  auto iwork = std::vector<int>(5 * n, 0);
  auto ifail = std::vector<int>(n, 0);

  HeevxFuncs<T>::heevx(&AerBlas::Jobz[0], &AerBlas::Range[0], &AerBlas::UpLo[0],
                       &n, heevx_copy.data(), &lda, &vl, &vu, &il, &iu, &abstol,
                       &m, eigenvalues.data(), eigenvectors.data(), &ldz,
                       work.data(), &lwork, rwork.data(), iwork.data(),
                       ifail.data(), &info);

  if (info) {
    throw std::runtime_error("Something went wrong in heevx call within "
                             "eigensystem_hermitian funcion. "
                             "Check that input matrix is really hermitian");
  }
}

#endif // _aer_framework_linalg_eigensystem_hpp_
