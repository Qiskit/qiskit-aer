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

#ifndef _aer_framework_linalg_eigen_psd_hpp_
#define _aer_framework_linalg_eigen_psd_hpp_

#include <type_traits>
#include "framework/matrix.hpp"

/**
 * Returns the eigenvalues and eigenvectors of a Hermitian
 * positive semi-definite (PSD) matrix.
 * @param psd_matrix: The Hermitian PSD matrix.
 * @param eigenvalues: On output: the eignevalues of the matrix (input is overwritten)
 * @param eigenvectors: On input: a copy of psd_matrix, on output: the eigenvectors of the matrix.
 *
 * @returns: void
 */
template <class T>
void eigensystem_psd(const matrix<std::complex<T>>& psd_matrix,
               /* out */ std::vector<T>& eigenvalues,
               /* out */ matrix<std::complex<T>>& eigenvectors);

//template <>
//void eigensystem_psd(const matrix<std::complex<float>>& psd_matrix,
//               /* out */ std::vector<float>& eigenvalues,
//               /* out */ matrix<std::complex<float>>& eigenvectors); 
//template <>
//void eigensystem_psd(const matrix<std::complex<double>>& psd_matrix,
//               /* out */ std::vector<double>& eigenvalues,
//               /* out */ matrix<std::complex<double>>& eigenvectors); 

template <class T>
void eigensystem_psd(const matrix<std::complex<T>>& psd_matrix,
               /* out */ std::vector<T>& eigenvalues,
               /* out */ matrix<std::complex<T>>& eigenvectors) {}

template <>
void eigensystem_psd(const matrix<std::complex<double>>& psd_matrix,
               /* out */ std::vector<double>& eigenvalues,
               /* out */ matrix<std::complex<double>>& eigenvectors) {
#ifdef DEBUG
  if ( psd_matrix.getRows() != psd_matrix.getCols() ) {
    std::cerr
        << "error: eig_psd initial conditions not satisified" << std::endl
        << "mat not square : " << psd_matrix.getCols() << " x " << psd_matrix.getRows() << std::endl
        << std::endl;
    exit(1);
  }
  if ( psd_matrix.size() != eigenvectors.size() ) {
    std::cerr
         << "error: eig_psd initial conditions not satisfied :" << std::endl;
         << "evecs size != mat size (should be a copy)" << std::endl
         << psd_matrix.size() << " != " << eigenvectors.size() << std::endl;
    exit(1);
  }
#endif
  eigenvalues.resize(psd_matrix.GetLD(), 0.0);
  double *d = eigenvalues.data();
  std::complex<double> *z = eigenvectors.GetMat();
 
  const char uplo = 'U'; //'L';
  size_t n = psd_matrix.GetLD();
  matrix<std::complex<double>> hetrd_copy(psd_matrix);
  std::complex<double>* a = hetrd_copy.GetMat();
  size_t lda = psd_matrix.GetLD();
  std::vector<double_t> e_vec(n-1, 0.0);
  double *e = e_vec.data();
  std::vector<std::complex<double>> tau_vec(n, std::complex<double>{0.0, 0.0});
  std::complex<double> *tau = tau_vec.data();
  size_t lwork = n;
  std::vector<std::complex<double>> hetrd_work_vec(n, std::complex<double>{0.0, 0.0}); 
  std::complex<double> *hetrd_work = hetrd_work_vec.data(); 
  int hetrd_info = 0;

  zhetrd_(&uplo, &n, a, &lda, d, e, tau, hetrd_work, &lwork, &hetrd_info);

#ifdef DEBUG
  // mat must be square, output references must be empty
  std::cout << "chetrd return: " << hetrd_info << std::endl;
  if ( hetrd_info != 0 ) {
    std::cerr << "error: chetrd returned non-zero exit code : " 
              << chetrd_info << std::endl;
        exit(1);
  }
#endif
 
  // Now call cpteqr
  const char compz = 'V'; //'N'/'V'/'I'
  size_t ldz = lda;
  std::vector<double> pteqr_work_vec(4*n, 0.0); 
  double *pteqr_work = pteqr_work_vec.data();
  int pteqr_info = 0;

  // On exit d contains eigenvalues, z contains eigenvectors
  // *** on exit e has been destroyed ***
  zpteqr_(&compz, &n, d, e, z, &ldz, pteqr_work, &pteqr_info);

#ifdef DEBUG
  std::cout << "cpteqr return: " << pteqr_info << std::endl;
  if ( pteqr_info != 0 ) {
    std::cerr << "error: cpteqr returned non-zero exit code : " 
              << pteqr_info << std::endl;
        exit(1);
  }
#endif
}
template <>
void eigensystem_psd(const matrix<std::complex<float>>& psd_matrix,
               /* out */ std::vector<float>& eigenvalues,
               /* out */ matrix<std::complex<float>>& eigenvectors) {
#ifdef DEBUG
  if ( psd_matrix.getRows() != psd_matrix.getCols() ) {
    std::cerr
        << "error: eig_psd initial conditions not satisified" << std::endl
        << "mat not square : " << psd_matrix.getCols() << " x " << psd_matrix.getRows() << std::endl
        << std::endl;
    exit(1);
  }
  if ( psd_matrix.size() != eigenvectors.size() ) {
    std::cerr
         << "error: eig_psd initial conditions not satisfied :" << std::endl;
         << "evecs size != mat size (should be a copy)" << std::endl
         << psd_matrix.size() << " != " << eigenvectors.size() << std::endl;
  }
#endif
  eigenvalues.resize(psd_matrix.GetLD(), 0.0);
  float *d = eigenvalues.data();
  std::complex<float> *z = eigenvectors.GetMat();
 
  const char uplo = 'U'; //'L';
  size_t n = psd_matrix.GetLD();
  matrix<std::complex<float>> hetrd_copy(psd_matrix);
  std::complex<float>* a = hetrd_copy.GetMat();
  size_t lda = psd_matrix.GetLD();
  std::vector<float_t> e_vec(n-1, 0.0);
  float *e = e_vec.data();
  std::vector<std::complex<float>> tau_vec(n, std::complex<float>{0.0, 0.0});
  std::complex<float> *tau = tau_vec.data();
  size_t lwork = n;
  std::vector<std::complex<float>> hetrd_work_vec(n, std::complex<float>{0.0, 0.0}); 
  std::complex<float> *hetrd_work = hetrd_work_vec.data(); 
  int hetrd_info = 0;

  chetrd_(&uplo, &n, a, &lda, d, e, tau, hetrd_work, &lwork, &hetrd_info);

#ifdef DEBUG
  // mat must be square, output references must be empty
  std::cout << "chetrd return: " << hetrd_info << std::endl;
  if ( hetrd_info != 0 ) {
    std::cerr << "error: chetrd returned non-zero exit code : " 
              << chetrd_info << std::endl;
        exit(1);
  }
#endif
 
  // Now call cpteqr
  const char compz = 'V'; //'N'/'V'/'I'
  size_t ldz = lda;
  std::vector<float> pteqr_work_vec(4*n, 0.0); 
  float *pteqr_work = pteqr_work_vec.data();
  int pteqr_info = 0;

  // On exit d contains eigenvalues, z contains eigenvectors
  // *** on exit e has been destroyed ***
  cpteqr_(&compz, &n, d, e, z, &ldz, pteqr_work, &pteqr_info);

#ifdef DEBUG
  std::cout << "cpteqr return: " << pteqr_info << std::endl;
  if ( pteqr_info != 0 ) {
    std::cerr << "error: cpteqr returned non-zero exit code : " 
              << pteqr_info << std::endl;
        exit(1);
  }
#endif
}

#endif
