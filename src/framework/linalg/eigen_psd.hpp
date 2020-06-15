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
#include "framework/blas_protos.hpp"
#include "framework/matrix.hpp"

/**
 * Returns the eigenvalues and eigenvectors of a Hermitian
 * positive semi-definite (PSD) matrix.
 * Uses the blas functions ?hetrd followed by ?pteqr
 * @param psd_matrix: The Hermitian PSD matrix.
 * @param eigenvalues: On output: the eignevalues of the matrix (input is overwritten)
 * @param eigenvectors: On input: a copy of psd_matrix, on output: the eigenvectors of the matrix.
 *
 * @returns: void
 */
template <class T>
void eigensystem_psd_hetrd(const matrix<std::complex<T>>& psd_matrix,
               /* out */ std::vector<T>& eigenvalues,
               /* out */ matrix<std::complex<T>>& eigenvectors);

/**
 * Returns the eigenvalues and eigenvectors of a Hermitian
 * positive semi-definite (PSD) matrix.
 * Uses the blas function ?heevx
 * @param psd_matrix: The Hermitian PSD matrix.
 * @param eigenvalues: On output: the eignevalues of the matrix (input is overwritten)
 * @param eigenvectors: On input: a copy of psd_matrix, on output: the eigenvectors of the matrix.
 *
 * @returns: void
 */
template <class T>
void eigensystem_psd_heevx(const matrix<std::complex<T>>& psd_matrix,
               /* out */ std::vector<T>& eigenvalues,
               /* out */ matrix<std::complex<T>>& eigenvectors);

/**
 * We define general templates for these functions as no-ops
 *   and use specialization for types which we have ops defined for
 * ** definitely not optimal, I'd rather a compiler error
 *    if someone tried to use this over e.g. int but so it goes **
 */
template <class T>
void eigensystem_psd_hetrd(const matrix<std::complex<T>>& psd_matrix,
               /* out */ std::vector<T>& eigenvalues,
               /* out */ matrix<std::complex<T>>& eigenvectors) {}

template <class T>
void eigensystem_psd_heevx(const matrix<std::complex<T>>& psd_matrix,
               /* out */ std::vector<T>& eigenvalues,
               /* out */ matrix<std::complex<T>>& eigenvectors) {}

/**
 * eigensystem specializations for floats/doubles
 */
template <>
void eigensystem_psd_hetrd(const matrix<std::complex<double>>& psd_matrix,
               /* out */ std::vector<double>& eigenvalues,
               /* out */ matrix<std::complex<double>>& eigenvectors) {
#ifdef DEBUG
  if ( psd_matrix.GetRows() != psd_matrix.GetColumns() ) {
    std::cerr
        << "error: eig_psd initial conditions not satisified" << std::endl
        << "mat not square : " << psd_matrix.GetColumns() << " x " << psd_matrix.GetRows() << std::endl
        << std::endl;
    exit(1);
  }
  if ( psd_matrix.size() != eigenvectors.size() ) {
    std::cerr
         << "error: eig_psd initial conditions not satisfied :" << std::endl
         << "evecs size != mat size (should be a copy)" << std::endl
         << psd_matrix.size() << " != " << eigenvectors.size() << std::endl;
    exit(1);
  }
#endif
  int n = psd_matrix.GetLD();
  char uplo = 'U'; //'L';
  char compz = 'V'; //'N'/'V'/'I'
  int lwork = 18 * psd_matrix.GetRows();
  int info = 0; 
  int lda = n;
  int ldz = lda;

  // no allocation here, we return by reference i.e. this space was already allocated
  std::complex<double> *z = eigenvectors.GetMat(); 
  // allocation here because hetrd needs 2 copies
  matrix<std::complex<double>> hetrd_copy(psd_matrix);
  // memory pointed to (owned) by a will be deallocated by matrix destructor
  std::complex<double> *a { hetrd_copy.GetMat() };
  double               *e { new double[n-1]{0.0} }; 
  double               *d { new double[n  ]{0.0} };

  std::complex<double> *tau   { new std::complex<double>[n]{std::complex<double>{0.0, 0.0} } };
  std::complex<double> *work1 { new std::complex<double>[lwork]{std::complex<double>{0.0, 0.0} } };

  AerBlas::f77::zhetrd(&uplo, &n, reinterpret_cast<__complex__ double*>(a), &lda, d, e, reinterpret_cast<__complex__ double*>(tau), reinterpret_cast<__complex__ double*>(work1), &lwork, &info);

#ifdef DEBUG
  std::cout << "diag_elems: ";
  for(auto v : eigenvalues )
    std::cout << v << ", ";
  std::cout << std::endl;
  // mat must be square, output references must be empty
  std::cout << "zhetrd return: " << info << std::endl;
  if ( info != 0 ) {
    std::cerr << "error: zhetrd returned non-zero exit code : " 
              << info << std::endl;
    info = 0;
  }
#endif

  // we don't allocate all at once to keep peak memory down
  delete[] work1;
  work1 = nullptr;
  double               *work2 { new double[4*n]{0.0 } };

  // Now call cpteqr
  // On exit d contains eigenvalues, z contains eigenvectors
  // *** on exit e has been destroyed ***
  AerBlas::f77::zpteqr(&compz, &n, d, e, reinterpret_cast<__complex__ double*>(z), &ldz, work2, &info);

#ifdef DEBUG
  std::cout << "eigenvalues: ";
  for(auto v : eigenvalues )
    std::cout << v << ", ";
  std::cout << std::endl;
 
  std::cout << "zpteqr return: " << info << std::endl;
  if ( info != 0 ) {
    std::cerr << "error: zpteqr returned non-zero exit code : " 
              << info << std::endl;
    info = 0;
  }
#endif

  // copy d into eigenvalues
  eigenvalues.clear();
  std::copy(&d[0], &d[n], std::back_inserter(eigenvalues));
  // free working memory
  delete[] d;
  d = nullptr;
  delete[] tau;
  tau = nullptr;
  delete[] work2;
  work2 = nullptr;
  // e was deallocated by pteqr
  e = nullptr;
}

template <>
void eigensystem_psd_hetrd(const matrix<std::complex<float>>& psd_matrix,
               /* out */ std::vector<float>& eigenvalues,
               /* out */ matrix<std::complex<float>>& eigenvectors) {
#ifdef DEBUG
  if ( psd_matrix.GetRows() != psd_matrix.GetColumns() ) {
    std::cerr
        << "error: eig_psd initial conditions not satisified" << std::endl
        << "mat not square : " << psd_matrix.GetColumns() << " x " << psd_matrix.GetRows() << std::endl
        << std::endl;
    exit(1);
  }
  if ( psd_matrix.size() != eigenvectors.size() ) {
    std::cerr
         << "error: eig_psd initial conditions not satisfied :" << std::endl
         << "evecs size != mat size (should be a copy)" << std::endl
         << psd_matrix.size() << " != " << eigenvectors.size() << std::endl;
    exit(1);
  }
#endif
  int n = psd_matrix.GetLD();
  char uplo = 'U'; //'L';
  char compz = 'V'; //'N'/'V'/'I'
  int lwork = 18 * psd_matrix.GetRows();
  int info = 0; 
  int lda = n;
  int ldz = lda;

  // no allocation here, we return by reference i.e. this space was already allocated
  std::complex<float> *z = eigenvectors.GetMat(); 
  // allocation here because hetrd needs 2 copies
  matrix<std::complex<float>> hetrd_copy(psd_matrix);
  // memory pointed to (owned) by a will be deallocated by matrix destructor
  std::complex<float> *a { hetrd_copy.GetMat() };
  float               *e { new float[n-1]{0.0} }; 
  float               *d { new float[n  ]{0.0} };

  std::complex<float> *tau   { new std::complex<float>[n]{std::complex<float>{0.0, 0.0} } };
  std::complex<float> *work1 { new std::complex<float>[lwork]{std::complex<float>{0.0, 0.0} } };

  AerBlas::f77::chetrd(&uplo, &n, reinterpret_cast<__complex__ float*>(a), &lda, d, e, reinterpret_cast<__complex__ float*>(tau), reinterpret_cast<__complex__ float*>(work1), &lwork, &info);

#ifdef DEBUG
  std::cout << "diag_elems: ";
  for(auto v : eigenvalues )
    std::cout << v << ", ";
  std::cout << std::endl;
  // mat must be square, output references must be empty
  std::cout << "chetrd return: " << info << std::endl;
  if ( info != 0 ) {
    std::cerr << "error: chetrd returned non-zero exit code : " 
              << info << std::endl;
    info = 0;
  }
#endif

  // we don't allocate all at once to keep peak memory down
  delete[] work1;
  work1 = nullptr;
  float               *work2 { new float[4*n]{0.0 } };

  // Now call cpteqr
  // On exit d contains eigenvalues, z contains eigenvectors
  // *** on exit e has been destroyed ***
  AerBlas::f77::cpteqr(&compz, &n, d, e, reinterpret_cast<__complex__ float*>(z), &ldz, work2, &info);

#ifdef DEBUG
  std::cout << "eigenvalues: ";
  for(auto v : eigenvalues )
    std::cout << v << ", ";
  std::cout << std::endl;
 
  std::cout << "cpteqr return: " << info << std::endl;
  if ( info != 0 ) {
    std::cerr << "error: cpteqr returned non-zero exit code : " 
              << info << std::endl;
    info = 0;
  }
#endif

  // copy d into eigenvalues
  eigenvalues.clear();
  std::copy(&d[0], &d[n], std::back_inserter(eigenvalues));
  // free working memory
  delete[] d;
  d = nullptr;
  delete[] tau;
  tau = nullptr;
  delete[] work2;
  work2 = nullptr;
  // e was deallocated by pteqr
  e = nullptr;
}

template <>
void eigensystem_psd_heevx(const matrix<std::complex<double>>& psd_matrix,
               std::vector<double>& eigenvalues,
               matrix<std::complex<double>>& eigenvectors) {
#ifdef DEBUG
  if ( psd_matrix.GetRows() != psd_matrix.GetColumns() ) {
    std::cerr
        << "error: eig_psd initial conditions not satisified" << std::endl
        << "mat not square : " << psd_matrix.GetColumns() << " x " << psd_matrix.GetRows() << std::endl
        << std::endl;
    exit(1);
  }
  if ( psd_matrix.size() != eigenvectors.size() ) {
    std::cerr
         << "error: eig_psd initial conditions not satisfied :" << std::endl
         << "evecs size != mat size (should be a copy)" << std::endl
         << psd_matrix.size() << " != " << eigenvectors.size() << std::endl;
    exit(1);
  }
#endif
  //JOBZ is CHARACTER*1
  //  = 'N':  Compute eigenvalues only;
  //  = 'V':  Compute eigenvalues and eigenvectors.
  //RANGE is CHARACTER*1
  //  = 'A': all eigenvalues will be found.
  //  = 'V': all eigenvalues in the half-open interval (VL,VU]
  //         will be found.
  //  = 'I': the IL-th through IU-th eigenvalues will be found.
  char jobz{'V'}, range{'A'}, uplo{'U'};
  int n{psd_matrix.GetLD()};
  int ldz{n}, lda{n}, lwork{2*n};
  int il{0}, iu{0}; // not referenced if range='A'
  double vl{0.0}, vu{0.0}; // not referenced if range='A'
  char cmach{'S'};
  double abstol{2.0*AerBlas::f77::dlamch(&cmach)};
  int m{0}; // number of eigenvalues found
  int info{0};

  // z is the matrix of eigenvectors to be returned
  eigenvectors.resize(ldz, n);
  matrix<std::complex<double>> heevx_copy{psd_matrix};
  std::complex<double>  *z{eigenvectors.GetMat()};
  std::complex<double>  *a{heevx_copy.GetMat()};
  std::complex<double>  *work{new std::complex<double>[lwork]{0.0, 0.0}};
  double                *rwork{new double[7*n]{0.0}};
  int                   *iwork{new int[5*n]{0}};
  int                   *ifail{new int[n]{0}};
  double                *w{new double[n]{0.0}};
 
  AerBlas::f77::zheevx(&jobz, &range, &uplo, &n, reinterpret_cast<__complex__ double*>(a), &lda, &vl, &vu, &il, &iu,
         &abstol, &m, w, reinterpret_cast<__complex__ double*>(z), &ldz, reinterpret_cast<__complex__ double*>(work), &lwork, rwork, iwork,
         ifail, &info);

#ifdef DEBUG
  std::cout << "zheevx return: " << info << std::endl;
  if ( info != 0 ) {
    std::cerr << "error: zheevx returned non-zero exit code : "
              << info << std::endl;
        exit(1);
  }
#endif

  // copy w into eigenvalues return
  // z is already a reference to the matrix
  //  of eigenvectors to be returned
  eigenvalues.clear();
  std::copy(&w[0], &w[n], std::back_inserter(eigenvalues));

  // free working memory
  a = nullptr;
  z = nullptr;
  delete[] w;
  w = nullptr;
  delete[] rwork;
  rwork = nullptr;
  delete[] iwork;
  iwork = nullptr;
  delete[] ifail;
  ifail = nullptr;

}

template <>
void eigensystem_psd_heevx(const matrix<std::complex<float>>& psd_matrix,
               std::vector<float>& eigenvalues,
               matrix<std::complex<float>>& eigenvectors) {
#ifdef DEBUG
  if ( psd_matrix.GetRows() != psd_matrix.GetColumns() ) {
    std::cerr
        << "error: eig_psd initial conditions not satisified" << std::endl
        << "mat not square : " << psd_matrix.GetColumns() << " x " << psd_matrix.GetRows() << std::endl
        << std::endl;
    exit(1);
  }
  if ( psd_matrix.size() != eigenvectors.size() ) {
    std::cerr
         << "error: eig_psd initial conditions not satisfied :" << std::endl
         << "evecs size != mat size (should be a copy)" << std::endl
         << psd_matrix.size() << " != " << eigenvectors.size() << std::endl;
    exit(1);
  }
#endif
  //JOBZ is CHARACTER*1
  //  = 'N':  Compute eigenvalues only;
  //  = 'V':  Compute eigenvalues and eigenvectors.
  //RANGE is CHARACTER*1
  //  = 'A': all eigenvalues will be found.
  //  = 'V': all eigenvalues in the half-open interval (VL,VU]
  //         will be found.
  //  = 'I': the IL-th through IU-th eigenvalues will be found.
  char jobz{'V'}, range{'A'}, uplo{'U'};
  int n{psd_matrix.GetLD()};
  int ldz{n}, lda{n}, lwork{2*n};
  int il{0}, iu{0}; // not referenced if range='A'
  float vl{0.0}, vu{0.0}; // not referenced if range='A'
  char cmach{'S'};
  float abstol{static_cast<float>(AerBlas::f77::slamch(&cmach) * 2.0)};
  int m{0}; // number of eigenvalues found
  int info{0};

  // z is the matrix of eigenvectors to be returned
  eigenvectors.resize(ldz, n);
  matrix<std::complex<float>> heevx_copy{psd_matrix};
  std::complex<float>  *z{eigenvectors.GetMat()};
  std::complex<float>  *a{heevx_copy.GetMat()};
  std::complex<float>  *work{new std::complex<float>[lwork]{0.0, 0.0}};
  float                *rwork{new float[7*n]{0.0}};
  int                  *iwork{new int[5*n]{0}};
  int                  *ifail{new int[n]{0}};
  float                *w{new float[n]{0.0}};
 
  AerBlas::f77::cheevx(&jobz, &range, &uplo, &n, reinterpret_cast<__complex__ float*>(a), &lda, &vl, &vu, &il, &iu,
         &abstol, &m, w, reinterpret_cast<__complex__ float*>(z), &ldz, reinterpret_cast<__complex__ float*>(work), &lwork, rwork, iwork,
         ifail, &info);

#ifdef DEBUG
  std::cout << "zheevx return: " << info << std::endl;
  if ( info != 0 ) {
    std::cerr << "error: zheevx returned non-zero exit code : "
              << info << std::endl;
        exit(1);
  }
#endif

  // copy w into eigenvalues return
  // z is already a reference to the matrix
  //  of eigenvectors to be returned
  eigenvalues.clear();
  std::copy(&w[0], &w[n], std::back_inserter(eigenvalues));

  // free working memory
  a = nullptr;
  z = nullptr;
  delete[] w;
  w = nullptr;
  delete[] rwork;
  rwork = nullptr;
  delete[] iwork;
  iwork = nullptr;
  delete[] ifail;
  ifail = nullptr;

}

#endif
