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

// Dependencies: BLAS
// These are the declarations for the various high-performance matrix routines
// used by the matrix class. An openblas install is required.

#ifndef _aer_framework_blas_protos_hpp
#define _aer_framework_blas_protos_hpp

#include <array>
#include <complex>
#include <iostream>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

//===========================================================================
// Prototypes for level 3 BLAS
//===========================================================================

// Single-Precison Real Matrix-Vector Multiplcation
void sgemv_(const char *TransA, const size_t *M, const size_t *N,
            const float *alpha, const float *A, const size_t *lda,
            const float *x, const size_t *incx, const float *beta, float *y,
            const size_t *lincy);
// Double-Precison Real Matrix-Vector Multiplcation
void dgemv_(const char *TransA, const size_t *M, const size_t *N,
            const double *alpha, const double *A, const size_t *lda,
            const double *x, const size_t *incx, const double *beta, double *y,
            const size_t *lincy);
// Single-Precison Complex Matrix-Vector Multiplcation
void cgemv_(const char *TransA, const size_t *M, const size_t *N,
            const std::complex<float> *alpha, const std::complex<float> *A,
            const size_t *lda, const std::complex<float> *x, const size_t *incx,
            const std::complex<float> *beta, std::complex<float> *y,
            const size_t *lincy);
// Double-Precison Real Matrix-Vector Multiplcation
void zgemv_(const char *TransA, const size_t *M, const size_t *N,
            const std::complex<double> *alpha, const std::complex<double> *A,
            const size_t *lda, const std::complex<double> *x,
            const size_t *incx, const std::complex<double> *beta,
            std::complex<double> *y, const size_t *lincy);
// Single-Precison Real Matrix-Matrix Multiplcation
void sgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const float *alpha,
            const float *A, const size_t *lda, const float *B,
            const size_t *lba, const float *beta, float *C, size_t *ldc);
// Double-Precison Real Matrix-Matrix Multiplcation
void dgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const double *alpha,
            const double *A, const size_t *lda, const double *B,
            const size_t *lba, const double *beta, double *C, size_t *ldc);
// Single-Precison Complex Matrix-Matrix Multiplcation
void cgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const std::complex<float> *alpha,
            const std::complex<float> *A, const size_t *lda,
            const std::complex<float> *B, const size_t *ldb,
            const std::complex<float> *beta, std::complex<float> *C,
            size_t *ldc);
// Double-Precison Complex Matrix-Matrix Multiplcation
void zgemm_(const char *TransA, const char *TransB, const size_t *M,
            const size_t *N, const size_t *K, const std::complex<double> *alpha,
            const std::complex<double> *A, const size_t *lda,
            const std::complex<double> *B, const size_t *ldb,
            const std::complex<double> *beta, std::complex<double> *C,
            size_t *ldc);

// Reduces a Single-Precison Complex Hermitian matrix A to real symmetric
// tridiagonal form
void chetrd_(char *TRANS, int *N, std::complex<float> *A, int *LDA, float *d,
             float *e, std::complex<float> *tau, std::complex<float> *work,
             int *lwork, int *info);

// Reduces a Double-Precison Complex Hermitian matrix A to real symmetric
// tridiagonal form T
void zhetrd_(char *TRANS, int *N, std::complex<double> *A, int *LDA, double *d,
             double *e, std::complex<double> *tau, std::complex<double> *work,
             int *lwork, int *info);

// Computes all eigenvalues and, optionally, eigenvectors of a
// Single-Precison Complex symmetric positive definite tridiagonal matrix
void cpteqr_(char *compz, int *n, float *d, float *e, std::complex<float> *z,
             int *ldz, std::complex<float> *work, int *info);

// Computes all eigenvalues and, optionally, eigenvectors of a
// Double-Precison Complex symmetric positive definite tridiagonal matrix
void zpteqr_(char *compz, int *n, double *d, double *e, std::complex<double> *z,
             int *ldz, std::complex<double> *work, int *info);

// Computes selected eigenvalues and, optionally, eigenvectors
// of a Single-Precison Complex Hermitian matrix A
void cheevx_(char *jobz, char *range, char *uplo, int *n,
             std::complex<float> *a, int *lda, float *vl, float *vu, int *il,
             int *iu, float *abstol, int *m, float *w, std::complex<float> *z,
             int *ldz, std::complex<float> *work, int *lwork, float *rwork,
             int *iwork, int *ifail, int *info);

// Computes selected eigenvalues and, optionally, eigenvectors
// of a Double-Precison Complex Hermitian matrix A
void zheevx_(char *jobz, char *range, char *uplo, int *n,
             std::complex<double> *a, int *lda, double *vl, double *vu, int *il,
             int *iu, double *abstol, int *m, double *w,
             std::complex<double> *z, int *ldz, std::complex<double> *work,
             int *lwork, double *rwork, int *iwork, int *ifail, int *info);

// Determines Single-Precision machine parameters.
float slamch_(char *cmach);

// Determines Double-Precision machine parameters.
double dlamch_(char *cmach);

#ifdef __cplusplus
}
#endif

namespace AerBlas {

std::array<char, 3> Trans = {'N', 'T', 'C'};
/*  Trans (input) CHARACTER*1.
                On entry, TRANSA specifies the form of op( A ) to be used in the
   matrix multiplication as follows:
                        = 'N' no transpose;
                        = 'T' transpose of A;
                        = 'C' hermitian conjugate of A.
*/
std::array<char, 2> UpLo = {'U', 'L'};
/*  UpLo    (input) CHARACTER*1
                        = 'U':  Upper triangle of A is stored;
                        = 'L':  Lower triangle of A is stored.
*/
std::array<char, 2> Jobz = {'V', 'N'};
/*  Jobz    (input) CHARACTER*1
                        = 'N':  Compute eigenvalues only;
                        = 'V':  Compute eigenvalues and eigenvectors.
*/
std::array<char, 3> Range = {'A', 'V', 'I'};
/*  Range   (input) CHARACTER*1
                                = 'A': all eigenvalues will be found.
                                = 'V': all eigenvalues in the half-open interval
   (VL,VU] will be found.
                                = 'I': the IL-th through IU-th eigenvalues will
   be found.
*/

} // namespace AerBlas

#endif // end _blas_protos_h_
