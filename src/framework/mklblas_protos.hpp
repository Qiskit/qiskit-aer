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

#define MKL_Complex16 std::complex<double>
#define MKL_Complex8 std::complex<float>
#define MKL_INT size_t
#include <mkl.h>

#ifdef __cplusplus
extern "C" {
#endif

//===========================================================================
// Prototypes for level 3 BLAS
//===========================================================================

// Reduces a Single-Precison Complex Hermitian matrix A to real symmetric
// tridiagonal form
// void chetrd_(char *TRANS, int *N, std::complex<float> *A, int *LDA, float *d,
//              float *e, std::complex<float> *tau, std::complex<float> *work,
//              int *lwork, int *info);
// 
// // Reduces a Double-Precison Complex Hermitian matrix A to real symmetric
// // tridiagonal form T
// void zhetrd_(char *TRANS, int *N, std::complex<double> *A, int *LDA, double *d,
//              double *e, std::complex<double> *tau, std::complex<double> *work,
//              int *lwork, int *info);
// 
// // Computes all eigenvalues and, optionally, eigenvectors of a
// // Single-Precison Complex symmetric positive definite tridiagonal matrix
// void cpteqr_(char *compz, int *n, float *d, float *e, std::complex<float> *z,
//              int *ldz, std::complex<float> *work, int *info);
// 
// // Computes all eigenvalues and, optionally, eigenvectors of a
// // Double-Precison Complex symmetric positive definite tridiagonal matrix
// void zpteqr_(char *compz, int *n, double *d, double *e, std::complex<double> *z,
//              int *ldz, std::complex<double> *work, int *info);

/***********************************************************
 * Function overwrite to keep the same call as in OpenBLAS *
 * *********************************************************/
void sgemv_(const CBLAS_TRANSPOSE *TransA, const size_t *M, const size_t *N,
            const float *alpha, const float *A, const size_t *lda,
            const float *x, const size_t *incx, const float *beta, float *y,
            const size_t *lincy)
{
    return cblas_sgemv(CblasColMajor, *TransA, *M, *N, *alpha, A, *lda, x, *incx, *beta, y, *lincy);
}

void dgemv_(const CBLAS_TRANSPOSE *TransA, const size_t *M, const size_t *N,
            const double *alpha, const double *A, const size_t *lda,
            const double *x, const size_t *incx, const double *beta, double *y,
            const size_t *lincy)
{
    return cblas_dgemv(CblasColMajor, *TransA, *M, *N, *alpha, A, *lda, x, *incx, *beta, y, *lincy);
}

void cgemv_(const CBLAS_TRANSPOSE *TransA, const size_t *M, const size_t *N,
            const std::complex<float> *alpha, const std::complex<float> *A,
            const size_t *lda, const std::complex<float> *x, const size_t *incx,
            const std::complex<float> *beta, std::complex<float> *y,
            const size_t *lincy)
{
    return cblas_cgemv(CblasColMajor, *TransA, *M, *N, alpha, A, *lda, x, *incx, beta, y, *lincy);
}

void zgemv_(const CBLAS_TRANSPOSE *TransA, const size_t *M, const size_t *N,
            const std::complex<double> *alpha, const std::complex<double> *A,
            const size_t *lda, const std::complex<double> *x,
            const size_t *incx, const std::complex<double> *beta,
            std::complex<double> *y, const size_t *lincy)
{
    return cblas_zgemv(CblasColMajor, *TransA, *M, *N, alpha, A, *lda, x, *incx, beta, y, *lincy);
}

void sgemm_(const CBLAS_TRANSPOSE *TransA, const CBLAS_TRANSPOSE *TransB, const size_t *M,
            const size_t *N, const size_t *K, const float *alpha,
            const float *A, const size_t *lda, const float *B,
            const size_t *ldb, const float *beta, float *C, size_t *ldc)
{
    return cblas_sgemm(CblasColMajor, *TransA, *TransB, *M, *N, *K, *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
}

void dgemm_(const CBLAS_TRANSPOSE *TransA, const CBLAS_TRANSPOSE *TransB, const size_t *M,
            const size_t *N, const size_t *K, const double *alpha,
            const double *A, const size_t *lda, const double *B,
            const size_t *ldb, const double *beta, double *C, size_t *ldc)
{
    return cblas_dgemm(CblasColMajor, *TransA, *TransB, *M, *N, *K, *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
}

void cgemm_(const CBLAS_TRANSPOSE *TransA, const CBLAS_TRANSPOSE *TransB, const size_t *M,
            const size_t *N, const size_t *K, const std::complex<float> *alpha,
            const std::complex<float> *A, const size_t *lda,
            const std::complex<float> *B, const size_t *ldb,
            const std::complex<float> *beta, std::complex<float> *C,
            size_t *ldc)
{
    return cblas_cgemm(CblasColMajor, *TransA, *TransB, *M, *N, *K, alpha, A, *lda, B, *ldb, beta, C, *ldc);
}

void zgemm_(const CBLAS_TRANSPOSE *TransA, const CBLAS_TRANSPOSE *TransB, const size_t *M,
            const size_t *N, const size_t *K, const std::complex<double> *alpha,
            const std::complex<double> *A, const size_t *lda,
            const std::complex<double> *B, const size_t *ldb,
            const std::complex<double> *beta, std::complex<double> *C,
            size_t *ldc)
{
    return cblas_zgemm(CblasColMajor, *TransA, *TransB, *M, *N, *K, alpha, A, *lda, B, *ldb, beta, C, *ldc);
}

#ifdef __cplusplus
}
#endif

namespace AerBlas {

// std::array<char, 3> Trans = {'N', 'T', 'C'};
std::array<CBLAS_TRANSPOSE, 3> Trans = {CblasNoTrans, CblasTrans, CblasConjTrans};

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
