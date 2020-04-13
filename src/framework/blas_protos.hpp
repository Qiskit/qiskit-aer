/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

/*
Dependencies: BLAS
These are the c++ wrappers for the various high-performance matrix routines 
used by the matrix class. A lapack/blas/mkl install is required to compile 
and link these headers.
*/

#ifndef _aer_framework_blas_protos_hpp
#define _aer_framework_blas_protos_hpp

#include <complex>
#include <iostream>
#include <vector>
#include <array>

#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102
//#include "lapacke.h"

/*******************************************************************************
 *
 * BLAS headers
 *
 ******************************************************************************/

const std::array<char, 3> Trans = {'N', 'T', 'C'};
/*  Trans (input) CHARACTER*1.
                On entry, TRANSA specifies the form of op( A ) to be used in the
   matrix multiplication as follows:
                        = 'N' no transpose;
                        = 'T' transpose of A;
                        = 'C' hermitian conjugate of A.
*/
const std::array<char, 2> UpLo = {'U', 'L'};
/*  UpLo    (input) CHARACTER*1
                        = 'U':  Upper triangle of A is stored;
                        = 'L':  Lower triangle of A is stored.
*/
const std::array<char, 2> Jobz = {'V', 'N'};
/*  Jobz    (input) CHARACTER*1
                        = 'N':  Compute eigenvalues only;
                        = 'V':  Compute eigenvalues and eigenvectors.
*/
const std::array<char, 3> Range = {'A', 'V', 'I'};
/*  Range   (input) CHARACTER*1
                                = 'A': all eigenvalues will be found.
                                = 'V': all eigenvalues in the half-open interval
   (VL,VU] will be found.
                                = 'I': the IL-th through IU-th eigenvalues will
   be found.
*/

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

// Reduces a complex Hermitian matrix A to symmetric tridiagonal form T by a unitary similarity transformation
void chetrd_(const char *uplo, const size_t *n, 
             std::complex<float> *a, size_t *lda, 
             float *d, float *e, std::complex<float> *tau, std::complex<float> *work,
             const size_t *lwork, int *info);
void zhetrd_(const char *uplo, const size_t *n,
             std::complex<double> *a, size_t *lda, 
             double *d, double *e, std::complex<double> *tau, std::complex<double> *work,
             const size_t *lwork, int *info);
// Calculates the eigenvectors/values of a real symmetric positive-definite tridiagonal matrix T
void cpteqr_(const char *compz, const size_t *n,
             float *d, float *e, std::complex<float>* z,
             size_t *ldz, float* work, int *info );
void zpteqr_(const char *compz, const size_t *n,
             double *d, double *e, std::complex<double>* z,
             size_t *ldz, double* work, int *info );

// computes the eigenvalues and, optionally, the left and/or right eigenvectors for HE matrices
void cheevx(const char* jobz, const char* range, const char* uplo,
            const size_t* n, std::complex<float>* a, const size_t* lda,
            const float* vl, const float* vu, const size_t* il, const size_t* iu,
            const float* abstol, const size_t* m, 

#ifdef __cplusplus
}
#endif

//------------------------------------------------------------------------------
// end _blas_protos_h_
//------------------------------------------------------------------------------
#endif
