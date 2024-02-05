// Dependencies: Intel MKL
// These are the declarations for the various high-performance matrix routines
// used by the matrix class. An Intel MKL install is required.

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

/***********************************************************
 * Function overwrite to keep the same call as in OpenBLAS *
 * *********************************************************/
void sgemv_(const CBLAS_TRANSPOSE *TransA, const size_t *M, const size_t *N,
            const float *alpha, const float *A, const size_t *lda,
            const float *x, const size_t *incx, const float *beta, float *y,
            const size_t *lincy) {
  return cblas_sgemv(CblasColMajor, *TransA, *M, *N, *alpha, A, *lda, x, *incx,
                     *beta, y, *lincy);
}

void dgemv_(const CBLAS_TRANSPOSE *TransA, const size_t *M, const size_t *N,
            const double *alpha, const double *A, const size_t *lda,
            const double *x, const size_t *incx, const double *beta, double *y,
            const size_t *lincy) {
  return cblas_dgemv(CblasColMajor, *TransA, *M, *N, *alpha, A, *lda, x, *incx,
                     *beta, y, *lincy);
}

void cgemv_(const CBLAS_TRANSPOSE *TransA, const size_t *M, const size_t *N,
            const std::complex<float> *alpha, const std::complex<float> *A,
            const size_t *lda, const std::complex<float> *x, const size_t *incx,
            const std::complex<float> *beta, std::complex<float> *y,
            const size_t *lincy) {
  return cblas_cgemv(CblasColMajor, *TransA, *M, *N, alpha, A, *lda, x, *incx,
                     beta, y, *lincy);
}

void zgemv_(const CBLAS_TRANSPOSE *TransA, const size_t *M, const size_t *N,
            const std::complex<double> *alpha, const std::complex<double> *A,
            const size_t *lda, const std::complex<double> *x,
            const size_t *incx, const std::complex<double> *beta,
            std::complex<double> *y, const size_t *lincy) {
  return cblas_zgemv(CblasColMajor, *TransA, *M, *N, alpha, A, *lda, x, *incx,
                     beta, y, *lincy);
}

void sgemm_(const CBLAS_TRANSPOSE *TransA, const CBLAS_TRANSPOSE *TransB,
            const size_t *M, const size_t *N, const size_t *K,
            const float *alpha, const float *A, const size_t *lda,
            const float *B, const size_t *ldb, const float *beta, float *C,
            size_t *ldc) {
  return cblas_sgemm(CblasColMajor, *TransA, *TransB, *M, *N, *K, *alpha, A,
                     *lda, B, *ldb, *beta, C, *ldc);
}

void dgemm_(const CBLAS_TRANSPOSE *TransA, const CBLAS_TRANSPOSE *TransB,
            const size_t *M, const size_t *N, const size_t *K,
            const double *alpha, const double *A, const size_t *lda,
            const double *B, const size_t *ldb, const double *beta, double *C,
            size_t *ldc) {
  return cblas_dgemm(CblasColMajor, *TransA, *TransB, *M, *N, *K, *alpha, A,
                     *lda, B, *ldb, *beta, C, *ldc);
}

void cgemm_(const CBLAS_TRANSPOSE *TransA, const CBLAS_TRANSPOSE *TransB,
            const size_t *M, const size_t *N, const size_t *K,
            const std::complex<float> *alpha, const std::complex<float> *A,
            const size_t *lda, const std::complex<float> *B, const size_t *ldb,
            const std::complex<float> *beta, std::complex<float> *C,
            size_t *ldc) {
  return cblas_cgemm(CblasColMajor, *TransA, *TransB, *M, *N, *K, alpha, A,
                     *lda, B, *ldb, beta, C, *ldc);
}

void zgemm_(const CBLAS_TRANSPOSE *TransA, const CBLAS_TRANSPOSE *TransB,
            const size_t *M, const size_t *N, const size_t *K,
            const std::complex<double> *alpha, const std::complex<double> *A,
            const size_t *lda, const std::complex<double> *B, const size_t *ldb,
            const std::complex<double> *beta, std::complex<double> *C,
            size_t *ldc) {
  return cblas_zgemm(CblasColMajor, *TransA, *TransB, *M, *N, *K, alpha, A,
                     *lda, B, *ldb, beta, C, *ldc);
}

#ifdef __cplusplus
}
#endif

namespace AerBlas {

// std::array<char, 3> Trans = {'N', 'T', 'C'};
std::array<CBLAS_TRANSPOSE, 3> Trans = {CblasNoTrans, CblasTrans,
                                        CblasConjTrans};

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
