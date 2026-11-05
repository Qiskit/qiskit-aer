// Dependencies: BLAS - LAPACK
// Modified: use signed 32-bit integers for the LP64 LAPACK interface.

#ifndef _aer_framework_lapack_protos_hpp
#define _aer_framework_lapack_protos_hpp

#include <array>
#include <complex>
#include <iostream>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// LAPACK SVD function
// https://netlib.org/lapack/explore-html/d3/da8/group__complex16_g_esing_gad6f0c85f3cca2968e1ef901d2b6014ee.html
void zgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             std::complex<double> *a, const int *lda, double *s,
             std::complex<double> *u, const int *ldu, std::complex<double> *vt,
             const int *ldvt, std::complex<double> *work, const int *lwork,
             double *rwork, int *info);

// D&C approach
// https://netlib.org/lapack/explore-html/d3/da8/group__complex16_g_esing_gaccb06ed106ce18814ad7069dcb43aa27.html
void zgesdd_(const char *jobz, const int *m, const int *n,
             std::complex<double> *a, const int *lda, double *s,
             std::complex<double> *u, const int *ldu, std::complex<double> *vt,
             const int *ldvt, std::complex<double> *work, const int *lwork,
             double *rwork, int *iwork, int *info);

#ifdef __cplusplus
}
#endif

#endif // end __lapack_protos_h_
