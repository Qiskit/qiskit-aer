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

//#include "openblas/cblas.h"
#include "openblas/f77blas.h"

/*
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>

#include "openblas/lapacke.h"
*/

#define ALIAS_FUNCTION(OriginalnamE, AliasnamE) \
template <typename... Args> \
inline auto AliasnamE(Args&&... args) -> decltype(OriginalnamE(std::forward<Args>(args)...)) { \
  return OriginalnamE(std::forward<Args>(args)...); \
}

namespace AerBlas {

std::array<char, 3> Trans = {'N', 'T', 'C'};
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

namespace f77 {
    ALIAS_FUNCTION(sgemv_, sgemv);
    ALIAS_FUNCTION(dgemv_, dgemv);
    ALIAS_FUNCTION(cgemv_, cgemv);
    ALIAS_FUNCTION(zgemv_, zgemv);
    ALIAS_FUNCTION(sgemm_, sgemm);
    ALIAS_FUNCTION(dgemm_, dgemm);
    ALIAS_FUNCTION(cgemm_, cgemm);
    ALIAS_FUNCTION(zgemm_, zgemm);
 
    ALIAS_FUNCTION(chetrd_, chetrd);
    ALIAS_FUNCTION(zhetrd_, zhetrd);
    ALIAS_FUNCTION(cpteqr_, cpteqr);
    ALIAS_FUNCTION(zpteqr_, zpteqr);
    ALIAS_FUNCTION(cheevx_, cheevx);
    ALIAS_FUNCTION(zheevx_, zheevx);
    ALIAS_FUNCTION(slamch_, slamch);
    ALIAS_FUNCTION(dlamch_, dlamch);
} // namespace f77

/*
namespace lapack {
    ALIAS_FUNCTION(LAPACKE_chetrd, chetrd);
    ALIAS_FUNCTION(LAPACKE_zhetrd, zhetrd);
    ALIAS_FUNCTION(LAPACKE_cpteqr, cpteqr);
    ALIAS_FUNCTION(LAPACKE_zpteqr, zpteqr);
    ALIAS_FUNCTION(LAPACKE_cheevx, cheevx);
    ALIAS_FUNCTION(LAPACKE_zheevx, zheevx);
    ALIAS_FUNCTION(LAPACKE_slamch, slamch);
    ALIAS_FUNCTION(LAPACKE_dlamch, dlamch);
} // namespace lapack
*/

} // namespace AerBlas

//------------------------------------------------------------------------------
// end _blas_protos_h_
//------------------------------------------------------------------------------
#endif
