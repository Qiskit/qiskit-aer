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



#ifndef _qv_qvintrin_hpp_
#define _qv_qvintrin_hpp_

#include <cstdint>
#include <utility>

#if defined(__PPC__) | defined(__PPC64__)

#elif defined(__arm__) || defined(__arm64__)

#else

#if !defined(QV_ISOLATED_AVX) && defined (__AVX2__)
#include <cpuid.h>
// only if x86
#include "qvintrin_avx.hpp"
#endif

#endif

namespace QV {

#ifndef QV_AVX_ISOLATION
bool is_intrinsics () {
#if defined(__PPC__) | defined(__PPC64__)
  return false;
#elif defined(__arm__) || defined(__arm64__)
  return false;
#else //Intel
#ifdef __AVX2__
  return is_avx2_supported();
#else
  return false;
#endif //Intel

#endif
}

bool apply_matrix_opt(
    float* qv_data,
    const uint64_t data_size,
    const uint64_t* qregs,
    const uint64_t qregs_size,
    const float* fmat,
    const uint64_t omp_threads) {
#if defined(__PPC__) | defined(__PPC64__)
  return false;
#elif defined(__arm__) || defined(__arm64__)
  return false;
#else //Intel
#ifdef __AVX2__
  return apply_matrix_avx <float> (
      (void *) qv_data, data_size, qregs, qregs_size, fmat, omp_threads);
#else
  return false;
#endif //Intel

#endif
}

bool apply_matrix_opt(
    double* qv_data,
    const uint64_t data_size,
    const uint64_t* qregs,
    const uint64_t qregs_size,
    const double* dmat,
    const uint64_t omp_threads) {

#if defined(__PPC__) | defined(__PPC64__)
  return false;
#elif defined(__arm__) || defined(__arm64__)
  return false;
#else //Intel
#ifdef __AVX2__
  return apply_matrix_avx <double> (
      qv_data, data_size, qregs, qregs_size, dmat, omp_threads);
#else
  return false;
#endif //Intel

#endif
}
#else
bool is_intrinsics();

bool apply_matrix_opt(
    float* qv_data,
    const uint64_t data_size,
    const uint64_t* qregs,
    const uint64_t qregs_size,
    const float* mat,
    const uint64_t omp_threads);

bool apply_matrix_opt(
    double* qv_data,
    const uint64_t data_size,
    const uint64_t* qregs,
    const uint64_t qregs_size,
    const double* mat,
    const uint64_t omp_threads);
#endif
}
//------------------------------------------------------------------------------
#endif // end module
