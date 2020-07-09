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
#define QV_ISOLATED_AVX
#include "qvintrin.hpp"

#if defined(__PPC__)  || defined(__PPC64__)
// PPC
#elif defined(__arm__) || defined(__arm64__)
// ARM
#else
// INTEL
#ifdef _MSC_VER
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#endif
#include "qvintrin_avx.hpp"
#endif

namespace QV {

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
#if defined(__PPC__)  || defined(__PPC64__)
  return false;
#elif defined(__arm__) || defined(__arm64__)
  return false;
#else //Intel
#ifdef __AVX2__
  return apply_matrix_avx <float> (
      qv_data, data_size, qregs, qregs_size, fmat, omp_threads);
#else
  return false;
#endif // Intel
#endif
}

bool apply_matrix_opt(
    double* qv_data,
    const uint64_t data_size,
    const uint64_t* qregs,
    const uint64_t qregs_size,
    const double* dmat,
    const uint64_t omp_threads) {
#if defined(__PPC__)  || defined(__PPC64__)
  return false;
#elif defined(__arm__) || defined(__arm64__)
  return false;
#else //Intel
#ifdef __AVX2__
  return apply_matrix_avx <double> (
      qv_data, data_size, qregs, qregs_size, dmat, omp_threads);
#else
  return false;
#endif // Intel
#endif
}
}

