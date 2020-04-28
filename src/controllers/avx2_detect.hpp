/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_controller_avx2_detect_hpp_
#define _aer_controller_avx2_detect_hpp_

#ifdef _WIN64
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#endif

namespace AER {

#ifdef __GNUC__
static void get_cpuid(void* p) {
  int* a = (int*) p;
  __cpuid(1, a[0], a[1], a[2], a[3]);
}
static void get_cpuid_count(void* p) {
  int* a = (int*) p;
  __cpuid_count(0x00000007, 0, a[0], a[1], a[2], a[3]);
}
#endif

inline bool is_avx2_supported() {
#if defined(__GNUC__)
  int info[4] = {0};
  get_cpuid(info);
  bool fma = (info[2] >> 12 & 1);
  bool avx = (info[2] >> 28 & 1);
  if (!fma || !avx)
    return false;
  get_cpuid_count(info);
  bool avx2 = (info[1] >> 5 & 1);
  return avx2;
#else
  return false;
#endif
}

// end namespace AER
}
#endif


