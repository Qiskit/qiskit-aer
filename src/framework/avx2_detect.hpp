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

#include <array>
#include <vector>
#include <bitset>

#include "misc/common_macros.hpp"
#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(GNUC_AVX2)
    #include <cpuid.h>
#endif


namespace {
inline void ccpuid(int cpu_info[4], int function_id){
#if defined(_MSC_VER)
  __cpuid(cpu_info, function_id);
#elif defined(GNUC_AVX2)
  __cpuid(function_id,
    cpu_info[0],
    cpu_info[1],
    cpu_info[2],
    cpu_info[3]);
#else // We don't support this platform intrinsics
  cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
#endif
}

inline void cpuidex(int cpu_info[4], int function_id, int subfunction_id){
#if defined(_MSC_VER)
  __cpuidex(cpu_info, function_id, subfunction_id);
#elif defined(GNUC_AVX2)
  __cpuid_count(function_id, subfunction_id, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#else // We don't support this platform intrinsics
   cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
#endif
}
}

namespace AER {

inline bool is_avx2_supported(){
#if defined(GNUC_AVX2) || defined(_MSC_VER)
  static bool cached = false;
  static bool is_supported = false;
  if(cached)
    return is_supported;

  std::array<int, 4> cpui;
  ccpuid(cpui.data(), 0);
  auto num_ids = cpui[0];
  if(num_ids < 7){
    cached = true;
    is_supported = false;
    return false;
  }

  std::vector<std::array<int, 4>> data;
  for (int i = 0; i <= num_ids; ++i){
    cpuidex(cpui.data(), i, 0);
    data.push_back(cpui);
  }

  std::bitset<32> f_1_ECX = data[1][2];
  std::bitset<32> f_7_EBX = data[7][1];

  bool is_fma_supported = (f_1_ECX[12] & 1);
  bool is_avx2_supported = (f_7_EBX[5] & 1);

  cached = true;
  is_supported = is_fma_supported && is_avx2_supported;
  return is_supported;
#else
    return false;
#endif
}
// end namespace AER
}
#endif


