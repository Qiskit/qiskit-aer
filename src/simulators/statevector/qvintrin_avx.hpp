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

#ifndef _qv_intrinsics_avx_hpp_
#define _qv_intrinsics_avx_hpp_

#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <memory>

#ifdef _WIN64
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#endif

namespace QV {

using reg_t = std::vector<uint64_t>;
template<size_t N> using areg_t = std::array<uint64_t, N>;
using indexes_t = std::unique_ptr<uint64_t[]>;

#define _mm_complex_multiply_pd(real_ret, imag_ret, real_left, imag_left, real_right, imag_right)\
{\
  real_ret = _mm256_mul_pd(real_left, real_right);\
  imag_ret = _mm256_mul_pd(real_left, imag_right);\
  real_ret = _mm256_fnmadd_pd(imag_left, imag_right, real_ret);\
  imag_ret = _mm256_fmadd_pd(imag_left, real_right, imag_ret);\
}

#define _mm_complex_multiply_add_pd(real_ret, imag_ret, real_left, imag_left, real_right, imag_right)\
{\
  real_ret = _mm256_fmadd_pd(real_left, real_right, real_ret);\
  imag_ret = _mm256_fmadd_pd(real_left, imag_right, imag_ret);\
  real_ret = _mm256_fnmadd_pd(imag_left, imag_right, real_ret);\
  imag_ret = _mm256_fmadd_pd(imag_left, real_right, imag_ret);\
}

#define _mm_complex_inner_product_pd(\
    dim, vreals, vimags, cmplxs, vret_real, vret_imag, vtmp_0, vtmp_1)\
{\
  vtmp_0 = _mm256_set1_pd(cmplxs[0]);\
  vtmp_1 = _mm256_set1_pd(cmplxs[1]);\
  _mm_complex_multiply_pd(vret_real, vret_imag, vreals[0], vimags[0], vtmp_0, vtmp_1);\
  for (unsigned ____i = 1; ____i < dim; ++____i) {\
    vtmp_0 = _mm256_set1_pd(cmplxs[____i * 2]);\
    vtmp_1 = _mm256_set1_pd(cmplxs[____i * 2 + 1]);\
    _mm_complex_multiply_add_pd(vret_real, vret_imag, vreals[____i], vimags[____i], vtmp_0, vtmp_1);\
  }\
}

#define _mm_complex_multiply_ps(real_ret, imag_ret, real_left, imag_left, real_right, imag_right)\
{\
  real_ret = _mm256_mul_ps(real_left, real_right);\
  imag_ret = _mm256_mul_ps(real_left, imag_right);\
  real_ret = _mm256_fnmadd_ps(imag_left, imag_right, real_ret);\
  imag_ret = _mm256_fmadd_ps(imag_left, real_right, imag_ret);\
}

#define _mm_complex_multiply_add_ps(real_ret, imag_ret, real_left, imag_left, real_right, imag_right)\
{\
  real_ret = _mm256_fmadd_ps(real_left, real_right, real_ret);\
  imag_ret = _mm256_fmadd_ps(real_left, imag_right, imag_ret);\
  real_ret = _mm256_fnmadd_ps(imag_left, imag_right, real_ret);\
  imag_ret = _mm256_fmadd_ps(imag_left, real_right, imag_ret);\
}

#define _mm_complex_inner_product_ps(\
    dim, vreals, vimags, cmplxs, vret_real, vret_imag, vtmp_0, vtmp_1)\
{\
  vtmp_0 = _mm256_set1_ps(cmplxs[0]);\
  vtmp_1 = _mm256_set1_ps(cmplxs[1]);\
  _mm_complex_multiply_ps(vret_real, vret_imag, vreals[0], vimags[0], vtmp_0, vtmp_1);\
  for (unsigned ____i = 1; ____i < dim; ++____i) {\
    vtmp_0 = _mm256_set1_ps(cmplxs[____i * 2]);\
    vtmp_1 = _mm256_set1_ps(cmplxs[____i * 2 + 1]);\
    _mm_complex_multiply_add_ps(vret_real, vret_imag, vreals[____i], vimags[____i], vtmp_0, vtmp_1);\
  }\
}

#define _mm_load_twoarray_complex_ps(real_addr_0, imag_addr_1, real_ret, imag_ret)\
{\
  real_ret = _mm256_load_ps(real_addr_0);\
  imag_ret = _mm256_load_ps(imag_addr_1);\
}

#define _mm_store_twoarray_complex_ps(real_ret, imag_ret, cmplx_addr_0, cmplx_addr_1)\
{\
  _mm256_store_ps(cmplx_addr_0, real_ret);\
  _mm256_store_ps(cmplx_addr_1, imag_ret);\
}

#define _mm_altload_complex_ps(cmplx_addr_0, cmplx_addr_1, real_ret, imag_ret)\
{\
  real_ret = _mm256_load_ps(cmplx_addr_0);\
  imag_ret = _mm256_load_ps(cmplx_addr_1);\
  tmp0 = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));\
  tmp1 = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));\
  real_ret = _mm256_blend_ps(real_ret, tmp1, 0b10101010);\
  imag_ret = _mm256_blend_ps(tmp0, imag_ret, 0b10101010);\
  real_ret = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));\
  imag_ret = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));\
}

#define _mm_altstore_complex_ps(real_ret, imag_ret, cmplx_addr_0, cmplx_addr_1)\
{\
  real_ret = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));\
  imag_ret = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));\
  tmp0 = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));\
  tmp1 = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));\
  real_ret = _mm256_blend_ps(real_ret, tmp1, 0b10101010);\
  imag_ret = _mm256_blend_ps(tmp0, imag_ret, 0b10101010);\
  _mm256_store_ps(cmplx_addr_0, real_ret);\
  _mm256_store_ps(cmplx_addr_1, imag_ret);\
}

#define _mm_altload_complex_pd(cmplx_addr_0, cmplx_addr_1, real_ret, imag_ret)\
{\
  real_ret = _mm256_load_pd(cmplx_addr_0);\
  imag_ret = _mm256_load_pd(cmplx_addr_1);\
  tmp0 = _mm256_permute4x64_pd(real_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);\
  tmp1 = _mm256_permute4x64_pd(imag_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);\
  real_ret = _mm256_blend_pd(real_ret, tmp1, 0b1010);\
  imag_ret = _mm256_blend_pd(tmp0, imag_ret, 0b1010);\
  real_ret = _mm256_permute4x64_pd(real_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);\
  imag_ret = _mm256_permute4x64_pd(imag_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);\
}

#define _mm_load_twoarray_complex_pd(real_addr_0, imag_addr_1, real_ret, imag_ret)\
{\
  real_ret = _mm256_load_pd(real_addr_0);\
  imag_ret = _mm256_load_pd(imag_addr_1);\
}

#define _mm_store_twoarray_complex_pd(real_ret, imag_ret, cmplx_addr_0, cmplx_addr_1)\
{\
  _mm256_store_pd(cmplx_addr_0, real_ret);\
  _mm256_store_pd(cmplx_addr_1, imag_ret);\
}

#define _mm_altstore_complex_pd(real_ret, imag_ret, cmplx_addr_0, cmplx_addr_1)\
{\
  real_ret = _mm256_permute4x64_pd(real_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);\
  imag_ret = _mm256_permute4x64_pd(imag_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);\
  tmp0 = _mm256_permute4x64_pd(real_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);\
  tmp1 = _mm256_permute4x64_pd(imag_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);\
  real_ret = _mm256_blend_pd(real_ret, tmp1, 0b1010);\
  imag_ret = _mm256_blend_pd(tmp0, imag_ret, 0b1010);\
  _mm256_store_pd(cmplx_addr_0, real_ret);\
  _mm256_store_pd(cmplx_addr_1, imag_ret);\
}

#define perm_d_q0q1_0   3 * 64 + 2 * 16 + 0 * 4 + 1 * 1
#define perm_d_q0q1_1   3 * 64 + 0 * 16 + 1 * 4 + 2 * 1
#define perm_d_q0q1_2   0 * 64 + 2 * 16 + 1 * 4 + 3 * 1
#define perm_d_q0       2 * 64 + 3 * 16 + 0 * 4 + 1 * 1
#define perm_d_q1       1 * 64 + 0 * 16 + 3 * 4 + 2 * 1

template<size_t N>
inline void _apply_matrix_float_avx_q0q1q2(  //
    void* reals, //
    void* imags, //
    void* mat, //
    const areg_t<1ULL << N> &inds, //
    const areg_t<N> qregs //
    ) {

  __m256i masks[7];
  __m256 real_ret, imag_ret, real_ret1, imag_ret1;
  __m256 vreals[1ULL << N], vimags[1ULL << N];
  __m256 tmp0, tmp1;

  float* freals = (float*) reals;
  float* fimags = (float*) imags;
  float* fmat = (float*) mat;

  masks[0] = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 0, 1);
  masks[1] = _mm256_set_epi32(7, 6, 5, 4, 3, 0, 1, 2);
  masks[2] = _mm256_set_epi32(7, 6, 5, 4, 0, 2, 1, 3);
  masks[3] = _mm256_set_epi32(7, 6, 5, 0, 3, 2, 1, 4);
  masks[4] = _mm256_set_epi32(7, 6, 0, 4, 3, 2, 1, 5);
  masks[5] = _mm256_set_epi32(7, 0, 5, 4, 3, 2, 1, 6);
  masks[6] = _mm256_set_epi32(0, 6, 5, 4, 3, 2, 1, 7);

  for (unsigned i = 0; i < (1ULL << N); i += 8) {
    auto idx = inds[i];
    if (freals == fimags) {
      _mm_altload_complex_ps(&freals[idx * 2], &freals[idx * 2 + 8], vreals[i], vimags[i]);
    } else {
      _mm_load_twoarray_complex_ps(&freals[idx], &fimags[idx], vreals[i], vimags[i]);
    }

    for (unsigned j = 1; j < 8; ++j) {
      vreals[i + j] = _mm256_permutevar8x32_ps(vreals[i], masks[j - 1]);
      vimags[i + j] = _mm256_permutevar8x32_ps(vimags[i], masks[j - 1]);
    }
  }

  unsigned midx = 0;
  for (unsigned i = 0; i < inds.size(); i += 8) {
    auto idx = inds[i];
    _mm_complex_inner_product_ps((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    for (unsigned j = 1; j < 8; ++j) {
      _mm_complex_inner_product_ps((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
      midx += (1ULL << (N + 1));

      real_ret1 = _mm256_permutevar8x32_ps(real_ret1, masks[j - 1]);
      imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, masks[j - 1]);

      switch (j) {
      case 1:
        real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00000010);
        imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00000010);
        break;
      case 2:
        real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00000100);
        imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00000100);
        break;
      case 3:
        real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00001000);
        imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00001000);
        break;
      case 4:
        real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00010000);
        imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00010000);
        break;
      case 5:
        real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00100000);
        imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00100000);
        break;
      case 6:
        real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b01000000);
        imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b01000000);
        break;
      case 7:
        real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b10000000);
        imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b10000000);
        break;
      }
    }

    if (freals == fimags) {
      _mm_altstore_complex_ps(real_ret, imag_ret, &freals[idx * 2], &freals[idx * 2 + 8]);
    } else {
      _mm_store_twoarray_complex_ps(real_ret, imag_ret, &freals[idx], &fimags[idx]);
    }
  }
}

template<size_t N>
inline void _apply_matrix_float_avx_qLqL( //
    void* reals, //
    void* imags, //
    void* mat, //
    const areg_t<1ULL << N> &inds, //
    const areg_t<N> qregs //
    ) {

  __m256i masks[3];
  __m256 real_ret, imag_ret, real_ret1, imag_ret1;
  __m256 vreals[1ULL << N], vimags[1ULL << N];
  __m256 tmp0, tmp1;

  if (qregs[1] == 1) {
    masks[0] = _mm256_set_epi32(7, 6, 4, 5, 3, 2, 0, 1);
    masks[1] = _mm256_set_epi32(7, 4, 5, 6, 3, 0, 1, 2);
    masks[2] = _mm256_set_epi32(4, 6, 5, 7, 0, 2, 1, 3);
  } else if (qregs[0] == 0) {
    masks[0] = _mm256_set_epi32(7, 6, 5, 4, 2, 3, 0, 1);
    masks[1] = _mm256_set_epi32(7, 2, 5, 0, 3, 6, 1, 4);
    masks[2] = _mm256_set_epi32(2, 6, 0, 4, 3, 7, 1, 5);
  } else { //if (q0 == 1 && q1 == 2) {
    masks[0] = _mm256_set_epi32(7, 6, 5, 4, 1, 0, 3, 2);
    masks[1] = _mm256_set_epi32(7, 6, 1, 0, 3, 2, 5, 4);
    masks[2] = _mm256_set_epi32(1, 0, 5, 4, 3, 2, 7, 6);
  }

  float* freals = (float*) reals;
  float* fimags = (float*) imags;
  float* fmat = (float*) mat;

  for (unsigned i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    if (freals == fimags) {
      _mm_altload_complex_ps(&freals[idx * 2], &freals[idx * 2 + 8], vreals[i], vimags[i]);
    } else {
      _mm_load_twoarray_complex_ps(&freals[idx], &fimags[idx], vreals[i], vimags[i]);
    }
    for (unsigned j = 0; j < 3; ++j) {
      vreals[i + j + 1] = _mm256_permutevar8x32_ps(vreals[i], masks[j]);
      vimags[i + j + 1] = _mm256_permutevar8x32_ps(vimags[i], masks[j]);
    }
  }

  unsigned midx = 0;
  for (unsigned i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    _mm_complex_inner_product_ps((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    for (unsigned j = 0; j < 3; ++j) {
      _mm_complex_inner_product_ps((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
      midx += (1ULL << (N + 1));

      real_ret1 = _mm256_permutevar8x32_ps(real_ret1, masks[j]);
      imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, masks[j]);

      switch (j) {
      case 0:
        real_ret = (qregs[1] == 1) ? _mm256_blend_ps(real_ret, real_ret1, 0b00100010) : // (0,1)
                   (qregs[0] == 0) ? _mm256_blend_ps(real_ret, real_ret1, 0b00001010) : // (0,2)
                                     _mm256_blend_ps(real_ret, real_ret1, 0b00001100); //  (1,2)
        imag_ret = (qregs[1] == 1) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b00100010) : // (0,1)
                   (qregs[0] == 0) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b00001010) : // (0,2)
                                     _mm256_blend_ps(imag_ret, imag_ret1, 0b00001100); //  (1,2)
        break;
      case 1:
        real_ret = (qregs[1] == 1) ? _mm256_blend_ps(real_ret, real_ret1, 0b01000100) :  // (0,1)
                   (qregs[0] == 0) ? _mm256_blend_ps(real_ret, real_ret1, 0b01010000) :  // (0,2)
                                     _mm256_blend_ps(real_ret, real_ret1, 0b00110000); //   (1,2)
        imag_ret = (qregs[1] == 1) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b01000100) :  // (0,1)
                   (qregs[0] == 0) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b01010000) :  // (0,2)
                                     _mm256_blend_ps(imag_ret, imag_ret1, 0b00110000); //   (1,2)
        break;
      case 2:
        real_ret = (qregs[1] == 1) ? _mm256_blend_ps(real_ret, real_ret1, 0b10001000) : // (0,1)
                   (qregs[0] == 0) ? _mm256_blend_ps(real_ret, real_ret1, 0b10100000) : // (0,2)
                                     _mm256_blend_ps(real_ret, real_ret1, 0b11000000); //  (1,2)
        imag_ret = (qregs[1] == 1) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10001000) : // (0,1)
                   (qregs[0] == 0) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10100000) : // (0,2)
                                     _mm256_blend_ps(imag_ret, imag_ret1, 0b11000000); //  (1,2)
        break;
      }
    }
    if (freals == fimags) {
      _mm_altstore_complex_ps(real_ret, imag_ret, &freals[idx * 2], &freals[idx * 2 + 8]);
    } else {
      _mm_store_twoarray_complex_ps(real_ret, imag_ret, &freals[idx], &fimags[idx]);
    }
  }
}

template<size_t N>
inline void _apply_matrix_float_avx_qL( //
    void* reals, //
    void* imags, //
    void* mat, //
    const areg_t<1ULL << N> &inds, //
    const areg_t<N> qregs //
    ) {

  __m256i mask;
  __m256 real_ret, imag_ret, real_ret1, imag_ret1;
  __m256 vreals[1ULL << N], vimags[1ULL << N];
  __m256 tmp0, tmp1;

  if (qregs[0] == 0) {
    mask = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
  } else if (qregs[0] == 1) {
    mask = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
  } else { //if (q0 == 2) {
    mask = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
  }

  float* freals = (float*) reals;
  float* fimags = (float*) imags;
  float* fmat = (float*) mat;

  for (unsigned i = 0; i < (1ULL << N); i += 2) {
    auto idx = inds[i];
    if (freals == fimags) {
      _mm_altload_complex_ps(&freals[idx * 2], &freals[idx * 2 + 8], vreals[i], vimags[i]);
    } else {
      _mm_load_twoarray_complex_ps(&freals[idx], &fimags[idx], vreals[i], vimags[i]);
    }
    vreals[i + 1] = _mm256_permutevar8x32_ps(vreals[i], mask);
    vimags[i + 1] = _mm256_permutevar8x32_ps(vimags[i], mask);
  }

  unsigned midx = 0;
  for (unsigned i = 0; i < (1ULL << N); i += 2) {
    auto idx = inds[i];
    _mm_complex_inner_product_ps((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    _mm_complex_inner_product_ps((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    real_ret1 = _mm256_permutevar8x32_ps(real_ret1, mask);
    imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, mask);

    real_ret = (qregs[0] == 0) ? _mm256_blend_ps(real_ret, real_ret1, 0b10101010) : // (0,H,H)
               (qregs[0] == 1) ? _mm256_blend_ps(real_ret, real_ret1, 0b11001100) : // (1,H,H)
                                 _mm256_blend_ps(real_ret, real_ret1, 0b11110000); //  (2,H,H)
    imag_ret = (qregs[0] == 0) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10101010) : // (0,H,H)
               (qregs[0] == 1) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b11001100) : // (1,H,H)
                                 _mm256_blend_ps(imag_ret, imag_ret1, 0b11110000); //  (2,H,H)

    if (freals == fimags) {
      _mm_altstore_complex_ps(real_ret, imag_ret, &freals[idx * 2], &freals[idx * 2 + 8]);
    } else {
      _mm_store_twoarray_complex_ps(real_ret, imag_ret, &freals[idx], &fimags[idx]);
    }
  }
}

template<size_t N>
inline void _apply_matrix_float_avx( //
    void* reals, //
    void* imags, //
    void* mat, //
    const areg_t<1ULL << N> &inds, //
    const areg_t<N> qregs //
    ) {

  __m256 real_ret, imag_ret;
  __m256 vreals[1ULL << N], vimags[1ULL << N];
  __m256 tmp0, tmp1;

  float* freals = (float*) reals;
  float* fimags = (float*) imags;
  float* fmat = (float*) mat;

  for (unsigned i = 0; i < (1ULL << N); ++i) {
    auto idx = inds[i];
    if (freals == fimags) {
      _mm_altload_complex_ps(&freals[idx * 2], &freals[idx * 2 + 8], vreals[i], vimags[i]);
    } else {
      _mm_load_twoarray_complex_ps(&freals[idx], &fimags[idx], vreals[i], vimags[i]);
    }
  }

  unsigned midx = 0;
  for (unsigned i = 0; i < (1ULL << N); ++i) {
    auto idx = inds[i];
    _mm_complex_inner_product_ps((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));
    if (freals == fimags) {
      _mm_altstore_complex_ps(real_ret, imag_ret, &freals[idx * 2], &freals[idx * 2 + 8]);
    } else {
      _mm_store_twoarray_complex_ps(real_ret, imag_ret, &freals[idx], &fimags[idx]);
    }
  }
}

template<size_t N>
inline void _apply_matrix_double_avx_q0q1( //
    void* reals, //
    void* imags, //
    void* mat, //
    const areg_t<1ULL << N> &inds, //
    const areg_t<N> qregs //
    ) {

  __m256d real_ret, imag_ret, real_ret1, imag_ret1;
  __m256d vreals[1ULL << N], vimags[1ULL << N];
  __m256d tmp0, tmp1;

  double* dreals = (double*) reals;
  double* dimags = (double*) imags;
  double* dmat = (double*) mat;

  for (unsigned i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    if (dreals == dimags) {
      _mm_altload_complex_pd(&dreals[idx * 2], &dreals[idx * 2 + 4], vreals[i], vimags[i]);
    } else {
      _mm_load_twoarray_complex_pd(&dreals[idx], &dimags[idx], vreals[i], vimags[i]);
    }
    for (unsigned j = 1; j < 4; ++j) {
      switch (j) {
      case 1:
        vreals[i + j] = _mm256_permute4x64_pd(vreals[i], perm_d_q0q1_0);
        vimags[i + j] = _mm256_permute4x64_pd(vimags[i], perm_d_q0q1_0);
        break;
      case 2:
        vreals[i + j] = _mm256_permute4x64_pd(vreals[i], perm_d_q0q1_1);
        vimags[i + j] = _mm256_permute4x64_pd(vimags[i], perm_d_q0q1_1);
        break;
      case 3:
        vreals[i + j] = _mm256_permute4x64_pd(vreals[i], perm_d_q0q1_2);
        vimags[i + j] = _mm256_permute4x64_pd(vimags[i], perm_d_q0q1_2);
        break;
      }
    }
  }

  unsigned midx = 0;
  for (unsigned i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    _mm_complex_inner_product_pd((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));
    for (unsigned j = 1; j < 4; ++j) {
      _mm_complex_inner_product_pd((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
      midx += (1ULL << (N + 1));
      switch (j) {
      case 1:
        real_ret1 = _mm256_permute4x64_pd(real_ret1, perm_d_q0q1_0);
        imag_ret1 = _mm256_permute4x64_pd(imag_ret1, perm_d_q0q1_0);
        real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b0010);
        imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b0010);
        break;
      case 2:
        real_ret1 = _mm256_permute4x64_pd(real_ret1, perm_d_q0q1_1);
        imag_ret1 = _mm256_permute4x64_pd(imag_ret1, perm_d_q0q1_1);
        real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b0100);
        imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b0100);
        break;
      case 3:
        real_ret1 = _mm256_permute4x64_pd(real_ret1, perm_d_q0q1_2);
        imag_ret1 = _mm256_permute4x64_pd(imag_ret1, perm_d_q0q1_2);
        real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b1000);
        imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b1000);
        break;
      }
    }
    if (dreals == dimags) {
      _mm_altstore_complex_pd(real_ret, imag_ret, &dreals[idx * 2], &dreals[idx * 2 + 4]);
    } else {
      _mm_store_twoarray_complex_pd(real_ret, imag_ret, &dreals[idx], &dimags[idx]);
    }
  }
}

template<size_t N>
inline void _apply_matrix_double_avx_qL( //
    void* reals, //
    void* imags, //
    void* mat, //
    const areg_t<1ULL << N> &inds, //
    const areg_t<N> qregs //
    ) {

  __m256d real_ret, imag_ret, real_ret1, imag_ret1;
  __m256d vreals[1ULL << N], vimags[1ULL << N];
  __m256d tmp0, tmp1;

  double* dreals = (double*) reals;
  double* dimags = (double*) imags;
  double* dmat = (double*) mat;

  for (unsigned i = 0; i < (1ULL << N); i += 2) {
    auto idx = inds[i];
    if (dreals == dimags) {
      _mm_altload_complex_pd(&dreals[idx * 2], &dreals[idx * 2 + 4], vreals[i], vimags[i]);
    } else {
      _mm_load_twoarray_complex_pd(&dreals[idx], &dimags[idx], vreals[i], vimags[i]);
    }
    if (qregs[0] == 0) {
      vreals[i + 1] = _mm256_permute4x64_pd(vreals[i], perm_d_q0);
      vimags[i + 1] = _mm256_permute4x64_pd(vimags[i], perm_d_q0);
    } else {
      vreals[i + 1] = _mm256_permute4x64_pd(vreals[i], perm_d_q1);
      vimags[i + 1] = _mm256_permute4x64_pd(vimags[i], perm_d_q1);
    }
  }

  unsigned midx = 0;
  for (unsigned i = 0; i < (1ULL << N); i += 2) {
    auto idx = inds[i];
    _mm_complex_inner_product_pd((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    _mm_complex_inner_product_pd((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    if (qregs[0] == 0) {
      real_ret1 = _mm256_permute4x64_pd(real_ret1, perm_d_q0);
      imag_ret1 = _mm256_permute4x64_pd(imag_ret1, perm_d_q0);
      real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b1010);
      imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b1010);
    } else {
      real_ret1 = _mm256_permute4x64_pd(real_ret1, perm_d_q1);
      imag_ret1 = _mm256_permute4x64_pd(imag_ret1, perm_d_q1);
      real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b1100);
      imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b1100);
    }

    if (dreals == dimags) {
      _mm_altstore_complex_pd(real_ret, imag_ret, &dreals[idx * 2], &dreals[idx * 2 + 4]);
    } else {
      _mm_store_twoarray_complex_pd(real_ret, imag_ret, &dreals[idx], &dimags[idx]);
    }
  }
}

template<size_t N>
inline void _apply_matrix_double_avx( //
    void* reals, //
    void* imags, //
    void* mat, //
    const areg_t<1ULL << N> &inds, //
    const areg_t<N> qregs //
    ) {

  __m256d real_ret, imag_ret;
  __m256d vreals[1ULL << N], vimags[1ULL << N];
  __m256d tmp0, tmp1;

  double* dreals = (double*) reals;
  double* dimags = (double*) imags;
  double* dmat = (double*) mat;

  for (unsigned i = 0; i < (1ULL << N); ++i) {
    auto idx = inds[i];
    if (dreals == dimags) {
      _mm_altload_complex_pd(&dreals[idx * 2], &dreals[idx * 2 + 4], vreals[i], vimags[i]);
    } else {
      _mm_load_twoarray_complex_pd(&dreals[idx], &dimags[idx], vreals[i], vimags[i]);
    }
  }

  unsigned midx = 0;
  for (unsigned i = 0; i < (1ULL << N); ++i) {
    auto idx = inds[i];
    _mm_complex_inner_product_pd((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));
    if (dreals == dimags) {
      _mm_altstore_complex_pd(real_ret, imag_ret, &dreals[idx * 2], &dreals[idx * 2 + 4]);
    } else {
      _mm_store_twoarray_complex_pd(real_ret, imag_ret, &dreals[idx], &dimags[idx]);
    }
  }
}

const std::array<uint64_t, 64> _BITS{{  //
    1ULL, 2ULL, 4ULL, 8ULL,  //
        16ULL, 32ULL, 64ULL, 128ULL,  //
        256ULL, 512ULL, 1024ULL, 2048ULL,  //
        4096ULL, 8192ULL, 16384ULL, 32768ULL,  //
        65536ULL, 131072ULL, 262144ULL, 524288ULL,  //
        1048576ULL, 2097152ULL, 4194304ULL, 8388608ULL,  //
        16777216ULL, 33554432ULL, 67108864ULL, 134217728ULL,  //
        268435456ULL, 536870912ULL, 1073741824ULL, 2147483648ULL,  //
        4294967296ULL, 8589934592ULL, 17179869184ULL, 34359738368ULL,  //
        68719476736ULL, 137438953472ULL, 274877906944ULL, 549755813888ULL,  //
        1099511627776ULL, 2199023255552ULL, 4398046511104ULL, 8796093022208ULL,  //
        17592186044416ULL, 35184372088832ULL, 70368744177664ULL, 140737488355328ULL,  //
        281474976710656ULL, 562949953421312ULL, 1125899906842624ULL, 2251799813685248ULL,  //
        4503599627370496ULL, 9007199254740992ULL, 18014398509481984ULL, 36028797018963968ULL,  //
        72057594037927936ULL, 144115188075855872ULL, 288230376151711744ULL, 576460752303423488ULL,  //
        1152921504606846976ULL, 2305843009213693952ULL, 4611686018427387904ULL, 9223372036854775808ULL  //
    }};

const std::array<uint64_t, 64> _MASKS{{  //
    0ULL, 1ULL, 3ULL, 7ULL,  //
        15ULL, 31ULL, 63ULL, 127ULL,  //
        255ULL, 511ULL, 1023ULL, 2047ULL,  //
        4095ULL, 8191ULL, 16383ULL, 32767ULL,  //
        65535ULL, 131071ULL, 262143ULL, 524287ULL,  //
        1048575ULL, 2097151ULL, 4194303ULL, 8388607ULL,  //
        16777215ULL, 33554431ULL, 67108863ULL, 134217727ULL,  //
        268435455ULL, 536870911ULL, 1073741823ULL, 2147483647ULL,  //
        4294967295ULL, 8589934591ULL, 17179869183ULL, 34359738367ULL,  //
        68719476735ULL, 137438953471ULL, 274877906943ULL, 549755813887ULL,  //
        1099511627775ULL, 2199023255551ULL, 4398046511103ULL, 8796093022207ULL,  //
        17592186044415ULL, 35184372088831ULL, 70368744177663ULL, 140737488355327ULL,  //
        281474976710655ULL, 562949953421311ULL, 1125899906842623ULL, 2251799813685247ULL,  //
        4503599627370495ULL, 9007199254740991ULL, 18014398509481983ULL, 36028797018963967ULL,  //
        72057594037927935ULL, 144115188075855871ULL, 288230376151711743ULL, 576460752303423487ULL,  //
        1152921504606846975ULL, 2305843009213693951ULL, 4611686018427387903ULL, 9223372036854775807ULL  //
    }};

template<typename list_t>
uint64_t _index0(const list_t &qubits_sorted, const uint64_t k) {
  uint64_t lowbits, retval = k;
  for (size_t j = 0; j < qubits_sorted.size(); j++) {
    lowbits = retval & _MASKS[qubits_sorted[j]];
    retval >>= qubits_sorted[j];
    retval <<= qubits_sorted[j] + 1;
    retval |= lowbits;
  }
  return retval;
}

template<size_t N>
areg_t<1ULL << N> _indexes(const areg_t<N> &qs, const areg_t<N> &qubits_sorted, const uint64_t k) {
  areg_t<1ULL << N> ret;
  ret[0] = _index0(qubits_sorted, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = _BITS[i];
    const auto bit = _BITS[qs[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}

template<typename Lambda, typename list_t, typename param_t>
void _apply_lambda(uint64_t data_size, const uint64_t skip, Lambda&& func, const list_t &qubits, const unsigned omp_threads, const param_t &params) {

  const auto NUM_QUBITS = qubits.size();
  const uint64_t END = data_size >> NUM_QUBITS;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

#pragma omp parallel for if (omp_threads > 1) num_threads(omp_threads)
  for (uint64_t k = 0; k < END; k += skip) {
    const auto inds = _indexes(qubits, qubits_sorted, k);
    std::forward < Lambda > (func)(inds, params);
  }
}

template<size_t N, typename data_t>
inline void _reorder(areg_t<N>& qreg, data_t* fmat) {
  auto dim = (1UL << N);
  auto qreg_orig = qreg;
  std::sort(qreg.begin(), qreg.end());

  unsigned masks[N];

  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      if (qreg_orig[i] == qreg[j])
        masks[i] = 1U << j;

  size_t indexes[1U << N];
  for (size_t i = 0; i < dim; ++i) {
    size_t index = 0U;
    for (size_t j = 0; j < N; ++j) {
      if (i & (1U << j))
        index |= masks[j];
    }
    indexes[i] = index;
  }

  std::vector<data_t> fmat_org;
  for (size_t i = 0; i < dim * dim * 2; ++i) {
    fmat_org.push_back(fmat[i]);
  }

  for (unsigned i = 0; i < dim; ++i) {
    for (unsigned j = 0; j < dim; ++j) {
      unsigned oldidx = i * dim + j;
      unsigned newidx = indexes[i] * dim + indexes[j];
      fmat[newidx * 2] = fmat_org[oldidx * 2];
      fmat[newidx * 2 + 1] = fmat_org[oldidx * 2 + 1];
    }
  }
}

template<size_t N>
inline bool _apply_matrix_float(  //
    void * _data0, //
    void * _data1, //
    uint64_t data_size, //
    const areg_t<N> qregs_, //
    const void* mat, //
    const unsigned omp_threads //
    ) {

  if (data_size <= 8)
    return false;

  float* data0 = (float*) _data0;
  float* data1 = (float*) _data1;
  float* fmat_orig = (float*) mat;

  float fmat[(1 << (N * 2 + 1))];
  memcpy(fmat, mat, sizeof(float) * (1 << (N * 2 + 1)));

// transpose
  for (size_t i = 0; i < (1 << N); ++i) {
    for (size_t j = 0; j < (1 << N); ++j) {
      fmat[(i * (1 << N) + j) * 2] = fmat_orig[(j * (1 << N) + i) * 2];
      fmat[(i * (1 << N) + j) * 2 + 1] = fmat_orig[(j * (1 << N) + i) * 2 + 1];
    }
  }

  auto qregs = qregs_;
  if (qregs.size() > 1) {
    _reorder(qregs, fmat);
  }

  if (qregs.size() > 2 && qregs[2] == 2) {
    auto lambda = [&](const areg_t<(1 << N)> &inds, float* _fmat)->void {
      _apply_matrix_float_avx_q0q1q2(data0, data1, _fmat, inds, qregs);
    };
    _apply_lambda(data_size, 1, lambda, qregs, omp_threads, (float*) fmat);
  } else if (qregs.size() > 1 && qregs[1] < 3) {
    auto lambda = [&](const areg_t<(1 << N)> &inds, float* _fmat)->void {
      _apply_matrix_float_avx_qLqL(data0, data1, _fmat, inds, qregs);
    };
    _apply_lambda(data_size, 2, lambda, qregs, omp_threads, (float*) fmat);
  } else if (qregs[0] < 3) {
    auto lambda = [&](const areg_t<(1 << N)> &inds, float* _fmat)->void {
      _apply_matrix_float_avx_qL(data0, data1, _fmat, inds, qregs);
    };
    _apply_lambda(data_size, 4, lambda, qregs, omp_threads, (float*) fmat);
  } else {
    auto lambda = [&](const areg_t<(1 << N)> &inds, float* _fmat)->void {
      _apply_matrix_float_avx(data0, data1, _fmat, inds, qregs);
    };
    _apply_lambda(data_size, 8, lambda, qregs, omp_threads, (float*) fmat);
  }
  return true;
}

template<size_t N>
inline bool _apply_matrix_double(  //
    void * _data0, //
    void * _data1, //
    uint64_t data_size, //
    const areg_t<N> qregs_, //
    const void* mat, //
    const unsigned omp_threads //
    ) {

  if (data_size <= 4)
    return false;

  double dmat[(1 << (N * 2 + 1))];
  memcpy(dmat, mat, sizeof(double) * (1 << (N * 2 + 1)));

  double* dmat_orig = (double*) mat;

// transpose
  for (size_t i = 0; i < (1 << N); ++i) {
    for (size_t j = 0; j < (1 << N); ++j) {
      dmat[(i * (1 << N) + j) * 2] = dmat_orig[(j * (1 << N) + i) * 2];
      dmat[(i * (1 << N) + j) * 2 + 1] = dmat_orig[(j * (1 << N) + i) * 2 + 1];
    }
  }

  auto qregs = qregs_;
  if (qregs.size() > 1) {
    _reorder(qregs, dmat);
  }

  double* data0 = (double*) _data0;
  double* data1 = (double*) _data1;

  if (qregs.size() > 1 && qregs[1] == 1) {
    auto lambda = [&](const areg_t<(1 << N)> &inds, double* _dmat)->void {
      _apply_matrix_double_avx_q0q1(data0, data1, _dmat, inds, qregs);
    };
    _apply_lambda(data_size, 1, lambda, qregs, omp_threads, (double*) dmat);
  } else if (qregs[0] < 2) {
    auto lambda = [&](const areg_t<(1 << N)> &inds, double* _dmat)->void {
      _apply_matrix_double_avx_qL(data0, data1, _dmat, inds, qregs);
    };
    _apply_lambda(data_size, 2, lambda, qregs, omp_threads, (double*) dmat);
  } else {
    auto lambda = [&](const areg_t<(1 << N)> &inds, double* _dmat)->void {
      _apply_matrix_double_avx(data0, data1, _dmat, inds, qregs);
    };
    _apply_lambda(data_size, 4, lambda, qregs, omp_threads, (double*) dmat);
  }
  return true;
}

template<typename data_t, size_t N>
inline bool apply_matrix_avx( //
    void * qv_data, //
    uint64_t data_size, //
    const areg_t<N> qregs, //
    const void* mat, //
    const unsigned omp_threads //
    ) {

  if (sizeof(data_t) == sizeof(float)) {
    return _apply_matrix_float(qv_data, qv_data, data_size, qregs, mat, omp_threads);
  } else {
    return _apply_matrix_double(qv_data, qv_data, data_size, qregs, mat, omp_threads);
  }
}

template<size_t N>
inline areg_t<N> to_array(reg_t vec) {
  areg_t<N> ret;
  for (unsigned i = 0; i < N; ++i)
    ret[i] = vec[i];
  return ret;
}

template<typename data_t>
inline bool apply_matrix_avx( //
    void * qv_data, //
    uint64_t data_size, //
    const reg_t qregs, //
    const void* mat, //
    const unsigned omp_threads //
    ) {
  switch (qregs.size()) {
  case 1:
    return apply_matrix_avx<data_t>(qv_data, data_size, to_array<1>(qregs), mat, omp_threads);
  case 2:
    return apply_matrix_avx<data_t>(qv_data, data_size, to_array<2>(qregs), mat, omp_threads);
  case 3:
    return apply_matrix_avx<data_t>(qv_data, data_size, to_array<3>(qregs), mat, omp_threads);
  case 4:
    return apply_matrix_avx<data_t>(qv_data, data_size, to_array<4>(qregs), mat, omp_threads);
  case 5:
    return apply_matrix_avx<data_t>(qv_data, data_size, to_array<5>(qregs), mat, omp_threads);
  case 6:
    return apply_matrix_avx<data_t>(qv_data, data_size, to_array<6>(qregs), mat, omp_threads);
  default:
    return false;
  }
}

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

} // namespace QV
#endif
