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
#include <type_traits>
#include "qubitvector.hpp"

namespace {

template<typename FloatType>
using m256_t = typename std::conditional<std::is_same<FloatType, double>::value, __m256d, __m256>::type;

// These Views are helpers for encapsulate QubitVector data_ type access.
// As the data_ variable is allocated dynamically and of the type:
// std::complex<double/float>*
template<typename FloatType>
struct RealView {
  RealView() = delete;
  // Unfortunately, shared_ptr implies allocations and we cannot afford
  // them in this piece of code, so this is the reason to use raw pointers.
  RealView(const std::complex<FloatType>* data) : data(data){}
  inline const FloatType* operator[](size_t i) const {
    #ifdef DEBUG
    if (i >= sizeof(data)) {
      throw std::runtime_error("Index out of bounds!");
    }
    #endif
    return reintepret_cast<FloatType*>(data)[2* i];
  }
  const std::complex<FloatType>* data = nullptr;
};

template<typename FloatType>
struct ImaginaryView {
  ImaginaryView() = delete;
  ImaginaryView(const std::complex<FloatType>* data) : data(data){}
  inline const FloatType* operator[](size_t i) const {
    #ifdef DEBUG
    if (i >= sizeof(data)) {
      throw std::runtime_error("Index out of bounds!");
    }
    #endif
    return reintepret_cast<FloatType*>(data)[2 * i + 1];
  }
  const std::complex<FloatType>* data = nullptr;
};


auto _mm256_mul(const m256_t<double> left, const m256_t<double> right){
  return _mm256_mul_pd(left, right);
}

auto _mm256_mul(const m256_t<float> left, const m256_t<float> right){
  return _mm256_mul_ps(left, right);
}

auto _mm256_fnmadd(const m256_t<double> left, const m256_t<double> right, const m256_t<double> ret){
  return _mm256_fnmadd_pd(left, right, ret);
}

auto _mm256_fnmadd(const m256_t<float> left, const m256_t<float> right, const m256_t<float> ret){
  return _mm256_fnmadd_ps(left, right, ret);
}

auto _mm256_fmadd(const m256_t<double> left, const m256_t<double> right, const m256_t<double> ret){
  return _mm256_fmadd_pd(left, right, ret);
}

auto _mm256_fmadd(const m256_t<float>& left, const m256_t<float>& right, const m256_t<float>& ret){
  return _mm256_fmadd_ps(left, right, ret);
}

auto _mm256_set1(double d){
  return _mm256_set1_pd(d);
}

auto _mm256_set1(float f){
  return _mm256_set1_ps(f);
}

auto _mm256_load(double const *d){
  return _mm256_load_pd(d);
}

auto _mm256_load(float const *f){
  return _mm256_load_ps(f);
}

void _mm256_store(float* f, const m256_t<float>& c){
  _mm256_store_ps(f, c);
}

void _mm256_store(double* d, const m256_t<double>& c){
  _mm256_store_pd(d, c);
}

// auto _mm256_permute(m256_t<double> a, int b){
//   return _mm256_permute4x64_pd(a, b);
// }

// auto _mm256_permute(m256_t<float> f, __m256i b){
//   return _mm256_permutevar8x32_ps(f, b);
// }

// auto _mm256_blend(m256_t<double> d, int b){
//   return _mm256_blend_pd(d, b);
// }

// auto _mm256_blend(m256_t<float> f, int b){
//   return _mm256_blend_ps(f, b);
// }


template<typename FloatType>
inline void _mm_complex_multiply(m256_t<FloatType>& real_ret, m256_t<FloatType>& imag_ret, m256_t<FloatType>& real_left,
  m256_t<FloatType>& imag_left, const m256_t<FloatType> real_right, const m256_t<FloatType> imag_right){
    real_ret = _mm256_mul(real_left, real_right);
    imag_ret = _mm256_mul(real_left, imag_right);
    real_ret = _mm256_fnmadd(imag_left, imag_right, real_ret);
    imag_ret = _mm256_fmadd(imag_left, real_right, imag_ret);
}

template<typename FloatType>
inline void _mm_complex_multiply_add(m256_t<FloatType>& real_ret, m256_t<FloatType>& imag_ret, m256_t<FloatType>& real_left,
  m256_t<FloatType>& imag_left, const m256_t<FloatType> real_right, const m256_t<FloatType> imag_right){
    real_ret = _mm256_fmadd(real_left, real_right, real_ret);
    imag_ret = _mm256_fmadd(real_left, imag_right, imag_ret);
    real_ret = _mm256_fnmadd(imag_left, imag_right, real_ret);
    imag_ret = _mm256_fmadd(imag_left, real_right, imag_ret);
}

template<typename FloatType>
inline void _mm_complex_inner_product(size_t dim, m256_t<FloatType> vreals[], m256_t<FloatType> vimags[], double * cmplxs,
  m256_t<FloatType>& vret_real, m256_t<FloatType>& vret_imag, m256_t<FloatType>& vtmp_0, m256_t<FloatType>& vtmp_1){
    vtmp_0 = _mm256_set1(cmplxs[0]);
    vtmp_1 = _mm256_set1(cmplxs[1]);
    _mm_complex_multiply(vret_real, vret_imag, vreals[0], vimags[0], vtmp_0, vtmp_1);
    for (size_t i = 1; i < dim; ++i){
      vtmp_0 = _mm256_set1(cmplxs[i * 2]);
      vtmp_1 = _mm256_set1(cmplxs[i * 2 + 1]);
      _mm_complex_multiply_add(vret_real, vret_imag, vreals[i], vimags[i], vtmp_0, vtmp_1);
    }
}

template<typename FloatType>
inline void _mm_load_twoarray_complex(const FloatType * real_addr_0, const FloatType * imag_addr_1,
  m256_t<FloatType>& real_ret, m256_t<FloatType>& imag_ret){
    real_ret = _mm256_load(real_addr_0);
    imag_ret = _mm256_load(imag_addr_1);
}

template<typename FloatType>
inline void _mm_store_twoarray_complex(const m256_t<FloatType>& real_ret, const m256_t<FloatType>& imag_ret,
  FloatType * cmplx_addr_0, FloatType * cmplx_addr_1){
    _mm256_store(cmplx_addr_0, real_ret);
    _mm256_store(cmplx_addr_1, imag_ret);
}

/* TODO: Commented until clarification
template<typename FloatType>
inline void _mm_altload_complex(FloatType const * cmplx_addr_0, FloatType const * cmplx_addr_1,
  m256_t<FloatType>& real_ret, m256_t<FloatType>& imag_ret){
    real_ret = _mm256_loads(cmplx_addr_0);
    imag_ret = _mm256_load(cmplx_addr_1);
    tmp0 = _mm256_permute(real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    tmp1 = _mm256_permute(imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    real_ret = _mm256_blend(real_ret, tmp1, 0b10101010);
    imag_ret = _mm256_blend(tmp0, imag_ret, 0b10101010);
    real_ret = _mm256_permute(real_ret, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
    imag_ret = _mm256_permute(imag_ret, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
}


template<typename FloatType>
inline void _mm_altstore_complex(m256_t<FloatType>& real_ret, m256_t<FloatType>& imag_ret,
  FloatType const * cmplx_addr_0, FloatType const * cmplx_addr_1) {
    real_ret = _mm256_permute(real_ret, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    imag_ret = _mm256_permute(imag_ret, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    tmp0 = _mm256_permute(real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    tmp1 = _mm256_permute(imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    real_ret = _mm256_blend(real_ret, tmp1, 0b10101010);
    imag_ret = _mm256_blend(tmp0, imag_ret, 0b10101010);
    _mm256_store(cmplx_addr_0, real_ret);
    _mm256_store(cmplx_addr_1, imag_ret);
}*/

template<size_t N, typename FloatType>
inline void reorder(QV::areg_t<N>& qregs, QV::cvector_t<FloatType>& mat) {
  if(qregs.size() < 2)
    return;

  auto dim = (1UL << N);
  auto qreg_orig = qregs;
  // TODO Haven't we already ordered
  std::sort(qregs.begin(), qregs.end());

  size_t masks[N];

  for(size_t i = 0; i < N; ++i)
    for(size_t j = 0; j < N; ++j)
      if(qreg_orig[i] == qregs[j])
        masks[i] = 1U << j;

  size_t indexes[1U << N];
  for(size_t i = 0; i < dim; ++i) {
    size_t index = 0U;
    for(size_t j = 0; j < N; ++j) {
      if(i & (1U << j))
        index |= masks[j];
    }
    indexes[i] = index;
  }

  QV::cvector_t<FloatType> mat_orig;
  mat_orig.reserve(mat.size());
  std::copy_n(mat.begin(), dim * dim * 2, std::back_inserter(mat_orig));

  // TODO Using standard interface for transposing instead of mmemory indexing
  for(size_t i = 0; i < dim; ++i) {
    for(size_t j = 0; j < dim; ++j) {
      size_t oldidx = i * dim + j;
      size_t newidx = indexes[i] * dim + indexes[j];
      reinterpret_cast<FloatType*>(mat.data())[newidx * 2] = reinterpret_cast<FloatType*>(mat_orig.data())[oldidx * 2];
      reinterpret_cast<FloatType*>(mat.data())[newidx * 2 + 1] = reinterpret_cast<FloatType*>(mat_orig.data())[oldidx * 2 + 1];
    }
  }
}

template<size_t N>
inline QV::areg_t<N> to_array(const QV::reg_t& vec) {
  QV::areg_t<N> ret;
  std::copy_n(vec.begin(), N, std::back_inserter(ret));
  return ret;
}

} // End anonymous namespace

namespace QV {

//using reg_t = std::vector<uint64_t>;
//template<size_t N> using areg_t = std::array<uint64_t, N>;
//using indexes_t = std::unique_ptr<uint64_t[]>;

template<size_t N>
inline void _apply_matrix_float_avx_q0q1q2(
    const RealView<float>& reals,
    const ImaginaryView<float>& imags,
    const std::complex<float>* mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N> qregs
){

  const std::array<__m256i, 7> _MASKS = {
    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 0, 1),
    _mm256_set_epi32(7, 6, 5, 4, 3, 0, 1, 2),
    _mm256_set_epi32(7, 6, 5, 4, 0, 2, 1, 3),
    _mm256_set_epi32(7, 6, 5, 0, 3, 2, 1, 4),
    _mm256_set_epi32(7, 6, 0, 4, 3, 2, 1, 5),
    _mm256_set_epi32(7, 0, 5, 4, 3, 2, 1, 6),
    _mm256_set_epi32(0, 6, 5, 4, 3, 2, 1, 7)
  };
  __m256 real_ret, imag_ret, real_ret1, imag_ret1;
  __m256 vreals[1ULL << N], vimags[1ULL << N];
  __m256 tmp0, tmp1;


  for(unsigned i = 0; i < (1ULL << N); i += 8){
    auto idx = inds[i];
    // if (freals == fimags) {
    //   _mm_altload_complex_ps(&freals[idx * 2], &freals[idx * 2 + 8], vreals[i], vimags[i]);
    // } else {
    //   _mm_load_twoarray_complex_ps(&freals[idx], &fimags[idx], vreals[i], vimags[i]);
    // }
    _mm_load_twoarray_complex<float>(&reals[idx], &imags[idx], vreals[i], vimags[i]);

    for (unsigned j = 1; j < 8; ++j) {
      vreals[i + j] = _mm256_permutevar8x32_ps(vreals[i], _MASKS[j - 1]);
      vimags[i + j] = _mm256_permutevar8x32_ps(vimags[i], _MASKS[j - 1]);
    }
  }

  unsigned midx = 0;
  for(unsigned i = 0; i < inds.size(); i += 8){
    auto idx = inds[i];
    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    for(size_t j = 1; j < 8; ++j){
      _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
      midx += (1ULL << (N + 1));

      real_ret1 = _mm256_permutevar8x32_ps(real_ret1, _MASKS[j - 1]);
      imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, _MASKS[j - 1]);

      // Yes, this needs to be constexpr due to restrictions with _mm256_blend_ps
      const auto shift_bit_mask = [](const size_t j) -> int {
          return 0b00000001 << j;
      };

      real_ret = _mm256_blend_ps(real_ret, real_ret1, shift_bit_mask(j));
      imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, shift_bit_mask(j));

      // switch (j) {
      // case 1:
      //   real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00000010);
      //   imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00000010);
      //   break;
      // case 2:
      //   real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00000100);
      //   imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00000100);
      //   break;
      // case 3:
      //   real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00001000);
      //   imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00001000);
      //   break;
      // case 4:
      //   real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00010000);
      //   imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00010000);
      //   break;
      // case 5:
      //   real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00100000);
      //   imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00100000);
      //   break;
      // case 6:
      //   real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b01000000);
      //   imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b01000000);
      //   break;
      // case 7:
      //   real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b10000000);
      //   imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b10000000);
      //   break;
      // }
    }

    // if (freals == fimags) {
    //   _mm_altstore_complex_ps(real_ret, imag_ret, &freals[idx * 2], &freals[idx * 2 + 8]);
    // } else {
    //   _mm_store_twoarray_complex_ps(real_ret, imag_ret, &freals[idx], &fimags[idx]);
    // }
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_float_avx_qLqL(
    const RealView<float>& reals,
    const ImaginaryView<float>& imags,
    const std::complex<float>* mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N> qregs
){

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

  for (size_t i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    // if (freals == fimags) {
    //   _mm_altload_complex_ps(&freals[idx * 2], &freals[idx * 2 + 8], vreals[i], vimags[i]);
    // } else {
    //   _mm_load_twoarray_complex_ps(&freals[idx], &fimags[idx], vreals[i], vimags[i]);
    // }
    _mm_load_twoarray_complex<float>(reals[idx], imags[idx], vreals[i], vimags[i]);

    for (size_t j = 0; j < 3; ++j) {
      vreals[i + j + 1] = _mm256_permutevar8x32_ps(vreals[i], masks[j]);
      vimags[i + j + 1] = _mm256_permutevar8x32_ps(vimags[i], masks[j]);
    }
  }

  size_t midx = 0;
  for (size_t i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    for (unsigned j = 0; j < 3; ++j){
      _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
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
    // if (freals == fimags) {
    //   _mm_altstore_complex_ps(real_ret, imag_ret, &freals[idx * 2], &freals[idx * 2 + 8]);
    // } else {
    //   _mm_store_twoarray_complex_ps(real_ret, imag_ret, &freals[idx], &fimags[idx]);
    // }
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_float_avx_qL(
    const RealView<float>& reals,
    const ImaginaryView<float>& imags,
    const std::complex<float>* mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N> qregs
){

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

  for (unsigned i = 0; i < (1ULL << N); i += 2){
    auto idx = inds[i];
    // if (freals == fimags) {
    //   _mm_altload_complex_ps(&freals[idx * 2], &freals[idx * 2 + 8], vreals[i], vimags[i]);
    // } else {
    //   _mm_load_twoarray_complex_ps(&freals[idx], &fimags[idx], vreals[i], vimags[i]);
    // }
    _mm_load_twoarray_complex<float>(reals[idx], imags[idx], vreals[i], vimags[i]);

    vreals[i + 1] = _mm256_permutevar8x32_ps(vreals[i], mask);
    vimags[i + 1] = _mm256_permutevar8x32_ps(vimags[i], mask);
  }

  unsigned midx = 0;
  for (unsigned i = 0; i < (1ULL << N); i += 2) {
    auto idx = inds[i];
    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    real_ret1 = _mm256_permutevar8x32_ps(real_ret1, mask);
    imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, mask);

    real_ret = (qregs[0] == 0) ? _mm256_blend_ps(real_ret, real_ret1, 0b10101010) : // (0,H,H)
               (qregs[0] == 1) ? _mm256_blend_ps(real_ret, real_ret1, 0b11001100) : // (1,H,H)
                                 _mm256_blend_ps(real_ret, real_ret1, 0b11110000); //  (2,H,H)
    imag_ret = (qregs[0] == 0) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10101010) : // (0,H,H)
               (qregs[0] == 1) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b11001100) : // (1,H,H)
                                 _mm256_blend_ps(imag_ret, imag_ret1, 0b11110000); //  (2,H,H)

    // if (freals == fimags) {
    //   _mm_altstore_complex_ps(real_ret, imag_ret, &freals[idx * 2], &freals[idx * 2 + 8]);
    // } else {
    //   _mm_store_twoarray_complex_ps(real_ret, imag_ret, &freals[idx], &fimags[idx]);
    // }
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_float_avx(
    const RealView<float>& reals,
    const ImaginaryView<float>& imags,
    const std::complex<float>* mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N> qregs
){

  __m256 real_ret, imag_ret;
  __m256 vreals[1ULL << N], vimags[1ULL << N];
  __m256 tmp0, tmp1;

  for(unsigned i = 0; i < (1ULL << N); ++i){
    auto idx = inds[i];
    // if (freals == fimags) {
    //   _mm_altload_complex_ps(&freals[idx * 2], &freals[idx * 2 + 8], vreals[i], vimags[i]);
    // } else {
    //   _mm_load_twoarray_complex_ps(&freals[idx], &fimags[idx], vreals[i], vimags[i]);
    // }
    _mm_load_twoarray_complex<float>(reals[idx], imags[idx], vreals[i], vimags[i]);
  }

  unsigned midx = 0;
  for(unsigned i = 0; i < (1ULL << N); ++i){
    auto idx = inds[i];
    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, mat[midx], real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));
    // if (freals == fimags) {
    //   _mm_altstore_complex_ps(real_ret, imag_ret, &freals[idx * 2], &freals[idx * 2 + 8]);
    // } else {
    //   _mm_store_twoarray_complex_ps(real_ret, imag_ret, &freals[idx], &fimags[idx]);
    // }
    _mm_store_twoarray_complex<float>(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_double_avx_q0q1(
    const RealView<double>& reals,
    const ImaginaryView<double>& imags,
    const std::complex<double>* mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N> qregs
){

  const int PERM_D_Q0Q1_0 = 3 * 64 + 2 * 16 + 0 * 4 + 1 * 1;
  const int PERM_D_Q0Q1_1 = 3 * 64 + 0 * 16 + 1 * 4 + 2 * 1;
  const int PERM_D_Q0Q1_2 = 0 * 64 + 2 * 16 + 1 * 4 + 3 * 1;


  __m256d real_ret, imag_ret, real_ret1, imag_ret1;
  __m256d vreals[1ULL << N], vimags[1ULL << N];
  __m256d tmp0, tmp1;


  for(size_t i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    // if (dreals == dimags) {
    //   _mm_altload_complex_pd(&dreals[idx * 2], &dreals[idx * 2 + 4], vreals[i], vimags[i]);
    // } else {
    //   _mm_load_twoarray_complex_pd(&dreals[idx], &dimags[idx], vreals[i], vimags[i]);
    // }
    _mm_load_twoarray_complex<double>(reals[idx], imags[idx], vreals[i], vimags[i]);
    for (size_t j = 1; j < 4; ++j) {
      switch (j) {
      case 1:
        vreals[i + j] = _mm256_permute4x64_pd(vreals[i], PERM_D_Q0Q1_0);
        vimags[i + j] = _mm256_permute4x64_pd(vimags[i], PERM_D_Q0Q1_0);
        break;
      case 2:
        vreals[i + j] = _mm256_permute4x64_pd(vreals[i], PERM_D_Q0Q1_1);
        vimags[i + j] = _mm256_permute4x64_pd(vimags[i], PERM_D_Q0Q1_1);
        break;
      case 3:
        vreals[i + j] = _mm256_permute4x64_pd(vreals[i], PERM_D_Q0Q1_2);
        vimags[i + j] = _mm256_permute4x64_pd(vimags[i], PERM_D_Q0Q1_2);
        break;
      }
    }
  }

  size_t midx = 0;
  for (size_t i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));
    for (size_t j = 1; j < 4; ++j) {
      _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
      midx += (1ULL << (N + 1));
      switch(j){
        case 1:
          real_ret1 = _mm256_permute4x64_pd(real_ret1, PERM_D_Q0Q1_0);
          imag_ret1 = _mm256_permute4x64_pd(imag_ret1, PERM_D_Q0Q1_0);
          real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b0010);
          imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b0010);
          break;
        case 2:
          real_ret1 = _mm256_permute4x64_pd(real_ret1, PERM_D_Q0Q1_1);
          imag_ret1 = _mm256_permute4x64_pd(imag_ret1, PERM_D_Q0Q1_1);
          real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b0100);
          imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b0100);
          break;
        case 3:
          real_ret1 = _mm256_permute4x64_pd(real_ret1, PERM_D_Q0Q1_2);
          imag_ret1 = _mm256_permute4x64_pd(imag_ret1, PERM_D_Q0Q1_2);
          real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b1000);
          imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b1000);
          break;
      }
    }
    // if (dreals == dimags) {
    //   _mm_altstore_complex_pd(real_ret, imag_ret, &dreals[idx * 2], &dreals[idx * 2 + 4]);
    // } else {
    //   _mm_store_twoarray_complex_pd(real_ret, imag_ret, &dreals[idx], &dimags[idx]);
    // }
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_double_avx_qL(
    const RealView<double>& reals,
    const ImaginaryView<double>& imags,
    const std::complex<double>* mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N> qregs
){

  const int PERM_D_Q0 = 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1;
  const int PERM_D_Q1 = 1 * 64 + 0 * 16 + 3 * 4 + 2 * 1;

  __m256d real_ret, imag_ret, real_ret1, imag_ret1;
  __m256d vreals[1ULL << N], vimags[1ULL << N];
  __m256d tmp0, tmp1;

  for(unsigned i = 0; i < (1ULL << N); i += 2){
    auto idx = inds[i];
    // if (dreals == dimags) {
    //   _mm_altload_complex_pd(&dreals[idx * 2], &dreals[idx * 2 + 4], vreals[i], vimags[i]);
    // } else {
    //   _mm_load_twoarray_complex_pd(&dreals[idx], &dimags[idx], vreals[i], vimags[i]);
    // }
    _mm_load_twoarray_complex<double>(reals[idx], imags[idx], vreals[i], vimags[i]);
    if (qregs[0] == 0) {
      vreals[i + 1] = _mm256_permute4x64_pd(vreals[i], PERM_D_Q0);
      vimags[i + 1] = _mm256_permute4x64_pd(vimags[i], PERM_D_Q0);
    } else {
      vreals[i + 1] = _mm256_permute4x64_pd(vreals[i], PERM_D_Q1);
      vimags[i + 1] = _mm256_permute4x64_pd(vimags[i], PERM_D_Q1);
    }
  }

  size_t midx = 0;
  for(size_t i = 0; i < (1ULL << N); i += 2){
    auto idx = inds[i];
    _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    if (qregs[0] == 0) {
      real_ret1 = _mm256_permute4x64_pd(real_ret1, PERM_D_Q0);
      imag_ret1 = _mm256_permute4x64_pd(imag_ret1, PERM_D_Q0);
      real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b1010);
      imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b1010);
    } else {
      real_ret1 = _mm256_permute4x64_pd(real_ret1, PERM_D_Q1);
      imag_ret1 = _mm256_permute4x64_pd(imag_ret1, PERM_D_Q1);
      real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b1100);
      imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b1100);
    }

    // if (dreals == dimags) {
    //   _mm_altstore_complex_pd(real_ret, imag_ret, &dreals[idx * 2], &dreals[idx * 2 + 4]);
    // } else {
    //   _mm_store_twoarray_complex_pd(real_ret, imag_ret, &dreals[idx], &dimags[idx]);
    // }
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_double_avx(
    const RealView<double>& reals,
    const ImaginaryView<double>& imags,
    const std::complex<double>* mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N> qregs
){

  __m256d real_ret, imag_ret;
  __m256d vreals[1ULL << N], vimags[1ULL << N];
  __m256d tmp0, tmp1;

  for(unsigned i = 0; i < (1ULL << N); ++i){
    auto idx = inds[i];
    // if (dreals == dimags) {
    //   _mm_altload_complex_pd(&dreals[idx * 2], &dreals[idx * 2 + 4], vreals[i], vimags[i]);
    // } else {
    //   _mm_load_twoarray_complex_pd(&dreals[idx], &dimags[idx], vreals[i], vimags[i]);
    // }
    _mm_load_twoarray_complex<double>(reals[idx], imags[idx], vreals[i], vimags[i]);
  }

  unsigned midx = 0;
  for(unsigned i = 0; i < (1ULL << N); ++i){
    auto idx = inds[i];
    _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&mat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));
    // if (dreals == dimags) {
    //   _mm_altstore_complex_pd(real_ret, imag_ret, &dreals[idx * 2], &dreals[idx * 2 + 4]);
    // } else {
    //   _mm_store_twoarray_complex_pd(real_ret, imag_ret, &dreals[idx], &dimags[idx]);
    // }
    _mm_store_twoarray_complex<double>(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<typename list_t>
uint64_t _index0(const list_t &qubits_sorted, const uint64_t k){
  uint64_t lowbits, retval = k;
  const std::array<uint64_t, 64> _MASKS = {
    0ULL, 1ULL, 3ULL, 7ULL,
    15ULL, 31ULL, 63ULL, 127ULL,
    255ULL, 511ULL, 1023ULL, 2047ULL,
    4095ULL, 8191ULL, 16383ULL, 32767ULL,
    65535ULL, 131071ULL, 262143ULL, 524287ULL,
    1048575ULL, 2097151ULL, 4194303ULL, 8388607ULL,
    16777215ULL, 33554431ULL, 67108863ULL, 134217727ULL,
    268435455ULL, 536870911ULL, 1073741823ULL, 2147483647ULL,
    4294967295ULL, 8589934591ULL, 17179869183ULL, 34359738367ULL,
    68719476735ULL, 137438953471ULL, 274877906943ULL, 549755813887ULL,
    1099511627775ULL, 2199023255551ULL, 4398046511103ULL, 8796093022207ULL,
    17592186044415ULL, 35184372088831ULL, 70368744177663ULL, 140737488355327ULL,
    281474976710655ULL, 562949953421311ULL, 1125899906842623ULL, 2251799813685247ULL,
    4503599627370495ULL, 9007199254740991ULL, 18014398509481983ULL, 36028797018963967ULL,
    72057594037927935ULL, 144115188075855871ULL, 288230376151711743ULL, 576460752303423487ULL,
    1152921504606846975ULL, 2305843009213693951ULL, 4611686018427387903ULL, 9223372036854775807ULL
  };

  for (size_t j = 0; j < qubits_sorted.size(); j++) {
    lowbits = retval & _MASKS[qubits_sorted[j]];
    retval >>= qubits_sorted[j];
    retval <<= qubits_sorted[j] + 1;
    retval |= lowbits;
  }
  return retval;
}

template<size_t N>
areg_t<1ULL << N> _indexes(const areg_t<N> &qs, const areg_t<N> &qubits_sorted, const uint64_t k){
  areg_t<1ULL << N> ret;
  const std::array<uint64_t, 64> _BITS = {
    1ULL, 2ULL, 4ULL, 8ULL,
    16ULL, 32ULL, 64ULL, 128ULL,
    256ULL, 512ULL, 1024ULL, 2048ULL,
    4096ULL, 8192ULL, 16384ULL, 32768ULL,
    65536ULL, 131072ULL, 262144ULL, 524288ULL,
    1048576ULL, 2097152ULL, 4194304ULL, 8388608ULL,
    16777216ULL, 33554432ULL, 67108864ULL, 134217728ULL,
    268435456ULL, 536870912ULL, 1073741824ULL, 2147483648ULL,
    4294967296ULL, 8589934592ULL, 17179869184ULL, 34359738368ULL,
    68719476736ULL, 137438953472ULL, 274877906944ULL, 549755813888ULL,
    1099511627776ULL, 2199023255552ULL, 4398046511104ULL, 8796093022208ULL,
    17592186044416ULL, 35184372088832ULL, 70368744177664ULL, 140737488355328ULL,
    281474976710656ULL, 562949953421312ULL, 1125899906842624ULL, 2251799813685248ULL,
    4503599627370496ULL, 9007199254740992ULL, 18014398509481984ULL, 36028797018963968ULL,
    72057594037927936ULL, 144115188075855872ULL, 288230376151711744ULL, 576460752303423488ULL,
    1152921504606846976ULL, 2305843009213693952ULL, 4611686018427387904ULL, 9223372036854775808ULL
  };

  ret[0] = _index0(qubits_sorted, k);
  for (size_t i = 0; i < N; i++){
    const auto n = _BITS[i];
    const auto bit = _BITS[qs[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}

template<typename Lambda, typename list_t, typename param_t>
void _apply_lambda(uint64_t data_size, const uint64_t skip, Lambda&& func, const list_t &qubits,
  const unsigned omp_threads, const param_t &params){

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

template<size_t N>
inline bool _apply_avx_kernel(
  const areg_t<N>& qregs,
  const std::complex<float>* data,
  uint64_t data_size,
  const std::complex<float>* mat,
  uint_t omp_threads
){

  RealView<float> real = {data};
  ImaginaryView<float> img = {data};

  if(qregs.size() > 2 && qregs[2] == 2){
    auto lambda = [&](const areg_t<(1 << N)> &inds, const std::complex<float>* fmat)->void {
      _apply_matrix_float_avx_q0q1q2(real, img, fmat, inds, qregs);
    };

    _apply_lambda(data_size, 1, lambda, qregs, omp_threads, mat);

  }else if (qregs.size() > 1 && qregs[1] < 3){
    auto lambda = [&](const areg_t<(1 << N)> &inds, const std::complex<float>* fmat)->void {
      _apply_matrix_float_avx_qLqL(real, img, fmat, inds, qregs);
    };

    _apply_lambda(data_size, 2, lambda, qregs, omp_threads, mat);

  }else if (qregs[0] < 3){
    auto lambda = [&](const areg_t<(1 << N)> &inds, const std::complex<float>* fmat)->void {
      _apply_matrix_float_avx_qL(real, img, fmat, inds, qregs);
    };

    _apply_lambda(data_size, 4, lambda, qregs, omp_threads, mat);

  }else{
    auto lambda = [&](const areg_t<(1 << N)> &inds, const std::complex<float>* fmat)->void {
      _apply_matrix_float_avx(real, img, fmat, inds, qregs);
    };

    _apply_lambda(data_size, 8, lambda, qregs, omp_threads, mat);

  }
  return true;
}

template<size_t N>
inline bool _apply_avx_kernel(
  const areg_t<N>& qregs,
  const std::complex<double>* data,
  uint64_t data_size,
  const std::complex<double>* mat,
  uint_t omp_threads
){

  const RealView<double> real = {data};
  const ImaginaryView<double> img = {data};

  if (qregs.size() > 1 && qregs[1] == 1) {
    auto lambda = [&](const areg_t<(1 << N)>& inds, const std::complex<double>* dmat) -> void {
      _apply_matrix_double_avx_q0q1(real, img, dmat, inds, qregs);
    };

    _apply_lambda(data_size, 1, lambda, qregs, omp_threads, mat);

  } else if (qregs[0] < 2) {
    auto lambda = [&](const areg_t<(1 << N)>& inds, const std::complex<double>* dmat) -> void {
      _apply_matrix_double_avx_qL(real, img, dmat, inds, qregs);
    };

    _apply_lambda(data_size, 2, lambda, qregs, omp_threads, mat);

  } else {
    auto lambda = [&](const areg_t<(1 << N)>& inds, const std::complex<double>* dmat) -> void {
      _apply_matrix_double_avx(real, img, dmat, inds, qregs);
    };

    _apply_lambda(data_size, 4, lambda, qregs, omp_threads, mat);

  }
}

template<typename FloatType, size_t N>
inline bool apply_matrix_avx(
    const std::complex<FloatType>* data,
    uint64_t data_size,
    const areg_t<N> qregs,
    const cvector_t<FloatType>& mat,
    uint_t omp_threads
){

  auto transpose = [](const cvector_t<FloatType>& matrix) -> cvector_t<FloatType> {
      cvector_t<double> transposed(matrix.size());

      // We deal with MxM matrices, so let's take rows for example
      auto rows = log2(matrix.size());
      for(size_t i = 0; i < rows; ++i){
          for(size_t j = 0; j < rows; ++j){
              transposed[ i * rows + j] = matrix[ j * rows + i ];
          }
      }
      return transposed;
  };

  auto transposed_mat = transpose(mat);

  auto ordered_qregs = qregs;
  reorder(ordered_qregs, transposed_mat);

  return _apply_avx_kernel<N>(ordered_qregs, data, data_size, transposed_mat.data(), omp_threads);
}


template<typename FloatType>
inline bool apply_matrix_avx(
  const std::complex<FloatType>* qv_data,
  uint64_t data_size,
  const reg_t& qregs,
  const cvector_t<FloatType>& mat,
  uint_t omp_threads
){

  switch (qregs.size()) {
  case 1:
    return apply_matrix_avx(qv_data, data_size, to_array<1>(qregs), mat, omp_threads);
  case 2:
    return apply_matrix_avx(qv_data, data_size, to_array<2>(qregs), mat, omp_threads);
  case 3:
    return apply_matrix_avx(qv_data, data_size, to_array<3>(qregs), mat, omp_threads);
  case 4:
    return apply_matrix_avx(qv_data, data_size, to_array<4>(qregs), mat, omp_threads);
  case 5:
    return apply_matrix_avx(qv_data, data_size, to_array<5>(qregs), mat, omp_threads);
  case 6:
    return apply_matrix_avx(qv_data, data_size, to_array<6>(qregs), mat, omp_threads);
  default:
  // TODO: Return Enum
    return false;
  }
}

} // namespace QV
#endif
