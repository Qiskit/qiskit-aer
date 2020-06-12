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
#include "indexes.hpp"
#include "qubitvector.hpp"

namespace {

template<typename FloatType>
using m256_t = typename std::conditional<std::is_same<FloatType, double>::value, __m256d, __m256>::type;

// These Views are helpers for encapsulate QubitVector::data_ access, specifically
// for SIMD vectorization.
// The data_ variable is allocated dynamically and of the type:
// std::complex<double/float>*, and SIMD operations requires getting
// a number of continuous values to vectorice the operation.
// As we are dealing with memory layout of the form:
//  [real|imag|real|imag|real|imag|...]
// we require special indexing to the elements.
template<typename FloatType>
struct RealVectorView {
  RealVectorView() = delete;
  // Unfortunately, shared_ptr implies allocations and we cannot afford
  // them in this piece of code, so this is the reason to use raw pointers.
  RealVectorView(std::complex<FloatType>* data) : data(data){}
  inline FloatType* operator[](size_t i){
    return &reinterpret_cast<FloatType*>(data)[i * 2];
  }
  inline const FloatType* operator[](size_t i) const {
    return &reinterpret_cast<const FloatType*>(data)[i * 2];
  }
  std::complex<FloatType>* data = nullptr;
};

template<typename FloatType>
struct ImaginaryVectorView : std::false_type {};

template<>
struct ImaginaryVectorView<double> {
  ImaginaryVectorView() = delete;
  ImaginaryVectorView(std::complex<double>* data) : data(data){}
  // SIMD vectorization takes n bytes depending on the underlaying type, so
  // for doubles, SIMD loads 4 consecutive values (4 * sizeof(double) = 32 bytes)
  inline double* operator[](size_t i){
    return &reinterpret_cast<double*>(data)[i * 2 + 4];
  }
  inline const double* operator[](size_t i) const {
    return &reinterpret_cast<const double*>(data)[i * 2 + 4];
  }
  std::complex<double>* data = nullptr;
};

template<>
struct ImaginaryVectorView<float> {
  ImaginaryVectorView() = delete;
  ImaginaryVectorView(std::complex<float>* data) : data(data){}
  // SIMD vectorization takes n bytes depending on the underlaying type, so
  // for floats, SIMD loads 8 consecutive (8 * sizeof(float) = 32 bytes)
  inline float* operator[](size_t i){
    return &reinterpret_cast<float*>(data)[i * 2 + 8];
  }
  inline const float* operator[](size_t i) const {
    return &reinterpret_cast<const float*>(data)[i * 2 + 8];
  }
  std::complex<float>* data = nullptr;
};

static auto _mm256_mul(const m256_t<double>& left, const m256_t<double>& right){
  return _mm256_mul_pd(left, right);
}

static auto _mm256_mul(const m256_t<float>& left, const m256_t<float>& right){
  return _mm256_mul_ps(left, right);
}

static auto _mm256_fnmadd(const m256_t<double>& left, const m256_t<double>& right, const m256_t<double>& ret){
  return _mm256_fnmadd_pd(left, right, ret);
}

static auto _mm256_fnmadd(const m256_t<float>& left, const m256_t<float>& right, const m256_t<float>& ret){
  return _mm256_fnmadd_ps(left, right, ret);
}

static auto _mm256_fmadd(const m256_t<double>& left, const m256_t<double>& right, const m256_t<double>& ret){
  return _mm256_fmadd_pd(left, right, ret);
}

static auto _mm256_fmadd(const m256_t<float>& left, const m256_t<float>& right, const m256_t<float>& ret){
  return _mm256_fmadd_ps(left, right, ret);
}

static auto _mm256_set1(double d){
  return _mm256_set1_pd(d);
}

static auto _mm256_set1(float f){
  return _mm256_set1_ps(f);
}

static auto _mm256_load(double const* d){
  return _mm256_load_pd(d);
}

static auto _mm256_load(float const* f){
  return _mm256_load_ps(f);
}

static void _mm256_store(float* f, const m256_t<float>& c){
  _mm256_store_ps(f, c);
}

static void _mm256_store(double* d, const m256_t<double>& c){
  _mm256_store_pd(d, c);
}

template<typename FloatType>
static inline void _mm_complex_multiply(m256_t<FloatType>& real_ret, m256_t<FloatType>& imag_ret, m256_t<FloatType>& real_left,
  m256_t<FloatType>& imag_left, const m256_t<FloatType>& real_right, const m256_t<FloatType>& imag_right){
    real_ret = _mm256_mul(real_left, real_right);
    imag_ret = _mm256_mul(real_left, imag_right);
    real_ret = _mm256_fnmadd(imag_left, imag_right, real_ret);
    imag_ret = _mm256_fmadd(imag_left, real_right, imag_ret);
}

template<typename FloatType>
static inline void _mm_complex_multiply_add(m256_t<FloatType>& real_ret, m256_t<FloatType>& imag_ret, m256_t<FloatType>& real_left,
  m256_t<FloatType>& imag_left, const m256_t<FloatType>& real_right, const m256_t<FloatType>& imag_right){
    real_ret = _mm256_fmadd(real_left, real_right, real_ret);
    imag_ret = _mm256_fmadd(real_left, imag_right, imag_ret);
    real_ret = _mm256_fnmadd(imag_left, imag_right, real_ret);
    imag_ret = _mm256_fmadd(imag_left, real_right, imag_ret);
}

template<typename FloatType>
static inline void _mm_complex_inner_product(size_t dim, m256_t<FloatType> vreals[], m256_t<FloatType> vimags[],
  const FloatType* cmplxs, m256_t<FloatType>& vret_real, m256_t<FloatType>& vret_imag,
  m256_t<FloatType>& vtmp_0, m256_t<FloatType>& vtmp_1){
    vtmp_0 = _mm256_set1(cmplxs[0]);
    vtmp_1 = _mm256_set1(cmplxs[1]);
    _mm_complex_multiply<FloatType>(vret_real, vret_imag, vreals[0], vimags[0], vtmp_0, vtmp_1);
    for (size_t i = 1; i < dim; ++i){
      vtmp_0 = _mm256_set1(cmplxs[i * 2]);
      vtmp_1 = _mm256_set1(cmplxs[i * 2 + 1]);
      _mm_complex_multiply_add<FloatType>(vret_real, vret_imag, vreals[i], vimags[i], vtmp_0, vtmp_1);
    }
}

static inline void _mm_load_twoarray_complex(const double * real_addr_0, const double * imag_addr_1,
  m256_t<double>& real_ret, m256_t<double>& imag_ret){
    real_ret = _mm256_load(real_addr_0);
    imag_ret = _mm256_load(imag_addr_1);
    auto tmp0 = _mm256_permute4x64_pd(real_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
    auto tmp1 = _mm256_permute4x64_pd(imag_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
    real_ret = _mm256_blend_pd(real_ret, tmp1, 0b1010);
    imag_ret = _mm256_blend_pd(tmp0, imag_ret, 0b1010);
    real_ret = _mm256_permute4x64_pd(real_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
    imag_ret = _mm256_permute4x64_pd(imag_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
}

static inline void _mm_load_twoarray_complex(const float * real_addr_0, const float * imag_addr_1,
  m256_t<float>& real_ret, m256_t<float>& imag_ret){
    real_ret = _mm256_load(real_addr_0);
    imag_ret = _mm256_load(imag_addr_1);
    auto tmp0 = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    auto tmp1 = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    real_ret = _mm256_blend_ps(real_ret, tmp1, 0b10101010);
    imag_ret = _mm256_blend_ps(tmp0, imag_ret, 0b10101010);
    real_ret = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
    imag_ret = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
}

static inline void _mm_store_twoarray_complex(m256_t<float>& real_ret, m256_t<float>& imag_ret,
  float * cmplx_addr_0, float * cmplx_addr_1){
    real_ret = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    imag_ret = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    auto tmp0 = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    auto tmp1 = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    real_ret = _mm256_blend_ps(real_ret, tmp1, 0b10101010);
    imag_ret = _mm256_blend_ps(tmp0, imag_ret, 0b10101010);
    _mm256_store(cmplx_addr_0, real_ret);
    _mm256_store(cmplx_addr_1, imag_ret);
}

static inline void _mm_store_twoarray_complex(m256_t<double>& real_ret, m256_t<double>& imag_ret,
  double * cmplx_addr_0, double * cmplx_addr_1){
    real_ret = _mm256_permute4x64_pd(real_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
    imag_ret = _mm256_permute4x64_pd(imag_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
    auto tmp0 = _mm256_permute4x64_pd(real_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
    auto tmp1 = _mm256_permute4x64_pd(imag_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
    real_ret = _mm256_blend_pd(real_ret, tmp1, 0b1010);
    imag_ret = _mm256_blend_pd(tmp0, imag_ret, 0b1010);
    _mm256_store(cmplx_addr_0, real_ret);
    _mm256_store(cmplx_addr_1, imag_ret);
}

template<size_t N, typename FloatType>
inline void reorder(QV::areg_t<N>& qregs, QV::cvector_t<FloatType>& mat) {
  if(qregs.size() < 2)
    return;

  auto dim = (1UL << N);
  auto qreg_orig = qregs;
  // TODO Haven't we already ordered?
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
  std::copy(mat.begin(),mat.end(), std::back_inserter(mat_orig));

  // TODO Using standard interface for transposing instead of memory indexing
  for(size_t i = 0; i < dim; ++i) {
    for(size_t j = 0; j < dim; ++j) {
      size_t oldidx = i * dim + j;
      size_t newidx = indexes[i] * dim + indexes[j];
      reinterpret_cast<FloatType*>(mat.data())[newidx * 2] = reinterpret_cast<FloatType*>(mat_orig.data())[oldidx * 2];
      reinterpret_cast<FloatType*>(mat.data())[newidx * 2 + 1] = reinterpret_cast<FloatType*>(mat_orig.data())[oldidx * 2 + 1];
    }
  }
}

template<size_t N, typename data_t>
inline void _reorder(QV::areg_t<N>& qreg, data_t* fmat) {
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
inline QV::areg_t<N> to_array(const QV::reg_t& vec) {
  QV::areg_t<N> ret;
  std::copy_n(vec.begin(), N, std::begin(ret));
  return ret;
}

} // End anonymous namespace

namespace QV {

template<size_t N>
inline void _apply_matrix_float_avx_q0q1q2(
    RealVectorView<float>& reals,
    ImaginaryVectorView<float>& imags,
    const QV::cvector_t<float>& mat,
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

  const float * fmat = reinterpret_cast<const float *>(mat.data());


  for(unsigned i = 0; i < (1ULL << N); i += 8){
    auto idx = inds[i];
    _mm_load_twoarray_complex(reals[idx], imags[idx], vreals[i], vimags[i]);

    for (unsigned j = 1; j < 8; ++j) {
      vreals[i + j] = _mm256_permutevar8x32_ps(vreals[i], _MASKS[j - 1]);
      vimags[i + j] = _mm256_permutevar8x32_ps(vimags[i], _MASKS[j - 1]);
    }
  }

  unsigned midx = 0;
  for(unsigned i = 0; i < inds.size(); i += 8){
    auto idx = inds[i];
    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    for(size_t j = 1; j < 8; ++j){
      _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
      midx += (1ULL << (N + 1));

      real_ret1 = _mm256_permutevar8x32_ps(real_ret1, _MASKS[j - 1]);
      imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, _MASKS[j - 1]);

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
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_float_avx_qLqL(
    RealVectorView<float>& reals,
    ImaginaryVectorView<float>& imags,
    const QV::cvector_t<float>& mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N> qregs
){

  __m256i masks[3];
  __m256 real_ret, imag_ret, real_ret1, imag_ret1;
  __m256 vreals[1ULL << N], vimags[1ULL << N];
  __m256 tmp0, tmp1;

   const float * fmat = reinterpret_cast<const float *>(mat.data());

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
    _mm_load_twoarray_complex(reals[idx], imags[idx], vreals[i], vimags[i]);

    for (size_t j = 0; j < 3; ++j) {
      vreals[i + j + 1] = _mm256_permutevar8x32_ps(vreals[i], masks[j]);
      vimags[i + j + 1] = _mm256_permutevar8x32_ps(vimags[i], masks[j]);
    }
  }

  size_t midx = 0;
  for (size_t i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    for (unsigned j = 0; j < 3; ++j){
      _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
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
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_float_avx_qL(
    RealVectorView<float>& reals,
    ImaginaryVectorView<float>& imags,
    const QV::cvector_t<float>& mat,
    const areg_t<1ULL << N>& inds,
    const areg_t<N>& qregs
){

  __m256i mask;
  __m256 real_ret, imag_ret, real_ret1, imag_ret1;
  __m256 vreals[1ULL << N], vimags[1ULL << N];
  __m256 tmp0, tmp1;

  const float * fmat = reinterpret_cast<const float *>(mat.data());

  if (qregs[0] == 0) {
    mask = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
  } else if (qregs[0] == 1) {
    mask = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
  } else { //if (q0 == 2) {
    mask = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
  }

  for (unsigned i = 0; i < (1ULL << N); i += 2){
    auto idx = inds[i];
    _mm_load_twoarray_complex(reals[idx], imags[idx], vreals[i], vimags[i]);

    vreals[i + 1] = _mm256_permutevar8x32_ps(vreals[i], mask);
    vimags[i + 1] = _mm256_permutevar8x32_ps(vimags[i], mask);
  }

  unsigned midx = 0;
  for (unsigned i = 0; i < (1ULL << N); i += 2) {
    auto idx = inds[i];
    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    real_ret1 = _mm256_permutevar8x32_ps(real_ret1, mask);
    imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, mask);

    real_ret = (qregs[0] == 0) ? _mm256_blend_ps(real_ret, real_ret1, 0b10101010) : // (0,H,H)
               (qregs[0] == 1) ? _mm256_blend_ps(real_ret, real_ret1, 0b11001100) : // (1,H,H)
                                 _mm256_blend_ps(real_ret, real_ret1, 0b11110000); //  (2,H,H)
    imag_ret = (qregs[0] == 0) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10101010) : // (0,H,H)
               (qregs[0] == 1) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b11001100) : // (1,H,H)
                                 _mm256_blend_ps(imag_ret, imag_ret1, 0b11110000); //  (2,H,H)

    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_float_avx(
    RealVectorView<float>& reals,
    ImaginaryVectorView<float>& imags,
    const QV::cvector_t<float>& mat,
    const areg_t<1ULL << N>& inds,
    const areg_t<N>& qregs
){

  __m256 real_ret, imag_ret;
  __m256 vreals[1ULL << N], vimags[1ULL << N];
  __m256 tmp0, tmp1;

  const float * fmat = reinterpret_cast<const float *>(mat.data());

  for(unsigned i = 0; i < (1ULL << N); ++i){
    auto idx = inds[i];
    _mm_load_twoarray_complex(reals[idx], imags[idx], vreals[i], vimags[i]);
  }

  unsigned midx = 0;
  for(unsigned i = 0; i < (1ULL << N); ++i){
    auto idx = inds[i];
    _mm_complex_inner_product<float>((1ULL << N), vreals, vimags, (&fmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_double_avx_q0q1(
    RealVectorView<double>& reals,
    ImaginaryVectorView<double>& imags,
    const QV::cvector_t<double>& mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N>& qregs
){

  const int PERM_D_Q0Q1_0 = 3 * 64 + 2 * 16 + 0 * 4 + 1 * 1;
  const int PERM_D_Q0Q1_1 = 3 * 64 + 0 * 16 + 1 * 4 + 2 * 1;
  const int PERM_D_Q0Q1_2 = 0 * 64 + 2 * 16 + 1 * 4 + 3 * 1;


  __m256d real_ret, imag_ret, real_ret1, imag_ret1;
  __m256d vreals[1ULL << N], vimags[1ULL << N];
  __m256d tmp0, tmp1;

  const double * dmat = reinterpret_cast<const double *>(mat.data());

  for(size_t i = 0; i < (1ULL << N); i += 4) {
    auto idx = inds[i];
    _mm_load_twoarray_complex(reals[idx], imags[idx], vreals[i], vimags[i]);
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
    _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));
    for (size_t j = 1; j < 4; ++j) {
      _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
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
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_double_avx_qL(
    RealVectorView<double>& reals,
    ImaginaryVectorView<double>& imags,
    const QV::cvector_t<double>& mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N>& qregs
){

  const int PERM_D_Q0 = 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1;
  const int PERM_D_Q1 = 1 * 64 + 0 * 16 + 3 * 4 + 2 * 1;

  __m256d real_ret, imag_ret, real_ret1, imag_ret1;
  __m256d vreals[1ULL << N], vimags[1ULL << N];
  __m256d tmp0, tmp1;

  const double * dmat = reinterpret_cast<const double *>(mat.data());

  for(unsigned i = 0; i < (1ULL << N); i += 2){
    auto idx = inds[i];
    _mm_load_twoarray_complex(reals[idx], imags[idx], vreals[i], vimags[i]);
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
    _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));

    _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret1, imag_ret1, tmp0, tmp1);
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
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

template<size_t N>
inline void _apply_matrix_double_avx(
    RealVectorView<double>& reals,
    ImaginaryVectorView<double>& imags,
    const QV::cvector_t<double>& mat,
    const areg_t<1ULL << N> &inds,
    const areg_t<N> qregs
){

  __m256d real_ret, imag_ret;
  __m256d vreals[1ULL << N], vimags[1ULL << N];
  __m256d tmp0, tmp1;

  const double * dmat = reinterpret_cast<const double *>(mat.data());

  for(unsigned i = 0; i < (1ULL << N); ++i){
    auto idx = inds[i];
    _mm_load_twoarray_complex(reals[idx], imags[idx], vreals[i], vimags[i]);
  }

  unsigned midx = 0;
  for(unsigned i = 0; i < (1ULL << N); ++i){
    auto idx = inds[i];
    _mm_complex_inner_product<double>((1ULL << N), vreals, vimags, (&dmat[midx]), real_ret, imag_ret, tmp0, tmp1);
    midx += (1ULL << (N + 1));
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[idx], imags[idx]);
  }
}

enum class Avx {
  NotApplied,
  Applied
};

template<size_t N>
inline Avx _apply_avx_kernel(
  const areg_t<N>& qregs,
  std::complex<float>* data,
  uint64_t data_size,
  const QV::cvector_t<float>& mat,
  uint_t omp_threads
){

  RealVectorView<float> real = {data};
  ImaginaryVectorView<float> img = {data};

  if(qregs.size() > 2 && qregs[2] == 2){
    auto lambda = [&](const areg_t<(1 << N)> &inds, const QV::cvector_t<float>& fmat)->void {
      _apply_matrix_float_avx_q0q1q2(real, img, fmat, inds, qregs);
    };

    QV::apply_lambda(0, data_size, 1, omp_threads, lambda, qregs, mat);

  }else if (qregs.size() > 1 && qregs[1] < 3){
    auto lambda = [&](const areg_t<(1 << N)> &inds, const QV::cvector_t<float>& fmat)->void {
      _apply_matrix_float_avx_qLqL(real, img, fmat, inds, qregs);
    };

    QV::apply_lambda(0, data_size, 2, omp_threads, lambda, qregs, mat);

  }else if (qregs[0] < 3){
    auto lambda = [&](const areg_t<(1 << N)> &inds, const QV::cvector_t<float>& fmat)->void {
      _apply_matrix_float_avx_qL(real, img, fmat, inds, qregs);
    };

    QV::apply_lambda(0, data_size, 4, omp_threads, lambda, qregs, mat);

  }else{
    auto lambda = [&](const areg_t<(1 << N)> &inds, const QV::cvector_t<float>& fmat)->void {
      _apply_matrix_float_avx(real, img, fmat, inds, qregs);
    };

    QV::apply_lambda(0, data_size, 8, omp_threads, lambda, qregs, mat);

  }
  return Avx::Applied;
}

template<size_t N>
inline Avx _apply_avx_kernel(
  const areg_t<N>& qregs,
  std::complex<double>* data,
  uint64_t data_size,
  const QV::cvector_t<double>& mat,
  uint_t omp_threads
){

  RealVectorView<double> real = {data};
  ImaginaryVectorView<double> img = {data};

  if (qregs.size() > 1 && qregs[1] == 1) {
    auto lambda = [&](const areg_t<(1 << N)>& inds, const QV::cvector_t<double>& dmat) -> void {
      _apply_matrix_double_avx_q0q1(real, img, dmat, inds, qregs);
    };

    QV::apply_lambda(0, data_size, 1, omp_threads, lambda, qregs, mat);

  } else if (qregs[0] < 2) {
    auto lambda = [&](const areg_t<(1 << N)>& inds, const QV::cvector_t<double>& dmat) -> void {
      _apply_matrix_double_avx_qL(real, img, dmat, inds, qregs);
    };

    QV::apply_lambda(0, data_size, 2, omp_threads, lambda, qregs, mat);

  } else {
    auto lambda = [&](const areg_t<(1 << N)>& inds, const QV::cvector_t<double>& dmat) -> void {
      _apply_matrix_double_avx(real, img, dmat, inds, qregs);
    };

    QV::apply_lambda(0, data_size, 4, omp_threads, lambda, qregs, mat);

  }
  return Avx::Applied;
}


template<typename FloatType>
typename std::enable_if<std::is_same<FloatType, double>::value, bool>::type
is_simd_applicable(uint64_t data_size){
  if (data_size <= 4)
    return false;
  return true;
}

template<typename FloatType>
typename std::enable_if<std::is_same<FloatType, float>::value, bool>::type
is_simd_applicable(uint64_t data_size){
  if (data_size <= 8)
    return false;
  return true;
}

template<typename FloatType, size_t N>
inline Avx apply_matrix_avx(
    std::complex<FloatType>* data,
    uint64_t data_size,
    const areg_t<N>& qregs,
    const cvector_t<FloatType>& mat,
    uint_t omp_threads
){

  if (!is_simd_applicable<FloatType>(data_size))
    return Avx::NotApplied;

  auto transpose = [](const cvector_t<FloatType>& matrix) -> cvector_t<FloatType> {
      cvector_t<FloatType> transposed(matrix.size());
      // We deal with MxM matrices, so let's take rows for example
      auto rows = sqrt(matrix.size());
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
  return _apply_avx_kernel<N>(ordered_qregs, data, data_size, transposed_mat, omp_threads);
}

template<typename FloatType>
inline Avx apply_matrix_avx(
  std::complex<FloatType>* qv_data,
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
    return Avx::NotApplied;
  }
}

} // namespace QV
#endif
