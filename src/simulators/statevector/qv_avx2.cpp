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

#include "qv_avx2.hpp"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

/**
 * DISCLAIMER: We want to compile this code in isolation of the rest of the
 *codebase, because it contains AVX specific instructions, so is very CPU arch
 *dependant. We will detect CPU features at runtime and derive the execution
 *path over here if AVX is supported, if it's not supported the QubitVector
 *normal class will be used instead (so no SIMD whatsoever). Because of this, we
 *don't want to depend on any other library, otherwise the linker could take AVX
 *version of the symbols from this file, and use them elsewhere, making the
 *runtime CPU detection useless because this other symbols, with AVX code inside
 *thenm, will be present in other places of the final executable, out of our
 *control.
 *
 * DISCLAIMER 2: There's a lot of raw pointers and pointer arithmetic, sorry.
 **/

namespace {

/** Remember we cannot use STL (or memcpy) **/
template <typename T, typename U>
void copy(T dest, const U orig, size_t size) {
  for (auto i = 0; i < size; ++i)
    dest[i] = orig[i];
}

inline void fill_indices(uint64_t index0,
                         uint64_t* indexes,
                         const size_t indexes_size,
                         const uint64_t* qregs,
                         const size_t qregs_size) {
  for (size_t i = 0; i < indexes_size; ++i)
    indexes[i] = index0;

  for (size_t n = 0; n < qregs_size; ++n)
    for (size_t i = 0; i < indexes_size; i += (1 << (n + 1)))
      for (size_t j = 0; j < (1U << n); ++j)
        indexes[i + j + (1U << n)] += (1ULL << qregs[n]);
}

const uint64_t MASKS[] = {0ULL,
                          1ULL,
                          3ULL,
                          7ULL,
                          15ULL,
                          31ULL,
                          63ULL,
                          127ULL,
                          255ULL,
                          511ULL,
                          1023ULL,
                          2047ULL,
                          4095ULL,
                          8191ULL,
                          16383ULL,
                          32767ULL,
                          65535ULL,
                          131071ULL,
                          262143ULL,
                          524287ULL,
                          1048575ULL,
                          2097151ULL,
                          4194303ULL,
                          8388607ULL,
                          16777215ULL,
                          33554431ULL,
                          67108863ULL,
                          134217727ULL,
                          268435455ULL,
                          536870911ULL,
                          1073741823ULL,
                          2147483647ULL,
                          4294967295ULL,
                          8589934591ULL,
                          17179869183ULL,
                          34359738367ULL,
                          68719476735ULL,
                          137438953471ULL,
                          274877906943ULL,
                          549755813887ULL,
                          1099511627775ULL,
                          2199023255551ULL,
                          4398046511103ULL,
                          8796093022207ULL,
                          17592186044415ULL,
                          35184372088831ULL,
                          70368744177663ULL,
                          140737488355327ULL,
                          281474976710655ULL,
                          562949953421311ULL,
                          1125899906842623ULL,
                          2251799813685247ULL,
                          4503599627370495ULL,
                          9007199254740991ULL,
                          18014398509481983ULL,
                          36028797018963967ULL,
                          72057594037927935ULL,
                          144115188075855871ULL,
                          288230376151711743ULL,
                          576460752303423487ULL,
                          1152921504606846975ULL,
                          2305843009213693951ULL,
                          4611686018427387903ULL,
                          9223372036854775807ULL};

inline uint64_t index0(const uint64_t* sorted_qubits,
                       const size_t sorted_qubits_size,
                       const uint64_t k) {
  uint64_t lowbits, retval = k;
  for (size_t j = 0; j < sorted_qubits_size; j++) {
    lowbits = retval & MASKS[sorted_qubits[j]];
    retval >>= sorted_qubits[j];
    retval <<= sorted_qubits[j] + 1;
    retval |= lowbits;
  }
  return retval;
}

template <typename Lambda, typename param_t>
void avx_apply_lambda(const uint64_t data_size,
                      const uint64_t skip,
                      Lambda&& func,
                      const uint64_t* sorted_qubits,
                      const size_t sorted_qubits_size,
                      const size_t omp_threads,
                      const param_t& params) {
  const int64_t END = data_size >> sorted_qubits_size;

#pragma omp parallel for if (omp_threads > 1) num_threads(omp_threads)
  for (int64_t k = 0; k < END; k += skip) {
    const auto index = index0(sorted_qubits, sorted_qubits_size, k);
    std::forward<Lambda>(func)(index, params);
  }
}

template <typename FloatType>
using m256_t = typename std::
    conditional<std::is_same<FloatType, double>::value, __m256d, __m256>::type;

// These Views are helpers for encapsulate access to real and imaginary parts of
// the original source (QubitVector::data_). SIMD operations requires getting a
// number of continuous values to vectorice the operation. As we are dealing
// with memory layout of the form:
//  [real|imag|real|imag|real|imag|...]
// we require special indexing to the elements.
template <typename FloatType>
struct RealVectorView {
  RealVectorView() = delete;
  // Unfortunately, shared_ptr implies allocations and we cannot afford
  // them in this piece of code, so this is the reason to use raw pointers.
  RealVectorView(FloatType* data) : data(data) {}
  inline FloatType* operator[](size_t i) { return &data[i * 2]; }
  inline const FloatType* operator[](size_t i) const { return &data[i * 2]; }
  FloatType* data = nullptr;
};

template <typename FloatType>
struct ImaginaryVectorView : std::false_type {};

template <>
struct ImaginaryVectorView<double> {
  ImaginaryVectorView() = delete;
  ImaginaryVectorView(double* data) : data(data) {}
  // SIMD vectorization takes n bytes depending on the underlaying type, so
  // for doubles, SIMD loads 4 consecutive values (4 * sizeof(double) = 32
  // bytes)
  inline double* operator[](size_t i) { return &data[i * 2 + 4]; }
  inline const double* operator[](size_t i) const { return &data[i * 2 + 4]; }
  double* data = nullptr;
};

template <>
struct ImaginaryVectorView<float> {
  ImaginaryVectorView() = delete;
  ImaginaryVectorView(float* data) : data(data) {}
  // SIMD vectorization takes n bytes depending on the underlaying type, so
  // for floats, SIMD loads 8 consecutive (8 * sizeof(float) = 32 bytes)
  inline float* operator[](size_t i) { return &data[i * 2 + 8]; }
  inline const float* operator[](size_t i) const { return &data[i * 2 + 8]; }
  float* data = nullptr;
};

static auto _mm256_mul(const m256_t<double>& left,
                       const m256_t<double>& right) {
  return _mm256_mul_pd(left, right);
}

static auto _mm256_mul(const m256_t<float>& left, const m256_t<float>& right) {
  return _mm256_mul_ps(left, right);
}

static auto _mm256_fnmadd(const m256_t<double>& left,
                          const m256_t<double>& right,
                          const m256_t<double>& ret) {
  return _mm256_fnmadd_pd(left, right, ret);
}

static auto _mm256_fnmadd(const m256_t<float>& left,
                          const m256_t<float>& right,
                          const m256_t<float>& ret) {
  return _mm256_fnmadd_ps(left, right, ret);
}

static auto _mm256_fmadd(const m256_t<double>& left,
                         const m256_t<double>& right,
                         const m256_t<double>& ret) {
  return _mm256_fmadd_pd(left, right, ret);
}

static auto _mm256_fmadd(const m256_t<float>& left,
                         const m256_t<float>& right,
                         const m256_t<float>& ret) {
  return _mm256_fmadd_ps(left, right, ret);
}

static auto _mm256_set1(double d) {
  return _mm256_set1_pd(d);
}

static auto _mm256_set1(float f) {
  return _mm256_set1_ps(f);
}

static auto _mm256_load(double const* d) {
  return _mm256_load_pd(d);
}

static auto _mm256_load(float const* f) {
  return _mm256_load_ps(f);
}

static void _mm256_store(float* f, const m256_t<float>& c) {
  _mm256_store_ps(f, c);
}

static void _mm256_store(double* d, const m256_t<double>& c) {
  _mm256_store_pd(d, c);
}

template <typename FloatType>
static inline void _mm_complex_multiply(m256_t<FloatType>& real_ret,
                                        m256_t<FloatType>& imag_ret,
                                        m256_t<FloatType>& real_left,
                                        m256_t<FloatType>& imag_left,
                                        const m256_t<FloatType>& real_right,
                                        const m256_t<FloatType>& imag_right) {
  real_ret = _mm256_mul(real_left, real_right);
  imag_ret = _mm256_mul(real_left, imag_right);
  real_ret = _mm256_fnmadd(imag_left, imag_right, real_ret);
  imag_ret = _mm256_fmadd(imag_left, real_right, imag_ret);
}

template <typename FloatType>
static inline void _mm_complex_multiply_add(
    m256_t<FloatType>& real_ret,
    m256_t<FloatType>& imag_ret,
    m256_t<FloatType>& real_left,
    m256_t<FloatType>& imag_left,
    const m256_t<FloatType>& real_right,
    const m256_t<FloatType>& imag_right) {
  real_ret = _mm256_fmadd(real_left, real_right, real_ret);
  imag_ret = _mm256_fmadd(real_left, imag_right, imag_ret);
  real_ret = _mm256_fnmadd(imag_left, imag_right, real_ret);
  imag_ret = _mm256_fmadd(imag_left, real_right, imag_ret);
}

template <typename FloatType>
static inline void _mm_complex_inner_product(size_t dim,
                                             m256_t<FloatType> vreals[],
                                             m256_t<FloatType> vimags[],
                                             const FloatType* cmplxs,
                                             m256_t<FloatType>& vret_real,
                                             m256_t<FloatType>& vret_imag,
                                             m256_t<FloatType>& vtmp_0,
                                             m256_t<FloatType>& vtmp_1) {
  vtmp_0 = _mm256_set1(cmplxs[0]);
  vtmp_1 = _mm256_set1(cmplxs[1]);
  _mm_complex_multiply<FloatType>(vret_real, vret_imag, vreals[0], vimags[0],
                                  vtmp_0, vtmp_1);
  for (size_t i = 1; i < dim; ++i) {
    vtmp_0 = _mm256_set1(cmplxs[i * 2]);
    vtmp_1 = _mm256_set1(cmplxs[i * 2 + 1]);
    _mm_complex_multiply_add<FloatType>(vret_real, vret_imag, vreals[i],
                                        vimags[i], vtmp_0, vtmp_1);
  }
}

static inline void _mm_load_twoarray_complex(const double* real_addr_0,
                                             const double* imag_addr_1,
                                             m256_t<double>& real_ret,
                                             m256_t<double>& imag_ret) {
  real_ret = _mm256_load(real_addr_0);
  imag_ret = _mm256_load(imag_addr_1);
  auto tmp0 = _mm256_permute4x64_pd(real_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
  auto tmp1 = _mm256_permute4x64_pd(imag_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
  real_ret = _mm256_blend_pd(real_ret, tmp1, 0b1010);
  imag_ret = _mm256_blend_pd(tmp0, imag_ret, 0b1010);
  real_ret = _mm256_permute4x64_pd(real_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
  imag_ret = _mm256_permute4x64_pd(imag_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
}

static inline void _mm_load_twoarray_complex(const float* real_addr_0,
                                             const float* imag_addr_1,
                                             m256_t<float>& real_ret,
                                             m256_t<float>& imag_ret) {
  real_ret = _mm256_load(real_addr_0);
  imag_ret = _mm256_load(imag_addr_1);
  auto tmp0 = _mm256_permutevar8x32_ps(
      real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
  auto tmp1 = _mm256_permutevar8x32_ps(
      imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
  real_ret = _mm256_blend_ps(real_ret, tmp1, 0b10101010);
  imag_ret = _mm256_blend_ps(tmp0, imag_ret, 0b10101010);
  real_ret = _mm256_permutevar8x32_ps(real_ret,
                                      _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
  imag_ret = _mm256_permutevar8x32_ps(imag_ret,
                                      _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
}

static inline void _mm_store_twoarray_complex(m256_t<float>& real_ret,
                                              m256_t<float>& imag_ret,
                                              float* cmplx_addr_0,
                                              float* cmplx_addr_1) {
  real_ret = _mm256_permutevar8x32_ps(real_ret,
                                      _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
  imag_ret = _mm256_permutevar8x32_ps(imag_ret,
                                      _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
  auto tmp0 = _mm256_permutevar8x32_ps(
      real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
  auto tmp1 = _mm256_permutevar8x32_ps(
      imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
  real_ret = _mm256_blend_ps(real_ret, tmp1, 0b10101010);
  imag_ret = _mm256_blend_ps(tmp0, imag_ret, 0b10101010);
  _mm256_store(cmplx_addr_0, real_ret);
  _mm256_store(cmplx_addr_1, imag_ret);
}

static inline void _mm_store_twoarray_complex(m256_t<double>& real_ret,
                                              m256_t<double>& imag_ret,
                                              double* cmplx_addr_0,
                                              double* cmplx_addr_1) {
  real_ret = _mm256_permute4x64_pd(real_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
  imag_ret = _mm256_permute4x64_pd(imag_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
  auto tmp0 = _mm256_permute4x64_pd(real_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
  auto tmp1 = _mm256_permute4x64_pd(imag_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
  real_ret = _mm256_blend_pd(real_ret, tmp1, 0b1010);
  imag_ret = _mm256_blend_pd(tmp0, imag_ret, 0b1010);
  _mm256_store(cmplx_addr_0, real_ret);
  _mm256_store(cmplx_addr_1, imag_ret);
}

template <typename FloatType>
inline void reorder(const uint64_t* qreg_orig,
                    const size_t qregs_size,
                    uint64_t* qreg,
                    FloatType* mat) {
  const auto DIMENSION = (1ULL << qregs_size);

  auto sort = [](uint64_t* unordered, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size - 1 - i; ++j) {
        if (unordered[j] > unordered[j + 1]) {
          auto tmp = unordered[j];
          unordered[j] = unordered[j + 1];
          unordered[j + 1] = tmp;
        }
      }
    }
  };
  sort(qreg, qregs_size);

  auto build_mask = [&](size_t* masks) {
    for (size_t i = 0; i < qregs_size; ++i)
      for (size_t j = 0; j < qregs_size; ++j)
        if (qreg_orig[i] == qreg[j])
          masks[i] = 1U << j;
  };
  size_t masks[qregs_size];
  build_mask(masks);

  auto build_index = [&](size_t* indexes) {
    for (size_t i = 0; i < DIMENSION; ++i) {
      size_t index = 0U;
      for (size_t j = 0; j < qregs_size; ++j) {
        if (i & (1U << j))
          index |= masks[j];
      }
      indexes[i] = index;
    }
  };
  size_t indexes[1U << qregs_size];
  build_index(indexes);

  FloatType mat_temp[1 << (qregs_size * 2 + 1)];
  copy(mat_temp, mat, DIMENSION * DIMENSION * 2);

  for (size_t i = 0; i < DIMENSION; ++i) {
    for (size_t j = 0; j < DIMENSION; ++j) {
      size_t oldidx = i * DIMENSION + j;
      size_t newidx = indexes[i] * DIMENSION + indexes[j];
      mat[newidx * 2] = mat_temp[oldidx * 2];
      mat[newidx * 2 + 1] = mat_temp[oldidx * 2 + 1];
    }
  }
}

}  // End anonymous namespace

namespace AER {
namespace QV {

inline void _apply_matrix_float_avx_q0q1q2(RealVectorView<float>& reals,
                                           ImaginaryVectorView<float>& imags,
                                           const float* mat,
                                           const uint64_t* qregs,
                                           const size_t qregs_size,
                                           const uint64_t index0) {
  const auto indexes_size = (1ULL << qregs_size);
  uint64_t indexes[indexes_size];
  fill_indices(index0, indexes, indexes_size, qregs, qregs_size);

  __m256 real_ret, imag_ret, real_ret1, imag_ret1;
  __m256 vreals[1ULL << qregs_size], vimags[1ULL << qregs_size];
  __m256 tmp0, tmp1;

  const __m256i _MASKS[7] = {_mm256_set_epi32(7, 6, 5, 4, 3, 2, 0, 1),
                             _mm256_set_epi32(7, 6, 5, 4, 3, 0, 1, 2),
                             _mm256_set_epi32(7, 6, 5, 4, 0, 2, 1, 3),
                             _mm256_set_epi32(7, 6, 5, 0, 3, 2, 1, 4),
                             _mm256_set_epi32(7, 6, 0, 4, 3, 2, 1, 5),
                             _mm256_set_epi32(7, 0, 5, 4, 3, 2, 1, 6),
                             _mm256_set_epi32(0, 6, 5, 4, 3, 2, 1, 7)};

  for (size_t i = 0; i < indexes_size; i += 8) {
    auto index = indexes[i];
    _mm_load_twoarray_complex(reals[index], imags[index], vreals[i], vimags[i]);

    for (size_t j = 1; j < 8; ++j) {
      vreals[i + j] = _mm256_permutevar8x32_ps(vreals[i], _MASKS[j - 1]);
      vimags[i + j] = _mm256_permutevar8x32_ps(vimags[i], _MASKS[j - 1]);
    }
  }

  size_t mindex = 0;
  for (size_t i = 0; i < indexes_size; i += 8) {
    auto index = indexes[i];
    _mm_complex_inner_product<float>((1ULL << qregs_size), vreals, vimags,
                                     (&mat[mindex]), real_ret, imag_ret, tmp0,
                                     tmp1);
    mindex += (1ULL << (qregs_size + 1));

    for (size_t j = 1; j < 8; ++j) {
      _mm_complex_inner_product<float>((1ULL << qregs_size), vreals, vimags,
                                       (&mat[mindex]), real_ret1, imag_ret1,
                                       tmp0, tmp1);
      mindex += (1ULL << (qregs_size + 1));

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
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[index], imags[index]);
  }
}

inline void _apply_matrix_float_avx_qLqL(RealVectorView<float>& reals,
                                         ImaginaryVectorView<float>& imags,
                                         const float* mat,
                                         const uint64_t* qregs,
                                         const size_t qregs_size,
                                         const uint64_t index0) {
  __m256i masks[3];
  __m256 real_ret, imag_ret, real_ret1, imag_ret1;
  __m256 vreals[1ULL << qregs_size], vimags[1ULL << qregs_size];
  __m256 tmp0, tmp1;

  const auto indexes_size = (1ULL << qregs_size);
  uint64_t indexes[indexes_size];
  fill_indices(index0, indexes, indexes_size, qregs, qregs_size);

  if (qregs[1] == 1) {
    masks[0] = _mm256_set_epi32(7, 6, 4, 5, 3, 2, 0, 1);
    masks[1] = _mm256_set_epi32(7, 4, 5, 6, 3, 0, 1, 2);
    masks[2] = _mm256_set_epi32(4, 6, 5, 7, 0, 2, 1, 3);
  } else if (qregs[0] == 0) {
    masks[0] = _mm256_set_epi32(7, 6, 5, 4, 2, 3, 0, 1);
    masks[1] = _mm256_set_epi32(7, 2, 5, 0, 3, 6, 1, 4);
    masks[2] = _mm256_set_epi32(2, 6, 0, 4, 3, 7, 1, 5);
  } else {  // if (q0 == 1 && q1 == 2) {
    masks[0] = _mm256_set_epi32(7, 6, 5, 4, 1, 0, 3, 2);
    masks[1] = _mm256_set_epi32(7, 6, 1, 0, 3, 2, 5, 4);
    masks[2] = _mm256_set_epi32(1, 0, 5, 4, 3, 2, 7, 6);
  }

  for (size_t i = 0; i < (1ULL << qregs_size); i += 4) {
    auto index = indexes[i];
    _mm_load_twoarray_complex(reals[index], imags[index], vreals[i], vimags[i]);

    for (size_t j = 0; j < 3; ++j) {
      vreals[i + j + 1] = _mm256_permutevar8x32_ps(vreals[i], masks[j]);
      vimags[i + j + 1] = _mm256_permutevar8x32_ps(vimags[i], masks[j]);
    }
  }

  size_t mindex = 0;
  for (size_t i = 0; i < (1ULL << qregs_size); i += 4) {
    auto index = indexes[i];
    _mm_complex_inner_product<float>((1ULL << qregs_size), vreals, vimags,
                                     (&mat[mindex]), real_ret, imag_ret, tmp0,
                                     tmp1);
    mindex += (1ULL << (qregs_size + 1));

    for (size_t j = 0; j < 3; ++j) {
      _mm_complex_inner_product<float>((1ULL << qregs_size), vreals, vimags,
                                       (&mat[mindex]), real_ret1, imag_ret1,
                                       tmp0, tmp1);
      mindex += (1ULL << (qregs_size + 1));

      real_ret1 = _mm256_permutevar8x32_ps(real_ret1, masks[j]);
      imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, masks[j]);

      switch (j) {
        case 0:
          real_ret = (qregs[1] == 1)
                         ? _mm256_blend_ps(real_ret, real_ret1, 0b00100010)
                         :  // (0,1)
                         (qregs[0] == 0)
                             ? _mm256_blend_ps(real_ret, real_ret1, 0b00001010)
                             :  // (0,2)
                             _mm256_blend_ps(real_ret, real_ret1,
                                             0b00001100);  //  (1,2)
          imag_ret = (qregs[1] == 1)
                         ? _mm256_blend_ps(imag_ret, imag_ret1, 0b00100010)
                         :  // (0,1)
                         (qregs[0] == 0)
                             ? _mm256_blend_ps(imag_ret, imag_ret1, 0b00001010)
                             :  // (0,2)
                             _mm256_blend_ps(imag_ret, imag_ret1,
                                             0b00001100);  //  (1,2)
          break;
        case 1:
          real_ret = (qregs[1] == 1)
                         ? _mm256_blend_ps(real_ret, real_ret1, 0b01000100)
                         :  // (0,1)
                         (qregs[0] == 0)
                             ? _mm256_blend_ps(real_ret, real_ret1, 0b01010000)
                             :  // (0,2)
                             _mm256_blend_ps(real_ret, real_ret1,
                                             0b00110000);  //   (1,2)
          imag_ret = (qregs[1] == 1)
                         ? _mm256_blend_ps(imag_ret, imag_ret1, 0b01000100)
                         :  // (0,1)
                         (qregs[0] == 0)
                             ? _mm256_blend_ps(imag_ret, imag_ret1, 0b01010000)
                             :  // (0,2)
                             _mm256_blend_ps(imag_ret, imag_ret1,
                                             0b00110000);  //   (1,2)
          break;
        case 2:
          real_ret = (qregs[1] == 1)
                         ? _mm256_blend_ps(real_ret, real_ret1, 0b10001000)
                         :  // (0,1)
                         (qregs[0] == 0)
                             ? _mm256_blend_ps(real_ret, real_ret1, 0b10100000)
                             :  // (0,2)
                             _mm256_blend_ps(real_ret, real_ret1,
                                             0b11000000);  //  (1,2)
          imag_ret = (qregs[1] == 1)
                         ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10001000)
                         :  // (0,1)
                         (qregs[0] == 0)
                             ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10100000)
                             :  // (0,2)
                             _mm256_blend_ps(imag_ret, imag_ret1,
                                             0b11000000);  //  (1,2)
          break;
      }
    }
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[index], imags[index]);
  }
}

inline void _apply_matrix_float_avx_qL(RealVectorView<float>& reals,
                                       ImaginaryVectorView<float>& imags,
                                       const float* mat,
                                       const uint64_t* qregs,
                                       const size_t qregs_size,
                                       const uint64_t index0) {
  __m256i mask;
  __m256 real_ret, imag_ret, real_ret1, imag_ret1;
  __m256 vreals[1ULL << qregs_size], vimags[1ULL << qregs_size];
  __m256 tmp0, tmp1;

  const auto indexes_size = (1ULL << qregs_size);
  uint64_t indexes[indexes_size];
  fill_indices(index0, indexes, indexes_size, qregs, qregs_size);

  if (qregs[0] == 0) {
    mask = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
  } else if (qregs[0] == 1) {
    mask = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
  } else {  // if (q0 == 2) {
    mask = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
  }

  for (size_t i = 0; i < (1ULL << qregs_size); i += 2) {
    auto index = indexes[i];
    _mm_load_twoarray_complex(reals[index], imags[index], vreals[i], vimags[i]);

    vreals[i + 1] = _mm256_permutevar8x32_ps(vreals[i], mask);
    vimags[i + 1] = _mm256_permutevar8x32_ps(vimags[i], mask);
  }

  size_t mindex = 0;
  for (size_t i = 0; i < (1ULL << qregs_size); i += 2) {
    auto index = indexes[i];
    _mm_complex_inner_product<float>((1ULL << qregs_size), vreals, vimags,
                                     (&mat[mindex]), real_ret, imag_ret, tmp0,
                                     tmp1);
    mindex += (1ULL << (qregs_size + 1));

    _mm_complex_inner_product<float>((1ULL << qregs_size), vreals, vimags,
                                     (&mat[mindex]), real_ret1, imag_ret1, tmp0,
                                     tmp1);
    mindex += (1ULL << (qregs_size + 1));

    real_ret1 = _mm256_permutevar8x32_ps(real_ret1, mask);
    imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, mask);

    real_ret =
        (qregs[0] == 0) ? _mm256_blend_ps(real_ret, real_ret1, 0b10101010)
                        :  // (0,H,H)
            (qregs[0] == 1) ? _mm256_blend_ps(real_ret, real_ret1, 0b11001100)
                            :                                      // (1,H,H)
                _mm256_blend_ps(real_ret, real_ret1, 0b11110000);  //  (2,H,H)
    imag_ret =
        (qregs[0] == 0) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10101010)
                        :  // (0,H,H)
            (qregs[0] == 1) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b11001100)
                            :                                      // (1,H,H)
                _mm256_blend_ps(imag_ret, imag_ret1, 0b11110000);  //  (2,H,H)

    _mm_store_twoarray_complex(real_ret, imag_ret, reals[index], imags[index]);
  }
}

inline void _apply_matrix_float_avx(RealVectorView<float>& reals,
                                    ImaginaryVectorView<float>& imags,
                                    const float* mat,
                                    const uint64_t* qregs,
                                    const size_t qregs_size,
                                    const uint64_t index0) {
  __m256 real_ret, imag_ret;
  __m256 vreals[1ULL << qregs_size], vimags[1ULL << qregs_size];
  __m256 tmp0, tmp1;

  const auto indexes_size = (1ULL << qregs_size);
  uint64_t indexes[indexes_size];
  fill_indices(index0, indexes, indexes_size, qregs, qregs_size);

  for (size_t i = 0; i < (1ULL << qregs_size); ++i) {
    auto index = indexes[i];
    _mm_load_twoarray_complex(reals[index], imags[index], vreals[i], vimags[i]);
  }

  size_t mindex = 0;
  for (size_t i = 0; i < (1ULL << qregs_size); ++i) {
    auto index = indexes[i];
    _mm_complex_inner_product<float>((1ULL << qregs_size), vreals, vimags,
                                     (&mat[mindex]), real_ret, imag_ret, tmp0,
                                     tmp1);
    mindex += (1ULL << (qregs_size + 1));
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[index], imags[index]);
  }
}

inline void _apply_matrix_double_avx_q0q1(RealVectorView<double>& reals,
                                          ImaginaryVectorView<double>& imags,
                                          const double* mat,
                                          const uint64_t* qregs,
                                          const size_t qregs_size,
                                          const uint64_t index0) {
  const int PERM_D_Q0Q1_0 = 3 * 64 + 2 * 16 + 0 * 4 + 1 * 1;
  const int PERM_D_Q0Q1_1 = 3 * 64 + 0 * 16 + 1 * 4 + 2 * 1;
  const int PERM_D_Q0Q1_2 = 0 * 64 + 2 * 16 + 1 * 4 + 3 * 1;

  __m256d real_ret, imag_ret, real_ret1, imag_ret1;
  __m256d vreals[1ULL << qregs_size], vimags[1ULL << qregs_size];
  __m256d tmp0, tmp1;

  const auto indexes_size = (1ULL << qregs_size);
  uint64_t indexes[indexes_size];
  fill_indices(index0, indexes, indexes_size, qregs, qregs_size);

  for (size_t i = 0; i < (1ULL << qregs_size); i += 4) {
    auto index = indexes[i];
    _mm_load_twoarray_complex(reals[index], imags[index], vreals[i], vimags[i]);
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

  size_t mindex = 0;
  for (size_t i = 0; i < (1ULL << qregs_size); i += 4) {
    auto index = indexes[i];
    _mm_complex_inner_product<double>((1ULL << qregs_size), vreals, vimags,
                                      (&mat[mindex]), real_ret, imag_ret, tmp0,
                                      tmp1);
    mindex += (1ULL << (qregs_size + 1));
    for (size_t j = 1; j < 4; ++j) {
      _mm_complex_inner_product<double>((1ULL << qregs_size), vreals, vimags,
                                        (&mat[mindex]), real_ret1, imag_ret1,
                                        tmp0, tmp1);
      mindex += (1ULL << (qregs_size + 1));
      switch (j) {
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
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[index], imags[index]);
  }
}

inline void _apply_matrix_double_avx_qL(RealVectorView<double>& reals,
                                        ImaginaryVectorView<double>& imags,
                                        const double* mat,
                                        const uint64_t* qregs,
                                        const size_t qregs_size,
                                        const uint64_t index0) {
  const int PERM_D_Q0 = 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1;
  const int PERM_D_Q1 = 1 * 64 + 0 * 16 + 3 * 4 + 2 * 1;

  __m256d real_ret, imag_ret, real_ret1, imag_ret1;
  __m256d vreals[1ULL << qregs_size], vimags[1ULL << qregs_size];
  __m256d tmp0, tmp1;

  const auto indexes_size = (1ULL << qregs_size);
  uint64_t indexes[indexes_size];
  fill_indices(index0, indexes, indexes_size, qregs, qregs_size);

  for (size_t i = 0; i < (1ULL << qregs_size); i += 2) {
    auto index = indexes[i];
    _mm_load_twoarray_complex(reals[index], imags[index], vreals[i], vimags[i]);
    if (qregs[0] == 0) {
      vreals[i + 1] = _mm256_permute4x64_pd(vreals[i], PERM_D_Q0);
      vimags[i + 1] = _mm256_permute4x64_pd(vimags[i], PERM_D_Q0);
    } else {
      vreals[i + 1] = _mm256_permute4x64_pd(vreals[i], PERM_D_Q1);
      vimags[i + 1] = _mm256_permute4x64_pd(vimags[i], PERM_D_Q1);
    }
  }

  size_t mindex = 0;
  for (size_t i = 0; i < (1ULL << qregs_size); i += 2) {
    auto index = indexes[i];
    _mm_complex_inner_product<double>((1ULL << qregs_size), vreals, vimags,
                                      (&mat[mindex]), real_ret, imag_ret, tmp0,
                                      tmp1);
    mindex += (1ULL << (qregs_size + 1));

    _mm_complex_inner_product<double>((1ULL << qregs_size), vreals, vimags,
                                      (&mat[mindex]), real_ret1, imag_ret1,
                                      tmp0, tmp1);
    mindex += (1ULL << (qregs_size + 1));

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
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[index], imags[index]);
  }
}

inline void _apply_matrix_double_avx(RealVectorView<double>& reals,
                                     ImaginaryVectorView<double>& imags,
                                     const double* mat,
                                     const uint64_t* qregs,
                                     const size_t qregs_size,
                                     const uint64_t index0) {
  __m256d real_ret, imag_ret;
  __m256d vreals[1ULL << qregs_size], vimags[1ULL << qregs_size];
  __m256d tmp0, tmp1;

  const auto indexes_size = (1ULL << qregs_size);
  uint64_t indexes[indexes_size];
  fill_indices(index0, indexes, indexes_size, qregs, qregs_size);

  for (size_t i = 0; i < (1ULL << qregs_size); ++i) {
    auto index = indexes[i];
    _mm_load_twoarray_complex(reals[index], imags[index], vreals[i], vimags[i]);
  }

  size_t mindex = 0;
  for (size_t i = 0; i < (1ULL << qregs_size); ++i) {
    auto index = indexes[i];
    _mm_complex_inner_product<double>((1ULL << qregs_size), vreals, vimags,
                                      (&mat[mindex]), real_ret, imag_ret, tmp0,
                                      tmp1);
    mindex += (1ULL << (qregs_size + 1));
    _mm_store_twoarray_complex(real_ret, imag_ret, reals[index], imags[index]);
  }
}

inline Avx _apply_avx_kernel(const uint64_t* qregs,
                             const size_t qregs_size,
                             float* data,
                             const uint64_t data_size,
                             const float* mat,
                             const size_t omp_threads) {
  RealVectorView<float> real = {data};
  ImaginaryVectorView<float> img = {data};

  if (qregs_size > 2 && qregs[2] == 2) {
    auto lambda = [&](const uint64_t index0, const float* mat) -> void {
      _apply_matrix_float_avx_q0q1q2(real, img, mat, qregs, qregs_size, index0);
    };

    avx_apply_lambda(data_size, 1, lambda, qregs, qregs_size, omp_threads, mat);

  } else if (qregs_size > 1 && qregs[1] < 3) {
    auto lambda = [&](const uint64_t index0, const float* mat) -> void {
      _apply_matrix_float_avx_qLqL(real, img, mat, qregs, qregs_size, index0);
    };

    avx_apply_lambda(data_size, 2, lambda, qregs, qregs_size, omp_threads, mat);

  } else if (qregs[0] < 3) {
    auto lambda = [&](const uint64_t index0, const float* mat) -> void {
      _apply_matrix_float_avx_qL(real, img, mat, qregs, qregs_size, index0);
    };

    avx_apply_lambda(data_size, 4, lambda, qregs, qregs_size, omp_threads, mat);

  } else {
    auto lambda = [&](const uint64_t index0, const float* mat) -> void {
      _apply_matrix_float_avx(real, img, mat, qregs, qregs_size, index0);
    };

    avx_apply_lambda(data_size, 8, lambda, qregs, qregs_size, omp_threads, mat);
  }
  return Avx::Applied;
}

inline Avx _apply_avx_kernel(const uint64_t* qregs,
                             const size_t qregs_size,
                             double* data,
                             const size_t data_size,
                             const double* mat,
                             const size_t omp_threads) {
  RealVectorView<double> real = {data};
  ImaginaryVectorView<double> img = {data};

  if (qregs_size > 1 && qregs[1] == 1) {
    auto lambda = [&](const uint64_t index0, const double* mat) -> void {
      _apply_matrix_double_avx_q0q1(real, img, mat, qregs, qregs_size, index0);
    };

    avx_apply_lambda(data_size, 1, lambda, qregs, qregs_size, omp_threads, mat);

  } else if (qregs[0] < 2) {
    auto lambda = [&](const uint64_t index0, const double* mat) -> void {
      _apply_matrix_double_avx_qL(real, img, mat, qregs, qregs_size, index0);
    };

    avx_apply_lambda(data_size, 2, lambda, qregs, qregs_size, omp_threads, mat);

  } else {
    auto lambda = [&](const uint64_t index0, const double* mat) -> void {
      _apply_matrix_double_avx(real, img, mat, qregs, qregs_size, index0);
    };

    avx_apply_lambda(data_size, 4, lambda, qregs, qregs_size, omp_threads, mat);
  }
  return Avx::Applied;
}

template <typename FloatType>
typename std::enable_if<std::is_same<FloatType, double>::value, bool>::type
is_simd_applicable(const size_t qregs_size, const uint64_t data_size) {
  if (data_size <= 4 || qregs_size > 6)
    return false;
  return true;
}

template <typename FloatType>
typename std::enable_if<std::is_same<FloatType, float>::value, bool>::type
is_simd_applicable(const size_t qregs_size, const uint64_t data_size) {
  if (data_size <= 8 || qregs_size > 6)
    return false;
  return true;
}

template <typename FloatType>
Avx apply_matrix_avx(FloatType* qv_data,
                     const uint64_t data_size,
                     const uint64_t* qregs,
                     const size_t qregs_size,
                     const FloatType* mat,
                     const size_t omp_threads) {
  if (!is_simd_applicable<FloatType>(qregs_size, data_size))
    return Avx::NotApplied;

  auto transpose = [&](const FloatType* matrix, FloatType* transposed) {
    for (size_t i = 0; i < (1U << qregs_size); ++i) {
      for (size_t j = 0; j < (1U << qregs_size); ++j) {
        // This is for accessing the real part of the vector of complex numbers,
        // which complies with this inner format: [r,i,r,i,r,i,r,i,...]
        transposed[(i * (1U << qregs_size) + j) * 2] =
            matrix[(j * (1U << qregs_size) + i) * 2];
        //  And this is for the imaginary part.
        transposed[(i * (1U << qregs_size) + j) * 2 + 1] =
            matrix[(j * (1U << qregs_size) + i) * 2 + 1];
      }
    }
  };

  const auto matrix_size = (1 << (qregs_size * 2 + 1));
  FloatType transposed_mat[matrix_size];
  copy(&transposed_mat[0], mat, matrix_size);
  transpose(mat, &transposed_mat[0]);

  uint64_t ordered_qregs[qregs_size];
  copy(&ordered_qregs[0], qregs, qregs_size);

  reorder<FloatType>(qregs, qregs_size, ordered_qregs, transposed_mat);

  return _apply_avx_kernel(ordered_qregs, qregs_size, qv_data, data_size,
                           transposed_mat, omp_threads);
}

template Avx apply_matrix_avx<double>(double*,
                                      const uint64_t data_size,
                                      const uint64_t* qregs,
                                      const size_t qregs_size,
                                      const double* mat,
                                      const size_t omp_threads);
template Avx apply_matrix_avx<float>(float* data,
                                     const uint64_t data_size,
                                     const uint64_t* qregs,
                                     const size_t qregs_size,
                                     const float* mat,
                                     const size_t omp_threads);

} /* End namespace QV */
} /* End namespace AER */
