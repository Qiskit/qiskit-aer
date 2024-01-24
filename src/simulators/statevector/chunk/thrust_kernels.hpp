/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _qv_thrust_kernels_hpp_
#define _qv_thrust_kernels_hpp_

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#ifdef AER_THRUST_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef AER_THRUST_ROCM
#include <hip/hip_runtime.h>
#endif
DISABLE_WARNING_POP

#include "misc/wrap_thrust.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/utils.hpp"

#ifdef AER_THRUST_GPU
#include "simulators/statevector/chunk/cuda_kernels.hpp"
#endif

namespace AER {
namespace QV {
namespace Chunk {

//========================================
//  base class of gate functions
//========================================
template <typename data_t>
class GateFuncBase {
protected:
  thrust::complex<data_t> *data_;   // pointer to state vector buffer
  thrust::complex<double> *matrix_; // storage for matrix on device
  uint_t *params_;    // storage for additional parameters on device
  uint_t base_index_; // start index of state vector
  uint_t chunk_bits_;
  uint_t *cregs_;
  uint_t num_creg_bits_;
  int_t conditional_bit_;
#ifndef AER_THRUST_GPU
  uint_t index_offset_;
#endif
public:
  GateFuncBase() {
    data_ = NULL;
    matrix_ = NULL;
    params_ = NULL;
    base_index_ = 0;
    chunk_bits_ = 0;
    cregs_ = NULL;
    num_creg_bits_ = 0;
    conditional_bit_ = -1;
#ifndef AER_THRUST_GPU
    index_offset_ = 0;
#endif
  }
  virtual void set_data(thrust::complex<data_t> *p) { data_ = p; }
  void set_matrix(thrust::complex<double> *mat) { matrix_ = mat; }
  void set_params(uint_t *p) { params_ = p; }
  void set_chunk_bits(uint_t bits) { chunk_bits_ = bits; }

  void set_base_index(uint_t i) { base_index_ = i; }
  void set_cregs_(uint_t *cbits, uint_t nreg) {
    cregs_ = cbits;
    num_creg_bits_ = nreg;
  }
  void set_conditional(int_t bit) { conditional_bit_ = bit; }

#ifndef AER_THRUST_GPU
  void set_index_offset(uint_t i) { index_offset_ = i; }
#endif

  __host__ __device__ thrust::complex<data_t> *data(void) { return data_; }

  virtual bool is_diagonal(void) { return false; }
  virtual int qubits_count(void) { return 1; }
  virtual int num_control_bits(void) { return 0; }
  virtual int control_mask(void) { return 1; }
  virtual bool use_cache(void) { return false; }
  virtual bool batch_enable(void) { return true; }

  virtual const char *name(void) { return "base function"; }
  virtual uint_t size(int num_qubits) {
    if (is_diagonal()) {
      chunk_bits_ = num_qubits;
      return (1ull << num_qubits);
    } else {
      chunk_bits_ = num_qubits - (qubits_count() - num_control_bits());
      return (1ull << (num_qubits - (qubits_count() - num_control_bits())));
    }
  }

  virtual __host__ __device__ uint_t thread_to_index(uint_t _tid) const {
    return _tid;
  }
  virtual __host__ __device__ void
  run_with_cache(uint_t _tid, uint_t _idx,
                 thrust::complex<data_t> *_cache) const {
    // implemente this in the kernel class
  }
  virtual __host__ __device__ double
  run_with_cache_sum(uint_t _tid, uint_t _idx,
                     thrust::complex<data_t> *_cache) const {
    // implemente this in the kernel class
    return 0.0;
  }

  virtual __host__ __device__ bool check_conditional(uint_t i) const {
    if (conditional_bit_ < 0)
      return true;

    uint_t iChunk = i >> chunk_bits_;
    uint_t n64, i64, ibit;
    n64 = (num_creg_bits_ + 63) >> 6;
    i64 = conditional_bit_ >> 6;
    ibit = conditional_bit_ & 63;
    return (((cregs_[iChunk * n64 + i64] >> ibit) & 1) != 0);
  }
};

//========================================
//  gate functions with cache
//========================================
template <typename data_t>
class GateFuncWithCache : public GateFuncBase<data_t> {
protected:
  uint_t nqubits_;

public:
  GateFuncWithCache(uint_t nq) { nqubits_ = nq; }

  bool use_cache(void) { return true; }

  __host__ __device__ virtual uint_t thread_to_index(uint_t _tid) const {
    uint_t idx, ii, t, j;
    uint_t *qubits;
    uint_t *qubits_sorted;

    qubits_sorted = this->params_;
    qubits = qubits_sorted + nqubits_;

    idx = 0;
    ii = _tid >> nqubits_;
    for (j = 0; j < nqubits_; j++) {
      t = ii & ((1ull << qubits_sorted[j]) - 1);
      idx += t;
      ii = (ii - t) << 1;

      if (((_tid >> j) & 1) != 0) {
        idx += (1ull << qubits[j]);
      }
    }
    idx += ii;
    return idx;
  }

  __host__ __device__ void sync_threads() const {
#ifdef CUDA_ARCH
    __syncthreads();
#endif
  }

  __host__ __device__ void operator()(const uint_t &i) const {
    if (!this->check_conditional(i))
      return;

    thrust::complex<data_t> cache[1024];
    uint_t j, idx;
    uint_t matSize = 1ull << nqubits_;

    // load data to cache
    for (j = 0; j < matSize; j++) {
      idx = thread_to_index((i << nqubits_) + j);
      cache[j] = this->data_[idx];
    }

    // execute using cache
    for (j = 0; j < matSize; j++) {
      idx = thread_to_index((i << nqubits_) + j);
      this->run_with_cache(j, idx, cache);
    }
  }

  virtual int qubits_count(void) { return nqubits_; }
};

template <typename data_t>
class GateFuncSumWithCache : public GateFuncBase<data_t> {
protected:
  uint_t nqubits_;

public:
  GateFuncSumWithCache(uint_t nq) { nqubits_ = nq; }

  bool use_cache(void) { return true; }

  __host__ __device__ virtual uint_t thread_to_index(uint_t _tid) const {
    uint_t idx, ii, t, j;
    uint_t *qubits;
    uint_t *qubits_sorted;

    qubits_sorted = this->params_;
    qubits = qubits_sorted + nqubits_;

    idx = 0;
    ii = _tid >> nqubits_;
    for (j = 0; j < nqubits_; j++) {
      t = ii & ((1ull << qubits_sorted[j]) - 1);
      idx += t;
      ii = (ii - t) << 1;

      if (((_tid >> j) & 1) != 0) {
        idx += (1ull << qubits[j]);
      }
    }
    idx += ii;
    return idx;
  }

  __host__ __device__ double operator()(const uint_t &i) const {
    if (!this->check_conditional(i))
      return 0.0;

    thrust::complex<data_t> cache[1024];
    uint_t j, idx;
    uint_t matSize = 1ull << nqubits_;
    double sum = 0.0;

    // load data to cache
    for (j = 0; j < matSize; j++) {
      idx = thread_to_index((i << nqubits_) + j);
      cache[j] = this->data_[idx];
    }

    // execute using cache
    for (j = 0; j < matSize; j++) {
      idx = thread_to_index((i << nqubits_) + j);
      sum += this->run_with_cache_sum(j, idx, cache);
    }
    return sum;
  }

  virtual int qubits_count(void) { return nqubits_; }
};

// stridded iterator to access diagonal probabilities
template <typename Iterator>
class strided_range {
public:
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct stride_functor
      : public thrust::unary_function<difference_type, difference_type> {
    difference_type stride;

    stride_functor(difference_type _stride) : stride(_stride) {}

    __host__ __device__ difference_type
    operator()(const difference_type &i) const {
      if (stride == 1) // statevector
        return i;

      // density matrix
      difference_type i_chunk;
      i_chunk = i / (stride - 1);
      difference_type ret = stride * i - i_chunk * (stride - 1);
      return ret;
    }
  };

  typedef typename thrust::counting_iterator<difference_type> CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator>
      TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator, TransformIterator>
      PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_range(Iterator _first, Iterator _last, difference_type _stride)
      : first(_first), last(_last), stride(_stride) {}

  iterator begin(void) const {
    return PermutationIterator(
        first, TransformIterator(CountingIterator(0), stride_functor(stride)));
  }

  iterator end(void) const {
    if (stride == 1) // statevector
      return begin() + (last - first);

    // density matrix
    return begin() + (last - first) / (stride - 1);
  }

protected:
  Iterator first;
  Iterator last;
  difference_type stride;
};

template <typename data_t>
struct complex_dot_scan
    : public thrust::unary_function<thrust::complex<data_t>,
                                    thrust::complex<data_t>> {
  __host__ __device__ thrust::complex<data_t>
  operator()(thrust::complex<data_t> x) {
    return thrust::complex<data_t>(x.real() * x.real() + x.imag() * x.imag(),
                                   0);
  }
};

template <typename data_t>
struct complex_norm : public thrust::unary_function<thrust::complex<data_t>,
                                                    thrust::complex<data_t>> {
  __host__ __device__ thrust::complex<double>
  operator()(thrust::complex<data_t> x) {
    return thrust::complex<double>((double)x.real() * (double)x.real(),
                                   (double)x.imag() * (double)x.imag());
  }
};

template <typename data_t>
struct complex_less {
  typedef thrust::complex<data_t> first_argument_type;
  typedef thrust::complex<data_t> second_argument_type;
  typedef bool result_type;
  __thrust_exec_check_disable__ __host__ __device__ bool
  operator()(const thrust::complex<data_t> &lhs,
             const thrust::complex<data_t> &rhs) const {
    return lhs.real() < rhs.real();
  }
}; // end less

class HostFuncBase {
protected:
public:
  HostFuncBase() {}

  virtual void execute() {}
};

//------------------------------------------------------------------------------
// State initialize component
//------------------------------------------------------------------------------
template <typename data_t>
class initialize_component_1qubit_func : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> s0, s1;
  uint_t mask;
  uint_t offset;

public:
  initialize_component_1qubit_func(int qubit, thrust::complex<double> state0,
                                   thrust::complex<double> state1) {
    s0 = state0;
    s1 = state1;

    mask = (1ull << qubit) - 1;
    offset = 1ull << qubit;
  }

  virtual __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> *vec0;
    thrust::complex<data_t> *vec1;

    vec0 = this->data_;
    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec0[i0];

    vec0[i0] = s0 * q0;
    vec1[i0] = s1 * q0;
  }

  const char *name(void) { return "initialize_component 1 qubit"; }
};

template <typename data_t>
class initialize_component_func : public GateFuncBase<data_t> {
protected:
  uint_t nqubits;
  uint_t offset;
  uint_t mat_pos;
  uint_t mat_num;

public:
  initialize_component_func(const int nq, const uint_t pos, const uint_t num) {
    nqubits = nq;
    mat_pos = pos;
    mat_num = num;
  }

  int qubits_count(void) { return nqubits; }
  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    thrust::complex<double> q0;
    thrust::complex<double> q;
    thrust::complex<double> *state;
    uint_t *qubits;
    uint_t *qubits_sorted;
    uint_t j, k;
    uint_t ii, idx, t;
    uint_t mask;

    // get parameters from iterator
    vec = this->data_;
    state = this->matrix_;
    qubits = this->params_;
    qubits_sorted = qubits + nqubits;

    idx = 0;
    ii = i;
    for (j = 0; j < nqubits; j++) {
      mask = (1ull << qubits_sorted[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    q0 = vec[idx];
    for (k = mat_pos; k < mat_pos + mat_num; k++) {
      ii = idx;
      for (j = 0; j < nqubits; j++) {
        if (((k >> j) & 1) != 0)
          ii += (1ull << qubits[j]);
      }
      if (ii == idx) {
        if (mat_pos > 0)
          continue;
      }
      q = q0 * state[k - mat_pos];
      vec[ii] = q;
    }
  }

  const char *name(void) { return "initialize_component"; }
};

//------------------------------------------------------------------------------
// Zero clear
//------------------------------------------------------------------------------
template <typename data_t>
class ZeroClear : public GateFuncBase<data_t> {
protected:
public:
  ZeroClear() {}
  bool is_diagonal(void) { return true; }
  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    vec = this->data_;
    vec[i] = 0.0;
  }
  const char *name(void) { return "zero"; }
};

//------------------------------------------------------------------------------
// Initialize state
//------------------------------------------------------------------------------
template <typename data_t>
class initialize_kernel : public GateFuncBase<data_t> {
protected:
  int num_qubits_state_;
  uint_t offset_;
  thrust::complex<data_t> init_val_;

public:
  initialize_kernel(thrust::complex<data_t> v, int nqs, uint_t offset) {
    num_qubits_state_ = nqs;
    offset_ = offset;
    init_val_ = v;
  }

  bool is_diagonal(void) { return true; }

  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    uint_t iChunk = (i >> num_qubits_state_);

    vec = this->data_;

    if (i == iChunk * offset_) {
      vec[i] = init_val_;
    } else {
      vec[i] = 0.0;
    }
  }
  const char *name(void) { return "initialize"; }
};

//------------------------------------------------------------------------------
// Matrix multiplication
//------------------------------------------------------------------------------
template <typename data_t>
class MatrixMult2x2 : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> m0, m1, m2, m3;
  int qubit;
  uint_t mask;
  uint_t offset0;

public:
  MatrixMult2x2(const cvector_t<double> &mat, int q) {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];

    mask = (1ull << qubit) - 1;

    offset0 = 1ull << qubit;
  }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> q0, q1;
    thrust::complex<data_t> *vec0;
    thrust::complex<data_t> *vec1;

    vec0 = this->data_;
    vec1 = vec0 + offset0;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec0[i0];
    q1 = vec1[i0];

    vec0[i0] = m0 * q0 + m2 * q1;
    vec1[i0] = m1 * q0 + m3 * q1;
  }
  const char *name(void) { return "mult2x2"; }
};

template <typename data_t>
class MatrixMult4x4 : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> m00, m10, m20, m30;
  thrust::complex<double> m01, m11, m21, m31;
  thrust::complex<double> m02, m12, m22, m32;
  thrust::complex<double> m03, m13, m23, m33;
  uint_t mask0;
  uint_t mask1;
  uint_t offset0;
  uint_t offset1;

public:
  MatrixMult4x4(const cvector_t<double> &mat, int qubit0, int qubit1) {
    m00 = mat[0];
    m01 = mat[1];
    m02 = mat[2];
    m03 = mat[3];

    m10 = mat[4];
    m11 = mat[5];
    m12 = mat[6];
    m13 = mat[7];

    m20 = mat[8];
    m21 = mat[9];
    m22 = mat[10];
    m23 = mat[11];

    m30 = mat[12];
    m31 = mat[13];
    m32 = mat[14];
    m33 = mat[15];

    offset0 = 1ull << qubit0;
    offset1 = 1ull << qubit1;
    if (qubit0 < qubit1) {
      mask0 = offset0 - 1;
      mask1 = offset1 - 1;
    } else {
      mask0 = offset1 - 1;
      mask1 = offset0 - 1;
    }
  }

  int qubits_count(void) { return 2; }
  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1, i2;
    thrust::complex<data_t> *vec0;
    thrust::complex<data_t> *vec1;
    thrust::complex<data_t> *vec2;
    thrust::complex<data_t> *vec3;
    thrust::complex<data_t> q0, q1, q2, q3;

    vec0 = this->data_;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    vec1 = vec0 + offset0;
    vec2 = vec0 + offset1;
    vec3 = vec2 + offset0;

    q0 = vec0[i0];
    q1 = vec1[i0];
    q2 = vec2[i0];
    q3 = vec3[i0];

    vec0[i0] = m00 * q0 + m10 * q1 + m20 * q2 + m30 * q3;
    vec1[i0] = m01 * q0 + m11 * q1 + m21 * q2 + m31 * q3;
    vec2[i0] = m02 * q0 + m12 * q1 + m22 * q2 + m32 * q3;
    vec3[i0] = m03 * q0 + m13 * q1 + m23 * q2 + m33 * q3;
  }
  const char *name(void) { return "mult4x4"; }
};

template <typename data_t>
class MatrixMult8x8 : public GateFuncBase<data_t> {
protected:
  uint_t offset0;
  uint_t offset1;
  uint_t offset2;
  uint_t mask0;
  uint_t mask1;
  uint_t mask2;

public:
  MatrixMult8x8(const reg_t &qubit, const reg_t &qubit_ordered) {
    offset0 = (1ull << qubit[0]);
    offset1 = (1ull << qubit[1]);
    offset2 = (1ull << qubit[2]);

    mask0 = (1ull << qubit_ordered[0]) - 1;
    mask1 = (1ull << qubit_ordered[1]) - 1;
    mask2 = (1ull << qubit_ordered[2]) - 1;
  }

  int qubits_count(void) { return 3; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1, i2, i3;
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0, q1, q2, q3, q4, q5, q6, q7;
    thrust::complex<double> m0, m1, m2, m3, m4, m5, m6, m7;
    thrust::complex<double> *pMat;

    vec = this->data_;
    pMat = this->matrix_;

    i0 = i & mask0;
    i3 = (i - i0) << 1;
    i1 = i3 & mask1;
    i3 = (i3 - i1) << 1;
    i2 = i3 & mask2;
    i3 = (i3 - i2) << 1;

    i0 = i0 + i1 + i2 + i3;

    q0 = vec[i0];
    q1 = vec[i0 + offset0];
    q2 = vec[i0 + offset1];
    q3 = vec[i0 + offset1 + offset0];
    q4 = vec[i0 + offset2];
    q5 = vec[i0 + offset2 + offset0];
    q6 = vec[i0 + offset2 + offset1];
    q7 = vec[i0 + offset2 + offset1 + offset0];

    m0 = pMat[0];
    m1 = pMat[8];
    m2 = pMat[16];
    m3 = pMat[24];
    m4 = pMat[32];
    m5 = pMat[40];
    m6 = pMat[48];
    m7 = pMat[56];

    vec[i0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 + m5 * q5 +
              m6 * q6 + m7 * q7;

    m0 = pMat[1];
    m1 = pMat[9];
    m2 = pMat[17];
    m3 = pMat[25];
    m4 = pMat[33];
    m5 = pMat[41];
    m6 = pMat[49];
    m7 = pMat[57];

    vec[i0 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 +
                        m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[2];
    m1 = pMat[10];
    m2 = pMat[18];
    m3 = pMat[26];
    m4 = pMat[34];
    m5 = pMat[42];
    m6 = pMat[50];
    m7 = pMat[58];

    vec[i0 + offset1] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 +
                        m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[3];
    m1 = pMat[11];
    m2 = pMat[19];
    m3 = pMat[27];
    m4 = pMat[35];
    m5 = pMat[43];
    m6 = pMat[51];
    m7 = pMat[59];

    vec[i0 + offset1 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 +
                                  m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[4];
    m1 = pMat[12];
    m2 = pMat[20];
    m3 = pMat[28];
    m4 = pMat[36];
    m5 = pMat[44];
    m6 = pMat[52];
    m7 = pMat[60];

    vec[i0 + offset2] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 + m4 * q4 +
                        m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[5];
    m1 = pMat[13];
    m2 = pMat[21];
    m3 = pMat[29];
    m4 = pMat[37];
    m5 = pMat[45];
    m6 = pMat[53];
    m7 = pMat[61];

    vec[i0 + offset2 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 +
                                  m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[6];
    m1 = pMat[14];
    m2 = pMat[22];
    m3 = pMat[30];
    m4 = pMat[38];
    m5 = pMat[46];
    m6 = pMat[54];
    m7 = pMat[62];

    vec[i0 + offset2 + offset1] = m0 * q0 + m1 * q1 + m2 * q2 + m3 * q3 +
                                  m4 * q4 + m5 * q5 + m6 * q6 + m7 * q7;

    m0 = pMat[7];
    m1 = pMat[15];
    m2 = pMat[23];
    m3 = pMat[31];
    m4 = pMat[39];
    m5 = pMat[47];
    m6 = pMat[55];
    m7 = pMat[63];

    vec[i0 + offset2 + offset1 + offset0] = m0 * q0 + m1 * q1 + m2 * q2 +
                                            m3 * q3 + m4 * q4 + m5 * q5 +
                                            m6 * q6 + m7 * q7;
  }
  const char *name(void) { return "mult8x8"; }
};

template <typename data_t>
class MatrixMult16x16 : public GateFuncBase<data_t> {
protected:
  uint_t offset0;
  uint_t offset1;
  uint_t offset2;
  uint_t offset3;
  uint_t mask0;
  uint_t mask1;
  uint_t mask2;
  uint_t mask3;

public:
  MatrixMult16x16(const reg_t &qubit, const reg_t &qubit_ordered) {
    offset0 = (1ull << qubit[0]);
    offset1 = (1ull << qubit[1]);
    offset2 = (1ull << qubit[2]);
    offset3 = (1ull << qubit[3]);

    mask0 = (1ull << qubit_ordered[0]) - 1;
    mask1 = (1ull << qubit_ordered[1]) - 1;
    mask2 = (1ull << qubit_ordered[2]) - 1;
    mask3 = (1ull << qubit_ordered[3]) - 1;
  }

  int qubits_count(void) { return 4; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1, i2, i3, i4, offset;
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0, q1, q2, q3, q4, q5, q6, q7;
    thrust::complex<data_t> q8, q9, q10, q11, q12, q13, q14, q15;
    thrust::complex<double> r;
    thrust::complex<double> *pMat;
    int j;

    vec = this->data_;
    pMat = this->matrix_;

    i0 = i & mask0;
    i4 = (i - i0) << 1;
    i1 = i4 & mask1;
    i4 = (i4 - i1) << 1;
    i2 = i4 & mask2;
    i4 = (i4 - i2) << 1;
    i3 = i4 & mask3;
    i4 = (i4 - i3) << 1;

    i0 = i0 + i1 + i2 + i3 + i4;

    q0 = vec[i0];
    q1 = vec[i0 + offset0];
    q2 = vec[i0 + offset1];
    q3 = vec[i0 + offset1 + offset0];
    q4 = vec[i0 + offset2];
    q5 = vec[i0 + offset2 + offset0];
    q6 = vec[i0 + offset2 + offset1];
    q7 = vec[i0 + offset2 + offset1 + offset0];
    q8 = vec[i0 + offset3];
    q9 = vec[i0 + offset3 + offset0];
    q10 = vec[i0 + offset3 + offset1];
    q11 = vec[i0 + offset3 + offset1 + offset0];
    q12 = vec[i0 + offset3 + offset2];
    q13 = vec[i0 + offset3 + offset2 + offset0];
    q14 = vec[i0 + offset3 + offset2 + offset1];
    q15 = vec[i0 + offset3 + offset2 + offset1 + offset0];

    offset = 0;
    for (j = 0; j < 16; j++) {
      r = pMat[0 + j] * q0;
      r += pMat[16 + j] * q1;
      r += pMat[32 + j] * q2;
      r += pMat[48 + j] * q3;
      r += pMat[64 + j] * q4;
      r += pMat[80 + j] * q5;
      r += pMat[96 + j] * q6;
      r += pMat[112 + j] * q7;
      r += pMat[128 + j] * q8;
      r += pMat[144 + j] * q9;
      r += pMat[160 + j] * q10;
      r += pMat[176 + j] * q11;
      r += pMat[192 + j] * q12;
      r += pMat[208 + j] * q13;
      r += pMat[224 + j] * q14;
      r += pMat[240 + j] * q15;

      offset = offset3 * (((uint_t)j >> 3) & 1) +
               offset2 * (((uint_t)j >> 2) & 1) +
               offset1 * (((uint_t)j >> 1) & 1) + offset0 * ((uint_t)j & 1);

      vec[i0 + offset] = r;
    }
  }
  const char *name(void) { return "mult16x16"; }
};

template <typename data_t>
class MatrixMultNxN : public GateFuncWithCache<data_t> {
protected:
public:
  MatrixMultNxN(uint_t nq) : GateFuncWithCache<data_t>(nq) { ; }

  __host__ __device__ void
  run_with_cache(uint_t _tid, uint_t _idx,
                 thrust::complex<data_t> *_cache) const {
    uint_t j;
    thrust::complex<data_t> q, r;
    thrust::complex<double> m;
    uint_t mat_size, irow;
    thrust::complex<data_t> *vec;
    thrust::complex<double> *pMat;

    vec = this->data_;
    pMat = this->matrix_;

    mat_size = 1ull << this->nqubits_;
    irow = _tid & (mat_size - 1);

    r = 0.0;
    for (j = 0; j < mat_size; j++) {
      m = pMat[irow + mat_size * j];
      q = _cache[(_tid & 1023) - irow + j];

      r += m * q;
    }

    vec[_idx] = r;
  }

  const char *name(void) { return "multNxN"; }
};

// in-place NxN matrix multiplication using LU factorization
template <typename data_t>
class MatrixMultNxN_LU : public GateFuncBase<data_t> {
protected:
  uint_t nqubits;
  uint_t matSize;
  uint_t nswap;

public:
  MatrixMultNxN_LU(const cvector_t<double> &mat, const reg_t &qb,
                   cvector_t<double> &matLU, reg_t &params) {
    uint_t i, j, k, imax;
    std::complex<double> c0, c1;
    double d, dmax;
    uint_t *pSwap;

    nqubits = qb.size();
    matSize = 1ull << nqubits;

    matLU = mat;
    params.resize(nqubits + matSize * 2);

    for (k = 0; k < nqubits; k++) {
      params[k] = qb[k];
    }

    // LU factorization of input matrix
    for (i = 0; i < matSize; i++) {
      params[nqubits + i] = i; // init pivot
    }
    for (i = 0; i < matSize; i++) {
      imax = i;
      dmax = std::abs(matLU[(i << nqubits) + params[nqubits + i]]);
      for (j = i + 1; j < matSize; j++) {
        d = std::abs(matLU[(i << nqubits) + params[nqubits + j]]);
        if (d > dmax) {
          dmax = d;
          imax = j;
        }
      }
      if (imax != i) {
        j = params[nqubits + imax];
        params[nqubits + imax] = params[nqubits + i];
        params[nqubits + i] = j;
      }

      if (dmax > 0) {
        c0 = matLU[(i << nqubits) + params[nqubits + i]];

        for (j = i + 1; j < matSize; j++) {
          c1 = matLU[(i << nqubits) + params[nqubits + j]] / c0;

          for (k = i + 1; k < matSize; k++) {
            matLU[(k << nqubits) + params[nqubits + j]] -=
                c1 * matLU[(k << nqubits) + params[nqubits + i]];
          }
          matLU[(i << nqubits) + params[nqubits + j]] = c1;
        }
      }
    }

    // making table for swapping pivotted result
    pSwap = new uint_t[matSize];
    nswap = 0;
    for (i = 0; i < matSize; i++) {
      pSwap[i] = params[nqubits + i];
    }
    i = 0;
    while (i < matSize) {
      if (pSwap[i] != i) {
        params[nqubits + matSize + nswap++] = i;
        j = pSwap[i];
        params[nqubits + matSize + nswap++] = j;
        k = pSwap[j];
        pSwap[j] = j;
        while (i != k) {
          j = k;
          params[nqubits + matSize + nswap++] = k;
          k = pSwap[j];
          pSwap[j] = j;
        }
        pSwap[i] = i;
      }
      i++;
    }
    delete[] pSwap;
  }

  int qubits_count(void) { return nqubits; }

  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> q, qt;
    thrust::complex<double> m;
    thrust::complex<double> r;
    uint_t j, k, l, iq;
    uint_t ii, idx, t;
    uint_t mask, offset_j, offset_k;
    thrust::complex<data_t> *vec;
    thrust::complex<double> *pMat;
    uint_t *qubits;
    uint_t *pivot;
    uint_t *table;

    vec = this->data_;
    pMat = this->matrix_;
    qubits = this->params_;

    pivot = qubits + nqubits;
    table = pivot + matSize;

    idx = 0;
    ii = i;
    for (j = 0; j < nqubits; j++) {
      mask = (1ull << qubits[j]) - 1;

      t = ii & mask;
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    // mult U
    for (j = 0; j < matSize; j++) {
      r = 0.0;
      for (k = j; k < matSize; k++) {
        l = (pivot[j] + (k << nqubits));
        m = pMat[l];

        offset_k = 0;
        for (iq = 0; iq < nqubits; iq++) {
          if (((k >> iq) & 1) != 0)
            offset_k += (1ull << qubits[iq]);
        }
        q = vec[offset_k + idx];

        r += m * q;
      }
      offset_j = 0;
      for (iq = 0; iq < nqubits; iq++) {
        if (((j >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      vec[offset_j + idx] = r;
    }

    // mult L
    for (j = matSize - 1; j > 0; j--) {
      offset_j = 0;
      for (iq = 0; iq < nqubits; iq++) {
        if (((j >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      r = vec[offset_j + idx];

      for (k = 0; k < j; k++) {
        l = (pivot[j] + (k << nqubits));
        m = pMat[l];

        offset_k = 0;
        for (iq = 0; iq < nqubits; iq++) {
          if (((k >> iq) & 1) != 0)
            offset_k += (1ull << qubits[iq]);
        }
        q = vec[offset_k + idx];

        r += m * q;
      }
      offset_j = 0;
      for (iq = 0; iq < nqubits; iq++) {
        if (((j >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      vec[offset_j + idx] = r;
    }

    // swap results
    if (nswap > 0) {
      offset_j = 0;
      for (iq = 0; iq < nqubits; iq++) {
        if (((table[0] >> iq) & 1) != 0)
          offset_j += (1ull << qubits[iq]);
      }
      q = vec[offset_j + idx];
      k = pivot[table[0]];
      for (j = 1; j < nswap; j++) {
        offset_j = 0;
        for (iq = 0; iq < nqubits; iq++) {
          if (((table[j] >> iq) & 1) != 0)
            offset_j += (1ull << qubits[iq]);
        }
        qt = vec[offset_j + idx];

        offset_k = 0;
        for (iq = 0; iq < nqubits; iq++) {
          if (((k >> iq) & 1) != 0)
            offset_k += (1ull << qubits[iq]);
        }
        vec[offset_k + idx] = q;
        q = qt;
        k = pivot[table[j]];
      }
      offset_k = 0;
      for (iq = 0; iq < nqubits; iq++) {
        if (((k >> iq) & 1) != 0)
          offset_k += (1ull << qubits[iq]);
      }
      vec[offset_k + idx] = q;
    }
  }
  const char *name(void) { return "multNxN"; }
};

template <typename data_t>
class MatrixMult2x2Controlled : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> m0, m1, m2, m3;
  uint_t mask;
  uint_t cmask;
  uint_t offset;
  int nqubits;

public:
  MatrixMult2x2Controlled(const cvector_t<double> &mat, const reg_t &qubits) {
    int i;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];
    nqubits = qubits.size();

    offset = 1ull << qubits[nqubits - 1];
    mask = (1ull << qubits[nqubits - 1]) - 1;
    cmask = 0;
    for (i = 0; i < nqubits - 1; i++) {
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void) { return nqubits; }
  int num_control_bits(void) { return nqubits - 1; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> q0, q1;
    thrust::complex<data_t> *vec0;
    thrust::complex<data_t> *vec1;

    vec0 = this->data_;

    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if (((i0 + this->base_index_) & cmask) == cmask) {
      q0 = vec0[i0];
      q1 = vec1[i0];

      vec0[i0] = m0 * q0 + m2 * q1;
      vec1[i0] = m1 * q0 + m3 * q1;
    }
  }
  const char *name(void) { return "matrix_Cmult2x2"; }
};

template <typename data_t>
class BatchedMatrixMult2x2 : public GateFuncBase<data_t> {
protected:
  uint_t matrix_begin_;
  uint_t num_shots_per_matrix_;
  uint_t mask_;
  uint_t cmask_;
  uint_t offset_;
  uint_t nqubits_;

public:
  BatchedMatrixMult2x2(const reg_t &qubits, uint_t imat,
                       uint_t nshots_per_mat) {
    uint_t i;
    nqubits_ = qubits.size();

    offset_ = 1ull << qubits[nqubits_ - 1];
    mask_ = (1ull << qubits[nqubits_ - 1]) - 1;
    cmask_ = 0;
    for (i = 0; i < nqubits_ - 1; i++) {
      cmask_ |= (1ull << qubits[i]);
    }
    matrix_begin_ = imat;
    num_shots_per_matrix_ = nshots_per_mat;
  }

  int qubits_count(void) { return 1; }
  int num_control_bits(void) { return nqubits_ - 1; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> q0, q1;
    thrust::complex<data_t> *vec0;
    thrust::complex<data_t> *vec1;

    vec0 = this->data_;

    vec1 = vec0 + offset_;

    i1 = i & mask_;
    i0 = (i - i1) << 1;
    i0 += i1;

    if (((i0 + this->base_index_) & cmask_) == cmask_) {
      thrust::complex<double> m0, m1, m2, m3;
      q0 = vec0[i0];
      q1 = vec1[i0];

      uint_t iChunk = (this->base_index_ + i) >> this->chunk_bits_;
      // matrix offset from the top of buffer
      uint_t i_mat = (iChunk / num_shots_per_matrix_) - matrix_begin_;
      thrust::complex<double> *mat = this->matrix_ + i_mat * 4ull;

      m0 = mat[0];
      m1 = mat[1];
      m2 = mat[2];
      m3 = mat[3];

      vec0[i0] = m0 * q0 + m2 * q1;
      vec1[i0] = m1 * q0 + m3 * q1;
    }
  }
  const char *name(void) { return "BatchedMatrixMult2x2"; }
};

template <typename data_t>
class BatchedMatrixMultNxN : public GateFuncWithCache<data_t> {
protected:
  uint_t matrix_begin_;
  uint_t num_shots_per_matrix_;

public:
  BatchedMatrixMultNxN(uint_t nq, uint_t imat, uint_t nshots_per_mat)
      : GateFuncWithCache<data_t>(nq) {
    matrix_begin_ = imat;
    num_shots_per_matrix_ = nshots_per_mat;
  }

  __host__ __device__ void
  run_with_cache(uint_t _tid, uint_t _idx,
                 thrust::complex<data_t> *_cache) const {
    uint_t j;
    thrust::complex<data_t> q, r;
    thrust::complex<double> m;
    uint_t mat_size, irow;
    thrust::complex<data_t> *vec;
    thrust::complex<double> *pMat;

    uint_t iChunk = (this->base_index_ + _tid) >> this->chunk_bits_;
    // matrix offset from the top of buffer
    uint_t i_mat = (iChunk / num_shots_per_matrix_) - matrix_begin_;

    mat_size = 1ull << this->nqubits_;

    vec = this->data_;
    pMat = this->matrix_ + i_mat * mat_size * mat_size;

    irow = _tid & (mat_size - 1);

    r = 0.0;
    for (j = 0; j < mat_size; j++) {
      m = pMat[irow + mat_size * j];
      q = _cache[(_tid & 1023) - irow + j];

      r += m * q;
    }

    vec[_idx] = r;
  }

  const char *name(void) { return "BatchedMatrixMultNxN"; }
};

//------------------------------------------------------------------------------
// Diagonal matrix multiplication
//------------------------------------------------------------------------------
template <typename data_t>
class DiagonalMult2x2 : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> m0, m1;
  int qubit;

public:
  DiagonalMult2x2(const cvector_t<double> &mat, int q) {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
  }

  bool is_diagonal(void) { return true; }

  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> q;
    thrust::complex<data_t> *vec;
    thrust::complex<double> m;
    uint_t gid;

    vec = this->data_;
    gid = this->base_index_;

    q = vec[i];
    if ((((i + gid) >> qubit) & 1) == 0) {
      m = m0;
    } else {
      m = m1;
    }

    vec[i] = m * q;
  }
  const char *name(void) { return "diagonal_mult2x2"; }
};

template <typename data_t>
class DiagonalMult4x4 : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> m0, m1, m2, m3;
  int qubit0;
  int qubit1;

public:
  DiagonalMult4x4(const cvector_t<double> &mat, int q0, int q1) {
    qubit0 = q0;
    qubit1 = q1;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];
  }

  bool is_diagonal(void) { return true; }
  int qubits_count(void) { return 2; }

  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> q;
    thrust::complex<data_t> *vec;
    thrust::complex<double> m;
    uint_t gid;

    vec = this->data_;
    gid = this->base_index_;

    q = vec[i];
    if ((((i + gid) >> qubit1) & 1) == 0) {
      if ((((i + gid) >> qubit0) & 1) == 0) {
        m = m0;
      } else {
        m = m1;
      }
    } else {
      if ((((i + gid) >> qubit0) & 1) == 0) {
        m = m2;
      } else {
        m = m3;
      }
    }

    vec[i] = m * q;
  }
  const char *name(void) { return "diagonal_mult4x4"; }
};

template <typename data_t>
class DiagonalMultNxN : public GateFuncBase<data_t> {
protected:
  uint_t nqubits;

public:
  DiagonalMultNxN(const reg_t &qb) { nqubits = qb.size(); }

  bool is_diagonal(void) { return true; }
  int qubits_count(void) { return nqubits; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t j, im;
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q;
    thrust::complex<double> m;
    thrust::complex<double> *pMat;
    uint_t *qubits;
    uint_t gid;

    vec = this->data_;
    gid = this->base_index_;

    pMat = this->matrix_;
    qubits = this->params_;

    im = 0;
    for (j = 0; j < nqubits; j++) {
      if ((((i + gid) >> qubits[j]) & 1) != 0) {
        im += (1 << j);
      }
    }

    q = vec[i];
    m = pMat[im];

    vec[i] = m * q;
  }
  const char *name(void) { return "diagonal_multNxN"; }
};

template <typename data_t>
class DiagonalMult2x2Controlled : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> m0, m1;
  uint_t mask;
  uint_t cmask;
  int nqubits;

public:
  DiagonalMult2x2Controlled(const cvector_t<double> &mat, const reg_t &qubits) {
    int i;
    nqubits = qubits.size();

    m0 = mat[0];
    m1 = mat[1];

    mask = (1ull << qubits[nqubits - 1]);
    cmask = 0;
    for (i = 0; i < nqubits - 1; i++) {
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void) { return 1; }
  int num_control_bits(void) { return nqubits - 1; }

  bool is_diagonal(void) { return true; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t gid;
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;
    thrust::complex<double> m;

    vec = this->data_;
    gid = this->base_index_;

    if (((i + gid) & cmask) == cmask) {
      if ((i + gid) & mask) {
        m = m1;
      } else {
        m = m0;
      }

      q0 = vec[i];
      vec[i] = m * q0;
    }
  }
  const char *name(void) { return "diagonal_Cmult2x2"; }
};

template <typename data_t>
class BatchedDiagonalMatrixMult2x2 : public GateFuncBase<data_t> {
protected:
  uint_t matrix_begin_;
  uint_t num_shots_per_matrix_;
  uint_t mask_;
  uint_t cmask_;
  uint_t offset_;
  uint_t nqubits_;

public:
  BatchedDiagonalMatrixMult2x2(const reg_t &qubits, uint_t imat,
                               uint_t nshots_per_mat) {
    uint_t i;
    nqubits_ = qubits.size();

    mask_ = (1ull << qubits[nqubits_ - 1]);
    cmask_ = 0;
    for (i = 0; i < nqubits_ - 1; i++) {
      cmask_ |= (1ull << qubits[i]);
    }
    matrix_begin_ = imat;
    num_shots_per_matrix_ = nshots_per_mat;
  }

  int qubits_count(void) { return 1; }
  int num_control_bits(void) { return nqubits_ - 1; }
  bool is_diagonal(void) { return true; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t gid;
    thrust::complex<data_t> q0;
    thrust::complex<double> m;
    thrust::complex<data_t> *vec;

    vec = this->data_;
    gid = this->base_index_;

    if (((i + gid) & cmask_) == cmask_) {
      uint_t iChunk = (i + gid) >> this->chunk_bits_;
      // matrix offset from the top of buffer
      uint_t i_mat = (iChunk / num_shots_per_matrix_) - matrix_begin_;
      thrust::complex<double> *mat = this->matrix_ + i_mat * 2ull;

      q0 = vec[i];
      if ((i + gid) & mask_) {
        m = mat[1];
      } else {
        m = mat[0];
      }
      vec[i] = m * q0;
    }
  }
  const char *name(void) { return "BatchedDiagonalMatrixMult2x2"; }
};

template <typename data_t>
class BatchedDiagonalMatrixMultNxN : public GateFuncBase<data_t> {
protected:
  uint_t matrix_begin_;
  uint_t num_shots_per_matrix_;
  uint_t nqubits_;

public:
  BatchedDiagonalMatrixMultNxN(const uint_t nq, uint_t imat,
                               uint_t nshots_per_mat) {
    nqubits_ = nq;

    matrix_begin_ = imat;
    num_shots_per_matrix_ = nshots_per_mat;
  }

  int qubits_count(void) { return nqubits_; }
  int num_control_bits(void) { return 0; }
  bool is_diagonal(void) { return true; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t j, im;
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q;
    thrust::complex<double> m;
    uint_t *qubits;
    uint_t gid;

    gid = this->base_index_;

    uint_t iChunk = (i + gid) >> this->chunk_bits_;
    // matrix offset from the top of buffer
    uint_t i_mat = (iChunk / num_shots_per_matrix_) - matrix_begin_;
    thrust::complex<double> *mat = this->matrix_ + i_mat * 2ull;

    vec = this->data_;
    qubits = this->params_;

    q = vec[i];

    im = 0;
    for (j = 0; j < nqubits_; j++) {
      if ((((i + gid) >> qubits[j]) & 1) != 0) {
        im += (1 << j);
      }
    }
    m = mat[im];
    vec[i] = m * q;
  }

  const char *name(void) { return "BatchedDiagonalMatrixMultNxN"; }
};

//------------------------------------------------------------------------------
// Permutation
//------------------------------------------------------------------------------
template <typename data_t>
class Permutation : public GateFuncBase<data_t> {
protected:
  uint_t nqubits;
  uint_t npairs;

public:
  Permutation(const reg_t &qubits_sorted, const reg_t &qubits,
              const std::vector<std::pair<uint_t, uint_t>> &pairs,
              reg_t &params) {
    uint_t j, k;
    uint_t offset0, offset1;

    nqubits = qubits.size();
    npairs = pairs.size();

    params.resize(nqubits + npairs * 2);

    for (j = 0; j < nqubits; j++) { // save masks
      params[j] = (1ull << qubits_sorted[j]) - 1;
    }
    // make offset for pairs
    for (j = 0; j < npairs; j++) {
      offset0 = 0;
      offset1 = 0;
      for (k = 0; k < nqubits; k++) {
        if (((pairs[j].first >> k) & 1) != 0) {
          offset0 += (1ull << qubits[k]);
        }
        if (((pairs[j].second >> k) & 1) != 0) {
          offset1 += (1ull << qubits[k]);
        }
      }
      params[nqubits + j * 2] = offset0;
      params[nqubits + j * 2 + 1] = offset1;
    }
  }
  int qubits_count(void) { return nqubits; }

  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    uint_t j;
    uint_t ii, idx, t;
    uint_t *mask;
    uint_t *pairs;

    vec = this->data_;
    mask = this->params_;
    pairs = mask + nqubits;

    idx = 0;
    ii = i;
    for (j = 0; j < nqubits; j++) {
      t = ii & mask[j];
      idx += t;
      ii = (ii - t) << 1;
    }
    idx += ii;

    for (j = 0; j < npairs; j++) {
      q0 = vec[idx + pairs[j * 2]];
      q1 = vec[idx + pairs[j * 2 + 1]];

      vec[idx + pairs[j * 2]] = q1;
      vec[idx + pairs[j * 2 + 1]] = q0;
    }
  }
  const char *name(void) { return "Permutation"; }
};

//------------------------------------------------------------------------------
// X gate
//------------------------------------------------------------------------------
template <typename data_t>
class CX_func : public GateFuncBase<data_t> {
protected:
  uint_t offset;
  uint_t mask;
  uint_t cmask;
  int nqubits;
  int qubit_t;

public:
  CX_func(const reg_t &qubits) {
    int i;
    nqubits = qubits.size();

    qubit_t = qubits[nqubits - 1];
    offset = 1ull << qubit_t;
    mask = offset - 1;

    cmask = 0;
    for (i = 0; i < nqubits - 1; i++) {
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void) { return nqubits; }
  int num_control_bits(void) { return nqubits - 1; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> q0, q1;
    thrust::complex<data_t> *vec0;
    thrust::complex<data_t> *vec1;

    vec0 = this->data_;
    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if (((i0 + this->base_index_) & cmask) == cmask) {
      q0 = vec0[i0];
      q1 = vec1[i0];

      vec0[i0] = q1;
      vec1[i0] = q0;
    }
  }
  const char *name(void) { return "CX"; }
};

//------------------------------------------------------------------------------
// Y gate
//------------------------------------------------------------------------------
template <typename data_t>
class CY_func : public GateFuncBase<data_t> {
protected:
  uint_t mask;
  uint_t cmask;
  uint_t offset;
  int nqubits;
  int qubit_t;

public:
  CY_func(const reg_t &qubits) {
    int i;
    nqubits = qubits.size();

    qubit_t = qubits[nqubits - 1];
    offset = (1ull << qubit_t);
    mask = (1ull << qubit_t) - 1;

    cmask = 0;
    for (i = 0; i < nqubits - 1; i++) {
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void) { return nqubits; }
  int num_control_bits(void) { return nqubits - 1; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> q0, q1;
    thrust::complex<data_t> *vec0;
    thrust::complex<data_t> *vec1;

    vec0 = this->data_;

    vec1 = vec0 + offset;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    if (((i0 + this->base_index_) & cmask) == cmask) {
      q0 = vec0[i0];
      q1 = vec1[i0];

      vec0[i0] = thrust::complex<data_t>(q1.imag(), -q1.real());
      vec1[i0] = thrust::complex<data_t>(-q0.imag(), q0.real());
    }
  }
  const char *name(void) { return "CY"; }
};

//------------------------------------------------------------------------------
// Swap gate
//------------------------------------------------------------------------------
template <typename data_t>
class CSwap_func : public GateFuncBase<data_t> {
protected:
  uint_t mask0;
  uint_t mask1;
  uint_t cmask;
  int nqubits;
  int qubit_t0;
  int qubit_t1;
  uint_t offset1;
  uint_t offset2;

public:
  CSwap_func(const reg_t &qubits) {
    int i;
    nqubits = qubits.size();

    if (qubits[nqubits - 2] < qubits[nqubits - 1]) {
      qubit_t0 = qubits[nqubits - 2];
      qubit_t1 = qubits[nqubits - 1];
    } else {
      qubit_t1 = qubits[nqubits - 2];
      qubit_t0 = qubits[nqubits - 1];
    }
    mask0 = (1ull << qubit_t0) - 1;
    mask1 = (1ull << qubit_t1) - 1;

    offset1 = 1ull << qubit_t0;
    offset2 = 1ull << qubit_t1;

    cmask = 0;
    for (i = 0; i < nqubits - 2; i++) {
      cmask |= (1ull << qubits[i]);
    }
  }

  int qubits_count(void) { return nqubits; }
  int num_control_bits(void) { return nqubits - 2; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1, i2;
    thrust::complex<data_t> q1, q2;
    thrust::complex<data_t> *vec1;
    thrust::complex<data_t> *vec2;

    vec1 = this->data_;

    vec2 = vec1 + offset2;
    vec1 = vec1 + offset1;

    i0 = i & mask0;
    i2 = (i - i0) << 1;
    i1 = i2 & mask1;
    i2 = (i2 - i1) << 1;

    i0 = i0 + i1 + i2;

    if (((i0 + this->base_index_) & cmask) == cmask) {
      q1 = vec1[i0];
      q2 = vec2[i0];
      vec1[i0] = q2;
      vec2[i0] = q1;
    }
  }
  const char *name(void) { return "CSWAP"; }
};

template <typename data_t>
class MultiSwap_func : public GateFuncWithCache<data_t> {
protected:
public:
  MultiSwap_func(uint_t nq) : GateFuncWithCache<data_t>(nq) {}

  __host__ __device__ void
  run_with_cache(uint_t _tid, uint_t _idx,
                 thrust::complex<data_t> *_cache) const {
    thrust::complex<data_t> *vec;
    uint_t pos = _tid & 1023;
    uint_t j;

    vec = this->data_;

    for (j = 0; j < this->nqubits_; j += 2) {
      if ((((pos >> j) & 1) ^ ((pos >> (j + 1)) & 1)) != 0) {
        pos ^= ((1ull << j) | (1ull << (j + 1)));
      }
    }
    vec[_idx] = _cache[pos];
  }
  const char *name(void) { return "MultiSWAP"; }
};

// swap operator between chunks
template <typename data_t>
class CSwapChunk_func : public GateFuncBase<data_t> {
protected:
  uint_t mask;
  thrust::complex<data_t> *vec0;
  thrust::complex<data_t> *vec1;
  bool write_back_;
  bool swap_all_;

public:
  CSwapChunk_func(const reg_t &qubits, uint_t block_bits,
                  thrust::complex<data_t> *pVec0,
                  thrust::complex<data_t> *pVec1, bool wb) {
    uint_t nqubits;
    uint_t qubit_t;
    nqubits = qubits.size();

    if (qubits[nqubits - 2] < qubits[nqubits - 1]) {
      qubit_t = qubits[nqubits - 2];
    } else {
      qubit_t = qubits[nqubits - 1];
    }
    mask = (1ull << qubit_t) - 1;

    vec0 = pVec0;
    vec1 = pVec1;

    write_back_ = wb;
    if (qubit_t >= block_bits)
      swap_all_ = true;
    else
      swap_all_ = false;
  }

  bool batch_enable(void) { return false; }
  bool is_diagonal(void) { return swap_all_; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> q0, q1;

    i0 = i & mask;
    i1 = (i - i0) << 1;
    i0 += i1;

    q0 = vec0[i0];
    q1 = vec1[i0];
    vec0[i0] = q1;
    if (write_back_)
      vec1[i0] = q0;
  }
  const char *name(void) { return "Chunk SWAP"; }
};

// buffer swap
template <typename data_t>
class BufferSwap_func : public GateFuncBase<data_t> {
protected:
  uint_t mask;
  thrust::complex<data_t> *vec0_;
  thrust::complex<data_t> *vec1_;
  uint_t size_;
  bool write_back_;

public:
  BufferSwap_func(thrust::complex<data_t> *pVec0,
                  thrust::complex<data_t> *pVec1, uint_t size, bool wb) {
    vec0_ = pVec0;
    vec1_ = pVec1;
    size_ = size;
    write_back_ = wb;
  }

  bool is_diagonal(void) { return true; }

  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> q0, q1;

    if (i < size_) {
      q1 = vec1_[i];
      if (write_back_) {
        q0 = vec0_[i];
        vec1_[i] = q0;
      }
      vec0_[i] = q1;
    }
  }
  const char *name(void) { return "buffer swap"; }
};

//------------------------------------------------------------------------------
// Phase gate
//------------------------------------------------------------------------------
template <typename data_t>
class phase_func : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> phase;
  uint_t mask;
  int nqubits;

public:
  phase_func(const reg_t &qubits, thrust::complex<double> p) {
    int i;
    nqubits = qubits.size();
    phase = p;

    mask = 0;
    for (i = 0; i < nqubits; i++) {
      mask |= (1ull << qubits[i]);
    }
  }
  bool is_diagonal(void) { return true; }

  __host__ __device__ void operator()(const uint_t &i) const {
    uint_t gid;
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;

    vec = this->data_;
    gid = this->base_index_;

    if (((i + gid) & mask) == mask) {
      q0 = vec[i];
      vec[i] = q0 * phase;
    }
  }
  const char *name(void) { return "phase"; }
};

//------------------------------------------------------------------------------
// Norm functions
//------------------------------------------------------------------------------
template <typename data_t>
class norm_func : public GateFuncBase<data_t> {
protected:
public:
  norm_func(void) {}
  bool is_diagonal(void) { return true; }
  bool batch_enable(void) { return true; }

  __host__ __device__ double operator()(const uint_t &i) const {
    thrust::complex<data_t> q;
    thrust::complex<data_t> *vec;
    double d;

    vec = this->data_;
    q = vec[i];
    d = (double)(q.real() * q.real() + q.imag() * q.imag());
    return d;
  }

  const char *name(void) { return "norm"; }
};

template <typename data_t>
class trace_func : public GateFuncBase<data_t> {
protected:
  uint_t rows_;

public:
  trace_func(uint_t nrow) { rows_ = nrow; }
  bool is_diagonal(void) { return true; }
  uint_t size(int num_qubits) {
    this->chunk_bits_ = num_qubits;
    return rows_;
  }

  __host__ __device__ double operator()(const uint_t &i) const {
    thrust::complex<data_t> q;
    thrust::complex<data_t> *vec;

    uint_t iChunk = (i / rows_);
    uint_t lid = i - (iChunk * rows_);
    uint_t idx = (iChunk << this->chunk_bits_) + lid * (rows_ + 1);

    vec = this->data_;
    q = vec[idx];
    return q.real();
  }

  const char *name(void) { return "trace"; }
};

template <typename data_t>
class NormMatrixMultNxN : public GateFuncSumWithCache<data_t> {
protected:
public:
  NormMatrixMultNxN(uint_t nq) : GateFuncSumWithCache<data_t>(nq) { ; }

  __host__ __device__ double
  run_with_cache_sum(uint_t _tid, uint_t _idx,
                     thrust::complex<data_t> *_cache) const {
    uint_t j;
    thrust::complex<data_t> q, r;
    thrust::complex<double> m;
    uint_t mat_size, irow;
    thrust::complex<double> *pMat;

    pMat = this->matrix_;

    mat_size = 1ull << this->nqubits_;
    irow = _tid & (mat_size - 1);

    r = 0.0;
    for (j = 0; j < mat_size; j++) {
      m = pMat[irow + mat_size * j];
      q = _cache[_tid - irow + j];

      r += m * q;
    }

    return (r.real() * r.real() + r.imag() * r.imag());
  }

  const char *name(void) { return "NormmultNxN"; }
};

template <typename data_t>
class NormDiagonalMultNxN : public GateFuncBase<data_t> {
protected:
  int nqubits;

public:
  NormDiagonalMultNxN(const reg_t &qb) { nqubits = qb.size(); }

  bool is_diagonal(void) { return true; }
  int qubits_count(void) { return nqubits; }

  __host__ __device__ double operator()(const uint_t &i) const {
    uint_t im, j, gid;
    thrust::complex<data_t> q;
    thrust::complex<double> m, r;
    thrust::complex<double> *pMat;
    thrust::complex<data_t> *vec;
    uint_t *qubits;

    vec = this->data_;
    pMat = this->matrix_;
    qubits = this->params_;
    gid = this->base_index_;

    im = 0;
    for (j = 0; j < nqubits; j++) {
      if (((i + gid) & (1ull << qubits[j])) != 0) {
        im += (1 << j);
      }
    }

    q = vec[i];
    m = pMat[im];

    r = m * q;
    return (r.real() * r.real() + r.imag() * r.imag());
  }
  const char *name(void) { return "Norm_diagonal_multNxN"; }
};

template <typename data_t>
class NormMatrixMult2x2 : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> m0, m1, m2, m3;
  int qubit;
  uint_t mask;
  uint_t offset;

public:
  NormMatrixMult2x2(const cvector_t<double> &mat, int q) {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
    m2 = mat[2];
    m3 = mat[3];

    offset = 1ull << qubit;
    mask = (1ull << qubit) - 1;
  }

  __host__ __device__ double operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0, q1;
    thrust::complex<double> r0, r1;
    double sum = 0.0;

    vec = this->data_;

    i1 = i & mask;
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec[i0];
    q1 = vec[offset + i0];

    r0 = m0 * q0 + m2 * q1;
    sum += r0.real() * r0.real() + r0.imag() * r0.imag();
    r1 = m1 * q0 + m3 * q1;
    sum += r1.real() * r1.real() + r1.imag() * r1.imag();
    return sum;
  }
  const char *name(void) { return "Norm_mult2x2"; }
};

template <typename data_t>
class NormDiagonalMult2x2 : public GateFuncBase<data_t> {
protected:
  thrust::complex<double> m0, m1;
  int qubit;

public:
  NormDiagonalMult2x2(cvector_t<double> &mat, int q) {
    qubit = q;
    m0 = mat[0];
    m1 = mat[1];
  }

  bool is_diagonal(void) { return true; }

  __host__ __device__ double operator()(const uint_t &i) const {
    uint_t gid;
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q;
    thrust::complex<double> m, r;

    vec = this->data_;
    gid = this->base_index_;

    q = vec[i];
    if ((((i + gid) >> qubit) & 1) == 0) {
      m = m0;
    } else {
      m = m1;
    }

    r = m * q;

    return (r.real() * r.real() + r.imag() * r.imag());
  }
  const char *name(void) { return "Norm_diagonal_mult2x2"; }
};

//------------------------------------------------------------------------------
// Probabilities
//------------------------------------------------------------------------------
template <typename data_t>
class probability_func : public GateFuncBase<data_t> {
protected:
  uint_t mask;
  uint_t cmask;

public:
  probability_func(const reg_t &qubits, int i) {
    int k;
    int nq = qubits.size();

    mask = 0;
    cmask = 0;
    for (k = 0; k < nq; k++) {
      mask |= (1ull << qubits[k]);

      if (((i >> k) & 1) != 0) {
        cmask |= (1ull << qubits[k]);
      }
    }
  }

  bool is_diagonal(void) { return true; }

  __host__ __device__ double operator()(const uint_t &i) const {
    thrust::complex<data_t> q;
    thrust::complex<data_t> *vec;
    double ret;

    vec = this->data_;

    ret = 0.0;

    if ((i & mask) == cmask) {
      q = vec[i];
      ret = q.real() * q.real() + q.imag() * q.imag();
    }
    return ret;
  }

  const char *name(void) { return "probabilities"; }
};

template <typename data_t>
class probability_1qubit_func : public GateFuncBase<data_t> {
protected:
  uint_t offset;

public:
  probability_1qubit_func(const uint_t qubit) { offset = 1ull << qubit; }

  __host__ __device__ thrust::complex<double>
  operator()(const uint_t &i) const {
    uint_t i0, i1;
    thrust::complex<data_t> q0, q1;
    thrust::complex<data_t> *vec0;
    thrust::complex<data_t> *vec1;
    thrust::complex<double> ret;
    double d0, d1;

    vec0 = this->data_;
    vec1 = vec0 + offset;

    i1 = i & (offset - 1);
    i0 = (i - i1) << 1;
    i0 += i1;

    q0 = vec0[i0];
    q1 = vec1[i0];

    d0 = (double)(q0.real() * q0.real() + q0.imag() * q0.imag());
    d1 = (double)(q1.real() * q1.real() + q1.imag() * q1.imag());

    ret = thrust::complex<double>(d0, d1);
    return ret;
  }

  const char *name(void) { return "probabilities_1qubit"; }
};

//------------------------------------------------------------------------------
// Expectation values
//------------------------------------------------------------------------------
inline __host__ __device__ uint_t pop_count_kernel(uint_t val) {
  uint_t count = val;
  count = (count & 0x5555555555555555) + ((count >> 1) & 0x5555555555555555);
  count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
  count = (count & 0x0f0f0f0f0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f0f0f0f0f);
  count = (count & 0x00ff00ff00ff00ff) + ((count >> 8) & 0x00ff00ff00ff00ff);
  count = (count & 0x0000ffff0000ffff) + ((count >> 16) & 0x0000ffff0000ffff);
  count = (count & 0x00000000ffffffff) + ((count >> 32) & 0x00000000ffffffff);
  return count;
}

// special case Z only
template <typename data_t>
class expval_pauli_Z_func : public GateFuncBase<data_t> {
protected:
  uint_t z_mask_;

public:
  expval_pauli_Z_func(uint_t z) { z_mask_ = z; }

  bool is_diagonal(void) { return true; }
  bool batch_enable(void) { return true; }

  __host__ __device__ double operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;
    double ret = 0.0;

    vec = this->data_;

    q0 = vec[i];
    ret = q0.real() * q0.real() + q0.imag() * q0.imag();

    if (z_mask_ != 0) {
      if (pop_count_kernel(i & z_mask_) & 1)
        ret = -ret;
    }

    return ret;
  }
  const char *name(void) { return "expval_pauli_Z"; }
};

template <typename data_t>
class expval_pauli_XYZ_func : public GateFuncBase<data_t> {
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  uint_t mask_l_;
  uint_t mask_u_;
  thrust::complex<data_t> phase_;

public:
  expval_pauli_XYZ_func(uint_t x, uint_t z, uint_t x_max,
                        std::complex<data_t> p) {
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;

    mask_u_ = ~((1ull << (x_max + 1)) - 1);
    mask_l_ = (1ull << x_max) - 1;
  }
  bool batch_enable(void) { return true; }

  __host__ __device__ double operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    thrust::complex<data_t> q0p;
    thrust::complex<data_t> q1p;
    double d0, d1, ret = 0.0;
    uint_t idx0, idx1;

    vec = this->data_;

    idx0 = ((i << 1) & mask_u_) | (i & mask_l_);
    idx1 = idx0 ^ x_mask_;

    q0 = vec[idx0];
    q1 = vec[idx1];
    q0p = q1 * phase_;
    q1p = q0 * phase_;
    d0 = q0.real() * q0p.real() + q0.imag() * q0p.imag();
    d1 = q1.real() * q1p.real() + q1.imag() * q1p.imag();

    if (z_mask_ != 0) {
      if (pop_count_kernel(idx0 & z_mask_) & 1)
        ret = -d0;
      else
        ret = d0;
      if (pop_count_kernel(idx1 & z_mask_) & 1)
        ret -= d1;
      else
        ret += d1;
    } else {
      ret = d0 + d1;
    }

    return ret;
  }
  const char *name(void) { return "expval_pauli_XYZ"; }
};

template <typename data_t>
class expval_pauli_inter_chunk_func : public GateFuncBase<data_t> {
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  thrust::complex<data_t> phase_;
  thrust::complex<data_t> *pair_chunk_;
  uint_t z_count_;
  uint_t z_count_pair_;

public:
  expval_pauli_inter_chunk_func(uint_t x, uint_t z, std::complex<data_t> p,
                                thrust::complex<data_t> *pair_chunk, uint_t zc,
                                uint_t zcp) {
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;

    pair_chunk_ = pair_chunk;
    z_count_ = zc;
    z_count_pair_ = zcp;
  }

  bool is_diagonal(void) { return true; }
  bool batch_enable(void) { return false; }

  __host__ __device__ double operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    thrust::complex<data_t> q0p;
    thrust::complex<data_t> q1p;
    double d0, d1, ret = 0.0;
    uint_t ip;

    vec = this->data_;

    ip = i ^ x_mask_;
    q0 = vec[i];
    q1 = pair_chunk_[ip];
    q0p = q1 * phase_;
    q1p = q0 * phase_;
    d0 = q0.real() * q0p.real() + q0.imag() * q0p.imag();
    d1 = q1.real() * q1p.real() + q1.imag() * q1p.imag();

    if ((pop_count_kernel(i & z_mask_) + z_count_) & 1)
      ret = -d0;
    else
      ret = d0;
    if ((pop_count_kernel(ip & z_mask_) + z_count_pair_) & 1)
      ret -= d1;
    else
      ret += d1;

    return ret;
  }
  const char *name(void) { return "expval_pauli_inter_chunk"; }
};

template <typename data_t>
class batched_expval_I_func : public GateFuncBase<data_t> {
protected:
  bool variance_;
  double param_;
  double param_var_;

public:
  batched_expval_I_func(bool var, thrust::complex<double> par) {
    variance_ = var;
    param_ = par.real();
    param_var_ = par.imag();
  }
  bool is_diagonal(void) { return true; }
  bool batch_enable(void) { return true; }

  __host__ __device__ thrust::complex<double>
  operator()(const uint_t &i) const {
    thrust::complex<data_t> q;
    thrust::complex<data_t> *vec;
    double d, dv = 0.0;

    vec = this->data_;
    q = vec[i];
    d = (double)(q.real() * q.real() + q.imag() * q.imag());

    if (variance_)
      dv = d * param_var_;
    d *= param_;
    return thrust::complex<double>(d, dv);
  }
  const char *name(void) { return "batched_expval_I_func"; }
};

template <typename data_t>
class batched_expval_pauli_Z_func : public GateFuncBase<data_t> {
protected:
  uint_t z_mask_;
  bool variance_;
  double param_;
  double param_var_;

public:
  batched_expval_pauli_Z_func(bool var, thrust::complex<double> par, uint_t z) {
    variance_ = var;
    param_ = par.real();
    param_var_ = par.imag();
    z_mask_ = z;
  }

  bool is_diagonal(void) { return true; }
  bool batch_enable(void) { return true; }

  __host__ __device__ thrust::complex<double>
  operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;
    double d, dv = 0.0;

    vec = this->data_;

    q0 = vec[i];
    d = q0.real() * q0.real() + q0.imag() * q0.imag();

    if (z_mask_ != 0) {
      if (pop_count_kernel(i & z_mask_) & 1)
        d = -d;
    }

    if (variance_)
      dv = d * param_var_;
    d *= param_;
    return thrust::complex<double>(d, dv);
  }
  const char *name(void) { return "batched_expval_pauli_Z_func"; }
};

template <typename data_t>
class batched_expval_pauli_XYZ_func : public GateFuncBase<data_t> {
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  uint_t mask_l_;
  uint_t mask_u_;
  thrust::complex<data_t> phase_;
  bool variance_;
  double param_;
  double param_var_;

public:
  batched_expval_pauli_XYZ_func(bool var, thrust::complex<double> par, uint_t x,
                                uint_t z, uint_t x_max,
                                std::complex<data_t> p) {
    variance_ = var;
    param_ = par.real();
    param_var_ = par.imag();

    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;

    mask_u_ = ~((1ull << (x_max + 1)) - 1);
    mask_l_ = (1ull << x_max) - 1;
  }
  bool batch_enable(void) { return true; }

  __host__ __device__ thrust::complex<double>
  operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    thrust::complex<data_t> q0p;
    thrust::complex<data_t> q1p;
    double d0, d1, ret, ret_v = 0.0;
    uint_t idx0, idx1;

    vec = this->data_;

    idx0 = ((i << 1) & mask_u_) | (i & mask_l_);
    idx1 = idx0 ^ x_mask_;

    q0 = vec[idx0];
    q1 = vec[idx1];
    q0p = q1 * phase_;
    q1p = q0 * phase_;
    d0 = q0.real() * q0p.real() + q0.imag() * q0p.imag();
    d1 = q1.real() * q1p.real() + q1.imag() * q1p.imag();

    if (z_mask_ != 0) {
      if (pop_count_kernel(idx0 & z_mask_) & 1)
        ret = -d0;
      else
        ret = d0;
      if (pop_count_kernel(idx1 & z_mask_) & 1)
        ret -= d1;
      else
        ret += d1;
    } else {
      ret = d0 + d1;
    }

    if (variance_)
      ret_v = ret * param_var_;
    ret *= param_;
    return thrust::complex<double>(ret, ret_v);
  }
  const char *name(void) { return "batched_expval_pauli_XYZ_func"; }
};

//------------------------------------------------------------------------------
// Pauli application
//------------------------------------------------------------------------------
template <typename data_t>
class multi_pauli_func : public GateFuncBase<data_t> {
protected:
  uint_t x_mask_;
  uint_t z_mask_;
  uint_t mask_l_;
  uint_t mask_u_;
  thrust::complex<data_t> phase_;
  uint_t nqubits_;

public:
  multi_pauli_func(uint_t x, uint_t z, uint_t x_max, std::complex<data_t> p) {
    x_mask_ = x;
    z_mask_ = z;
    phase_ = p;

    mask_u_ = ~((1ull << (x_max + 1)) - 1);
    mask_l_ = (1ull << x_max) - 1;
  }

  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;
    thrust::complex<data_t> q1;
    uint_t idx0, idx1;

    vec = this->data_;

    idx0 = ((i << 1) & mask_u_) | (i & mask_l_);
    idx1 = idx0 ^ x_mask_;

    q0 = vec[idx0];
    q1 = vec[idx1];

    if (z_mask_ != 0) {
      if (pop_count_kernel(idx0 & z_mask_) & 1)
        q0 *= -1;

      if (pop_count_kernel(idx1 & z_mask_) & 1)
        q1 *= -1;
    }
    vec[idx0] = q1 * phase_;
    vec[idx1] = q0 * phase_;
  }
  const char *name(void) { return "multi_pauli"; }
};

// special case Z only
template <typename data_t>
class multi_pauli_Z_func : public GateFuncBase<data_t> {
protected:
  uint_t z_mask_;
  thrust::complex<data_t> phase_;

public:
  multi_pauli_Z_func(uint_t z, std::complex<data_t> p) {
    z_mask_ = z;
    phase_ = p;
  }

  bool is_diagonal(void) { return true; }

  __host__ __device__ void operator()(const uint_t &i) const {
    thrust::complex<data_t> *vec;
    thrust::complex<data_t> q0;

    vec = this->data_;

    q0 = vec[i];

    if (z_mask_ != 0) {
      if (pop_count_kernel(i & z_mask_) & 1)
        q0 = -q0;
    }
    vec[i] = q0 * phase_;
  }
  const char *name(void) { return "multi_pauli_Z"; }
};

//------------------------------------------------------------------------------
} // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
