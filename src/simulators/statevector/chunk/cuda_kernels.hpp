/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020. 2021.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _qv_cuda_kernels_hpp_
#define _qv_cuda_kernels_hpp_

#include "misc/gpu_static_properties.hpp"

namespace AER {
namespace QV {
namespace Chunk {

template <typename data_t, typename kernel_t>
__global__ void dev_apply_function(kernel_t func, uint_t count) {
  uint_t i;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    if (func.check_conditional(i))
      func(i);
  }
}

template <typename data_t, typename kernel_t>
__global__ void dev_apply_function_with_cache(kernel_t func, uint_t count) {
  // One cache entry per thread.
  __shared__ thrust::complex<data_t> cache[_MAX_THD];
  uint_t i, idx;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;

  if (!func.check_conditional(i))
    return;

  idx = func.thread_to_index(i);

  cache[threadIdx.x] = func.data()[idx];
  __syncthreads();

  func.run_with_cache(i, idx, cache);
}

template <typename data_t, typename kernel_t>
__global__ void dev_apply_function_sum(double *pReduceBuffer, kernel_t func,
                                       uint_t buf_size, uint_t count) {
  // One cache entry per warp/wavefront
  __shared__ double cache[_MAX_THD / _WS];
  double sum;
  uint_t i, j, iChunk, nw;

  iChunk = blockIdx.y + blockIdx.z * gridDim.y;
  i = threadIdx.x + blockIdx.x * blockDim.x + iChunk * gridDim.x * blockDim.x;
  if (i >= count)
    return;

  if (!func.check_conditional(i))
    return;

  sum = func(i);

  // reduce in warp
  nw = min(blockDim.x, _WS);
  for (j = 1; j < nw; j *= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, j, 32);
  }

  if (blockDim.x > _WS) {
    // reduce in thread block
    if ((threadIdx.x & (_WS - 1)) == 0) {
      cache[(threadIdx.x / _WS)] = sum;
    }
    __syncthreads();
    if (threadIdx.x < _WS) {
      if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS))
        sum = cache[threadIdx.x];
      else
        sum = 0.0;

      // reduce in warp
      nw = _WS;
      for (j = 1; j < nw; j *= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, j, 32);
      }
    }
  }
  if (threadIdx.x == 0) {
    pReduceBuffer[blockIdx.x + buf_size * iChunk] = sum;
  }
}

template <typename data_t, typename kernel_t>
__global__ void
dev_apply_function_sum_with_cache(double *pReduceBuffer, kernel_t func,
                                  uint_t buf_size, uint_t count) {
  // One cache entry per thread.
  __shared__ thrust::complex<data_t> cache[_MAX_THD];
  uint_t i, idx;
  uint_t j, iChunk, nw;
  double sum;

  iChunk = blockIdx.y + blockIdx.z * gridDim.y;
  i = threadIdx.x + blockIdx.x * blockDim.x + iChunk * gridDim.x * blockDim.x;
  if (i >= count)
    return;

  if (!func.check_conditional(i))
    return;

  idx = func.thread_to_index(i);

  cache[threadIdx.x] = func.data()[idx];
  __syncthreads();

  sum = func.run_with_cache_sum(threadIdx.x, idx, cache);

  // reduce in warp
  nw = min(blockDim.x, _WS);
  for (j = 1; j < nw; j *= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, j, 32);
  }

  if (blockDim.x > _WS) {
    // reduce in thread block
    __syncthreads();
    if ((threadIdx.x & (_WS - 1)) == 0) {
      ((double *)cache)[(threadIdx.x / _WS)] = sum;
    }
    __syncthreads();
    if (threadIdx.x < _WS) {
      if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS))
        sum = ((double *)cache)[threadIdx.x];
      else
        sum = 0.0;

      // reduce in warp
      nw = _WS;
      for (j = 1; j < nw; j *= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, j, 32);
      }
    }
  }
  if (threadIdx.x == 0) {
    pReduceBuffer[blockIdx.x + buf_size * iChunk] = sum;
  }
}

__global__ void dev_reduce_sum(double *pReduceBuffer, uint_t n,
                               uint_t buf_size) {
  // One cache entry per warp/wavefront
  __shared__ double cache[_MAX_THD / _WS];
  double sum;
  uint_t i, j, iChunk, nw;

  iChunk = blockIdx.y + blockIdx.z * gridDim.y;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n)
    sum = pReduceBuffer[i + buf_size * iChunk];
  else
    sum = 0.0;

  // reduce in warp
  nw = min(blockDim.x, _WS);
  for (j = 1; j < nw; j *= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, j, 32);
  }

  if (blockDim.x > _WS) {
    // reduce in thread block
    if ((threadIdx.x & (_WS - 1)) == 0) {
      cache[(threadIdx.x / _WS)] = sum;
    }
    __syncthreads();
    if (threadIdx.x < _WS) {
      if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS))
        sum = cache[threadIdx.x];
      else
        sum = 0.0;

      // reduce in warp
      nw = _WS;
      for (j = 1; j < nw; j *= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, j, 32);
      }
    }
  }
  if (threadIdx.x == 0) {
    pReduceBuffer[blockIdx.x + buf_size * iChunk] = sum;
  }
}

template <typename data_t, typename kernel_t>
__global__ void
dev_apply_function_sum_complex(thrust::complex<double> *pReduceBuffer,
                               kernel_t func, uint_t buf_size, uint_t count,
                               bool init) {
  // One cache entry per warp/wavefront
  __shared__ thrust::complex<double> cache[_MAX_THD / _WS];
  thrust::complex<double> sum;
  double tr, ti;
  uint_t i, j, iChunk, nw;

  iChunk = blockIdx.y + blockIdx.z * gridDim.y;
  i = threadIdx.x + blockIdx.x * blockDim.x + iChunk * gridDim.x * blockDim.x;
  if (i >= count)
    return;

  if (!func.check_conditional(i))
    return;

  sum = 0.0;
  if (!init && threadIdx.x == 0 && blockIdx.x == 0) {
    sum = pReduceBuffer[buf_size * iChunk];
  }
  sum += func(i);

  // reduce in warp
  nw = min(blockDim.x, _WS);
  for (j = 1; j < nw; j *= 2) {
    tr = __shfl_xor_sync(0xffffffff, sum.real(), j, 32);
    ti = __shfl_xor_sync(0xffffffff, sum.imag(), j, 32);
    sum += thrust::complex<double>(tr, ti);
  }

  if (blockDim.x > _WS) {
    // reduce in thread block
    if ((threadIdx.x & (_WS - 1)) == 0) {
      cache[(threadIdx.x / _WS)] = sum;
    }
    __syncthreads();
    if (threadIdx.x < _WS) {
      if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS))
        sum = cache[threadIdx.x];
      else
        sum = 0.0;

      // reduce in warp
      nw = _WS;
      for (j = 1; j < nw; j *= 2) {
        tr = __shfl_xor_sync(0xffffffff, sum.real(), j, 32);
        ti = __shfl_xor_sync(0xffffffff, sum.imag(), j, 32);
        sum += thrust::complex<double>(tr, ti);
      }
    }
  }
  if (threadIdx.x == 0) {
    pReduceBuffer[blockIdx.x + buf_size * iChunk] = sum;
  }
}

__global__ void dev_reduce_sum_complex(thrust::complex<double> *pReduceBuffer,
                                       uint_t n, uint_t buf_size) {
  // One cache entry per warp/wavefront
  __shared__ thrust::complex<double> cache[_MAX_THD / _WS];
  thrust::complex<double> sum;
  double tr, ti;
  uint_t i, j, iChunk, nw;

  iChunk = blockIdx.y + blockIdx.z * gridDim.y;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n)
    sum = pReduceBuffer[i + buf_size * iChunk];
  else
    sum = 0.0;

  // reduce in warp
  nw = min(blockDim.x, _WS);
  for (j = 1; j < nw; j *= 2) {
    tr = __shfl_xor_sync(0xffffffff, sum.real(), j, 32);
    ti = __shfl_xor_sync(0xffffffff, sum.imag(), j, 32);
    sum += thrust::complex<double>(tr, ti);
  }

  if (blockDim.x > _WS) {
    // reduce in thread block
    if ((threadIdx.x & (_WS - 1)) == 0) {
      cache[(threadIdx.x / _WS)] = sum;
    }
    __syncthreads();
    if (threadIdx.x < _WS) {
      if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS))
        sum = cache[threadIdx.x];
      else
        sum = 0.0;

      // reduce in warp
      nw = _WS;
      for (j = 1; j < nw; j *= 2) {
        tr = __shfl_xor_sync(0xffffffff, sum.real(), j, 32);
        ti = __shfl_xor_sync(0xffffffff, sum.imag(), j, 32);
        sum += thrust::complex<double>(tr, ti);
      }
    }
  }
  if (threadIdx.x == 0) {
    pReduceBuffer[blockIdx.x + buf_size * iChunk] = sum;
  }
}

__global__ void dev_reduce_sum_uint(uint_t *pReduceBuffer, uint_t n,
                                    uint_t buf_size) {
  // One cache entry per warp/wavefront
  __shared__ uint_t cache[_MAX_THD / _WS];
  uint_t sum;
  uint_t i, j, iChunk, nw;

  iChunk = blockIdx.y + blockIdx.z * gridDim.y;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n)
    sum = pReduceBuffer[i + buf_size * iChunk];
  else
    sum = 0;

  // reduce in warp
  nw = min(blockDim.x, _WS);
  for (j = 1; j < nw; j *= 2) {
    sum += __shfl_xor_sync(0xffffffff, sum, j, 32);
  }

  if (blockDim.x > _WS) {
    // reduce in thread block
    if ((threadIdx.x & (_WS - 1)) == 0) {
      cache[(threadIdx.x / _WS)] = sum;
    }
    __syncthreads();
    if (threadIdx.x < _WS) {
      if (threadIdx.x < ((blockDim.x + _WS - 1) / _WS))
        sum = cache[threadIdx.x];
      else
        sum = 0;

      // reduce in warp
      nw = _WS;
      for (j = 1; j < nw; j *= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, j, 32);
      }
    }
  }
  if (threadIdx.x == 0) {
    pReduceBuffer[blockIdx.x + buf_size * iChunk] = sum;
  }
}

//------------------------------------------------------------------------------
} // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
