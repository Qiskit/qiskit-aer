/**
 * This code is part of Qiskit.
 *
 * (C) Copyright AMD 2025.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_rocm_wavefront_utils_hpp_
#define _aer_rocm_wavefront_utils_hpp_

#ifdef AER_THRUST_ROCM

#include <hip/hip_runtime.h>

namespace AER {
namespace ROCm {

//============================================================================
// AMD Wavefront Utilities
// Provides optimized warp/wavefront operations for AMD GPUs
// AMD uses wavefront size of 64 (CDNA) or 32 (RDNA) vs NVIDIA's 32
//============================================================================

// Get wavefront size at compile time based on architecture
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  // CDNA architectures (MI100, MI200, MI300) use 64-wide wavefronts
  #define AER_AMD_WAVEFRONT_SIZE 64
  #define AER_AMD_ARCH_CDNA 1
#elif defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
      defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
  // RDNA architectures (RX 6000, RX 7000) use 32-wide wavefronts
  #define AER_AMD_WAVEFRONT_SIZE 32
  #define AER_AMD_ARCH_RDNA 1
#else
  // Default to 64 for unknown AMD architectures
  #define AER_AMD_WAVEFRONT_SIZE 64
#endif

// Runtime wavefront size detection
__device__ inline int get_wavefront_size() {
#if defined(AER_AMD_ARCH_CDNA)
  return 64;
#elif defined(AER_AMD_ARCH_RDNA)
  return 32;
#else
  return __builtin_amdgcn_wavefrontsize();
#endif
}

// Get lane ID within wavefront
__device__ inline int get_lane_id() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// Wavefront-level primitives optimized for AMD

// Ballot - get mask of lanes where predicate is true
__device__ inline uint64_t wavefront_ballot(int predicate) {
#if AER_AMD_WAVEFRONT_SIZE == 64
  return __ballot(predicate);
#else
  // For 32-wide wavefronts, cast to uint64_t
  return static_cast<uint64_t>(__ballot(predicate));
#endif
}

// Count number of active lanes (population count)
__device__ inline int wavefront_active_count() {
  uint64_t mask = wavefront_ballot(1);
  return __popcll(mask);
}

// Wavefront reduction: sum across all lanes
template <typename T>
__device__ inline T wavefront_reduce_sum(T value) {
  const int lane_id = get_lane_id();
  const int wavefront_size = get_wavefront_size();
  
#if AER_AMD_WAVEFRONT_SIZE == 64
  // 64-wide wavefront: 6 shuffle iterations (log2(64) = 6)
  #pragma unroll
  for (int offset = 32; offset > 0; offset /= 2) {
    value += __shfl_xor(value, offset, wavefront_size);
  }
#else
  // 32-wide wavefront: 5 shuffle iterations (log2(32) = 5)
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor(value, offset, wavefront_size);
  }
#endif
  
  return value;
}

// Wavefront reduction: max across all lanes
template <typename T>
__device__ inline T wavefront_reduce_max(T value) {
  const int lane_id = get_lane_id();
  const int wavefront_size = get_wavefront_size();
  
#if AER_AMD_WAVEFRONT_SIZE == 64
  #pragma unroll
  for (int offset = 32; offset > 0; offset /= 2) {
    T other = __shfl_xor(value, offset, wavefront_size);
    value = (value > other) ? value : other;
  }
#else
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    T other = __shfl_xor(value, offset, wavefront_size);
    value = (value > other) ? value : other;
  }
#endif
  
  return value;
}

// Wavefront reduction: min across all lanes
template <typename T>
__device__ inline T wavefront_reduce_min(T value) {
  const int lane_id = get_lane_id();
  const int wavefront_size = get_wavefront_size();
  
#if AER_AMD_WAVEFRONT_SIZE == 64
  #pragma unroll
  for (int offset = 32; offset > 0; offset /= 2) {
    T other = __shfl_xor(value, offset, wavefront_size);
    value = (value < other) ? value : other;
  }
#else
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    T other = __shfl_xor(value, offset, wavefront_size);
    value = (value < other) ? value : other;
  }
#endif
  
  return value;
}

// Prefix sum (scan) across wavefront
template <typename T>
__device__ inline T wavefront_prefix_sum(T value) {
  const int lane_id = get_lane_id();
  const int wavefront_size = get_wavefront_size();
  
#if AER_AMD_WAVEFRONT_SIZE == 64
  // 64-wide inclusive prefix sum
  #pragma unroll
  for (int offset = 1; offset < 64; offset *= 2) {
    T temp = __shfl_up(value, offset, wavefront_size);
    if (lane_id >= offset) {
      value += temp;
    }
  }
#else
  // 32-wide inclusive prefix sum
  #pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    T temp = __shfl_up(value, offset, wavefront_size);
    if (lane_id >= offset) {
      value += temp;
    }
  }
#endif
  
  return value;
}

// Broadcast value from lane 0 to all lanes
template <typename T>
__device__ inline T wavefront_broadcast(T value) {
  return __shfl(value, 0, get_wavefront_size());
}

// Exchange values between lanes with specified index
template <typename T>
__device__ inline T wavefront_shuffle(T value, int src_lane) {
  return __shfl(value, src_lane, get_wavefront_size());
}

// Shuffle XOR for butterfly reduction patterns
template <typename T>
__device__ inline T wavefront_shuffle_xor(T value, int lane_mask) {
  return __shfl_xor(value, lane_mask, get_wavefront_size());
}

//============================================================================
// Block-level reductions using wavefront primitives
// Optimized for AMD's memory hierarchy and wavefront size
//============================================================================

// Block reduction: sum across entire thread block
template <typename T>
__device__ inline T block_reduce_sum(T value) {
  // Shared memory for inter-wavefront communication
  __shared__ T wavefront_sums[64]; // Max 64 wavefronts per block
  
  const int lane_id = get_lane_id();
  const int wavefront_id = threadIdx.x / get_wavefront_size();
  const int num_wavefronts = (blockDim.x + get_wavefront_size() - 1) / get_wavefront_size();
  
  // Reduce within wavefront
  T wavefront_sum = wavefront_reduce_sum(value);
  
  // First lane of each wavefront writes to shared memory
  if (lane_id == 0) {
    wavefront_sums[wavefront_id] = wavefront_sum;
  }
  __syncthreads();
  
  // First wavefront reduces the wavefront sums
  T result = 0;
  if (wavefront_id == 0 && lane_id < num_wavefronts) {
    result = wavefront_sums[lane_id];
    result = wavefront_reduce_sum(result);
  }
  
  // Broadcast result to all threads
  __syncthreads();
  if (wavefront_id == 0 && lane_id == 0) {
    wavefront_sums[0] = result;
  }
  __syncthreads();
  
  return wavefront_sums[0];
}

// Block reduction: max across entire thread block
template <typename T>
__device__ inline T block_reduce_max(T value) {
  __shared__ T wavefront_maxs[64];
  
  const int lane_id = get_lane_id();
  const int wavefront_id = threadIdx.x / get_wavefront_size();
  const int num_wavefronts = (blockDim.x + get_wavefront_size() - 1) / get_wavefront_size();
  
  T wavefront_max = wavefront_reduce_max(value);
  
  if (lane_id == 0) {
    wavefront_maxs[wavefront_id] = wavefront_max;
  }
  __syncthreads();
  
  T result = value;
  if (wavefront_id == 0 && lane_id < num_wavefronts) {
    result = wavefront_maxs[lane_id];
    result = wavefront_reduce_max(result);
  }
  
  __syncthreads();
  if (wavefront_id == 0 && lane_id == 0) {
    wavefront_maxs[0] = result;
  }
  __syncthreads();
  
  return wavefront_maxs[0];
}

//============================================================================
// Memory access patterns optimized for AMD GPU memory hierarchy
//============================================================================

// Coalesced memory access helper
// AMD GPUs prefer 64-byte aligned accesses (cache line size)
template <typename T>
__device__ inline T coalesced_load(const T* addr) {
  // Ensure alignment for optimal memory throughput
  return __ldg(addr);
}

// Vectorized load for better memory bandwidth utilization
template <typename T>
__device__ inline void vectorized_load_128(T* dest, const T* src) {
  // Load 128 bits (16 bytes) at once
  // Optimal for HBM on MI200/MI300
  *reinterpret_cast<uint4*>(dest) = *reinterpret_cast<const uint4*>(src);
}

//============================================================================
// Architecture-specific optimizations
//============================================================================

// Check if we're running on CDNA architecture (MI100/MI200/MI300)
__device__ inline bool is_cdna_arch() {
#ifdef AER_AMD_ARCH_CDNA
  return true;
#else
  return false;
#endif
}

// Check if we're running on RDNA architecture (RX 6000/7000)
__device__ inline bool is_rdna_arch() {
#ifdef AER_AMD_ARCH_RDNA
  return true;
#else
  return false;
#endif
}

// Get optimal block size for architecture
__device__ inline int get_optimal_block_size() {
#ifdef AER_AMD_ARCH_CDNA
  // CDNA: 64-wide wavefronts, optimal 256 threads (4 wavefronts)
  return 256;
#elif defined(AER_AMD_ARCH_RDNA)
  // RDNA: 32-wide wavefronts, optimal 128-256 threads
  return 128;
#else
  return 256;
#endif
}

// Get optimal number of threads for memory-bound kernels
__device__ inline int get_memory_optimal_threads() {
#ifdef AER_AMD_ARCH_CDNA
  // CDNA has high memory bandwidth, can sustain more threads
  return 1024;
#else
  return 512;
#endif
}

} // namespace ROCm
} // namespace AER

#endif // AER_THRUST_ROCM
#endif // _aer_rocm_wavefront_utils_hpp_
