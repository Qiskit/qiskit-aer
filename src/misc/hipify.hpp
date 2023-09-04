/**
 * This code is part of Qiskit.
 *
 * (C) Copyright AMD 2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */
#ifndef __HIPIFY_H__
#define __HIPIFY_H__

#include "misc/gpu_static_properties.hpp"

// Define an equivalent for __shfl*_sync. This assumes that all threads
// in the wavefront are active, i.e. the mask is all ones.

template <unsigned mask, int width, typename T>
__device__ T __shfl_xor_aux(T var, int laneMask) {
  // Assert based on the values that make sense in CUDA.
  static_assert(mask == 0xffffffff,
                "Shuffle XOR implementation assumes all wavefront is active.");
  static_assert(width == 32,
                "Shuffle XOR implementation assumes on the whole wavefront.");
  // In AMDGCN all wavefront intrinsics are synchronous.
  return __shfl_xor(var, laneMask, _WS);
}
#define __shfl_xor_sync(mask, var, laneMask, width)                            \
  __shfl_xor_aux<mask, width>(var, laneMask);

template <unsigned mask, int width, typename T>
__device__ T __shfl_aux(T var, int lane) {
  // Assert based on the values that make sense in CUDA.
  static_assert(mask == 0xffffffff,
                "Shuffle implementation assumes all wavefront is active.");
  static_assert(width == 32,
                "Shuffle implementation assumes on the whole wavefront.");
  // In AMDGCN all wavefront intrinsics are synchronous.
  return __shfl(var, lane, _WS);
}
#define __shfl_sync(mask, var, lane, width) __shfl_aux<mask, width>(var, lane);

//
// HIP types
//
#define cudaDataType hipDataType
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

#endif //__HIPIFY_H__
