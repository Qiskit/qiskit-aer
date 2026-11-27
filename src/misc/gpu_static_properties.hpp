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
#ifndef __GPU_STATIC_PRIORITIES_H__
#define __GPU_STATIC_PRIORITIES_H__

#ifdef AER_THRUST_ROCM
#include <hip/hip_runtime.h>
// AMD GPU wavefront sizes:
// - CDNA (MI100/MI200/MI300): 64
// - RDNA (RX 6000/7000): 32
// We use 64 for CDNA as it's the most common data center GPU.
// CRITICAL: Must be a true compile-time constant for __shared__ array declarations
// warpSize in ROCm is a runtime builtin, not usable in constant expressions
constexpr int _WS = 64;
// Maximum number of threads in a block.
constexpr int _MAX_THD = 1024;
#endif // AER_THRUST_ROCM

#ifdef AER_THRUST_CUDA
// In CUDA warpSize could not be a compile-time constant so we use 32 directly.
#define _WS 32
// Maximum number of threads in a block.
#define _MAX_THD 1024
#endif // AER_THRUST_CUDA

#endif //__GPU_STATIC_PRIORITIES_H__
