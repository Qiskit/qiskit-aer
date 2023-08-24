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

#ifdef AER_THRUST_CUDA
// In CUDA warpSize could not be a compile-time constant so we use 32 directly.
#define _WS 32
// Maximum number of threads in a block.
#define _MAX_THD 1024
#endif // AER_THRUST_CUDA

#endif //__GPU_STATIC_PRIORITIES_H__
