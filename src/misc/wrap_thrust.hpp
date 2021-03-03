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

#ifndef _aer_misc_wrap_thrust_hpp_
#define _aer_misc_wrap_thrust_hpp_

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC system_header
#endif

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <thrust/for_each.h>
#include <thrust/complex.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/detail/vector_base.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/fill.h>

#ifdef AER_THRUST_CUDA
#include <thrust/device_vector.h>
#endif
#include <thrust/host_vector.h>

#include <thrust/system/omp/execution_policy.h>
DISABLE_WARNING_POP

#endif // inclusion guard
