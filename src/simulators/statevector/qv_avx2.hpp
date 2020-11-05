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

#ifndef _qv_avx2_hpp_
#define _qv_avx2_hpp_

#include <cstdint>
#include <cstring>

namespace AER {
namespace QV {

enum class Avx { NotApplied, Applied };

template <typename FloatType>
Avx apply_matrix_avx(FloatType* data,
                     const uint64_t data_size,
                     const uint64_t* qregs,
                     const size_t qregs_size,
                     const FloatType* mat,
                     const size_t omp_threads);


template <typename FloatType>
Avx apply_diagonal_matrix_avx(FloatType* data,
                              const uint64_t data_size,
                              const uint64_t* qregs,
                              const size_t qregs_size,
                              const FloatType* vec,
                              const size_t omp_threads);

} // end namespace QV
} // end namespace AER
#endif
