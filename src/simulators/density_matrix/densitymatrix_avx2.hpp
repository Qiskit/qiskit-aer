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

#ifndef _qv_density_matrix_avx2_hpp_
#define _qv_density_matrix_avx2_hpp_

#include <cmath>
#include "densitymatrix.hpp"
#include "simulators/statevector/base_avx2.hpp"

namespace AER {
namespace QV {

template <typename data_t = double>
class DensityMatrixAvx2 : public BaseAvx2<DensityMatrix<data_t, BaseAvx2<data_t>>, data_t>{
public:
    using BaseAvx2<DensityMatrix<data_t, BaseAvx2<data_t>>, data_t>::BaseAvx2;

    size_t calculate_num_threads();
};

// We do not define this functions in case we don't use AVX2
// so it can compile, as this class won't be used
#if defined(_MSC_VER) || defined(GNUC_AVX2)

template <typename data_t>
size_t DensityMatrixAvx2<data_t>::calculate_num_threads() {
  if ((BaseAvx2<DensityMatrix<data_t, BaseAvx2<data_t>>, data_t>::Base::num_qubits_ << 1UL ) > BaseAvx2<DensityMatrix<data_t, BaseAvx2<data_t>>, data_t>::Base::omp_threshold_ && BaseAvx2<DensityMatrix<data_t, BaseAvx2<data_t>>, data_t>::Base::omp_threads_ > 1) {
    return BaseAvx2<DensityMatrix<data_t, BaseAvx2<data_t>>, data_t>::Base::omp_threads_;
  }
  return 1;
}
#endif

}
}
//------------------------------------------------------------------------------
#endif // end module
