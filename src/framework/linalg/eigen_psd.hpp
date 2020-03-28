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

#ifndef _aer_framework_linalg_eigen_psd_hpp_
#define _aer_framework_linalg_eigen_psd_hpp_

#include "framework/matrix.hpp"

template <class float_t>
void eigen_psd(const matrix<std::complex<float_t>>& psd_matrix,
               /* out */ std::vector<std::complex<float_t>> &eigen_values,
               /* out */ matrix<std::complex<float_t>> &eigen_vectors){

}

#endif