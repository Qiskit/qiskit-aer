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

#include <type_traits>
#include "framework/matrix.hpp"

/**
 * Returns the eigenvalues and eigenvectors of a Hermitian
 * positive semi-definite (PSD) matrix.
 * @param psd_matrix: The Hermitian PSD matrix.
 * @param eigenvalues: The eignevalues of the matrix.
 * @param eigenvectors: The eigenvectors of the matrix.
 *
 * @returns: true.
 */
template <class float_t>
void eigensystem_psd(const matrix<std::complex<float_t>>& psd_matrix,
               /* out */ std::vector<std::complex<float_t>>& eigenvalues,
               /* out */ std::vector<std::vector<std::complex<float_t>>>& eigenvectors){


}

#endif
