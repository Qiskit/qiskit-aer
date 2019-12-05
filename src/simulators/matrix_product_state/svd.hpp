/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */


#ifndef SVD_HPP_
#define SVD_HPP_

#include <complex>
#include <vector>
#include "framework/utils.hpp"
#include "framework/types.hpp"

namespace AER {
// Data types
using long_complex_t = std::complex<long double>;

enum status {SUCCESS, FAILURE};

  cmatrix_t reshape_before_SVD(std::vector<cmatrix_t> data);
std::vector<cmatrix_t> reshape_U_after_SVD(cmatrix_t U);
rvector_t reshape_S_after_SVD(rvector_t S);
std::vector<cmatrix_t> reshape_V_after_SVD(const cmatrix_t V);
uint_t num_of_SV(rvector_t S, double threshold);
void reduce_zeros(cmatrix_t &U, rvector_t &S, cmatrix_t &V);
status csvd(cmatrix_t &C, cmatrix_t &U,rvector_t &S,cmatrix_t &V);
void csvd_wrapper(cmatrix_t &C, cmatrix_t &U,rvector_t &S,cmatrix_t &V);

} //namespace AER

#endif /* SVD_HPP_ */
