/*
 * SVD.hpp
 *
 *  Created on: Sep 9, 2018
 *      Author: eladgold
 */

#ifndef SVD_HPP_
#define SVD_HPP_

#include <complex>
#include <vector>
#include "framework/utils.hpp"

#define DEBUG false
#define SHOW_SVD false

using namespace std;
// Data types
using long_complex_t = std::complex<long double>;
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;
using rmatrix_t = matrix<double>;
using cmatrix_t = matrix<complex_t>;

cmatrix_t reshape_before_SVD(vector<cmatrix_t> data);
vector<cmatrix_t> reshape_U_after_SVD(cmatrix_t U);
rvector_t reshape_S_after_SVD(rvector_t S);
vector<cmatrix_t> reshape_V_after_SVD(cmatrix_t V);
void csvd(cmatrix_t &C, cmatrix_t &U,rvector_t &S,cmatrix_t &V);
void csvd_warp(cmatrix_t &C, cmatrix_t &U,rvector_t &S,cmatrix_t &V);

#endif /* SVD_HPP_ */
