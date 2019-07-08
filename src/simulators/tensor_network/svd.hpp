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
#include "framework/types.hpp"

using namespace std;
namespace AER {
// Data types
using long_complex_t = std::complex<long double>;
  //using complex_t = std::complex<double>;
  //using cvector_t = std::vector<complex_t>;
  //using rvector_t = std::vector<double>;
  //using rmatrix_t = matrix<double>;
  //using cmatrix_t = matrix<complex_t>;

enum status {SUCCESS, FAILURE};

cmatrix_t reshape_before_SVD(vector<cmatrix_t> data);
vector<cmatrix_t> reshape_U_after_SVD(cmatrix_t U);
rvector_t reshape_S_after_SVD(rvector_t S);
vector<cmatrix_t> reshape_V_after_SVD(const cmatrix_t V);
uint_t num_of_SV(rvector_t S, double threshold);
void reduce_zeros(cmatrix_t &U, rvector_t &S, cmatrix_t &V);
status csvd(cmatrix_t &C, cmatrix_t &U,rvector_t &S,cmatrix_t &V);
void csvd_wrapper(cmatrix_t &C, cmatrix_t &U,rvector_t &S,cmatrix_t &V);

} //namespace AER

#endif /* SVD_HPP_ */
