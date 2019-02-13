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

using namespace std;
// Data types
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;

void csvd (complex_t **a, int m, int n, double *s, complex_t **u, complex_t **v );

#endif /* SVD_HPP_ */
