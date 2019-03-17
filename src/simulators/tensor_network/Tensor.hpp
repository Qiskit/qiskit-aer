/*
 * Tensor.hpp
 *
 *  Created on: Aug 23, 2018
 *      Author: eladgold
 */

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

//#define NN 1024
#define NN 131072
#include <cstdio>
#include <iostream>
#include <complex>
#include <vector>
#include "SVD.hpp"
#include "matrix.hpp"
#include <math.h>
#include <string.h>
#include <exception>


using namespace std;

// Data types
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;

class Tensor
{
public:
	Tensor(complex_t alpha, complex_t beta);
	Tensor(const Tensor& rhs);
	~Tensor(){}

	Tensor& operator=(const Tensor& rhs);
	void print();
	void print(int row);
	complex_t get_data(int i, int a) const;
	void insert_data(int i, int a, complex_t data);
	int get_dim() const;
	void insert_dim(int new_dim);
	void dim_mult();
	void dim_mult(int dim);
	void dim_div();
	void dim_div(int dim);

protected:
	complex_t data_[2][NN];
	int dim_;
};

#endif // TENSOR_HPP_
