/*
 * Tensor.cpp
 *
 *  Created on: Aug 23, 2018
 *      Author: eladgold
 */

#include "Tensor.hpp"
#include <math.h>


Tensor::Tensor(complex_t alpha, complex_t beta)
{
	dim_ = 1;
	data_[0][0] = alpha;
	data_[1][0] = beta;
	for (int i = 1; i < NN; i++)
	{
		data_[0][i] = 0;
		data_[1][i] = 0;
	}
}

Tensor::Tensor(const Tensor& rhs)
{
	dim_ = rhs.dim_;
	for (int i = 0; i < NN; i++)
	{
		data_[0][i] = rhs.data_[0][i];
		data_[1][i] = rhs.data_[1][i];
	}
}

Tensor& Tensor::operator=(const Tensor& rhs)
{
	if (this != &rhs)
	{
		dim_ = rhs.dim_;
		for (int i = 0; i < NN; i++)
		{
			data_[0][i] = rhs.data_[0][i];
			data_[1][i] = rhs.data_[1][i];
		}
	}
	return *this;
}

complex_t Tensor::get_data(int i, int a) const
{
	return data_[i][a];
}

void Tensor::insert_data(int i, int a, complex_t data)
{
  if (a >= NN){
    cout << "ERROR: array size exceeded, a = " << a <<endl;
  }
  try{
  if (a < NN && i < 2)
    data_[i][a] = data;
  else
    throw;
  }
  catch(...) {
  }

  //  catch(meravexception &e) {
  //    cout << e.what() << "ERROR: exceeding size of Tensor, i = " << i << ", a = " << a << endl;

  //  }
}

void Tensor::insert_dim(int new_dim)
{
	dim_ = new_dim;
}

int Tensor::get_dim() const
{
	return dim_;
}

void Tensor::dim_mult()
{
	dim_*=2;
}

void Tensor::dim_mult(int dim)
{
	dim_*=dim;
}

void Tensor::dim_div()
{
	dim_/=2;
}

void Tensor::dim_div(int dim)
{
  if (dim == 0)
    cout << "in dim_div, trying to divide by 0" <<endl;
  dim_/=dim;
}

void Tensor::print()
{
	for(int j=0; j<2; j++)
	{
		for (int i=0; i<dim_; i++)
		{
			cout << data_[j][i] << " ";
		}
		cout << endl;
	}
}

void Tensor::print(int row)
{
	for (int i=0; i<dim_; i++)
	{
		cout << data_[row][i].real() << " ";
	}
	cout << endl;
}
