/*
 * Tensor.hpp
 *
 *  Created on: Aug 23, 2018
 *      Author: eladgold
 */

#ifndef _tensor_tensor_hpp_
#define _tensor_tensor_hpp_

#define SQR_HALF sqrt(0.5)

#include <cstdio>
#include <iostream>
#include <complex>
#include <vector>
#include <math.h>
#include <string.h>
#include <exception>

#include "SVD.hpp"
#include "CSVD.cpp"
#include "framework/matrix.hpp"
#include "framework/utils.hpp"

//namespace AER {
namespace TensorState {

// Data types
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;
using cmatrix_t = matrix<complex_t>;

// Input: vector S that contains the real singular values from the SVD decomposition
// Output: number of elements in S that are greater than 0 (actually greater than threshold)
uint num_of_SV(rvector_t S, double threshold)
{
	uint sum = 0;
	for(uint i = 0; i < S.size(); ++i)
	{
	  if(std::norm(S[i]) > threshold)
		sum++;
	}
	if (sum == 0)
	  cout << "SV_Num == 0"<< '\n';
	return sum;
}

class Tensor
{
public:
	Tensor(){}
	explicit Tensor(complex_t& alpha, complex_t& beta){
		matrix<complex_t> A = matrix<complex_t>(1), B = matrix<complex_t>(1);
		A(0,0) = alpha;
		B(0,0) = beta;
		data_.push_back(A);
		data_.push_back(B);
	}

	Tensor(const Tensor& rhs){
		data_ = rhs.data_;
	}

	virtual ~Tensor(){}

	Tensor& operator=(const Tensor& rhs){
		if (this != &rhs){
			data_ = rhs.data_;
		}
		return *this;
	}

	void print(bool statevector = false) {
	  if(statevector == false)
		  for(uint i = 0; i < data_.size(); i++)
			{
			  data_[i].SetOutputStyle(Matrix);
			  std::cout << "i = " << i << endl;
			  std::cout << data_[i];
			}
	  else
	  {
		  std::cout << "[";
		  for(uint i = 0; i < data_.size(); i++)
			{
			  std::cout << data_[i](0,0)<< " ";
			}
		  std::cout << "]" << endl;
	  }
	}

	cvector_t get_data_new(uint a1, uint a2) const
	{
		cvector_t Res;
		for(uint i = 0; i < data_.size(); i++)
			Res.push_back(data_[i](a1,a2));
		return Res;
	}
	matrix<complex_t> get_data(uint i) const
	{
		return data_[i];
	}


	void insert_data(uint a1, uint a2, vector<complex_t> data)
	{
		for(uint i = 0; i < data_.size(); i++)
			data_[i](a1,a2) = data[i];
	}


	uint get_dim() const {
		return data_.size();
	}

	void apply_x()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		swap(data_[0],data_[1]);
	}
	void apply_y()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[0] = data_[0] * complex_t(0, 1);
		data_[1] = data_[1] * complex_t(0, -1);
		swap(data_[0],data_[1]);
	}
	void apply_z()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * (-1.0);
	}
	void apply_h()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		cvector_t temp;
		for (uint a1 = 0; a1 < data_[0].GetRows(); a1++)
			for (uint a2 = 0; a2 < data_[0].GetColumns(); a2++)
			{
				temp = get_data_new(a1,a2);
				temp = AER::Utils::Matrix::H*temp;
				insert_data(a1,a2,temp);
			}
	}
	void apply_s()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * complex_t(0, 1);
	}
	void apply_sdg()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * complex_t(0, -1);
	}
	void apply_t()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * complex_t(SQR_HALF, SQR_HALF);
	}
	void apply_tdg()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * complex_t(SQR_HALF, -SQR_HALF);
	}
	void apply_u1(double lambda)
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		cvector_t temp;
		for (uint a1 = 0; a1 < data_[0].GetRows(); a1++)
			for (uint a2 = 0; a2 < data_[0].GetColumns(); a2++)
			{
				temp = get_data_new(a1,a2);
				temp = AER::Utils::Matrix::U1(lambda)*temp;
				insert_data(a1,a2,temp);
			}
	}
	void apply_u2(double phi, double lambda)
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		cvector_t temp;
		for (uint a1 = 0; a1 < data_[0].GetRows(); a1++)
			for (uint a2 = 0; a2 < data_[0].GetColumns(); a2++)
			{
				temp = get_data_new(a1,a2);
				temp = AER::Utils::Matrix::U2(phi,lambda)*temp;
				insert_data(a1,a2,temp);
			}
	}
	void apply_u3(double theta, double phi, double lambda)
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		cvector_t temp;
		for (uint a1 = 0; a1 < data_[0].GetRows(); a1++)
			for (uint a2 = 0; a2 < data_[0].GetColumns(); a2++)
			{
				temp = get_data_new(a1,a2);
				temp = AER::Utils::Matrix::U3(theta, phi,lambda)*temp;
				insert_data(a1,a2,temp);
			}
	}
	void apply_matrix(cmatrix_t &mat)
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}

		cvector_t temp;
		for (uint a1 = 0; a1 < data_[0].GetRows(); a1++)
			for (uint a2 = 0; a2 < data_[0].GetColumns(); a2++)
			{
				temp = get_data_new(a1,a2);
				temp = mat * temp;
				insert_data(a1,a2,temp);
			}
	}
	void apply_cnot(bool swapped = false)
	{
		if (data_.size() != 4)
		{
			cout << "ERROR: The tensor doesn't represent 2 qubits" << '\n';
			assert(false);
		}
		if(!swapped)
			swap(data_[2],data_[3]);
		else
			swap(data_[1],data_[3]);
	}
	void apply_swap()
	{
		if (data_.size() != 4)
		{
			cout << "ERROR: The tensor doesn't represent 2 qubits" << '\n';
			assert(false);
		}
		swap(data_[1],data_[2]);
	}
	void apply_cz()
	{
		if (data_.size() != 4)
		{
			cout << "ERROR: The tensor doesn't represent 2 qubits" << '\n';
			assert(false);
		}
		data_[3] = data_[3] * (-1.0);
	}
  
protected:
	/*
	The data structure of a Gamma tensor in MPS- a vector of matrices of the same dimensions.
	Size of the vector is for the real index, dimensions of the matrices are for the bond indices. (3-dimensions tensors)
	*/	
	vector<cmatrix_t> data_;

/* Functions mul/div Gamma by Lambda are used to keep the MPS in the canonical form
 * */
friend void mul_Gamma_by_left_Lambda(rvector_t &Lambda, Tensor &Gamma)
{
	if (Lambda == rvector_t {1.0}) return;
	uint rows = Gamma.data_[0].GetRows(), cols = Gamma.data_[0].GetColumns();
	for(uint i = 0; i < Gamma.data_.size(); i++)
		for(uint a1 = 0; a1 < rows; a1++)
			for(uint a2 = 0; a2 < cols; a2++)
				Gamma.data_[i](a1,a2) *= Lambda[a1];
}

friend void mul_Gamma_by_right_Lambda(Tensor &Gamma, rvector_t &Lambda)
{
	if (Lambda == rvector_t {1.0}) return;
	uint rows = Gamma.data_[0].GetRows(), cols = Gamma.data_[0].GetColumns();
	for(uint i = 0; i < Gamma.data_.size(); i++)
		for(uint a1 = 0; a1 < rows; a1++)
			for(uint a2 = 0; a2 < cols; a2++)
				Gamma.data_[i](a1,a2) *= Lambda[a2];
}

friend void div_Gamma_by_left_Lambda(rvector_t &Lambda, Tensor &Gamma)
{
	if (Lambda == rvector_t {1.0}) return;
	uint rows = Gamma.data_[0].GetRows(), cols = Gamma.data_[0].GetColumns();
	for(uint i = 0; i < Gamma.data_.size(); i++)
		for(uint a1 = 0; a1 < rows; a1++)
			for(uint a2 = 0; a2 < cols; a2++)
				Gamma.data_[i](a1,a2) /= Lambda[a1];
}

friend void div_Gamma_by_right_Lambda(Tensor &Gamma, rvector_t &Lambda)
{
	if (Lambda == rvector_t {1.0}) return;
	uint rows = Gamma.data_[0].GetRows(), cols = Gamma.data_[0].GetColumns();
	for(uint i = 0; i < Gamma.data_.size(); i++)
		for(uint a1 = 0; a1 < rows; a1++)
			for(uint a2 = 0; a2 < cols; a2++)
				Gamma.data_[i](a1,a2) /= Lambda[a2];
}

// Contract to gammas and one lambda of the MPS. Usually being used before 2-qubits gate.
friend Tensor contract(Tensor &A, rvector_t &lambda, Tensor &B)
{
	Tensor Res;
	mul_Gamma_by_right_Lambda(A,lambda);
	for(uint i = 0; i < A.data_.size(); i++)
		for(uint j = 0; j < B.data_.size(); j++)
			Res.data_.push_back(A.data_[i] * B.data_[j]);
	return Res;
}

// Decompose a tensor into 2 gammas and lambda of the MPS. Usually being used after 2-qubits gate.
friend void Decompose(Tensor &temp, Tensor &left_gamma, rvector_t &lambda, Tensor &right_gamma)
{
	matrix<complex_t> C = reshape_before_SVD(temp.data_);
	matrix<complex_t> U,V;
	rvector_t S(min(C.GetRows(), C.GetColumns()));


	C.SetOutputStyle(Matrix);
	U.SetOutputStyle(Matrix);
	V.SetOutputStyle(Matrix);
	if(SHOW_SVD) cout << "C =" << endl << C ;
	csvd(C,U,S,V);
	if(SHOW_SVD) {
		cout << "U = " << endl << U ;
		cout << "S = " << endl;
		for (uint i = 0; i != S.size(); ++i)
		    cout << S[i] << ' , ';
		cout << endl;
		cout << "V = " << endl << V ;
	}
	uint SV_num = num_of_SV(S, 1e-16);
	U.resize(U.GetRows(),SV_num);
	S.resize(SV_num);
	V.resize(V.GetRows(),SV_num);
	left_gamma.data_  = reshape_U_after_SVD(U);
	lambda            = S;
	right_gamma.data_ = reshape_V_after_SVD(V);
}


};




//-------------------------------------------------------------------------
} // end namespace TensorState
//-------------------------------------------------------------------------
//} // end namespace AER
//-------------------------------------------------------------------------
#endif
