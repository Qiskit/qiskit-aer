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

namespace AER {
namespace TensorNetworkState {

// Data types
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;
using cmatrix_t = matrix<complex_t>;

uint num_of_SV(rvector_t S, double threshold);

//**************************************************************
// function name: num_of_SV
// Description: Computes the number of none-zero singular values
//				in S
// Parameters: rvector_t S - vector of singular values from the
//			   SVD decomposition
//			   false to print as vector of matrices.
// Returns: number of elements in S that are greater than 0
//			(actually greater than threshold)
//**************************************************************
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

class MPS_Tensor
{
public:
  // Constructors of MPS_Tensor class
  MPS_Tensor(){}
  explicit MPS_Tensor(complex_t& alpha, complex_t& beta){
    matrix<complex_t> A = matrix<complex_t>(1), B = matrix<complex_t>(1);
    A(0,0) = alpha;
    B(0,0) = beta;
    data_.push_back(A);
    data_.push_back(B);
  }
  MPS_Tensor(const MPS_Tensor& rhs){
    data_ = rhs.data_;
  }
  // Destructor
  virtual ~MPS_Tensor(){}
  
  // Assignment operator
  MPS_Tensor& operator=(const MPS_Tensor& rhs){
    if (this != &rhs){
      data_ = rhs.data_;
    }
    return *this;
  }
  void print(bool statevector = false);
  cvector_t get_data(uint a1, uint a2) const;
  cmatrix_t get_data(uint i) const {
    return data_[i];
  }
  void insert_data(uint a1, uint a2, cvector_t data);

  //**************************************************************
  // function name: get_dim
  // Description: Get the dimension of the physical index of the tensor
  // Parameters: none.
  // Returns: uint of the dimension of the physical index of the tensor.
  //**************************************************************
  uint get_dim() const {
    return data_.size();
  }
  void apply_x();
  void apply_y();
  void apply_z();
  void apply_h();
  void apply_s();
  void apply_sdg();
  void apply_t();
  void apply_tdg();
  void apply_u1(double lambda);
  void apply_u2(double phi, double lambda);
  void apply_u3(double theta, double phi, double lambda);
  void apply_matrix(cmatrix_t &mat);
  void apply_cnot(bool swapped = false);
  void apply_swap();
  void apply_cz();
  void mul_Gamma_by_left_Lambda(const rvector_t &Lambda);
  void mul_Gamma_by_right_Lambda(const rvector_t &Lambda);
  void div_Gamma_by_left_Lambda(const rvector_t &Lambda);
  void div_Gamma_by_right_Lambda(const rvector_t &Lambda);
  static MPS_Tensor contract(const MPS_Tensor &left_gamma, const rvector_t &lambda, const MPS_Tensor &right_gamma);
  static void Decompose(MPS_Tensor &temp, MPS_Tensor &left_gamma, rvector_t &lambda, MPS_Tensor &right_gamma);

private:
  void mul_Gamma_by_Lambda(const rvector_t &Lambda, 
			   bool right, /* or left */
			   bool mul    /* or div */);
	/*
	The data structure of a Gamma tensor in MPS- a vector of matrices of
	the same dimensions. Size of the vector is for the physical index,
	dimensions of the matrices are for the bond indices. (3-dimensions tensors).
	Notation: i will represent the physical index, a1,a2 will represent the
	matrix indexes
	*/	
  vector<cmatrix_t> data_;
};

//=========================================================================
// Implementation
//=========================================================================

  //**************************************************************
    // function name: print
    // Description: Add a new command to the history
    // Parameters: bool statevector: true to print as a state vector
    //			   false to print as vector of matrices.
    // Returns: none.
    //**************************************************************
    void MPS_Tensor::print(bool statevector) {
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
  
  //**************************************************************
    // function name: get_data
    // Description: Get the data in some axis of the MPS_Tensor
    // 1.	Parameters: uint a1, uint a2 - indexes of data in matrix
    // 		Returns: cvector_t of data in (a1,a2) in all matrices
    // 2.	Parameters: uint i - index of a matrix in the MPS_Tensor
    // 		Returns: cmatrix_t of the data
    //**************************************************************
    cvector_t MPS_Tensor::get_data(uint a1, uint a2) const
   {
    cvector_t Res;
    for(uint i = 0; i < data_.size(); i++)
      Res.push_back(data_[i](a1,a2));
    return Res;
   }

  //**************************************************************
    // function name: insert_data
    // Description: Insert data to some axis of the MPS_Tensor
    // Parameters: uint a1, uint a2 - indexes of data in matrix
    // Parameters: cvector_t data - data to insert.
    // Returns: void.
    //**************************************************************
    void MPS_Tensor::insert_data(uint a1, uint a2, cvector_t data)
  {
    for(uint i = 0; i < data_.size(); i++)
      data_[i](a1,a2) = data[i];
  }



	//**************************************************************
	// function name: apply_x,y,z,...
	// Description: Apply some gate on the tensor. tensor must represent
	//				the number of qubits the gate expect
	// Parameters: none.
	// Returns: none.
	//**************************************************************
	void MPS_Tensor::apply_x()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		swap(data_[0],data_[1]);
	}
	void MPS_Tensor::apply_y()
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
	void MPS_Tensor::apply_z()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * (-1.0);
	}
	void MPS_Tensor::apply_h()
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
				temp = get_data(a1,a2);
				temp = AER::Utils::Matrix::H*temp;
				insert_data(a1,a2,temp);
			}
	}
	void MPS_Tensor::apply_s()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * complex_t(0, 1);
	}
	void MPS_Tensor::apply_sdg()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * complex_t(0, -1);
	}
	void MPS_Tensor::apply_t()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * complex_t(SQR_HALF, SQR_HALF);
	}
	void MPS_Tensor::apply_tdg()
	{
		if (data_.size() != 2)
		{
			cout << "ERROR: The tensor doesn't represent one qubit" << '\n';
			assert(false);
		}
		data_[1] = data_[1] * complex_t(SQR_HALF, -SQR_HALF);
	}
	void MPS_Tensor::apply_u1(double lambda)
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
				temp = get_data(a1,a2);
				temp = AER::Utils::Matrix::u1(lambda)*temp;
				insert_data(a1,a2,temp);
			}
	}
	void MPS_Tensor::apply_u2(double phi, double lambda)
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
				temp = get_data(a1,a2);
				temp = AER::Utils::Matrix::u2(phi,lambda)*temp;
				insert_data(a1,a2,temp);
			}
	}
	void MPS_Tensor::apply_u3(double theta, double phi, double lambda)
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
				temp = get_data(a1,a2);
				temp = AER::Utils::Matrix::u3(theta, phi,lambda)*temp;
				insert_data(a1,a2,temp);
			}
	}
	void MPS_Tensor::apply_matrix(cmatrix_t &mat)
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
				temp = get_data(a1,a2);
				temp = mat * temp;
				insert_data(a1,a2,temp);
			}
	}
	void MPS_Tensor::apply_cnot(bool swapped)
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
	void MPS_Tensor::apply_swap()
	{
		if (data_.size() != 4)
		{
			cout << "ERROR: The tensor doesn't represent 2 qubits" << '\n';
			assert(false);
		}
		swap(data_[1],data_[2]);
	}
	void MPS_Tensor::apply_cz()
	{
		if (data_.size() != 4)
		{
			cout << "ERROR: The tensor doesn't represent 2 qubits" << '\n';
			assert(false);
		}
		data_[3] = data_[3] * (-1.0);
	}
  

/* TL;DR - Functions mul/div Gamma by Lambda are used to keep the MPS in the
 * canonical form.
 *
 * Before applying a 2-qubit gate, we must contract these qubits to relevant Gamma tensors.
 * To maintain the canonical form, we must consider the Lambda tensors from
 * the sides of the Gamma tensors. This is what the multiply functions do. After the
 * decomposition of the result of the gate, we need to divide back by what we
 * multiplied before. This is what the division functions do.
 * */
void MPS_Tensor::mul_Gamma_by_left_Lambda(const rvector_t &Lambda)
{
  mul_Gamma_by_Lambda(Lambda, false,/*left*/ true /*mul*/);
}

void MPS_Tensor::mul_Gamma_by_right_Lambda(const rvector_t &Lambda)
{
  mul_Gamma_by_Lambda(Lambda, true,/*right*/ true /*mul*/);
}

void MPS_Tensor::div_Gamma_by_left_Lambda(const rvector_t &Lambda)
{
  mul_Gamma_by_Lambda(Lambda, false,/*left*/ false /*div*/);
}

void MPS_Tensor::div_Gamma_by_right_Lambda(const rvector_t &Lambda)
{
  mul_Gamma_by_Lambda(Lambda, true,/*right*/ false /*div*/);
}

void MPS_Tensor::mul_Gamma_by_Lambda(const rvector_t &Lambda, 
			 bool right, /* or left */
			 bool mul    /* or div */)
{
	if (Lambda == rvector_t {1.0}) return;
	uint rows = data_[0].GetRows(), cols = data_[0].GetColumns();
	for(uint i = 0; i < data_.size(); i++)
		for(uint a1 = 0; a1 < rows; a1++)
		  for(uint a2 = 0; a2 < cols; a2++) {
		    uint factor = right ? a2 : a1;
		    if (mul) {
		      data_[i](a1,a2) *= Lambda[factor];
		    } else{
		      data_[i](a1,a2) /= Lambda[factor];
		    }
		  }
}

//************************************************************************
// function name: contract
// Description: Contract two Gamma tensors and the Lambda between
// 				them. Usually used before 2-qubits gate.
// Parameters: MPS_Tensor &left_gamma, &right_gamma , rvector_t &lambda -
// 			   tensors to contract.
// Returns: The result tensor of the contract
//*************************************************************************
MPS_Tensor MPS_Tensor::contract(const MPS_Tensor &left_gamma, const rvector_t &lambda, const MPS_Tensor &right_gamma)
{
	MPS_Tensor Res;
	MPS_Tensor new_left = left_gamma;
	new_left.mul_Gamma_by_right_Lambda(lambda);
	for(uint i = 0; i < new_left.data_.size(); i++)
		for(uint j = 0; j < right_gamma.data_.size(); j++)
			Res.data_.push_back(new_left.data_[i] * right_gamma.data_[j]);
	return Res;
}

// Decompose a tensor into 2 gammas and lambda of the MPS. Usually being used after 2-qubits gate.
//************************************************************************
// function name: Decompose
// Description: Decompose a tensor into two Gamma tensors and the Lambda between
// 				them. Usually used after applying a 2-qubit gate.
// Parameters: MPS_Tensor &temp - the tensor to decompose.
//			   MPS_Tensor &left_gamma, &right_gamma , rvector_t &lambda -
// 			   tensors for the result.
// Returns: none.
//*************************************************************************
void MPS_Tensor::Decompose(MPS_Tensor &temp, MPS_Tensor &left_gamma, rvector_t &lambda, MPS_Tensor &right_gamma)
{
	matrix<complex_t> C = reshape_before_SVD(temp.data_);
	matrix<complex_t> U,V;
	rvector_t S(min(C.GetRows(), C.GetColumns()));


	if(SHOW_SVD)
	{
		C.SetOutputStyle(Matrix);
		U.SetOutputStyle(Matrix);
		V.SetOutputStyle(Matrix);
		cout << "C =" << endl << C ;
	}

	csvd(C,U,S,V);

	if(SHOW_SVD) {
		cout << "U = " << endl << U ;
		cout << "S = " << endl;
		for (uint i = 0; i != S.size(); ++i)
		    cout << S[i] << " , ";
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

//-------------------------------------------------------------------------
} // end namespace MPS_TensorState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
