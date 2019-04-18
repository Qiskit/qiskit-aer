/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */


#include <bitset>
#include <math.h>

// for analysis of memory consumption
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "framework/utils.hpp"

#include "tensor_state.hpp"
#include "tensor.hpp"

//namespace AER {
namespace TensorState {

template <class T>
void myswap(T &a, T &b){
	T temp = a;
	a = b;
	b = temp;
}

TensorState::TensorState(size_t size){
  if (DEBUG) cout << "in TS ctor"<<endl;
  size_ = size;
  /*
    q_reg = new Tensor*[size];
    lambda_reg = new Tensor*[size-1];
    entangled_dim_between_qubits = new uint[size];
    for(uint i = 0; i < size; ++i)
    {
    entangled_dim_between_qubits[i] = 1;
    }
    
    complex_t alpha = 1.0f;
    complex_t beta = 0.0f;
    for(uint i = 0; i < size_; i++)
    q_reg[i] = new Tensor(alpha,beta);
    for(uint i = 0; i < size_-1; i++)
    lambda_reg[i] = new Tensor(alpha,beta);
  */
}

TensorState::~TensorState(){}

void TensorState::set_num_qubits(size_t size) {
  size_ = size;
}

void TensorState::initialize()
{
  if ( size_ == 0) {
    cout << "must set size before initialize" <<endl;
    return;
  }
  
  complex_t alpha = 1.0f;
  complex_t beta = 0.0f;
  for(uint i = 0; i < size_; i++)
      q_reg_.push_back(Tensor(alpha,beta));
  for(uint i = 0; i < size_-1; i++)
      lambda_reg_.push_back(rvector_t {1.0}) ;

}

void TensorState::initialize(const TensorState &other){
    if (this != &other) {
      size_ = other.size_;
      q_reg_ = other.q_reg_;
      lambda_reg_ = other.lambda_reg_;
    }     
}

void TensorState::initialize(uint num_qubits, const cvector_t &vecState) {
  cout << "TensorState::initialize not supported yet" <<endl;
}

void TensorState::apply_cnot(uint index_A, uint index_B)
{
	//for MPS
	if(index_A + 1 < index_B)
	{
		apply_swap(index_A,index_B);
		apply_cnot(index_B,index_B);
		apply_swap(index_A,index_B);
		return;
	}
	else if(index_A  > index_B + 1)
	{
		apply_swap(index_A,index_B);
		apply_cnot(index_A,index_A);
		apply_swap(index_A,index_B);
		return;
	}
	bool swapped = false;
	if(index_A >  index_B)
	{
		myswap<uint>(index_A, index_B);
		swapped = true;
	}

	Tensor A = q_reg_[index_A], B = q_reg_[index_B];
	rvector_t left_lambda, right_lambda;
	//There is no lambda in the edges of the MPS
	left_lambda  = (index_A != 0) 	    ? lambda_reg_[index_A-1] : rvector_t {1.0};
	right_lambda = (index_B != size_-1) ? lambda_reg_[index_B  ] : rvector_t {1.0};

	mul_Gamma_by_left_Lambda(left_lambda, q_reg_[index_A]);
	mul_Gamma_by_right_Lambda(q_reg_[index_B], right_lambda);
	Tensor temp = contract(q_reg_[index_A],lambda_reg_[index_A], q_reg_[index_B]);

	if(DEBUG) temp.print();
	temp.apply_cnot(swapped);
	if(DEBUG) temp.print();
	Tensor left_gamma,right_gamma;
	rvector_t lambda;
	//	if(DEBUG) cout << "started new DeCompose" << endl;
	Decompose(temp, left_gamma, lambda, right_gamma);
	//	if(DEBUG) cout << "finished new DeCompose" << endl;
	div_Gamma_by_left_Lambda(left_lambda, left_gamma);
	div_Gamma_by_right_Lambda(right_gamma, right_lambda);
	q_reg_[index_A] = left_gamma;
	lambda_reg_[index_A] = lambda;
	q_reg_[index_B] = right_gamma;
	//	if(DEBUG) cout << "finished apply_cnot" << endl;
}

void TensorState::apply_swap(uint index_A, uint index_B)
{
	//for MPS
	if(index_A + 1 < index_B)
	{
		uint i;
		for(i = index_A; i < index_B; i++)
		{
			apply_swap(i,i+1);
		}
		for(i = index_B-1; i > index_A; i--)
		{
			apply_swap(i,i-1);
		}
		return;
	}
	if(index_A > index_B)
	{
		myswap<uint>(index_A, index_B);
	}
	Tensor A = q_reg_[index_A], B = q_reg_[index_B];
	rvector_t left_lambda, right_lambda;
	//There is no lambda in the edges of the MPS
	left_lambda  = (index_A != 0) 	    ? lambda_reg_[index_A-1] : rvector_t {1.0};
	right_lambda = (index_B != size_-1) ? lambda_reg_[index_B  ] : rvector_t {1.0};

	mul_Gamma_by_left_Lambda(left_lambda, q_reg_[index_A]);
	mul_Gamma_by_right_Lambda(q_reg_[index_B], right_lambda);
	Tensor temp = contract(q_reg_[index_A],lambda_reg_[index_A], q_reg_[index_B]);

	if(DEBUG) temp.print();
	temp.apply_swap();
	if(DEBUG) temp.print();
	Tensor left_gamma,right_gamma;
	rvector_t lambda;
	//	if(DEBUG) cout << "started new DeCompose" << endl;
	Decompose(temp, left_gamma, lambda, right_gamma);
	//	if(DEBUG) cout << "finished new DeCompose" << endl;
	div_Gamma_by_left_Lambda(left_lambda, left_gamma);
	div_Gamma_by_right_Lambda(right_gamma, right_lambda);
	q_reg_[index_A] = left_gamma;
	lambda_reg_[index_A] = lambda;
	q_reg_[index_B] = right_gamma;
	//	if(DEBUG) cout << "finished apply_swap" << endl;
}

void TensorState::apply_cz(uint index_A, uint index_B)
{
	//for MPS
	if(index_A + 1 < index_B)
	{
		apply_swap(index_A,index_B);
		apply_cnot(index_B,index_B);
		apply_swap(index_A,index_B);
		return;
	}
	else if(index_A  > index_B + 1)
	{
		apply_swap(index_A,index_B);
		apply_cnot(index_A,index_A);
		apply_swap(index_A,index_B);
		return;
	}
	if(index_A >  index_B)
	{
		myswap<uint>(index_A, index_B);
	}

	Tensor A = q_reg_[index_A], B = q_reg_[index_B];
	rvector_t left_lambda, right_lambda;
	//There is no lambda in the edges of the MPS
	left_lambda  = (index_A != 0) 	    ? lambda_reg_[index_A-1] : rvector_t {1.0};
	right_lambda = (index_B != size_-1) ? lambda_reg_[index_B  ] : rvector_t {1.0};

	mul_Gamma_by_left_Lambda(left_lambda, q_reg_[index_A]);
	mul_Gamma_by_right_Lambda(q_reg_[index_B], right_lambda);
	Tensor temp = contract(q_reg_[index_A],lambda_reg_[index_A], q_reg_[index_B]);

	if(DEBUG) temp.print();
	temp.apply_cz();
	if(DEBUG) temp.print();
	Tensor left_gamma,right_gamma;
	rvector_t lambda;
	//	if(DEBUG) cout << "started new DeCompose" << endl;
	Decompose(temp, left_gamma, lambda, right_gamma);
	//	if(DEBUG) cout << "finished new DeCompose" << endl;
	div_Gamma_by_left_Lambda(left_lambda, left_gamma);
	div_Gamma_by_right_Lambda(right_gamma, right_lambda);
	q_reg_[index_A] = left_gamma;
	lambda_reg_[index_A] = lambda;
	q_reg_[index_B] = right_gamma;
	//	if(DEBUG) cout << "finished apply_cnot" << endl;
}



double TensorState::Expectation_value(vector<uint> indexes, string matrices)
{
	sort(indexes.begin(), indexes.end());
	Tensor psi = state_vec(indexes.front(), indexes.back());
	uint size = psi.get_dim();
	cmatrix_t rho(size,size);
	for(uint i = 0; i < size; i++)
	{
		for(uint j = 0; j < size; j++)
		{
			rho(i,j) = AER::Utils::sum( AER::Utils::elementwise_multiplication(psi.get_data(i), AER::Utils::conj(psi.get_data(j))) );
		}
	}

	if(DEBUG) {rho.SetOutputStyle(Matrix); cout << "rho =\n" << rho;}

	cmatrix_t M(1), temp;
	M(0,0) = complex_t(1);
	for(char& gate : matrices)
	{
//		cout << temp;
	    if (gate == 'X')
	      temp = AER::Utils::Matrix::X;
	    else if (gate == 'Y')
	      temp = AER::Utils::Matrix::Y;
	    else if (gate == 'Z')
	      temp = AER::Utils::Matrix::Z;
	    else if (gate == 'I')
	      temp = AER::Utils::Matrix::I;
//	    else if (gate == 'H')
//		  temp = AER::Utils::Matrix::H;
//	    else if (gate == 'S')
//		  temp = AER::Utils::Matrix::S;
//	    else if (gate == 'T')
//		  temp = AER::Utils::Matrix::T;
//	    else if (gate == 'Sdg')
//		  temp = AER::Utils::Matrix::Sdg;
//	    else if (gate == 'Tdg')
//		  temp = AER::Utils::Matrix::Tdg;
	    M = AER::Utils::tensor_product(M, temp);
	}

	if(DEBUG) {M.SetOutputStyle(Matrix); cout << "M =\n" << M;}


	// Trace(rho*M). not using methods for efficiency
	complex_t res = 0;
	for (uint i = 0; i < M.GetRows(); i++)
		for (uint j = 0; j < M.GetRows(); j++)
				res += M(i,j)*rho(j,i);
	return real(res);
}

void TensorState::printTN()
{
	for(uint i=0; i<size_; i++)
	{
	  cout << "Gamma [" << i << "] :" << endl;
	  q_reg_[i].print();
	  if(i < size_- 1)
	    {
	      cout << "Lambda [" << i << "] (size = " << lambda_reg_[i].size() << "):" << endl;
	      cout << lambda_reg_[i] << endl;
	    }
	}
	cout << endl;
}

Tensor TensorState::state_vec(uint first_index, uint last_index)
{
	Tensor temp = q_reg_[first_index];
	rvector_t left_lambda, right_lambda;
	left_lambda  = (first_index != 0) 	    ? lambda_reg_[first_index-1] : rvector_t {1.0};
	right_lambda = (last_index != size_-1) ? lambda_reg_[last_index  ] : rvector_t {1.0};

	mul_Gamma_by_left_Lambda(left_lambda, temp);
	for(uint i = first_index+1; i < last_index+1; i++)
		temp = contract(temp, lambda_reg_[i-1], q_reg_[i]);
	// now temp is a tensor of 2^n matrices of size 1X1
	mul_Gamma_by_right_Lambda(temp, right_lambda);
	return temp;
}

//-------------------------------------------------------------------------
} // end namespace TensorState
//-------------------------------------------------------------------------
//} // end namespace AER
//-------------------------------------------------------------------------
