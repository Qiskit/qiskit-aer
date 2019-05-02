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

namespace AER {
namespace TensorState {

uint reverse_bits(uint num, uint len);

template <class T>
void myswap(T &a, T &b){
	T temp = a;
	a = b;
	b = temp;
}

uint reverse_bits(uint num, uint len) {
  uint sum = 0;
  //  std::assert(num < pow(2, len));
  for (uint i=0; i<len; ++i) {
    if ((num & 0x1) == 1) {
      sum += pow(2, len-1-i);
    }
    num = num>>1;
    if (num == 0) {
      break;
    }
  }
  return sum;
}
  /*
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
  */
void TensorState::initialize(uint num_qubits)
{
  if ( num_qubits == 0) {
    cout << "size must be larger than 0" <<endl;
    return;
  }
  size_ = num_qubits;
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

/*
void TensorState::initialize(uint num_qubits, const cvector_t &vecState) {
  cout << "TensorState::initialize not supported yet" <<endl;
}
  */
void TensorState::apply_cnot(uint index_A, uint index_B)
{
	//for MPS
	if(index_A + 1 < index_B)
	{
		apply_swap(index_A,index_B-1); //bring first qubit next to second qubit (recursive)
		apply_cnot(index_B-1,index_B); //apply gate
		apply_swap(index_A,index_B-1); //bring first qubit back (recursive)
	  return;
	}
	else if(index_A  > index_B + 1)
	{
		apply_swap(index_A-1,index_B);
		apply_cnot(index_A,index_A-1);
		apply_swap(index_A-1,index_B);
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

	q_reg_[index_A].mul_Gamma_by_left_Lambda(left_lambda);
	q_reg_[index_B].mul_Gamma_by_right_Lambda(right_lambda);
	Tensor temp = Tensor::contract(q_reg_[index_A],lambda_reg_[index_A], q_reg_[index_B]);

	if(DEBUG) temp.print();
	temp.apply_cnot(swapped);
	if(DEBUG) temp.print();
	Tensor left_gamma,right_gamma;
	rvector_t lambda;
	Tensor::Decompose(temp, left_gamma, lambda, right_gamma);
	left_gamma.div_Gamma_by_left_Lambda(left_lambda);
	right_gamma.div_Gamma_by_right_Lambda(right_lambda);
	q_reg_[index_A] = left_gamma;
	lambda_reg_[index_A] = lambda;
	q_reg_[index_B] = right_gamma;
}

void TensorState::apply_swap(uint index_A, uint index_B)
{
	if(index_A > index_B)
	{
		myswap<uint>(index_A, index_B);
	}
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

	Tensor A = q_reg_[index_A], B = q_reg_[index_B];
	rvector_t left_lambda, right_lambda;
	//There is no lambda in the edges of the MPS
	left_lambda  = (index_A != 0) 	    ? lambda_reg_[index_A-1] : rvector_t {1.0};
	right_lambda = (index_B != size_-1) ? lambda_reg_[index_B  ] : rvector_t {1.0};

	q_reg_[index_A].mul_Gamma_by_left_Lambda(left_lambda);
	q_reg_[index_B].mul_Gamma_by_right_Lambda(right_lambda);
	Tensor temp = Tensor::contract(q_reg_[index_A],lambda_reg_[index_A], q_reg_[index_B]);

	if(DEBUG) temp.print();
	temp.apply_swap();
	if(DEBUG) temp.print();
	Tensor left_gamma,right_gamma;
	rvector_t lambda;
	//	if(DEBUG) cout << "started new DeCompose" << endl;
	Tensor::Decompose(temp, left_gamma, lambda, right_gamma);
	//	if(DEBUG) cout << "finished new DeCompose" << endl;
	left_gamma.div_Gamma_by_left_Lambda(left_lambda);
	right_gamma.div_Gamma_by_right_Lambda(right_lambda);
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
		apply_swap(index_A,index_B-1);
		apply_cz(index_B-1,index_B);
		apply_swap(index_A,index_B-1);
	  return;
	}
	else if(index_A  > index_B + 1)
	{
		apply_swap(index_A-1,index_B);
		apply_cz(index_A,index_A-1);
		apply_swap(index_A-1,index_B);
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

	q_reg_[index_A].mul_Gamma_by_left_Lambda(left_lambda);
	q_reg_[index_B].mul_Gamma_by_right_Lambda(right_lambda);
	Tensor temp = Tensor::contract(q_reg_[index_A], lambda_reg_[index_A], q_reg_[index_B]);

	if(DEBUG) temp.print();
	temp.apply_cz();
	if(DEBUG) temp.print();
	Tensor left_gamma,right_gamma;
	rvector_t lambda;
	Tensor::Decompose(temp, left_gamma, lambda, right_gamma);
	left_gamma.div_Gamma_by_left_Lambda(left_lambda);
	right_gamma.div_Gamma_by_right_Lambda(right_lambda);
	q_reg_[index_A] = left_gamma;
	lambda_reg_[index_A] = lambda;
	q_reg_[index_B] = right_gamma;
}

void TensorState::change_position(uint src, uint dst)
{
	if(src == dst)
		return;
	else if(src < dst)
		for(uint i = src; i < dst; i++)
			apply_swap(i,i+1);
	else
		for(uint i = src; i > dst; i--)
			apply_swap(i,i-1);
}

vector<uint> calc_new_indexes(vector<uint> indexes)
{
	uint n = indexes.size();
	uint avg = round(accumulate( indexes.begin(), indexes.end(), 0.0)/ n );
	vector<uint> new_indexes( n );
	std::iota( std::begin( new_indexes ), std::end( new_indexes ), avg-n/2);
	return new_indexes;
}

cmatrix_t TensorState::Density_matrix(const reg_t &qubits)
{
  // ***** Assuming ascending sorted qubits register *****
//  vector<uint> internalIndexes = qubits;
	vector<uint> internalIndexes;
    for (uint_t index : qubits)
      internalIndexes.push_back((uint)index);

  //  std::sort(internalIndexes.begin(), internalIndexes.end()); -- Assuming sorted

  TensorState temp_TN;
  temp_TN.initialize(*this);
  vector<uint> new_indexes = calc_new_indexes(internalIndexes);
  uint avg = new_indexes[new_indexes.size()/2];
  vector<uint>::iterator it = lower_bound(internalIndexes.begin(), internalIndexes.end(), avg);
  int mid = std::distance(internalIndexes.begin(), it);
  for(int i = mid; i < internalIndexes.size(); i++)
  {
    temp_TN.change_position(internalIndexes[i],new_indexes[i]);
  }
  for(int i = mid-1; i >= 0; i--)
  {
    temp_TN.change_position(internalIndexes[i],new_indexes[i]);
  }
  Tensor psi = temp_TN.state_vec(new_indexes.front(), new_indexes.back());
  uint size = psi.get_dim();
  cmatrix_t rho(size,size);
  for(uint i = 0; i < size; i++) {
    for(uint j = 0; j < size; j++) {
      rho(i,j) = AER::Utils::sum( AER::Utils::elementwise_multiplication(psi.get_data(i), AER::Utils::conj(psi.get_data(j))) );
    }
  }
  return rho;
}

double TensorState::Expectation_value(const reg_t &qubits, const string &matrices)
{
  // ***** Assuming ascending sorted qubits register *****
  cmatrix_t rho = Density_matrix(qubits);

  cmatrix_t M(1), temp;
  M(0,0) = complex_t(1);
  for(const char& gate : matrices)
  {
    if (gate == 'X')
	  temp = AER::Utils::Matrix::X;
    else if (gate == 'Y')
	  temp = AER::Utils::Matrix::Y;
    else if (gate == 'Z')
	  temp = AER::Utils::Matrix::Z;
    else if (gate == 'I')
	  temp = AER::Utils::Matrix::I;
    M = AER::Utils::tensor_product(M, temp);
  }
  // Trace(rho*M). not using methods for efficiency
  complex_t res = 0;
  for (uint i = 0; i < M.GetRows(); i++)
    for (uint j = 0; j < M.GetRows(); j++)
      res += M(i,j)*rho(j,i);
  return real(res);
}

double TensorState::Expectation_value(const reg_t &qubits, const cmatrix_t &M)
{
  // ***** Assuming ascending sorted qubits register *****
  cmatrix_t rho = Density_matrix(qubits);

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

Tensor TensorState::state_vec(uint first_index, uint last_index) const
{
	Tensor temp = q_reg_[first_index];
	rvector_t left_lambda, right_lambda;
	left_lambda  = (first_index != 0) ? lambda_reg_[first_index-1] : rvector_t {1.0};
	right_lambda = (last_index != size_-1) ? lambda_reg_[last_index] : rvector_t {1.0};

	temp.mul_Gamma_by_left_Lambda(left_lambda);
	for(uint i = first_index+1; i < last_index+1; i++)
	  temp = Tensor::contract(temp, lambda_reg_[i-1], q_reg_[i]);
	// now temp is a tensor of 2^n matrices of size 1X1
	temp.mul_Gamma_by_right_Lambda(right_lambda);
	return temp;
}

//-------------------------------------------------------------------------
} // end namespace TensorState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
