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

#include "MPS.hpp"
#include "MPS_tensor.hpp"

namespace AER {
namespace TensorNetworkState {

uint reverse_bits(uint num, uint len);
vector<uint> calc_new_indexes(vector<uint> indexes);

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

vector<uint> calc_new_indexes(vector<uint> indexes)
{
	uint n = indexes.size();
	uint avg = round(accumulate( indexes.begin(), indexes.end(), 0.0)/ n );
	vector<uint> new_indexes( n );
	std::iota( std::begin( new_indexes ), std::end( new_indexes ), avg-n/2);
	return new_indexes;
}

void MPS::initialize(uint num_qubits)
{
  num_qubits_ = num_qubits;
  complex_t alpha = 1.0f;
  complex_t beta = 0.0f;
  for(uint i = 0; i < num_qubits_; i++)
      q_reg_.push_back(MPS_Tensor(alpha,beta));
  for(uint i = 0; i < num_qubits_-1; i++)
      lambda_reg_.push_back(rvector_t {1.0}) ;

}

void MPS::initialize(const MPS &other){
    if (this != &other) {
      num_qubits_ = other.num_qubits_;
      q_reg_ = other.q_reg_;
      lambda_reg_ = other.lambda_reg_;
    }     
}

void MPS::apply_cnot(uint index_A, uint index_B)
{
  apply_2_qubit_gate(index_A, index_B, cx);
}

void MPS::apply_cz(uint index_A, uint index_B)
{
  apply_2_qubit_gate(index_A, index_B, cz);
}

void MPS::apply_swap(uint index_A, uint index_B)
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

	MPS_Tensor A = q_reg_[index_A], B = q_reg_[index_B];
	rvector_t left_lambda, right_lambda;
	//There is no lambda in the edges of the MPS
	left_lambda  = (index_A != 0) 	    ? lambda_reg_[index_A-1] : rvector_t {1.0};
	right_lambda = (index_B != num_qubits_-1) ? lambda_reg_[index_B  ] : rvector_t {1.0};

	q_reg_[index_A].mul_Gamma_by_left_Lambda(left_lambda);
	q_reg_[index_B].mul_Gamma_by_right_Lambda(right_lambda);
	MPS_Tensor temp = MPS_Tensor::contract(q_reg_[index_A],lambda_reg_[index_A], q_reg_[index_B]);

	if(DEBUG) temp.print();
	temp.apply_swap();
	if(DEBUG) temp.print();
	MPS_Tensor left_gamma,right_gamma;
	rvector_t lambda;
	MPS_Tensor::Decompose(temp, left_gamma, lambda, right_gamma);
	left_gamma.div_Gamma_by_left_Lambda(left_lambda);
	right_gamma.div_Gamma_by_right_Lambda(right_lambda);
	q_reg_[index_A] = left_gamma;
	lambda_reg_[index_A] = lambda;
	q_reg_[index_B] = right_gamma;
}

void MPS::apply_2_qubit_gate(uint index_A, uint index_B, Gates gate_type)
{
	//for MPS
	if(index_A + 1 < index_B)
	{
		apply_swap(index_A,index_B-1);
		apply_2_qubit_gate(index_B-1,index_B, gate_type);
		apply_swap(index_A,index_B-1);
	  return;
	}
	else if(index_A > index_B + 1)
	{
		apply_swap(index_A-1,index_B);
		apply_2_qubit_gate(index_A,index_A-1, gate_type);
		apply_swap(index_A-1,index_B);
		return;
	}

	bool swapped = false;
	if(index_A >  index_B)
	{
		myswap<uint>(index_A, index_B);
		swapped = true;
	}

	MPS_Tensor A = q_reg_[index_A], B = q_reg_[index_B];
	rvector_t left_lambda, right_lambda;
	//There is no lambda in the edges of the MPS
	left_lambda  = (index_A != 0) 	    ? lambda_reg_[index_A-1] : rvector_t {1.0};
	right_lambda = (index_B != num_qubits_-1) ? lambda_reg_[index_B  ] : rvector_t {1.0};

	q_reg_[index_A].mul_Gamma_by_left_Lambda(left_lambda);
	q_reg_[index_B].mul_Gamma_by_right_Lambda(right_lambda);
	MPS_Tensor temp = MPS_Tensor::contract(q_reg_[index_A], lambda_reg_[index_A], q_reg_[index_B]);

	if(DEBUG) temp.print();
	switch (gate_type) {
	case cx:
	  temp.apply_cnot(swapped);
	  break;
	case cz:
	  temp.apply_cz();
	  break;
        default:
	  throw std::invalid_argument("illegal gate for apply_2_qubit_gate"); 
	}
	if(DEBUG) temp.print();
	MPS_Tensor left_gamma,right_gamma;
	rvector_t lambda;
	MPS_Tensor::Decompose(temp, left_gamma, lambda, right_gamma);
	left_gamma.div_Gamma_by_left_Lambda(left_lambda);
	right_gamma.div_Gamma_by_right_Lambda(right_lambda);
	q_reg_[index_A] = left_gamma;
	lambda_reg_[index_A] = lambda;
	q_reg_[index_B] = right_gamma;
}

void MPS::change_position(uint src, uint dst)
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

cmatrix_t MPS::Density_matrix(const reg_t &qubits) const
{
  // ***** Assuming ascending sorted qubits register *****
  vector<uint> internalIndexes;
  for (uint_t index : qubits)
    internalIndexes.push_back((uint)index);
  //  std::sort(internalIndexes.begin(), internalIndexes.end()); -- Assuming sorted

  MPS temp_TN;
  temp_TN.initialize(*this);
  vector<uint> new_indexes = calc_new_indexes(internalIndexes);
  uint avg = new_indexes[new_indexes.size()/2];
  vector<uint>::iterator it = lower_bound(internalIndexes.begin(), internalIndexes.end(), avg);
  int mid = std::distance(internalIndexes.begin(), it);
  for(uint i = mid; i < internalIndexes.size(); i++)
  {
    temp_TN.change_position(internalIndexes[i],new_indexes[i]);
  }
  for(int i = mid-1; i >= 0; i--)
  {
    temp_TN.change_position(internalIndexes[i],new_indexes[i]);
  }
  MPS_Tensor psi = temp_TN.state_vec(new_indexes.front(), new_indexes.back());
  uint size = psi.get_dim();
  cmatrix_t rho(size,size);
  for(uint i = 0; i < size; i++) {
    for(uint j = 0; j < size; j++) {
      rho(i,j) = AER::Utils::sum( AER::Utils::elementwise_multiplication(psi.get_data(i), AER::Utils::conj(psi.get_data(j))) );
    }
  }
  return rho;
}

double MPS::Expectation_value(const reg_t &qubits, const string &matrices) const
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

double MPS::Expectation_value(const reg_t &qubits, const cmatrix_t &M) const
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

void MPS::printTN()
{
	for(uint i=0; i<num_qubits_; i++)
	{
	  cout << "Gamma [" << i << "] :" << endl;
	  q_reg_[i].print();
	  if(i < num_qubits_- 1)
	    {
	      cout << "Lambda [" << i << "] (size = " << lambda_reg_[i].size() << "):" << endl;
	      cout << lambda_reg_[i] << endl;
	    }
	}
	cout << endl;
}

MPS_Tensor MPS::state_vec(uint first_index, uint last_index) const
{
	MPS_Tensor temp = q_reg_[first_index];
	rvector_t left_lambda, right_lambda;
	left_lambda  = (first_index != 0) ? lambda_reg_[first_index-1] : rvector_t {1.0};
	right_lambda = (last_index != num_qubits_-1) ? lambda_reg_[last_index] : rvector_t {1.0};

	temp.mul_Gamma_by_left_Lambda(left_lambda);
	for(uint i = first_index+1; i < last_index+1; i++)
	  temp = MPS_Tensor::contract(temp, lambda_reg_[i-1], q_reg_[i]);
	// now temp is a tensor of 2^n matrices of size 1X1
	temp.mul_Gamma_by_right_Lambda(right_lambda);
	return temp;
}
void MPS::full_state_vector(cvector_t& statevector) const
{
  MPS_Tensor mps_vec = state_vec(0, num_qubits_-1);
  uint length = pow(2, num_qubits_);
  for (uint i = 0; i < length; i++) {
    statevector.push_back(mps_vec.get_data(reverse_bits(i, num_qubits_))(0,0));
  }
}

//-------------------------------------------------------------------------
} // end namespace MPS
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
