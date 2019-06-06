/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */


#include <bitset>
#include <math.h>

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include <utility>
#include <iostream>

#include "framework/utils.hpp"

#include "matrix_product_state.hpp"
#include "matrix_product_state_tensor.hpp"

namespace AER {
namespace TensorNetworkState {

uint_t reverse_bits(uint_t num, uint_t len);
vector<uint_t> calc_new_indexes(vector<uint_t> indexes);

uint_t reverse_bits(uint_t num, uint_t len) {
  uint_t sum = 0;
  //  std::assert(num < pow(2, len));
  for (uint_t i=0; i<len; ++i) {
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

vector<uint_t> calc_new_indexes(vector<uint_t> indexes)
{
	uint_t n = indexes.size();
	uint_t avg = round(accumulate( indexes.begin(), indexes.end(), 0.0)/ n );
	vector<uint_t> new_indexes( n );
	std::iota( std::begin( new_indexes ), std::end( new_indexes ), avg-n/2);
	return new_indexes;
}

void MPS::initialize(uint_t num_qubits)
{
  num_qubits_ = num_qubits;
  complex_t alpha = 1.0f;
  complex_t beta = 0.0f;
  for(uint_t i = 0; i < num_qubits_-1; i++) {
      q_reg_.push_back(MPS_Tensor(alpha,beta));
      lambda_reg_.push_back(rvector_t {1.0});
  }
  // need to add one more Gamma tensor, because above loop only initialized up to n-1 
  q_reg_.push_back(MPS_Tensor(alpha,beta));
}

void MPS::initialize(const MPS &other){
    if (this != &other) {
      num_qubits_ = other.num_qubits_;
      q_reg_ = other.q_reg_;
      lambda_reg_ = other.lambda_reg_;
    }     
}

void MPS::apply_h(uint_t index) 
{
    cmatrix_t h_matrix = AER::Utils::Matrix::H;
    q_reg_[index].apply_matrix(h_matrix);
}

void MPS::apply_u1(uint_t index, double lambda)
{
  cmatrix_t u1_matrix = AER::Utils::Matrix::u1(lambda);
  q_reg_[index].apply_matrix(u1_matrix);
}

void MPS::apply_u2(uint_t index, double phi, double lambda)
{
  cmatrix_t u2_matrix = AER::Utils::Matrix::u2(phi, lambda);
  q_reg_[index].apply_matrix(u2_matrix);
}

void MPS::apply_u3(uint_t index, double theta, double phi, double lambda)
{
  cmatrix_t u3_matrix = AER::Utils::Matrix::u3(theta, phi, lambda);
  q_reg_[index].apply_matrix(u3_matrix);
}

void MPS::apply_cnot(uint_t index_A, uint_t index_B)
{
  apply_2_qubit_gate(index_A, index_B, cx, cmatrix_t(1));
}

void MPS::apply_cz(uint_t index_A, uint_t index_B)
{
  apply_2_qubit_gate(index_A, index_B, cz, cmatrix_t(1));
}
void MPS::apply_cu(uint_t index_A, uint_t index_B, cmatrix_t mat)
{
  apply_2_qubit_gate(index_A, index_B, cu, mat);
}
void MPS::apply_su4(uint_t index_A, uint_t index_B, cmatrix_t mat)
{
  apply_2_qubit_gate(index_A, index_B, su4, mat);
}

void MPS::apply_swap(uint_t index_A, uint_t index_B)
{
	if(index_A > index_B)
	{
	  std::swap(index_A, index_B);
	}
	//for MPS
	if(index_A + 1 < index_B)
	{
		uint_t i;
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

	temp.apply_swap();
	MPS_Tensor left_gamma,right_gamma;
	rvector_t lambda;
	MPS_Tensor::Decompose(temp, left_gamma, lambda, right_gamma);
	left_gamma.div_Gamma_by_left_Lambda(left_lambda);
	right_gamma.div_Gamma_by_right_Lambda(right_lambda);
	q_reg_[index_A] = left_gamma;
	lambda_reg_[index_A] = lambda;
	q_reg_[index_B] = right_gamma;
}

//-------------------------------------------------------------------------
// MPS::apply_2_qubit_gate - outline of the algorithm
// 1. Swap qubits A and B until they are consecutive
// 2. Contract MPS_Tensor[A] and MPS_Tensor[B], yielding a temporary four-matrix MPS_Tensor 
//    that represents the entangled states of A and B.
// 3. Apply the gate
// 4. Decompose the temporary MPS_Tensor (using SVD) into U*S*V, where U and V are matrices
//    and S is a diagonal matrix
// 5. U is split by rows to yield two MPS_Tensors representing qubit A (in reshape_U_after_SVD), 
//    V is split by columns to yield two MPS_Tensors representing qubit B (in reshape_V_after_SVD),
//    the diagonal of S becomes the Lambda-vector in between A and B.
//-------------------------------------------------------------------------

void MPS::apply_2_qubit_gate(uint_t index_A, uint_t index_B, Gates gate_type, cmatrix_t mat)
{
	//for MPS
	if(index_A + 1 < index_B)
	{
		apply_swap(index_A,index_B-1);
		apply_2_qubit_gate(index_B-1,index_B, gate_type, mat);
		apply_swap(index_A,index_B-1);
	  return;
	}
	else if(index_A > index_B + 1)
	{
		apply_swap(index_A-1,index_B);
		apply_2_qubit_gate(index_A,index_A-1, gate_type, mat);
		apply_swap(index_A-1,index_B);
		return;
	}

	bool swapped = false;
	if(index_A >  index_B)
	{
	  std::swap(index_A, index_B);
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

	switch (gate_type) {
	case cx:
	  temp.apply_cnot(swapped);
	  break;
	case cz:
	  temp.apply_cz();
	  break;
	case cu:
	{
	  cmatrix_t Zeros = AER::Utils::Matrix::I-AER::Utils::Matrix::I;
	  cmatrix_t temp1 = AER::Utils::concatenate(AER::Utils::Matrix::I, Zeros , 1),
			  	temp2 = AER::Utils::concatenate(Zeros, mat, 1);
	  cmatrix_t cu = AER::Utils::concatenate(temp1, temp2 ,0) ;
	  temp.apply_matrix(cu);
	  break;
	}
	case su4:
	  temp.apply_matrix(mat);
	  break;

	default:
	  throw std::invalid_argument("illegal gate for apply_2_qubit_gate"); 
	}
	MPS_Tensor left_gamma,right_gamma;
	rvector_t lambda;
	MPS_Tensor::Decompose(temp, left_gamma, lambda, right_gamma);
	left_gamma.div_Gamma_by_left_Lambda(left_lambda);
	right_gamma.div_Gamma_by_right_Lambda(right_lambda);
	q_reg_[index_A] = left_gamma;
	lambda_reg_[index_A] = lambda;
	q_reg_[index_B] = right_gamma;
}

void MPS::change_position(uint_t src, uint_t dst)
{
	if(src == dst)
		return;
	else if(src < dst)
		for(uint_t i = src; i < dst; i++)
			apply_swap(i,i+1);
	else
		for(uint_t i = src; i > dst; i--)
			apply_swap(i,i-1);
}

cmatrix_t MPS::density_matrix(const reg_t &qubits) const
{
  // ***** Assuming ascending sorted qubits register *****
  vector<uint_t> internalIndexes;
  for (uint_t index : qubits)
    internalIndexes.push_back((uint_t)index);
  //  std::sort(internalIndexes.begin(), internalIndexes.end()); -- Assuming sorted

  MPS temp_TN;
  temp_TN.initialize(*this);
  vector<uint_t> new_indexes = calc_new_indexes(internalIndexes);
  uint_t avg = new_indexes[new_indexes.size()/2];
  vector<uint_t>::iterator it = lower_bound(internalIndexes.begin(), internalIndexes.end(), avg);
  int mid = std::distance(internalIndexes.begin(), it);
  for(uint_t i = mid; i < internalIndexes.size(); i++)
  {
    temp_TN.change_position(internalIndexes[i],new_indexes[i]);
  }
  for(int i = mid-1; i >= 0; i--)
  {
    temp_TN.change_position(internalIndexes[i],new_indexes[i]);
  }
  MPS_Tensor psi = temp_TN.state_vec(new_indexes.front(), new_indexes.back());
  uint_t size = psi.get_dim();
  cmatrix_t rho(size,size);
  for(uint_t i = 0; i < size; i++) {
    for(uint_t j = 0; j < size; j++) {
      rho(i,j) = AER::Utils::sum( AER::Utils::elementwise_multiplication(psi.get_data(i), AER::Utils::conj(psi.get_data(j))) );
    }
  }
  return rho;
}

double MPS::Expectation_value(const reg_t &qubits, const string &matrices) const
{
  // ***** Assuming ascending sorted qubits register *****
  cmatrix_t rho = density_matrix(qubits);
  string matrices_reverse = matrices;
  reverse(matrices_reverse.begin(), matrices_reverse.end());
  cmatrix_t M(1), temp;
  M(0,0) = complex_t(1);
  for(const char& gate : matrices_reverse)
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
  for (uint_t i = 0; i < M.GetRows(); i++)
    for (uint_t j = 0; j < M.GetRows(); j++)
      res += M(i,j)*rho(j,i);
  return real(res);
}

double MPS::Expectation_value(const reg_t &qubits, const cmatrix_t &M) const
{
  // ***** Assuming ascending sorted qubits register *****
  cmatrix_t rho = density_matrix(qubits);

  // Trace(rho*M). not using methods for efficiency
  complex_t res = 0;
  for (uint_t i = 0; i < M.GetRows(); i++)
    for (uint_t j = 0; j < M.GetRows(); j++)
      res += M(i,j)*rho(j,i);
  return real(res);
}

ostream& MPS::print(ostream& out) const
{
	for(uint_t i=0; i<num_qubits_; i++)
	{
	  out << "Gamma [" << i << "] :" << endl;
	  q_reg_[i].print(out);
	  if(i < num_qubits_- 1)
	    {
	      out << "Lambda [" << i << "] (size = " << lambda_reg_[i].size() << "):" << endl;
	      out << lambda_reg_[i] << endl;
	    }
	}
	out << endl;
	return out;
}

MPS_Tensor MPS::state_vec(uint_t first_index, uint_t last_index) const
{
	MPS_Tensor temp = q_reg_[first_index];
	rvector_t left_lambda, right_lambda;
	left_lambda  = (first_index != 0) ? lambda_reg_[first_index-1] : rvector_t {1.0};
	right_lambda = (last_index != num_qubits_-1) ? lambda_reg_[last_index] : rvector_t {1.0};

	temp.mul_Gamma_by_left_Lambda(left_lambda);
	for(uint_t i = first_index+1; i < last_index+1; i++)
	  temp = MPS_Tensor::contract(temp, lambda_reg_[i-1], q_reg_[i]);
	// now temp is a tensor of 2^n matrices of size 1X1
	temp.mul_Gamma_by_right_Lambda(right_lambda);
	return temp;
}

void MPS::full_state_vector(cvector_t& statevector) const
{
  MPS_Tensor mps_vec = state_vec(0, num_qubits_-1);
  uint_t length = pow(2, num_qubits_);
  for (uint_t i = 0; i < length; i++) {
    statevector.push_back(mps_vec.get_data(reverse_bits(i, num_qubits_))(0,0));
  }
#ifdef DEBUG
  cout << *this;
#endif
}

void MPS::probabilities_vector(rvector_t& probvector) const
{
  MPS_Tensor mps_vec = state_vec(0, num_qubits_-1);
  uint_t length = pow(2, num_qubits_);
  complex_t data = 0;
  for (uint_t i = 0; i < length; i++) {
    data = mps_vec.get_data(reverse_bits(i, num_qubits_))(0,0);
    probvector.push_back(std::norm(data));
  }
}

// for now supporting only the fully vector (all qubits)
reg_t MPS::sample_measure(std::vector<double> &rands) 
{
  rvector_t probvector;
  probabilities_vector(probvector);
  const int_t SHOTS = rands.size();
  reg_t samples = {0};
  uint_t length = probvector.size();
  samples.assign(SHOTS, 0);
     for (int_t i = 0; i < SHOTS; ++i) {
        double rand = rands[i];
        double p = .0;
        uint_t sample;
        for (sample = 0; sample < length; ++sample) {
          p += probvector[sample];
          if (rand < p)
            break; 
        }
        samples[i] = sample;
      }
     return samples;
}

//-------------------------------------------------------------------------
} // end namespace MPS
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
