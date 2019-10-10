/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */


#include <bitset>
#include <math.h>

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include <utility>
#include <iostream>

#include "framework/utils.hpp"
#include "framework/matrix.hpp"

#include "matrix_product_state_internal.hpp"
#include "matrix_product_state_tensor.hpp"

namespace AER {
namespace MatrixProductState {

static const cmatrix_t zero_measure = 
      AER::Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}},
	                                 {{0, 0}, {0, 0}}});
static const cmatrix_t one_measure = 
      AER::Utils::make_matrix<complex_t>({{{0, 0}, {0, 0}},
			                 {{0, 0}, {1, 0}}});

//------------------------------------------------------------------------
// local function declarations
//------------------------------------------------------------------------
//--------------------------------------------------------------------------
// Function name: reverse_all_bits
// Description: The ordering of the amplitudes in the statevector in this module is 
//    000, 001, 010, 011, 100, 101, 110, 111.
//    The ordering of the amplitudes in the statevector in Qasm in general is 
//    000, 100, 010, 110, 001, 101, 011, 111.
//    This function converts the statevector from one representation to the other.
// Input: the input statevector and the number of qubits
// Returns: the statevector in reverse order
//----------------------------------------------------------------	
cvector_t reverse_all_bits(const cvector_t& statevector, uint_t num_qubits);
uint_t reverse_bits(uint_t num, uint_t len);
vector<uint_t> calc_new_indexes(vector<uint_t> indexes);

// The following two functions are helper functions used by 
// initialize_from_statevector
cmatrix_t reshape_matrix(cmatrix_t input_matrix);
cmatrix_t mul_matrix_by_lambda(const cmatrix_t &mat,
		               const rvector_t &lambda);

//------------------------------------------------------------------------
// local function implementations
//------------------------------------------------------------------------

uint_t reverse_bits(uint_t num, uint_t len) {
  uint_t sum = 0;
  //  std::assert(num < pow(2, len));
  for (uint_t i=0; i<len; ++i) {
    if ((num & 0x1) == 1) {
      sum += 1ULL << (len-1-i);   // adding pow(2, len-1-i)
    }
    num = num>>1;
    if (num == 0) {
      break;
    }
  }
  return sum;
}

cvector_t reverse_all_bits(const cvector_t& statevector, uint_t num_qubits)
{
  uint_t length = statevector.size();   // length = pow(2, num_qubits_)
  cvector_t output_vector(length);
  #pragma omp parallel for
  for (int_t i = 0; i < static_cast<int_t>(length); i++) {
    output_vector[i] = statevector[reverse_bits(i, num_qubits)];
  }

  return output_vector;
}

vector<uint_t> calc_new_indexes(vector<uint_t> indexes)
{
	uint_t n = indexes.size();
	uint_t avg = round(accumulate( indexes.begin(), indexes.end(), 0.0)/ n );
	vector<uint_t> new_indexes( n );
	std::iota( std::begin( new_indexes ), std::end( new_indexes ), avg-n/2);
	return new_indexes;
}

cmatrix_t mul_matrix_by_lambda(const cmatrix_t &mat,
			       const rvector_t &lambda)
{
  if (lambda == rvector_t {1.0}) return mat;
  cmatrix_t res_mat(mat);
  uint_t num_rows = mat.GetRows(), num_cols = mat.GetColumns();

  #ifdef _WIN32
     #pragma omp parallel for
  #else
     #pragma omp parallel for collapse(2)
  #endif
  for(int_t row = 0; row < static_cast<int_t>(num_rows); row++) {
    for(int_t col = 0; col < static_cast<int_t>(num_cols); col++) {
	res_mat(row, col) = mat(row, col) * lambda[col];
      }
  }
  return res_mat;
}

cmatrix_t reshape_matrix(cmatrix_t input_matrix) {
  vector<cmatrix_t> res(2);
  AER::Utils::split(input_matrix, res[0], res[1], 1);
  cmatrix_t reshaped_matrix = AER::Utils::concatenate(res[0], res[1], 0);
  return reshaped_matrix;
}

//------------------------------------------------------------------------
// implementation of MPS methods
//------------------------------------------------------------------------

void MPS::initialize(uint_t num_qubits)
{
  num_qubits_ = num_qubits;
  q_reg_.clear();
  lambda_reg_.clear();
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
void MPS::apply_cu1(uint_t index_A, uint_t index_B, double lambda)
{
  cmatrix_t u1_matrix = AER::Utils::Matrix::u1(lambda);
  apply_2_qubit_gate(index_A, index_B, cu1, u1_matrix);
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
	case id:
	  break;
	case cu1:
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

void MPS::apply_matrix(const reg_t & qubits, const cmatrix_t &mat) 
{
  switch (qubits.size()) {
  case 1: 
    q_reg_[qubits[0]].apply_matrix(mat);
    break;
  case 2:
    apply_2_qubit_gate(qubits[0], qubits[1], su4, mat);
    break;
  default:
    throw std::invalid_argument("currently support apply_matrix for 1 or 2 qubits only");
  }
}

void MPS::apply_diagonal_matrix(const AER::reg_t &qubits, const cvector_t &vmat) {
  //temporarily support by converting the vector to a full matrix whose diagonal is vmat
  uint_t dim = vmat.size();
  cmatrix_t diag_mat(dim, dim);
  for (uint_t i=0; i<dim; i++) {
    for (uint_t j=0; j<dim; j++){
      diag_mat(i, i) = ( i==j ? vmat[i] : 0.0);
    }
  }
  apply_matrix(qubits, diag_mat);
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
    internalIndexes.push_back(index);

  MPS temp_MPS;
  temp_MPS.initialize(*this);
  vector<uint_t> new_indexes = calc_new_indexes(internalIndexes);
  uint_t avg = new_indexes[new_indexes.size()/2];
  vector<uint_t>::iterator it = lower_bound(internalIndexes.begin(), internalIndexes.end(), avg);
  int mid = std::distance(internalIndexes.begin(), it);
  for(uint_t i = mid; i < internalIndexes.size(); i++)
  {
    temp_MPS.change_position(internalIndexes[i],new_indexes[i]);
  }
  for(int i = mid-1; i >= 0; i--)
  {
    temp_MPS.change_position(internalIndexes[i],new_indexes[i]);
  }
  MPS_Tensor psi = temp_MPS.state_vec(new_indexes.front(), new_indexes.back());
  uint_t size = psi.get_dim();
  cmatrix_t rho(size,size);
  #ifdef _WIN32
     #pragma omp parallel for
  #else
     #pragma omp parallel for collapse(2)
  #endif
  for(int_t i = 0; i < static_cast<int_t>(size); i++) {
    for(int_t j = 0; j < static_cast<int_t>(size); j++) {
      rho(i,j) = AER::Utils::sum( AER::Utils::elementwise_multiplication(psi.get_data(i), AER::Utils::conjugate(psi.get_data(j))) );
    }
  }
  return rho;
}

double MPS::expectation_value(const reg_t &qubits, const string &matrices) const
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

double MPS::expectation_value(const reg_t &qubits, const cmatrix_t &M) const
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

vector<reg_t> MPS::get_matrices_sizes() const
{
	vector<reg_t> result;
	for(uint_t i=0; i<num_qubits_; i++)
	{
		result.push_back(q_reg_[i].get_size());
	}
	return result;
}

MPS_Tensor MPS::state_vec(uint_t first_index, uint_t last_index) const
{
	MPS_Tensor temp = q_reg_[first_index];
	rvector_t left_lambda, right_lambda;
	left_lambda  = (first_index != 0) ? lambda_reg_[first_index-1] : rvector_t {1.0};
	right_lambda = (last_index != num_qubits_-1) ? lambda_reg_[last_index] : rvector_t {1.0};

	temp.mul_Gamma_by_left_Lambda(left_lambda);
	for(uint_t i = first_index+1; i < last_index+1; i++) {
	  temp = MPS_Tensor::contract(temp, lambda_reg_[i-1], q_reg_[i]);
	}
	// now temp is a tensor of 2^n matrices of size 1X1
	temp.mul_Gamma_by_right_Lambda(right_lambda);
	return temp;
}

void MPS::full_state_vector(cvector_t& statevector) const
{
  MPS_Tensor mps_vec = state_vec(0, num_qubits_-1);
  uint_t length = 1ULL << num_qubits_;   // length = pow(2, num_qubits_)
  statevector.resize(length);
  #pragma omp parallel for
  for (int_t i = 0; i < static_cast<int_t>(length); i++) {
    statevector[i] = mps_vec.get_data(reverse_bits(i, num_qubits_))(0,0);
  }
#ifdef DEBUG
  cout << *this;
#endif
}

void MPS::probabilities_vector(rvector_t& probvector) const
{
  MPS_Tensor mps_vec = state_vec(0, num_qubits_-1);
  uint_t length = 1ULL << num_qubits_;   // length = pow(2, num_qubits_)
  probvector.resize(length);
  #pragma omp parallel for
  for (int_t i = 0; i < static_cast<int_t>(length); i++) {
    probvector[i] = std::norm(mps_vec.get_data(reverse_bits(i, num_qubits_))(0,0));
  }
}

reg_t MPS::apply_measure(const reg_t &qubits, 
			 RngEngine &rng) {
  reg_t qubits_to_update;
  reg_t outcome_vector;
  outcome_vector.resize(qubits.size());
  for (uint_t i=0; i<qubits.size(); i++) {
    outcome_vector[i] = apply_measure(qubits[i], rng);
  }
  return outcome_vector;
}

uint_t MPS::apply_measure(uint_t qubit, 
			 RngEngine &rng) {
  reg_t qubits_to_update;
  qubits_to_update.push_back(qubit);

  // step 1 - measure qubit 0 in Z basis
  double exp_val = expectation_value(qubits_to_update, "Z");
  
  // step 2 - compute probability for 0 or 1 result
  double prob0 = (1 + exp_val ) / 2;
  double prob1 = 1 - prob0;

  // step 3 - randomly choose a measurement value for qubit 0
  double rnd = rng.rand(0, 1);
  uint_t measurement;
  cmatrix_t measurement_matrix(4);
  
  if (rnd < prob0) {
    measurement = 0;
    measurement_matrix = zero_measure;
    measurement_matrix = measurement_matrix * (1 / sqrt(prob0));
  } else {
    measurement = 1;
    measurement_matrix = one_measure;
    measurement_matrix = measurement_matrix * (1 / sqrt(prob1));
  }

  apply_matrix(qubits_to_update, measurement_matrix);

  // step 4 - propagate the changes to all qubits to the right
  for (uint_t i=qubit; i<num_qubits_-1; i++) {
    if (lambda_reg_[i].size() == 1) 
      break;   // no need to propagate if no entanglement
    apply_2_qubit_gate(i, i+1, id, cmatrix_t(1));
  }

  // and propagate the changes to all qubits to the left
  for (int_t i=qubit; i>0; i--) {
    if (lambda_reg_[i-1].size() == 1) 
      break;   // no need to propagate if no entanglement
    apply_2_qubit_gate(i-1, i, id, cmatrix_t(1));
  }
    
  return measurement;
}

void MPS::initialize_from_statevector(uint_t num_qubits, const cvector_t state_vector) {
  if (!q_reg_.empty())
    q_reg_.clear();
  if (!lambda_reg_.empty())
    lambda_reg_.clear();
  num_qubits_ = 0;

  cmatrix_t statevector_as_matrix(1, state_vector.size());
  #pragma omp parallel for
  for (int_t i=0; i<(int_t)state_vector.size(); i++) {
    statevector_as_matrix(0, i) = state_vector[i];
  }
  
  // remaining_matrix is the matrix that remains after each iteration
  // It is initialized to the input statevector after reshaping
  cmatrix_t remaining_matrix, reshaped_matrix; 
  cmatrix_t U, V;
  rvector_t S(1.0);
  bool first_iter = true;

  for (uint_t i=0; i<num_qubits-1; i++) {

    // step 1 - prepare matrix for next iteration (except for first iteration):
    //    (i) mul remaining matrix by left lambda 
    //    (ii) dagger and reshape
    if (first_iter) {
      remaining_matrix = statevector_as_matrix;
    } else {
      cmatrix_t temp = mul_matrix_by_lambda(V, S); 
      remaining_matrix = AER::Utils::dagger(temp);
    }
    reshaped_matrix = reshape_matrix(remaining_matrix);

    // step 2 - SVD
    S.clear();
    S.resize(min(reshaped_matrix.GetRows(), reshaped_matrix.GetColumns()));
    csvd_wrapper(reshaped_matrix, U, S, V);
    reduce_zeros(U, S, V);

    // step 3 - update q_reg_ with new gamma and new lambda
    //          increment number of qubits in the MPS structure
    vector<cmatrix_t> left_data = reshape_U_after_SVD(U);
    MPS_Tensor left_gamma(left_data[0], left_data[1]); 
    if (!first_iter)
      left_gamma.div_Gamma_by_left_Lambda(lambda_reg_.back()); 
    q_reg_.push_back(left_gamma);
    lambda_reg_.push_back(S);
    num_qubits_++;

    first_iter = false;
  }

  // step 4 - create the rightmost gamma and update q_reg_
  vector<cmatrix_t> right_data = reshape_V_after_SVD(V);
  
  MPS_Tensor right_gamma(right_data[0], right_data[1]) ;
  q_reg_.push_back(right_gamma);
  num_qubits_++;
}
 

//-------------------------------------------------------------------------
} // end namespace MPS
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
