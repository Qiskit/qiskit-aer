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

//------------------------------------------------------------------------
// Function name: squeeze_qubits
// Description: Takes a list of qubits, and squeezes them into a list of the same size,
//     that begins at 0, and where all qubits are consecutive. Note that relative 
//     order between qubits is preserved.
//     Example: [8, 4, 6, 0, 9] -> [3, 1, 2, 0, 4]
// Input: original_qubits 
// Returns: squeezed_qubits
//
//------------------------------------------------------------------------
  void squeeze_qubits(const reg_t &original_qubits, reg_t &squeezed_qubits);

//------------------------------------------------------------------------
// Function name: reorder_all_qubits
// Description: The ordering of the amplitudes in the statevector in this module is 
//    [n, (n-1),.., 2, 1, 0], i.e., msb is leftmost and lsb is rightmost.
//    Sometimes, we need to provide a different ordering of the amplitudes, such as 
//    in snapshot_probabilities. For example, instead of [2, 1, 0] the user requests 
//    the probabilities of [1, 0, 2].
//    Note that the ordering in the qubits vector is the same as that of the mps solver,
//    i.e., qubits are numbered from left to right, e.g., 210
// Input: orig_probvector - the ordered vector of probabilities
//        qubits - a list containing the new ordering
// Returns: new_probvector - the vector in the new ordering
//    e.g., 011->101 (for the ordering [1, 0, 2]
//
//------------------------------------------------------------------------
template <class T>
void reorder_all_qubits(const std::vector<T>& orig_probvector, 
			reg_t qubits, 
			std::vector<T>& new_probvector);
uint_t reorder_qubits(const reg_t qubits, uint_t index);

//--------------------------------------------------------------------------
// Function name: reverse_all_bits
// Description: The ordering of the amplitudes in the statevector in this module is 
//    000, 001, 010, 011, 100, 101, 110, 111.
//    The ordering of the amplitudes in the statevector in Qasm in general is 
//    000, 100, 010, 110, 001, 101, 011, 111.
//    This function converts the statevector from one representation to the other.
//    This is a special case of reorder_qubits
// Input: the input statevector and the number of qubits
// Output: the statevector in reverse order
//----------------------------------------------------------------	
template <class T>
std::vector<T> reverse_all_bits(const std::vector<T>& statevector, uint_t num_qubits);
uint_t reverse_bits(uint_t num, uint_t len);
std::vector<uint_t> calc_new_indices(const reg_t &indices);

// The following two functions are helper functions used by 
// initialize_from_statevector
cmatrix_t reshape_matrix(cmatrix_t input_matrix);
cmatrix_t mul_matrix_by_lambda(const cmatrix_t &mat,
		               const rvector_t &lambda);


//------------------------------------------------------------------------
// local function implementations
//------------------------------------------------------------------------
void squeeze_qubits(const reg_t &original_qubits, reg_t &squeezed_qubits) {
  std::vector<uint_t> sorted_qubits;
  for (uint_t index : original_qubits) {
    sorted_qubits.push_back(index);
  }
  sort(sorted_qubits.begin(), sorted_qubits.end());
  for (uint_t i=0; i<original_qubits.size(); i++) {
    for (uint_t j=0; j<sorted_qubits.size(); j++) {
      if (original_qubits[i] == sorted_qubits[j]) {
	squeezed_qubits[i] = j;
	break;
      } 
    }    
  }
}

template <class T>
void reorder_all_qubits(const std::vector<T>& orig_probvector, 
			reg_t qubits,
			std::vector<T>& new_probvector) {
  uint_t new_index;
  uint_t length = 1ULL << qubits.size();   // length = pow(2, num_qubits)
  // if qubits are [k0, k1,...,kn], move them to [0, 1, .. , n], but preserve relative
  // ordering
  reg_t squeezed_qubits(qubits.size());
  squeeze_qubits(qubits, squeezed_qubits);

  for (uint_t i=0; i < length; i++) {
    new_index = reorder_qubits(squeezed_qubits, i);
    new_probvector[new_index] = orig_probvector[i];
  } 
}

uint_t reorder_qubits(const reg_t qubits, uint_t index) {
  uint_t new_index = 0;

  int_t current_pos = 0, current_val = 0, new_pos = 0, shift =0;
  uint_t num_qubits = qubits.size();
  for (uint_t i=0; i<num_qubits; i++) {
    current_pos = num_qubits-1-qubits[i];
    current_val = 0x1 << current_pos;
    new_pos = num_qubits-1-i;
    shift = new_pos - current_pos;
    if (index & current_val) {
      if (shift > 0) {
	new_index += current_val << shift;
      } else if (shift < 0) {
	new_index += current_val >> -shift; 
      } else {
	new_index += current_val;
      }
      
    }
  }
  return new_index;
}

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

template <class T>
std::vector<T> reverse_all_bits(const std::vector<T>& statevector, uint_t num_qubits)
{
  uint_t length = statevector.size();   // length = pow(2, num_qubits_)
  std::vector<T> output_vector(length);
  #pragma omp parallel for
  for (int_t i = 0; i < static_cast<int_t>(length); i++) {
    output_vector[i] = statevector[reverse_bits(i, num_qubits)];
  }

  return output_vector;
}

std::vector<uint_t> calc_new_indices(const reg_t &indices) {
  // assumes indices vector is sorted
  uint_t n = indices.size();
  uint_t mid_index = indices[(n-1)/2];
  uint_t first = mid_index - (n-1)/2;
  std::vector<uint_t> new_indices(n);
  std::iota( std::begin( new_indices ), std::end( new_indices ), first);
  return new_indices;
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
  std::vector<cmatrix_t> res(2);
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

void MPS::apply_ccx(const reg_t &qubits)
{
  apply_3_qubit_gate(qubits, mcx, cmatrix_t(1));
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
	MPS_Tensor temp = MPS_Tensor::contract(q_reg_[index_A], lambda_reg_[index_A], q_reg_[index_B]);

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
void MPS::apply_2_qubit_gate(uint_t index_A, uint_t index_B, Gates gate_type, const cmatrix_t &mat)
{
  // We first move the two qubits to be in consecutive positions
  // If index_B > index_A, we move the qubit at index_B to index_A+1
  // If index_B < index_A, we move the qubit at index_B to index_A-1, and then
  // swap between the qubits
  uint_t A = index_A;
  bool swapped = false, greater = false, smaller = false;

  if (index_B > index_A+1) {
    greater = true;
    change_position(index_B, index_A+1);  // Move B to be right after A
  } else if (index_A > 0 && index_B < index_A-1) {
    smaller = true;
    change_position(index_B, index_A-1);  // Move B to be right before A
  }
  if (index_B < index_A) {
    A = index_A - 1;
    swapped = true;
  }
  // After we moved the qubits as necessary, 
  // the operation is always between qubits A and A+1
  rvector_t left_lambda, right_lambda;
  //There is no lambda on the edges of the MPS
  left_lambda  = (A != 0) 	    ? lambda_reg_[A-1] : rvector_t {1.0};
  right_lambda = (A+1 != num_qubits_-1) ? lambda_reg_[A+1] : rvector_t {1.0};
  
  q_reg_[A].mul_Gamma_by_left_Lambda(left_lambda);
  q_reg_[A+1].mul_Gamma_by_right_Lambda(right_lambda);
  MPS_Tensor temp = MPS_Tensor::contract(q_reg_[A], lambda_reg_[A], q_reg_[A+1]);
  
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
    // We reverse the order of the qubits, according to the Qiskit convention.
    // Effectively, this reverses swap for 2-qubit gates
    temp.apply_matrix(mat, !swapped);
    break;
    
  default:
    throw std::invalid_argument("illegal gate for apply_2_qubit_gate"); 
  }
  MPS_Tensor left_gamma,right_gamma;
  rvector_t lambda;
  MPS_Tensor::Decompose(temp, left_gamma, lambda, right_gamma);
  left_gamma.div_Gamma_by_left_Lambda(left_lambda);
  right_gamma.div_Gamma_by_right_Lambda(right_lambda);
  q_reg_[A] = left_gamma;
  lambda_reg_[A] = lambda;
  q_reg_[A+1] = right_gamma;

  if (greater) {
    change_position(index_A+1, index_B);  // Move B back to its original position
  } else if (smaller) {
    change_position(index_A-1, index_B);
  }
}

void MPS::apply_3_qubit_gate(const reg_t &qubits,
			     Gates gate_type, const cmatrix_t &mat)
{
  if (qubits.size() != 3) {
    std::stringstream ss;
    ss << "error: apply_3_qubit gate must receive 3 qubits";
    throw std::runtime_error(ss.str());
  }
  bool ordered = true;
  reg_t new_qubits(qubits.size());
  reg_t sorted_qubits(qubits.size());

  centralize_and_sort_qubits(qubits, sorted_qubits, new_qubits, ordered);

  // The controlled (or target) qubit, is qubit[2]. Since in new_qubits the qubits are sorted,
  // the relative position of the controlled qubit will be 0, 1, or 2 depending on
  // where qubit[2] was moved to in new_qubits
  uint_t target=0;
  if (qubits[2] > qubits[0] && qubits[2] > qubits[1])
    target = 2;
  else if (qubits[2] < qubits[0] && qubits[2] < qubits[1])
    target = 0;
  else
    target = 1;

  // extract the tensor containing only the 3 qubits on which we apply the gate
  uint_t first = new_qubits.front();
  MPS_Tensor sub_tensor(state_vec_as_MPS(first, first+2));

  // apply the gate to sub_tensor
  switch (gate_type) {
  case mcx:
       sub_tensor.apply_ccx(target);
    break;

  default:
    throw std::invalid_argument("illegal gate for apply_3_qubit_gate"); 
  }

  // state_mat is a matrix containing the flattened representation of the sub-tensor 
  // into a single matrix. Note that sub_tensor will contain 8 matrices for 3-qubit
  // gates. state_mat will be the concatenation of them all.
  cmatrix_t state_mat = sub_tensor.get_data(0);
  for (uint_t i=1; i<sub_tensor.get_data().size(); i++)
    state_mat = AER::Utils::concatenate(state_mat, sub_tensor.get_data(i), 1) ;

  // We convert the matrix back into a 3-qubit MPS structure
  MPS sub_MPS;
  sub_MPS.initialize_from_matrix(qubits.size(), state_mat);

  // copy the 3-qubit MPS back to the corresponding positions in the original MPS
  for (uint_t i=0; i<sub_MPS.num_qubits(); i++) {
    q_reg_[first+i] = sub_MPS.q_reg_[i];
  }
  lambda_reg_[first] = sub_MPS.lambda_reg_[0];
  lambda_reg_[first+1] = sub_MPS.lambda_reg_[1];
  if (first > 0)
    q_reg_[first].div_Gamma_by_left_Lambda(lambda_reg_[first-1]);
  if (first+2 < num_qubits_-1)
    q_reg_[first+2].div_Gamma_by_right_Lambda(lambda_reg_[first+2]);

  // This is the reverse of centralize_qubits which we did at the beginning
  if (!ordered)
    move_qubits_to_original_location(first, qubits, sorted_qubits);
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

void MPS::centralize_qubits(const reg_t &qubits,
			    reg_t &new_indices, bool & ordered) {
  reg_t sorted_indices;
  centralize_and_sort_qubits(qubits, sorted_indices, new_indices, ordered);
}

void MPS::centralize_and_sort_qubits(const reg_t &qubits, reg_t &sorted_indices,
			             reg_t &new_indices, bool & ordered) {
  sorted_indices = qubits;
  uint_t num_qubits = qubits.size();

  ordered = false;
  if (num_qubits == 1) {
    new_indices = qubits;
    ordered = true;
    return;
  }

  for (uint_t index=0; index < num_qubits-1; index++) {
    if (qubits[index] > qubits[index+1]){
      ordered = false;
      break;
    }
  }
  if (!ordered)
      sort(sorted_indices.begin(), sorted_indices.end());

  new_indices = calc_new_indices(sorted_indices);
  // We wish to minimize the number of swaps. Therefore we center the 
  // new indices around the median
  uint_t mid_index = (num_qubits-1)/2;
  
  for(uint_t i = mid_index; i < sorted_indices.size(); i++) {
    change_position(sorted_indices[i], new_indices[i]);
  }
  for(int i = mid_index-1; i >= 0; i--) {
    change_position(sorted_indices[i], new_indices[i]);
  }
}

void MPS::move_qubits_to_original_location(uint_t first, const reg_t &original_qubits,
					   const reg_t &sorted_qubits) {
  uint_t num_qubits = original_qubits.size();
  uint_t mid_index = (num_qubits-1)/2;

  for(uint_t i = 0; i < mid_index; i++)
  {
    change_position(first+i, sorted_qubits[i]);
  }

  for(uint_t i = num_qubits-1; i > mid_index; i--)
  {
    change_position(first+i, sorted_qubits[i]);
  }
  // note that the qubit at mid_index does not need to move, because we didn't move it
  // during centralize_qubits
}

void MPS::change_position(uint_t src, uint_t dst) {
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
  MPS temp_MPS;
  temp_MPS.initialize(*this);
  reg_t new_qubits;
  bool ordered = true;
  
  temp_MPS.centralize_qubits(qubits, new_qubits, ordered);

  MPS_Tensor psi = temp_MPS.state_vec_as_MPS(new_qubits.front(), new_qubits.back());
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

rvector_t MPS::trace_of_density_matrix(const reg_t &qubits) const
{
  MPS temp_MPS;
  temp_MPS.initialize(*this);
  bool ordered = true;
  reg_t new_qubits;
  temp_MPS.centralize_qubits(qubits, new_qubits, ordered);

  MPS_Tensor psi = temp_MPS.state_vec_as_MPS(new_qubits.front(), new_qubits.back());

  uint_t size = psi.get_dim();
  rvector_t trace_rho(size);

  for(int_t i = 0; i < static_cast<int_t>(size); i++) {
    trace_rho[i] = real(AER::Utils::sum( AER::Utils::elementwise_multiplication(psi.get_data(i), AER::Utils::conjugate(psi.get_data(i))) ));
  }
  return trace_rho;
}

void MPS::MPS_with_new_indices(const reg_t &qubits, 
			      MPS& temp_MPS, 
			      uint_t &front, uint_t &back) const {
  // ***** Assuming ascending sorted qubits register *****
  std::vector<uint_t> internalIndices;
  for (uint_t index : qubits)
    internalIndices.push_back(index);

  temp_MPS.initialize(*this);
  std::vector<uint_t> new_indices = calc_new_indices(internalIndices);
  uint_t avg = new_indices[new_indices.size()/2];
  std::vector<uint_t>::iterator it = lower_bound(internalIndices.begin(), internalIndices.end(), avg);
  int mid = std::distance(internalIndices.begin(), it);
  for(uint_t i = mid; i < internalIndices.size(); i++)
  {
    temp_MPS.change_position(internalIndices[i], new_indices[i]);
  }
  for(int i = mid-1; i >= 0; i--)
  {
    temp_MPS.change_position(internalIndices[i], new_indices[i]);
  }
  front = internalIndices.front();
  back = internalIndices.back();
}

double MPS::expectation_value(const reg_t &qubits, const cmatrix_t &M) const
{
  cmatrix_t rho = density_matrix(qubits);

  // Trace(rho*M). not using methods for efficiency
  complex_t res = 0;
  for (uint_t i = 0; i < M.GetRows(); i++)
    for (uint_t j = 0; j < M.GetRows(); j++)
      res += M(i,j)*rho(j,i);
  return real(res);
}

//---------------------------------------------------------------
// Function: expectation_value_pauli
// Algorithm: For more details, see "The density-matrix renormalization group in the age of matrix 
//            product states" by Ulrich Schollwock.
// For the illustration, assume computing the expectation 
// value on qubits numbered q0, q1, q2, q3. There may be additional qubits
// before q0 or after q3 
// Initial state: 
//      q0     q1     q2     q3                               
//   -a0-o--a1--o--a2--o--a3--o---  
//       |      |      |      |  
//   -a0-o--a1--o--a2--o--a3--o---                                     
//                       
//                                 
// We can actually think of this as       q0  q1  q2  q3
//                                       --o---o---o---o--
//                                      |  |   |   |   |  |
//                                       --o---o---o---o--
// because expectation value on the left and right are 1. 

// After Step 4:
//       q1     q2     q3
//     a1/o--a2--o--a3--o--
//      o |      |      |  |
//     a1\o--a2--o--a3--o-- 
//
// After step 8:
//       q1     q2     q3
//        o--a2--o--a3--o--
//     a1||i     |      |  |
//        o--a2--o--a3--o-- 
//
// After step 9:
//              q2     q3
//            a2/o--a3--o--
//             o |      |  |
//            a2\o--a3--o-- 
//---------------------------------------------------------------

complex_t MPS::expectation_value_pauli(const reg_t &qubits, const std::string &matrices) const
{
  MPS temp_MPS;
  uint_t first_index = 0, last_index = 0;
  MPS_with_new_indices(qubits, temp_MPS, first_index, last_index);

  // Preliminary step - reverse the order of the matrices because 
  // they are ordered in reverse to that of the qubits (in the interface)
  std::string reversed_matrices = matrices;
  reverse(reversed_matrices.begin(), reversed_matrices.end());
  char gate = reversed_matrices[0];

  // Step 1 - multiply tensor of q0 by its left lambda
  MPS_Tensor left_tensor = temp_MPS.q_reg_[first_index];
  if (first_index > 0) {
    left_tensor.mul_Gamma_by_left_Lambda(temp_MPS.lambda_reg_[first_index-1]);
  }

  // The last gamma must be multiplied also by its right lambda.
  // Here we handle the special case that we are calculating exp val 
  // on a single qubit
  // we need to mul every gamma by its right lambda
  if (first_index==last_index && first_index < num_qubits_-1) {
      left_tensor.mul_Gamma_by_right_Lambda(temp_MPS.lambda_reg_[first_index]);
  }

  // Step 2 - prepare the dagger of left_tensor
  MPS_Tensor left_tensor_dagger(AER::Utils::dagger(left_tensor.get_data(0)), 
				AER::Utils::dagger(left_tensor.get_data(1)));
  
  // Step 3 - Apply the gate to q0
  left_tensor.apply_pauli(gate);

  // Step 4 - contract Gamma0' with Gamma0 over dimensions a0 and i
  // Before contraction, Gamma0' has size a1 x a0 x i, Gamma0 has size i x a0 x a1
  // result = left_contract is a matrix of size a1 x a1
  cmatrix_t final_contract;
  MPS_Tensor::contract_2_dimensions(left_tensor_dagger, left_tensor, 
				    final_contract);

  for (uint_t qubit_num=first_index+1; qubit_num<=last_index; qubit_num++) {

    // Step 5 - multiply next Gamma by its left lambda (same as Step 1)
    // next gamma has dimensions a0 x a1 x i 
    MPS_Tensor next_gamma = temp_MPS.q_reg_[qubit_num];
    next_gamma.mul_Gamma_by_left_Lambda(temp_MPS.lambda_reg_[qubit_num-1]);

    // Last qubit must be multiplied by rightmost lambda
    if (qubit_num==last_index && qubit_num < num_qubits_-1)
      next_gamma.mul_Gamma_by_right_Lambda(temp_MPS.lambda_reg_[qubit_num]);

    // Step 6 - prepare the dagger of the next gamma (same as Step 2)
    // next_gamma_dagger has dimensions a1' x a0' x i
    MPS_Tensor next_gamma_dagger(AER::Utils::dagger(next_gamma.get_data(0)), 
				 AER::Utils::dagger(next_gamma.get_data(1)));
    
    // Step 7 - apply gate (same as Step 3)
    gate = reversed_matrices[qubit_num - first_index];
    next_gamma.apply_pauli(gate);
    
    // Step 8 - contract final_contract from previous stage with next gamma over a1
    // final_contract has dimensions a1 x a1, Gamma1 has dimensions a1 x a2 x i (where i=2)
    // result is a tensor of size a1 x a2 x i
    MPS_Tensor next_contract(final_contract * next_gamma.get_data(0), 
			     final_contract * next_gamma.get_data(1));

    // Step 9 - contract next_contract (a1 x a2 x i) 
    // with next_gamma_dagger (i x a2 x a1) (same as Step 4)
    // here we need to contract across two dimensions: a1 and i
    // result is a matrix of size a2 x a2
    MPS_Tensor::contract_2_dimensions(next_gamma_dagger, next_contract, 
				      final_contract);      
  }
 
  // Step 10 - contract over final matrix of size aN x aN
  // We need to contract the final matrix with itself
  // Compute this by taking the trace of final_contract
  complex_t result = AER::Utils::trace(final_contract);

  return result;
}

std::ostream& MPS::print(std::ostream& out) const
{
	for(uint_t i=0; i<num_qubits_; i++)
	{
	  out << "Gamma [" << i << "] :" << std::endl;
	  q_reg_[i].print(out);
	  if(i < num_qubits_- 1)
	    {
	      out << "Lambda [" << i << "] (size = " << lambda_reg_[i].size() << "):" << std::endl;
	      out << lambda_reg_[i] << std::endl;
	    }
	}
	out << std::endl;
	return out;
}

std::vector<reg_t> MPS::get_matrices_sizes() const
{
  std::vector<reg_t> result;
  for(uint_t i=0; i<num_qubits_; i++)
    {
      result.push_back(q_reg_[i].get_size());
    }
  return result;
}

MPS_Tensor MPS::state_vec_as_MPS(const reg_t &qubits) const {
  MPS temp_MPS;
  temp_MPS.initialize(*this);
  bool ordered = true;
  reg_t new_qubits;
  temp_MPS.centralize_qubits(qubits, new_qubits, ordered);
  return temp_MPS.state_vec_as_MPS(new_qubits.front(), new_qubits.back());
}

MPS_Tensor MPS::state_vec_as_MPS(uint_t first_index, uint_t last_index) const
{
	MPS_Tensor temp = q_reg_[first_index];
	rvector_t left_lambda, right_lambda;
	left_lambda  = (first_index != 0) ? lambda_reg_[first_index-1] : rvector_t {1.0};
	right_lambda = (last_index != num_qubits_-1) ? lambda_reg_[last_index] : rvector_t {1.0};

	temp.mul_Gamma_by_left_Lambda(left_lambda);
	// special case of a single qubit
	if (first_index == last_index) {
	  temp.mul_Gamma_by_right_Lambda(right_lambda);
	  return temp;
	}
	  
	for(uint_t i = first_index+1; i < last_index+1; i++) {
	  temp = MPS_Tensor::contract(temp, lambda_reg_[i-1], q_reg_[i]);
	}
	// now temp is a tensor of 2^n matrices of size 1X1
	temp.mul_Gamma_by_right_Lambda(right_lambda);

	return temp;
}

void MPS::full_state_vector(cvector_t& statevector) const
{
  MPS_Tensor mps_vec = state_vec_as_MPS(0, num_qubits_-1);
  uint_t length = 1ULL << num_qubits_;   // length = pow(2, num_qubits_)
  statevector.resize(length);
  #pragma omp parallel for
  for (int_t i = 0; i < static_cast<int_t>(length); i++) {
    statevector[i] = mps_vec.get_data(reverse_bits(i, num_qubits_))(0,0);
  }
#ifdef DEBUG
  std::cout << *this;
#endif
}

void MPS::get_probabilities_vector(rvector_t& probvector, 
			       const reg_t &qubits) const
{
  cvector_t state_vec;
  uint_t num_qubits = qubits.size();
  uint_t length = 1ULL << num_qubits;   // length = pow(2, num_qubits)
  probvector.resize(length);

  // compute the probability vector assuming the qubits are in ascending order
  rvector_t ordered_probvector = trace_of_density_matrix(qubits);

  // reorder the probabilities according to the specification in 'qubits'
  rvector_t temp_probvector(ordered_probvector.size()); 
  reorder_all_qubits(ordered_probvector, qubits, temp_probvector);

  // reverse to be consistent with qasm ordering
  probvector = reverse_all_bits(temp_probvector, num_qubits);
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
  double exp_val = real(expectation_value_pauli(qubits_to_update, "Z"));
  
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

void MPS::initialize_from_matrix(uint_t num_qubits, const cmatrix_t mat) {
  if (!q_reg_.empty())
    q_reg_.clear();
  if (!lambda_reg_.empty())
    lambda_reg_.clear();
  num_qubits_ = 0;

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
      remaining_matrix = mat;
    } else {
      cmatrix_t temp = mul_matrix_by_lambda(V, S); 
      remaining_matrix = AER::Utils::dagger(temp);
    }
    reshaped_matrix = reshape_matrix(remaining_matrix);

    // step 2 - SVD
    S.clear();
    S.resize(std::min(reshaped_matrix.GetRows(), reshaped_matrix.GetColumns()));
    csvd_wrapper(reshaped_matrix, U, S, V);
    reduce_zeros(U, S, V);

    // step 3 - update q_reg_ with new gamma and new lambda
    //          increment number of qubits in the MPS structure
    std::vector<cmatrix_t> left_data = reshape_U_after_SVD(U);
    MPS_Tensor left_gamma(left_data[0], left_data[1]); 
    if (!first_iter)
      left_gamma.div_Gamma_by_left_Lambda(lambda_reg_.back()); 
    q_reg_.push_back(left_gamma);
    lambda_reg_.push_back(S);
    num_qubits_++;

    first_iter = false;
  }

  // step 4 - create the rightmost gamma and update q_reg_
  std::vector<cmatrix_t> right_data = reshape_V_after_SVD(V);
  
  MPS_Tensor right_gamma(right_data[0], right_data[1]) ;
  q_reg_.push_back(right_gamma);
  num_qubits_++;
}
 

//-------------------------------------------------------------------------
} // end namespace MPS
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
