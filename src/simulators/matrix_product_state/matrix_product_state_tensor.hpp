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


#ifndef _tensor_tensor_hpp_
#define _tensor_tensor_hpp_

#define SQR_HALF sqrt(0.5)
#define NUMBER_OF_PRINTED_DIGITS 3

#include <cstdio>
#include <iostream>
#include <complex>
#include <vector>
#include <math.h>
#include <string.h>
#include <exception>

#include "svd.hpp"
#include "svd.cpp"
#include "framework/matrix.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace MatrixProductState {

// Data types
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;
using cmatrix_t = matrix<complex_t>;

//============================================================================
// MPS_Tensor class
//============================================================================
// The MPS_Tensor class is used to represent the data structure of a single
// Gamma-tensor (corresponding to a single qubit) in the MPS algorithm.
// In the stable state, each MPS_Tensor consists of two matrices -
// the matrix with index 0 (data_[0]) represents the amplitude of |0> and
// the matrix with index 1 (data_[1]) represents the amplitude of |1>.
// When applying a two-qubit gate, we temporarily create an MPS_Tensor of four matrices,
// corresponding to |00>, |01>, |10>, |11>.
// These are later decomposed back to the stable state of two matrices MPS_Tensor (per qubit).
//----------------------------------------------------------------

class MPS_Tensor
{
public:
  // Constructors of MPS_Tensor class
  MPS_Tensor(){}
  explicit MPS_Tensor(complex_t& alpha, complex_t& beta){
    //    matrix<complex_t> A = matrix<complex_t>(1), B = matrix<complex_t>(1);
    cmatrix_t A = cmatrix_t(1), B = cmatrix_t(1);
    A(0,0) = alpha;
    B(0,0) = beta;
    data_.push_back(A);
    data_.push_back(B);
  }
  MPS_Tensor(const MPS_Tensor& rhs){
    data_ = rhs.data_;
  }

  MPS_Tensor(const cmatrix_t& data0, const cmatrix_t& data1){
    if (!data_.empty())
      data_.clear();
    data_.push_back(data0);
    data_.push_back(data1);
  }

  MPS_Tensor(const std::vector<cmatrix_t> &data){
    if (!data_.empty())
      data_.clear();
    for (uint_t i=0; i<data.size(); i++)
      data_.push_back(data[i]);
  }

  // Destructor
  virtual ~MPS_Tensor(){}

  // Assignment operator
  MPS_Tensor& operator=(const MPS_Tensor& rhs){
    if (this != &rhs){
      data_.clear();
      data_ = rhs.data_;
    }
    return *this;
  }
  virtual std::ostream& print(std::ostream& out) const;
  reg_t get_size() const;
  cvector_t get_data(uint_t a1, uint_t a2) const;
  cmatrix_t get_data(uint_t i) const {
    return data_[i];
  }
  const std::vector<cmatrix_t> get_data() const {
    return data_;
  }
  void insert_data(uint_t a1, uint_t a2, cvector_t data);

  //------------------------------------------------------------------
  // function name: get_dim
  // Description: Get the dimension of the physical index of the tensor
  // Parameters: none.
  // Returns: uint_t of the dimension of the physical index of the tensor.
  //------------------------------------------------------------------
  uint_t get_dim() const {
    return data_.size();
  }
  void apply_pauli(char gate);
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
  void apply_matrix(const cmatrix_t &mat, bool swapped=false);
  void apply_cnot(bool swapped = false);
  void apply_swap();
  void apply_cz();
  void apply_ccx(uint_t target_qubit);
  void mul_Gamma_by_left_Lambda(const rvector_t &Lambda);
  void mul_Gamma_by_right_Lambda(const rvector_t &Lambda);
  void div_Gamma_by_left_Lambda(const rvector_t &Lambda);
  void div_Gamma_by_right_Lambda(const rvector_t &Lambda);
  static MPS_Tensor contract(const MPS_Tensor &left_gamma, const rvector_t &lambda, const MPS_Tensor &right_gamma, bool mul_by_lambda);
  static void Decompose(MPS_Tensor &temp, MPS_Tensor &left_gamma, rvector_t &lambda, MPS_Tensor &right_gamma);
  static void reshape_for_3_qubits_before_SVD(const std::vector<cmatrix_t> data, MPS_Tensor &reshaped_tensor);
static void contract_2_dimensions(const MPS_Tensor &left_gamma, 
				  const MPS_Tensor &right_gamma,
				  cmatrix_t &result);

private:
  void mul_Gamma_by_Lambda(const rvector_t &Lambda,
			   bool right, /* or left */
			   bool mul    /* or div */);

  std::vector<cmatrix_t> data_;
};

//=========================================================================
// Implementation
//=========================================================================

//---------------------------------------------------------------
// function name: print
// Description: Prints the Tensor. All the submatrices are aligned by rows.
//-------------------------------------------------------------
std::ostream& MPS_Tensor::print(std::ostream& out) const {
    complex_t value;

    out << "[" << std::endl;
    if (data_.size() > 0){
        //Printing the matrices row by row (i.e., not matrix by matrix)

        for (uint_t row = 0; row < data_[0].GetRows(); row++){
            for(uint_t i = 0; i < data_.size(); i++)
            {
                out << " |";

                for (uint_t column = 0; column < data_[0].GetColumns(); column++){

                    value = data_[i](row, column);

                    out << "(" << std::fixed << std::setprecision(NUMBER_OF_PRINTED_DIGITS) << value.real() << ", ";
                    out << std::fixed  << std::setprecision(NUMBER_OF_PRINTED_DIGITS) << value.imag() << ")," ;
                }
                out << "| ,";
            }
            out << std::endl;
        }
    }
    out << "]" << std::endl;

    return out;
}

//**************************************************************
// function name: get_size
// Description: get size of the matrices of the tensor.
// Parameters: none.
// Returns: reg_t of size 2, for rows and columns.
//**************************************************************
reg_t MPS_Tensor::get_size() const
{
	reg_t result;
	result.push_back(data_[0].GetRows());
	result.push_back(data_[0].GetColumns());
	return result;
}

//----------------------------------------------------------------
// function name: get_data
// Description: Get the data in some axis of the MPS_Tensor
// 1.	Parameters: uint_t a1, uint_t a2 - indexes of data in matrix
// 		Returns: cvector_t of data in (a1,a2) in all matrices
// 2.	Parameters: uint_t i - index of a matrix in the MPS_Tensor
// 		Returns: cmatrix_t of the data
//---------------------------------------------------------------
cvector_t MPS_Tensor::get_data(uint_t a1, uint_t a2) const
{
  cvector_t Res;
  for(uint_t i = 0; i < data_.size(); i++)
    Res.push_back(data_[i](a1,a2));
  return Res;
}

//---------------------------------------------------------------
// function name: insert_data
// Description: Insert data to some axis of the MPS_Tensor
// Parameters: uint_t a1, uint_t a2 - indexes of data in matrix
// Parameters: cvector_t data - data to insert.
// Returns: void.
//---------------------------------------------------------------
void MPS_Tensor::insert_data(uint_t a1, uint_t a2, cvector_t data)
{
  for(uint_t i = 0; i < data_.size(); i++)
    data_[i](a1,a2) = data[i];
}

void MPS_Tensor::apply_pauli(char gate) {
  switch (gate) {
  case 'X':
     apply_x();
     break;
  case 'Y':
     apply_y();
     break;
  case 'Z':
     apply_z();
     break;
  case 'I':
     break;
  default:
     throw std::invalid_argument("illegal gate for contract_with_self"); 
  }

}
//---------------------------------------------------------------
// function name: apply_x,y,z,...
// Description: Apply some gate on the tensor. tensor must represent
//		the number of qubits the gate expect
// Parameters: none.
// Returns: none.
//---------------------------------------------------------------
void MPS_Tensor::apply_x()
{
  std::swap(data_[0],data_[1]);
}
  void MPS_Tensor::apply_y()
  {
    data_[0] = data_[0] * complex_t(0, 1);
    data_[1] = data_[1] * complex_t(0, -1);
    std::swap(data_[0],data_[1]);
  }

void MPS_Tensor::apply_z()
{
  data_[1] = data_[1] * (-1.0);
}

void MPS_Tensor::apply_s()
{
  data_[1] = data_[1] * complex_t(0, 1);
}

void MPS_Tensor::apply_sdg()
{
  data_[1] = data_[1] * complex_t(0, -1);
}

void MPS_Tensor::apply_t()
{
  data_[1] = data_[1] * complex_t(SQR_HALF, SQR_HALF);
}

void MPS_Tensor::apply_tdg()
{
  data_[1] = data_[1] * complex_t(SQR_HALF, -SQR_HALF);
}

void MPS_Tensor::apply_matrix(const cmatrix_t &mat, bool swapped)
{
  if (swapped)
    swap(data_[1], data_[2]);

  MPS_Tensor new_tensor;
  // initialize by multiplying first column of mat by data_[0]
  for (uint_t i=0; i<mat.GetRows(); i++) 
    new_tensor.data_.push_back(mat(i, 0) * data_[0]);

  // add all other columns 
  for (uint_t i=0; i<mat.GetRows(); i++) {
    for (uint_t j=1; j<mat.GetColumns(); j++) {
      new_tensor.data_[i] += mat(i, j) * data_[j];
    }
  }
  *this = new_tensor;
   
  if (swapped)
    swap(data_[1], data_[2]);
}

void MPS_Tensor::apply_cnot(bool swapped)
{
  if(!swapped)
    std::swap(data_[2],data_[3]);
  else
    std::swap(data_[1],data_[3]);
}

void MPS_Tensor::apply_swap()
{
  std::swap(data_[1],data_[2]);
}

void MPS_Tensor::apply_cz()
{
  data_[3] = data_[3] * (-1.0);
}

void MPS_Tensor::apply_ccx(uint_t target_qubit)
{
  switch (target_qubit) {
  case 0:
    swap(data_[3], data_[7]);
    break;
  case 1:
    swap(data_[5], data_[7]);
    break;
  case 2:
    swap(data_[6], data_[7]);
    break;
  default:
   throw std::invalid_argument("Target qubit for cxx must be 0, 1, or 2"); 
  }
}
//-------------------------------------------------------------------------
// The following functions mul/div Gamma by Lambda are used to keep the MPS in the
// canonical form.
//
// Before applying a 2-qubit gate, we must contract these qubits to relevant Gamma tensors.
// To maintain the canonical form, we must consider the Lambda tensors from
// the sides of the Gamma tensors. This is what the multiply functions do. After the
// decomposition of the result of the gate, we need to divide back by what we
// multiplied before. This is what the division functions do.
//-------------------------------------------------------------------------
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
  uint_t rows = data_[0].GetRows(), cols = data_[0].GetColumns();
  for(uint_t i = 0; i < data_.size(); i++)
    for(uint_t a1 = 0; a1 < rows; a1++)
      for(uint_t a2 = 0; a2 < cols; a2++) {
	uint_t factor = right ? a2 : a1;
	if (mul) {
	  data_[i](a1,a2) *= Lambda[factor];
	} else{
	  data_[i](a1,a2) /= Lambda[factor];
	}
      }
}

//---------------------------------------------------------------
// function name: contract
// Description: Contract two Gamma tensors and the Lambda between
// 		them. Usually used before 2-qubits gate.
// Parameters: MPS_Tensor &left_gamma, &right_gamma , rvector_t &lambda -
// 	       tensors to contract.
// Returns: The result tensor of the contract
//---------------------------------------------------------------
MPS_Tensor MPS_Tensor::contract(const MPS_Tensor &left_gamma, 
				const rvector_t &lambda, 
				const MPS_Tensor &right_gamma,
				bool mul_by_lambda=true)
{
  MPS_Tensor Res;
  MPS_Tensor new_left = left_gamma;
  if (mul_by_lambda) {
    new_left.mul_Gamma_by_right_Lambda(lambda);
  }
  for(uint_t i = 0; i < new_left.data_.size(); i++)
    for(uint_t j = 0; j < right_gamma.data_.size(); j++) {

      Res.data_.push_back(new_left.data_[i] * right_gamma.data_[j]);
    }
  return Res;
}

//---------------------------------------------------------------
// Function name: contract_2_dimensions
// Description: Contract two Gamma tensors across 2 dimensions: left_columns/right_rows and
//                                                              left_size/right_size
// Parameters: MPS_Tensor &left_gamma, &right_gamma - the tensors to contract.
// Returns: The result matrix of the contract
// Assumptions:
//   1. We assume lambda was already multiplied into the gammas before this function
//   2. We assume the tensors (t1 and t2) are of the form:
//      t1   
//      o--a1--o
//     ||
//      o--a2--o
//      t2
//  There is a double bond between tensor 1 and 2, and each of them has an additional bond of 
//  dimension a1 and a2 respectively. The result matrix will be of size a2 x a1
//---------------------------------------------------------------
void MPS_Tensor::contract_2_dimensions(const MPS_Tensor &left_gamma, 
			               const MPS_Tensor &right_gamma,
				       cmatrix_t &result)
{
  int_t left_rows = left_gamma.data_[0].GetRows();
  int_t left_columns = left_gamma.data_[0].GetColumns();
  int_t left_size = left_gamma.get_dim();
  int_t right_rows = right_gamma.data_[0].GetRows();
  int_t right_columns = right_gamma.data_[0].GetColumns();
  int_t right_size = right_gamma.get_dim();

  // left_columns/right_rows and left_size/right_size
  if (left_columns != right_rows)   
    throw std::runtime_error("left_columns != right_rows");

  if (left_size != right_size)
    throw std::runtime_error("left_size != right_size");
  result.resize(left_rows, right_columns);

  #ifdef _WIN32
     #pragma omp parallel for
  #else
     #pragma omp parallel for collapse(2)
  #endif
  for (int_t l_row=0; l_row<left_rows; l_row++)
    for (int_t r_col=0; r_col<right_columns; r_col++)
      result(l_row, r_col) = 0;
  
  #ifdef _WIN32
     #pragma omp parallel for
  #else
     #pragma omp parallel for collapse(2)
  #endif
  for (int_t l_row=0; l_row<left_rows; l_row++)
    for (int_t r_col=0; r_col<right_columns; r_col++) {

      for (int_t size=0; size<left_size; size++)
	  for (int_t index=0; index<left_columns ; index++) {
 	      result(l_row, r_col) += left_gamma.data_[size](l_row, index) *
		                      right_gamma.data_[size](index, r_col);      

	    }
	  }
}
//---------------------------------------------------------------
// function name: Decompose
// Description: Decompose a tensor into two Gamma tensors and the Lambda between
// 				them. Usually used after applying a 2-qubit gate.
// Parameters: MPS_Tensor &temp - the tensor to decompose.
//			   MPS_Tensor &left_gamma, &right_gamma , rvector_t &lambda -
// 			   tensors for the result.
// Returns: none.
//---------------------------------------------------------------
void MPS_Tensor::Decompose(MPS_Tensor &temp, MPS_Tensor &left_gamma, rvector_t &lambda, MPS_Tensor &right_gamma)
{
  matrix<complex_t> C;
  C = reshape_before_SVD(temp.data_);
  matrix<complex_t> U,V;
  rvector_t S(std::min(C.GetRows(), C.GetColumns()));

#ifdef DEBUG
  std::cout << "Input matrix before SVD =" << std::endl << C ;
#endif

  csvd_wrapper(C, U, S, V);
  reduce_zeros(U, S, V);

#ifdef DEBUG
  std::cout << "matrices after SVD:" <<std::endl;
  std::cout << "U = " << std::endl << U ;
  std::cout << "S = " << std::endl;
  for (uint_t i = 0; i != S.size(); ++i)
    std::cout << S[i] << " , ";
  std::cout << std::endl;
  std::cout << "V* = " << std::endl << V ;
#endif

  left_gamma.data_  = reshape_U_after_SVD(U);
  lambda            = S;
  right_gamma.data_ = reshape_V_after_SVD(V);
}

  void MPS_Tensor::reshape_for_3_qubits_before_SVD(const std::vector<cmatrix_t> data, 
				     MPS_Tensor &reshaped_tensor)
{
// Turns 4 matrices A0,A1,A2,A3,A4,A5,A6,A7 to big matrix:
//  A0 A1 A2 A3
//  A4 A5 A6 A7

  cmatrix_t temp0_1 = AER::Utils::concatenate(data[0], data[1], 1),
            temp2_3 = AER::Utils::concatenate(data[2], data[3], 1),
            temp4_5 = AER::Utils::concatenate(data[4], data[5], 1),
            temp6_7 = AER::Utils::concatenate(data[6], data[7], 1);
  std::cout << temp0_1 ;
  std::vector<cmatrix_t> new_data_vector;
  new_data_vector.push_back(temp0_1);
  new_data_vector.push_back(temp2_3);
  new_data_vector.push_back(temp4_5);
  new_data_vector.push_back(temp6_7);
  reshaped_tensor = MPS_Tensor(new_data_vector);
}

//-------------------------------------------------------------------------
} // end namespace MatrixProductState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
