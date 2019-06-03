/**
 * Copyright 2019, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_tensor_state_hpp_
#define _aer_tensor_state_hpp_

#include "framework/json.hpp"
#include "framework/types.hpp"
#include "framework/utils.hpp"
#include "framework/operations.hpp"
#include "matrix_product_state_tensor.hpp"

namespace AER {
namespace TensorNetworkState {

// Allowed gates enum class
enum Gates {
  id, h, x, y, z, s, sdg, t, tdg, u1, u2, u3, // single qubit
  cx, cz, cu, swap, su4 // two qubit
};

class MPS{
public:
  MPS(uint_t num_qubits = 0):
    num_qubits_(num_qubits) {}
  ~MPS() {}

  //**************************************************************
  // function name: initialize
  // Description: Initialize the tensor network with some state.
  // 1.	Parameters: none. Initializes all qubits to |0>.
  // 2.	Parameters: const MPS &other - Copy another tensor network
  // TODO:
  // 3.	Parameters: uint_t num_qubits, const cvector_t &vecState -
  //  				Initializes qubits with a statevector.
  // Returns: none.
  //**************************************************************
  virtual void initialize(uint_t num_qubits=0);
  void initialize(const MPS &other);
  //void initialize(uint_t num_qubits, const cvector_t &vecState);

  //**************************************************************
  // function name: num_qubits
    // Description: Get the number of qubits in the tensor network
    // Parameters: none.
    // Returns: none.
  //**************************************************************
  uint_t num_qubits() const{return num_qubits_;}
  
  //**************************************************************
    // function name: set_num_qubits
    // Description: Set the number of qubits in the tensor network
    // Parameters: size_t num_qubits - number of qubits to set.
    // Returns: none.
  //**************************************************************
  void set_num_qubits(uint_t num_qubits) {
    num_qubits_ = num_qubits;
  }
  bool empty() const {
    return(num_qubits_ == 0);
  }
  

  //**************************************************************
    // function name: apply_x,y,z,...
    // Description: Apply a gate on some qubits by their indexes.
    // Parameters: uint_t index of the qubit/qubits.
    // Returns: none.
  //**************************************************************
  void apply_h(uint_t index);
  void apply_x(uint_t index){q_reg_[index].apply_x();}
  void apply_y(uint_t index){q_reg_[index].apply_y();}
  void apply_z(uint_t index){q_reg_[index].apply_z();}
  void apply_s(uint_t index){q_reg_[index].apply_s();}
  void apply_sdg(uint_t index){q_reg_[index].apply_sdg();}
  void apply_t(uint_t index){q_reg_[index].apply_t();}
  void apply_tdg(uint_t index){q_reg_[index].apply_tdg();}
  void apply_u1(uint_t index, double lambda);
  void apply_u2(uint_t index, double phi, double lambda);
  void apply_u3(uint_t index, double theta, double phi, double lambda);
  //void old_apply_swap(uint_t index_A, uint_t index_B);
  void apply_cnot(uint_t index_A, uint_t index_B);
  void apply_swap(uint_t index_A, uint_t index_B);
  void apply_cz(uint_t index_A, uint_t index_B);
  void apply_cu(uint_t index_A, uint_t index_B, cmatrix_t mat);
  void apply_su4(uint_t index_A, uint_t index_B, cmatrix_t mat);
  void apply_2_qubit_gate(uint_t index_A, uint_t index_B, Gates gate_type, cmatrix_t mat);


  void apply_matrix(const AER::reg_t &qubits, const cvector_t &vmat) 
                      {cout << "apply_matrix not supported yet" <<endl;}
  void apply_diagonal_matrix(const AER::reg_t &qubits, const cvector_t &vmat) 
                      {cout << "apply_diagonalmatrix not supported yet" <<endl;}

  //************************************************************************
    // function name: change_position
    // Description: Move qubit from src to dst in the MPS. Used only
    //   for expectation value calculations. Similar to swap, but doesn't
    //   move qubit in dst back to src, therefore being used only on the temp TN
    //   in Expectation_value function.
    // Parameters: uint_t src, source of the qubit.
    //			 uint_t dst, destination of the qubit.
    // Returns: none.
  //************************************************************************
  void change_position(uint_t src, uint_t dst);

  cmatrix_t Density_matrix(const reg_t &qubits) const;

  //  double Expectation_value(const vector<uint_t> &indexes, const string &matrices);
  double Expectation_value(const reg_t &qubits, const string &matrices) const;
  double Expectation_value(const reg_t &qubits, const cmatrix_t &M) const;

  //**************************************************************
  // function name: printTN
  // Description: Prints the tensor network
  // Parameters: none.
  // Returns: none.
  //**************************************************************
  void printTN();

  //*********************************************************************
  // function name: state_vec
  // Description: Computes the state vector of a subset of qubits.
  // 	The regular use is with for all qubits. in this case the output is
  //  	MPS_Tensor with a 2^n vector of 1X1 matrices.
  //  	If not used for all qubits,	the result tensor will contain a
  //   	2^(distance between edges) vector of matrices of some size. This
  //	method is being used for computing expectation value of subset of qubits.
  // Parameters: none.
  // Returns: none.
  //**********************************************************************
  MPS_Tensor state_vec(uint_t first_index, uint_t last_index) const;
  void full_state_vector(cvector_t &state_vector) const;
  void probabilities_vector(rvector_t& probvector) const;

  //methods from qasm_controller that are not supported yet
  void set_omp_threads(int threads) {
    if (threads > 0)
    omp_threads_ = threads;
  }
  void set_omp_threshold(int omp_qubit_threshold) {
    if (omp_qubit_threshold > 0)
      omp_threshold_ = omp_qubit_threshold; 
  }          
  void set_json_chop_threshold(double json_chop_threshold) {
    json_chop_threshold_ = json_chop_threshold;
  }
  void set_sample_measure_index_size(int index_size){
    sample_measure_index_size_ = index_size;
  }
  void enable_gate_opt() {
           cout << "enable_gate_opt not supported yet" <<endl;}

  rvector_t probabilities(const AER::reg_t &qubits) const
  {
	  rvector_t probvector;
	  probabilities_vector(probvector);
	  return probvector;
  }

  //  void store_measure(const AER::reg_t outcome, const AER::reg_t &cmemory, const AER::reg_t &cregister) const{
  //           cout << " store_measure not supported yet" <<endl;}

  double norm(const AER::reg_t &reg_qubits, cvector_t &vmat) const {
           cout << "norm not supported yet" <<endl;
           return 0;}

  //  std::vector<reg_t> sample_measure(std::vector<double> &rnds);
  reg_t sample_measure(std::vector<double> &rnds);
    

protected:
    uint_t num_qubits_;
  /*
    The data structure of a MPS- a vector of Gamma tensors and a vector of Lambda vectors.
  */
  vector<MPS_Tensor> q_reg_;
  vector<rvector_t> lambda_reg_;
  //-----------------------------------------------------------------------
  // Config settings
  //----------------------------------------------------------------------- 
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 14;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 10; // Sample measure indexing qubit size
  double json_chop_threshold_ = 0;  // Threshold for choping small values
                                    // in JSON serialization
};



//-------------------------------------------------------------------------
} // end namespace MPS
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif /* _aer_tensor_state_hpp_ */
