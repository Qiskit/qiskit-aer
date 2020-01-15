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


#ifndef _aer_matrix_product_state_hpp_
#define _aer_matrix_product_state_hpp_

#include "framework/json.hpp"
#include "framework/utils.hpp"
#include "framework/operations.hpp"
#include "matrix_product_state_tensor.hpp"

namespace AER {
namespace MatrixProductState {

// Allowed gates enum class
enum Gates {
  id, h, x, y, z, s, sdg, t, tdg, u1, u2, u3, // single qubit
  cx, cz, cu1, swap, su4, // two qubit
  mcx // three qubit
};

enum class Direction {RIGHT, LEFT};

//=========================================================================
// MPS class
//=========================================================================
//
// The structure used to store the state is a vector of n Gamma-tensors
// (each implemented by two matrices),
// and n-1 Lambda tensors (each implemented by a single vector),
// where n is the number of qubits in the circuit.
// Qubit i is controlled by Gamma-tensor i and Lambda-tensors i and i+1,
// for 0<=i<=n-1.
// -------------------------------------------------------------------------

class MPS{
public:
  MPS(uint_t num_qubits = 0):
    num_qubits_(num_qubits) {}
  ~MPS() {}

  //--------------------------------------------------------------------------
  // function name: initialize
  // Description: Initialize the MPS with some state.
  // 1.	Parameters: none. Initializes all qubits to |0>.
  // 2.	Parameters: const MPS &other - Copy another MPS
  // TODO:
  // 3.	Parameters: uint_t num_qubits, const cvector_t &vecState -
  //  				Initializes qubits with a statevector.
  // Returns: none.
  //----------------------------------------------------------------
  virtual void initialize(uint_t num_qubits=0);
  void initialize(const MPS &other);
  //void initialize(uint_t num_qubits, const cvector_t &vecState);

  //----------------------------------------------------------------
  // function name: num_qubits
  // Description: Get the number of qubits in the MPS
  // Parameters: none.
  // Returns: none.
  //----------------------------------------------------------------
  uint_t num_qubits() const{return num_qubits_;}

  //----------------------------------------------------------------
  // function name: set_num_qubits
  // Description: Set the number of qubits in the MPS
  // Parameters: size_t num_qubits - number of qubits to set.
  // Returns: none.
  //----------------------------------------------------------------
  void set_num_qubits(uint_t num_qubits) {
    num_qubits_ = num_qubits;
  }

  bool empty() const {
    return(num_qubits_ == 0);
  }

  //----------------------------------------------------------------
  // function name: apply_x,y,z,...
  // Description: Apply a gate on some qubits by their indexes.
  // Parameters: uint_t index of the qubit/qubits.
  // Returns: none.
  //----------------------------------------------------------------
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
  void apply_cnot(uint_t index_A, uint_t index_B);
  void apply_swap(uint_t index_A, uint_t index_B);
  void apply_cz(uint_t index_A, uint_t index_B);
  void apply_cu1(uint_t index_A, uint_t index_B, double lambda);
  void apply_2_qubit_gate(uint_t index_A, uint_t index_B, Gates gate_type, 
			  const cmatrix_t &mat);
  void apply_ccx(const reg_t &qubits);  
  void apply_3_qubit_gate(const reg_t &qubits, Gates gate_type, const cmatrix_t &mat);

  void apply_matrix(const reg_t & qubits, const cmatrix_t &mat);

  void apply_matrix(const AER::reg_t &qubits, const cvector_t &vmat)
  {
    apply_diagonal_matrix(qubits, vmat);
  }

  void apply_diagonal_matrix(const AER::reg_t &qubits, const cvector_t &vmat);

  cmatrix_t density_matrix(const reg_t &qubits) const;
  rvector_t trace_of_density_matrix(const reg_t &qubits) const;

  //---------------------------------------------------------------
  // Function: expectation_value
  // Description: Computes expectation value of the given qubits on the given matrix.
  // Parameters: The qubits for which we compute expectation value.
  //             M - the matrix
  // Returns: The expectation value. 
  //------------------------------------------------------------------
  double expectation_value(const reg_t &qubits, const cmatrix_t &M) const;

  //---------------------------------------------------------------
  // Function: expectation_value_pauli
  // Description: Computes expectation value of the given qubits on a string of Pauli matrices.
  // Parameters: The qubits for which we compute expectation value.
  //             A string of matrices of the set {X, Y, Z, I}. The matrices are given in 
  //             reverse order relative to the qubits.
  // Returns: The expectation value in the form of a complex number. The real part is the 
  //          actual expectation value.
  //------------------------------------------------------------------
  complex_t expectation_value_pauli(const reg_t &qubits, const std::string &matrices) const;

  //------------------------------------------------------------------
  // function name: MPS_with_new_indices
  // Description: Creates a copy of *this where the indices of the
  //   selected qubits have been moved for more efficient computation
  //   of the expectation value
  // Parameters: The qubits for which we compute expectation value.
  // Returns: new MPS.
  //----------------------------------------------------------------
  void MPS_with_new_indices(const reg_t &qubits, MPS& temp_MPS,
			    uint_t &front, uint_t &back) const;

  //----------------------------------------------------------------
  // function name: print
  // Description: prints the MPS
  //----------------------------------------------------------------
  virtual std::ostream&  print(std::ostream& out) const;

   //----------------------------------------------------------------
   // function name: get_matrices_sizes
   // Description: returns the size of the inner matrices of the MPS
   //----------------------------------------------------------------
  std::vector<reg_t> get_matrices_sizes() const;

  //----------------------------------------------------------------
  // function name: state_vec_as_MPS
  // Description: Computes the state vector of a subset of qubits.
  // 	The regular use is with for all qubits. in this case the output is
  //  	MPS_Tensor with a 2^n vector of 1X1 matrices.
  //  	If not used for all qubits,	the result tensor will contain a
  //   	2^(distance between edges) vector of matrices of some size. This
  //	method is used for computing expectation value of a subset of qubits.
  // Parameters: none.
  // Returns: none.
  //----------------------------------------------------------------
  MPS_Tensor state_vec_as_MPS(const reg_t &qubits) const;

  // This function computes the state vector for all the consecutive qubits 
  // between first_index and last_index
  MPS_Tensor state_vec_as_MPS(uint_t first_index, uint_t last_index) const;
  void full_state_vector(cvector_t &state_vector) const;
  void get_probabilities_vector(rvector_t& probvector, const reg_t &qubits) const;

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
    std::cout << "enable_gate_opt not supported yet" <<std::endl;
  }

  //  void store_measure(const AER::reg_t outcome, const AER::reg_t &cmemory, const AER::reg_t &cregister) const{
  //           cout << " store_measure not supported yet" <<endl;}

  double norm(const uint_t qubit, cvector_t &vmat) const {
    cmatrix_t mat = AER::Utils::devectorize_matrix(vmat);
    reg_t qubits = {qubit};
    return expectation_value(qubits, mat);
  }

  double norm(const reg_t &qubits, cvector_t &vmat) const {
    cmatrix_t mat = AER::Utils::devectorize_matrix(vmat);
    return expectation_value(qubits, mat);
  }

  reg_t apply_measure(const reg_t &qubits,
		      RngEngine &rng);

  uint_t apply_measure(uint_t qubit,
		       RngEngine &rng);

  //----------------------------------------------------------------
  // function name: initialize_from_statevector
  // Description: This function receives as input a state_vector and
  //      initializes the internal structures of the MPS according to its
  //      state.
  // Parameters: number of qubits, state_vector to initialize from
  // Returns: none.
  //----------------------------------------------------------------

  void initialize_from_statevector(uint_t num_qubits, cvector_t state_vector) {
    cmatrix_t statevector_as_matrix(1, state_vector.size());
#pragma omp parallel for
    for (int_t i=0; i<static_cast<int_t>(state_vector.size()); i++) {
      statevector_as_matrix(0, i) = state_vector[i];
    }
    initialize_from_matrix(num_qubits, statevector_as_matrix);
  }
  void initialize_from_matrix(uint_t num_qubits, cmatrix_t mat);

protected:
  //----------------------------------------------------------------
  // function name: centralize_qubits
  // Description: Creates a new MPS where a subset of the qubits is
  // moved to be in consecutive positions. Used for
  // computations involving a subset of the qubits.
  // Parameters: Input: new_MPS - the MPS with the shifted qubits
  //                    qubits - the subset of qubits
  //             Returns: new_first, new_last - new positions of the 
  //                    first and last qubits respectively
  //                    ordered - are the qubits in ascending order
  // Returns: none.
  //----------------------------------------------------------------
  void centralize_qubits(const reg_t &qubits,
			 reg_t &new_qubits, bool &ordered);

  //----------------------------------------------------------------
  // function name: centralize_and_sort_qubits
  // Description: Similar to centralize_qubits, but also returns the sorted qubit vector
  //----------------------------------------------------------------
  void centralize_and_sort_qubits(const reg_t &qubits, reg_t &sorted_indexes,
			 reg_t &new_qubits, bool &ordered);

  //----------------------------------------------------------------
  // function name: move_qubits_to_original_location
  // Description: This function reverses the effect of centralize_qubits.
  //      It returns the qubits that were previously centralized, to their original positions.
  // Parameters: Input: first - the index of the first qubit that was moved
  //                    original_qubits - the subset of qubits that were moved
  //                    sorted_qubits - the original_qubits in sorted order
  //             Returns: the MPS (this) where the qubits have been moved back to their original
  //                 position.
  // Returns: none.
  //----------------------------------------------------------------
  void move_qubits_to_original_location(uint_t first, const reg_t &original_qubits, 
					const reg_t &sorted_qubits);

  //----------------------------------------------------------------

  // function name: change_position
  // Description: Move qubit from src to dst in the MPS. Used only
  //   for expectation value calculations. Similar to swap, but doesn't
  //   move qubit in dst back to src, therefore being used only on the temp MPS
  //   in Expectation_value function.
  // Parameters: uint_t src, source of the qubit.
  //			 uint_t dst, destination of the qubit.
  // Returns: none.
  //----------------------------------------------------------------
  void change_position(uint_t src, uint_t dst);

  uint_t num_qubits_;
  std::vector<MPS_Tensor> q_reg_;
  std::vector<rvector_t> lambda_reg_;
  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 14;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 10; // Sample measure indexing qubit size
  double json_chop_threshold_ = 1E-8;  // Threshold for choping small values
                                    // in JSON serialization
};

inline std::ostream &operator<<(std::ostream &out, const rvector_t &vec) {
  out << "[";
  uint_t size = vec.size();
  for (uint_t i = 0; i < size-1; ++i) {
    out << vec[i];
    out << ", ";
  }
  out << vec[size-1] << "]";
  return out;
}

inline std::ostream&
operator <<(std::ostream& out, const MPS& mps)
{
  return mps.print(out);
}

//-------------------------------------------------------------------------
} // end namespace MPS
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif /* _aer_matrix_product_state_hpp_ */
