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

#include <cstdarg>

#include "framework/json.hpp"
#include "framework/operations.hpp"
#include "framework/utils.hpp"
#include "matrix_product_state_tensor.hpp"

namespace AER {
namespace MatrixProductState {

// Allowed gates enum class
enum Gates {
  id,
  h,
  x,
  y,
  z,
  s,
  sdg,
  sx,
  sxdg,
  t,
  tdg,
  u1,
  u2,
  u3,
  r,
  rx,
  ry,
  rz, // single qubit
  cx,
  cy,
  cz,
  cu1,
  swap,
  su4,
  rxx,
  ryy,
  rzz,
  rzx,
  csx, // two qubit
  ccx,
  cswap, // three qubit
  pauli
};

// enum class Direction {RIGHT, LEFT};

enum class Sample_measure_alg { APPLY_MEASURE, PROB, MEASURE_ALL, HEURISTIC };
enum class MPS_swap_direction { SWAP_LEFT, SWAP_RIGHT };

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

class MPS {
public:
  MPS(uint_t num_qubits = 0) : num_qubits_(num_qubits) {}
  ~MPS() {}

  //--------------------------------------------------------------------------
  // Function name: initialize
  // Description: Initialize the MPS with some state.
  // 1.	Parameters: none. Initializes all qubits to |0>.
  // 2.	Parameters: const MPS &other - Copy another MPS
  // TODO:
  // 3.	Parameters: uint_t num_qubits, const cvector_t &vecState -
  //  				Initializes qubits with a statevector.
  // Returns: none.
  //----------------------------------------------------------------
  virtual void initialize(uint_t num_qubits = 0);
  void initialize(const MPS &other);
  //  void initialize(const cvector_t &statevector);

  void apply_initialize(const reg_t &qubits, const cvector_t &statevector,
                        RngEngine &rng);
  void initialize_from_mps(const mps_container_t &mps);

  //----------------------------------------------------------------
  // Function name: num_qubits
  // Description: Get the number of qubits in the MPS
  // Parameters: none.
  // Returns: none.
  //----------------------------------------------------------------
  uint_t num_qubits() const { return num_qubits_; }

  //----------------------------------------------------------------
  // Function name: set_num_qubits
  // Description: Set the number of qubits in the MPS
  // Parameters: size_t num_qubits - number of qubits to set.
  // Returns: none.
  //----------------------------------------------------------------
  void set_num_qubits(uint_t num_qubits) { num_qubits_ = num_qubits; }

  bool empty() const { return (num_qubits_ == 0); }

  // the following 3 static methods are used as a reporting mechanism
  // for MPS debug data
  static void clear_log() { logging_str_.clear(); }

  static void print_to_log() { // Base function for recursive function
  }

  template <typename T, typename... Targs>
  static void print_to_log(const T &value, const Targs &...Fargs) {
    if (mps_log_data_) {
      logging_str_ << value;
      MPS::print_to_log(Fargs...); // recursive call
    }
  }

  static std::string output_log() {
    if (mps_log_data_)
      return logging_str_.str();
    else
      return "";
  }

  //----------------------------------------------------------------
  void move_all_qubits_to_sorted_ordering();

  /////////////////////////////////////////////////////////////////
  // API functions
  /////////////////////////////////////////////////////////////////

  //----------------------------------------------------------------
  // Function name: apply_x,y,z,...
  // Description: Apply a gate on some qubits by their indexes.
  // Parameters: uint_t index of the qubit/qubits.
  // Returns: none.
  //----------------------------------------------------------------
  void apply_h(uint_t index);
  void apply_sx(uint_t index);
  void apply_sxdg(uint_t index);
  void apply_r(uint_t index, double phi, double lam);
  void apply_rx(uint_t index, double theta);
  void apply_ry(uint_t index, double theta);
  void apply_rz(uint_t index, double theta);
  void apply_x(uint_t index) { get_qubit(index).apply_x(); }
  void apply_y(uint_t index) { get_qubit(index).apply_y(); }
  void apply_z(uint_t index) { get_qubit(index).apply_z(); }
  void apply_s(uint_t index) { get_qubit(index).apply_s(); }
  void apply_sdg(uint_t index) { get_qubit(index).apply_sdg(); }
  void apply_t(uint_t index) { get_qubit(index).apply_t(); }
  void apply_tdg(uint_t index) { get_qubit(index).apply_tdg(); }
  void apply_u1(uint_t index, double lambda) {
    get_qubit(index).apply_u1(lambda);
  }
  void apply_u2(uint_t index, double phi, double lambda);
  void apply_u3(uint_t index, double theta, double phi, double lambda);
  void apply_cnot(uint_t index_A, uint_t index_B);
  void apply_swap(uint_t index_A, uint_t index_B, bool swap_gate);

  void apply_cy(uint_t index_A, uint_t index_B);
  void apply_cz(uint_t index_A, uint_t index_B);
  void apply_csx(uint_t index_A, uint_t index_B);
  void apply_cu1(uint_t index_A, uint_t index_B, double lambda);
  void apply_rxx(uint_t index_A, uint_t index_B, double theta);
  void apply_ryy(uint_t index_A, uint_t index_B, double theta);
  void apply_rzz(uint_t index_A, uint_t index_B, double theta);
  void apply_rzx(uint_t index_A, uint_t index_B, double theta);

  void apply_ccx(const reg_t &qubits);
  void apply_cswap(const reg_t &qubits);

  void apply_matrix(const reg_t &qubits, const cmatrix_t &mat,
                    bool is_diagonal = false);

  void apply_matrix(const AER::reg_t &qubits, const cvector_t &vmat) {
    apply_diagonal_matrix(qubits, vmat);
  }

  void apply_diagonal_matrix(const AER::reg_t &qubits, const cvector_t &vmat);

  cmatrix_t density_matrix(const reg_t &qubits) const;

  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &kmats,
                   RngEngine &rng);

  //---------------------------------------------------------------
  // Function: expectation_value
  // Description: Computes expectation value of the given qubits on the given
  // matrix. Parameters: The qubits for which we compute expectation value.
  //             M - the matrix
  // Returns: The expectation value.
  //------------------------------------------------------------------
  double expectation_value(const reg_t &qubits, const cmatrix_t &M) const;

  //---------------------------------------------------------------
  // Function: expectation_value_pauli
  // Description: Computes expectation value of the given qubits on a string of
  // Pauli matrices. Parameters: The qubits for which we compute expectation
  // value.
  //             A string of matrices of the set {X, Y, Z, I}. The matrices are
  //             given in reverse order relative to the qubits.
  // Returns: The expectation value in the form of a complex number. The real
  // part is the
  //          actual expectation value.
  //------------------------------------------------------------------
  complex_t expectation_value_pauli(const reg_t &qubits,
                                    const std::string &matrices) const;

  //------------------------------------------------------------------
  // Function name: MPS_with_new_indices
  // Description: Moves the indices of the selected qubits for more efficient
  // computation
  //   of the expectation value
  // Parameters: The qubits for which we compute expectation value.
  // Returns: centralized_qubits - the qubits, after sorting and centralizing
  //
  //----------------------------------------------------------------
  void MPS_with_new_indices(const reg_t &qubits, reg_t &centralized_qubits,
                            MPS &temp_MPS) const;

  //----------------------------------------------------------------
  // Function name: print
  // Description: prints the MPS in the current ordering of the qubits
  // (qubit_order_)
  //----------------------------------------------------------------
  virtual std::ostream &print(std::ostream &out) const;

  Vector<complex_t> full_statevector();

  Vector<complex_t> get_amplitude_vector(const reg_t &base_values);

  //----------------------------------------------------------------
  // Function name: get_single_amplitude
  // Description: Returns the amplitude of the input base_value
  //----------------------------------------------------------------
  complex_t get_single_amplitude(const std::string &base_value);

  void get_probabilities_vector(rvector_t &probvector,
                                const reg_t &qubits) const;

  //----------------------------------------------------------------
  // Function name: get_prob_single_qubit_internal
  // Description: Returns the probability of measuring outcome 0 (or 1)
  //   for a single qubit in the standard basis.
  //   It does the same as get_probabilities_vector but is faster for
  //   a single qubit, and is used during measurement.
  // Parameters: qubit - the qubit for which we want the probability
  //             outcome - probability for 0 or 1
  //             mat - the '0' (or '1')matrix for the given qubit, multiplied
  //                   by its left and right lambdas. Contracting it with
  //                   its conjugate gives the probability for outcome '0' (or
  //                   '1') It is returned because it may be useful for further
  //                   computations.
  // Returns: the probability for the given outcome.
  //----------------------------------------------------------------

  double get_prob_single_qubit_internal(uint_t qubit, uint_t outcome,
                                        cmatrix_t &mat) const;
  //----------------------------------------------------------------
  // Function name: get_accumulated_probabilities_vector
  // Description: Computes the accumulated probabilities from 0
  // Parameters: qubits - the qubits for which we compute probabilities
  // Returns: acc_probvector - the vector of accumulated probabilities
  //          index_vec - the base values whose probabilities are not 0
  // For example:
  // if probabilities vector is: 0.5 (00), 0.0 (01), 0.2 (10), 0.3 (11), then
  // acc_probvector = 0.0,    0.5,    0.7,   1.0
  // index_vec =      0 (00), 2 (10), 3(11)
  //----------------------------------------------------------------

  void get_accumulated_probabilities_vector(rvector_t &acc_probvector,
                                            reg_t &index_vec,
                                            const reg_t &qubits) const;

  static void set_omp_threads(uint_t threads) {
    if (threads > 0)
      omp_threads_ = threads;
  }
  static void set_omp_threshold(uint_t omp_qubit_threshold) {
    if (omp_qubit_threshold > 0)
      omp_threshold_ = omp_qubit_threshold;
  }
  static void set_json_chop_threshold(double json_chop_threshold) {
    json_chop_threshold_ = json_chop_threshold;
  }

  static void set_sample_measure_alg(Sample_measure_alg alg) {
    sample_measure_alg_ = alg;
  }

  static void set_enable_gate_opt(bool enable_gate_opt) {
    enable_gate_opt_ = enable_gate_opt;
  }

  static void set_mps_log_data(bool mps_log_data) {
    mps_log_data_ = mps_log_data;
  }

  static void set_mps_swap_direction(MPS_swap_direction direction) {
    mps_swap_direction_ = direction;
  }

  static uint_t get_omp_threads() { return omp_threads_; }
  static uint_t get_omp_threshold() { return omp_threshold_; }
  static double get_json_chop_threshold() { return json_chop_threshold_; }
  static Sample_measure_alg get_sample_measure_alg() {
    return sample_measure_alg_;
  }

  static bool get_enable_gate_opt() { return enable_gate_opt_; }

  static bool get_mps_log_data() { return mps_log_data_; }

  static MPS_swap_direction get_swap_direction() { return mps_swap_direction_; }

  //----------------------------------------------------------------
  // Function name: norm
  // Description: the norm is defined as <psi|A^dagger . A|psi>.
  // It is equivalent to returning the expectation value of A^\dagger A,
  // Returns: double (the norm)
  //----------------------------------------------------------------

  double norm() const;
  double norm(const reg_t &qubits) const;
  double norm(const reg_t &qubits, const cvector_t &vmat) const;
  double norm(const reg_t &qubits, const cmatrix_t &mat) const;

  reg_t apply_measure(const reg_t &qubits, const rvector_t &rnds);
  reg_t apply_measure_internal(const reg_t &qubits, const rvector_t &rands);
  reg_t sample_measure(uint_t shots, RngEngine &rng) const;

  //----------------------------------------------------------------
  // Function name: initialize_from_statevector_internal
  // Description: This function receives as input a state_vector and
  //      initializes the internal structures of the MPS according to its
  //      state.
  // Parameters: qubits - with the internal ordering
  //             statevector to initialize from
  // Returns: none.
  //----------------------------------------------------------------

  void initialize_from_statevector_internal(const reg_t &qubits,
                                            const cvector_t &state_vector);
  void reset(const reg_t &qubits, RngEngine &rng);

  reg_t get_bond_dimensions() const;
  void print_bond_dimensions() const;
  uint_t get_max_bond_dimensions() const;

  mps_container_t copy_to_mps_container();
  mps_container_t move_to_mps_container();

private:
  MPS_Tensor &get_qubit(uint_t index) { return q_reg_[get_qubit_index(index)]; }

  uint_t get_qubit_index(uint_t index) const {
    return qubit_ordering_.location_[index];
  }

  reg_t get_internal_qubits(const reg_t &qubits) const;

  // The following methods are the internal versions of the api functions.
  // They are each called from the corresponding api function with
  // the internal ordering of the qubits - using get_internal_qubits

  // if swap_gate==true, this is an actual swap_gate of the circuit
  // if swap_gate==false, this is an internal swap, necessary for
  // some internal algorithm
  void apply_swap_internal(uint_t index_A, uint_t index_B,
                           bool swap_gate = false);
  void print_to_log_internal_swap(uint_t qubit0, uint_t qubit1) const;

  void apply_2_qubit_gate(uint_t index_A, uint_t index_B, Gates gate_type,
                          const cmatrix_t &mat, bool is_diagonal = false);
  void common_apply_2_qubit_gate(uint_t index_A, Gates gate_type,
                                 const cmatrix_t &mat, bool swapped,
                                 bool is_diagonal = false);
  void apply_3_qubit_gate(const reg_t &qubits, Gates gate_type,
                          const cmatrix_t &mat, bool is_diagonal = false);
  void apply_matrix_internal(const reg_t &qubits, const cmatrix_t &mat,
                             bool is_diagonal = false);

  // Certain local operations need to be propagated to the neighboring qubits.
  // Such operations include apply_measure and apply_kraus
  void propagate_to_neighbors_internal(uint_t min_qubit, uint_t max_qubit,
                                       uint_t next_measured_qubit);

  // apply_matrix for more than 2 qubits
  void apply_multi_qubit_gate(const reg_t &qubits, const cmatrix_t &mat,
                              bool is_diagonal = false);

  void apply_kraus_internal(const reg_t &qubits,
                            const std::vector<cmatrix_t> &kmats,
                            RngEngine &rng);

  // The following two are helper functions for apply_multi_qubit_gate
  void apply_unordered_multi_qubit_gate(const reg_t &qubits,
                                        const cmatrix_t &mat,
                                        bool is_diagonal = false);
  void apply_matrix_to_target_qubits(const reg_t &target_qubits,
                                     const cmatrix_t &mat,
                                     bool is_diagonal = false);
  cmatrix_t density_matrix_internal(const reg_t &qubits) const;
  rvector_t diagonal_of_density_matrix(const reg_t &qubits) const;

  double expectation_value_internal(const reg_t &qubits,
                                    const cmatrix_t &M) const;
  complex_t expectation_value_pauli_internal(const reg_t &qubits,
                                             const std::string &matrices,
                                             uint_t first_index,
                                             uint_t last_index,
                                             uint_t num_Is) const;

  //----------------------------------------------------------------
  // Function name: get_matrices_sizes
  // Description: returns the size of the inner matrices of the MPS
  //----------------------------------------------------------------
  std::vector<reg_t> get_matrices_sizes() const;

  //----------------------------------------------------------------
  // Function name: state_vec_as_MPS
  // Description: Computes the state vector of a subset of qubits.
  // 	The regular use is with for all qubits. in this case the output is
  //  	MPS_Tensor with a 2^n vector of 1X1 matrices.
  //  	If not used for all qubits,	the result tensor will contain a
  //   	2^(distance between edges) vector of matrices of some size. This
  //	method is used for computing expectation value of a subset of qubits.
  // Parameters: none.
  // Returns: none.
  //----------------------------------------------------------------
  MPS_Tensor state_vec_as_MPS(const reg_t &qubits);

  // This function computes the state vector for all the consecutive qubits
  // between first_index and last_index
  MPS_Tensor state_vec_as_MPS(uint_t first_index, uint_t last_index) const;

  Vector<complex_t> full_state_vector_internal(const reg_t &qubits);

  void get_probabilities_vector_internal(rvector_t &probvector,
                                         const reg_t &qubits) const;

  uint_t apply_measure_internal_single_qubit(uint_t qubit, const double rnd,
                                             uint_t next_measured_qubit);

  uint_t sample_measure_single_qubit(uint_t qubit, double &prob, double rnd,
                                     cmatrix_t &mat) const;

  // The return result contains the elements of input_qubits sorted according to
  // their order in ordering_.order_.
  // The vector returned in sub_ordering contains the indices of the qubits to
  // be measured in ordering_.order_.
  reg_t sort_qubits_by_ordering(const reg_t &input_qubits, reg_t &sub_ordering);
  reg_t sort_measured_values(const reg_t &input_outcome, reg_t &sub_ordering);
  //----------------------------------------------------------------
  // Function name: get_single_probability0
  // Description: Returns the probability that `qubit` will measure 0, given all
  // the measurements of the previous qubits that are accumulated in mat.
  //----------------------------------------------------------------
  double get_single_probability0(uint_t qubit, const cmatrix_t &mat) const;

  //----------------------------------------------------------------
  // Function name: initialize_from_matrix
  // Description: This method is similar to initialize_from_statevector, only
  // here
  //      the statevector has been converted to a 1xn matrix. The motivation is
  //      that the algorithm works by iteratively reshaping the statevector into
  //      a matrix and extracting one dimension every time to create one tensor
  //      of the mps.
  // Parameters: num_qubits - the number of qubits
  //             mat - contains the reshaped statevector to initialize from
  // Returns: none.
  //----------------------------------------------------------------

  void initialize_from_matrix(uint_t num_qubits, const cmatrix_t &mat);
  void initialize_component_internal(const reg_t &qubits,
                                     const cvector_t &statevector,
                                     RngEngine &rng);

  void reset_internal(const reg_t &qubits, RngEngine &rng);
  void measure_reset_update_internal(const reg_t &qubits,
                                     const reg_t &meas_state);

  //----------------------------------------------------------------
  // Function name: centralize_qubits
  // Description: Creates a new MPS where a subset of the qubits is
  // moved to be in consecutive positions. Used for
  // computations involving a subset of the qubits.
  //----------------------------------------------------------------
  void centralize_qubits(const reg_t &qubits, reg_t &centralized_qubits);

  //----------------------------------------------------------------
  // Function name: find_centralized_indices
  // Description: Performs the first part of centralize_qubits, i.e., returns
  // the
  //    new target indices, but does not actually change the MPS structure.
  //----------------------------------------------------------------
  void find_centralized_indices(const reg_t &qubits, reg_t &sorted_indices,
                                reg_t &centralized_qubits) const;

  //----------------------------------------------------------------
  // Function name: move_qubits_to_centralized_indices
  // Description: Performs the second part of centralize_qubits, i.e., moves the
  // qubits to the centralized indices
  //----------------------------------------------------------------
  void move_qubits_to_centralized_indices(const reg_t &sorted_indices,
                                          const reg_t &centralized_qubits);

  // Function name: change_position
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

  struct ordering {
    // order_ stores the current ordering of the qubits,
    // location_ stores the location of each qubit in the vector. It is derived
    // from order_ at the end of every swap operation for performance reasons
    // for example: starting position order_ = location_ = 01234
    // cx(0,4) -> order_ = 04123, location_ = 02341
    reg_t order_;
    reg_t location_;
  } qubit_ordering_;

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------
  static uint_t omp_threads_; // Disable multithreading by default
  static uint_t
      omp_threshold_; // Qubit threshold for multithreading when enabled
  static Sample_measure_alg
      sample_measure_alg_; // Algorithm for computing sample_measure

  static double json_chop_threshold_; // Threshold for choping small values
                                      // in JSON serialization
  static bool enable_gate_opt_;       // allow optimizations on gates
  static std::stringstream logging_str_;
  static bool mps_log_data_;
  static MPS_swap_direction mps_swap_direction_;
};

inline std::ostream &operator<<(std::ostream &out, const rvector_t &vec) {
  out << "[";
  uint_t size = vec.size();
  for (uint_t i = 0; i < size - 1; ++i) {
    out << vec[i];
    out << ", ";
  }
  out << vec[size - 1] << "]";
  return out;
}

inline std::ostream &operator<<(std::ostream &out, MPS &mps) {
  return mps.print(out);
}

inline void to_json(json_t &js, const MPS &mps) {}

//-------------------------------------------------------------------------
} // namespace MatrixProductState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif /* _aer_matrix_product_state_hpp_ */
