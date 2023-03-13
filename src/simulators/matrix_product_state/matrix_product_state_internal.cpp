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

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <utility>

#include "framework/linalg/almost_equal.hpp"
#include "framework/matrix.hpp"
#include "framework/utils.hpp"

#include "matrix_product_state_internal.hpp"
#include "matrix_product_state_tensor.hpp"

namespace AER {
namespace MatrixProductState {

static const cmatrix_t zero_measure =
    AER::Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}}, {{0, 0}, {0, 0}}});
static const cmatrix_t one_measure =
    AER::Utils::make_matrix<complex_t>({{{0, 0}, {0, 0}}, {{0, 0}, {1, 0}}});
uint_t MPS::omp_threads_ = 1;
uint_t MPS::omp_threshold_ = 14;
enum Sample_measure_alg MPS::sample_measure_alg_ =
    Sample_measure_alg::HEURISTIC;
enum MPS_swap_direction MPS::mps_swap_direction_ =
    MPS_swap_direction::SWAP_LEFT;
double MPS::json_chop_threshold_ = 1E-8;
std::stringstream MPS::logging_str_;
bool MPS::mps_log_data_ = 0;

//------------------------------------------------------------------------
// local function declarations
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// Function name: squeeze_qubits
// Description: Takes a list of qubits, and squeezes them into a list of the
// same size,
//     that begins at 0, and where all qubits are consecutive. Note that
//     relative order between qubits is preserved. Example: [8, 4, 6, 0, 9] ->
//     [3, 1, 2, 0, 4]
// Input: original_qubits
// Returns: squeezed_qubits
//
//------------------------------------------------------------------------
void squeeze_qubits(const reg_t &original_qubits, reg_t &squeezed_qubits);

//------------------------------------------------------------------------
// Function name: reorder_all_qubits
// Description: The ordering of the amplitudes in the statevector in this module
// is
//    [n, (n-1),.., 2, 1, 0], i.e., msb is leftmost and lsb is rightmost.
//    Sometimes, we need to provide a different ordering of the amplitudes, such
//    as in snapshot_probabilities. For example, instead of [2, 1, 0] the user
//    requests the probabilities of [1, 0, 2]. Note that the qubits are numbered
//    from left to right, e.g., 210
// Input: orig_probvector - the ordered vector of probabilities/amplitudes
//        qubits - a list containing the new ordering
// Returns: new_probvector - the vector in the new ordering
//    e.g., 011->101 (for the ordering [1, 0, 2]
//
//------------------------------------------------------------------------
template <class vec_t>
void reorder_all_qubits(const vec_t &orig_probvector, const reg_t &qubits,
                        vec_t &new_probvector);
uint_t reorder_qubits(const reg_t &qubits, uint_t index);

//------------------------------------------------------------------------
// Function name: permute_all_qubits
// Description: Given an input ordering of the qubits, an output ordering of the
//    qubits and a statevector, create the statevector that represents the same
//    state in the output ordering of the qubits.
// Input: orig_statevector - the ordered vector of probabilities/amplitudes
//        input_qubits - a list containing the original ordering of the qubits
//        output_qubits - a list containing the new ordering
// Returns: new_statevector - the vector in the new ordering
// Note that the qubits are numbered from left to right, i.e., the msb is 0
//
//------------------------------------------------------------------------
template <class vec_t>
void permute_all_qubits(const vec_t &orig_statevector,
                        const reg_t &input_qubits, const reg_t &output_qubits,
                        vec_t &new_statevector);

//------------------------------------------------------------------------
// Function name: permute_qubits
// Description: Helper function for permute_all_qubits
//    Given a single index in the amplitude vector, returns the index in the new
//    statevector that will be assigned the value of this index.
// Input: index - the original index
//        input_qubits - the original ordering of the qubits
//        output_qubits - a list containing the new ordering
// Returns: new index
// Note that the qubits are numbered from left to right, i.e., the msb is 0
// For example, if input_qubits=[0,1,2], and output_qubits=[1,0,2], for
// index=[0x100], output index =[0x010]
//------------------------------------------------------------------------
uint_t permute_qubits(const reg_t &input_qubits, uint_t index,
                      const reg_t &output_qubits);

//--------------------------------------------------------------------------
// Function name: reverse_all_bits
// Description: The ordering of the amplitudes in the statevector in this module
// is
//    000, 001, 010, 011, 100, 101, 110, 111.
//    The ordering of the amplitudes in the statevector in Qasm in general is
//    000, 100, 010, 110, 001, 101, 011, 111.
//    This function converts the statevector from one representation to the
//    other. This is a special case of reorder_qubits
// Input: the input statevector and the number of qubits
// Output: the statevector in reverse order
//----------------------------------------------------------------
template <class vec_t>
vec_t reverse_all_bits(const vec_t &statevector, uint_t num_qubits);

// with 5 qubits, the number 2 in binary is 00010,
// when reversed it is 01000, which is the number 8
uint_t reverse_bits(uint_t num, uint_t len);

std::vector<uint_t> calc_new_indices(const reg_t &indices);

// The following two functions are helper functions used by
// initialize_from_statevector
cmatrix_t reshape_matrix(cmatrix_t input_matrix);
cmatrix_t mul_matrix_by_lambda(const cmatrix_t &mat, const rvector_t &lambda);

std::string sort_paulis_by_qubits(const std::string &paulis,
                                  const reg_t &qubits);

bool is_ordered(const reg_t &qubits);

uint_t binary_search(const rvector_t &acc_probvector, uint_t start, uint_t end,
                     double rnd);
//------------------------------------------------------------------------
// local function implementations
//------------------------------------------------------------------------
void squeeze_qubits(const reg_t &original_qubits, reg_t &squeezed_qubits) {
  std::vector<uint_t> sorted_qubits;
  for (uint_t index : original_qubits) {
    sorted_qubits.push_back(index);
  }
  sort(sorted_qubits.begin(), sorted_qubits.end());
  for (uint_t i = 0; i < original_qubits.size(); i++) {
    for (uint_t j = 0; j < sorted_qubits.size(); j++) {
      if (original_qubits[i] == sorted_qubits[j]) {
        squeezed_qubits[i] = j;
        break;
      }
    }
  }
}

template <class vec_t>
void reorder_all_qubits(const vec_t &orig_probvector, const reg_t &qubits,
                        vec_t &new_probvector) {
  uint_t new_index;
  uint_t length = 1ULL << qubits.size(); // length = pow(2, num_qubits)

  // if qubits are [k0, k1,...,kn], move them to [0, 1, .. , n], but preserve
  // relative ordering
  reg_t squeezed_qubits(qubits.size());
  squeeze_qubits(qubits, squeezed_qubits);

  for (uint_t i = 0; i < length; i++) {
    new_index = reorder_qubits(squeezed_qubits, i);
    new_probvector[new_index] = orig_probvector[i];
  }
}

uint_t reorder_qubits(const reg_t &qubits, uint_t index) {
  uint_t new_index = 0;

  int_t current_pos = 0, current_val = 0, new_pos = 0, shift = 0;
  uint_t num_qubits = qubits.size();
  for (uint_t i = 0; i < num_qubits; i++) {
    current_pos = num_qubits - 1 - qubits[i];
    current_val = 1ULL << current_pos;
    new_pos = num_qubits - 1 - i;
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

template <class vec_t>
void permute_all_qubits(const vec_t &orig_statevector,
                        const reg_t &input_qubits, const reg_t &output_qubits,
                        vec_t &new_statevector) {
  uint_t new_index;
  uint_t length = 1ULL << input_qubits.size(); // length = pow(2, num_qubits)
  // if qubits are [k0, k1,...,kn], move them to [0, 1, .. , n], but preserve
  // relative ordering
  reg_t squeezed_qubits(input_qubits.size());
  squeeze_qubits(input_qubits, squeezed_qubits);

  for (uint_t i = 0; i < length; i++) {
    new_index = permute_qubits(squeezed_qubits, i, output_qubits);
    new_statevector[new_index] = orig_statevector[i];
  }
}

uint_t permute_qubits(const reg_t &input_qubits, uint_t index,
                      const reg_t &output_qubits) {
  uint_t new_index = 0;
  // pos is from right to left, i.e., msb has pos 0, and lsb has pos
  // num_qubits-1 This is in line with the ordering of the qubits in the MPS
  // structure
  int_t current_pos = 0, current_val = 0, new_pos = 0, shift = 0;
  uint_t num_qubits = input_qubits.size();

  for (uint_t in = 0; in < num_qubits; in++) {
    current_pos = in;
    for (uint_t out = 0; out < num_qubits; out++) {
      if (input_qubits[in] == output_qubits[out]) {
        new_pos = out;
        break;
      }
    }
    shift = new_pos - current_pos;
    current_val = 1ULL << current_pos;

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

// with 5 qubits, the number 2 in binary is 00010,
// when reversed it is 01000, which is the number 8
uint_t reverse_bits(uint_t num, uint_t len) {
  uint_t sum = 0;
  for (uint_t i = 0; i < len; ++i) {
    if ((num & 0x1) == 1) {
      sum += 1ULL << (len - 1 - i); // adding pow(2, len-1-i)
    }
    num = num >> 1;
    if (num == 0) {
      break;
    }
  }
  return sum;
}

template <class vec_t>
vec_t reverse_all_bits(const vec_t &statevector, uint_t num_qubits) {
  uint_t length = statevector.size(); // length = pow(2, num_qubits_)
  vec_t output_vector;
  output_vector.resize(length);

#pragma omp parallel for if (length > MPS::get_omp_threshold() &&              \
                             MPS::get_omp_threads() > 1)                       \
    num_threads(MPS::get_omp_threads())
  for (int_t i = 0; i < static_cast<int_t>(length); i++) {
    output_vector[i] = statevector[reverse_bits(i, num_qubits)];
  }

  return output_vector;
}

std::vector<uint_t> calc_new_indices(const reg_t &indices) {
  // assumes indices vector is sorted
  uint_t n = indices.size();
  uint_t mid_index = indices[(n - 1) / 2];
  uint_t first = mid_index - (n - 1) / 2;
  std::vector<uint_t> new_indices(n);
  std::iota(std::begin(new_indices), std::end(new_indices), first);
  return new_indices;
}

cmatrix_t mul_matrix_by_lambda(const cmatrix_t &mat, const rvector_t &lambda) {
  if (lambda == rvector_t{1.0})
    return mat;
  cmatrix_t res_mat(mat);
  uint_t num_rows = mat.GetRows(), num_cols = mat.GetColumns();

#ifdef _WIN32
#pragma omp parallel for if (num_rows * num_cols >                             \
                                 MPS_Tensor::MATRIX_OMP_THRESHOLD &&           \
                             MPS::get_omp_threads() > 1)                       \
    num_threads(MPS::get_omp_threads())
#else
#pragma omp parallel for collapse(                                             \
    2) if (num_rows * num_cols > MPS_Tensor::MATRIX_OMP_THRESHOLD &&           \
           MPS::get_omp_threads() > 1) num_threads(MPS::get_omp_threads())
#endif
  for (int_t row = 0; row < static_cast<int_t>(num_rows); row++) {
    for (int_t col = 0; col < static_cast<int_t>(num_cols); col++) {
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

std::string sort_paulis_by_qubits(const std::string &paulis,
                                  const reg_t &qubits) {
  uint_t min = UINT_MAX;
  uint_t min_index = 0;

  std::string new_paulis;
  std::vector<uint_t> temp_qubits = qubits;
  // find min_index, the next smallest index in qubits
  for (uint_t i = 0; i < paulis.size(); i++) {
    min = temp_qubits[0];
    for (uint_t qubit = 0; qubit < qubits.size(); qubit++)
      if (temp_qubits[qubit] <= min) {
        min = temp_qubits[qubit];
        min_index = qubit;
      }
    // select the corresponding pauli, and put it next in
    // the sorted vector
    new_paulis.push_back(paulis[min_index]);
    // make sure we don't select this index again by setting it to UINT_MAX
    temp_qubits[min_index] = UINT_MAX;
  }
  return new_paulis;
}

bool is_ordered(const reg_t &qubits) {
  bool ordered = true;
  for (uint_t index = 0; index < qubits.size() - 1; index++) {
    if (qubits[index] + 1 != qubits[index + 1]) {
      ordered = false;
      break;
    }
  }
  return ordered;
}
//------------------------------------------------------------------------
// implementation of MPS methods
//------------------------------------------------------------------------

void MPS::initialize(uint_t num_qubits) {
  num_qubits_ = num_qubits;
  q_reg_.clear();
  lambda_reg_.clear();
  complex_t alpha = 1.0f;
  complex_t beta = 0.0f;

  for (uint_t i = 0; i < num_qubits_; i++) {
    q_reg_.push_back(MPS_Tensor(alpha, beta));
  }
  for (uint_t i = 1; i < num_qubits_; i++) {
    lambda_reg_.push_back(rvector_t{1.0});
  }

  qubit_ordering_.order_.clear();
  qubit_ordering_.order_.resize(num_qubits);
  std::iota(qubit_ordering_.order_.begin(), qubit_ordering_.order_.end(), 0);

  qubit_ordering_.location_.clear();
  qubit_ordering_.location_.resize(num_qubits);
  std::iota(qubit_ordering_.location_.begin(), qubit_ordering_.location_.end(),
            0);
}

void MPS::initialize(const MPS &other) {
  if (this != &other) {
    num_qubits_ = other.num_qubits_;
    q_reg_ = other.q_reg_;
    lambda_reg_ = other.lambda_reg_;
    qubit_ordering_.order_ = other.qubit_ordering_.order_;
    qubit_ordering_.location_ = other.qubit_ordering_.location_;
  }
}

reg_t MPS::get_internal_qubits(const reg_t &qubits) const {
  reg_t internal_qubits(qubits.size());
  for (uint_t i = 0; i < qubits.size(); i++) {
    internal_qubits[i] = get_qubit_index(qubits[i]);
  }
  return internal_qubits;
}

void MPS::apply_h(uint_t index) {
  get_qubit(index).apply_matrix(AER::Linalg::Matrix::H);
}

void MPS::apply_sx(uint_t index) {
  get_qubit(index).apply_matrix(AER::Linalg::Matrix::SX);
}

void MPS::apply_sxdg(uint_t index) {
  get_qubit(index).apply_matrix(AER::Linalg::Matrix::SXDG);
}

void MPS::apply_r(uint_t index, double phi, double lam) {
  get_qubit(index).apply_matrix(AER::Linalg::Matrix::r(phi, lam));
}

void MPS::apply_rx(uint_t index, double theta) {
  get_qubit(index).apply_matrix(AER::Linalg::Matrix::rx(theta));
}

void MPS::apply_ry(uint_t index, double theta) {
  get_qubit(index).apply_matrix(AER::Linalg::Matrix::ry(theta));
}

void MPS::apply_rz(uint_t index, double theta) {
  get_qubit(index).apply_matrix(AER::Linalg::Matrix::rz(theta));
}

void MPS::apply_u2(uint_t index, double phi, double lambda) {
  get_qubit(index).apply_matrix(AER::Linalg::Matrix::u2(phi, lambda));
}

void MPS::apply_u3(uint_t index, double theta, double phi, double lambda) {
  get_qubit(index).apply_matrix(AER::Linalg::Matrix::u3(theta, phi, lambda));
}

void MPS::apply_cnot(uint_t index_A, uint_t index_B) {
  apply_2_qubit_gate(get_qubit_index(index_A), get_qubit_index(index_B), cx,
                     cmatrix_t(1, 1));
}

void MPS::apply_cy(uint_t index_A, uint_t index_B) {
  apply_2_qubit_gate(get_qubit_index(index_A), get_qubit_index(index_B), cy,
                     cmatrix_t(1, 1));
}

void MPS::apply_cz(uint_t index_A, uint_t index_B) {
  apply_2_qubit_gate(get_qubit_index(index_A), get_qubit_index(index_B), cz,
                     cmatrix_t(1, 1));
}

void MPS::apply_cu1(uint_t index_A, uint_t index_B, double lambda) {
  cmatrix_t lambda_in_mat(1, 1);
  lambda_in_mat(0, 0) = lambda;
  apply_2_qubit_gate(get_qubit_index(index_A), get_qubit_index(index_B), cu1,
                     lambda_in_mat);
}

void MPS::apply_csx(uint_t index_A, uint_t index_B) {
  cmatrix_t sx_matrix = AER::Linalg::Matrix::SX;
  apply_2_qubit_gate(get_qubit_index(index_A), get_qubit_index(index_B), csx,
                     sx_matrix);
}

void MPS::apply_rxx(uint_t index_A, uint_t index_B, double theta) {
  cmatrix_t rxx_matrix = AER::Linalg::Matrix::rxx(theta);
  apply_2_qubit_gate(get_qubit_index(index_A), get_qubit_index(index_B), su4,
                     rxx_matrix);
}

void MPS::apply_ryy(uint_t index_A, uint_t index_B, double theta) {
  cmatrix_t ryy_matrix = AER::Linalg::Matrix::ryy(theta);
  apply_2_qubit_gate(get_qubit_index(index_A), get_qubit_index(index_B), su4,
                     ryy_matrix);
}

void MPS::apply_rzz(uint_t index_A, uint_t index_B, double theta) {
  cmatrix_t rzz_matrix = AER::Linalg::Matrix::rzz(theta);
  apply_2_qubit_gate(get_qubit_index(index_A), get_qubit_index(index_B), su4,
                     rzz_matrix);
}

void MPS::apply_rzx(uint_t index_A, uint_t index_B, double theta) {
  cmatrix_t rzx_matrix = AER::Linalg::Matrix::rzx(theta);
  apply_2_qubit_gate(get_qubit_index(index_A), get_qubit_index(index_B), su4,
                     rzx_matrix);
}

void MPS::apply_ccx(const reg_t &qubits) {
  reg_t internal_qubits = get_internal_qubits(qubits);
  apply_3_qubit_gate(internal_qubits, ccx, cmatrix_t(1, 1));
}

void MPS::apply_cswap(const reg_t &qubits) {
  reg_t internal_qubits = get_internal_qubits(qubits);
  apply_3_qubit_gate(internal_qubits, cswap, cmatrix_t(1, 1));
}

void MPS::apply_swap(uint_t index_A, uint_t index_B, bool swap_gate) {
  apply_swap_internal(get_qubit_index(index_A), get_qubit_index(index_B),
                      swap_gate);
}

void MPS::apply_swap_internal(uint_t index_A, uint_t index_B, bool swap_gate) {
  uint_t actual_A = index_A;
  uint_t actual_B = index_B;
  if (actual_A > actual_B) {
    std::swap(actual_A, actual_B);
  }

  if (actual_A + 1 < actual_B) {
    uint_t i;
    for (i = actual_A; i < actual_B; i++) {
      apply_swap_internal(i, i + 1, swap_gate);
    }
    for (i = actual_B - 1; i > actual_A; i--) {
      apply_swap_internal(i, i - 1, swap_gate);
    }
    return;
  }
  // when actual_A+1 == actual_B then we can really do the swap between A and
  // A+1
  common_apply_2_qubit_gate(actual_A, Gates::swap,
                            cmatrix_t(1, 1) /*dummy matrix*/,
                            false /*swapped*/);

  if (!swap_gate) {
    // we move the qubit at index_A one position to the right
    // and the qubit at index_B (or index_A+1) is moved one position
    // to the left
    std::swap(qubit_ordering_.order_[index_A], qubit_ordering_.order_[index_B]);
    // For logging purposes:
    print_to_log_internal_swap(index_A, index_B);

    // update qubit locations after all the swaps
    for (uint_t i = 0; i < num_qubits_; i++)
      qubit_ordering_.location_[qubit_ordering_.order_[i]] = i;
  }
}

void MPS::print_to_log_internal_swap(uint_t qubit0, uint_t qubit1) const {
  if (mps_log_data_) {
    print_to_log("internal_swap on qubits ", qubit0, ",", qubit1);
  }
  print_bond_dimensions();
}

void MPS::print_bond_dimensions() const {
  print_to_log(", BD=[");
  reg_t bd = get_bond_dimensions();
  for (uint_t index = 0; index < bd.size(); index++) {
    print_to_log(bd[index]);
    if (index < bd.size() - 1)
      print_to_log(" ");
  }
  print_to_log("],  ");
}
//-------------------------------------------------------------------------
// MPS::apply_2_qubit_gate - outline of the algorithm
// 1. Swap qubits A and B until they are consecutive
// 2. Contract MPS_Tensor[A] and MPS_Tensor[B], yielding a temporary four-matrix
// MPS_Tensor
//    that represents the entangled states of A and B.
// 3. Apply the gate
// 4. Decompose the temporary MPS_Tensor (using SVD) into U*S*V, where U and V
// are matrices
//    and S is a diagonal matrix
// 5. U is split by rows to yield two MPS_Tensors representing qubit A (in
// reshape_U_after_SVD),
//    V is split by columns to yield two MPS_Tensors representing qubit B (in
//    reshape_V_after_SVD), the diagonal of S becomes the Lambda-vector in
//    between A and B.
//-------------------------------------------------------------------------
void MPS::apply_2_qubit_gate(uint_t index_A, uint_t index_B, Gates gate_type,
                             const cmatrix_t &mat, bool is_diagonal) {
  // We first move the two qubits to be in consecutive positions
  // By default, the right qubit is moved to the position after the left qubit.
  // However, the user can choose to move the left qubit to be in the position
  // before the right qubit by changing the MPS_swap_direction to SWAP_RIGHT.
  // The direction of the swaps may affect performance, depending on the
  // circuit.

  bool swapped = false;
  uint_t low_qubit = 0, high_qubit = 0;

  if (index_B > index_A) {
    low_qubit = index_A;
    high_qubit = index_B;
  } else {
    low_qubit = index_B;
    high_qubit = index_A;
    swapped = true;
  }
  if (mps_swap_direction_ == MPS_swap_direction::SWAP_LEFT) {
    // Move high_qubit to be right after low_qubit
    change_position(high_qubit, low_qubit + 1);
  } else { // mps_swap_right
    // Move low_qubit to be right before high_qubit
    change_position(low_qubit, high_qubit - 1);
    low_qubit = high_qubit - 1;
  }
  common_apply_2_qubit_gate(low_qubit, gate_type, mat, swapped, is_diagonal);
}

void MPS::common_apply_2_qubit_gate(
    uint_t A, // the gate is applied to A and A+1
    Gates gate_type, const cmatrix_t &mat, bool swapped, bool is_diagonal) {
  // After we moved the qubits as necessary,
  // the operation is always between qubits A and A+1

  // There is no lambda on the edges of the MPS
  if (A != 0)
    q_reg_[A].mul_Gamma_by_left_Lambda(lambda_reg_[A - 1]);
  if (A + 1 != num_qubits_ - 1)
    q_reg_[A + 1].mul_Gamma_by_right_Lambda(lambda_reg_[A + 1]);

  MPS_Tensor temp =
      MPS_Tensor::contract(q_reg_[A], lambda_reg_[A], q_reg_[A + 1]);

  switch (gate_type) {
  case cx:
    temp.apply_cnot(swapped);
    break;
  case cy:
    temp.apply_cy(swapped);
    break;
  case cz:
    temp.apply_cz();
    break;
  case swap:
    temp.apply_swap();
    break;
  case id:
    break;
  case cu1:
    temp.apply_cu1(std::real(mat(0, 0)));
    break;
  case csx:
    temp.apply_control_2_qubits(mat, swapped, is_diagonal);
    break;
  case su4:
    // We reverse the order of the qubits, according to the Qiskit convention.
    // Effectively, this reverses swap for 2-qubit gates
    temp.apply_matrix_2_qubits(mat, !swapped, is_diagonal);
    break;

  default:
    throw std::invalid_argument("illegal gate for apply_2_qubit_gate");
  }

  MPS_Tensor left_gamma, right_gamma;
  rvector_t lambda;
  double discarded_value =
      MPS_Tensor::Decompose(temp, left_gamma, lambda, right_gamma);
  if (discarded_value > json_chop_threshold_)
    MPS::print_to_log("discarded_value=", discarded_value, ", ");

  if (A != 0)
    left_gamma.div_Gamma_by_left_Lambda(lambda_reg_[A - 1]);
  if (A + 1 != num_qubits_ - 1)
    right_gamma.div_Gamma_by_right_Lambda(lambda_reg_[A + 1]);

  q_reg_[A] = left_gamma;
  lambda_reg_[A] = lambda;
  q_reg_[A + 1] = right_gamma;
}

void MPS::apply_3_qubit_gate(const reg_t &qubits, Gates gate_type,
                             const cmatrix_t &mat, bool is_diagonal) {
  if (qubits.size() != 3) {
    std::stringstream ss;
    ss << "error: apply_3_qubit gate must receive 3 qubits";
    throw std::runtime_error(ss.str());
  }

  reg_t new_qubits(qubits.size());
  centralize_qubits(qubits, new_qubits);

  // extract the tensor containing only the 3 qubits on which we apply the gate
  uint_t first = new_qubits.front();
  MPS_Tensor sub_tensor(state_vec_as_MPS(first, first + 2));

  // apply the gate to sub_tensor
  switch (gate_type) {
  case ccx:
    // The controlled (or target) qubit, is qubit[2]. Since in new_qubits the
    // qubits are sorted, the relative position of the controlled qubit will be
    // 0, 1, or 2 depending on where qubit[2] was moved to in new_qubits
    uint_t target;
    if (qubits[2] > qubits[0] && qubits[2] > qubits[1])
      target = 2;
    else if (qubits[2] < qubits[0] && qubits[2] < qubits[1])
      target = 0;
    else
      target = 1;
    sub_tensor.apply_ccx(target);
    break;
  case cswap:
    uint_t control;
    if (qubits[0] < qubits[1] && qubits[0] < qubits[2])
      control = 0;
    else if (qubits[0] > qubits[1] && qubits[0] > qubits[2])
      control = 2;
    else
      control = 1;
    sub_tensor.apply_cswap(control);
    break;

  default:
    throw std::invalid_argument("illegal gate for apply_3_qubit_gate");
  }

  // state_mat is a matrix containing the flattened representation of the
  // sub-tensor into a single matrix. Note that sub_tensor will contain 8
  // matrices for 3-qubit gates. state_mat will be the concatenation of them
  // all.
  cmatrix_t state_mat = sub_tensor.get_data(0);
  for (uint_t i = 1; i < sub_tensor.get_data().size(); i++)
    state_mat = AER::Utils::concatenate(state_mat, sub_tensor.get_data(i), 1);

  // We convert the matrix back into a 3-qubit MPS structure
  MPS sub_MPS;
  sub_MPS.initialize_from_matrix(qubits.size(), state_mat);

  // copy the 3-qubit MPS back to the corresponding positions in the original
  // MPS
  for (uint_t i = 0; i < sub_MPS.num_qubits(); i++) {
    q_reg_[first + i] = sub_MPS.q_reg_[i];
  }
  lambda_reg_[first] = sub_MPS.lambda_reg_[0];
  lambda_reg_[first + 1] = sub_MPS.lambda_reg_[1];
  if (first > 0)
    q_reg_[first].div_Gamma_by_left_Lambda(lambda_reg_[first - 1]);
  if (first + 2 < num_qubits_ - 1)
    q_reg_[first + 2].div_Gamma_by_right_Lambda(lambda_reg_[first + 2]);
}

void MPS::apply_matrix(const reg_t &qubits, const cmatrix_t &mat,
                       bool is_diagonal) {
  reg_t internal_qubits = get_internal_qubits(qubits);
  apply_matrix_internal(internal_qubits, mat, is_diagonal);
}

void MPS::apply_matrix_internal(const reg_t &qubits, const cmatrix_t &mat,
                                bool is_diagonal) {
  switch (qubits.size()) {
  case 1:
    q_reg_[qubits[0]].apply_matrix(mat, is_diagonal);
    break;
  case 2:
    apply_2_qubit_gate(qubits[0], qubits[1], su4, mat, is_diagonal);
    break;
  default:
    apply_multi_qubit_gate(qubits, mat, is_diagonal);
  }
}

void MPS::apply_multi_qubit_gate(const reg_t &qubits, const cmatrix_t &mat,
                                 bool is_diagonal) {
  // bring the qubits to consecutive positions
  uint_t num_qubits = qubits.size();
  uint_t length = 1ULL << num_qubits;
  reg_t squeezed_qubits(num_qubits);
  squeeze_qubits(qubits, squeezed_qubits);

  // reverse to match the ordering in qiskit, which is the reverse of mps
  std::reverse(squeezed_qubits.begin(), squeezed_qubits.end());

  // reorder on a dummy vector to get the new ordering
  reg_t ordered_vec(length);
  std::iota(std::begin(ordered_vec), std::end(ordered_vec), 0);
  reg_t new_vec(length);
  reorder_all_qubits(ordered_vec, squeezed_qubits, new_vec);

  // change qubit order in the matrix - instead of doing swaps on the qubits
  uint_t nqubits = qubits.size();
  uint_t sidelen = 1 << nqubits;
  cmatrix_t new_mat(sidelen, sidelen);
  for (uint_t col = 0; col < sidelen; ++col) {
    for (uint_t row = 0; row < sidelen; ++row) {
      if (row == col)
        new_mat(new_vec[row], new_vec[row]) = mat(row, row);
      else
        new_mat(new_vec[row], new_vec[col]) = mat(row, col);
    }
  }

  if (is_ordered(qubits))
    apply_matrix_to_target_qubits(qubits, new_mat, is_diagonal);
  else
    apply_unordered_multi_qubit_gate(qubits, new_mat, is_diagonal);
}

void MPS::apply_unordered_multi_qubit_gate(const reg_t &qubits,
                                           const cmatrix_t &mat,
                                           bool is_diagonal) {
  reg_t new_qubits(qubits.size());
  centralize_qubits(qubits, new_qubits);
  apply_matrix_to_target_qubits(new_qubits, mat, is_diagonal);
}

void MPS::apply_matrix_to_target_qubits(const reg_t &target_qubits,
                                        const cmatrix_t &mat,
                                        bool is_diagonal) {
  uint_t num_qubits = target_qubits.size();
  uint_t first = target_qubits.front();
  MPS_Tensor sub_tensor(state_vec_as_MPS(first, first + num_qubits - 1));
  sub_tensor.apply_matrix(mat, is_diagonal);

  // state_mat is a matrix containing the flattened representation of the
  // sub-tensor into a single matrix. E.g., sub_tensor will contain 8 matrices
  // for 3-qubit gates. state_mat will be the concatenation of them all.
  cmatrix_t state_mat = sub_tensor.get_data(0);
  for (uint_t i = 1; i < sub_tensor.get_data().size(); i++)
    state_mat = AER::Utils::concatenate(state_mat, sub_tensor.get_data(i), 1);

  // We convert the matrix back into an MPS structure
  MPS sub_MPS;
  sub_MPS.initialize_from_matrix(num_qubits, state_mat);

  if (num_qubits == num_qubits_) {
    q_reg_.clear();
    q_reg_ = sub_MPS.q_reg_;
    lambda_reg_ = sub_MPS.lambda_reg_;
  } else {
    // copy the sub_MPS back to the corresponding positions in the original MPS
    for (uint_t i = 0; i < sub_MPS.num_qubits(); i++) {
      q_reg_[first + i] = sub_MPS.q_reg_[i];
    }
    for (uint_t i = 0; i < num_qubits - 1; i++) {
      lambda_reg_[first + i] = sub_MPS.lambda_reg_[i];
    }
    if (first > 0)
      q_reg_[first].div_Gamma_by_left_Lambda(lambda_reg_[first - 1]);
    if (first + num_qubits - 1 < num_qubits_ - 1)
      q_reg_[first + num_qubits - 1].div_Gamma_by_right_Lambda(
          lambda_reg_[first + num_qubits - 1]);
  }
}

void MPS::apply_diagonal_matrix(const AER::reg_t &qubits,
                                const cvector_t &vmat) {
  // converting the vector to a 1xn matrix whose first (and single) row is vmat
  uint_t dim = vmat.size();
  cmatrix_t diag_mat(1, dim);
  for (uint_t i = 0; i < dim; i++) {
    diag_mat(0, i) = vmat[i];
  }
  apply_matrix(qubits, diag_mat, true /*is_diagonal*/);
}

void MPS::apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &kmats,
                      RngEngine &rng) {
  reg_t internal_qubits = get_internal_qubits(qubits);
  apply_kraus_internal(internal_qubits, kmats, rng);
}
void MPS::apply_kraus_internal(const reg_t &qubits,
                               const std::vector<cmatrix_t> &kmats,
                               RngEngine &rng) {
  // Check edge case for empty Kraus set (this shouldn't happen)
  if (kmats.empty())
    return; // end function early
  // Choose a real in [0, 1) to choose the applied kraus operator once
  // the accumulated probability is greater than r.
  // We know that the Kraus noise must be normalized
  // So we only compute probabilities for the first N-1 kraus operators
  // and infer the probability of the last one from 1 - sum of the previous

  double r = rng.rand(0., 1.);
  double accum = 0.;
  bool complete = false;

  cmatrix_t rho = density_matrix_internal(qubits);

  cmatrix_t sq_kmat;
  double p = 0;

  // Loop through N-1 kraus operators
  for (size_t j = 0; j < kmats.size() - 1; j++) {
    sq_kmat = AER::Utils::dagger(kmats[j]) * kmats[j];
    // Calculate probability
    p = real(AER::Utils::trace(rho * sq_kmat));
    accum += p;

    // check if we need to apply this operator
    if (accum > r) {
      // rescale mat so projection is normalized
      cmatrix_t temp_mat = kmats[j] * (1 / std::sqrt(p));
      apply_matrix_internal(qubits, temp_mat);
      complete = true;
      break;
    }
  }
  // check if we haven't applied a kraus operator yet
  if (!complete) {
    // Compute probability from accumulated
    double renorm = 1 / std::sqrt(1. - accum);
    cmatrix_t temp_mat = kmats.back() * renorm;
    apply_matrix_internal(qubits, temp_mat);
  }

  uint_t min_qubit = qubits[0];
  uint_t max_qubit = qubits[0];
  for (uint_t i = qubits[0]; i < qubits.size(); i++) {
    min_qubit = std::min(min_qubit, qubits[i]);
    max_qubit = std::max(max_qubit, qubits[i]);
  }
  propagate_to_neighbors_internal(min_qubit, max_qubit, num_qubits_ - 1);
}

void MPS::centralize_qubits(const reg_t &qubits, reg_t &centralized_qubits) {
  reg_t sorted_indices;
  find_centralized_indices(qubits, sorted_indices, centralized_qubits);
  move_qubits_to_centralized_indices(sorted_indices, centralized_qubits);
}

void MPS::find_centralized_indices(const reg_t &qubits, reg_t &sorted_indices,
                                   reg_t &centralized_qubits) const {
  sorted_indices = qubits;
  uint_t num_qubits = qubits.size();

  if (num_qubits == 1) {
    centralized_qubits = qubits;
    return;
  }

  bool ordered = true;
  for (uint_t index = 0; index < num_qubits - 1; index++) {
    if (qubits[index] > qubits[index + 1]) {
      ordered = false;
      break;
    }
  }
  if (!ordered)
    sort(sorted_indices.begin(), sorted_indices.end());

  centralized_qubits = calc_new_indices(sorted_indices);
}

void MPS::move_qubits_to_centralized_indices(const reg_t &sorted_indices,
                                             const reg_t &centralized_qubits) {
  // We wish to minimize the number of swaps. Therefore we center the
  // new indices around the median
  uint_t mid_index = (centralized_qubits.size() - 1) / 2;
  for (uint_t i = mid_index; i < sorted_indices.size(); i++) {
    change_position(sorted_indices[i], centralized_qubits[i]);
  }
  for (int i = mid_index - 1; i >= 0; i--) {
    change_position(sorted_indices[i], centralized_qubits[i]);
  }
}

void MPS::move_all_qubits_to_sorted_ordering() {
  // qubit_ordering_.order_ can simply be initialized
  for (uint_t left_index = 0; left_index < num_qubits_; left_index++) {
    // find the qubit with the smallest index
    uint_t min_index = left_index;
    for (uint_t i = left_index + 1; i < num_qubits_; i++) {
      if (qubit_ordering_.order_[i] == min_index) {
        min_index = i;
        break;
      }
    }
    // Move this qubit back to its original position
    for (uint_t j = min_index; j > left_index; j--) {
      // swap the qubits until smallest reaches its original position
      apply_swap_internal(j, j - 1);
    }
  }
}

void MPS::change_position(uint_t src, uint_t dst) {
  if (src == dst)
    return;
  if (src < dst)
    for (uint_t i = src; i < dst; i++) {
      apply_swap_internal(i, i + 1, false);
    }
  else
    for (uint_t i = src; i > dst; i--) {
      apply_swap_internal(i, i - 1, false);
    }
}

cmatrix_t MPS::density_matrix(const reg_t &qubits) const {
  reg_t internal_qubits = get_internal_qubits(qubits);
  return density_matrix_internal(internal_qubits);
}

cmatrix_t MPS::density_matrix_internal(const reg_t &qubits) const {
  MPS temp_MPS;
  temp_MPS.initialize(*this);
  MPS_Tensor psi = temp_MPS.state_vec_as_MPS(qubits);
  uint_t size = psi.get_dim();
  cmatrix_t rho(size, size);

  // We do the reordering of qubits on a dummy vector in order to not do the
  // reordering on psi, since psi is a vector of matrices and this would be more
  // costly in performance
  reg_t ordered_vector(size), temp_vector(size), actual_vec(size);
  std::iota(std::begin(ordered_vector), std::end(ordered_vector), 0);
  reorder_all_qubits(ordered_vector, qubits, temp_vector);
  actual_vec = reverse_all_bits(temp_vector, qubits.size());

#ifdef _WIN32
#pragma omp parallel for if (size > omp_threshold_ && omp_threads_ > 1)        \
    num_threads(omp_threads_)
#else
#pragma omp parallel for collapse(2) if (size > omp_threshold_ &&              \
                                         omp_threads_ > 1)                     \
    num_threads(omp_threads_)
#endif

  for (int_t i = 0; i < static_cast<int_t>(size); i++) {
    for (int_t j = 0; j < static_cast<int_t>(size); j++) {
      rho(i, j) = AER::Utils::sum(AER::Utils::elementwise_multiplication(
          psi.get_data(actual_vec[i]),
          AER::Utils::conjugate(psi.get_data(actual_vec[j]))));
    }
  }

  return rho;
}

rvector_t MPS::diagonal_of_density_matrix(const reg_t &qubits) const {
  reg_t new_qubits;
  MPS temp_MPS;
  temp_MPS.initialize(*this);
  temp_MPS.centralize_qubits(qubits, new_qubits);

  MPS_Tensor psi =
      temp_MPS.state_vec_as_MPS(new_qubits.front(), new_qubits.back());

  uint_t size = psi.get_dim();
  rvector_t diagonal_rho(size);

  for (int_t i = 0; i < static_cast<int_t>(size); i++) {
    diagonal_rho[i] =
        real(AER::Utils::sum(AER::Utils::elementwise_multiplication(
            psi.get_data(i), AER::Utils::conjugate(psi.get_data(i)))));
  }
  return diagonal_rho;
}

void MPS::MPS_with_new_indices(const reg_t &qubits, reg_t &centralized_qubits,
                               MPS &temp_MPS) const {
  temp_MPS.initialize(*this);
  temp_MPS.centralize_qubits(qubits, centralized_qubits);
}

double MPS::expectation_value(const reg_t &qubits, const cmatrix_t &M) const {
  reg_t internal_qubits = get_internal_qubits(qubits);
  double expval = expectation_value_internal(internal_qubits, M);
  return expval;
}

double MPS::expectation_value_internal(const reg_t &qubits,
                                       const cmatrix_t &M) const {
  cmatrix_t rho;
  rho = density_matrix_internal(qubits);

  // Trace(rho*M). not using methods for efficiency
  complex_t res = 0;
  for (uint_t i = 0; i < M.GetRows(); i++)
    for (uint_t j = 0; j < M.GetRows(); j++)
      res += M(i, j) * rho(j, i);
  // Trace(rho*M). not using methods for efficiency
  return real(res);
}

//---------------------------------------------------------------
// Function: expectation_value_pauli
// Algorithm: For more details, see "The density-matrix renormalization group in
// the age of matrix
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

complex_t MPS::expectation_value_pauli(const reg_t &qubits,
                                       const std::string &matrices) const {
  reg_t internal_qubits = get_internal_qubits(qubits);

  // instead of computing the expectation value on the specified qubits,
  // we find the min and max of these qubits, and compute the expectation value
  // on all the qubits in between, inserting I matrices for those qubits
  // that were not in the original vector "qubits".
  // This enhancement was done for performance reasons
  reg_t extended_qubits = internal_qubits;

  const auto min =
      std::min_element(begin(internal_qubits), end(internal_qubits));
  const auto max =
      std::max_element(begin(internal_qubits), end(internal_qubits));
  uint_t min_qubit = *min;
  uint_t max_qubit = *max;

  // The number of qubits added  to extended_qubits
  uint_t num_Is = 0;

  // Add all the additional qubits at the end of the vector of extended_qubits
  // The I matrices are added in expectation_value_pauli_internal, after they
  // are reversed
  for (uint_t i = min_qubit; i <= max_qubit; i++) {
    auto itr = std::find(internal_qubits.begin(), internal_qubits.end(), i);
    if (itr == internal_qubits.end()) {
      extended_qubits.push_back(i);
      num_Is++;
    }
  }

  return expectation_value_pauli_internal(extended_qubits, matrices, min_qubit,
                                          max_qubit, num_Is);
}

complex_t MPS::expectation_value_pauli_internal(const reg_t &qubits,
                                                const std::string &matrices,
                                                uint_t first_index,
                                                uint_t last_index,
                                                uint_t num_Is) const {
  // when computing the expectation value. We only have to sort the pauli
  // matrices to be in the same ordering as the qubits

  // Preliminary step - reverse the order of the matrices because
  // they are ordered in reverse to that of the qubits (in the interface)
  std::string reversed_matrices = matrices;
  reverse(reversed_matrices.begin(), reversed_matrices.end());
  for (uint_t i = 0; i < num_Is; i++)
    reversed_matrices.append("I");
  // sort the paulis according to the initial ordering of the qubits
  auto sorted_matrices = sort_paulis_by_qubits(reversed_matrices, qubits);

  char gate = sorted_matrices[0];

  // Step 1 - multiply tensor of q0 by its left lambda
  MPS_Tensor left_tensor = q_reg_[first_index];

  if (first_index > 0) {
    left_tensor.mul_Gamma_by_left_Lambda(lambda_reg_[first_index - 1]);
  }

  // The last gamma must be multiplied also by its right lambda.
  // Here we handle the special case that we are calculating exp val
  // on a single qubit
  // we need to mul every gamma by its right lambda
  if (first_index == last_index && first_index < num_qubits_ - 1) {
    left_tensor.mul_Gamma_by_right_Lambda(lambda_reg_[first_index]);
  }

  // Step 2 - prepare the dagger of left_tensor
  MPS_Tensor left_tensor_dagger(AER::Utils::dagger(left_tensor.get_data(0)),
                                AER::Utils::dagger(left_tensor.get_data(1)));
  // Step 3 - Apply the gate to q0
  left_tensor.apply_pauli(gate);

  // Step 4 - contract Gamma0' with Gamma0 over dimensions a0 and i
  // Before contraction, Gamma0' has size a1 x a0 x i, Gamma0 has size i x a0 x
  // a1 result = left_contract is a matrix of size a1 x a1
  cmatrix_t final_contract;
  MPS_Tensor::contract_2_dimensions(left_tensor_dagger, left_tensor,
                                    omp_threads_, final_contract);
  for (uint_t qubit_num = first_index + 1; qubit_num <= last_index;
       qubit_num++) {
    // Step 5 - multiply next Gamma by its left lambda (same as Step 1)
    // next gamma has dimensions a0 x a1 x i
    MPS_Tensor next_gamma = q_reg_[qubit_num];
    next_gamma.mul_Gamma_by_left_Lambda(lambda_reg_[qubit_num - 1]);

    // Last qubit must be multiplied by rightmost lambda
    if (qubit_num == last_index && qubit_num < num_qubits_ - 1)
      next_gamma.mul_Gamma_by_right_Lambda(lambda_reg_[qubit_num]);

    // Step 6 - prepare the dagger of the next gamma (same as Step 2)
    // next_gamma_dagger has dimensions a1' x a0' x i
    MPS_Tensor next_gamma_dagger(AER::Utils::dagger(next_gamma.get_data(0)),
                                 AER::Utils::dagger(next_gamma.get_data(1)));

    // Step 7 - apply gate (same as Step 3)
    gate = sorted_matrices[qubit_num - first_index];
    next_gamma.apply_pauli(gate);

    // Step 8 - contract final_contract from previous stage with next gamma over
    // a1 final_contract has dimensions a1 x a1, Gamma1 has dimensions a1 x a2 x
    // i (where i=2) result is a tensor of size a1 x a2 x i
    MPS_Tensor next_contract(final_contract * next_gamma.get_data(0),
                             final_contract * next_gamma.get_data(1));

    // Step 9 - contract next_contract (a1 x a2 x i)
    // with next_gamma_dagger (i x a2 x a1) (same as Step 4)
    // here we need to contract across two dimensions: a1 and i
    // result is a matrix of size a2 x a2
    MPS_Tensor::contract_2_dimensions(next_gamma_dagger, next_contract,
                                      omp_threads_, final_contract);
  }

  // Step 10 - contract over final matrix of size aN x aN
  // We need to contract the final matrix with itself
  // Compute this by taking the trace of final_contract
  complex_t result = AER::Utils::trace(final_contract);
  return result;
}

std::ostream &MPS::print(std::ostream &out) const {
  for (uint_t i = 0; i < num_qubits_; i++) {
    out << "Gamma [" << i << "] :" << std::endl;
    q_reg_[i].print(out);
    if (i < num_qubits_ - 1) {
      out << "Lambda [" << i << "] (size = " << lambda_reg_[i].size()
          << "):" << std::endl;
      out << lambda_reg_[i] << std::endl;
    }
  }
  out << std::endl;
  return out;
}

std::vector<reg_t> MPS::get_matrices_sizes() const {
  std::vector<reg_t> result;
  for (uint_t i = 0; i < num_qubits_; i++) {
    result.push_back(q_reg_[i].get_size());
  }
  return result;
}

reg_t MPS::get_bond_dimensions() const {
  reg_t result;
  for (uint_t i = 0; i < num_qubits_ - 1; i++) {
    result.push_back(lambda_reg_[i].size());
  }
  return result;
}

uint_t MPS::get_max_bond_dimensions() const {
  uint_t max = 0;
  for (uint_t i = 0; i < num_qubits_ - 1; i++) {
    if (lambda_reg_[i].size() > max)
      max = lambda_reg_[i].size();
  }
  return max;
}

MPS_Tensor MPS::state_vec_as_MPS(const reg_t &qubits) {
  reg_t new_qubits;
  centralize_qubits(qubits, new_qubits);
  return state_vec_as_MPS(new_qubits.front(), new_qubits.back());
}

MPS_Tensor MPS::state_vec_as_MPS(uint_t first_index, uint_t last_index) const {
  MPS_Tensor temp = q_reg_[first_index];

  if (first_index != 0)
    temp.mul_Gamma_by_left_Lambda(lambda_reg_[first_index - 1]);

  // special case of a single qubit
  if ((first_index == last_index) && (last_index != num_qubits_ - 1)) {
    temp.mul_Gamma_by_right_Lambda(lambda_reg_[last_index]);
    return temp;
  }

  for (uint_t i = first_index + 1; i < last_index + 1; i++) {
    temp = MPS_Tensor::contract(temp, lambda_reg_[i - 1], q_reg_[i]);
  }
  // now temp is a tensor of 2^n matrices
  if (last_index != num_qubits_ - 1)
    temp.mul_Gamma_by_right_Lambda(lambda_reg_[last_index]);
  return temp;
}

Vector<complex_t> MPS::full_statevector() {
  reg_t qubits(num_qubits_);
  std::iota(std::begin(qubits), std::end(qubits), 0);
  reg_t internal_qubits = get_internal_qubits(qubits);
  return full_state_vector_internal(internal_qubits);
}

Vector<complex_t> MPS::full_state_vector_internal(const reg_t &qubits) {
  // mps_vec contains the state vector with the qubits in ascending order
  MPS_Tensor mps_vec = state_vec_as_MPS(qubits);

  uint_t num_qubits = qubits.size();
  uint_t length = 1ULL << num_qubits; // length = pow(2, num_qubits)
  Vector<complex_t> statevector(length, false);
  // statevector is constructed in ascending order
#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) \
    num_threads(omp_threads_)
  for (int_t i = 0; i < static_cast<int_t>(length); i++) {
    statevector[i] = mps_vec.get_data(i)(0, 0);
  }
  Vector<complex_t> temp_statevector(length, false);
  // temp_statevector will contain the statevector in the ordering defined in
  // "qubits"
  reorder_all_qubits(statevector, qubits, temp_statevector);
  // reverse to be consistent with qasm ordering
  return reverse_all_bits(temp_statevector, num_qubits);
}

Vector<complex_t> MPS::get_amplitude_vector(const reg_t &base_values) {
  uint_t num_values = base_values.size();
  std::string base_value;
  Vector<complex_t> amplitude_vector(num_values, false);

#pragma omp parallel for if (num_values > omp_threshold_ && omp_threads_ > 1)  \
    num_threads(omp_threads_)
  for (int_t i = 0; i < static_cast<int_t>(num_values); i++) {
    // Since the qubits may not be ordered, we determine the actual index
    // by the internal order of the qubits, to obtain the actual_base_value
    uint_t actual_base_value =
        reorder_qubits(qubit_ordering_.order_, base_values[i]);
    base_value = AER::Utils::int2string(actual_base_value);
    amplitude_vector[i] = get_single_amplitude(base_value);
  }
  return amplitude_vector;
}

complex_t MPS::get_single_amplitude(const std::string &base_value) {
  // We take the bits of the base value from right to left in order not to
  // expand the base values to the full width of 2^n We contract from left to
  // right because the representation in Qiskit is from left to right, i.e.,
  // 1=1000, 2=0100, ...

  int_t pos = base_value.length() - 1;
  uint_t bit = base_value[pos] == '0' ? 0 : 1;
  pos--;
  cmatrix_t temp = q_reg_[0].get_data(bit);

  for (uint_t qubit = 0; qubit < num_qubits_ - 1; qubit++) {
    if (pos >= 0)
      bit = base_value[pos] == '0' ? 0 : 1;
    else
      bit = 0;
    for (uint_t row = 0; row < temp.GetRows(); row++) {
      for (uint_t col = 0; col < temp.GetColumns(); col++) {
        temp(row, col) *= lambda_reg_[qubit][col];
      }
    }
    temp = temp * q_reg_[qubit + 1].get_data(bit);
    pos--;
  }

  return temp(0, 0);
}

void MPS::get_probabilities_vector(rvector_t &probvector,
                                   const reg_t &qubits) const {
  reg_t internal_qubits = get_internal_qubits(qubits);
  get_probabilities_vector_internal(probvector, internal_qubits);
}

void MPS::get_probabilities_vector_internal(rvector_t &probvector,
                                            const reg_t &qubits) const {
  cvector_t state_vec;
  uint_t num_qubits = qubits.size();
  uint_t size = 1ULL << num_qubits; // length = pow(2, num_qubits)
  probvector.resize(size);

  // compute the probability vector assuming the qubits are in ascending order
  rvector_t ordered_probvector = diagonal_of_density_matrix(qubits);

  // reorder the probabilities according to the specification in 'qubits'
  rvector_t temp_probvector(size);
  reorder_all_qubits(ordered_probvector, qubits, temp_probvector);

  // reverse to be consistent with qasm ordering
  probvector = reverse_all_bits(temp_probvector, num_qubits);
}

double MPS::get_prob_single_qubit_internal(uint_t qubit, uint_t outcome,
                                           cmatrix_t &mat) const {
  mat = q_reg_[qubit].get_data(outcome);
  if (qubit > 0) {
    // Multiply mat by left lambda
    for (uint_t col = 0; col < mat.GetColumns(); col++)
      for (uint_t row = 0; row < mat.GetRows(); row++)
        mat(row, col) *= lambda_reg_[qubit - 1][row];
  }
  if (qubit < num_qubits_ - 1) {
    // Multiply mat by right lambda
    for (uint_t row = 0; row < mat.GetRows(); row++)
      for (uint_t col = 0; col < mat.GetColumns(); col++)
        mat(row, col) *= lambda_reg_[qubit][col];
  }
  double prob = real(AER::Utils::sum(
      AER::Utils::elementwise_multiplication(mat, AER::Utils::conjugate(mat))));
  return prob;
}

void MPS::get_accumulated_probabilities_vector(rvector_t &acc_probvector,
                                               reg_t &index_vec,
                                               const reg_t &qubits) const {
  rvector_t probvector;
  get_probabilities_vector(probvector, qubits);
  uint_t size = probvector.size();
  uint_t j = 1;
  acc_probvector.push_back(0.0);
  for (uint_t i = 0; i < size; i++) {
    if (!Linalg::almost_equal(probvector[i], 0.0)) {
      index_vec.push_back(i);
      acc_probvector.push_back(acc_probvector[j - 1] + probvector[i]);
      j++;
    }
  }
}

uint_t binary_search(const rvector_t &acc_probvector, uint_t start, uint_t end,
                     double rnd) {
  if (start >= end - 1) {
    return start;
  }
  uint_t mid = (start + end) / 2;
  if (rnd <= acc_probvector[mid])
    return binary_search(acc_probvector, start, mid, rnd);
  else
    return binary_search(acc_probvector, mid, end, rnd);
}

double MPS::norm() const {
  reg_t qubits(num_qubits_);
  return norm(qubits);
}

double MPS::norm(const reg_t &qubits) const {
  reg_t temp_qubits = qubits;
  std::iota(std::begin(temp_qubits), std::end(temp_qubits), 0);
  double trace = 0;
  MPS temp_MPS;
  temp_MPS.initialize(*this);
  rvector_t vec = temp_MPS.diagonal_of_density_matrix(temp_qubits);
  for (uint_t i = 0; i < vec.size(); i++)
    trace += vec[i];
  return trace;
}

double MPS::norm(const reg_t &qubits, const cvector_t &vmat) const {
  return norm(qubits, AER::Utils::devectorize_matrix(vmat));
}

double MPS::norm(const reg_t &qubits, const cmatrix_t &mat) const {
  cmatrix_t norm_mat = AER::Utils::dagger(mat) * mat;
  return expectation_value(qubits, norm_mat);
}

reg_t MPS::apply_measure(const reg_t &qubits, const rvector_t &rnds) {
  // Unlike other api methods, we do not call the respective internal method
  // with internal_qubits. apply_measure_internal will take care of the
  // ordering.
  return apply_measure_internal(qubits, rnds);
}

reg_t MPS::apply_measure_internal(const reg_t &qubits, const rvector_t &rands) {
  // We begin measuring the qubits from the leftmost qubit in the internal
  // MPS structure that is in 'qubits'. We measure the qubits from left to
  // right. For every qubit, q, that is measured, we must propagate the effect
  // of its measurement to its neighbors, l and r, and then to their neighbors,
  // and so on. If r (or l) is measured next, then there is no need to propagate
  // to its next neighbor because we can propagate the effects of measuring q
  // and r together.
  // We measure the qubits in the order they appear in the MPS structure.
  // We check if r needs to be measured. If so, we simply measure it. If not, we
  // propagate the effect of measuring q all the way to the right, until
  // we reach a qubit that should be measured. Then we measure that qubit
  // and continue the propagation to the right.
  // In both cases, we propagate the effect all the way to the left, because
  // no more qubits will be measured on the left
  reg_t qubits_to_update;
  uint_t size = qubits.size();
  reg_t outcome_vector(size);

  // We sort 'qubits' according to `ordering_.order_`.
  // This means the qubits will be measured in the order they appear in the MPS
  // structure. This allows more efficient propagation of values between qubits.
  // This order will be defined in sorted_qubits.
  reg_t sub_ordering(qubits.size());
  reg_t sorted_qubits = sort_qubits_by_ordering(qubits, sub_ordering);

  uint_t next_measured_qubit = num_qubits_ - 1;
  for (uint_t i = 0; i < size; i++) {
    if (i < size - 1) {
      next_measured_qubit = sorted_qubits[i + 1];
    } else {
      next_measured_qubit = num_qubits_ - 1;
    }
    outcome_vector[i] = apply_measure_internal_single_qubit(
        sorted_qubits[i], rands[i], next_measured_qubit);
  }
  reg_t sorted_outcome_vector;
  // The values in outcome_vector are sorted to suit qubit ordering of
  // 0,1,2,...,n, because that is the ordering expected by the Aer simulator.
  sorted_outcome_vector = sort_measured_values(outcome_vector, sub_ordering);
  return sorted_outcome_vector;
}

uint_t MPS::apply_measure_internal_single_qubit(uint_t qubit, const double rnd,
                                                uint_t next_measured_qubit) {
  reg_t qubits_to_update;
  qubits_to_update.push_back(qubit);
  cmatrix_t dummy_mat;

  // compute probability for 0 or 1 result
  double prob0 = get_prob_single_qubit_internal(qubit, 0, dummy_mat);
  double prob1 = 1 - prob0;
  uint_t measurement;
  cmatrix_t measurement_matrix(2, 2);

  if (rnd < prob0) {
    measurement = 0;
    measurement_matrix = zero_measure;
    measurement_matrix = measurement_matrix * (1 / sqrt(prob0));
  } else {
    measurement = 1;
    measurement_matrix = one_measure;
    measurement_matrix = measurement_matrix * (1 / sqrt(prob1));
  }
  apply_matrix_internal(qubits_to_update, measurement_matrix);

  if (num_qubits_ > 1)
    propagate_to_neighbors_internal(qubit, qubit, next_measured_qubit);

  return measurement;
}

void MPS::propagate_to_neighbors_internal(uint_t min_qubit, uint_t max_qubit,
                                          uint_t next_measured_qubit) {
  // propagate the changes to all qubits to the right
  for (uint_t i = max_qubit; i < next_measured_qubit; i++) {
    if (lambda_reg_[i].size() == 1)
      break; // no need to propagate if no entanglement
    apply_2_qubit_gate(i, i + 1, id, cmatrix_t(1, 1));
  }
  // propagate the changes to all qubits to the left
  for (int_t i = min_qubit; i > 0; i--) {
    if (lambda_reg_[i - 1].size() == 1)
      break; // no need to propagate if no entanglement
    apply_2_qubit_gate(i - 1, i, id, cmatrix_t(1, 1));
  }
}

// Here is an example to demonstrate what the following method does:
// Assume qubit_ordering_.order_ == [0, 3, 1, 2,] and
// we are measuring input_qubits == [0, 2, 3]
// sub_ordering will contain the same qubits as in input_qubits, but in the
// order they appear in qubit_ordering_.order_, i.e., [0, 3, 2].
// sorted_qubits will contain the indices of input_qubits within
// qubit_ordering_.order_, i.e., sorted_qubits == [0, 1, 3]
reg_t MPS::sort_qubits_by_ordering(const reg_t &input_qubits,
                                   reg_t &sub_ordering) {
  reg_t sorted_qubits(input_qubits.size());
  uint_t next = 0;
  for (uint_t i = 0; i < num_qubits_; i++) {
    for (uint_t j = 0; j < input_qubits.size(); j++) {
      if (input_qubits[j] == qubit_ordering_.order_[i]) {
        sorted_qubits[next] = i;
        sub_ordering[next] = qubit_ordering_.order_[i];
        next++;
        break;
      }
    }
  }
  return sorted_qubits;
}

// This method sorts the measurement outcomes in ascending order (0,1,2,...)
// because this is the order expected by AerSimulator. For example, If
// sub_ordering == [0,3,2,1] and input_outcome == [1000, 0100, 0010, 0001], then
// sorted_outcome = [1000, 0001, 0010, 0100].
reg_t MPS::sort_measured_values(const reg_t &input_outcome,
                                reg_t &sub_ordering) {
  reg_t sorted_outcome(input_outcome.size());
  uint_t next = 0;
  for (uint_t min_index = 0; min_index < num_qubits_; min_index++) {
    for (uint_t index = 0; index < input_outcome.size(); index++) {
      if (sub_ordering[index] == min_index) {
        sorted_outcome[next] = input_outcome[index];
        next++;
      }
    }
  }
  return sorted_outcome;
}

// The algorithm implemented here is based on https://arxiv.org/abs/1709.01662.
// Given a particular base value, e.g., 11010, its probability is computed by
// contracting the suitable matrices per qubit (from right to left), i.e.,
// mat(0) for qubit 0, mat(1) for qubit 1, mat(0) for qubit 2, mat(1) for qubit
// 3, mat(1) for qubit 4. We build the randomly selected base value for every
// shot as follows: For the first qubit, compute its probability for 0 and then
// randomly select
//        the measurement. 'mat' is initialized to the suitable matrix (0 or 1).
// For qubit i, we store in 'mat' the contraction of the matrices that were
// selected up to i-1.
//        We compute the probability that qubit i is 0 by contracting with
//        matrix 0. We randomly select a measurement according to this
//        probability. We then update 'mat' by contracting it with the suitable
//        matrix (0 or 1).

reg_t MPS::sample_measure(uint_t shots, RngEngine &rng) const {
  double prob = 1;
  reg_t current_measure(num_qubits_);
  cmatrix_t mat;
  rvector_t rnds(num_qubits_);
  for (uint_t i = 0; i < num_qubits_; ++i) {
    rnds[i] = rng.rand(0., 1.);
  }
  for (uint_t i = 0; i < num_qubits_; i++) {
    current_measure[i] = sample_measure_single_qubit(i, prob, rnds[i], mat);
  }
  // Rearrange internal ordering of the qubits to sorted ordering
  reg_t ordered_outcome(num_qubits_);
  for (uint_t i = 0; i < num_qubits_; i++) {
    ordered_outcome[qubit_ordering_.order_[i]] = current_measure[i];
  }
  return ordered_outcome;
}

uint_t MPS::sample_measure_single_qubit(uint_t qubit, double &prob, double rnd,
                                        cmatrix_t &mat) const {
  double prob0 = 0;
  if (qubit == 0) {
    reg_t qubits_to_update;
    qubits_to_update.push_back(qubit);
    // step 1 - measure qubit in Z basis
    double exp_val = real(expectation_value_pauli_internal(
        qubits_to_update, "Z", qubit, qubit, 0));
    // step 2 - compute probability for 0 or 1 result
    prob0 = (1 + exp_val) / 2;
  } else {
    prob0 = get_single_probability0(qubit, mat);
    prob0 /= prob;
  }
  uint_t measurement = (rnd < prob0) ? 0 : 1;
  double new_prob = (measurement == 0) ? prob0 : 1 - prob0;
  prob *= new_prob;

  // Now update mat for the next qubit
  // mat represents the accumulated product of the matrices of the current
  // measurement outcome
  if (qubit == 0) {
    mat = q_reg_[qubit].get_data(measurement);
    if (qubit != 0) // multiply mat by left lambda
      for (uint_t col = 0; col < mat.GetColumns(); col++)
        for (uint_t row = 0; row < mat.GetRows(); row++)
          mat(row, col) *= lambda_reg_[qubit - 1][row];
  } else {
    mat = mat * q_reg_[qubit].get_data(measurement);
  }
  if (qubit != num_qubits_ - 1) { // multiply mat by right lambda
    for (uint_t row = 0; row < mat.GetRows(); row++)
      for (uint_t col = 0; col < mat.GetColumns(); col++)
        mat(row, col) *= lambda_reg_[qubit][col];
  }
  return measurement;
}

double MPS::get_single_probability0(uint_t qubit, const cmatrix_t &mat) const {
  // multiply by the matrix for measurement of 0
  cmatrix_t temp_mat = mat * q_reg_[qubit].get_data(0);

  if (qubit != num_qubits_ - 1) {
    for (uint_t row = 0; row < temp_mat.GetRows(); row++) {
      for (uint_t col = 0; col < temp_mat.GetColumns(); col++) {
        temp_mat(row, col) *= lambda_reg_[qubit][col];
      }
    }
  }
  // prob0 = the probability to measure 0
  double prob0 = real(AER::Utils::sum(AER::Utils::elementwise_multiplication(
      temp_mat, AER::Utils::conjugate(temp_mat))));
  return prob0;
}

void MPS::apply_initialize(const reg_t &qubits, const cvector_t &statevector,
                           RngEngine &rng) {
  uint_t num_qubits = qubits.size();
  reg_t internal_qubits = get_internal_qubits(qubits);
  uint_t num_amplitudes = statevector.size();
  cvector_t reordered_statevector(num_amplitudes);
  reg_t output_qubits(num_qubits);

  // We reorder the statevector since initialize_from_statevector_internal
  // assumes order 3,2,1,0
  for (uint_t i = 0; i < num_qubits; i++)
    output_qubits[i] = num_qubits - 1 - i;
  permute_all_qubits(statevector, internal_qubits, output_qubits,
                     reordered_statevector);

  if (num_qubits == num_qubits_)
    initialize_from_statevector_internal(internal_qubits,
                                         reordered_statevector);
  else
    initialize_component_internal(internal_qubits, reordered_statevector, rng);
}

void MPS::initialize_from_statevector_internal(const reg_t &qubits,
                                               const cvector_t &statevector) {
  uint_t num_qubits = qubits.size();
  cmatrix_t statevector_as_matrix(1, statevector.size());

#pragma omp parallel for if (num_qubits_ > MPS::get_omp_threshold() &&         \
                             MPS::get_omp_threads() > 1)                       \
    num_threads(MPS::get_omp_threads())
  for (int_t i = 0; i < static_cast<int_t>(statevector.size()); i++) {
    statevector_as_matrix(0, i) = statevector[i];
  }
  if ((1ULL << num_qubits) != statevector.size()) {
    std::stringstream ss;
    ss << "error: length of statevector should be 2^num_qubits";
    throw std::runtime_error(ss.str());
  }
  initialize_from_matrix(num_qubits, statevector_as_matrix);
}

void MPS::initialize_from_matrix(uint_t num_qubits, const cmatrix_t &mat) {
  if (!q_reg_.empty())
    q_reg_.clear();
  if (!lambda_reg_.empty())
    lambda_reg_.clear();
  qubit_ordering_.order_.clear();
  qubit_ordering_.order_.resize(num_qubits);
  std::iota(qubit_ordering_.order_.begin(), qubit_ordering_.order_.end(), 0);
  qubit_ordering_.location_.clear();
  qubit_ordering_.location_.resize(num_qubits);
  std::iota(qubit_ordering_.location_.begin(), qubit_ordering_.location_.end(),
            0);
  num_qubits_ = 0;

  if (num_qubits == 1) {
    num_qubits_ = 1;
    complex_t a = mat(0, 0);
    complex_t b = mat(0, 1);
    q_reg_.push_back(MPS_Tensor(a, b));
    return;
  }

  // remaining_matrix is the matrix that remains after each iteration
  // It is initialized to the input statevector after reshaping
  cmatrix_t remaining_matrix, reshaped_matrix;
  cmatrix_t U, V;
  rvector_t S(1.0);
  bool first_iter = true;
  for (uint_t i = 0; i < num_qubits - 1; i++) {
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
    reduce_zeros(U, S, V, MPS_Tensor::get_max_bond_dimension(),
                 MPS_Tensor::get_truncation_threshold());

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

  MPS_Tensor right_gamma(right_data[0], right_data[1]);
  q_reg_.push_back(right_gamma);
  num_qubits_++;
}

//--------------------------------------------------
// Algorithm for initialize_component:
// 1. Centralize 'qubits'
// 2. Compute the norm of 'qubits'
// 3. Normalize the values in 'statevector' to the norm computed in (3)
// 4. Create a new MPS consisting of 'qubits'
// 5. Initialize it to 'statevector'
// 6. Reset 'qubits' in *this (the original MPS) - note that this stage may
// affect
//    qubits that are not in 'qubits', if they are entangled with 'qubits'.
// 7. Cut out the old section of 'qubits' in the original MPS
// 8. Stick the new section of 'qubits' in the original MPS
//---------------------------------------------------
void MPS::initialize_component_internal(const reg_t &qubits,
                                        const cvector_t &statevector,
                                        RngEngine &rng) {
  uint_t num_qubits = qubits.size();
  uint_t num_amplitudes = statevector.size();
  reg_t new_qubits(num_qubits);
  centralize_qubits(qubits, new_qubits);

  uint_t first = new_qubits.front();
  uint_t last = new_qubits.back();
  MPS_Tensor qubits_tensor = state_vec_as_MPS(first, last);
  double qubits_norm = norm(new_qubits);

  cmatrix_t mat(1, num_amplitudes);
  for (uint_t i = 0; i < num_amplitudes; i++) {
    complex_t normalized_i = statevector[i] / qubits_norm;
    mat(0, i) = normalized_i;
  }
  // Note that new_qubits are sorted by default, since they are new qubits
  // Therefore we don't need to sort before reset_internal
  reset_internal(new_qubits, rng);
  MPS qubits_mps;
  qubits_mps.initialize_from_matrix(num_qubits, mat);
  for (uint_t i = first; i <= last; i++) {
    q_reg_[i] = qubits_mps.q_reg_[i - first];
  }
}

void MPS::reset(const reg_t &qubits, RngEngine &rng) {
  move_all_qubits_to_sorted_ordering();

  // At this point internal_qubits should actually be identical to qubits,
  // but keeping this call to be consistent with other apply_ methods
  reg_t internal_qubits = get_internal_qubits(qubits);
  reset_internal(internal_qubits, rng);
}

void MPS::reset_internal(const reg_t &qubits, RngEngine &rng) {
  rvector_t rands;
  rands.reserve(qubits.size());
  for (uint_t i = 0; i < qubits.size(); ++i)
    rands.push_back(rng.rand(0., 1.));

  // note that qubits should be sorted by the caller to this method
  // Simulate unobserved measurement
  reg_t outcome_vector = apply_measure_internal(qubits, rands);
  // Apply update to reset state
  measure_reset_update_internal(qubits, outcome_vector);
}

void MPS::measure_reset_update_internal(const reg_t &qubits,
                                        const reg_t &meas_state) {
  for (uint_t i = 0; i < qubits.size(); i++) {
    if (meas_state[i] != 0) {
      q_reg_[qubits[i]].apply_x();
    }
  }
}

mps_container_t MPS::copy_to_mps_container() {
  move_all_qubits_to_sorted_ordering();
  mps_container_t ret;
  for (uint_t i = 0; i < num_qubits(); i++) {
    ret.first.push_back(
        std::make_pair(q_reg_[i].get_data(0), q_reg_[i].get_data(1)));
  }
  for (uint_t i = 0; i < num_qubits() - 1; i++) {
    ret.second.push_back(lambda_reg_[i]);
  }
  return ret;
}

mps_container_t MPS::move_to_mps_container() {
  move_all_qubits_to_sorted_ordering();
  mps_container_t ret;
  for (uint_t i = 0; i < num_qubits(); i++) {
    ret.first.push_back(std::make_pair(std::move(q_reg_[i].get_data(0)),
                                       std::move(q_reg_[i].get_data(1))));
  }
  std::vector<std::vector<double>> lambda_vec;
  for (uint_t i = 0; i < num_qubits() - 1; i++) {
    ret.second.push_back(std::move(lambda_reg_[i]));
  }
  initialize(MPS());
  return ret;
}

void MPS::initialize_from_mps(const mps_container_t &mps) {

  uint_t num_qubits = mps.first.size();
  // clear and restart all internal structures
  q_reg_.clear();
  lambda_reg_.clear();
  q_reg_.resize(num_qubits);
  lambda_reg_.resize(num_qubits - 1);
  qubit_ordering_.order_.clear();
  qubit_ordering_.order_.resize(num_qubits);
  std::iota(qubit_ordering_.order_.begin(), qubit_ordering_.order_.end(), 0);

  qubit_ordering_.location_.clear();
  qubit_ordering_.location_.resize(num_qubits);
  std::iota(qubit_ordering_.location_.begin(), qubit_ordering_.location_.end(),
            0);

  // initialize values from mps_container_t
  for (uint_t i = 0; i < num_qubits; i++) {
    MPS_Tensor next_tensor(mps.first[i].first, mps.first[i].second);
    q_reg_[i] = std::move(next_tensor);
  }
  for (uint_t i = 0; i < num_qubits - 1; i++) {
    lambda_reg_[i] = mps.second[i];
  }
}

//-------------------------------------------------------------------------
} // namespace MatrixProductState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
