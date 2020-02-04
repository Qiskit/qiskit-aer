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

#ifndef _aer_transpile_fusion_hpp_
#define _aer_transpile_fusion_hpp_

#include "transpile/circuitopt.hpp"

namespace AER {
namespace Transpile {

using uint_t = uint_t;
using op_t = Operations::Op;
using optype_t = Operations::OpType;
using oplist_t = std::vector<op_t>;
using opset_t = Operations::OpSet;
using reg_t = std::vector<uint_t>;

class Fusion : public CircuitOptimization {
public:
  // constructor
  Fusion(uint_t max_qubit = 5, uint_t threshold = 20, double cost_factor = 1.8);

  /*
   * Fusion optimization uses following configuration options
   * - fusion_enable (bool): Enable fusion optimization in circuit optimization
   *       passes [Default: True]
   * - fusion_verbose (bool): Output gates generated in fusion optimization
   *       into metadata [Default: False]
   * - fusion_max_qubit (int): Maximum number of qubits for a operation generated
   *       in a fusion optimization [Default: 5]
   * - fusion_threshold (int): Threshold that number of qubits must be greater
   *       than to enable fusion optimization [Default: 20]
   * - fusion_cost_factor (double): a cost function to estimate an aggregate
   *       gate [Default: 1.8]
   */
  void set_config(const json_t &config) override;

  void optimize_circuit(Circuit& circ,
                        Noise::NoiseModel& noise,
                        const opset_t &opset,
                        ExperimentData &data) const override;

private:
  bool can_ignore(const op_t& op) const;

  bool can_apply_fusion(const op_t& op) const;

  double get_cost(const op_t& op) const;

  bool aggregate_operations(oplist_t& ops, const int fusion_start, const int fusion_end) const;

  op_t generate_fusion_operation(const std::vector<op_t>& fusioned_ops) const;

  void swap_cols_and_rows(const uint_t idx1,
                          const uint_t idx2,
                          cmatrix_t &mat,
                          uint_t dim) const;

  cmatrix_t sort_matrix(const reg_t &src,
                        const reg_t &sorted,
                        const cmatrix_t &mat) const;

  cmatrix_t expand_matrix(const reg_t& src_qubits,
                          const reg_t& dst_sorted_qubits,
                          const cmatrix_t& mat) const;

  bool is_diagonal(const oplist_t& ops,
                   const uint_t from,
                   const uint_t until) const;

  double estimate_cost(const oplist_t& ops,
                       const uint_t from,
                       const uint_t until) const;

  void add_fusion_qubits(reg_t& fusion_qubits, const op_t& op) const;

  cmatrix_t matrix(const op_t& op) const;

#ifdef DEBUG
  void dump(const Circuit& circuit) const;
#endif

  const static std::vector<std::string> supported_gates;

private:
  uint_t max_qubit_;
  uint_t threshold_;
  double cost_factor_;
  bool verbose_ = false;
  bool active_ = true;
};

const std::vector<std::string> Fusion::supported_gates({
  "id",   // Pauli-Identity gate
  "x",    // Pauli-X gate
  "y",    // Pauli-Y gate
  "z",    // Pauli-Z gate
  "s",    // Phase gate (aka sqrt(Z) gate)
  "sdg",  // Conjugate-transpose of Phase gate
  "h",    // Hadamard gate (X + Z / sqrt(2))
  "t",    // T-gate (sqrt(S))
  "tdg",  // Conjguate-transpose of T gate
  // Waltz Gates
  "u0",   // idle gate in multiples of X90
  "u1",   // zero-X90 pulse waltz gate
  "u2",   // single-X90 pulse waltz gate
  "u3",   // two X90 pulse waltz gate
  "U",    // two X90 pulse waltz gate
  // Two-qubit gates
  "CX",   // Controlled-X gate (CNOT)
  "cx",   // Controlled-X gate (CNOT)
  "cu1",  // Controlled-U1 gate
  "cz",   // Controlled-Z gate
  "swap" // SWAP gate
  // Three-qubit gates
  //"ccx"   // Controlled-CX gate (Toffoli): TODO
});

Fusion::Fusion(uint_t max_qubit, uint_t threshold, double cost_factor):
    max_qubit_(max_qubit), threshold_(threshold), cost_factor_(cost_factor) {
}

void Fusion::set_config(const json_t &config) {

  CircuitOptimization::set_config(config);

  if (JSON::check_key("fusion_verbose", config_))
    JSON::get_value(verbose_, "fusion_verbose", config_);

  if (JSON::check_key("fusion_enable", config_))
    JSON::get_value(active_, "fusion_enable", config_);

  if (JSON::check_key("fusion_max_qubit", config_))
    JSON::get_value(max_qubit_, "fusion_max_qubit", config_);

  if (JSON::check_key("fusion_threshold", config_))
    JSON::get_value(threshold_, "fusion_threshold", config_);

  if (JSON::check_key("fusion_cost_factor", config_))
    JSON::get_value(cost_factor_, "fusion_cost_factor", config_);
}


#ifdef DEBUG
void Fusion::dump(const Circuit& circuit) const {
  int idx = 0;
  for (const op_t& op : circuit.ops) {
    std::cout << "  " << idx++ << ":\t" << op.name << " " << op.qubits << std::endl;
    for (const cmatrix_t&  mat: op.mats) {
      const uint_t row = mat.GetRows();
      const uint_t column = mat.GetColumns();
      for (uint_t i = 0; i < row; ++i) {
        for (uint_t j = 0; j < column; ++j) {
          if (j == 0) std::cout << "      ";
          else std::cout << ", ";
          std::cout << mat(i, j);
        }
        std::cout << std::endl;
      }
    }
  }
}
#endif

void Fusion::optimize_circuit(Circuit& circ,
                              Noise::NoiseModel& noise,
                              const opset_t &allowed_opset,
                              ExperimentData &data) const {

  if (circ.num_qubits < threshold_ || !active_)
    return;

  bool applied = false;

  uint_t fusion_start = 0;
  for (uint_t op_idx = 0; op_idx < circ.ops.size(); ++op_idx) {
    if (can_ignore(circ.ops[op_idx]))
      continue;
    if (!can_apply_fusion(circ.ops[op_idx])) {
      applied |= fusion_start != op_idx && aggregate_operations(circ.ops, fusion_start, op_idx);
      fusion_start = op_idx + 1;
    }
  }

  if (fusion_start < circ.ops.size() && aggregate_operations(circ.ops, fusion_start, circ.ops.size()))
      applied = true;

  if (applied) {

    size_t idx = 0;
    for (size_t i = 0; i < circ.ops.size(); ++i) {
      if (circ.ops[i].name != "nop") {
        if (i != idx)
          circ.ops[idx] = circ.ops[i];
        ++idx;
      }
    }

    if (idx != circ.ops.size())
      circ.ops.erase(circ.ops.begin() + idx, circ.ops.end());

    if (verbose_)
      data.add_metadata("fusion_verbose", circ.ops);
  }

#ifdef DEBUG
  dump(circ.ops);
#endif
}

bool Fusion::can_ignore(const op_t& op) const {
  switch (op.type) {
  case optype_t::barrier:
    return true;
  case optype_t::gate:
    return op.name == "id" || op.name == "u0";
  default:
    return false;
  }
}

bool Fusion::can_apply_fusion(const op_t& op) const {
  if (op.conditional)
    return false;
  switch (op.type) {
  case optype_t::barrier:
    return false;
  case optype_t::matrix:
    return op.mats.size() == 1 && op.mats[0].size() <= 4;
  case optype_t::gate:
    return (std::find(supported_gates.begin(), supported_gates.end(), op.name) != supported_gates.end());
  case optype_t::reset:
  case optype_t::measure:
  case optype_t::bfunc:
  case optype_t::roerror:
  case optype_t::snapshot:
  case optype_t::kraus:
  default:
    return false;
  }
}

double Fusion::get_cost(const op_t& op) const {
  if (can_ignore(op))
    return .0;
  else
    return cost_factor_;
}

bool Fusion::aggregate_operations(oplist_t& ops, const int fusion_start, const int fusion_end) const {

  // costs[i]: estimated cost to execute from 0-th to i-th in original.ops
  std::vector<double> costs;
  // fusion_to[i]: best path to i-th in original.ops
  std::vector<int> fusion_to;

  // set costs and fusion_to of fusion_start
  fusion_to.push_back(fusion_start);
  costs.push_back(get_cost(ops[fusion_start]));

  bool applied = false;
  // calculate the minimal path to each operation in the circuit
  for (int i = fusion_start + 1; i < fusion_end; ++i) {
    // init with fusion from i-th to i-th
    fusion_to.push_back(i);
    costs.push_back(costs[i - fusion_start - 1] + get_cost(ops[i]));

    for (int num_fusion = 2; num_fusion <=  static_cast<int> (max_qubit_); ++num_fusion) {
      // calculate cost if {num_fusion}-qubit fusion is applied
      reg_t fusion_qubits;
      add_fusion_qubits(fusion_qubits, ops[i]);

      for (int j = i - 1; j >= fusion_start; --j) {
        add_fusion_qubits(fusion_qubits, ops[j]);

        if (static_cast<int> (fusion_qubits.size()) > num_fusion) // exceed the limit of fusion
          break;

        // calculate a new cost of (i-th) by adding
        double estimated_cost = estimate_cost(ops, (uint_t) j, i) // fusion gate from j-th to i-th, and
            + (j == 0 ? 0.0 : costs[j - 1 - fusion_start]); // cost of (j-1)-th

        // update cost
        if (estimated_cost <= costs[i - fusion_start]) {
          costs[i - fusion_start] = estimated_cost;
          fusion_to[i - fusion_start] = j;
          applied = true;
        }
      }
    }
  }

  if (!applied)
    return false;

  // generate a new circuit with the minimal path to the last operation in the circuit
  for (int i = fusion_end - 1; i >= fusion_start;) {

    int to = fusion_to[i - fusion_start];

    if (to != i) {
      std::vector<op_t> fusioned_ops;
      for (int j = to; j <= i; ++j) {
        fusioned_ops.push_back(ops[j]);
        ops[j].name = "nop";
      }
      if (!fusioned_ops.empty())
        ops[i] = generate_fusion_operation(fusioned_ops);
    }
    i = to - 1;
  }

  return true;
}

op_t Fusion::generate_fusion_operation(const std::vector<op_t>& fusioned_ops) const {

  std::vector<reg_t> regs;
  std::vector<cmatrix_t> mats;

  for (const op_t& fusioned_op: fusioned_ops) {
    regs.push_back(fusioned_op.qubits);
    mats.push_back(matrix(fusioned_op));
  }

  reg_t sorted_qubits;
  for (const reg_t& reg: regs)
    for (const uint_t qubit: reg)
      if (std::find(sorted_qubits.begin(), sorted_qubits.end(), qubit) == sorted_qubits.end())
        sorted_qubits.push_back(qubit);

  std::sort(sorted_qubits.begin(), sorted_qubits.end());

  std::vector<cmatrix_t> sorted_mats;

  for (size_t i = 0; i < regs.size(); ++i) {
    const reg_t& reg = regs[i];
    const cmatrix_t& mat = mats[i];
    sorted_mats.push_back(expand_matrix(reg, sorted_qubits, mat));
  }

  auto U = sorted_mats[0];
  const auto dim = 1ULL << sorted_qubits.size();

  for (size_t m = 1; m < sorted_mats.size(); m++) {

    cmatrix_t u_tmp(U.GetRows(), U.GetColumns());
    const cmatrix_t& u = sorted_mats[m];

    for (size_t i = 0; i < dim; ++i)
      for (size_t j = 0; j < dim; ++j)
        for (size_t k = 0; k < dim; ++k)
          u_tmp(i, j) += u(i, k) * U(k, j);

    U = u_tmp;
  }

  return Operations::make_fusion(sorted_qubits, U, fusioned_ops);
}

cmatrix_t Fusion::expand_matrix(const reg_t& src_qubits, const reg_t& dst_sorted_qubits, const cmatrix_t& mat) const {

  const auto dst_dim = 1ULL << dst_sorted_qubits.size();

  // generate a matrix for op
  cmatrix_t u(dst_dim, dst_dim);
  std::vector<bool> filled(dst_dim, false);

  if (src_qubits.size() == 1) { //1-qubit operation
    // 1. identify delta
    const auto index = std::distance(dst_sorted_qubits.begin(),
                               std::find(dst_sorted_qubits.begin(), dst_sorted_qubits.end(), src_qubits[0]));

    const auto delta = 1ULL << index;

    // 2. find vmat(0, 0) position in U
    for (uint_t i = 0; i < dst_dim; ++i) {

      if (filled[i])
        continue;

      //  3. allocate op.u to u based on u(i, i) and delta
      u(i          , (i + 0)    ) = mat(0, 0);
      u(i          , (i + delta)) = mat(0, 1);
      u((i + delta), (i + 0)    ) = mat(1, 0);
      u((i + delta), (i + delta)) = mat(1, 1);
      filled[i] = filled[i + delta] = true;
    }
  } else if (src_qubits.size() == 2) { //2-qubit operation

    reg_t sorted_src_qubits = src_qubits;
    std::sort(sorted_src_qubits.begin(), sorted_src_qubits.end());
    const cmatrix_t sorted_mat = sort_matrix(src_qubits, sorted_src_qubits, mat);

    // 1. identify low and high delta
    auto low = std::distance(dst_sorted_qubits.begin(),
                               std::find(dst_sorted_qubits.begin(), dst_sorted_qubits.end(), sorted_src_qubits[0]));

    auto high = std::distance(dst_sorted_qubits.begin(),
                                std::find(dst_sorted_qubits.begin(), dst_sorted_qubits.end(), sorted_src_qubits[1]));

    auto low_delta = 1UL << low;
    auto high_delta = 1UL << high;

    // 2. find op.u(0, 0) position in U
    for (uint_t i = 0; i < dst_dim; ++i) {
      if (filled[i])
        continue;

      //  3. allocate vmat to u based on u(i, i) and delta
      u(i                           , (i + 0)                     ) = sorted_mat(0, 0);
      u(i                           , (i + low_delta)             ) = sorted_mat(0, 1);
      u(i                           , (i + high_delta)            ) = sorted_mat(0, 2);
      u(i                           , (i + low_delta + high_delta)) = sorted_mat(0, 3);
      u((i + low_delta)             , (i + 0)                     ) = sorted_mat(1, 0);
      u((i + low_delta)             , (i + low_delta)             ) = sorted_mat(1, 1);
      u((i + low_delta)             , (i + high_delta)            ) = sorted_mat(1, 2);
      u((i + low_delta)             , (i + low_delta + high_delta)) = sorted_mat(1, 3);
      u((i + high_delta)            , (i + 0)                     ) = sorted_mat(2, 0);
      u((i + high_delta)            , (i + low_delta)             ) = sorted_mat(2, 1);
      u((i + high_delta)            , (i + high_delta)            ) = sorted_mat(2, 2);
      u((i + high_delta)            , (i + low_delta + high_delta)) = sorted_mat(2, 3);
      u((i + low_delta + high_delta), (i + 0)                     ) = sorted_mat(3, 0);
      u((i + low_delta + high_delta), (i + low_delta)             ) = sorted_mat(3, 1);
      u((i + low_delta + high_delta), (i + high_delta)            ) = sorted_mat(3, 2);
      u((i + low_delta + high_delta), (i + low_delta + high_delta)) = sorted_mat(3, 3);

      filled[i] = true;
      filled[i + low_delta] = true;
      filled[i + high_delta] = true;
      filled[i + low_delta + high_delta] = true;
    }
    //TODO: } else if (src_qubits.size() == 3) {
  } else {
    throw std::runtime_error("Fusion::illegal qubit number: " + std::to_string(src_qubits.size()));
  }

  return u;
}

//------------------------------------------------------------------------------
// Gate-swap optimized helper functions
//------------------------------------------------------------------------------
void Fusion::swap_cols_and_rows(const uint_t idx1, const uint_t idx2,
                                cmatrix_t &mat, uint_t dim) const {

  uint_t mask1 = (1UL << idx1);
  uint_t mask2 = (1UL << idx2);

  for (uint_t first = 0; first < dim; ++first) {
    if ((first & mask1) && !(first & mask2)) {
      uint_t second = (first ^ mask1) | mask2;

      for (uint_t i = 0; i < dim; ++i) {
        complex_t cache = mat(first, i);
        mat(first, i) = mat(second, i);
        mat(second, i) = cache;
      }
      for (uint_t i = 0; i < dim; ++i) {
        complex_t cache = mat(i, first);
        mat(i, first) = mat(i, second);
        mat(i, second) = cache;
      }
    }
  }
}

cmatrix_t Fusion::sort_matrix(const reg_t &src,
                              const reg_t &sorted,
                              const cmatrix_t &mat) const {

  const uint_t dim = mat.GetRows();
  auto ret = mat;
  auto current = src;

  while (current != sorted) {
    uint_t from;
    uint_t to;
    for (from = 0; from < current.size(); ++from)
      if (current[from] != sorted[from])
        break;
    if (from == current.size())
      break;
    for (to = from + 1; to < current.size(); ++to)
      if (current[from] == sorted[to])
        break;
    if (to == current.size()) {
      std::stringstream ss;
      ss << "Fusion::sort_matrix we should not reach here";
      throw std::runtime_error(ss.str());
    }
    swap_cols_and_rows(from, to, ret, dim);

    uint_t cache = current[from];
    current[from] = current[to];
    current[to] = cache;
  }

  return ret;
}

bool Fusion::is_diagonal(const std::vector<op_t>& ops,
                         const uint_t from,
                         const uint_t until) const {

  // check unitary matrix of ops between "from" and "to" is a diagonal matrix

  for (uint_t i = from; i <= until; ++i) {
    //   ┌───┐┌────┐┌───┐
    //  ─┤ X ├┤ U1 ├┤ X ├
    //   └─┬─┘└────┘└─┬─┘
    //  ───■──────────■─-
    if ((i + 2) <= until
        && ops[i + 0].name == "cx"
        && ops[i + 1].name == "u1"
        && ops[i + 2].name == "cx"
        && ops[i + 0].qubits[1] == ops[i + 1].qubits[0]
        && ops[i + 1].qubits[0] == ops[i + 2].qubits[1]
        && ops[i + 0].qubits[0] == ops[i + 2].qubits[0] )
    {
      i += 2;
      continue;
    }
    if (ops[i].name == "u1" || ops[i].name == "cu1")
      continue;
    return false;
  }
  return true;
}

double Fusion::estimate_cost(const std::vector<op_t>& ops,
                             const uint_t from,
                             const uint_t until) const {
  if (is_diagonal(ops, from, until))
    return cost_factor_;

  reg_t fusion_qubits;
  for (uint_t i = from; i <= until; ++i)
    add_fusion_qubits(fusion_qubits, ops[i]);
  return pow(cost_factor_, (double) std::max(fusion_qubits.size() - 1, size_t(1)));
}

void Fusion::add_fusion_qubits(reg_t& fusion_qubits, const op_t& op) const {
  for (const auto qubit: op.qubits){
    if (find(fusion_qubits.begin(), fusion_qubits.end(), qubit) == fusion_qubits.end()){
      fusion_qubits.push_back(qubit);
    }
  }
}

cmatrix_t Fusion::matrix(const op_t& op) const {
  if (op.type == optype_t::gate) {
    if (op.name == "id") {   // Pauli-Identity gate
      return Utils::Matrix::I;
    } else if (op.name == "x") {    // Pauli-X gate
      return Utils::Matrix::X;
    } else if (op.name == "y") {    // Pauli-Y gate
      return Utils::Matrix::Y;
    } else if (op.name == "z") {    // Pauli-Z gate
      return Utils::Matrix::Z;
    } else if (op.name == "s") {    // Phase gate (aka sqrt(Z) gate)
      return Utils::Matrix::S;
    } else if (op.name == "sdg") {  // Conjugate-transpose of Phase gate
      return Utils::Matrix::SDG;
    } else if (op.name == "h") {    // Hadamard gate (X + Z / sqrt(2))
      return Utils::Matrix::H;
    } else if (op.name == "t") {    // T-gate (sqrt(S))
      return Utils::Matrix::T;
    } else if (op.name == "tdg") {  // Conjguate-transpose of T gate
      return Utils::Matrix::TDG;
    } else if (op.name == "u0") {   // idle gate in multiples of X90
      return Utils::Matrix::I;
    } else if (op.name == "u1") {   // zero-X90 pulse waltz gate
      return Utils::make_matrix<complex_t>( {
        { {1, 0}, {0, 0} },
        { {0, 0}, std::exp( complex_t(0, 1.) * std::real(op.params[0])) }}
      );
    } else if (op.name == "cu1") {   // zero-X90 pulse waltz gate
      return Utils::make_matrix<complex_t>( {
        { {1, 0}, {0, 0}, {0, 0}, {0, 0} },
        { {0, 0}, {1, 0}, {0, 0}, {0, 0} },
        { {0, 0}, {0, 0}, {1, 0}, {0, 0} },
        { {0, 0}, {0, 0}, {0, 0}, std::exp( complex_t(0, 1.) * std::real(op.params[0])) }}
      );
    } else if (op.name == "u2") {   // single-X90 pulse waltz gate
      return Utils::Matrix::u3( M_PI / 2., std::real(op.params[0]), std::real(op.params[1]));
    } else if (op.name == "u3" || op.name == "U") {   // two X90 pulse waltz gate
      return Utils::Matrix::u3( std::real(op.params[0]), std::real(op.params[1]), std::real(op.params[2]));
    // Two-qubit gates
    } else if (op.name == "CX" || op.name == "cx") {   // Controlled-X gate (CNOT)
      return Utils::Matrix::CX;
    } else if (op.name == "cz") {   // Controlled-Z gate
      return Utils::Matrix::CZ;
    } else if (op.name == "swap") { // SWAP gate
      return Utils::Matrix::SWAP;
    // Three-qubit gates
//    } else if (op.name == "ccx") {   // Controlled-CX gate (Toffoli)
//      return Utils::Matrix::CCX;
    } else {
      std::stringstream msg;
      msg << "invalid operation:" << op.name << "\'.matrix()";
      throw std::runtime_error(msg.str());
    }
  } else if (op.type == optype_t::matrix) {
    return op.mats[0];
  } else {
    std::stringstream msg;
    msg << "Fusion: unexpected operation type:" << op.type;
    throw std::runtime_error(msg.str());
  }
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------


#endif
