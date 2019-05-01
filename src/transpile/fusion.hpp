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
  Fusion(uint_t max_qubit = 5, uint_t threshold = 16, double cost_factor = 2.5);

  void set_config(const json_t &config) override;

  void optimize_circuit(Circuit& circ,
                        const opset_t &opset,
                        OutputData &data) const override;

  bool can_ignore(const op_t& op) const;

  bool can_apply_fusion(const op_t& op) const;

  oplist_t aggregate(const oplist_t& buffer) const;

  void swap_cols_and_rows(const uint_t idx1,
                          const uint_t idx2,
                          cmatrix_t &mat,
                          uint_t dim) const;

  cmatrix_t sort_matrix(const reg_t &src,
                        const reg_t &sorted,
                        const cmatrix_t &mat) const;

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
  const double cost_factor_;
  bool verbose_;
  bool active_;
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
  "cz",   // Controlled-Z gate
  "swap" // SWAP gate
  // Three-qubit gates
  //"ccx"   // Controlled-CX gate (Toffoli): TODO
});

Fusion::Fusion(uint_t max_qubit, uint_t threshold, double cost_factor):
    max_qubit_(max_qubit), threshold_(threshold), cost_factor_(cost_factor),
    verbose_(false), active_(false) {
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
                              const opset_t &allowed_opset,
                              OutputData &data) const {

  if (circ.num_qubits < threshold_
      || !active_
      || allowed_opset.optypes.find(optype_t::matrix_sequence) == allowed_opset.optypes.end())
    return;

  bool ret = false;

  oplist_t optimized_ops;

  optimized_ops.clear();
  oplist_t buffer;

  for (const op_t op: circ.ops) {
    if (can_ignore(op))
      continue;
    if (!can_apply_fusion(op)) {
      if (!buffer.empty()) {
        ret = true;
        oplist_t optimized_buffer = aggregate(buffer);
        optimized_ops.insert(optimized_ops.end(), optimized_buffer.begin(), optimized_buffer.end());
        buffer.clear();
      }
      optimized_ops.push_back(op);
    } else {
      buffer.push_back(op);
    }
  }

  if (!buffer.empty()) {
    oplist_t optimized_buffer = aggregate(buffer);
    optimized_ops.insert(optimized_ops.end(), optimized_buffer.begin(), optimized_buffer.end());
    ret = true;
  }

  if (ret) {
    circ.ops = optimized_ops;
  }

  if (verbose_) {
    data.add_additional_data("metadata",
                             json_t::object({{"fusion_verbose", optimized_ops}}));
  }

#ifdef DEBUG
  dump(optimized_ops);
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

oplist_t Fusion::aggregate(const oplist_t& original) const {

  // costs[i]: estimated cost to execute from 0-th to i-th in original.ops
  std::vector<double> costs;
  // fusion_to[i]: best path to i-th in original.ops
  std::vector<int> fusion_to;

  int applied_total = 0;
  // calculate the minimal path to each operation in the circuit
  for (size_t i = 0; i < original.size(); ++i) {
    bool applied = false;

    // first, fusion from i-th to i-th
    fusion_to.push_back(i);

    // calculate the initial cost from i-th to i-th
    if (i == 0) {
      // if this is the first op, no fusion
      costs.push_back(cost_factor_);
      continue;
    }
    // otherwise, i-th cost is calculated from (i-1)-th cost
    costs.push_back(costs[i - 1] + cost_factor_);

    for (uint_t num_fusion = 2; num_fusion <= max_qubit_; ++num_fusion) {
      // calculate cost if {num_fusion}-qubit fusion is applied
      reg_t fusion_qubits;
      add_fusion_qubits(fusion_qubits, original[i]);

      for (int j = i - 1; j >= 0; --j) {
        add_fusion_qubits(fusion_qubits, original[j]);

        if (fusion_qubits.size() > num_fusion) // exceed the limit of fusion
          break;

        // calculate a new cost of (i-th) by adding
        double estimated_cost = estimate_cost(original, (uint_t) j, i) // fusion gate from j-th to i-th, and
            + (j == 0 ? 0.0 : costs[j - 1]); // cost of (j-1)-th

        // update cost
        if (estimated_cost < costs[i]) {
          costs[i] = estimated_cost;
          fusion_to[i] = j;
          applied = true;
        }
      }
    }
    if (applied)
      ++applied_total;
  }

  if (applied_total / static_cast<double> (original.size()) < 0.25)
    return original;

  // generate a new circuit with the minimal path to the last operation in the circuit
  oplist_t optimized;

  for (int i = original.size() - 1; i >= 0;) {
    int to = fusion_to[i];

    if (to == i) {
      optimized.push_back(original[i]);
    } else {
      std::vector<reg_t> regs;
      std::vector<cmatrix_t> mats;
      for (int j = to; j <= i; ++j) {
        regs.push_back(original[j].qubits);
        mats.push_back(matrix(original[j]));
      }
      optimized.push_back(Operations::make_matrix_sequence(regs, mats));
    }
    i = to - 1;
  }

  std::reverse(optimized.begin(), optimized.end());

  return optimized;
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

double Fusion::estimate_cost(const std::vector<op_t>& ops,
                             const uint_t from,
                             const uint_t until) const {
  reg_t fusion_qubits;
  for (uint_t i = from; i <= until; ++i)
    add_fusion_qubits(fusion_qubits, ops[i]);
  return pow(cost_factor_, (double) fusion_qubits.size());
}

void Fusion::add_fusion_qubits(reg_t& fusion_qubits, const op_t& op) const {
  for (uint_t i = 0; i < op.qubits.size(); ++i)
    if (find(fusion_qubits.begin(), fusion_qubits.end(), op.qubits[i]) == fusion_qubits.end())
      fusion_qubits.push_back(op.qubits[i]);
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
    throw std::runtime_error("Fusion: unexpected operation type");
  }
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------


#endif
