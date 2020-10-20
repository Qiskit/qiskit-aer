/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_transpile_fusion_diagonal_hpp_
#define _aer_transpile_fusion_diagonal_hpp_

#include <chrono>

#include "transpile/circuitopt.hpp"
#include "framework/avx2_detect.hpp"
#include "../fusion_method.hpp"

namespace AER {
namespace Transpile {

class DiagonalFusion {
public:
  DiagonalFusion(std::shared_ptr<FusionMethod> method_ = std::make_shared<FusionMethod>())
    : method(method_), max_qubit(method_->get_default_max_qubit() * 2), threshold(method_->get_default_threshold_qubit() + 5), active(method_->support_diagonal()) { }

  virtual ~DiagonalFusion() {}

  void set_config(const json_t &config);

  std::string name() const { return "diagonal"; };

  uint_t get_threshold() const { return threshold; }

  bool aggregate_operations(oplist_t& ops, const int fusion_start, const int fusion_end) const;

#ifdef DEBUG
  void dump_op_in_circuit(const oplist_t& ops, uint_t op_idx) const;
#endif

private:
  bool is_diagonal_op(const op_t& op) const;

  int get_next_diagonal_end(const oplist_t& ops, const int from, std::set<uint_t>& fusing_qubits) const;

  const std::shared_ptr<FusionMethod> method;
  uint_t max_qubit;
  uint_t threshold;
  bool active;
};

void DiagonalFusion::set_config(const json_t &config) {
  if (JSON::check_key("fusion_enable", config))
    JSON::get_value(active, "fusion_enable", config);
  if (JSON::check_key("fusion_enable.diagonal", config))
    JSON::get_value(active, "fusion_enable.diagonal", config);
}

#ifdef DEBUG
void DiagonalFusion::dump_op_in_circuit(const oplist_t& ops, uint_t op_idx) const {
  std::cout << std::setw(3) << op_idx << ": ";
  if (ops[op_idx].type == optype_t::nop) {
    std::cout << std::setw(10) << "nop" << ": ";
  } else {
    std::cout << std::setw(10) << ops[op_idx].name << ": ";
    if (ops[op_idx].qubits.size() > 0) {
      auto qubits = ops[op_idx].qubits;
      std::sort(qubits.begin(), qubits.end());
      int pos = 0;
      for (int j = 0; j < qubits.size(); ++j) {
        int q_pos = 1 + qubits[j] * 2;
        for (int k = 0; k < (q_pos - pos); ++k) {
          std::cout << " ";
        }
        pos = q_pos + 1;
        std::cout << "X";
      }
    }
  }
  std::cout << std::endl;
}
#endif

bool DiagonalFusion::is_diagonal_op(const op_t& op) const {

  if (op.type == Operations::OpType::gate) {
    if (op.name == "u1" || op.name == "cu1" || op.name == "mcu1")
      return true;
    if (op.name == "u3")
      return op.params[0] == std::complex<double>(0.) && op.params[1] == std::complex<double>(0.);
    else
      return false;
  }

  if (op.type == Operations::OpType::diagonal_matrix)
    return true;

  if (op.type == Operations::OpType::matrix)
    return op.mats.size() == 1 && Utils::is_diagonal(op.mats[0], 1e-7);

  return false;
}

int DiagonalFusion::get_next_diagonal_end(const oplist_t& ops,
                                          const int from,
                                          std::set<uint_t>& fusing_qubits) const {

  if (is_diagonal_op(ops[from])) {
    for (const auto qubit: ops[from].qubits)
      fusing_qubits.insert(qubit);
    return from;
  }

  if (ops[from].type == Operations::OpType::gate) {
    auto pos = from;

    // find first cx list
    for (; pos < ops.size(); ++pos)
      if (ops[from].type != Operations::OpType::gate || ops[pos].name != "cx")
        break;

    if (pos == from || pos == ops.size())
      return -1;

    auto cx_end = pos - 1;

    bool found = false;
    // find diagonals
    for (; pos < ops.size(); ++pos)
      if (is_diagonal_op(ops[pos]))
        found = true;
      else
        break;

    if (!found)
      return -1;

    if (pos == ops.size())
      return -1;

    auto u1_end = pos;

    // find second cx list that is the reverse of the first
    for (; pos < ops.size(); ++pos) {
      if (ops[pos].type == Operations::OpType::gate
          && ops[pos].name == ops[cx_end].name
          && ops[pos].qubits == ops[cx_end].qubits) {
        if (cx_end == from)
          break;
        --cx_end;
      } else {
        return -1;
      }
    }

    for (auto i = from; i < u1_end; ++i)
      for (const auto qubit: ops[i].qubits)
        fusing_qubits.insert(qubit);

    return pos;

  } else {
    return -1;
  }

}

bool DiagonalFusion::aggregate_operations(oplist_t& ops,
                                         const int fusion_start,
                                         const int fusion_end) const {

  if (!active)
    return false;

#ifdef DEBUG
  std::cout << "before diagonal: " << std::endl;
  for (int op_idx = fusion_start; op_idx < fusion_end; ++op_idx)
    dump_op_in_circuit(ops, op_idx);
#endif

  // current impl is sensitive to ordering of gates
  for (int op_idx = fusion_start; op_idx < fusion_end; ++op_idx) {

    std::set<uint_t> checking_qubits_set;
    auto next_diagonal_end = get_next_diagonal_end(ops, op_idx, checking_qubits_set);

    if (next_diagonal_end < 0)
      continue;

    if (checking_qubits_set.size() > max_qubit)
      continue;

    std::set<uint_t> fusing_qubits_set = checking_qubits_set;
    auto next_diagonal_start = next_diagonal_end + 1;

    int cnt = 0;
    while (true) {
      auto next_diagonal_end = get_next_diagonal_end(ops, next_diagonal_start, checking_qubits_set);
      if (next_diagonal_end < 0)
        break;
      if (checking_qubits_set.size() > max_qubit)
        break;
      next_diagonal_start = next_diagonal_end + 1;
      fusing_qubits_set = checking_qubits_set;
    }

    std::vector<op_t> fusing_ops;
    for (; op_idx < next_diagonal_start; ++op_idx) {
      fusing_ops.push_back(ops[op_idx]); //copy
      ops[op_idx].type = optype_t::nop;
    }
    --op_idx;
    std::vector<uint_t> fusing_qubits;
    for (const auto q : fusing_qubits_set)
      fusing_qubits.push_back(q);
    ops[op_idx] = method->generate_diagonal_fusion_operation(fusing_ops, fusing_qubits);
  }

#ifdef DEBUG
  std::cout << "after diagonal: " << std::endl;
  for (int op_idx = fusion_start; op_idx < fusion_end; ++op_idx)
    dump_op_in_circuit(ops, op_idx);
#endif
  return true;
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------

#endif
