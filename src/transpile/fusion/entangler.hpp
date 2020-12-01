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

#ifndef _aer_transpile_fusion_entangler_hpp_
#define _aer_transpile_fusion_entangler_hpp_

#include <chrono>

#include "transpile/circuitopt.hpp"
#include "../fusion_method.hpp"

namespace AER {
namespace Transpile {

class EntanglerFusion {
public:
  EntanglerFusion(std::shared_ptr<FusionMethod> method_ = std::make_shared<FusionMethod>())
    : method(method_), max_qubit(64), threshold(method_->get_default_threshold_qubit()), active(true) { }

  virtual ~EntanglerFusion() {}

  void set_config(const json_t &config);

  std::string name() const { return "entangler"; };

  uint_t get_threshold() const { return threshold; }

  bool aggregate_operations(uint_t num_qubits, oplist_t& ops, const int fusion_start, const int fusion_end) const;

#ifdef DEBUG
  void dump_op_in_circuit(const oplist_t& ops, uint_t op_idx) const;
#endif

private:
  const std::shared_ptr<FusionMethod> method;
  uint_t max_qubit;
  uint_t threshold;
  bool active;
};

void EntanglerFusion::set_config(const json_t &config) {
  if (JSON::check_key("fusion_enable", config))
    JSON::get_value(active, "fusion_enable", config);
  if (JSON::check_key("fusion_enable.entangler", config))
    JSON::get_value(active, "fusion_enable.entangler", config);
}

#ifdef DEBUG
void EntanglerFusion::dump_op_in_circuit(const oplist_t& ops, uint_t op_idx) const {
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

bool EntanglerFusion::aggregate_operations(uint_t num_qubits,
                                           oplist_t& ops,
                                           const int fusion_start,
                                           const int fusion_end) const {

  if (!active)
    return false;

#ifdef DEBUG
  std::cout << "before entangler: " << std::endl;
  for (int op_idx = fusion_start; op_idx < fusion_end; ++op_idx)
    dump_op_in_circuit(ops, op_idx);
#endif

  // current impl is sensitive to ordering of gates
  for (int op_idx = fusion_start; op_idx < fusion_end; ++op_idx) {

    if (ops[op_idx].name != "CX" && ops[op_idx].name != "cx")
      continue;

    auto start_op_idx = op_idx;
    uint_t num_of_cx = 1;
    ++op_idx;
    for (; op_idx < fusion_end; ++op_idx) {
      if (ops[op_idx].name != "CX" && ops[op_idx].name != "cx")
        break;
      ++num_of_cx;
    }

    if (num_of_cx < (num_qubits * 2))
      continue;

    auto end_op_idx = op_idx;
    std::vector<uint_t> ctrl_qubits;
    std::vector<uint_t> tgt_qubits;
    std::vector<op_t> fusing_ops;
    for (op_idx = start_op_idx ; op_idx < end_op_idx; ++op_idx) {
      ctrl_qubits.push_back(ops[op_idx].qubits[0]);
      tgt_qubits.push_back(ops[op_idx].qubits[1]);
      ops[op_idx].type = optype_t::nop;
    }
    ops[start_op_idx] = Operations::make_cxlist(ctrl_qubits, tgt_qubits);
  }

#ifdef DEBUG
  std::cout << "after entangler: " << std::endl;
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
