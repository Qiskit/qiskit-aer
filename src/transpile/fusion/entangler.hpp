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

bool EntanglerFusion::aggregate_operations(uint_t num_qubits,
                                           oplist_t& ops,
                                           const int fusion_start,
                                           const int fusion_end) const {

  if (!active)
    return false;

  uint_t num_of_cx = 0;

  for (int op_idx = fusion_start; op_idx < fusion_end; ++op_idx)
    if (ops[op_idx].name == "CX" || ops[op_idx].name == "cx")
      ++num_of_cx;

  if (num_of_cx < (fusion_end - fusion_start) /4)
    return false;

  std::vector<int> searched;

  // current impl is sensitive to ordering of gates
  for (int op_idx = fusion_start; op_idx < fusion_end; ++op_idx) {

    if (ops[op_idx].type == optype_t::nop)
      continue;

    if (ops[op_idx].name != "CX" && ops[op_idx].name != "cx")
      continue;

    if (std::find(searched.begin(), searched.end(), op_idx) != searched.end())
      continue;

    auto start_op_idx = op_idx;
    std::vector<uint_t> idx_list;
    std::vector<uint_t> entangled_qubits;
    std::vector<uint_t> opened_qubits;

    idx_list.push_back(op_idx);
    entangled_qubits.push_back(ops[op_idx].qubits[0]);
    entangled_qubits.push_back(ops[op_idx].qubits[1]);

    ++op_idx;

    for (; op_idx < fusion_end; ++op_idx) {
      if (ops[op_idx].type == optype_t::nop)
        continue;

      if (ops[op_idx].type != optype_t::gate
          && ops[op_idx].type != optype_t::matrix
          && ops[op_idx].type != optype_t::diagonal_matrix) {
        break;
      }

      if (ops[op_idx].name == "CX" || ops[op_idx].name == "cx") {
        bool direct_dependency_ctrl = std::find(opened_qubits.begin(), opened_qubits.end(), ops[op_idx].qubits[0]) == opened_qubits.end();
        bool direct_dependency_tgt = std::find(opened_qubits.begin(), opened_qubits.end(), ops[op_idx].qubits[1]) == opened_qubits.end();
        if (direct_dependency_ctrl && direct_dependency_tgt) {
          idx_list.push_back(op_idx);
          entangled_qubits.push_back(ops[op_idx].qubits[0]);
          entangled_qubits.push_back(ops[op_idx].qubits[1]);
        } else if (direct_dependency_ctrl) {
          opened_qubits.push_back(ops[op_idx].qubits[0]);
        } else if (direct_dependency_tgt) {
          opened_qubits.push_back(ops[op_idx].qubits[1]);
        }
      } else {
        bool dependency = false;
        for (auto qubit : ops[op_idx].qubits) {
          if (std::find(entangled_qubits.begin(), entangled_qubits.end(), qubit) != entangled_qubits.end()) {
            dependency = true;
            break;
          }
        }
        if (dependency)
          for (auto qubit : ops[op_idx].qubits)
            if (std::find(opened_qubits.begin(), opened_qubits.end(), qubit) == opened_qubits.end())
              opened_qubits.push_back(qubit);
      }

      if (opened_qubits.size() == num_qubits)
        break;
    }

    if (idx_list.size() < (num_qubits * 2)) {
      searched.insert(searched.end(), idx_list.begin(), idx_list.end());
    } else {
      std::vector<uint_t> ctrl_qubits;
      std::vector<uint_t> tgt_qubits;
      std::vector<op_t> fusing_ops;
      for (auto op_idx : idx_list) {
        ctrl_qubits.push_back(ops[op_idx].qubits[0]);
        tgt_qubits.push_back(ops[op_idx].qubits[1]);
        ops[op_idx].type = optype_t::nop;
      }
      ops[start_op_idx] = Operations::make_cxlist(ctrl_qubits, tgt_qubits);
    }
    op_idx = start_op_idx;
  }

  return true;
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------

#endif
