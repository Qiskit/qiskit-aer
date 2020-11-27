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
    : method(method_), threshold(method_->get_default_threshold_qubit() + 5), active(method_->support_diagonal()) { }

  virtual ~DiagonalFusion() {}

  void set_config(const json_t &config);

  std::string name() const { return "diagonal"; };

  uint_t get_threshold() const { return threshold; }

  bool aggregate_operations(oplist_t& ops, const int fusion_start, const int fusion_end) const;

#ifdef DEBUG
  void dump_op_in_circuit(const oplist_t& ops, uint_t op_idx) const;
#endif

private:
  const std::shared_ptr<FusionMethod> method;
  uint_t threshold;
  bool active;
};

void DiagonalFusion::set_config(const json_t &config) {
  if (JSON::check_key("fusion_enable", config))
    JSON::get_value(active, "fusion_enable", config);
  if (JSON::check_key("fusion_enable.diagonal", config))
    JSON::get_value(active, "fusion_enable.diagonal", config);
  if (JSON::check_key("fusion_threshold.diagonal", config))
    JSON::get_value(threshold, "fusion_threshold.diagonal", config);
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

    std::vector<reg_t> qubits_list;
    std::vector<cvector_t> params_list;

    while(ops[op_idx].type == Operations::OpType::diagonal_matrix) {
      qubits_list.push_back(ops[op_idx].qubits);
      params_list.push_back(ops[op_idx].params);
      ++op_idx;
    }

    if (qubits_list.size() < 2)
      continue;

    for (int i = op_idx - qubits_list.size(); i < op_idx; ++i)
      ops[i].type = optype_t::nop;

    ops[op_idx] = Operations::make_multi_diagonal(qubits_list, params_list, std::string("fusion"));
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
