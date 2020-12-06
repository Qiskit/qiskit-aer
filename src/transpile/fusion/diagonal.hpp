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

#ifndef _aer_transpile_fusion_diagonal_merge_hpp_
#define _aer_transpile_fusion_diagonal_merge_hpp_

#include <chrono>

#include "transpile/circuitopt.hpp"
#include "framework/avx2_detect.hpp"
#include "../fusion_method.hpp"

namespace AER {
namespace Transpile {

class DiagonalMerge {
public:
  DiagonalMerge(std::shared_ptr<FusionMethod> method_ = std::make_shared<FusionMethod>())
    : method(method_), threshold(method_->get_default_threshold_qubit() + 5), active(false) { }

  virtual ~DiagonalMerge() {}

  void set_config(const json_t &config);

  std::string name() const { return "diagonal_merge"; };

  uint_t get_threshold() const { return threshold; }

  bool aggregate_operations(uint_t num_qubits, oplist_t& ops, const int fusion_start, const int fusion_end) const;

private:
  const std::shared_ptr<FusionMethod> method;
  uint_t threshold;
  bool active;
};

void DiagonalMerge::set_config(const json_t &config) {
  if (JSON::check_key("fusion_enable.diagonal_merge", config))
    JSON::get_value(active, "fusion_enable.diagonal_merge", config);
  if (JSON::check_key("fusion_threshold.diagonal_merge", config))
    JSON::get_value(threshold, "fusion_threshold.diagonal_merge", config);
}

bool DiagonalMerge::aggregate_operations(uint_t num_qubits,
                                          oplist_t& ops,
                                          const int fusion_start,
                                          const int fusion_end) const {

  if (!active)
    return false;

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

    ops[op_idx - qubits_list.size()] = Operations::make_multi_diagonal(qubits_list, params_list, std::string("fusion"));
  }

  return true;
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------

#endif
