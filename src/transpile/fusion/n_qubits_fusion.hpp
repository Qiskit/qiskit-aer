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

#ifndef _aer_transpile_fusion_two_hpp_
#define _aer_transpile_fusion_two_hpp_

#include <chrono>
#include <algorithm>

#include "transpile/circuitopt.hpp"
#include "framework/avx2_detect.hpp"
#include "../fusion_method.hpp"

namespace AER {
namespace Transpile {

template<size_t N>
class NQubitFusion {
public:
  NQubitFusion(std::shared_ptr<FusionMethod> method_ = std::make_shared<FusionMethod>())
    : method(method_), qubit_threshold(method_->get_default_threshold_qubit()), gate_threshold_ratio(10) { }

  virtual ~NQubitFusion() {}

  void set_config(const json_t &config);

  std::string name() const { return "two_qubit_fusion"; };

  uint_t get_threshold() const { return qubit_threshold; }

  bool aggregate_operations(uint_t num_qubits, oplist_t& ops, const int fusion_start, const int fusion_end) const;

private:
  const size_t n = N;
  const std::shared_ptr<FusionMethod> method;
  uint_t qubit_threshold;
  uint_t gate_threshold_ratio;
  bool active = true;
};

template<size_t N>
void NQubitFusion<N>::set_config(const json_t &config) {
  if (JSON::check_key("fusion_enable", config))
    JSON::get_value(active, "fusion_enable", config);

  if (JSON::check_key("fusion_enable.n_qubits", config))
    JSON::get_value(active, "fusion_enable.n_qubits", config);

  std::stringstream opt_name;
  opt_name << "fusion_enable." << N << "_qubits";

  if (JSON::check_key(opt_name.str(), config))
    JSON::get_value(active, opt_name.str(), config);

  if (JSON::check_key("fusion_threshold.n_qubits.ratio", config))
    JSON::get_value(gate_threshold_ratio, "fusion_threshold.n_qubits.ratio", config);

  opt_name.str("");
  opt_name << "fusion_threshold." << N << "_qubits.ratio";

  if (JSON::check_key(opt_name.str(), config))
    JSON::get_value(gate_threshold_ratio, opt_name.str(), config);
}

template<size_t N>
bool NQubitFusion<N>::aggregate_operations(uint_t num_qubits,
                                           oplist_t& ops,
                                           const int fusion_start,
                                           const int fusion_end) const {

  if (!active)
    return false;

  std::vector<std::pair<uint_t, optype_t>> nops;
  std::vector<std::pair<uint_t, std::vector<op_t>>> targets;
  uint_t fused = 0;

  for (uint_t op_idx = fusion_start; op_idx < fusion_end; ++op_idx) {

    if (ops[op_idx].qubits.size() != N || !method->can_apply_fusion(ops[op_idx]) || ops[op_idx].type == optype_t::nop)
      continue;

    std::vector<uint_t> fusing_op_idxs = { op_idx };

    bool backward = true;
    for (bool backward : {true, false} ) {
      std::reverse(fusing_op_idxs.begin(), fusing_op_idxs.end());
      std::vector<int> fusing_qubits;
      fusing_qubits.insert(fusing_qubits.end(), ops[op_idx].qubits.begin(), ops[op_idx].qubits.end());
      for (int fusing_op_idx = (backward? op_idx - 1: op_idx + 1);
          (backward && fusing_op_idx >= fusion_start) || (!backward && fusing_op_idx < fusion_end);
          fusing_op_idx += (backward? -1: 1)) {
        if (ops[fusing_op_idx].type == optype_t::nop)
          continue;
        if (!method->can_apply_fusion(ops[fusing_op_idx]))
          break;

        bool fused = true;
        for (const auto qubit: ops[fusing_op_idx].qubits) {
          fused &= (std::find(fusing_qubits.begin(), fusing_qubits.end(), qubit) != fusing_qubits.end());
          if (!fused)
            break;
        }

        if (fused) {
          fusing_op_idxs.push_back(fusing_op_idx);
          continue;
        }

        for (const int op_qubit: ops[fusing_op_idx].qubits)
          std::replace(fusing_qubits.begin(), fusing_qubits.end(), op_qubit, -1);

        bool search_next = false;
        for (const auto fusing_qubit: fusing_qubits)
          search_next |= (fusing_qubit != -1);
        if (!search_next)
          break;
      }
    }

    if (fusing_op_idxs.size() <= 1)
      continue;

    std::vector<op_t> fusing_ops;
    for (auto fusing_op_idx : fusing_op_idxs) {
      fusing_ops.push_back(ops[fusing_op_idx]); //copy
      nops.push_back(std::make_pair(fusing_op_idx, ops[fusing_op_idx].type));
      ops[fusing_op_idx].type = optype_t::nop;
      ++fused;
    }
    targets.push_back(std::make_pair(op_idx, fusing_ops));
    ++fused;
  }

  for (const auto& target: targets)
    ops[target.first] = method->generate_fusion_operation(target.second, ops[target.first].qubits);

  return true;
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------

#endif
