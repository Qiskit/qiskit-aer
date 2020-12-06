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

#ifndef _aer_transpile_cost_based_fusion_hpp_
#define _aer_transpile_cost_based_fusion_hpp_

#include <chrono>

#include "transpile/circuitopt.hpp"
#include "framework/avx2_detect.hpp"
#include "../fusion_method.hpp"

namespace AER {
namespace Transpile {

class CostBasedFusion {
public:
  // constructor
  /*
   * Cost-based fusion optimization uses following configuration options
   * - fusion_cost_factor (double): a cost function to estimate an aggregate
   *       gate [Default: 1.8]
   */
  CostBasedFusion(
      std::shared_ptr<FusionMethod> method_ = std::make_shared<FusionMethod>(),
      uint_t max_fused_qubits_ = 5,
      double cost_factor_ = 1.8):
        method(method_), max_fused_qubits(max_fused_qubits_), threshold(method_->get_default_threshold_qubit()), cost_factor(cost_factor_){ }

  virtual ~CostBasedFusion() {}

  void set_config(const json_t &config);

  std::string name() const { return "cost_based"; };

  uint_t get_threshold() const { return threshold; }

  bool aggregate_operations(uint_t num_qubits, oplist_t& ops, const int fusion_start, const int fusion_end) const;

private:
  bool can_ignore(const op_t& op) const;

  bool aggregate_operations_kernel(oplist_t& ops,
                             const int fusion_start,
                             const int fusion_end) const;

  bool is_diagonal(const oplist_t& ops,
                   const uint_t from,
                   const uint_t until) const;

  double estimate_cost(const oplist_t& ops,
                       const uint_t from,
                       const uint_t until) const;

  void add_fusion_qubits(reg_t& fusion_qubits, const op_t& op) const;

private:
  const std::shared_ptr<FusionMethod> method;
  uint_t max_fused_qubits;
  uint_t threshold;
  double cost_factor;
  double mat_costs[64];
  double diag_costs[64];
  bool active = true;
  bool debug = false;
};

void CostBasedFusion::set_config(const json_t &config) {
  if (JSON::check_key("fusion_max_qubit", config))
    JSON::get_value(max_fused_qubits, "fusion_max_qubit", config);

  if (JSON::check_key("fusion_cost_factor", config))
    JSON::get_value(cost_factor, "fusion_cost_factor", config);

  for (int i = 1; i < 64; ++i) {
    std::stringstream cost_name;
    cost_name << "fusion_cost.mat." << i;

    if (JSON::check_key(cost_name.str(), config))
      JSON::get_value(mat_costs[i], cost_name.str(), config);
    else
      mat_costs[i] = 0.;
  }

  for (int i = 1; i < 64; ++i) {
    std::stringstream cost_name;
    cost_name << "fusion_cost.diag." << i;

    if (JSON::check_key(cost_name.str(), config))
      JSON::get_value(diag_costs[i], cost_name.str(), config);
    else
      diag_costs[i] = 0.;
  }

  if (JSON::check_key("fusion_enable.cost_based", config))
    JSON::get_value(active, "fusion_enable.cost_based", config);

  if (JSON::check_key("fusion_debug.cost_based", config))
    JSON::get_value(debug, "fusion_debug.cost_based", config);

}

bool CostBasedFusion::can_ignore(const op_t& op) const {
  switch (op.type) {
    case optype_t::barrier:
      return true;
    case optype_t::gate:
      return op.name == "id" || op.name == "u0";
    default:
      return false;
  }
}

bool CostBasedFusion::aggregate_operations(uint_t num_qubits,
                                  oplist_t& ops,
                                  const int fusion_start_,
                                  const int fusion_end) const {

  if (!active)
    return false;

  int fusion_start = fusion_start_;
  bool applied = false;
  for (uint_t op_idx = fusion_start; op_idx < fusion_end; ++op_idx) {
    if (can_ignore(ops[op_idx]))
      continue;
    if (ops[op_idx].qubits.size() > max_fused_qubits
        || !method->can_apply_fusion(ops[op_idx])
        || op_idx == (fusion_end - 1)) {
      applied |= fusion_start != op_idx &&
          aggregate_operations_kernel(ops, fusion_start, op_idx);
      fusion_start = op_idx + 1;
    }
  }

  return applied;

}


bool CostBasedFusion::aggregate_operations_kernel(oplist_t& ops,
                                         const int fusion_start,
                                         const int fusion_end) const {

  // costs[i]: estimated cost to execute from 0-th to i-th in original.ops
  std::vector<double> costs;
  // fusion_to[i]: best path to i-th in original.ops
  std::vector<int> fusion_to;

  // set costs and fusion_to of fusion_start
  fusion_to.push_back(fusion_start);
  costs.push_back(estimate_cost(ops, (uint_t) fusion_start, fusion_start));
  if (debug)
    std::cout << "cost: " << fusion_start << " - " << fusion_start << " " << costs[0] << std::endl;

  bool applied = false;
  // calculate the minimal path to each operation in the circuit
  for (int i = fusion_start + 1; i < fusion_end; ++i) {
    // init with fusion from i-th to i-th
    fusion_to.push_back(i);
    costs.push_back(costs[i - fusion_start - 1] + estimate_cost(ops, (uint_t) i, i));
    if (debug)
      std::cout << "cost: " << i << " - " << i << " " << costs[costs.size() - 1] << std::endl;

    for (int num_fusion = 2; num_fusion <=  static_cast<int> (max_fused_qubits); ++num_fusion) {
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

        if (debug)
          std::cout << "cost: " << j << " - " << i << " " << estimated_cost << std::endl;

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
      std::set<uint_t> fusioned_qubits;
      for (int j = to; j <= i; ++j) {
        fusioned_ops.push_back(ops[j]);
        fusioned_qubits.insert(ops[j].qubits.cbegin(), ops[j].qubits.cend());
        ops[j].type = optype_t::nop;
      }
      if (!fusioned_ops.empty()) {
        if (method->support_diagonal() && is_diagonal(fusioned_ops, 0, fusioned_ops.size() - 1))
          ops[i] = method->generate_diagonal_fusion_operation(
              fusioned_ops,
              reg_t(fusioned_qubits.begin(), fusioned_qubits.end()));
        else
          ops[i] = method->generate_fusion_operation(
              fusioned_ops,
              reg_t(fusioned_qubits.begin(), fusioned_qubits.end()));
      }
    }
    i = to - 1;
  }
  return true;
}

//------------------------------------------------------------------------------
// Gate-swap optimized helper functions
//------------------------------------------------------------------------------

bool CostBasedFusion::is_diagonal(const std::vector<op_t>& ops,
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
    if (ops[i].name == "u1" || ops[i].name == "cu1" || ops[i].name == "cp")
      continue;
    return false;
  }
  return true;
}

double CostBasedFusion::estimate_cost(const std::vector<op_t>& ops,
                             const uint_t from,
                             const uint_t until) const {

  reg_t fusion_qubits;
  for (uint_t i = from; i <= until; ++i)
    add_fusion_qubits(fusion_qubits, ops[i]);

  auto num_of_fusion_qubits = fusion_qubits.size();

  if (!num_of_fusion_qubits)
    return 0.;

  if (is_diagonal(ops, from, until)) {

    if (diag_costs[num_of_fusion_qubits] > 0.)
      return diag_costs[num_of_fusion_qubits];

    if (num_of_fusion_qubits < 5)
      return cost_factor;

    double last_configured_cost = 1.0;
    double last_configured_qubit = 0;
    for (auto i = 1; i < num_of_fusion_qubits; ++i) {
      if (diag_costs[num_of_fusion_qubits] > 0.) {
        last_configured_cost = diag_costs[num_of_fusion_qubits];
        last_configured_qubit = i;
      }
    }
    return last_configured_cost * pow(cost_factor, (double) (fusion_qubits.size() - last_configured_qubit));
  } else {

    if (mat_costs[num_of_fusion_qubits] > 0.)
      return mat_costs[num_of_fusion_qubits];

    if (num_of_fusion_qubits < 2)
      return 1.;

    double last_configured_cost = 1.0;
    double last_configured_qubit = 0;
    for (auto i = 1; i < num_of_fusion_qubits; ++i) {
      if (mat_costs[num_of_fusion_qubits] > 0.) {
        last_configured_cost = mat_costs[num_of_fusion_qubits];
        last_configured_qubit = i;
      }
    }
    return last_configured_cost * pow(cost_factor, (double) (fusion_qubits.size() - last_configured_qubit));
  }
}

void CostBasedFusion::add_fusion_qubits(reg_t& fusion_qubits, const op_t& op) const {
  for (const auto &qubit: op.qubits){
    if (find(fusion_qubits.begin(), fusion_qubits.end(), qubit) == fusion_qubits.end()){
      fusion_qubits.push_back(qubit);
    }
  }
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------

#endif
