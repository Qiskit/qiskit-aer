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

#include <chrono>

#include "transpile/circuitopt.hpp"
#include "framework/avx2_detect.hpp"
#include "simulators/unitary/unitary_state.hpp"
#include "simulators/superoperator/superoperator_state.hpp"

namespace AER {
namespace Transpile {


class FusionMethod {
public:
  // Return name of method
  virtual std::string name() = 0;

  virtual bool support_diagonal() const = 0;

  // Aggregate a subcircuit of operations into a single operation
  virtual op_t generate_operation(std::vector<op_t>& fusioned_ops, bool diagonal = false) const {
    std::set<uint_t> fusioned_qubits;
    for (auto & op: fusioned_ops)
      fusioned_qubits.insert(op.qubits.begin(), op.qubits.end());

    reg_t remapped2orig(fusioned_qubits.begin(), fusioned_qubits.end());
    std::unordered_map<uint_t, uint_t> orig2remapped;
    reg_t arg_qubits;
    arg_qubits.assign(fusioned_qubits.size(), 0);
    for (size_t i = 0; i < remapped2orig.size(); i++) {
      orig2remapped[remapped2orig[i]] = i;
      arg_qubits[i] = i;
    }

    // Remap qubits
    for (auto & op: fusioned_ops)
      for (size_t i = 0; i < op.qubits.size(); i++)
        op.qubits[i] = orig2remapped[op.qubits[i]];

    auto fusioned_op = generate_operation_internal(fusioned_ops, arg_qubits);

    // Revert qubits
    for (size_t i = 0; i < fusioned_op.qubits.size(); i++)
      fusioned_op.qubits[i] = remapped2orig[fusioned_op.qubits[i]];

    if (diagonal) {
      std::vector<complex_t> vec;
      vec.assign((1UL << fusioned_op.qubits.size()), 0);
      for (size_t i = 0; i < vec.size(); ++i)
        vec[i] = fusioned_op.mats[0](i, i);
      fusioned_op = Operations::make_diagonal(fusioned_op.qubits, std::move(vec), std::string("fusion"));
    }

    return fusioned_op;
  };

  virtual op_t generate_operation_internal(const std::vector<op_t>& fusioned_ops,
                                           const reg_t &fusioned_qubits) const = 0;

  virtual bool can_apply(const op_t& op, uint_t max_fused_qubits) const = 0;

  virtual bool can_ignore(const op_t& op) const {
    switch (op.type) {
      case optype_t::barrier:
        return true;
      case optype_t::gate:
        return op.name == "id" || op.name == "u0";
      default:
        return false;
    }
  }

  static FusionMethod& find_method(const Circuit& circ,
                                  const opset_t &allowed_opset,
                                  const bool allow_superop,
                                  const bool allow_kraus);

  static bool exist_non_unitary(const std::vector<op_t>& fusioned_ops) {
    for (auto & op: fusioned_ops)
      if (noise_opset_.contains(op.type))
        return true;
    return false;
  };

private:
  const static Operations::OpSet noise_opset_;
};

const Operations::OpSet FusionMethod::noise_opset_(
  {Operations::OpType::kraus,
   Operations::OpType::superop,
   Operations::OpType::reset},
  {}, {}
);

class UnitaryFusion : public FusionMethod {
public:
  virtual std::string name() override { return "unitary"; };

  virtual bool support_diagonal() const override { return true; }

  virtual op_t generate_operation_internal (const std::vector<op_t>& fusioned_ops,
                                           const reg_t &qubits) const override {
    // Run simulation
    RngEngine dummy_rng;
    ExperimentResult dummy_result;

    // Unitary simulation
    QubitUnitary::State<> unitary_simulator;
    unitary_simulator.initialize_qreg(qubits.size());
    unitary_simulator.apply_ops(fusioned_ops.cbegin(), fusioned_ops.cend(), dummy_result, dummy_rng);
    return Operations::make_unitary(qubits, unitary_simulator.qreg().move_to_matrix(),
                                    std::string("fusion"));
  };

  virtual bool can_apply(const op_t& op, uint_t max_fused_qubits) const {
    if (op.conditional)
      return false;
    switch (op.type) {
      case optype_t::matrix:
        return op.mats.size() == 1 && op.qubits.size() <= max_fused_qubits;
      case optype_t::diagonal_matrix:
        return op.qubits.size() <= max_fused_qubits;
      case optype_t::gate: {
        if (op.qubits.size() > max_fused_qubits)
          return false;
        return QubitUnitary::StateOpSet.contains_gates(op.name);
      }
      default:
        return false;
    }
  };
};

class SuperOpFusion : public UnitaryFusion {
public:
  virtual std::string name() override { return "superop"; };

  virtual bool support_diagonal() const override { return false; }

  virtual op_t generate_operation_internal(const std::vector<op_t>& fusioned_ops,
                                           const reg_t &qubits) const override {

    if (!exist_non_unitary(fusioned_ops))
      return UnitaryFusion::generate_operation_internal(fusioned_ops, qubits);

    // Run simulation
    RngEngine dummy_rng;
    ExperimentResult dummy_result;

    // For both Kraus and SuperOp method we simulate using superoperator
    // simulator
    QubitSuperoperator::State<> superop_simulator;
    superop_simulator.initialize_qreg(qubits.size());
    superop_simulator.apply_ops(fusioned_ops.cbegin(), fusioned_ops.cend(), dummy_result, dummy_rng);
    auto superop = superop_simulator.qreg().move_to_matrix();

    return Operations::make_superop(qubits, std::move(superop));
  };

  virtual bool can_apply(const op_t& op, uint_t max_fused_qubits) const {
    if (op.conditional)
      return false;
    switch (op.type) {
      case optype_t::kraus:
      case optype_t::reset:
      case optype_t::superop: {
        return op.qubits.size() <= max_fused_qubits;
      }
      case optype_t::gate: {
        if (op.qubits.size() > max_fused_qubits)
          return false;
        return QubitSuperoperator::StateOpSet.contains_gates(op.name);
      }
      default:
        return UnitaryFusion::can_apply(op, max_fused_qubits);
    }
  };
};

class KrausFusion : public UnitaryFusion {
public:
  virtual std::string name() override { return "kraus"; };

  virtual bool support_diagonal() const override { return false; }

  virtual op_t generate_operation_internal(const std::vector<op_t>& fusioned_ops,
                                           const reg_t &qubits) const override {

    if (!exist_non_unitary(fusioned_ops))
      return UnitaryFusion::generate_operation_internal(fusioned_ops, qubits);

    // Run simulation
    RngEngine dummy_rng;
    ExperimentResult dummy_result;

    // For both Kraus and SuperOp method we simulate using superoperator
    // simulator
    QubitSuperoperator::State<> superop_simulator;
    superop_simulator.initialize_qreg(qubits.size());
    superop_simulator.apply_ops(fusioned_ops.cbegin(), fusioned_ops.cend(), dummy_result, dummy_rng);
    auto superop = superop_simulator.qreg().move_to_matrix();

    // If Kraus method we convert superop to canonical Kraus representation
    size_t dim = 1 << qubits.size();
    return Operations::make_kraus(qubits, Utils::superop2kraus(superop, dim));
  };

  virtual bool can_apply(const op_t& op, uint_t max_fused_qubits) const {
    if (op.conditional)
      return false;
    switch (op.type) {
      case optype_t::kraus:
      case optype_t::reset:
      case optype_t::superop: {
        return op.qubits.size() <= max_fused_qubits;
      }
      case optype_t::gate: {
        if (op.qubits.size() > max_fused_qubits)
          return false;
        return QubitSuperoperator::StateOpSet.contains_gates(op.name);
      }
      default:
        return UnitaryFusion::can_apply(op, max_fused_qubits);
    }
  };
};

FusionMethod& FusionMethod::find_method(const Circuit& circ,
                                       const opset_t &allowed_opset,
                                       const bool allow_superop,
                                       const bool allow_kraus) {
  static UnitaryFusion unitary;
  static SuperOpFusion superOp;
  static KrausFusion kraus;

  if (allow_superop && allowed_opset.contains(optype_t::superop) &&
      (circ.opset().contains(optype_t::kraus)
       || circ.opset().contains(optype_t::superop)
       || circ.opset().contains(optype_t::reset))) {
    return superOp;
  } else if (allow_kraus && allowed_opset.contains(optype_t::kraus) &&
      (circ.opset().contains(optype_t::kraus)
       || circ.opset().contains(optype_t::superop))) {
    return kraus;
  } else {
    return unitary;
  }
}

class Fuser {
public:
  virtual std::string name() const = 0;

  virtual void set_config(const json_t &config) = 0;

  virtual void set_metadata(ExperimentResult &result) const { }; //nop

  virtual bool aggregate_operations(oplist_t& ops,
                                    const int fusion_start,
                                    const int fusion_end,
                                    const uint_t max_fused_qubits,
                                    const FusionMethod& method) const = 0;

  virtual void allocate_new_operation(oplist_t& ops,
                                      const uint_t idx,
                                      const std::vector<uint_t>& fusioned_ops_idxs,
                                      const FusionMethod& method,
                                      const bool diagonal = false) const;
};

void Fuser::allocate_new_operation(oplist_t& ops,
                                   const uint_t idx,
                                   const std::vector<uint_t>& idxs,
                                   const FusionMethod& method,
                                   const bool diagonal) const {

  oplist_t fusing_ops;
  for (uint_t i: idxs)
    fusing_ops.push_back(ops[i]);
  ops[idx] = method.generate_operation(fusing_ops, diagonal);
  for (auto i: idxs)
    if (i != idx)
      ops[i].type = optype_t::nop;
}

class CostBasedFusion : public Fuser {
public:
  CostBasedFusion() {
    std::fill_n(costs_, 64, -1);
  };

  virtual std::string name() const override { return "cost_base"; };

  virtual void set_config(const json_t &config) override;

  virtual void set_metadata(ExperimentResult &result) const override;

  virtual bool aggregate_operations(oplist_t& ops,
                                    const int fusion_start,
                                    const int fusion_end,
                                    const uint_t max_fused_qubits,
                                    const FusionMethod& method) const override;

private:
  bool is_diagonal(const oplist_t& ops,
                   const uint_t from,
                   const uint_t until) const;

  double estimate_cost(const oplist_t& ops,
                       const uint_t from,
                       const uint_t until) const;

  void add_fusion_qubits(reg_t& fusion_qubits, const op_t& op) const;

private:
  bool active = true;
  double cost_factor = 1.8;
  double costs_[64];
};

template<size_t N>
class NQubitFusion : public Fuser {
public:
  NQubitFusion(): opt_name(std::to_string(N) + "_qubits"),
                  activate_prop_name("fusion_enable." + std::to_string(N) + "_qubits") {
  }

  virtual void set_config(const json_t &config) override;

  virtual std::string name() const override {
    return opt_name;
  };

  virtual bool aggregate_operations(oplist_t& ops,
                                    const int fusion_start,
                                    const int fusion_end,
                                    const uint_t max_fused_qubits,
                                    const FusionMethod& method) const override;

  bool exclude_escaped_qubits(std::vector<uint_t>& fusing_qubits,
                                const op_t& tgt_op) const;
private:
  bool active = true;
  const std::string opt_name;
  const std::string activate_prop_name;
  uint_t qubit_threshold = 5;
};

template<size_t N>
void NQubitFusion<N>::set_config(const json_t &config) {
  if (JSON::check_key("fusion_enable.n_qubits", config))
    JSON::get_value(active, "fusion_enable.n_qubits", config);

  if (JSON::check_key(activate_prop_name, config))
    JSON::get_value(active, activate_prop_name, config);
}

template<size_t N>
bool NQubitFusion<N>::exclude_escaped_qubits(std::vector<uint_t>& fusing_qubits,
                                             const op_t& tgt_op) const {
  bool included = true;
  for (const auto qubit: tgt_op.qubits)
    included &= (std::find(fusing_qubits.begin(), fusing_qubits.end(), qubit) != fusing_qubits.end());

  if (included)
    return false;

  for (const int op_qubit: tgt_op.qubits) {
    auto found = std::find(fusing_qubits.begin(), fusing_qubits.end(), op_qubit);
    if (found != fusing_qubits.end())
      fusing_qubits.erase(found);
  }
  return true;
}

template<size_t N>
bool NQubitFusion<N>::aggregate_operations(oplist_t& ops,
                                           const int fusion_start,
                                           const int fusion_end,
                                           const uint_t max_fused_qubits,
                                           const FusionMethod& method) const {
  if (!active)
    return false;

  std::vector<std::pair<uint_t, std::vector<op_t>>> targets;
  bool fused = false;

  for (uint_t op_idx = fusion_start; op_idx < fusion_end; ++op_idx) {
    // skip operations to be ignored
    if (!method.can_apply(ops[op_idx], max_fused_qubits) || ops[op_idx].type == optype_t::nop)
      continue;

    // 1. find a N-qubit operation
    if (ops[op_idx].qubits.size() != N)
      continue;

    std::vector<uint_t> fusing_op_idxs = { op_idx };

    std::vector<uint_t> fusing_qubits;
    fusing_qubits.insert(fusing_qubits.end(), ops[op_idx].qubits.begin(), ops[op_idx].qubits.end());

    // 2. fuse operations with backwarding
    for (int fusing_op_idx = op_idx - 1; fusing_op_idx >= fusion_start; --fusing_op_idx) {
      auto& tgt_op = ops[fusing_op_idx];
      if (tgt_op.type == optype_t::nop)
        continue;
      if (!method.can_apply(tgt_op, max_fused_qubits))
        break;
      // check all the qubits are in fusing_qubits
      if (!exclude_escaped_qubits(fusing_qubits, tgt_op))
        fusing_op_idxs.push_back(fusing_op_idx); // All the qubits of tgt_op are in fusing_qubits
      else if (fusing_qubits.empty())
          break;
    }

    std::reverse(fusing_op_idxs.begin(), fusing_op_idxs.end());
    fusing_qubits.clear();
    fusing_qubits.insert(fusing_qubits.end(), ops[op_idx].qubits.begin(), ops[op_idx].qubits.end());

    // 3. fuse operations with forwarding
    for (int fusing_op_idx = op_idx + 1; fusing_op_idx < fusion_end; ++fusing_op_idx) {
      auto& tgt_op = ops[fusing_op_idx];
      if (tgt_op.type == optype_t::nop)
        continue;
      if (!method.can_apply(tgt_op, max_fused_qubits))
        break;
      // check all the qubits are in fusing_qubits
      if (!exclude_escaped_qubits(fusing_qubits, tgt_op))
        fusing_op_idxs.push_back(fusing_op_idx); // All the qubits of tgt_op are in fusing_qubits
      else if (fusing_qubits.empty())
          break;
    }

    if (fusing_op_idxs.size() <= 1)
      continue;

    // 4. generate a fused operation
    allocate_new_operation(ops, op_idx, fusing_op_idxs, method, false);

    fused = true;
  }

  return fused;
}

class DiagonalFusion : public Fuser {
public:
  DiagonalFusion() = default;

  virtual ~DiagonalFusion() = default;

  virtual std::string name() const override { return "diagonal"; };

  virtual void set_config(const json_t &config) override;

  virtual bool aggregate_operations(oplist_t& ops,
                                    const int fusion_start,
                                    const int fusion_end,
                                    const uint_t max_fused_qubits,
                                    const FusionMethod& method) const override;

private:
  bool is_diagonal_op(const op_t& op) const;

  int get_next_diagonal_end(const oplist_t& ops, const int from, const int end, std::set<uint_t>& fusing_qubits) const;

  const std::shared_ptr<FusionMethod> method_;
  uint_t min_qubit = 3;
  bool active = true;
};

void DiagonalFusion::set_config(const json_t &config) {
  if (JSON::check_key("fusion_enable.diagonal", config))
    JSON::get_value(active, "fusion_enable.diagonal", config);
  if (JSON::check_key("fusion_min_qubit.diagonal", config))
    JSON::get_value(min_qubit, "fusion_min_qubit.diagonal", config);
}

bool DiagonalFusion::is_diagonal_op(const op_t& op) const {

  if (op.type == Operations::OpType::diagonal_matrix)
    return true;

  if (op.type == Operations::OpType::gate) {
    if (op.name == "p" || op.name == "cp" || op.name == "u1" || op.name == "cu1"
        || op.name == "mcu1" || op.name== "rz" || op.name== "rzz")
      return true;
    if (op.name == "u3")
      return op.params[0] == std::complex<double>(0.) && op.params[1] == std::complex<double>(0.);
    else
      return false;
  }

  return false;
}

// Returns an index in `ops` or `-1`.
// If gates from `from` to the returned index are fused to a gate, the fused gate is a diagonal gate.
// The returned index is equal or more than `from` and is lower than `end`.
// If -1 is returned, no pattern to generate a diagonal gate is identified from `from`.
int DiagonalFusion::get_next_diagonal_end(const oplist_t& ops,
                                          const int from,
                                          const int end,
                                          std::set<uint_t>& fusing_qubits) const {

  if (is_diagonal_op(ops[from])) {
    for (const auto qubit: ops[from].qubits)
      fusing_qubits.insert(qubit);
    return from;
  }

  if (ops[from].type != Operations::OpType::gate)
    return -1;

  auto pos = from;

  // find a diagonal gate that has the same lists of CX before and after it
  //      ┌───┐                                   ┌───┐
  // q_0: ┤ X ├───────────────────────────────────┤ X ├
  //      └─┬─┘┌───┐            ┌──────────┐ ┌───┐└─┬─┘
  // q_1: ──■──┤ X ├────────────┤ diagonal ├─┤ X ├──■──
  //           └─┬─┘┌──────────┐└──────────┘ └─┬─┘
  // q_2: ───────■──┤ diagonal ├───────────────■───────
  //                └──────────┘
  //        ■ [from,pos]

  // find first cx list
  for (; pos < end; ++pos)
    if (ops[from].type != Operations::OpType::gate || ops[pos].name != "cx")
      break;

  if (pos == from || pos == end)
    return -1;

  auto cx_end = pos - 1;

  //      ┌───┐                                   ┌───┐
  // q_0: ┤ X ├───────────────────────────────────┤ X ├
  //      └─┬─┘┌───┐            ┌──────────┐ ┌───┐└─┬─┘
  // q_1: ──■──┤ X ├────────────┤ diagonal ├─┤ X ├──■──
  //           └─┬─┘┌──────────┐└──────────┘ └─┬─┘
  // q_2: ───────■──┤ diagonal ├───────────────■───────
  //                └──────────┘
  //        ■ [from]     ■ [pos]
  //             ■ [cx_end]

  bool found = false;
  // find diagonals
  for (; pos < end; ++pos)
    if (is_diagonal_op(ops[pos]))
      found = true;
    else
      break;

  if (!found)
    return -1;

  if (pos == end)
    return -1;

  auto u1_end = pos;

  //      ┌───┐                                   ┌───┐
  // q_0: ┤ X ├───────────────────────────────────┤ X ├
  //      └─┬─┘┌───┐            ┌──────────┐ ┌───┐└─┬─┘
  // q_1: ──■──┤ X ├────────────┤ diagonal ├─┤ X ├──■──
  //           └─┬─┘┌──────────┐└──────────┘ └─┬─┘
  // q_2: ───────■──┤ diagonal ├───────────────■───────
  //                └──────────┘
  //        ■ [from]                           ■ [pos,u1_end]
  //             ■ [cx_end]

  // find second cx list that is the reverse of the first
  for (; pos < end; ++pos) {
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

  if (pos == end)
    return -1;

  //      ┌───┐                                   ┌───┐
  // q_0: ┤ X ├───────────────────────────────────┤ X ├
  //      └─┬─┘┌───┐            ┌──────────┐ ┌───┐└─┬─┘
  // q_1: ──■──┤ X ├────────────┤ diagonal ├─┤ X ├──■──
  //           └─┬─┘┌──────────┐└──────────┘ └─┬─┘
  // q_2: ───────■──┤ diagonal ├───────────────■───────
  //                └──────────┘
  //        ■ [from]                                ■ [pos]
  //        ■ [cx_end]                         ■ [u1_end]

  for (auto i = from; i < u1_end; ++i)
    for (const auto qubit: ops[i].qubits)
      fusing_qubits.insert(qubit);

  return pos;

}

bool DiagonalFusion::aggregate_operations(oplist_t& ops,
                                          const int fusion_start,
                                          const int fusion_end,
                                          const uint_t max_fused_qubits,
                                          const FusionMethod& method) const {

  if (!active || !method.support_diagonal())
    return false;

  // current impl is sensitive to ordering of gates
  for (int op_idx = fusion_start; op_idx < fusion_end; ++op_idx) {

    // find instructions to generate a diagonal gate from op_idx
    std::set<uint_t> checking_qubits_set;
    auto next_diagonal_end = get_next_diagonal_end(ops, op_idx, fusion_end, checking_qubits_set);

    if (next_diagonal_end < 0)
      continue;

    if (checking_qubits_set.size() > max_fused_qubits)
      continue;

    // find instructions to generate the next diagonal gates
    auto next_diagonal_start = next_diagonal_end + 1;

    while (true) {
      auto nde = get_next_diagonal_end(ops, next_diagonal_start, fusion_end, checking_qubits_set);
      if (nde < 0 || checking_qubits_set.size() > max_fused_qubits)
        break;
      next_diagonal_start = nde + 1;
    }

    if (checking_qubits_set.size() < min_qubit)
      continue;

    std::vector<uint_t> fusing_op_idxs;
    while(op_idx < next_diagonal_start) {
      fusing_op_idxs.push_back(op_idx);
      ++op_idx;
    }

    --op_idx;
    allocate_new_operation(ops, op_idx, fusing_op_idxs, method, true);
  }

  return true;
}

class Fusion : public CircuitOptimization {
public:
  // constructor
  /*
   * Fusion optimization uses following configuration options
   * - fusion_enable (bool): Enable fusion optimization in circuit optimization
   *       passes [Default: True]
   * - fusion_verbose (bool): Output gates generated in fusion optimization
   *       into metadata [Default: False]
   * - fusion_max_qubit (int): Maximum number of qubits for a operation generated
   *       in a fusion optimization [Default: 5]
   * - fusion_threshold (int): Threshold that number of qubits must be greater
   *       than to enable fusion optimization [Default: 14]
   * - fusion_cost_factor (double): a cost function to estimate an aggregate
   *       gate [Default: 1.8]
   */
  Fusion();
  
  void set_config(const json_t &config) override;

  virtual void set_parallelization(uint_t num) { parallelization_ = num; };

  virtual void set_parallelization_threshold(uint_t num) { parallel_threshold_ = num; };

  virtual void optimize_circuit(Circuit& circ,
                                Noise::NoiseModel& noise,
                                const opset_t &allowed_opset,
                                ExperimentResult &result) const override;

  // Qubit threshold for activating fusion pass
  uint_t max_qubit = 5;
  uint_t threshold = 14;

  bool verbose = false;
  bool active = true;
  bool allow_superop = false;
  bool allow_kraus = false;

  // Number of threads to fuse operations
  uint_t parallelization_ = 1;
  // Number of gates to enable parallelization
  uint_t parallel_threshold_ = 10000;

private:
  void optimize_circuit(Circuit& circ,
                        const Noise::NoiseModel& noise,
                        const opset_t &allowed_opset,
                        const uint_t ops_start,
                        const uint_t ops_end,
                        const std::shared_ptr<Fuser>& fuser,
                        const FusionMethod& method) const;

#ifdef DEBUG
  void dump(const Circuit& circuit) const {
    auto& ops = circuit.ops;
    for (uint_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
      std::cout << std::setw(3) << op_idx << ": ";
      if (ops[op_idx].type == optype_t::nop) {
        std::cout << std::setw(15) << "nop" << ": ";
      } else {
        std::cout << std::setw(15) << ops[op_idx].name << "-" << ops[op_idx].qubits.size() << ": ";
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
  }
#endif

private:
  std::vector<std::shared_ptr<Fuser>> fusers;
};

Fusion::Fusion() {
  fusers.push_back(std::make_shared<DiagonalFusion>());
  fusers.push_back(std::make_shared<NQubitFusion<1>>());
  fusers.push_back(std::make_shared<NQubitFusion<2>>());
  fusers.push_back(std::make_shared<CostBasedFusion>());
}

void Fusion::set_config(const json_t &config) {

  CircuitOptimization::set_config(config);

  if (JSON::check_key("fusion_verbose", config_))
    JSON::get_value(verbose, "fusion_verbose", config_);

  if (JSON::check_key("fusion_enable", config_))
    JSON::get_value(active, "fusion_enable", config_);

  if (JSON::check_key("fusion_max_qubit", config_))
    JSON::get_value(max_qubit, "fusion_max_qubit", config_);

  if (JSON::check_key("fusion_threshold", config_))
    JSON::get_value(threshold, "fusion_threshold", config_);

  for (std::shared_ptr<Fuser>& fuser: fusers)
    fuser->set_config(config_);

  if (JSON::check_key("fusion_allow_kraus", config))
    JSON::get_value(allow_kraus, "fusion_allow_kraus", config);

  if (JSON::check_key("fusion_allow_superop", config))
    JSON::get_value(allow_superop, "fusion_allow_superop", config);

  if (JSON::check_key("fusion_parallelization_threshold", config_))
    JSON::get_value(parallel_threshold_, "fusion_parallelization_threshold", config_);
}

void Fusion::optimize_circuit(Circuit& circ,
                              Noise::NoiseModel& noise,
                              const opset_t &allowed_opset,
                              ExperimentResult &result) const {

#ifdef DEBUG
    std::cout << "original" << std::endl;
    dump(circ);
#endif

  // Start timer
  using clock_t = std::chrono::high_resolution_clock;
  auto timer_start = clock_t::now();

  // Check if fusion should be skipped
  if (!active || !allowed_opset.contains(optype_t::matrix)) {
    result.metadata.add(false, "fusion", "enabled");
    return;
  }

  result.metadata.add(true, "fusion", "enabled");
  result.metadata.add(threshold, "fusion", "threshold");
  result.metadata.add(max_qubit, "fusion", "max_fused_qubits");

  // Check qubit threshold
  if (circ.num_qubits <= threshold || circ.ops.size() < 2) {
    result.metadata.add(false, "fusion", "applied");
    return;
  }

  // Determine fusion method
  FusionMethod& method = FusionMethod::find_method(circ, allowed_opset, allow_superop, allow_kraus);
  result.metadata.add(method.name(), "fusion", "method");

  bool applied = false;
  for (const std::shared_ptr<Fuser>& fuser: fusers) {
    fuser->set_metadata(result);

    if (circ.ops.size() < parallel_threshold_ || parallelization_ <= 1) {
      optimize_circuit(circ, noise, allowed_opset, 0, circ.ops.size(), fuser, method);
      result.metadata.add(1, "fusion", "parallelization");
    } else {
      // determine unit for each OMP thread
      int_t unit = circ.ops.size() / parallelization_;
      if (circ.ops.size() % parallelization_)
        ++unit;

#pragma omp parallel for if (parallelization_ > 1) num_threads(parallelization_)
      for (int_t i = 0; i < parallelization_; i++) {
        int_t start = unit * i;
        int_t end = std::min(start + unit, (int_t) circ.ops.size());
        optimize_circuit(circ, noise, allowed_opset, start, end, fuser, method);
      }
      result.metadata.add(parallelization_, "fusion", "parallelization");
    }

    size_t idx = 0;
    for (size_t i = 0; i < circ.ops.size(); ++i) {
      if (circ.ops[i].type != optype_t::nop) {
        if (i != idx)
          circ.ops[idx] = circ.ops[i];
        ++idx;
      }
    }

    if (idx != circ.ops.size()) {
      applied = true;
      circ.ops.erase(circ.ops.begin() + idx, circ.ops.end());
      circ.set_params();
    }

#ifdef DEBUG
    std::cout << fuser->name() << std::endl;
    dump(circ);
#endif

  }
  result.metadata.add(applied, "fusion", "applied");
  if (applied && verbose)
    result.metadata.add(circ.ops, "fusion", "output_ops");

  auto timer_stop = clock_t::now();
  result.metadata.add(std::chrono::duration<double>(timer_stop - timer_start).count(), "fusion", "time_taken");
}

void Fusion::optimize_circuit(Circuit& circ,
                              const Noise::NoiseModel& noise,
                              const opset_t &allowed_opset,
                              const uint_t ops_start,
                              const uint_t ops_end,
                              const std::shared_ptr<Fuser>& fuser,
                              const FusionMethod& method) const {

  uint_t fusion_start = ops_start;
  uint_t op_idx;
  for (op_idx = ops_start; op_idx < ops_end; ++op_idx) {
    if (method.can_ignore(circ.ops[op_idx]))
      continue;
    if (!method.can_apply(circ.ops[op_idx], max_qubit) || op_idx == (ops_end - 1)) {
      fuser->aggregate_operations(circ.ops, fusion_start, op_idx, max_qubit, method);
      fusion_start = op_idx + 1;
    }
  }
}

void CostBasedFusion::set_metadata(ExperimentResult &result) const {
  result.metadata.add(cost_factor, "fusion", "cost_factor");
}

void CostBasedFusion::set_config(const json_t &config) {

  if (JSON::check_key("fusion_cost_factor", config))
    JSON::get_value(cost_factor, "fusion_cost_factor", config);

  if (JSON::check_key("fusion_enable.cost_based", config))
    JSON::get_value(active, "fusion_enable.cost_based", config);

  for (int i = 0; i < 64; ++i) {
    auto prop_name = "fusion_cost." + std::to_string(i + 1);
    if (JSON::check_key(prop_name, config))
      JSON::get_value(costs_[i], prop_name, config);
  }
}

bool CostBasedFusion::aggregate_operations(oplist_t& ops,
                                  const int fusion_start,
                                  const int fusion_end,
                                  const uint_t max_fused_qubits,
                                  const FusionMethod& method) const {
  if (!active)
    return false;

  // costs[i]: estimated cost to execute from 0-th to i-th in original.ops
  std::vector<double> costs;
  // fusion_to[i]: best path to i-th in original.ops
  std::vector<int> fusion_to;

  // set costs and fusion_to of fusion_start
  fusion_to.push_back(fusion_start);
  costs.push_back(method.can_ignore(ops[fusion_start])? .0 : cost_factor);

  bool applied = false;
  // calculate the minimal path to each operation in the circuit
  for (int i = fusion_start + 1; i < fusion_end; ++i) {
    // init with fusion from i-th to i-th
    fusion_to.push_back(i);
    costs.push_back(costs[i - fusion_start - 1] + (method.can_ignore(ops[i])? .0 : cost_factor));

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
      std::vector<uint_t> fusing_op_idxs;
      for (int j = to; j <= i; ++j)
        fusing_op_idxs.push_back(j);
      if (!fusing_op_idxs.empty())
        allocate_new_operation(ops, i, fusing_op_idxs, method, false);
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
  if (is_diagonal(ops, from, until))
    return 1.0;

  reg_t fusion_qubits;
  for (uint_t i = from; i <= until; ++i)
    add_fusion_qubits(fusion_qubits, ops[i]);

  auto configured_cost = costs_[fusion_qubits.size() - 1];
  if (configured_cost > 0)
    return configured_cost;

  if(is_avx2_supported()){
    switch (fusion_qubits.size()) {
      case 1:
        // [[ falling through :) ]]
      case 2:
        return 1.0;
      case 3:
        return 1.1;
      case 4:
        return 3;
      default:
        return pow(cost_factor, (double) std::max(fusion_qubits.size() - 2, size_t(1)));
    }
  }
  return pow(cost_factor, (double) std::max(fusion_qubits.size() - 1, size_t(1)));
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
