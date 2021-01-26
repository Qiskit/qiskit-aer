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

using uint_t = uint_t;
using op_t = Operations::Op;
using optype_t = Operations::OpType;
using oplist_t = std::vector<op_t>;
using opset_t = Operations::OpSet;
using reg_t = std::vector<uint_t>;


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
  Fusion(uint_t _max_qubit = 5, uint_t _threshold = 14, double _cost_factor = 1.8)
    : max_qubit(_max_qubit), threshold(_threshold), cost_factor(_cost_factor) {}
  
  // Allowed fusion methods:
  // - Unitary: only fuse gates into unitary instructions
  // - SuperOp: fuse gates, reset, kraus, and superops into kraus instuctions
  // - Kraus: fuse gates, reset, kraus, and superops into kraus instuctions
  enum class Method {unitary, kraus, superop};

  void set_config(const json_t &config) override;

  virtual void set_parallelization(uint_t num) { parallelization_ = num; };

  virtual void set_parallelization_threshold(uint_t num) { parallel_threshold_ = num; };

  virtual void optimize_circuit(Circuit& circ,
                                Noise::NoiseModel& noise,
                                const opset_t &allowed_opset,
                                ExperimentResult &result) const override;

  // Qubit threshold for activating fusion pass
  uint_t max_qubit;
  uint_t threshold;
  double cost_factor;
  bool verbose = false;
  bool active = true;
  bool allow_superop = false;
  bool allow_kraus = false;

  // Number of threads to fuse operations
  uint_t parallelization_ = 1;
  // Number of gates to enable parallelization
  uint_t parallel_threshold_ = 10000;

private:
  bool can_ignore(const op_t& op) const;

  bool can_apply_fusion(const op_t& op,
                        uint_t max_max_fused_qubits,
                        Method method) const;

  double get_cost(const op_t& op) const;

  void optimize_circuit(Circuit& circ,
                        Noise::NoiseModel& noise,
                        const opset_t &allowed_opset,
                        uint_t ops_start,
                        uint_t ops_end) const;

  bool aggregate_operations(oplist_t& ops,
                            const int fusion_start,
                            const int fusion_end,
                            uint_t max_fused_qubits,
                            Method method) const;

  // Aggregate a subcircuit of operations into a single operation
  op_t generate_fusion_operation(const std::vector<op_t>& fusioned_ops,
                                 const reg_t &num_qubits,
                                 Method method) const;

  bool is_diagonal(const oplist_t& ops,
                   const uint_t from,
                   const uint_t until) const;

  double estimate_cost(const oplist_t& ops,
                       const uint_t from,
                       const uint_t until) const;

  void add_fusion_qubits(reg_t& fusion_qubits, const op_t& op) const;

#ifdef DEBUG
  void dump(const Circuit& circuit) const;
#endif

private:
  const static Operations::OpSet noise_opset_;
};


const Operations::OpSet Fusion::noise_opset_(
  {Operations::OpType::kraus,
   Operations::OpType::superop,
   Operations::OpType::reset},
  {}, {}
);


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

  if (JSON::check_key("fusion_cost_factor", config))
    JSON::get_value(cost_factor, "fusion_cost_factor", config);
  
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
  result.metadata.add(cost_factor, "fusion", "cost_factor");
  result.metadata.add(max_qubit, "fusion", "max_fused_qubits");

  // Check qubit threshold
  if (circ.num_qubits <= threshold || circ.ops.size() < 2) {
    result.metadata.add(false, "fusion", "applied");
    return;
  }
  // Determine fusion method
  // TODO: Support Kraus fusion method
  Method method = Method::unitary;
  if (allow_superop && allowed_opset.contains(optype_t::superop) &&
      (circ.opset().contains(optype_t::kraus)
       || circ.opset().contains(optype_t::superop)
       || circ.opset().contains(optype_t::reset))) {
    method = Method::superop;
  } else if (allow_kraus && allowed_opset.contains(optype_t::kraus) &&
      (circ.opset().contains(optype_t::kraus)
       || circ.opset().contains(optype_t::superop))) {
    method = Method::kraus;
  }
  if (method == Method::unitary) {
    result.metadata.add("unitary", "fusion", "method");
  } else if (method == Method::superop) {
    result.metadata.add("superop", "fusion", "method");
  } else if (method == Method::kraus) {
    result.metadata.add("kraus", "fusion", "method");
  }

  if (circ.ops.size() < parallel_threshold_ || parallelization_ <= 1) {
    optimize_circuit(circ, noise, allowed_opset, 0, circ.ops.size());
  } else {
    // determine unit for each OMP thread
    int_t unit = circ.ops.size() / parallelization_;
    if (circ.ops.size() % parallelization_)
      ++unit;

#pragma omp parallel for if (parallelization_ > 1) num_threads(parallelization_)
    for (int_t i = 0; i < parallelization_; i++) {
      int_t start = unit * i;
      int_t end = std::min(start + unit, (int_t) circ.ops.size());
      optimize_circuit(circ, noise, allowed_opset, start, end);
    }
  }

  result.metadata.add(parallelization_, "fusion", "parallelization");

  auto timer_stop = clock_t::now();
  result.metadata.add(std::chrono::duration<double>(timer_stop - timer_start).count(), "fusion", "time_taken");

  size_t idx = 0;
  for (size_t i = 0; i < circ.ops.size(); ++i) {
    if (circ.ops[i].type != optype_t::nop) {
      if (i != idx)
        circ.ops[idx] = circ.ops[i];
      ++idx;
    }
  }

  if (idx == circ.ops.size()) {
    result.metadata.add(false, "fusion", "applied");
  } else {
    circ.ops.erase(circ.ops.begin() + idx, circ.ops.end());
    result.metadata.add(true, "fusion", "applied");
    circ.set_params();

    if (verbose)
      result.metadata.add(circ.ops, "fusion", "output_ops");
  }
}

void Fusion::optimize_circuit(Circuit& circ,
                              Noise::NoiseModel& noise,
                              const opset_t &allowed_opset,
                              uint_t ops_start,
                              uint_t ops_end) const {

  // Determine fusion method
  // TODO: Support Kraus fusion method
  Method method = Method::unitary;
  if (allow_superop && allowed_opset.contains(optype_t::superop) &&
      (circ.opset().contains(optype_t::kraus)
       || circ.opset().contains(optype_t::superop)
       || circ.opset().contains(optype_t::reset))) {
    method = Method::superop;
  } else if (allow_kraus && allowed_opset.contains(optype_t::kraus) &&
      (circ.opset().contains(optype_t::kraus)
       || circ.opset().contains(optype_t::superop))) {
    method = Method::kraus;
  }

  uint_t fusion_start = ops_start;
  uint_t op_idx;
  for (op_idx = ops_start; op_idx < ops_end; ++op_idx) {
    if (can_ignore(circ.ops[op_idx]))
      continue;
    if (!can_apply_fusion(circ.ops[op_idx], max_qubit, method) || op_idx == (ops_end - 1)) {
      aggregate_operations(circ.ops, fusion_start, op_idx, max_qubit, method);
      fusion_start = op_idx + 1;
    }
  }
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

bool Fusion::can_apply_fusion(const op_t& op, uint_t max_fused_qubits, Method method) const {
  if (op.conditional)
    return false;
  switch (op.type) {
    case optype_t::matrix:
      return op.mats.size() == 1 && op.qubits.size() <= max_fused_qubits;
    case optype_t::kraus:
    case optype_t::reset:
    case optype_t::superop: {
      return method != Method::unitary && op.qubits.size() <= max_fused_qubits;
    }
    case optype_t::gate: {
      if (op.qubits.size() > max_fused_qubits)
        return false;
      return (method == Method::unitary)
        ? QubitUnitary::StateOpSet.contains_gates(op.name)
        : QubitSuperoperator::StateOpSet.contains_gates(op.name);
    }
    case optype_t::measure:
    case optype_t::bfunc:
    case optype_t::roerror:
    case optype_t::snapshot:
    case optype_t::barrier:
    default:
      return false;
  }
}

double Fusion::get_cost(const op_t& op) const {
  if (can_ignore(op))
    return .0;
  else
    return cost_factor;
}


op_t Fusion::generate_fusion_operation(const std::vector<op_t>& fusioned_ops,
                                       const reg_t &qubits,
                                       Method method) const {
  // Run simulation
  RngEngine dummy_rng;
  ExperimentResult dummy_result;

  if (method == Method::unitary) {
    // Unitary simulation
    QubitUnitary::State<> unitary_simulator;
    unitary_simulator.initialize_qreg(qubits.size());
    unitary_simulator.apply_ops(fusioned_ops, dummy_result, dummy_rng);
    return Operations::make_unitary(qubits, unitary_simulator.move_to_matrix(),
                                    std::string("fusion"));
  }

  // For both Kraus and SuperOp method we simulate using superoperator
  // simulator
  QubitSuperoperator::State<> superop_simulator;
  superop_simulator.initialize_qreg(qubits.size());
  superop_simulator.apply_ops(fusioned_ops, dummy_result, dummy_rng);
  auto superop = superop_simulator.move_to_matrix();

  if (method == Method::superop) {
    return Operations::make_superop(qubits, std::move(superop));
  }

  // If Kraus method we convert superop to canonical Kraus representation
  size_t dim = 1 << qubits.size();
  return Operations::make_kraus(qubits, Utils::superop2kraus(superop, dim));
}

bool Fusion::aggregate_operations(oplist_t& ops,
                                  const int fusion_start,
                                  const int fusion_end,
                                  uint_t max_fused_qubits,
                                  Method method) const {

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
      std::vector<op_t> fusioned_ops;
      std::set<uint_t> fusioned_qubits;
      for (int j = to; j <= i; ++j) {
        fusioned_ops.push_back(ops[j]);
        fusioned_qubits.insert(ops[j].qubits.cbegin(), ops[j].qubits.cend());
        ops[j].type = optype_t::nop;
      }
      if (!fusioned_ops.empty()) {
        // We need to remap qubits in fusion subcircuits for simulation
        // TODO: This could be done above during the fusion cost calculation
        reg_t qubits(fusioned_qubits.begin(), fusioned_qubits.end());
        std::unordered_map<uint_t, uint_t> qubit_mapping;
        for (size_t j = 0; j < qubits.size(); j++) {
          qubit_mapping[qubits[j]] = j;
        }
        // Remap qubits and determine method
        bool non_unitary = false;
        for (auto & op: fusioned_ops) {
          non_unitary |= noise_opset_.contains(op.type);
          for (size_t j = 0; j < op.qubits.size(); j++) {
            op.qubits[j] = qubit_mapping[op.qubits[j]];
          }
        }
        Method required_method = (non_unitary) ? method : Method::unitary;
        ops[i] = generate_fusion_operation(fusioned_ops, qubits, required_method);
      }
    }
    i = to - 1;
  }
  return true;
}

//------------------------------------------------------------------------------
// Gate-swap optimized helper functions
//------------------------------------------------------------------------------

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
    if (ops[i].name == "u1" || ops[i].name == "cu1" || ops[i].name == "cp")
      continue;
    return false;
  }
  return true;
}

double Fusion::estimate_cost(const std::vector<op_t>& ops,
                             const uint_t from,
                             const uint_t until) const {
  if (is_diagonal(ops, from, until))
    return cost_factor;

  reg_t fusion_qubits;
  for (uint_t i = from; i <= until; ++i)
    add_fusion_qubits(fusion_qubits, ops[i]);

  if(is_avx2_supported()){
    switch (fusion_qubits.size()) {
      case 1:
        // [[ falling through :) ]]
      case 2:
        return cost_factor;
      case 3:
        return cost_factor * 1.1;
      case 4:
        return cost_factor * 3;
      default:
        return pow(cost_factor, (double) std::max(fusion_qubits.size() - 1, size_t(1)));
    }
  }
  return pow(cost_factor, (double) std::max(fusion_qubits.size() - 1, size_t(1)));
}

void Fusion::add_fusion_qubits(reg_t& fusion_qubits, const op_t& op) const {
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
