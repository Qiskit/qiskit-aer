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
#include "simulators/unitary/unitary_state.hpp"

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
  Fusion(uint_t _max_qubit = 5, uint_t _threshold = 20, double _cost_factor = 1.8)
    : max_qubit(_max_qubit), threshold(_threshold), cost_factor(_cost_factor) {}
  /*
   * Fusion optimization uses following configuration options
   * - fusion_enable (bool): Enable fusion optimization in circuit optimization
   *       passes [Default: True]
   * - fusion_verbose (bool): Output gates generated in fusion optimization
   *       into metadata [Default: False]
   * - fusion_max_qubit (int): Maximum number of qubits for a operation generated
   *       in a fusion optimization [Default: 5]
   * - fusion_threshold (int): Threshold that number of qubits must be greater
   *       than to enable fusion optimization [Default: 20]
   * - fusion_cost_factor (double): a cost function to estimate an aggregate
   *       gate [Default: 1.8]
   */
  void set_config(const json_t &config) override;

  void optimize_circuit(Circuit& circ,
                        Noise::NoiseModel& noise,
                        const opset_t &allowed_opset,
                        ExperimentData &data) const override;

  // Qubit threshold for activating fusion pass
  uint_t max_qubit;
  uint_t threshold;
  double cost_factor;
  bool verbose = false;
  bool active = true;

private:
  bool can_ignore(const op_t& op) const;

  bool can_apply_fusion(const op_t& op) const;

  double get_cost(const op_t& op) const;

  bool aggregate_operations(oplist_t& ops, const int fusion_start, const int fusion_end) const;

  // Aggregate a subcircuit of operations into a single operation
  op_t generate_fusion_operation(const std::vector<op_t>& fusioned_ops,
                                 const reg_t &num_qubits) const;

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

};


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

  if (JSON::check_key("fusion_cost_factor", config_))
    JSON::get_value(cost_factor, "fusion_cost_factor", config_);
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
                              Noise::NoiseModel& noise,
                              const opset_t &allowed_opset,
                              ExperimentData &data) const {
  // Check if fusion should be skipped
  if (!active || circ.num_qubits < threshold
      || !allowed_opset.contains(optype_t::matrix))
    return;

  bool applied = false;

  uint_t fusion_start = 0;
  for (uint_t op_idx = 0; op_idx < circ.ops.size(); ++op_idx) {
    if (can_ignore(circ.ops[op_idx]))
      continue;
    if (!can_apply_fusion(circ.ops[op_idx])) {
      applied |= fusion_start != op_idx && aggregate_operations(circ.ops, fusion_start, op_idx);
      fusion_start = op_idx + 1;
    }
  }

  if (fusion_start < circ.ops.size() && aggregate_operations(circ.ops, fusion_start, circ.ops.size()))
      applied = true;

  if (applied) {

    size_t idx = 0;
    for (size_t i = 0; i < circ.ops.size(); ++i) {
      if (circ.ops[i].name != "nop") {
        if (i != idx)
          circ.ops[idx] = circ.ops[i];
        ++idx;
      }
    }

    if (idx != circ.ops.size())
      circ.ops.erase(circ.ops.begin() + idx, circ.ops.end());

    if (verbose)
      data.add_metadata("fusion_verbose", circ.ops);
  }

#ifdef DEBUG
  dump(circ.ops);
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
  case optype_t::matrix:
    return op.mats.size() == 1 && op.qubits.size() <= max_qubit;
  case optype_t::gate: {
    return (op.qubits.size() <= max_qubit &&
      QubitUnitary::StateOpSet.contains_gates(op.name));
  }
  case optype_t::kraus:
  case optype_t::reset:
  case optype_t::superop:
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
                                       const reg_t &qubits) const {
  // Run simulation
  ExperimentData dummy_data;
  RngEngine dummy_rng;

  QubitUnitary::State<> unitary_simulator;
  unitary_simulator.initialize_qreg(qubits.size());
  unitary_simulator.apply_ops(fusioned_ops, dummy_data, dummy_rng);
  return Operations::make_unitary(qubits,
                                  unitary_simulator.qreg().matrix(),
                                  std::string("fusion"));
}


bool Fusion::aggregate_operations(oplist_t& ops, const int fusion_start, const int fusion_end) const {

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

    for (int num_fusion = 2; num_fusion <=  static_cast<int> (max_qubit); ++num_fusion) {
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
        std::copy(ops[j].qubits.cbegin(), ops[j].qubits.cend(),
                  std::inserter(fusioned_qubits, fusioned_qubits.end()));
        ops[j].name = "nop";
      }
      if (!fusioned_ops.empty()) {
        // We need to remap qubits in fusion subcircuits for simulation
        // TODO: This could be done above during the fusion cost calculation
        reg_t qubits(fusioned_qubits.begin(), fusioned_qubits.end());
        std::unordered_map<uint_t, uint_t> qubit_mapping;
        for (size_t j = 0; j < qubits.size(); j++) {
          qubit_mapping[qubits[j]] = j;
        }
        // Remap qubits
        for (auto & op: fusioned_ops) {
          for (size_t j = 0; j < op.qubits.size(); j++) {
            op.qubits[j] = qubit_mapping[op.qubits[j]];
          }
        }
        ops[i] = generate_fusion_operation(fusioned_ops, qubits);
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
    if (ops[i].name == "u1" || ops[i].name == "cu1")
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
  return pow(cost_factor, (double) std::max(fusion_qubits.size() - 1, size_t(1)));
}

void Fusion::add_fusion_qubits(reg_t& fusion_qubits, const op_t& op) const {
  for (const auto qubit: op.qubits){
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
