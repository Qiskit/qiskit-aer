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

#ifndef _aer_transpile_truncate_qubits_hpp_
#define _aer_transpile_truncate_qubits_hpp_

#include "transpile/circuitopt.hpp"

namespace AER {
namespace Transpile {


using uint_t = uint_t;
using op_t = Operations::Op;
using optype_t = Operations::OpType;
using oplist_t = std::vector<op_t>;
using opset_t = Operations::OpSet;
using reg_t = std::vector<uint_t>;


class TruncateQubits : public CircuitOptimization {
public:

  void set_config(const json_t &config) override;

  // truncate unnecessary qubits
  void optimize_circuit(Circuit& circ,
                        const opset_t &opset,
                        OutputData &data) const override;

private:
  // check this optimization can be applied
  bool can_apply(const Circuit& circ) const;

  // check this optimization can be applied
  bool can_apply(const op_t& op) const;

  // generate a new mapping. a value of reg_t is original and its index is the new mapping
  reg_t generate_mapping(const Circuit& circ) const;

  // remap qubits in an operation
  op_t remap_qubits(const op_t op, const reg_t new_mapping)const;

  // show debug info
  bool verbose_ = false;

  // disabled in config
  bool active_ = true;
};

void TruncateQubits::set_config(const json_t &config) {

  CircuitOptimization::set_config(config);

  if (JSON::check_key("truncate_verbose", config_))
    JSON::get_value(verbose_, "truncate_verbose", config_);

  if (JSON::check_key("truncate_enable", config_))
    JSON::get_value(active_, "truncate_enable", config_);

  if (JSON::check_key("initial_statevector", config_))
    active_ = false;

}

void TruncateQubits::optimize_circuit(Circuit& circ,
                             const opset_t &allowed_opset,
                             OutputData &data) const {

  if (!active_ || !can_apply(circ))
    return;

  reg_t new_mapping = generate_mapping(circ);

  if (new_mapping.size() == circ.num_qubits)
    return;

  oplist_t new_ops;
  for (const op_t& old_op: circ.ops)
    new_ops.push_back(remap_qubits(old_op, new_mapping));

  circ.ops = new_ops;
  circ.num_qubits = new_mapping.size();

  if (verbose_) {
    data.add_additional_data("metadata",
                             json_t::object({{"truncate_verbose", new_mapping}}));
  }

}

reg_t TruncateQubits::generate_mapping(const Circuit& circ) const {
  size_t not_used = circ.num_qubits + 1;
  reg_t mapping = reg_t(circ.num_qubits, not_used);

  for (const op_t& op: circ.ops)
    for (size_t qubit: op.qubits)
      mapping[qubit] = qubit;

  mapping.erase(std::remove(mapping.begin(), mapping.end(), not_used), mapping.end());

  return mapping;
}

op_t TruncateQubits::remap_qubits(const op_t op, const reg_t new_mapping) const {
  op_t new_op = op;
  new_op.qubits.clear();

  for (const size_t qubit: op.qubits) {
    size_t new_qubit = std::distance(new_mapping.begin(), find(new_mapping.begin(), new_mapping.end(), qubit));
    new_op.qubits.push_back(new_qubit);
  }
  return new_op;

}

bool TruncateQubits::can_apply(const Circuit& circ) const {

  for (const op_t& op: circ.ops)
    if (!can_apply(op))
      return false;

  return true;
}

bool TruncateQubits::can_apply(const op_t& op) const {
  switch (op.type) {
  case optype_t::matrix_sequence: //TODO
  case optype_t::kraus: //TODO
  case optype_t::snapshot:
  case optype_t::noise_switch:
    return false;
  default:
    return true;
  }
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
