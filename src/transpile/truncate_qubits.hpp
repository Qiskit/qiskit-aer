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

class TruncateQubits : public CircuitOptimization {
public:

  void set_config(const json_t &config) override;

  // truncate unnecessary qubits
  void optimize_circuit(Circuit& circ,
                        const Operations::OpSet &opset,
                        OutputData &data) const override;

private:
  // check this optimization can be applied
  bool can_apply(const Circuit& circ) const;

  // check this optimization can be applied
  bool can_apply(const Operations::Op& op) const;

  // generate a new mapping. a value of reg_t is original and its index is the new mapping
  reg_t generate_mapping(const Circuit& circ) const;

  // remap qubits in an operation
  reg_t remap_qubits(const reg_t &qubits,
                     const reg_t &mapping) const;

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
                                      const Operations::OpSet &allowed_opset,
                                      OutputData &data) const {

  if (!active_ || !can_apply(circ))
    return;

  reg_t new_mapping = generate_mapping(circ);

  if (new_mapping.size() == circ.num_qubits)
    return;

  for (Operations::Op& op: circ.ops) {
    // Remap qubits
    reg_t new_qubits = remap_qubits(op.qubits, new_mapping);
    // Remap regs
    std::vector<reg_t> new_regs;
    for (reg_t &reg : op.regs)
      new_regs.push_back(remap_qubits(reg, new_mapping));

    op.qubits = new_qubits;
    op.regs = new_regs;
  }

  circ.num_qubits = new_mapping.size();

  if (verbose_) {
    data.add_additional_data("metadata",
                             json_t::object({{"truncate_verbose", new_mapping}}));
  }

}

reg_t TruncateQubits::generate_mapping(const Circuit& circ) const {
  size_t not_used = circ.num_qubits + 1;
  reg_t mapping = reg_t(circ.num_qubits, not_used);

  for (const Operations::Op& op: circ.ops) {
    for (size_t qubit: op.qubits)
      mapping[qubit] = qubit;
    for (const reg_t &reg: op.regs)
      for (size_t qubit: reg)
        mapping[qubit] = qubit;
  }

  mapping.erase(std::remove(mapping.begin(), mapping.end(), not_used),
                mapping.end());

  return mapping;
}

reg_t TruncateQubits::remap_qubits(const reg_t &qubits,
                                   const reg_t &mapping) const {
  reg_t new_qubits;
  for (const size_t qubit: qubits) {
    size_t new_qubit = std::distance(mapping.begin(),
                                     find(mapping.begin(), mapping.end(), qubit));
    new_qubits.push_back(new_qubit);
  }
  return new_qubits;
}

bool TruncateQubits::can_apply(const Circuit& circ) const {
  for (const Operations::Op& op: circ.ops)
    if (!can_apply(op))
      return false;
  return true;
}

bool TruncateQubits::can_apply(const Operations::Op& op) const {
  switch (op.type) {
  case Operations::OpType::snapshot: {
    const stringset_t allowed({
      "memory",
      "register",
      "probabilities",
      "probabilities_with_variance",
      "expectation_value_pauli",
      "expectation_value_pauli_with_variance"
    });
    return allowed.find(op.name) != allowed.end();
  }
  default:
    return true;
  }
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
