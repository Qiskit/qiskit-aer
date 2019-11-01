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

#include <unordered_map>

#include "transpile/circuitopt.hpp"

namespace AER {
namespace Transpile {

class TruncateQubits : public CircuitOptimization {
public:

  using mapping_t = std::unordered_map<uint_t, uint_t>;

  void set_config(const json_t &config) override;

  // Truncate unused qubits
  void optimize_circuit(Circuit& circ,
                        Noise::NoiseModel& noise,
                        const Operations::OpSet &opset,
                        ExperimentData &data) const override;

private:
  // check this optimization can be applied
  bool can_apply(const Circuit& circ) const;

  // check this optimization can be applied
  bool can_apply(const Operations::Op& op) const;

  // Generate a list of qubits that are used in the input circuit and noise model
  reg_t get_active_qubits(const Circuit& circ,
                          const Noise::NoiseModel& noise) const;

  // generate a new mapping. a value of reg_t is original and its index is the new mapping
  mapping_t generate_mapping(const reg_t& active_qubits,
                             const Circuit& circ,
                             const Noise::NoiseModel& noise) const;

  // remap qubits in an operation
  void remap_qubits(reg_t &qubits,
                    const mapping_t &mapping) const;

  // show debug info
  bool verbose_ = false;

  // disabled in config
  bool active_ = true;
};

void TruncateQubits::set_config(const json_t &config) {

  CircuitOptimization::set_config(config);

  if (JSON::check_key("truncate_verbose", config)) {
    JSON::get_value(verbose_, "truncate_verbose", config);
  }
  if (JSON::check_key("truncate_enable", config)) {
    JSON::get_value(active_, "truncate_enable", config);
  }
  if (JSON::check_key("initial_statevector", config)) {
    active_ = false;
  }
}

void TruncateQubits::optimize_circuit(Circuit& circ,
                                      Noise::NoiseModel& noise,
                                      const Operations::OpSet &allowed_opset,
                                      ExperimentData &data) const {
  
  // Check if circuit operations allow remapping
  // Remapped circuits must return the same output data as the
  // original circuit
  if (!active_ || !can_apply(circ))
    return;

  // Get qubits actually used in the circuit
  // If this is all qubits we don't need to remap
  reg_t active_qubits = get_active_qubits(circ, noise);
  if (active_qubits.size() == circ.num_qubits)
    return;

  // Generate the qubit mapping {original_qubit: new_qubit}
  mapping_t mapping = generate_mapping(active_qubits, circ, noise);

  // Remap circuit operations
  for (Operations::Op& op: circ.ops) {
    remap_qubits(op.qubits, mapping);
    // Remap regs
    for (reg_t &reg : op.regs) {
      remap_qubits(reg, mapping);
    }
  }
  // Update the number of qubits in the circuit 
  circ.num_qubits = active_qubits.size();

  // Remap noise model
  noise.remap_qubits(mapping);

  if (verbose_) {
    json_t truncate_metadata;
    truncate_metadata["active_qubits"] = active_qubits;
    truncate_metadata["mapping"] = mapping;
    data.add_metadata("truncate_qubits", truncate_metadata);
  }
}

reg_t TruncateQubits::get_active_qubits(const Circuit& circ,
                                        const Noise::NoiseModel& noise) const {

  size_t not_used = circ.num_qubits + 1;
  reg_t active_qubits = reg_t(circ.num_qubits, not_used);

  for (const Operations::Op& op: circ.ops) {
    for (size_t qubit: op.qubits)
      active_qubits[qubit] = qubit;
    for (const reg_t &reg: op.regs)
      for (size_t qubit: reg)
        active_qubits[qubit] = qubit;
    
    // Add noise model qubits
    // The label for checking noise is either stored in string_params
    // or is the op name
    std::string label = "";
    if (op.string_params.size() == 1) {
      label = op.string_params[0];
    }
    if (label == "")
      label = op.name;
    const auto noise_reg = noise.nonlocal_noise_qubits(label, op.qubits);
    for (size_t qubit: noise_reg) {
      // A noise model might have more qubits in it than are in
      // the original circuit. In this case we only add qubits
      // up to the number of qubits in the circuit
      if (qubit < circ.num_qubits)
        active_qubits[qubit] = qubit;
    }
  }

  // Erase unused qubits for the list
  active_qubits.erase(std::remove(active_qubits.begin(), active_qubits.end(), not_used),
                active_qubits.end());
  return active_qubits;
}

TruncateQubits::mapping_t
TruncateQubits::generate_mapping(const reg_t& active_qubits, 
                                 const Circuit& circ,
                                 const Noise::NoiseModel& noise) const {
  // Convert to mapping
  mapping_t mapping;
  for (const auto & qubit : active_qubits) {
    size_t new_qubit = std::distance(active_qubits.begin(),
                                     find(active_qubits.begin(),
                                     active_qubits.end(), qubit));
    mapping[qubit] = new_qubit;
  }

  // Now we need to complete the mapping by adding qubits not in the input space
  // This is required for remapping the noise model.
  if (!noise.is_ideal()) {
    uint_t unused_qubit = active_qubits.size();
    for (uint_t qubit=0; qubit<circ.num_qubits; qubit++) {
      if (mapping.find(qubit) == mapping.end()) {
        mapping[qubit] = unused_qubit;
        unused_qubit++; // increment unused qubit position
      }
    }
  }
  return mapping;
}


void TruncateQubits::remap_qubits(reg_t& qubits, const mapping_t &mapping) const {
  for (size_t j=0; j<qubits.size(); j++) {
   qubits[j] = mapping.at(qubits[j]);
  }
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
