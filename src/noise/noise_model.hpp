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

#ifndef _aer_noise_model_hpp_
#define _aer_noise_model_hpp_

#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/operations.hpp"
#include "framework/types.hpp"
#include "framework/rng.hpp"
#include "framework/circuit.hpp"
#include "noise/abstract_error.hpp"

// For JSON parsing of specific error types
#include "noise/quantum_error.hpp"
#include "noise/readout_error.hpp"

namespace AER {
namespace Noise {

//=========================================================================
// Noise Model class
//=========================================================================

// This allows specification of default and individual noise parameters for
// each circuit operation that may either be local (applied to the qubits
// in the operation) or nonlocal (applied to different qubits). The errors
// may each also be applied either before or after the operation as per
// the specification of the AbstractError subclass.

class NoiseModel {
public:

  using NoiseOps = std::vector<Operations::Op>;

  NoiseModel() = default;
  NoiseModel(const json_t &js) {load_from_json(js);}

  // Sample a noisy implementation of a full circuit
  // An RngEngine is passed in as a reference so that sampling
  // can be done in a thread-safe manner
  Circuit sample_noise(const Circuit &circ, RngEngine &rng) const;

  // Load a noise model from JSON
  void load_from_json(const json_t &js);

  // Add a QuantumError to the noise model
  void add_quantum_error(const QuantumError &error,
                         const stringset_t &op_labels,
                         const std::vector<reg_t> &op_qubits = {},
                         const std::vector<reg_t> &noise_qubits = {});
  
  // Add a ReadoutError to the noise model
  void add_readout_error(const ReadoutError &error,
                         const std::vector<reg_t> &op_qubits = {});
  
  // Return true if the noise model is ideal
  inline bool ideal() const {
    return !(local_quantum_errors_ || nonlocal_quantum_errors_) && readout_errors_.empty();
  }

  // Set which single qubit gates should use the X90 waltz error model
  inline void set_x90_gates(const stringset_t &x90_gates) {
    x90_gates_ = x90_gates;
  }

  // Set threshold for applying u1 rotation angles.
  // an Op for u1(theta) will only be added if |theta| > 0 and |theta - 2*pi| > 0
  inline void set_u1_threshold(double threshold) {
    u1_threshold_ = threshold;
  }

  // Return the opset for the noise model
  inline const Operations::OpSet& opset() const {return opset_;}

private:

  // Sample noise for the current operation
  NoiseOps sample_noise(const Operations::Op &op, RngEngine &rng) const;

  // Sample noise for the current operation
  void sample_readout_noise(const Operations::Op &op,
                            NoiseOps &noise_after,
                            RngEngine &rng)  const;

  void sample_local_quantum_noise(const Operations::Op &op,
                                  NoiseOps &noise_before,
                                  NoiseOps &noise_after,
                                  RngEngine &rng)  const;

  void sample_nonlocal_quantum_noise(const Operations::Op &op,
                                     NoiseOps &noise_before,
                                     NoiseOps &noise_after,
                                     RngEngine &rng)  const;

  // Sample noise for the current operation
  NoiseOps sample_noise_helper(const Operations::Op &op,
                               RngEngine &rng) const;

  // Sample a noisy implementation of a two-X90 pulse u3 gate
  NoiseOps sample_noise_x90_u3(uint_t qubit, complex_t theta,
                               complex_t phi, complex_t lamba,
                               RngEngine &rng) const;
  
  // Sample a noisy implementation of a single-X90 pulse u2 gate
  NoiseOps sample_noise_x90_u2(uint_t qubit, complex_t phi, complex_t lambda,
                               RngEngine &rng) const;

  // Add a local quantum error to the noise model for specific qubits
  void add_local_quantum_error(const QuantumError &error,
                               const stringset_t &op_labels,
                               const std::vector<reg_t> &op_qubits);

  // Add a non-local Error type to the model for specific qubits
  void add_nonlocal_quantum_error(const QuantumError &error,
                                  const stringset_t &op_labels,
                                  const std::vector<reg_t> &op_qubits,
                                  const std::vector<reg_t> &noise_qubits);

  // Flags which say whether the local or nonlocal error tables are used
  bool local_quantum_errors_ = false;
  bool nonlocal_quantum_errors_ = false;

  // List of quantum errors
  std::vector<QuantumError> quantum_errors_;

  // List of readout errors
  std::vector<ReadoutError> readout_errors_;

  using inner_table_t = stringmap_t<std::vector<size_t>>;
  using outer_table_t = stringmap_t<inner_table_t>;

  // Table indexes a name with a vector of the position of noise operations
  inner_table_t readout_error_table_;
  outer_table_t local_quantum_error_table_;

  // Nonlocal noise lookup table. Things get messy here...
  // the outer map is a table from gate strings to gate qubit maps
  // the gate qubit map is a map from qubits to another map of target qubits
  // which is then the final qubit map from target qubits to error_ptr positions
  stringmap_t<outer_table_t> nonlocal_quantum_error_table_;

  // Helper function to convert reg to string for key of unordered maps/sets
  std::string reg2string(const reg_t &reg) const;
  reg_t string2reg(std::string s) const;

  // Table of single-qubit gates to use a Waltz X90 based error model
  stringset_t x90_gates_;

  // Lookup table for gate strings to enum
  enum class WaltzGate {id, x, y, z, h, s, sdg, t, tdg, u0, u1, u2, u3};
  const static stringmap_t<WaltzGate> waltz_gate_table_;

  // waltz threshold for applying u1 rotations if |theta - 2n*pi | > threshold
  double u1_threshold_ = 1e-10;

  // Joint OpSet of all errors
  Operations::OpSet opset_;
};


//=========================================================================
// Noise Model class
//=========================================================================

NoiseModel::NoiseOps NoiseModel::sample_noise(const Operations::Op &op,
                                              RngEngine &rng) const {
  // Look to see if gate is a waltz gate for this error model
  auto it = x90_gates_.find(op.name);
  if (it == x90_gates_.end()) {
    // Non-X90 based gate, run according to base model
    return sample_noise_helper(op, rng);
  }
  // Decompose ops in terms of their waltz implementation
  auto gate = waltz_gate_table_.find(op.name);
  if (gate != waltz_gate_table_.end()) {
    switch (gate->second) {
      case WaltzGate::u3:
        return sample_noise_x90_u3(op.qubits[0], op.params[0], op.params[1], op.params[2], rng);
      case WaltzGate::u2:
        return sample_noise_x90_u2(op.qubits[0], op.params[0], op.params[1], rng);
      case WaltzGate::x:
        return sample_noise_x90_u3(op.qubits[0], M_PI, 0., M_PI, rng);
      case WaltzGate::y:
        return sample_noise_x90_u3(op.qubits[0],  M_PI, 0.5 * M_PI, 0.5 * M_PI, rng);
      case WaltzGate::h:
        return sample_noise_x90_u2(op.qubits[0], 0., M_PI, rng);
      default:
        // The rest of the Waltz operations are noise free (u1 only)
        return {op};
    }
  } else {
    // something went wrong if we end up here
    throw std::invalid_argument("Invalid waltz gate.");
  }
}


Circuit NoiseModel::sample_noise(const Circuit &circ, RngEngine &rng) const {
    bool noise_active = true; // set noise active to on-state
    Circuit noisy_circ = circ; // copy input circuit
    noisy_circ.measure_sampling_flag = false; // disable measurement opt flag
    noisy_circ.ops.clear(); // delete ops
    noisy_circ.ops.reserve(2 * circ.ops.size()); // just to be safe?
    // Sample a noisy realization of the circuit
    for (const auto &op: circ.ops) {
      switch (op.type) {
        // Operations that cannot have noise
        case Operations::OpType::barrier:
          noisy_circ.ops.push_back(op);
          break;
        case Operations::OpType::snapshot:
          noisy_circ.ops.push_back(op);
          break;
        case Operations::OpType::kraus:
          noisy_circ.ops.push_back(op);
          break;
        case Operations::OpType::roerror:
          noisy_circ.ops.push_back(op);
          break;
        case Operations::OpType::bfunc:
          noisy_circ.ops.push_back(op);
          break;
        // Switch noise on or off during current circuit sample
        case Operations::OpType::noise_switch:
          noise_active = static_cast<int>(std::real(op.params[0]));
          break;
        default:
          if (noise_active) {
            NoiseOps noisy_op = sample_noise(op, rng);
            noisy_circ.ops.insert(noisy_circ.ops.end(), noisy_op.begin(), noisy_op.end());
          }
          break;
      }
    }
    return noisy_circ;
}


void NoiseModel::add_readout_error(const ReadoutError &error,
                                         const std::vector<reg_t> &op_qubits) {
  // Add roerror to noise model ops
  opset_.optypes.insert(Operations::OpType::roerror);
  // Add error term as unique pointer
  readout_errors_.push_back(error);
  // Get position of error in error vector
  const auto error_pos = readout_errors_.size() - 1;

  // Add error index to the error table
  if (op_qubits.empty()) {
    readout_error_table_[""].push_back(error_pos);
  } else {
    for (const auto &qubits : op_qubits)
      readout_error_table_[reg2string(qubits)].push_back(error_pos);
  }
}


void NoiseModel::add_quantum_error(const QuantumError &error,
                                   const stringset_t &op_labels,
                                   const std::vector<reg_t> &op_qubits,
                                   const std::vector<reg_t> &noise_qubits) {
  // Add error opset to noise model opset
  opset_.insert(error.opset());

  // Add error to noise model
  if (op_qubits.empty()) {
    // Add default local error
    add_local_quantum_error(error, op_labels, {reg_t()});
  } else if (noise_qubits.empty()) {
    // Add local error for specific qubits
    add_local_quantum_error(error, op_labels, op_qubits);
  } else {
    // Add non local error for specific qubits and target qubits
    add_nonlocal_quantum_error(error, op_labels, op_qubits, noise_qubits);
  }
}


void NoiseModel::add_local_quantum_error(const QuantumError &error,
                                         const stringset_t &op_labels,
                                         const std::vector<reg_t> &op_qubits) {
  // Turn on local error flag
  if (!op_labels.empty()) {
    local_quantum_errors_ = true;
  }
  // Add error term as unique pointer
  quantum_errors_.push_back(error);
  // Get position of error in error vector
  const auto error_pos = quantum_errors_.size() - 1;
  // Add error index to the error table
  for (const auto &gate: op_labels)
    for (const auto &qubits : op_qubits)
      local_quantum_error_table_[gate][reg2string(qubits)].push_back(error_pos);
}


void NoiseModel::add_nonlocal_quantum_error(const QuantumError &error,
                                            const stringset_t &op_labels,
                                            const std::vector<reg_t> &op_qubits,
                                            const std::vector<reg_t> &noise_qubits) {

  // Turn on nonlocal error flag
  if (!op_labels.empty() && !op_qubits.empty() && !noise_qubits.empty()) {
    nonlocal_quantum_errors_ = true;
  }
  // Add error term as unique pointer
  quantum_errors_.push_back(error);
  // Get position of error in error vector
  const auto error_pos = quantum_errors_.size() - 1;
  // Add error index to the error table
  for (const auto &gate: op_labels)
    for (const auto &qubits_gate : op_qubits)
      for (const auto &qubits_noise : noise_qubits)
        nonlocal_quantum_error_table_[gate][reg2string(qubits_gate)][reg2string(qubits_noise)].push_back(error_pos);
}


NoiseModel::NoiseOps NoiseModel::sample_noise_helper(const Operations::Op &op,
                                                     RngEngine &rng) const {                                                
  // Return operator set
  NoiseOps noise_before;
  NoiseOps noise_after;
  // Apply local errors first
  sample_local_quantum_noise(op, noise_before, noise_after, rng);
  // Apply nonlocal errors second
  sample_nonlocal_quantum_noise(op, noise_before, noise_after, rng);
  // Apply readout error to measure ops
  if (op.type == Operations::OpType::measure) {
    sample_readout_noise(op, noise_after, rng);
  }

  // combine the original op with the noise ops before and after
  noise_before.reserve(noise_before.size() + noise_after.size() + 1);
  noise_before.push_back(op);
  noise_before.insert(noise_before.end(), noise_after.begin(), noise_after.end());
  return noise_before;
}


void NoiseModel::sample_readout_noise(const Operations::Op &op,
                                      NoiseOps &noise_after,
                                      RngEngine &rng) const {
  // If no readout errors are defined pass
  if (readout_errors_.empty()) {
    return;
  }
  // Check if measure op writes only to memory, or also to registers
  // We will use the same error model for both memory and registers
  bool has_registers = !op.registers.empty();
  
  //
  std::string op_qubits = reg2string(op.qubits);

  // Check if the qubits are listed in the readout error model
  auto iter_default = readout_error_table_.find(std::string());

  // Format qubit sets
  std::vector<std::string> qubit_keys;
  std::vector<reg_t> memory_sets, registers_sets;
  if (readout_error_table_.find(op_qubits) == readout_error_table_.end()) {
    // Since measure can be defined on multiple qubits
    // but error model may be specified only on single qubits we add
    // each one separately. If a multi-qubit model is found for specified
    // qubits however, that will be used instead.
    for (const auto &q : op.qubits) {
      qubit_keys.push_back(std::to_string(q));
    }
    // Add the classical register sets for measure ops
    for (const auto &q : op.memory) {
      memory_sets.push_back({q});
    }
    if (has_registers) {
      for (const auto &q : op.registers) {
        registers_sets.push_back({q});
      }
    }
  } else {
    // for gate operations we use the qubits as specified
    qubit_keys.push_back(op_qubits);
    memory_sets.push_back(op.memory);
    if (has_registers)
      registers_sets.push_back(op.registers);
  }
  // Iterate over qubits
  for (size_t qs=0; qs < qubit_keys.size(); ++qs) {
    auto iter_qubits = readout_error_table_.find(qubit_keys[qs]);
    if (iter_qubits != readout_error_table_.end() ||
        iter_default != readout_error_table_.end()) {
      auto &error_positions = (iter_qubits != readout_error_table_.end())
        ? iter_qubits->second
        : iter_default->second;
      for (auto &pos : error_positions) {
        // Sample Readout error
        auto noise_ops = readout_errors_[pos].sample_noise(memory_sets[qs], rng);
        if (has_registers) {
          for (auto& noise_op: noise_ops) {
            noise_op.registers = registers_sets[qs];
          }
        }
        // Add noise after the error 
        noise_after.insert(noise_after.end(), noise_ops.begin(), noise_ops.end());
      }
    }
  }
}


void NoiseModel::sample_local_quantum_noise(const Operations::Op &op,
                                            NoiseOps &noise_before,
                                            NoiseOps &noise_after,
                                            RngEngine &rng) const {
  
  // If no errors are defined pass
  if (local_quantum_errors_ == false)
    return;

  // Get op name, or label if it is a gate or unitary matrix
  std::string name = (op.type == Operations::OpType::matrix ||
                      op.type == Operations::OpType::gate)
    ? op.string_params[0]
    : op.name;

  // Check if op is a measure or reset
  bool is_measure_or_reset = (op.type == Operations::OpType::measure ||
                              op.type == Operations::OpType::reset);

  // Convert qubits to string for table lookup
  std::string op_qubits = reg2string(op.qubits);

  // Get the qubit error map for gate name
  auto iter = local_quantum_error_table_.find(name);
  if (iter != local_quantum_error_table_.end()) {
    // Check if the qubits are listed in the inner model
    const auto qubit_map = iter->second;
    // Get the default qubit model in case a specific qubit model is not found
    // The default model is stored under the empty key string ""
    auto iter_default = qubit_map.find(std::string());
    // Format qubit sets
    std::vector<std::string> qubit_keys;
    if (is_measure_or_reset && qubit_map.find(op_qubits) == qubit_map.end()) {
      // Since measure and reset ops can be defined on multiple qubits
      // but error model may be specified only on single qubits we add
      // each one separately. If a multi-qubit model is found for specified
      // qubits however, that will be used instead.
      for (const auto &q : op.qubits) {
        qubit_keys.push_back(std::to_string(q) + std::string(","));
      }
    } else {
      // for gate operations we use the qubits as specified
      qubit_keys.push_back(op_qubits);
    }
    for (size_t qs=0; qs < qubit_keys.size(); ++qs) {
      auto iter_qubits = qubit_map.find(qubit_keys[qs]);
      if (iter_qubits != qubit_map.end() ||
          iter_default != qubit_map.end()) {
        auto &error_positions = (iter_qubits != qubit_map.end())
          ? iter_qubits->second
          : iter_default->second;
        for (auto &pos : error_positions) {
          auto noise_ops = quantum_errors_[pos].sample_noise(string2reg(qubit_keys[qs]), rng);
          // Duplicate same sampled error operations
          if (quantum_errors_[pos].errors_after())
            noise_after.insert(noise_after.end(), noise_ops.begin(), noise_ops.end());
          else
            noise_before.insert(noise_before.end(), noise_ops.begin(), noise_ops.end());
        }
      }
    }
  }
}


void NoiseModel::sample_nonlocal_quantum_noise(const Operations::Op &op,
                                               NoiseOps &noise_before,
                                               NoiseOps &noise_after,
                                               RngEngine &rng) const {
  
  // If no errors are defined pass
  if (nonlocal_quantum_errors_ == false)
    return;
  
  // Get op name, or label if it is a gate or unitary matrix
  std::string name = (op.type == Operations::OpType::matrix ||
                      op.type == Operations::OpType::gate)
    ? op.string_params[0]
    : op.name;

  // Convert qubits to string for table lookup
  std::string qubits_str = reg2string(op.qubits);
  // Get the inner error map for  gate name
  auto iter = nonlocal_quantum_error_table_.find(name);
  if (iter != nonlocal_quantum_error_table_.end()) {
    const auto qubit_map = iter->second;
    // Format qubit sets
    std::vector<std::string> qubit_keys;

    if ((op.type == Operations::OpType::measure || op.type == Operations::OpType::reset)
        && qubit_map.find(qubits_str) == qubit_map.end()) {
      // Since measure and reset ops can be defined on multiple qubits
      // but error model may be specified only on single qubits we add
      // each one separately. If a multi-qubit model is found for specified
      // qubits however, that will be used instead.
      for (const auto &q : op.qubits)
        qubit_keys.push_back(std::to_string(q));
    } else {
      // for gate operations we use the qubits as specified
      qubit_keys.push_back(reg2string(op.qubits));
    }
    for (const auto &qubits: qubit_keys) {
      // Check if the qubits are listed in the inner model
      auto iter_qubits = qubit_map.find(qubits);
      if (iter_qubits != qubit_map.end()) {
        for (auto &target_pair : iter_qubits->second) {
          auto &target_qubits = target_pair.first;
          auto &error_positions = target_pair.second;
          for (auto &pos : error_positions) {
            auto ops = quantum_errors_[pos].sample_noise(string2reg(target_qubits), rng);
            if (quantum_errors_[pos].errors_after())
              noise_after.insert(noise_after.end(), ops.begin(), ops.end());
            else
              noise_before.insert(noise_before.end(), ops.begin(), ops.end());
          }
        }
      }
    }
  }
}


const stringmap_t<NoiseModel::WaltzGate>
NoiseModel::waltz_gate_table_ = {
  {"u3", WaltzGate::u3}, {"u2", WaltzGate::u2}, {"u1", WaltzGate::u1}, {"u0", WaltzGate::u0},
  {"id", WaltzGate::id}, {"x", WaltzGate::x}, {"y", WaltzGate::y}, {"z", WaltzGate::z},
  {"h", WaltzGate::h}, {"s", WaltzGate::s}, {"sdg", WaltzGate::sdg},
  {"t", WaltzGate::t}, {"tdg", WaltzGate::tdg}
};


NoiseModel::NoiseOps NoiseModel::sample_noise_x90_u3(uint_t qubit,
                                                       complex_t theta,
                                                       complex_t phi,
                                                       complex_t lambda,
                                                       RngEngine &rng) const {
  NoiseOps ret;
  const auto x90 = Operations::make_unitary({qubit}, Utils::Matrix::X90, "x90");
  if (std::abs(lambda) > u1_threshold_
      && std::abs(lambda - 2 * M_PI) > u1_threshold_
      && std::abs(lambda + 2 * M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, lambda)); // add 1st U1
  auto sample = sample_noise_helper(x90, rng); // sample noise for 1st X90
  ret.insert(ret.end(), sample.begin(), sample.end()); // add 1st noisy X90
  if (std::abs(theta + M_PI) > u1_threshold_
      && std::abs(theta - M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, theta + M_PI)); // add 2nd U1
  sample = sample_noise_helper(x90, rng); // sample noise for 2nd X90
  ret.insert(ret.end(), sample.begin(), sample.end()); // add 2nd noisy X90
  if (std::abs(phi + M_PI) > u1_threshold_
      && std::abs(phi - M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, phi + M_PI)); // add 3rd U1
  return ret;
}


NoiseModel::NoiseOps NoiseModel::sample_noise_x90_u2(uint_t qubit,
                                                       complex_t phi,
                                                       complex_t lambda,
                                                       RngEngine &rng) const {
  NoiseOps ret;
  const auto x90 = Operations::make_unitary({qubit}, Utils::Matrix::X90, "x90");
  if (std::abs(lambda - 0.5 * M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, lambda - 0.5 * M_PI)); // add 1st U1
  auto sample = sample_noise_helper(x90, rng); // sample noise for 1st X90
  ret.insert(ret.end(), sample.begin(), sample.end()); // add 1st noisy X90
  if (std::abs(phi + 0.5 * M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, phi + 0.5 * M_PI)); // add 2nd U1
  return ret;
}


std::string NoiseModel::reg2string(const reg_t &reg) const {
  std::stringstream result;
  std::copy(reg.begin(), reg.end(), std::ostream_iterator<reg_t::value_type>(result, ","));
  return result.str();
}


reg_t NoiseModel::string2reg(std::string s) const {
  reg_t result;
  size_t pos = 0;
  while ((pos = s.find(",")) != std::string::npos) {
    result.push_back(std::stoi(s.substr(0, pos)));
    s.erase(0, pos + 1);
  }
  return result;
}

//=========================================================================
// JSON Conversion
//=========================================================================

/*
  Schemas:
  {
    "error_model": {
      "errors": [error js],
      "x90_gates": which gates should be implemented as waltz gates and use the "x90" term
    }
  }

  Specific types additional parameters
  Quantum Error
  {
    "type": "qerror",
    "operations": ["x", "y", ...],  // ops to apply error to
    "gate_qubits": [[0], [1]]      // error only apples when op is on these qubits (blank for all)
    "noise_qubits": [[2], ...]     // error term will be applied to these qubits (blank for input qubits)
                                   // if false noise will be applied before op
    "probabilities": [p0, p1, ..],
    "instructions": [qobj_circuit_instrs0, qobj_circuit_instrs1, ...]
  }

  Readout Error (note: readout errors can only be local)
  {
    "type": "roerror",
    "operations": ["measure"],
    "probabilities": [[P(0|0), P(0|1)], [P(1|0), P(1|1)]]
    "gate_qubits": [[0]]  // error only apples when op is on these qubits (blank for all)
  }
*/

void NoiseModel::load_from_json(const json_t &js) {

  // If JSON is empty stop
  if (js.empty())
    return;

  // Check JSON is an object
  if (!js.is_object()) {
    throw std::invalid_argument("Invalid noise_params JSON: not an object.");
  }

  // See if any single qubit gates have a waltz error model applied to them
  if (JSON::check_key("x90_gates", js)) {
    set_x90_gates(js["x90_gates"]);
  }

  if (JSON::check_key("errors", js)) {
    if (!js["errors"].is_array()) {
      throw std::invalid_argument("Invalid noise_params JSON: \"error\" field is not a list");
    }
    for (const auto &gate_js : js["errors"]) {
      std::string type;
      JSON::get_value(type, "type", gate_js);
      stringset_t ops; // want set so ops are unique, and we can pull out measure
      JSON::get_value(ops, "operations", gate_js);
      std::vector<reg_t> gate_qubits;
      JSON::get_value(gate_qubits, "gate_qubits", gate_js);
      std::vector<reg_t> noise_qubits;
      JSON::get_value(noise_qubits, "noise_qubits", gate_js);

      // We treat measure as a separate error op so that it can be applied before
      // the measure operation, rather than after like the other gates
      if (ops.find("measure") != ops.end() && type != "roerror") {
        ops.erase("measure"); // remove measure from set of ops
        if (type != "qerror")
          throw std::invalid_argument("NoiseModel: Invalid noise type (" + type + ")");
        QuantumError error;
        error.load_from_json(gate_js);
        error.set_errors_before(); // set errors before the op
        add_quantum_error(error, {"measure"}, gate_qubits, noise_qubits);
      }
      // Load the remaining ops as errors that come after op
      if (type == "qerror") {
        QuantumError error;
        error.load_from_json(gate_js);
        add_quantum_error(error, ops, gate_qubits, noise_qubits);
      } else if (type == "roerror") {
        // We do not allow non-local readout errors
        if (!noise_qubits.empty()) {
          throw std::invalid_argument("Readout error must be a local error");
        }
        ReadoutError error; // readout error goes after
        error.load_from_json(gate_js);
        add_readout_error(error, gate_qubits);
      }else {
        throw std::invalid_argument("NoiseModel: Invalid noise type (" + type + ")");
      }
    }
  }
}

inline void from_json(const json_t &js, NoiseModel &model) {
  model = NoiseModel(js);
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
