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
#include "framework/linalg/matrix_utils.hpp"
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

  using Method = QuantumError::Method;
  using NoiseOps = std::vector<Operations::Op>;

  NoiseModel() = default;
  NoiseModel(const json_t &js) {load_from_json(js);}

  // Sample a noisy implementation of a full circuit
  // An RngEngine is passed in as a reference so that sampling
  // can be done in a thread-safe manner.
  // Sample methods are:
  // standard: each noisy op will be returned along with additional noise ops
  // superop: each noisy gate or reset will be returned as a single superop 
  Circuit sample_noise(const Circuit &circ,
                       RngEngine &rng) const;

  // Set sample mode to circuit
  // This is the default method for noise sampling that can work for
  // any simulator that supports the sampled noise instructions
  void activate_circuit_method();

  // Set sample mode to superoperator
  // This will cause all QuantumErrors stored in the noise model
  // to calculate their superoperator representations and raise
  // an exception if they cannot be converted.
  void activate_superop_method();

  // Set sample mode to kraus
  // This will cause all QuantumErrors stored in the noise model
  // to calculate their canonical Kraus representations and raise
  // an exception if they cannot be converted.
  void activate_kraus_method();

  //-----------------------------------------------------------------------
  // Checking if errors types are in noise model
  //-----------------------------------------------------------------------

  // Return True if noise model contains readout errors
  inline bool has_readout_errors() const {
    return !readout_errors_.empty();
  }

  // Return True if noise model contains quantum errors
  inline bool has_quantum_errors() const {
    return local_quantum_errors_ || nonlocal_quantum_errors_;
  }

  // Return True if noise model contains nonlocal quantum errors
  inline bool has_nonlocal_quantum_errors() const {
    return nonlocal_quantum_errors_;
  }

  // Return True if noise model contains local quantum errors
  inline bool has_local_quantum_errors() const {
    return local_quantum_errors_;
  }

  // Return true if the noise model is ideal
  inline bool is_ideal() const {
    return !has_readout_errors() && !has_quantum_errors();
  }

  //-----------------------------------------------------------------------
  // Add errors to noise model
  //-----------------------------------------------------------------------

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

  //-----------------------------------------------------------------------
  // Utils
  //-----------------------------------------------------------------------

  // Remap the qubits in the noise model.
  // A remapping is entered as a map {old: new}
  // Any qubits not in the mapping are assumed to be mapped to themselves.
  // Hence the sets of all keys and all values of the map must be equal.
  void remap_qubits(const std::unordered_map<uint_t, uint_t> &mapping);

  // Return vector of noise qubits for non local error on specified label and qubits
  // If no nonlocal error exists an empty set is returned.
  std::set<uint_t> nonlocal_noise_qubits(const std::string label, const reg_t &qubits) const;

  // Set threshold for applying u1 rotation angles.
  // an Op for u1(theta) will only be added if |theta| > 0 and |theta - 2*pi| > 0
  inline void set_u1_threshold(double threshold) {
    u1_threshold_ = threshold;
  }

  // Return the opset for the noise model
  inline const Operations::OpSet& opset() const {return opset_;}

private:

  // Sample noise for the current operation.
  NoiseOps sample_noise(const Operations::Op &op,
                        RngEngine &rng) const;

  // Sample noise for the current operation
  void sample_readout_noise(const Operations::Op &op,
                            NoiseOps &noise_after,
                            RngEngine &rng)  const;

  void sample_local_quantum_noise(const Operations::Op &op,
                                  NoiseOps &noise_before,
                                  NoiseOps &noise_after,
                                  RngEngine &rng)  const;

  void sample_nonlocal_quantum_noise(const Operations::Op &op,
                                     NoiseOps &noise_ops,
                                     NoiseOps &noise_after,
                                     RngEngine &rng) const;

  // Sample noise for the current operation
  NoiseOps sample_noise_helper(const Operations::Op &op,
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

  // Set of qubits used in noise model
  std::set<uint_t> noise_qubits_;

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
  std::string remap_string(const std::string key,
                           const std::unordered_map<uint_t, uint_t> &mapping) const;

  // Helper function to try and convert an instruction to superop matrix
  // If conversion isn't possible this returns an empty matrix
  cmatrix_t op2superop(const Operations::Op &op) const;

  // Try and convert an instruction to unitary matrix
  // If conversion isn't possible this returns an empty matrix
  cmatrix_t op2unitary(const Operations::Op &op) const;

  // Lookup table for gate strings to enum
  enum class WaltzGate {id, x, y, z, h, s, sdg, t, tdg, u0, u1, u2, u3};
  const static stringmap_t<WaltzGate> waltz_gate_table_;

  // Parameterized Gates
  enum class ParamGate {u1, u2, u3, r, rx, ry, rz, rxx, ryy, rzz, rzx, cp};
  const static stringmap_t<ParamGate> param_gate_table_;

  // waltz threshold for applying u1 rotations if |theta - 2n*pi | > threshold
  double u1_threshold_ = 1e-10;

  // Joint OpSet of all errors
  Operations::OpSet opset_;

  // Sampling method
  Method method_ = Method::circuit;
};

//=========================================================================
// Parameterized Gates
//=========================================================================

const stringmap_t<NoiseModel::ParamGate>
NoiseModel::param_gate_table_ = {
  {"u", ParamGate::u3},
  {"u3", ParamGate::u3},
  {"u2", ParamGate::u2},
  {"u1", ParamGate::u1},
  {"r", ParamGate::r},
  {"rx", ParamGate::rx},
  {"ry", ParamGate::ry},
  {"rz", ParamGate::rz},
  {"rxx", ParamGate::rxx},
  {"ryy", ParamGate::ryy},
  {"rzz", ParamGate::rzz},
  {"rzx", ParamGate::rzx},
  {"p", ParamGate::u1},
  {"cp", ParamGate::cp},
  {"cu1", ParamGate::cp}
};


//=========================================================================
// Noise sampling
//=========================================================================

NoiseModel::NoiseOps NoiseModel::sample_noise(const Operations::Op &op,
                                              RngEngine &rng) const {
  // Noise operations
  NoiseOps noise_ops = sample_noise_helper(op, rng);

  // If original op is conditional, make all the noise operations also conditional
  if (op.conditional) {
    for (auto& noise_op : noise_ops) {
      noise_op.conditional = op.conditional;
      noise_op.conditional_reg = op.conditional_reg;
      noise_op.bfunc = op.bfunc;
    }
  }
  return noise_ops;
}


Circuit NoiseModel::sample_noise(const Circuit &circ,
                                 RngEngine &rng) const {
    bool noise_active = true; // set noise active to on-state
    Circuit noisy_circ;
    // Copy metadata
    noisy_circ.seed = circ.seed;
    noisy_circ.shots = circ.shots;
    noisy_circ.header = circ.header;

    // Reserve double length of ops just to be safe
    noisy_circ.ops.reserve(2 * circ.ops.size());

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
        case Operations::OpType::superop:
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
    // Update circuit parameters
    noisy_circ.set_params();
    return noisy_circ;
}

void NoiseModel::activate_circuit_method() {
  method_ = Method::circuit;
}

void NoiseModel::activate_superop_method() {
  // Set internal sampling method
  method_ = Method::superop;
  // Compute superoperators
  for (auto& qerror : quantum_errors_) {
    qerror.compute_superoperator();
  }
}


void NoiseModel::activate_kraus_method() {
  // Set internal sampling method
  method_ = Method::kraus;
  // Compute kraus
  for (auto& qerror : quantum_errors_) {
    qerror.compute_kraus();
  }
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
    for (const auto &qubits : op_qubits) {
      readout_error_table_[reg2string(qubits)].push_back(error_pos);
      for (const auto &qubit : qubits)
        noise_qubits_.insert(qubit);
    }
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
    for (const auto &qubits : op_qubits) {
      local_quantum_error_table_[gate][reg2string(qubits)].push_back(error_pos);
      for (const auto &qubit : qubits)
        noise_qubits_.insert(qubit);
    }
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
    for (const auto &qubits_gate : op_qubits) {
      for (const auto &qubit : qubits_gate)
        noise_qubits_.insert(qubit);
      for (const auto &qubits_noise : noise_qubits) {
        nonlocal_quantum_error_table_[gate][reg2string(qubits_gate)][reg2string(qubits_noise)].push_back(error_pos);
        for (const auto &qubit : qubits_noise)
          noise_qubits_.insert(qubit);
      }
    }
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

  // Combine errors
  auto &noise_ops = noise_before;
  noise_ops.reserve(noise_before.size() + noise_after.size() + 1);
  noise_ops.push_back(op);
  noise_ops.insert(noise_ops.end(),
                   std::make_move_iterator(noise_after.begin()),
                   std::make_move_iterator(noise_after.end()));
  
  
  if (op.type != Operations::OpType::measure &&
      noise_ops.size() == 2 &&
      noise_ops[0].qubits == noise_ops[1].qubits) {
    // Try and fuse operations
    // If either are superoperators combine superoperators
    // else if either are unitaries combine unitaries
    // otherwise return the full list
    auto& first_op = noise_ops[0];
    auto& second_op = noise_ops[1];

    if (second_op.type == Operations::OpType::superop) {
      auto& current = second_op;
      const auto mat = op2superop(first_op);
      if (!mat.empty()) {
        current.mats[0] = current.mats[0] * mat;
        return NoiseOps({current});
      }
    } else if (first_op.type == Operations::OpType::superop) {
      auto& current = first_op;
      const auto mat = op2superop(second_op);
      if (!mat.empty()) {
        current.mats[0] = mat * current.mats[0];
        return NoiseOps({current});
      }
    } else if (second_op.type == Operations::OpType::matrix) { 
      auto& current = noise_before[1];
      const auto mat = op2unitary(first_op);
      if (!mat.empty()) {
        current.mats[0] = current.mats[0] * mat;
        return NoiseOps({current});
      }
    } else if (first_op.type == Operations::OpType::matrix) {
      auto& current = first_op;
      const auto mat = op2unitary(second_op);
      if (!mat.empty()) {
        current.mats[0] = mat * current.mats[0];
        return NoiseOps({current});
      }
    }
  }
  // Otherwise return the list of ops
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
    for(auto qubit_key: qubit_keys){
      auto iter_qubits = qubit_map.find(qubit_key);
      if (iter_qubits != qubit_map.end() ||
          iter_default != qubit_map.end()) {
        auto &error_positions = (iter_qubits != qubit_map.end())
          ? iter_qubits->second
          : iter_default->second;
        for (auto &pos : error_positions) {
          auto noise_ops = quantum_errors_[pos].sample_noise(string2reg(qubit_key), rng, method_);
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
            auto ops = quantum_errors_[pos].sample_noise(string2reg(target_qubits), rng,
                                                         method_);
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


cmatrix_t NoiseModel::op2superop(const Operations::Op &op) const {
  switch (op.type) {
    case Operations::OpType::superop:
      return op.mats[0];
    case Operations::OpType::kraus: {
      return Utils::kraus_superop(op.mats);
    }
    case Operations::OpType::reset:
      return Linalg::SMatrix::reset(1ULL << op.qubits.size());
    case  Operations::OpType::matrix:
      return Utils::unitary_superop(op.mats[0]);
    case Operations::OpType::gate:  {
      auto it = param_gate_table_.find(op.name);
      if (it != param_gate_table_.end()) {
        // Get parameterized gate superop
        switch (it -> second) {
          case ParamGate::u1:
            return Linalg::SMatrix::u1(op.params[0]);
          case ParamGate::u2:
            return Linalg::SMatrix::u2(op.params[0], op.params[1]);
          case ParamGate::u3:
            return Linalg::SMatrix::u3(op.params[0], op.params[1], op.params[2]);
          case ParamGate::r:
            return Linalg::SMatrix::r(op.params[0], op.params[1]);
          case ParamGate::rx:
            return Linalg::SMatrix::rx(op.params[0]);
          case ParamGate::ry:
            return Linalg::SMatrix::ry(op.params[0]);
          case ParamGate::rz:
            return Linalg::SMatrix::rz(op.params[0]);
          case ParamGate::rxx:
            return Linalg::SMatrix::rxx(op.params[0]);
          case ParamGate::ryy:
            return Linalg::SMatrix::ryy(op.params[0]);
          case ParamGate::rzz:
            return Linalg::SMatrix::rzz(op.params[0]);
          case ParamGate::rzx:
            return Linalg::SMatrix::rzx(op.params[0]);
          case ParamGate::cp:
            return Linalg::SMatrix::cphase(op.params[0]);
        }
      } else {
        // Check if we can convert this gate to a standard superoperator matrix
        if (Linalg::SMatrix::allowed_name(op.name)) {
          return Linalg::SMatrix::from_name(op.name);
        }
      }
    }
    default:
      return cmatrix_t();
  }
}

cmatrix_t NoiseModel::op2unitary(const Operations::Op &op) const {
  switch (op.type) {
  case Operations::OpType::matrix:
    return op.mats[0];
  case Operations::OpType::gate:  {
    auto it = param_gate_table_.find(op.name);
    if (it != param_gate_table_.end()) {
      // Get parameterized gate superop
      switch (it -> second) {
        case ParamGate::u1:
          return Linalg::Matrix::u1(op.params[0]);
        case ParamGate::u2:
          return Linalg::Matrix::u2(op.params[0], op.params[1]);
        case ParamGate::u3:
          return Linalg::Matrix::u3(op.params[0], op.params[1], op.params[2]);
        case ParamGate::r:
          return Linalg::Matrix::r(op.params[0], op.params[1]);
        case ParamGate::rx:
          return Linalg::Matrix::rx(op.params[0]);
        case ParamGate::ry:
          return Linalg::Matrix::ry(op.params[0]);
        case ParamGate::rz:
          return Linalg::Matrix::rz(op.params[0]);
        case ParamGate::rxx:
          return Linalg::Matrix::rxx(op.params[0]);
        case ParamGate::ryy:
          return Linalg::Matrix::ryy(op.params[0]);
        case ParamGate::rzz:
          return Linalg::Matrix::rzz(op.params[0]);
        case ParamGate::rzx:
          return Linalg::Matrix::rzx(op.params[0]);
        case ParamGate::cp:
          return Linalg::Matrix::cphase(op.params[0]);
      }
    } else {
      // Check if we can convert this gate to a standard superoperator matrix
      if (Linalg::Matrix::allowed_name(op.name)) {
        return Linalg::Matrix::from_name(op.name);
      }
    }
  }
  default:
    return cmatrix_t();
  }
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
// Qubit Remapping
//=========================================================================

std::set<uint_t> NoiseModel::nonlocal_noise_qubits(const std::string label,
                                                   const reg_t& qubits) const {
  std::set<uint_t> all_noise_qubits;
  // Check if label has noise
  const auto outer_it = nonlocal_quantum_error_table_.find(label);
  if (outer_it != nonlocal_quantum_error_table_.end()) {
    const auto outer_table = outer_it->second;
    const auto it = outer_table.find(reg2string(qubits));
    // Check if label on specified qubits has noise
    if (it != outer_table.end()) {
      // Add all noise qubit errors to the return value
      const auto inner_table = it->second;
      for (const auto &pair : inner_table) {
        auto noise_qubits = string2reg(pair.first);
        for (const auto &qubit : noise_qubits) {
          all_noise_qubits.insert(qubit);
        }
      }
    }
  }
  return all_noise_qubits;
}


std::string NoiseModel::remap_string(const std::string key,
                                     const std::unordered_map<uint_t, uint_t> &mapping) const{
  reg_t qubits = string2reg(key);
  for (size_t j=0; j<qubits.size(); j++)
    qubits[j] = mapping.at(qubits[j]);
  return reg2string(qubits);
}


void NoiseModel::remap_qubits(const std::unordered_map<uint_t, uint_t> &mapping) {

  // If noise model is ideal we have no need to remap
  if (is_ideal())
    return;

  // We only need the mapping for qubits in the noise model.
  // We add qubits not specified in the mapping as trivial mapping to themselves
  // We also validate the mapping while building the full mapping
  std::unordered_map<uint_t, uint_t> full_mapping = mapping;
  // Add noise qubits not specified in mapping
  for (const auto &qubit : noise_qubits_) {
    if (full_mapping.find(qubit) == full_mapping.end()) {
      full_mapping[qubit] = qubit;
    }
  }

  // Check mapping is valid
  std::set<uint_t> qubits_in;
  std::set<uint_t> qubits_out;
  for (const auto &pair: full_mapping) {
    qubits_in.insert(pair.first);
    qubits_out.insert(pair.second);
  }
  if (qubits_in != qubits_out) {
    std::stringstream msg;
    msg << "NoiseModel: invalid qubit re-mapping " << full_mapping;
    throw std::invalid_argument(msg.str());
  }

  // Remap readout error
  if (has_readout_errors()) {
    inner_table_t new_readout_error_table;
    for (const auto &pair : readout_error_table_) {
      new_readout_error_table[remap_string(pair.first, full_mapping)] = pair.second;
    }
    readout_error_table_ = new_readout_error_table;
    new_readout_error_table.clear();
  }

  // Remap local quantum error
  if (has_local_quantum_errors()) {
    for (auto& outer_pair : local_quantum_error_table_) {
      // Get reference to the inner table we need to change the keys for
      auto& inner_table = outer_pair.second;
      // Make a temporary table to store remapped table
      inner_table_t new_table;
      for (const auto &inner_pair : inner_table) {
        new_table[remap_string(inner_pair.first, full_mapping)] = inner_pair.second;
      }
      // Replace inner table with the remapped table
      inner_table = new_table;
    }
  }

  // Remap nonlocal quantum error
  if (has_nonlocal_quantum_errors()) {
    for (auto& pair : nonlocal_quantum_error_table_) {
      // Get reference to the middle table we need to change the keys for
      auto& outer_table = pair.second;
      // Make a temporary table to store remapped outer table
      outer_table_t new_outer_table;
      for (auto& outer_pair : outer_table) {
        // Remap inner table
        auto& inner_table = outer_pair.second;
        inner_table_t new_inner_table;
        for (const auto &inner_pair : inner_table) {
          new_inner_table[remap_string(inner_pair.first, full_mapping)] = inner_pair.second;
        }
        // Update outer table with remapped inner table
        new_outer_table[remap_string(outer_pair.first, full_mapping)] = new_inner_table;
      }
      outer_table = new_outer_table;
    }
  }
}


//=========================================================================
// JSON Conversion
//=========================================================================

/*
  Schemas:
  {
    "error_model": {
      "errors": [error js]
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
