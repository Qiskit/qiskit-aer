/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    noise_model.hpp
 * @brief   Noise model base class for qiskit-aer simulator engines
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_base_noise_model_hpp_
#define _aer_base_noise_model_hpp_

#include "framework/operations.hpp"
#include "framework/types.hpp"
#include "framework/rng.hpp"
#include "framework/circuit.hpp"
#include "noise/abstract_error.hpp"

// move JSON parsing
#include "noise/gate_error.hpp"


namespace AER {
namespace Base {

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

  // Sample noise for the current operation
  virtual NoiseOps sample_noise(const Operations::Op &op);

  // Sample a noisy implementation of a full circuit
  virtual Circuit sample_noise(const Circuit &circ);

  // Load a noise model from JSON
  void load_from_json(const json_t &js);

  // Set the RngEngine seed to a fixed value
  inline void set_rng_seed(uint_t seed) { rng_ = RngEngine(seed);}

  // Add a local error to the noise model for specific qubits
  template <class DerivedError>
  void add_local_error(const DerivedError &error, 
                 const std::vector<std::string> &op_labels,
                 const std::vector<reg_t> &op_qubit_sets);


  // Add a local Error type to the model default for all qubits
  // without a specific error present
  template <class DerivedError>
  void add_local_error(const DerivedError &error, 
                 const std::vector<std::string> &op_labels) {
    add_local_error(error, op_labels, {reg_t()});
  }

  // Add a non-local Error type to the model for specific qubits
  template <class DerivedError>
  void add_nonlocal_error(const DerivedError &error, 
                          const std::vector<std::string> &op_labels,
                          const std::vector<reg_t> &op_qubit_sets,
                          const std::vector<reg_t> &noise_qubit_sets);

  // Return true if there is the noise model is ideal
  // ie. there is no noise added
  inline bool ideal() {
    return !(local_errors_ || nonlocal_errors_);
  }

  // Set which single qubit gates should use the X90 waltz error model
  inline void set_waltz_gates(const std::set<std::string> &waltz_gates) {
    waltz_gates_ = waltz_gates;
  }

  // Set threshold for applying u1 rotation angles.
  // an Op for u1(theta) will only be added if |theta| > 0 and |theta - 2*pi| > 0
  inline void set_u1_threshold(double threshold) {
    u1_threshold_ = threshold;
  }

private:
  // Flags which say whether the local or nonlocal error tables are used
  bool local_errors_ = false;
  bool nonlocal_errors_ = false;
  
  // Flag to control whether noise has been switched off during circuit
  // noise sampling
  bool noise_active_ = true;

  // Table of errors
  std::vector<std::unique_ptr<Noise::AbstractError>> error_ptrs_;

  // Table indexes a name with a vector of the position of noise operations
  // Cant make inner map and unordered map due to lack of hashing for vector<uint_t>
  using qubit_map_t = std::map<reg_t, std::vector<size_t>>;
  std::unordered_map<std::string, qubit_map_t> local_error_table_;

  // Nonlocal noise lookup table. Things get messy here...
  // the outer map is a table from gate strings to gate qubit maps
  // the gate qubit map is a map from qubits to another map of target qubits
  // which is then the final qubit map from target qubits to error_ptr positions
  using nonlocal_qubit_map_t = std::map<reg_t, qubit_map_t>;
  std::unordered_map<std::string, nonlocal_qubit_map_t> nonlocal_error_table_;

  // Table of single-qubit gates to use a Waltz X90 based error model
  std::set<std::string> waltz_gates_;

  // Lookup table for gate strings to enum
  enum class Gate {id, x, y, z, h, s, sdg, t, tdg, u0, u1, u2, u3};
  const static std::unordered_map<std::string, Gate> waltz_gate_table_;

  // waltz threshold for applying u1 rotations if |theta - 2n*pi | > threshold
  double u1_threshold_ = 1e-10;

  // Rng engine
  RngEngine rng_; // initialized with random seed

  // Sample noise for the current operation
  NoiseOps sample_noise_helper(const Operations::Op &op);

  // Sample a noisy implementation of a two-X90 pulse u3 gate
  NoiseOps sample_noise_waltz_u3(uint_t qubit, complex_t theta,
                                 complex_t phi, complex_t lam); // TODO
  
  // Sample a noisy implementation of a single-X90 pulse u2 gate
  NoiseOps sample_noise_waltz_u2(uint_t qubit, complex_t phi, complex_t lam); // TODO
};


//=========================================================================
// Noise Model class
//=========================================================================

NoiseModel::NoiseOps NoiseModel::sample_noise(const Operations::Op &op) {
  // Look to see if gate is a waltz gate for this error model
  auto it = waltz_gates_.find(op.name);
  if (it == waltz_gates_.end()) {
    // Non-X90 based gate, run according to base model
    return sample_noise_helper(op);
  }
  // Decompose ops in terms of their waltz implementation
  auto gate = waltz_gate_table_.find(op.name);
  if (gate != waltz_gate_table_.end()) {
    switch (gate->second) {
      case Gate::u3:
        return sample_noise_waltz_u3(op.qubits[0], op.params[0], op.params[1], op.params[2]);
      case Gate::u2:
        return sample_noise_waltz_u2(op.qubits[0], op.params[0], op.params[1]);
      case Gate::x:
        return sample_noise_waltz_u3(op.qubits[0], M_PI, 0., M_PI);
      case Gate::y:
        return sample_noise_waltz_u3(op.qubits[0],  M_PI, 0.5 * M_PI, 0.5 * M_PI);
      case Gate::h:
        return sample_noise_waltz_u2(op.qubits[0], 0., M_PI);
      default:
        // The rest of the Waltz operations are noise free (u1 only)
        return {op};
    }
  } else {
    // something went wrong if we end up here
    throw std::invalid_argument("Invalid waltz gate.");
  }
}


Circuit NoiseModel::sample_noise(const Circuit &circ) {
    noise_active_ = true; // set noise active to on-state
    Circuit noisy_circ = circ; // copy input circuit
    noisy_circ.measure_sampling_flag = false; // disable measurement opt flag 
    noisy_circ.ops.clear(); // delete ops
    noisy_circ.ops.reserve(2 * circ.ops.size()); // just to be safe?
    // Sample a noisy realization of the circuit
    for (const auto &op: circ.ops) {
      if (op.name == "noise_switch") {
        // check for noise switch operation
        noise_active_ = std::real(op.params[0]);
      } else if (noise_active_) {
        NoiseOps noisy_op = sample_noise(op);
        // insert noisy op sequence into the circuit
        noisy_circ.ops.insert(noisy_circ.ops.end(), noisy_op.begin(), noisy_op.end());
      }
    }
    return noisy_circ;
}


template <class DerivedError>
void NoiseModel::add_local_error(const DerivedError &error,
                           const std::vector<std::string> &op_labels,
                           const std::vector<reg_t> &qubit_sets) {  
  // Turn on local error flag
  if (!op_labels.empty()) {
    local_errors_ = true;
  }
  // Add error term as unique pointer                                          
  error_ptrs_.push_back(std::unique_ptr<Noise::AbstractError>(std::make_unique<DerivedError>(error)));
  // position is error vector is length - 1
  const auto error_pos = error_ptrs_.size() - 1;
  // Add error index to the error table
  for (const auto &gate: op_labels)
    for (const auto &qubits : qubit_sets)
      local_error_table_[gate][qubits].push_back(error_pos); 
}


template <class DerivedError>
void NoiseModel::add_nonlocal_error(const DerivedError &error,
                                    const std::vector<std::string> &op_labels,
                                    const std::vector<reg_t> &qubit_sets,
                                    const std::vector<reg_t> &noise_qubit_sets) {  
  // Turn on nonlocal error flag
  if (!op_labels.empty() && !qubit_sets.empty() && !noise_qubit_sets.empty()) {
    nonlocal_errors_ = true;
  }
  // Add error term as unique pointer                                          
  error_ptrs_.push_back(std::unique_ptr<Noise::AbstractError>(std::make_unique<DerivedError>(error)));
  // position is error vector is length - 1
  const auto error_pos = error_ptrs_.size() - 1;
  // Add error index to the error table
  for (const auto &gate: op_labels)
    for (const auto &qubits_gate : qubit_sets)
      for (const auto &qubits_noise : noise_qubit_sets)
        nonlocal_error_table_[gate][qubits_gate][qubits_noise].push_back(error_pos); 
}


NoiseModel::NoiseOps NoiseModel::sample_noise_helper(const Operations::Op &op) {

  // Return operator set
  NoiseOps noise_before;
  NoiseOps noise_after;
  // Apply local errors first
  if (local_errors_) {
    // Get the qubit error map for  gate name
    auto iter = local_error_table_.find(op.name);
    if (iter != local_error_table_.end()) {
      // Check if the qubits are listed in the inner model
      const auto qubit_map = iter->second;
      auto iter_qubits = qubit_map.find(op.qubits);
      auto iter_default = qubit_map.find({});
      if (iter_qubits != qubit_map.end() ||
          iter_default != qubit_map.end()) {
        auto &error_positions = (iter_qubits != qubit_map.end())
          ? iter_qubits->second
          : iter_default->second;
        for (auto &pos : error_positions) {
          auto ops = error_ptrs_[pos]->sample_noise(op.qubits, rng_);
          if (error_ptrs_[pos]->errors_after())
            noise_after.insert(noise_after.end(), ops.begin(), ops.end());
          else
            noise_before.insert(noise_before.end(), ops.begin(), ops.end());
        }
      }
    }
  }
  // Apply nonlocal errors second
  if (nonlocal_errors_) {
    // Get the inner error map for  gate name
    auto iter = nonlocal_error_table_.find(op.name);
    if (iter != nonlocal_error_table_.end()) {
      // Check if the qubits are listed in the inner model
      const auto qubit_map = iter->second;
      auto iter_qubits = qubit_map.find(op.qubits);
      if (iter_qubits != qubit_map.end()) {
        for (auto &target_pair : iter_qubits->second) {
          auto &target_qubits = target_pair.first;
          auto &error_positions = target_pair.second;
          for (auto &pos : error_positions) {
            auto ops = error_ptrs_[pos]->sample_noise(target_qubits, rng_);
            if (error_ptrs_[pos]->errors_after())
              noise_after.insert(noise_after.end(), ops.begin(), ops.end());
            else
              noise_before.insert(noise_before.end(), ops.begin(), ops.end());
          }
        }
      }
    }
  }
  // combine the original op with the noise ops before and after
  noise_before.reserve(noise_before.size() + noise_after.size() + 1);
  noise_before.push_back(op);
  noise_before.insert(noise_before.end(), noise_after.begin(), noise_after.end());
  return noise_before;
}


const std::unordered_map<std::string, NoiseModel::Gate>
NoiseModel::waltz_gate_table_ = {
  {"u3", Gate::u3}, {"u2", Gate::u2}, {"u1", Gate::u1}, {"u0", Gate::u0},
  {"id", Gate::id}, {"x", Gate::x}, {"y", Gate::y}, {"z", Gate::z},
  {"h", Gate::h}, {"s", Gate::s}, {"sdg", Gate::sdg},
  {"t", Gate::t}, {"tdg", Gate::tdg}
};


NoiseModel::NoiseOps NoiseModel::sample_noise_waltz_u3(uint_t qubit,
                                                       complex_t theta,
                                                       complex_t phi,
                                                       complex_t lam) {
  NoiseOps ret;
  const auto x90 = Operations::make_mat({qubit}, Utils::Matrix::X90, "x90");
  if (std::abs(lam) > u1_threshold_
      && std::abs(lam - 2 * M_PI) > u1_threshold_
      && std::abs(lam + 2 * M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, lam)); // add 1st U1
  // TODO:: need to add X90 as named noise model
  auto sample = sample_noise_helper(x90); // sample noise for 1st X90
  ret.insert(ret.end(), sample.begin(), sample.end()); // add 1st noisy X90
  if (std::abs(theta + M_PI) > u1_threshold_
      && std::abs(theta - M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, theta + M_PI)); // add 2nd U1
  sample = sample_noise_helper(x90); // sample noise for 2nd X90
  ret.insert(ret.end(), sample.begin(), sample.end()); // add 2nd noisy X90
  if (std::abs(phi + M_PI) > u1_threshold_
      && std::abs(phi - M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, phi + M_PI)); // add 3rd U1
  return ret;                               
}


NoiseModel::NoiseOps NoiseModel::sample_noise_waltz_u2(uint_t qubit,
                                                       complex_t phi,
                                                       complex_t lam) {
  NoiseOps ret;
  const auto x90 = Operations::make_mat({qubit}, Utils::Matrix::X90, "x90");
  if (std::abs(lam - 0.5 * M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, lam - 0.5 * M_PI)); // add 1st U1
  auto sample = sample_noise_helper(x90); // sample noise for 1st X90
  ret.insert(ret.end(), sample.begin(), sample.end()); // add 1st noisy X90
  if (std::abs(phi + 0.5 * M_PI) > u1_threshold_)
    ret.push_back(Operations::make_u1(qubit, phi + 0.5 * M_PI)); // add 2nd U1
  return ret;                               
}


//=========================================================================
// JSON Conversion
//=========================================================================

// Currently we only support a Kraus and unitary gate errors
// the noise config should be an array of gate_error objects:
//  [gate_error1, gate_error2, ...]
// where each gate_error object is of the form
//  {
//    "type": "gate_error",   // string
//    "operations": ["x", "y", "z"], // list string
//    "params": [mat1, mat2, mat3]
//  }
// TODO add reset errors and X90 based errors

void NoiseModel::load_from_json(const json_t &js) {
  // Check json is an array
  if (!js.is_array()) {
    throw std::invalid_argument("Noise params JSON is not an array");
  }

  for (const auto &gate_js : js) {
    std::string type;
    JSON::get_value(type, "type", gate_js);
    std::vector<std::string> ops;
    JSON::get_value(ops, "operations", gate_js);
    std::vector<cmatrix_t> mats;
    JSON::get_value(mats, "params", gate_js);

    if (type == "gate_error") {
      Noise::GateError error(mats);
      add_local_error(error, ops);
    } else {
      throw std::invalid_argument("Invalid noise type (" + type + ")");
    }
  }
}

void from_json(const json_t &js, NoiseModel &model) {
  model = NoiseModel(js);
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif