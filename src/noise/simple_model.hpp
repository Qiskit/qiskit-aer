/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    simple_model.hpp
 * @brief   Simple noise model for Qiskit-Aer simulator
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _aer_noise_simple_model_hpp_
#define _aer_noise_simple_model_hpp_

#include <unordered_map>

#include "framework/json.hpp"
#include "base/noise.hpp"
#include "noise/gate_error.hpp"

namespace AER {
namespace Noise {


//=========================================================================
// Simple Noise Model class
//=========================================================================

// This noise model allows adding a single Error subclasses for each circuit
// operation string.
// It assumes the error acts on the same qubits as the operation, and that
// the error operation is the same for all qubits.

class SimpleModel : public Model {
public: 

  SimpleModel() = default;
  SimpleModel(const json_t &js) {load_from_json(js);}

  // Sample noise for the current operation
  NoiseOps sample_noise(const Operations::Op &op) override;

  // Load a noise model from JSON
  void load_from_json(const json_t &js);

  // Add a Error subclass to the model
  template <class DerivedError>
  void add_error(const DerivedError &error,
                 const std::vector<std::string> &op_labels);

private:
  std::vector<std::unique_ptr<Error>> errors_;
  // Table indexes a name with a vector of the position of noise operations
  std::unordered_map<std::string, std::vector<size_t>> error_table_;
};

//-------------------------------------------------------------------------
// Implementation
//-------------------------------------------------------------------------

NoiseOps SimpleModel::sample_noise(const Operations::Op &op) {
  // Get the error model
  auto iter = error_table_.find(op.name);
  if (iter == error_table_.end())
    return {op};// error not found

  NoiseOps ret;
  for (const auto &pos : iter->second) {
    auto ops = errors_[pos]->sample_noise(op, op.qubits, rng_);
    ret.insert(ret.end(), ops.begin(), ops.end());
  }
  return ret;
}

template <class DerivedError>
void SimpleModel::add_error(const DerivedError &error,
                            const std::vector<std::string> &op_labels) {  
  // Add error term as unique pointer                                          
  errors_.push_back(std::unique_ptr<Error>(std::make_unique<DerivedError>(error))); 
  for (const auto &gate: op_labels) {
    error_table_[gate].push_back(errors_.size() - 1); // position is vector length - 1
  }
}

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

void SimpleModel::load_from_json(const json_t &js) {
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
      GateError error(mats);
      add_error(error, ops);
    } else {
      throw std::invalid_argument("Invalid noise type (" + type + ")");
    }
  }
}

void from_json(const json_t &js, SimpleModel &model) {
  model = SimpleModel(js);
}

//-------------------------------------------------------------------------
} // end namespace Noise
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif