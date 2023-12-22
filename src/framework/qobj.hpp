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

#ifndef _aer_framework_qobj_hpp_
#define _aer_framework_qobj_hpp_

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/circuit.hpp"
#include "noise/noise_model.hpp"

namespace AER {

//============================================================================
// Qobj data structure
//============================================================================

class Qobj {
public:
  //----------------------------------------------------------------
  // Constructors
  //----------------------------------------------------------------

  // Default constructor and destructors
  Qobj() = default;
  virtual ~Qobj() = default;

  // Deserialization constructor
  template <typename inputdata_t>
  Qobj(const inputdata_t &input);

  //----------------------------------------------------------------
  // Data
  //----------------------------------------------------------------
  std::string id;            // qobj identifier passed to result
  std::string type = "QASM"; // currently we only support QASM
  std::vector<std::shared_ptr<Circuit>> circuits; // List of circuits
  json_t header;                 // (optional) passed through to result
  json_t config;                 // (optional) qobj level config data
  Noise::NoiseModel noise_model; // (optional) noise model
};

//============================================================================
// JSON initialization and deserialization
//============================================================================

// JSON deserialization
inline void from_json(const json_t &js, Qobj &qobj) { qobj = Qobj(js); }

template <typename inputdata_t>
Qobj::Qobj(const inputdata_t &input) {
  // Check required fields
  if (Parser<inputdata_t>::get_value(id, "qobj_id", input) == false) {
    throw std::invalid_argument(R"(Invalid qobj: no "qobj_id" field)");
  };
  Parser<inputdata_t>::get_value(type, "type", input);
  if (type != "QASM") {
    throw std::invalid_argument(R"(Invalid qobj: "type" != "QASM".)");
  };
  if (Parser<inputdata_t>::check_key("experiments", input) == false) {
    throw std::invalid_argument(R"(Invalid qobj: no "experiments" field.)");
  }

  // Apply qubit truncation
  bool truncation = true;

  // Parse config
  if (Parser<inputdata_t>::get_value(config, "config", input)) {
    // Check for truncation option
    Parser<json_t>::get_value(truncation, "enable_truncation", config);

    // Load noise model
    if (Parser<json_t>::get_value(noise_model, "noise_model", config)) {
      // If noise model has non-local errors disable trunction
      if (noise_model.has_nonlocal_quantum_errors()) {
        truncation = false;
      }
    }
  } else {
    config = json_t::object();
  }

  // Parse header
  if (!Parser<inputdata_t>::get_value(header, "header", input)) {
    header = json_t::object();
  }

  // Check for fixed simulator seed
  // If simulator seed is set, each experiment will be set to a fixed (but
  // different) seed Otherwise a random seed will be chosen for each experiment
  int_t seed = -1;
  uint_t seed_shift = 0;
  bool has_simulator_seed = Parser<json_t>::get_value(
      seed, "seed_simulator", config); // config always json
  const auto &circs = Parser<inputdata_t>::get_list("experiments", input);
  const size_t num_circs = circs.size();

  // Check if parameterized qobj
  // It should be of the form
  // [exp0_params, exp1_params, ...]
  // where:
  //    expk_params = [((i, j), pars), ....]
  //    i is the instruction index in the experiment
  //    j is the param index in the instruction
  //    pars = [par0, par1, ...] is a list of different parameterizations
  using pos_t = std::pair<int_t, int_t>;
  using exp_params_t = std::vector<std::pair<pos_t, std::vector<double>>>;
  std::vector<exp_params_t> param_table;
  Parser<json_t>::get_value(param_table, "parameterizations", config);

  // Validate parameterizations for number of circuis
  if (!param_table.empty() && param_table.size() != num_circs) {
    throw std::invalid_argument(
        R"(Invalid parameterized qobj: "parameterizations" length does not match number of circuits.)");
  }

  // Load circuits
  for (size_t i = 0; i < num_circs; i++) {
    if (param_table.empty() || param_table[i].empty()) {
      // Get base circuit from qobj
      auto circuit = std::make_shared<Circuit>(
          static_cast<inputdata_t>(circs[i]), config, truncation);
      // Non parameterized circuit
      circuits.push_back(circuit);
    } else {
      // Get base circuit from qobj without truncation
      auto circuit = std::make_shared<Circuit>(
          static_cast<inputdata_t>(circs[i]), config, false);
      // Load different parameterizations of the initial circuit
      const auto circ_params = param_table[i];
      const size_t num_params = circ_params[0].second.size();
      const size_t num_instr = circuit->ops.size();
      for (size_t j = 0; j < num_params; j++) {
        // Make a copy of the initial circuit
        auto param_circuit = std::make_shared<Circuit>(*circuit);
        for (const auto &params : circ_params) {
          const auto instr_pos = params.first.first;
          const auto param_pos = params.first.second;
          // Validation
          if (instr_pos == AER::Config::GLOBAL_PHASE_POS) {
            // negative position is for global phase
            param_circuit->global_phase_angle = params.second[j];
          } else {
            if ((uint_t)instr_pos >= num_instr) {
              throw std::invalid_argument(
                  R"(Invalid parameterized qobj: instruction position out of range)");
            }
            auto &op = param_circuit->ops[instr_pos];
            if ((uint_t)param_pos >= op.params.size()) {
              throw std::invalid_argument(
                  R"(Invalid parameterized qobj: instruction param position out of range)");
            }
            if (j >= params.second.size()) {
              throw std::invalid_argument(
                  R"(Invalid parameterized qobj: parameterization value out of range)");
            }
            // Update the param
            op.params[param_pos] = params.second[j];
          }
        }
        // Run truncation.
        // TODO: Truncation should be performed and parameters should be
        // resolved after it. However, parameters are associated with indices of
        // instructions, which can be changed in truncation. Therefore, current
        // implementation performs truncation for each parameter set.
        if (truncation)
          param_circuit->set_params(true);
        circuits.push_back(param_circuit);
      }
    }
  }
  // Override random seed with fixed seed if set
  // We shift the seed for each successive experiment
  // So that results aren't correlated between experiments
  if (!has_simulator_seed) {
    seed = circuits[0]->seed;
  }
  for (auto &circuit : circuits) {
    circuit->seed = seed + seed_shift;
    seed_shift += 2113; // Shift the seed
  }
}

//------------------------------------------------------------------------------
} // namespace AER
//------------------------------------------------------------------------------
#endif
