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

#ifndef _aer_controller_execute_hpp_
#define _aer_controller_execute_hpp_

#include "misc/hacks.hpp"
#include <string>

#include "framework/config.hpp"
#include "framework/matrix.hpp"
#include "framework/python_parser.hpp"
#include "framework/results/result.hpp"

//=========================================================================
// Controller Execute interface
//=========================================================================

namespace AER {

template <class controller_t, typename inputdata_t>
Result controller_execute(const inputdata_t &qobj) {
  controller_t controller;

  // Fix for MacOS and OpenMP library double initialization crash.
  // Issue: https://github.com/Qiskit/qiskit-aer/issues/1
  if (Parser<inputdata_t>::check_key("config", qobj)) {
    std::string path;
    const auto &config = Parser<inputdata_t>::get_value("config", qobj);
    Parser<inputdata_t>::get_value(path, "library_dir", config);
    Hacks::maybe_load_openmp(path);
  }
  return controller.execute(qobj);
}

template <class controller_t>
Result controller_execute(std::vector<std::shared_ptr<Circuit>> &input_circs,
                          AER::Noise::NoiseModel &noise_model,
                          AER::Config &config) {
  controller_t controller;

  bool truncate = config.enable_truncation;

  if (noise_model.has_nonlocal_quantum_errors())
    truncate = false;

  const size_t num_circs = input_circs.size();

  // Check if parameterized circuits
  // It should be of the form
  // [exp0_params, exp1_params, ...]
  // where:
  //    expk_params = [((i, j), pars), ....]
  //    i is the instruction index in the experiment
  //    j is the param index in the instruction
  //    pars = [par0, par1, ...] is a list of different parameterizations
  using pos_t = std::pair<int_t, int_t>;
  using exp_params_t = std::vector<std::pair<pos_t, std::vector<double>>>;
  std::vector<exp_params_t> &param_table = config.param_table;

  // check if runtime binding is enable
  bool runtime_parameter_bind = false;
  if (config.runtime_parameter_bind_enable.has_value())
    runtime_parameter_bind = config.runtime_parameter_bind_enable.value();

  // Validate parameterizations for number of circuis
  if (!param_table.empty() && param_table.size() != num_circs) {
    throw std::invalid_argument(
        R"(Invalid parameterized circuits: "parameterizations" length does not match number of circuits.)");
  }

  std::vector<std::shared_ptr<Circuit>> circs;
  std::vector<std::shared_ptr<Circuit>> template_circs;

  using myclock_t = std::chrono::high_resolution_clock;
  auto timer_start = myclock_t::now();
  try {
    // Load circuits
    for (size_t i = 0; i < num_circs; i++) {
      auto &circ = input_circs[i];
      if (param_table.empty() || param_table[i].empty()) {
        // Non parameterized circuit
        circ->set_params(truncate);
        circ->set_metadata(config, truncate);
        circs.push_back(circ);
        template_circs.push_back(circ);
      } else {
        // Get base circuit without truncation
        circ->set_params(false);
        circ->set_metadata(config, truncate);
        // Load different parameterizations of the initial circuit
        const auto &circ_params = param_table[i];
        const size_t num_params = circ_params[0].second.size();
        const size_t num_instr = circ->ops.size();

        if (runtime_parameter_bind && num_params > 1) {
          // Make a copy of the initial circuit
          auto param_circ = std::make_shared<Circuit>(*circ);
          param_circ->num_bind_params = num_params;

          for (const auto &params : circ_params) {
            const auto instr_pos = params.first.first;
            const auto param_pos = params.first.second;
            // Validation
            if (instr_pos == AER::Config::GLOBAL_PHASE_POS) {
              // negative position is for global phase
              param_circ->global_phase_for_params.resize(num_params);
              for (size_t j = 0; j < num_params; j++)
                param_circ->global_phase_for_params[j] = params.second[j];
            } else if ((uint_t)instr_pos >= num_instr) {
              throw std::invalid_argument(
                  R"(Invalid parameterized qobj: instruction position out of range)");
            }
            auto &op = param_circ->ops[instr_pos];
            if (!op.has_bind_params) {
              if ((uint_t)param_pos >= op.params.size()) {
                throw std::invalid_argument(
                    R"(Invalid parameterized qobj: instruction param position out of range)");
              }
              // resize parameter array
              op.params.resize(op.params.size() * num_params);
              op.has_bind_params = true;
            }
            uint_t stride = op.params.size() / num_params;
            for (size_t j = 0; j < num_params; j++)
              op.params[param_pos + stride * j] = params.second[j];
          }
          // Run truncation.
          // TODO: Truncation should be performed and parameters should be
          // resolved after it. However, parameters are associated with indices
          // of instructions, which can be changed in truncation. Therefore,
          // current implementation performs truncation for each parameter set.
          if (truncate) {
            param_circ->set_params(true);
            param_circ->set_metadata(config, true);
          }
          circs.push_back(param_circ);
          for (size_t j = 0; j < num_params; j++)
            template_circs.push_back(circ);
        } else {
          for (size_t j = 0; j < num_params; j++) {
            // Make a copy of the initial circuit
            auto param_circ = std::make_shared<Circuit>(*circ);
            for (const auto &params : circ_params) {
              const auto instr_pos = params.first.first;
              const auto param_pos = params.first.second;
              // Validation
              if (instr_pos == AER::Config::GLOBAL_PHASE_POS) {
                // negative position is for global phase
                circ->global_phase_angle = params.second[j];
              } else {
                if ((uint_t)instr_pos >= num_instr) {
                  std::cout << "Invalid parameterization: instruction position "
                               "out of range: "
                            << instr_pos << std::endl;
                  throw std::invalid_argument(
                      R"(Invalid parameterization: instruction position out of range)");
                }
                auto &op = param_circ->ops[instr_pos];
                if ((uint_t)param_pos >= op.params.size()) {
                  throw std::invalid_argument(
                      R"(Invalid parameterization: instruction param position out of range)");
                }
                if (j >= params.second.size()) {
                  throw std::invalid_argument(
                      R"(Invalid parameterization: parameterization value out of range)");
                }
                // Update the param
                op.params[param_pos] = params.second[j];
              }
            }
            // Run truncation.
            // TODO: Truncation should be performed and parameters should be
            // resolved after it. However, parameters are associated with
            // indices of instructions, which can be changed in truncation.
            // Therefore, current implementation performs truncation for each
            // parameter set.
            if (truncate) {
              param_circ->set_params(true);
              param_circ->set_metadata(config, true);
            }
            circs.push_back(param_circ);
            template_circs.push_back(circ);
          }
        }
      }
    }
  } catch (std::exception &e) {
    Result result;

    result.status = Result::Status::error;
    result.message = std::string("Failed to load circuits: ") + e.what();
    return result;
  }
  int_t seed = -1;
  uint_t seed_shift = 0;

  if (config.seed_simulator.has_value())
    seed = config.seed_simulator.value();
  else
    seed = circs[0]->seed;

  if (runtime_parameter_bind) {
    for (auto &circ : circs) {
      circ->seed = seed + seed_shift;
      circ->seed_for_params.resize(circ->num_bind_params);
      for (uint_t i = 0; i < circ->num_bind_params; i++) {
        circ->seed_for_params[i] = seed + seed_shift;
        seed_shift += 2113;
      }
    }
  } else {
    for (auto &circ : circs) {
      circ->seed = seed + seed_shift;
      seed_shift += 2113;
    }
  }
  auto time_taken =
      std::chrono::duration<double>(myclock_t::now() - timer_start).count();

  // Fix for MacOS and OpenMP library double initialization crash.
  // Issue: https://github.com/Qiskit/qiskit-aer/issues/1
  Hacks::maybe_load_openmp(config.library_dir);
  controller.set_config(config);
  auto ret = controller.execute(circs, noise_model, config);

  for (size_t i = 0; i < ret.results.size(); ++i)
    ret.results[i].circ_id = template_circs[i]->circ_id;
  ret.metadata.add(time_taken, "time_taken_parameter_binding");

  return ret;
}

} // end namespace AER
#endif
