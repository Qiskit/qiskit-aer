/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
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

#include <chrono>

#include "transpile/circuitopt.hpp"
#include "framework/avx2_detect.hpp"
#include "fusion_method.hpp"
#include "fusion/diagonal_fusion.hpp"
#include "fusion/diagonal.hpp"
#include "fusion/entangler.hpp"
#include "fusion/n_qubits_fusion.hpp"
#include "fusion/cost_based_fusion.hpp"

namespace AER {
namespace Transpile {

template<typename Fuser>
class FusionOptimization : public CircuitOptimization {
public:
  // constructor
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
   * - fusion_parallel_threshold (int): Threshold that number of qubits must be greater
   *       than to enable parallelization [Default: 100000]
   */
  FusionOptimization(std::shared_ptr<FusionMethod> method_ = std::make_shared<FusionMethod>())
    : method(method_), fuser(Fuser(method_)) { }
  
  virtual ~FusionOptimization() {}

  void set_config(const json_t &config) override;

  void set_parallelization(uint_t num) { parallelization = num; };

  void set_active(const bool active_) { active = active_; }

  void optimize_circuit(Circuit& circ,
                        Noise::NoiseModel& noise,
                        const opset_t &allowed_opset,
                        ExperimentResult &data) const override;

private:
  void dump_op_in_circuit(const oplist_t& ops, uint_t op_idx) const;

private:
  const std::shared_ptr<FusionMethod> method;
  Fuser fuser;
  bool verbose = false;
  bool debug = false;
  bool active = true;
  uint_t parallelization = 1;
  uint_t parallel_threshold = 10000;
};

template<typename Fuser>
void FusionOptimization<Fuser>::set_config(const json_t &config) {

  CircuitOptimization::set_config(config);

  if (JSON::check_key("fusion_verbose", config_))
    JSON::get_value(verbose, "fusion_verbose", config_);

  if (JSON::check_key("fusion_enable", config_))
    JSON::get_value(active, "fusion_enable", config_);

  if (JSON::check_key("fusion_parallel_threshold", config))
    JSON::get_value(parallel_threshold, "fusion_parallel_threshold", config);

  if (JSON::check_key("fusion_debug", config))
    JSON::get_value(debug, "fusion_debug", config);

  std::stringstream fusion_debug;
  fusion_debug << "fusion_debug." <<fuser.name();

  if (JSON::check_key(fusion_debug.str(), config))
    JSON::get_value(debug, fusion_debug.str(), config);

  fuser.set_config(config);
}

template<typename Fuser>
void FusionOptimization<Fuser>::dump_op_in_circuit(const oplist_t& ops, uint_t op_idx) const {
  std::cout << std::setw(3) << op_idx << ": ";
  if (ops[op_idx].type == optype_t::nop) {
    std::cout << std::setw(10) << "nop" << ": ";
  } else {
    std::cout << std::setw(10) << ops[op_idx].name << ": ";
    if (ops[op_idx].qubits.size() > 0) {
      auto qubits = ops[op_idx].qubits;
      std::sort(qubits.begin(), qubits.end());
      int pos = 0;
      for (int j = 0; j < qubits.size(); ++j) {
        int q_pos = 1 + qubits[j] * 2;
        for (int k = 0; k < (q_pos - pos); ++k) {
          std::cout << " ";
        }
        pos = q_pos + 1;
        std::cout << "X";
      }
    }
  }
  std::cout << std::endl;
}

template<typename Fuser>
void FusionOptimization<Fuser>::optimize_circuit(Circuit& circ,
                              Noise::NoiseModel& noise,
                              const opset_t &allowed_opset,
                              ExperimentResult &data) const {
  // Check if fusion should be skipped
  if (!active || !allowed_opset.contains(optype_t::diagonal_matrix))
    return;

  // Start timer
  using clock_t = std::chrono::high_resolution_clock;
  auto timer_start = clock_t::now();

  // Fusion metadata container
  json_t metadata;
  metadata["applied"] = false;
  metadata["method"] = method->name();
  metadata["threshold"] = fuser.get_threshold();

  if (circ.num_qubits < fuser.get_threshold()) {
    data.add_metadata(fuser.name(), metadata);
    return;
  }
  // Apply fusion
  int applied = 0;

  if (debug) {
    std::cout << "before " << fuser.name() << ":" << std::endl;
    for (int op_idx = 0; op_idx < circ.ops.size(); ++op_idx)
      dump_op_in_circuit(circ.ops, op_idx);
  }

  if (circ.ops.size() < parallel_threshold) {
    applied = fuser.aggregate_operations(circ.num_qubits, circ.ops, 0, circ.ops.size())? 1:0;
  } else {
    auto unit = circ.ops.size() / parallelization;
    if (circ.ops.size() % parallelization)
      ++unit;
#pragma omp parallel for num_threads(parallelization) reduction(+:applied)
    for(int_t fusion_start0 = 0; fusion_start0 < circ.ops.size(); fusion_start0 += unit) {
      auto fusion_end0 = (fusion_start0 + unit) < circ.ops.size()? fusion_start0 + unit: circ.ops.size();
      if (fuser.aggregate_operations(circ.num_qubits, circ.ops, fusion_start0, fusion_end0))
        applied += 1;
    }
  }

  if (applied) {
    size_t idx = 0;
    for (size_t i = 0; i < circ.ops.size(); ++i) {
      if (circ.ops[i].type != optype_t::nop) {
        if (i != idx)
          circ.ops[idx] = circ.ops[i];
        ++idx;
      }
    }

    if (idx != circ.ops.size())
      circ.ops.erase(circ.ops.begin() + idx, circ.ops.end());
    metadata["applied"] = true;

    // Update circuit params for fused circuit
    circ.set_params();
  }

  // Final metadata
  if (verbose && applied) {
    metadata["input_ops"] = circ.ops;
    metadata["output_ops"] = circ.ops;
  }
  auto timer_stop = clock_t::now();
  metadata["time_taken"] = std::chrono::duration<double>(timer_stop - timer_start).count();
  data.add_metadata(fuser.name(), metadata);

  if (debug) {
    std::cout << "after " << fuser.name() << ":" << std::endl;
    for (int op_idx = 0; op_idx < circ.ops.size(); ++op_idx)
      dump_op_in_circuit(circ.ops, op_idx);
  }
}

class Fusion : public CircuitOptimization {
public:
  Fusion(std::shared_ptr<FusionMethod> method = std::make_shared<FusionMethod>())
    : two_qubit_fusion(method), three_qubit_fusion(method), cost_based_fusion(method) { }

  virtual ~Fusion() {}

  void set_config(const json_t &config) override;

  void set_parallelization(uint_t num);

  void optimize_circuit(Circuit& circ,
                        Noise::NoiseModel& noise,
                        const opset_t &allowed_opset,
                        ExperimentResult &data) const override;

private:
  Transpile::FusionOptimization<Transpile::DiagonalFusion> diagonal_fusion;
  Transpile::FusionOptimization<Transpile::DiagonalMerge> diagonal_merge;
  Transpile::FusionOptimization<Transpile::EntanglerFusion> entangler_fusion;
  Transpile::FusionOptimization<Transpile::NQubitFusion<2>> two_qubit_fusion;
  Transpile::FusionOptimization<Transpile::NQubitFusion<3>> three_qubit_fusion;
  Transpile::FusionOptimization<Transpile::CostBasedFusion> cost_based_fusion;
};

void Fusion::set_config(const json_t &config) {
  diagonal_fusion.set_config(config);
  diagonal_merge.set_config(config);
  entangler_fusion.set_config(config);
  two_qubit_fusion.set_config(config);
  three_qubit_fusion.set_config(config);
  cost_based_fusion.set_config(config);
}

void Fusion::set_parallelization(uint_t num) {
  diagonal_fusion.set_parallelization(num);
  diagonal_merge.set_parallelization(num);
  entangler_fusion.set_parallelization(num);
  two_qubit_fusion.set_parallelization(num);
  three_qubit_fusion.set_parallelization(num);
  cost_based_fusion.set_parallelization(num);
}

void Fusion::optimize_circuit(Circuit& circ,
                                  Noise::NoiseModel& noise,
                                  const opset_t &allowed_opset,
                                  ExperimentResult &data) const {

  Noise::NoiseModel dummy_noise; //ignore noise
  diagonal_fusion.optimize_circuit(circ, dummy_noise, allowed_opset, data);
  entangler_fusion.optimize_circuit(circ, dummy_noise, allowed_opset, data);
  three_qubit_fusion.optimize_circuit(circ, dummy_noise, allowed_opset, data);
  two_qubit_fusion.optimize_circuit(circ, dummy_noise, allowed_opset, data);
  cost_based_fusion.optimize_circuit(circ, dummy_noise, allowed_opset, data);
  diagonal_merge.optimize_circuit(circ, dummy_noise, allowed_opset, data);
}


//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------

#endif
