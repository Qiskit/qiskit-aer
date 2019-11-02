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

#ifndef _aer_transpile_delay_measure_hpp_
#define _aer_transpile_delay_measure_hpp_

#include <unordered_map>

#include "transpile/circuitopt.hpp"

namespace AER {
namespace Transpile {

class DelayMeasure : public CircuitOptimization {
public:

  // DelayMeasure uses following configuration options
  // - delay_measure_verbose (bool): if true, output generated gates in metadata (default: false)
  // - delay_measure_enable (bool): if true, activate optimization (default: true)
  void set_config(const json_t &config) override;

  // Push back measurements to tail of circuit if no non-measure
  // instructions follow on the same qubits
  // This allows measure sampling to be used on more circuits
  void optimize_circuit(Circuit& circ,
                        Noise::NoiseModel& noise,
                        const Operations::OpSet &opset,
                        ExperimentData &data) const override;

private:
  // show debug info
  bool verbose_ = false;

  // disabled in config
  bool active_ = true;
};

void DelayMeasure::set_config(const json_t &config) {
  CircuitOptimization::set_config(config);
  JSON::get_value(verbose_, "delay_measure_verbose", config);
  JSON::get_value(active_, "delay_measure_enable", config);
}

void DelayMeasure::optimize_circuit(Circuit& circ,
                                    Noise::NoiseModel& noise,
                                    const Operations::OpSet &allowed_opset,
                                    ExperimentData &data) const {
  // Pass if we have quantum errors in the noise model
  if (active_ == false || circ.shots <= 1 || noise.has_quantum_errors())
    return; 

  // Get position of first measure / readout error in circ
  auto it = circ.ops.begin();
  while (it != circ.ops.end()) {
    const auto type = it->type;
    if (type == Operations::OpType::measure ||
        type == Operations::OpType::roerror)
      break;
    ++it;
  }
  // If there are no measure instructions we don't need to optimize
  if (it == circ.ops.end())
    return;

  // Store measure and non measure instructions in the tail
  // for possible later remapping
  size_t pos = std::distance(circ.ops.begin(), it);
  std::vector<Operations::Op> meas_ops;
  meas_ops.reserve(circ.ops.size() - pos);
  std::vector<Operations::Op> non_meas_ops;
  non_meas_ops.reserve(circ.ops.size() - pos);

  // Scan circuit to find position of first measure for all qubits;
  std::unordered_set<uint_t> meas_qubits;
  while (it != circ.ops.end()) {
    // If any operations are conditional we abort
    if (it->conditional)
      return;
    const auto type = it->type;
    const auto qubits = it->qubits;
    switch (type) {
      case Operations::OpType::measure:
      case Operations::OpType::roerror: {
        meas_qubits.insert(qubits.begin(), qubits.end());
        meas_ops.push_back(*it);
        break;  
      }
      case Operations::OpType::snapshot: {
        return;
      }
      default: {
        for (const auto& qubit : qubits) {
          if (meas_qubits.find(qubit) != meas_qubits.end()) {
            return;
          }
        }
        non_meas_ops.push_back(*it);
      }
    }
    ++it;
  }
  // Check if there are any non-meas instructions in the tail
  if (non_meas_ops.empty())
    return;

  // Now modify the circuit to add meas instructions
  // after non meas instructions
  circ.ops.erase(circ.ops.begin() + pos, circ.ops.end());
  circ.ops.insert(circ.ops.end(), non_meas_ops.begin(), non_meas_ops.end());
  circ.ops.insert(circ.ops.end(), meas_ops.begin(), meas_ops.end());
  
  if (verbose_)
      data.add_metadata("delay_measure_verbose", circ.ops);
}


//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
