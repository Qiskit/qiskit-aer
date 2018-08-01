/**
 * Copyright 2017, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

/**
 * @file    state_vector_engine.hpp
 * @brief   QISKIT Simulator QubitVector engine class
 * @author  Christopher J. Wood <cjwood@us.ibm.com>
 */

#ifndef _qubitvector_qv_engine_hpp_
#define _qubitvector_qv_engine_hpp_

#include <sstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h> // OpenMP
#endif

#include "simulators/qubitvector/qv_state.hpp"
#include "engines/qasm_engine.hpp"


namespace AER {
namespace QubitVector {

//============================================================================
// Optimized QubitVector QASM Engine class for Qiskit-Aer
//============================================================================

// This is an optimized derived class of the Engines::QasmEngine class for
// use with the QubitVector::State class.
// Optimizations:
// - If a circuit contains no reset operaitons, and all measurements are at the
//   end it can sample measurement outcomes from the final state rather than
//   simulate every shot.
class QasmEngine : public Engines::QasmEngine<state_t> {

public:
  // Default constructor
  explicit QasmEngine() : Engines::QasmEngine<state_t>() {};
  ~QasmEngine() = default;

  //----------------------------------------------------------------
  // Engines::QasmEngine class overrides
  //----------------------------------------------------------------

  virtual void execute(Base::State<state_t> *be,
                       const Circuit &circ,
                       uint_t nshots) override;

protected:

  //----------------------------------------------------------------
  // Additional Methods
  //----------------------------------------------------------------

  bool check_opt_meas(const Circuit &circ) const;

  void execute_with_sampling(Base::State<state_t> *state,
                        const Circuit &circ,
                        uint_t shots);

  template<class T>
  void sample_counts(Base::State<state_t> *state,
                     const Circuit &circ,
                     uint_t shots,
                     const std::vector<T> &probs,
                     const reg_t &memory,
                     const reg_t &registers);

  bool destructive_sample_ = true;
};

//============================================================================
// Implmentations:
//============================================================================

void QasmEngine::execute( Base::State<state_t> *state, const Circuit &circ, uint_t shots) {
                                
  // Check for sampling measurement optimization
  bool meas_opt = check_opt_meas(circ);
  if (meas_opt) {
    execute_with_sampling(state, circ, shots);
  } else {
    Engines::QasmEngine<state_t>::execute(state, circ, shots); // execute without sampling
  }
}


bool QasmEngine::check_opt_meas(const Circuit &circ) const {
  // Find first instance of a measurement and check there
  // are no reset operations before the measurement
  auto start = circ.ops.begin();
  while (start != circ.ops.end()) {
    const auto name = start->name;
    if (name == "reset" || name == "kraus" || name == "roerr")
      return false;
    if (name == "measure")
      break;
    ++start;
  }
  // Check all remaining operations are measurements
  while (start != circ.ops.end()) {
    if (start->name != "measure")
      return false;
    ++start;
  }
  // If we made it this far we can apply the optimization
  return true;
}


void QasmEngine::execute_with_sampling(Base::State<state_t> *state,
                                       const Circuit &circ,
                                       uint_t shots) {                                    
  initialize(state, circ);
  // Find position of first measurement operation
  uint_t pos = 0;
  while (pos < circ.ops.size() && circ.ops[pos].name != "measure") {
    pos++;
  }
  // Execute operations before measurements
  for(auto it = circ.ops.cbegin(); it!=(circ.ops.cbegin() + pos); ++it) {
    apply_op(state, *it);
  }

  // Get measurement operations and set of measured qubits
  std::vector<Operations::Op> meas(circ.ops.begin() + pos, circ.ops.end());
  std::vector<uint_t> meas_qubits; // measured qubits
  std::map<uint_t, uint_t> memory_map; // map of memory locations to qubit measured
  std::map<uint_t, uint_t> registers_map;// map of register locations to qubit measured
  for (const auto &op : meas) {
    for (size_t j=0; j < op.qubits.size(); ++j) {
      meas_qubits.push_back(op.qubits[j]);
      if (!op.memory.empty())
        memory_map[op.qubits[j]] = op.memory[j];
      if (!op.registers.empty())
        registers_map[op.qubits[j]] = op.registers[j];
    }
  }
  // Sort the qubits and delete duplicates
  sort(meas_qubits.begin(), meas_qubits.end());
  meas_qubits.erase(unique(meas_qubits.begin(), meas_qubits.end()), meas_qubits.end());

  // Convert memory and register maps to ordered lists
  reg_t memory;
  if (!memory_map.empty())
    for (const auto &q: meas_qubits)
      memory.push_back(memory_map[q]);
  reg_t registers;
  if (!registers_map.empty())
    for (const auto &q: meas_qubits)
      registers.push_back(registers_map[q]);
  
  // Initialize outcome map for measured qubits
  
  std::map<uint_t, uint_t> outcomes;
  for (auto &qubit : meas_qubits)
    outcomes[qubit] = 0;

  if (destructive_sample_ && meas_qubits.size() == circ.num_qubits) {
    cvector_t &cprobs = state->data().vector();
    for (uint_t j=0; j < cprobs.size(); j++) {
      cprobs[j] = std::real(cprobs[j] * std::conj(cprobs[j]));
    }
    // Sample measurement outcomes
    sample_counts(state, circ, shots, cprobs, memory, registers);
  } else {
    // TODO partial trace over other qubits so it can also be done in place
    // Sample measurement outcomes
    rvector_t probs = state->data().probabilities(meas_qubits);
    sample_counts(state, circ, shots, probs, memory, registers);
  }
}


//------------------------------------------------------------------------------
// Templated so works for real or complex probability vector
template<class T>
void QasmEngine::sample_counts(Base::State<state_t> *state,
                               const Circuit &circ,
                               uint_t shots,
                               const std::vector<T> &probs,
                               const reg_t &memory,
                               const reg_t &registers) {
  // Sample measurement outcomes
  auto &rng = state->access_rng();
  for (uint_t s = 0; s < shots; ++s) {
    double p = 0.;
    double r = rng.rand(0, 1);
    uint_t val;
    for (val = 0; val < (1ULL << circ.num_qubits); val++) {
      if (r < (p += std::real(probs[val])))
        break;
    }
    // convert outcome to register
    const reg_t reg = Utils::int2reg(val, 2, circ.num_qubits);
    initialize_creg(circ); // reset creg for sampling
    store_measure(reg, memory, registers);
    compute_result(state);
  }
}


//------------------------------------------------------------------------------
} // end namespace QubitVector
} // end namespace AER
//------------------------------------------------------------------------------

#endif
