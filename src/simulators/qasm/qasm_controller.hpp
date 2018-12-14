/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_qasm_controller_hpp_
#define _aer_qasm_controller_hpp_

#include "base/controller.hpp"
#include "simulators/qubitvector/qv_state.hpp"

namespace AER {
namespace Simulator {

//=========================================================================
// QasmController class
//=========================================================================

/**************************************************************************
 * Config settings:
 * 
 * From QubitVector::State class
 * 
 * - "initial_statevector" (json complex vector): Use a custom initial
 *      statevector for the simulation [Default: null].
 * - "chop_threshold" (double): Threshold for truncating small values to
 *      zero in result data [Default: 1e-15]
 * - "statevector_parallel_threshold" (int): Threshold that number of qubits
 *      must be greater than to enable OpenMP parallelization at State
 *      level [Default: 13]
 * - "statevector_sample_measure_opt" (int): Threshold that number of qubits
 *      must be greater than to enable indexing optimization during
 *      measure sampling [Default: 10]
 * - "statevector_hpc_gate_opt" (bool): Enable large qubit gate optimizations.
 *      [Default: False]
 * 
 * From BaseController Class
 *
 * - "noise_model" (json): A noise model to use for simulation [Default: null]
 * - "max_parallel_threads" (int): Set the maximum OpenMP threads that may
 *      be used across all levels of parallelization. Set to 0 for maximum
 *      available. [Default : 0]
 * - "max_parallel_experiments" (int): Set number of circuits that may be
 *      executed in parallel. Set to 0 to use the number of max parallel
 *      threads [Default: 1]
 * - "max_parallel_shots" (int): Set number of shots that maybe be executed
 *      in parallel for each circuit. Sset to 0 to use the number of max
 *      parallel threads [Default: 1].
 * - "counts" (bool): Return counts objecy in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 * - "register" (bool): Return register array in circuit data [Default: False]
 * 
 **************************************************************************/

class QasmController : public Base::Controller {
public:
  //-----------------------------------------------------------------------
  // Base class config override
  //-----------------------------------------------------------------------
  
  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  // Allowed config options:
  // - "initial_statevector: complex_vector"
  // Plus Base Controller config options
  virtual void set_config(const json_t &config) override;

  // Clear the current config
  void virtual clear_config() override;

private:

  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  virtual OutputData run_circuit(const Circuit &circ,
                                 uint_t shots,
                                 uint_t rng_seed,
                                 int num_threads_state) const override;

  //----------------------------------------------------------------
  // Run circuit without optimization
  //----------------------------------------------------------------

  // Execute n-shots of a circuit on the input state
  template <class State_t>
  void run_circuit_default(const Circuit &circ,
                           uint_t shots,
                           State_t &state,
                           OutputData &data,
                           RngEngine &rng) const;

  //----------------------------------------------------------------
  // Measure sampling optimizations
  //----------------------------------------------------------------
  
  // Execute n-shots of a circuit performing measurement sampling
  // if the input circuit supports it
  template <class State_t>
  void run_circuit_measure_sampler(const Circuit &circ,
                                   uint_t shots,
                                   State_t &state,
                                   OutputData &data,
                                   RngEngine &rng) const;


  // Sample measurement outcomes for the input measure ops from the
  // current state of the input State_t
  template <class State_t>
  void measure_sampler(const std::vector<Operations::Op> &meas_ops,
                       uint_t shots,
                       State_t &state,
                       OutputData &data,
                       RngEngine &rng) const;

  // Check if measure sampling optimization if valid for the input circuit
  // if so return a pair {true, pos} where pos is the position of the
  // first measurement operation in the input circuit
  std::pair<bool, size_t> check_measure_sampling_opt(const Circuit &circ) const;
  
  //-----------------------------------------------------------------------
  // Custom initial state
  //-----------------------------------------------------------------------        
  cvector_t initial_state_;
};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Config
//-------------------------------------------------------------------------

void QasmController::set_config(const json_t &config) {
  // Set base controller config
  Base::Controller::set_config(config);
  //Add custom initial state
  if (JSON::get_value(initial_state_, "initial_statevector", config)) {
    // Check initial state is normalized
    if (!Utils::is_unit_vector(initial_state_, 1e-10))
      throw std::runtime_error("QasmController: initial_statevector is not a unit vector");
  }
}

void QasmController::clear_config() {
  Base::Controller::clear_config();
  initial_state_ = cvector_t();
}

//-------------------------------------------------------------------------
// Base class override
//-------------------------------------------------------------------------

OutputData QasmController::run_circuit(const Circuit &circ,
                                      uint_t shots,
                                      uint_t rng_seed,
                                      int num_threads_state) const {  
  // Check if circuit can run on a statevector simulator
  // TODO: Should we make validate circuit a static method of the class?
  bool statevec_valid = QubitVector::State<>().validate_circuit(circ);  
  // throw exception listing the invalid instructions
  if (statevec_valid == false) {
    QubitVector::State<>().validate_circuit_except(circ);
  }

  // Check for custom initial state, and if so check it matches num qubits
  if (!initial_state_.empty()) {
    if (initial_state_.size() != 1ULL << circ.num_qubits) {
      uint_t num_qubits(std::log2(initial_state_.size()));
      std::stringstream msg;
      msg << "QasmController: " << num_qubits << "-qubit initial state ";
      msg << "cannot be used for a " << circ.num_qubits << "-qubit circuit.";
      throw std::runtime_error(msg.str());
    }
  }

  // Initialize statevector
  QubitVector::State<> state;
  state.set_config(Base::Controller::config_);
  state.set_available_threads(num_threads_state);
  
  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  OutputData data;
  data.set_config(Base::Controller::config_);
  
  // Check if there is noise for the implementation
  if (noise_model_.ideal()) {
    // Implement without noise
    run_circuit_measure_sampler(circ, shots, state, data, rng);
  } else {
    // Sample noise for each shot
    while (shots-- > 0) {
      Circuit noise_circ = noise_model_.sample_noise(circ, rng);
      run_circuit_default(noise_circ, 1, state, data, rng);
    }
  }
  return data;
}


//-------------------------------------------------------------------------
// Run circuit helpers
//-------------------------------------------------------------------------

template <class State_t>
void QasmController::run_circuit_default(const Circuit &circ,
                                         uint_t shots,
                                         State_t &state,
                                         OutputData &data,
                                         RngEngine &rng) const {  
  while (shots-- > 0) {
    if (initial_state_.empty())
      state.initialize_qreg(circ.num_qubits);
    else
      state.initialize_qreg(circ.num_qubits, initial_state_);
    state.initialize_creg(circ.num_memory, circ.num_registers);
    state.apply_ops(circ.ops, data, rng);
    state.add_creg_to_data(data);
  }
}


template <class State_t>
void QasmController::run_circuit_measure_sampler(const Circuit &circ,
                                                 uint_t shots,
                                                 State_t &state,
                                                 OutputData &data,
                                                 RngEngine &rng) const {
  // Check if optimization is valid
  auto check = check_measure_sampling_opt(circ);
  // Perform standard execution if we cannot apply the optimization
  // or the execution is only for a single shot
  if (shots == 1 || check.first == false) {
    run_circuit_default(circ, shots, state, data, rng);
    return;
  } 
  auto pos = check.second; // Position of first measurement op
  // Run circuit instructions before first measure
  std::vector<Operations::Op> ops(circ.ops.begin(), circ.ops.begin() + pos);
  if (initial_state_.empty())
    state.initialize_qreg(circ.num_qubits);
  else
    state.initialize_qreg(circ.num_qubits, initial_state_);
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(ops, data, rng);

  // Get measurement operations and set of measured qubits
  ops = std::vector<Operations::Op>(circ.ops.begin() + pos, circ.ops.end());
  measure_sampler(ops, shots, state, data, rng);
}


//-------------------------------------------------------------------------
// Measure sampling optimization
//-------------------------------------------------------------------------

std::pair<bool, size_t> 
QasmController::check_measure_sampling_opt(const Circuit &circ) const {
  // Find first instance of a measurement and check there
  // are no reset operations before the measurement
  auto start = circ.ops.begin();
  while (start != circ.ops.end()) {
    const auto type = start->type;
    if (type == Operations::OpType::reset ||
        type == Operations::OpType::kraus ||
        type == Operations::OpType::roerror) {
      return std::make_pair(false, 0);
    }
    if (type == Operations::OpType::measure)
      break;
    ++start;
  }
  // Record position for if optimization passes
  auto start_meas = start;
  // Check all remaining operations are measurements
  while (start != circ.ops.end()) {
    if (start->type != Operations::OpType::measure) {
      return std::make_pair(false, 0);
    }
    ++start;
  }
  // If we made it this far we can apply the optimization
  size_t meas_pos = start_meas - circ.ops.begin();
  return std::make_pair(true, meas_pos);
}


template <class State_t>
void QasmController::measure_sampler(const std::vector<Operations::Op> &meas_ops,
                                     uint_t shots,
                                     State_t &state,
                                     OutputData &data,
                                     RngEngine &rng) const {                    
  // Check if meas_circ is empty, and if so return initial creg
  if (meas_ops.empty()) {
    while (shots-- > 0) {
      state.add_creg_to_data(data);
    }
    return;
  }
  // Get measured qubits from circuit sort and delete duplicates
  std::vector<uint_t> meas_qubits; // measured qubits
  for (const auto &op : meas_ops) {
    for (size_t j=0; j < op.qubits.size(); ++j)
      meas_qubits.push_back(op.qubits[j]);
  }
  sort(meas_qubits.begin(), meas_qubits.end());
  meas_qubits.erase(unique(meas_qubits.begin(), meas_qubits.end()), meas_qubits.end());
  // Generate the samples
  auto all_samples = state.sample_measure(meas_qubits, shots, rng);

  // Make qubit map of position in vector of measured qubits
  std::unordered_map<uint_t, uint_t> qubit_map;
  for (uint_t j=0; j < meas_qubits.size(); ++j) {
      qubit_map[meas_qubits[j]] = j;
  }

  // Maps of memory and register to qubit position
  std::unordered_map<uint_t, uint_t> memory_map;
  std::unordered_map<uint_t, uint_t> register_map;
  for (const auto &op : meas_ops) {
    for (size_t j=0; j < op.qubits.size(); ++j) {
      auto pos = qubit_map[op.qubits[j]];
      if (!op.memory.empty())
        memory_map[op.memory[j]] = pos;
      if (!op.registers.empty())
        register_map[op.registers[j]] = pos;
    }
  }

  // Process samples
  // Convert opts to circuit so we can get the needed creg sizes
  // NB: this function could probably be moved somewhere else like Utils or Ops
  Circuit meas_circ(meas_ops);
  ClassicalRegister creg;
  while (!all_samples.empty()) {
    auto sample = all_samples.back();
    creg.initialize(meas_circ.num_memory, meas_circ.num_registers);

    // process memory bit measurements
    for (const auto &pair : memory_map) {
      creg.store_measure(reg_t({sample[pair.second]}), reg_t({pair.first}), reg_t());
    }
    auto memory = creg.memory_hex();
    data.add_memory_count(memory);
    data.add_memory_singleshot(memory);

    // process register bit measurements
    for (const auto &pair : register_map) {
      creg.store_measure(reg_t({sample[pair.second]}), reg_t(), reg_t({pair.first}));
    }
    data.add_register_singleshot(creg.register_hex());

    // pop off processed sample
    all_samples.pop_back();
  }
}

//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
