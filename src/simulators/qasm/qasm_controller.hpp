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
#include "simulators/stabilizer/stabilizer_state.hpp"


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
  // Simulation types
  //-----------------------------------------------------------------------

  // Simulation methods for the Qasm Controller
  enum class Method {automatic, statevector, stabilizer};

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
  // Utility functions
  //----------------------------------------------------------------

  // Return the simulation method to use for the input circuit
  // If a custom method is specified in the config this will be
  // used. If the default automatic method is set this will choose
  // the appropriate method based on the input circuit.
  Method simulation_method(const Circuit &circ) const;

  // Initialize a State subclass to a given initial state
  template <class State_t, class Initstate_t>
  void initialize_state(const Circuit &circ,
                        State_t &state,
                        const Initstate_t &initial_state) const;

  // Optimize a circuit based on simulator config, and store the optimized
  // circuit in output_circ
  // To optimize in place use output_circ = input_circ
  // NOTE: That is a place-holder and no optimization passes are implemented
  template <class State_t>
  void optimize_circuit(const Circuit& input_circ, Circuit& output_circ) const;

  //----------------------------------------------------------------
  // Run circuit helpers
  //----------------------------------------------------------------

  // Execute n-shots of a circuit on the input state
  template <class State_t, class Initstate_t>
  OutputData run_circuit_helper(const Circuit &circ,
                                uint_t shots,
                                uint_t rng_seed,
                                int num_threads_state,
                                const Initstate_t &initial_state) const;

  // Execute a single shot a circuit by initializing the state vector
  // to initial_state, running all ops in circ, and updating data with
  // simulation output.
  template <class State_t, class Initstate_t>
  void run_single_shot(const Circuit &circ,
                       State_t &state,
                       const Initstate_t &initial_state,
                       OutputData &data,
                       RngEngine &rng) const;

  // Execute a n-shots of a circuit without noise.
  // If possible this is done using measure sampling to only simulate
  // a single shot up to the first measurement, then sampling measure
  // outcomes for each shot.
  template <class State_t, class Initstate_t>
  void run_circuit_without_noise(const Circuit &circ,
                                 uint_t shots,
                                 State_t &state,
                                 const Initstate_t &initial_state,
                                 OutputData &data,
                                 RngEngine &rng) const;

  // Execute n-shots of a circuit with noise by sampling a new noisy
  // instance of the circuit for each shot.
  template <class State_t, class Initstate_t>
  void run_circuit_with_noise(const Circuit &circ,
                              uint_t shots,
                              State_t &state,
                              const Initstate_t &initial_state,
                              OutputData &data,
                              RngEngine &rng) const;

  //----------------------------------------------------------------
  // Measure sampling optimization
  //----------------------------------------------------------------

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
  // Config
  //-----------------------------------------------------------------------

  // Simulation method
  Method simulation_method_ = Method::automatic;

  // Initial statevector for Statevector simulation method
  cvector_t initial_statevector_;

  // TODO: initial stabilizer state
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

  // Override automatic simulation method with a fixed method
  std::string method;
  if (JSON::get_value(method, "method", config)) {
    if (method == "statevector")
      simulation_method_ = Method::statevector;
    else if (method == "stabilizer")
      simulation_method_ = Method::stabilizer;
    else if (method != "automatic")
      throw std::runtime_error(std::string("QasmController: Invalid simulation method.") + method);
  }

  //Add custom initial state
  if (JSON::get_value(initial_statevector_, "initial_statevector", config)) {
    // Raise error if method is set to stabilizer
    if (simulation_method_ == Method::stabilizer) {
      throw std::runtime_error(std::string("QasmController: Using an initial statevector") +
                               std::string(" is not valid with stabilizer simulation method.") +
                               method);
    }
    // Override simulator method to statevector
    simulation_method_ = Method::statevector;
    // Check initial state is normalized
    if (!Utils::is_unit_vector(initial_statevector_, 1e-10)) {
      throw std::runtime_error("QasmController: initial_statevector is not a unit vector");
    }
  }
}

void QasmController::clear_config() {
  Base::Controller::clear_config();
  simulation_method_ = Method::automatic;
  initial_statevector_ = cvector_t();
}

//-------------------------------------------------------------------------
// Base class override
//-------------------------------------------------------------------------

OutputData QasmController::run_circuit(const Circuit &circ,
                                       uint_t shots,
                                       uint_t rng_seed,
                                       int num_threads_state) const {
  // Execute according to simulation method
  switch (simulation_method(circ)) {
    case Method::statevector:
      // Statvector simulation
      return run_circuit_helper<QubitVector::State<>>(circ,
                                                      shots,
                                                      rng_seed,
                                                      num_threads_state,
                                                      initial_statevector_); // allow custom initial state
    case Method::stabilizer:
      // Stabilizer simulation
      // TODO: Stabilizer doesn't yet support custom state initialization
      return run_circuit_helper<Stabilizer::State>(circ,
                                                   shots,
                                                   rng_seed,
                                                   num_threads_state,
                                                   Clifford::Clifford()); // no custom initial state
    default:
      // We shouldn't get here, so throw an exception if we do
      throw std::runtime_error("QasmController:Invalid simulation method");
  }
}

//-------------------------------------------------------------------------
// Utility methods
//-------------------------------------------------------------------------

QasmController::Method QasmController::simulation_method(const Circuit &circ) const {
  // Check conditions for automatic simulation types
  auto method = simulation_method_;
  if (method == Method::automatic) {
    // Check if Clifford circuit and noise model
    if (validate_state(Stabilizer::State(), circ, noise_model_, false))
      method = Method::stabilizer;
    // Default method is statevector
    else
      method = Method::statevector;
  }
  return method;
}


template <class State_t, class Initstate_t>
void QasmController::initialize_state(const Circuit &circ,
                                      State_t &state,
                                      const Initstate_t &initial_state) const {
  if (initial_state.empty()) {
    state.initialize_qreg(circ.num_qubits);
  } else {
    state.initialize_qreg(circ.num_qubits, initial_state);
  }
  state.initialize_creg(circ.num_memory, circ.num_registers);
}


template <class State_t>
void QasmController::optimize_circuit(const Circuit &input_circ,
                                      Circuit &output_circ) const {
  output_circ = input_circ;
  // Add optimization passes here
}


//-------------------------------------------------------------------------
// Run circuit helpers
//-------------------------------------------------------------------------

template <class State_t, class Initstate_t>
OutputData QasmController::run_circuit_helper(const Circuit &circ,
                                              uint_t shots,
                                              uint_t rng_seed,
                                              int num_threads_state,
                                              const Initstate_t &initial_state) const {  
  // Initialize new state object
  State_t state;

  // Valid state again and raise exeption if invalid ops
  validate_state(state, circ, noise_model_, true);

  // Set state config
  state.set_config(Base::Controller::config_);
  state.set_available_threads(num_threads_state);

  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  OutputData data;
  data.set_config(Base::Controller::config_);
  data.add_additional_data("metadata",
                           json_t::object({{"method", state.name()}}));

  // Check if there is noise for the implementation
  if (noise_model_.ideal()) {
    run_circuit_without_noise(circ, shots, state, initial_state, data, rng);
  } else {
    run_circuit_with_noise(circ, shots, state, initial_state, data, rng);
  }
  return data;
}


template <class State_t, class Initstate_t>
void QasmController::run_single_shot(const Circuit &circ,
                                     State_t &state,
                                     const Initstate_t &initial_state,
                                     OutputData &data,
                                     RngEngine &rng) const {
  initialize_state(circ, state, initial_state);
  state.apply_ops(circ.ops, data, rng);
  state.add_creg_to_data(data);
}


template <class State_t, class Initstate_t>
void QasmController::run_circuit_with_noise(const Circuit &circ,
                                            uint_t shots,
                                            State_t &state,
                                            const Initstate_t &initial_state,
                                            OutputData &data,
                                            RngEngine &rng) const {
  // Sample a new noise circuit and optimize for each shot
  while(shots-- > 0) {
    Circuit noise_circ = noise_model_.sample_noise(circ, rng);
    optimize_circuit<State_t>(noise_circ, noise_circ);
    run_single_shot(noise_circ, state, initial_state, data, rng);
  }                                   
}


template <class State_t, class Initstate_t>
void QasmController::run_circuit_without_noise(const Circuit &circ,
                                               uint_t shots,
                                               State_t &state,
                                               const Initstate_t &initial_state,
                                               OutputData &data,
                                               RngEngine &rng) const {
  // Optimize circuit for state type
  Circuit opt_circ;
  optimize_circuit<State_t>(circ, opt_circ);

  // Check if measure sampler and optimization are valid
  auto check = check_measure_sampling_opt(opt_circ);
  if (check.first == false || shots == 1) {
    // Perform standard execution if we cannot apply the
    // measurement sampling optimization
    while(shots-- > 0) {
      run_single_shot(opt_circ, state, initial_state, data, rng);
    }
  } else {
    // Implement measure sampler
    auto pos = check.second; // Position of first measurement op

    // Run circuit instructions before first measure
    std::vector<Operations::Op> ops(opt_circ.ops.begin(), opt_circ.ops.begin() + pos);
    initialize_state(opt_circ, state, initial_state);
    state.apply_ops(ops, data, rng);

    // Get measurement operations and set of measured qubits
    ops = std::vector<Operations::Op>(opt_circ.ops.begin() + pos, opt_circ.ops.end());
    measure_sampler(ops, shots, state, data, rng);
  }                                
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
  //size_t meas_pos = start_meas - circ.ops.begin();
  size_t meas_pos = std::distance(circ.ops.begin(), start_meas);
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
