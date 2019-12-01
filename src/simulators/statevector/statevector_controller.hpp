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

#ifndef _aer_statevector_controller_hpp_
#define _aer_statevector_controller_hpp_

#include "base/controller.hpp"
#include "statevector_state.hpp"
#include <list>

// TODO: remove the following debug pring before final push
#define PRINT_TRACE  std::cout << __FILE__ << ":" << __FUNCTION__ << "." << __LINE__ << std::endl;


namespace AER {
namespace Simulator {

//=========================================================================
// StatevectorController class
//=========================================================================

/**************************************************************************
 * Config settings:
 * 
 * From Statevector::State class
 * 
 * - "initial_statevector" (json complex vector): Use a custom initial
 *      statevector for the simulation [Default: null].
 * - "zero_threshold" (double): Threshold for truncating small values to
 *      zero in result data [Default: 1e-10]
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
 * - "max_parallel_threads" (int): Set the maximum OpenMP threads that may
 *      be used across all levels of parallelization. Set to 0 for maximum
 *      available. [Default : 0]
 * - "max_parallel_experiments" (int): Set number of circuits that may be
 *      executed in parallel. Set to 0 to use the number of max parallel
 *      threads [Default: 1]
 * - "counts" (bool): Return counts object in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 * - "register" (bool): Return register array in circuit data [Default: False]
 * 
 **************************************************************************/

class StatevectorController : public Base::Controller {
public:
  //-----------------------------------------------------------------------
  // Base class config override
  //-----------------------------------------------------------------------
  StatevectorController();

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  // Allowed config options:
  // - "initial_statevector: complex_vector"
  // Plus Base Controller config options
  virtual void set_config(const json_t &config) override;

  // Clear the current config
  void virtual clear_config() override;

protected:
  struct ManyWorldNode {
  	~ManyWorldNode() {delete _0_outcome_; delete _1_outcome_; } // TODO: test the recursive deletion

	  Statevector::State<> statevec_;
	  ManyWorldNode *_0_outcome_ = NULL;
	  ManyWorldNode *_1_outcome_ = NULL;
	  double _0_outcome_prob_ = 0;  // TODO: what if we're at a different measurement basis ?
	  double _1_outcome_prob_ = 0; // TODO: what if we're at a different measurement basis ?
	  // TODO: store also the split qubit
  };


  virtual size_t required_memory_mb(const Circuit& circuit,
                                    const Noise::NoiseModel& noise) const override;

private:

  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // This simulator will only return a single shot, regardless of the
  // input shot number
  virtual ExperimentData run_circuit(const Circuit &circ,
                                 const Noise::NoiseModel& noise,
                                 const json_t &config,
                                 uint_t shots,
                                 uint_t rng_seed) const override;


  ExperimentData run_circuit_many_worlds(const Circuit &circ,
          const Noise::NoiseModel& noise,
          const json_t &config,
          uint_t shots,
          uint_t rng_seed) const;

  void run_circuit_many_worlds_aux(ManyWorldNode &curr_world, const cvector_t &initial_state,
  				const Circuit &circ,
					std::list<Operations::Op> ops, // Call-by-value as it's going to change during the recursion
					const Noise::NoiseModel& noise,
          const json_t &config,
					ExperimentData &data,
					RngEngine &rng
          ) const;

  ManyWorldNode* initialize_world(const Circuit &circ,
			const cvector_t &initial_state,
      const Noise::NoiseModel& noise,
      const json_t &config) const;

  //-----------------------------------------------------------------------
  // Custom initial state
  //-----------------------------------------------------------------------        
  cvector_t initial_state_;
};

//=========================================================================
// Implementations
//=========================================================================

StatevectorController::StatevectorController() : Base::Controller() {
	PRINT_TRACE
  // Disable qubit truncation by default
  Base::Controller::truncate_qubits_ = false;
}

//-------------------------------------------------------------------------
// Config
//-------------------------------------------------------------------------

void StatevectorController::set_config(const json_t &config) {
	PRINT_TRACE
  // Set base controller config
  Base::Controller::set_config(config);

  //Add custom initial state
  if (JSON::get_value(initial_state_, "initial_statevector", config)) {
    // Check initial state is normalized
    if (!Utils::is_unit_vector(initial_state_, validation_threshold_))
      throw std::runtime_error("StatevectorController: initial_statevector is not a unit vector");
  }
}

void StatevectorController::clear_config() {
	PRINT_TRACE
  Base::Controller::clear_config();
  initial_state_ = cvector_t();
}

size_t StatevectorController::required_memory_mb(const Circuit& circ,
                                                 const Noise::NoiseModel& noise) const {
	PRINT_TRACE
  Statevector::State<> state;
  return state.required_memory_mb(circ.num_qubits, circ.ops);
}

//-------------------------------------------------------------------------
// Run circuit
//-------------------------------------------------------------------------

ExperimentData StatevectorController::run_circuit(const Circuit &circ,
                                              const Noise::NoiseModel& noise,
                                              const json_t &config,
                                              uint_t shots,
                                              uint_t rng_seed) const {
	PRINT_TRACE
	std::cout <<" " << std::endl;

  // Initialize  state
  Statevector::State<> state;

  // Validate circuit and throw exception if invalid operations exist
  validate_state(state, circ, noise, true);

  // Check for custom initial state, and if so check it matches num qubits
  if (!initial_state_.empty()) {
    if (initial_state_.size() != 1ULL << circ.num_qubits) {
      uint_t num_qubits(std::log2(initial_state_.size()));
      std::stringstream msg;
      msg << "StatevectorController: " << num_qubits << "-qubit initial state ";
      msg << "cannot be used for a " << circ.num_qubits << "-qubit circuit.";
      throw std::runtime_error(msg.str());
    }
  }

  // Set config
  state.set_config(config);
  state.set_parallalization(parallel_state_update_);
  
  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  ExperimentData data;
  data.set_config(config);
  
  // Run single shot collecting measure data or snapshots
  if (initial_state_.empty())
    state.initialize_qreg(circ.num_qubits);
  else
    state.initialize_qreg(circ.num_qubits, initial_state_);
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(circ.ops, data, rng);
  state.add_creg_to_data(data);
  
  // Add final state to the data
  data.add_additional_data("statevector", state.qreg().vector());

  return data;
}

StatevectorController::ManyWorldNode*
StatevectorController::initialize_world(const Circuit &circ,
						const cvector_t &initial_state,
		        const Noise::NoiseModel& noise,
		        const json_t &config) const {
	ManyWorldNode *node = new ManyWorldNode;

	// Initializing the current world state
	node->statevec_.set_config(config);
	node->statevec_.set_parallalization(parallel_state_update_);

  // Run single shot collecting measure data or snapshots
  if (initial_state.empty())
  	node->statevec_.initialize_qreg(circ.num_qubits);
  else
  	node->statevec_.initialize_qreg(circ.num_qubits, initial_state);

  node->statevec_.initialize_creg(circ.num_memory, circ.num_registers);

  return node;
}


ExperimentData StatevectorController::run_circuit_many_worlds(const Circuit &circ,
        const Noise::NoiseModel& noise,
        const json_t &config,
        uint_t shots,
        uint_t rng_seed) const {


  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  ExperimentData data;
  data.set_config(config);

  ManyWorldNode *root_world = initialize_world(circ, initial_state_, noise, config);


  std::list<Operations::Op> root_world_ops;
  for (auto const &op : circ.ops )
  	root_world_ops.push_back( op );

  run_circuit_many_worlds_aux(*root_world, initial_state_, circ, root_world_ops, noise, config, data, rng);

 // state.add_creg_to_data(data); // TODO: uncomment & what to do with this ?

  // Add final state to the data
  // data.add_additional_data("statevector", state.qreg().vector()); // TODO: uncomment & what to do with this ?

  delete root_world;

  return data;
}

void StatevectorController::run_circuit_many_worlds_aux(StatevectorController::ManyWorldNode &curr_world,
				const cvector_t &initial_state, const Circuit &circ,
				std::list<Operations::Op> ops,
        const Noise::NoiseModel& noise,
        const json_t &config,
				ExperimentData &data,
				RngEngine &rng) const {

  // Simulating all the ops up to the first measure op
  std::vector<Operations::Op> curr_world_ops;
  while ( not ops.empty() and (ops.begin()->type != Operations::OpType::measure) ) {
  	curr_world_ops.push_back( ops.front() );
  	ops.pop_front();
  }

  curr_world.statevec_.apply_ops(curr_world_ops, data, rng);

  // Handling the measure op
    // TODO: complete


  // Split into 2 worlds and continue
//  ManyWorldNode *_0_world = initialize_world(circ, , noise, config);
//  ManyWorldNode *_1_world = initialize_world(circ, , noise, config);


}



//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
