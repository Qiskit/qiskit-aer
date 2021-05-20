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

#ifndef _aer_framework_circuit_hpp_
#define _aer_framework_circuit_hpp_

#include <random>

#include "framework/operations.hpp"
#include "framework/opset.hpp"

namespace AER {

//============================================================================
// Circuit class for Qiskit-Aer
//============================================================================

// A circuit is a list of Ops along with a specification of maximum needed
// qubits, memory bits, and register bits for the input operators.
class Circuit {
public:
  using Op = Operations::Op;
  using OpType = Operations::OpType;

  // Circuit operations
  std::vector<Op> ops;

  // Circuit parameters updated by from ops by set_params
  uint_t num_qubits = 0;        // maximum number of qubits needed for ops
  uint_t num_memory = 0;        // maximum number of memory clbits needed for ops
  uint_t num_registers = 0;     // maximum number of registers clbits needed for ops
  
  // Measurement params
  bool has_conditional = false;      // True if any ops are conditional
  bool can_sample = true;            // True if circuit tail contains measure, roerror, barrier.
  size_t first_measure_pos = 0;      // Position of first measure instruction
  bool can_sample_initialize = true; // True if circuit contains at most 1 initialize
                                     // and it is the first instruction in the circuit

  // Circuit metadata constructed from json QobjExperiment
  uint_t shots = 1;
  uint_t seed;
  json_t header;
  double global_phase_angle = 0;

  // Constructor
  // The constructor automatically calculates the num_qubits, num_memory, num_registers
  // parameters by scanning the input list of ops.
  Circuit() {set_random_seed();}
  Circuit(const std::vector<Op> &_ops);

  // Construct a circuit from JSON
  template<typename inputdata_t>
  Circuit(const inputdata_t& circ);

  template<typename inputdata_t>
  Circuit(const inputdata_t& circ, const json_t& qobj_config);

  //-----------------------------------------------------------------------
  // Set containers
  //-----------------------------------------------------------------------

  // Return the opset for the circuit
  inline const Operations::OpSet& opset() const {return opset_;}

  // Return the used qubits for the circuit
  inline const std::set<uint_t>& qubits() const {return qubitset_;}

  // Return the used memory for the circuit
  inline const std::set<uint_t>& memory() const {return memoryset_;}

  // Return the used registers for the circuit
  inline const std::set<uint_t>& registers() const {return registerset_;}

  //-----------------------------------------------------------------------
  // Utility methods 
  //-----------------------------------------------------------------------
  
  // Automatically set the number of qubits, memory, registers, and check
  // for conditionals based on ops
  void set_params();

  // Set the circuit rng seed to random value
  inline void set_random_seed() {seed = std::random_device()();}

private:
  Operations::OpSet opset_;       // Set of operation types contained in circuit
  std::set<uint_t> qubitset_;     // Set of qubits used in the circuit
  std::set<uint_t> memoryset_;    // Set of memory bits used in the circuit
  std::set<uint_t> registerset_;  // Set of register bits used in the circuit
  std::set<std::string> saveset_; // Set of key names for save data instructions
};

// Json conversion function
inline void from_json(const json_t& js, Circuit &circ) {circ = Circuit(js);}


//============================================================================
// Implementation: Circuit methods
//============================================================================

void Circuit::set_params() {

  // Clear current containers
  opset_ = Operations::OpSet();
  qubitset_.clear();
  memoryset_.clear();
  registerset_.clear();
  saveset_.clear();
  can_sample = true;
  first_measure_pos = 0;

  // Check maximum qubit, and register size
  // Memory size is loaded from qobj config
  // Also check if measure sampling is in principle possible
  bool first_measure = true;
  size_t initialize_qubits = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    const auto &op = ops[i];
    has_conditional |= op.conditional;
    opset_.insert(op);
    qubitset_.insert(op.qubits.begin(), op.qubits.end());
    memoryset_.insert(op.memory.begin(), op.memory.end());
    registerset_.insert(op.registers.begin(), op.registers.end());

    // Check for duplicate save keys
    if (Operations::SAVE_TYPES.find(op.type) != Operations::SAVE_TYPES.end()) {
      auto pair = saveset_.insert(op.string_params[0]);
      if (!pair.second) {
        throw std::invalid_argument("Duplicate key \"" + op.string_params[0] +
                                    "\" in save instruction.");
      }
    }

    // Keep track of minimum width of any non-initial initialize instructions
    if (i > 0 && op.type == OpType::initialize) {
      initialize_qubits = (initialize_qubits == 0)
        ? op.qubits.size()
        : std::min(op.qubits.size(), initialize_qubits);
    }

    // Compute measure sampling check
    if (can_sample) {
      if (first_measure) {
        if (op.type == OpType::measure || op.type == OpType::roerror) {
          first_measure = false;
        } else {
          first_measure_pos++;
        }
      } else if(!(op.type == OpType::barrier ||
                  op.type == OpType::measure ||
                  op.type == OpType::roerror)) {
        can_sample = false;
      }
    }
  }

  // Get required number of qubits, memory, registers from set maximums
  // Since these are std::set containers the largest element is the
  // Last element of the (non-empty) container. 
  num_qubits = (qubitset_.empty()) ? 0 : 1 + *qubitset_.rbegin();
  num_memory = (memoryset_.empty()) ? 0 : 1 + *memoryset_.rbegin();
  num_registers = (registerset_.empty()) ? 0 : 1 + *registerset_.rbegin();

  // Check that any non-initial initialize is defined on full width of qubits
  if (initialize_qubits > 0 && initialize_qubits < num_qubits) {
    can_sample_initialize = false;
  }
}

Circuit::Circuit(const std::vector<Op> &_ops) : Circuit() {
  ops = _ops;
  set_params();
}

template<typename inputdata_t>
Circuit::Circuit(const inputdata_t &circ) : Circuit(circ, json_t()) {}

template<typename inputdata_t>
Circuit::Circuit(const inputdata_t &circ, const json_t &qobj_config) : Circuit() {

  // Get config
  json_t config = qobj_config;
  if (Parser<inputdata_t>::check_key("config", circ)) {
    json_t circ_config;
    Parser<inputdata_t>::get_value(circ_config, "config", circ);
    for (auto it = circ_config.cbegin(); it != circ_config.cend(); ++it) {
      config[it.key()] = it.value(); // overwrite circuit level config values
    }
  }
  // Load instructions
  if (Parser<inputdata_t>::check_key("instructions", circ) == false) {
    throw std::invalid_argument("Invalid Qobj experiment: no \"instructions\" field.");
  }
  ops.clear(); // remove any current operations
  const inputdata_t &input_ops = Parser<inputdata_t>::get_list("instructions", circ);
  for(auto the_op: input_ops){
    ops.emplace_back(Operations::input_to_op(the_op));
  }

  // Set circuit parameters from ops
  set_params();

  // Load metadata
  Parser<inputdata_t>::get_value(header, "header", circ);
  Parser<json_t>::get_value(shots, "shots", config);
  Parser<json_t>::get_value(global_phase_angle, "global_phase", header);

  // Check for specified memory slots
  uint_t memory_slots = 0;
  Parser<json_t>::get_value(memory_slots, "memory_slots", config);
  if (memory_slots < num_memory) {
    throw std::invalid_argument("Invalid Qobj experiment: not enough memory slots.");
  }
  // override memory slot number
  num_memory = memory_slots;

  // Check for specified n_qubits
  if (Parser<json_t>::check_key("n_qubits", config)) {
    // uint_t n_qubits = config["n_qubits"];
    uint_t n_qubits;
    Parser<json_t>::get_value(n_qubits, "n_qubits", config);
    if (n_qubits < num_qubits) {
      throw std::invalid_argument("Invalid Qobj experiment: n_qubits < instruction qubits.");
    }
    // override qubit number
    num_qubits = n_qubits;
  }
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
