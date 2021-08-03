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
  Circuit(const std::vector<Op> &_ops, bool optimize = false);
  Circuit(std::vector<Op> &&_ops, bool optimize = false);

  // Construct a circuit from JSON
  template<typename inputdata_t>
  Circuit(const inputdata_t& circ, bool optimize = false);

  template<typename inputdata_t>
  Circuit(const inputdata_t& circ, const json_t& qobj_config, bool optimize = false);

  //-----------------------------------------------------------------------
  // Set containers
  //-----------------------------------------------------------------------

  // Return the opset for the circuit
  inline const auto& opset() const {return opset_;}

  // Return the used qubits for the circuit
  inline const auto& qubits() const {return qubitset_;}

  // Return the used memory for the circuit
  inline const auto& memory() const {return memoryset_;}

  // Return the used registers for the circuit
  inline const auto& registers() const {return registerset_;}

  // Return the used registers for the circuit
  inline const auto& qubit_map() const {return qubitmap_;}

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

  // Mapping from loaded op qubits to remapped truncated circuit qubits
  std::unordered_map<uint_t, uint_t> qubitmap_;

  // True if qubitmap is trivial so circuit qubits are the same
  // as original qubits
  bool trivial_map_ = true;

  // Add type, qubit, memory, conditional metadata information from op
  void add_op_metadata(const Op& op);

  // Reset circuit metadata
  void reset_metadata();

  // Set for op types that generate result data from simulations
  const static std::unordered_set<OpType> result_ops_;

  // Optimized initialization method that performs truncation of
  // unnecessary qubits, remapping of remaining qubits, checking
  // of measure sampling optimization, and delay of measurements
  // to end of circuit 
  void optimized_initialize(std::vector<Op>&& instructions);
  void optimized_initialize(const std::vector<Op>& instructions);

  // Helper functions for optimized initialization

  // Reverse iterate over set of input ops to build set of measure
  // and save instruction predecessors to determine which qubits
  // are necessary to simulate
  std::vector<Op> get_reversed_ops(std::vector<Op>&& instructions);
  std::vector<Op> get_reversed_ops(const std::vector<Op>& instructions);

  // Reverse iterate over reversed ops to build the remapped circuit
  // of ops required to produce the expected output
  std::vector<Op> get_forward_ops(std::vector<Op>&& reversed_ops);

  // Helper function for get_reversed_ops
  bool check_result_ancestor(const Op& op,
                             std::set<uint_t>& measured_qubits,
                             std::set<uint_t>& ancestor_qubits,
                             std::set<uint_t>& modified_qubits,
                             bool& measure_opt) const;

  
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

Circuit::Circuit(const std::vector<Op> &_ops, bool optimize) : Circuit() {
  if (optimize) {
    optimized_initialize(_ops);
  } else {
    ops = _ops;
    set_params();
  }
}

Circuit::Circuit(std::vector<Op> &&_ops, bool optimize) : Circuit() {
  if (optimize) {
    optimized_initialize(std::move(_ops));
  } else {
    ops = std::move(_ops);
    set_params();
  }
}

template<typename inputdata_t>
Circuit::Circuit(const inputdata_t &circ, bool optimize) : Circuit(circ, json_t(), optimize) {}

template<typename inputdata_t>
Circuit::Circuit(const inputdata_t &circ, const json_t &qobj_config, bool optimize) : Circuit() {
  // Get config
  json_t config = qobj_config;
  if (Parser<inputdata_t>::check_key("config", circ)) {
    json_t circ_config;
    Parser<inputdata_t>::get_value(circ_config, "config", circ);
    for (auto it = circ_config.cbegin(); it != circ_config.cend(); ++it) {
      config[it.key()] = it.value(); // overwrite circuit level config values
    }
  }
  
  // Load metadata
  Parser<inputdata_t>::get_value(header, "header", circ);
  Parser<json_t>::get_value(shots, "shots", config);
  Parser<json_t>::get_value(global_phase_angle, "global_phase", header);

  // Load instructions
  if (Parser<inputdata_t>::check_key("instructions", circ) == false) {
    throw std::invalid_argument("Invalid Qobj experiment: no \"instructions\" field.");
  }
  const auto input_ops = Parser<inputdata_t>::get_list("instructions", circ);
  
  // Convert to Ops
  // TODO: If parser could support reverse iteration through the list of ops without
  // conversion we could call `get_reversed_ops` on the inputdata without first
  // converting. 
  std::vector<Op> converted_ops;
  for(auto the_op: input_ops){
    converted_ops.emplace_back(Operations::input_to_op(the_op));
  }

  // Optimized initialization
  if (optimize) {
    optimized_initialize(converted_ops);
    return;
  }

  // Old non-optimized initialization
  ops = std::move(converted_ops);

  // Set circuit parameters from ops
  set_params();

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

//-------------------------------------------------------------------------
// Circuit initialization optimization
//-------------------------------------------------------------------------

const std::unordered_set<Operations::OpType> Circuit::result_ops_ = {
  Operations::OpType::measure, Operations::OpType::roerror, Operations::OpType::snapshot,
  Operations::OpType::save_state, Operations::OpType::save_expval, Operations::OpType::save_expval_var,
  Operations::OpType::save_statevec, Operations::OpType::save_statevec_dict,
  Operations::OpType::save_densmat, Operations::OpType::save_probs, Operations::OpType::save_probs_ket,
  Operations::OpType::save_amps, Operations::OpType::save_amps_sq, Operations::OpType::save_stabilizer,
  Operations::OpType::save_unitary, Operations::OpType::save_mps, Operations::OpType::save_superop,
};


void Circuit::reset_metadata() {

  opset_ = Operations::OpSet();
  qubitset_.clear();
  memoryset_.clear();
  registerset_.clear();
  saveset_.clear();
  qubitmap_.clear();

  num_qubits = 0;
  num_memory = 0;
  num_registers = 0;
  
  has_conditional = false;
  can_sample = true;
  first_measure_pos = 0;
  can_sample_initialize = true;
}

void Circuit::add_op_metadata(const Op& op) {
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
}


void Circuit::optimized_initialize(const std::vector<Op>& instructions) {
  reset_metadata();
  ops = get_forward_ops(get_reversed_ops(instructions));
}

void Circuit::optimized_initialize(std::vector<Op>&& instructions) {
  reset_metadata();
  ops = get_forward_ops(get_reversed_ops(std::move(instructions)));
}


std::vector<Operations::Op> Circuit::get_forward_ops(std::vector<Op> && reversed_ops) {

  // Generate mapping of original qubits to ancestor set
  uint_t idx = 0;
  std::unordered_map<uint_t, uint_t> qubit_mapping;
  std::unordered_map<uint_t, uint_t> inverse_qubit_mapping;
  for (const auto& qubit: qubitset_) {
    if (trivial_map_ && idx != qubit) {
      trivial_map_ = false;
    }
    qubit_mapping[qubit] = idx;
    inverse_qubit_mapping[idx] = qubit;
    idx++;
  }

  // Set circuit size parameters
  num_qubits = qubitset_.size();
  num_memory = memoryset_.size();
  num_registers = registerset_.size();

  // Construct remapped circuit
  std::vector<Op> circuit_ops;
  std::vector<Op> measurement_ops;

  // Keep track of any initialize instructions that aren't first instruction
  size_t pos = 0;
  size_t min_initialize_qubits = 0;
  bool has_results = false;

  while (!reversed_ops.empty()) {
    auto& op = reversed_ops.back();

    // Check if circuit has generated any result data
    if (!has_results && (result_ops_.find(op.type) != result_ops_.end())) {
      has_results = true;
    }

    // Remap op qubits
    if (!trivial_map_) {
      reg_t new_qubits;
      for (auto& qubit : op.qubits) {
        new_qubits.push_back(qubit_mapping[qubit]);
      }
      op.qubits = new_qubits;
    }

    // Check for initialize optimizations
    if (op.type == OpType::initialize) {
      if (!has_results && (op.qubits.size() == num_qubits)) {
        // We are overriding any previous instructions that haven't generated
        // data so we can discard them
        ops.clear();
        min_initialize_qubits = 0;
        pos = 0;
      } else if (pos > 0) {
        min_initialize_qubits = (min_initialize_qubits == 0)
          ? op.qubits.size()
          : std::min(op.qubits.size(), min_initialize_qubits);
      }
    }

    // Move op to remapped circuit
    if (!has_conditional && can_sample && (
          op.type == OpType::measure || op.type == OpType::roerror)) {
      measurement_ops.push_back(std::move(op));
    } else {
      circuit_ops.push_back(std::move(op));
    }

    // Pop off empty tail and increase position counter
    reversed_ops.pop_back();
    pos++;
  }

  // Add measurement ops to tail
  first_measure_pos = circuit_ops.size();
  for (auto&& op : measurement_ops) {
    circuit_ops.push_back(std::move(op));
  }

  // Check if non-initial initialize instructions disable measure sampling
  // because they aren't defined on the full circuit width.
  if (min_initialize_qubits > 0 && min_initialize_qubits < num_qubits) {
    can_sample_initialize = false;
  }

  // Set qubit map to original circuit qubits
  qubitmap_ = std::move(inverse_qubit_mapping);

  return circuit_ops;
}

std::vector<Operations::Op> Circuit::get_reversed_ops(const std::vector<Operations::Op>& instructions) {
  std::vector<Operations::Op> inst_cpy = instructions;
  return get_reversed_ops(std::move(inst_cpy));
}

std::vector<Operations::Op> Circuit::get_reversed_ops(std::vector<Operations::Op>&& instructions) {
  std::vector<Operations::Op> reversed_ancestors;

  // Reverse iterate over instructions and collect ancestors
  // for any result producing instructions
  std::set<uint_t> measured_qubits;
  std::set<uint_t> ancestor_qubits;
  std::set<uint_t> modified_qubits;

  while (!instructions.empty()) {
    auto& op = instructions.back();
    if (check_result_ancestor(op, measured_qubits, ancestor_qubits, modified_qubits, can_sample)) {
      add_op_metadata(op);
      reversed_ancestors.push_back(std::move(op));
    }
    instructions.pop_back();
  }
  return reversed_ancestors;
}


bool Circuit::check_result_ancestor(const Op& op,
                                    std::set<uint_t>& measured_qubits,
                                    std::set<uint_t>& ancestor_qubits,
                                    std::set<uint_t>& modified_qubits,
                                    bool& measure_opt) const {
  if (op.type == OpType::barrier || op.type == OpType::nop) {
    return false;
  }
  if (op.type == OpType::bfunc) {
    measure_opt = false;
    return true;
  }

  // Check if op is a a result generating instrunction, or an ancestor
  // of result generating instructions
  bool ancestor = false;
  for (const auto& qubit: op.qubits) {
    if (result_ops_.find(op.type) != result_ops_.end()) {
      // Instruction generates result data
      measured_qubits.insert(qubit);
      ancestor_qubits.insert(qubit);
      if (measure_opt && op.type == OpType::measure
          && (modified_qubits.find(qubit) == modified_qubits.end())) {
        // Instruction is an intermediate measurement that can't be
        // pushed to the tail of a circuit and hence disables
        // measure sampling optimization
        measure_opt = false;
      }
    } else if (measure_opt && measured_qubits.find(qubit) != measured_qubits.end()) {
      if (op.conditional) {
        // Measure opt is also prevented if the op is conditional
        measure_opt = false;
      } else {
        // Instruction modifies a qubit that will be measured so could
        // potentially prevent measure sampling if there is an intermediate
        // measurement on this qubit
        modified_qubits.insert(qubit);
      }
    }

    if (!ancestor && (ancestor_qubits.find(qubit) != ancestor_qubits.end())) {
      // If qubit is in ancestor qubits all op qubits are ancestors
      ancestor = true;
    }
  }
  
  // If instruction isn't an ancestor we are done
  if (!ancestor) {
    return false;
  }

  // If instruction is ancestor add all qubits to ancestor qubit set
  for (const auto& qubit: op.qubits) {
    ancestor_qubits.insert(qubit);
  }
  return true;
}


//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
