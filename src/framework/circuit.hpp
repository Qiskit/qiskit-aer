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
  bool remapped_qubits = false;      // True if qubits have been remapped

  // Constructor
  // The constructor automatically calculates the num_qubits, num_memory, num_registers
  // parameters by scanning the input list of ops.
  Circuit() {set_random_seed();}
  Circuit(const std::vector<Op> &_ops, bool truncation = false);
  Circuit(std::vector<Op> &&_ops, bool truncation = false);

  // Construct a circuit from JSON
  template<typename inputdata_t>
  Circuit(const inputdata_t& circ, bool truncation = false);

  template<typename inputdata_t>
  Circuit(const inputdata_t& circ, const json_t& qobj_config, bool truncation = false);

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

  // Return the mapping of input op qubits to circuit qubits
  inline const auto& qubit_map() const {return qubitmap_;}

  //-----------------------------------------------------------------------
  // Utility methods 
  //-----------------------------------------------------------------------

  // Automatically set the number of qubits, memory, registers, and check
  // for conditionals and measure sampling based on ops.
  // If `truncation = true` also perform truncation of
  // unnecessary qubits, remapping of remaining qubits, checking
  // of measure sampling optimization, and delay of measurements
  // to end of circuit
  void set_params(bool truncation = false);

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

  // Add type, qubit, memory, conditional metadata information from op
  void add_op_metadata(const Op& op);

  // Reset circuit metadata
  void reset_metadata();

  // Helper function for optimized set params
  bool check_result_ancestor(const Op& op,
                             std::unordered_set<uint_t>& ancestor_qubits) const;
  
  // Helper function for optimized set params
  void remap_qubits(Op& op) const;
};


// Json conversion function
inline void from_json(const json_t& js, Circuit &circ) {circ = Circuit(js);}


//============================================================================
// Implementation: Circuit methods
//============================================================================

Circuit::Circuit(const std::vector<Op> &_ops, bool truncation) : Circuit() {
  ops = _ops;
  set_params(truncation);
}

Circuit::Circuit(std::vector<Op> &&_ops, bool truncation) : Circuit() {
  ops = std::move(_ops);
  set_params(truncation);
}

template<typename inputdata_t>
Circuit::Circuit(const inputdata_t &circ, bool truncation) : Circuit(circ, json_t(), truncation) {}

template<typename inputdata_t>
Circuit::Circuit(const inputdata_t &circ, const json_t &qobj_config, bool truncation) : Circuit() {
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
  ops = std::move(converted_ops);
  set_params(truncation);

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
    if (!truncation) {
      // Override minimal circuit qubit number with qobj number if truncation
      // is explicitly disabled.
      num_qubits = n_qubits;
    }
  }
}

//-------------------------------------------------------------------------
// Circuit initialization optimization
//-------------------------------------------------------------------------

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


void Circuit::set_params(bool truncation) {
  // Clear current circuit metadata  
  reset_metadata();
  if (ops.empty()) return;

  // Analyze input ops from tail to head to get locations of ancestor,
  // first measurement position and last initialize position
  const auto size = ops.size();
  std::vector<bool> ancestor(size, false);
  first_measure_pos = size;
  bool has_measure = false;
  size_t num_ancestors = 0;
  size_t last_ancestor_pos = 0;
  size_t last_initialize_pos = 0;
  bool ops_to_remove = false;

  std::unordered_set<uint_t> ancestor_qubits;
  for (size_t i = 0; i < size; ++ i) {
    const size_t rpos = size - i - 1;
    const auto& op = ops[rpos];
    if (!truncation || check_result_ancestor(op, ancestor_qubits)) {
      add_op_metadata(op);
      ancestor[rpos] = true;
      num_ancestors++;
      if (op.type == OpType::measure) {
        first_measure_pos = rpos;
        has_measure = true;
      } else if (op.type == OpType::initialize && last_initialize_pos == 0) {
        last_initialize_pos = rpos;
      }
      if (last_ancestor_pos == 0) {
        last_ancestor_pos = rpos;
      }
    } else if (truncation && !ops_to_remove){
      ops_to_remove = true;
    }
  }

  // Set qubit size and check for truncaiton
  remapped_qubits = false;
  if (truncation) {
    // Generate mapping of original qubits to ancestor set
    uint_t idx = 0;
    for (const auto& qubit: qubitset_) {
      if (!remapped_qubits && idx != qubit) {
        // qubits will be remapped
        remapped_qubits = true;
      }
      qubitmap_[qubit] = idx;
      idx++;
    }
  }

  // Set qubit and memory size
  num_memory = (memoryset_.empty()) ? 0 : 1 + *memoryset_.rbegin();
  num_registers = (registerset_.empty()) ? 0 : 1 + *registerset_.rbegin();
  if (remapped_qubits) {
    num_qubits = qubitset_.size();
  } else {
    num_qubits = (qubitset_.empty()) ? 0 : 1 + *qubitset_.rbegin();
  }

  // Check if can sample initialize
  if (last_initialize_pos > 0 &&
      ops[last_initialize_pos].qubits.size() < num_qubits) {
    can_sample_initialize = false;
    can_sample = false;
  }

  // Check measurement opt and split tail meas and non-meas ops
  std::vector<uint_t> tail_pos;
  std::vector<Op> tail_meas_ops;
  if (has_measure && can_sample) {
    std::unordered_set<uint_t> meas_qubits;
    std::unordered_set<uint_t> modified_qubits;

    for (uint_t pos = first_measure_pos; pos <= last_ancestor_pos; ++pos) {
      if (ops_to_remove && !ancestor[pos]) {
        // Skip if not ancestor
        continue;
      }

      const auto& op = ops[pos];
      if (op.conditional) {
        can_sample = false;
        break;
      }

      switch (op.type) {
        case OpType::measure:
        case OpType::roerror: {
          meas_qubits.insert(op.qubits.begin(), op.qubits.end());
          tail_meas_ops.push_back(op);
          break;  
        }
        case OpType::snapshot:
        case OpType::save_state:
        case OpType::save_expval:
        case OpType::save_expval_var:
        case OpType::save_statevec:
        case OpType::save_statevec_dict:
        case OpType::save_densmat:
        case OpType::save_probs:
        case OpType::save_probs_ket:
        case OpType::save_amps:
        case OpType::save_amps_sq:
        case OpType::save_stabilizer:
        case OpType::save_clifford:
        case OpType::save_unitary:
        case OpType::save_mps:
        case OpType::save_superop: 
        {
          can_sample = false;
          break;
        }
        default: {
          for (const auto &qubit : op.qubits) {
            if (meas_qubits.find(qubit) != meas_qubits.end()) {
              can_sample = false;
              break;
            }
          }
          tail_pos.push_back(pos);
        }
      }
      if (!can_sample) {
        break;
      }
    }
  }

  // Counter for current position in ops as we shuffle ops
  size_t op_idx = 0;
  size_t head_end = 0;
  std::set<std::string> marks;
  std::set<std::string> dests;
  if (has_measure && can_sample) {
    head_end = first_measure_pos;
  } else if (num_ancestors > 0) {
    head_end = last_ancestor_pos + 1;
  }
  for (size_t pos = 0; pos < head_end; ++pos) {
    if (ops_to_remove && !ancestor[pos]) {
      // Skip if not ancestor
      continue;
    }
    if (remapped_qubits) {
      remap_qubits(ops[pos]);
    }
    if (pos != op_idx) {
      ops[op_idx] = std::move(ops[pos]);
    }
    if (ops[op_idx].type == OpType::jump) {
      dests.insert(ops[op_idx].string_params[0]);
    } else if (ops[op_idx].type == OpType::mark) {
      auto& mark_name = ops[op_idx].string_params[0];
      if (marks.find(mark_name) != marks.end()) {
        std::stringstream msg;
        msg << "Duplicated mark destination:\"" << mark_name << "\"." << std::endl;
        throw std::runtime_error(msg.str());
      }
      marks.insert(mark_name);
    }
    if (pos == first_measure_pos) {
      first_measure_pos = op_idx;
    }
    op_idx++;
  }

  for (auto dest : dests) {
    if (marks.find(dest) == marks.end()) {
      std::stringstream msg;
      msg << "Invalid jump destination:\"" << dest << "\"." << std::endl;
      throw std::runtime_error(msg.str());
    }
  }

  if (has_measure && can_sample) {
    // Apply remapping to tail ops
    for (size_t tidx = 0; tidx < tail_pos.size(); ++tidx) {
      const auto tpos = tail_pos[tidx];
      if (!ops_to_remove && !ancestor[tpos]) {
        continue;
      }
      auto& op = ops[tpos];
      if (remapped_qubits) {
        remap_qubits(ops[tpos]);
      }
      if (tpos != op_idx) {
        ops[op_idx] = std::move(op);
      }
      op_idx++;
    }
    // Now add remaining delayed measure ops
    first_measure_pos = op_idx;
    for (auto & op : tail_meas_ops) {
      if (remapped_qubits) {
        remap_qubits(op);
      }
      ops[op_idx] = std::move(op);
      op_idx++;
    }
  }
  // Handle edge case of truncation with no measurements
  first_measure_pos = std::min(op_idx, first_measure_pos);

  // Resize to remove discarded ops
  ops.resize(op_idx);
}


void Circuit::remap_qubits(Op& op) const {
  reg_t new_qubits;
  for (auto& qubit : op.qubits) {
    new_qubits.push_back(qubitmap_.at(qubit));
  }
  op.qubits = std::move(new_qubits);
}


bool Circuit::check_result_ancestor(const Op& op, std::unordered_set<uint_t>& ancestor_qubits) const {
  switch (op.type) {
    case OpType::barrier:
    case OpType::nop: {
      return false;
    }
    case OpType::bfunc: {
      return true;
    }
    // Result generating types
    case OpType::measure:
    case OpType::roerror:
    case OpType::snapshot:
    case OpType::save_state:
    case OpType::save_expval:
    case OpType::save_expval_var:
    case OpType::save_statevec:
    case OpType::save_statevec_dict:
    case OpType::save_densmat:
    case OpType::save_probs:
    case OpType::save_probs_ket:
    case OpType::save_amps:
    case OpType::save_amps_sq:
    case OpType::save_stabilizer:
    case OpType::save_clifford:
    case OpType::save_unitary:
    case OpType::save_mps:
    case OpType::save_superop: {
      ancestor_qubits.insert(op.qubits.begin(), op.qubits.end());
      return true;
    }
    default: {
      for (const auto& qubit : op.qubits) {
        if (ancestor_qubits.find(qubit) != ancestor_qubits.end()) {
          ancestor_qubits.insert(op.qubits.begin(), op.qubits.end());
          return true;
        }
      }
      return false;
    }
  }
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
