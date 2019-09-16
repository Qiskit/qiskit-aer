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

#ifndef _aer_stabilizer_state_hpp
#define _aer_stabilizer_state_hpp

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "base/state.hpp"
#include "clifford.hpp"

namespace AER {
namespace Stabilizer {

//============================================================================
// Stabilizer state gates
//============================================================================

enum class Gates {id, x, y, z, h, s, sdg, cx, cz, swap};

// Allowed snapshots enum class
enum class Snapshots {
  stabilizer, cmemory, cregister,
  probs, probs_var
  /* TODO: the following snapshots still need to be implemented */
  //expval_pauli, expval_pauli_var, //  TODO
};

//============================================================================
// Stabilizer Table state class
//============================================================================

class State : public Base::State<Clifford::Clifford> {

public:
  using BaseState = Base::State<Clifford::Clifford>;

  State() = default;
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {return "stabilizer";}

  // Return the set of qobj instruction types supported by the State
  virtual Operations::OpSet::optypeset_t allowed_ops() const override {
    return Operations::OpSet::optypeset_t({
      Operations::OpType::gate,
      Operations::OpType::measure,
      Operations::OpType::reset,
      Operations::OpType::snapshot,
      Operations::OpType::barrier,
      Operations::OpType::bfunc,
      Operations::OpType::roerror
    });
  }

  // Return the set of qobj gate instruction names supported by the State
  virtual stringset_t allowed_gates() const override {
    return {"CX", "cx", "cz", "swap", "id", "x", "y", "z", "h", "s", "sdg"};
  }

  // Return the set of qobj snapshot types supported by the State
  virtual stringset_t allowed_snapshots() const override {
    return {"stabilizer", "memory", "register"};
  }

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exeption will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentData &data,
                         RngEngine &rng) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit state
  virtual void initialize_qreg(uint_t num_qubits,
                               const Clifford::Clifford &state) override;

  // TODO: currently returns 0
  // Returns the required memory for storing an n-qubit state in megabytes.
  virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops)
                                    const override;

  // Load any settings for the State class from a config JSON
  virtual void set_config(const json_t &config) override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t& qubits,
                                            uint_t shots,
                                            RngEngine &rng) override;

protected:

  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(const Operations::Op &op);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual void apply_measure(const reg_t &qubits,
                             const reg_t &cmemory,
                             const reg_t &cregister,
                             RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits, RngEngine &rng);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const Operations::Op &op, ExperimentData &data);

  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Implement a measurement on all specified qubits and return the outcome
  reg_t apply_measure_and_update(const reg_t &qubits, RngEngine &rng);

  //-----------------------------------------------------------------------
  // Special snapshot types
  //
  // IMPORTANT: These methods are not marked const to allow modifying state
  // during snapshot, but after the snapshot is applied the simulator
  // should be left in the pre-snapshot state.
  //-----------------------------------------------------------------------

  // Snapshot the stabilizer state of the simulator.
  // This returns a list of stabilizer generators
  void snapshot_stabilizer(const Operations::Op &op, ExperimentData &data);
                            
  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(const Operations::Op &op,
                              ExperimentData &data,
                              bool variance);

  /* TODO
  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const Operations::Op &op,
                             ExperimentData &data,
                             bool variance);
  */

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Set maximum number of qubits for which snapshot
  // probabilities can be implemented
  size_t max_qubits_snapshot_probs_ = 32;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;

  // Table of allowed snapshot types to enum class members
  const static stringmap_t<Snapshots> snapshotset_;

};


//============================================================================
// Implementation: Allowed ops and gateset
//============================================================================

const stringmap_t<Gates> State::gateset_({
  // Single qubit gates
  {"id", Gates::id},   // Pauli-Identity gate
  {"x", Gates::x},    // Pauli-X gate
  {"y", Gates::y},    // Pauli-Y gate
  {"z", Gates::z},    // Pauli-Z gate
  {"s", Gates::s},    // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg}, // Conjugate-transpose of Phase gate
  {"h", Gates::h},    // Hadamard gate (X + Z / sqrt(2))
  // Two-qubit gates
  {"CX", Gates::cx},  // Controlled-X gate (CNOT)
  {"cx", Gates::cx},  // Controlled-X gate (CNOT),
  {"cz", Gates::cz},   // Controlled-Z gate
  {"swap", Gates::swap} // SWAP gate
});

const stringmap_t<Snapshots> State::snapshotset_({
  {"stabilizer", Snapshots::stabilizer},
  {"memory", Snapshots::cmemory},
  {"register", Snapshots::cregister},
  {"probabilities", Snapshots::probs},
  {"probabilities_with_variance", Snapshots::probs_var}
  //{"expectation_value_pauli", Snapshots::expval_pauli}, // TODO
  //{"expectation_value_pauli_with_variance", Snapshots::expval_pauli_var} // TODO
});


//============================================================================
// Implementation: Base class method overrides
//============================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------

void State::initialize_qreg(uint_t num_qubits) {
  BaseState::qreg_ = Clifford::Clifford(num_qubits);
}

void State::initialize_qreg(uint_t num_qubits,
                            const Clifford::Clifford &state) {
  // Check dimension of state
  if (state.num_qubits() != num_qubits) {
    throw std::invalid_argument("Stabilizer::State::initialize: initial state does not match qubit number");
  }
  BaseState::qreg_ = state;
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

size_t State::required_memory_mb(uint_t num_qubits,
                                 const std::vector<Operations::Op> &ops)
                                 const  {
  (void)ops; // avoid unused variable compiler warning
  // The Clifford object requires very little memory.
  // A Pauli vector consists of 2 binary vectors each with
  // Binary vector = (4 + n // 64) 64-bit ints
  // Pauli = 2 * binary vector
  size_t mem = 16 * (4 + num_qubits); // Pauli bytes
  // Clifford = 2n * Pauli + 2n phase ints
  mem = 2 * num_qubits * (mem + 16); // Clifford bytes
  mem = mem >> 20; // Clifford mb
  return mem;
}

void State::set_config(const json_t &config) {
  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "zero_threshold", config);

  // Load max snapshot qubit size and set hard limit of 64 qubits.
  JSON::get_value(max_qubits_snapshot_probs_, "stabilizer_max_snapshot_probabilities", config);
  max_qubits_snapshot_probs_ = std::max<uint_t>(max_qubits_snapshot_probs_, 64);
}

//=========================================================================
// Implementation: apply operations
//=========================================================================

void State::apply_ops(const std::vector<Operations::Op> &ops,
                      ExperimentData &data,
                      RngEngine &rng) {
  // Simple loop over vector of input operations
  for (const auto op: ops) {
    if(BaseState::creg_.check_conditional(op)) {
      switch (op.type) {
        case Operations::OpType::barrier:
          break;
        case Operations::OpType::reset:
          apply_reset(op.qubits, rng);
          break;
        case Operations::OpType::measure:
          apply_measure(op.qubits, op.memory, op.registers, rng);
          break;
        case Operations::OpType::bfunc:
          BaseState::creg_.apply_bfunc(op);
          break;
        case Operations::OpType::roerror:
          BaseState::creg_.apply_roerror(op, rng);
          break;
        case Operations::OpType::gate:
          apply_gate(op);
          break;
        case Operations::OpType::snapshot:
          apply_snapshot(op, data);
          break;
        default:
          throw std::invalid_argument("Stabilizer::State::invalid instruction \'" +
                                      op.name + "\'.");
      }
    }
  }
}

void State::apply_gate(const Operations::Op &op) {
  // Check Op is supported by State
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument("Stabilizer::State::invalid gate instruction \'" +
                                op.name + "\'.");
  switch (it -> second) {
    case Gates::id:
      break;
    case Gates::x:
      BaseState::qreg_.append_x(op.qubits[0]);
      break;
    case Gates::y:
      BaseState::qreg_.append_y(op.qubits[0]);
      break;
    case Gates::z:
      BaseState::qreg_.append_z(op.qubits[0]);
      break;
    case Gates::h:
      BaseState::qreg_.append_h(op.qubits[0]);
      break;
    case Gates::s:
      BaseState::qreg_.append_s(op.qubits[0]);
      break;
    case Gates::sdg:
      BaseState::qreg_.append_z(op.qubits[0]);
      BaseState::qreg_.append_s(op.qubits[0]);
      break;
    case Gates::cx:
      BaseState::qreg_.append_cx(op.qubits[0], op.qubits[1]);
      break;
    case Gates::cz:
      BaseState::qreg_.append_h(op.qubits[1]);
      BaseState::qreg_.append_cx(op.qubits[0], op.qubits[1]);
      BaseState::qreg_.append_h(op.qubits[1]);
      break;
    case Gates::swap:
      BaseState::qreg_.append_cx(op.qubits[0], op.qubits[1]);
      BaseState::qreg_.append_cx(op.qubits[1], op.qubits[0]);
      BaseState::qreg_.append_cx(op.qubits[0], op.qubits[1]);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument("Stabilizer::State::invalid gate instruction \'" +
                                  op.name + "\'.");
  }
}

//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================


void State::apply_measure(const reg_t &qubits,
                          const reg_t &cmemory,
                          const reg_t &cregister,
                          RngEngine &rng) {
  // Apply measurement and get classical outcome
  reg_t outcome = apply_measure_and_update(qubits, rng);
  // Add measurement outcome to creg
  BaseState::creg_.store_measure(outcome, cmemory, cregister);
}


void State::apply_reset(const reg_t &qubits, RngEngine &rng) {

  // Apply measurement and get classical outcome
  reg_t outcome = apply_measure_and_update(qubits, rng);
  // Use the outcome to apply X gate to any qubits left in the
  // |1> state after measure, then discard outcome.
  for (size_t j=0; j < qubits.size(); j++) {
    if (outcome[j] == 1) {
      qreg_.append_x(qubits[j]);
    }
  }
}


reg_t State::apply_measure_and_update(const reg_t &qubits,
                                      RngEngine &rng) {
  // Measurement outcome probabilities in the clifford
  // table are either deterministic or random.
  // We generate the distribution for the random case
  // which is used to generate the random integer
  // needed by the measure function.
  const rvector_t dist = {0.5, 0.5};
  reg_t outcome;
  // Measure each qubit
  for (const auto q : qubits) {
    uint_t r = rng.rand_int(dist);
    outcome.push_back(qreg_.measure_and_update(q, r));
  }
  return outcome;
}

std::vector<reg_t> State::sample_measure(const reg_t &qubits,
                                         uint_t shots,
                                         RngEngine &rng) {
  // TODO: see if we can improve efficiency by directly sampling from Clifford table
  auto qreg_cache = BaseState::qreg_;
  std::vector<reg_t> samples;
  samples.reserve(shots);
  while (shots-- > 0) { // loop over shots
    samples.push_back(apply_measure_and_update(qubits, rng));
    BaseState::qreg_ = qreg_cache; // restore pre-measurement data from cache
  }
  return samples;
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

void State::apply_snapshot(const Operations::Op &op,
                           ExperimentData &data) {

// Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument("Stabilizer::State::invalid snapshot instruction \'" + 
                                op.name + "\'.");
  switch (it->second) {
    case Snapshots::stabilizer:
      snapshot_stabilizer(op, data);
      break;
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(op, data);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(op, data);
      break;
    case Snapshots::probs: {
      snapshot_probabilities(op, data, false);
    } break;
    case Snapshots::probs_var: {
      snapshot_probabilities(op, data, true);
    } break;
    default:
      // We shouldn't get here unless there is a bug in the snapshotset
      throw std::invalid_argument("Stabilizer::State::invalid snapshot instruction \'" +
                                  op.name + "\'.");
  }
}


void State::snapshot_stabilizer(const Operations::Op &op, ExperimentData &data) {
  // We don't want to snapshot the full Clifford table, only the
  // stabilizer part. First Convert simulator clifford table to JSON
  json_t clifford = BaseState::qreg_;
  // Then extract the stabilizer generator list
  data.add_singleshot_snapshot("stabilizer",
                               op.string_params[0],
                               clifford["stabilizers"]);
}


void State::snapshot_probabilities(const Operations::Op &op,
                                   ExperimentData &data,
                                   bool variance) {
  // Check number of qubits being measured is less than 64.
  // otherwise we cant use 64-bit int logic.
  // Practical limits are much lower. For example:
  // A 32-qubit probability vector takes approx 16 GB of memory
  // to store.
  const size_t num_qubits = op.qubits.size();
  if (num_qubits > max_qubits_snapshot_probs_) {
    std::string msg = "Stabilizer::State::snapshot_probabilities: "
      "cannot return measure probabilities for " + std::to_string(num_qubits) +
      "-qubit measurement. Maximum is set to " +
      std::to_string(max_qubits_snapshot_probs_);
    throw std::runtime_error(msg);
  }

  // build X-stabilizer matrix for measure qubits
  std::vector<uint_t> x_stab;
  for(const auto& qubit : op.qubits){
    reg_t row;
    for(const auto& qubit2 : op.qubits){
      row.push_back(qreg_.stabilizer(qubit).X[qubit2]);
    }
    x_stab.push_back(Utils::reg2int(row, 2));
  }

  // Make a copy of the clifford table
  // and sample a single measurement outcome
  // We don't need to use the RNG here since we will be
  // enumerating over all outcomes later.
  auto qreg_copy = BaseState::qreg_;
  reg_t sample;
  for (const auto& q : op.qubits) {
    sample.push_back(qreg_copy.measure_and_update(q, 0));
  }
  const uint_t sample_outcome = Utils::reg2int(sample, 2); // convert to int
  // All other non-zero outcomes are
  // sample’= sample + b * x_stab (mod 2),
  // where b is a bitstring, and occur with same probabilities
  stringmap_t<double> probs;

  const uint_t num_outcomes = 1ULL << num_qubits;
  for (uint_t b=0; b < num_outcomes; b++) {
    uint_t outcome = sample_outcome;
    for (size_t j=0; j < num_qubits; j++) {
      // Check if j-th bit is 1
      if (b & (1ULL << j))
        outcome ^= x_stab[j];
    }
    // Check if outcome is in already in the probabilities
    // map and if not add it. We will renormalize at the end.
    const std::string outcome_hex = Utils::int2hex(outcome);
    if (probs.find(outcome_hex) == probs.end()) {
      probs[outcome_hex] = 1.0;
    }
  }

  // Renormalize outcomes
  auto renorm = probs.size();
  for (auto &pair : probs) {
    pair.second /= renorm;
  }

  // Add snapshot to data
  data.add_average_snapshot("probabilities", op.string_params[0],
                            BaseState::creg_.memory_hex(), probs, variance);
}


//------------------------------------------------------------------------------
} // end namespace Stabilizer
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
