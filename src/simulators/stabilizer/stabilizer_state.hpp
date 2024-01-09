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

#include "clifford.hpp"
#include "framework/config.hpp"
#include "framework/json.hpp"
#include "framework/utils.hpp"
#include "simulators/state.hpp"

namespace AER {
namespace Stabilizer {

//============================================================================
// Stabilizer state gates
//============================================================================
using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {OpType::gate, OpType::measure, OpType::reset, OpType::barrier,
     OpType::bfunc, OpType::qerror_loc, OpType::roerror, OpType::save_expval,
     OpType::save_expval_var, OpType::save_probs, OpType::save_probs_ket,
     OpType::save_amps_sq, OpType::save_stabilizer, OpType::save_clifford,
     OpType::save_state, OpType::set_stabilizer, OpType::jump, OpType::mark},
    // Gates
    {"CX", "cx",  "cy", "cz",   "swap",  "id",    "x",   "y",  "z",  "h",
     "s",  "sdg", "sx", "sxdg", "delay", "pauli", "ecr", "rx", "ry", "rz"});

enum class Gates {
  id,
  x,
  y,
  z,
  h,
  s,
  sdg,
  sx,
  sxdg,
  cx,
  cy,
  cz,
  swap,
  pauli,
  ecr,
  rx,
  ry,
  rz
};

//============================================================================
// Stabilizer Table state class
//============================================================================

class State : public QuantumState::State<Clifford::Clifford> {

public:
  using BaseState = QuantumState::State<Clifford::Clifford>;

  State() : BaseState(StateOpSet) {}

  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return "stabilizer"; }

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  virtual void apply_op(const Operations::Op &op, ExperimentResult &result,
                        RngEngine &rng, bool final_op = false) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // TODO: currently returns 0
  // Returns the required memory for storing an n-qubit state in megabytes.
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Load any settings for the State class from a config JSON
  virtual void set_config(const Config &config) override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                            RngEngine &rng) override;

  bool
  validate_parameters(const std::vector<Operations::Op> &ops) const override;

protected:
  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(const Operations::Op &op);

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_pauli(const reg_t &qubits, const std::string &pauli);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual void apply_measure(const reg_t &qubits, const reg_t &cmemory,
                             const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits, RngEngine &rng);

  // Set the state of the simulator to a given Clifford
  void apply_set_stabilizer(const Clifford::Clifford &clifford);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save Clifford state of simulator
  void apply_save_stabilizer(const Operations::Op &op,
                             ExperimentResult &result);

  // Save probabilities
  void apply_save_probs(const Operations::Op &op, ExperimentResult &result);

  // Helper function for saving amplitudes squared
  void apply_save_amplitudes_sq(const Operations::Op &op,
                                ExperimentResult &result);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) override;

  // Return the probability of an outcome bitstring.
  double get_probability(const reg_t &qubits, const std::string &outcome);

  template <typename T>
  void get_probabilities_auxiliary(const reg_t &qubits, std::string outcome,
                                   double outcome_prob, T &probs);

  void get_probability_helper(const reg_t &qubits, const std::string &outcome,
                              std::string &outcome_carry, double &prob_carry);

  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Implement a measurement on all specified qubits and return the outcome
  reg_t apply_measure_and_update(const reg_t &qubits, RngEngine &rng);

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
};

//============================================================================
// Implementation: Allowed ops and gateset
//============================================================================

const stringmap_t<Gates> State::gateset_({
    // Single qubit gates
    {"delay", Gates::id},  // Delay gate
    {"id", Gates::id},     // Pauli-Identity gate
    {"x", Gates::x},       // Pauli-X gate
    {"y", Gates::y},       // Pauli-Y gate
    {"z", Gates::z},       // Pauli-Z gate
    {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
    {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
    {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
    {"sx", Gates::sx},     // Sqrt X gate.
    {"sxdg", Gates::sxdg}, // Inverse Sqrt X gate.
    // Two-qubit gates
    {"CX", Gates::cx},       // Controlled-X gate (CNOT)
    {"cx", Gates::cx},       // Controlled-X gate (CNOT),
    {"cy", Gates::cy},       // Controlled-Y gate
    {"cz", Gates::cz},       // Controlled-Z gate
    {"swap", Gates::swap},   // SWAP gate
    {"pauli", Gates::pauli}, // Pauli gate
    {"ecr", Gates::ecr},     // ECR gate
    {"rx", Gates::rx},       // RX gate (only support k * pi/2 cases)
    {"ry", Gates::ry},       // RY gate (only support k * pi/2 cases)
    {"rz", Gates::rz}        // RZ gate (only support k * pi/2 cases)
});

//============================================================================
// Implementation: Base class method overrides
//============================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------

void State::initialize_qreg(uint_t num_qubits) {
  BaseState::qreg_.initialize(num_qubits);
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

size_t State::required_memory_mb(uint_t num_qubits,
                                 const std::vector<Operations::Op> &ops) const {
  (void)ops; // avoid unused variable compiler warning
  // The Clifford object requires very little memory.
  // A Pauli vector consists of 2 binary vectors each with
  // Binary vector = (4 + n // 64) 64-bit ints
  // Pauli = 2 * binary vector
  size_t mem = 16 * (4 + num_qubits); // Pauli bytes
  // Clifford = 2n * Pauli + 2n phase ints
  mem = 2 * num_qubits * (mem + 16); // Clifford bytes
  mem = mem >> 20;                   // Clifford mb
  return mem;
}

void State::set_config(const Config &config) {
  // Set threshold for truncating snapshots
  json_chop_threshold_ = config.zero_threshold;

  // Load max snapshot qubit size and set hard limit of 64 qubits.
  max_qubits_snapshot_probs_ = config.stabilizer_max_snapshot_probabilities;
  max_qubits_snapshot_probs_ = std::max<uint_t>(max_qubits_snapshot_probs_, 64);
}

bool State::validate_parameters(const std::vector<Operations::Op> &ops) const {
  for (uint_t i = 0; i < ops.size(); i++) {
    if (ops[i].type == OpType::gate) {
      // check parameter of R gates
      if (ops[i].name == "rx" || ops[i].name == "ry" || ops[i].name == "rz") {
        double pi2 = std::real(ops[i].params[0]) * 2.0 / M_PI;
        double pi2_int = (double)std::round(pi2);

        if (!AER::Linalg::almost_equal(pi2, pi2_int)) {
          return false;
        }
      }
    }
  }
  return true;
}

//=========================================================================
// Implementation: apply operations
//=========================================================================

void State::apply_op(const Operations::Op &op, ExperimentResult &result,
                     RngEngine &rng, bool final_op) {
  if (BaseState::creg().check_conditional(op)) {
    switch (op.type) {
    case OpType::barrier:
    case OpType::qerror_loc:
      break;
    case OpType::reset:
      apply_reset(op.qubits, rng);
      break;
    case OpType::measure:
      apply_measure(op.qubits, op.memory, op.registers, rng);
      break;
    case OpType::bfunc:
      BaseState::creg().apply_bfunc(op);
      break;
    case OpType::roerror:
      BaseState::creg().apply_roerror(op, rng);
      break;
    case OpType::gate:
      apply_gate(op);
      break;
    case OpType::set_stabilizer:
      apply_set_stabilizer(op.clifford);
      break;
    case OpType::save_expval:
    case OpType::save_expval_var:
      apply_save_expval(op, result);
      break;
    case OpType::save_probs:
    case OpType::save_probs_ket:
      apply_save_probs(op, result);
      break;
    case OpType::save_amps_sq:
      apply_save_amplitudes_sq(op, result);
      break;
    case OpType::save_state:
    case OpType::save_stabilizer:
    case OpType::save_clifford:
      apply_save_stabilizer(op, result);
      break;
    default:
      throw std::invalid_argument("Stabilizer::State::invalid instruction \'" +
                                  op.name + "\'.");
    }
  }
}

void State::apply_gate(const Operations::Op &op) {
  int_t pi2;
  // Check Op is supported by State
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
        "Stabilizer::State::invalid gate instruction \'" + op.name + "\'.");
  switch (it->second) {
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
  case Gates::sx:
    BaseState::qreg_.append_z(op.qubits[0]);
    BaseState::qreg_.append_s(op.qubits[0]);
    BaseState::qreg_.append_h(op.qubits[0]);
    BaseState::qreg_.append_z(op.qubits[0]);
    BaseState::qreg_.append_s(op.qubits[0]);
    break;
  case Gates::sxdg:
    BaseState::qreg_.append_s(op.qubits[0]);
    BaseState::qreg_.append_h(op.qubits[0]);
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
  case Gates::cy:
    BaseState::qreg_.append_z(op.qubits[1]);
    BaseState::qreg_.append_s(op.qubits[1]);
    BaseState::qreg_.append_cx(op.qubits[0], op.qubits[1]);
    BaseState::qreg_.append_s(op.qubits[1]);
    break;
  case Gates::swap:
    BaseState::qreg_.append_cx(op.qubits[0], op.qubits[1]);
    BaseState::qreg_.append_cx(op.qubits[1], op.qubits[0]);
    BaseState::qreg_.append_cx(op.qubits[0], op.qubits[1]);
    break;
  case Gates::pauli:
    apply_pauli(op.qubits, op.string_params[0]);
    break;
  case Gates::ecr:
    BaseState::qreg_.append_h(op.qubits[1]);
    BaseState::qreg_.append_s(op.qubits[0]);
    BaseState::qreg_.append_z(op.qubits[1]); // sdg(1)
    BaseState::qreg_.append_s(op.qubits[1]); // sdg(1)
    BaseState::qreg_.append_h(op.qubits[1]);
    BaseState::qreg_.append_cx(op.qubits[0], op.qubits[1]);
    BaseState::qreg_.append_x(op.qubits[0]);
    BaseState::qreg_.append_x(op.qubits[1]);
    break;
  case Gates::rx:
    pi2 = (int_t)std::round(std::real(op.params[0]) * 2.0 / M_PI) & 3;
    if (pi2 == 1) {
      // HSH
      BaseState::qreg_.append_h(op.qubits[0]);
      BaseState::qreg_.append_s(op.qubits[0]);
      BaseState::qreg_.append_h(op.qubits[0]);
    } else if (pi2 == 2) {
      // X
      BaseState::qreg_.append_x(op.qubits[0]);
    } else if (pi2 == 3) {
      // HSdgH
      BaseState::qreg_.append_h(op.qubits[0]);
      BaseState::qreg_.append_z(op.qubits[0]);
      BaseState::qreg_.append_s(op.qubits[0]);
      BaseState::qreg_.append_h(op.qubits[0]);
    }
    break;
  case Gates::ry:
    pi2 = (int_t)std::round(std::real(op.params[0]) * 2.0 / M_PI) & 3;
    if (pi2 == 1) {
      BaseState::qreg_.append_h(op.qubits[0]);
      BaseState::qreg_.append_x(op.qubits[0]);
    } else if (pi2 == 2) {
      // Y
      BaseState::qreg_.append_y(op.qubits[0]);
    } else if (pi2 == 3) {
      BaseState::qreg_.append_x(op.qubits[0]);
      BaseState::qreg_.append_h(op.qubits[0]);
    }
    break;
  case Gates::rz:
    pi2 = (int_t)std::round(std::real(op.params[0]) * 2.0 / M_PI) & 3;
    if (pi2 == 1) {
      // S
      BaseState::qreg_.append_s(op.qubits[0]);
    } else if (pi2 == 2) {
      // Z
      BaseState::qreg_.append_z(op.qubits[0]);
    } else if (pi2 == 3) {
      // Sdg
      BaseState::qreg_.append_z(op.qubits[0]);
      BaseState::qreg_.append_s(op.qubits[0]);
    }
    break;
  default:
    // We shouldn't reach here unless there is a bug in gateset
    throw std::invalid_argument(
        "Stabilizer::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

void State::apply_pauli(const reg_t &qubits, const std::string &pauli) {
  const auto size = qubits.size();
  for (size_t i = 0; i < qubits.size(); ++i) {
    const auto qubit = qubits[size - 1 - i];
    switch (pauli[i]) {
    case 'I':
      break;
    case 'X':
      BaseState::qreg_.append_x(qubit);
      break;
    case 'Y':
      BaseState::qreg_.append_y(qubit);
      break;
    case 'Z':
      BaseState::qreg_.append_z(qubit);
      break;
    default:
      throw std::invalid_argument("invalid Pauli \'" +
                                  std::to_string(pauli[i]) + "\'.");
    }
  }
}

//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

void State::apply_measure(const reg_t &qubits, const reg_t &cmemory,
                          const reg_t &cregister, RngEngine &rng) {
  // Apply measurement and get classical outcome
  reg_t outcome = apply_measure_and_update(qubits, rng);
  // Add measurement outcome to creg
  BaseState::creg().store_measure(outcome, cmemory, cregister);
}

void State::apply_reset(const reg_t &qubits, RngEngine &rng) {

  // Apply measurement and get classical outcome
  reg_t outcome = apply_measure_and_update(qubits, rng);
  // Use the outcome to apply X gate to any qubits left in the
  // |1> state after measure, then discard outcome.
  for (size_t j = 0; j < qubits.size(); j++) {
    if (outcome[j] == 1) {
      qreg_.append_x(qubits[j]);
    }
  }
}

reg_t State::apply_measure_and_update(const reg_t &qubits, RngEngine &rng) {
  // Measurement outcome probabilities in the clifford
  // table are either deterministic or random.
  // We generate the distribution for the random case
  // which is used to generate the random integer
  // needed by the measure function.
  const rvector_t dist = {0.5, 0.5};
  reg_t outcome;
  // Measure each qubit
  for (const auto &q : qubits) {
    uint_t r = rng.rand_int(dist);
    outcome.push_back(qreg_.measure_and_update(q, r));
  }
  return outcome;
}

std::vector<reg_t> State::sample_measure(const reg_t &qubits, uint_t shots,
                                         RngEngine &rng) {
  // TODO: see if we can improve efficiency by directly sampling from Clifford
  // table
  auto qreg_cache = BaseState::qreg_;
  std::vector<reg_t> samples;
  samples.reserve(shots);
  while (shots-- > 0) { // loop over shots
    samples.push_back(apply_measure_and_update(qubits, rng));
    BaseState::qreg_ = qreg_cache; // restore pre-measurement data from cache
  }
  return samples;
}

void State::apply_set_stabilizer(const Clifford::Clifford &clifford) {
  if (clifford.num_qubits() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(
        "set stabilizer must be defined on full width of qubits (" +
        std::to_string(clifford.num_qubits()) +
        " != " + std::to_string(BaseState::qreg_.num_qubits()) + ").");
  }
  BaseState::qreg_.apply_set_stabilizer(clifford);
}

//=========================================================================
// Implementation: Save data
//=========================================================================

void State::apply_save_stabilizer(const Operations::Op &op,
                                  ExperimentResult &result) {
  std::string key = op.string_params[0];
  OpType op_type = op.type;
  switch (op_type) {
  case OpType::save_clifford: {
    if (key == "_method_") {
      key = "clifford";
    }
    break;
  }
  case OpType::save_stabilizer:
  case OpType::save_state: {
    if (key == "_method_") {
      key = "stabilizer";
    }
    op_type = OpType::save_stabilizer;
    break;
  }
  default:
    // We shouldn't ever reach here...
    throw std::invalid_argument(
        "Invalid save state instruction for stabilizer");
  }
  json_t clifford = BaseState::qreg_;
  result.save_data_pershot(creg(), key, std::move(clifford), op_type,
                           op.save_type);
}

void State::apply_save_probs(const Operations::Op &op,
                             ExperimentResult &result) {
  // Check number of qubits being measured is less than 64.
  // otherwise we cant use 64-bit int logic.
  // Practical limits are much lower. For example:
  // A 32-qubit probability vector takes approx 16 GB of memory
  // to store.
  const size_t num_qubits = op.qubits.size();
  if (num_qubits > max_qubits_snapshot_probs_) {
    std::string msg = "Stabilizer::State::snapshot_probabilities: "
                      "cannot return measure probabilities for " +
                      std::to_string(num_qubits) +
                      "-qubit measurement. Maximum is set to " +
                      std::to_string(max_qubits_snapshot_probs_);
    throw std::runtime_error(msg);
  }
  if (op.type == OpType::save_probs_ket) {
    std::map<std::string, double> probs;
    get_probabilities_auxiliary(op.qubits, std::string(op.qubits.size(), 'X'),
                                1, probs);
    result.save_data_average(creg(), op.string_params[0], std::move(probs),
                             op.type, op.save_type);
  } else {
    std::vector<double> probs(1ULL << op.qubits.size(), 0.);
    get_probabilities_auxiliary(op.qubits, std::string(op.qubits.size(), 'X'),
                                1, probs);
    result.save_data_average(creg(), op.string_params[0], std::move(probs),
                             op.type, op.save_type);
  }
}

void State::apply_save_amplitudes_sq(const Operations::Op &op,
                                     ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes_sq instructions (empty params).");
  }
  uint_t num_qubits = op.qubits.size();
  if (num_qubits != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(
        "Save amplitude square must be defined on full width of qubits.");
  }
  rvector_t amps_sq(op.int_params.size(),
                    1.0); // Must be initialized in 1 for helper func
  for (size_t i = 0; i < op.int_params.size(); i++) {
    amps_sq[i] = get_probability(op.qubits,
                                 Utils::int2bin(op.int_params[i], num_qubits));
  }
  result.save_data_average(creg(), op.string_params[0], std::move(amps_sq),
                           op.type, op.save_type);
}

double State::expval_pauli(const reg_t &qubits, const std::string &pauli) {
  return BaseState::qreg_.expval_pauli(qubits, pauli);
}

static void set_value_helper(std::map<std::string, double> &probs,
                             const std::string &outcome, double value) {
  probs[Utils::bin2hex(outcome)] = value;
}

static void set_value_helper(std::vector<double> &probs,
                             const std::string &outcome, double value) {
  probs[std::stoull(outcome, 0, 2)] = value;
}

template <typename T>
void State::get_probabilities_auxiliary(const reg_t &qubits,
                                        std::string outcome,
                                        double outcome_prob, T &probs) {
  int_t qubit_for_branching = -1;
  for (uint_t i = 0; i < qubits.size(); ++i) {
    uint_t qubit = qubits[qubits.size() - i - 1];
    if (outcome[i] == 'X') {
      if (BaseState::qreg_.is_deterministic_outcome(qubit)) {
        bool single_qubit_outcome =
            BaseState::qreg_.measure_and_update(qubit, 0);
        if (single_qubit_outcome) {
          outcome[i] = '1';
        } else {
          outcome[i] = '0';
        }
      } else {
        qubit_for_branching = i;
      }
    }
  }

  if (qubit_for_branching == -1) {
    set_value_helper(probs, outcome, outcome_prob);
    return;
  }

  for (uint_t single_qubit_outcome = 0; single_qubit_outcome < 2;
       ++single_qubit_outcome) {
    std::string new_outcome = outcome;
    if (single_qubit_outcome) {
      new_outcome[qubit_for_branching] = '1';
    } else {
      new_outcome[qubit_for_branching] = '0';
    }

    auto copy_of_qreg = BaseState::qreg_;
    BaseState::qreg_.measure_and_update(
        qubits[qubits.size() - qubit_for_branching - 1], single_qubit_outcome);
    get_probabilities_auxiliary(qubits, new_outcome, 0.5 * outcome_prob, probs);
    BaseState::qreg_ = copy_of_qreg;
  }
}

double State::get_probability(const reg_t &qubits, const std::string &outcome) {
  std::string outcome_carry = std::string(qubits.size(), 'X');
  double prob = 1.0;
  get_probability_helper(qubits, outcome, outcome_carry, prob);
  return prob;
}

void State::get_probability_helper(const reg_t &qubits,
                                   const std::string &outcome,
                                   std::string &outcome_carry,
                                   double &prob_carry) {
  int_t qubit_for_branching = -1;
  for (uint_t i = 0; i < qubits.size(); ++i) {
    uint_t qubit = qubits[qubits.size() - i - 1];
    if (outcome_carry[i] == 'X') {
      if (BaseState::qreg_.is_deterministic_outcome(qubit)) {
        bool single_qubit_outcome =
            BaseState::qreg_.measure_and_update(qubit, 0);
        if (single_qubit_outcome) {
          outcome_carry[i] = '1';
        } else {
          outcome_carry[i] = '0';
        }
        if (outcome[i] != outcome_carry[i]) {
          prob_carry = 0.0;
          return;
        }
      } else {
        qubit_for_branching = i;
      }
    }
  }
  if (qubit_for_branching == -1) {
    return;
  }
  outcome_carry[qubit_for_branching] = outcome[qubit_for_branching];
  uint_t single_qubit_outcome = (outcome[qubit_for_branching] == '1') ? 1 : 0;
  auto cached_qreg = BaseState::qreg_;
  BaseState::qreg_.measure_and_update(
      qubits[qubits.size() - qubit_for_branching - 1], single_qubit_outcome);
  prob_carry *= 0.5;
  get_probability_helper(qubits, outcome, outcome_carry, prob_carry);
  BaseState::qreg_ = cached_qreg;
}

//------------------------------------------------------------------------------
} // end namespace Stabilizer
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
