/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2022.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_tensor_net_state_hpp
#define _aer_tensor_net_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/config.hpp"
#include "framework/json.hpp"
#include "framework/opset.hpp"
#include "framework/utils.hpp"
#include "simulators/state.hpp"
#include "tensor_net.hpp"

#include "simulators/tensor_network/tensor_net.hpp"

namespace AER {

namespace TensorNetwork {

using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {OpType::gate,
     OpType::measure,
     OpType::reset,
     OpType::initialize,
     OpType::barrier,
     OpType::bfunc,
     OpType::roerror,
     OpType::matrix,
     OpType::diagonal_matrix,
     OpType::multiplexer,
     OpType::kraus,
     OpType::superop,
     OpType::qerror_loc,
     OpType::sim_op,
     OpType::set_statevec,
     OpType::set_densmat,
     OpType::save_expval,
     OpType::save_expval_var,
     OpType::save_probs,
     OpType::save_probs_ket,
     OpType::save_amps,
     OpType::save_amps_sq,
     OpType::save_state,
     OpType::save_statevec,
     OpType::save_statevec_dict,
     OpType::save_densmat,
     OpType::jump,
     OpType::mark},
    // Gates
    {"u1",   "u2",    "u3",     "u",     "U",     "CX",       "cx",
     "cz",   "cy",    "cp",     "cu1",   "cu2",   "cu3",      "swap",
     "id",   "p",     "x",      "y",     "z",     "h",        "s",
     "sdg",  "t",     "tdg",    "r",     "rx",    "ry",       "rz",
     "rxx",  "ryy",   "rzz",    "rzx",   "ccx",   "cswap",    "mcx",
     "mcy",  "mcz",   "mcu1",   "mcu2",  "mcu3",  "mcswap",   "mcphase",
     "mcr",  "mcrx",  "mcry",   "mcry",  "sx",    "sxdg",     "csx",
     "mcsx", "csxdg", "mcsxdg", "delay", "pauli", "mcx_gray", "cu",
     "mcu",  "mcp",   "ecr"});

// Allowed gates enum class
enum class Gates {
  id,
  h,
  s,
  sdg,
  t,
  tdg,
  rxx,
  ryy,
  rzz,
  rzx,
  mcx,
  mcy,
  mcz,
  mcr,
  mcrx,
  mcry,
  mcrz,
  mcp,
  mcu2,
  mcu3,
  mcu,
  mcswap,
  mcsx,
  mcsxdg,
  pauli,
  ecr
};

//=========================================================================
// TensorNet State subclass
//=========================================================================

template <class tensor_net_t = TensorNetwork::TensorNet<double>>
class State : public QuantumState::State<tensor_net_t> {
public:
  using BaseState = QuantumState::State<tensor_net_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return tensor_net_t::name(); }

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  virtual void apply_op(const Operations::Op &op, ExperimentResult &result,
                        RngEngine &rng, bool final_op = false) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is independent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                            RngEngine &rng) override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  void set_config(const Config &config) override;

  // Initializes an n-qubit state to the all |0> state
  void initialize_qreg(const uint_t num_qubits) override;

  // Initializes to a specific n-qubit state
  void initialize_qreg(const tensor_net_t &tensor);

  void initialize_from_vector(const cvector_t<double> &params);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) override;

  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------
  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp() {}

  auto move_to_vector();
  auto copy_to_vector();

  void enable_density_matrix(bool flg) { enable_density_matrix_ = flg; }

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
  virtual void apply_measure(const reg_t &qubits, const reg_t &cmemory,
                             const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits, RngEngine &rng);

  // Initialize the specified qubits to a given state |psi>
  // by applying a reset to the these qubits and then
  // computing the tensor product with the new state |psi>
  // /psi> is given in params
  void apply_initialize(const reg_t &qubits, const cvector_t<double> &params,
                        RngEngine &rng);

  void initialize_from_matrix(const cmatrix_t &params);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const Operations::Op &op);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t<double> &vmat);

  // apply diagonal matrix
  void apply_diagonal_matrix(const reg_t &qubits,
                             const cvector_t<double> &diag);

  // Apply a vector of control matrices to given qubits (identity on all other
  // qubits)
  void apply_multiplexer(const reg_t &control_qubits,
                         const reg_t &target_qubits,
                         const std::vector<cmatrix_t> &mmat);

  // Apply stacked (flat) version of multiplexer matrix to target qubits (using
  // control qubits to select matrix instance)
  void apply_multiplexer(const reg_t &control_qubits,
                         const reg_t &target_qubits, const cmatrix_t &mat);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &krausops,
                   RngEngine &rng);

  // Apply the global phase
  void apply_global_phase(void);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current state of the statevector simulator
  // If `last_op` is True this will use move semantics to move the simulator
  // state to the results, otherwise it will use copy semantics to leave
  // the current simulator state unchanged.
  void apply_save_statevector(const Operations::Op &op,
                              ExperimentResult &result, bool last_op);

  // Save the current state of the statevector simulator as a ket-form map.
  void apply_save_statevector_dict(const Operations::Op &op,
                                   ExperimentResult &result);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const Operations::Op &op,
                                 ExperimentResult &result);

  // Helper function for computing expectation value
  void apply_save_probs(const Operations::Op &op, ExperimentResult &result);

  // Helper function for saving amplitudes and amplitudes squared
  void apply_save_amplitudes(const Operations::Op &op,
                             ExperimentResult &result);

  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Return vector of measure probabilities for specified qubits
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  // TODO: move to private (no longer part of base class)
  rvector_t measure_probs(const reg_t &qubits) const;

  // Sample the measurement outcome for qubits
  // return a pair (m, p) of the outcome m, and its corresponding
  // probability p.
  // Outcome is given as an int: Eg for two-qubits {q0, q1} we have
  // 0 -> |q1 = 0, q0 = 0> state
  // 1 -> |q1 = 0, q0 = 1> state
  // 2 -> |q1 = 1, q0 = 0> state
  // 3 -> |q1 = 1, q0 = 1> state
  std::pair<uint_t, double> sample_measure_with_prob(const reg_t &qubits,
                                                     RngEngine &rng);

  void measure_reset_update(const std::vector<uint_t> &qubits,
                            const uint_t final_state, const uint_t meas_state,
                            const double meas_prob);

  //-----------------------------------------------------------------------
  // Single-qubit gate helpers
  //-----------------------------------------------------------------------

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Multi-controlled u3
  //-----------------------------------------------------------------------

  // Apply N-qubit multi-controlled single qubit gate specified by
  // 4 parameters u4(theta, phi, lambda, gamma)
  // NOTE: if N=1 this is just a regular u4 gate.
  void apply_gate_mcu(const reg_t &qubits, const double theta, const double phi,
                      const double lambda, const double gamma);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  uint_t num_sampling_qubits_ = 10;
  bool use_cuTensorNet_autotuning_ = false;

  bool enable_density_matrix_ = true;
};

//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

template <class tensor_net_t>
const stringmap_t<Gates> State<tensor_net_t>::gateset_(
    {                         // 1-qubit gates
     {"delay", Gates::id},    // Delay gate
     {"id", Gates::id},       // Pauli-Identity gate
     {"x", Gates::mcx},       // Pauli-X gate
     {"y", Gates::mcy},       // Pauli-Y gate
     {"z", Gates::mcz},       // Pauli-Z gate
     {"s", Gates::s},         // Phase gate (aka sqrt(Z) gate)
     {"sdg", Gates::sdg},     // Conjugate-transpose of Phase gate
     {"h", Gates::h},         // Hadamard gate (X + Z / sqrt(2))
     {"t", Gates::t},         // T-gate (sqrt(S))
     {"tdg", Gates::tdg},     // Conjguate-transpose of T gate
     {"p", Gates::mcp},       // Parameterized phase gate
     {"sx", Gates::mcsx},     // Sqrt(X) gate
     {"sxdg", Gates::mcsxdg}, // Inverse Sqrt(X) gate
     /* 1-qubit rotation Gates */
     {"r", Gates::mcr},   // R rotation gate
     {"rx", Gates::mcrx}, // Pauli-X rotation gate
     {"ry", Gates::mcry}, // Pauli-Y rotation gate
     {"rz", Gates::mcrz}, // Pauli-Z rotation gate
     /* Waltz Gates */
     {"u1", Gates::mcp},  // zero-X90 pulse waltz gate
     {"u2", Gates::mcu2}, // single-X90 pulse waltz gate
     {"u3", Gates::mcu3}, // two X90 pulse waltz gate
     {"u", Gates::mcu3},  // two X90 pulse waltz gate
     {"U", Gates::mcu3},  // two X90 pulse waltz gate
     /* 2-qubit gates */
     {"CX", Gates::mcx},       // Controlled-X gate (CNOT)
     {"cx", Gates::mcx},       // Controlled-X gate (CNOT)
     {"cy", Gates::mcy},       // Controlled-Y gate
     {"cz", Gates::mcz},       // Controlled-Z gate
     {"cp", Gates::mcp},       // Controlled-Phase gate
     {"cu1", Gates::mcp},      // Controlled-u1 gate
     {"cu2", Gates::mcu2},     // Controlled-u2 gate
     {"cu3", Gates::mcu3},     // Controlled-u3 gate
     {"cu", Gates::mcu},       // Controlled-u4 gate
     {"cp", Gates::mcp},       // Controlled-Phase gate
     {"swap", Gates::mcswap},  // SWAP gate
     {"rxx", Gates::rxx},      // Pauli-XX rotation gate
     {"ryy", Gates::ryy},      // Pauli-YY rotation gate
     {"rzz", Gates::rzz},      // Pauli-ZZ rotation gate
     {"rzx", Gates::rzx},      // Pauli-ZX rotation gate
     {"csx", Gates::mcsx},     // Controlled-Sqrt(X) gate
     {"csxdg", Gates::mcsxdg}, // Controlled-Sqrt(X)dg gate
     {"ecr", Gates::ecr},      // ECR Gate
     /* 3-qubit gates */
     {"ccx", Gates::mcx},      // Controlled-CX gate (Toffoli)
     {"cswap", Gates::mcswap}, // Controlled SWAP gate (Fredkin)
     /* Multi-qubit controlled gates */
     {"mcx", Gates::mcx},       // Multi-controlled-X gate
     {"mcy", Gates::mcy},       // Multi-controlled-Y gate
     {"mcz", Gates::mcz},       // Multi-controlled-Z gate
     {"mcr", Gates::mcr},       // Multi-controlled R-rotation gate
     {"mcrx", Gates::mcrx},     // Multi-controlled X-rotation gate
     {"mcry", Gates::mcry},     // Multi-controlled Y-rotation gate
     {"mcrz", Gates::mcrz},     // Multi-controlled Z-rotation gate
     {"mcphase", Gates::mcp},   // Multi-controlled-Phase gate
     {"mcp", Gates::mcp},       // Multi-controlled-Phase gate
     {"mcu1", Gates::mcp},      // Multi-controlled-u1
     {"mcu2", Gates::mcu2},     // Multi-controlled-u2
     {"mcu3", Gates::mcu3},     // Multi-controlled-u3
     {"mcu", Gates::mcu},       // Multi-controlled-u4
     {"mcswap", Gates::mcswap}, // Multi-controlled SWAP gate
     {"mcsx", Gates::mcsx},     // Multi-controlled-Sqrt(X) gate
     {"mcsxdg", Gates::mcsxdg}, // Multi-controlled-Sqrt(X)dg gate
     {"pauli", Gates::pauli},   // Multi-qubit Pauli gate
     {"mcx_gray", Gates::mcx}});

//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------

template <class tensor_net_t>
void State<tensor_net_t>::initialize_qreg(const uint_t num_qubits) {
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.set_num_sampling_qubits(num_sampling_qubits_);
  BaseState::qreg_.initialize();

  apply_global_phase();
}

template <class tensor_net_t>
void State<tensor_net_t>::initialize_qreg(const tensor_net_t &tensor) {
  BaseState::qreg_.initialize(tensor);
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

template <class tensor_net_t>
void State<tensor_net_t>::apply_global_phase(void) {
  BaseState::qreg_.apply_diagonal_matrix(
      {0}, {BaseState::global_phase_, BaseState::global_phase_});
}

template <class tensor_net_t>
size_t State<tensor_net_t>::required_memory_mb(
    uint_t num_qubits, const std::vector<Operations::Op> &ops) const {
  return 0;
}

template <class tensor_net_t>
void State<tensor_net_t>::set_config(const Config &config) {
  // Set threshold for truncating snapshots
  json_chop_threshold_ = config.zero_threshold;

  num_sampling_qubits_ = config.tensor_network_num_sampling_qubits;
  use_cuTensorNet_autotuning_ = config.use_cuTensorNet_autotuning;
}

template <class tensor_net_t>
auto State<tensor_net_t>::move_to_vector() {
  return BaseState::qreg_.move_to_vector();
}

template <class tensor_net_t>
auto State<tensor_net_t>::copy_to_vector() {
  return BaseState::qreg_.copy_to_vector();
}

//=========================================================================
// Implementation: apply operations
//=========================================================================
template <class tensor_net_t>
void State<tensor_net_t>::apply_op(const Operations::Op &op,
                                   ExperimentResult &result, RngEngine &rng,
                                   bool final_op) {
  if (BaseState::creg().check_conditional(op)) {
    switch (op.type) {
    case OpType::barrier:
    case OpType::nop:
    case OpType::qerror_loc:
      break;
    case OpType::reset:
      apply_reset(op.qubits, rng);
      break;
    case OpType::initialize:
      apply_initialize(op.qubits, op.params, rng);
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
    case OpType::matrix:
      apply_matrix(op);
      break;
    case OpType::diagonal_matrix:
      apply_diagonal_matrix(op.qubits, op.params);
      break;
    case OpType::multiplexer:
      apply_multiplexer(op.regs[0], op.regs[1],
                        op.mats); // control qubits ([0]) & target qubits([1])
      break;
    case OpType::superop:
      BaseState::qreg_.apply_superop_matrix(
          op.qubits, Utils::vectorize_matrix(op.mats[0]));
      break;
    case OpType::kraus:
      apply_kraus(op.qubits, op.mats, rng);
      break;
    case OpType::set_statevec:
      initialize_from_vector(op.params);
      break;
    case OpType::set_densmat:
      initialize_from_matrix(op.mats[0]);
      break;
    case OpType::save_expval:
    case OpType::save_expval_var:
      BaseState::apply_save_expval(op, result);
      break;
    case OpType::save_densmat:
      apply_save_density_matrix(op, result);
      break;
    case OpType::save_state:
    case OpType::save_statevec:
      apply_save_statevector(op, result, final_op);
      break;
    case OpType::save_statevec_dict:
      apply_save_statevector_dict(op, result);
      break;
    case OpType::save_probs:
    case OpType::save_probs_ket:
      apply_save_probs(op, result);
      break;
    case OpType::save_amps:
    case OpType::save_amps_sq:
      apply_save_amplitudes(op, result);
      break;
    default:
      throw std::invalid_argument("TensorNet::State::invalid instruction \'" +
                                  op.name + "\'.");
    }
  }
}

//=========================================================================
// Implementation: Save data
//=========================================================================

template <class tensor_net_t>
void State<tensor_net_t>::apply_save_probs(const Operations::Op &op,
                                           ExperimentResult &result) {
  // get probs as hexadecimal
  auto probs = measure_probs(op.qubits);
  if (op.type == Operations::OpType::save_probs_ket) {
    // Convert to ket dict
    result.save_data_average(BaseState::creg(), op.string_params[0],
                             Utils::vec2ket(probs, json_chop_threshold_, 16),
                             op.type, op.save_type);
  } else {
    result.save_data_average(BaseState::creg(), op.string_params[0],
                             std::move(probs), op.type, op.save_type);
  }
}

template <class tensor_net_t>
double State<tensor_net_t>::expval_pauli(const reg_t &qubits,
                                         const std::string &pauli) {
  return BaseState::qreg_.expval_pauli(qubits, pauli);
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_save_statevector(const Operations::Op &op,
                                                 ExperimentResult &result,
                                                 bool last_op) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  std::string key = (op.string_params[0] == "_method_") ? "tensor_network"
                                                        : op.string_params[0];

  if (last_op) {
    result.save_data_pershot(BaseState::creg(), key, move_to_vector(),
                             OpType::save_statevec, op.save_type);
  } else {
    result.save_data_pershot(BaseState::creg(), key, copy_to_vector(),
                             OpType::save_statevec, op.save_type);
  }
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_save_statevector_dict(
    const Operations::Op &op, ExperimentResult &result) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }

  auto state_ket = BaseState::qreg_.vector_ket(json_chop_threshold_);
  std::map<std::string, complex_t> result_state_ket;
  for (auto const &it : state_ket) {
    result_state_ket[it.first] = it.second;
  }
  result.save_data_pershot(BaseState::creg(), op.string_params[0],
                           std::move(result_state_ket), op.type, op.save_type);
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_save_density_matrix(const Operations::Op &op,
                                                    ExperimentResult &result) {
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    reduced_state[0] = BaseState::qreg_.norm();
  } else {
    reduced_state = BaseState::qreg_.reduced_density_matrix(op.qubits);
  }
  result.save_data_average(BaseState::creg(), op.string_params[0],
                           std::move(reduced_state), op.type, op.save_type);
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_save_amplitudes(const Operations::Op &op,
                                                ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  if (op.type == Operations::OpType::save_amps) {
    Vector<complex_t> amps(size, false);
    for (int_t i = 0; i < size; ++i) {
      amps[i] = BaseState::qreg_.get_state(op.int_params[i]);
    }
    result.save_data_pershot(BaseState::creg(), op.string_params[0],
                             std::move(amps), op.type, op.save_type);
  } else {
    rvector_t amps_sq(size, 0);
    for (int_t i = 0; i < size; ++i) {
      amps_sq[i] = BaseState::qreg_.probability(op.int_params[i]);
    }
    result.save_data_average(BaseState::creg(), op.string_params[0],
                             std::move(amps_sq), op.type, op.save_type);
  }
}

//=========================================================================
// Implementation: Matrix multiplication
//=========================================================================

template <class tensor_net_t>
void State<tensor_net_t>::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
        "QubitVectorState::invalid gate instruction \'" + op.name + "\'.");
  switch (it->second) {
  case Gates::mcx:
    // Includes X, CX, CCX, etc
    BaseState::qreg_.apply_mcx(op.qubits);
    break;
  case Gates::mcy:
    // Includes Y, CY, CCY, etc
    BaseState::qreg_.apply_mcy(op.qubits);
    break;
  case Gates::mcz:
    // Includes Z, CZ, CCZ, etc
    BaseState::qreg_.apply_mcphase(op.qubits, -1);
    break;
  case Gates::mcr:
    BaseState::qreg_.apply_mcu(op.qubits,
                               Linalg::VMatrix::r(op.params[0], op.params[1]));
    break;
  case Gates::mcrx:
    BaseState::qreg_.apply_rotation(op.qubits, TensorNetwork::Rotation::x,
                                    std::real(op.params[0]));
    break;
  case Gates::mcry:
    BaseState::qreg_.apply_rotation(op.qubits, TensorNetwork::Rotation::y,
                                    std::real(op.params[0]));
    break;
  case Gates::mcrz:
    BaseState::qreg_.apply_rotation(op.qubits, TensorNetwork::Rotation::z,
                                    std::real(op.params[0]));
    break;
  case Gates::rxx:
    BaseState::qreg_.apply_rotation(op.qubits, TensorNetwork::Rotation::xx,
                                    std::real(op.params[0]));
    break;
  case Gates::ryy:
    BaseState::qreg_.apply_rotation(op.qubits, TensorNetwork::Rotation::yy,
                                    std::real(op.params[0]));
    break;
  case Gates::rzz:
    BaseState::qreg_.apply_rotation(op.qubits, TensorNetwork::Rotation::zz,
                                    std::real(op.params[0]));
    break;
  case Gates::rzx:
    BaseState::qreg_.apply_rotation(op.qubits, TensorNetwork::Rotation::zx,
                                    std::real(op.params[0]));
    break;
  case Gates::ecr:
    BaseState::qreg_.apply_matrix(op.qubits, Linalg::VMatrix::ECR);
  case Gates::id:
    break;
  case Gates::h:
    apply_gate_mcu(op.qubits, M_PI / 2., 0., M_PI, 0.);
    break;
  case Gates::s:
    apply_gate_phase(op.qubits[0], complex_t(0., 1.));
    break;
  case Gates::sdg:
    apply_gate_phase(op.qubits[0], complex_t(0., -1.));
    break;
  case Gates::t: {
    const double isqrt2{1. / std::sqrt(2)};
    apply_gate_phase(op.qubits[0], complex_t(isqrt2, isqrt2));
  } break;
  case Gates::tdg: {
    const double isqrt2{1. / std::sqrt(2)};
    apply_gate_phase(op.qubits[0], complex_t(isqrt2, -isqrt2));
  } break;
  case Gates::mcswap:
    // Includes SWAP, CSWAP, etc
    BaseState::qreg_.apply_mcswap(op.qubits);
    break;
  case Gates::mcu3:
    // Includes u3, cu3, etc
    apply_gate_mcu(op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                   std::real(op.params[2]), 0.);
    break;
  case Gates::mcu:
    // Includes u3, cu3, etc
    apply_gate_mcu(op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                   std::real(op.params[2]), std::real(op.params[3]));
    break;
  case Gates::mcu2:
    // Includes u2, cu2, etc
    apply_gate_mcu(op.qubits, M_PI / 2., std::real(op.params[0]),
                   std::real(op.params[1]), 0.);
    break;
  case Gates::mcp:
    // Includes u1, cu1, p, cp, mcp etc
    BaseState::qreg_.apply_mcphase(op.qubits,
                                   std::exp(complex_t(0, 1) * op.params[0]));
    break;
  case Gates::mcsx:
    // Includes sx, csx, mcsx etc
    BaseState::qreg_.apply_mcu(op.qubits, Linalg::VMatrix::SX);
    break;
  case Gates::mcsxdg:
    BaseState::qreg_.apply_mcu(op.qubits, Linalg::VMatrix::SXDG);
    break;
  case Gates::pauli:
    BaseState::qreg_.apply_pauli(op.qubits, op.string_params[0]);
    break;
  default:
    // We shouldn't reach here unless there is a bug in gateset
    throw std::invalid_argument(
        "TensorNet::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_multiplexer(const reg_t &control_qubits,
                                            const reg_t &target_qubits,
                                            const cmatrix_t &mat) {
  if (control_qubits.empty() == false && target_qubits.empty() == false &&
      mat.size() > 0) {
    cvector_t<double> vmat = Utils::vectorize_matrix(mat);
    BaseState::qreg_.apply_multiplexer(control_qubits, target_qubits, vmat);
  }
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_matrix(const Operations::Op &op) {
  if (op.qubits.empty() == false && op.mats[0].size() > 0) {
    if (Utils::is_diagonal(op.mats[0], .0)) {
      apply_diagonal_matrix(op.qubits, Utils::matrix_diagonal(op.mats[0]));
    } else {
      BaseState::qreg_.apply_matrix(op.qubits,
                                    Utils::vectorize_matrix(op.mats[0]));
    }
  }
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t<double> &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    apply_diagonal_matrix(qubits, vmat);
  } else {
    BaseState::qreg_.apply_matrix(qubits, vmat);
  }
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t<double> &diag) {
  BaseState::qreg_.apply_diagonal_matrix(qubits, diag);
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_gate_mcu(const reg_t &qubits, double theta,
                                         double phi, double lambda,
                                         double gamma) {
  BaseState::qreg_.apply_mcu(qubits,
                             Linalg::VMatrix::u4(theta, phi, lambda, gamma));
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_gate_phase(uint_t qubit, complex_t phase) {
  cvector_t<double> diag = {{1., phase}};
  apply_diagonal_matrix(reg_t({qubit}), diag);
}

//=========================================================================
// Implementation: Reset, Initialize and Measurement Sampling
//=========================================================================

template <class tensor_net_t>
void State<tensor_net_t>::apply_measure(const reg_t &qubits,
                                        const reg_t &cmemory,
                                        const reg_t &cregister,
                                        RngEngine &rng) {
  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Implement measurement update
  measure_reset_update(qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());
  BaseState::creg().store_measure(outcome, cmemory, cregister);
}

template <class tensor_net_t>
rvector_t State<tensor_net_t>::measure_probs(const reg_t &qubits) const {
  return BaseState::qreg_.probabilities(qubits);
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_reset(const reg_t &qubits, RngEngine &rng) {
  // if there is no save_statevec, reset can be applied as density matrix mode
  if (enable_density_matrix_)
    BaseState::qreg_.apply_reset(qubits);
  else {
    // Simulate unobserved measurement
    const auto meas = sample_measure_with_prob(qubits, rng);
    // Apply update to reset state
    measure_reset_update(qubits, 0, meas.first, meas.second);
  }
}

template <class tensor_net_t>
std::pair<uint_t, double>
State<tensor_net_t>::sample_measure_with_prob(const reg_t &qubits,
                                              RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);
  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class tensor_net_t>
void State<tensor_net_t>::measure_reset_update(
    const std::vector<uint_t> &qubits, const uint_t final_state,
    const uint_t meas_state, const double meas_prob) {
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state

  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t<double> mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    BaseState::qreg_.apply_diagonal_matrix(qubits, mdiag);

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      BaseState::qreg_.apply_mcx(qubits);
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t<double> mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    BaseState::qreg_.apply_diagonal_matrix(qubits, mdiag);

    // If it doesn't agree with the reset state update
    // This function could be optimized as a permutation update
    if (final_state != meas_state) {
      // build vectorized permutation matrix
      cvector_t<double> perm(dim * dim, 0.);
      perm[final_state * dim + meas_state] = 1.;
      perm[meas_state * dim + final_state] = 1.;
      for (size_t j = 0; j < dim; j++) {
        if (j != final_state && j != meas_state)
          perm[j * dim + j] = 1.;
      }
      // apply permutation to swap state
      apply_matrix(qubits, perm);
    }
  }
}

template <class tensor_net_t>
std::vector<reg_t> State<tensor_net_t>::sample_measure(const reg_t &qubits,
                                                       uint_t shots,
                                                       RngEngine &rng) {
  // Generate flat register for storing
  std::vector<double> rnds(shots);

  for (uint_t i = 0; i < shots; ++i)
    rnds[i] = rng.rand(0, 1);

  std::vector<reg_t> samples = BaseState::qreg_.sample_measure(rnds);
  std::vector<reg_t> ret(shots);

  if (omp_get_num_threads() > 1) {
    for (uint_t i = 0; i < shots; ++i) {
      ret[i].resize(qubits.size());
      for (uint_t j = 0; j < qubits.size(); j++)
        ret[i][j] = samples[i][qubits[j]];
    }
  } else {
#pragma omp parallel for
    for (int_t i = 0; i < (int_t)shots; ++i) {
      ret[i].resize(qubits.size());
      for (uint_t j = 0; j < qubits.size(); j++)
        ret[i][j] = samples[i][qubits[j]];
    }
  }
  return ret;
}

template <class tensor_net_t>
void State<tensor_net_t>::apply_initialize(const reg_t &qubits,
                                           const cvector_t<double> &params_in,
                                           RngEngine &rng) {
  auto sorted_qubits = qubits;
  std::sort(sorted_qubits.begin(), sorted_qubits.end());
  // apply global phase here
  cvector_t<double> tmp;
  if (BaseState::has_global_phase_) {
    tmp.resize(params_in.size());
    auto apply_global_phase = [&tmp, params_in, this](int_t i) {
      tmp[i] = params_in[i] * BaseState::global_phase_;
    };
    Utils::apply_omp_parallel_for((qubits.size() > 14), 0, params_in.size(),
                                  apply_global_phase, BaseState::threads_);
  }
  const cvector_t<double> &params = tmp.empty() ? params_in : tmp;
  if (qubits.size() == BaseState::qreg_.num_qubits()) {
    // If qubits is all ordered qubits in the statevector
    // we can just initialize the whole state directly
    if (qubits == sorted_qubits) {
      initialize_from_vector(params);
      return;
    }
  }

  apply_reset(qubits, rng);

  BaseState::qreg_.initialize_component(qubits, params);
}

template <class tensor_net_t>
void State<tensor_net_t>::initialize_from_vector(
    const cvector_t<double> &params) {
  BaseState::qreg_.initialize();

  reg_t qubits(BaseState::qreg_.num_qubits());
  for (uint_t i = 0; i < BaseState::qreg_.num_qubits(); i++)
    qubits[i] = i;
  BaseState::qreg_.initialize_component(qubits, params);
}

template <class tensor_net_t>
void State<tensor_net_t>::initialize_from_matrix(const cmatrix_t &params) {
  BaseState::qreg_.initialize();
  reg_t qubits(BaseState::qreg_.num_qubits());

  BaseState::qreg_.initialize_from_matrix(params);
}

//=========================================================================
// Implementation: Multiplexer Circuit
//=========================================================================

template <class tensor_net_t>
void State<tensor_net_t>::apply_multiplexer(
    const reg_t &control_qubits, const reg_t &target_qubits,
    const std::vector<cmatrix_t> &mmat) {
  // (1) Pack vector of matrices into single (stacked) matrix ... note: matrix
  // dims: rows = DIM[qubit.size()] columns = DIM[|target bits|]
  cmatrix_t multiplexer_matrix = Utils::stacked_matrix(mmat);

  // (2) Treat as single, large(r), chained/batched matrix operator
  apply_multiplexer(control_qubits, target_qubits, multiplexer_matrix);
}

//=========================================================================
// Implementation: Kraus Noise
//=========================================================================
template <class tensor_net_t>
void State<tensor_net_t>::apply_kraus(const reg_t &qubits,
                                      const std::vector<cmatrix_t> &kmats,
                                      RngEngine &rng) {
  // Check edge case for empty Kraus set (this shouldn't happen)
  if (kmats.empty())
    return; // end function early

  // if there is no save_statevec, use density matrix mode
  if (enable_density_matrix_) {
    BaseState::qreg_.apply_superop_matrix(
        qubits, Utils::vectorize_matrix(Utils::kraus_superop(kmats)));
    return;
  }

  // Choose a real in [0, 1) to choose the applied kraus operator once
  // the accumulated probability is greater than r.
  // We know that the Kraus noise must be normalized
  // So we only compute probabilities for the first N-1 kraus operators
  // and infer the probability of the last one from 1 - sum of the previous

  double r;
  double accum = 0.;
  double p;
  bool complete = false;

  r = rng.rand(0., 1.);

  // Loop through N-1 kraus operators
  for (size_t j = 0; j < kmats.size() - 1; j++) {
    // Calculate probability
    std::vector<std::complex<double>> vmat = Utils::vectorize_matrix(kmats[j]);

    p = BaseState::qreg_.norm(qubits, vmat);
    accum += p;

    // check if we need to apply this operator
    if (accum > r) {
      // rescale vmat so projection is normalized
      Utils::scalar_multiply_inplace(vmat, 1 / std::sqrt(p));
      // apply Kraus projection operator
      apply_matrix(qubits, vmat);
      complete = true;
      break;
    }
  }

  // check if we haven't applied a kraus operator yet
  if (complete == false) {
    // Compute probability from accumulated
    complex_t renorm = 1 / std::sqrt(1. - accum);
    auto vmat = Utils::vectorize_matrix(renorm * kmats.back());
    apply_matrix(qubits, vmat);
  }
}

//-------------------------------------------------------------------------
} // namespace TensorNetwork
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
