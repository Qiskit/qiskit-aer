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

//=========================================================================
// Tensor Network State - simulation method
//=========================================================================
// For this simulation method, we represent the state of the circuit using a
// tensor network structure, the specifically matrix product state. The idea is
// based on the following paper (there exist other sources as well): The
// density-matrix renormalization group in the age of matrix product states by
// Ulrich Schollwock.
//
//--------------------------------------------------------------------------

#ifndef _matrix_product_state_hpp
#define _matrix_product_state_hpp

#include <algorithm>
#include <sstream>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/json.hpp"
#include "framework/linalg/almost_equal.hpp"
#include "framework/utils.hpp"
#include "matrix_product_state_internal.cpp"
#include "matrix_product_state_internal.hpp"
#include "simulators/state.hpp"

#include "matrix_product_state_size_estimator.hpp"

namespace AER {
namespace MatrixProductState {

static uint_t instruction_number = 0;

using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    {OpType::gate,
     OpType::measure,
     OpType::reset,
     OpType::initialize,
     OpType::barrier,
     OpType::bfunc,
     OpType::roerror,
     OpType::qerror_loc,
     OpType::matrix,
     OpType::diagonal_matrix,
     OpType::kraus,
     OpType::save_expval,
     OpType::save_expval_var,
     OpType::save_densmat,
     OpType::save_statevec,
     OpType::save_probs,
     OpType::save_probs_ket,
     OpType::save_amps,
     OpType::save_amps_sq,
     OpType::save_mps,
     OpType::save_state,
     OpType::set_mps,
     OpType::set_statevec,
     OpType::jump,
     OpType::mark},
    // Gates
    {"id",  "x",    "y",   "z",   "s",     "sdg",   "h",    "t",  "tdg", "p",
     "u1",  "u2",   "u3",  "u",   "U",     "CX",    "cx",   "cy", "cz",  "cp",
     "cu1", "swap", "ccx", "sx",  "sxdg",  "r",     "rx",   "ry", "rz",  "rxx",
     "ryy", "rzz",  "rzx", "csx", "delay", "cswap", "pauli"});

//=========================================================================
// Matrix Product State subclass
//=========================================================================

using matrixproductstate_t = MPS;

class State : public QuantumState::State<matrixproductstate_t> {
public:
  using BaseState = QuantumState::State<matrixproductstate_t>;

  State() : BaseState(StateOpSet) {}
  State(uint_t num_qubits) : State() { qreg_.initialize((uint_t)num_qubits); }
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return "matrix_product_state"; }

  bool empty() const { return qreg_.empty(); }

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  virtual void apply_op(const Operations::Op &op, ExperimentResult &result,
                        RngEngine &rng, bool final_op = false) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // We currently set the threshold to 1 in qasm_controller.hpp, i.e., no
  // parallelization
  virtual void set_config(const Config &config) override;

  virtual void add_metadata(ExperimentResult &result) const override;

  // prints the bond dimensions after each instruction to the metadata
  void output_bond_dimensions(const Operations::Op &op) const;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                            RngEngine &rng) override;

  // Computes sample_measure by copying the MPS to a temporary structure, and
  // applying a measurement on the temporary MPS. This is done for every shot,
  // so is not efficient for a large number of shots
  std::vector<reg_t> sample_measure_using_apply_measure(const reg_t &qubits,
                                                        uint_t shots,
                                                        RngEngine &rng);
  std::vector<reg_t> sample_measure_all(uint_t shots, RngEngine &rng);
  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  void initialize_omp();

protected:
  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(const Operations::Op &op);

  // Initialize the specified qubits to a given state |psi>
  // by creating the MPS state with the new state |psi>.
  // |psi> is given in params
  // Currently only supports intialization of all qubits
  void apply_initialize(const reg_t &qubits, const cvector_t &params,
                        RngEngine &rng);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function, then "measure"
  // should be contained in the set defined by 'allowed_ops'
  virtual void apply_measure(const reg_t &qubits, const reg_t &cmemory,
                             const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits, RngEngine &rng);

  // Apply a matrix to given qubits (identity on all other qubits)
  // We assume matrix to be 2x2
  void apply_matrix(const reg_t &qubits, const cmatrix_t &mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t &vmat);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &kmats,
                   RngEngine &rng);

  // Apply multi-qubit Pauli
  void apply_pauli(const reg_t &qubits, const std::string &pauli);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current state of the simulator
  void apply_save_mps(const Operations::Op &op, ExperimentResult &result,
                      bool last_op);

  // Compute and save the statevector for the current simulator state
  void apply_save_statevector(const Operations::Op &op,
                              ExperimentResult &result);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const Operations::Op &op,
                                 ExperimentResult &result);

  // Helper function for computing expectation value
  void apply_save_probs(const Operations::Op &op, ExperimentResult &result);

  // Helper function for saving amplitudes and amplitudes squared
  void apply_save_amplitudes(const Operations::Op &op,
                             ExperimentResult &result);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) override;

  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Return vector of measure probabilities for specified qubits
  // If a state subclass supports this function, then "measure"
  // must be contained in the set defined by 'allowed_ops'
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

  //-----------------------------------------------------------------------
  // Single-qubit gate helpers
  //-----------------------------------------------------------------------

  // Apply a waltz gate specified by parameters u3(theta, phi, lambda)
  void apply_gate_u3(const uint_t qubit, const double theta, const double phi,
                     const double lambda);

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;
};

//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

const stringmap_t<Gates>
    State::gateset_({                   // Single qubit gates
                     {"id", Gates::id}, // Pauli-Identity gate
                     {"delay", Gates::id},
                     {"x", Gates::x},       // Pauli-X gate
                     {"y", Gates::y},       // Pauli-Y gate
                     {"z", Gates::z},       // Pauli-Z gate
                     {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
                     {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
                     {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
                     {"sx", Gates::sx},     // Sqrt(X) gate
                     {"sxdg", Gates::sxdg}, // Inverse Sqrt(X) gate
                     {"t", Gates::t},       // T-gate (sqrt(S))
                     {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
                     {"r", Gates::r},       // R rotation gate
                     {"rx", Gates::rx},     // Pauli-X rotation gate
                     {"ry", Gates::ry},     // Pauli-Y rotation gate
                     {"rz", Gates::rz},     // Pauli-Z rotation gate
                     /* Waltz Gates */
                     {"p", Gates::u1},  // zero-X90 pulse waltz gate
                     {"u1", Gates::u1}, // zero-X90 pulse waltz gate
                     {"u2", Gates::u2}, // single-X90 pulse waltz gate
                     {"u3", Gates::u3}, // two X90 pulse waltz gate
                     {"u", Gates::u3},  // two X90 pulse waltz gate
                     {"U", Gates::u3},  // two X90 pulse waltz gate
                     /* Two-qubit gates */
                     {"CX", Gates::cx},   // Controlled-X gate (CNOT)
                     {"cx", Gates::cx},   // Controlled-X gate (CNOT)
                     {"cy", Gates::cy},   // Controlled-Y gate
                     {"cz", Gates::cz},   // Controlled-Z gate
                     {"cu1", Gates::cu1}, // Controlled-U1 gate
                     {"cp", Gates::cu1},  // Controlled-U1 gate
                     {"csx", Gates::csx},
                     {"swap", Gates::swap}, // SWAP gate
                     {"rxx", Gates::rxx},   // Pauli-XX rotation gate
                     {"ryy", Gates::ryy},   // Pauli-YY rotation gate
                     {"rzz", Gates::rzz},   // Pauli-ZZ rotation gate
                     {"rzx", Gates::rzx},   // Pauli-ZX rotation gate
                     /* Three-qubit gates */
                     {"ccx", Gates::ccx}, // Controlled-CX gate (Toffoli)
                     {"cswap", Gates::cswap},
                     /* Pauli */
                     {"pauli", Gates::pauli}});

//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------

void State::initialize_qreg(uint_t num_qubits = 0) {
  qreg_.initialize(num_qubits);
  if (BaseState::has_global_phase_) {
    BaseState::qreg_.apply_diagonal_matrix(
        {0}, {BaseState::global_phase_, BaseState::global_phase_});
  }
}

void State::initialize_omp() {
  if (BaseState::threads_ > 0)
    qreg_.set_omp_threads(
        BaseState::threads_); // set allowed OMP threads in MPS
}

size_t State::required_memory_mb(uint_t num_qubits,
                                 const std::vector<Operations::Op> &ops) const {
  if (num_qubits > 1) {
    MPSSizeEstimator est(num_qubits);
    uint_t size = est.estimate(ops);
    return (size >> 20);
  }
  return 0;
}

void State::set_config(const Config &config) {
  // Set threshold for truncating Schmidt coefficients
  MPS_Tensor::set_truncation_threshold(
      config.matrix_product_state_truncation_threshold);

  if (config.matrix_product_state_max_bond_dimension.has_value())
    MPS_Tensor::set_max_bond_dimension(
        config.matrix_product_state_max_bond_dimension.value());
  else
    MPS_Tensor::set_max_bond_dimension(UINT64_MAX);

  // Set threshold for truncating snapshots
  MPS::set_json_chop_threshold(config.chop_threshold);

  // Set OMP num threshold
  MPS::set_omp_threshold(config.mps_parallel_threshold);

  // Set OMP threads
  MPS::set_omp_threads(config.mps_omp_threads);

  // Set the algorithm for sample measure
  if (config.mps_sample_measure_algorithm.compare("mps_probabilities") == 0)
    MPS::set_sample_measure_alg(Sample_measure_alg::PROB);
  else
    MPS::set_sample_measure_alg(Sample_measure_alg::APPLY_MEASURE);

  // Set mps_log_data
  MPS::set_mps_log_data(config.mps_log_data);

  // Set the direction for the internal swaps
  std::string direction;
  if (config.mps_swap_direction.compare("mps_swap_right") == 0)
    MPS::set_mps_swap_direction(MPS_swap_direction::SWAP_RIGHT);
  else
    MPS::set_mps_swap_direction(MPS_swap_direction::SWAP_LEFT);
}

void State::add_metadata(ExperimentResult &result) const {
  result.metadata.add(MPS_Tensor::get_truncation_threshold(),
                      "matrix_product_state_truncation_threshold");
  result.metadata.add(MPS_Tensor::get_max_bond_dimension(),
                      "matrix_product_state_max_bond_dimension");
  result.metadata.add(MPS::get_sample_measure_alg(),
                      "matrix_product_state_sample_measure_algorithm");
  if (MPS::get_mps_log_data())
    result.metadata.add("{" + MPS::output_log() + "}", "MPS_log_data");
}

void State::output_bond_dimensions(const Operations::Op &op) const {
  MPS::print_to_log("I", instruction_number, ":", op.name, " on qubits ",
                    op.qubits[0]);
  for (uint_t index = 1; index < op.qubits.size(); index++) {
    MPS::print_to_log(",", op.qubits[index]);
  }
  qreg_.print_bond_dimensions();
  instruction_number++;
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
      apply_matrix(op.qubits, op.mats[0]);
      break;
    case OpType::diagonal_matrix:
      BaseState::qreg_.apply_diagonal_matrix(op.qubits, op.params);
      break;
    case OpType::kraus:
      apply_kraus(op.qubits, op.mats, rng);
      break;
    case OpType::set_statevec: {
      reg_t all_qubits(qreg_.num_qubits());
      std::iota(all_qubits.begin(), all_qubits.end(), 0);
      qreg_.apply_initialize(all_qubits, op.params, rng);
      break;
    }
    case OpType::set_mps:
      qreg_.initialize_from_mps(op.mps);
      break;
    case OpType::save_expval:
    case OpType::save_expval_var:
      BaseState::apply_save_expval(op, result);
      break;
    case OpType::save_densmat:
      apply_save_density_matrix(op, result);
      break;
    case OpType::save_statevec:
      apply_save_statevector(op, result);
      break;
    case OpType::save_state:
    case OpType::save_mps:
      apply_save_mps(op, result, final_op);
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
      throw std::invalid_argument(
          "MatrixProductState::State::invalid instruction \'" + op.name +
          "\'.");
    }
    // qreg_.print(std::cout);
    //  print out bond dimensions only if they may have changed since previous
    //  print
    if (MPS::get_mps_log_data() &&
        (op.type == OpType::gate || op.type == OpType::measure ||
         op.type == OpType::initialize || op.type == OpType::reset ||
         op.type == OpType::matrix) &&
        op.qubits.size() > 1) {
      output_bond_dimensions(op);
    }
  }
}

//=========================================================================
// Implementation: Save data
//=========================================================================

void State::apply_save_mps(const Operations::Op &op, ExperimentResult &result,
                           bool last_op) {
  if (op.qubits.size() != qreg_.num_qubits()) {
    throw std::invalid_argument(
        "Save MPS was not applied to all qubits."
        " Only the full matrix product state can be saved.");
  }
  std::string key = (op.string_params[0] == "_method_") ? "matrix_product_state"
                                                        : op.string_params[0];
  if (last_op) {
    result.save_data_pershot(creg(), key, qreg_.move_to_mps_container(),
                             OpType::save_mps, op.save_type);
  } else {
    result.save_data_pershot(creg(), key, qreg_.copy_to_mps_container(),
                             OpType::save_mps, op.save_type);
  }
}

void State::apply_save_probs(const Operations::Op &op,
                             ExperimentResult &result) {
  rvector_t probs;
  qreg_.get_probabilities_vector(probs, op.qubits);
  if (op.type == OpType::save_probs_ket) {
    result.save_data_average(
        creg(), op.string_params[0],
        Utils::vec2ket(probs, MPS::get_json_chop_threshold(), 16), op.type,
        op.save_type);
  } else {
    result.save_data_average(creg(), op.string_params[0], std::move(probs),
                             op.type, op.save_type);
  }
}

void State::apply_save_amplitudes(const Operations::Op &op,
                                  ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save amplitudes instructions (empty params).");
  }
  Vector<complex_t> amps = qreg_.get_amplitude_vector(op.int_params);
  if (op.type == OpType::save_amps_sq) {
    // Square amplitudes
    std::vector<double> amps_sq(op.int_params.size());
    std::transform(amps.data(), amps.data() + amps.size(), amps_sq.begin(),
                   [](complex_t val) -> double { return pow(abs(val), 2); });
    result.save_data_average(creg(), op.string_params[0], std::move(amps_sq),
                             op.type, op.save_type);
  } else {
    result.save_data_pershot(creg(), op.string_params[0], std::move(amps),
                             op.type, op.save_type);
  }
}

double State::expval_pauli(const reg_t &qubits, const std::string &pauli) {
  return BaseState::qreg_.expectation_value_pauli(qubits, pauli).real();
}

void State::apply_save_statevector(const Operations::Op &op,
                                   ExperimentResult &result) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(
        "Save statevector was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  result.save_data_pershot(creg(), op.string_params[0],
                           qreg_.full_statevector(), op.type, op.save_type);
}

void State::apply_save_density_matrix(const Operations::Op &op,
                                      ExperimentResult &result) {
  cmatrix_t reduced_state;
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);
    reduced_state[0] = qreg_.norm();
  } else {
    reduced_state = qreg_.density_matrix(op.qubits);
  }

  result.save_data_average(creg(), op.string_params[0],
                           std::move(reduced_state), op.type, op.save_type);
}

void State::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
        "MatrixProductState::State::invalid gate instruction \'" + op.name +
        "\'.");
  switch (it->second) {
  case Gates::ccx:
    qreg_.apply_ccx(op.qubits);
    break;
  case Gates::cswap:
    qreg_.apply_cswap(op.qubits);
    break;
  case Gates::u3:
    qreg_.apply_u3(op.qubits[0], std::real(op.params[0]),
                   std::real(op.params[1]), std::real(op.params[2]));
    break;
  case Gates::u2:
    qreg_.apply_u2(op.qubits[0], std::real(op.params[0]),
                   std::real(op.params[1]));
    break;
  case Gates::u1:
    qreg_.apply_u1(op.qubits[0], std::real(op.params[0]));
    break;
  case Gates::cx:
    qreg_.apply_cnot(op.qubits[0], op.qubits[1]);
    break;
  case Gates::id: {
    break;
  }
  case Gates::x:
    qreg_.apply_x(op.qubits[0]);
    break;
  case Gates::y:
    qreg_.apply_y(op.qubits[0]);
    break;
  case Gates::z:
    qreg_.apply_z(op.qubits[0]);
    break;
  case Gates::h:
    qreg_.apply_h(op.qubits[0]);
    break;
  case Gates::s:
    qreg_.apply_s(op.qubits[0]);
    break;
  case Gates::sdg:
    qreg_.apply_sdg(op.qubits[0]);
    break;
  case Gates::sx:
    qreg_.apply_sx(op.qubits[0]);
    break;
  case Gates::sxdg:
    qreg_.apply_sxdg(op.qubits[0]);
    break;
  case Gates::t:
    qreg_.apply_t(op.qubits[0]);
    break;
  case Gates::tdg:
    qreg_.apply_tdg(op.qubits[0]);
    break;
  case Gates::r:
    qreg_.apply_r(op.qubits[0], std::real(op.params[0]),
                  std::real(op.params[1]));
    break;
  case Gates::rx:
    qreg_.apply_rx(op.qubits[0], std::real(op.params[0]));
    break;
  case Gates::ry:
    qreg_.apply_ry(op.qubits[0], std::real(op.params[0]));
    break;
  case Gates::rz:
    qreg_.apply_rz(op.qubits[0], std::real(op.params[0]));
    break;
  case Gates::swap:
    qreg_.apply_swap(op.qubits[0], op.qubits[1], true);
    break;
  case Gates::cy:
    qreg_.apply_cy(op.qubits[0], op.qubits[1]);
    break;
  case Gates::cz:
    qreg_.apply_cz(op.qubits[0], op.qubits[1]);
    break;
  case Gates::csx:
    qreg_.apply_csx(op.qubits[0], op.qubits[1]);
    break;
  case Gates::cu1:
    qreg_.apply_cu1(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::rxx:
    qreg_.apply_rxx(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::ryy:
    qreg_.apply_ryy(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::rzz:
    qreg_.apply_rzz(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::rzx:
    qreg_.apply_rzx(op.qubits[0], op.qubits[1], std::real(op.params[0]));
    break;
  case Gates::pauli:
    apply_pauli(op.qubits, op.string_params[0]);
    break;
  default:
    // We shouldn't reach here unless there is a bug in gateset
    throw std::invalid_argument(
        "MatrixProductState::State::invalid gate instruction \'" + op.name +
        "\'.");
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
      BaseState::qreg_.apply_x(qubit);
      break;
    case 'Y':
      BaseState::qreg_.apply_y(qubit);
      break;
    case 'Z':
      BaseState::qreg_.apply_z(qubit);
      break;
    default:
      throw std::invalid_argument("invalid Pauli \'" +
                                  std::to_string(pauli[i]) + "\'.");
    }
  }
}

void State::apply_matrix(const reg_t &qubits, const cmatrix_t &mat) {
  if (!qubits.empty() && mat.size() > 0)
    qreg_.apply_matrix(qubits, mat);
}

void State::apply_matrix(const reg_t &qubits, const cvector_t &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    qreg_.apply_diagonal_matrix(qubits, vmat);
  } else {
    qreg_.apply_matrix(qubits, vmat);
  }
}

void State::apply_kraus(const reg_t &qubits,
                        const std::vector<cmatrix_t> &kmats, RngEngine &rng) {
  qreg_.apply_kraus(qubits, kmats, rng);
}

//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

void State::apply_initialize(const reg_t &qubits, const cvector_t &params,
                             RngEngine &rng) {
  // apply global phase here
  if (BaseState::has_global_phase_) {
    cvector_t tmp(params.size());
    auto apply_global_phase = [&tmp, params, this](int_t i) {
      tmp[i] = params[i] * BaseState::global_phase_;
    };
    Utils::apply_omp_parallel_for((qubits.size() > 14), 0, params.size(),
                                  apply_global_phase, BaseState::threads_);
    qreg_.apply_initialize(qubits, tmp, rng);
  } else {
    qreg_.apply_initialize(qubits, params, rng);
  }
}

void State::apply_measure(const reg_t &qubits, const reg_t &cmemory,
                          const reg_t &cregister, RngEngine &rng) {
  rvector_t rands;
  rands.reserve(qubits.size());
  for (uint_t i = 0; i < qubits.size(); ++i)
    rands.push_back(rng.rand(0., 1.));
  reg_t outcome = qreg_.apply_measure(qubits, rands);
  creg().store_measure(outcome, cmemory, cregister);
}

rvector_t State::measure_probs(const reg_t &qubits) const {
  rvector_t probvector;
  qreg_.get_probabilities_vector(probvector, qubits);
  return probvector;
}

std::vector<reg_t> State::sample_measure(const reg_t &qubits, uint_t shots,
                                         RngEngine &rng) {
  // There are two alternative algorithms for sample measure
  // We choose the one that is optimal relative to the total number
  // of qubits,and the number of shots.
  // The parameters used below are based on experimentation.
  // The user can override this by setting the parameter
  // "mps_sample_measure_algorithm"
  if (MPS::get_sample_measure_alg() == Sample_measure_alg::PROB &&
      qubits.size() == qreg_.num_qubits()) {
    return sample_measure_all(shots, rng);
  }
  return sample_measure_using_apply_measure(qubits, shots, rng);
}

std::vector<reg_t>
State::sample_measure_using_apply_measure(const reg_t &qubits, uint_t shots,
                                          RngEngine &rng) {
  std::vector<reg_t> all_samples;
  all_samples.resize(shots);
  std::vector<rvector_t> rnds_list;
  rnds_list.reserve(shots);
  for (uint_t i = 0; i < shots; ++i) {
    rvector_t rands;
    rands.reserve(qubits.size());
    for (uint_t j = 0; j < qubits.size(); ++j)
      rands.push_back(rng.rand(0., 1.));
    rnds_list.push_back(rands);
  }

#pragma omp parallel if (BaseState::threads_ > 1)                              \
    num_threads(BaseState::threads_)
  {
    MPS temp;
#pragma omp for
    for (int_t i = 0; i < static_cast<int_t>(shots); i++) {
      temp.initialize(qreg_);
      auto single_result = temp.apply_measure_internal(qubits, rnds_list[i]);
      all_samples[i] = single_result;
    }
  }
  return all_samples;
}

std::vector<reg_t> State::sample_measure_all(uint_t shots, RngEngine &rng) {
  std::vector<reg_t> all_samples;
  all_samples.resize(shots);

  for (uint_t i = 0; i < shots; i++) {
    auto single_result = qreg_.sample_measure(shots, rng);
    all_samples[i] = single_result;
  }
  return all_samples;
}

void State::apply_reset(const reg_t &qubits, RngEngine &rng) {
  qreg_.reset(qubits, rng);
}

std::pair<uint_t, double> State::sample_measure_with_prob(const reg_t &qubits,
                                                          RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);

  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

//-------------------------------------------------------------------------
} // end namespace MatrixProductState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif