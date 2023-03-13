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

#ifndef _superoperator_state_hpp
#define _superoperator_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/config.hpp"
#include "framework/json.hpp"
#include "framework/utils.hpp"
#include "simulators/state.hpp"
#include "superoperator.hpp"
#ifdef AER_THRUST_SUPPORTED
#include "superoperator_thrust.hpp"
#endif

namespace AER {
namespace QubitSuperoperator {

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {Operations::OpType::gate, Operations::OpType::reset,
     Operations::OpType::barrier, Operations::OpType::qerror_loc,
     Operations::OpType::bfunc, Operations::OpType::roerror,
     Operations::OpType::matrix, Operations::OpType::diagonal_matrix,
     Operations::OpType::kraus, Operations::OpType::superop,
     Operations::OpType::save_state, Operations::OpType::save_superop,
     Operations::OpType::set_unitary, Operations::OpType::set_superop,
     Operations::OpType::jump, Operations::OpType::mark},
    // Gates
    {"U",   "CX", "u1",   "u2",  "u3",    "u",     "cx",  "cy",  "cz",  "swap",
     "id",  "x",  "y",    "z",   "h",     "s",     "sdg", "t",   "tdg", "ccx",
     "r",   "rx", "ry",   "rz",  "rxx",   "ryy",   "rzz", "rzx", "p",   "cp",
     "cu1", "sx", "sxdg", "x90", "delay", "pauli", "ecr"});

// Allowed gates enum class
enum class Gates {
  u2,
  u1,
  u3,
  id,
  x,
  y,
  z,
  h,
  s,
  sdg,
  sx,
  sxdg,
  t,
  tdg,
  r,
  rx,
  ry,
  rz,
  cx,
  cy,
  cz,
  cp,
  swap,
  rxx,
  ryy,
  rzz,
  rzx,
  ccx,
  pauli,
  ecr
};

//=========================================================================
// QubitUnitary State subclass
//=========================================================================

template <class data_t = QV::Superoperator<double>>
class State : public QuantumState::State<data_t> {
public:
  using BaseState = QuantumState::State<data_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return "superop"; }

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  virtual void apply_op(const Operations::Op &op, ExperimentResult &result,
                        RngEngine &rng, bool final_op = false) override;

  // Initializes an n-qubit unitary to the identity matrix
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << 4 * num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // Config: {"omp_qubit_threshold": 3}
  virtual void set_config(const Config &config) override;

  virtual bool allocate(uint_t num_qubits, uint_t block_bits,
                        uint_t num_parallel_shots = 1) override;

  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp();

  auto move_to_matrix() { return BaseState::qreg_.move_to_matrix(); }

protected:
  //-----------------------------------------------------------------------
  // Apply Instructions
  //-----------------------------------------------------------------------

  // Applies a Gate operation to the state class.
  // This should support all and only the operations defined in
  // allowed_operations.
  void apply_gate(const Operations::Op &op);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cmatrix_t &mat);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t &vmat);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &krausops);

  // Apply an N-qubit Pauli gate
  void apply_pauli(const reg_t &qubits, const std::string &pauli);

  //-----------------------------------------------------------------------
  // Multi-controlled u3
  //-----------------------------------------------------------------------

  // Apply N-qubit multi-controlled single qubit waltz gate specified by
  // parameters u3(theta, phi, lambda)
  // NOTE: if N=1 this is just a regular u3 gate.
  void apply_gate_u3(const uint_t qubit, const double theta, const double phi,
                     const double lambda);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current superop matrix
  void apply_save_state(const Operations::Op &op, ExperimentResult &result,
                        bool last_op = false);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) override;

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 3;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;
};

//============================================================================
// Implementation: Allowed ops and gateset
//============================================================================

template <class data_t>
const stringmap_t<Gates> State<data_t>::gateset_({
    // Single qubit gates
    {"delay", Gates::id},  // Delay gate
    {"id", Gates::id},     // Pauli-Identity gate
    {"x", Gates::x},       // Pauli-X gate
    {"y", Gates::y},       // Pauli-Y gate
    {"z", Gates::z},       // Pauli-Z gate
    {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
    {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
    {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
    {"t", Gates::t},       // T-gate (sqrt(S))
    {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
    {"x90", Gates::sx},    // Pi/2 X (equiv to Sqrt(X) gate)
    {"sx", Gates::sx},     // Sqrt(X) gate
    {"sxdg", Gates::sxdg}, // Sqrt(X)^hc gate
    {"r", Gates::r},       // R rotation gate
    {"rx", Gates::rx},     // Pauli-X rotation gate
    {"ry", Gates::ry},     // Pauli-Y rotation gate
    {"rz", Gates::rz},     // Pauli-Z rotation gate
    // Waltz Gates
    {"p", Gates::u1},  // Phase gate
    {"u1", Gates::u1}, // zero-X90 pulse waltz gate
    {"u2", Gates::u2}, // single-X90 pulse waltz gate
    {"u3", Gates::u3}, // two X90 pulse waltz gate
    {"u", Gates::u3},  // two X90 pulse waltz gate
    {"U", Gates::u3},  // two X90 pulse waltz gate
    // Two-qubit gates
    {"CX", Gates::cx},     // Controlled-X gate (CNOT)
    {"cx", Gates::cx},     // Controlled-X gate (CNOT)
    {"cy", Gates::cy},     // Controlled-Y gate
    {"cz", Gates::cz},     // Controlled-Z gate
    {"cp", Gates::cp},     // Controlled-Phase gate
    {"cu1", Gates::cp},    // Controlled-Phase gate
    {"swap", Gates::swap}, // SWAP gate
    {"rxx", Gates::rxx},   // Pauli-XX rotation gate
    {"ryy", Gates::ryy},   // Pauli-YY rotation gate
    {"rzz", Gates::rzz},   // Pauli-ZZ rotation gate
    {"rzx", Gates::rzx},   // Pauli-ZX rotation gate
    {"ecr", Gates::ecr},   // ECR Gate
    // Three-qubit gates
    {"ccx", Gates::ccx},    // Controlled-CX gate (Toffoli)
    {"pauli", Gates::pauli} // Multiple pauli operations at once
});

//============================================================================
// Implementation: Base class method overrides
//============================================================================

template <class data_t>
void State<data_t>::apply_op(const Operations::Op &op, ExperimentResult &result,
                             RngEngine &rng, bool final_op) {
  if (BaseState::creg().check_conditional(op)) {
    switch (op.type) {
    case Operations::OpType::barrier:
    case Operations::OpType::qerror_loc:
      break;
    case Operations::OpType::gate:
      apply_gate(op);
      break;
    case Operations::OpType::bfunc:
      BaseState::creg().apply_bfunc(op);
      break;
    case Operations::OpType::roerror:
      BaseState::creg().apply_roerror(op, rng);
      break;
    case Operations::OpType::reset:
      apply_reset(op.qubits);
      break;
    case Operations::OpType::matrix:
      apply_matrix(op.qubits, op.mats[0]);
      break;
    case Operations::OpType::diagonal_matrix:
      BaseState::qreg_.apply_diagonal_matrix(op.qubits, op.params);
      break;
    case Operations::OpType::kraus:
      apply_kraus(op.qubits, op.mats);
      break;
    case Operations::OpType::superop:
      BaseState::qreg_.apply_superop_matrix(
          op.qubits, Utils::vectorize_matrix(op.mats[0]));
      break;
    case Operations::OpType::set_unitary:
    case Operations::OpType::set_superop:
      BaseState::qreg_.initialize_from_matrix(op.mats[0]);
      break;
    case Operations::OpType::save_state:
    case Operations::OpType::save_superop:
      apply_save_state(op, result, final_op);
      break;
    default:
      throw std::invalid_argument(
          "QubitSuperoperator::State::invalid instruction \'" + op.name +
          "\'.");
    }
  }
}

template <class data_t>
size_t State<data_t>::required_memory_mb(
    uint_t num_qubits, const std::vector<Operations::Op> &ops) const {
  // An n-qubit unitary as 2^4n complex doubles
  // where each complex double is 16 bytes
  (void)ops; // avoid unused variable compiler warning
  size_t shift_mb = std::max<int_t>(0, num_qubits + 4 - 20);
  size_t mem_mb = 1ULL << (4 * shift_mb);
  return mem_mb;
}

template <class data_t>
void State<data_t>::set_config(const Config &config) {
  // Set OMP threshold for state update functions
  if (config.superoperator_parallel_threshold.has_value())
    omp_qubit_threshold_ = config.superoperator_parallel_threshold.value();

  // Set threshold for truncating snapshots
  json_chop_threshold_ = config.zero_threshold;
  BaseState::qreg_.set_json_chop_threshold(json_chop_threshold_);
}

template <class data_t>
void State<data_t>::initialize_qreg(uint_t num_qubits) {
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize();
}

template <class data_t>
void State<data_t>::initialize_omp() {
  BaseState::qreg_.set_omp_threshold(omp_qubit_threshold_);
  if (BaseState::threads_ > 0)
    BaseState::qreg_.set_omp_threads(
        BaseState::threads_); // set allowed OMP threads in qubitvector
}

template <class data_t>
bool State<data_t>::allocate(uint_t num_qubits, uint_t block_bits,
                             uint_t num_parallel_shots) {
  return BaseState::qreg_.chunk_setup(num_qubits * 4, num_qubits * 4, 0, 1);
}

//=========================================================================
// Implementation: Reset
//=========================================================================

template <class data_t>
void State<data_t>::apply_reset(const reg_t &qubits) {
  // TODO: This can be more efficient by adding reset
  // to base class rather than doing a matrix multiplication
  // where all but 1 row is zeros.
  const auto reset_op = Linalg::SMatrix::reset(1ULL << qubits.size());
  BaseState::qreg_.apply_superop_matrix(qubits,
                                        Utils::vectorize_matrix(reset_op));
}

//=========================================================================
// Implementation: Kraus Noise
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_kraus(const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats) {
  BaseState::qreg_.apply_superop_matrix(
      qubits, Utils::vectorize_matrix(Utils::kraus_superop(kmats)));
}

//=========================================================================
// Implementation: Gates
//=========================================================================

template <class data_t>
void State<data_t>::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument("Unitary::State::invalid gate instruction \'" +
                                op.name + "\'.");
  switch (it->second) {
  case Gates::u3:
    apply_gate_u3(op.qubits[0], std::real(op.params[0]),
                  std::real(op.params[1]), std::real(op.params[2]));
    break;
  case Gates::u2:
    apply_gate_u3(op.qubits[0], M_PI / 2., std::real(op.params[0]),
                  std::real(op.params[1]));
    break;
  case Gates::u1:
    BaseState::qreg_.apply_phase(op.qubits[0],
                                 std::exp(complex_t(0., 1.) * op.params[0]));
    break;
  case Gates::r:
    apply_matrix(op.qubits, Linalg::VMatrix::r(op.params[0], op.params[1]));
    break;
  case Gates::rx:
    apply_matrix(op.qubits, Linalg::VMatrix::rx(op.params[0]));
    break;
  case Gates::ry:
    apply_matrix(op.qubits, Linalg::VMatrix::ry(op.params[0]));
    break;
  case Gates::rz:
    apply_matrix(op.qubits, Linalg::VMatrix::rz_diag(op.params[0]));
    break;
  case Gates::rxx:
    apply_matrix(op.qubits, Linalg::VMatrix::rxx(op.params[0]));
    break;
  case Gates::ryy:
    apply_matrix(op.qubits, Linalg::VMatrix::ryy(op.params[0]));
    break;
  case Gates::rzz:
    apply_matrix(op.qubits, Linalg::VMatrix::rzz_diag(op.params[0]));
    break;
  case Gates::rzx:
    apply_matrix(op.qubits, Linalg::VMatrix::rzx(op.params[0]));
    break;
  case Gates::ecr:
    apply_matrix(op.qubits, Linalg::VMatrix::ECR);
    break;
  case Gates::cx:
    BaseState::qreg_.apply_cnot(op.qubits[0], op.qubits[1]);
    break;
  case Gates::cy:
    apply_matrix(op.qubits, Linalg::VMatrix::CY);
    break;
  case Gates::cz:
    BaseState::qreg_.apply_cphase(op.qubits[0], op.qubits[1], -1);
    break;
  case Gates::cp:
    BaseState::qreg_.apply_cphase(op.qubits[0], op.qubits[1],
                                  std::exp(complex_t(0., 1.) * op.params[0]));
    break;
  case Gates::id:
    break;
  case Gates::x:
    BaseState::qreg_.apply_x(op.qubits[0]);
    break;
  case Gates::y:
    BaseState::qreg_.apply_y(op.qubits[0]);
    break;
  case Gates::z:
    BaseState::qreg_.apply_phase(op.qubits[0], -1);
    break;
  case Gates::h:
    apply_gate_u3(op.qubits[0], M_PI / 2., 0., M_PI);
    break;
  case Gates::s:
    BaseState::qreg_.apply_phase(op.qubits[0], complex_t(0., 1.));
    break;
  case Gates::sdg:
    BaseState::qreg_.apply_phase(op.qubits[0], complex_t(0., -1.));
    break;
  case Gates::sx:
    BaseState::qreg_.apply_unitary_matrix(op.qubits, Linalg::VMatrix::SX);
    break;
  case Gates::sxdg:
    BaseState::qreg_.apply_unitary_matrix(op.qubits, Linalg::VMatrix::SXDG);
    break;
  case Gates::t: {
    const double isqrt2{1. / std::sqrt(2)};
    BaseState::qreg_.apply_phase(op.qubits[0], complex_t(isqrt2, isqrt2));
  } break;
  case Gates::tdg: {
    const double isqrt2{1. / std::sqrt(2)};
    BaseState::qreg_.apply_phase(op.qubits[0], complex_t(isqrt2, -isqrt2));
  } break;
  case Gates::swap: {
    BaseState::qreg_.apply_swap(op.qubits[0], op.qubits[1]);
  } break;
  case Gates::ccx:
    BaseState::qreg_.apply_toffoli(op.qubits[0], op.qubits[1], op.qubits[2]);
    break;
  case Gates::pauli:
    apply_pauli(op.qubits, op.string_params[0]);
    break;
  default:
    // We shouldn't reach here unless there is a bug in gateset
    throw std::invalid_argument(
        "Superoperator::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

template <class data_t>
void State<data_t>::apply_matrix(const reg_t &qubits, const cmatrix_t &mat) {
  if (qubits.empty() == false && mat.size() > 0) {
    BaseState::qreg_.apply_unitary_matrix(qubits, Utils::vectorize_matrix(mat));
  }
}

template <class data_t>
void State<data_t>::apply_matrix(const reg_t &qubits, const cvector_t &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    BaseState::qreg_.apply_diagonal_unitary_matrix(qubits, vmat);
  } else {
    BaseState::qreg_.apply_unitary_matrix(qubits, vmat);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_gate_u3(const uint_t qubit, double theta,
                                      double phi, double lambda) {
  const auto u3 = Linalg::VMatrix::u3(theta, phi, lambda);
  BaseState::qreg_.apply_unitary_matrix(reg_t({qubit}), u3);
}

template <class statevec_t>
void State<statevec_t>::apply_pauli(const reg_t &qubits,
                                    const std::string &pauli) {
  // Pauli as a superoperator is (-1)^num_y P\otimes P
  complex_t coeff = (std::count(pauli.begin(), pauli.end(), 'Y') % 2) ? -1 : 1;
  BaseState::qreg_.apply_pauli(BaseState::qreg_.superop_qubits(qubits),
                               pauli + pauli, coeff);
}

template <class statevec_t>
void State<statevec_t>::apply_save_state(const Operations::Op &op,
                                         ExperimentResult &result,
                                         bool last_op) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(op.name + " was not applied to all qubits."
                                          " Only the full state can be saved.");
  }
  // Default key
  std::string key =
      (op.string_params[0] == "_method_") ? "superop" : op.string_params[0];
  if (last_op) {
    result.save_data_pershot(BaseState::creg(), key,
                             BaseState::qreg_.move_to_matrix(),
                             Operations::OpType::save_superop, op.save_type);
  } else {
    result.save_data_pershot(BaseState::creg(), key,
                             BaseState::qreg_.copy_to_matrix(),
                             Operations::OpType::save_superop, op.save_type);
  }
}

template <class data_t>
double State<data_t>::expval_pauli(const reg_t &qubits,
                                   const std::string &pauli) {
  throw std::runtime_error(
      "SuperOp simulator does not support Pauli expectation values.");
}

//------------------------------------------------------------------------------
} // end namespace QubitSuperoperator
} // end namespace AER
//------------------------------------------------------------------------------
#endif
