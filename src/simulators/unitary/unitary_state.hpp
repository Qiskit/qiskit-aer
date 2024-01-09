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

#ifndef _unitary_state_hpp
#define _unitary_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include "framework/json.hpp"
#include "framework/utils.hpp"
#include "simulators/chunk_utils.hpp"
#include "simulators/state.hpp"
#include "unitarymatrix.hpp"
#include <math.h>
#ifdef AER_THRUST_SUPPORTED
#include "unitarymatrix_thrust.hpp"
#endif

namespace AER {

namespace QubitUnitary {

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {Operations::OpType::gate, Operations::OpType::barrier,
     Operations::OpType::bfunc, Operations::OpType::roerror,
     Operations::OpType::qerror_loc, Operations::OpType::matrix,
     Operations::OpType::diagonal_matrix, Operations::OpType::save_unitary,
     Operations::OpType::save_state, Operations::OpType::set_unitary,
     Operations::OpType::jump, Operations::OpType::mark},
    // Gates
    {"u1",     "u2",      "u3",    "u",      "U",     "CX",    "cx",   "cz",
     "cy",     "cp",      "cu1",   "cu2",    "cu3",   "swap",  "id",   "p",
     "x",      "y",       "z",     "h",      "s",     "sdg",   "t",    "tdg",
     "r",      "rx",      "ry",    "rz",     "rxx",   "ryy",   "rzz",  "rzx",
     "ccx",    "cswap",   "mcx",   "mcy",    "mcz",   "mcu1",  "mcu2", "mcu3",
     "mcswap", "mcphase", "mcr",   "mcrx",   "mcry",  "mcry",  "sx",   "sxdg",
     "csx",    "mcsx",    "csxdg", "mcsxdg", "delay", "pauli", "cu",   "mcu",
     "mcp",    "ecr"});

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
// QubitUnitary State subclass
//=========================================================================

template <class unitary_matrix_t = QV::UnitaryMatrix<double>>
class State : public virtual QuantumState::State<unitary_matrix_t> {
public:
  using BaseState = QuantumState::State<unitary_matrix_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return "unitary"; }

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  virtual void apply_op(const Operations::Op &op, ExperimentResult &result,
                        RngEngine &rng, bool final_op = false) override;

  // memory allocation (previously called before inisitalize_qreg)
  bool allocate(uint_t num_qubits, uint_t block_bits,
                uint_t num_parallel_shots = 1) override;

  // Initializes an n-qubit unitary to the identity matrix
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << 2 * num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // Config: {"omp_qubit_threshold": 7}
  virtual void set_config(const Config &config) override;

  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  // Initializes to a specific n-qubit unitary given as a complex matrix
  virtual void initialize_qreg(uint_t num_qubits, const cmatrix_t &unitary);

  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp();

  auto move_to_matrix();
  auto copy_to_matrix();

  // Apply the global phase
  void apply_global_phase();

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

  // Apply a diagonal matrix
  void apply_diagonal_matrix(const reg_t &qubits, const cvector_t &diag);

  //-----------------------------------------------------------------------
  // 1-Qubit Gates
  //-----------------------------------------------------------------------

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t qubit, const complex_t phase);

  void apply_gate_phase(const reg_t &qubits, const complex_t phase);

  //-----------------------------------------------------------------------
  // Multi-controlled u
  //-----------------------------------------------------------------------

  // Apply N-qubit multi-controlled single qubit gate specified by
  // 4 parameters u4(theta, phi, lambda, gamma)
  // NOTE: if N=1 this is just a regular u4 gate.
  void apply_gate_mcu(const reg_t &qubits, double theta, double phi,
                      double lambda, double gamma);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the unitary matrix for the simulator
  void apply_save_unitary(const Operations::Op &op, ExperimentResult &result,
                          bool last_op);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) override;

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 6;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;
};

//============================================================================
// Implementation: Allowed ops and gateset
//============================================================================

template <class unitary_matrix_t>
const stringmap_t<Gates> State<unitary_matrix_t>::gateset_({
    // Single qubit gates
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
    {"sxdg", Gates::mcsxdg}, // Sqrt(X)^hc gate
    // 1-qubit rotation Gates
    {"r", Gates::mcr},   // R rotation gate
    {"rx", Gates::mcrx}, // Pauli-X rotation gate
    {"ry", Gates::mcry}, // Pauli-Y rotation gate
    {"rz", Gates::mcrz}, // Pauli-Z rotation gate
    // Waltz Gates
    {"p", Gates::mcp},   // Parameterized phase gate
    {"u1", Gates::mcp},  // zero-X90 pulse waltz gate
    {"u2", Gates::mcu2}, // single-X90 pulse waltz gate
    {"u3", Gates::mcu3}, // two X90 pulse waltz gate
    {"u", Gates::mcu3},  // two X90 pulse waltz gate
    {"U", Gates::mcu3},  // two X90 pulse waltz gate
    // Two-qubit gates
    {"CX", Gates::mcx},       // Controlled-X gate (CNOT)
    {"cx", Gates::mcx},       // Controlled-X gate (CNOT)
    {"cy", Gates::mcy},       // Controlled-Z gate
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
    // Three-qubit gates
    {"ccx", Gates::mcx},      // Controlled-CX gate (Toffoli)
    {"cswap", Gates::mcswap}, // Controlled-SWAP gate (Fredkin)
    // Multi-qubit controlled gates
    {"mcx", Gates::mcx},       // Multi-controlled-X gate
    {"mcy", Gates::mcy},       // Multi-controlled-Y gate
    {"mcz", Gates::mcz},       // Multi-controlled-Z gate
    {"mcr", Gates::mcr},       // Multi-controlled R-rotation gate
    {"mcrx", Gates::mcrx},     // Multi-controlled X-rotation gate
    {"mcry", Gates::mcry},     // Multi-controlled Y-rotation gate
    {"mcrz", Gates::mcrz},     // Multi-controlled Z-rotation gate
    {"mcphase", Gates::mcp},   // Multi-controlled-Phase gate
    {"mcu1", Gates::mcp},      // Multi-controlled-u1
    {"mcu2", Gates::mcu2},     // Multi-controlled-u2
    {"mcu3", Gates::mcu3},     // Multi-controlled-u3
    {"mcu", Gates::mcu},       // Multi-controlled-u4
    {"mcp", Gates::mcp},       // Multi-controlled-Phase gate
    {"mcswap", Gates::mcswap}, // Multi-controlled SWAP gate
    {"mcsx", Gates::mcsx},     // Multi-controlled-Sqrt(X) gate
    {"mcsxdg", Gates::mcsxdg}, // Multi-controlled-Sqrt(X)dg gate
    {"pauli", Gates::pauli}    // Multiple pauli operations at once
});

//============================================================================
// Implementation: Base class method overrides
//============================================================================

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_op(const Operations::Op &op,
                                       ExperimentResult &result, RngEngine &rng,
                                       bool final_op) {
  if (BaseState::creg().check_conditional(op)) {
    switch (op.type) {
    case Operations::OpType::barrier:
    case Operations::OpType::qerror_loc:
      break;
    case Operations::OpType::bfunc:
      BaseState::creg().apply_bfunc(op);
      break;
    case Operations::OpType::roerror:
      BaseState::creg().apply_roerror(op, rng);
      break;
    case Operations::OpType::gate:
      apply_gate(op);
      break;
    case Operations::OpType::set_unitary:
      BaseState::qreg_.initialize_from_matrix(op.mats[0]);
      break;
    case Operations::OpType::save_state:
    case Operations::OpType::save_unitary:
      apply_save_unitary(op, result, final_op);
      break;
    case Operations::OpType::matrix:
      apply_matrix(op.qubits, op.mats[0]);
      break;
    case Operations::OpType::diagonal_matrix:
      apply_diagonal_matrix(op.qubits, op.params);
      break;
    default:
      throw std::invalid_argument(
          "QubitUnitary::State::invalid instruction \'" + op.name + "\'.");
    }
  }
}

template <class unitary_matrix_t>
size_t State<unitary_matrix_t>::required_memory_mb(
    uint_t num_qubits, const std::vector<Operations::Op> &ops) const {
  (void)ops; // avoid unused variable compiler warning
  unitary_matrix_t tmp;
  return tmp.required_memory_mb(2 * num_qubits);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::set_config(const Config &config) {
  BaseState::set_config(config);

  // Set OMP threshold for state update functions
  if (config.unitary_parallel_threshold.has_value())
    omp_qubit_threshold_ = config.unitary_parallel_threshold.value();

  // Set threshold for truncating snapshots
  json_chop_threshold_ = config.zero_threshold;

  BaseState::qreg_.set_json_chop_threshold(json_chop_threshold_);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_qreg(uint_t num_qubits) {
  initialize_omp();

  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize();

  apply_global_phase();
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_qreg(uint_t num_qubits,
                                              const cmatrix_t &unitary) {
  // Check dimension of unitary
  if (unitary.size() != 1ULL << (2 * num_qubits)) {
    throw std::invalid_argument(
        "Unitary::State::initialize: initial state does not match qubit "
        "number");
  }
  initialize_omp();

  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize_from_matrix(unitary);

  apply_global_phase();
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_omp() {
  BaseState::qreg_.set_omp_threshold(omp_qubit_threshold_);
  if (BaseState::threads_ > 0)
    BaseState::qreg_.set_omp_threads(
        BaseState::threads_); // set allowed OMP threads in qubitvector
}

template <class unitary_matrix_t>
bool State<unitary_matrix_t>::allocate(uint_t num_qubits, uint_t block_bits,
                                       uint_t num_parallel_shots) {
  if (BaseState::max_matrix_qubits_ > 0)
    BaseState::qreg_.set_max_matrix_bits(BaseState::max_matrix_qubits_);

  BaseState::qreg_.set_target_gpus(BaseState::target_gpus_);
  BaseState::qreg_.chunk_setup(block_bits * 2, num_qubits * 2, 0, 1);

  return true;
}

template <class unitary_matrix_t>
auto State<unitary_matrix_t>::move_to_matrix() {
  return BaseState::qreg_.move_to_matrix();
}

template <class unitary_matrix_t>
auto State<unitary_matrix_t>::copy_to_matrix() {
  return BaseState::qreg_.copy_to_matrix();
}

//=========================================================================
// Implementation: Gates
//=========================================================================

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate(const Operations::Op &op) {
  // CPU qubit vector does not handle chunk ID inside kernel, so modify op here
  if (BaseState::num_global_qubits_ > BaseState::qreg_.num_qubits() &&
      !BaseState::qreg_.support_global_indexing()) {
    reg_t qubits_in, qubits_out;
    if (op.name[0] == 'c' || op.name.find("mc") == 0) {
      Chunk::get_inout_ctrl_qubits(op, BaseState::qreg_.num_qubits(), qubits_in,
                                   qubits_out);
    }
    if (qubits_out.size() > 0) {
      uint_t mask = 0;
      for (uint_t i = 0; i < qubits_out.size(); i++) {
        mask |= (1ull << (qubits_out[i] - BaseState::qreg_.num_qubits()));
      }
      if ((BaseState::qreg_.chunk_index() & mask) == mask) {
        Operations::Op new_op = Chunk::correct_gate_op_in_chunk(op, qubits_in);
        apply_gate(new_op);
      }
      return;
    }
  }

  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument("Unitary::State::invalid gate instruction \'" +
                                op.name + "\'.");
  Gates g = it->second;
  switch (g) {
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
    BaseState::qreg_.apply_mcu(op.qubits, Linalg::VMatrix::rx(op.params[0]));
    break;
  case Gates::mcry:
    BaseState::qreg_.apply_mcu(op.qubits, Linalg::VMatrix::ry(op.params[0]));
    break;
  case Gates::mcrz:
    BaseState::qreg_.apply_mcu(op.qubits, Linalg::VMatrix::rz(op.params[0]));
    break;
  case Gates::rxx:
    BaseState::qreg_.apply_matrix(op.qubits,
                                  Linalg::VMatrix::rxx(op.params[0]));
    break;
  case Gates::ryy:
    BaseState::qreg_.apply_matrix(op.qubits,
                                  Linalg::VMatrix::ryy(op.params[0]));
    break;
  case Gates::rzz:
    apply_diagonal_matrix(op.qubits, Linalg::VMatrix::rzz_diag(op.params[0]));
    break;
  case Gates::rzx:
    BaseState::qreg_.apply_matrix(op.qubits,
                                  Linalg::VMatrix::rzx(op.params[0]));
    break;
  case Gates::ecr:
    BaseState::qreg_.apply_matrix(op.qubits, Linalg::VMatrix::ECR);
    break;
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
  case Gates::pauli:
    BaseState::qreg_.apply_pauli(op.qubits, op.string_params[0]);
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
    // Includes u, cu, etc
    apply_gate_mcu(op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                   std::real(op.params[2]), std::real(op.params[3]));
    break;
  case Gates::mcu2:
    // Includes u2, cu2, etc
    apply_gate_mcu(op.qubits, M_PI / 2., std::real(op.params[0]),
                   std::real(op.params[1]), 0.);
    break;
  case Gates::mcp:
    // Includes u1, cu1, p, cp, mcp, etc
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
  default:
    // We shouldn't reach here unless there is a bug in gateset
    throw std::invalid_argument("Unitary::State::invalid gate instruction \'" +
                                op.name + "\'.");
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_matrix(const reg_t &qubits,
                                           const cmatrix_t &mat) {
  if (qubits.empty() == false && mat.size() > 0) {
    apply_matrix(qubits, Utils::vectorize_matrix(mat));
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_matrix(const reg_t &qubits,
                                           const cvector_t &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    apply_diagonal_matrix(qubits, vmat);
  } else {
    BaseState::qreg_.apply_matrix(qubits, vmat);
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                    const cvector_t &diag) {
  if (BaseState::num_global_qubits_ > BaseState::qreg_.num_qubits()) {
    if (!BaseState::qreg_.support_global_indexing()) {
      reg_t qubits_in = qubits;
      cvector_t diag_in = diag;
      Chunk::block_diagonal_matrix(BaseState::qreg_.chunk_index(),
                                   BaseState::qreg_.num_qubits(), qubits_in,
                                   diag_in);
      BaseState::qreg_.apply_diagonal_matrix(qubits_in, diag_in);
    } else {
      reg_t qubits_chunk = qubits;
      for (uint_t i = 0; i < qubits.size(); i++) {
        if (qubits_chunk[i] >= BaseState::qreg_.num_qubits())
          qubits_chunk[i] += BaseState::qreg_.num_qubits();
      }
      BaseState::qreg_.apply_diagonal_matrix(qubits_chunk, diag);
    }
  } else {
    BaseState::qreg_.apply_diagonal_matrix(qubits, diag);
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_phase(uint_t qubit, complex_t phase) {
  cvector_t diag(2);
  diag[0] = 1.0;
  diag[1] = phase;
  apply_diagonal_matrix(reg_t({qubit}), diag);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_phase(const reg_t &qubits,
                                               complex_t phase) {
  cvector_t diag((1 << qubits.size()), 1.0);
  diag[(1 << qubits.size()) - 1] = phase;
  apply_diagonal_matrix(qubits, diag);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_mcu(const reg_t &qubits, double theta,
                                             double phi, double lambda,
                                             double gamma) {
  const auto u4 = Linalg::Matrix::u4(theta, phi, lambda, gamma);
  BaseState::qreg_.apply_mcu(qubits, Utils::vectorize_matrix(u4));
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_global_phase() {
  if (BaseState::has_global_phase_) {
    apply_diagonal_matrix({0},
                          {BaseState::global_phase_, BaseState::global_phase_});
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_save_unitary(const Operations::Op &op,
                                                 ExperimentResult &result,
                                                 bool last_op) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full unitary can be saved.");
  }
  std::string key =
      (op.string_params[0] == "_method_") ? "unitary" : op.string_params[0];

  if (last_op) {
    result.save_data_pershot(BaseState::creg(), key, move_to_matrix(),
                             Operations::OpType::save_unitary, op.save_type);
  } else {
    result.save_data_pershot(BaseState::creg(), key, copy_to_matrix(),
                             Operations::OpType::save_unitary, op.save_type);
  }
}

template <class unitary_matrix_t>
double State<unitary_matrix_t>::expval_pauli(const reg_t &qubits,
                                             const std::string &pauli) {
  throw std::runtime_error(
      "Unitary simulator does not support Pauli expectation values.");
}

//------------------------------------------------------------------------------
} // namespace QubitUnitary
} // end namespace AER
//------------------------------------------------------------------------------
#endif
