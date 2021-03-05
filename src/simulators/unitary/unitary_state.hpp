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
#include <math.h>
#include "simulators/state.hpp"
#include "framework/json.hpp"
#include "framework/utils.hpp"
#include "simulators/state.hpp"
#include "unitarymatrix.hpp"
#ifdef AER_THRUST_SUPPORTED
#include "unitarymatrix_thrust.hpp"
#endif

namespace AER {

//predefinition of QubitUnitary::State for friend class declaration to access static members
namespace QubitUnitaryChunk {
template <class unitary_matrix_t> class State;
}

namespace QubitUnitary {

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {Operations::OpType::gate, Operations::OpType::barrier,
     Operations::OpType::matrix, Operations::OpType::diagonal_matrix,
     Operations::OpType::snapshot, Operations::OpType::save_unitary},
    // Gates
    {"u1",     "u2",      "u3",  "u",    "U",    "CX",   "cx",   "cz",
     "cy",     "cp",      "cu1", "cu2",  "cu3",  "swap", "id",   "p",
     "x",      "y",       "z",   "h",    "s",    "sdg",  "t",    "tdg",
     "r",      "rx",      "ry",  "rz",   "rxx",  "ryy",  "rzz",  "rzx",
     "ccx",    "cswap",   "mcx", "mcy",  "mcz",  "mcu1", "mcu2", "mcu3",
     "mcswap", "mcphase", "mcr", "mcrx", "mcry", "mcry", "sx",   "csx",
     "mcsx",   "delay", "pauli"},
    // Snapshots
    {"unitary"});

// Allowed gates enum class
enum class Gates {
  id, h, s, sdg, t, tdg, rxx, ryy, rzz, rzx,
  mcx, mcy, mcz, mcr, mcrx, mcry, mcrz, mcp, mcu2, mcu3, mcswap, mcsx, pauli,
};

//=========================================================================
// QubitUnitary State subclass
//=========================================================================

template <class unitary_matrix_t = QV::UnitaryMatrix<double>>
class State : public Base::State<unitary_matrix_t> {
  friend class QubitUnitaryChunk::State<unitary_matrix_t>;
public:
  using BaseState = Base::State<unitary_matrix_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return "unitary"; }

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exeption will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentResult &result, RngEngine &rng,
                         bool final_ops = false) override;

  // Initializes an n-qubit unitary to the identity matrix
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit unitary matrix
  virtual void initialize_qreg(uint_t num_qubits,
                               const unitary_matrix_t &unitary) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << 2 * num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // Config: {"omp_qubit_threshold": 7}
  virtual void set_config(const json_t &config) override;

  virtual void allocate(uint_t num_qubits) override;

  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  // Initializes to a specific n-qubit unitary given as a complex matrix
  virtual void initialize_qreg(uint_t num_qubits, const cmatrix_t &unitary);

  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp();

  auto move_to_matrix()
  {
    return BaseState::qreg_.move_to_matrix();
  }
protected:
  //-----------------------------------------------------------------------
  // Apply Instructions
  //-----------------------------------------------------------------------

  // Applies a Gate operation to the state class.
  // This should support all and only the operations defined in
  // allowed_operations.
  void apply_gate(const Operations::Op &op);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const Operations::Op &op, ExperimentResult &result);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cmatrix_t &mat);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t &vmat);

  //-----------------------------------------------------------------------
  // 1-Qubit Gates
  //-----------------------------------------------------------------------

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Multi-controlled u3
  //-----------------------------------------------------------------------

  // Apply N-qubit multi-controlled single qubit waltz gate specified by
  // parameters u3(theta, phi, lambda)
  // NOTE: if N=1 this is just a regular u3 gate.
  void apply_gate_mcu3(const reg_t &qubits, const double theta,
                       const double phi, const double lambda);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the unitary matrix for the simulator
  void apply_save_unitary(const Operations::Op &op,
                          ExperimentResult &result,
                          bool last_op);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string& pauli) override;

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Apply the global phase
  void apply_global_phase();

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
    {"delay", Gates::id},// Delay gate
    {"id", Gates::id},   // Pauli-Identity gate
    {"x", Gates::mcx},   // Pauli-X gate
    {"y", Gates::mcy},   // Pauli-Y gate
    {"z", Gates::mcz},   // Pauli-Z gate
    {"s", Gates::s},     // Phase gate (aka sqrt(Z) gate)
    {"sdg", Gates::sdg}, // Conjugate-transpose of Phase gate
    {"h", Gates::h},     // Hadamard gate (X + Z / sqrt(2))
    {"t", Gates::t},     // T-gate (sqrt(S))
    {"tdg", Gates::tdg}, // Conjguate-transpose of T gate
    {"p", Gates::mcp},   // Parameterized phase gate
    {"sx", Gates::mcsx}, // Sqrt(X) gate
    // 1-qubit rotation Gates
    {"r", Gates::mcr},   // R rotation gate
    {"rx", Gates::mcrx}, // Pauli-X rotation gate
    {"ry", Gates::mcry}, // Pauli-Y rotation gate
    {"rz", Gates::mcrz}, // Pauli-Z rotation gate
    // Waltz Gates
    {"p", Gates::mcp},   // Parameterized phase gate 
    {"u1", Gates::mcp}, // zero-X90 pulse waltz gate
    {"u2", Gates::mcu2}, // single-X90 pulse waltz gate
    {"u3", Gates::mcu3}, // two X90 pulse waltz gate
    {"u", Gates::mcu3}, // two X90 pulse waltz gate
    {"U", Gates::mcu3}, // two X90 pulse waltz gate
    // Two-qubit gates
    {"CX", Gates::mcx},      // Controlled-X gate (CNOT)
    {"cx", Gates::mcx},      // Controlled-X gate (CNOT)
    {"cy", Gates::mcy},      // Controlled-Z gate
    {"cz", Gates::mcz},      // Controlled-Z gate
    {"cp", Gates::mcp},      // Controlled-Phase gate 
    {"cu1", Gates::mcp},    // Controlled-u1 gate
    {"cu2", Gates::mcu2},    // Controlled-u2
    {"cu3", Gates::mcu3},    // Controlled-u3 gate
    {"cp", Gates::mcp},      // Controlled-Phase gate 
    {"swap", Gates::mcswap}, // SWAP gate
    {"rxx", Gates::rxx},     // Pauli-XX rotation gate
    {"ryy", Gates::ryy},     // Pauli-YY rotation gate
    {"rzz", Gates::rzz},     // Pauli-ZZ rotation gate
    {"rzx", Gates::rzx},     // Pauli-ZX rotation gate
    {"csx", Gates::mcsx},    // Controlled-Sqrt(X) gate
    // Three-qubit gates
    {"ccx", Gates::mcx},      // Controlled-CX gate (Toffoli)
    {"cswap", Gates::mcswap}, // Controlled-SWAP gate (Fredkin)
    // Multi-qubit controlled gates
    {"mcx", Gates::mcx},      // Multi-controlled-X gate
    {"mcy", Gates::mcy},      // Multi-controlled-Y gate
    {"mcz", Gates::mcz},      // Multi-controlled-Z gate
    {"mcr", Gates::mcr},      // Multi-controlled R-rotation gate
    {"mcrx", Gates::mcrx},    // Multi-controlled X-rotation gate
    {"mcry", Gates::mcry},    // Multi-controlled Y-rotation gate
    {"mcrz", Gates::mcrz},    // Multi-controlled Z-rotation gate
    {"mcphase", Gates::mcp},  // Multi-controlled-Phase gate 
    {"mcu1", Gates::mcp},    // Multi-controlled-u1
    {"mcu2", Gates::mcu2},    // Multi-controlled-u2
    {"mcu3", Gates::mcu3},    // Multi-controlled-u3
    {"mcphase", Gates::mcp},  // Multi-controlled-Phase gate 
    {"mcswap", Gates::mcswap},// Multi-controlled SWAP gate
    {"mcsx", Gates::mcsx},    // Multi-controlled-Sqrt(X) gate
    {"pauli", Gates::pauli}  // Multiple pauli operations at once
});

template <class unitary_matrix_t>
void State<unitary_matrix_t>::allocate(uint_t num_qubits)
{
  BaseState::qreg_.chunk_setup(num_qubits*2,num_qubits*2,0,1);
}

//============================================================================
// Implementation: Base class method overrides
//============================================================================

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_ops(
    const std::vector<Operations::Op> &ops, ExperimentResult &result,
    RngEngine &rng, bool final_ops) {
  // Simple loop over vector of input operations
  for (size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];
    switch (op.type) {
      case Operations::OpType::barrier:
        break;
      case Operations::OpType::gate:
        // Note conditionals will always fail since no classical registers
        if (BaseState::creg_.check_conditional(op))
          apply_gate(op);
        break;
      case Operations::OpType::save_unitary:
        apply_save_unitary(op, result, final_ops && ops.size() == i + 1);
        break;
      case Operations::OpType::snapshot:
        apply_snapshot(op, result);
        break;
      case Operations::OpType::matrix:
        apply_matrix(op.qubits, op.mats[0]);
        break;
      case Operations::OpType::diagonal_matrix:
        BaseState::qreg_.apply_diagonal_matrix(op.qubits, op.params);
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
  // An n-qubit unitary as 2^2n complex doubles
  // where each complex double is 16 bytes
  (void)ops; // avoid unused variable compiler warning
  size_t shift_mb = std::max<int_t>(0, num_qubits + 4 - 20);
  size_t mem_mb = 1ULL << (2 * shift_mb);
  return mem_mb;
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::set_config(const json_t &config) {
  BaseState::set_config(config);

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "unitary_parallel_threshold", config);

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "zero_threshold", config);
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
                                              const unitary_matrix_t &unitary) {
  // Check dimension of state
  if (unitary.num_qubits() != num_qubits) {
    throw std::invalid_argument(
        "Unitary::State::initialize: initial state does not match qubit "
        "number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  const size_t sz = 1ULL << BaseState::qreg_.size();
  BaseState::qreg_.initialize_from_data(unitary.data(), sz);
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

//=========================================================================
// Implementation: Gates
//=========================================================================

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate(const Operations::Op &op) {
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
      BaseState::qreg_.apply_mcu(op.qubits, Linalg::VMatrix::r(op.params[0], op.params[1]));
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
      BaseState::qreg_.apply_matrix(op.qubits, Linalg::VMatrix::rxx(op.params[0]));
      break;
    case Gates::ryy:
      BaseState::qreg_.apply_matrix(op.qubits, Linalg::VMatrix::ryy(op.params[0]));
      break;
    case Gates::rzz:
      BaseState::qreg_.apply_diagonal_matrix(op.qubits, Linalg::VMatrix::rzz_diag(op.params[0]));
      break;
    case Gates::rzx:
      BaseState::qreg_.apply_matrix(op.qubits, Linalg::VMatrix::rzx(op.params[0]));
      break;
    case Gates::id:
      break;
    case Gates::h:
      apply_gate_mcu3(op.qubits, M_PI / 2., 0., M_PI);
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
      apply_gate_mcu3(op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                      std::real(op.params[2]));
      break;
    case Gates::mcu2:
      // Includes u2, cu2, etc
      apply_gate_mcu3(op.qubits, M_PI / 2., std::real(op.params[0]),
                      std::real(op.params[1]));
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
    BaseState::qreg_.apply_diagonal_matrix(qubits, vmat);
  } else {
    BaseState::qreg_.apply_matrix(qubits, vmat);
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_phase(uint_t qubit, complex_t phase) {
  cmatrix_t diag(1, 2);
  diag(0, 0) = 1.0;
  diag(0, 1) = phase;
  apply_matrix(reg_t({qubit}), diag);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_mcu3(const reg_t &qubits, double theta,
                                              double phi, double lambda) {
  const auto u3 = Linalg::Matrix::u3(theta, phi, lambda);
  BaseState::qreg_.apply_mcu(qubits, Utils::vectorize_matrix(u3));
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_snapshot(const Operations::Op &op,
                                             ExperimentResult &result) {
  // Look for snapshot type in snapshotset
  if (op.name == "unitary" || op.name == "state") {
    result.legacy_data.add_pershot_snapshot("unitary", op.string_params[0],
                              BaseState::qreg_.copy_to_matrix());
    BaseState::snapshot_state(op, result);
  } else {
    throw std::invalid_argument(
        "Unitary::State::invalid snapshot instruction \'" + op.name + "\'.");
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_global_phase() {
  if (BaseState::has_global_phase_) {
    BaseState::qreg_.apply_diagonal_matrix(
      {0}, {BaseState::global_phase_, BaseState::global_phase_}
    );
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_save_unitary(const Operations::Op &op,
                                                 ExperimentResult &result,
                                                 bool last_op) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(
        op.name + " was not applied to all qubits."
        " Only the full unitary can be saved.");
  }
  if (last_op) {
    BaseState::save_data_pershot(result, op.string_params[0],
                                 BaseState::qreg_.move_to_matrix(),
                                 op.save_type);
  } else {
    BaseState::save_data_pershot(result, op.string_params[0],
                                 BaseState::qreg_.copy_to_matrix(),
                                 op.save_type);
  }
}

template <class unitary_matrix_t>
double  State<unitary_matrix_t>::expval_pauli(const reg_t &qubits,
                                              const std::string& pauli) {
  throw std::runtime_error("Unitary simulator does not support Pauli expectation values.");
}

//------------------------------------------------------------------------------
} // namespace QubitUnitary
} // end namespace AER
//------------------------------------------------------------------------------
#endif
