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
#include "unitarymatrix.hpp"
#ifdef AER_THRUST_SUPPORTED
#include "unitarymatrix_thrust.hpp"
#endif

namespace AER {
namespace QubitUnitary {

// Allowed gates enum class
enum class Gates {
  id,
  h,
  s,
  sdg,
  t,
  tdg,  // single qubit
  // multi-qubit controlled (including single-qubit non-controlled)
  mcx,
  mcy,
  mcz,
  mcu1,
  mcu2,
  mcu3,
  mcswap
};

//=========================================================================
// QubitUnitary State subclass
//=========================================================================

template <class unitary_matrix_t = QV::UnitaryMatrix<double>>
class State : public Base::State<unitary_matrix_t> {
 public:
  using BaseState = Base::State<unitary_matrix_t>;

  State() = default;
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return "unitary"; }

  // Return the set of qobj instruction types supported by the State
  virtual Operations::OpSet::optypeset_t allowed_ops() const override {
    return Operations::OpSet::optypeset_t(
        {Operations::OpType::gate, Operations::OpType::barrier,
         Operations::OpType::matrix, Operations::OpType::snapshot});
  }

  // Return the set of qobj gate instruction names supported by the State
  virtual stringset_t allowed_gates() const override {
    return {"u1",  "u2",  "u3",   "cx",   "cz",   "cy",   "cu1",
            "cu2", "cu3", "swap", "id",   "x",    "y",    "z",
            "h",   "s",   "sdg",  "t",    "tdg",  "ccx",  "cswap",
            "mcx", "mcy", "mcz",  "mcu1", "mcu2", "mcu3", "mcswap"};
  }

  // Return the set of qobj snapshot types supported by the State
  virtual stringset_t allowed_snapshots() const override { return {"unitary"}; }

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exeption will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentData &data, RngEngine &rng) override;

  // Initializes an n-qubit unitary to the identity matrix
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit unitary matrix
  virtual void initialize_qreg(uint_t num_qubits,
                               const unitary_matrix_t &unitary) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << 2 * num_qubits bytes
  virtual size_t required_memory_mb(
      uint_t num_qubits, const std::vector<Operations::Op> &ops) const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // Config: {"omp_qubit_threshold": 7}
  virtual void set_config(const json_t &config) override;

  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  // Initializes to a specific n-qubit unitary given as a complex matrix
  virtual void initialize_qreg(uint_t num_qubits, const cmatrix_t &unitary);

  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp();

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
  virtual void apply_snapshot(const Operations::Op &op, ExperimentData &data);

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
    {"id", Gates::id},    // Pauli-Identity gate
    {"x", Gates::mcx},    // Pauli-X gate
    {"y", Gates::mcy},    // Pauli-Y gate
    {"z", Gates::mcz},    // Pauli-Z gate
    {"s", Gates::s},      // Phase gate (aka sqrt(Z) gate)
    {"sdg", Gates::sdg},  // Conjugate-transpose of Phase gate
    {"h", Gates::h},      // Hadamard gate (X + Z / sqrt(2))
    {"t", Gates::t},      // T-gate (sqrt(S))
    {"tdg", Gates::tdg},  // Conjguate-transpose of T gate
    // Waltz Gates
    {"u1", Gates::mcu1},  // zero-X90 pulse waltz gate
    {"u2", Gates::mcu2},  // single-X90 pulse waltz gate
    {"u3", Gates::mcu3},  // two X90 pulse waltz gate
    // Two-qubit gates
    {"cx", Gates::mcx},       // Controlled-X gate (CNOT)
    {"cy", Gates::mcy},       // Controlled-Z gate
    {"cz", Gates::mcz},       // Controlled-Z gate
    {"cu1", Gates::mcu1},     // Controlled-u1 gate
    {"cu2", Gates::mcu2},     // Controlled-u2
    {"cu3", Gates::mcu3},     // Controlled-u3 gate
    {"swap", Gates::mcswap},  // SWAP gate
    // Three-qubit gates
    {"ccx", Gates::mcx},       // Controlled-CX gate (Toffoli)
    {"cswap", Gates::mcswap},  // Controlled-SWAP gate (Fredkin)
    // Multi-qubit controlled gates
    {"mcx", Gates::mcx},       // Multi-controlled-X gate
    {"mcy", Gates::mcy},       // Multi-controlled-Y gate
    {"mcz", Gates::mcz},       // Multi-controlled-Z gate
    {"mcu1", Gates::mcu1},     // Multi-controlled-u1
    {"mcu2", Gates::mcu2},     // Multi-controlled-u2
    {"mcu3", Gates::mcu3},     // Multi-controlled-u3
    {"mcswap", Gates::mcswap}  // Multi-controlled-SWAP gate
});

//============================================================================
// Implementation: Base class method overrides
//============================================================================

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_ops(
    const std::vector<Operations::Op> &ops, ExperimentData &data,
    RngEngine &rng) {
  // Simple loop over vector of input operations
  for (const auto op : ops) {
    switch (op.type) {
      case Operations::OpType::barrier:
        break;
      case Operations::OpType::gate:
        // Note conditionals will always fail since no classical registers
        if (BaseState::creg_.check_conditional(op)) apply_gate(op);
        break;
      case Operations::OpType::snapshot:
        apply_snapshot(op, data);
        break;
      case Operations::OpType::matrix:
        apply_matrix(op.qubits, op.mats[0]);
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
  (void)ops;  // avoid unused variable compiler warning
  size_t shift_mb = std::max<int_t>(0, num_qubits + 4 - 20);
  size_t mem_mb = 1ULL << (2 * shift_mb);
  return mem_mb;
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::set_config(const json_t &config) {
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
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_qreg(
    uint_t num_qubits, const unitary_matrix_t &unitary) {
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
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_qreg(
    uint_t num_qubits, const cmatrix_t &unitary) {
  // Check dimension of unitary
  if (unitary.size() != 1ULL << (2 * num_qubits)) {
    throw std::invalid_argument(
        "Unitary::State::initialize: initial state does not match qubit "
        "number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize_from_matrix(unitary);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_omp() {
  BaseState::qreg_.set_omp_threshold(omp_qubit_threshold_);
  if (BaseState::threads_ > 0)
    BaseState::qreg_.set_omp_threads(
        BaseState::threads_);  // set allowed OMP threads in qubitvector
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
      apply_gate_mcu3(op.qubits, std::real(op.params[0]),
                      std::real(op.params[1]), std::real(op.params[2]));
      break;
    case Gates::mcu2:
      // Includes u2, cu2, etc
      apply_gate_mcu3(op.qubits, M_PI / 2., std::real(op.params[0]),
                      std::real(op.params[1]));
      break;
    case Gates::mcu1:
      // Includes u1, cu1, etc
      BaseState::qreg_.apply_mcphase(op.qubits,
                                     std::exp(complex_t(0, 1) * op.params[0]));
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument(
          "Unitary::State::invalid gate instruction \'" + op.name + "\'.");
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
void State<unitary_matrix_t>::apply_gate_phase(uint_t qubit,
                                                       complex_t phase) {
  cmatrix_t diag(1, 2);
  diag(0, 0) = 1.0;
  diag(0, 1) = phase;
  apply_matrix(reg_t({qubit}), diag);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_mcu3(const reg_t &qubits,
                                                      double theta, double phi,
                                                      double lambda) {
  const auto u3 = Utils::Matrix::u3(theta, phi, lambda);
  BaseState::qreg_.apply_mcu(qubits, Utils::vectorize_matrix(u3));
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_snapshot(const Operations::Op &op,
                                                     ExperimentData &data) {
  // Look for snapshot type in snapshotset
  if (op.name == "unitary" || op.name == "state") {
    data.add_pershot_snapshot("unitary", op.string_params[0],
                              BaseState::qreg_.matrix());
    BaseState::snapshot_state(op, data);
  } else {
    throw std::invalid_argument(
        "Unitary::State::invalid snapshot instruction \'" + op.name + "\'.");
  }
}

//------------------------------------------------------------------------------
}  // namespace QubitUnitary
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
