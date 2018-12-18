/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _unitary_state_hpp
#define _unitary_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "base/state.hpp"
#include "qubitmatrix.hpp"


namespace AER {
namespace QubitUnitary {

// Allowed gates enum class
enum class Gates {
  u1, u2, u3, id, x, y, z, h, s, sdg, t, tdg, // single qubit
  cx, cz, swap, // two qubit
  ccx // three qubit
};


//=========================================================================
// QubitUnitary State subclass
//=========================================================================

template <class statematrix_t = cmatrix_t>
class State : public Base::State<QM::QubitMatrix<statematrix_t>> {
public:
  using BaseState = Base::State<QM::QubitMatrix<statematrix_t>>;

  State() = default;
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the set of qobj instruction types supported by the State
  inline virtual std::unordered_set<Operations::OpType> allowed_ops() const override {
    return std::unordered_set<Operations::OpType>({
      Operations::OpType::gate,
      Operations::OpType::barrier,
      Operations::OpType::matrix,
      Operations::OpType::snapshot
    });
  }

  // Return the set of qobj gate instruction names supported by the State
  inline virtual stringset_t allowed_gates() const override {
    return {"U", "CX", "u1", "u2", "u3", "cx", "cz", "swap",
            "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "ccx"};
  }

  // Return the set of qobj snapshot types supported by the State
  inline virtual stringset_t allowed_snapshots() const override {
    return {"unitary"};
  }

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exeption will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         OutputData &data,
                         RngEngine &rng) override;

  // Initializes an n-qubit unitary to the identity matrix
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit unitary matrix
  virtual void initialize_qreg(uint_t num_qubits,
                               const QM::QubitMatrix<statematrix_t> &unitary) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << 2 * num_qubits bytes
  virtual uint_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops) override;

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
  virtual void apply_snapshot(const Operations::Op &op, OutputData &data);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cmatrix_t & mat);

  //-----------------------------------------------------------------------
  // 1-Qubit Gates
  //-----------------------------------------------------------------------

  void apply_gate_u3(const uint_t qubit, const double theta, const double phi,
                     const double lambda);

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 6;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-15;

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;
};


//============================================================================
// Implementation: Allowed ops and gateset
//============================================================================

template <class statemat_t>
const stringmap_t<Gates> State<statemat_t>::gateset_({
  // Single qubit gates
  {"id", Gates::id},   // Pauli-Identity gate
  {"x", Gates::x},    // Pauli-X gate
  {"y", Gates::y},    // Pauli-Y gate
  {"z", Gates::z},    // Pauli-Z gate
  {"s", Gates::s},    // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg}, // Conjugate-transpose of Phase gate
  {"h", Gates::h},    // Hadamard gate (X + Z / sqrt(2))
  {"t", Gates::t},    // T-gate (sqrt(S))
  {"tdg", Gates::tdg}, // Conjguate-transpose of T gate
  // Waltz Gates
  {"u1", Gates::u1},  // zero-X90 pulse waltz gate
  {"u2", Gates::u2},  // single-X90 pulse waltz gate
  {"u3", Gates::u3},  // two X90 pulse waltz gate
  {"U", Gates::u3},   // two X90 pulse waltz gate
  // Two-qubit gates
  {"CX", Gates::cx},  // Controlled-X gate (CNOT)
  {"cx", Gates::cx},  // Controlled-X gate (CNOT)
  {"cz", Gates::cz},  // Controlled-Z gate
  {"swap", Gates::swap}, // SWAP gate
  // Three-qubit gates
  {"ccx", Gates::ccx}  // Controlled-CX gate (Toffoli)
});

//============================================================================
// Implementation: Base class method overrides
//============================================================================

template <class statemat_t>
void State<statemat_t>::apply_ops(const std::vector<Operations::Op> &ops,
                                  OutputData &data,
                                  RngEngine &rng) {
  // Simple loop over vector of input operations
  for (const auto op: ops) {
    switch (op.type) {
      case Operations::OpType::barrier:
        break;
      case Operations::OpType::gate:
        // Note conditionals will always fail since no classical registers
        if (BaseState::creg_.check_conditional(op))
          apply_gate(op);
        break;
      case Operations::OpType::snapshot:
        apply_snapshot(op, data);
        break;
      case Operations::OpType::matrix:
        apply_matrix(op.qubits, op.mats[0]);
        break;
      default:
        throw std::invalid_argument("QubitUnitary::State::invalid instruction \'" +
                                    op.name + "\'.");
    }
  }
}

template <class statemat_t>
uint_t State<statemat_t>::required_memory_mb(uint_t num_qubits,
                                 const std::vector<Operations::Op> &ops) {
  // An n-qubit unitary as 2^2n complex doubles
  // where each complex double is 16 bytes
  (void)ops; // avoid unused variable compiler warning
  uint_t shift_mb = std::max<int_t>(0, num_qubits + 4 - 20);
  uint_t mem_mb = 1ULL << (2 * shift_mb);
  return mem_mb;
}


template <class statemat_t>
void State<statemat_t>::set_config(const json_t &config) {
  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "unitary_parallel_threshold", config);

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "chop_threshold", config);
  BaseState::qreg_.set_json_chop_threshold(json_chop_threshold_);
}


template <class statemat_t>
void State<statemat_t>::initialize_qreg(uint_t num_qubits) {
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize();
}


template <class statemat_t>
void State<statemat_t>::initialize_qreg(uint_t num_qubits,
                                        const QM::QubitMatrix<statemat_t> &unitary) {
  // Check dimension of state
  if (unitary.num_qubits() != num_qubits) {
    throw std::invalid_argument("QubitMatrix::State::initialize: initial state does not match qubit number");
  }
  BaseState::qreg_ = unitary;
  initialize_omp();
}


template <class statemat_t>
void State<statemat_t>::initialize_qreg(uint_t num_qubits,
                                        const cmatrix_t &unitary) {
  // Check dimension of unitary
  if (unitary.size() != 1ULL << (2 * num_qubits)) {
    throw std::invalid_argument("QubitMatrix::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize(unitary);
}


template <class statemat_t>
void State<statemat_t>::initialize_omp() {
  BaseState::qreg_.set_omp_threshold(omp_qubit_threshold_);
  if (BaseState::threads_ > 0)
    BaseState::qreg_.set_omp_threads(BaseState::threads_); // set allowed OMP threads in qubitvector
}


//=========================================================================
// Implementation: Gates
//=========================================================================

template <class statemat_t>
void State<statemat_t>::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument("QubitMatrix::State::invalid gate instruction \'" +
                                op.name + "\'.");
  Gates g = it -> second;
  switch (g) {
    case Gates::u3:
      apply_gate_u3(op.qubits[0],
                    std::real(op.params[0]),
                    std::real(op.params[1]),
                    std::real(op.params[2]));
      break;
    case Gates::u2:
      apply_gate_u3(op.qubits[0],
                    M_PI / 2.,
                    std::real(op.params[0]),
                    std::real(op.params[1]));
      break;
    case Gates::u1:
      apply_gate_phase(op.qubits[0], std::exp(complex_t(0., 1.) * op.params[0]));
      break;
    case Gates::cx:
      BaseState::qreg_.apply_cnot(op.qubits[0], op.qubits[1]);
      break;
    case Gates::cz:
      BaseState::qreg_.apply_cz(op.qubits[0], op.qubits[1]);
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
      BaseState::qreg_.apply_z(op.qubits[0]);
      break;
    case Gates::h:
      apply_gate_u3(op.qubits[0], M_PI / 2., 0., M_PI);
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
    case Gates::swap: {
      BaseState::qreg_.apply_swap(op.qubits[0], op.qubits[1]);
    } break;
    case Gates::ccx: {
      BaseState::qreg_.apply_toffoli(op.qubits[0], op.qubits[1], op.qubits[2]);
    } break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument("QubitMatrix::State::invalid gate instruction \'" +
                                  op.name + "\'.");
  }
}


template <class statemat_t>
void State<statemat_t>::apply_matrix(const reg_t &qubits, const cmatrix_t &mat) {
  if (qubits.empty() == false && mat.size() > 0) {
    if (mat.GetRows() == 1){
      BaseState::qreg_.apply_diagonal_matrix(qubits, mat);
    } else {
      BaseState::qreg_.apply_matrix(qubits, mat);
    }
  }
}


template <class statemat_t>
void State<statemat_t>::apply_gate_u3(uint_t qubit, double theta, double phi, double lambda) {
  apply_matrix(reg_t({qubit}), Utils::Matrix::U3(theta, phi, lambda));
}


template <class statemat_t>
void State<statemat_t>::apply_gate_phase(uint_t qubit, complex_t phase) {
  cmatrix_t diag(1, 2);
  diag(0, 0) = 1.0;
  diag(0, 1) = phase;
  apply_matrix(reg_t({qubit}), diag);
}


template <class statemat_t>
void State<statemat_t>::apply_snapshot(const Operations::Op &op,
                                       OutputData &data) {
  // Look for snapshot type in snapshotset
  if (op.name == "unitary" || op.name == "state") {
    BaseState::snapshot_state(op, data);
  } else {
    throw std::invalid_argument("QubitMatrix::State::invalid snapshot instruction \'" +
                                op.name + "\'.");
  }
}

//------------------------------------------------------------------------------
} // end namespace QubitUnitary::
} // end namespace AER
//------------------------------------------------------------------------------
#endif
