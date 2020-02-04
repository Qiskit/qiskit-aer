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

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "simulators/state.hpp"
#include "superoperator.hpp"


namespace AER {
namespace QubitSuperoperator {

// Allowed gates enum class
enum class Gates {
  u1, u2, u3, id, x, y, z, h, s, sdg, t, tdg, // single qubit
  cx, cz, swap, // two qubit
  ccx // three qubit
};

// Allowed snapshots enum class
enum class Snapshots {superop};

//=========================================================================
// QubitUnitary State subclass
//=========================================================================

template <class data_t = QV::Superoperator<double>>
class State : public Base::State<data_t> {
public:
  using BaseState = Base::State<data_t>;

  State() = default;
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {return "superoperator";}

  // Return the set of qobj instruction types supported by the State
  virtual Operations::OpSet::optypeset_t allowed_ops() const override {
    return Operations::OpSet::optypeset_t({
      Operations::OpType::gate,
      Operations::OpType::reset,
      Operations::OpType::snapshot,
      Operations::OpType::barrier,
      Operations::OpType::matrix,
      Operations::OpType::kraus,
      Operations::OpType::superop
    });
  }

  // Return the set of qobj gate instruction names supported by the State
  virtual stringset_t allowed_gates() const override {
    return {"U", "CX", "u1", "u2", "u3", "cx", "cz", "swap",
            "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "ccx"};
  }

  // Return the set of qobj snapshot types supported by the State
  virtual stringset_t allowed_snapshots() const override {
    return {"superoperator"};
  }

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exeption will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentData &data,
                         RngEngine &rng) override;

  // Initializes an n-qubit unitary to the identity matrix
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit unitary superop
  virtual void initialize_qreg(uint_t num_qubits,
                               const data_t &unitary) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << 4 * num_qubits bytes
  virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops)
                                    const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // Config: {"omp_qubit_threshold": 3}
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
  void apply_matrix(const reg_t &qubits, const cmatrix_t & mat);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t & vmat);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits,
                   const std::vector<cmatrix_t> &krausops);

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
  void apply_gate_u3(const uint_t qubit,
                     const double theta,
                     const double phi,
                     const double lambda);

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
  {"id", Gates::id},     // Pauli-Identity gate
  {"x", Gates::x},       // Pauli-X gate
  {"y", Gates::y},       // Pauli-Y gate
  {"z", Gates::z},       // Pauli-Z gate
  {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
  {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
  {"t", Gates::t},       // T-gate (sqrt(S))
  {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
  // Waltz Gates
  {"u1", Gates::u1},     // zero-X90 pulse waltz gate
  {"u2", Gates::u2},     // single-X90 pulse waltz gate
  {"u3", Gates::u3},     // two X90 pulse waltz gate
  {"U", Gates::u3},      // two X90 pulse waltz gate
  // Two-qubit gates
  {"CX", Gates::cx},     // Controlled-X gate (CNOT)
  {"cx", Gates::cx},     // Controlled-X gate (CNOT)
  {"cz", Gates::cz},     // Controlled-Z gate
  {"swap", Gates::swap}, // SWAP gate
  // Three-qubit gates
  {"ccx", Gates::ccx}    // Controlled-CX gate (Toffoli)
});

//============================================================================
// Implementation: Base class method overrides
//============================================================================

template <class data_t>
void State<data_t>::apply_ops(const std::vector<Operations::Op> &ops,
                                  ExperimentData &data,
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
      case Operations::OpType::reset:
        apply_reset(op.qubits);
        break;
      case Operations::OpType::matrix:
        apply_matrix(op.qubits, op.mats[0]);
        break;
      case Operations::OpType::kraus:
        apply_kraus(op.qubits, op.mats);
        break;
      case Operations::OpType::superop:
        BaseState::qreg_.apply_superop_matrix(op.qubits, Utils::vectorize_matrix(op.mats[0]));
        break;
      case Operations::OpType::snapshot:
        apply_snapshot(op, data);
        break;
      default:
        throw std::invalid_argument("QubitSuperoperator::State::invalid instruction \'" +
                                    op.name + "\'.");
    }
  }
}

template <class data_t>
size_t State<data_t>::required_memory_mb(uint_t num_qubits,
                                 const std::vector<Operations::Op> &ops)
                                 const {
  // An n-qubit unitary as 2^4n complex doubles
  // where each complex double is 16 bytes
  (void)ops; // avoid unused variable compiler warning
  size_t shift_mb = std::max<int_t>(0, num_qubits + 4 - 20);
  size_t mem_mb = 1ULL << (4 * shift_mb);
  return mem_mb;
}


template <class data_t>
void State<data_t>::set_config(const json_t &config) {
  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "superoperator_parallel_threshold", config);

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "zero_threshold", config);
  BaseState::qreg_.set_json_chop_threshold(json_chop_threshold_);
}


template <class data_t>
void State<data_t>::initialize_qreg(uint_t num_qubits) {
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize();
}


template <class data_t>
void State<data_t>::initialize_qreg(uint_t num_qubits,
                                    const data_t &supermat) {
  // Check dimension of state
  if (supermat.num_qubits() != num_qubits) {
    throw std::invalid_argument("QubitSuperoperator::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  const size_t sz = 1ULL << BaseState::qreg_.size();
  BaseState::qreg_.initialize_from_data(supermat.data(), sz);
}


template <class data_t>
void State<data_t>::initialize_qreg(uint_t num_qubits,
                                    const cmatrix_t &mat) {
  
  // Check dimension of unitary
  const auto sz_uni = 1ULL << (2 * num_qubits);
  const auto sz_super = 1ULL << (4 * num_qubits);
  if (mat.size() != sz_uni && mat.size() != sz_super) {
    throw std::invalid_argument(
      "QubitSuperoperator::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize_from_matrix(mat);
}


template <class data_t>
void State<data_t>::initialize_omp() {
  BaseState::qreg_.set_omp_threshold(omp_qubit_threshold_);
  if (BaseState::threads_ > 0)
    BaseState::qreg_.set_omp_threads(BaseState::threads_); // set allowed OMP threads in qubitvector
}

//=========================================================================
// Implementation: Reset
//=========================================================================

template <class data_t>
void State<data_t>::apply_reset(const reg_t &qubits) {
  // TODO: This can be more efficient by adding reset
  // to base class rather than doing a matrix multiplication
  // where all but 1 row is zeros.
  const auto reset_op = Utils::SMatrix::reset(1ULL << qubits.size());
  BaseState::qreg_.apply_superop_matrix(qubits, Utils::vectorize_matrix(reset_op));
}

//=========================================================================
// Implementation: Kraus Noise
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_kraus(const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats) {
  // Convert to Superoperator
  const auto nrows = kmats[0].GetRows();
  cmatrix_t superop(nrows * nrows, nrows * nrows);
  for (const auto kraus : kmats) {
    superop += Utils::tensor_product(Utils::conjugate(kraus), kraus);
  }
  BaseState::qreg_.apply_superop_matrix(qubits, Utils::vectorize_matrix(superop));
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
  switch (it -> second) {
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
    case Gates::ccx:
      BaseState::qreg_.apply_toffoli(op.qubits[0], op.qubits[1], op.qubits[2]);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument("Superoperator::State::invalid gate instruction \'" +
                                  op.name + "\'.");
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


template <class data_t>
void State<data_t>::apply_gate_phase(uint_t qubit, complex_t phase) {
  cvector_t diag(2);
  diag[0] = 1.0;
  diag[1] = phase;
  BaseState::qreg_.apply_diagonal_unitary_matrix(reg_t({qubit}), diag);
}


template <class statevec_t>
void State<statevec_t>::apply_gate_u3(const uint_t qubit,
                                      double theta,
                                      double phi,
                                      double lambda) {
  const auto u3 = Utils::VMatrix::u3(theta, phi, lambda);
  BaseState::qreg_.apply_unitary_matrix(reg_t({qubit}), u3);
}


template <class data_t>
void State<data_t>::apply_snapshot(const Operations::Op &op,
                                   ExperimentData &data) {
  // Look for snapshot type in snapshotset
  if (op.name == "superopertor" || op.name == "state") {
    BaseState::snapshot_state(op, data, "superoperator");
  } else {
    throw std::invalid_argument("QubitSuperoperator::State::invalid snapshot instruction \'" +
                                op.name + "\'.");
  }
}

//------------------------------------------------------------------------------
} // end namespace QubitSuperoperator
} // end namespace AER
//------------------------------------------------------------------------------
#endif
