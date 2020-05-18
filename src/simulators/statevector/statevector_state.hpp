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

#ifndef _statevector_state_hpp
#define _statevector_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "simulators/state.hpp"
#include "qubitvector.hpp"
#ifdef AER_THRUST_SUPPORTED
#include "qubitvector_thrust.hpp"
#endif


namespace AER {
namespace Statevector {

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  // Op types
  {Operations::OpType::gate, Operations::OpType::measure,
    Operations::OpType::reset, Operations::OpType::initialize,
    Operations::OpType::snapshot, Operations::OpType::barrier,
    Operations::OpType::bfunc, Operations::OpType::roerror,
    Operations::OpType::matrix, Operations::OpType::diagonal_matrix,
    Operations::OpType::multiplexer, Operations::OpType::kraus},
  // Gates
  {"u1",  "u2",  "u3",   "cx",   "cz",   "cy",   "cu1",
    "cu2", "cu3", "swap", "id",   "x",    "y",    "z",
    "h",   "s",   "sdg",  "t",    "tdg",  "ccx",  "cswap",
    "mcx", "mcy", "mcz",  "mcu1", "mcu2", "mcu3", "mcswap"},
  // Snapshots
  {"statevector", "memory", "register", "probabilities",
    "probabilities_with_variance", "expectation_value_pauli",
    "expectation_value_pauli_with_variance",
    "expectation_value_matrix_single_shot", "expectation_value_matrix",
    "expectation_value_matrix_with_variance",
    "expectation_value_pauli_single_shot"}
);

// Allowed gates enum class
enum class Gates {
  id, h, s, sdg, t, tdg, // single qubit
  // multi-qubit controlled (including single-qubit non-controlled)
  mcx, mcy, mcz, mcu1, mcu2, mcu3, mcswap
};

// Allowed snapshots enum class
enum class Snapshots {
  statevector, cmemory, cregister,
  probs, probs_var,
  expval_pauli, expval_pauli_var, expval_pauli_shot,
  expval_matrix, expval_matrix_var, expval_matrix_shot
};

// Enum class for different types of expectation values
enum class SnapshotDataType {average, average_var, pershot};

//=========================================================================
// QubitVector State subclass
//=========================================================================

template <class statevec_t = QV::QubitVector<double>>
class State : public Base::State<statevec_t> {
public:
  using BaseState = Base::State<statevec_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {return statevec_t::name();}

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exception will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentData &data,
                         RngEngine &rng) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit state
  virtual void initialize_qreg(uint_t num_qubits,
                               const statevec_t &state) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops)
                                    const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  virtual void set_config(const json_t &config) override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t& qubits,
                                            uint_t shots,
                                            RngEngine &rng) override;

  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  // Initializes to a specific n-qubit state given as a complex std::vector
  virtual void initialize_qreg(uint_t num_qubits, const cvector_t &state);

  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp();

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

  // Initialize the specified qubits to a given state |psi>
  // by applying a reset to the these qubits and then
  // computing the tensor product with the new state |psi>
  // /psi> is given in params
  void apply_initialize(const reg_t &qubits, const cvector_t &params, RngEngine &rng);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const Operations::Op &op, ExperimentData &data);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const Operations::Op &op);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t & vmat); 

  // Apply a vector of control matrices to given qubits (identity on all other qubits)
  void apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const std::vector<cmatrix_t> &mmat);

  // Apply stacked (flat) version of multiplexer matrix to target qubits (using control qubits to select matrix instance)
  void apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const cmatrix_t &mat);


  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits,
                   const std::vector<cmatrix_t> &krausops,
                   RngEngine &rng);

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
  std::pair<uint_t, double>
  sample_measure_with_prob(const reg_t &qubits, RngEngine &rng);


  void measure_reset_update(const std::vector<uint_t> &qubits,
                            const uint_t final_state,
                            const uint_t meas_state,
                            const double meas_prob);

  //-----------------------------------------------------------------------
  // Special snapshot types
  //
  // IMPORTANT: These methods are not marked const to allow modifying state
  // during snapshot, but after the snapshot is applied the simulator
  // should be left in the pre-snapshot state.
  //-----------------------------------------------------------------------

  // Snapshot current amplitudes
  void snapshot_statevector(const Operations::Op &op,
                            ExperimentData &data,
                            SnapshotDataType type);

  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(const Operations::Op &op,
                              ExperimentData &data,
                              SnapshotDataType type);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const Operations::Op &op,
                             ExperimentData &data,
                             SnapshotDataType type);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(const Operations::Op &op,
                              ExperimentData &data,
                              SnapshotDataType type);

  //-----------------------------------------------------------------------
  // Single-qubit gate helpers
  //-----------------------------------------------------------------------

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Multi-controlled u3
  //-----------------------------------------------------------------------
  
  // Apply N-qubit multi-controlled single qubit waltz gate specified by
  // parameters u3(theta, phi, lambda)
  // NOTE: if N=1 this is just a regular u3 gate.
  void apply_gate_mcu3(const reg_t& qubits,
                       const double theta,
                       const double phi,
                       const double lambda);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 14;

  // QubitVector sample measure index size
  int sample_measure_index_size_ = 10;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;

  // Table of allowed snapshot types to enum class members
  const static stringmap_t<Snapshots> snapshotset_;

};


//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

template <class statevec_t>
const stringmap_t<Gates> State<statevec_t>::gateset_({
  // Single qubit gates
  {"id", Gates::id},     // Pauli-Identity gate
  {"x", Gates::mcx},     // Pauli-X gate
  {"y", Gates::mcy},     // Pauli-Y gate
  {"z", Gates::mcz},     // Pauli-Z gate
  {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
  {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
  {"t", Gates::t},       // T-gate (sqrt(S))
  {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
  // Waltz Gates
  {"u1", Gates::mcu1},   // zero-X90 pulse waltz gate
  {"u2", Gates::mcu2},   // single-X90 pulse waltz gate
  {"u3", Gates::mcu3},   // two X90 pulse waltz gate
  // Two-qubit gates
  {"cx", Gates::mcx},        // Controlled-X gate (CNOT)
  {"cy", Gates::mcy},        // Controlled-Y gate
  {"cz", Gates::mcz},        // Controlled-Z gate
  {"cu1", Gates::mcu1},      // Controlled-u1 gate
  {"cu2", Gates::mcu2},      // Controlled-u2 gate
  {"cu3", Gates::mcu3},      // Controlled-u3 gate
  {"swap", Gates::mcswap},   // SWAP gate
  // 3-qubit gates
  {"ccx", Gates::mcx},       // Controlled-CX gate (Toffoli)
  {"cswap", Gates::mcswap},  // Controlled SWAP gate (Fredkin)
  // Multi-qubit controlled gates
  {"mcx", Gates::mcx},      // Multi-controlled-X gate
  {"mcy", Gates::mcy},      // Multi-controlled-Y gate
  {"mcz", Gates::mcz},      // Multi-controlled-Z gate
  {"mcu1", Gates::mcu1},    // Multi-controlled-u1
  {"mcu2", Gates::mcu2},    // Multi-controlled-u2
  {"mcu3", Gates::mcu3},    // Multi-controlled-u3
  {"mcswap", Gates::mcswap} // Multi-controlled SWAP gate

});


template <class statevec_t>
const stringmap_t<Snapshots> State<statevec_t>::snapshotset_({
  {"statevector", Snapshots::statevector},
  {"probabilities", Snapshots::probs},
  {"expectation_value_pauli", Snapshots::expval_pauli},
  {"expectation_value_matrix", Snapshots::expval_matrix},
  {"probabilities_with_variance", Snapshots::probs_var},
  {"expectation_value_pauli_with_variance", Snapshots::expval_pauli_var},
  {"expectation_value_matrix_with_variance", Snapshots::expval_matrix_var},
  {"expectation_value_pauli_single_shot", Snapshots::expval_pauli_shot},
  {"expectation_value_matrix_single_shot", Snapshots::expval_matrix_shot},
  {"memory", Snapshots::cmemory},
  {"register", Snapshots::cregister}
});


//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------

template <class statevec_t>
void State<statevec_t>::initialize_qreg(uint_t num_qubits) {
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize();
}

template <class statevec_t>
void State<statevec_t>::initialize_qreg(uint_t num_qubits,
                                   const statevec_t &state) {
  // Check dimension of state
  if (state.num_qubits() != num_qubits) {
    throw std::invalid_argument("QubitVector::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize_from_data(state.data(), 1ULL << num_qubits);
}

template <class statevec_t>
void State<statevec_t>::initialize_qreg(uint_t num_qubits,
                                        const cvector_t &state) {
  if (state.size() != 1ULL << num_qubits) {
    throw std::invalid_argument("QubitVector::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize_from_vector(state);
}

template <class statevec_t>
void State<statevec_t>::initialize_omp() {
  BaseState::qreg_.set_omp_threshold(omp_qubit_threshold_);
  if (BaseState::threads_ > 0)
    BaseState::qreg_.set_omp_threads(BaseState::threads_); // set allowed OMP threads in qubitvector
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

template <class statevec_t>
size_t State<statevec_t>::required_memory_mb(uint_t num_qubits,
                                             const std::vector<Operations::Op> &ops)
                                             const {
  // An n-qubit state vector as 2^n complex doubles
  // where each complex double is 16 bytes
  (void)ops; // avoid unused variable compiler warning
  return BaseState::qreg_.required_memory_mb(num_qubits);
}

template <class statevec_t>
void State<statevec_t>::set_config(const json_t &config) {

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "zero_threshold", config);
  BaseState::qreg_.set_json_chop_threshold(json_chop_threshold_);

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold", config);

  // Set the sample measure indexing size
  int index_size;
  if (JSON::get_value(index_size, "statevector_sample_measure_opt", config)) {
    BaseState::qreg_.set_sample_measure_index_size(index_size);
  };
}


//=========================================================================
// Implementation: apply operations
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_ops(const std::vector<Operations::Op> &ops,
                                 ExperimentData &data,
                                 RngEngine &rng) {

  // Simple loop over vector of input operations
  for (const auto & op: ops) {
    if(BaseState::creg_.check_conditional(op)) {
      switch (op.type) {
        case Operations::OpType::barrier:
          break;
        case Operations::OpType::reset:
          apply_reset(op.qubits, rng);
          break;
        case Operations::OpType::initialize:
          apply_initialize(op.qubits, op.params, rng);
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
        case Operations::OpType::matrix:
          apply_matrix(op);
          break;
        case Operations::OpType::diagonal_matrix:
          BaseState::qreg_.apply_diagonal_matrix(op.qubits, op.params);
          break;
        case Operations::OpType::multiplexer:
          apply_multiplexer(op.regs[0], op.regs[1], op.mats); // control qubits ([0]) & target qubits([1])
          break;
        case Operations::OpType::kraus:
          apply_kraus(op.qubits, op.mats, rng);
          break;
        default:
          throw std::invalid_argument("QubitVector::State::invalid instruction \'" +
                                      op.name + "\'.");
      }
    }
  }
}


//=========================================================================
// Implementation: Snapshots
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_snapshot(const Operations::Op &op,
                                       ExperimentData &data) {

  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument("QubitVectorState::invalid snapshot instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
    case Snapshots::statevector:
      data.add_pershot_snapshot("statevector", op.string_params[0], BaseState::qreg_.vector());
      break;
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(op, data);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(op, data);
      break;
    case Snapshots::probs: {
      // get probs as hexadecimal
      snapshot_probabilities(op, data, SnapshotDataType::average);
    } break;
    case Snapshots::expval_pauli: {
      snapshot_pauli_expval(op, data, SnapshotDataType::average);
    } break;
    case Snapshots::expval_matrix: {
      snapshot_matrix_expval(op, data, SnapshotDataType::average);
    }  break;
    case Snapshots::probs_var: {
      // get probs as hexadecimal
      snapshot_probabilities(op, data, SnapshotDataType::average_var);
    } break;
    case Snapshots::expval_pauli_var: {
      snapshot_pauli_expval(op, data, SnapshotDataType::average_var);
    } break;
    case Snapshots::expval_matrix_var: {
      snapshot_matrix_expval(op, data, SnapshotDataType::average_var);
    }  break;
    case Snapshots::expval_pauli_shot: {
      snapshot_pauli_expval(op, data, SnapshotDataType::pershot);
    } break;
    case Snapshots::expval_matrix_shot: {
      snapshot_matrix_expval(op, data, SnapshotDataType::pershot);
    }  break;
    default:
      // We shouldn't get here unless there is a bug in the snapshotset
      throw std::invalid_argument("QubitVector::State::invalid snapshot instruction \'" +
                                  op.name + "\'.");
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_probabilities(const Operations::Op &op,
                                               ExperimentData &data,
                                               SnapshotDataType type) {
  // get probs as hexadecimal
  auto probs = Utils::vec2ket(measure_probs(op.qubits),
                              json_chop_threshold_, 16);
  bool variance = type == SnapshotDataType::average_var;
  data.add_average_snapshot("probabilities", op.string_params[0],
                            BaseState::creg_.memory_hex(), probs, variance);
}


template <class statevec_t>
void State<statevec_t>::snapshot_pauli_expval(const Operations::Op &op,
                                              ExperimentData &data,
                                              SnapshotDataType type) {
  // Check empty edge case
  if (op.params_expval_pauli.empty()) {
    throw std::invalid_argument("Invalid expval snapshot (Pauli components are empty).");
  }

  // Accumulate expval components
  complex_t expval(0., 0.);
  for (const auto &param : op.params_expval_pauli) {
    const auto& coeff = param.first;
    const auto& pauli = param.second;
    expval += coeff * BaseState::qreg_.expval_pauli(op.qubits, pauli);
  }

  // Add to snapshot
  Utils::chop_inplace(expval, json_chop_threshold_);
  switch (type) {
    case SnapshotDataType::average:
      data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::creg_.memory_hex(), expval, false);
      break;
    case SnapshotDataType::average_var:
      data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::creg_.memory_hex(), expval, true);
      break;
    case SnapshotDataType::pershot:
      data.add_pershot_snapshot("expectation_values", op.string_params[0], expval);
      break;
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_matrix_expval(const Operations::Op &op,
                                               ExperimentData &data,
                                               SnapshotDataType type) {
  // Check empty edge case
  if (op.params_expval_matrix.empty()) {
    throw std::invalid_argument("Invalid matrix snapshot (components are empty).");
  }
  reg_t qubits = op.qubits;
  // Cache the current quantum state
  BaseState::qreg_.checkpoint();
  bool first = true; // flag for first pass so we don't unnecessarily revert from checkpoint

  // Compute expval components
  complex_t expval(0., 0.);
  for (const auto &param : op.params_expval_matrix) {
    complex_t coeff = param.first;
    // Revert the quantum state to cached checkpoint
    if (first)
      first = false;
    else
      BaseState::qreg_.revert(true);
    // Apply each matrix component
    for (const auto &pair: param.second) {
      reg_t sub_qubits;
      for (const auto pos : pair.first) {
        sub_qubits.push_back(qubits[pos]);
      }
      const cmatrix_t &mat = pair.second;
      cvector_t vmat = (mat.GetColumns() == 1)
        ? Utils::vectorize_matrix(Utils::projector(Utils::vectorize_matrix(mat))) // projector case
        : Utils::vectorize_matrix(mat); // diagonal or square matrix case
      if (vmat.size() == 1ULL << qubits.size()) {
        BaseState::qreg_.apply_diagonal_matrix(sub_qubits, vmat);
      } else {
        BaseState::qreg_.apply_matrix(sub_qubits, vmat);
      }

    }
    expval += coeff*BaseState::qreg_.inner_product();
  }
  // add to snapshot
  Utils::chop_inplace(expval, json_chop_threshold_);
  switch (type) {
    case SnapshotDataType::average:
      data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::creg_.memory_hex(), expval, false);
      break;
    case SnapshotDataType::average_var:
      data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::creg_.memory_hex(), expval, true);
      break;
    case SnapshotDataType::pershot:
      data.add_pershot_snapshot("expectation_values", op.string_params[0], expval);
      break;
  }
  // Revert to original state
  BaseState::qreg_.revert(false);
}


//=========================================================================
// Implementation: Matrix multiplication
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument("QubitVectorState::invalid gate instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
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
      apply_gate_mcu3(op.qubits,
                      std::real(op.params[0]),
                      std::real(op.params[1]),
                      std::real(op.params[2]));
      break;
    case Gates::mcu2:
      // Includes u2, cu2, etc
      apply_gate_mcu3(op.qubits,
                      M_PI / 2.,
                      std::real(op.params[0]),
                      std::real(op.params[1]));
      break;
    case Gates::mcu1:
      // Includes u1, cu1, etc
      BaseState::qreg_.apply_mcphase(op.qubits, std::exp(complex_t(0, 1) * op.params[0]));
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument("QubitVector::State::invalid gate instruction \'" +
                                  op.name + "\'.");
  }
}


template <class statevec_t>
void State<statevec_t>::apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const cmatrix_t &mat) {
  if (control_qubits.empty() == false && target_qubits.empty() == false && mat.size() > 0) {
    cvector_t vmat = Utils::vectorize_matrix(mat);
    BaseState::qreg_.apply_multiplexer(control_qubits, target_qubits, vmat);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(const Operations::Op &op) {
  if (op.qubits.empty() == false && op.mats[0].size() > 0) {
    if (Utils::is_diagonal(op.mats[0], .0)) {
      BaseState::qreg_.apply_diagonal_matrix(op.qubits, Utils::matrix_diagonal(op.mats[0]));
    } else {
      BaseState::qreg_.apply_matrix(op.qubits, Utils::vectorize_matrix(op.mats[0]));
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(const reg_t &qubits, const cvector_t &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    BaseState::qreg_.apply_diagonal_matrix(qubits, vmat);
  } else {
    BaseState::qreg_.apply_matrix(qubits, vmat);
  }
}


template <class statevec_t>
void State<statevec_t>::apply_gate_mcu3(const reg_t& qubits,
                                        double theta,
                                        double phi,
                                        double lambda) {
  BaseState::qreg_.apply_mcu(qubits, Utils::VMatrix::u3(theta, phi, lambda));
}

template <class statevec_t>
void State<statevec_t>::apply_gate_phase(uint_t qubit, complex_t phase) {
  cvector_t diag = {{1., phase}};
  apply_matrix(reg_t({qubit}), diag);
}


//=========================================================================
// Implementation: Reset, Initialize and Measurement Sampling
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_measure(const reg_t &qubits,
                                      const reg_t &cmemory,
                                      const reg_t &cregister,
                                      RngEngine &rng) {
  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Implement measurement update
  measure_reset_update(qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());
  BaseState::creg_.store_measure(outcome, cmemory, cregister);
}

template <class statevec_t>
rvector_t State<statevec_t>::measure_probs(const reg_t &qubits) const {
  return BaseState::qreg_.probabilities(qubits);
}

template <class statevec_t>
std::vector<reg_t> State<statevec_t>::sample_measure(const reg_t &qubits,
                                                     uint_t shots,
                                                     RngEngine &rng) {
  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  for (uint_t i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));

  auto allbit_samples = BaseState::qreg_.sample_measure(rnds);

  // Convert to reg_t format
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);
  for (int_t val : allbit_samples) {
    reg_t allbit_sample = Utils::int2reg(val, 2, BaseState::qreg_.num_qubits());
    reg_t sample;
    sample.reserve(qubits.size());
    for (uint_t qubit : qubits) {
      sample.push_back(allbit_sample[qubit]);
    }
    all_samples.push_back(sample);
  }
  return all_samples;
}


template <class statevec_t>
void State<statevec_t>::apply_reset(const reg_t &qubits,
                                    RngEngine &rng) {
  // Simulate unobserved measurement
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Apply update to reset state
  measure_reset_update(qubits, 0, meas.first, meas.second);
}

template <class statevec_t>
std::pair<uint_t, double>
State<statevec_t>::sample_measure_with_prob(const reg_t &qubits,
                                            RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);
  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class statevec_t>
void State<statevec_t>::measure_reset_update(const std::vector<uint_t> &qubits,
                                 const uint_t final_state,
                                 const uint_t meas_state,
                                 const double meas_prob) {
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement final_state

  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    apply_matrix(qubits, mdiag);

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      BaseState::qreg_.apply_mcx(qubits);
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    apply_matrix(qubits, mdiag);

    // If it doesn't agree with the reset state update
    // This function could be optimized as a permutation update
    if (final_state != meas_state) {
      // build vectorized permutation matrix
      cvector_t perm(dim * dim, 0.);
      perm[final_state * dim + meas_state] = 1.;
      perm[meas_state * dim + final_state] = 1.;
      for (size_t j=0; j < dim; j++) {
        if (j != final_state && j != meas_state)
          perm[j * dim + j] = 1.;
      }
      // apply permutation to swap state
      apply_matrix(qubits, perm);
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_initialize(const reg_t &qubits,
                                         const cvector_t &params,
                                         RngEngine &rng) {

   if (qubits.size() == BaseState::qreg_.num_qubits()) {
   // If qubits is all ordered qubits in the statevector
   // we can just initialize the whole state directly
   auto sorted_qubits = qubits;
   std::sort(sorted_qubits.begin(), sorted_qubits.end());
      if (qubits == sorted_qubits) {
        initialize_qreg(qubits.size(), params);
      return;
      }
   }
   // Apply reset to qubits
   apply_reset(qubits, rng);
   // Apply initialize_component
   BaseState::qreg_.initialize_component(qubits, params);
}

//=========================================================================
// Implementation: Multiplexer Circuit
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const std::vector<cmatrix_t> &mmat) {
	// (1) Pack vector of matrices into single (stacked) matrix ... note: matrix dims: rows = DIM[qubit.size()] columns = DIM[|target bits|]
	cmatrix_t multiplexer_matrix = Utils::stacked_matrix(mmat);

	// (2) Treat as single, large(r), chained/batched matrix operator
	apply_multiplexer(control_qubits, target_qubits, multiplexer_matrix);
}


//=========================================================================
// Implementation: Kraus Noise
//=========================================================================
template <class statevec_t>
void State<statevec_t>::apply_kraus(const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats,
                                    RngEngine &rng) {

  // Check edge case for empty Kraus set (this shouldn't happen)
  if (kmats.empty())
    return; // end function early


  // Choose a real in [0, 1) to choose the applied kraus operator once
  // the accumulated probability is greater than r.
  // We know that the Kraus noise must be normalized
  // So we only compute probabilities for the first N-1 kraus operators
  // and infer the probability of the last one from 1 - sum of the previous

  double r = rng.rand(0., 1.);
  double accum = 0.;
  bool complete = false;

  // Loop through N-1 kraus operators
  for (size_t j=0; j < kmats.size() - 1; j++) {

    // Calculate probability
    cvector_t vmat = Utils::vectorize_matrix(kmats[j]);
    double p = BaseState::qreg_.norm(qubits, vmat);
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
    apply_matrix(qubits, Utils::vectorize_matrix(renorm * kmats.back()));
  }
}

//-------------------------------------------------------------------------
} // end namespace QubitVector
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
