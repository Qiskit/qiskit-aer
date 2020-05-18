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

#ifndef _aer_densitymatrix_state_hpp
#define _aer_densitymatrix_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "simulators/state.hpp"
#include "densitymatrix.hpp"
#ifdef AER_THRUST_SUPPORTED
#include "densitymatrix_thrust.hpp"
#endif


namespace AER {
namespace DensityMatrix {

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  // Op types
  {Operations::OpType::gate, Operations::OpType::measure,
    Operations::OpType::reset, Operations::OpType::snapshot,
    Operations::OpType::barrier, Operations::OpType::bfunc,
    Operations::OpType::roerror, Operations::OpType::matrix,
    Operations::OpType::diagonal_matrix, Operations::OpType::kraus,
    Operations::OpType::superop},
  // Gates
  {"U", "CX", "u1", "u2", "u3", "cx", "cz", "swap", "id", "x", "y",
    "z", "h", "s", "sdg", "t", "tdg", "ccx"},
  // Snapshots
  {"density_matrix", "memory", "register", "probabilities",
    "probabilities_with_variance", "expectation_value_pauli",
    "expectation_value_pauli_with_variance"}
);

// Allowed gates enum class
enum class Gates {
  u1, u2, u3, id, x, y, z, h, s, sdg, t, tdg, // single qubit
  cx, cz, swap, // two qubit
  ccx // three qubit
};

// Allowed snapshots enum class
enum class Snapshots {
  cmemory, cregister, densitymatrix,
  probs, probs_var,
  expval_pauli, expval_pauli_var
  /* TODO: The following expectation value snapshots still need to be implemented */
  //,expval_matrix, expval_matrix_var
};

//=========================================================================
// DensityMatrix State subclass
//=========================================================================

template <class densmat_t = QV::DensityMatrix<double>>
class State : public Base::State<densmat_t> {
public:
  using BaseState = Base::State<densmat_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {return densmat_t::name();}

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exeption will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentData &data,
                         RngEngine &rng) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit state
  virtual void initialize_qreg(uint_t num_qubits,
                               const densmat_t &state) override;

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

  // Initializes to a specific n-qubit state given as a complex matrix
  virtual void initialize_qreg(uint_t num_qubits, const cmatrix_t &state);

  // Initialize OpenMP settings for the underlying DensityMatrix class
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

  // Reset the specified qubits to the |0> state by tracing out qubits
  void apply_reset(const reg_t &qubits);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const Operations::Op &op, ExperimentData &data);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cmatrix_t & mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t & vmat);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &kraus);

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

  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(const Operations::Op &op,
                              ExperimentData &data,
                              bool variance);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const Operations::Op &op,
                             ExperimentData &data,
                             bool variance);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(const Operations::Op &op,
                              ExperimentData &data,
                              bool variance);

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

  // OpenMP qubit threshold
  // NOTE: This is twice the number of qubits in the DensityMatrix since it
  // refers to the equivalent qubit number in the underlying QubitVector class
  int omp_qubit_threshold_ = 14;

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

template <class densmat_t>
const stringmap_t<Gates> State<densmat_t>::gateset_({
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


template <class densmat_t>
const stringmap_t<Snapshots> State<densmat_t>::snapshotset_({
  {"density_matrix", Snapshots::densitymatrix},
  {"probabilities", Snapshots::probs},
  {"probabilities_with_variance", Snapshots::probs_var},
  {"memory", Snapshots::cmemory},
  {"register", Snapshots::cregister},
  {"expectation_value_pauli", Snapshots::expval_pauli},
  {"expectation_value_pauli_with_variance", Snapshots::expval_pauli_var}
});


//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits) {
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize();
}

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits,
                                   const densmat_t &state) {
  // Check dimension of state
  if (state.num_qubits() != num_qubits) {
    throw std::invalid_argument("DensityMatrix::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize_from_data(state.data(), 1ULL << 2 * num_qubits);
}

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits,
                                        const cmatrix_t &state) {
  if (state.size() != 1ULL << 2 * num_qubits) {
    throw std::invalid_argument("DensityMatrix::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize_from_matrix(state);
}

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits,
                                        const cvector_t &state) {
  if (state.size() != 1ULL << 2 * num_qubits) {
    throw std::invalid_argument("DensityMatrix::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();
  BaseState::qreg_.set_num_qubits(num_qubits);
  BaseState::qreg_.initialize_from_vector(state);
}

template <class densmat_t>
void State<densmat_t>::initialize_omp() {
  BaseState::qreg_.set_omp_threshold(omp_qubit_threshold_);
  if (BaseState::threads_ > 0)
    BaseState::qreg_.set_omp_threads(BaseState::threads_); // set allowed OMP threads in qubitvector
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

template <class densmat_t>
size_t State<densmat_t>::required_memory_mb(uint_t num_qubits,
                                            const std::vector<Operations::Op> &ops)
                                            const {
  // An n-qubit state vector as 2^n complex doubles
  // where each complex double is 16 bytes
  (void)ops; // avoid unused variable compiler warning
  size_t shift_mb = std::max<int_t>(0, num_qubits + 4 - 20);
  size_t mem_mb = 1ULL << shift_mb;
  return mem_mb;
}

template <class densmat_t>
void State<densmat_t>::set_config(const json_t &config) {

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "chop_threshold", config);
  BaseState::qreg_.set_json_chop_threshold(json_chop_threshold_);

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold", config);
}


//=========================================================================
// Implementation: apply operations
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_ops(const std::vector<Operations::Op> &ops,
                                 ExperimentData &data,
                                 RngEngine &rng) {
  // Simple loop over vector of input operations
  for (const auto op: ops) {
    // If conditional op check conditional
    if (BaseState::creg_.check_conditional(op) == false)
      return;
    switch (op.type) {
      case Operations::OpType::barrier:
        break;
      case Operations::OpType::reset:
        apply_reset(op.qubits);
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
        apply_matrix(op.qubits, op.mats[0]);
        break;
      case Operations::OpType::diagonal_matrix:
        BaseState::qreg_.apply_diagonal_matrix(op.qubits, op.params);
        break;
      case Operations::OpType::superop:
        BaseState::qreg_.apply_superop_matrix(op.qubits, Utils::vectorize_matrix(op.mats[0]));
        break;
      case Operations::OpType::kraus:
        apply_kraus(op.qubits, op.mats);
        break;
      default:
        throw std::invalid_argument("DensityMatrix::State::invalid instruction \'" +
                                    op.name + "\'.");
    }
  }
}


//=========================================================================
// Implementation: Snapshots
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_snapshot(const Operations::Op &op,
                                       ExperimentData &data) {

  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument("DensityMatrixState::invalid snapshot instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
    case Snapshots::densitymatrix:
      data.add_average_snapshot("density_matrix",
                                op.string_params[0],
                                BaseState::creg_.memory_hex(),
                                BaseState::qreg_.matrix(),
                                false);
      break;
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(op, data);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(op, data);
      break;
    case Snapshots::probs:
      // get probs as hexadecimal
      snapshot_probabilities(op, data, false);
      break;
    case Snapshots::probs_var:
      // get probs as hexadecimal
      snapshot_probabilities(op, data, true);
      break;
    case Snapshots::expval_pauli: {
      snapshot_pauli_expval(op, data, false);
    } break;
    case Snapshots::expval_pauli_var: {
      snapshot_pauli_expval(op, data, true);
    } break;
    /* TODO
    case Snapshots::expval_matrix: {
      snapshot_matrix_expval(op, data, false);
    }  break;
    case Snapshots::expval_matrix_var: {
      snapshot_matrix_expval(op, data, true);
    }  break;
    */
    default:
      // We shouldn't get here unless there is a bug in the snapshotset
      throw std::invalid_argument("DensityMatrix::State::invalid snapshot instruction \'" +
                                  op.name + "\'.");
  }
}

template <class densmat_t>
void State<densmat_t>::snapshot_probabilities(const Operations::Op &op,
                                              ExperimentData &data,
                                              bool variance) {
  // get probs as hexadecimal
  auto probs = Utils::vec2ket(measure_probs(op.qubits),
                              json_chop_threshold_, 16);
  data.add_average_snapshot("probabilities",
                            op.string_params[0],
                            BaseState::creg_.memory_hex(),
                            probs,
                            variance);
}


template <class densmat_t>
void State<densmat_t>::snapshot_pauli_expval(const Operations::Op &op,
                                             ExperimentData &data,
                                             bool variance) {
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
  data.add_average_snapshot("expectation_value",
                            op.string_params[0],
                            BaseState::creg_.memory_hex(),
                            expval, variance);
}

//=========================================================================
// Implementation: Matrix multiplication
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument("DensityMatrixState::invalid gate instruction \'" + 
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
      throw std::invalid_argument("DensityMatrix::State::invalid gate instruction \'" +
                                  op.name + "\'.");
  }
}


template <class densmat_t>
void State<densmat_t>::apply_matrix(const reg_t &qubits, const cmatrix_t &mat) {
  if (mat.GetRows() == 1) {
    BaseState::qreg_.apply_diagonal_unitary_matrix(qubits, Utils::vectorize_matrix(mat));
  } else {
    BaseState::qreg_.apply_unitary_matrix(qubits, Utils::vectorize_matrix(mat));
  }
}

template <class densmat_t>
void State<densmat_t>::apply_gate_u3(uint_t qubit, double theta, double phi, double lambda) {
  BaseState::qreg_.apply_unitary_matrix(reg_t({qubit}), Utils::VMatrix::u3(theta, phi, lambda));
}

template <class densmat_t>
void State<densmat_t>::apply_gate_phase(uint_t qubit, complex_t phase) {
  cvector_t diag = {{1., phase}};
  BaseState::qreg_.apply_diagonal_unitary_matrix(reg_t({qubit}), diag);
}


//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_measure(const reg_t &qubits,
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

template <class densmat_t>
rvector_t State<densmat_t>::measure_probs(const reg_t &qubits) const {
  return BaseState::qreg_.probabilities(qubits);
}

template <class densmat_t>
std::vector<reg_t> State<densmat_t>::sample_measure(const reg_t &qubits,
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


template <class densmat_t>
void State<densmat_t>::apply_reset(const reg_t &qubits) {
  // TODO: This can be more efficient by adding reset
  // to base class rather than doing a matrix multiplication
  // where all but 1 row is zeros.
  const auto reset_op = Utils::SMatrix::reset(1ULL << qubits.size());
  BaseState::qreg_.apply_superop_matrix(qubits, Utils::vectorize_matrix(reset_op));
}

template <class densmat_t>
std::pair<uint_t, double>
State<densmat_t>::sample_measure_with_prob(const reg_t &qubits,
                                            RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);
  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class densmat_t>
void State<densmat_t>::measure_reset_update(const reg_t &qubits,
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
    BaseState::qreg_.apply_diagonal_unitary_matrix(qubits, mdiag);

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      BaseState::qreg_.apply_x(qubits[0]);
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    BaseState::qreg_.apply_diagonal_unitary_matrix(qubits, mdiag);

    // If it doesn't agree with the reset state update
    // TODO This function could be optimized as a permutation update
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
      BaseState::qreg_.apply_unitary_matrix(qubits, perm);
    }
  }
}


//=========================================================================
// Implementation: Kraus Noise
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_kraus(const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats) {
  // Convert to Superoperator
  const auto nrows = kmats[0].GetRows();
  cmatrix_t superop(nrows * nrows, nrows * nrows);
  for (const auto kraus : kmats) {
    superop += Utils::tensor_product(Utils::conjugate(kraus), kraus);
  }
  BaseState::qreg_.apply_superop_matrix(qubits, Utils::vectorize_matrix(superop));
}

//-------------------------------------------------------------------------
} // end namespace DensityMatrix
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
