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
// For this simulation method, we represent the state of the circuit using a tensor
// network structure, the specifically matrix product state. The idea is based on
// the following paper (there exist other sources as well):
// The density-matrix renormalization group in the age of matrix product states by
// Ulrich Schollwock.
//
//--------------------------------------------------------------------------

#ifndef _matrix_product_state_hpp
#define _matrix_product_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/json.hpp"
#include "simulators/state.hpp"
#include "matrix_product_state_internal.hpp"
#include "matrix_product_state_internal.cpp"


namespace AER {
namespace MatrixProductState {

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  {Operations::OpType::gate, Operations::OpType::measure,
  Operations::OpType::reset, Operations::OpType::initialize,
  Operations::OpType::snapshot, Operations::OpType::barrier,
  Operations::OpType::bfunc, Operations::OpType::roerror,
  Operations::OpType::matrix},
  // Gates
  {"id", "x", "y", "z", "s", "sdg", "h", "t", "tdg", "u1", "u2", "u3",
    "U", "CX", "cx", "cz", "cu1", "swap", "ccx"},
  // Snapshots
  {"statevector", "memory", "register", "probabilities",
    "expectation_value_pauli", "expectation_value_pauli_with_variance",
    "expectation_value_pauli_single_shot", "expectation_value_matrix",
    "expectation_value_matrix_with_variance",
    "expectation_value_matrix_single_shot"}
);

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
// Matrix Product State subclass
//=========================================================================

using matrixproductstate_t = MPS;

class State : public Base::State<matrixproductstate_t> {
public:
  using BaseState = Base::State<matrixproductstate_t>;

  State() : BaseState(StateOpSet) {}
  State(uint_t num_qubits) : State() {qreg_.initialize((uint_t)num_qubits);}
  virtual ~State() = default;


  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {
	  return "matrix_product_state";
  }

  bool empty() const {
    return qreg_.empty();
  }

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exception will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentData &data,
                         RngEngine &rng) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit state given as a complex std::vector
  virtual void initialize_qreg(uint_t num_qubits, const matrixproductstate_t &state) override;

  void initialize_qreg(uint_t num_qubits, const cvector_t &statevector);

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
    virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops)
                                    const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // We currently set the threshold to 1 in qasm_controller.hpp, i.e., no parallelization
  virtual void set_config(const json_t &config) override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t& qubits,
                                            uint_t shots,
                                            RngEngine &rng) override;

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
  void apply_initialize(const reg_t &qubits,
			const cvector_t &params,
			RngEngine &rng);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function, then "measure"
  // should be contained in the set defineed by 'allowed_ops'
  virtual void apply_measure(const reg_t &qubits,
                             const reg_t &cmemory,
                             const reg_t &cregister,
                             RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits, RngEngine &rng);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exception will be raised.
  virtual void apply_snapshot(const Operations::Op &op, ExperimentData &data);

  // Apply a matrix to given qubits (identity on all other qubits)
  // We assume matrix to be 2x2
  void apply_matrix(const reg_t &qubits, const cmatrix_t & mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t & vmat);

  // Apply a Kraus error operation
  //void apply_kraus(const reg_t &qubits,
  //                 const std::vector<cmatrix_t> &krausops,
  //                 RngEngine &rng);

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
  std::pair<uint_t, double>
  sample_measure_with_prob(const reg_t &qubits, RngEngine &rng);


  void measure_reset_update(const reg_t &qubits,
                            const uint_t final_state,
                            const reg_t &meas_state);

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
                              SnapshotDataType type);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const Operations::Op &op,
                             ExperimentData &data,
                             SnapshotDataType type);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(const Operations::Op &op,
                              ExperimentData &data,
                              SnapshotDataType type);

  // Snapshot the state vector
  void snapshot_state(const Operations::Op &op,
		      ExperimentData &data,
		      std::string name = "");

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

  // Table of allowed snapshot types to enum class members
  const static stringmap_t<Snapshots> snapshotset_;

};


//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

const stringmap_t<Gates> State::gateset_({
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
  {"cu1", Gates::cu1},     // Controlled-U1 gate
  {"swap", Gates::swap}, // SWAP gate
  // Three-qubit gates
   {"ccx", Gates::mcx}    // Controlled-CX gate (Toffoli)
});

const stringmap_t<Snapshots> State::snapshotset_({
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

void State::initialize_qreg(uint_t num_qubits) {
  qreg_.initialize((uint_t)num_qubits);
}

void State::initialize_qreg(uint_t num_qubits, const matrixproductstate_t &state) {
  // Check dimension of state
  if (qreg_.num_qubits() != num_qubits) {
    throw std::invalid_argument("MatrixProductState::State::initialize: initial state does not match qubit number");
  }
#ifdef DEBUG
  cout << "initialize with state not supported yet";
#endif
}

void State::initialize_qreg(uint_t num_qubits, const cvector_t &statevector) {
  // Check dimension of state
  if (qreg_.num_qubits() != num_qubits) {
    throw std::invalid_argument("MatrixProductState::State::initialize: initial state does not match qubit number");
  }

  // internal bit ordering is the opposite of ordering in Qasm, so must
  // reverse order before starting
  cvector_t mps_format_state_vector = reverse_all_bits(statevector, num_qubits);

  qreg_.initialize_from_statevector(num_qubits, mps_format_state_vector);
}

size_t State::required_memory_mb(uint_t num_qubits,
			      const std::vector<Operations::Op> &ops) const {
    // for each qubit we have a tensor structure.
    // Initially, each tensor contains 2 matrices with a single complex double
    // Depending on the number of 2-qubit gates,
    // these matrices may double their size
    // for now - compute only initial size
    // later - FIXME
    size_t mem_mb = 16 * 2 * num_qubits;
    return mem_mb;
}

void State::set_config(const json_t &config) {

  // Set threshold for truncating snapshots
  uint_t json_chop_threshold;
  if (JSON::get_value(json_chop_threshold, "chop_threshold", config))
    MPS::set_json_chop_threshold(json_chop_threshold);

  // Set OMP num threshold
  uint_t omp_qubit_threshold;
  if (JSON::get_value(omp_qubit_threshold, "mps_parallel_threshold", config))
    MPS::set_omp_threshold(omp_qubit_threshold);

  // Set OMP threads
  uint_t omp_threads;
  if (JSON::get_value(omp_threads, "mps_omp_threads", config))
    MPS::set_omp_threads(omp_threads);

  // Set the sample measure indexing size
  int index_size;
  if (JSON::get_value(index_size, "mps_sample_measure_opt", config)) {
    MPS::set_sample_measure_index_size(index_size);
  }
}

//=========================================================================
// Implementation: apply operations
//=========================================================================

void State::apply_ops(const std::vector<Operations::Op> &ops,
                      ExperimentData &data,
                      RngEngine &rng) {

  // Simple loop over vector of input operations
  for (const auto op: ops) {
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
          apply_matrix(op.qubits, op.mats[0]);
          break;
        default:
          throw std::invalid_argument("MatrixProductState::State::invalid instruction \'" +
                                      op.name + "\'.");
      }
    }
  }
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

void State::snapshot_pauli_expval(const Operations::Op &op,
				  ExperimentData &data,
				  SnapshotDataType type){
  if (op.params_expval_pauli.empty()) {
    throw std::invalid_argument("Invalid expval snapshot (Pauli components are empty).");
  }

  //Compute expval components
  complex_t expval(0., 0.);

  for (const auto &param : op.params_expval_pauli) {
    complex_t coeff = param.first;
    std::string pauli_matrices = param.second;
    complex_t pauli_expval = qreg_.expectation_value_pauli(op.qubits, pauli_matrices);
    expval += coeff * pauli_expval;
  }

  // add to snapshot
  Utils::chop_inplace(expval, MPS::get_json_chop_threshold());
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

void State::snapshot_matrix_expval(const Operations::Op &op,
				   ExperimentData &data,
				   SnapshotDataType type){
  if (op.params_expval_matrix.empty()) {
    throw std::invalid_argument("Invalid matrix snapshot (components are empty).");
  }
  complex_t expval(0., 0.);
  double one_expval = 0;

  for (const auto &param : op.params_expval_matrix) {
    complex_t coeff = param.first;

    for (const auto &pair: param.second) {
      reg_t sub_qubits;
      for (const auto pos : pair.first) {
        sub_qubits.push_back(op.qubits[pos]);
      }
      const cmatrix_t &mat = pair.second;
      one_expval = qreg_.expectation_value(sub_qubits, mat);
      expval += coeff * one_expval;
    }
  }
  // add to snapshot
  Utils::chop_inplace(expval, MPS::get_json_chop_threshold());
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

void State::snapshot_state(const Operations::Op &op,
			   ExperimentData &data,
			   std::string name) {
  cvector_t statevector;
  qreg_.full_state_vector(statevector);
  data.add_pershot_snapshot("statevector", op.string_params[0], statevector);
}

void State::snapshot_probabilities(const Operations::Op &op,
				   ExperimentData &data,
				   SnapshotDataType type) {
  rvector_t prob_vector;
  qreg_.get_probabilities_vector(prob_vector, op.qubits);
  auto probs = Utils::vec2ket(prob_vector, MPS::get_json_chop_threshold(), 16);
  bool variance = type == SnapshotDataType::average_var;
  data.add_average_snapshot("probabilities", op.string_params[0], 
  			    BaseState::creg_.memory_hex(), probs, variance);

}

void State::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
      "MatrixProductState::State::invalid gate instruction \'" + op.name + "\'.");
  switch (it -> second) {
  case Gates::mcx:
      qreg_.apply_ccx(op.qubits);
      break;
    case Gates::u3:
      qreg_.apply_u3(op.qubits[0],
                    std::real(op.params[0]),
                    std::real(op.params[1]),
                    std::real(op.params[2]));
      break;
    case Gates::u2:
      qreg_.apply_u2(op.qubits[0],
                    std::real(op.params[0]),
                    std::real(op.params[1]));
      break;
    case Gates::u1:
      qreg_.apply_u1(op.qubits[0],
		     std::real(op.params[0]));
      break;
    case Gates::cx:
      qreg_.apply_cnot(op.qubits[0], op.qubits[1]);
      break;
    case Gates::id:
    {
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
    case Gates::t:
      qreg_.apply_t(op.qubits[0]);
      break;
    case Gates::tdg:
      qreg_.apply_tdg(op.qubits[0]);
      break;
    case Gates::swap:
      qreg_.apply_swap(op.qubits[0], op.qubits[1], true);
      break;
    case Gates::cz:
      qreg_.apply_cz(op.qubits[0], op.qubits[1]);
      break;
    case Gates::cu1:
      qreg_.apply_cu1(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument(
        "MatrixProductState::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

  void State::apply_matrix(const reg_t &qubits, const cmatrix_t &mat) {
   if (!qubits.empty() && mat.size() > 0) {
     qreg_.apply_matrix(qubits, mat);
     return;
   }
  }

  void State::apply_matrix(const reg_t &qubits, const cvector_t &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    qreg_.apply_diagonal_matrix(qubits, vmat);
  } else {
    qreg_.apply_matrix(qubits, vmat);
  }
}

//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

void State::apply_initialize(const reg_t &qubits,
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
    // partial initialization not supported yet
   throw std::invalid_argument("MPS_State: Partial initialization not supported yet.");
}

void State::apply_measure(const reg_t &qubits,
                          const reg_t &cmemory,
                          const reg_t &cregister,
                          RngEngine &rng) {
  reg_t outcome = qreg_.apply_measure(qubits, rng);
  creg_.store_measure(outcome, cmemory, cregister);
}

rvector_t State::measure_probs(const reg_t &qubits) const{
  rvector_t probvector;
  qreg_.get_probabilities_vector(probvector, qubits);
  return probvector;
}

std::vector<reg_t> State::sample_measure(const reg_t &qubits,
                                         uint_t shots,
                                         RngEngine &rng) {

  MPS temp;
  std::vector<reg_t> all_samples;
  all_samples.resize(shots);
  reg_t single_result;

  for (int_t i=0; i<static_cast<int_t>(shots);  i++) {
    temp.initialize(qreg_);
    single_result = temp.apply_measure(qubits, rng);
    all_samples[i] = single_result;
  }

  return all_samples;
}

void State::apply_snapshot(const Operations::Op &op, ExperimentData &data) {
  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument("MatrixProductState::invalid snapshot instruction \'" +
                                op.name + "\'.");
  switch (it -> second) {
  case Snapshots::statevector: {
      snapshot_state(op, data, "statevector");
      break;
  }
  case Snapshots::cmemory:
    BaseState::snapshot_creg_memory(op, data);
    break;
  case Snapshots::cregister:
    BaseState::snapshot_creg_register(op, data);
    break;
  case Snapshots::probs: {
      // get probs as hexadecimal
      snapshot_probabilities(op, data, SnapshotDataType::average);
      break;
  }
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
    throw std::invalid_argument("MatrixProductState::State::invalid snapshot instruction \'" +
				op.name + "\'.");
  }
}

void State::apply_reset(const reg_t &qubits,
                        RngEngine &rng) {
  // Simulate unobserved measurement
  reg_t outcome = qreg_.apply_measure(qubits, rng);
  // Apply update to reset state
  measure_reset_update(qubits, 0, outcome);
}

void State::measure_reset_update(const reg_t &qubits,
				 const uint_t final_state,
				 const reg_t &meas_state) {
  for (uint_t i=0; i<qubits.size(); i++) {
    if(meas_state[i] != final_state) {
      qreg_.apply_x(qubits[i]);
    }
  }
}

std::pair<uint_t, double>
State::sample_measure_with_prob(const reg_t &qubits,
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
