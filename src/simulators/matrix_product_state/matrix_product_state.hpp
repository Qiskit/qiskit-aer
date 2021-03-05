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
#include "framework/utils.hpp"
#include "simulators/state.hpp"
#include "matrix_product_state_internal.hpp"
#include "matrix_product_state_internal.cpp"
#include "framework/linalg/almost_equal.hpp"


namespace AER {
namespace MatrixProductState {

using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  {OpType::gate, OpType::measure,
   OpType::reset, OpType::initialize,
   OpType::snapshot, OpType::barrier,
   OpType::bfunc, OpType::roerror,
   OpType::matrix, OpType::diagonal_matrix,
   OpType::kraus, OpType::save_expval,
   OpType::save_expval_var, OpType::save_densmat,
   OpType::save_statevec, OpType::save_probs,
   OpType::save_probs_ket, OpType::save_amps,
   OpType::save_amps_sq},
  // Gates
  {"id", "x",  "y", "z", "s",  "sdg", "h",  "t",   "tdg",  "p", "u1",
   "u2", "u3", "u", "U", "CX", "cx",  "cy", "cz", "cp", "cu1", "swap", "ccx",
   "sx", "r", "rx", "ry", "rz", "rxx", "ryy", "rzz", "rzx", "csx", "delay",
   "cswap"},
  // Snapshots
  {"statevector", "amplitudes", "memory", "register", "probabilities",
    "expectation_value_pauli", "expectation_value_pauli_with_variance",
    "expectation_value_pauli_single_shot", "expectation_value_matrix",
    "expectation_value_matrix_with_variance",
      "expectation_value_matrix_single_shot",
      "density_matrix", "density_matrix_with_variance"}
);

// Allowed snapshots enum class
enum class Snapshots {
  statevector, amplitudes, cmemory, cregister,
    probs, probs_var, densmat, densmat_var,
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
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops = false) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit state given as a complex std::vector
  void initialize_qreg(uint_t num_qubits, const cvector_t &statevector);

  virtual void initialize_qreg(uint_t num_qubits, const matrixproductstate_t &state) override;

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

  virtual void add_metadata(ExperimentResult &result) const override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t& qubits,
                                            uint_t shots,
                                            RngEngine &rng) override;

  // Computes sample_measure by first computing the probabilities and then
  // randomly chooses measurement outcomes based on the probability weights
  std::vector<reg_t> 
  sample_measure_using_probabilities(const reg_t &qubits,
				     uint_t shots,
				     RngEngine &rng);

  // Computes sample_measure by copying the MPS to a temporary structure, and
  // applying a measurement on the temporary MPS. This is done for every shot,
  // so is not efficient for a large number of shots
  std::vector<reg_t> 
  sample_measure_using_apply_measure(const reg_t &qubits,
				     uint_t shots,
				     RngEngine &rng) const;

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
  // should be contained in the set defined by 'allowed_ops'
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
  virtual void apply_snapshot(const Operations::Op &op, ExperimentResult &result);

  // Apply a matrix to given qubits (identity on all other qubits)
  // We assume matrix to be 2x2
  void apply_matrix(const reg_t &qubits, const cmatrix_t & mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t & vmat);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits,
                   const std::vector<cmatrix_t> &kmats,
                   RngEngine &rng);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Compute and save the statevector for the current simulator state
  void apply_save_statevector(const Operations::Op &op,
                              ExperimentResult &result);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const Operations::Op &op,
                                 ExperimentResult &result);

  // Helper function for computing expectation value
  void apply_save_probs(const Operations::Op &op,
                        ExperimentResult &result);

  // Helper function for saving amplitudes and amplitudes squared
  void apply_save_amplitudes(const Operations::Op &op,
                             ExperimentResult &result);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string& pauli) override;

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

  //-----------------------------------------------------------------------
  // Special snapshot types
  //
  // IMPORTANT: These methods are not marked const to allow modifying state
  // during snapshot, but after the snapshot is applied the simulator
  // should be left in the pre-snapshot state.
  //-----------------------------------------------------------------------

  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(const Operations::Op &op,
                              ExperimentResult &result,
                              SnapshotDataType type);

 void snapshot_density_matrix(const Operations::Op &op,
			     ExperimentResult &result,
	     		     SnapshotDataType type);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const Operations::Op &op,
                             ExperimentResult &result,
                             SnapshotDataType type);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(const Operations::Op &op,
                              ExperimentResult &result,
                              SnapshotDataType type);

  // Snapshot the state vector
  void snapshot_state(const Operations::Op &op,
		      ExperimentResult &result,
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
  {"delay", Gates::id},
  {"x", Gates::x},       // Pauli-X gate
  {"y", Gates::y},       // Pauli-Y gate
  {"z", Gates::z},       // Pauli-Z gate
  {"s", Gates::s},       // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg},   // Conjugate-transpose of Phase gate
  {"h", Gates::h},       // Hadamard gate (X + Z / sqrt(2))
  {"sx", Gates::sx},     // Sqrt(X) gate
  {"t", Gates::t},       // T-gate (sqrt(S))
  {"tdg", Gates::tdg},   // Conjguate-transpose of T gate
  {"r", Gates::r},       // R rotation gate
  {"rx", Gates::rx},     // Pauli-X rotation gate
  {"ry", Gates::ry},     // Pauli-Y rotation gate
  {"rz", Gates::rz},     // Pauli-Z rotation gate
  // Waltz Gates
  {"p", Gates::u1},      // zero-X90 pulse waltz gate
  {"u1", Gates::u1},     // zero-X90 pulse waltz gate
  {"u2", Gates::u2},     // single-X90 pulse waltz gate
  {"u3", Gates::u3},     // two X90 pulse waltz gate
  {"u", Gates::u3},      // two X90 pulse waltz gate
  {"U", Gates::u3},      // two X90 pulse waltz gate
  // Two-qubit gates
  {"CX", Gates::cx},     // Controlled-X gate (CNOT)
  {"cx", Gates::cx},     // Controlled-X gate (CNOT)
  {"cy", Gates::cy},     // Controlled-Y gate
  {"cz", Gates::cz},     // Controlled-Z gate
  {"cu1", Gates::cu1},   // Controlled-U1 gate
  {"cp", Gates::cu1},    // Controlled-U1 gate
  {"csx", Gates::csx},
  {"swap", Gates::swap}, // SWAP gate
  {"rxx", Gates::rxx},   // Pauli-XX rotation gate
  {"ryy", Gates::ryy},   // Pauli-YY rotation gate
  {"rzz", Gates::rzz},   // Pauli-ZZ rotation gate
  {"rzx", Gates::rzx},   // Pauli-ZX rotation gate
  // Three-qubit gates
  {"ccx", Gates::ccx},   // Controlled-CX gate (Toffoli)
  {"cswap", Gates::cswap}
});

const stringmap_t<Snapshots> State::snapshotset_({
  {"statevector", Snapshots::statevector},
  {"amplitudes", Snapshots::amplitudes},
  {"probabilities", Snapshots::probs},
  {"expectation_value_pauli", Snapshots::expval_pauli},
  {"expectation_value_matrix", Snapshots::expval_matrix},
  {"probabilities_with_variance", Snapshots::probs_var},
  {"density_matrix", Snapshots::densmat},
  {"density_matrix_with_variance", Snapshots::densmat_var},
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

void State::initialize_qreg(uint_t num_qubits=0) {
  qreg_.initialize(num_qubits);
}

void State::initialize_qreg(uint_t num_qubits, const cvector_t &statevector) {
  if (qreg_.num_qubits() != num_qubits)
    throw std::invalid_argument("MatrixProductState::State::initialize_qreg: initial state does not match qubit number");
  reg_t qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  qreg_.initialize_from_statevector_internal(qubits, statevector);
}

void State::initialize_qreg(uint_t num_qubits, const matrixproductstate_t &state) {
  // Check dimension of state
  if (qreg_.num_qubits() != num_qubits) {
    throw std::invalid_argument("MatrixProductState::State::initialize_qreg: initial state does not match qubit number");
  }
#ifdef DEBUG
  std::cout << "initialize with state not supported yet";
#endif
}

void State::initialize_omp() {
  if (BaseState::threads_ > 0)
    qreg_.set_omp_threads(BaseState::threads_); // set allowed OMP threads in MPS
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
  // Set threshold for truncating Schmidt coefficients
  double threshold;
  if (JSON::get_value(threshold, "matrix_product_state_truncation_threshold", config))
    MPS_Tensor::set_truncation_threshold(threshold);
  else
    MPS_Tensor::set_truncation_threshold(1e-16);

  uint_t max_bond_dimension;
  if (JSON::get_value(max_bond_dimension, "matrix_product_state_max_bond_dimension", config)) 
    MPS_Tensor::set_max_bond_dimension(max_bond_dimension);
  else
    MPS_Tensor::set_max_bond_dimension(UINT64_MAX);

  // Set threshold for truncating snapshots
  uint_t json_chop_threshold;
  if (JSON::get_value(json_chop_threshold, "chop_threshold", config))
    MPS::set_json_chop_threshold(json_chop_threshold);
  else
    MPS::set_json_chop_threshold(1E-8);

  // Set OMP num threshold
  uint_t omp_qubit_threshold;
  if (JSON::get_value(omp_qubit_threshold, "mps_parallel_threshold", config))
    MPS::set_omp_threshold(omp_qubit_threshold);
  else
     MPS::set_omp_threshold(14);

  // Set OMP threads
  uint_t omp_threads;
  if (JSON::get_value(omp_threads, "mps_omp_threads", config))
    MPS::set_omp_threads(omp_threads);
  else
    MPS::set_omp_threads(1);

// Set the algorithm for sample measure
  std::string alg;
  if (JSON::get_value(alg, "mps_sample_measure_algorithm", config)) {
    if (alg.compare("mps_probabilities") == 0) {
      MPS::set_sample_measure_alg(Sample_measure_alg::PROB);
    } else if (alg.compare("mps_apply_measure") == 0) {
      MPS::set_sample_measure_alg(Sample_measure_alg::APPLY_MEASURE);
    }
  } else {
    MPS::set_sample_measure_alg(Sample_measure_alg::HEURISTIC);
  }
}

void State::add_metadata(ExperimentResult &result) const {
  result.metadata.add(
    MPS_Tensor::get_truncation_threshold(),
    "matrix_product_state_truncation_threshold");
  result.metadata.add(
    MPS_Tensor::get_max_bond_dimension(),
    "matrix_product_state_max_bond_dimension");
  result.metadata.add(
    MPS::get_sample_measure_alg(),
    "matrix_product_state_sample_measure_algorithm");
} 

//=========================================================================
// Implementation: apply operations
//=========================================================================

void State::apply_ops(const std::vector<Operations::Op> &ops,
                      ExperimentResult &result,
                      RngEngine &rng, bool final_ops) {

  // Simple loop over vector of input operations
  for (const auto &op: ops) {
    if(BaseState::creg_.check_conditional(op)) {
      switch (op.type) {
        case OpType::barrier:
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
          BaseState::creg_.apply_bfunc(op);
          break;
        case OpType::roerror:
          BaseState::creg_.apply_roerror(op, rng);
          break;
        case OpType::gate:
          apply_gate(op);
          break;
        case OpType::snapshot:
          apply_snapshot(op, result);
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
        case OpType::save_probs:
        case OpType::save_probs_ket:
          apply_save_probs(op, result);
          break;
        case OpType::save_amps:
        case OpType::save_amps_sq:
          apply_save_amplitudes(op, result);
          break;
        default:
          throw std::invalid_argument("MatrixProductState::State::invalid instruction \'" +
                                      op.name + "\'.");
      }
    }
  }
}

//=========================================================================
// Implementation: Save data
//=========================================================================

void State::apply_save_probs(const Operations::Op &op,
                             ExperimentResult &result) {
  rvector_t probs;
  qreg_.get_probabilities_vector(probs, op.qubits);
  if (op.type == OpType::save_probs_ket) {
    BaseState::save_data_average(result, op.string_params[0],
                                 Utils::vec2ket(probs, MPS::get_json_chop_threshold(), 16),
                                 op.save_type);
  } else {
    BaseState::save_data_average(result, op.string_params[0],
                                 std::move(probs), op.save_type);
  }
}

void State::apply_save_amplitudes(const Operations::Op &op,
                             ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument("Invalid save amplitudes instructions (empty params).");
  }
  Vector<complex_t> amps = qreg_.get_amplitude_vector(op.int_params);
  if (op.type == OpType::save_amps_sq) {
    // Square amplitudes
    std::vector<double> amps_sq(op.int_params.size());
    std::transform(amps.data(), amps.data() + amps.size(), amps_sq.begin(),
      [](complex_t val) -> double { return pow(abs(val), 2); });
    BaseState::save_data_average(result, op.string_params[0],
                                 std::move(amps_sq), op.save_type);
  } else {
    BaseState::save_data_pershot(result, op.string_params[0],
                                 std::move(amps), op.save_type);
  }
}

double State::expval_pauli(const reg_t &qubits,
                           const std::string& pauli) {
  return BaseState::qreg_.expectation_value_pauli(qubits, pauli).real();
}

void State::apply_save_statevector(const Operations::Op &op,
                                   ExperimentResult &result) {
  if (op.qubits.size() != BaseState::qreg_.num_qubits()) {
    throw std::invalid_argument(
        "Save statevector was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  BaseState::save_data_pershot(result, op.string_params[0],
                               qreg_.full_statevector(), op.save_type);
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

  BaseState::save_data_average(result, op.string_params[0],
                               std::move(reduced_state), op.save_type);
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

void State::snapshot_pauli_expval(const Operations::Op &op,
				  ExperimentResult &result,
				  SnapshotDataType type){
  if (op.params_expval_pauli.empty()) {
    throw std::invalid_argument("Invalid expval snapshot (Pauli components are empty).");
  }

  //Compute expval components
  complex_t expval(0., 0.);

  for (const auto &param : op.params_expval_pauli) {
    complex_t coeff = param.first;
    std::string pauli_matrices = param.second;
    expval += coeff * expval_pauli(op.qubits, pauli_matrices);
  }

  // add to snapshot
  Utils::chop_inplace(expval, MPS::get_json_chop_threshold());
  switch (type) {
    case SnapshotDataType::average:
      result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::creg_.memory_hex(), expval, false);
      break;
    case SnapshotDataType::average_var:
      result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::creg_.memory_hex(), expval, true);
      break;
    case SnapshotDataType::pershot:
      result.legacy_data.add_pershot_snapshot("expectation_values", op.string_params[0], expval);
      break;
  }
}

void State::snapshot_matrix_expval(const Operations::Op &op,
				   ExperimentResult &result,
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
      result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::creg_.memory_hex(), expval, false);
      break;
    case SnapshotDataType::average_var:
      result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::creg_.memory_hex(), expval, true);
      break;
    case SnapshotDataType::pershot:
      result.legacy_data.add_pershot_snapshot("expectation_values", op.string_params[0], expval);
      break;
  }
}

void State::snapshot_state(const Operations::Op &op,
			   ExperimentResult &result,
			   std::string name) {
  result.legacy_data.add_pershot_snapshot(
    "statevector", op.string_params[0], qreg_.full_statevector());
}

void State::snapshot_probabilities(const Operations::Op &op,
				   ExperimentResult &result,
				   SnapshotDataType type) {
  rvector_t prob_vector;
  qreg_.get_probabilities_vector(prob_vector, op.qubits);
  auto probs = Utils::vec2ket(prob_vector, MPS::get_json_chop_threshold(), 16);

  bool variance = type == SnapshotDataType::average_var;
  result.legacy_data.add_average_snapshot("probabilities", op.string_params[0], 
  			    BaseState::creg_.memory_hex(), probs, variance);

}

void State::snapshot_density_matrix(const Operations::Op &op,
			     ExperimentResult &result,
			     SnapshotDataType type) {
  cmatrix_t reduced_state;
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);
    reduced_state[0] = qreg_.norm();
  } else {
    reduced_state = qreg_.density_matrix(op.qubits);
  }

  // Add density matrix to result data
  switch (type) {
    case SnapshotDataType::average:
      result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                            BaseState::creg_.memory_hex(), std::move(reduced_state), false);
      break;
    case SnapshotDataType::average_var:
      result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                            BaseState::creg_.memory_hex(), std::move(reduced_state), true);
      break;
    case SnapshotDataType::pershot:
      result.legacy_data.add_pershot_snapshot("density_matrix", op.string_params[0], std::move(reduced_state));
      break;
  }
}

void State::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
      "MatrixProductState::State::invalid gate instruction \'" + op.name + "\'.");
  switch (it -> second) {
    case Gates::ccx:
      qreg_.apply_ccx(op.qubits);
      break;
    case Gates::cswap:
      qreg_.apply_cswap(op.qubits);
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
    case Gates::sx:
      qreg_.apply_sx(op.qubits[0]);
      break;
    case Gates::t:
      qreg_.apply_t(op.qubits[0]);
      break;
    case Gates::tdg:
      qreg_.apply_tdg(op.qubits[0]);
      break;
    case Gates::r:
      qreg_.apply_r(op.qubits[0], 
		    std::real(op.params[0]),
		    std::real(op.params[1]));
      break;
    case Gates::rx:
      qreg_.apply_rx(op.qubits[0], 
		     std::real(op.params[0]));
      break;
    case Gates::ry:
      qreg_.apply_ry(op.qubits[0], 
		     std::real(op.params[0]));
      break;
    case Gates::rz:
      qreg_.apply_rz(op.qubits[0], 
		     std::real(op.params[0]));
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
      qreg_.apply_cu1(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    case Gates::rxx:
      qreg_.apply_rxx(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    case Gates::ryy:
      qreg_.apply_ryy(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    case Gates::rzz:
      qreg_.apply_rzz(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    case Gates::rzx:
      qreg_.apply_rzx(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument(
        "MatrixProductState::State::invalid gate instruction \'" + op.name + "\'.");
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
                   const std::vector<cmatrix_t> &kmats,
                   RngEngine &rng) {
  qreg_.apply_kraus(qubits, kmats, rng);
}


//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

void State::apply_initialize(const reg_t &qubits,
			     const cvector_t &params,
			     RngEngine &rng) {
  qreg_.apply_initialize(qubits, params, rng);
}

void State::apply_measure(const reg_t &qubits,
                          const reg_t &cmemory,
                          const reg_t &cregister,
                          RngEngine &rng) {
  reg_t outcome = qreg_.apply_measure(qubits, rng);
  creg_.store_measure(outcome, cmemory, cregister);
}

rvector_t State::measure_probs(const reg_t &qubits) const {
  rvector_t probvector;
  qreg_.get_probabilities_vector(probvector, qubits);
  return probvector;
}

std::vector<reg_t> State::sample_measure(const reg_t &qubits,
                                         uint_t shots,
                                         RngEngine &rng) {
  // There are two alternative algorithms for sample measure
  // We choose the one that is optimal relative to the total number 
  // of qubits,and the number of shots.
  // The parameters used below are based on experimentation.
  // The user can override this by setting the parameter "mps_sample_measure_algorithm"
  uint_t num_qubits = qubits.size();
  if (MPS::get_sample_measure_alg() == Sample_measure_alg::PROB){
    return sample_measure_using_probabilities(qubits, shots, rng);
  }
  if (MPS::get_sample_measure_alg() == Sample_measure_alg::APPLY_MEASURE ||
      num_qubits >26 )
     return sample_measure_using_apply_measure(qubits, shots, rng);

  double num_qubits_dbl = static_cast<double>(num_qubits);
  double shots_dbl = static_cast<double>(shots);

  // Sample_measure_alg::HEURISTIC
  uint_t max_bond_dim = qreg_.get_max_bond_dimensions();

  if (num_qubits <10)
    return sample_measure_using_probabilities(qubits, shots, rng);
  if (max_bond_dim <= 2) {
    if (shots_dbl < 12.0 * pow(1.85, (num_qubits_dbl-10.0)))
       return sample_measure_using_apply_measure(qubits, shots, rng);
    else
      return sample_measure_using_probabilities(qubits, shots, rng);
  } else if (max_bond_dim <= 4) {
    if (shots_dbl < 3.0 * pow(1.75, (num_qubits_dbl-10.0)))
       return sample_measure_using_apply_measure(qubits, shots, rng);
    else
      return sample_measure_using_probabilities(qubits, shots, rng);
  } else if (max_bond_dim <= 8) {
    if (shots_dbl < 2.5 * pow(1.65, (num_qubits_dbl-10.0)))
       return sample_measure_using_apply_measure(qubits, shots, rng);
    else
      return sample_measure_using_probabilities(qubits, shots, rng);
  } else if (max_bond_dim <= 16) {
    if (shots_dbl < 0.5 * pow(1.75, (num_qubits_dbl-10.0)))
       return sample_measure_using_apply_measure(qubits, shots, rng);
    else
      return sample_measure_using_probabilities(qubits, shots, rng);
  } 
  return sample_measure_using_probabilities(qubits, shots, rng);
}
	     
std::vector<reg_t> State::
sample_measure_using_probabilities(const reg_t &qubits,
				   uint_t shots,
				   RngEngine &rng) {

  // Generate flat register for storing
  rvector_t rnds;
  rnds.reserve(shots);
  for (uint_t i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));

  auto allbit_samples = qreg_.sample_measure_using_probabilities(rnds, qubits);

  // Convert to reg_t format
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);
  for (int_t val : allbit_samples) {
    reg_t allbit_sample = Utils::int2reg(val, 2, qubits.size());
    reg_t sample;
    sample.reserve(qubits.size());
    for (uint_t j=0; j<qubits.size(); j++){
      sample.push_back(allbit_sample[j]);
    }
    all_samples.push_back(sample);
  }
  return all_samples;
}

std::vector<reg_t> State::
  sample_measure_using_apply_measure(const reg_t &qubits, 
				     uint_t shots, 
				     RngEngine &rng) const {
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

void State::apply_snapshot(const Operations::Op &op, ExperimentResult &result) {
  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument("MatrixProductState::invalid snapshot instruction \'" +
                                op.name + "\'.");
  switch (it -> second) {
  case Snapshots::statevector: {
      snapshot_state(op, result, "statevector");
      break;
  }
  case Snapshots::cmemory:
    BaseState::snapshot_creg_memory(op, result);
    break;
  case Snapshots::cregister:
    BaseState::snapshot_creg_register(op, result);
    break;
  case Snapshots::probs: {
      // get probs as hexadecimal
      snapshot_probabilities(op, result, SnapshotDataType::average);
      break;
  }
  case Snapshots::densmat: {
      snapshot_density_matrix(op, result, SnapshotDataType::average);
  } break;
  case Snapshots::expval_pauli: {
    snapshot_pauli_expval(op, result, SnapshotDataType::average);
  } break;
  case Snapshots::expval_matrix: {
    snapshot_matrix_expval(op, result, SnapshotDataType::average);
  }  break;
  case Snapshots::probs_var: {
    // get probs as hexadecimal
    snapshot_probabilities(op, result, SnapshotDataType::average_var);
  } break;
  case Snapshots::densmat_var: {
      snapshot_density_matrix(op, result, SnapshotDataType::average_var);
  } break;
  case Snapshots::expval_pauli_var: {
    snapshot_pauli_expval(op, result, SnapshotDataType::average_var);
  } break;
  case Snapshots::expval_matrix_var: {
    snapshot_matrix_expval(op, result, SnapshotDataType::average_var);
  }  break;
  case Snapshots::expval_pauli_shot: {
    snapshot_pauli_expval(op, result, SnapshotDataType::pershot);
  } break;
  case Snapshots::expval_matrix_shot: {
    snapshot_matrix_expval(op, result, SnapshotDataType::pershot);
  }  break;
  default:
    // We shouldn't get here unless there is a bug in the snapshotset
    throw std::invalid_argument("MatrixProductState::State::invalid snapshot instruction \'" +
				op.name + "\'.");
  }
}

void State::apply_reset(const reg_t &qubits,
                        RngEngine &rng) {
  qreg_.reset(qubits, rng);
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
