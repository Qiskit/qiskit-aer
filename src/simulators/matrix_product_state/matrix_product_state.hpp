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
#include <sstream>
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

using matrixproductstate_t = MPS;

static uint_t instruction_number = 0;

using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  {OpType::gate, OpType::measure,
   OpType::reset, OpType::initialize,
   OpType::snapshot, OpType::barrier,
   OpType::bfunc, OpType::roerror, OpType::qerror_loc,
   OpType::matrix, OpType::diagonal_matrix,
   OpType::kraus, OpType::save_expval,
   OpType::save_expval_var, OpType::save_densmat,
   OpType::save_statevec, OpType::save_probs,
   OpType::save_probs_ket, OpType::save_amps,
   OpType::save_amps_sq, OpType::save_mps, OpType::save_state,
   OpType::set_mps, OpType::set_statevec,
   OpType::jump, OpType::mark
  },
  // Gates
  {"id", "x",  "y", "z", "s",  "sdg", "h",  "t",   "tdg",  "p", "u1",
   "u2", "u3", "u", "U", "CX", "cx",  "cy", "cz", "cp", "cu1", "swap", "ccx",
   "sx", "sxdg", "r", "rx", "ry", "rz", "rxx", "ryy", "rzz", "rzx", "csx", "delay",
   "cswap", "pauli"},
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


class State : public QuantumState::State< matrixproductstate_t> {
public:
  using BaseState = QuantumState::State< matrixproductstate_t>;

  State() : BaseState(StateOpSet) {}
  State(uint_t num_qubits) : State() {BaseState::state_.qreg().initialize((uint_t)num_qubits);}
  virtual ~State() = default;


  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {
	  return "matrix_product_state";
  }

  /*    //this is not used?
  bool empty() const {
    return qreg_.empty();
  }
  */

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  void apply_op(QuantumState::RegistersBase& state,
                        const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine &rng,
                        bool final_op = false) override;


  // Initializes to a specific n-qubit state given as a complex std::vector
  void initialize_qreg_from_data(uint_t num_qubits, const cvector_t &statevector);

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual size_t required_memory_mb(uint_t num_qubits,
                                  QuantumState::OpItr first, QuantumState::OpItr last)
                                  const override;

  virtual void add_metadata(ExperimentResult &result) const override;

  // prints the bond dimensions after each instruction to the metadata
  void output_bond_dimensions(const Operations::Op &op) const;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure_state(QuantumState::RegistersBase& state_in, const reg_t& qubits,
                                            uint_t shots,
                                            RngEngine &rng) override;

  // Computes sample_measure by copying the MPS to a temporary structure, and
  // applying a measurement on the temporary MPS. This is done for every shot,
  // so is not efficient for a large number of shots
  std::vector<reg_t> 
  sample_measure_using_apply_measure(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits,
				     uint_t shots,
				     RngEngine &rng);
std::vector<reg_t> sample_measure_all(QuantumState::Registers<matrixproductstate_t>& state, uint_t shots, 
				      RngEngine &rng);
  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  void initialize_omp(QuantumState::Registers<matrixproductstate_t>& state);

protected:
  // Initializes an n-qubit state to the all |0> state
  void initialize_state(QuantumState::RegistersBase& state_in, uint_t num_qubits) override;

  void initialize_state(QuantumState::RegistersBase& state_in, uint_t num_qubits, const matrixproductstate_t &state) override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // We currently set the threshold to 1 in qasm_controller.hpp, i.e., no parallelization
  void set_state_config(QuantumState::RegistersBase& state_in, const json_t &config) override;

  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(QuantumState::Registers<matrixproductstate_t>& state,const Operations::Op &op);

  // Initialize the specified qubits to a given state |psi>
  // by creating the MPS state with the new state |psi>.
  // |psi> is given in params
  // Currently only supports intialization of all qubits
  void apply_initialize(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits,
			const cvector_t &params,
			RngEngine &rng);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function, then "measure"
  // should be contained in the set defined by 'allowed_ops'
  virtual void apply_measure(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits,
                             const reg_t &cmemory,
                             const reg_t &cregister,
                             RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits, RngEngine &rng);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exception will be raised.
  virtual void apply_snapshot(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op, ExperimentResult &result);

  // Apply a matrix to given qubits (identity on all other qubits)
  // We assume matrix to be 2x2
  void apply_matrix(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits, const cmatrix_t & mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits, const cvector_t & vmat);

  // Apply a Kraus error operation
  void apply_kraus(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits,
                   const std::vector<cmatrix_t> &kmats,
                   RngEngine &rng);

  // Apply multi-qubit Pauli
  void apply_pauli(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits, const std::string& pauli);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current state of the simulator
  void apply_save_mps(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                      ExperimentResult &result,
                      bool last_op);
                            
  // Compute and save the statevector for the current simulator state
  void apply_save_statevector(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                              ExperimentResult &result);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                                 ExperimentResult &result);

  // Helper function for computing expectation value
  void apply_save_probs(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                        ExperimentResult &result);

  // Helper function for saving amplitudes and amplitudes squared
  void apply_save_amplitudes(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                             ExperimentResult &result);

  // Helper function for computing expectation value
  virtual double expval_pauli(QuantumState::RegistersBase& state, const reg_t &qubits,
                              const std::string& pauli) override;

  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Return vector of measure probabilities for specified qubits
  // If a state subclass supports this function, then "measure"
  // must be contained in the set defined by 'allowed_ops'
  rvector_t measure_probs(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits) const;

  // Sample the measurement outcome for qubits
  // return a pair (m, p) of the outcome m, and its corresponding
  // probability p.
  // Outcome is given as an int: Eg for two-qubits {q0, q1} we have
  // 0 -> |q1 = 0, q0 = 0> state
  // 1 -> |q1 = 0, q0 = 1> state
  // 2 -> |q1 = 1, q0 = 0> state
  // 3 -> |q1 = 1, q0 = 1> state
  std::pair<uint_t, double>
  sample_measure_with_prob(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits, RngEngine &rng);

  //-----------------------------------------------------------------------
  // Special snapshot types
  //
  // IMPORTANT: These methods are not marked const to allow modifying state
  // during snapshot, but after the snapshot is applied the simulator
  // should be left in the pre-snapshot state.
  //-----------------------------------------------------------------------

  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                              ExperimentResult &result,
                              SnapshotDataType type);

 void snapshot_density_matrix(QuantumState::Registers<matrixproductstate_t>& state,const Operations::Op &op,
			     ExperimentResult &result,
	     		     SnapshotDataType type);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(QuantumState::Registers<matrixproductstate_t>& state,const Operations::Op &op,
                             ExperimentResult &result,
                             SnapshotDataType type);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(QuantumState::Registers<matrixproductstate_t>& state,const Operations::Op &op,
                              ExperimentResult &result,
                              SnapshotDataType type);

  // Snapshot the state vector
  void snapshot_state(QuantumState::Registers<matrixproductstate_t>& state,const Operations::Op &op,
		      ExperimentResult &result,
		      std::string name = "");

  //-----------------------------------------------------------------------
  // Single-qubit gate helpers
  //-----------------------------------------------------------------------

  // Apply a waltz gate specified by parameters u3(theta, phi, lambda)
  void apply_gate_u3(QuantumState::Registers<matrixproductstate_t>& state,const uint_t qubit, const double theta, const double phi,
                     const double lambda);

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(QuantumState::Registers<matrixproductstate_t>& state,const uint_t qubit, const complex_t phase);

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
  {"sxdg", Gates::sxdg}, // Inverse Sqrt(X) gate
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
  {"cswap", Gates::cswap},
  // Pauli
  {"pauli", Gates::pauli}
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

void State::initialize_state(QuantumState::RegistersBase& state_in, uint_t num_qubits=0) 
{
  QuantumState::Registers<matrixproductstate_t>& state = dynamic_cast<QuantumState::Registers<matrixproductstate_t>&>(state_in);
  if(state.qregs().size() == 0)
    state.allocate(1);
  state.qreg().initialize(num_qubits);
}

void State::initialize_qreg_from_data(uint_t num_qubits, const cvector_t &statevector) 
{
  QuantumState::Registers<matrixproductstate_t>& state = BaseState::state_;
  if(state.qregs().size() == 0)
    state.allocate(1);

  if (state.qreg().num_qubits() != num_qubits)
    throw std::invalid_argument("MatrixProductState::State::initialize_qreg: initial state does not match qubit number");
  reg_t qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  state.qreg().initialize_from_statevector_internal(qubits, statevector);
}

void State::initialize_state(QuantumState::RegistersBase& state_in, uint_t num_qubits, const matrixproductstate_t &mpstate) 
{
  throw std::invalid_argument("MatrixProductState::State::initialize_qreg: initialize with state not supported yet");
}

void State::initialize_omp(QuantumState::Registers<matrixproductstate_t>& state) {
  if (BaseState::threads_ > 0)
    state.qreg().set_omp_threads(BaseState::threads_); // set allowed OMP threads in MPS
}


size_t State::required_memory_mb(uint_t num_qubits,
			      QuantumState::OpItr first, QuantumState::OpItr last) const {
    // for each qubit we have a tensor structure.
    // Initially, each tensor contains 2 matrices with a single complex double
    // Depending on the number of 2-qubit gates,
    // these matrices may double their size
    // for now - compute only initial size
    // later - FIXME
    size_t mem_mb = 16 * 2 * num_qubits;
    return mem_mb;
}

void State::set_state_config(QuantumState::RegistersBase& state_in, const json_t &config) 
{
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
    } else {
      MPS::set_sample_measure_alg(Sample_measure_alg::APPLY_MEASURE);
    }
  }
  // Set mps_log_data
  bool mps_log_data;
  if (JSON::get_value(mps_log_data, "mps_log_data", config))
    MPS::set_mps_log_data(mps_log_data);

// Set the direction for the internal swaps
  std::string direction;
  if (JSON::get_value(direction, "mps_swap_direction", config)) {
    if (direction.compare("mps_swap_right") == 0)
      MPS::set_mps_swap_direction(MPS_swap_direction::SWAP_RIGHT);
    else
      MPS::set_mps_swap_direction(MPS_swap_direction::SWAP_LEFT);
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
  if (MPS::get_mps_log_data())
    result.metadata.add("{" + MPS::output_log() + "}", "MPS_log_data");
} 

void State::output_bond_dimensions(const Operations::Op &op) const {
  MPS::print_to_log("I", instruction_number, ":", op.name, " on qubits ", op.qubits[0]);
  for (uint_t index=1; index<op.qubits.size(); index++) {
    MPS::print_to_log(",", op.qubits[index]);
  }
  BaseState::state_.qreg().print_bond_dimensions();
  instruction_number++;
}


//=========================================================================
// Implementation: apply operations
//=========================================================================

void State::apply_op(QuantumState::RegistersBase& state_in, const Operations::Op &op,
                      ExperimentResult &result,
                      RngEngine &rng, bool final_op) 
{
  QuantumState::Registers<matrixproductstate_t>& state = dynamic_cast<QuantumState::Registers<matrixproductstate_t>&>(state_in);

  if (state.creg().check_conditional(op)) {
    switch (op.type) {
      case OpType::barrier:
      case OpType::qerror_loc:
        break;
      case OpType::reset:
        apply_reset(state, op.qubits, rng);
        break;
      case OpType::initialize:
        apply_initialize(state, op.qubits, op.params, rng);
        break;
      case OpType::measure:
        apply_measure(state, op.qubits, op.memory, op.registers, rng);
        break;
      case OpType::bfunc:
        state.creg().apply_bfunc(op);
        break;
      case OpType::roerror:
        state.creg().apply_roerror(op, rng);
        break;
      case OpType::gate:
        apply_gate(state, op);
        break;
      case OpType::snapshot:
        apply_snapshot(state, op, result);
        break;
      case OpType::matrix:
        apply_matrix(state, op.qubits, op.mats[0]);
        break;
      case OpType::diagonal_matrix:
        state.qreg().apply_diagonal_matrix(op.qubits, op.params);
        break;
      case OpType::kraus:
        apply_kraus(state, op.qubits, op.mats, rng);
        break;
      case OpType::set_statevec: {
          reg_t all_qubits(state.qreg().num_qubits());
          std::iota(all_qubits.begin(), all_qubits.end(), 0);
          state.qreg().apply_initialize(all_qubits, op.params, rng);
          break;
        }
      case OpType::set_mps:
        state.qreg().initialize_from_mps(op.mps);
        break;
      case OpType::save_expval:
      case OpType::save_expval_var:
        apply_save_expval(state, op, result);
        break;
      case OpType::save_densmat:
        apply_save_density_matrix(state, op, result);
        break;
      case OpType::save_statevec:
        apply_save_statevector(state, op, result);
        break;
      case OpType::save_state:
      case OpType::save_mps:
        apply_save_mps(state, op, result, final_op);
        break;
      case OpType::save_probs:
      case OpType::save_probs_ket:
        apply_save_probs(state, op, result);
        break;
      case OpType::save_amps:
      case OpType::save_amps_sq:
        apply_save_amplitudes(state, op, result);
        break;
      default:
        throw std::invalid_argument("MatrixProductState::State::invalid instruction \'" +
                                    op.name + "\'.");
    }
    //qreg_.print(std::cout);
    // print out bond dimensions only if they may have changed since previous print
    if (MPS::get_mps_log_data()
        && (op.type == OpType::gate ||op.type == OpType::measure || 
            op.type == OpType::initialize || op.type == OpType::reset || 
            op.type == OpType::matrix) && 
            op.qubits.size() > 1) {
      output_bond_dimensions(op);
    }
  }
}

//=========================================================================
// Implementation: Save data
//=========================================================================

void State::apply_save_mps(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                           ExperimentResult &result,
                           bool last_op) 
{
  if (op.qubits.size() != state.qreg().num_qubits()) {
    throw std::invalid_argument(
        "Save MPS was not applied to all qubits."
        " Only the full matrix product state can be saved.");
  }
  std::string key = (op.string_params[0] == "_method_")
                      ? "matrix_product_state"
                      : op.string_params[0];
  if (last_op) {
    result.save_data_pershot(state.creg(), key, state.qreg().move_to_mps_container(),
		                         OpType::save_mps, op.save_type);
  } else {
    result.save_data_pershot(state.creg(), key, state.qreg().copy_to_mps_container(),
                             OpType::save_mps, op.save_type);
  }
}

void State::apply_save_probs(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                             ExperimentResult &result) 
{
  rvector_t probs;
  state.qreg().get_probabilities_vector(probs, op.qubits);
  if (op.type == OpType::save_probs_ket) {
    result.save_data_average(state.creg(), op.string_params[0],
                             Utils::vec2ket(probs, MPS::get_json_chop_threshold(), 16),
                             op.type, op.save_type);
  } else {
    result.save_data_average(state.creg(), op.string_params[0],
                             std::move(probs), op.type, op.save_type);
  }
}

void State::apply_save_amplitudes(QuantumState::Registers<matrixproductstate_t>& state,const Operations::Op &op,
                             ExperimentResult &result) 
{
  if (op.int_params.empty()) {
    throw std::invalid_argument("Invalid save amplitudes instructions (empty params).");
  }
  Vector<complex_t> amps = state.qreg().get_amplitude_vector(op.int_params);
  if (op.type == OpType::save_amps_sq) {
    // Square amplitudes
    std::vector<double> amps_sq(op.int_params.size());
    std::transform(amps.data(), amps.data() + amps.size(), amps_sq.begin(),
      [](complex_t val) -> double { return pow(abs(val), 2); });
    result.save_data_average(state.creg(), op.string_params[0],
                             std::move(amps_sq), op.type, op.save_type);
  } else {
    result.save_data_pershot(state.creg(), op.string_params[0],
                             std::move(amps), op.type, op.save_type);
  }
}

double State::expval_pauli(QuantumState::RegistersBase& state_in, const reg_t &qubits,
                           const std::string& pauli) 
{
  QuantumState::Registers<matrixproductstate_t>& state = dynamic_cast<QuantumState::Registers<matrixproductstate_t>&>(state_in);
  return state.qreg().expectation_value_pauli(qubits, pauli).real();
}

void State::apply_save_statevector(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                                   ExperimentResult &result) 
{
  if (op.qubits.size() != state.qreg().num_qubits()) {
    throw std::invalid_argument(
        "Save statevector was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  result.save_data_pershot(state.creg(), op.string_params[0],
                               state.qreg().full_statevector(), op.type, op.save_type);
}


void State::apply_save_density_matrix(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
                                      ExperimentResult &result) 
{
  cmatrix_t reduced_state;
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);
    reduced_state[0] = state.qreg().norm();
  } else {
    reduced_state = state.qreg().density_matrix(op.qubits);
  }

  result.save_data_average(state.creg(), op.string_params[0],
                           std::move(reduced_state), op.type, op.save_type);
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

void State::snapshot_pauli_expval(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
				  ExperimentResult &result,
				  SnapshotDataType type)
{
  if (op.params_expval_pauli.empty()) {
    throw std::invalid_argument("Invalid expval snapshot (Pauli components are empty).");
  }

  //Compute expval components
  complex_t expval(0., 0.);

  for (const auto &param : op.params_expval_pauli) {
    complex_t coeff = param.first;
    std::string pauli_matrices = param.second;
    expval += coeff * expval_pauli(state, op.qubits, pauli_matrices);
  }

  // add to snapshot
  Utils::chop_inplace(expval, MPS::get_json_chop_threshold());
  switch (type) {
    case SnapshotDataType::average:
      result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            state.creg().memory_hex(), expval, false);
      break;
    case SnapshotDataType::average_var:
      result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            state.creg().memory_hex(), expval, true);
      break;
    case SnapshotDataType::pershot:
      result.legacy_data.add_pershot_snapshot("expectation_values", op.string_params[0], expval);
      break;
  }
}

void State::snapshot_matrix_expval(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
				   ExperimentResult &result,
				   SnapshotDataType type)
{
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
      one_expval = state.qreg().expectation_value(sub_qubits, mat);
      expval += coeff * one_expval;
    }
  }
  // add to snapshot
  Utils::chop_inplace(expval, MPS::get_json_chop_threshold());
  switch (type) {
    case SnapshotDataType::average:
      result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            state.creg().memory_hex(), expval, false);
      break;
    case SnapshotDataType::average_var:
      result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            state.creg().memory_hex(), expval, true);
      break;
    case SnapshotDataType::pershot:
      result.legacy_data.add_pershot_snapshot("expectation_values", op.string_params[0], expval);
      break;
  }
}

void State::snapshot_state(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
			   ExperimentResult &result,
			   std::string name) 
{
  result.legacy_data.add_pershot_snapshot(
    "statevector", op.string_params[0], state.qreg().full_statevector());
}

void State::snapshot_probabilities(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
				   ExperimentResult &result,
				   SnapshotDataType type) 
{
  rvector_t prob_vector;
  state.qreg().get_probabilities_vector(prob_vector, op.qubits);
  auto probs = Utils::vec2ket(prob_vector, MPS::get_json_chop_threshold(), 16);

  bool variance = type == SnapshotDataType::average_var;
  result.legacy_data.add_average_snapshot("probabilities", op.string_params[0], 
                state.creg().memory_hex(), probs, variance);

}

void State::snapshot_density_matrix(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op,
			     ExperimentResult &result,
			     SnapshotDataType type) 
{
  cmatrix_t reduced_state;
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);
    reduced_state[0] = state.qreg().norm();
  } else {
    reduced_state = state.qreg().density_matrix(op.qubits);
  }

  // Add density matrix to result data
  switch (type) {
    case SnapshotDataType::average:
      result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                            state.creg().memory_hex(), std::move(reduced_state), false);
      break;
    case SnapshotDataType::average_var:
      result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                            state.creg().memory_hex(), std::move(reduced_state), true);
      break;
    case SnapshotDataType::pershot:
      result.legacy_data.add_pershot_snapshot("density_matrix", op.string_params[0], std::move(reduced_state));
      break;
  }
}

void State::apply_gate(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op) 
{
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
      "MatrixProductState::State::invalid gate instruction \'" + op.name + "\'.");
  switch (it -> second) {
    case Gates::ccx:
      state.qreg().apply_ccx(op.qubits);
      break;
    case Gates::cswap:
      state.qreg().apply_cswap(op.qubits);
      break;
    case Gates::u3:
      state.qreg().apply_u3(op.qubits[0],
                    std::real(op.params[0]),
                    std::real(op.params[1]),
                    std::real(op.params[2]));
      break;
    case Gates::u2:
      state.qreg().apply_u2(op.qubits[0],
                    std::real(op.params[0]),
                    std::real(op.params[1]));
      break;
    case Gates::u1:
      state.qreg().apply_u1(op.qubits[0],
		     std::real(op.params[0]));
      break;
    case Gates::cx:
      state.qreg().apply_cnot(op.qubits[0], op.qubits[1]);
      break;
    case Gates::id:
    {
        break;
    }
    case Gates::x:
      state.qreg().apply_x(op.qubits[0]);
      break;
    case Gates::y:
      state.qreg().apply_y(op.qubits[0]);
      break;
    case Gates::z:
      state.qreg().apply_z(op.qubits[0]);
      break;
    case Gates::h:
      state.qreg().apply_h(op.qubits[0]);
      break;
    case Gates::s:
      state.qreg().apply_s(op.qubits[0]);
      break;
    case Gates::sdg:
      state.qreg().apply_sdg(op.qubits[0]);
      break;
    case Gates::sx:
      state.qreg().apply_sx(op.qubits[0]);
      break;
    case Gates::sxdg:
      state.qreg().apply_sxdg(op.qubits[0]);
      break;
    case Gates::t:
      state.qreg().apply_t(op.qubits[0]);
      break;
    case Gates::tdg:
      state.qreg().apply_tdg(op.qubits[0]);
      break;
    case Gates::r:
      state.qreg().apply_r(op.qubits[0], 
		    std::real(op.params[0]),
		    std::real(op.params[1]));
      break;
    case Gates::rx:
      state.qreg().apply_rx(op.qubits[0], 
		     std::real(op.params[0]));
      break;
    case Gates::ry:
      state.qreg().apply_ry(op.qubits[0], 
		     std::real(op.params[0]));
      break;
    case Gates::rz:
      state.qreg().apply_rz(op.qubits[0], 
		     std::real(op.params[0]));
      break;
    case Gates::swap:
      state.qreg().apply_swap(op.qubits[0], op.qubits[1], true);
      break;
    case Gates::cy:
      state.qreg().apply_cy(op.qubits[0], op.qubits[1]);
      break;
    case Gates::cz:
      state.qreg().apply_cz(op.qubits[0], op.qubits[1]);
      break;
    case Gates::csx:
      state.qreg().apply_csx(op.qubits[0], op.qubits[1]);
      break;
    case Gates::cu1:
      state.qreg().apply_cu1(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    case Gates::rxx:
      state.qreg().apply_rxx(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    case Gates::ryy:
      state.qreg().apply_ryy(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    case Gates::rzz:
      state.qreg().apply_rzz(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    case Gates::rzx:
      state.qreg().apply_rzx(op.qubits[0], op.qubits[1],
    		      std::real(op.params[0]));
      break;
    case Gates::pauli:
      apply_pauli(state, op.qubits, op.string_params[0]);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument(
        "MatrixProductState::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

void State::apply_pauli(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits, const std::string& pauli) 
{
  const auto size = qubits.size();
  for (size_t i = 0; i < qubits.size(); ++i) {
    const auto qubit = qubits[size - 1 - i];
    switch (pauli[i]) {
      case 'I':
        break;
      case 'X':
        state.qreg().apply_x(qubit);
        break;
      case 'Y':
        state.qreg().apply_y(qubit);
        break;
      case 'Z':
        state.qreg().apply_z(qubit);
        break;
      default:
        throw std::invalid_argument("invalid Pauli \'" + std::to_string(pauli[i]) + "\'.");
    }
  }
}

void State::apply_matrix(QuantumState::Registers<matrixproductstate_t>& state,const reg_t &qubits, const cmatrix_t &mat) 
{
  if (!qubits.empty() && mat.size() > 0)
    state.qreg().apply_matrix(qubits, mat);
}

void State::apply_matrix(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits, const cvector_t &vmat) 
{
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    state.qreg().apply_diagonal_matrix(qubits, vmat);
  } else {
    state.qreg().apply_matrix(qubits, vmat);
  }
}

void State::apply_kraus(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits,
                   const std::vector<cmatrix_t> &kmats,
                   RngEngine &rng) 
{
  state.qreg().apply_kraus(qubits, kmats, rng);
}


//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

void State::apply_initialize(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits,
			     const cvector_t &params,
			     RngEngine &rng) 
{
  state.qreg().apply_initialize(qubits, params, rng);
}

void State::apply_measure(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits,
                          const reg_t &cmemory,
                          const reg_t &cregister,
                          RngEngine &rng) 
{
  rvector_t rands;
  rands.reserve(qubits.size());
  for (int_t i = 0; i < qubits.size(); ++i)
    rands.push_back(rng.rand(0., 1.));
  reg_t outcome = state.qreg().apply_measure(qubits, rands);
  state.creg().store_measure(outcome, cmemory, cregister);
}

rvector_t State::measure_probs(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits) const 
{
  rvector_t probvector;
  state.qreg().get_probabilities_vector(probvector, qubits);
  return probvector;
}

std::vector<reg_t> State::sample_measure_state(QuantumState::RegistersBase& state_in, const reg_t& qubits,
					 uint_t shots,
                                         RngEngine &rng)
{
  // There are two alternative algorithms for sample measure
  // We choose the one that is optimal relative to the total number 
  // of qubits,and the number of shots.
  // The parameters used below are based on experimentation.
  // The user can override this by setting the parameter "mps_sample_measure_algorithm"
  QuantumState::Registers<matrixproductstate_t>& state = dynamic_cast<QuantumState::Registers<matrixproductstate_t>&>(state_in);

  if (MPS::get_sample_measure_alg() == Sample_measure_alg::PROB && 
      qubits.size() == state.qreg().num_qubits()){
    return sample_measure_all(state, shots, rng);
  }
  return sample_measure_using_apply_measure(state, qubits, shots, rng);
}
	     
std::vector<reg_t> State::
  sample_measure_using_apply_measure(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits, 
				     uint_t shots, 
				     RngEngine &rng) 
{
  std::vector<reg_t> all_samples;
  all_samples.resize(shots);
  // input is always sorted in qasm_controller, therefore, we must return the qubits 
  // to their original location (sorted)
  state.qreg().move_all_qubits_to_sorted_ordering();
  reg_t sorted_qubits = qubits;
  std::sort(sorted_qubits.begin(), sorted_qubits.end());

  std::vector<rvector_t> rnds_list;
  rnds_list.reserve(shots);
  for (int_t i = 0; i < shots; ++i) {
    rvector_t rands;
    rands.reserve(qubits.size());
    for (int_t j = 0; j < qubits.size(); ++j)
      rands.push_back(rng.rand(0., 1.));
    rnds_list.push_back(rands);
  }

  #pragma omp parallel if (BaseState::threads_ > 1) num_threads(BaseState::threads_)
  {
    MPS temp;
    #pragma omp for
    for (int_t i=0; i<static_cast<int_t>(shots);  i++) {
      temp.initialize(state.qreg());
      auto single_result = temp.apply_measure_internal(sorted_qubits, rnds_list[i]);
      all_samples[i] = single_result;
    }
  }

  return all_samples;
}

std::vector<reg_t> State::sample_measure_all(QuantumState::Registers<matrixproductstate_t>& state, uint_t shots, 
					     RngEngine &rng) 
{
  std::vector<reg_t> all_samples;
  all_samples.resize(shots);

  for (uint_t i=0; i<shots;  i++) {
    auto single_result = state.qreg().sample_measure(shots, rng);
    all_samples[i] = single_result;
  }
  return all_samples;
}

void State::apply_snapshot(QuantumState::Registers<matrixproductstate_t>& state, const Operations::Op &op, ExperimentResult &result) 
{
  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument("MatrixProductState::invalid snapshot instruction \'" +
                                op.name + "\'.");
  switch (it -> second) {
  case Snapshots::statevector: {
      snapshot_state(state, op, result, "statevector");
      break;
  }
  case Snapshots::cmemory:
    BaseState::snapshot_creg_memory(state, op, result);
    break;
  case Snapshots::cregister:
    BaseState::snapshot_creg_register(state, op, result);
    break;
  case Snapshots::probs: {
      // get probs as hexadecimal
      snapshot_probabilities(state, op, result, SnapshotDataType::average);
      break;
  }
  case Snapshots::densmat: {
      snapshot_density_matrix(state, op, result, SnapshotDataType::average);
  } break;
  case Snapshots::expval_pauli: {
    snapshot_pauli_expval(state, op, result, SnapshotDataType::average);
  } break;
  case Snapshots::expval_matrix: {
    snapshot_matrix_expval(state, op, result, SnapshotDataType::average);
  }  break;
  case Snapshots::probs_var: {
    // get probs as hexadecimal
    snapshot_probabilities(state, op, result, SnapshotDataType::average_var);
  } break;
  case Snapshots::densmat_var: {
      snapshot_density_matrix(state, op, result, SnapshotDataType::average_var);
  } break;
  case Snapshots::expval_pauli_var: {
    snapshot_pauli_expval(state, op, result, SnapshotDataType::average_var);
  } break;
  case Snapshots::expval_matrix_var: {
    snapshot_matrix_expval(state, op, result, SnapshotDataType::average_var);
  }  break;
  case Snapshots::expval_pauli_shot: {
    snapshot_pauli_expval(state, op, result, SnapshotDataType::pershot);
  } break;
  case Snapshots::expval_matrix_shot: {
    snapshot_matrix_expval(state, op, result, SnapshotDataType::pershot);
  }  break;
  default:
    // We shouldn't get here unless there is a bug in the snapshotset
    throw std::invalid_argument("MatrixProductState::State::invalid snapshot instruction \'" +
				op.name + "\'.");
  }
}

void State::apply_reset(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits,
                        RngEngine &rng) 
{
  state.qreg().reset(qubits, rng);
}

std::pair<uint_t, double>
State::sample_measure_with_prob(QuantumState::Registers<matrixproductstate_t>& state, const reg_t &qubits,
                                RngEngine &rng) 
{
  rvector_t probs = measure_probs(state, qubits);

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
