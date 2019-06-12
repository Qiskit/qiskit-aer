/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
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

#ifndef _tensor_tensor_state_hpp
#define _tensor_tensor_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/json.hpp"
#include "base/state.hpp"
#include "matrix_product_state.hpp"
#include "matrix_product_state.cpp"

namespace AER {
namespace TensorNetworkState { 

// Allowed snapshots enum class
enum class Snapshots {
  statevector, cmemory, cregister,
  probs, probs_var,
  expval_pauli, expval_pauli_var,
  expval_matrix, expval_matrix_var
};


//=========================================================================
// Tensor Network State subclass
//=========================================================================

using tensorstate_t = MPS;

class State : public Base::State<tensorstate_t> {
public:
  using BaseState = Base::State<tensorstate_t>;
  
  State() = default;

  State(uint_t num_qubits) {
    qreg_.initialize((uint_t)num_qubits);
  }
  virtual ~State() = default;


  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {
	  return "tensorstate";
  }

  bool empty() const {
    return qreg_.empty();
  }

  // Return the set of qobj instruction types supported by the State
  virtual Operations::OpSet::optypeset_t allowed_ops() const override {
  	return Operations::OpSet::optypeset_t({

	  //TODO: Review these operations
      Operations::OpType::gate,
      Operations::OpType::measure,
      Operations::OpType::reset,
      Operations::OpType::snapshot,
      Operations::OpType::barrier,
      Operations::OpType::bfunc,
      Operations::OpType::roerror,
      Operations::OpType::matrix,
      Operations::OpType::kraus
    });
  }

  // Return the set of qobj gate instruction names supported by the State
  virtual stringset_t allowed_gates() const override {
	stringset_t allowed_gates;
    for(auto& gate: gateset_){
      allowed_gates.insert(gate.first);
    }
    return allowed_gates;
  }

  // Return the set of qobj snapshot types supported by the State
  virtual stringset_t allowed_snapshots() const override {
	//TODO: Review this
    return {"statevector", "memory", "register",
            "probabilities", "probabilities_with_variance",
            "expectation_value_pauli", "expectation_value_pauli_with_variance",
            "expectation_value_matrix", "expectation_value_matrix_with_variance"};
  }

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exception will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         OutputData &data,
                         RngEngine &rng) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit state given as a complex std::vector
  virtual void initialize_qreg(uint_t num_qubits, const tensorstate_t &state) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
    virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops) override;

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
  virtual void apply_snapshot(const Operations::Op &op, OutputData &data);

  // Apply a matrix to given qubits (identity on all other qubits)
  // We assume matrix to be 2x2
  void apply_matrix(const reg_t &qubits, const cmatrix_t & mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const reg_t &qubits, const cvector_t & vmat);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits,
                   const std::vector<cmatrix_t> &krausops,
                   RngEngine &rng);

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
                              OutputData &data,
                              bool variance);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const Operations::Op &op,
                             OutputData &data,
                             bool variance);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(const Operations::Op &op,
                              OutputData &data,
                              bool variance);

  // Snapshot the state vector
  void snapshot_state(const Operations::Op &op,
		      OutputData &data,
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

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 14;

  // QubitVector sample measure index size
  int sample_measure_index_size_ = 10;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-15;

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
  {"cu", Gates::cu},     // Controlled-U gate
  {"cu1", Gates::cu},     // Controlled-U gate
  {"swap", Gates::swap}, // SWAP gate
  {"su4", Gates::su4},   // general su4 matrix gate
  // Three-qubit gates
  // TODO: No Toffoli support?
  //{"ccx", Gates::ccx}    // Controlled-CX gate (Toffoli)
});

const stringmap_t<Snapshots> State::snapshotset_({
  {"statevector", Snapshots::statevector},
  {"probabilities", Snapshots::probs},
  {"expectation_value_pauli", Snapshots::expval_pauli},
  {"expectation_value_matrix", Snapshots::expval_matrix},
  {"probabilities_with_variance", Snapshots::probs_var},
  {"expectation_value_pauli_with_variance", Snapshots::expval_pauli_var},
  {"expectation_value_matrix_with_variance", Snapshots::expval_matrix_var},
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
  initialize_omp();
  qreg_.initialize((uint_t)num_qubits);
}

void State::initialize_qreg(uint_t num_qubits, const tensorstate_t &state) {
  // Check dimension of state
  if (qreg_.num_qubits() != num_qubits) {
    throw std::invalid_argument("TensorNetwork::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();
  //qreg_.initialize((uint_t)num_qubits, state);
  cout << "initialize with state not supported yet" <<endl;
}

void State::initialize_omp() {
  qreg_.set_omp_threshold(omp_qubit_threshold_);
  if (BaseState::threads_ > 0)
    qreg_.set_omp_threads(BaseState::threads_); // set allowed OMP threads in MPS
}

size_t State::required_memory_mb(uint_t num_qubits,
			      const std::vector<Operations::Op> &ops) {
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
  JSON::get_value(json_chop_threshold_, "chop_threshold", config);
  qreg_.set_json_chop_threshold(json_chop_threshold_);

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold", config);

  // Set the sample measure indexing size
  int index_size;
  if (JSON::get_value(index_size, "statevector_sample_measure_opt", config)) {
    qreg_.set_sample_measure_index_size(index_size);
  };

  // Enable sorted gate optimzations
  bool gate_opt = false;
  JSON::get_value(gate_opt, "statevector_gate_opt", config);
  if (gate_opt)
    qreg_.enable_gate_opt();
}

//=========================================================================
// Implementation: apply operations
//=========================================================================

void State::apply_ops(const std::vector<Operations::Op> &ops,
                      OutputData &data,
                      RngEngine &rng) {

  // Simple loop over vector of input operations
  for (const auto op: ops) {
    switch (op.type) {
      case Operations::OpType::barrier:
        break;
      case Operations::OpType::reset:
        apply_reset(op.qubits, rng);
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
        //if (BaseState::creg_.check_conditional(op))
          apply_gate(op);
        break;
      case Operations::OpType::snapshot:
        apply_snapshot(op, data);
        break;
      case Operations::OpType::matrix:
        apply_matrix(op.qubits, op.mats[0]);
        break;
      case Operations::OpType::kraus:
        apply_kraus(op.qubits, op.mats, rng);
        break;
      default:
        throw std::invalid_argument("TensorNetworkState::State::invalid instruction \'" +
                                    op.name + "\'.");
    }
  }
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

void State::snapshot_pauli_expval(const Operations::Op &op,
				  OutputData &data,
				  bool variance){
  if (op.params_expval_pauli.empty()) {
    throw std::invalid_argument("Invalid expval snapshot (Pauli components are empty).");
  }

  //Cache the current quantum state
  //BaseState::qreg_.checkpoint(); 
  //bool first = true; // flag for first pass so we don't unnecessarily revert from checkpoint

  //Compute expval components
  double expval = 0;
  string pauli_matrices;

  for (const auto &param : op.params_expval_pauli) {
    pauli_matrices += param.second;
  }
  expval = qreg_.Expectation_value(op.qubits, pauli_matrices);
    // Pauli expectation values should always be real for a valid state
    // so we truncate the imaginary part
  //expval += coeff * std::real(BaseState::qreg_.inner_product());
  data.add_singleshot_snapshot("expectation_value", op.string_params[0], expval);
  
  //qreg_.revert(false);
  // Revert to original state
  //BaseState::qreg_.revert(false);
}

void State::snapshot_matrix_expval(const Operations::Op &op,
				   OutputData &data,
				   bool variance){
  if (op.params_expval_matrix.empty()) {
    throw std::invalid_argument("Invalid matrix snapshot (components are empty).");
  }

  for (const auto &param : op.params_expval_matrix) {
    complex_t coeff = param.first;

    for (const auto &pair: param.second) {
      const reg_t &qubits = pair.first;
      const cmatrix_t &mat = pair.second;
      double expval = 0;
      expval = qreg_.Expectation_value(qubits, mat);
    // Pauli expectation values should always be real for a valid state
    // so we truncate the imaginary part
    //expval += coeff * std::real(BaseState::qreg_.inner_product());
      data.add_singleshot_snapshot("expectation_value", op.string_params[0], expval);
    }
  }
}

void State::snapshot_state(const Operations::Op &op,
			   OutputData &data,
			   std::string name) {
  TensorNetworkState::MPS_Tensor full_tensor = qreg_.state_vec(0, qreg_.num_qubits()-1);
  cvector_t statevector;
  qreg_.full_state_vector(statevector);
  data.add_singleshot_snapshot("statevector", op.string_params[0], statevector);
}

void State::snapshot_probabilities(const Operations::Op &op,
				   OutputData &data,
				   bool variance) {
  TensorNetworkState::MPS_Tensor full_tensor = qreg_.state_vec(0, qreg_.num_qubits()-1);
  rvector_t prob_vector;
  qreg_.probabilities_vector(prob_vector);
  data.add_singleshot_snapshot("probabilities", op.string_params[0], prob_vector);
}

void State::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
      "TensorNetwork::State::invalid gate instruction \'" + op.name + "\'.");

  switch (it -> second) {
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
      qreg_.apply_swap(op.qubits[0], op.qubits[1]);
      break;
    case Gates::cz:
      qreg_.apply_cz(op.qubits[0], op.qubits[1]);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument(
        "TensorNetwork::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

void State::apply_matrix(const reg_t &qubits, const cmatrix_t &mat) {
  if (!qubits.empty() && mat.size() > 0) {
    apply_matrix(qubits, Utils::vectorize_matrix(mat));
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

void State::apply_gate_u3(uint_t qubit, double theta, double phi, double lambda) {
  apply_matrix(reg_t({qubit}), Utils::Matrix::u3(theta, phi, lambda));
}

void State::apply_gate_phase(uint_t qubit, complex_t phase) {
  cvector_t diag = {{1., phase}};
  apply_matrix(reg_t({qubit}), diag);
}


//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

void State::apply_measure(const reg_t &qubits,
                          const reg_t &cmemory,
                          const reg_t &cregister,
                          RngEngine &rng) {

//	  double threshold = 1e-10;
//	  vector<uint_t> indexes {0,2};
//	  string matrices = "XYZ" ;
//	  cout << "Expectation value on XYZ qubits 0-2 = " << endl ;
//	  cout << qreg_.Expectation_value(indexes,matrices) << endl;
//
//	  result = qreg_.state_vec(0,qreg_.num_qubits()-1);
//	  result.print(true);

  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Implement measurement update
  measure_reset_update(qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());
  BaseState::creg_.store_measure(outcome, cmemory, cregister);
  
}

rvector_t State::measure_probs(const reg_t &qubits) const {
  return qreg_.probabilities(qubits);
}

std::vector<reg_t> State::sample_measure(const reg_t &qubits,
                                         uint_t shots,
                                         RngEngine &rng) {
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);

  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  for (uint_t i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));

  reg_t allbit_samples = qreg_.sample_measure(rnds);

  for (int_t val : allbit_samples) {
    reg_t allbit_sample = Utils::int2reg(val, 2, qreg_.num_qubits());
    reg_t sample;
    sample.reserve(qubits.size());
    for (uint_t qubit : qubits) {
      sample.push_back(allbit_sample[qubit]);
    }
    all_samples.push_back(sample);
  }
  
  return all_samples;
}

void State::apply_snapshot(const Operations::Op &op, OutputData &data) {
  // Look for snapshot type in snapshotset

  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument("Tensor_Network_State::invalid snapshot instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
  case Snapshots::statevector: {
      snapshot_state(op, data, "statevector"); 
      break; 
      }
      /*    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(op, data);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(op, data);
      break;
      */
  case Snapshots::probs: {
      // get probs as hexadecimal
      snapshot_probabilities(op, data, false);
      break;
    } 
    case Snapshots::expval_pauli: {
      snapshot_pauli_expval(op, data, false);
      break;
    }
    case Snapshots::expval_matrix: {
      snapshot_matrix_expval(op, data, false);
      break;
    }
      /*
    case Snapshots::probs_var: {
      // get probs as hexadecimal
      snapshot_probabilities(op, data, true);
    } break;
    case Snapshots::expval_pauli_var: {
      snapshot_pauli_expval(op, data, true);
    } break;
    case Snapshots::expval_matrix_var: {
      snapshot_matrix_expval(op, data, true);
      }  break;*/
    default:
      // We shouldn't get here unless there is a bug in the snapshotset
      throw std::invalid_argument("TensorNetworkState::State::invalid snapshot instruction \'" +
                                  op.name + "\'."); 
  }
}

void State::apply_reset(const reg_t &qubits,
                        RngEngine &rng) {

  // Simulate unobserved measurement
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Apply update tp reset state
  measure_reset_update(qubits, 0, meas.first, meas.second);
}

std::pair<uint_t, double>
State::sample_measure_with_prob(const reg_t &qubits,
                                RngEngine &rng) {
  rvector_t probs = measure_probs(qubits);

  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

void State::measure_reset_update(const std::vector<uint_t> &qubits,
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
      qreg_.apply_x(qubits[0]);
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

//=========================================================================
// Implementation: Kraus Noise
// This function has not been checked yet
//=========================================================================
void State::apply_kraus(const reg_t &qubits,
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
    double p = qreg_.norm(qubits, vmat);
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
} // end namespace TensorNetworkState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
