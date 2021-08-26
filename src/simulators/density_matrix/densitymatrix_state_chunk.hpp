/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_densitymatrix_state_chunk_hpp
#define _aer_densitymatrix_state_chunk_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "simulators/state_chunk.hpp"
#include "densitymatrix.hpp"
#ifdef AER_THRUST_SUPPORTED
#include "densitymatrix_thrust.hpp"
#endif

namespace AER {
namespace DensityMatrixChunk {

using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {OpType::gate, OpType::measure,
     OpType::reset, OpType::snapshot,
     OpType::barrier, OpType::bfunc,
     OpType::roerror, OpType::matrix,
     OpType::diagonal_matrix, OpType::kraus,
     OpType::superop, OpType::set_statevec,
     OpType::set_densmat, OpType::save_expval,
     OpType::save_expval_var, OpType::save_densmat,
     OpType::save_probs, OpType::save_probs_ket,
     OpType::save_amps_sq, OpType::save_state
     },
    // Gates
    {"U",    "CX",  "u1", "u2",  "u3", "u",   "cx",   "cy",  "cz",
     "swap", "id",  "x",  "y",   "z",  "h",   "s",    "sdg", "t",
     "tdg",  "ccx", "r",  "rx",  "ry", "rz",  "rxx",  "ryy", "rzz",
     "rzx",  "p",   "cp", "cu1", "sx", "sxdg", "x90", "delay", "pauli"},
    // Snapshots
    {"density_matrix", "memory", "register", "probabilities",
     "probabilities_with_variance", "expectation_value_pauli",
     "expectation_value_pauli_with_variance"});


//=========================================================================
// DensityMatrix State subclass
//=========================================================================

template <class densmat_t = QV::DensityMatrix<double>>
class State : public Base::StateChunk<densmat_t> {
public:
  using BaseState = Base::StateChunk<densmat_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() {}

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {return densmat_t::name();}

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

  auto move_to_matrix();
  auto copy_to_matrix();
protected:

  template <typename list_t>
  void initialize_from_vector(const list_t &vec);

  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------
  virtual void apply_op(const int_t iChunk,const Operations::Op &op,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops) override;

  //swap between chunks
  virtual void apply_chunk_swap(const reg_t &qubits) override;

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(const uint_t iChunk, const Operations::Op &op);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual void apply_measure(const reg_t &qubits,
                             const reg_t &cmemory,
                             const reg_t &cregister,
                             RngEngine &rng);

  // Reset the specified qubits to the |0> state by tracing out qubits
  void apply_reset(const int_t iChunk, const reg_t &qubits);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const Operations::Op &op,
                              ExperimentResult &result,
                              bool last_op = false);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cmatrix_t & mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t & vmat);

  //apply diagonal matrix
  void apply_diagonal_unitary_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t & diag);

  // Apply a Kraus error operation
  void apply_kraus(const reg_t &qubits, const std::vector<cmatrix_t> &kraus);

  // Apply an N-qubit Pauli gate
  void apply_pauli(const reg_t &qubits, const std::string &pauli);

  // apply phase
  void apply_phase(const uint_t iChunk,const uint_t qubit, const complex_t phase);
  void apply_phase(const uint_t iChunk,const reg_t& qubits, const complex_t phase);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current density matrix
  void apply_save_state(const Operations::Op &op,
                        ExperimentResult &result,
                        bool last_op = false);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const Operations::Op &op,
                                 ExperimentResult &result,
                                 bool last_op = false);

  // Helper function for computing expectation value
  void apply_save_probs(const Operations::Op &op,
                        ExperimentResult &result);

  // Helper function for saving amplitudes squared
  void apply_save_amplitudes_sq(const Operations::Op &op,
                                ExperimentResult &result);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string& pauli) override;

  // Return the reduced density matrix for the simulator
  cmatrix_t reduced_density_matrix(const reg_t &qubits, bool last_op = false);
  cmatrix_t reduced_density_matrix_helper(const reg_t &qubits,
                                          const reg_t &qubits_sorted);

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

  // Snapshot reduced density matrix
  void snapshot_density_matrix(const Operations::Op &op,
                               ExperimentResult &result,
                               bool last_op = false);

  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(const Operations::Op &op,
                              ExperimentResult &result,
                              bool variance);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const Operations::Op &op,
                             ExperimentResult &result,
                             bool variance);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(const Operations::Op &op,
                              ExperimentResult &result,
                              bool variance);


  //-----------------------------------------------------------------------
  // Single-qubit gate helpers
  //-----------------------------------------------------------------------

  // Apply a waltz gate specified by parameters u3(theta, phi, lambda)
  void apply_gate_u3(const int_t iChunk, const uint_t qubit, const double theta, const double phi,
                     const double lambda);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // OpenMP qubit threshold
  // NOTE: This is twice the number of qubits in the DensityMatrix since it
  // refers to the equivalent qubit number in the underlying QubitVector class
  int omp_qubit_threshold_ = 14;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  int qubit_scale() override
  {
    return 2;
  }

  virtual bool is_applied_to_each_chunk(const Operations::Op &op);
};

//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

template <class densmat_t>
bool State<densmat_t>::is_applied_to_each_chunk(const Operations::Op &op)
{
  if(op.type == Operations::OpType::reset){
    return true;
  }
  return BaseState::is_applied_to_each_chunk(op);
}

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------
template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits) 
{
  int_t i;

  initialize_omp();

  if(BaseState::chunk_bits_ == BaseState::num_qubits_){
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_);
      BaseState::qregs_[i].zero();
      BaseState::qregs_[i].initialize();
    }
  }
  else{   //multi-chunk distribution
    for(i=0;i<BaseState::num_local_chunks_;i++){
      //this function should be called in-order
      BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_);
    }

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      if(BaseState::global_chunk_index_ + i == 0 || this->num_qubits_ == this->chunk_bits_){
        BaseState::qregs_[i].initialize();
      }
      else{
        BaseState::qregs_[i].zero();
      }
    }
  }
}

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits,
                                   const densmat_t &state) 
{
  // Check dimension of state
  if (state.num_qubits() != num_qubits) {
    throw std::invalid_argument("DensityMatrix::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();

  int_t iChunk;
  if(BaseState::chunk_bits_ == BaseState::num_qubits_){
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);
      BaseState::qregs_[iChunk].initialize_from_data(state.data(), 1ULL << 2 * num_qubits);
    }
  }
  else{   //multi-chunk distribution
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      //this function should be called in-order
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);
    }

    auto input = state.copy_to_matrix();

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      uint_t irow_chunk = ((iChunk + BaseState::global_chunk_index_) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_))) << (BaseState::chunk_bits_);
      uint_t icol_chunk = ((iChunk + BaseState::global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_)))-1)) << (BaseState::chunk_bits_);

      //copy part of state for this chunk
      uint_t i,row,col;
      cvector_t tmp(1ull << (BaseState::chunk_bits_*2));
      for(i=0;i<(1ull << (BaseState::chunk_bits_*2));i++){
        uint_t icol = i & ((1ull << (BaseState::chunk_bits_))-1);
        uint_t irow = i >> (BaseState::chunk_bits_);
        tmp[i] = input[icol_chunk + icol + ((irow_chunk + irow) << (BaseState::num_qubits_))];
      }
      BaseState::qregs_[iChunk].initialize_from_vector(tmp);
    }
  }
}

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits,
                                        const cmatrix_t &state) 
{
  if (state.size() != 1ULL << 2 * num_qubits) {
    throw std::invalid_argument("DensityMatrix::State::initialize: initial state does not match qubit number");
  }
  initialize_omp();

  int_t iChunk;
  if(BaseState::chunk_bits_ == BaseState::num_qubits_){
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);
      BaseState::qregs_[iChunk].initialize_from_matrix(state);
    }
  }
  else{   //multi-chunk distribution
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      //this function should be called in-order
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);
    }

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      uint_t irow_chunk = ((iChunk + BaseState::global_chunk_index_) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_))) << (BaseState::chunk_bits_);
      uint_t icol_chunk = ((iChunk + BaseState::global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_)))-1)) << (BaseState::chunk_bits_);

      //copy part of state for this chunk
      uint_t i,row,col;
      cvector_t tmp(1ull << (BaseState::chunk_bits_*2));
      for(i=0;i<(1ull << (BaseState::chunk_bits_*2));i++){
        uint_t icol = i & ((1ull << (BaseState::chunk_bits_))-1);
        uint_t irow = i >> (BaseState::chunk_bits_);
        tmp[i] = state[icol_chunk + icol + ((irow_chunk + irow) << (BaseState::num_qubits_))];
      }
      BaseState::qregs_[iChunk].initialize_from_vector(tmp);
    }
  }
}

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits,
                                        const cvector_t &state) 
{
  if (state.size() != 1ULL << 2 * num_qubits) {
    throw std::invalid_argument("DensityMatrix::State::initialize: initial state does not match qubit number");
  }

  initialize_omp();

  int_t iChunk;
  if(BaseState::chunk_bits_ == BaseState::num_qubits_){
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);
      BaseState::qregs_[iChunk].initialize_from_vector(state);
    }
  }
  else{   //multi-chunk distribution
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      //this function should be called in-order
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);
    }

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      uint_t irow_chunk = ((iChunk + BaseState::global_chunk_index_) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_))) << (BaseState::chunk_bits_);
      uint_t icol_chunk = ((iChunk + BaseState::global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_)))-1)) << (BaseState::chunk_bits_);

      //copy part of state for this chunk
      uint_t i,row,col;
      cvector_t tmp(1ull << (BaseState::chunk_bits_*2));
      for(i=0;i<(1ull << (BaseState::chunk_bits_*2));i++){
        uint_t icol = i & ((1ull << (BaseState::chunk_bits_))-1);
        uint_t irow = i >> (BaseState::chunk_bits_);
        tmp[i] = state[icol_chunk + icol + ((irow_chunk + irow) << (BaseState::num_qubits_))];
      }
      BaseState::qregs_[iChunk].initialize_from_vector(tmp);
    }
  }

}

template <class densmat_t>
void State<densmat_t>::initialize_omp() 
{
  uint_t i;
  for(i=0;i<BaseState::num_local_chunks_;i++){
    BaseState::qregs_[i].set_omp_threshold(omp_qubit_threshold_);
    if (BaseState::threads_ > 0)
      BaseState::qregs_[i].set_omp_threads(BaseState::threads_); // set allowed OMP threads in qubitvector
  }
}

template <class densmat_t>
template <typename list_t>
void State<densmat_t>::initialize_from_vector(const list_t &vec)
{
  if((1ull << (BaseState::num_qubits_*2)) == vec.size()){
    BaseState::initialize_from_vector(vec);
  }
  else if((1ull << (BaseState::num_qubits_*2)) == vec.size() * vec.size()) {
    int_t iChunk;
    if(BaseState::chunk_bits_ == BaseState::num_qubits_){
      for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
        BaseState::qregs_[iChunk].initialize_from_vector(AER::Utils::tensor_product(AER::Utils::conjugate(vec), vec));
      }
    }
    else{   //multi-chunk distribution

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
      for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
        uint_t irow_chunk = ((iChunk + BaseState::global_chunk_index_) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_))) << (BaseState::chunk_bits_);
        uint_t icol_chunk = ((iChunk + BaseState::global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_)))-1)) << (BaseState::chunk_bits_);

        //copy part of state for this chunk
        uint_t i,row,col;
        list_t vec1(1ull << BaseState::chunk_bits_);
        list_t vec2(1ull << BaseState::chunk_bits_);

        for(i=0;i<(1ull << BaseState::chunk_bits_);i++){
          vec1[i] = vec[(irow_chunk << BaseState::chunk_bits_) + i];
          vec2[i] = std::conj(vec[(icol_chunk << BaseState::chunk_bits_) + i]);
        }
        BaseState::qregs_[iChunk].initialize_from_vector(AER::Utils::tensor_product(vec1, vec2));
      }
    }
  }
  else {
    throw std::runtime_error("DensityMatrixChunk::initialize input vector is incorrect length. Expected: " +
                             std::to_string((1ull << (BaseState::num_qubits_*2))) + " Received: " +
                             std::to_string(vec.size()));
  }
}


template <class densmat_t>
auto State<densmat_t>::move_to_matrix()
{
  if(BaseState::num_global_chunks_ == 1){
    return BaseState::qregs_[0].move_to_matrix();
  }
  return BaseState::apply_to_matrix(false);
}

template <class densmat_t>
auto State<densmat_t>::copy_to_matrix()
{
  if(BaseState::num_global_chunks_ == 1){
    return BaseState::qregs_[0].copy_to_matrix();
  }
  return BaseState::apply_to_matrix(true);
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
void State<densmat_t>::set_config(const json_t &config) 
{
  BaseState::set_config(config);

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "chop_threshold", config);
  uint_t i;
  for(i=0;i<BaseState::num_local_chunks_;i++){
    BaseState::qregs_[i].set_json_chop_threshold(json_chop_threshold_);
  }

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold", config);

}


//=========================================================================
// Implementation: apply operations
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_op(const int_t iChunk,const Operations::Op &op,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops) {
  if (BaseState::creg_.check_conditional(op)) {
    switch (op.type) {
      case Operations::OpType::barrier:
        break;
      case Operations::OpType::reset:
        apply_reset(iChunk,op.qubits);
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
        apply_gate(iChunk,op);
        break;
      case Operations::OpType::snapshot:
        apply_snapshot(op, result, final_ops);
        break;
      case Operations::OpType::matrix:
        apply_matrix(iChunk,op.qubits, op.mats[0]);
        break;
      case Operations::OpType::diagonal_matrix:
        apply_diagonal_unitary_matrix(iChunk,op.qubits, op.params);
        break;
      case Operations::OpType::superop:
        BaseState::qregs_[iChunk].apply_superop_matrix(op.qubits, Utils::vectorize_matrix(op.mats[0]));
        break;
      case OpType::kraus:
        apply_kraus(op.qubits, op.mats);
        break;
      case OpType::set_statevec:
        initialize_from_vector(op.params);
        break;
      case OpType::set_densmat:
        BaseState::initialize_from_matrix(op.mats[0]);
        break;
      case Operations::OpType::save_expval:
      case Operations::OpType::save_expval_var:
        BaseState::apply_save_expval(op, result);
        break;
      case Operations::OpType::save_state:
        apply_save_state(op, result, final_ops);
        break;
      case Operations::OpType::save_densmat:
        apply_save_density_matrix(op, result, final_ops);
        break;
      case Operations::OpType::save_probs:
      case Operations::OpType::save_probs_ket:
        apply_save_probs(op, result);
        break;
      case Operations::OpType::save_amps_sq:
          apply_save_amplitudes_sq(op, result);
          break;
      default:
        throw std::invalid_argument("DensityMatrix::State::invalid instruction \'" +
                                    op.name + "\'.");
    }
  }
}

//swap between chunks
template <class densmat_t>
void State<densmat_t>::apply_chunk_swap(const reg_t &qubits)
{
  uint_t q0,q1;
  q0 = qubits[0];
  q1 = qubits[1];

  std::swap(BaseState::qubit_map_[q0],BaseState::qubit_map_[q1]);

  if(qubits[0] >= BaseState::chunk_bits_){
    q0 += BaseState::chunk_bits_;
  }
  if(qubits[1] >= BaseState::chunk_bits_){
    q1 += BaseState::chunk_bits_;
  }
  reg_t qs0 = {{q0, q1}};
  BaseState::apply_chunk_swap(qs0);

  if(qubits[0] >= BaseState::chunk_bits_){
    q0 += (BaseState::num_qubits_ - BaseState::chunk_bits_);
  }
  else{
    q0 += BaseState::chunk_bits_;
  }
  if(qubits[1] >= BaseState::chunk_bits_){
    q1 += (BaseState::num_qubits_ - BaseState::chunk_bits_);
  }
  else{
    q1 += BaseState::chunk_bits_;
  }
  reg_t qs1 = {{q0, q1}};
  BaseState::apply_chunk_swap(qs1);
}

//=========================================================================
// Implementation: Save data
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_save_probs(const Operations::Op &op,
                                            ExperimentResult &result) {
  auto probs = measure_probs(op.qubits);
  if (op.type == Operations::OpType::save_probs_ket) {
    BaseState::save_data_average(result, op.string_params[0],
                                 Utils::vec2ket(probs, json_chop_threshold_, 16),
                                 op.save_type);
  } else {
    BaseState::save_data_average(result, op.string_params[0],
                                 std::move(probs), op.save_type);
  }
}

template <class densmat_t>
void State<densmat_t>::apply_save_amplitudes_sq(const Operations::Op &op,
                                                ExperimentResult &result) 
{
  if (op.int_params.empty()) {
    throw std::invalid_argument("Invalid save_amplitudes_sq instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  int_t iChunk;
  rvector_t amps_sq(size,0);
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
  for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
    uint_t irow,icol;
    irow = (BaseState::global_chunk_index_ + iChunk) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
    icol = (BaseState::global_chunk_index_ + iChunk) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
    if(irow != icol)
      continue;

#pragma omp parallel for if (size > pow(2, omp_qubit_threshold_) &&        \
                                 BaseState::threads_ > 1)                       \
                          num_threads(BaseState::threads_)
    for (int_t i = 0; i < size; ++i) {
      uint_t idx = BaseState::mapped_index(op.int_params[i]);
      if(idx >= (irow << BaseState::chunk_bits_) && idx < ((irow+1) << BaseState::chunk_bits_))
        amps_sq[i] = BaseState::qregs_[iChunk].probability(idx - (irow << BaseState::chunk_bits_));
    }
  }
#ifdef AER_MPI
  BaseState::reduce_sum(amps_sq);
#endif
  BaseState::save_data_average(result, op.string_params[0],
                               std::move(amps_sq), op.save_type);
}

template <class densmat_t>
double State<densmat_t>::expval_pauli(const reg_t &qubits,
                                      const std::string& pauli) 
{
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;
  std::string pauli_in_chunk;
  std::string pauli_out_chunk;
  int_t i,n;
  double expval(0.);

  //get inner/outer chunk pauli string
  n = pauli.size();
  for(i=0;i<n;i++){
    if(qubits[i] < BaseState::chunk_bits_){
      qubits_in_chunk.push_back(qubits[i]);
      pauli_in_chunk.push_back(pauli[n-i-1]);
    }
    else{
      qubits_out_chunk.push_back(qubits[i]);
      pauli_out_chunk.push_back(pauli[n-i-1]);
    }
  }

  int_t nrows = 1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_));

  if(qubits_out_chunk.size() > 0){  //there are bits out of chunk
    std::complex<double> phase = 1.0;

    std::reverse(pauli_out_chunk.begin(),pauli_out_chunk.end());
    std::reverse(pauli_in_chunk.begin(),pauli_in_chunk.end());

    uint_t x_mask, z_mask, num_y, x_max;
    std::tie(x_mask, z_mask, num_y, x_max) = AER::QV::pauli_masks_and_phase(qubits_out_chunk, pauli_out_chunk);

    z_mask >>= (BaseState::chunk_bits_);
    if(x_mask != 0){
      x_mask >>= (BaseState::chunk_bits_);
      x_max -= (BaseState::chunk_bits_);

      AER::QV::add_y_phase(num_y,phase);

      const uint_t mask_u = ~((1ull << (x_max + 1)) - 1);
      const uint_t mask_l = (1ull << x_max) - 1;

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) reduction(+:expval)
      for(i=0;i<nrows/2;i++){
        uint_t irow = ((i << 1) & mask_u) | (i & mask_l);
        uint_t iChunk = (irow ^ x_mask) + irow * nrows;

        if(BaseState::chunk_index_begin_[BaseState::distributed_rank_] <= iChunk && BaseState::chunk_index_end_[BaseState::distributed_rank_] > iChunk){  //on this process
          double sign = 2.0;
          if (z_mask && (AER::Utils::popcount(irow & z_mask) & 1))
            sign = -2.0;
          expval += sign * BaseState::qregs_[iChunk-BaseState::global_chunk_index_].expval_pauli_non_diagonal_chunk(qubits_in_chunk, pauli_in_chunk,phase);
        }
      }
    }
    else{
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) reduction(+:expval)
      for(i=0;i<nrows;i++){
        uint_t iChunk = i * (nrows+1);
        if(BaseState::chunk_index_begin_[BaseState::distributed_rank_] <= iChunk && BaseState::chunk_index_end_[BaseState::distributed_rank_] > iChunk){  //on this process
          double sign = 1.0;
          if (z_mask && (AER::Utils::popcount(i & z_mask) & 1))
            sign = -1.0;
          expval += sign * BaseState::qregs_[iChunk-BaseState::global_chunk_index_].expval_pauli(qubits_in_chunk, pauli_in_chunk,1.0);
        }
      }
    }
  }
  else{ //all bits are inside chunk
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) reduction(+:expval)
    for(i=0;i<nrows;i++){
      uint_t iChunk = i * (nrows+1);
      if(BaseState::chunk_index_begin_[BaseState::distributed_rank_] <= iChunk && BaseState::chunk_index_end_[BaseState::distributed_rank_] > iChunk){  //on this process
        expval += BaseState::qregs_[iChunk-BaseState::global_chunk_index_].expval_pauli(qubits, pauli,1.0);
      }
    }
  }

#ifdef AER_MPI
  BaseState::reduce_sum(expval);
#endif
  return expval;
}

template <class densmat_t>
void State<densmat_t>::apply_save_density_matrix(const Operations::Op &op,
                                                 ExperimentResult &result,
                                                 bool last_op) 
{
  BaseState::save_data_average(result, op.string_params[0],
                               reduced_density_matrix(op.qubits, last_op),
                               op.save_type);
}

template <class densmat_t>
void State<densmat_t>::apply_save_state(const Operations::Op &op,
                                        ExperimentResult &result,
                                        bool last_op) 
{
  // Renamp single data type to average
  Operations::Op op_cpy = op;
  switch (op.save_type) {
    case Operations::DataSubType::single:
      op_cpy.save_type = Operations::DataSubType::average;
      break;
    case Operations::DataSubType::c_single:
      op_cpy.save_type = Operations::DataSubType::c_average;
      break;
    default:
      break;
  }
  // Default key
  op_cpy.string_params[0] = (op.string_params[0] == "_method_")
                              ? "density_matrix"
                              : op.string_params[0];
  apply_save_density_matrix(op_cpy, result, last_op);
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_snapshot(const Operations::Op &op,
                                      ExperimentResult &result,
                                      bool last_op) 
{
  // Look for snapshot type in snapshotset
  auto it = DensityMatrix::State<densmat_t>::snapshotset_.find(op.name);
  if (it == DensityMatrix::State<densmat_t>::snapshotset_.end())
    throw std::invalid_argument("DensityMatrixState::invalid snapshot instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
    case DensityMatrix::Snapshots::densitymatrix:
      snapshot_density_matrix(op, result, last_op);
      break;
    case DensityMatrix::Snapshots::cmemory:
      BaseState::snapshot_creg_memory(op, result);
      break;
    case DensityMatrix::Snapshots::cregister:
      BaseState::snapshot_creg_register(op, result);
      break;
    case DensityMatrix::Snapshots::probs:
      // get probs as hexadecimal
      snapshot_probabilities(op, result, false);
      break;
    case DensityMatrix::Snapshots::probs_var:
      // get probs as hexadecimal
      snapshot_probabilities(op, result, true);
      break;
    case DensityMatrix::Snapshots::expval_pauli: {
      snapshot_pauli_expval(op, result, false);
    } break;
    case DensityMatrix::Snapshots::expval_pauli_var: {
      snapshot_pauli_expval(op, result, true);
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
                                              ExperimentResult &result,
                                              bool variance) 
{
  // get probs as hexadecimal
  auto probs = Utils::vec2ket(measure_probs(op.qubits),
                              json_chop_threshold_, 16);

  result.legacy_data.add_average_snapshot("probabilities",
                            op.string_params[0],
                            BaseState::creg_.memory_hex(),
                            std::move(probs),
                            variance);
}


template <class densmat_t>
void State<densmat_t>::snapshot_pauli_expval(const Operations::Op &op,
                                             ExperimentResult &result,
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
    expval += coeff * expval_pauli(op.qubits, pauli);
  }

  // Add to snapshot
  Utils::chop_inplace(expval, json_chop_threshold_);
  result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::creg_.memory_hex(), expval, variance);
}

template <class densmat_t>
void State<densmat_t>::snapshot_density_matrix(const Operations::Op &op,
                                               ExperimentResult &result,
                                               bool last_op)
{
  result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                            BaseState::creg_.memory_hex(),
                            reduced_density_matrix(op.qubits, last_op), false);
}


template <class densmat_t>
cmatrix_t State<densmat_t>::reduced_density_matrix(const reg_t& qubits, bool last_op) 
{
  cmatrix_t reduced_state;
  // Check if tracing over all qubits
  if (qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    std::complex<double> sum = 0.0;
    for(int_t i=0;i<BaseState::num_local_chunks_;i++){
      sum += BaseState::qregs_[i].trace();
    }
#ifdef AER_MPI
    BaseState::reduce_sum(sum);
#endif
    reduced_state[0] = sum;
  } else {
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

    if ((qubits.size() == BaseState::num_qubits_) && (qubits == qubits_sorted)) {
      if (last_op) {
        reduced_state = move_to_matrix();
      } else {
        reduced_state = copy_to_matrix();
      }
    } else {
      reduced_state = reduced_density_matrix_helper(qubits, qubits_sorted);
    }
  }
  return reduced_state;
}
  
template <class densmat_t>
cmatrix_t State<densmat_t>::reduced_density_matrix_helper(const reg_t &qubits,
                                          const reg_t &qubits_sorted) 
{
  int_t iChunk;
  uint_t size = 1ull << (BaseState::chunk_bits_*2);
  uint_t mask = (1ull << (BaseState::chunk_bits_)) - 1;
  uint_t num_threads = BaseState::qregs_[0].get_omp_threads();

  size_t size_required = (sizeof(std::complex<double>) << (qubits.size()*2)) + (sizeof(std::complex<double>) << (BaseState::chunk_bits_*2))*BaseState::num_local_chunks_;
  if((size_required>>20) > Utils::get_system_memory_mb()){
    throw std::runtime_error(std::string("There is not enough memory to store density matrix"));
  }
  cmatrix_t reduced_state(1ull << qubits.size(),1ull << qubits.size(),true);

  if(BaseState::distributed_rank_ == 0){
    auto tmp = BaseState::qregs_[0].copy_to_matrix();
    for(iChunk=0;iChunk<BaseState::num_global_chunks_;iChunk++){
      int_t i;
      uint_t irow_chunk = (iChunk >> ((BaseState::num_qubits_ - BaseState::chunk_bits_))) << BaseState::chunk_bits_;
      uint_t icol_chunk = (iChunk & ((1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_)))-1)) << BaseState::chunk_bits_;

      if(iChunk < BaseState::num_local_chunks_)
        tmp = BaseState::qregs_[iChunk].copy_to_matrix();
#ifdef AER_MPI
      else
        BaseState::recv_data(tmp.data(),size,0,iChunk);
#endif
#pragma omp parallel for if(num_threads > 1) num_threads(num_threads)
      for(i=0;i<size;i++){
        uint_t irow = (i >> (BaseState::chunk_bits_)) + irow_chunk;
        uint_t icol = (i & mask) + icol_chunk;
        uint_t irow_out = 0;
        uint_t icol_out = 0;
        int j;
        for(j=0;j<qubits.size();j++){
          if((irow >> qubits[j]) & 1){
            irow &= ~(1ull << qubits[j]);
            irow_out += (1ull << j);
          }
          if((icol >> qubits[j]) & 1){
            icol &= ~(1ull << qubits[j]);
            icol_out += (1ull << j);
          }
        }
        if(irow == icol){   //only diagonal base can be reduced
          uint_t idx = ((irow_out) << qubits.size()) + icol_out;
#pragma omp critical
          reduced_state[idx] += tmp[i];
        }
      }
    }
  }
  else{
#ifdef AER_MPI
    //send matrices to process 0
    for(iChunk=0;iChunk<BaseState::num_global_chunks_;iChunk++){
      uint_t iProc = BaseState::get_process_by_chunk(iChunk);
      if(iProc == BaseState::distributed_rank_){
        auto tmp = BaseState::qregs_[iChunk-BaseState::global_chunk_index_].copy_to_matrix();
        BaseState::send_data(tmp.data(),size,iChunk,0);
      }
    }
#endif
  }

  return reduced_state;
}


//=========================================================================
// Implementation: Matrix multiplication
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_gate(const uint_t iChunk, const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = DensityMatrix::State<densmat_t>::gateset_.find(op.name);
  if (it == DensityMatrix::State<densmat_t>::gateset_.end())
    throw std::invalid_argument("DensityMatrixState::invalid gate instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
    case DensityMatrix::Gates::u3:
      apply_gate_u3(iChunk, op.qubits[0],
                    std::real(op.params[0]),
                    std::real(op.params[1]),
                    std::real(op.params[2]));
      break;
    case DensityMatrix::Gates::u2:
      apply_gate_u3(iChunk, op.qubits[0],
                    M_PI / 2.,
                    std::real(op.params[0]),
                    std::real(op.params[1]));
      break;
    case DensityMatrix::Gates::u1:
      apply_phase(iChunk,op.qubits[0], std::exp(complex_t(0., 1.) * op.params[0]));
      break;
    case DensityMatrix::Gates::cx:
      BaseState::qregs_[iChunk].apply_cnot(op.qubits[0], op.qubits[1]);
      break;
    case DensityMatrix::Gates::cy:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::CY);
      break;
    case DensityMatrix::Gates::cz:
      BaseState::qregs_[iChunk].apply_cphase(op.qubits[0], op.qubits[1], -1);
      break;
    case DensityMatrix::Gates::cp:
      BaseState::qregs_[iChunk].apply_cphase(op.qubits[0], op.qubits[1],
                                    std::exp(complex_t(0., 1.) * op.params[0]));
      break;
    case DensityMatrix::Gates::id:
      break;
    case DensityMatrix::Gates::x:
      BaseState::qregs_[iChunk].apply_x(op.qubits[0]);
      break;
    case DensityMatrix::Gates::y:
      BaseState::qregs_[iChunk].apply_y(op.qubits[0]);
      break;
    case DensityMatrix::Gates::z:
      BaseState::qregs_[iChunk].apply_phase(op.qubits[0], -1);
      break;
    case DensityMatrix::Gates::h:
      apply_gate_u3(iChunk, op.qubits[0], M_PI / 2., 0., M_PI);
      break;
    case DensityMatrix::Gates::s:
      apply_phase(iChunk,op.qubits[0], complex_t(0., 1.));
      break;
    case DensityMatrix::Gates::sdg:
      apply_phase(iChunk,op.qubits[0], complex_t(0., -1.));
      break;
    case DensityMatrix::Gates::sx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::SX);
      break;
    case DensityMatrix::Gates::sxdg:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::SXDG);
      break;
    case DensityMatrix::Gates::t: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_phase(iChunk,op.qubits[0], complex_t(isqrt2, isqrt2));
    } break;
    case DensityMatrix::Gates::tdg: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_phase(iChunk,op.qubits[0], complex_t(isqrt2, -isqrt2));
    } break;
    case DensityMatrix::Gates::swap: {
      BaseState::qregs_[iChunk].apply_swap(op.qubits[0], op.qubits[1]);
    } break;
    case DensityMatrix::Gates::ccx:
      BaseState::qregs_[iChunk].apply_toffoli(op.qubits[0], op.qubits[1], op.qubits[2]);
      break;
    case DensityMatrix::Gates::r:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::r(op.params[0], op.params[1]));
      break;
    case DensityMatrix::Gates::rx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::rx(op.params[0]));
      break;
    case DensityMatrix::Gates::ry:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::ry(op.params[0]));
      break;
    case DensityMatrix::Gates::rz:
      apply_diagonal_unitary_matrix(iChunk,op.qubits, Linalg::VMatrix::rz_diag(op.params[0]));
      break;
    case DensityMatrix::Gates::rxx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::rxx(op.params[0]));
      break;
    case DensityMatrix::Gates::ryy:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::ryy(op.params[0]));
      break;
    case DensityMatrix::Gates::rzz:
      apply_diagonal_unitary_matrix(iChunk,op.qubits, Linalg::VMatrix::rzz_diag(op.params[0]));
      break;
    case DensityMatrix::Gates::rzx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::rzx(op.params[0]));
      break;
    case DensityMatrix::Gates::pauli:
      apply_pauli(op.qubits, op.string_params[0]);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument("DensityMatrix::State::invalid gate instruction \'" +
                                  op.name + "\'.");
  }
}


template <class densmat_t>
void State<densmat_t>::apply_matrix(const int_t iChunk, const reg_t &qubits, const cmatrix_t &mat) {
  if (mat.GetRows() == 1) {
    apply_diagonal_unitary_matrix(iChunk,qubits, Utils::vectorize_matrix(mat));
  } else {
    BaseState::qregs_[iChunk].apply_unitary_matrix(qubits, Utils::vectorize_matrix(mat));
  }
}

template <class densmat_t>
void State<densmat_t>::apply_gate_u3(const int_t iChunk, uint_t qubit, double theta, double phi, double lambda) {
  BaseState::qregs_[iChunk].apply_unitary_matrix(reg_t({qubit}), Linalg::VMatrix::u3(theta, phi, lambda));
}

template <class densmat_t>
void State<densmat_t>::apply_pauli(const reg_t &qubits,
                                   const std::string &pauli) 
{
  int_t i;
  // Pauli as a superoperator is (-1)^num_y P\otimes P
  complex_t coeff = (std::count(pauli.begin(), pauli.end(), 'Y') % 2) ? -1 : 1;

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
  for(i=0;i<BaseState::num_local_chunks_;i++){
    BaseState::qregs_[i].apply_pauli(
        BaseState::qregs_[i].superop_qubits(qubits), pauli + pauli, coeff);
  }
}

template <class densmat_t>
void State<densmat_t>::apply_diagonal_unitary_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t & diag)
{
  if(BaseState::gpu_optimization_){
    //GPU computes all chunks in one kernel, so pass qubits and diagonal matrix as is
    BaseState::qregs_[iChunk].apply_diagonal_unitary_matrix(qubits,diag);
  }
  else{
    reg_t qubits_in = qubits;
    reg_t qubits_row = qubits;
    cvector_t diag_in = diag;
    cvector_t diag_row = diag;

    BaseState::block_diagonal_matrix(iChunk,qubits_in,diag_in);

    for(int_t i=0;i<qubits.size();i++){
      if(qubits[i] >= BaseState::chunk_bits_)
        qubits_row[i] = qubits[i] + BaseState::num_qubits_ - BaseState::chunk_bits_;
    }
    BaseState::block_diagonal_matrix(iChunk,qubits_row,diag_row);

    reg_t qubits_chunk(qubits_in.size()*2);
    for(int_t i=0;i<qubits_in.size();i++){
      qubits_chunk[i] = qubits_in[i];
      qubits_chunk[i+qubits_in.size()] = qubits_in[i] + BaseState::chunk_bits_;
    }
    BaseState::qregs_[iChunk].apply_diagonal_matrix(qubits_chunk,AER::Utils::tensor_product(AER::Utils::conjugate(diag_row),diag_in));
  }
}

template <class densmat_t>
void State<densmat_t>::apply_phase(const uint_t iChunk,const uint_t qubit, const complex_t phase)
{
  cvector_t diag(2);
  diag[0] = 1.0;
  diag[1] = phase;
  apply_diagonal_unitary_matrix(iChunk,reg_t({qubit}), diag);
}

template <class densmat_t>
void State<densmat_t>::apply_phase(const uint_t iChunk,const reg_t& qubits, const complex_t phase)
{
  cvector_t diag((1 << qubits.size()),1.0);
  diag[(1 << qubits.size()) - 1] = phase;
  apply_diagonal_unitary_matrix(iChunk,qubits, diag);
}

//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_measure(const reg_t &qubits,
                                      const reg_t &cmemory,
                                      const reg_t &cregister,
                                      RngEngine &rng) 
{
  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(qubits, rng);
  // Implement measurement update
  measure_reset_update(qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());

  BaseState::creg_.store_measure(outcome, cmemory, cregister);
}

template <class densmat_t>
rvector_t State<densmat_t>::measure_probs(const reg_t &qubits) const 
{
  uint_t dim = 1ull << qubits.size();
  rvector_t sum(dim,0.0);
  int_t i,j,k;
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;

  for(i=0;i<qubits.size();i++){
    if(qubits[i] < BaseState::chunk_bits_){
      qubits_in_chunk.push_back(qubits[i]);
    }
    else{
      qubits_out_chunk.push_back(qubits[i]);
    }
  }

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i,j,k) 
  for(i=0;i<BaseState::num_local_chunks_;i++){
    uint_t irow,icol;
    irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
    icol = (BaseState::global_chunk_index_ + i) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));

    if(irow == icol){   //diagonal chunk
      if(qubits_in_chunk.size() > 0){
        auto chunkSum = BaseState::qregs_[i].probabilities(qubits_in_chunk);
        if(qubits_in_chunk.size() == qubits.size()){
          for(j=0;j<dim;j++){
#pragma omp atomic
            sum[j] += chunkSum[j];
          }
        }
        else{
          for(j=0;j<chunkSum.size();j++){
            int idx = 0;
            int i_in = 0;
            for(k=0;k<qubits.size();k++){
              if(qubits[k] < (BaseState::chunk_bits_)){
                idx += (((j >> i_in) & 1) << k);
                i_in++;
              }
              else{
                if((((i + BaseState::global_chunk_index_) << (BaseState::chunk_bits_)) >> qubits[k]) & 1){
                  idx += 1ull << k;
                }
              }
            }
#pragma omp atomic
            sum[idx] += chunkSum[j];
          }
        }
      }
      else{ //there is no bit in chunk
        auto tr = std::real(BaseState::qregs_[i].trace());
        int idx = 0;
        for(k=0;k<qubits_out_chunk.size();k++){
          if((((i + BaseState::global_chunk_index_) << (BaseState::chunk_bits_)) >> qubits_out_chunk[k]) & 1){
            idx += 1ull << k;
          }
        }
#pragma omp atomic
        sum[idx] += tr;
      }
    }
  }

#ifdef AER_MPI
  BaseState::reduce_sum(sum);
#endif

  return sum;
}

template <class densmat_t>
std::vector<reg_t> State<densmat_t>::sample_measure(const reg_t &qubits,
                                                     uint_t shots,
                                                     RngEngine &rng) 
{
  int_t i,j;
  std::vector<double> chunkSum(BaseState::num_local_chunks_+1,0);
  double sum,localSum;
  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  rvector_t allbit_samples(shots,0);

  for (i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));

   //calculate per chunk sum
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
  for(i=0;i<BaseState::num_local_chunks_;i++){
    uint_t irow,icol;
    irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
    icol = (BaseState::global_chunk_index_ + i) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
    if(irow == icol)   //only diagonal chunk has probabilities
      chunkSum[i] = std::real( BaseState::qregs_[i].trace() );
    else
      chunkSum[i] = 0.0;
  }
  localSum = 0.0;
  for(i=0;i<BaseState::num_local_chunks_;i++){
    sum = localSum;
    localSum += chunkSum[i];
    chunkSum[i] = sum;
  }
  chunkSum[BaseState::num_local_chunks_] = localSum;

  double globalSum = 0.0;
  if(BaseState::nprocs_ > 1){
    std::vector<double> procTotal(BaseState::nprocs_);

    for(i=0;i<BaseState::nprocs_;i++){
      procTotal[i] = localSum;
    }
    BaseState::gather_value(procTotal);

    for(i=0;i<BaseState::myrank_;i++){
      globalSum += procTotal[i];
    }
  }

  rvector_t local_samples(shots,0);

  //get rnds positions for each chunk
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i,j) 
  for(i=0;i<BaseState::num_local_chunks_;i++){
    uint_t irow,icol;
    irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
    icol = (BaseState::global_chunk_index_ + i) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
    if(irow != icol)
      continue;

    uint_t nIn;
    std::vector<uint_t> vIdx;
    std::vector<double> vRnd;

    //find rnds in this chunk
    nIn = 0;
    for(j=0;j<shots;j++){
      if(rnds[j] >= chunkSum[i] + globalSum && rnds[j] < chunkSum[i+1] + globalSum){
        vRnd.push_back(rnds[j] - (globalSum + chunkSum[i]));
        vIdx.push_back(j);
        nIn++;
      }
    }

    if(nIn > 0){
      auto chunkSamples = BaseState::qregs_[i].sample_measure(vRnd);
      uint_t ir;
      ir = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));

      for(j=0;j<nIn;j++){
        local_samples[vIdx[j]] = (ir << BaseState::chunk_bits_) + chunkSamples[j];
      }
    }
  }

#ifdef AER_MPI
  BaseState::reduce_sum(local_samples);
#endif
  allbit_samples = local_samples;

  // Convert to reg_t format
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);
  for (int_t val : allbit_samples) {
    reg_t allbit_sample = Utils::int2reg(val, 2, BaseState::num_qubits_);
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
void State<densmat_t>::apply_reset(const int_t iChunk,const reg_t &qubits) 
{
  // TODO: This can be more efficient by adding reset
  // to base class rather than doing a matrix multiplication
  // where all but 1 row is zeros.
  const auto reset_op = Linalg::SMatrix::reset(1ULL << qubits.size());

  BaseState::qregs_[iChunk].apply_superop_matrix(qubits, Utils::vectorize_matrix(reset_op));
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
                                             const double meas_prob) 
{
  int_t i;

  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement final_state
  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      apply_diagonal_unitary_matrix(i,qubits, mdiag);
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      if(qubits[0] < BaseState::chunk_bits_){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
        for(i=0;i<BaseState::num_local_chunks_;i++){
          BaseState::qregs_[i].apply_x(qubits[0]);
        }
      }
      else{
        BaseState::apply_chunk_x(qubits[0]);
        BaseState::apply_chunk_x(qubits[0]+BaseState::chunk_bits_);
      }
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      apply_diagonal_unitary_matrix(i,qubits, mdiag);
    }

    // If it doesn't agree with the reset state update
    // TODO This function could be optimized as a permutation update
    if (final_state != meas_state) {
      reg_t qubits_in_chunk;
      reg_t qubits_out_chunk;

      for(i=0;i<qubits.size();i++){
        if(qubits[i] < BaseState::chunk_bits_){
          qubits_in_chunk.push_back(qubits[i]);
        }
        else{
          qubits_out_chunk.push_back(qubits[i]);
        }
      }

      if(qubits_in_chunk.size() > 0){   //in chunk exchange
        const size_t dim_in = 1ULL << qubits_in_chunk.size();
        // build vectorized permutation matrix
        cvector_t perm(dim_in * dim_in, 0.);
        perm[final_state * dim_in + meas_state] = 1.;
        perm[meas_state * dim_in + final_state] = 1.;
        for (size_t j=0; j < dim_in; j++) {
          if (j != final_state && j != meas_state)
            perm[j * dim_in + j] = 1.;
        }

        // apply permutation to swap state
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
        for(i=0;i<BaseState::num_local_chunks_;i++){
          BaseState::qregs_[i].apply_unitary_matrix(qubits, perm);
        }
      }
      if(qubits_out_chunk.size() > 0){  //out of chunk exchange
        for(i=0;i<qubits_out_chunk.size();i++){
          BaseState::apply_chunk_x(qubits_out_chunk[i]);
          BaseState::apply_chunk_x(qubits_out_chunk[i]+(BaseState::num_qubits_ - BaseState::chunk_bits_));
        }
      }
    }
  }
}


//=========================================================================
// Implementation: Kraus Noise
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_kraus(const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats)
{
  int_t i;
  // Convert to Superoperator
  const auto nrows = kmats[0].GetRows();
  cmatrix_t superop(nrows * nrows, nrows * nrows);
  for (const auto& kraus : kmats) {
    superop += Utils::tensor_product(Utils::conjugate(kraus), kraus);
  }
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
  for(i=0;i<BaseState::num_local_chunks_;i++){
    BaseState::qregs_[i].apply_superop_matrix(qubits, Utils::vectorize_matrix(superop));
  }
}

//-------------------------------------------------------------------------
} // end namespace DensityMatrix
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
