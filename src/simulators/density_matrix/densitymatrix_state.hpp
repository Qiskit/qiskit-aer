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

#include "densitymatrix.hpp"
#include "framework/json.hpp"
#include "framework/opset.hpp"
#include "framework/utils.hpp"
#include "simulators/state_chunk.hpp"
#ifdef AER_THRUST_SUPPORTED
#include "densitymatrix_thrust.hpp"
#endif

namespace AER {

namespace DensityMatrix {

using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {OpType::gate, OpType::measure,
     OpType::reset, OpType::snapshot,
     OpType::barrier, OpType::bfunc, OpType::qerror_loc,
     OpType::roerror, OpType::matrix,
     OpType::diagonal_matrix, OpType::kraus,
     OpType::superop, OpType::set_statevec,
     OpType::set_densmat, OpType::save_expval,
     OpType::save_expval_var, OpType::save_densmat,
     OpType::save_probs, OpType::save_probs_ket,
     OpType::save_amps_sq, OpType::save_state,
     OpType::jump, OpType::mark
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

// Allowed gates enum class
enum class Gates {
  u1, u2, u3, r, rx,ry, rz, id, x, y, z, h, s, sdg, sx, sxdg, t, tdg,
  cx, cy, cz, swap, rxx, ryy, rzz, rzx, ccx, cp, pauli
};

// Allowed snapshots enum class
enum class Snapshots {
  cmemory, cregister,
  densitymatrix,
  probs, probs_var,
  expval_pauli, expval_pauli_var
  /* TODO: The following expectation value snapshots still need to be
     implemented */
  //,expval_matrix, expval_matrix_var
};

//=========================================================================
// DensityMatrix State subclass
//=========================================================================

template <class densmat_t = QV::DensityMatrix<double>>
class State : public QuantumState::StateChunk<densmat_t> {
public:
  using BaseState = QuantumState::StateChunk<densmat_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return densmat_t::name(); }

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  virtual void apply_op(const int_t iChunk, const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine &rng,
                        bool final_op = false) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  virtual void set_config(const json_t &config) override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                            RngEngine &rng) override;

  // Initialize OpenMP settings for the underlying DensityMatrix class
  void initialize_omp();

  auto move_to_matrix(const int_t iChunk);
  auto copy_to_matrix(const int_t iChunk);

protected:
  template <typename list_t>
  void initialize_from_vector(const int_t iChunk, const list_t &vec);

  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------
  //apply op to multiple shots , return flase if op is not supported to execute in a batch
  bool apply_batched_op(const int_t iChunk, const Operations::Op &op,
                                ExperimentResult &result,
                                std::vector<RngEngine> &rng,
                                bool final_op = false) override;

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(const int_t iChunk, const Operations::Op &op);

  //apply (multi) control gate by statevector
  void apply_gate_statevector(const int_t iChunk, const Operations::Op &op);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual void apply_measure(const int_t iChunk, const reg_t &qubits, const reg_t &cmemory,
                             const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by tracing out qubits
  void apply_reset(const int_t iChunk, const reg_t &qubits);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const int_t iChunk, const Operations::Op &op,
                              ExperimentResult &result,
                              bool last_op = false);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cmatrix_t &mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t &vmat);

  //apply diagonal matrix
  void apply_diagonal_unitary_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t & diag);

  // Apply a Kraus error operation
  void apply_kraus(const int_t iChunk, const reg_t &qubits, const std::vector<cmatrix_t> &kraus);

  // Apply an N-qubit Pauli gate
  void apply_pauli(const int_t iChunk, const reg_t &qubits, const std::string &pauli);

  // apply phase
  void apply_phase(const int_t iChunk, const uint_t qubit, const complex_t phase);
  void apply_phase(const int_t iChunk, const reg_t& qubits, const complex_t phase);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current full density matrix
  void apply_save_state(const int_t iChunk, const Operations::Op &op,
                        ExperimentResult &result,
                        bool last_op = false);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const int_t iChunk, const Operations::Op &op,
                                 ExperimentResult &result,
                                 bool last_op = false);

  // Helper function for computing expectation value
  void apply_save_probs(const int_t iChunk, const Operations::Op &op,
                        ExperimentResult &result);

  // Helper function for saving amplitudes squared
  void apply_save_amplitudes_sq(const int_t iChunk, const Operations::Op &op,
                                ExperimentResult &result);

  // Helper function for computing expectation value
  virtual double expval_pauli(const int_t iChunk, const reg_t &qubits,
                              const std::string& pauli) override;

  // Return the reduced density matrix for the simulator
  cmatrix_t reduced_density_matrix(const int_t iChunk, const reg_t &qubits, bool last_op = false);
  cmatrix_t reduced_density_matrix_helper(const int_t iChunk, const reg_t &qubits,
                                          const reg_t &qubits_sorted);

  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Return vector of measure probabilities for specified qubits
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  // TODO: move to private (no longer part of base class)
  rvector_t measure_probs(const int_t iChunk, const reg_t &qubits) const;

  // Sample the measurement outcome for qubits
  // return a pair (m, p) of the outcome m, and its corresponding
  // probability p.
  // Outcome is given as an int: Eg for two-qubits {q0, q1} we have
  // 0 -> |q1 = 0, q0 = 0> state
  // 1 -> |q1 = 0, q0 = 1> state
  // 2 -> |q1 = 1, q0 = 0> state
  // 3 -> |q1 = 1, q0 = 1> state
  std::pair<uint_t, double> sample_measure_with_prob(const int_t iChunk, const reg_t &qubits,
                                                     RngEngine &rng);

  void measure_reset_update(const int_t iChunk, const std::vector<uint_t> &qubits,
                            const uint_t final_state, const uint_t meas_state,
                            const double meas_prob);

  //-----------------------------------------------------------------------
  // Special snapshot types
  //
  // IMPORTANT: These methods are not marked const to allow modifying state
  // during snapshot, but after the snapshot is applied the simulator
  // should be left in the pre-snapshot state.
  //-----------------------------------------------------------------------

  // Snapshot reduced density matrix
  void snapshot_density_matrix(const int_t iChunk, const Operations::Op &op,
                               ExperimentResult &result,
                               bool last_op = false);
                          
  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                              bool variance);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                             bool variance);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
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

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;

  // Table of allowed snapshot types to enum class members
  const static stringmap_t<Snapshots> snapshotset_;

  //scale for density matrix = 2
  //this function is used in the base class to scale chunk qubits for multi-chunk distribution
  int qubit_scale(void) override
  {
    return 2;
  }

  //-----------------------------------------------------------------------
  //Functions for multi-chunk distribution
  //-----------------------------------------------------------------------
  //swap between chunks
  void apply_chunk_swap(const reg_t &qubits) override;

  //apply multiple swaps between chunks
  void apply_multi_chunk_swap(const reg_t &qubits) override;
};

//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

template <class densmat_t>
const stringmap_t<Gates> State<densmat_t>::gateset_({
    // Single qubit gates
    {"delay", Gates::id},// Delay gate
    {"id", Gates::id},   // Pauli-Identity gate
    {"x", Gates::x},     // Pauli-X gate
    {"y", Gates::y},     // Pauli-Y gate
    {"z", Gates::z},     // Pauli-Z gate
    {"s", Gates::s},     // Phase gate (aka sqrt(Z) gate)
    {"sdg", Gates::sdg}, // Conjugate-transpose of Phase gate
    {"h", Gates::h},     // Hadamard gate (X + Z / sqrt(2))
    {"t", Gates::t},     // T-gate (sqrt(S))
    {"tdg", Gates::tdg}, // Conjguate-transpose of T gate
    {"x90", Gates::sx},  // Pi/2 X (equiv to Sqrt(X) gate)
    {"sx", Gates::sx},   // Sqrt(X) gate
    {"sxdg", Gates::sxdg},// Inverse Sqrt(X) gate
    {"r", Gates::r},     // R rotation gate
    {"rx", Gates::rx},   // Pauli-X rotation gate
    {"ry", Gates::ry},   // Pauli-Y rotation gate
    {"rz", Gates::rz},   // Pauli-Z rotation gate
    // Waltz Gates
    {"p", Gates::u1},  // Phase gate
    {"u1", Gates::u1}, // zero-X90 pulse waltz gate
    {"u2", Gates::u2}, // single-X90 pulse waltz gate
    {"u3", Gates::u3}, // two X90 pulse waltz gate
    {"u", Gates::u3}, // two X90 pulse waltz gate
    {"U", Gates::u3},  // two X90 pulse waltz gate
    // Two-qubit gates
    {"CX", Gates::cx},     // Controlled-X gate (CNOT)
    {"cx", Gates::cx},     // Controlled-X gate (CNOT)
    {"cy", Gates::cy},     // Controlled-Y gate
    {"cz", Gates::cz},     // Controlled-Z gate
    {"cp", Gates::cp},     // Controlled-Phase gate
    {"cu1", Gates::cp},    // Controlled-Phase gate
    {"swap", Gates::swap}, // SWAP gate
    {"rxx", Gates::rxx},   // Pauli-XX rotation gate
    {"ryy", Gates::ryy},   // Pauli-YY rotation gate
    {"rzz", Gates::rzz},   // Pauli-ZZ rotation gate
    {"rzx", Gates::rzx},   // Pauli-ZX rotation gate
    // Three-qubit gates
    {"ccx", Gates::ccx},   // Controlled-CX gate (Toffoli)
    // Pauli gate
    {"pauli", Gates::pauli} // Multi-qubit Pauli gate
});

template <class densmat_t>
const stringmap_t<Snapshots> State<densmat_t>::snapshotset_(
    {{"density_matrix", Snapshots::densitymatrix},
     {"probabilities", Snapshots::probs},
     {"probabilities_with_variance", Snapshots::probs_var},
     {"memory", Snapshots::cmemory},
     {"register", Snapshots::cregister},
     {"expectation_value_pauli", Snapshots::expval_pauli},
     {"expectation_value_pauli_with_variance", Snapshots::expval_pauli_var}});

//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------
template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits) 
{
  if(BaseState::qregs_.size() == 0)
    BaseState::allocate(num_qubits,num_qubits,1);
  initialize_omp();

  for(int_t i=0;i<BaseState::qregs_.size();i++){
    BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_);
  }

  if(BaseState::multi_chunk_distribution_){
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++){
          if(BaseState::global_chunk_index_ + iChunk == 0){
            BaseState::qregs_[iChunk].initialize();
          }
          else{
            BaseState::qregs_[iChunk].zero();
          }
        }
      }
    }
    else{
      for(int_t i=0;i<BaseState::qregs_.size();i++){
        if(BaseState::global_chunk_index_ + i == 0){
          BaseState::qregs_[i].initialize();
        }
        else{
          BaseState::qregs_[i].zero();
        }
      }
    }
  }
  else{
    for(int_t i=0;i<BaseState::qregs_.size();i++){
      BaseState::qregs_[i].initialize();
    }
  }
}

template <class densmat_t> void State<densmat_t>::initialize_omp() 
{
  uint_t i;
  for(i=0;i<BaseState::qregs_.size();i++){
    BaseState::qregs_[i].set_omp_threshold(omp_qubit_threshold_);
    if (BaseState::threads_ > 0)
      BaseState::qregs_[i].set_omp_threads(BaseState::threads_); // set allowed OMP threads in qubitvector
  }
}

template <class densmat_t>
template <typename list_t>
void State<densmat_t>::initialize_from_vector(const int_t iChunkIn, const list_t &vec)
{
  if((1ull << (BaseState::num_qubits_*2)) == vec.size()){
    BaseState::initialize_from_vector(iChunkIn, vec);
  }
  else if((1ull << (BaseState::num_qubits_*2)) == vec.size() * vec.size()) {
    int_t iChunk;
    if(BaseState::multi_chunk_distribution_){
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++){
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
      else{
        for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
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
    else{
      BaseState::qregs_[iChunkIn].initialize_from_vector(AER::Utils::tensor_product(AER::Utils::conjugate(vec), vec));
    }
  }
  else {
    throw std::runtime_error("DensityMatrixChunk::initialize input vector is incorrect length. Expected: " +
                             std::to_string((1ull << (BaseState::num_qubits_*2))) + " Received: " +
                             std::to_string(vec.size()));
  }
}

template <class densmat_t>
auto State<densmat_t>::move_to_matrix(const int_t iChunk)
{
  if(!BaseState::multi_chunk_distribution_)
    return BaseState::qregs_[iChunk].move_to_matrix();
  return BaseState::apply_to_matrix(false);
}

template <class densmat_t>
auto State<densmat_t>::copy_to_matrix(const int_t iChunk)
{
  if(!BaseState::multi_chunk_distribution_)
    return BaseState::qregs_[iChunk].copy_to_matrix();
  return BaseState::apply_to_matrix(true);
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

template <class densmat_t>
size_t State<densmat_t>::required_memory_mb(
    uint_t num_qubits, const std::vector<Operations::Op> &ops) const 
{
  (void)ops; // avoid unused variable compiler warning
  (void)ops; // avoid unused variable compiler warning
  densmat_t tmp;
  return tmp.required_memory_mb(2*num_qubits);
}

template <class densmat_t>
void State<densmat_t>::set_config(const json_t &config) 
{
  BaseState::set_config(config);

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "chop_threshold", config);
  uint_t i;
  for(i=0;i<BaseState::qregs_.size();i++){
    BaseState::qregs_[i].set_json_chop_threshold(json_chop_threshold_);
  }

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold",
                  config);
}


//=========================================================================
// Implementation: apply operations
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_op(const int_t iChunk, const Operations::Op &op,
                                 ExperimentResult &result,
                                 RngEngine &rng,
                                 bool final_ops) 
{
  if(BaseState::check_conditional(iChunk, op)) {
    switch (op.type) {
      case OpType::barrier:
      case OpType::qerror_loc:
        break;
      case OpType::reset:
        apply_reset(iChunk, op.qubits);
        break;
      case OpType::measure:
        apply_measure(iChunk, op.qubits, op.memory, op.registers, rng);
        break;
      case OpType::bfunc:
        BaseState::cregs_[0].apply_bfunc(op);
        break;
      case OpType::roerror:
        BaseState::cregs_[0].apply_roerror(op, rng);
        break;
      case OpType::gate:
        apply_gate(iChunk, op);
        break;
      case OpType::snapshot:
        apply_snapshot(iChunk, op, result, final_ops);
        break;
      case OpType::matrix:
        apply_matrix(iChunk, op.qubits, op.mats[0]);
        break;
      case OpType::diagonal_matrix:
        apply_diagonal_unitary_matrix(iChunk, op.qubits, op.params);
        break;
      case OpType::superop:
        BaseState::qregs_[iChunk].apply_superop_matrix(op.qubits, Utils::vectorize_matrix(op.mats[0]));
        break;
      case OpType::kraus:
        apply_kraus(iChunk, op.qubits, op.mats);
        break;
      case OpType::set_statevec:
        initialize_from_vector(iChunk, op.params);
        break;
      case OpType::set_densmat:
        BaseState::initialize_from_matrix(iChunk, op.mats[0]);
        break;
      case OpType::save_expval:
      case OpType::save_expval_var:
        BaseState::apply_save_expval(iChunk, op, result);
        break;
      case OpType::save_state:
        apply_save_state(iChunk, op, result, final_ops);
        break;
      case OpType::save_densmat:
        apply_save_density_matrix(iChunk, op, result, final_ops);
        break;
      case OpType::save_probs:
      case OpType::save_probs_ket:
        apply_save_probs(iChunk, op, result);
        break;
      case OpType::save_amps_sq:
        apply_save_amplitudes_sq(iChunk, op, result);
        break;
      default:
        throw std::invalid_argument("DensityMatrix::State::invalid instruction \'" +
                                    op.name + "\'.");
    }
  }
}

template <class densmat_t>
bool State<densmat_t>::apply_batched_op(const int_t iChunk, const Operations::Op &op,
                                  ExperimentResult &result,
                                  std::vector<RngEngine> &rng,
                                  bool final_ops) 
{
  if(op.conditional)
    BaseState::qregs_[iChunk].set_conditional(op.conditional_reg);

  switch (op.type) {
    case OpType::barrier:
    case OpType::nop:
    case OpType::qerror_loc:
      break;
    case OpType::reset:
      BaseState::qregs_[iChunk].apply_reset(op.qubits);
      break;
    case OpType::measure:
      BaseState::qregs_[iChunk].apply_batched_measure(op.qubits,rng,op.memory,op.registers);
      break;
    case OpType::bfunc:
      BaseState::qregs_[iChunk].apply_bfunc(op);
      break;
    case OpType::roerror:
      BaseState::qregs_[iChunk].apply_roerror(op, rng);
      break;
    case OpType::gate:
      apply_gate(iChunk, op);
      break;
    case OpType::matrix:
      apply_matrix(iChunk, op.qubits, op.mats[0]);
      break;
    case OpType::diagonal_matrix:
      BaseState::qregs_[iChunk].apply_diagonal_unitary_matrix(op.qubits, op.params);
      break;
    case OpType::superop:
      BaseState::qregs_[iChunk].apply_superop_matrix(op.qubits, Utils::vectorize_matrix(op.mats[0]));
      break;
    case OpType::kraus:
      apply_kraus(iChunk, op.qubits, op.mats);
      break;
    default:
      //other operations should be called to indivisual chunks by apply_op
      return false;
  }
  return true;
}

//=========================================================================
// Implementation: Save data
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_save_probs(const int_t iChunk, const Operations::Op &op,
                                            ExperimentResult &result) 
{
  auto probs = measure_probs(iChunk, op.qubits);
  auto cr = this->creg(BaseState::get_global_shot_index(iChunk));
  if (op.type == OpType::save_probs_ket) {
    result.save_data_average(cr, op.string_params[0],
                             Utils::vec2ket(probs, json_chop_threshold_, 16),
                             op.type, op.save_type);
  } else {
    result.save_data_average(cr, op.string_params[0],
                             std::move(probs), op.type, op.save_type);
  }
}

template <class densmat_t>
void State<densmat_t>::apply_save_amplitudes_sq(const int_t iChunkIn, const Operations::Op &op,
                                                ExperimentResult &result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument("Invalid save_amplitudes_sq instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  rvector_t amps_sq(size);

  if(BaseState::multi_chunk_distribution_){
    int_t iChunk;
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
      uint_t irow,icol;
      irow = (BaseState::global_chunk_index_ + iChunk) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
      icol = (BaseState::global_chunk_index_ + iChunk) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
      if(irow != icol)
        continue;

#pragma omp parallel for if (size > pow(2, omp_qubit_threshold_) &&        \
                                 BaseState::threads_ > 1 && !BaseState::chunk_omp_parallel_)  \
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
  }
  else{
#pragma omp parallel for if (size > pow(2, omp_qubit_threshold_) &&        \
                                 BaseState::threads_ > 1)                       \
                          num_threads(BaseState::threads_)
    for (int_t i = 0; i < size; ++i) {
      amps_sq[i] = BaseState::qregs_[iChunkIn].probability(op.int_params[i]);
    }
  }
  auto cr = this->creg(BaseState::get_global_shot_index(iChunkIn));
  result.save_data_average(cr, op.string_params[0],
                           std::move(amps_sq), op.type, op.save_type);
}

template <class densmat_t>
double State<densmat_t>::expval_pauli(const int_t iChunk, const reg_t &qubits,
                                      const std::string& pauli)  {
  if(!BaseState::multi_chunk_distribution_)
    return BaseState::qregs_[iChunk].expval_pauli(qubits, pauli);

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
void State<densmat_t>::apply_save_density_matrix(const int_t iChunk, const Operations::Op &op,
                                                 ExperimentResult &result,
                                                 bool last_op) {
  auto cr = this->creg(BaseState::get_global_shot_index(iChunk));
  result.save_data_average(cr, op.string_params[0],
                           reduced_density_matrix(iChunk, op.qubits, last_op),
                           op.type, op.save_type);
}

template <class densmat_t>
void State<densmat_t>::apply_save_state(const int_t iChunk, const Operations::Op &op,
                                        ExperimentResult &result,
                                        bool last_op) 
{
  if (op.qubits.size() != BaseState::num_qubits_) {
    throw std::invalid_argument(
        op.name + " was not applied to all qubits."
        " Only the full state can be saved.");
  }
  // Renamp single data type to average
  Operations::DataSubType save_type;
  switch (op.save_type) {
    case Operations::DataSubType::single:
      save_type = Operations::DataSubType::average;
      break;
    case Operations::DataSubType::c_single:
      save_type = Operations::DataSubType::c_average;
      break;
    default:
      save_type = op.save_type;
  }

  // Default key
  std::string key = (op.string_params[0] == "_method_")
                      ? "density_matrix"
                      : op.string_params[0];
  auto cr = this->creg(BaseState::get_global_shot_index(iChunk));
  if (last_op) {
    result.save_data_average(cr, key, move_to_matrix(iChunk),
                             OpType::save_densmat, save_type);
  } else {
    result.save_data_average(cr, key, copy_to_matrix(iChunk),
                             OpType::save_densmat, save_type);
  }
}


template <class densmat_t>
cmatrix_t State<densmat_t>::reduced_density_matrix(const int_t iChunk, const reg_t& qubits, bool last_op) 
{
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);
    if(!BaseState::multi_chunk_distribution_){
      reduced_state[0] = BaseState::qregs_[iChunk].trace();
    }
    else{
      std::complex<double> sum = 0.0;
      for(int_t i=0;i<BaseState::qregs_.size();i++){
        sum += BaseState::qregs_[i].trace();
      }
#ifdef AER_MPI
      BaseState::reduce_sum(sum);
#endif
      reduced_state[0] = sum;
    }
  } else {

    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

    if ((qubits.size() == BaseState::num_qubits_) && (qubits == qubits_sorted)) {
      if (last_op) {
        reduced_state = move_to_matrix(iChunk);
      } else {
        reduced_state = copy_to_matrix(iChunk);
      }
    } else {
      reduced_state = reduced_density_matrix_helper(iChunk, qubits, qubits_sorted);
    }
  }
  return reduced_state;
}

template <class densmat_t>
cmatrix_t State<densmat_t>::reduced_density_matrix_helper(const int_t iChunkIn, const reg_t &qubits,
                                          const reg_t &qubits_sorted) 
{
  if(!BaseState::multi_chunk_distribution_){
    // Get superoperator qubits
    const reg_t squbits = BaseState::qregs_[iChunkIn].superop_qubits(qubits);
    const reg_t squbits_sorted = BaseState::qregs_[iChunkIn].superop_qubits(qubits_sorted);

    // Get dimensions
    const size_t N = qubits.size();
    const size_t DIM = 1ULL << N;
    const int_t VDIM = 1ULL << (2 * N);
    const size_t END = 1ULL << (BaseState::qregs_[iChunkIn].num_qubits() - N);
    const size_t SHIFT = END + 1;

    // Copy vector to host memory
    auto vmat = BaseState::qregs_[iChunkIn].vector();
    cmatrix_t reduced_state(DIM, DIM, false);
    {
      // Fill matrix with first iteration
      const auto inds = QV::indexes(squbits, squbits_sorted, 0);
      for (int_t i = 0; i < VDIM; ++i) {
        reduced_state[i] = std::move(vmat[inds[i]]);
      }
    }
    // Accumulate with remaning blocks
    for (size_t k = 1; k < END; k++) {
      const auto inds = QV::indexes(squbits, squbits_sorted, k * SHIFT);
      for (int_t i = 0; i < VDIM; ++i) {
        reduced_state[i] += complex_t(std::move(vmat[inds[i]]));
      }
    }
    return reduced_state;
  }

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
// Implementation: Snapshots
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_snapshot(const int_t iChunk, const Operations::Op &op,
                                      ExperimentResult &result,
                                      bool last_op) {

  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument(
        "DensityMatrixState::invalid snapshot instruction \'" + op.name +
        "\'.");
  switch (it->second) {
    case Snapshots::densitymatrix:
      snapshot_density_matrix(iChunk, op, result, last_op);
      break;
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(iChunk, op, result);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(iChunk, op, result);
      break;
    case Snapshots::probs:
      // get probs as hexadecimal
      snapshot_probabilities(iChunk, op, result, false);
      break;
    case Snapshots::probs_var:
      // get probs as hexadecimal
      snapshot_probabilities(iChunk, op, result, true);
      break;
    case Snapshots::expval_pauli: {
      snapshot_pauli_expval(iChunk, op, result, false);
    } break;
    case Snapshots::expval_pauli_var: {
      snapshot_pauli_expval(iChunk, op, result, true);
    } break;
    /* TODO
    case Snapshots::expval_matrix: {
      snapshot_matrix_expval(op, result, false);
    }  break;
    case Snapshots::expval_matrix_var: {
      snapshot_matrix_expval(op, result, true);
    }  break;
    */
    default:
      // We shouldn't get here unless there is a bug in the snapshotset
      throw std::invalid_argument(
        "DensityMatrix::State::invalid snapshot instruction \'" + op.name +
        "\'.");
  }
}

template <class densmat_t>
void State<densmat_t>::snapshot_probabilities(const int_t iChunk, const Operations::Op &op,
                                              ExperimentResult &result,
                                              bool variance) 
{
  int_t ishot = BaseState::get_global_shot_index(iChunk);
  // get probs as hexadecimal
  auto probs = Utils::vec2ket(measure_probs(iChunk, op.qubits),
                              json_chop_threshold_, 16);
  result.legacy_data.add_average_snapshot("probabilities",
                            op.string_params[0],
                            BaseState::cregs_[ishot].memory_hex(),
                            std::move(probs),
                            variance);
}

template <class densmat_t>
void State<densmat_t>::snapshot_pauli_expval(const int_t iChunk, const Operations::Op &op,
                                             ExperimentResult &result,
                                             bool variance) 
{
  // Check empty edge case
  if (op.params_expval_pauli.empty()) {
    throw std::invalid_argument(
        "Invalid expval snapshot (Pauli components are empty).");
  }
  int_t ishot = BaseState::get_global_shot_index(iChunk);

  // Accumulate expval components
  complex_t expval(0., 0.);
  for (const auto &param : op.params_expval_pauli) {
    const auto &coeff = param.first;
    const auto &pauli = param.second;
    expval += coeff * expval_pauli(iChunk, op.qubits, pauli);
  }

  // Add to snapshot
  Utils::chop_inplace(expval, json_chop_threshold_);
  result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::cregs_[ishot].memory_hex(), expval, variance);
}

template <class densmat_t>
void State<densmat_t>::snapshot_density_matrix(const int_t iChunk, const Operations::Op &op,
                                               ExperimentResult &result,
                                               bool last_op) 
{
  int_t ishot = BaseState::get_global_shot_index(iChunk);
  result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                                          BaseState::cregs_[ishot].memory_hex(),
                                          reduced_density_matrix(iChunk, op.qubits, last_op),
                                          false);
}

//=========================================================================
// Implementation: Matrix multiplication
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_gate(const int_t iChunk, const Operations::Op &op) 
{
  if(!BaseState::global_chunk_indexing_){
    reg_t qubits_in,qubits_out;
    bool ctrl_chunk = true;
    bool ctrl_chunk_sp = true;
    BaseState::get_inout_ctrl_qubits(op,qubits_out,qubits_in);
    if(qubits_out.size() > 0){
      uint_t mask = 0;
      for(int i=0;i<qubits_out.size();i++){
        mask |= (1ull << (qubits_out[i] - BaseState::chunk_bits_));
      }
      if(((BaseState::global_chunk_index_ + iChunk) & mask) != mask){
        ctrl_chunk = false;
      }
      if((((BaseState::global_chunk_index_ + iChunk) >> (BaseState::num_qubits_ - BaseState::chunk_bits_)) & mask) != mask){
        ctrl_chunk_sp = false;
      }
      if(!ctrl_chunk && !ctrl_chunk_sp)
        return;   //do nothing for this chunk
      else{
        Operations::Op new_op = BaseState::remake_gate_in_chunk_qubits(op,qubits_in);
        if(ctrl_chunk && ctrl_chunk_sp)
          apply_gate(iChunk,new_op);  //apply gate by using op with internal qubits
        else if(ctrl_chunk)
          apply_gate_statevector(iChunk,new_op);
        else{
          for(int i=0;i<new_op.qubits.size();i++)
            new_op.qubits[i] += BaseState::chunk_bits_;
          apply_gate_statevector(iChunk,new_op);
        }
        return;
      }
    }
  }

  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
        "DensityMatrixState::invalid gate instruction \'" + op.name + "\'.");
  switch (it->second) {
    case Gates::u3:
      apply_gate_u3(iChunk, op.qubits[0], std::real(op.params[0]),
                    std::real(op.params[1]), std::real(op.params[2]));
      break;
    case Gates::u2:
      apply_gate_u3(iChunk, op.qubits[0], M_PI / 2., std::real(op.params[0]),
                    std::real(op.params[1]));
      break;
    case Gates::u1:
      apply_phase(iChunk, op.qubits[0], std::exp(complex_t(0., 1.) * op.params[0]));
      break;
    case Gates::cx:
      BaseState::qregs_[iChunk].apply_cnot(op.qubits[0], op.qubits[1]);
      break;
    case Gates::cy:
      BaseState::qregs_[iChunk].apply_cy(op.qubits[0], op.qubits[1]);
      break;
    case Gates::cz:
      BaseState::qregs_[iChunk].apply_cphase(op.qubits[0], op.qubits[1], -1);
      break;
    case Gates::cp:
      BaseState::qregs_[iChunk].apply_cphase(op.qubits[0], op.qubits[1],
                                    std::exp(complex_t(0., 1.) * op.params[0]));
      break;
    case Gates::id:
      break;
    case Gates::x:
      BaseState::qregs_[iChunk].apply_x(op.qubits[0]);
      break;
    case Gates::y:
      BaseState::qregs_[iChunk].apply_y(op.qubits[0]);
      break;
    case Gates::z:
      apply_phase(iChunk, op.qubits[0], -1);
      break;
    case Gates::h:
      apply_gate_u3(iChunk, op.qubits[0], M_PI / 2., 0., M_PI);
      break;
    case Gates::s:
      apply_phase(iChunk, op.qubits[0], complex_t(0., 1.));
      break;
    case Gates::sdg:
      apply_phase(iChunk, op.qubits[0], complex_t(0., -1.));
      break;
    case Gates::sx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::SX);
      break;
    case Gates::sxdg:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::SXDG);
      break;
    case Gates::t: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_phase(iChunk, op.qubits[0], complex_t(isqrt2, isqrt2));
    } break;
    case Gates::tdg: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_phase(iChunk, op.qubits[0], complex_t(isqrt2, -isqrt2));
    } break;
    case Gates::swap: {
      BaseState::qregs_[iChunk].apply_swap(op.qubits[0], op.qubits[1]);
    } break;
    case Gates::ccx:
      BaseState::qregs_[iChunk].apply_toffoli(op.qubits[0], op.qubits[1], op.qubits[2]);
      break;
    case Gates::r:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::r(op.params[0], op.params[1]));
      break;
    case Gates::rx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::rx(op.params[0]));
      break;
    case Gates::ry:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::ry(op.params[0]));
      break;
    case Gates::rz:
      apply_diagonal_unitary_matrix(iChunk, op.qubits, Linalg::VMatrix::rz_diag(op.params[0]));
      break;
    case Gates::rxx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::rxx(op.params[0]));
      break;
    case Gates::ryy:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::ryy(op.params[0]));
      break;
    case Gates::rzz:
      apply_diagonal_unitary_matrix(iChunk, op.qubits, Linalg::VMatrix::rzz_diag(op.params[0]));
      break;
    case Gates::rzx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::rzx(op.params[0]));
      break;
    case Gates::pauli:
      apply_pauli(iChunk, op.qubits, op.string_params[0]);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument(
        "DensityMatrix::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

template <class densmat_t>
void State<densmat_t>::apply_gate_statevector(const int_t iChunk, const Operations::Op &op)
{
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
        "DensityMatrixState::invalid gate instruction \'" + op.name + "\'.");
  switch (it->second) {
    case Gates::x:
    case Gates::cx:
      BaseState::qregs_[iChunk].apply_mcx(op.qubits);
      break;
    case Gates::u1:
      if(op.qubits[op.qubits.size()-1] < BaseState::chunk_bits_){
        BaseState::qregs_[iChunk].apply_mcphase(op.qubits,
                                    std::exp(complex_t(0., 1.) * op.params[0]));
      }
      else{
        BaseState::qregs_[iChunk].apply_mcphase(op.qubits,
                                    std::conj(std::exp(complex_t(0., 1.) * op.params[0])));
      }
      break;
    case Gates::y:
      BaseState::qregs_[iChunk].apply_mcy(op.qubits);
      break;
    case Gates::z:
      BaseState::qregs_[iChunk].apply_mcphase(op.qubits, -1);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument(
        "DensityMatrix::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

template <class densmat_t>
void State<densmat_t>::apply_matrix(const int_t iChunk, const reg_t &qubits, const cmatrix_t &mat) {
  if (mat.GetRows() == 1) {
    apply_diagonal_unitary_matrix(iChunk, 
        qubits, Utils::vectorize_matrix(mat));
  } else {
    BaseState::qregs_[iChunk].apply_unitary_matrix(qubits, Utils::vectorize_matrix(mat));
  }
}

template <class densmat_t>
void State<densmat_t>::apply_gate_u3(const int_t iChunk, uint_t qubit, double theta, double phi,
                                     double lambda) {
  BaseState::qregs_[iChunk].apply_unitary_matrix(
      reg_t({qubit}), Linalg::VMatrix::u3(theta, phi, lambda));
}

template <class densmat_t>
void State<densmat_t>::apply_diagonal_unitary_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t & diag)
{
  if(BaseState::global_chunk_indexing_ || !BaseState::multi_chunk_distribution_){
    //GPU computes all chunks in one kernel, so pass qubits and diagonal matrix as is
    BaseState::qregs_[iChunk].apply_diagonal_unitary_matrix(qubits,diag);
  }
  else{
    reg_t qubits_in = qubits;
    reg_t qubits_row = qubits;
    cvector_t diag_in = diag;
    cvector_t diag_row = diag;

    BaseState::block_diagonal_matrix(iChunk,qubits_in,diag_in);

    if(qubits_in.size() == qubits.size()){
      BaseState::qregs_[iChunk].apply_diagonal_unitary_matrix(qubits,diag);
    }
    else{
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
}

template <class densmat_t>
void State<densmat_t>::apply_phase(const int_t iChunk, const uint_t qubit, const complex_t phase)
{
  cvector_t diag(2);
  diag[0] = 1.0;
  diag[1] = phase;
  apply_diagonal_unitary_matrix(iChunk, reg_t({qubit}), diag);
}

template <class densmat_t>
void State<densmat_t>::apply_phase(const int_t iChunk, const reg_t& qubits, const complex_t phase)
{
  cvector_t diag((1 << qubits.size()),1.0);
  diag[(1 << qubits.size()) - 1] = phase;
  apply_diagonal_unitary_matrix(iChunk, qubits, diag);
}

template <class densmat_t>
void State<densmat_t>::apply_pauli(const int_t iChunk, const reg_t &qubits,
                                   const std::string &pauli) 
{
  // Pauli as a superoperator is (-1)^num_y P\otimes P
  complex_t coeff = (std::count(pauli.begin(), pauli.end(), 'Y') % 2) ? -1 : 1;
  BaseState::qregs_[iChunk].apply_pauli(
      BaseState::qregs_[iChunk].superop_qubits(qubits), pauli + pauli, coeff);
}

//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_measure(const int_t iChunk, const reg_t &qubits, const reg_t &cmemory,
                                     const reg_t &cregister, RngEngine &rng) 
{
  int_t ishot = BaseState::get_global_shot_index(iChunk);
  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(iChunk, qubits, rng);
  // Implement measurement update
  measure_reset_update(iChunk, qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());
  BaseState::cregs_[ishot].store_measure(outcome, cmemory, cregister);
}

template <class densmat_t>
rvector_t State<densmat_t>::measure_probs(const int_t iChunk, const reg_t &qubits) const 
{
  if(!BaseState::multi_chunk_distribution_)
    return BaseState::qregs_[iChunk].probabilities(qubits);

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

  if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for private(i,j,k)
    for(int_t ig=0;ig<BaseState::num_groups_;ig++){
      for(i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++){
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
    }
  }
  else{
    for(i=0;i<BaseState::qregs_.size();i++){
      uint_t irow,icol;
      irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
      icol = (BaseState::global_chunk_index_ + i) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));

      if(irow == icol){   //diagonal chunk
        if(qubits_in_chunk.size() > 0){
          auto chunkSum = BaseState::qregs_[i].probabilities(qubits_in_chunk);
          if(qubits_in_chunk.size() == qubits.size()){
            for(j=0;j<dim;j++){
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
          sum[idx] += tr;
        }
      }
    }
  }

#ifdef AER_MPI
  BaseState::reduce_sum(sum);
#endif

  return sum;
  
}

template <class densmat_t>
void State<densmat_t>::apply_reset(const int_t iChunk, const reg_t &qubits) 
{
  BaseState::qregs_[iChunk].apply_reset(qubits);
}

template <class densmat_t>
std::pair<uint_t, double>
State<densmat_t>::sample_measure_with_prob(const int_t iChunk, const reg_t &qubits,
                                           RngEngine &rng) 
{
  rvector_t probs = measure_probs(iChunk, qubits);
  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class densmat_t>
void State<densmat_t>::measure_reset_update(const int_t iChunk, const reg_t &qubits,
                                            const uint_t final_state,
                                            const uint_t meas_state,
                                            const double meas_prob) 
{
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    if(!BaseState::multi_chunk_distribution_)
      apply_diagonal_unitary_matrix(iChunk, qubits, mdiag);
    else{
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
            apply_diagonal_unitary_matrix(i, qubits, mdiag);
        }
      }
      else{
        for(int_t i=0;i<BaseState::qregs_.size();i++)
          apply_diagonal_unitary_matrix(i, qubits, mdiag);
      }
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      if(!BaseState::multi_chunk_distribution_)
        BaseState::qregs_[iChunk].apply_x(qubits[0]);
      else{
        if(qubits[0] < BaseState::chunk_bits_){
          if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
            for(int_t ig=0;ig<BaseState::num_groups_;ig++){
              for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
                BaseState::qregs_[i].apply_x(qubits[0]);
            }
          }
          else{
            for(int_t i=0;i<BaseState::qregs_.size();i++)
              BaseState::qregs_[i].apply_x(qubits[0]);
          }
        }
        else{
          BaseState::apply_chunk_x(qubits[0]);
          BaseState::apply_chunk_x(qubits[0]+BaseState::chunk_bits_);
        }
      }
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    if(!BaseState::multi_chunk_distribution_)
      apply_diagonal_unitary_matrix(iChunk, qubits, mdiag);
    else{
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
            apply_diagonal_unitary_matrix(i, qubits, mdiag);
        }
      }
      else{
        for(int_t i=0;i<BaseState::qregs_.size();i++)
          apply_diagonal_unitary_matrix(i, qubits, mdiag);
      }
    }

    // If it doesn't agree with the reset state update
    // TODO This function could be optimized as a permutation update
    if (final_state != meas_state) {
      // build vectorized permutation matrix
      cvector_t perm(dim * dim, 0.);
      perm[final_state * dim + meas_state] = 1.;
      perm[meas_state * dim + final_state] = 1.;
      for (size_t j = 0; j < dim; j++) {
        if (j != final_state && j != meas_state)
          perm[j * dim + j] = 1.;
      }
      // apply permutation to swap state
      if(!BaseState::multi_chunk_distribution_)
        BaseState::qregs_[iChunk].apply_unitary_matrix(qubits, perm);
      else{
        reg_t qubits_in_chunk;
        reg_t qubits_out_chunk;

        for(int_t i=0;i<qubits.size();i++){
          if(qubits[i] < BaseState::chunk_bits_){
            qubits_in_chunk.push_back(qubits[i]);
          }
          else{
            qubits_out_chunk.push_back(qubits[i]);
          }
        }
        if(qubits_in_chunk.size() > 0){   //in chunk exchange
          if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
            for(int_t ig=0;ig<BaseState::num_groups_;ig++){
              for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
                BaseState::qregs_[i].apply_unitary_matrix(qubits, perm);
            }
          }
          else{
            for(int_t i=0;i<BaseState::qregs_.size();i++)
              BaseState::qregs_[i].apply_unitary_matrix(qubits, perm);
          }
        }
        if(qubits_out_chunk.size() > 0){  //out of chunk exchange
          for(int_t i=0;i<qubits_out_chunk.size();i++){
            BaseState::apply_chunk_x(qubits_out_chunk[i]);
            BaseState::apply_chunk_x(qubits_out_chunk[i]+(BaseState::num_qubits_ - BaseState::chunk_bits_));
          }
        }
      }
    }
  }
}

template <class densmat_t>
std::vector<reg_t> State<densmat_t>::sample_measure(const reg_t &qubits,
                                                    uint_t shots,
                                                    RngEngine &rng) 
{
  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  for (uint_t i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));
  reg_t allbit_samples(shots,0);

  if(!BaseState::multi_chunk_distribution_)
    allbit_samples = BaseState::qregs_[0].sample_measure(rnds);
  else{
    int_t i,j;
    std::vector<double> chunkSum(BaseState::qregs_.size()+1,0);
    double sum,localSum;
   //calculate per chunk sum
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for private(i) 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++){
          uint_t irow,icol;
          irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
          icol = (BaseState::global_chunk_index_ + i) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
          if(irow == icol)   //only diagonal chunk has probabilities
            chunkSum[i] = std::real( BaseState::qregs_[i].trace() );
          else
            chunkSum[i] = 0.0;
        }
      }
    }
    else{
      for(i=0;i<BaseState::qregs_.size();i++){
        uint_t irow,icol;
        irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
        icol = (BaseState::global_chunk_index_ + i) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
        if(irow == icol)   //only diagonal chunk has probabilities
          chunkSum[i] = std::real( BaseState::qregs_[i].trace() );
        else
          chunkSum[i] = 0.0;
      }
    }
    localSum = 0.0;
    for(i=0;i<BaseState::qregs_.size();i++){
      sum = localSum;
      localSum += chunkSum[i];
      chunkSum[i] = sum;
    }
    chunkSum[BaseState::qregs_.size()] = localSum;

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

    reg_t local_samples(shots,0);

    //get rnds positions for each chunk
    for(i=0;i<BaseState::qregs_.size();i++){
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

        for(j=0;j<chunkSamples.size();j++){
          local_samples[vIdx[j]] = (ir << BaseState::chunk_bits_) + chunkSamples[j];
        }
      }
    }

#ifdef AER_MPI
    BaseState::reduce_sum(local_samples);
#endif
    allbit_samples = local_samples;
  }

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


//=========================================================================
// Implementation: Kraus Noise
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_kraus(const int_t iChunk, const reg_t &qubits,
                                   const std::vector<cmatrix_t> &kmats) 
{
  BaseState::qregs_[iChunk].apply_superop_matrix(
      qubits, Utils::vectorize_matrix(Utils::kraus_superop(kmats)));
}


//-----------------------------------------------------------------------
//Functions for multi-chunk distribution
//-----------------------------------------------------------------------
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

template <class densmat_t>
void State<densmat_t>::apply_multi_chunk_swap(const reg_t &qubits)
{
  reg_t qubits_density;

  for(int_t i=0;i<qubits.size();i+=2){
    uint_t q0,q1;
    q0 = qubits[i*2];
    q1 = qubits[i*2+1];

    std::swap(BaseState::qubit_map_[q0],BaseState::qubit_map_[q1]);

    if(q1 >= BaseState::chunk_bits_){
      q1 += BaseState::chunk_bits_;
    }
    qubits_density.push_back(q0);
    qubits_density.push_back(q1);

    q0 += BaseState::chunk_bits_;
    if(q1 >= BaseState::chunk_bits_){
      q1 += (BaseState::num_qubits_ - BaseState::chunk_bits_*2);
    }
  }

  BaseState::apply_multi_chunk_swap(qubits_density);
}


//-------------------------------------------------------------------------
} // end namespace DensityMatrix
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
