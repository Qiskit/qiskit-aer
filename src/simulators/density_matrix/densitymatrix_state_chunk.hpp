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

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  // Op types
  {Operations::OpType::gate, Operations::OpType::measure,
    Operations::OpType::reset, Operations::OpType::snapshot,
    Operations::OpType::barrier, Operations::OpType::bfunc,
    Operations::OpType::roerror, Operations::OpType::matrix,
    Operations::OpType::diagonal_matrix, Operations::OpType::kraus,
    Operations::OpType::superop, Operations::OpType::sim_op},
  // Gates
  {"U", "CX", "u1", "u2", "u3",  "cx",  "cy", "cz",  "swap", "id",
     "x", "y",  "z",  "h",  "s",   "sdg", "t",   "tdg",  "ccx",
     "r", "rx", "ry", "rz", "rxx", "ryy", "rzz", "rzx",  "p",
     "cp","cu1", "sx", "x90", "delay", "swap_chunk"},
  // Snapshots
  {"density_matrix", "memory", "register", "probabilities",
    "probabilities_with_variance", "expectation_value_pauli",
    "expectation_value_pauli_with_variance"}
);

// Allowed gates enum class
enum class Gates {
  u1, u2, u3, r, rx,ry, rz, id, x, y, z, h, s, sdg, sx, t, tdg,
  cx, cy, cz, swap, rxx, ryy, rzz, rzx, ccx, cp
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

  virtual void allocate(uint_t num_qubits,uint_t shots);

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
  uint_t apply_blocking(const std::vector<Operations::Op> &ops, uint_t op_begin);

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(const uint_t iChunk, const Operations::Op &op);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual void apply_measure(const int_t iChunk, const reg_t &qubits,
                             const reg_t &cmemory,
                             const reg_t &cregister,
                             RngEngine &rng);

  // Reset the specified qubits to the |0> state by tracing out qubits
  void apply_reset(const int_t iChunk, const reg_t &qubits);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const int_t iChunk, const Operations::Op &op, ExperimentData &data);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cmatrix_t & mat);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t & vmat);

  // Apply a Kraus error operation
  void apply_kraus(const int_t iChunk, const reg_t &qubits, const std::vector<cmatrix_t> &kraus);

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
  std::pair<uint_t, double>
  sample_measure_with_prob(const int_t iChunk, const reg_t &qubits, RngEngine &rng);


  void measure_reset_update(const int_t iChunk, const std::vector<uint_t> &qubits,
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
  void snapshot_density_matrix(const int_t iChunk, const Operations::Op &op,
                               ExperimentData &data);

  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(const int_t iChunk, const Operations::Op &op,
                              ExperimentData &data,
                              bool variance);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const int_t iChunk, const Operations::Op &op,
                             ExperimentData &data,
                             bool variance);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(const int_t iChunk, const Operations::Op &op,
                              ExperimentData &data,
                              bool variance);

  // Return the reduced density matrix for the simulator
  cmatrix_t reduced_density_matrix(const int_t iChunk, const reg_t &qubits, const reg_t& qubits_sorted);
  cmatrix_t reduced_density_matrix_cpu(const int_t iChunk, const reg_t &qubits, const reg_t& qubits_sorted);
  cmatrix_t reduced_density_matrix_thrust(const int_t iChunk, const reg_t &qubits, const reg_t& qubits_sorted);

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
    {"r", Gates::r},     // R rotation gate
    {"rx", Gates::rx},   // Pauli-X rotation gate
    {"ry", Gates::ry},   // Pauli-Y rotation gate
    {"rz", Gates::rz},   // Pauli-Z rotation gate
    // Waltz Gates
    {"p", Gates::u1},  // Phase gate
    {"u1", Gates::u1}, // zero-X90 pulse waltz gate
    {"u2", Gates::u2}, // single-X90 pulse waltz gate
    {"u3", Gates::u3}, // two X90 pulse waltz gate
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
    {"ccx", Gates::ccx} // Controlled-CX gate (Toffoli)
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
void State<densmat_t>::allocate(uint_t num_qubits,uint_t shots)
{
  int_t i;
  uint_t nchunks;

  BaseState::num_shots_ = shots;

  BaseState::setup_chunk_bits(num_qubits,2);

//  printf("  TEST chunk_bits = %d, num_qubits = %d (%d), shots = %d, num_chunks = %d\n",this->chunk_bits_,this->num_qubits_,num_qubits,shots,BaseState::num_local_chunks_);

  BaseState::chunk_omp_parallel_ = false;
  if(BaseState::chunk_bits_ < BaseState::num_qubits_){
    if(BaseState::qregs_[0].name() == "density_matrix_gpu"){
      BaseState::chunk_omp_parallel_ = true;   //CUDA backend requires thread parallelization of chunk loop
    }
  }

  nchunks = BaseState::num_local_chunks_;
  for(i=0;i<BaseState::num_local_chunks_;i++){
    if(this->multi_shot_parallelization_){
      BaseState::qregs_[i].chunk_setup(BaseState::chunk_bits_,BaseState::num_qubits_,0,nchunks);
    }
    else{
      BaseState::qregs_[i].chunk_setup(BaseState::chunk_bits_,BaseState::num_qubits_,i + BaseState::global_chunk_index_,nchunks);
    }
    //only first one allocates chunks, others only set chunk index
    nchunks = 0;
  }
}

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits) 
{
  int_t i;

  initialize_omp();

  if(BaseState::chunk_bits_ == BaseState::num_qubits_){
//#pragma omp barrier
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_/2);
      BaseState::qregs_[i].zero();
    }
//#pragma omp barrier
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].initialize();
    }
  }
  else{   //multi-chunk distribution

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_/2);
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
                                   const densmat_t &state) {
  // Check dimension of state
  if (state.num_qubits() != num_qubits) {
    throw std::invalid_argument("DensityMatrix::State::initialize: initial state does not match qubit number");
  }
  //TO DO : need multiple states to initialize ...
                                          printf("  TEST density init qreg from state\n");
}

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits,
                                        const cmatrix_t &state) {
  if (state.size() != 1ULL << 2 * num_qubits) {
    throw std::invalid_argument("DensityMatrix::State::initialize: initial state does not match qubit number");
  }
  //TO DO : need multiple states to initialize ...
                                          printf("  TEST density init qreg from matrix\n");
}

template <class densmat_t>
void State<densmat_t>::initialize_qreg(uint_t num_qubits,
                                        const cvector_t &state) {
  if (state.size() != 1ULL << 2 * num_qubits) {
    throw std::invalid_argument("DensityMatrix::State::initialize: initial state does not match qubit number");
  }

  uint_t i;

  initialize_omp();

  for(i=0;i<BaseState::num_local_chunks_;i++){
    BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_/2);

    BaseState::qregs_[i].initialize_from_vector(state);
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
void State<densmat_t>::apply_ops(const std::vector<Operations::Op> &ops,
                                 ExperimentData &data,
                                 RngEngine &rng) 
{
  uint_t iOp,nOp;
  int_t iChunk;

  nOp = ops.size();
  iOp = 0;

  // Simple loop over vector of input operations
  while(iOp < nOp){
    // If conditional op check conditional
    if (BaseState::cregs_[0].check_conditional(ops[iOp])) {
      switch (ops[iOp].type) {
        case Operations::OpType::barrier:
          break;
        case Operations::OpType::reset:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            apply_reset(iChunk, ops[iOp].qubits);
          break;
        case Operations::OpType::measure:
          apply_measure(-1, ops[iOp].qubits, ops[iOp].memory, ops[iOp].registers, rng);
          break;
        case Operations::OpType::bfunc:
          for(iChunk=0;iChunk<BaseState::cregs_.size();iChunk++)
            BaseState::cregs_[iChunk].apply_bfunc(ops[iOp]);
          break;
        case Operations::OpType::roerror:
          for(iChunk=0;iChunk<BaseState::cregs_.size();iChunk++)
            BaseState::cregs_[iChunk].apply_roerror(ops[iOp], rng);
          break;
        case Operations::OpType::gate:
          if(ops[iOp].name == "swap_chunk"){
            uint_t q0,q1;
            q0 = ops[iOp].qubits[0];
            q1 = ops[iOp].qubits[1];
            if(ops[iOp].qubits[0] >= BaseState::chunk_bits_/2){
              q0 += BaseState::chunk_bits_/2;
            }
            if(ops[iOp].qubits[1] >= BaseState::chunk_bits_/2){
              q1 += BaseState::chunk_bits_/2;
            }
            reg_t qs0 = {{q0, q1}};
            BaseState::apply_chunk_swap(qs0);

            if(ops[iOp].qubits[0] >= BaseState::chunk_bits_/2){
              q0 += (BaseState::num_qubits_ - BaseState::chunk_bits_)/2;
            }
            else{
              q0 += BaseState::chunk_bits_/2;
            }
            if(ops[iOp].qubits[1] >= BaseState::chunk_bits_/2){
              q1 += (BaseState::num_qubits_ - BaseState::chunk_bits_)/2;
            }
            else{
              q1 += BaseState::chunk_bits_/2;
            }
            reg_t qs1 = {{q0, q1}};
            BaseState::apply_chunk_swap(qs1);
          }
          else{
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
            for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
              apply_gate(iChunk,ops[iOp]);
          }
          break;
        case Operations::OpType::snapshot:
          apply_snapshot(-1,ops[iOp], data);
          break;
        case Operations::OpType::matrix:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            apply_matrix(iChunk,ops[iOp].qubits, ops[iOp].mats[0]);
          break;
        case Operations::OpType::diagonal_matrix:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            BaseState::qregs_[iChunk].apply_diagonal_matrix(ops[iOp].qubits, ops[iOp].params);
          break;
        case Operations::OpType::superop:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            BaseState::qregs_[iChunk].apply_superop_matrix(ops[iOp].qubits, Utils::vectorize_matrix(ops[iOp].mats[0]));
          break;
        case Operations::OpType::kraus:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            apply_kraus(iChunk, ops[iOp].qubits, ops[iOp].mats);
          break;
        case Operations::OpType::sim_op:
          if(ops[iOp].name == "begin_blocking"){
            iOp = apply_blocking(ops,iOp + 1);
          }
          break;
        default:
          throw std::invalid_argument("DensityMatrix::State::invalid instruction \'" +
                                      ops[iOp].name + "\'.");
      }
    }
    iOp++;
  }
}

template <class densmat_t>
uint_t State<densmat_t>::apply_blocking(const std::vector<Operations::Op> &ops, uint_t op_begin)
{
  uint_t iOp,nOp,iEnd;
  int_t iChunk;

  nOp = ops.size();
  iEnd = op_begin;

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk,iOp) 
  for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
    bool inBlock = true;
    iOp = op_begin;

    BaseState::qregs_[iChunk].fetch_chunk();

    while(iOp < nOp){
      if(BaseState::cregs_[0].check_conditional(ops[iOp])) {
        switch (ops[iOp].type){
          case Operations::OpType::gate:
            apply_gate(iChunk,ops[iOp]);
            break;
          case Operations::OpType::matrix:
            apply_matrix(iChunk,ops[iOp].qubits,ops[iOp].mats[0]);
            break;
          case Operations::OpType::diagonal_matrix:
            BaseState::qregs_[iChunk].apply_diagonal_matrix(ops[iOp].qubits, ops[iOp].params);
            break;
          case Operations::OpType::superop:
            BaseState::qregs_[iChunk].apply_superop_matrix(ops[iOp].qubits, Utils::vectorize_matrix(ops[iOp].mats[0]));
            break;
          case Operations::OpType::sim_op:
            if(ops[iOp].name == "end_blocking"){
              inBlock = false;
#ifdef _MSC_VER
#pragma omp critical
              {
#else
#pragma omp atomic write
#endif
              iEnd = iOp;
#ifdef _MSC_VER
              }
#endif
            }
            break;
          default:
            throw std::invalid_argument("QubitVector::State::invalid instruction \'" +
                                        ops[iOp].name + "\'.");
        }
      }

      if(!inBlock){
        break;
      }
      iOp++;
    }

    if(iOp >= nOp){
#ifdef _MSC_VER
#pragma omp critical
              {
#else
#pragma omp atomic write
#endif
      iEnd = iOp;
#ifdef _MSC_VER
              }
#endif
    }

    BaseState::qregs_[iChunk].release_chunk();
  }

  return iEnd;
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_snapshot(const int_t iChunk, const Operations::Op &op,
                                       ExperimentData &data) 
{
  int_t i;
  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument("DensityMatrixState::invalid snapshot instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
    case Snapshots::densitymatrix:
      snapshot_density_matrix(iChunk, op, data);
      break;
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(iChunk,op, data);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(iChunk,op, data);
      break;
    case Snapshots::probs:
      // get probs as hexadecimal
      snapshot_probabilities(iChunk, op, data, false);
      break;
    case Snapshots::probs_var:
      // get probs as hexadecimal
      snapshot_probabilities(iChunk, op, data, true);
      break;
    case Snapshots::expval_pauli: {
      snapshot_pauli_expval(iChunk, op, data, false);
    } break;
    case Snapshots::expval_pauli_var: {
      snapshot_pauli_expval(iChunk, op, data, true);
    } break;
    /* TODO
    case Snapshots::expval_matrix: {
      snapshot_matrix_expval(iChunk, op, data, false);
    }  break;
    case Snapshots::expval_matrix_var: {
      snapshot_matrix_expval(iChunk, op, data, true);
    }  break;
    */
    default:
      // We shouldn't get here unless there is a bug in the snapshotset
      throw std::invalid_argument("DensityMatrix::State::invalid snapshot instruction \'" +
                                  op.name + "\'.");
  }
}

template <class densmat_t>
void State<densmat_t>::snapshot_probabilities(const int_t iChunk, const Operations::Op &op,
                                              ExperimentData &data,
                                              bool variance) 
{
  // get probs as hexadecimal
  auto probs = Utils::vec2ket(measure_probs(iChunk,op.qubits),
                              json_chop_threshold_, 16);
  if(iChunk < 0){
    data.add_average_snapshot("probabilities",
                            op.string_params[0],
                            BaseState::cregs_[0].memory_hex(),
                            probs,
                            variance);
  }
  else{
    data.add_average_snapshot("probabilities",
                            op.string_params[0],
                            BaseState::cregs_[iChunk].memory_hex(),
                            probs,
                            variance);
  }
}


template <class densmat_t>
void State<densmat_t>::snapshot_pauli_expval(const int_t iChunk, const Operations::Op &op,
                                             ExperimentData &data,
                                             bool variance) 
{
  int_t i,ireg;
  // Check empty edge case
  if (op.params_expval_pauli.empty()) {
    throw std::invalid_argument("Invalid expval snapshot (Pauli components are empty).");
  }

  // Accumulate expval components
  complex_t expval(0., 0.);
  for (const auto &param : op.params_expval_pauli) {
    const auto& coeff = param.first;
    const auto& pauli = param.second;

    if(iChunk < 0){
      double exp_re = 0.0;
      double exp_im = 0.0;
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) reduction(+:exp_re,exp_im)
      for(i=0;i<BaseState::num_local_chunks_;i++){
        uint_t irow,icol;
        irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_)/2);
        icol = (BaseState::global_chunk_index_ + i) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)/2));
        if(irow == icol){   //only diagonal chunks are calculated
          auto exp_tmp = coeff * BaseState::qregs_[i].expval_pauli(op.qubits, pauli);
          exp_re += exp_tmp.real();
          exp_im += exp_tmp.imag();
        }
      }
      complex_t t(exp_re,exp_im);
      expval += t;
    }
    else{
      expval = coeff * BaseState::qregs_[iChunk].expval_pauli(op.qubits, pauli);
    }
  }

#ifdef AER_MPI
  complex_t sum;
  MPI_Allreduce(&expval,&sum,2,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
  expval = sum;
#endif

  if(iChunk < 0){
    ireg = 0;
  }
  else{
    ireg = iChunk;
  }

  // Add to snapshot
  Utils::chop_inplace(expval, json_chop_threshold_);
  data.add_average_snapshot("expectation_value",
                            op.string_params[0],
                            BaseState::cregs_[ireg].memory_hex(),
                            expval, variance);
}

template <class densmat_t>
void State<densmat_t>::snapshot_density_matrix(const int_t iChunkIn, const Operations::Op &op,
                                               ExperimentData &data) 
{
  int_t iChunk,nChunk;
  if(iChunkIn < 0){
    iChunk = 0;
    nChunk = BaseState::num_local_chunks_;
  }
  else{
    iChunk = iChunkIn;
    nChunk = iChunkIn+1;
  }
  for(;iChunk<nChunk;iChunk++){
    cmatrix_t reduced_state;

    // Check if tracing over all qubits
    if (op.qubits.empty()) {
      reduced_state = cmatrix_t(1, 1);
      reduced_state[iChunk] = BaseState::qregs_[iChunk].trace();
    } else {

      auto qubits_sorted = op.qubits;
      std::sort(qubits_sorted.begin(), qubits_sorted.end());

      if ((op.qubits.size() == BaseState::qregs_[iChunk].num_qubits()) && (op.qubits == qubits_sorted)) {
        reduced_state = BaseState::qregs_[iChunk].copy_to_matrix();
      } else {
        reduced_state = reduced_density_matrix(iChunk,op.qubits, qubits_sorted);
      }
    }

    int_t ireg = iChunk;
    if(BaseState::cregs_.size() == 1){
      ireg = 0;
    }
    data.add_average_snapshot("density_matrix",
                              op.string_params[0],
                              BaseState::cregs_[ireg].memory_hex(),
                              std::move(reduced_state),
                              false);
  }
}


template <class statevec_t>
cmatrix_t State<statevec_t>::reduced_density_matrix(const int_t iChunk, const reg_t &qubits,
                                                    const reg_t &qubits_sorted) {
  return reduced_density_matrix_cpu(iChunk,qubits, qubits_sorted);
}

#ifdef AER_THRUST_SUPPORTED
// Thrust specialization must copy memory from device to host
template <>
cmatrix_t State<QV::DensityMatrixThrust<float>>::reduced_density_matrix(const int_t iChunk, const reg_t &qubits,
                                                                       const reg_t &qubits_sorted) {
  
  return reduced_density_matrix_thrust(iChunk,qubits, qubits_sorted);
}

template <>
cmatrix_t State<QV::DensityMatrixThrust<double>>::reduced_density_matrix(const int_t iChunk, const reg_t &qubits,
                                                                       const reg_t &qubits_sorted) {
  
  return reduced_density_matrix_thrust(iChunk,qubits, qubits_sorted);
}

#endif

template <class densmat_t>
cmatrix_t State<densmat_t>::reduced_density_matrix_cpu(const int_t iChunk, const reg_t& qubits, const reg_t& qubits_sorted) {

  // Get superoperator qubits
  const reg_t squbits = BaseState::qregs_[iChunk].superop_qubits(qubits);
  const reg_t squbits_sorted = BaseState::qregs_[iChunk].superop_qubits(qubits_sorted);

  // Get dimensions
  const size_t N = qubits.size();
  const size_t DIM = 1ULL << N;
  const size_t VDIM = 1ULL << (2 * N);
  const size_t END = 1ULL << (BaseState::qregs_[iChunk].num_qubits() - N);  
  const size_t SHIFT = END + 1;

  // TODO: If we are not going to apply any additional instructions after
  //       this function we could move the memory when constructing rather
  //       than copying
  const auto& vmat = BaseState::qregs_[iChunk].data();
  cmatrix_t reduced_state(DIM, DIM, false);
  {
    // Fill matrix with first iteration
    const auto inds = QV::indexes(squbits, squbits_sorted, 0);
    for (int_t i = 0; i < VDIM; ++i) {
      reduced_state[i] = complex_t(vmat[inds[i]]);
    }
  }
  // Accumulate with remaning blocks
  for (size_t k = 1; k < END; k++) {
    const auto inds = QV::indexes(squbits, squbits_sorted, k * SHIFT);
    for (int_t i = 0; i < VDIM; ++i) {
      reduced_state[i] += complex_t(vmat[inds[i]]);
    }
  }
  return reduced_state;
}
  

template <class densmat_t>
cmatrix_t State<densmat_t>::reduced_density_matrix_thrust(const int_t iChunk, const reg_t& qubits, const reg_t& qubits_sorted) {

  // Get superoperator qubits
  const reg_t squbits = BaseState::qregs_[iChunk].superop_qubits(qubits);
  const reg_t squbits_sorted = BaseState::qregs_[iChunk].superop_qubits(qubits_sorted);

  // Get dimensions
  const size_t N = qubits.size();
  const size_t DIM = 1ULL << N;
  const size_t VDIM = 1ULL << (2 * N);
  const size_t END = 1ULL << (BaseState::qregs_[iChunk].num_qubits() - N);  
  const size_t SHIFT = END + 1;

  // Copy vector to host memory
  auto vmat = BaseState::qregs_[iChunk].vector();
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


//=========================================================================
// Implementation: Matrix multiplication
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_gate(const uint_t iChunk, const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument("DensityMatrixState::invalid gate instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
    case Gates::u3:
      apply_gate_u3(iChunk, op.qubits[0],
                    std::real(op.params[0]),
                    std::real(op.params[1]),
                    std::real(op.params[2]));
      break;
    case Gates::u2:
      apply_gate_u3(iChunk, op.qubits[0],
                    M_PI / 2.,
                    std::real(op.params[0]),
                    std::real(op.params[1]));
      break;
    case Gates::u1:
      BaseState::qregs_[iChunk].apply_phase(op.qubits[0], std::exp(complex_t(0., 1.) * op.params[0]));
      break;
    case Gates::cx:
      BaseState::qregs_[iChunk].apply_cnot(op.qubits[0], op.qubits[1]);
      break;
    case Gates::cy:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::CY);
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
      BaseState::qregs_[iChunk].apply_phase(op.qubits[0], -1);
      break;
    case Gates::h:
      apply_gate_u3(iChunk, op.qubits[0], M_PI / 2., 0., M_PI);
      break;
    case Gates::s:
      BaseState::qregs_[iChunk].apply_phase(op.qubits[0], complex_t(0., 1.));
      break;
    case Gates::sdg:
      BaseState::qregs_[iChunk].apply_phase(op.qubits[0], complex_t(0., -1.));
      break;
    case Gates::sx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::SX);
      break;
    case Gates::t: {
      const double isqrt2{1. / std::sqrt(2)};
      BaseState::qregs_[iChunk].apply_phase(op.qubits[0], complex_t(isqrt2, isqrt2));
    } break;
    case Gates::tdg: {
      const double isqrt2{1. / std::sqrt(2)};
      BaseState::qregs_[iChunk].apply_phase(op.qubits[0], complex_t(isqrt2, -isqrt2));
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
      BaseState::qregs_[iChunk].apply_diagonal_unitary_matrix(op.qubits, Linalg::VMatrix::rz_diag(op.params[0]));
      break;
    case Gates::rxx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::rxx(op.params[0]));
      break;
    case Gates::ryy:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::ryy(op.params[0]));
      break;
    case Gates::rzz:
      BaseState::qregs_[iChunk].apply_diagonal_unitary_matrix(op.qubits, Linalg::VMatrix::rzz_diag(op.params[0]));
      break;
    case Gates::rzx:
      BaseState::qregs_[iChunk].apply_unitary_matrix(op.qubits, Linalg::VMatrix::rzx(op.params[0]));
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
    BaseState::qregs_[iChunk].apply_diagonal_unitary_matrix(qubits, Utils::vectorize_matrix(mat));
  } else {
    BaseState::qregs_[iChunk].apply_unitary_matrix(qubits, Utils::vectorize_matrix(mat));
  }
}

template <class densmat_t>
void State<densmat_t>::apply_gate_u3(const int_t iChunk, uint_t qubit, double theta, double phi, double lambda) {
  BaseState::qregs_[iChunk].apply_unitary_matrix(reg_t({qubit}), Linalg::VMatrix::u3(theta, phi, lambda));
}



//=========================================================================
// Implementation: Reset and Measurement Sampling
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_measure(const int_t iChunk, const reg_t &qubits,
                                      const reg_t &cmemory,
                                      const reg_t &cregister,
                                      RngEngine &rng) 
{
  // Actual measurement outcome
  const auto meas = sample_measure_with_prob(iChunk, qubits, rng);
  // Implement measurement update
  measure_reset_update(iChunk, qubits, meas.first, meas.first, meas.second);
  const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());

  int_t ireg = 0;
  if(iChunk >= 0){
    ireg = iChunk;
  }
  BaseState::cregs_[ireg].store_measure(outcome, cmemory, cregister);
}

template <class densmat_t>
rvector_t State<densmat_t>::measure_probs(const int_t iChunk, const reg_t &qubits) const 
{
  if(iChunk < 0){
    uint_t dim = 1ull << qubits.size();
    rvector_t sum(dim,0.0);
    int_t i,j,k;
    reg_t qubits_in_chunk;
    reg_t qubits_out_chunk;

    for(i=0;i<qubits.size();i++){
      if(qubits[i] < BaseState::chunk_bits_/2){
        qubits_in_chunk.push_back(qubits[i]);
      }
      else{
        qubits_out_chunk.push_back(qubits[i]);
      }
    }

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i,j,k) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      uint_t irow,icol;
      irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_)/2);
      icol = (BaseState::global_chunk_index_ + i) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)/2));

      if(irow == icol){   //diagonal chunk
        auto chunkSum = BaseState::qregs_[i].probabilities(qubits);
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
              if(qubits[k] < BaseState::chunk_bits_){
                idx += (((j >> i_in) & 1) << k);
                i_in++;
              }
              else{
                if((((i + BaseState::global_chunk_index_) << (BaseState::chunk_bits_/2)) >> qubits[k]) & 1){
                  idx += 1ull << k;
                }
              }
            }
#pragma omp atomic
            sum[idx] += chunkSum[j];
          }
        }
      }
    }

    return sum;
  }
  else{
    return BaseState::qregs_[iChunk].probabilities(qubits);
  }
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
  reg_t allbit_samples(shots,0);

  printf(" TEST: smapling\n");

  for (i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));

  if(this->multi_shot_parallelization_){
    for(i=0;i<BaseState::num_local_chunks_;i++){
      std::vector<double> vRnd(1);
      vRnd[0] = rnds[i];
      auto samples = BaseState::qregs_[i].sample_measure(vRnd);

      allbit_samples[i] = samples[0];
    }
  }
  else{
     //calculate per chunk sum
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      uint_t irow,icol;
      irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_)/2);
      icol = (BaseState::global_chunk_index_ + i) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)/2));
      if(irow == icol)   //diagonal chunk
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
#ifdef AER_MPI
    if(BaseState::nprocs_ > 1){// && isMultiShot_ == false){
      std::vector<double> procTotal(BaseState::nprocs_);

      for(i=0;i<BaseState::nprocs_;i++){
        procTotal[i] = localSum;
      }

      MPI_Alltoall(&procTotal[0],1,MPI_DOUBLE_PRECISION,&procTotal[0],1,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD);

      for(i=0;i<BaseState::myrank_;i++){
        globalSum += procTotal[i];
      }
    }
#endif

    reg_t local_samples(shots,0);

    //get rnds positions for each chunk
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i,j) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
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
        uint_t irow;
        irow = (BaseState::global_chunk_index_ + i) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_)/2);

        for(j=0;j<nIn;j++){
          local_samples[vIdx[j]] = (irow << BaseState::chunk_bits_/2) + chunkSamples[j];
        }
      }
    }

#ifdef AER_MPI
    MPI_Allreduce(&local_samples[0],&allbit_samples[0],shots,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
#else
    allbit_samples = local_samples;
#endif
  }

  // Convert to reg_t format
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);
  for (int_t val : allbit_samples) {
    reg_t allbit_sample = Utils::int2reg(val, 2, BaseState::num_qubits_/2);
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
void State<densmat_t>::apply_reset(const int_t iChunk, const reg_t &qubits) 
{
  // TODO: This can be more efficient by adding reset
  // to base class rather than doing a matrix multiplication
  // where all but 1 row is zeros.
  const auto reset_op = Linalg::SMatrix::reset(1ULL << qubits.size());

  if(iChunk < 0){
    int_t i;
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].apply_superop_matrix(qubits, Utils::vectorize_matrix(reset_op));
    }
  }
  else{
    BaseState::qregs_[iChunk].apply_superop_matrix(qubits, Utils::vectorize_matrix(reset_op));
  }
}

template <class densmat_t>
std::pair<uint_t, double>
State<densmat_t>::sample_measure_with_prob(const int_t iChunk, const reg_t &qubits,
                                            RngEngine &rng) {
  rvector_t probs = measure_probs(iChunk,qubits);
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
  int_t i;

  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement final_state
  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
      for(i=0;i<BaseState::num_local_chunks_;i++){
        BaseState::qregs_[i].apply_diagonal_unitary_matrix(qubits, mdiag);
      }
    }
    else{
      BaseState::qregs_[iChunk].apply_diagonal_unitary_matrix(qubits, mdiag);
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
        for(i=0;i<BaseState::num_local_chunks_;i++){
          BaseState::qregs_[i].apply_x(qubits[0]);
        }
      }
      else{
        BaseState::qregs_[iChunk].apply_x(qubits[0]);
      }
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
      for(i=0;i<BaseState::num_local_chunks_;i++){
        BaseState::qregs_[i].apply_diagonal_unitary_matrix(qubits, mdiag);
      }
    }
    else{
      BaseState::qregs_[iChunk].apply_diagonal_unitary_matrix(qubits, mdiag);
    }

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
      if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
        for(i=0;i<BaseState::num_local_chunks_;i++){
          BaseState::qregs_[i].apply_unitary_matrix(qubits, perm);
        }
      }
      else{
        BaseState::qregs_[iChunk].apply_unitary_matrix(qubits, perm);
      }
    }
  }
}


//=========================================================================
// Implementation: Kraus Noise
//=========================================================================

template <class densmat_t>
void State<densmat_t>::apply_kraus(const int_t iChunk, const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats) 
{
  int_t i;
  // Convert to Superoperator
  const auto nrows = kmats[0].GetRows();
  cmatrix_t superop(nrows * nrows, nrows * nrows);
  for (const auto kraus : kmats) {
    superop += Utils::tensor_product(Utils::conjugate(kraus), kraus);
  }
  if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].apply_superop_matrix(qubits, Utils::vectorize_matrix(superop));
    }
  }
  else{
    BaseState::qregs_[iChunk].apply_superop_matrix(qubits, Utils::vectorize_matrix(superop));
  }
}

//-------------------------------------------------------------------------
} // end namespace DensityMatrix
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
