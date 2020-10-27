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

#ifndef _statevector_state_chunk_hpp
#define _statevector_state_chunk_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "simulators/state_chunk.hpp"
#include "qubitvector.hpp"
#ifdef AER_THRUST_SUPPORTED
#define AER_QUBITVECTOR_THRUST_FIRST_DEF
#include "qubitvector_thrust.hpp"
#undef AER_QUBITVECTOR_THRUST_FIRST_DEF
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif


namespace AER {
namespace StatevectorChunk {

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
  // Op types
  {Operations::OpType::gate, Operations::OpType::measure,
    Operations::OpType::reset, Operations::OpType::initialize,
    Operations::OpType::snapshot, Operations::OpType::barrier,
    Operations::OpType::bfunc, Operations::OpType::roerror,
    Operations::OpType::matrix, Operations::OpType::diagonal_matrix,
    Operations::OpType::multiplexer, Operations::OpType::kraus, Operations::OpType::sim_op},
    // Gates
    {"u1",   "u2",   "u3",   "cx",   "cz",   "cy",     "cp",      "cu1",
     "cu2",  "cu3",  "swap", "id",   "p",    "x",      "y",       "z",
     "h",    "s",    "sdg",  "t",    "tdg",  "r",      "rx",      "ry",
     "rz",   "rxx",  "ryy",  "rzz",  "rzx",  "ccx",    "cswap",   "mcx",
     "mcy",  "mcz",  "mcu1", "mcu2", "mcu3", "mcswap", "mcphase", "mcr",
     "mcrx", "mcry", "mcry", "sx",   "csx",  "mcsx", "delay"},
  // Snapshots
    {"statevector", "memory", "register", "probabilities",
     "probabilities_with_variance", "expectation_value_pauli", "density_matrix",
     "density_matrix_with_variance", "expectation_value_pauli_with_variance",
     "expectation_value_matrix_single_shot", "expectation_value_matrix",
     "expectation_value_matrix_with_variance",
     "expectation_value_pauli_single_shot"}
);

// Allowed gates enum class
enum class Gates {
  id, h, s, sdg, t, tdg,
  rxx, ryy, rzz, rzx,
  mcx, mcy, mcz, mcr, mcrx, mcry,
  mcrz, mcp, mcu2, mcu3, mcswap, mcsx
};

// Allowed snapshots enum class
enum class Snapshots {
  statevector,
  cmemory,
  cregister,
  probs,
  probs_var,
  densmat,
  densmat_var,
  expval_pauli,
  expval_pauli_var,
  expval_pauli_shot,
  expval_matrix,
  expval_matrix_var,
  expval_matrix_shot
};

// Enum class for different types of expectation values
enum class SnapshotDataType {average, average_var, pershot};

//=========================================================================
// QubitVector State subclass
//=========================================================================

template <class statevec_t = QV::QubitVector<double>>
class State : public Base::StateChunk<statevec_t> {
public:
  using BaseState = Base::StateChunk<statevec_t>;

  State() : BaseState(StateOpSet)
  {
  }
  virtual ~State();

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {return statevec_t::name();}

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exception will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops = false) override;

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

  virtual void allocate(uint_t num_qubits,uint_t shots);

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

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const int_t iChunk, const reg_t &qubits, RngEngine &rng);

  // Initialize the specified qubits to a given state |psi>
  // by applying a reset to the these qubits and then
  // computing the tensor product with the new state |psi>
  // /psi> is given in params
  void apply_initialize(const int_t iChunk, const reg_t &qubits, const cvector_t &params, RngEngine &rng);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const int_t iChunk,const Operations::Op &op, ExperimentResult &result, bool last_op = false);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const Operations::Op &op);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t & vmat); 

  // Apply a vector of control matrices to given qubits (identity on all other qubits)
  void apply_multiplexer(const int_t iChunk, const reg_t &control_qubits, const reg_t &target_qubits, const std::vector<cmatrix_t> &mmat);

  // Apply stacked (flat) version of multiplexer matrix to target qubits (using control qubits to select matrix instance)
  void apply_multiplexer(const int_t iChunk, const reg_t &control_qubits, const reg_t &target_qubits, const cmatrix_t &mat);


  // Apply a Kraus error operation
  void apply_kraus(const int_t iChunk, const reg_t &qubits,
                   const std::vector<cmatrix_t> &krausops,
                   RngEngine &rng);

  void apply_mcswap(const int_t iChunk,const reg_t &qubits);

  /*
  //swap between chunks
  void apply_chunk_swap(const reg_t &qubits);
  */

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

  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                              SnapshotDataType type);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                             SnapshotDataType type);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                              SnapshotDataType type);

  // Snapshot reduced density matrix
  void snapshot_density_matrix(int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                               SnapshotDataType type);

  // Return the reduced density matrix for the simulator
  cmatrix_t density_matrix(int_t iChunk, const reg_t &qubits);

  // Helper function to convert a vector to a reduced density matrix
  template <class T> cmatrix_t vec2density(const reg_t &qubits, const T &vec);

  //-----------------------------------------------------------------------
  // Single-qubit gate helpers
  //-----------------------------------------------------------------------

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const int_t iChunk, const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Multi-controlled u3
  //-----------------------------------------------------------------------
  
  // Apply N-qubit multi-controlled single qubit waltz gate specified by
  // parameters u3(theta, phi, lambda)
  // NOTE: if N=1 this is just a regular u3 gate.
  void apply_gate_mcu3(const uint_t iChunk, const reg_t& qubits,
                       const double theta,
                       const double phi,
                       const double lambda);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Apply the global phase
  void apply_global_phase();

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

template <class statevec_t>
State<statevec_t>::~State()
{

}

//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

template <class statevec_t>
const stringmap_t<Gates> State<statevec_t>::gateset_({
    // 1-qubit gates
    {"delay", Gates::id},// Delay gate
    {"id", Gates::id},   // Pauli-Identity gate
    {"x", Gates::mcx},   // Pauli-X gate
    {"y", Gates::mcy},   // Pauli-Y gate
    {"z", Gates::mcz},   // Pauli-Z gate
    {"s", Gates::s},     // Phase gate (aka sqrt(Z) gate)
    {"sdg", Gates::sdg}, // Conjugate-transpose of Phase gate
    {"h", Gates::h},     // Hadamard gate (X + Z / sqrt(2))
    {"t", Gates::t},     // T-gate (sqrt(S))
    {"tdg", Gates::tdg}, // Conjguate-transpose of T gate
    {"p", Gates::mcp},   // Parameterized phase gate 
    {"sx", Gates::mcsx}, // Sqrt(X) gate
    // 1-qubit rotation Gates
    {"r", Gates::mcr},   // R rotation gate
    {"rx", Gates::mcrx}, // Pauli-X rotation gate
    {"ry", Gates::mcry}, // Pauli-Y rotation gate
    {"rz", Gates::mcrz}, // Pauli-Z rotation gate
    // Waltz Gates
    {"p", Gates::mcp},   // Parameterized phase gate 
    {"u1", Gates::mcp},  // zero-X90 pulse waltz gate
    {"u2", Gates::mcu2}, // single-X90 pulse waltz gate
    {"u3", Gates::mcu3}, // two X90 pulse waltz gate
    // 2-qubit gates
    {"cx", Gates::mcx},      // Controlled-X gate (CNOT)
    {"cy", Gates::mcy},      // Controlled-Y gate
    {"cz", Gates::mcz},      // Controlled-Z gate
    {"cp", Gates::mcp},      // Controlled-Phase gate 
    {"cu1", Gates::mcp},    // Controlled-u1 gate
    {"cu2", Gates::mcu2},    // Controlled-u2 gate
    {"cu3", Gates::mcu3},    // Controlled-u3 gate
    {"cp", Gates::mcp},      // Controlled-Phase gate 
    {"swap", Gates::mcswap}, // SWAP gate
    {"rxx", Gates::rxx},     // Pauli-XX rotation gate
    {"ryy", Gates::ryy},     // Pauli-YY rotation gate
    {"rzz", Gates::rzz},     // Pauli-ZZ rotation gate
    {"rzx", Gates::rzx},     // Pauli-ZX rotation gate
    {"csx", Gates::mcsx},    // Controlled-Sqrt(X) gate
    // 3-qubit gates
    {"ccx", Gates::mcx},      // Controlled-CX gate (Toffoli)
    {"cswap", Gates::mcswap}, // Controlled SWAP gate (Fredkin)
    // Multi-qubit controlled gates
    {"mcx", Gates::mcx},      // Multi-controlled-X gate
    {"mcy", Gates::mcy},      // Multi-controlled-Y gate
    {"mcz", Gates::mcz},      // Multi-controlled-Z gate
    {"mcr", Gates::mcr},      // Multi-controlled R-rotation gate
    {"mcrx", Gates::mcrx},    // Multi-controlled X-rotation gate
    {"mcry", Gates::mcry},    // Multi-controlled Y-rotation gate
    {"mcrz", Gates::mcrz},    // Multi-controlled Z-rotation gate
    {"mcphase", Gates::mcp},  // Multi-controlled-Phase gate 
    {"mcu1", Gates::mcp},     // Multi-controlled-u1
    {"mcu2", Gates::mcu2},    // Multi-controlled-u2
    {"mcu3", Gates::mcu3},    // Multi-controlled-u3
    {"mcphase", Gates::mcp},  // Multi-controlled-Phase gate 
    {"mcswap", Gates::mcswap},// Multi-controlled SWAP gate
    {"mcsx", Gates::mcsx}     // Multi-controlled-Sqrt(X) gate
});


template <class statevec_t>
const stringmap_t<Snapshots> State<statevec_t>::snapshotset_(
    {{"statevector", Snapshots::statevector},
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
     {"register", Snapshots::cregister}});


//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------
template <class statevec_t>
void State<statevec_t>::allocate(uint_t num_qubits,uint_t shots)
{
  int_t i;
  uint_t nchunks;

  BaseState::num_shots_ = shots;

  BaseState::setup_chunk_bits(num_qubits);

  BaseState::chunk_omp_parallel_ = false;
  if(BaseState::chunk_bits_ < BaseState::num_qubits_){
    if(BaseState::qregs_[0].name() == "statevector_gpu"){
      BaseState::chunk_omp_parallel_ = true;   //CUDA backend requires thread parallelization of chunk loop
    }
  }

  nchunks = BaseState::num_local_chunks_;
  for(i=0;i<BaseState::num_local_chunks_;i++){
    if(this->multi_shot_parallelization_){
      BaseState::qregs_[i].chunk_setup(num_qubits,num_qubits,0,nchunks);
    }
    else{
      BaseState::qregs_[i].chunk_setup(BaseState::chunk_bits_,num_qubits,i + BaseState::global_chunk_index_,nchunks);
    }
    //only first one allocates chunks, others only set chunk index
    nchunks = 0;
  }
}

template <class statevec_t>
void State<statevec_t>::initialize_qreg(uint_t num_qubits) 
{
  int_t i;

  initialize_omp();

  if(BaseState::chunk_bits_ == BaseState::num_qubits_){
//#pragma omp barrier
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_);
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
      BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_);
      if(BaseState::global_chunk_index_ + i == 0 || this->num_qubits_ == this->chunk_bits_){
        BaseState::qregs_[i].initialize();
      }
      else{
        BaseState::qregs_[i].zero();
      }
    }
  }
  apply_global_phase();
}

template <class statevec_t>
void State<statevec_t>::initialize_qreg(uint_t num_qubits,
                               const statevec_t &state)
{
  if (state.num_qubits() != num_qubits) {
    throw std::invalid_argument("QubitVector::State::initialize: initial state does not match qubit number");
  }

  printf(" TEST init statevec\n");

  //TO DO : need multiple states to initialize ...

  apply_global_phase();
}


template <class statevec_t>
void State<statevec_t>::initialize_qreg(uint_t num_qubits,
                                        const cvector_t &state) 
{
  if (state.size() != 1ULL << num_qubits) {
    throw std::invalid_argument("QubitVector::State::initialize: initial state does not match qubit number");
  }

  uint_t i,chunk_offset;

  initialize_omp();

  for(i=0;i<BaseState::num_local_chunks_;i++){
    BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_);

    if(this->num_qubits_ == this->chunk_bits_){
      BaseState::qregs_[i].initialize_from_vector(state);
    }
    else{
      chunk_offset = (i + BaseState::global_chunk_index_) << BaseState::chunk_bits_;

      BaseState::qregs_[i].initialize_from_vector(state);
    }
  }

  apply_global_phase();
}

template <class statevec_t>
void State<statevec_t>::initialize_omp()
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
template <class statevec_t>
void State<statevec_t>::apply_global_phase() 
{
  if (BaseState::has_global_phase_) {
    int_t i;
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].apply_diagonal_matrix({0}, {BaseState::global_phase_, BaseState::global_phase_});
    }
  }
}

template <class statevec_t>
size_t State<statevec_t>::required_memory_mb(uint_t num_qubits,
                                             const std::vector<Operations::Op> &ops)
                                             const 
{
  // An n-qubit state vector as 2^n complex doubles
  // where each complex double is 16 bytes
  (void)ops; // avoid unused variable compiler warning
//  return BaseState::qreg_.required_memory_mb(num_qubits);
  statevec_t tmp;
  return tmp.required_memory_mb(num_qubits);
}

template <class statevec_t>
void State<statevec_t>::set_config(const json_t &config) 
{
  BaseState::set_config(config);

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "zero_threshold", config);
//  BaseState::qreg_.set_json_chop_threshold(json_chop_threshold_);

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold", config);

  // Set the sample measure indexing size
  JSON::get_value(sample_measure_index_size_, "statevector_sample_measure_opt", config);
//    BaseState::qreg_.set_sample_measure_index_size(index_size);

}


//=========================================================================
// Implementation: apply operations
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_ops(const std::vector<Operations::Op> &ops,
                                 ExperimentResult &result,
                                 RngEngine &rng,
                                 bool final_ops)
{
  uint_t iOp,nOp;
  int_t iChunk;

  nOp = ops.size();
  iOp = 0;
  // Simple loop over vector of input operations
  while(iOp < nOp){
    if(BaseState::cregs_[0].check_conditional(ops[iOp])) {
      switch (ops[iOp].type) {
        case Operations::OpType::barrier:
          break;
        case Operations::OpType::reset:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            apply_reset(iChunk,ops[iOp].qubits, rng);
          break;
        case Operations::OpType::initialize:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            apply_initialize(iChunk, ops[iOp].qubits, ops[iOp].params, rng);
          break;
        case Operations::OpType::measure:
          apply_measure(-1, ops[iOp].qubits, ops[iOp].memory, ops[iOp].registers, rng);
          break;
        case Operations::OpType::bfunc:
          BaseState::cregs_[0].apply_bfunc(ops[iOp]);
          break;
        case Operations::OpType::roerror:
          BaseState::cregs_[0].apply_roerror(ops[iOp], rng);
          break;
        case Operations::OpType::gate:
          if(ops[iOp].name == "swap_chunk"){
            BaseState::apply_chunk_swap(ops[iOp].qubits);
          }
          else{
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
            for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
              apply_gate(iChunk,ops[iOp]);
          }
          break;
        case Operations::OpType::snapshot:
          apply_snapshot(-1, ops[iOp], result, final_ops && nOp == iOp + 1);
          break;
        case Operations::OpType::matrix:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            apply_matrix(iChunk,ops[iOp]);
          break;
        case Operations::OpType::multiplexer:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            apply_multiplexer(iChunk,ops[iOp].regs[0], ops[iOp].regs[1], ops[iOp].mats); // control qubits ([0]) & target qubits([1])
          break;
        case Operations::OpType::kraus:
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
          for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
            apply_kraus(iChunk, ops[iOp].qubits, ops[iOp].mats, rng);
          break;
        case Operations::OpType::sim_op:
          if(ops[iOp].name == "begin_blocking"){
            iOp = apply_blocking(ops,iOp + 1);
          }
          else if(ops[iOp].name == "begin_register_blocking"){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
            for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
              BaseState::qregs_[iChunk].enter_register_blocking(ops[iOp].qubits);
          }
          else if(ops[iOp].name == "end_register_blocking"){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
            for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++)
              BaseState::qregs_[iChunk].leave_register_blocking();
          }
          break;
        default:
          throw std::invalid_argument("QubitVector::State::invalid instruction \'" +
                                      ops[iOp].name + "\'.");
      }
    }
    iOp++;
  }


}

template <class statevec_t>
uint_t State<statevec_t>::apply_blocking(const std::vector<Operations::Op> &ops, uint_t op_begin)
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
      if(BaseState::cregs_[iChunk].check_conditional(ops[iOp])) {
        switch (ops[iOp].type){
          case Operations::OpType::gate:
            apply_gate(iChunk,ops[iOp]);
            break;
          case Operations::OpType::matrix:
            apply_matrix(iChunk,ops[iOp]);
            break;
          case Operations::OpType::multiplexer:
            apply_multiplexer(iChunk,ops[iOp].regs[0], ops[iOp].regs[1], ops[iOp].mats); // control qubits ([0]) & target qubits([1])
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

template <class statevec_t>
void State<statevec_t>::apply_snapshot(const int_t iChunk, const Operations::Op &op,
                                       ExperimentResult &result,
                                       bool last_op) 
{
  int_t i;
  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument(
        "QubitVectorState::invalid snapshot instruction \'" + op.name + "\'.");
  switch (it->second) {
    case Snapshots::statevector:
      if (last_op) {
        if(iChunk < 0){
          for(i=0;i<BaseState::num_local_chunks_;i++)
            result.data.add_pershot_snapshot("statevector", op.string_params[0], BaseState::qregs_[i].move_to_vector());
        }
        else{
          result.data.add_pershot_snapshot("statevector", op.string_params[0], BaseState::qregs_[iChunk].move_to_vector());
        }
      } else {
        if(iChunk < 0){
          for(i=0;i<BaseState::num_local_chunks_;i++)
            result.data.add_pershot_snapshot("statevector", op.string_params[0], BaseState::qregs_[i].copy_to_vector());
        }
        else{
          result.data.add_pershot_snapshot("statevector", op.string_params[0], BaseState::qregs_[iChunk].copy_to_vector());
        }
      }
      break;
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(iChunk,op, result);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(iChunk,op, result);
      break;
    case Snapshots::probs: {
      // get probs as hexadecimal
      snapshot_probabilities(iChunk,op, result, SnapshotDataType::average);
    }break;
    case Snapshots::densmat: {
      snapshot_density_matrix(iChunk,op, result, SnapshotDataType::average);
    } break;
    case Snapshots::expval_pauli: {
      snapshot_pauli_expval(iChunk,op, result, SnapshotDataType::average);
    } break;
    case Snapshots::expval_matrix: {
      snapshot_matrix_expval(iChunk,op, result, SnapshotDataType::average);
    } break;
    case Snapshots::probs_var: {
      // get probs as hexadecimal
      snapshot_probabilities(iChunk,op, result, SnapshotDataType::average_var);
    } break;
    case Snapshots::densmat_var: {
      snapshot_density_matrix(iChunk,op, result, SnapshotDataType::average_var);
    } break;
    case Snapshots::expval_pauli_var: {
      snapshot_pauli_expval(iChunk,op, result, SnapshotDataType::average_var);
    } break;
    case Snapshots::expval_matrix_var: {
      snapshot_matrix_expval(iChunk,op, result, SnapshotDataType::average_var);
    } break;
    case Snapshots::expval_pauli_shot: {
      snapshot_pauli_expval(iChunk,op, result, SnapshotDataType::pershot);
    } break;
    case Snapshots::expval_matrix_shot: {
      snapshot_matrix_expval(iChunk,op, result, SnapshotDataType::pershot);
    } break;
    default:
      // We shouldn't get here unless there is a bug in the snapshotset
      throw std::invalid_argument(
          "QubitVector::State::invalid snapshot instruction \'" + op.name +
          "\'.");
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_probabilities(const int_t iChunk, const Operations::Op &op,
                                               ExperimentResult &result,
                                               SnapshotDataType type) 
{
  // get probs as hexadecimal
  auto probs = Utils::vec2ket(measure_probs(iChunk,op.qubits),
                              json_chop_threshold_, 16);
  bool variance = type == SnapshotDataType::average_var;

  if(iChunk < 0){
    result.data.add_average_snapshot("probabilities", op.string_params[0],
                              BaseState::cregs_[0].memory_hex(), probs, variance);
  }
  else{
    result.data.add_average_snapshot("probabilities", op.string_params[0],
                              BaseState::cregs_[iChunk].memory_hex(), probs, variance);
  }
}


template <class statevec_t>
void State<statevec_t>::snapshot_pauli_expval(const int_t iChunk, const Operations::Op &op,
                                               ExperimentResult &result,
                                               SnapshotDataType type) 
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
        auto exp_tmp = coeff * BaseState::qregs_[i].expval_pauli(op.qubits, pauli);
        exp_re += exp_tmp.real();
        exp_im += exp_tmp.imag();
      }
      complex_t t(exp_re,exp_im);
      expval += t;
    }
    else{
      expval += coeff * BaseState::qregs_[iChunk].expval_pauli(op.qubits, pauli);
    }
  }

#ifdef AER_MPI
  if(iChunk < 0){
    complex_t sum;
    MPI_Allreduce(&expval,&sum,2,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
    expval = sum;
  }
#endif

  if(iChunk < 0){
    ireg = 0;
  }
  else{
    ireg = iChunk;
  }

  // Add to snapshot
  Utils::chop_inplace(expval, json_chop_threshold_);
  switch (type) {
    case SnapshotDataType::average:
      result.data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::cregs_[ireg].memory_hex(), expval, false);
      break;
    case SnapshotDataType::average_var:
      result.data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::cregs_[ireg].memory_hex(), expval, true);
      break;
    case SnapshotDataType::pershot:
      result.data.add_pershot_snapshot("expectation_values", op.string_params[0], expval);
      break;
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_matrix_expval(const int_t iChunk, const Operations::Op &op,
                                               ExperimentResult &result,
                                               SnapshotDataType type) 
{
  int_t i,ireg;
  // Check empty edge case
  if (op.params_expval_matrix.empty()) {
    throw std::invalid_argument("Invalid matrix snapshot (components are empty).");
  }
  reg_t qubits = op.qubits;
  // Cache the current quantum state
  if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++)
      BaseState::qregs_[i].checkpoint();
  }
  else{
    BaseState::qregs_[iChunk].checkpoint();
  }
  bool first = true; // flag for first pass so we don't unnecessarily revert from checkpoint

  // Compute expval components
  complex_t expval(0., 0.);
  for (const auto &param : op.params_expval_matrix) {
    complex_t coeff = param.first;
    // Revert the quantum state to cached checkpoint
    if (first)
      first = false;
    else{
      if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
        for(i=0;i<BaseState::num_local_chunks_;i++)
          BaseState::qregs_[i].revert(true);
      }
      else{
          BaseState::qregs_[iChunk].revert(true);
      }
    }
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
        if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i)
          for(i=0;i<BaseState::num_local_chunks_;i++)
            BaseState::qregs_[i].apply_diagonal_matrix(sub_qubits, vmat);
        }
        else{
          BaseState::qregs_[iChunk].apply_diagonal_matrix(sub_qubits, vmat);
        }
      } else {
        if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i)
          for(i=0;i<BaseState::num_local_chunks_;i++)
            BaseState::qregs_[i].apply_matrix(sub_qubits, vmat);
        }
        else{
          BaseState::qregs_[iChunk].apply_matrix(sub_qubits, vmat);
        }
      }

    }

    if(iChunk < 0){
      double exp_re = 0.0;
      double exp_im = 0.0;
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) reduction(+:exp_re,exp_im)
      for(i=0;i<BaseState::num_local_chunks_;i++){
        auto exp_tmp = coeff*BaseState::qregs_[i].inner_product();
        exp_re += exp_tmp.real();
        exp_im += exp_tmp.imag();
      }
      complex_t t(exp_re,exp_im);
      expval += t;
    }
    else{
      expval += coeff*BaseState::qregs_[iChunk].inner_product();
    }
  }

#ifdef AER_MPI
  if(iChunk < 0){
    complex_t sum;
    MPI_Allreduce(&expval,&sum,2,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
    expval = sum;
  }
#endif

  if(iChunk < 0){
    ireg = 0;
  }
  else{
    ireg = iChunk;
  }

  // add to snapshot
  Utils::chop_inplace(expval, json_chop_threshold_);
  switch (type) {
    case SnapshotDataType::average:
      result.data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::cregs_[ireg].memory_hex(), expval, false);
      break;
    case SnapshotDataType::average_var:
      result.data.add_average_snapshot("expectation_value", op.string_params[0],
                            BaseState::cregs_[ireg].memory_hex(), expval, true);
      break;
    case SnapshotDataType::pershot:
      result.data.add_pershot_snapshot("expectation_values", op.string_params[0], expval);
      break;
  }
  // Revert to original state
  if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++)
      BaseState::qregs_[i].revert(false);
  }
  else{
    BaseState::qregs_[iChunk].revert(false);
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_density_matrix(int_t iChunk, const const Operations::Op &op,
                                               ExperimentResult &result,
                                               SnapshotDataType type) 
{
  int_t i;

  if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      cmatrix_t reduced_state;

      // Check if tracing over all qubits
      if (op.qubits.empty()) {
        reduced_state = cmatrix_t(1, 1);
        reduced_state[0] = BaseState::qregs_[i].norm();
      } else {
        reduced_state = density_matrix(i,op.qubits);
      }

      // Add density matrix to result data
      switch (type) {
      case SnapshotDataType::average:
        result.data.add_average_snapshot("density_matrix", op.string_params[0],
                                  BaseState::cregs_[0].memory_hex(),
                                  std::move(reduced_state), false);
        break;
      case SnapshotDataType::average_var:
        result.data.add_average_snapshot("density_matrix", op.string_params[0],
                                  BaseState::cregs_[0].memory_hex(),
                                  std::move(reduced_state), true);
        break;
      case SnapshotDataType::pershot:
        result.data.add_pershot_snapshot("density_matrix", op.string_params[0],
                                  std::move(reduced_state));
        break;
      }
    }
  }
  else{
    cmatrix_t reduced_state;
    // Check if tracing over all qubits
    if (op.qubits.empty()) {
      reduced_state = cmatrix_t(1, 1);
      reduced_state[0] = BaseState::qregs_[iChunk].norm();
    } else {
      reduced_state = density_matrix(iChunk,op.qubits);
    }

    // Add density matrix to result data
    switch (type) {
    case SnapshotDataType::average:
      result.data.add_average_snapshot("density_matrix", op.string_params[0],
                                BaseState::cregs_[iChunk].memory_hex(),
                                std::move(reduced_state), false);
      break;
    case SnapshotDataType::average_var:
      result.data.add_average_snapshot("density_matrix", op.string_params[0],
                                BaseState::cregs_[iChunk].memory_hex(),
                                std::move(reduced_state), true);
      break;
    case SnapshotDataType::pershot:
      result.data.add_pershot_snapshot("density_matrix", op.string_params[0],
                                std::move(reduced_state));
      break;
    }
  }
}

template <class statevec_t>
cmatrix_t State<statevec_t>::density_matrix(int_t iChunk,const reg_t &qubits) {
  return vec2density(qubits, BaseState::qregs_[iChunk].data());
}

#ifdef AER_THRUST_SUPPORTED
template <>
cmatrix_t State<QV::QubitVectorThrust<float>>::density_matrix(int_t iChunk,const reg_t &qubits) {
  return vec2density(qubits, BaseState::qregs_[iChunk].copy_to_vector());
}

template <>
cmatrix_t State<QV::QubitVectorThrust<double>>::density_matrix(int_t iChunk,const reg_t &qubits) {
  return vec2density(qubits, BaseState::qregs_[iChunk].copy_to_vector());
}
#endif

template <class statevec_t>
template <class T>
cmatrix_t State<statevec_t>::vec2density(const reg_t &qubits, const T &vec) {
  const size_t N = qubits.size();
  const size_t DIM = 1ULL << N;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Return full density matrix
  cmatrix_t densmat(DIM, DIM);
  if ((N == BaseState::qregs_[0].num_qubits()) && (qubits == qubits_sorted)) {
    const int_t mask = QV::MASKS[N];
#pragma omp parallel for if (2 * N > omp_qubit_threshold_ &&                   \
                             BaseState::threads_ > 1)                          \
    num_threads(BaseState::threads_)
    for (int_t rowcol = 0; rowcol < int_t(DIM * DIM); ++rowcol) {
      const int_t row = rowcol >> N;
      const int_t col = rowcol & mask;
      densmat(row, col) = complex_t(vec[row]) * complex_t(std::conj(vec[col]));
    }
  } else {
    const size_t END = 1ULL << (BaseState::qregs_[0].num_qubits() - N);
    // Initialize matrix values with first block
    {
      const auto inds = QV::indexes(qubits, qubits_sorted, 0);
      for (size_t row = 0; row < DIM; ++row)
        for (size_t col = 0; col < DIM; ++col) {
          densmat(row, col) =
              complex_t(vec[inds[row]]) * complex_t(std::conj(vec[inds[col]]));
        }
    }
    // Accumulate remaining blocks
    for (size_t k = 1; k < END; k++) {
      // store entries touched by U
      const auto inds = QV::indexes(qubits, qubits_sorted, k);
      for (size_t row = 0; row < DIM; ++row)
        for (size_t col = 0; col < DIM; ++col) {
          densmat(row, col) +=
              complex_t(vec[inds[row]]) * complex_t(std::conj(vec[inds[col]]));
        }
    }
  }
  return densmat;
}


//=========================================================================
// Implementation: Matrix multiplication
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_gate(const uint_t iChunk, const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument("QubitVectorState::invalid gate instruction \'" + 
                                op.name + "\'.");

  switch (it -> second) {
    case Gates::mcx:
      // Includes X, CX, CCX, etc
      BaseState::qregs_[iChunk].apply_mcx(op.qubits);
      break;
    case Gates::mcy:
      // Includes Y, CY, CCY, etc
      BaseState::qregs_[iChunk].apply_mcy(op.qubits);
      break;
    case Gates::mcz:
      // Includes Z, CZ, CCZ, etc
      BaseState::qregs_[iChunk].apply_mcphase(op.qubits, -1);
      break;
    case Gates::mcr:
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::r(op.params[0], op.params[1]));
      break;
    case Gates::mcrx:
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::rx(op.params[0]));
      break;
    case Gates::mcry:
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::ry(op.params[0]));
      break;
    case Gates::mcrz:
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::rz(op.params[0]));
      break;
    case Gates::rxx:
      BaseState::qregs_[iChunk].apply_matrix(op.qubits, Linalg::VMatrix::rxx(op.params[0]));
      break;
    case Gates::ryy:
      BaseState::qregs_[iChunk].apply_matrix(op.qubits, Linalg::VMatrix::ryy(op.params[0]));
      break;
    case Gates::rzz:
      BaseState::qregs_[iChunk].apply_diagonal_matrix(op.qubits, Linalg::VMatrix::rzz_diag(op.params[0]));
      break;
    case Gates::rzx:
      BaseState::qregs_[iChunk].apply_matrix(op.qubits, Linalg::VMatrix::rzx(op.params[0]));
      break;
    case Gates::id:
      break;
    case Gates::h:
      apply_gate_mcu3(iChunk,op.qubits, M_PI / 2., 0., M_PI);
      break;
    case Gates::s:
      apply_gate_phase(iChunk,op.qubits[0], complex_t(0., 1.));
      break;
    case Gates::sdg:
      apply_gate_phase(iChunk,op.qubits[0], complex_t(0., -1.));
      break;
    case Gates::t: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_gate_phase(iChunk,op.qubits[0], complex_t(isqrt2, isqrt2));
    } break;
    case Gates::tdg: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_gate_phase(iChunk,op.qubits[0], complex_t(isqrt2, -isqrt2));
    } break;
    case Gates::mcswap:
      // Includes SWAP, CSWAP, etc
      apply_mcswap(iChunk, op.qubits);
      break;
    case Gates::mcu3:
      // Includes u3, cu3, etc
      apply_gate_mcu3(iChunk,op.qubits,
                      std::real(op.params[0]),
                      std::real(op.params[1]),
                      std::real(op.params[2]));
      break;
    case Gates::mcu2:
      // Includes u2, cu2, etc
      apply_gate_mcu3(iChunk,op.qubits,
                      M_PI / 2.,
                      std::real(op.params[0]),
                      std::real(op.params[1]));
      break;
    case Gates::mcp:
      // Includes u1, cu1, p, cp, mcp etc
      BaseState::qregs_[iChunk].apply_mcphase(op.qubits,
                                     std::exp(complex_t(0, 1) * op.params[0]));
      break;
    case Gates::mcsx:
      // Includes sx, csx, mcsx etc
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::SX);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument("QubitVector::State::invalid gate instruction \'" +
                                  op.name + "\'.");
  }

}


template <class statevec_t>
void State<statevec_t>::apply_multiplexer(const int_t iChunk, const reg_t &control_qubits, const reg_t &target_qubits, const cmatrix_t &mat) {
  if (control_qubits.empty() == false && target_qubits.empty() == false && mat.size() > 0) {
    cvector_t vmat = Utils::vectorize_matrix(mat);

    BaseState::qregs_[iChunk].apply_multiplexer(control_qubits, target_qubits, vmat);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(const int_t iChunk, const Operations::Op &op) 
{
  if (op.qubits.empty() == false && op.mats[0].size() > 0) {
    if (Utils::is_diagonal(op.mats[0], .0)) {
      BaseState::qregs_[iChunk].apply_diagonal_matrix(op.qubits, Utils::matrix_diagonal(op.mats[0]));
    } else {
      BaseState::qregs_[iChunk].apply_matrix(op.qubits, Utils::vectorize_matrix(op.mats[0]));
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    BaseState::qregs_[iChunk].apply_diagonal_matrix(qubits, vmat);
  } else {
    BaseState::qregs_[iChunk].apply_matrix(qubits, vmat);
  }
}


template <class statevec_t>
void State<statevec_t>::apply_gate_mcu3(const uint_t iChunk, const reg_t& qubits,
                                        double theta,
                                        double phi,
                                        double lambda) 
{
  BaseState::qregs_[iChunk].apply_mcu(qubits, Linalg::VMatrix::u3(theta, phi, lambda));
}

template <class statevec_t>
void State<statevec_t>::apply_gate_phase(const int_t iChunk, uint_t qubit, complex_t phase) {
  cvector_t diag = {{1., phase}};
  apply_matrix(iChunk,reg_t({qubit}), diag);
}

template <class statevec_t>
void State<statevec_t>::apply_mcswap(const int_t iChunk,const reg_t &qubits)
{
  BaseState::qregs_[iChunk].apply_mcswap(qubits);
}

//=========================================================================
// Implementation: Reset, Initialize and Measurement Sampling
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_measure(const int_t iChunk, const reg_t &qubits,
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

template <class statevec_t>
rvector_t State<statevec_t>::measure_probs(const int_t iChunk, const reg_t &qubits) const 
{

  if(iChunk < 0){
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
            if(qubits[k] < BaseState::chunk_bits_){
              idx += (((j >> i_in) & 1) << k);
              i_in++;
            }
            else{
              if((((i + BaseState::global_chunk_index_) << BaseState::chunk_bits_) >> qubits[k]) & 1){
                idx += 1ull << k;
              }
            }
          }
#pragma omp atomic 
          sum[idx] += chunkSum[j];
        }
      }
    }

#ifdef AER_MPI
    rvector_t tmp(dim);
    MPI_Allreduce(&sum[0],&tmp[0],dim,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
    for(i=0;i<dim;i++){
      sum[i] = tmp[i];
    }
#endif
    return sum;
  }
  else{
    return BaseState::qregs_[iChunk].probabilities(qubits);
  }
}

template <class statevec_t>
std::vector<reg_t> State<statevec_t>::sample_measure(const reg_t &qubits,
                                                     uint_t shots,
                                                     RngEngine &rng) 
{
  int_t i,j;
  // Generate flat register for storing
  std::vector<double> rnds;
  rnds.reserve(shots);
  reg_t allbit_samples(shots,0);

  for (i = 0; i < shots; ++i)
    rnds.push_back(rng.rand(0, 1));

  if(this->multi_shot_parallelization_){
    /*
    //multi-shot parallelization
    reg_t local_samples(BaseState::num_local_chunks_,0);

    //step 0: store rnds
    for(i=0;i<BaseState::num_local_chunks_;i++){
      std::vector<double> vRnd(1);
      vRnd[0] = rnds[i];
      auto samples = BaseState::qregs_[i].sample_measure(vRnd);
    }
#pragma omp barrier

    //step 1: do sample measure
    for(i=0;i<BaseState::num_local_chunks_;i++){
      std::vector<double> vRnd(1);
      vRnd[0] = rnds[i];
      auto samples = BaseState::qregs_[i].sample_measure(vRnd);
    }
#pragma omp barrier

    //step 2: get results
    */
    for(i=0;i<BaseState::num_local_chunks_;i++){
      std::vector<double> vRnd(1);
      vRnd[0] = rnds[i];
      auto samples = BaseState::qregs_[i].sample_measure(vRnd);

      allbit_samples[i] = samples[0];
    }
  }
  else{
    std::vector<double> chunkSum(BaseState::num_local_chunks_+1,0);
    double sum,localSum;
    //calculate per chunk sum
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      chunkSum[i] = BaseState::qregs_[i].norm();
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

        for(j=0;j<nIn;j++){
          local_samples[vIdx[j]] = ((BaseState::global_chunk_index_ + i) << BaseState::chunk_bits_) + chunkSamples[j];
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

  if(qubits.size() == 0){
    //return all bits if qubits is empty (for multi-shot parallelization)
    for (int_t val : allbit_samples) {
      reg_t allbit_sample = Utils::int2reg(val, 2, BaseState::num_qubits_);
      reg_t sample;
      sample.reserve(BaseState::num_qubits_);
      for (uint_t qubit = 0 ; qubit < BaseState::num_qubits_; qubit++) {
        sample.push_back(allbit_sample[qubit]);
      }
      all_samples.push_back(sample);
    }
  }
  else{
    for (int_t val : allbit_samples) {
      reg_t allbit_sample = Utils::int2reg(val, 2, BaseState::num_qubits_);
      reg_t sample;
      sample.reserve(qubits.size());
      for (uint_t qubit : qubits) {
        sample.push_back(allbit_sample[qubit]);
      }
      all_samples.push_back(sample);
    }
  }
  return all_samples;
}


template <class statevec_t>
void State<statevec_t>::apply_reset(const int_t iChunk, const reg_t &qubits,
                                    RngEngine &rng) {
  // Simulate unobserved measurement
  const auto meas = sample_measure_with_prob(iChunk, qubits, rng);
  // Apply update to reset state
  measure_reset_update(iChunk, qubits, 0, meas.first, meas.second);
}

template <class statevec_t>
std::pair<uint_t, double>
State<statevec_t>::sample_measure_with_prob(const int_t iChunk, const reg_t &qubits,
                                            RngEngine &rng) {
  rvector_t probs = measure_probs(iChunk,qubits);
  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class statevec_t>
void State<statevec_t>::measure_reset_update(const int_t iChunk, const std::vector<uint_t> &qubits,
                                 const uint_t final_state,
                                 const uint_t meas_state,
                                 const double meas_prob) 
{
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement final_state

  int_t i;
  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);
    if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
      for(i=0;i<BaseState::num_local_chunks_;i++)
        apply_matrix(i, qubits, mdiag);
    }
    else{
      apply_matrix(iChunk, qubits, mdiag);
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
        for(i=0;i<BaseState::num_local_chunks_;i++)
          BaseState::qregs_[i].apply_mcx(qubits);
      }
      else{
        BaseState::qregs_[iChunk].apply_mcx(qubits);
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
      for(i=0;i<BaseState::num_local_chunks_;i++)
        apply_matrix(i,qubits, mdiag);
    }
    else{
      apply_matrix(iChunk,qubits, mdiag);
    }

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
      if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
        for(i=0;i<BaseState::num_local_chunks_;i++)
          apply_matrix(i,qubits, perm);
      }
      else{
        apply_matrix(iChunk,qubits, perm);
      }
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_initialize(const int_t iChunk, const reg_t &qubits,
                                         const cvector_t &params,
                                         RngEngine &rng) 
{
  uint_t i;

  if (qubits.size() == BaseState::num_qubits_) {
    // If qubits is all ordered qubits in the statevector
    // we can just initialize the whole state directly
    auto sorted_qubits = qubits;
    std::sort(sorted_qubits.begin(), sorted_qubits.end());
    if (qubits == sorted_qubits) {
      initialize_qreg(qubits.size(), params);
      return;
    }
  }

  /*
  //disable batched execution
  if(iChunk < 0){
    for(i=0;i<BaseState::num_local_chunks_;i++)
      BaseState::qregs_[i].set_batch_bits(0);
  }
  else{
    BaseState::qregs_[iChunk].set_batch_bits(0);
  }
  */

  // Apply reset to qubits
  apply_reset(iChunk,qubits, rng);
  // Apply initialize_component
  if(iChunk < 0){
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].initialize_component(qubits, params);
//      BaseState::qregs_[i].set_batch_bits(1);
    }
  }
  else{
    BaseState::qregs_[iChunk].initialize_component(qubits, params);
//    BaseState::qregs_[iChunk].set_batch_bits(1);
  }
}

//=========================================================================
// Implementation: Multiplexer Circuit
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_multiplexer(const int_t iChunk, const reg_t &control_qubits, const reg_t &target_qubits, const std::vector<cmatrix_t> &mmat) {
	// (1) Pack vector of matrices into single (stacked) matrix ... note: matrix dims: rows = DIM[qubit.size()] columns = DIM[|target bits|]
	cmatrix_t multiplexer_matrix = Utils::stacked_matrix(mmat);

	// (2) Treat as single, large(r), chained/batched matrix operator
	apply_multiplexer(iChunk,control_qubits, target_qubits, multiplexer_matrix);
}


//=========================================================================
// Implementation: Kraus Noise
//=========================================================================
template <class statevec_t>
void State<statevec_t>::apply_kraus(const int_t iChunk, const reg_t &qubits,
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

  int_t i,j;
  cvector_t vmat;
  double local_accum;

  // Loop through N-1 kraus operators
  for (size_t j=0; j < kmats.size() - 1; j++) {
    // Calculate probability
    vmat = Utils::vectorize_matrix(kmats[j]);

    local_accum = 0.0;
    if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) reduction(+:local_accum)
      for(i=0;i<BaseState::num_local_chunks_;i++){
        local_accum += BaseState::qregs_[i].norm(qubits, vmat);
      }

#ifdef AER_MPI
      double global_accum;
      MPI_Allreduce(&local_accum,&global_accum,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
      local_accum = global_accum;
#endif
    }
    else{
      local_accum = BaseState::qregs_[iChunk].norm(qubits, vmat);
    }

    accum += local_accum;

    // check if we need to apply this operator
    if (accum > r) {
      complete = true;
      break;
    }
  }
  if(complete){
    // rescale vmat so projection is normalized
    Utils::scalar_multiply_inplace(vmat, 1 / std::sqrt(local_accum));
    if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i)
      for(i=0;i<BaseState::num_local_chunks_;i++){
        // apply Kraus projection operator
        apply_matrix(i,qubits, vmat);
      }
    }
    else{
      apply_matrix(iChunk,qubits, vmat);
    }
  }
  else{
    // if we haven't applied a kraus operator yet
    // Compute probability from accumulated
    complex_t renorm = 1 / std::sqrt(1. - accum);
    auto mat = Utils::vectorize_matrix(renorm * kmats.back());
    if(iChunk < 0){
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i)
      for(i=0;i<BaseState::num_local_chunks_;i++){
        apply_matrix(i,qubits, mat);
      }
    }
    else{
      apply_matrix(iChunk,qubits, mat);
    }
  }
}

//-------------------------------------------------------------------------
} // end namespace QubitVector
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
