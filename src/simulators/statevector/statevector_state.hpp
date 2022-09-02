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

#include "framework/json.hpp"
#include "framework/utils.hpp"
#include "simulators/state_chunk.hpp"
#include "qubitvector.hpp"
#ifdef AER_THRUST_SUPPORTED
#include "qubitvector_thrust.hpp"
#endif

namespace AER {

namespace Statevector {

using OpType = Operations::OpType;

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {OpType::gate, OpType::measure,
     OpType::reset, OpType::initialize,
     OpType::snapshot, OpType::barrier,
     OpType::bfunc, OpType::roerror,
     OpType::matrix, OpType::diagonal_matrix,
     OpType::multiplexer, OpType::kraus, OpType::qerror_loc,
     OpType::sim_op, OpType::set_statevec,
     OpType::save_expval, OpType::save_expval_var,
     OpType::save_probs, OpType::save_probs_ket,
     OpType::save_amps, OpType::save_amps_sq,
     OpType::save_state, OpType::save_statevec,
     OpType::save_statevec_dict, OpType::save_densmat,
     OpType::jump, OpType::mark
     },
    // Gates
    {"u1",     "u2",      "u3",  "u",    "U",    "CX",   "cx",   "cz",
     "cy",     "cp",      "cu1", "cu2",  "cu3",  "swap", "id",   "p",
     "x",      "y",       "z",   "h",    "s",    "sdg",  "t",    "tdg",
     "r",      "rx",      "ry",  "rz",   "rxx",  "ryy",  "rzz",  "rzx",
     "ccx",    "cswap",   "mcx", "mcy",  "mcz",  "mcu1", "mcu2", "mcu3",
     "mcswap", "mcphase", "mcr", "mcrx", "mcry", "mcry", "sx",   "sxdg",
     "csx", "mcsx", "csxdg", "mcsxdg",  "delay", "pauli", "mcx_gray", "cu", "mcu", "mcp"},
    // Snapshots
    {"statevector", "memory", "register", "probabilities",
     "probabilities_with_variance", "expectation_value_pauli", "density_matrix",
     "density_matrix_with_variance", "expectation_value_pauli_with_variance",
     "expectation_value_matrix_single_shot", "expectation_value_matrix",
     "expectation_value_matrix_with_variance",
     "expectation_value_pauli_single_shot"});

// Allowed gates enum class
enum class Gates {
  id, h, s, sdg, t, tdg,
  rxx, ryy, rzz, rzx,
  mcx, mcy, mcz, mcr, mcrx, mcry,
  mcrz, mcp, mcu2, mcu3, mcu, mcswap, mcsx, mcsxdg, pauli
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
enum class SnapshotDataType { average, average_var, pershot };

//=========================================================================
// QubitVector State subclass
//=========================================================================

template <class statevec_t = QV::QubitVector<double>>
class State : public QuantumState::StateChunk<statevec_t> {
public:
  using BaseState = QuantumState::StateChunk<statevec_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return statevec_t::name(); }

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  void apply_op(const int_t iChunk, const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine& rng,
                        bool final_op = false) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit state
  virtual void initialize_statevector(uint_t num_qubits,
                                      statevec_t &&state);

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is independent of the number of ops
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

  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp();

  auto move_to_vector(const int_t iChunk);
  auto copy_to_vector(const int_t iChunk);

protected:
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

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual void apply_measure(const int_t iChunk, const reg_t &qubits, const reg_t &cmemory,
                             const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const int_t iChunk, const reg_t &qubits, RngEngine &rng);

  // Initialize the specified qubits to a given state |psi>
  // by applying a reset to the these qubits and then
  // computing the tensor product with the new state |psi>
  // /psi> is given in params
  void apply_initialize(const int_t iChunk, const reg_t &qubits, const cvector_t &params,
                        RngEngine &rng);

  void initialize_from_vector(const int_t iChunk, const cvector_t &params);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const int_t iChunk, const Operations::Op &op, ExperimentResult &result, bool last_op = false);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const Operations::Op &op);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t &vmat);

  //apply diagonal matrix
  void apply_diagonal_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t & diag); 

  // Apply a vector of control matrices to given qubits (identity on all other
  // qubits)
  void apply_multiplexer(const int_t iChunk, const reg_t &control_qubits,
                         const reg_t &target_qubits,
                         const std::vector<cmatrix_t> &mmat);

  // Apply stacked (flat) version of multiplexer matrix to target qubits (using
  // control qubits to select matrix instance)
  void apply_multiplexer(const int_t iChunk, const reg_t &control_qubits,
                         const reg_t &target_qubits, const cmatrix_t &mat);

  // Apply a Kraus error operation
  void apply_kraus(const int_t iChunk, const reg_t &qubits, const std::vector<cmatrix_t> &krausops,
                   RngEngine &rng);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current state of the statevector simulator
  // If `last_op` is True this will use move semantics to move the simulator
  // state to the results, otherwise it will use copy semantics to leave
  // the current simulator state unchanged.
  void apply_save_statevector(const int_t iChunk, const Operations::Op &op,
                              ExperimentResult &result,
                              bool last_op);

  // Save the current state of the statevector simulator as a ket-form map.
  void apply_save_statevector_dict(const int_t iChunk, const Operations::Op &op,
                                  ExperimentResult &result);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(const int_t iChunk, const Operations::Op &op,
                                 ExperimentResult &result);

  // Helper function for computing expectation value
  void apply_save_probs(const int_t iChunk, const Operations::Op &op,
                        ExperimentResult &result);

  // Helper function for saving amplitudes and amplitudes squared
  void apply_save_amplitudes(const int_t iChunk, const Operations::Op &op,
                             ExperimentResult &result);

  // Helper function for computing expectation value
  virtual double expval_pauli(const int_t iChunk, const reg_t &qubits,
                              const std::string& pauli) override;
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
  // Apply a supported snapshot instruction
  //
  // IMPORTANT: These methods are not marked const to allow modifying state
  // during snapshot, but after the snapshot is applied the simulator
  // should be left in the pre-snapshot state.
  //-----------------------------------------------------------------------

  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                              SnapshotDataType type);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                             SnapshotDataType type);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                              SnapshotDataType type);

  // Snapshot reduced density matrix
  void snapshot_density_matrix(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                               SnapshotDataType type);

  // Return the reduced density matrix for the simulator
  cmatrix_t density_matrix(const int_t iChunk, const reg_t &qubits);

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

  // Apply N-qubit multi-controlled single qubit gate specified by
  // 4 parameters u4(theta, phi, lambda, gamma)
  // NOTE: if N=1 this is just a regular u4 gate.
  void apply_gate_mcu(const int_t iChunk, const reg_t &qubits, const double theta,
                      const double phi, const double lambda,
                      const double gamma);

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
    {"sxdg", Gates::mcsxdg}, // Inverse Sqrt(X) gate
    // 1-qubit rotation Gates
    {"r", Gates::mcr},   // R rotation gate
    {"rx", Gates::mcrx}, // Pauli-X rotation gate
    {"ry", Gates::mcry}, // Pauli-Y rotation gate
    {"rz", Gates::mcrz}, // Pauli-Z rotation gate
    // Waltz Gates
    {"u1", Gates::mcp},  // zero-X90 pulse waltz gate
    {"u2", Gates::mcu2}, // single-X90 pulse waltz gate
    {"u3", Gates::mcu3}, // two X90 pulse waltz gate
    {"u", Gates::mcu3}, // two X90 pulse waltz gate
    {"U", Gates::mcu3}, // two X90 pulse waltz gate
    // 2-qubit gates
    {"CX", Gates::mcx},      // Controlled-X gate (CNOT)
    {"cx", Gates::mcx},      // Controlled-X gate (CNOT)
    {"cy", Gates::mcy},      // Controlled-Y gate
    {"cz", Gates::mcz},      // Controlled-Z gate
    {"cp", Gates::mcp},      // Controlled-Phase gate 
    {"cu1", Gates::mcp},     // Controlled-u1 gate
    {"cu2", Gates::mcu2},    // Controlled-u2 gate
    {"cu3", Gates::mcu3},    // Controlled-u3 gate
    {"cu", Gates::mcu},      // Controlled-u4 gate
    {"cp", Gates::mcp},      // Controlled-Phase gate 
    {"swap", Gates::mcswap}, // SWAP gate
    {"rxx", Gates::rxx},     // Pauli-XX rotation gate
    {"ryy", Gates::ryy},     // Pauli-YY rotation gate
    {"rzz", Gates::rzz},     // Pauli-ZZ rotation gate
    {"rzx", Gates::rzx},     // Pauli-ZX rotation gate
    {"csx", Gates::mcsx},    // Controlled-Sqrt(X) gate
    {"csxdg", Gates::mcsxdg}, // Controlled-Sqrt(X)dg gate
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
    {"mcp", Gates::mcp},      // Multi-controlled-Phase gate 
    {"mcu1", Gates::mcp},     // Multi-controlled-u1
    {"mcu2", Gates::mcu2},    // Multi-controlled-u2
    {"mcu3", Gates::mcu3},    // Multi-controlled-u3
    {"mcu", Gates::mcu},      // Multi-controlled-u4
    {"mcswap", Gates::mcswap},// Multi-controlled SWAP gate
    {"mcsx", Gates::mcsx},    // Multi-controlled-Sqrt(X) gate
    {"mcsxdg", Gates::mcsxdg}, // Multi-controlled-Sqrt(X)dg gate
    {"pauli", Gates::pauli},   // Multi-qubit Pauli gate
    {"mcx_gray", Gates::mcx}
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
void State<statevec_t>::initialize_qreg(uint_t num_qubits) 
{
  int_t i;
  if(BaseState::qregs_.size() == 0)
    BaseState::allocate(num_qubits,num_qubits,1);

  initialize_omp();

  for(i=0;i<BaseState::qregs_.size();i++){
    BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_);
  }

  if(BaseState::multi_chunk_distribution_){
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++){
          if(BaseState::global_chunk_index_ + iChunk == 0 || this->num_qubits_ == this->chunk_bits_){
            BaseState::qregs_[iChunk].initialize();
          }
          else{
            BaseState::qregs_[iChunk].zero();
          }
        }
      }
    }
    else{
      for(i=0;i<BaseState::qregs_.size();i++){
        if(BaseState::global_chunk_index_ + i == 0 || this->num_qubits_ == this->chunk_bits_){
          BaseState::qregs_[i].initialize();
        }
        else{
          BaseState::qregs_[i].zero();
        }
      }
    }
  }
  else{
    for(i=0;i<BaseState::qregs_.size();i++){
      BaseState::qregs_[i].initialize();
    }
  }
  apply_global_phase();
}

template <class statevec_t>
void State<statevec_t>::initialize_statevector(uint_t num_qubits,
                                               statevec_t &&state) 
{
  if (state.num_qubits() != num_qubits) {
    throw std::invalid_argument("QubitVector::State::initialize: initial state does not match qubit number");
  }

  if (BaseState::qregs_.size() == 1) {
    BaseState::qregs_[0] = std::move(state);
  } else {
    if(BaseState::qregs_.size() == 0)
      BaseState::allocate(num_qubits,num_qubits,1);
    initialize_omp();

    int_t iChunk;
    for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);
    }

    if(BaseState::multi_chunk_distribution_){
      uint_t local_offset = BaseState::global_chunk_index_ << BaseState::chunk_bits_;
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for private(iChunk)
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++)
            BaseState::qregs_[iChunk].initialize_from_data(state.data() + local_offset + (iChunk << BaseState::chunk_bits_), 1ull << BaseState::chunk_bits_);
        }
      }
      else{
        for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++)
          BaseState::qregs_[iChunk].initialize_from_data(state.data() + local_offset + (iChunk << BaseState::chunk_bits_), 1ull << BaseState::chunk_bits_);
      }
    }
    else{
      for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
        BaseState::qregs_[iChunk].initialize_from_data(state.data(), 1ull << BaseState::chunk_bits_);
      }
    }
  }

  apply_global_phase();
}

template <class statevec_t> void State<statevec_t>::initialize_omp() 
{
  uint_t i;

  for(i=0;i<BaseState::qregs_.size();i++){
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
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++)
          BaseState::qregs_[iChunk].apply_diagonal_matrix({0}, {BaseState::global_phase_, BaseState::global_phase_});
      }
    }
    else{
      for(i=0;i<BaseState::qregs_.size();i++)
        BaseState::qregs_[i].apply_diagonal_matrix({0}, {BaseState::global_phase_, BaseState::global_phase_});
    }
  }
}

template <class statevec_t>
size_t State<statevec_t>::required_memory_mb(uint_t num_qubits,
                                             const std::vector<Operations::Op> &ops)
                                             const 
{
  (void)ops; // avoid unused variable compiler warning
  statevec_t tmp;
  return tmp.required_memory_mb(num_qubits);
}

template <class statevec_t>
void State<statevec_t>::set_config(const json_t &config) {
  BaseState::set_config(config);

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "zero_threshold", config);
  for(int_t i=0;i<BaseState::qregs_.size();i++){
    BaseState::qregs_[i].set_json_chop_threshold(json_chop_threshold_);
  }

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold", config);

  // Set the sample measure indexing size
  int index_size;
  if (JSON::get_value(index_size, "statevector_sample_measure_opt", config)) {
    for(int_t i=0;i<BaseState::qregs_.size();i++){
      BaseState::qregs_[i].set_sample_measure_index_size(index_size);
    }
  }
}


template <class statevec_t>
auto State<statevec_t>::move_to_vector(const int_t iChunkIn)
{
  if(BaseState::multi_chunk_distribution_){
    size_t size_required = 2*(sizeof(std::complex<double>) << BaseState::num_qubits_) + (sizeof(std::complex<double>) << BaseState::chunk_bits_)*BaseState::num_local_chunks_;
    if((size_required >> 20) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to store states"));
    }
    int_t iChunk;
    auto state = BaseState::qregs_[0].move_to_vector();
    state.resize(BaseState::num_local_chunks_ << BaseState::chunk_bits_);

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk)
    for(iChunk=1;iChunk<BaseState::qregs_.size();iChunk++){
      auto tmp = BaseState::qregs_[iChunk].move_to_vector();
      uint_t j,offset = iChunk << BaseState::chunk_bits_;
      for(j=0;j<tmp.size();j++){
        state[offset + j] = tmp[j];
      }
    }

#ifdef AER_MPI
    BaseState::gather_state(state);
#endif
    return state;
  }
  else {
    return std::move(BaseState::qregs_[iChunkIn].move_to_vector());
  }
}

template <class statevec_t>
auto State<statevec_t>::copy_to_vector(const int_t iChunkIn)
{
  if(BaseState::multi_chunk_distribution_){
    size_t size_required = 2*(sizeof(std::complex<double>) << BaseState::num_qubits_) + (sizeof(std::complex<double>) << BaseState::chunk_bits_)*BaseState::num_local_chunks_;
    if((size_required >> 20) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to store states"));
    }
    int_t iChunk;
    auto state = BaseState::qregs_[0].copy_to_vector();
    state.resize(BaseState::num_local_chunks_ << BaseState::chunk_bits_);

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk)
    for(iChunk=1;iChunk<BaseState::qregs_.size();iChunk++){
      auto tmp = BaseState::qregs_[iChunk].copy_to_vector();
      uint_t j,offset = iChunk << BaseState::chunk_bits_;
      for(j=0;j<tmp.size();j++){
        state[offset + j] = tmp[j];
      }
    }

#ifdef AER_MPI
    BaseState::gather_state(state);
#endif
    return state;
  }
  else
    return BaseState::qregs_[iChunkIn].copy_to_vector();
}


//=========================================================================
// Implementation: apply operations
//=========================================================================
template <class statevec_t>
void State<statevec_t>::apply_op(const int_t iChunk, const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine& rng,
                        bool final_op)
{
  if(BaseState::check_conditional(iChunk, op)) {
    switch (op.type) {
      case OpType::barrier:
      case OpType::nop:
      case OpType::qerror_loc:
        break;
      case OpType::reset:
        apply_reset(iChunk, op.qubits, rng);
        break;
      case OpType::initialize:
        apply_initialize(iChunk, op.qubits, op.params, rng);
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
        apply_snapshot(iChunk, op, result, final_op);
        break;
      case OpType::matrix:
        apply_matrix(iChunk, op);
        break;
      case OpType::diagonal_matrix:
        apply_diagonal_matrix(iChunk, op.qubits, op.params);
        break;
      case OpType::multiplexer:
        apply_multiplexer(iChunk, op.regs[0], op.regs[1],
                          op.mats); // control qubits ([0]) & target qubits([1])
        break;
      case OpType::kraus:
        apply_kraus(iChunk, op.qubits, op.mats, rng);
        break;
      case OpType::sim_op:
        if(op.name == "begin_register_blocking"){
          BaseState::qregs_[iChunk].enter_register_blocking(op.qubits);
        }
        else if(op.name == "end_register_blocking"){
          BaseState::qregs_[iChunk].leave_register_blocking();
        }
        break;
      case OpType::set_statevec:
        initialize_from_vector(iChunk, op.params);
        break;
      case OpType::save_expval:
      case OpType::save_expval_var:
        BaseState::apply_save_expval(iChunk, op, result);
        break;
      case OpType::save_densmat:
        apply_save_density_matrix(iChunk, op, result);
        break;
      case OpType::save_state:
      case OpType::save_statevec:
        apply_save_statevector(iChunk, op, result, final_op);
        break;
      case OpType::save_statevec_dict:
        apply_save_statevector_dict(iChunk, op, result);
        break;
      case OpType::save_probs:
      case OpType::save_probs_ket:
        apply_save_probs(iChunk, op, result);
        break;
      case OpType::save_amps:
      case OpType::save_amps_sq:
        apply_save_amplitudes(iChunk, op, result);
        break;
      default:
        throw std::invalid_argument(
            "QubitVector::State::invalid instruction \'" + op.name + "\'.");
    }
  }
}

template <class statevec_t>
bool State<statevec_t>::apply_batched_op(const int_t iChunk, 
                                  const Operations::Op &op,
                                  ExperimentResult &result,
                                  std::vector<RngEngine> &rng,
                                  bool final_op) 
{
  if(op.conditional){
    BaseState::qregs_[iChunk].set_conditional(op.conditional_reg);
  }

  switch (op.type) {
    case OpType::barrier:
    case OpType::nop:
    case OpType::qerror_loc:
      break;
    case OpType::reset:
      BaseState::qregs_[iChunk].apply_batched_reset(op.qubits,rng);
      break;
    case OpType::initialize:
      BaseState::qregs_[iChunk].apply_batched_reset(op.qubits,rng);
      BaseState::qregs_[iChunk].initialize_component(op.qubits, op.params);
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
      apply_matrix(iChunk, op);
      break;
    case OpType::diagonal_matrix:
      BaseState::qregs_[iChunk].apply_diagonal_matrix(op.qubits, op.params);
      break;
    case OpType::multiplexer:
      apply_multiplexer(iChunk, op.regs[0], op.regs[1],
                        op.mats); // control qubits ([0]) & target qubits([1])
      break;
    case OpType::kraus:
      BaseState::qregs_[iChunk].apply_batched_kraus(op.qubits, op.mats,rng);
      break;
    case OpType::sim_op:
      if(op.name == "begin_register_blocking"){
        BaseState::qregs_[iChunk].enter_register_blocking(op.qubits);
      }
      else if(op.name == "end_register_blocking"){
        BaseState::qregs_[iChunk].leave_register_blocking();
      }
      else{
        return false;
      }
      break;
    case OpType::set_statevec:
      BaseState::qregs_[iChunk].initialize_from_vector(op.params);
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

template <class statevec_t>
void State<statevec_t>::apply_save_probs(const int_t iChunk, const Operations::Op &op,
                                         ExperimentResult &result) 
{
  // get probs as hexadecimal
  auto probs = measure_probs(iChunk, op.qubits);
  if (op.type == Operations::OpType::save_probs_ket) {
    // Convert to ket dict
    result.save_data_average(BaseState::chunk_creg(iChunk), op.string_params[0],
                             Utils::vec2ket(probs, json_chop_threshold_, 16),
                             op.type, op.save_type);
  } else {
    result.save_data_average(BaseState::chunk_creg(iChunk), op.string_params[0],
                             std::move(probs), op.type, op.save_type);
  }
}


template <class statevec_t>
double State<statevec_t>::expval_pauli(const int_t iChunk, const reg_t &qubits,
                                       const std::string& pauli) 
{
  if(!BaseState::multi_chunk_distribution_)
    return BaseState::qregs_[iChunk].expval_pauli(qubits, pauli);

  //multi-chunk distribution
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

  if(qubits_out_chunk.size() > 0){  //there are bits out of chunk
    std::complex<double> phase = 1.0;

    std::reverse(pauli_out_chunk.begin(),pauli_out_chunk.end());
    std::reverse(pauli_in_chunk.begin(),pauli_in_chunk.end());

    uint_t x_mask, z_mask, num_y, x_max;
    std::tie(x_mask, z_mask, num_y, x_max) = AER::QV::pauli_masks_and_phase(qubits_out_chunk, pauli_out_chunk);

    AER::QV::add_y_phase(num_y,phase);

    if(x_mask != 0){    //pairing state is out of chunk
      bool on_same_process = true;
#ifdef AER_MPI
      int proc_bits = 0;
      uint_t procs = BaseState::distributed_procs_;
      while(procs > 1){
        if((procs & 1) != 0){
          proc_bits = -1;
          break;
        }
        proc_bits++;
        procs >>= 1;
      }
      if(x_mask & (~((1ull << (BaseState::num_qubits_ - proc_bits)) - 1)) != 0){    //data exchange between processes is required
        on_same_process = false;
      }
#endif

      x_mask >>= BaseState::chunk_bits_;
      z_mask >>= BaseState::chunk_bits_;
      x_max -= BaseState::chunk_bits_;

      const uint_t mask_u = ~((1ull << (x_max + 1)) - 1);
      const uint_t mask_l = (1ull << x_max) - 1;
      if(on_same_process){
        auto apply_expval_pauli_chunk = [this, x_mask, z_mask, x_max,mask_u,mask_l, qubits_in_chunk, pauli_in_chunk, phase](int_t iGroup)
        {
          double expval = 0.0;
          for(int_t iChunk = BaseState::top_chunk_of_group_[iGroup];iChunk < BaseState::top_chunk_of_group_[iGroup + 1];iChunk++){
            uint_t pair_chunk = iChunk ^ x_mask;
            if(iChunk < pair_chunk){
              uint_t z_count,z_count_pair;
              z_count = AER::Utils::popcount(iChunk & z_mask);
              z_count_pair = AER::Utils::popcount(pair_chunk & z_mask);

              expval += BaseState::qregs_[iChunk-BaseState::global_chunk_index_].expval_pauli(qubits_in_chunk, pauli_in_chunk,BaseState::qregs_[pair_chunk],z_count,z_count_pair,phase);
            }
          }
          return expval;
        };
        expval += Utils::apply_omp_parallel_for_reduction((BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0),0,BaseState::num_global_chunks_/2,apply_expval_pauli_chunk);
      }
      else{
        for(int_t i=0;i<BaseState::num_global_chunks_/2;i++){
          uint_t iChunk = ((i << 1) & mask_u) | (i & mask_l);
          uint_t pair_chunk = iChunk ^ x_mask;
          uint_t iProc = BaseState::get_process_by_chunk(pair_chunk);
          if(BaseState::chunk_index_begin_[BaseState::distributed_rank_] <= iChunk && BaseState::chunk_index_end_[BaseState::distributed_rank_] > iChunk){  //on this process
            uint_t z_count,z_count_pair;
            z_count = AER::Utils::popcount(iChunk & z_mask);
            z_count_pair = AER::Utils::popcount(pair_chunk & z_mask);

            if(iProc == BaseState::distributed_rank_){  //pair is on the same process
              expval += BaseState::qregs_[iChunk-BaseState::global_chunk_index_].expval_pauli(qubits_in_chunk, pauli_in_chunk,BaseState::qregs_[pair_chunk - BaseState::global_chunk_index_],z_count,z_count_pair,phase);
            }
            else{
              BaseState::recv_chunk(iChunk-BaseState::global_chunk_index_,pair_chunk);
              //refer receive buffer to calculate expectation value
              expval += BaseState::qregs_[iChunk-BaseState::global_chunk_index_].expval_pauli(qubits_in_chunk, pauli_in_chunk,BaseState::qregs_[iChunk-BaseState::global_chunk_index_],z_count,z_count_pair,phase);
            }
          }
          else if(iProc == BaseState::distributed_rank_){  //pair is on this process
            BaseState::send_chunk(iChunk-BaseState::global_chunk_index_,pair_chunk);
          }
        }
      }
    }
    else{ //no exchange between chunks
      z_mask >>= BaseState::chunk_bits_;
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for reduction(+:expval)
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          double e_tmp = 0.0;
          for(int_t iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++){
            double sign = 1.0;
            if (z_mask && (AER::Utils::popcount((iChunk + BaseState::global_chunk_index_) & z_mask) & 1))
              sign = -1.0;
            e_tmp += sign * BaseState::qregs_[iChunk].expval_pauli(qubits_in_chunk, pauli_in_chunk);
          }
          expval += e_tmp;
        }
      }
      else{
        for(i=0;i<BaseState::qregs_.size();i++){
          double sign = 1.0;
          if (z_mask && (AER::Utils::popcount((i + BaseState::global_chunk_index_) & z_mask) & 1))
            sign = -1.0;
          expval += sign * BaseState::qregs_[i].expval_pauli(qubits_in_chunk, pauli_in_chunk);
        }
      }
    }
  }
  else{ //all bits are inside chunk
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for reduction(+:expval)
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        double e_tmp = 0.0;
        for(int_t iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++)
          e_tmp += BaseState::qregs_[iChunk].expval_pauli(qubits, pauli);
        expval += e_tmp;
      }
    }
    else{
      for(i=0;i<BaseState::qregs_.size();i++)
        expval += BaseState::qregs_[i].expval_pauli(qubits, pauli);
    }
  }

#ifdef AER_MPI
  BaseState::reduce_sum(expval);
#endif
  return expval;
}

template <class statevec_t>
void State<statevec_t>::apply_save_statevector(const int_t iChunk, const Operations::Op &op,
                                               ExperimentResult &result,
                                               bool last_op) 
{
  if (op.qubits.size() != BaseState::num_qubits_) {
    throw std::invalid_argument(
        op.name + " was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  std::string key = (op.string_params[0] == "_method_")
                      ? "statevector"
                      : op.string_params[0];

  if (last_op) {
    auto v = move_to_vector(iChunk);
    result.save_data_pershot(BaseState::chunk_creg(iChunk), key, std::move(v),
                                  OpType::save_statevec, op.save_type);
  } else {
    result.save_data_pershot(BaseState::chunk_creg(iChunk), key, copy_to_vector(iChunk),
                                OpType::save_statevec, op.save_type);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_save_statevector_dict(const int_t iChunk, const Operations::Op &op,
                                                   ExperimentResult &result) 
{
  if (op.qubits.size() != BaseState::num_qubits_) {
    throw std::invalid_argument(
        op.name + " was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  if(BaseState::multi_chunk_distribution_){
    auto vec = copy_to_vector(iChunk);
    std::map<std::string, complex_t> result_state_ket;
    for (size_t k = 0; k < vec.size(); ++k) {
      if (std::abs(vec[k]) >= json_chop_threshold_){
        std::string key = Utils::int2hex(k);
        result_state_ket.insert({key, vec[k]});
      }
    }
    result.save_data_pershot(BaseState::chunk_creg(iChunk), op.string_params[0],
                                 std::move(result_state_ket), op.type, op.save_type);
  }
  else{
    auto state_ket = BaseState::qregs_[iChunk].vector_ket(json_chop_threshold_);
    std::map<std::string, complex_t> result_state_ket;
    for (auto const& it : state_ket){
      result_state_ket[it.first] = it.second;
    }
    result.save_data_pershot(BaseState::chunk_creg(iChunk), op.string_params[0],
                                 std::move(result_state_ket), op.type, op.save_type);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_save_density_matrix(const int_t iChunk, const Operations::Op &op,
                                                  ExperimentResult &result) 
{
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    if(!BaseState::multi_chunk_distribution_){
      reduced_state[0] = BaseState::qregs_[iChunk].norm();
    }
    else{
      double sum = 0.0;
      if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for reduction(+:sum)
        for(int_t i=0;i<BaseState::qregs_.size();i++)
          sum += BaseState::qregs_[i].norm();
      }
      else{
        for(int_t i=0;i<BaseState::qregs_.size();i++)
          sum += BaseState::qregs_[i].norm();
      }
#ifdef AER_MPI
      BaseState::reduce_sum(sum);
#endif
      reduced_state[0] = sum;
    }
  } else {
    reduced_state = density_matrix(iChunk, op.qubits);
  }

  result.save_data_average(BaseState::chunk_creg(iChunk), op.string_params[0],
                           std::move(reduced_state), op.type, op.save_type);
}

template <class statevec_t>
void State<statevec_t>::apply_save_amplitudes(const int_t iChunkIn, const Operations::Op &op,
                                              ExperimentResult &result) 
{
  if (op.int_params.empty()) {
    throw std::invalid_argument("Invalid save_amplitudes instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  if (op.type == Operations::OpType::save_amps) {
    Vector<complex_t> amps(size, false);
    if(!BaseState::multi_chunk_distribution_){
      for (int_t i = 0; i < size; ++i) {
        amps[i] = BaseState::qregs_[iChunkIn].get_state(op.int_params[i]);
      }
    }
    else{
      for (int_t i = 0; i < size; ++i) {
        uint_t idx = BaseState::mapped_index(op.int_params[i]);
        uint_t iChunk = idx >> BaseState::chunk_bits_;
        amps[i] = 0.0;
        if(iChunk >= BaseState::global_chunk_index_ && iChunk < BaseState::global_chunk_index_ + BaseState::qregs_.size()){
          amps[i] = BaseState::qregs_[iChunk - BaseState::global_chunk_index_].get_state(idx - (iChunk << BaseState::chunk_bits_));
        }
#ifdef AER_MPI
        complex_t amp = amps[i];
        BaseState::reduce_sum(amp);
        amps[i] = amp;
#endif
      }
    }
    result.save_data_pershot(BaseState::chunk_creg(iChunkIn), op.string_params[0],
                                 std::move(amps), op.type, op.save_type);
  }
  else{
    rvector_t amps_sq(size,0);
    if(!BaseState::multi_chunk_distribution_){
      for (int_t i = 0; i < size; ++i) {
        amps_sq[i] = BaseState::qregs_[iChunkIn].probability(op.int_params[i]);
      }
    }
    else{
      for (int_t i = 0; i < size; ++i) {
        uint_t idx = BaseState::mapped_index(op.int_params[i]);
        uint_t iChunk = idx >> BaseState::chunk_bits_;
        if(iChunk >= BaseState::global_chunk_index_ && iChunk < BaseState::global_chunk_index_ + BaseState::qregs_.size()){
          amps_sq[i] = BaseState::qregs_[iChunk - BaseState::global_chunk_index_].probability(idx - (iChunk << BaseState::chunk_bits_));
        }
      }
#ifdef AER_MPI
      BaseState::reduce_sum(amps_sq);
#endif
    }
   result.save_data_average(BaseState::chunk_creg(iChunkIn), op.string_params[0],
                            std::move(amps_sq), op.type, op.save_type);
  }
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_snapshot(const int_t iChunk, const Operations::Op &op,
                                       ExperimentResult &result,
                                       bool last_op) {

  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument(
        "QubitVectorState::invalid snapshot instruction \'" + op.name + "\'.");
  switch (it->second) {
    case Snapshots::statevector:
      if (last_op) {
        result.legacy_data.add_pershot_snapshot("statevector", op.string_params[0],
                                         move_to_vector(iChunk));
      } else {
        result.legacy_data.add_pershot_snapshot("statevector", op.string_params[0],
                                         copy_to_vector(iChunk));
      }
      break;
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(iChunk, op, result);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(iChunk, op, result);
      break;
    case Snapshots::probs: {
      // get probs as hexadecimal
      snapshot_probabilities(iChunk, op, result, SnapshotDataType::average);
    } break;
    case Snapshots::densmat: {
      snapshot_density_matrix(iChunk, op, result, SnapshotDataType::average);
    } break;
    case Snapshots::expval_pauli: {
      snapshot_pauli_expval(iChunk, op, result, SnapshotDataType::average);
    } break;
    case Snapshots::expval_matrix: {
      snapshot_matrix_expval(iChunk, op, result, SnapshotDataType::average);
    } break;
    case Snapshots::probs_var: {
      // get probs as hexadecimal
      snapshot_probabilities(iChunk, op, result, SnapshotDataType::average_var);
    } break;
    case Snapshots::densmat_var: {
      snapshot_density_matrix(iChunk, op, result, SnapshotDataType::average_var);
    } break;
    case Snapshots::expval_pauli_var: {
      snapshot_pauli_expval(iChunk, op, result, SnapshotDataType::average_var);
    } break;
    case Snapshots::expval_matrix_var: {
      snapshot_matrix_expval(iChunk, op, result, SnapshotDataType::average_var);
    } break;
    case Snapshots::expval_pauli_shot: {
      snapshot_pauli_expval(iChunk, op, result, SnapshotDataType::pershot);
    } break;
    case Snapshots::expval_matrix_shot: {
      snapshot_matrix_expval(iChunk, op, result, SnapshotDataType::pershot);
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
  int_t ishot = BaseState::get_global_shot_index(iChunk);

  auto probs =
      Utils::vec2ket(measure_probs(iChunk, op.qubits), json_chop_threshold_, 16);
  bool variance = type == SnapshotDataType::average_var;
  result.legacy_data.add_average_snapshot("probabilities", op.string_params[0],
                                   BaseState::cregs_[ishot].memory_hex(),
                                   std::move(probs), variance);
}

template <class statevec_t>
void State<statevec_t>::snapshot_pauli_expval(const int_t iChunk, const Operations::Op &op,
                                              ExperimentResult &result,
                                              SnapshotDataType type) 
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
  switch (type) {
  case SnapshotDataType::average:
    result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                              BaseState::cregs_[ishot].memory_hex(), expval, false);
    break;
  case SnapshotDataType::average_var:
    result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                              BaseState::cregs_[ishot].memory_hex(), expval, true);
    break;
  case SnapshotDataType::pershot:
    result.legacy_data.add_pershot_snapshot("expectation_values", op.string_params[0],
                              expval);
    break;
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_matrix_expval(const int_t iChunk, const Operations::Op &op,
                                               ExperimentResult &result,
                                               SnapshotDataType type) 
{
  // Check empty edge case
  if (op.params_expval_matrix.empty()) {
    throw std::invalid_argument(
        "Invalid matrix snapshot (components are empty).");
  }
  int_t ishot = BaseState::get_global_shot_index(iChunk);

  reg_t qubits = op.qubits;
  // Cache the current quantum state
  if(!BaseState::multi_chunk_distribution_)
    BaseState::qregs_[iChunk].checkpoint();
  else{
    if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for 
      for(int_t i=0;i<BaseState::qregs_.size();i++)
        BaseState::qregs_[i].checkpoint();
    }
    else{
      for(int_t i=0;i<BaseState::qregs_.size();i++)
        BaseState::qregs_[i].checkpoint();
    }
  }

  bool first = true; // flag for first pass so we don't unnecessarily revert
                     // from checkpoint

  // Compute expval components
  complex_t expval(0., 0.);
  for (const auto &param : op.params_expval_matrix) {
    complex_t coeff = param.first;
    // Revert the quantum state to cached checkpoint
    if (first)
      first = false;
    else{
      if(!BaseState::multi_chunk_distribution_)
        BaseState::qregs_[iChunk].revert(true);
      else{
        if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for 
          for(int_t i=0;i<BaseState::qregs_.size();i++)
            BaseState::qregs_[i].revert(true);
        }
        else{
          for(int_t i=0;i<BaseState::qregs_.size();i++)
            BaseState::qregs_[i].revert(true);
        }
      }
    }
    // Apply each matrix component
    for (const auto &pair : param.second) {
      reg_t sub_qubits;
      for (const auto &pos : pair.first) {
        sub_qubits.push_back(qubits[pos]);
      }
      const cmatrix_t &mat = pair.second;
      cvector_t vmat =
          (mat.GetColumns() == 1)
              ? Utils::vectorize_matrix(Utils::projector(
                    Utils::vectorize_matrix(mat))) // projector case
              : Utils::vectorize_matrix(mat); // diagonal or square matrix case

      if (vmat.size() == 1ULL << qubits.size()) {
        if(!BaseState::multi_chunk_distribution_)
          apply_diagonal_matrix(iChunk, sub_qubits, vmat);
        else{
          if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for
            for(int_t i=0;i<BaseState::qregs_.size();i++)
              apply_diagonal_matrix(i, sub_qubits, vmat);
          }
          else{
            for(int_t i=0;i<BaseState::qregs_.size();i++)
              apply_diagonal_matrix(i, sub_qubits, vmat);
          }
        }
      } else {
        if(!BaseState::multi_chunk_distribution_)
          BaseState::qregs_[iChunk].apply_matrix(sub_qubits, vmat);
        else{
          if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for 
            for(int_t i=0;i<BaseState::qregs_.size();i++)
              BaseState::qregs_[i].apply_matrix(sub_qubits, vmat);
          }
          else{
            for(int_t i=0;i<BaseState::qregs_.size();i++)
              BaseState::qregs_[i].apply_matrix(sub_qubits, vmat);
          }
        }
      }
    }
    double exp_re = 0.0;
    double exp_im = 0.0;
    if(!BaseState::multi_chunk_distribution_){
      auto exp_tmp = coeff*BaseState::qregs_[iChunk].inner_product();
      exp_re += exp_tmp.real();
      exp_im += exp_tmp.imag();
    }
    else{
      if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for reduction(+:exp_re,exp_im)
        for(int_t i=0;i<BaseState::qregs_.size();i++){
          auto exp_tmp = coeff*BaseState::qregs_[i].inner_product();
          exp_re += exp_tmp.real();
          exp_im += exp_tmp.imag();
        }
      }
      else{
        for(int_t i=0;i<BaseState::qregs_.size();i++){
          auto exp_tmp = coeff*BaseState::qregs_[i].inner_product();
          exp_re += exp_tmp.real();
          exp_im += exp_tmp.imag();
        }
      }
    }
    complex_t t(exp_re,exp_im);
    expval += t;
  }
#ifdef AER_MPI
  if(BaseState::multi_chunk_distribution_)
    BaseState::reduce_sum(expval);
#endif

  // add to snapshot
  Utils::chop_inplace(expval, json_chop_threshold_);
  switch (type) {
  case SnapshotDataType::average:
    result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                              BaseState::cregs_[ishot].memory_hex(), expval, false);
    break;
  case SnapshotDataType::average_var:
    result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                              BaseState::cregs_[ishot].memory_hex(), expval, true);
    break;
  case SnapshotDataType::pershot:
    result.legacy_data.add_pershot_snapshot("expectation_values", op.string_params[0],
                              expval);
    break;
  }
  // Revert to original state
  if(!BaseState::multi_chunk_distribution_)
    BaseState::qregs_[iChunk].revert(false);
  else{
    if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for 
      for(int_t i=0;i<BaseState::qregs_.size();i++)
        BaseState::qregs_[i].revert(false);
    }
    else{
      for(int_t i=0;i<BaseState::qregs_.size();i++)
        BaseState::qregs_[i].revert(false);
    }
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_density_matrix(const int_t iChunk, const Operations::Op &op,
                                                ExperimentResult &result,
                                                SnapshotDataType type) {
  cmatrix_t reduced_state;
  int_t ishot = BaseState::get_global_shot_index(iChunk);

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    if(!BaseState::multi_chunk_distribution_)
      reduced_state[0] = BaseState::qregs_[iChunk].norm();
    else{
      double sum = 0.0;
      if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for reduction(+:sum)
        for(int_t i=0;i<BaseState::qregs_.size();i++)
          sum += BaseState::qregs_[i].norm();
      }
      else{
        for(int_t i=0;i<BaseState::qregs_.size();i++)
          sum += BaseState::qregs_[i].norm();
      }
#ifdef AER_MPI
      BaseState::reduce_sum(sum);
#endif
      reduced_state[0] = sum;
    }
  } else {
    reduced_state = density_matrix(iChunk, op.qubits);
  }

  // Add density matrix to result data
  switch (type) {
  case SnapshotDataType::average:
    result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                              BaseState::cregs_[ishot].memory_hex(),
                              std::move(reduced_state), false);
    break;
  case SnapshotDataType::average_var:
    result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                              BaseState::cregs_[ishot].memory_hex(),
                              std::move(reduced_state), true);
    break;
  case SnapshotDataType::pershot:
    result.legacy_data.add_pershot_snapshot("density_matrix", op.string_params[0],
                              std::move(reduced_state));
    break;
  }
}

template <class statevec_t>
cmatrix_t State<statevec_t>::density_matrix(const int_t iChunk, const reg_t &qubits) 
{
  return vec2density(qubits, copy_to_vector(iChunk));
}

template <class statevec_t>
template <class T>
cmatrix_t State<statevec_t>::vec2density(const reg_t &qubits, const T &vec) {
  const size_t N = qubits.size();
  const size_t DIM = 1ULL << N;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Return full density matrix
  cmatrix_t densmat(DIM, DIM);
  if ((N == BaseState::num_qubits_) && (qubits == qubits_sorted)) {
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
    const size_t END = 1ULL << (BaseState::num_qubits_ - N);
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
void State<statevec_t>::apply_gate(const int_t iChunk, const Operations::Op &op) 
{
  if(!BaseState::global_chunk_indexing_){
    reg_t qubits_in,qubits_out;
    BaseState::get_inout_ctrl_qubits(op,qubits_out,qubits_in);
    if(qubits_out.size() > 0){
      uint_t mask = 0;
      for(int i=0;i<qubits_out.size();i++){
        mask |= (1ull << (qubits_out[i] - BaseState::chunk_bits_));
      }
      if(((BaseState::global_chunk_index_ + iChunk) & mask) == mask){
        Operations::Op new_op = BaseState::remake_gate_in_chunk_qubits(op,qubits_in);
        apply_gate(iChunk, new_op);
      }
      return;
    }
  }

  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument(
        "QubitVectorState::invalid gate instruction \'" + op.name + "\'.");
  switch (it->second) {
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
      BaseState::qregs_[iChunk].apply_rotation(op.qubits, QV::Rotation::x, std::real(op.params[0]));
      break;
    case Gates::mcry:
      BaseState::qregs_[iChunk].apply_rotation(op.qubits, QV::Rotation::y, std::real(op.params[0]));
      break;
    case Gates::mcrz:
      BaseState::qregs_[iChunk].apply_rotation(op.qubits, QV::Rotation::z, std::real(op.params[0]));
      break;
    case Gates::rxx:
      BaseState::qregs_[iChunk].apply_rotation(op.qubits, QV::Rotation::xx, std::real(op.params[0]));
      break;
    case Gates::ryy:
      BaseState::qregs_[iChunk].apply_rotation(op.qubits, QV::Rotation::yy, std::real(op.params[0]));
      break;
    case Gates::rzz:
      BaseState::qregs_[iChunk].apply_rotation(op.qubits, QV::Rotation::zz, std::real(op.params[0]));
      break;
    case Gates::rzx:
      BaseState::qregs_[iChunk].apply_rotation(op.qubits, QV::Rotation::zx, std::real(op.params[0]));
      break;
    case Gates::id:
      break;
    case Gates::h:
      apply_gate_mcu(iChunk, op.qubits, M_PI / 2., 0., M_PI, 0.);
      break;
    case Gates::s:
      apply_gate_phase(iChunk, op.qubits[0], complex_t(0., 1.));
      break;
    case Gates::sdg:
      apply_gate_phase(iChunk, op.qubits[0], complex_t(0., -1.));
      break;
    case Gates::t: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_gate_phase(iChunk, op.qubits[0], complex_t(isqrt2, isqrt2));
    } break;
    case Gates::tdg: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_gate_phase(iChunk, op.qubits[0], complex_t(isqrt2, -isqrt2));
    } break;
    case Gates::mcswap:
      // Includes SWAP, CSWAP, etc
      BaseState::qregs_[iChunk].apply_mcswap(op.qubits);
      break;
    case Gates::mcu3:
      // Includes u3, cu3, etc
      apply_gate_mcu(iChunk, op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                     std::real(op.params[2]), 0.);
      break;
    case Gates::mcu:
      // Includes u3, cu3, etc
      apply_gate_mcu(iChunk, op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                      std::real(op.params[2]), std::real(op.params[3]));
      break;
    case Gates::mcu2:
      // Includes u2, cu2, etc
      apply_gate_mcu(iChunk, op.qubits, M_PI / 2., std::real(op.params[0]),
                     std::real(op.params[1]), 0.);
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
    case Gates::mcsxdg:
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::SXDG);
      break;
    case Gates::pauli:
      BaseState::qregs_[iChunk].apply_pauli(op.qubits, op.string_params[0]);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument(
          "QubitVector::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

template <class statevec_t>
void State<statevec_t>::apply_multiplexer(const int_t iChunk, const reg_t &control_qubits,
                                          const reg_t &target_qubits,
                                          const cmatrix_t &mat) 
{
  if (control_qubits.empty() == false && target_qubits.empty() == false &&
      mat.size() > 0) {
    cvector_t vmat = Utils::vectorize_matrix(mat);
    BaseState::qregs_[iChunk].apply_multiplexer(control_qubits, target_qubits, vmat);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(const int_t iChunk, const Operations::Op &op) 
{
  if (op.qubits.empty() == false && op.mats[0].size() > 0) {
    if (Utils::is_diagonal(op.mats[0], .0)) {
      apply_diagonal_matrix(iChunk, op.qubits, Utils::matrix_diagonal(op.mats[0]));
    } else {
      BaseState::qregs_[iChunk].apply_matrix(op.qubits,
                                    Utils::vectorize_matrix(op.mats[0]));
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(const int_t iChunk, const reg_t &qubits,
                                     const cvector_t &vmat) 
{
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    apply_diagonal_matrix(iChunk, qubits, vmat);
  } else {
    BaseState::qregs_[iChunk].apply_matrix(qubits, vmat);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_diagonal_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t & diag)
{
  if(BaseState::global_chunk_indexing_ || !BaseState::multi_chunk_distribution_){
    //GPU computes all chunks in one kernel, so pass qubits and diagonal matrix as is
    BaseState::qregs_[iChunk].apply_diagonal_matrix(qubits,diag);
  }
  else{
    reg_t qubits_in = qubits;
    cvector_t diag_in = diag;

    BaseState::block_diagonal_matrix(iChunk,qubits_in,diag_in);
    BaseState::qregs_[iChunk].apply_diagonal_matrix(qubits_in,diag_in);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_gate_mcu(const int_t iChunk, const reg_t &qubits, double theta,
                                       double phi, double lambda, double gamma) 
{
  BaseState::qregs_[iChunk].apply_mcu(qubits, Linalg::VMatrix::u4(theta, phi, lambda, gamma));
}

template <class statevec_t>
void State<statevec_t>::apply_gate_phase(const int_t iChunk, uint_t qubit, complex_t phase) 
{
  cvector_t diag = {{1., phase}};
  apply_diagonal_matrix(iChunk, reg_t({qubit}), diag);
}

//=========================================================================
// Implementation: Reset, Initialize and Measurement Sampling
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_measure(const int_t iChunk, const reg_t &qubits, const reg_t &cmemory,
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

template <class statevec_t>
rvector_t State<statevec_t>::measure_probs(const int_t iChunk, const reg_t &qubits) const 
{
  if(!BaseState::multi_chunk_distribution_)
    return BaseState::qregs_[iChunk].probabilities(qubits);

  uint_t dim = 1ull << qubits.size();
  rvector_t sum(dim,0.0);
  int_t i,j,k;
  reg_t qubits_in_chunk;
  reg_t qubits_out_chunk;

  BaseState::qubits_inout(qubits,qubits_in_chunk,qubits_out_chunk);

  if(qubits_in_chunk.size() > 0){
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for private(i,j,k) 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++){
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
      }
    }
    else{
      for(i=0;i<BaseState::qregs_.size();i++){
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
            sum[idx] += chunkSum[j];
          }
        }
      }
    }
  }
  else{ //there is no bit in chunk
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for private(i,j,k) 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++){
          auto nr = std::real(BaseState::qregs_[i].norm());
          int idx = 0;
          for(k=0;k<qubits_out_chunk.size();k++){
            if((((i + BaseState::global_chunk_index_) << (BaseState::chunk_bits_)) >> qubits_out_chunk[k]) & 1){
              idx += 1ull << k;
            }
          }
#pragma omp atomic
          sum[idx] += nr;
        }
      }
    }
    else{
      for(i=0;i<BaseState::qregs_.size();i++){
        auto nr = std::real(BaseState::qregs_[i].norm());
        int idx = 0;
        for(k=0;k<qubits_out_chunk.size();k++){
          if((((i + BaseState::global_chunk_index_) << (BaseState::chunk_bits_)) >> qubits_out_chunk[k]) & 1){
            idx += 1ull << k;
          }
        }
        sum[idx] += nr;
      }
    }
  }

#ifdef AER_MPI
  BaseState::reduce_sum(sum);
#endif

  return sum;

}

template <class statevec_t>
void State<statevec_t>::apply_reset(const int_t iChunk, const reg_t &qubits, RngEngine &rng) {
  // Simulate unobserved measurement
  const auto meas = sample_measure_with_prob(iChunk, qubits, rng);
  // Apply update to reset state
  measure_reset_update(iChunk, qubits, 0, meas.first, meas.second);
}

template <class statevec_t>
std::pair<uint_t, double>
State<statevec_t>::sample_measure_with_prob(const int_t iChunk, const reg_t &qubits,
                                            RngEngine &rng) {
  rvector_t probs = measure_probs(iChunk, qubits);
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
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state

  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    cvector_t mdiag(2, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    if(!BaseState::multi_chunk_distribution_)
      BaseState::qregs_[iChunk].apply_diagonal_matrix(qubits, mdiag);
    else{
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for  
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_diagonal_matrix(ic, qubits, mdiag);
        }
      }
      else{
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_diagonal_matrix(ic, qubits, mdiag);
        }
      }
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      if(!BaseState::multi_chunk_distribution_)
        BaseState::qregs_[iChunk].apply_mcx(qubits);
      else
        BaseState::apply_chunk_x(qubits[0]);
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    if(!BaseState::multi_chunk_distribution_)
      BaseState::qregs_[iChunk].apply_diagonal_matrix(qubits, mdiag);
    else{
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_diagonal_matrix(ic, qubits, mdiag);
        }
      }
      else{
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_diagonal_matrix(ic, qubits, mdiag);
        }
      }
    }

    // If it doesn't agree with the reset state update
    // This function could be optimized as a permutation update
    if (final_state != meas_state) {
      reg_t qubits_in_chunk;
      reg_t qubits_out_chunk;

      BaseState::qubits_inout(qubits,qubits_in_chunk,qubits_out_chunk);

      if(!BaseState::multi_chunk_distribution_ || qubits_in_chunk.size() == qubits.size()){   //all bits are inside chunk
        // build vectorized permutation matrix
        cvector_t perm(dim * dim, 0.);
        perm[final_state * dim + meas_state] = 1.;
        perm[meas_state * dim + final_state] = 1.;
        for (size_t j = 0; j < dim; j++) {
          if (j != final_state && j != meas_state)
            perm[j * dim + j] = 1.;
        }
        // apply permutation to swap state
        apply_matrix(iChunk, qubits, perm);
      }
      else{
        for(int_t i=0;i<qubits.size();i++){
          if(((final_state >> i) & 1) != ((meas_state >> i) & 1)){
            BaseState::apply_chunk_x(qubits[i]);
          }
        }
      }
    }
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

  if(!BaseState::multi_chunk_distribution_)
    allbit_samples = BaseState::qregs_[0].sample_measure(rnds);
  else{
    std::vector<double> chunkSum(BaseState::qregs_.size()+1,0);
    double sum,localSum;

    //calculate per chunk sum
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++){
          bool batched = BaseState::qregs_[ic].enable_batch(true);   //return sum of all chunks in group
          chunkSum[ic] = BaseState::qregs_[ic].norm();
          BaseState::qregs_[ic].enable_batch(batched);
        }
      }
    }
    else{
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++){
          bool batched = BaseState::qregs_[ic].enable_batch(true);   //return sum of all chunks in group
          chunkSum[ic] = BaseState::qregs_[ic].norm();
          BaseState::qregs_[ic].enable_batch(batched);
        }
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

        for(j=0;j<chunkSamples.size();j++){
          local_samples[vIdx[j]] = ((BaseState::global_chunk_index_ + i) << BaseState::chunk_bits_) + chunkSamples[j];
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

template <class statevec_t>
void State<statevec_t>::apply_initialize(const int_t iChunk, const reg_t &qubits,
                                         const cvector_t &params,
                                         RngEngine &rng) 
{
  auto sorted_qubits = qubits;
  std::sort(sorted_qubits.begin(), sorted_qubits.end());
  if (qubits.size() == BaseState::num_qubits_) {
    // If qubits is all ordered qubits in the statevector
    // we can just initialize the whole state directly
    if (qubits == sorted_qubits) {
      initialize_from_vector(iChunk, params);
      return;
    }
  }
  // Apply reset to qubits
  apply_reset(iChunk, qubits, rng);

  // Apply initialize_component
  if(!BaseState::multi_chunk_distribution_)
    BaseState::qregs_[iChunk].initialize_component(qubits, params);
  else{
    reg_t qubits_in_chunk;
    reg_t qubits_out_chunk;
    BaseState::qubits_inout(qubits,qubits_in_chunk,qubits_out_chunk);

    if(qubits_out_chunk.size() == 0){   //no qubits outside of chunk
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
            BaseState::qregs_[i].initialize_component(qubits, params);
        }
      }
      else{
        for(int_t i=0;i<BaseState::qregs_.size();i++)
          BaseState::qregs_[i].initialize_component(qubits, params);
      }
    }
    else{
      //scatter base states
      if(qubits_in_chunk.size() > 0){
        //scatter inside chunks
        const size_t dim = 1ULL << qubits_in_chunk.size();
        cvector_t perm(dim * dim, 0.);
        for(int_t i=0;i<dim;i++){
          perm[i] = 1.0;
        }

        if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for 
          for(int_t i=0;i<BaseState::qregs_.size();i++)
            apply_matrix(i, qubits_in_chunk, perm );
        }
        else{
          for(int_t i=0;i<BaseState::qregs_.size();i++)
            apply_matrix(i, qubits_in_chunk, perm );
        }
      }
      if(qubits_out_chunk.size() > 0){
        //then scatter outside chunk
        auto sorted_qubits_out = qubits_out_chunk;
        std::sort(sorted_qubits_out.begin(), sorted_qubits_out.end());

        for(int_t i=0;i<(1ull << (BaseState::num_qubits_ - BaseState::chunk_bits_ - qubits_out_chunk.size()));i++){
          uint_t baseChunk = 0;
          uint_t j,ii,t;
          ii = i;
          for(j=0;j<qubits_out_chunk.size();j++){
            t = ii & ((1ull << qubits_out_chunk[j])-1);
            baseChunk += t;
            ii = (ii - t) << 1;
          }
          baseChunk += ii;
          baseChunk >>= BaseState::chunk_bits_;

          for(j=1;j<(1ull << qubits_out_chunk.size());j++){
            int_t ic = baseChunk;
            for(t=0;t<qubits_out_chunk.size();t++){
              if((j >> t) & 1)
                ic += (1ull << (qubits_out_chunk[t] - BaseState::chunk_bits_));
            }

            if(ic >= BaseState::chunk_index_begin_[BaseState::distributed_rank_] && ic < BaseState::chunk_index_end_[BaseState::distributed_rank_]){    //on this process
              if(baseChunk >= BaseState::chunk_index_begin_[BaseState::distributed_rank_] && baseChunk < BaseState::chunk_index_end_[BaseState::distributed_rank_]){    //base chunk is on this process
                BaseState::qregs_[ic].initialize_from_data(BaseState::qregs_[baseChunk].data(),1ull << BaseState::chunk_bits_);
              }
              else{
                BaseState::recv_chunk(ic,baseChunk);
                //using swap chunk function to release send/recv buffers for Thrust
                reg_t swap(2);
                swap[0] = BaseState::chunk_bits_;
                swap[1] = BaseState::chunk_bits_;
                BaseState::qregs_[ic].apply_chunk_swap(swap,baseChunk);
              }
            }
            else if(baseChunk >= BaseState::chunk_index_begin_[BaseState::distributed_rank_] && baseChunk < BaseState::chunk_index_end_[BaseState::distributed_rank_]){    //base chunk is on this process
              BaseState::send_chunk(baseChunk - BaseState::global_chunk_index_,ic);
            }
          }
        }
      }

      //initialize by params
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
            apply_diagonal_matrix(i, qubits,params );
        }
      }
      else{
        for(int_t i=0;i<BaseState::qregs_.size();i++)
          apply_diagonal_matrix(i, qubits,params );
      }
    }
  }
}

template <class statevec_t>
void State<statevec_t>::initialize_from_vector(const int_t iChunk, const cvector_t &params)
{
  if(!BaseState::multi_chunk_distribution_)
    BaseState::qregs_[iChunk].initialize_from_vector(params);
  else{   //multi-chunk distribution
    uint_t local_offset = BaseState::global_chunk_index_ << BaseState::chunk_bits_;

#pragma omp parallel for if(BaseState::chunk_omp_parallel_)
    for(int_t i=0;i<BaseState::qregs_.size();i++){
      //copy part of state for this chunk
      cvector_t tmp(1ull << BaseState::chunk_bits_);
      std::copy(params.begin() + local_offset + (i << BaseState::chunk_bits_),
                params.begin() + local_offset + ((i+1) << BaseState::chunk_bits_),
                tmp.begin());
      BaseState::qregs_[i].initialize_from_vector(tmp);
    }
  }
}

//=========================================================================
// Implementation: Multiplexer Circuit
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_multiplexer(const int_t iChunk, const reg_t &control_qubits,
                                          const reg_t &target_qubits,
                                          const std::vector<cmatrix_t> &mmat) {
  // (1) Pack vector of matrices into single (stacked) matrix ... note: matrix
  // dims: rows = DIM[qubit.size()] columns = DIM[|target bits|]
  cmatrix_t multiplexer_matrix = Utils::stacked_matrix(mmat);

  // (2) Treat as single, large(r), chained/batched matrix operator
  apply_multiplexer(iChunk, control_qubits, target_qubits, multiplexer_matrix);
}

//=========================================================================
// Implementation: Kraus Noise
//=========================================================================
template <class statevec_t>
void State<statevec_t>::apply_kraus(const int_t iChunk, const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats,
                                    RngEngine &rng) 
{
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
  double p;
  bool complete = false;

  // Loop through N-1 kraus operators
  for (size_t j = 0; j < kmats.size() - 1; j++) {

    // Calculate probability
    cvector_t vmat = Utils::vectorize_matrix(kmats[j]);

    if(!BaseState::multi_chunk_distribution_){
      p = BaseState::qregs_[iChunk].norm(qubits, vmat);
      accum += p;
    }
    else{
      p = 0.0;
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for reduction(+:p)
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
            p += BaseState::qregs_[i].norm(qubits, vmat);
        }
      }
      else{
        for(int_t i=0;i<BaseState::qregs_.size();i++)
          p += BaseState::qregs_[i].norm(qubits, vmat);
      }

#ifdef AER_MPI
      BaseState::reduce_sum(p);
#endif
      accum += p;
    }

    // check if we need to apply this operator
    if (accum > r) {
      // rescale vmat so projection is normalized
      Utils::scalar_multiply_inplace(vmat, 1 / std::sqrt(p));
      // apply Kraus projection operator
      if(!BaseState::multi_chunk_distribution_)
        apply_matrix(iChunk, qubits, vmat);
      else{
        if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
          for(int_t ig=0;ig<BaseState::num_groups_;ig++){
            for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
              apply_matrix(ic, qubits, vmat);
          }
        }
        else{
          for(int_t ig=0;ig<BaseState::num_groups_;ig++){
            for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
              apply_matrix(ic, qubits, vmat);
          }
        }
      }
      complete = true;
      break;
    }
  }

  // check if we haven't applied a kraus operator yet
  if (complete == false) {
    // Compute probability from accumulated
    complex_t renorm = 1 / std::sqrt(1. - accum);
    auto vmat = Utils::vectorize_matrix(renorm * kmats.back());
    if(!BaseState::multi_chunk_distribution_)
      apply_matrix(iChunk, qubits, vmat);
    else{
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_matrix(ic, qubits, vmat);
        }
      }
      else{
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_matrix(ic, qubits, vmat);
        }
      }
    }
  }
}


//-------------------------------------------------------------------------
} // namespace Statevector
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
