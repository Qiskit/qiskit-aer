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
     OpType::jump, OpType::mark,
     OpType::sample_noise
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
  void apply_op(QuantumState::RegistersBase& state, 
                const Operations::Op &op,
                ExperimentResult &result,
                RngEngine& rng,
                bool final_op = false) override;

  //apply_op for specific chunk
  void apply_op_chunk(uint_t iChunk, QuantumState::RegistersBase& state, 
                        const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine& rng,
                        bool final_op = false) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is independent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     QuantumState::OpItr first, QuantumState::OpItr last) const override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure_state(QuantumState::RegistersBase& state, const reg_t &qubits, uint_t shots,
                                            RngEngine &rng) override;

  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------
  // Initializes to a specific n-qubit state given as a complex std::vector
  void initialize_qreg_from_data(uint_t num_qubits, const cvector_t &state);

  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp(QuantumState::Registers<statevec_t>& state);

  auto move_to_vector(QuantumState::Registers<statevec_t>& state);
  auto copy_to_vector(QuantumState::Registers<statevec_t>& state);

  //Does this state support runtime noise sampling?
  bool runtime_noise_sampling_supported(void) override {return true;}

protected:
  // Initialize classical memory and register to default value (all-0)
  void initialize_creg_state(QuantumState::RegistersBase& state, uint_t num_memory, uint_t num_register) override;

  // Initialize classical memory and register to specific values
  void initialize_creg_state(QuantumState::RegistersBase& state, 
                       uint_t num_memory,
                       uint_t num_register,
                       const std::string &memory_hex,
                       const std::string &register_hex) override;
  void initialize_creg_state(QuantumState::RegistersBase& state, const ClassicalRegister& creg) override;

  // Initializes an n-qubit state to the all |0> state
  void initialize_state(QuantumState::RegistersBase& state, uint_t num_qubits) override;

  // Initializes to a specific n-qubit state
  void initialize_state(QuantumState::RegistersBase& state, uint_t num_qubits,
                               const statevec_t &vector) override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  void set_state_config(QuantumState::RegistersBase& state_in, const json_t &config) override;

  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------
  //apply op to multiple shots , return flase if op is not supported to execute in a batch
  bool apply_batched_op(const int_t iChunk, QuantumState::RegistersBase& state, const Operations::Op &op,
                                ExperimentResult &result,
                                std::vector<RngEngine> &rng,
                                bool final_op = false) override;


  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(statevec_t& qreg, const Operations::Op &op);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual void apply_measure(QuantumState::Registers<statevec_t>& state, const reg_t &qubits, const reg_t &cmemory,
                             const reg_t &cregister, RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(QuantumState::Registers<statevec_t>& state, const reg_t &qubits, RngEngine &rng);

  // Initialize the specified qubits to a given state |psi>
  // by applying a reset to the these qubits and then
  // computing the tensor product with the new state |psi>
  // /psi> is given in params
  void apply_initialize(QuantumState::Registers<statevec_t>& state, const reg_t &qubits, const cvector_t &params,
                        RngEngine &rng);

  void initialize_from_vector(QuantumState::Registers<statevec_t>& state, const cvector_t &params);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(QuantumState::Registers<statevec_t>& state, const Operations::Op &op, ExperimentResult &result, bool last_op = false);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(statevec_t& qreg, const Operations::Op &op);

  // Apply a vectorized matrix to given qubits (identity on all other qubits)
  void apply_matrix(statevec_t& qreg, const reg_t &qubits, const cvector_t &vmat);

  //apply diagonal matrix
  void apply_diagonal_matrix(statevec_t& qreg, const reg_t &qubits, const cvector_t & diag); 

  // Apply a vector of control matrices to given qubits (identity on all other
  // qubits)
  void apply_multiplexer(statevec_t& qreg, const reg_t &control_qubits,
                         const reg_t &target_qubits,
                         const std::vector<cmatrix_t> &mmat);

  // Apply stacked (flat) version of multiplexer matrix to target qubits (using
  // control qubits to select matrix instance)
  void apply_multiplexer(statevec_t& qreg, const reg_t &control_qubits,
                         const reg_t &target_qubits, const cmatrix_t &mat);

  // Apply a Kraus error operation
  void apply_kraus(QuantumState::Registers<statevec_t>& state, const reg_t &qubits, const std::vector<cmatrix_t> &krausops,
                   RngEngine &rng);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the current state of the statevector simulator
  // If `last_op` is True this will use move semantics to move the simulator
  // state to the results, otherwise it will use copy semantics to leave
  // the current simulator state unchanged.
  void apply_save_statevector(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                              ExperimentResult &result,
                              bool last_op);

  // Save the current state of the statevector simulator as a ket-form map.
  void apply_save_statevector_dict(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                  ExperimentResult &result);

  // Save the current density matrix or reduced density matrix
  void apply_save_density_matrix(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                 ExperimentResult &result);

  // Helper function for computing expectation value
  void apply_save_probs(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                        ExperimentResult &result);

  // Helper function for saving amplitudes and amplitudes squared
  void apply_save_amplitudes(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                             ExperimentResult &result);

  // Apply a save expectation value instruction
  void apply_save_expval(QuantumState::Registers<statevec_t>& state, const Operations::Op &op, ExperimentResult &result);

  // Helper function for computing expectation value
  double expval_pauli(QuantumState::RegistersBase& state, const reg_t &qubits,
                              const std::string& pauli) override;
  //-----------------------------------------------------------------------
  // Measurement Helpers
  //-----------------------------------------------------------------------

  // Return vector of measure probabilities for specified qubits
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  // TODO: move to private (no longer part of base class)
  rvector_t measure_probs(QuantumState::Registers<statevec_t>& state, const reg_t &qubits) const;

  // Sample the measurement outcome for qubits
  // return a pair (m, p) of the outcome m, and its corresponding
  // probability p.
  // Outcome is given as an int: Eg for two-qubits {q0, q1} we have
  // 0 -> |q1 = 0, q0 = 0> state
  // 1 -> |q1 = 0, q0 = 1> state
  // 2 -> |q1 = 1, q0 = 0> state
  // 3 -> |q1 = 1, q0 = 1> state
  std::pair<uint_t, double> sample_measure_with_prob(QuantumState::Registers<statevec_t>& state, const reg_t &qubits,
                                                     RngEngine &rng);
  rvector_t sample_measure_with_prob_shot_branching(QuantumState::Registers<statevec_t>& state, const reg_t &qubits);

  void measure_reset_update(QuantumState::Registers<statevec_t>& state, const std::vector<uint_t> &qubits,
                            const uint_t final_state, const uint_t meas_state,
                            const double meas_prob);
  void measure_reset_update_shot_branching(
                             QuantumState::Registers<statevec_t>& state, const std::vector<uint_t> &qubits,
                             const int_t final_state,
                             const rvector_t& meas_probs);

  //-----------------------------------------------------------------------
  // Special snapshot types
  // Apply a supported snapshot instruction
  //
  // IMPORTANT: These methods are not marked const to allow modifying state
  // during snapshot, but after the snapshot is applied the simulator
  // should be left in the pre-snapshot state.
  //-----------------------------------------------------------------------

  // Snapshot current qubit probabilities for a measurement (average)
  void snapshot_probabilities(QuantumState::Registers<statevec_t>& state, const Operations::Op &op, ExperimentResult &result,
                              SnapshotDataType type);

  // Snapshot the expectation value of a Pauli operator
  void snapshot_pauli_expval(QuantumState::Registers<statevec_t>& state, const Operations::Op &op, ExperimentResult &result,
                             SnapshotDataType type);

  // Snapshot the expectation value of a matrix operator
  void snapshot_matrix_expval(QuantumState::Registers<statevec_t>& state, const Operations::Op &op, ExperimentResult &result,
                              SnapshotDataType type);

  // Snapshot reduced density matrix
  void snapshot_density_matrix(QuantumState::Registers<statevec_t>& state, const Operations::Op &op, ExperimentResult &result,
                               SnapshotDataType type);

  // Return the reduced density matrix for the simulator
  cmatrix_t density_matrix(QuantumState::Registers<statevec_t>& state, const reg_t &qubits);

  // Helper function to convert a vector to a reduced density matrix
  template <class T> cmatrix_t vec2density(const reg_t &qubits, const T &vec);

  //-----------------------------------------------------------------------
  // Single-qubit gate helpers
  //-----------------------------------------------------------------------

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(statevec_t& qreg, const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Multi-controlled u3
  //-----------------------------------------------------------------------

  // Apply N-qubit multi-controlled single qubit gate specified by
  // 4 parameters u4(theta, phi, lambda, gamma)
  // NOTE: if N=1 this is just a regular u4 gate.
  void apply_gate_mcu(statevec_t& qreg, const reg_t &qubits, const double theta,
                      const double phi, const double lambda,
                      const double gamma);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Apply the global phase
  void apply_global_phase(QuantumState::RegistersBase& state) override;

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

  bool shot_branching_supported(void) override
  {
    if(BaseState::multi_chunk_distribution_)
      return false;   //disable shot branching if multi-chunk distribution is used
    return true;
  }

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
void State<statevec_t>::initialize_state(QuantumState::RegistersBase& state_in, uint_t num_qubits) 
{
  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  int_t i;
  if(state.qregs().size() == 0)
    BaseState::allocate(num_qubits,BaseState::chunk_bits_,1);

  initialize_omp(state);

  for(i=0;i<state.qregs().size();i++){
    state.qregs()[i].set_num_qubits(BaseState::chunk_bits_);
  }

  if(BaseState::multi_chunk_distribution_){
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++){
          if(BaseState::global_chunk_index_ + iChunk == 0 || this->num_qubits_ == this->chunk_bits_){
            state.qregs()[iChunk].initialize();
          }
          else{
            state.qregs()[iChunk].zero();
          }
        }
      }
    }
    else{
      for(i=0;i<state.qregs().size();i++){
        if(BaseState::global_chunk_index_ + i == 0 || this->num_qubits_ == this->chunk_bits_){
          state.qregs()[i].initialize();
        }
        else{
          state.qregs()[i].zero();
        }
      }
    }
  }
  else{
    for(i=0;i<state.qregs().size();i++){
      state.qregs()[i].initialize();
    }
  }
  apply_global_phase(state);
}

template <class statevec_t>
void State<statevec_t>::initialize_state(QuantumState::RegistersBase& state_in, uint_t num_qubits,
                                        const statevec_t &vector) 
{
  if (vector.num_qubits() != num_qubits) {
    throw std::invalid_argument("QubitVector::State::initialize: initial state does not match qubit number");
  }

  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  if(state.qregs().size() == 0)
    BaseState::allocate(num_qubits,BaseState::chunk_bits_,1);
  initialize_omp(state);

  int_t iChunk;
  for(iChunk=0;iChunk<state.qregs().size();iChunk++){
    state.qregs()[iChunk].set_num_qubits(BaseState::chunk_bits_);
  }

  if(BaseState::multi_chunk_distribution_){
    uint_t local_offset = BaseState::global_chunk_index_ << BaseState::chunk_bits_;
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for private(iChunk)
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++)
          state.qregs()[iChunk].initialize_from_data(vector.data() + local_offset + (iChunk << BaseState::chunk_bits_), 1ull << BaseState::chunk_bits_);
      }
    }
    else{
      for(iChunk=0;iChunk<state.qregs().size();iChunk++)
        state.qregs()[iChunk].initialize_from_data(vector.data() + local_offset + (iChunk << BaseState::chunk_bits_), 1ull << BaseState::chunk_bits_);
    }
  }
  else{
    for(iChunk=0;iChunk<state.qregs().size();iChunk++){
      state.qregs()[iChunk].initialize_from_data(vector.data(), 1ull << BaseState::chunk_bits_);
    }
  }
  apply_global_phase(state);
}

template <class statevec_t>
void State<statevec_t>::initialize_qreg_from_data(uint_t num_qubits,
                                        const cvector_t &vector) 
{
  if (vector.size() != 1ULL << num_qubits) {
    throw std::invalid_argument("QubitVector::State::initialize: initial state does not match qubit number");
  }

  QuantumState::Registers<statevec_t>& state = BaseState::state_;

  if(BaseState::state_.qregs().size() == 0)
    BaseState::allocate(num_qubits,BaseState::chunk_bits_,1);

  initialize_omp(BaseState::state_);

  int_t iChunk;
  for(iChunk=0;iChunk<BaseState::state_.qregs().size();iChunk++){
    BaseState::state_.qregs()[iChunk].set_num_qubits(BaseState::chunk_bits_);
  }

  initialize_from_vector(BaseState::state_, vector);
  apply_global_phase(BaseState::state_);
}


template <class statevec_t>
void State<statevec_t>::initialize_qreg_from_data(uint_t num_qubits,
                                        const cvector_t &vector) 
{
  if (vector.size() != 1ULL << num_qubits) {
    throw std::invalid_argument("QubitVector::State::initialize: initial state does not match qubit number");
  }

  QuantumState::Registers<statevec_t>& state = BaseState::state_;

  if(BaseState::state_.qregs().size() == 0)
    BaseState::allocate(num_qubits,BaseState::chunk_bits_,1);

  initialize_omp(BaseState::state_);

  int_t iChunk;
  for(iChunk=0;iChunk<BaseState::state_.qregs().size();iChunk++){
    BaseState::state_.qregs()[iChunk].set_num_qubits(BaseState::chunk_bits_);
  }

  initialize_from_vector(BaseState::state_, vector);
  apply_global_phase(BaseState::state_);
}

template <class statevec_t> void State<statevec_t>::initialize_omp(QuantumState::Registers<statevec_t>& state) 
{
  uint_t i;

  for(i=0;i<state.qregs().size();i++){
    state.qregs()[i].set_omp_threshold(omp_qubit_threshold_);
    if (BaseState::threads_ > 0)
      state.qregs()[i].set_omp_threads(BaseState::threads_); // set allowed OMP threads in qubitvector
  }
}

template <class statevec_t>
void State<statevec_t>::initialize_creg_state(QuantumState::RegistersBase& state_in, uint_t num_memory, uint_t num_register) 
{
  BaseState::initialize_creg_state(state_in, num_memory, num_register);

  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  for(int_t i=0;i<state.qregs().size();i++)
    state.qreg(i).initialize_creg(num_memory, num_register);
}


template <class statevec_t>
void State<statevec_t>::initialize_creg_state(QuantumState::RegistersBase& state_in, 
                                     uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) 
{
  BaseState::initialize_creg_state(state_in, num_memory, num_register, memory_hex, register_hex);

  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  for(int_t i=0;i<state.qregs().size();i++)
    state.qreg(i).initialize_creg(num_memory, num_register, memory_hex, register_hex);
}

template <class statevec_t>
void State<statevec_t>::initialize_creg_state(QuantumState::RegistersBase& state_in, const ClassicalRegister& creg)
{
  BaseState::initialize_creg_state(state_in, creg);

  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  for(int_t i=0;i<state.qregs().size();i++)
    state.qreg(i).initialize_creg(creg.memory_size(), creg.register_size(), creg.memory_hex(), creg.register_hex());
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

template <class statevec_t>
void State<statevec_t>::apply_global_phase(QuantumState::RegistersBase& state_in) 
{
  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  if (BaseState::has_global_phase_) {
    int_t i;
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++)
          state.qreg(iChunk).apply_diagonal_matrix({0}, {BaseState::global_phase_, BaseState::global_phase_});
      }
    }
    else{
      for(i=0;i<state.qregs().size();i++)
        state.qreg(i).apply_diagonal_matrix({0}, {BaseState::global_phase_, BaseState::global_phase_});
    }
  }
}

template <class statevec_t>
size_t State<statevec_t>::required_memory_mb(uint_t num_qubits,
                                             QuantumState::OpItr first, QuantumState::OpItr last)
                                             const 
{
  (void)first; // avoid unused variable compiler warning
  (void)last;
  statevec_t tmp;
  return tmp.required_memory_mb(num_qubits);
}

template <class statevec_t>
void State<statevec_t>::set_state_config(QuantumState::RegistersBase& state_in, const json_t &config) 
{
  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);
  double thresh;

  // Set OMP threshold for state update functions
  if(omp_get_num_threads() > 1){
#pragma omp critical
    {
      // Set threshold for truncating snapshots
      JSON::get_value(json_chop_threshold_, "zero_threshold", config);
      JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold", config);
      thresh = json_chop_threshold_;
    }
  }
  else{
    // Set threshold for truncating snapshots
    JSON::get_value(json_chop_threshold_, "zero_threshold", config);
    JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold", config);
    thresh = json_chop_threshold_;
  }

  // Set the sample measure indexing size
  int index_size;
  JSON::get_value(index_size, "statevector_sample_measure_opt", config);
  for(int_t i=0;i<state.qregs().size();i++){
    state.qregs()[i].set_json_chop_threshold(thresh);
    state.qregs()[i].set_sample_measure_index_size(index_size);
  }
}


template <class statevec_t>
auto State<statevec_t>::move_to_vector(QuantumState::Registers<statevec_t>& state)
{
  if(BaseState::multi_chunk_distribution_){
    size_t size_required = 2*(sizeof(std::complex<double>) << BaseState::num_qubits_) + (sizeof(std::complex<double>) << BaseState::chunk_bits_)*BaseState::num_local_chunks_;
    if((size_required >> 20) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to store states"));
    }
    int_t iChunk;
    auto out = state.qreg(0).move_to_vector();
    out.resize(BaseState::num_local_chunks_ << BaseState::chunk_bits_);

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk)
    for(iChunk=1;iChunk<state.qregs().size();iChunk++){
      auto tmp = state.qreg(iChunk).move_to_vector();
      uint_t j,offset = iChunk << BaseState::chunk_bits_;
      for(j=0;j<tmp.size();j++){
        out[offset + j] = tmp[j];
      }
    }

#ifdef AER_MPI
    BaseState::gather_state(out);
#endif
    return out;
  }
  else
    return state.qreg().move_to_vector();
}

template <class statevec_t>
auto State<statevec_t>::copy_to_vector(QuantumState::Registers<statevec_t>& state)
{
  if(BaseState::multi_chunk_distribution_){
    size_t size_required = 2*(sizeof(std::complex<double>) << BaseState::num_qubits_) + (sizeof(std::complex<double>) << BaseState::chunk_bits_)*BaseState::num_local_chunks_;
    if((size_required >> 20) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to store states"));
    }
    int_t iChunk;
    auto out = state.qreg(0).copy_to_vector();
    out.resize(BaseState::num_local_chunks_ << BaseState::chunk_bits_);

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk)
    for(iChunk=1;iChunk<state.qregs().size();iChunk++){
      auto tmp = state.qreg(iChunk).copy_to_vector();
      uint_t j,offset = iChunk << BaseState::chunk_bits_;
      for(j=0;j<tmp.size();j++){
        out[offset + j] = tmp[j];
      }
    }

#ifdef AER_MPI
    BaseState::gather_state(out);
#endif
    return out;
  }
  else
  return state.qreg().copy_to_vector();
}


//=========================================================================
// Implementation: apply operations
//=========================================================================
template <class statevec_t>
void State<statevec_t>::apply_op(QuantumState::RegistersBase& state_in,
                        const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine& rng,
                        bool final_op)
{
  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  if(BaseState::enable_batch_execution_ && state.qreg().batched_optimization_supported()){
    if(op.conditional){
      state.qreg().set_conditional(op.conditional_reg);
    }
    if(state.additional_ops().size() == 0)
      state.qreg().enable_batch(true);
    else
      state.qreg().enable_batch(false);
  }
  else if(!state.creg().check_conditional(op))
    return;

  switch (op.type) {
    case OpType::barrier:
    case OpType::nop:
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
      for(int_t i=0;i<state.qregs().size();i++)
        apply_gate(state.qreg(i), op);
      break;
    case OpType::snapshot:
      apply_snapshot(state, op, result, final_op);
      break;
    case OpType::matrix:
      for(int_t i=0;i<state.qregs().size();i++)
        apply_matrix(state.qreg(i), op);
      break;
    case OpType::diagonal_matrix:
      for(int_t i=0;i<state.qregs().size();i++)
        apply_diagonal_matrix(state.qreg(i), op.qubits, op.params);
      break;
    case OpType::multiplexer:
      for(int_t i=0;i<state.qregs().size();i++){
        apply_multiplexer(state.qreg(i), op.regs[0], op.regs[1],
                        op.mats); // control qubits ([0]) & target qubits([1])
      }
      break;
    case OpType::kraus:
      apply_kraus(state, op.qubits, op.mats, rng);
      break;
    case OpType::sim_op:
      if(op.name == "begin_register_blocking"){
        state.qreg().enter_register_blocking(op.qubits);
      }
      else if(op.name == "end_register_blocking"){
        state.qreg().leave_register_blocking();
      }
      break;
    case OpType::set_statevec:
      initialize_from_vector(state, op.params);
      break;
    case OpType::save_expval:
    case OpType::save_expval_var:
      apply_save_expval(state, op, result);
      break;
    case OpType::save_densmat:
      apply_save_density_matrix(state, op, result);
      break;
    case OpType::save_state:
    case OpType::save_statevec:
      apply_save_statevector(state, op, result, final_op);
      break;
    case OpType::save_statevec_dict:
      apply_save_statevector_dict(state, op, result);
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
      throw std::invalid_argument(
          "QubitVector::State::invalid instruction \'" + op.name + "\'.");
  }
}

template <class statevec_t>
void State<statevec_t>::apply_op_chunk(uint_t iChunk, QuantumState::RegistersBase& state_in, 
                        const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine& rng,
                        bool final_op)
{
  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  if(state.creg().check_conditional(op)) {
    switch (op.type) {
      case OpType::barrier:
      case OpType::nop:
      case OpType::qerror_loc:
        break;
      case OpType::bfunc:
        state.creg().apply_bfunc(op);
        break;
      case OpType::roerror:
        state.creg().apply_roerror(op, rng);
        break;
      case OpType::gate:
        apply_gate(state.qreg(iChunk), op);
        break;
      case OpType::matrix:
        apply_matrix(state.qreg(iChunk), op);
        break;
      case OpType::diagonal_matrix:
        apply_diagonal_matrix(state.qreg(iChunk), op.qubits, op.params);
        break;
      case OpType::multiplexer:
        apply_multiplexer(state.qreg(iChunk), op.regs[0], op.regs[1],
                          op.mats); // control qubits ([0]) & target qubits([1])
        break;
      case OpType::sim_op:
        if(op.name == "begin_register_blocking"){
          state.qreg(iChunk).enter_register_blocking(op.qubits);
        }
        else if(op.name == "end_register_blocking"){
          state.qreg(iChunk).leave_register_blocking();
        }
        break;
      default:
        throw std::invalid_argument(
            "QubitVector::State::invalid instruction \'" + op.name + "\'.");
    }
  }
}

template <class statevec_t>
bool State<statevec_t>::apply_batched_op(const int_t iChunk, QuantumState::RegistersBase& state_in, 
                                  const Operations::Op &op,
                                  ExperimentResult &result,
                                  std::vector<RngEngine> &rng,
                                  bool final_op) 
{
  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  if(op.conditional){
    state.qreg(iChunk).set_conditional(op.conditional_reg);
  }

  switch (op.type) {
    case OpType::barrier:
    case OpType::nop:
    case OpType::qerror_loc:
      break;
    case OpType::reset:
      state.qreg(iChunk).apply_batched_reset(op.qubits,rng);
      break;
    case OpType::initialize:
      state.qreg(iChunk).apply_batched_reset(op.qubits,rng);
      state.qreg(iChunk).initialize_component(op.qubits, op.params);
      break;
    case OpType::measure:
      state.qreg(iChunk).apply_batched_measure(op.qubits,rng,op.memory,op.registers);
      break;
    case OpType::bfunc:
      state.qreg(iChunk).apply_bfunc(op);
      break;
    case OpType::roerror:
      state.qreg(iChunk).apply_roerror(op, rng);
      break;
    case OpType::gate:
      apply_gate(state.qreg(iChunk), op);
      break;
    case OpType::matrix:
      apply_matrix(state.qreg(iChunk), op);
      break;
    case OpType::diagonal_matrix:
      state.qreg(iChunk).apply_diagonal_matrix(op.qubits, op.params);
      break;
    case OpType::multiplexer:
      apply_multiplexer(state.qreg(iChunk), op.regs[0], op.regs[1],
                        op.mats); // control qubits ([0]) & target qubits([1])
      break;
    case OpType::kraus:
      state.qreg(iChunk).apply_batched_kraus(op.qubits, op.mats,rng);
      break;
    case OpType::sim_op:
      if(op.name == "begin_register_blocking"){
        state.qreg(iChunk).enter_register_blocking(op.qubits);
      }
      else if(op.name == "end_register_blocking"){
        state.qreg(iChunk).leave_register_blocking();
      }
      else{
        return false;
      }
      break;
    case OpType::set_statevec:
      state.qreg(iChunk).initialize_from_vector(op.params);
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
void State<statevec_t>::apply_save_probs(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                         ExperimentResult &result) 
{
  // get probs as hexadecimal
  auto probs = measure_probs(state, op.qubits);
  if (op.type == Operations::OpType::save_probs_ket) {
    // Convert to ket dict
    result.save_data_average(state.creg(), op.string_params[0],
                             Utils::vec2ket(probs, json_chop_threshold_, 16),
                             op.type, op.save_type);
  } else {
    result.save_data_average(state.creg(), op.string_params[0],
                             std::move(probs), op.type, op.save_type);
  }
}


template <class statevec_t>
double State<statevec_t>::expval_pauli(QuantumState::RegistersBase& state_in, const reg_t &qubits,
                                       const std::string& pauli) 
{
  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  if(!BaseState::multi_chunk_distribution_)
    return state.qreg().expval_pauli(qubits, pauli);

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
        auto apply_expval_pauli_chunk = [this, x_mask, z_mask, x_max,mask_u,mask_l, qubits_in_chunk, pauli_in_chunk, phase, state](int_t iGroup)
        {
          double expval = 0.0;
          for(int_t iChunk = BaseState::top_chunk_of_group_[iGroup];iChunk < BaseState::top_chunk_of_group_[iGroup + 1];iChunk++){
            uint_t pair_chunk = iChunk ^ x_mask;
            if(iChunk < pair_chunk){
              uint_t z_count,z_count_pair;
              z_count = AER::Utils::popcount(iChunk & z_mask);
              z_count_pair = AER::Utils::popcount(pair_chunk & z_mask);

              expval += state.qreg(iChunk-BaseState::global_chunk_index_).expval_pauli(qubits_in_chunk, pauli_in_chunk,state.qreg(pair_chunk),z_count,z_count_pair,phase);
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
              expval += state.qreg(iChunk-BaseState::global_chunk_index_).expval_pauli(qubits_in_chunk, pauli_in_chunk,state.qreg(pair_chunk - BaseState::global_chunk_index_),z_count,z_count_pair,phase);
            }
            else{
              BaseState::recv_chunk(state, iChunk-BaseState::global_chunk_index_,pair_chunk);
              //refer receive buffer to calculate expectation value
              expval += state.qreg(iChunk-BaseState::global_chunk_index_).expval_pauli(qubits_in_chunk, pauli_in_chunk,state.qreg(iChunk-BaseState::global_chunk_index_),z_count,z_count_pair,phase);
            }
          }
          else if(iProc == BaseState::distributed_rank_){  //pair is on this process
            BaseState::send_chunk(state, iChunk-BaseState::global_chunk_index_,pair_chunk);
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
            e_tmp += sign * state.qreg(iChunk).expval_pauli(qubits_in_chunk, pauli_in_chunk);
          }
          expval += e_tmp;
        }
      }
      else{
        for(i=0;i<state.qregs().size();i++){
          double sign = 1.0;
          if (z_mask && (AER::Utils::popcount((i + BaseState::global_chunk_index_) & z_mask) & 1))
            sign = -1.0;
          expval += sign * state.qreg(i).expval_pauli(qubits_in_chunk, pauli_in_chunk);
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
          e_tmp += state.qreg(iChunk).expval_pauli(qubits, pauli);
        expval += e_tmp;
      }
    }
    else{
      for(i=0;i<state.qregs().size();i++)
        expval += state.qreg(i).expval_pauli(qubits, pauli);
    }
  }

#ifdef AER_MPI
  BaseState::reduce_sum(expval);
#endif
  return expval;
}

template <class statevec_t>
void State<statevec_t>::apply_save_statevector(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
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

  if(BaseState::multi_chunk_distribution_ || state.num_shots() > 1){
    if (last_op) {
      result.save_data_pershot(state.creg(), key, move_to_vector(state),
                               OpType::save_statevec, op.save_type, state.num_shots());
    } else {
      result.save_data_pershot(state.creg(), key, copy_to_vector(state),
                                  OpType::save_statevec, op.save_type, state.num_shots());
    }
  }
  else{
    //for batched multi-shot, save each qreg
    for(int_t i=0;i<state.num_qregs();i++){
      if (last_op) {
        result.save_data_pershot(state.creg(), key, state.qreg(i).move_to_vector(),
                                 OpType::save_statevec, op.save_type, 1);
      } else {
        result.save_data_pershot(state.creg(), key, state.qreg(i).copy_to_vector(),
                                    OpType::save_statevec, op.save_type, 1);
      }
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_save_statevector_dict(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                                   ExperimentResult &result) 
{
  if (op.qubits.size() != BaseState::num_qubits_) {
    throw std::invalid_argument(
        op.name + " was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  if(BaseState::multi_chunk_distribution_){
    auto vec = copy_to_vector(state);
    std::map<std::string, complex_t> result_state_ket;
    for (size_t k = 0; k < vec.size(); ++k) {
      if (std::abs(vec[k]) >= json_chop_threshold_){
        std::string key = Utils::int2hex(k);
        result_state_ket.insert({key, vec[k]});
      }
    }
    result.save_data_pershot(state.creg(), op.string_params[0],
                                 std::move(result_state_ket), op.type, op.save_type, state.num_shots());
  }
  else{
    for(int_t i=0;i<state.num_qregs();i++){
      auto state_ket = state.qreg().vector_ket(json_chop_threshold_);
      std::map<std::string, complex_t> result_state_ket;
      for (auto const& it : state_ket){
        result_state_ket[it.first] = it.second;
      }
      result.save_data_pershot(state.creg(), op.string_params[0],
                                   std::move(result_state_ket), op.type, op.save_type, state.num_shots());
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_save_density_matrix(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                                  ExperimentResult &result) 
{
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    if(BaseState::multi_chunk_distribution_){
      double sum = 0.0;
      if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for reduction(+:sum)
        for(int_t i=0;i<state.qregs().size();i++)
          sum += state.qreg(i).norm();
      }
      else{
        for(int_t i=0;i<state.qregs().size();i++)
          sum += state.qreg(i).norm();
      }
#ifdef AER_MPI
      BaseState::reduce_sum(sum);
#endif
      reduced_state[0] = sum;

      result.save_data_average(state.creg(), op.string_params[0],
                               std::move(reduced_state), op.type, op.save_type);
    }
    else{
      for(int_t i=0;i<state.num_qregs();i++){
        reduced_state[0] = state.qreg().norm();
        result.save_data_average(state.creg(), op.string_params[0],
                                 reduced_state, op.type, op.save_type);
      }
    }
  }
  else {
    if(BaseState::multi_chunk_distribution_){
      reduced_state = density_matrix(state, op.qubits);
      result.save_data_average(state.creg(), op.string_params[0],
                               std::move(reduced_state), op.type, op.save_type);
    }
    else{
      for(int_t i=0;i<state.num_qregs();i++){
        reduced_state = vec2density(op.qubits, state.qreg(i).copy_to_vector());
        result.save_data_average(state.creg(), op.string_params[0],
                                 std::move(reduced_state), op.type, op.save_type);
      }
    }
  }

}

template <class statevec_t>
void State<statevec_t>::apply_save_amplitudes(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                              ExperimentResult &result) 
{
  if (op.int_params.empty()) {
    throw std::invalid_argument("Invalid save_amplitudes instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  if (op.type == Operations::OpType::save_amps) {
    if(BaseState::multi_chunk_distribution_){
      Vector<complex_t> amps(size, false);
      for (int_t i = 0; i < size; ++i) {
        uint_t idx = state.get_mapped_index(op.int_params[i]);
        uint_t iChunk = idx >> BaseState::chunk_bits_;
        amps[i] = 0.0;
        if(iChunk >= BaseState::global_chunk_index_ && iChunk < BaseState::global_chunk_index_ + state.qregs().size()){
          amps[i] = state.qreg(iChunk - BaseState::global_chunk_index_).get_state(idx - (iChunk << BaseState::chunk_bits_));
        }
#ifdef AER_MPI
        complex_t amp = amps[i];
        BaseState::reduce_sum(amp);
        amps[i] = amp;
#endif
      }
      result.save_data_pershot(state.creg(), op.string_params[0],
                                   std::move(amps), op.type, op.save_type, state.num_shots());
    }
    else{
      for(int_t j=0;j<state.num_qregs();j++){
        Vector<complex_t> amps(size, false);
        for (int_t i = 0; i < size; ++i) {
          amps[i] = state.qreg(j).get_state(op.int_params[i]);
        }
        result.save_data_pershot(state.creg(), op.string_params[0],
                                     std::move(amps), op.type, op.save_type, state.num_shots());
      }
    }
  }
  else{
    if(BaseState::multi_chunk_distribution_){
      rvector_t amps_sq(size,0);
      for (int_t i = 0; i < size; ++i) {
        uint_t idx = state.get_mapped_index(op.int_params[i]);
        uint_t iChunk = idx >> BaseState::chunk_bits_;
        if(iChunk >= BaseState::global_chunk_index_ && iChunk < BaseState::global_chunk_index_ + state.qregs().size()){
          amps_sq[i] = state.qreg(iChunk - BaseState::global_chunk_index_).probability(idx - (iChunk << BaseState::chunk_bits_));
        }
      }
#ifdef AER_MPI
      BaseState::reduce_sum(amps_sq);
#endif
     result.save_data_average(state.creg(), op.string_params[0],
                              std::move(amps_sq), op.type, op.save_type);
    }
    else{
      for(int_t j=0;j<state.num_qregs();j++){
        rvector_t amps_sq(size,0);
        for (int_t i = 0; i < size; ++i) {
          amps_sq[i] = state.qreg(j).probability(op.int_params[i]);
        }
        result.save_data_average(state.creg(), op.string_params[0],
                                  std::move(amps_sq), op.type, op.save_type);
      }
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_save_expval(QuantumState::Registers<statevec_t>& state, 
                                       const Operations::Op &op,
                                       ExperimentResult &result)
{
  if(BaseState::multi_chunk_distribution_){
    BaseState::apply_save_expval(state,op,result);
    return;
  }

  // Check empty edge case
  if (op.expval_params.empty()) {
    throw std::invalid_argument(
        "Invalid save expval instruction (Pauli components are empty).");
  }
  bool variance = (op.type == OpType::save_expval_var);

  for(int_t i=0;i<state.num_qregs();i++){
    // Accumulate expval components
    double expval(0.);
    double sq_expval(0.);
    for (const auto &param : op.expval_params) {
      // param is tuple (pauli, coeff, sq_coeff)
      const auto val =   state.qreg(i).expval_pauli(op.qubits, std::get<0>(param));
      expval += std::get<1>(param) * val;
      if (variance) {
        sq_expval += std::get<2>(param) * val;
      }
    }
    if (variance) {
      std::vector<double> expval_var(2);
      expval_var[0] = expval;  // mean
      expval_var[1] = sq_expval - expval * expval;  // variance
      result.save_data_average(state.creg(), op.string_params[0], expval_var, op.type, op.save_type);
    } else {
      result.save_data_average(state.creg(), op.string_params[0], expval, op.type, op.save_type);
    }
  }
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_snapshot(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
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
                                         move_to_vector(state));
      } else {
        result.legacy_data.add_pershot_snapshot("statevector", op.string_params[0],
                                         copy_to_vector(state));
      }
      break;
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(state, op, result);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(state, op, result);
      break;
    case Snapshots::probs: {
      // get probs as hexadecimal
      snapshot_probabilities(state, op, result, SnapshotDataType::average);
    } break;
    case Snapshots::densmat: {
      snapshot_density_matrix(state, op, result, SnapshotDataType::average);
    } break;
    case Snapshots::expval_pauli: {
      snapshot_pauli_expval(state, op, result, SnapshotDataType::average);
    } break;
    case Snapshots::expval_matrix: {
      snapshot_matrix_expval(state, op, result, SnapshotDataType::average);
    } break;
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
    } break;
    case Snapshots::expval_pauli_shot: {
      snapshot_pauli_expval(state, op, result, SnapshotDataType::pershot);
    } break;
    case Snapshots::expval_matrix_shot: {
      snapshot_matrix_expval(state, op, result, SnapshotDataType::pershot);
    } break;
    default:
      // We shouldn't get here unless there is a bug in the snapshotset
      throw std::invalid_argument(
          "QubitVector::State::invalid snapshot instruction \'" + op.name +
          "\'.");
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_probabilities(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                               ExperimentResult &result,
                                               SnapshotDataType type) 
{
  // get probs as hexadecimal
  auto probs =
      Utils::vec2ket(measure_probs(state, op.qubits), json_chop_threshold_, 16);
  bool variance = type == SnapshotDataType::average_var;
  result.legacy_data.add_average_snapshot("probabilities", op.string_params[0],
                                   state.creg().memory_hex(),
                                   std::move(probs), variance);
}

template <class statevec_t>
void State<statevec_t>::snapshot_pauli_expval(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                              ExperimentResult &result,
                                              SnapshotDataType type) 
{
  // Check empty edge case
  if (op.params_expval_pauli.empty()) {
    throw std::invalid_argument(
        "Invalid expval snapshot (Pauli components are empty).");
  }

  // Accumulate expval components
  complex_t expval(0., 0.);
  for (const auto &param : op.params_expval_pauli) {
    const auto &coeff = param.first;
    const auto &pauli = param.second;
    expval += coeff * expval_pauli(state, op.qubits, pauli);
  }

  // Add to snapshot
  Utils::chop_inplace(expval, json_chop_threshold_);
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
    result.legacy_data.add_pershot_snapshot("expectation_values", op.string_params[0],
                              expval);
    break;
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_matrix_expval(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                               ExperimentResult &result,
                                               SnapshotDataType type) 
{
  // Check empty edge case
  if (op.params_expval_matrix.empty()) {
    throw std::invalid_argument(
        "Invalid matrix snapshot (components are empty).");
  }

  reg_t qubits = op.qubits;
  // Cache the current quantum state
  if(!BaseState::multi_chunk_distribution_)
    state.qreg().checkpoint();
  else{
    if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for 
      for(int_t i=0;i<state.qregs().size();i++)
        state.qreg(i).checkpoint();
    }
    else{
      for(int_t i=0;i<state.qregs().size();i++)
        state.qreg(i).checkpoint();
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
        state.qreg().revert(true);
      else{
        if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for 
          for(int_t i=0;i<state.qregs().size();i++)
            state.qreg(i).revert(true);
        }
        else{
          for(int_t i=0;i<state.qregs().size();i++)
            state.qreg(i).revert(true);
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
          apply_diagonal_matrix(state.qreg(), sub_qubits, vmat);
        else{
          if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for
            for(int_t i=0;i<state.qregs().size();i++)
              apply_diagonal_matrix(state.qreg(i), sub_qubits, vmat);
          }
          else{
            for(int_t i=0;i<state.qregs().size();i++)
              apply_diagonal_matrix(state.qreg(i), sub_qubits, vmat);
          }
        }
      } else {
        if(!BaseState::multi_chunk_distribution_)
          state.qreg().apply_matrix(sub_qubits, vmat);
        else{
          if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for 
            for(int_t i=0;i<state.qregs().size();i++)
              state.qreg(i).apply_matrix(sub_qubits, vmat);
          }
          else{
            for(int_t i=0;i<state.qregs().size();i++)
              state.qreg(i).apply_matrix(sub_qubits, vmat);
          }
        }
      }
    }
    double exp_re = 0.0;
    double exp_im = 0.0;
    if(!BaseState::multi_chunk_distribution_){
      auto exp_tmp = coeff*state.qreg().inner_product();
      exp_re += exp_tmp.real();
      exp_im += exp_tmp.imag();
    }
    else{
      if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for reduction(+:exp_re,exp_im)
        for(int_t i=0;i<state.qregs().size();i++){
          auto exp_tmp = coeff*state.qreg(i).inner_product();
          exp_re += exp_tmp.real();
          exp_im += exp_tmp.imag();
        }
      }
      else{
        for(int_t i=0;i<state.qregs().size();i++){
          auto exp_tmp = coeff*state.qreg(i).inner_product();
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
                              state.creg().memory_hex(), expval, false);
    break;
  case SnapshotDataType::average_var:
    result.legacy_data.add_average_snapshot("expectation_value", op.string_params[0],
                              state.creg().memory_hex(), expval, true);
    break;
  case SnapshotDataType::pershot:
    result.legacy_data.add_pershot_snapshot("expectation_values", op.string_params[0],
                              expval);
    break;
  }
  // Revert to original state
  if(!BaseState::multi_chunk_distribution_)
    state.qreg().revert(false);
  else{
    if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for 
      for(int_t i=0;i<state.qregs().size();i++)
        state.qreg(i).revert(false);
    }
    else{
      for(int_t i=0;i<state.qregs().size();i++)
        state.qreg(i).revert(false);
    }
  }
}

template <class statevec_t>
void State<statevec_t>::snapshot_density_matrix(QuantumState::Registers<statevec_t>& state, const Operations::Op &op,
                                                ExperimentResult &result,
                                                SnapshotDataType type) {
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    if(!BaseState::multi_chunk_distribution_)
      reduced_state[0] = state.qreg().norm();
    else{
      double sum = 0.0;
      if(BaseState::chunk_omp_parallel_){
#pragma omp parallel for reduction(+:sum)
        for(int_t i=0;i<state.qregs().size();i++)
          sum += state.qreg(i).norm();
      }
      else{
        for(int_t i=0;i<state.qregs().size();i++)
          sum += state.qreg(i).norm();
      }
#ifdef AER_MPI
      BaseState::reduce_sum(sum);
#endif
      reduced_state[0] = sum;
    }
  } else {
    reduced_state = density_matrix(state, op.qubits);
  }

  // Add density matrix to result data
  switch (type) {
  case SnapshotDataType::average:
    result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                              state.creg().memory_hex(),
                              std::move(reduced_state), false);
    break;
  case SnapshotDataType::average_var:
    result.legacy_data.add_average_snapshot("density_matrix", op.string_params[0],
                              state.creg().memory_hex(),
                              std::move(reduced_state), true);
    break;
  case SnapshotDataType::pershot:
    result.legacy_data.add_pershot_snapshot("density_matrix", op.string_params[0],
                              std::move(reduced_state));
    break;
  }
}

template <class statevec_t>
cmatrix_t State<statevec_t>::density_matrix(QuantumState::Registers<statevec_t>& state, const reg_t &qubits) 
{
  return vec2density(qubits, copy_to_vector(state));
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
void State<statevec_t>::apply_gate(statevec_t& qreg, const Operations::Op &op) 
{
  if(!BaseState::global_chunk_indexing_){
    reg_t qubits_in,qubits_out;
    BaseState::get_inout_ctrl_qubits(op,qubits_out,qubits_in);
    if(qubits_out.size() > 0){
      uint_t mask = 0;
      for(int i=0;i<qubits_out.size();i++){
        mask |= (1ull << (qubits_out[i] - BaseState::chunk_bits_));
      }
      if((qreg.chunk_index() & mask) == mask){
        Operations::Op new_op = BaseState::remake_gate_in_chunk_qubits(op,qubits_in);
        apply_gate(qreg, new_op);
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
      qreg.apply_mcx(op.qubits);
      break;
    case Gates::mcy:
      // Includes Y, CY, CCY, etc
      qreg.apply_mcy(op.qubits);
      break;
    case Gates::mcz:
      // Includes Z, CZ, CCZ, etc
      qreg.apply_mcphase(op.qubits, -1);
      break;
    case Gates::mcr:
      qreg.apply_mcu(op.qubits, Linalg::VMatrix::r(op.params[0], op.params[1]));
      break;
    case Gates::mcrx:
      qreg.apply_rotation(op.qubits, QV::Rotation::x, std::real(op.params[0]));
      break;
    case Gates::mcry:
      qreg.apply_rotation(op.qubits, QV::Rotation::y, std::real(op.params[0]));
      break;
    case Gates::mcrz:
      qreg.apply_rotation(op.qubits, QV::Rotation::z, std::real(op.params[0]));
      break;
    case Gates::rxx:
      qreg.apply_rotation(op.qubits, QV::Rotation::xx, std::real(op.params[0]));
      break;
    case Gates::ryy:
      qreg.apply_rotation(op.qubits, QV::Rotation::yy, std::real(op.params[0]));
      break;
    case Gates::rzz:
      qreg.apply_rotation(op.qubits, QV::Rotation::zz, std::real(op.params[0]));
      break;
    case Gates::rzx:
      qreg.apply_rotation(op.qubits, QV::Rotation::zx, std::real(op.params[0]));
      break;
    case Gates::id:
      break;
    case Gates::h:
      apply_gate_mcu(qreg, op.qubits, M_PI / 2., 0., M_PI, 0.);
      break;
    case Gates::s:
      apply_gate_phase(qreg, op.qubits[0], complex_t(0., 1.));
      break;
    case Gates::sdg:
      apply_gate_phase(qreg, op.qubits[0], complex_t(0., -1.));
      break;
    case Gates::t: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_gate_phase(qreg, op.qubits[0], complex_t(isqrt2, isqrt2));
    } break;
    case Gates::tdg: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_gate_phase(qreg, op.qubits[0], complex_t(isqrt2, -isqrt2));
    } break;
    case Gates::mcswap:
      // Includes SWAP, CSWAP, etc
      qreg.apply_mcswap(op.qubits);
      break;
    case Gates::mcu3:
      // Includes u3, cu3, etc
      apply_gate_mcu(qreg, op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                     std::real(op.params[2]), 0.);
      break;
    case Gates::mcu:
      // Includes u3, cu3, etc
      apply_gate_mcu(qreg, op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                      std::real(op.params[2]), std::real(op.params[3]));
      break;
    case Gates::mcu2:
      // Includes u2, cu2, etc
      apply_gate_mcu(qreg, op.qubits, M_PI / 2., std::real(op.params[0]),
                     std::real(op.params[1]), 0.);
      break;
    case Gates::mcp:
      // Includes u1, cu1, p, cp, mcp etc
      qreg.apply_mcphase(op.qubits,
                                     std::exp(complex_t(0, 1) * op.params[0]));
      break;
    case Gates::mcsx:
      // Includes sx, csx, mcsx etc
      qreg.apply_mcu(op.qubits, Linalg::VMatrix::SX);
      break;
    case Gates::mcsxdg:
      qreg.apply_mcu(op.qubits, Linalg::VMatrix::SXDG);
      break;
    case Gates::pauli:
      qreg.apply_pauli(op.qubits, op.string_params[0]);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument(
          "QubitVector::State::invalid gate instruction \'" + op.name + "\'.");
  }
}

template <class statevec_t>
void State<statevec_t>::apply_multiplexer(statevec_t& qreg, const reg_t &control_qubits,
                                          const reg_t &target_qubits,
                                          const cmatrix_t &mat) 
{
  if (control_qubits.empty() == false && target_qubits.empty() == false &&
      mat.size() > 0) {
    cvector_t vmat = Utils::vectorize_matrix(mat);
    qreg.apply_multiplexer(control_qubits, target_qubits, vmat);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(statevec_t& qreg, const Operations::Op &op) 
{
  if (op.qubits.empty() == false && op.mats[0].size() > 0) {
    if (Utils::is_diagonal(op.mats[0], .0)) {
      apply_diagonal_matrix(qreg, op.qubits, Utils::matrix_diagonal(op.mats[0]));
    } else {
      qreg.apply_matrix(op.qubits,
                                    Utils::vectorize_matrix(op.mats[0]));
    }
  }
}

template <class statevec_t>
void State<statevec_t>::apply_matrix(statevec_t& qreg, const reg_t &qubits,
                                     const cvector_t &vmat) 
{
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    apply_diagonal_matrix(qreg, qubits, vmat);
  } else {
    qreg.apply_matrix(qubits, vmat);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_diagonal_matrix(statevec_t& qreg, const reg_t &qubits, const cvector_t & diag)
{
  if(BaseState::global_chunk_indexing_ || !BaseState::multi_chunk_distribution_){
    //GPU computes all chunks in one kernel, so pass qubits and diagonal matrix as is
    qreg.apply_diagonal_matrix(qubits,diag);
  }
  else{
    reg_t qubits_in = qubits;
    cvector_t diag_in = diag;

    BaseState::block_diagonal_matrix(qreg.chunk_index(),qubits_in,diag_in);
    qreg.apply_diagonal_matrix(qubits_in,diag_in);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_gate_mcu(statevec_t& qreg, const reg_t &qubits, double theta,
                                       double phi, double lambda, double gamma) 
{
  qreg.apply_mcu(qubits, Linalg::VMatrix::u4(theta, phi, lambda, gamma));
}

template <class statevec_t>
void State<statevec_t>::apply_gate_phase(statevec_t& qreg, uint_t qubit, complex_t phase) 
{
  cvector_t diag = {{1., phase}};
  apply_diagonal_matrix(qreg, reg_t({qubit}), diag);
}

//=========================================================================
// Implementation: Reset, Initialize and Measurement Sampling
//=========================================================================

template <class statevec_t>
rvector_t State<statevec_t>::measure_probs(QuantumState::Registers<statevec_t>& state, const reg_t &qubits) const 
{
  if(!BaseState::multi_chunk_distribution_)
    return state.qreg().probabilities(qubits);

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
          auto chunkSum = state.qreg(i).probabilities(qubits_in_chunk);

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
      for(i=0;i<state.qregs().size();i++){
        auto chunkSum = state.qreg(i).probabilities(qubits_in_chunk);

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
          auto nr = std::real(state.qreg(i).norm());
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
      for(i=0;i<state.qregs().size();i++){
        auto nr = std::real(state.qreg(i).norm());
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
void State<statevec_t>::apply_measure(QuantumState::Registers<statevec_t>& state, const reg_t &qubits, const reg_t &cmemory,
                                      const reg_t &cregister, RngEngine &rng) 
{
  //shot branching
  if(BaseState::enable_shot_branching_){
    rvector_t probs = sample_measure_with_prob_shot_branching(state, qubits);

    //save result to cregs
    for(int_t i=0;i<probs.size();i++){
      const reg_t outcome = Utils::int2reg(i, 2, qubits.size());
      state.branch(i).creg_.store_measure(outcome, cmemory, cregister);
    }

    measure_reset_update_shot_branching(state, qubits, -1, probs);
  }
  else{
    // Actual measurement outcome
    const auto meas = sample_measure_with_prob(state, qubits, rng);
    // Implement measurement update
    measure_reset_update(state, qubits, meas.first, meas.first, meas.second);
    const reg_t outcome = Utils::int2reg(meas.first, 2, qubits.size());
    state.creg().store_measure(outcome, cmemory, cregister);
  }
}

template <class statevec_t>
void State<statevec_t>::apply_reset(QuantumState::Registers<statevec_t>& state, const reg_t &qubits, RngEngine &rng) 
{
  //shot branching
  if(BaseState::enable_shot_branching_){
    rvector_t probs = sample_measure_with_prob_shot_branching(state, qubits);

    measure_reset_update_shot_branching(state, qubits, 0, probs);
  }
  else{
    // Simulate unobserved measurement
    const auto meas = sample_measure_with_prob(state, qubits, rng);
    // Apply update to reset state
    measure_reset_update(state, qubits, 0, meas.first, meas.second);
  }
}

template <class statevec_t>
std::pair<uint_t, double>
State<statevec_t>::sample_measure_with_prob(QuantumState::Registers<statevec_t>& state, const reg_t &qubits,
                                            RngEngine &rng) 
{
  rvector_t probs = measure_probs(state, qubits);

  // Randomly pick outcome and return pair
  uint_t outcome = rng.rand_int(probs);
  return std::make_pair(outcome, probs[outcome]);
}

template <class statevec_t>
rvector_t State<statevec_t>::sample_measure_with_prob_shot_branching(QuantumState::Registers<statevec_t>& state, const reg_t &qubits)
{
  rvector_t probs = measure_probs(state, qubits);
  uint_t nshots = state.num_shots();
  reg_t shot_branch(nshots);

  for(int_t i=0;i<nshots;i++){
    shot_branch[i] = state.rng_shots(i).rand_int(probs);
  }

  //branch shots
  state.branch_shots(shot_branch, probs.size());

  return probs;
}

template <class statevec_t>
void State<statevec_t>::measure_reset_update(QuantumState::Registers<statevec_t>& state, const std::vector<uint_t> &qubits,
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
      state.qreg().apply_diagonal_matrix(qubits, mdiag);
    else{
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for  
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_diagonal_matrix(state.qreg(ic), qubits, mdiag);
        }
      }
      else{
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_diagonal_matrix(state.qreg(ic), qubits, mdiag);
        }
      }
    }

    // If it doesn't agree with the reset state update
    if (final_state != meas_state) {
      if(!BaseState::multi_chunk_distribution_)
        state.qreg().apply_mcx(qubits);
      else
        BaseState::apply_chunk_x(state, qubits[0]);
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    cvector_t mdiag(dim, 0.);
    mdiag[meas_state] = 1. / std::sqrt(meas_prob);

    if(!BaseState::multi_chunk_distribution_)
      state.qreg().apply_diagonal_matrix(qubits, mdiag);
    else{
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_diagonal_matrix(state.qreg(ic), qubits, mdiag);
        }
      }
      else{
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
            apply_diagonal_matrix(state.qreg(ic), qubits, mdiag);
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
        if(!BaseState::multi_chunk_distribution_)
          apply_matrix(state.qreg(), qubits, perm);
        else{
          if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
            for(int_t ig=0;ig<BaseState::num_groups_;ig++){
              for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
                apply_matrix(state.qreg(ic), qubits, perm);
            }
          }
          else{
            for(int_t ig=0;ig<BaseState::num_groups_;ig++){
              for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
                apply_matrix(state.qreg(ic), qubits, perm);
            }
          }
        }
      }
      else{
        for(int_t i=0;i<qubits.size();i++){
          if(((final_state >> i) & 1) != ((meas_state >> i) & 1)){
            BaseState::apply_chunk_x(state, qubits[i]);
          }
        }
      }
    }
  }
}

template <class statevec_t>
void State<statevec_t>::measure_reset_update_shot_branching(
                                             QuantumState::Registers<statevec_t>& state, const std::vector<uint_t> &qubits,
                                             const int_t final_state,
                                             const rvector_t& meas_probs)
{
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state

  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    for(int_t i=0;i<2;i++){
      cvector_t mdiag(2, 0.);
      mdiag[i] = 1. / std::sqrt(meas_probs[i]);

      Operations::Op op;
      op.type = OpType::diagonal_matrix;
      op.qubits = qubits;
      op.params = mdiag;
      state.add_op_after_branch(i, op);

      if(final_state >= 0 && final_state != i) {
        Operations::Op op;
        op.type = OpType::gate;
        op.name = "mcx";
        op.qubits = qubits;
        state.add_op_after_branch(i, op);
      }
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    for(int_t i=0;i<dim;i++){
      cvector_t mdiag(dim, 0.);
      mdiag[i] = 1. / std::sqrt(meas_probs[i]);

      Operations::Op op;
      op.type = OpType::diagonal_matrix;
      op.qubits = qubits;
      op.params = mdiag;
      state.add_op_after_branch(i, op);

      if(final_state >= 0 && final_state != i) {
        // build vectorized permutation matrix
        cvector_t perm(dim * dim, 0.);
        perm[final_state * dim + i] = 1.;
        perm[i * dim + final_state] = 1.;
        for (size_t j = 0; j < dim; j++) {
          if (j != final_state && j != i)
            perm[j * dim + j] = 1.;
        }
        Operations::Op op;
        op.type = OpType::matrix;
        op.qubits = qubits;
        op.mats.push_back(Utils::devectorize_matrix(perm));
        state.add_op_after_branch(i, op);
      }
    }
  }
}

template <class statevec_t>
std::vector<reg_t> State<statevec_t>::sample_measure_state(QuantumState::RegistersBase& state_in, const reg_t &qubits,
                                                     uint_t shots,
                                                     RngEngine &rng) 
{
  QuantumState::Registers<statevec_t>& state = dynamic_cast<QuantumState::Registers<statevec_t>&>(state_in);

  int_t i,j;
  // Generate flat register for storing
  std::vector<double> rnds(shots);
  reg_t allbit_samples(shots,0);

  if(!BaseState::multi_chunk_distribution_){
    bool tmp = state.qregs()[0].enable_batch(false);
    if(state.num_shots() > 1){
      double norm = state.qregs()[0].norm();

      //use independent rng for each shot
      for (i = 0; i < state.num_shots(); ++i)
        rnds[i] = state.rng_shots(i).rand(0, norm);
    }
    else{
      for (i = 0; i < shots; ++i)
        rnds[i] = rng.rand(0, 1);
    }

    allbit_samples = state.qregs()[0].sample_measure(rnds);
    state.qregs()[0].enable_batch(tmp);
  }
  else{
    std::vector<double> chunkSum(state.qregs().size()+1,0);
    double sum,localSum;

    //calculate per chunk sum
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++){
          bool batched = state.qregs()[ic].enable_batch(true);   //return sum of all chunks in group
          chunkSum[ic] = state.qregs()[ic].norm();
          state.qregs()[ic].enable_batch(batched);
        }
      }
    }
    else{
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++){
          bool batched = state.qregs()[ic].enable_batch(true);   //return sum of all chunks in group
          chunkSum[ic] = state.qregs()[ic].norm();
          state.qregs()[ic].enable_batch(batched);
        }
      }
    }

    localSum = 0.0;
    for(i=0;i<state.qregs().size();i++){
      sum = localSum;
      localSum += chunkSum[i];
      chunkSum[i] = sum;
    }
    chunkSum[state.qregs().size()] = localSum;

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

    for (i = 0; i < shots; ++i)
      rnds[i] = rng.rand(0, 1);

    reg_t local_samples(shots,0);

    //get rnds positions for each chunk
    for(i=0;i<state.qregs().size();i++){
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
        auto chunkSamples = state.qregs()[i].sample_measure(vRnd);

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
void State<statevec_t>::apply_initialize(QuantumState::Registers<statevec_t>& state, const reg_t &qubits,
                                         const cvector_t &params,
                                         RngEngine &rng) 
{
  auto sorted_qubits = qubits;
  std::sort(sorted_qubits.begin(), sorted_qubits.end());
  if (qubits.size() == BaseState::num_qubits_) {
    // If qubits is all ordered qubits in the statevector
    // we can just initialize the whole state directly
    if (qubits == sorted_qubits) {
      initialize_from_vector(state, params);
      return;
    }
  }

  if(BaseState::enable_shot_branching_){
    if(state.additional_ops().size() == 0){
      apply_reset(state, qubits, rng);

      Operations::Op op;
      op.type = OpType::initialize;
      op.name = "initialize";
      op.qubits = qubits;
      op.params = params;
      for(int_t i=0;i<state.num_branch();i++){
        state.add_op_after_branch(i, op);
      }

      return; //initialize will be done in next call because of shot branching in reset
    }
  }
  else{
    // Apply reset to qubits
    apply_reset(state, qubits, rng);
  }

  // Apply initialize_component
  if(!BaseState::multi_chunk_distribution_)
    state.qreg().initialize_component(qubits, params);
  else{
    reg_t qubits_in_chunk;
    reg_t qubits_out_chunk;
    BaseState::qubits_inout(qubits,qubits_in_chunk,qubits_out_chunk);

    if(qubits_out_chunk.size() == 0){   //no qubits outside of chunk
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
            state.qreg(i).initialize_component(qubits, params);
        }
      }
      else{
        for(int_t i=0;i<state.qregs().size();i++)
          state.qreg(i).initialize_component(qubits, params);
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
          for(int_t i=0;i<state.qregs().size();i++)
            apply_matrix(state.qreg(i), qubits_in_chunk, perm );
        }
        else{
          for(int_t i=0;i<state.qregs().size();i++)
            apply_matrix(state.qreg(i), qubits_in_chunk, perm );
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
                state.qreg(ic).initialize_from_data(state.qreg(baseChunk).data(),1ull << BaseState::chunk_bits_);
              }
              else{
                BaseState::recv_chunk(state, ic,baseChunk);
                //using swap chunk function to release send/recv buffers for Thrust
                reg_t swap(2);
                swap[0] = BaseState::chunk_bits_;
                swap[1] = BaseState::chunk_bits_;
                state.qreg(ic).apply_chunk_swap(swap,baseChunk);
              }
            }
            else if(baseChunk >= BaseState::chunk_index_begin_[BaseState::distributed_rank_] && baseChunk < BaseState::chunk_index_end_[BaseState::distributed_rank_]){    //base chunk is on this process
              BaseState::send_chunk(state, baseChunk - BaseState::global_chunk_index_,ic);
            }
          }
        }
      }

      //initialize by params
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
            apply_diagonal_matrix(state.qreg(i), qubits,params );
        }
      }
      else{
        for(int_t i=0;i<state.qregs().size();i++)
          apply_diagonal_matrix(state.qreg(i), qubits,params );
      }
    }
  }
}

template <class statevec_t>
void State<statevec_t>::initialize_from_vector(QuantumState::Registers<statevec_t>& state, const cvector_t &params)
{
  if(!BaseState::multi_chunk_distribution_)
    state.qreg().initialize_from_vector(params);
  else{   //multi-chunk distribution
    uint_t local_offset = BaseState::global_chunk_index_ << BaseState::chunk_bits_;

#pragma omp parallel for if(BaseState::chunk_omp_parallel_)
    for(int_t i=0;i<state.qregs().size();i++){
      //copy part of state for this chunk
      cvector_t tmp(1ull << BaseState::chunk_bits_);
      std::copy(params.begin() + local_offset + (i << BaseState::chunk_bits_),
                params.begin() + local_offset + ((i+1) << BaseState::chunk_bits_),
                tmp.begin());
      state.qreg(i).initialize_from_vector(tmp);
    }
  }
}

//=========================================================================
// Implementation: Multiplexer Circuit
//=========================================================================

template <class statevec_t>
void State<statevec_t>::apply_multiplexer(statevec_t& qreg, const reg_t &control_qubits,
                                          const reg_t &target_qubits,
                                          const std::vector<cmatrix_t> &mmat) {
  // (1) Pack vector of matrices into single (stacked) matrix ... note: matrix
  // dims: rows = DIM[qubit.size()] columns = DIM[|target bits|]
  cmatrix_t multiplexer_matrix = Utils::stacked_matrix(mmat);

  // (2) Treat as single, large(r), chained/batched matrix operator
  apply_multiplexer(qreg, control_qubits, target_qubits, multiplexer_matrix);
}

//=========================================================================
// Implementation: Kraus Noise
//=========================================================================
template <class statevec_t>
void State<statevec_t>::apply_kraus(QuantumState::Registers<statevec_t>& state, const reg_t &qubits,
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

  double r;
  double accum = 0.;
  double p;
  bool complete = false;

  reg_t shot_branch;
  uint_t nshots;
  rvector_t rshots,pmats;
  uint_t nshots_multiplied = 0;

  if(BaseState::enable_shot_branching_){
    nshots = state.num_shots();
    shot_branch.resize(nshots);
    rshots.resize(nshots);
    for(int_t i=0;i<nshots;i++){
      shot_branch[i] = kmats.size() - 1;
      rshots[i] = state.rng_shots(i).rand(0., 1.);
    }
    pmats.resize(kmats.size());
  }
  else{
    r = rng.rand(0., 1.);
  }

  // Loop through N-1 kraus operators
  for (size_t j = 0; j < kmats.size() - 1; j++) {

    // Calculate probability
    cvector_t vmat = Utils::vectorize_matrix(kmats[j]);

    if(!BaseState::multi_chunk_distribution_){
      p = state.qreg().norm(qubits, vmat);
      accum += p;
    }
    else{
      p = 0.0;
      if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for reduction(+:p)
        for(int_t ig=0;ig<BaseState::num_groups_;ig++){
          for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
            p += state.qreg(i).norm(qubits, vmat);
        }
      }
      else{
        for(int_t i=0;i<state.qregs().size();i++)
          p += state.qreg(i).norm(qubits, vmat);
      }

#ifdef AER_MPI
      BaseState::reduce_sum(p);
#endif
      accum += p;
    }

    // check if we need to apply this operator
    if(BaseState::enable_shot_branching_){
      pmats[j] = p;
      for(int_t i=0;i<nshots;i++){
        if(shot_branch[i] >= kmats.size() - 1){
          if(accum > rshots[i]){
            shot_branch[i] = j;
            nshots_multiplied++;
          }
        }
      }
      if(nshots_multiplied >= nshots){
        complete = true;
        break;
      }
    }
    else{
      if (accum > r) {
        // rescale vmat so projection is normalized
        Utils::scalar_multiply_inplace(vmat, 1 / std::sqrt(p));
        // apply Kraus projection operator
        if(!BaseState::multi_chunk_distribution_)
          apply_matrix(state.qreg(), qubits, vmat);
        else{
          if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
            for(int_t ig=0;ig<BaseState::num_groups_;ig++){
              for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
                apply_matrix(state.qreg(ic), qubits, vmat);
            }
          }
          else{
            for(int_t ig=0;ig<BaseState::num_groups_;ig++){
              for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
                apply_matrix(state.qreg(ic), qubits, vmat);
            }
          }
        }
        complete = true;
        break;
      }
    }
  }

  // check if we haven't applied a kraus operator yet
  if(BaseState::enable_shot_branching_){
    pmats[pmats.size()-1] = 1. - accum;

    state.branch_shots(shot_branch, kmats.size());
    for(int_t i=0;i<kmats.size();i++){
      Operations::Op op;
      op.type = OpType::matrix;
      op.qubits = qubits;
      op.mats.push_back(kmats[i]);
      Utils::scalar_multiply_inplace(op.mats[0], 1/std::sqrt(pmats[i]));
      state.add_op_after_branch(i, op);
    }
  }
  else{
    if (complete == false) {
      // Compute probability from accumulated
      complex_t renorm = 1 / std::sqrt(1. - accum);
      auto vmat = Utils::vectorize_matrix(renorm * kmats.back());
      if(!BaseState::multi_chunk_distribution_)
        apply_matrix(state.qreg(), qubits, vmat);
      else{
        if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 1){
#pragma omp parallel for 
          for(int_t ig=0;ig<BaseState::num_groups_;ig++){
            for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
              apply_matrix(state.qreg(ic), qubits, vmat);
          }
        }
        else{
          for(int_t ig=0;ig<BaseState::num_groups_;ig++){
            for(int_t ic = BaseState::top_chunk_of_group_[ig];ic < BaseState::top_chunk_of_group_[ig + 1];ic++)
              apply_matrix(state.qreg(ic), qubits, vmat);
          }
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
