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

#ifndef _unitary_state_hpp
#define _unitary_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include "simulators/state.hpp"
#include "framework/json.hpp"
#include "framework/utils.hpp"
#include "simulators/state_chunk.hpp"
#include "unitarymatrix.hpp"
#ifdef AER_THRUST_SUPPORTED
#include "unitarymatrix_thrust.hpp"
#endif

namespace AER {

namespace QubitUnitary {

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {Operations::OpType::gate, Operations::OpType::barrier,
     Operations::OpType::bfunc, Operations::OpType::roerror,
     Operations::OpType::qerror_loc,
     Operations::OpType::matrix, Operations::OpType::diagonal_matrix,
     Operations::OpType::snapshot, Operations::OpType::save_unitary,
     Operations::OpType::save_state, Operations::OpType::set_unitary,
     Operations::OpType::jump, Operations::OpType::mark
    },
    // Gates
    {"u1",     "u2",      "u3",  "u",    "U",    "CX",   "cx",   "cz",
     "cy",     "cp",      "cu1", "cu2",  "cu3",  "swap", "id",   "p",
     "x",      "y",       "z",   "h",    "s",    "sdg",  "t",    "tdg",
     "r",      "rx",      "ry",  "rz",   "rxx",  "ryy",  "rzz",  "rzx",
     "ccx",    "cswap",   "mcx", "mcy",  "mcz",  "mcu1", "mcu2", "mcu3",
     "mcswap", "mcphase", "mcr", "mcrx", "mcry", "mcry", "sx",   "sxdg", "csx",
     "mcsx",   "csxdg", "mcsxdg", "delay", "pauli", "cu",   "mcu", "mcp"},
    // Snapshots
    {"unitary"});

// Allowed gates enum class
enum class Gates {
  id, h, s, sdg, t, tdg, rxx, ryy, rzz, rzx,
  mcx, mcy, mcz, mcr, mcrx, mcry, mcrz, mcp,
  mcu2, mcu3, mcu, mcswap, mcsx, mcsxdg, pauli,
};

//=========================================================================
// QubitUnitary State subclass
//=========================================================================

template <class unitary_matrix_t = QV::UnitaryMatrix<double>>
class State : public virtual Base::StateChunk<unitary_matrix_t> {
public:
  using BaseState = Base::StateChunk<unitary_matrix_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return "unitary"; }

  // Apply an operation
  // If the op is not in allowed_ops an exeption will be raised.
  virtual void apply_op(const int_t iChunk, const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine &rng,
                        bool final_op = false) override;

  // Initializes an n-qubit unitary to the identity matrix
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit unitary matrix
  virtual void initialize_qreg(uint_t num_qubits,
                               const unitary_matrix_t &unitary) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << 2 * num_qubits bytes
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  // Config: {"omp_qubit_threshold": 7}
  virtual void set_config(const json_t &config) override;

  //-----------------------------------------------------------------------
  // Additional methods
  //-----------------------------------------------------------------------

  // Initializes to a specific n-qubit unitary given as a complex matrix
  virtual void initialize_qreg(uint_t num_qubits, const cmatrix_t &unitary);

  // Initialize OpenMP settings for the underlying QubitVector class
  void initialize_omp();

  auto move_to_matrix(const int_t iChunk);
  auto copy_to_matrix(const int_t iChunk);
protected:
  //-----------------------------------------------------------------------
  // Apply Instructions
  //-----------------------------------------------------------------------
  //apply op to multiple shots , return flase if op is not supported to execute in a batch
  bool apply_batched_op(const int_t iChunk, const Operations::Op &op,
                                ExperimentResult &result,
                                std::vector<RngEngine> &rng,
                                bool final_op = false) override;

  // Applies a Gate operation to the state class.
  // This should support all and only the operations defined in
  // allowed_operations.
  void apply_gate(const int_t iChunk, const Operations::Op &op);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const int_t iChunk, const Operations::Op &op, ExperimentResult &result);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cmatrix_t &mat);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t &vmat);

  // Apply a diagonal matrix
  void apply_diagonal_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t &diag);

  //swap between chunks
  virtual void apply_chunk_swap(const reg_t &qubits) override;

  //-----------------------------------------------------------------------
  // 1-Qubit Gates
  //-----------------------------------------------------------------------

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const int_t iChunk, const uint_t qubit, const complex_t phase);

  void apply_gate_phase(const int_t iChunk, const reg_t& qubits, const complex_t phase);

  //-----------------------------------------------------------------------
  // Multi-controlled u
  //-----------------------------------------------------------------------

  // Apply N-qubit multi-controlled single qubit gate specified by
  // 4 parameters u4(theta, phi, lambda, gamma)
  // NOTE: if N=1 this is just a regular u4 gate.
  void apply_gate_mcu(const int_t iChunk, const reg_t &qubits, double theta,
                      double phi, double lambda, double gamma);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the unitary matrix for the simulator
  void apply_save_unitary(const int_t iChunk, const Operations::Op &op,
                          ExperimentResult &result,
                          bool last_op);

  // Helper function for computing expectation value
  virtual double expval_pauli(const int_t iChunk, const reg_t &qubits,
                              const std::string& pauli) override;

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Apply the global phase
  void apply_global_phase();

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 6;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;

  //scale for unitary = 2
  //this function is used in the base class to scale chunk qubits for multi-chunk distribution
  int qubit_scale(void) override
  {
    return 2;
  }
};

//============================================================================
// Implementation: Allowed ops and gateset
//============================================================================

template <class unitary_matrix_t>
const stringmap_t<Gates> State<unitary_matrix_t>::gateset_({
    // Single qubit gates
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
    {"sxdg", Gates::mcsxdg}, // Sqrt(X)^hc gate
    // 1-qubit rotation Gates
    {"r", Gates::mcr},   // R rotation gate
    {"rx", Gates::mcrx}, // Pauli-X rotation gate
    {"ry", Gates::mcry}, // Pauli-Y rotation gate
    {"rz", Gates::mcrz}, // Pauli-Z rotation gate
    // Waltz Gates
    {"p", Gates::mcp},   // Parameterized phase gate 
    {"u1", Gates::mcp}, // zero-X90 pulse waltz gate
    {"u2", Gates::mcu2}, // single-X90 pulse waltz gate
    {"u3", Gates::mcu3}, // two X90 pulse waltz gate
    {"u", Gates::mcu3}, // two X90 pulse waltz gate
    {"U", Gates::mcu3}, // two X90 pulse waltz gate
    // Two-qubit gates
    {"CX", Gates::mcx},      // Controlled-X gate (CNOT)
    {"cx", Gates::mcx},      // Controlled-X gate (CNOT)
    {"cy", Gates::mcy},      // Controlled-Z gate
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
    {"csxdg", Gates::mcsxdg},// Controlled-Sqrt(X)dg gate
    // Three-qubit gates
    {"ccx", Gates::mcx},      // Controlled-CX gate (Toffoli)
    {"cswap", Gates::mcswap}, // Controlled-SWAP gate (Fredkin)
    // Multi-qubit controlled gates
    {"mcx", Gates::mcx},      // Multi-controlled-X gate
    {"mcy", Gates::mcy},      // Multi-controlled-Y gate
    {"mcz", Gates::mcz},      // Multi-controlled-Z gate
    {"mcr", Gates::mcr},      // Multi-controlled R-rotation gate
    {"mcrx", Gates::mcrx},    // Multi-controlled X-rotation gate
    {"mcry", Gates::mcry},    // Multi-controlled Y-rotation gate
    {"mcrz", Gates::mcrz},    // Multi-controlled Z-rotation gate
    {"mcphase", Gates::mcp},  // Multi-controlled-Phase gate 
    {"mcu1", Gates::mcp},    // Multi-controlled-u1
    {"mcu2", Gates::mcu2},    // Multi-controlled-u2
    {"mcu3", Gates::mcu3},    // Multi-controlled-u3
    {"mcu", Gates::mcu},      // Multi-controlled-u4
    {"mcp", Gates::mcp},      // Multi-controlled-Phase gate 
    {"mcswap", Gates::mcswap},// Multi-controlled SWAP gate
    {"mcsx", Gates::mcsx},    // Multi-controlled-Sqrt(X) gate
    {"mcsxdg", Gates::mcsxdg},    // Multi-controlled-Sqrt(X)dg gate
    {"pauli", Gates::pauli}  // Multiple pauli operations at once
});

//============================================================================
// Implementation: Base class method overrides
//============================================================================

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_op(const int_t iChunk, 
    const Operations::Op &op, ExperimentResult &result,
    RngEngine &rng, bool final_op)
{
  if(BaseState::check_conditional(iChunk, op)) {
    switch (op.type) {
      case Operations::OpType::barrier:
      case Operations::OpType::qerror_loc:
        break;
      case Operations::OpType::bfunc:
        BaseState::cregs_[0].apply_bfunc(op);
        break;
      case Operations::OpType::roerror:
        BaseState::cregs_[0].apply_roerror(op, rng);
        break;
      case Operations::OpType::gate:
        apply_gate(iChunk, op);
        break;
      case Operations::OpType::set_unitary:
        BaseState::initialize_from_matrix(iChunk, op.mats[0]);
        break;
      case Operations::OpType::save_state:
      case Operations::OpType::save_unitary:
        apply_save_unitary(iChunk, op, result, final_op);
        break;
      case Operations::OpType::snapshot:
        apply_snapshot(iChunk, op, result);
        break;
      case Operations::OpType::matrix:
        apply_matrix(iChunk, op.qubits, op.mats[0]);
        break;
      case Operations::OpType::diagonal_matrix:
        apply_diagonal_matrix(iChunk, op.qubits, op.params);
        break;
      default:
        throw std::invalid_argument(
            "QubitUnitary::State::invalid instruction \'" + op.name + "\'.");
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
    case Operations::OpType::barrier:
    case Operations::OpType::nop:
    case Operations::OpType::qerror_loc:
      break;
    case Operations::OpType::bfunc:
      BaseState::qregs_[iChunk].apply_bfunc(op);
      break;
    case Operations::OpType::roerror:
      BaseState::qregs_[iChunk].apply_roerror(op, rng);
      break;
    case Operations::OpType::gate:
      apply_gate(iChunk, op);
      break;
    case Operations::OpType::matrix:
      apply_matrix(iChunk, op.qubits, op.mats[0]);
      break;
    case Operations::OpType::diagonal_matrix:
      BaseState::qregs_[iChunk].apply_diagonal_matrix(op.qubits, op.params);
      break;
    default:
      //other operations should be called to indivisual chunks by apply_op
      return false;
  }
  return true;
}


template <class unitary_matrix_t>
size_t State<unitary_matrix_t>::required_memory_mb(
    uint_t num_qubits, const std::vector<Operations::Op> &ops) const {
  // An n-qubit unitary as 2^2n complex doubles
  // where each complex double is 16 bytes
  (void)ops; // avoid unused variable compiler warning
  size_t shift_mb = std::max<int_t>(0, num_qubits + 4 - 20);
  size_t mem_mb = 1ULL << (2 * shift_mb);
  return mem_mb;
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::set_config(const json_t &config) 
{
  BaseState::set_config(config);

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "unitary_parallel_threshold", config);

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "zero_threshold", config);

  for(int_t i=0;i<BaseState::qregs_.size();i++)
    BaseState::qregs_[i].set_json_chop_threshold(json_chop_threshold_);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_qreg(uint_t num_qubits) 
{
  if(BaseState::qregs_.size() == 0)
    BaseState::allocate(num_qubits,num_qubits,1);

  initialize_omp();

  int_t iChunk;
  for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
    BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);
  }

  if(BaseState::multi_chunk_distribution_){
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for private(iChunk) 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++){
          uint_t irow,icol;
          irow = (BaseState::global_chunk_index_ + iChunk) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
          icol = (BaseState::global_chunk_index_ + iChunk) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
          if(irow == icol)
            BaseState::qregs_[iChunk].initialize();
          else
            BaseState::qregs_[iChunk].zero();
        }
      }
    }
    else{
      for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
        uint_t irow,icol;
        irow = (BaseState::global_chunk_index_ + iChunk) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_));
        icol = (BaseState::global_chunk_index_ + iChunk) - (irow << ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
        if(irow == icol)
          BaseState::qregs_[iChunk].initialize();
        else
          BaseState::qregs_[iChunk].zero();
      }
    }
  }
  else{
    for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
      BaseState::qregs_[iChunk].initialize();
    }
  }
  apply_global_phase();
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_qreg(uint_t num_qubits,
                                              const unitary_matrix_t &unitary) 
{
  // Check dimension of state
  if (unitary.num_qubits() != num_qubits) {
    throw std::invalid_argument(
        "Unitary::State::initialize: initial state does not match qubit "
        "number");
  }
  if(BaseState::qregs_.size() == 0)
    BaseState::allocate(num_qubits,num_qubits,1);
  initialize_omp();

  int_t iChunk;
  for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++)
    BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);

  if(BaseState::multi_chunk_distribution_){
    auto input = unitary.copy_to_matrix();
    uint_t mask = (1ull << (BaseState::chunk_bits_)) - 1;

    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for private(iChunk) 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++){
          uint_t irow_chunk = ((iChunk + BaseState::global_chunk_index_) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
          uint_t icol_chunk = ((iChunk + BaseState::global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_)))-1));

          //copy part of state for this chunk
          uint_t i,row,col;
          cvector_t tmp(1ull << BaseState::chunk_bits_);
          for(i=0;i<(1ull << BaseState::chunk_bits_);i++){
            uint_t icol = i >> (BaseState::chunk_bits_);
            uint_t irow = i & mask;
            uint_t idx = ((icol+(irow_chunk << BaseState::chunk_bits_)) << (BaseState::num_qubits_)) + (icol_chunk << BaseState::chunk_bits_) + irow;
            tmp[i] = input[idx];
          }
          BaseState::qregs_[iChunk].initialize_from_vector(tmp);
        }
      }
    }
    else{
      for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
        uint_t irow_chunk = ((iChunk + BaseState::global_chunk_index_) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
        uint_t icol_chunk = ((iChunk + BaseState::global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_)))-1));

        //copy part of state for this chunk
        uint_t i,row,col;
        cvector_t tmp(1ull << BaseState::chunk_bits_);
        for(i=0;i<(1ull << BaseState::chunk_bits_);i++){
          uint_t icol = i >> (BaseState::chunk_bits_);
          uint_t irow = i & mask;
          uint_t idx = ((icol+(irow_chunk << BaseState::chunk_bits_)) << (BaseState::num_qubits_)) + (icol_chunk << BaseState::chunk_bits_) + irow;
          tmp[i] = input[idx];
        }
        BaseState::qregs_[iChunk].initialize_from_vector(tmp);
      }
    }
  }
  else{
    BaseState::qregs_[iChunk].initialize_from_data(unitary.data(), 1ULL << 2 * num_qubits);
  }
  apply_global_phase();
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_qreg(uint_t num_qubits,
                                              const cmatrix_t &unitary) 
{
  // Check dimension of unitary
  if (unitary.size() != 1ULL << (2 * num_qubits)) {
    throw std::invalid_argument(
        "Unitary::State::initialize: initial state does not match qubit "
        "number");
  }
  if(BaseState::qregs_.size() == 0)
    BaseState::allocate(num_qubits,num_qubits,1);
  initialize_omp();

  int_t iChunk;
  for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++)
    BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);

  if(BaseState::multi_chunk_distribution_){
    uint_t mask = (1ull << (BaseState::chunk_bits_)) - 1;
    for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
      //this function should be called in-order
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_);
    }

    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for private(iChunk) 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(iChunk = BaseState::top_chunk_of_group_[ig];iChunk < BaseState::top_chunk_of_group_[ig + 1];iChunk++){
          uint_t irow_chunk = ((iChunk + BaseState::global_chunk_index_) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
          uint_t icol_chunk = ((iChunk + BaseState::global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_)))-1));

          //copy part of state for this chunk
          uint_t i,row,col;
          cvector_t tmp(1ull << BaseState::chunk_bits_);
          for(i=0;i<(1ull << BaseState::chunk_bits_);i++){
            uint_t icol = i >> (BaseState::chunk_bits_);
            uint_t irow = i & mask;
            uint_t idx = ((icol+(irow_chunk << BaseState::chunk_bits_)) << (BaseState::num_qubits_)) + (icol_chunk << BaseState::chunk_bits_) + irow;
            tmp[i] = unitary[idx];
          }
          BaseState::qregs_[iChunk].initialize_from_vector(tmp);
        }
      }
    }
    else{
      for(iChunk=0;iChunk<BaseState::qregs_.size();iChunk++){
        uint_t irow_chunk = ((iChunk + BaseState::global_chunk_index_) >> ((BaseState::num_qubits_ - BaseState::chunk_bits_)));
        uint_t icol_chunk = ((iChunk + BaseState::global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - BaseState::chunk_bits_)))-1));

        //copy part of state for this chunk
        uint_t i,row,col;
        cvector_t tmp(1ull << BaseState::chunk_bits_);
        for(i=0;i<(1ull << BaseState::chunk_bits_);i++){
          uint_t icol = i >> (BaseState::chunk_bits_);
          uint_t irow = i & mask;
          uint_t idx = ((icol+(irow_chunk << BaseState::chunk_bits_)) << (BaseState::num_qubits_)) + (icol_chunk << BaseState::chunk_bits_) + irow;
          tmp[i] = unitary[idx];
        }
        BaseState::qregs_[iChunk].initialize_from_vector(tmp);
      }
    }
  }
  else{
    BaseState::qregs_[iChunk].initialize_from_matrix(unitary);
  }
  apply_global_phase();
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_omp() 
{
  uint_t i;
  for(i=0;i<BaseState::qregs_.size();i++){
    BaseState::qregs_[i].set_omp_threshold(omp_qubit_threshold_);
    if (BaseState::threads_ > 0)
      BaseState::qregs_[i].set_omp_threads(BaseState::threads_); // set allowed OMP threads in qubitvector
  }
}

template <class unitary_matrix_t>
auto State<unitary_matrix_t>::move_to_matrix(const int_t iChunk)
{
  if(!BaseState::multi_chunk_distribution_)
    return BaseState::qregs_[iChunk].move_to_matrix();
  return BaseState::apply_to_matrix(false);
}

template <class unitary_matrix_t>
auto State<unitary_matrix_t>::copy_to_matrix(const int_t iChunk)
{
  if(!BaseState::multi_chunk_distribution_)
    return BaseState::qregs_[iChunk].copy_to_matrix();
  return BaseState::apply_to_matrix(true);
}

//=========================================================================
// Implementation: Gates
//=========================================================================

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate(const int_t iChunk, const Operations::Op &op) 
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
    throw std::invalid_argument("Unitary::State::invalid gate instruction \'" +
                                op.name + "\'.");
  Gates g = it->second;
  switch (g) {
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
      apply_diagonal_matrix(iChunk, op.qubits, Linalg::VMatrix::rzz_diag(op.params[0]));
      break;
    case Gates::rzx:
      BaseState::qregs_[iChunk].apply_matrix(op.qubits, Linalg::VMatrix::rzx(op.params[0]));
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
    case Gates::pauli:
        BaseState::qregs_[iChunk].apply_pauli(op.qubits, op.string_params[0]);
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
      // Includes u, cu, etc
      apply_gate_mcu(iChunk, op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                      std::real(op.params[2]), std::real(op.params[3]));
      break;
    case Gates::mcu2:
      // Includes u2, cu2, etc
      apply_gate_mcu(iChunk, op.qubits, M_PI / 2., std::real(op.params[0]),
                     std::real(op.params[1]), 0.);
      break;
    case Gates::mcp:
      // Includes u1, cu1, p, cp, mcp, etc
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
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument("Unitary::State::invalid gate instruction \'" +
                                  op.name + "\'.");
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_matrix(const int_t iChunk, const reg_t &qubits,
                                           const cmatrix_t &mat) 
{
  if (qubits.empty() == false && mat.size() > 0) {
    apply_matrix(iChunk, qubits, Utils::vectorize_matrix(mat));
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_matrix(const int_t iChunk, const reg_t &qubits,
                                           const cvector_t &vmat) 
{
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    apply_diagonal_matrix(iChunk, qubits, vmat);
  } else {
    BaseState::qregs_[iChunk].apply_matrix(qubits, vmat);
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_diagonal_matrix(const int_t iChunk, const reg_t &qubits, const cvector_t &diag)
{
  if(BaseState::global_chunk_indexing_ || !BaseState::multi_chunk_distribution_){
    //GPU computes all chunks in one kernel, so pass qubits and diagonal matrix as is
    reg_t qubits_chunk = qubits;
    for(uint_t i=0;i<qubits.size();i++){
      if(qubits_chunk[i] >= BaseState::chunk_bits_){
        qubits_chunk[i] += BaseState::chunk_bits_;
      }
    }
    BaseState::qregs_[iChunk].apply_diagonal_matrix(qubits_chunk,diag);
  }
  else{
    reg_t qubits_in = qubits;
    cvector_t diag_in = diag;

    BaseState::block_diagonal_matrix(iChunk,qubits_in,diag_in);
    BaseState::qregs_[iChunk].apply_diagonal_matrix(qubits_in,diag_in);
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_phase(const int_t iChunk, uint_t qubit, complex_t phase) 
{
  cvector_t diag(2);
  diag[0] = 1.0;
  diag[1] = phase;
  apply_diagonal_matrix(iChunk, reg_t({qubit}), diag);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_phase(const int_t iChunk, const reg_t& qubits, complex_t phase)
{
  cvector_t diag((1 << qubits.size()),1.0);
  diag[(1 << qubits.size()) - 1] = phase;
  apply_diagonal_matrix(iChunk, qubits, diag);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_mcu(const int_t iChunk, const reg_t &qubits, double theta,
                                              double phi, double lambda, double gamma) 
{
  const auto u4 = Linalg::Matrix::u4(theta, phi, lambda, gamma);
  BaseState::qregs_[iChunk].apply_mcu(qubits, Utils::vectorize_matrix(u4));
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_snapshot(const int_t iChunk, const Operations::Op &op,
                                             ExperimentResult &result) 
{
  // Look for snapshot type in snapshotset
  if (op.name == "unitary" || op.name == "state") {
    result.legacy_data.add_pershot_snapshot("unitary", op.string_params[0],
                              copy_to_matrix(iChunk));
    BaseState::snapshot_state(iChunk, op, result);
  } else {
    throw std::invalid_argument(
        "Unitary::State::invalid snapshot instruction \'" + op.name + "\'.");
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_global_phase() 
{
  if (BaseState::has_global_phase_) {
    if(BaseState::chunk_omp_parallel_ && BaseState::num_groups_ > 0){
#pragma omp parallel for 
      for(int_t ig=0;ig<BaseState::num_groups_;ig++){
        for(int_t i = BaseState::top_chunk_of_group_[ig];i < BaseState::top_chunk_of_group_[ig + 1];i++)
          apply_diagonal_matrix(i, {0}, {BaseState::global_phase_, BaseState::global_phase_});
      }
    }
    else{
      for(int_t i=0;i<BaseState::qregs_.size();i++)
        apply_diagonal_matrix(i, {0}, {BaseState::global_phase_, BaseState::global_phase_});
    }
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_save_unitary(const int_t iChunk, const Operations::Op &op,
                                                 ExperimentResult &result,
                                                 bool last_op) 
{
  if (op.qubits.size() != BaseState::num_qubits_) {
    throw std::invalid_argument(
        op.name + " was not applied to all qubits."
        " Only the full unitary can be saved.");
  }
  std::string key = (op.string_params[0] == "_method_") ? "unitary" : op.string_params[0];

  if (last_op) {
    BaseState::save_data_pershot(iChunk, result, key, move_to_matrix(iChunk),
                                 Operations::OpType::save_unitary,
                                 op.save_type);
  } else {
    BaseState::save_data_pershot(iChunk, result, key, copy_to_matrix(iChunk),
                                 Operations::OpType::save_unitary,
                                 op.save_type);
  }
}

template <class unitary_matrix_t>
double  State<unitary_matrix_t>::expval_pauli(const int_t iChunk, const reg_t &qubits,
                                              const std::string& pauli) 
{
  throw std::runtime_error("Unitary simulator does not support Pauli expectation values.");
}

//swap between chunks
template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_chunk_swap(const reg_t &qubits)
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
}

//------------------------------------------------------------------------------
} // namespace QubitUnitary
} // end namespace AER
//------------------------------------------------------------------------------
#endif
