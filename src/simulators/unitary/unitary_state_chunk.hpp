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

#ifndef _unitary_state_chunk_hpp
#define _unitary_state_chunk_hpp

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

//#include "unitary_state.hpp"

namespace AER {
namespace QubitUnitaryChunk {

// OpSet of supported instructions
const Operations::OpSet StateOpSet(
    // Op types
    {Operations::OpType::gate, Operations::OpType::barrier,
     Operations::OpType::matrix, Operations::OpType::diagonal_matrix},
    // Gates
    {"u1",     "u2",      "u3",  "u",    "U",    "CX",   "cx",   "cz",
     "cy",     "cp",      "cu1", "cu2",  "cu3",  "swap", "id",   "p",
     "x",      "y",       "z",   "h",    "s",    "sdg",  "t",    "tdg",
     "r",      "rx",      "ry",  "rz",   "rxx",  "ryy",  "rzz",  "rzx",
     "ccx",    "cswap",   "mcx", "mcy",  "mcz",  "mcu1", "mcu2", "mcu3",
     "mcswap", "mcphase", "mcr", "mcrx", "mcry", "mcry", "sx",   "csx",
     "mcsx",   "delay", "pauli"},
    // Snapshots
    {});

// Allowed gates enum class
enum class Gates {
  id, h, s, sdg, t, tdg, rxx, ryy, rzz, rzx,
  mcx, mcy, mcz, mcr, mcrx, mcry, mcrz, mcp, mcu2, mcu3, mcswap, mcsx, pauli,
};

//=========================================================================
// QubitUnitary State subclass
//=========================================================================

template <class unitary_matrix_t = QV::UnitaryMatrix<double>>
class State : public Base::StateChunk<unitary_matrix_t> {
public:
  using BaseState = Base::StateChunk<unitary_matrix_t>;

  State() : BaseState(StateOpSet) {}
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override { return "unitary"; }

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

  auto move_to_matrix();

protected:
  //-----------------------------------------------------------------------
  // Apply Instructions
  //-----------------------------------------------------------------------
  virtual void apply_op(const int_t iChunk,const Operations::Op &op,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops = false) override;

  // Applies a Gate operation to the state class.
  // This should support all and only the operations defined in
  // allowed_operations.
  void apply_gate(const uint_t iChunk,const Operations::Op &op);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const Operations::Op &op, ExperimentResult &result);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const uint_t iChunk,const reg_t &qubits, const cmatrix_t &mat);

  // Apply a matrix to given qubits (identity on all other qubits)
  void apply_matrix(const uint_t iChunk,const reg_t &qubits, const cvector_t &vmat);

  //-----------------------------------------------------------------------
  // 1-Qubit Gates
  //-----------------------------------------------------------------------

  // Optimize phase gate with diagonal [1, phase]
  void apply_gate_phase(const uint_t iChunk,const uint_t qubit, const complex_t phase);

  //-----------------------------------------------------------------------
  // Multi-controlled u3
  //-----------------------------------------------------------------------

  // Apply N-qubit multi-controlled single qubit waltz gate specified by
  // parameters u3(theta, phi, lambda)
  // NOTE: if N=1 this is just a regular u3 gate.
  void apply_gate_mcu3(const uint_t iChunk,const reg_t &qubits, const double theta,
                       const double phi, const double lambda);

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
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

  int qubit_scale() override
  {
    return 2;
  }
};


//============================================================================
// Implementation: Base class method overrides
//============================================================================
template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_op(const int_t iChunk,const Operations::Op &op,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops)
{
  switch (op.type) {
    case Operations::OpType::barrier:
      break;
    case Operations::OpType::gate:
      // Note conditionals will always fail since no classical registers
      if (BaseState::creg_.check_conditional(op))
        apply_gate(iChunk,op);
      break;
    case Operations::OpType::snapshot:
      apply_snapshot(op, result);
      break;
    case Operations::OpType::matrix:
      apply_matrix(iChunk,op.qubits, op.mats[0]);
      break;
    case Operations::OpType::diagonal_matrix:
      BaseState::qregs_[iChunk].apply_diagonal_matrix(op.qubits, op.params);
      break;
    default:
      throw std::invalid_argument(
          "QubitUnitary::State::invalid instruction \'" + op.name + "\'.");
  }
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
void State<unitary_matrix_t>::set_config(const json_t &config) {
  BaseState::set_config(config);

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "unitary_parallel_threshold", config);

  // Set threshold for truncating snapshots
  JSON::get_value(json_chop_threshold_, "zero_threshold", config);
  uint_t i;
  for(i=0;i<BaseState::num_local_chunks_;i++){
    BaseState::qregs_[i].set_json_chop_threshold(json_chop_threshold_);
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_qreg(uint_t num_qubits) 
{
  int_t i;

  initialize_omp();

  if(BaseState::chunk_bits_ == BaseState::num_qubits_){
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].set_num_qubits(BaseState::chunk_bits_/2);
      BaseState::qregs_[i].zero();
    }
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
  initialize_omp();

  int_t iChunk;
  if(BaseState::chunk_bits_ == BaseState::num_qubits_){
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_/2);
    }
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      BaseState::qregs_[iChunk].initialize_from_data(unitary.data(), 1ull << BaseState::chunk_bits_);
    }
  }
  else{   //multi-chunk distribution
    uint_t local_offset = BaseState::global_chunk_index_ << BaseState::chunk_bits_;

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_/2);
      BaseState::qregs_[iChunk].initialize_from_data(unitary.data() + local_offset + (iChunk << BaseState::chunk_bits_), 1ull << BaseState::chunk_bits_);
    }
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
  initialize_omp();

  int_t iChunk;
  if(BaseState::chunk_bits_ == BaseState::num_qubits_){
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_/2);
    }
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      BaseState::qregs_[iChunk].initialize_from_matrix(unitary);
    }
  }
  else{   //multi-chunk distribution
    uint_t local_offset = BaseState::global_chunk_index_ << BaseState::chunk_bits_;

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<BaseState::num_local_chunks_;iChunk++){
      //copy part of state for this chunk
      uint_t i;
      cvector_t tmp(1ull << BaseState::chunk_bits_);
      for(i=0;i<(1ull << BaseState::chunk_bits_);i++){
        tmp[i] = unitary[local_offset + (iChunk << BaseState::chunk_bits_) + i];
      }

      BaseState::qregs_[iChunk].set_num_qubits(BaseState::chunk_bits_/2);
      BaseState::qregs_[iChunk].initialize_from_vector(tmp);
    }
  }

  apply_global_phase();
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::initialize_omp() 
{
  uint_t i;
  for(i=0;i<BaseState::num_local_chunks_;i++){
    BaseState::qregs_[i].set_omp_threshold(omp_qubit_threshold_);
    if (BaseState::threads_ > 0)
      BaseState::qregs_[i].set_omp_threads(BaseState::threads_); // set allowed OMP threads in qubitvector
  }
}

template <class unitary_matrix_t>
auto State<unitary_matrix_t>::move_to_matrix()
{
  if(BaseState::num_global_chunks_ == 1){
    return BaseState::qregs_[0].move_to_matrix();
  }
  else{
    int_t iChunk;
    auto state = BaseState::qregs_[0].vector();   //using vector to gather distributed matrix

    //TO DO check memory availability
    state.resize(BaseState::num_local_chunks_ << BaseState::chunk_bits_);

#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(iChunk)
    for(iChunk=1;iChunk<BaseState::num_local_chunks_;iChunk++){
      auto tmp = BaseState::qregs_[iChunk].vector();
      uint_t j,offset = iChunk << BaseState::chunk_bits_;
      for(j=0;j<tmp.size();j++){
        state[offset + j] = tmp[j];
      }
    }

#ifdef AER_MPI
    BaseState::gather_state(state);
#endif

    //type of matrix cam not be discovered from State class, so make from matrix
    auto matrix = BaseState::qregs_[0].move_to_matrix();
    matrix.resize(1ull << (BaseState::num_qubits_/2),1ull << (BaseState::num_qubits_/2));
    matrix.copy_from_buffer(1ull << (BaseState::num_qubits_/2),1ull << (BaseState::num_qubits_/2),&state[0]);
    return matrix;
  }
}

//=========================================================================
// Implementation: Gates
//=========================================================================

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate(const uint_t iChunk,const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = QubitUnitary::State<unitary_matrix_t>::gateset_.find(op.name);
  if (it == QubitUnitary::State<unitary_matrix_t>::gateset_.end())
    throw std::invalid_argument("Unitary::State::invalid gate instruction \'" +
                                op.name + "\'.");
  QubitUnitary::Gates g = it->second;
  switch (g) {
    case QubitUnitary::Gates::mcx:
      // Includes X, CX, CCX, etc
      BaseState::qregs_[iChunk].apply_mcx(op.qubits);
      break;
    case QubitUnitary::Gates::mcy:
      // Includes Y, CY, CCY, etc
      BaseState::qregs_[iChunk].apply_mcy(op.qubits);
      break;
    case QubitUnitary::Gates::mcz:
      // Includes Z, CZ, CCZ, etc
      BaseState::qregs_[iChunk].apply_mcphase(op.qubits, -1);
      break;
    case QubitUnitary::Gates::mcr:
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::r(op.params[0], op.params[1]));
      break;
    case QubitUnitary::Gates::mcrx:
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::rx(op.params[0]));
      break;
    case QubitUnitary::Gates::mcry:
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::ry(op.params[0]));
      break;
    case QubitUnitary::Gates::mcrz:
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::rz(op.params[0]));
      break;
    case QubitUnitary::Gates::rxx:
      BaseState::qregs_[iChunk].apply_matrix(op.qubits, Linalg::VMatrix::rxx(op.params[0]));
      break;
    case QubitUnitary::Gates::ryy:
      BaseState::qregs_[iChunk].apply_matrix(op.qubits, Linalg::VMatrix::ryy(op.params[0]));
      break;
    case QubitUnitary::Gates::rzz:
      BaseState::qregs_[iChunk].apply_diagonal_matrix(op.qubits, Linalg::VMatrix::rzz_diag(op.params[0]));
      break;
    case QubitUnitary::Gates::rzx:
      BaseState::qregs_[iChunk].apply_matrix(op.qubits, Linalg::VMatrix::rzx(op.params[0]));
      break;
    case QubitUnitary::Gates::id:
      break;
    case QubitUnitary::Gates::h:
      apply_gate_mcu3(iChunk,op.qubits, M_PI / 2., 0., M_PI);
      break;
    case QubitUnitary::Gates::s:
      apply_gate_phase(iChunk,op.qubits[0], complex_t(0., 1.));
      break;
    case QubitUnitary::Gates::sdg:
      apply_gate_phase(iChunk,op.qubits[0], complex_t(0., -1.));
      break;
    case QubitUnitary::Gates::pauli:
        BaseState::qregs_[iChunk].apply_pauli(op.qubits, op.string_params[0]);
        break;
    case QubitUnitary::Gates::t: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_gate_phase(iChunk,op.qubits[0], complex_t(isqrt2, isqrt2));
    } break;
    case QubitUnitary::Gates::tdg: {
      const double isqrt2{1. / std::sqrt(2)};
      apply_gate_phase(iChunk,op.qubits[0], complex_t(isqrt2, -isqrt2));
    } break;
    case QubitUnitary::Gates::mcswap:
      // Includes SWAP, CSWAP, etc
      BaseState::qregs_[iChunk].apply_mcswap(op.qubits);
      break;
    case QubitUnitary::Gates::mcu3:
      // Includes u3, cu3, etc
      apply_gate_mcu3(iChunk,op.qubits, std::real(op.params[0]), std::real(op.params[1]),
                      std::real(op.params[2]));
      break;
    case QubitUnitary::Gates::mcu2:
      // Includes u2, cu2, etc
      apply_gate_mcu3(iChunk,op.qubits, M_PI / 2., std::real(op.params[0]),
                      std::real(op.params[1]));
      break;
    case QubitUnitary::Gates::mcp:
      // Includes u1, cu1, p, cp, mcp, etc
      BaseState::qregs_[iChunk].apply_mcphase(op.qubits,
                                     std::exp(complex_t(0, 1) * op.params[0]));
      break;
    case QubitUnitary::Gates::mcsx:
      // Includes sx, csx, mcsx etc
      BaseState::qregs_[iChunk].apply_mcu(op.qubits, Linalg::VMatrix::SX);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument("Unitary::State::invalid gate instruction \'" +
                                  op.name + "\'.");
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_matrix(const uint_t iChunk,const reg_t &qubits,
                                           const cmatrix_t &mat) {
  if (qubits.empty() == false && mat.size() > 0) {
    apply_matrix(iChunk,qubits, Utils::vectorize_matrix(mat));
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_matrix(const uint_t iChunk,const reg_t &qubits,
                                           const cvector_t &vmat) {
  // Check if diagonal matrix
  if (vmat.size() == 1ULL << qubits.size()) {
    BaseState::qregs_[iChunk].apply_diagonal_matrix(qubits, vmat);
  } else {
    BaseState::qregs_[iChunk].apply_matrix(qubits, vmat);
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_phase(const uint_t iChunk,uint_t qubit, complex_t phase) {
  cmatrix_t diag(1, 2);
  diag(0, 0) = 1.0;
  diag(0, 1) = phase;
  apply_matrix(iChunk,reg_t({qubit}), diag);
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_gate_mcu3(const uint_t iChunk,const reg_t &qubits, double theta,
                                              double phi, double lambda) {
  const auto u3 = Linalg::Matrix::u3(theta, phi, lambda);
  BaseState::qregs_[iChunk].apply_mcu(qubits, Utils::vectorize_matrix(u3));
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_snapshot(const Operations::Op &op,
                                             ExperimentResult &result) {
  // Look for snapshot type in snapshotset
  if (op.name == "unitary" || op.name == "state") {
    auto matrix = move_to_matrix();
    if(BaseState::myrank_ == 0){
      result.legacy_data.add_pershot_snapshot("unitary", op.string_params[0],
                              matrix);
    }
  } else {
    throw std::invalid_argument(
        "Unitary::State::invalid snapshot instruction \'" + op.name + "\'.");
  }
}

template <class unitary_matrix_t>
void State<unitary_matrix_t>::apply_global_phase() {
  if (BaseState::has_global_phase_) {
    int_t i;
#pragma omp parallel for if(BaseState::chunk_omp_parallel_) private(i) 
    for(i=0;i<BaseState::num_local_chunks_;i++){
      BaseState::qregs_[i].apply_diagonal_matrix(
        {0}, {BaseState::global_phase_, BaseState::global_phase_}
      );
    }
  }
}

template <class unitary_matrix_t>
double  State<unitary_matrix_t>::expval_pauli(const reg_t &qubits,
                                              const std::string& pauli) {
  throw std::runtime_error("Unitary simulator does not support Pauli expectation values.");
}

//------------------------------------------------------------------------------
} // namespace QubitUnitary
} // end namespace AER
//------------------------------------------------------------------------------
#endif
