/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019. 2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _unitary_parallel_executor_hpp
#define _unitary_parallel_executor_hpp

#include "simulators/parallel_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace QubitUnitary {

//-------------------------------------------------------------------------
// Parallel executor for QubitUnitar
//-------------------------------------------------------------------------

template <class unitary_matrix_t>
class ParallelExecutor : public Executor::ParallelExecutor<unitary_matrix_t> {
  using BaseExecutor = Executor::ParallelExecutor<unitary_matrix_t>;

protected:
public:
  ParallelExecutor() {}
  virtual ~ParallelExecutor() {}

  auto move_to_matrix(void);
  auto copy_to_matrix(void);

protected:
  void set_config(const Config &config) override;

  // apply parallel operations
  void apply_parallel_op(const Operations::Op &op, ExperimentResult &result,
                         RngEngine &rng, bool final_op) override;

  void initialize_qreg(uint_t num_qubits) override;

  //-----------------------------------------------------------------------
  // Apply Instructions
  //-----------------------------------------------------------------------
  // swap between chunks
  void apply_chunk_swap(const reg_t &qubits) override;

  //-----------------------------------------------------------------------
  // Save data instructions
  //-----------------------------------------------------------------------

  // Save the unitary matrix for the simulator
  void apply_save_unitary(const Operations::Op &op, ExperimentResult &result,
                          bool last_op);

  // Helper function for computing expectation value
  double expval_pauli(const reg_t &qubits, const std::string &pauli) override;

  // scale for unitary = 2
  // this function is used in the base class to scale chunk qubits for
  // multi-chunk distribution
  uint_t qubit_scale(void) override { return 2; }
};

template <class unitary_matrix_t>
void ParallelExecutor<unitary_matrix_t>::set_config(const Config &config) {
  BaseExecutor::set_config(config);
}

template <class unitary_matrix_t>
void ParallelExecutor<unitary_matrix_t>::initialize_qreg(uint_t num_qubits) {
  int_t iChunk;
  for (iChunk = 0; iChunk < BaseExecutor::states_.size(); iChunk++) {
    BaseExecutor::states_[iChunk].qreg().set_num_qubits(
        BaseExecutor::chunk_bits_);
  }

  if (BaseExecutor::chunk_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for private(iChunk)
    for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
      for (iChunk = BaseExecutor::top_state_of_group_[ig];
           iChunk < BaseExecutor::top_state_of_group_[ig + 1]; iChunk++) {
        uint_t irow, icol;
        irow = (BaseExecutor::global_state_index_ + iChunk) >>
               ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));
        icol =
            (BaseExecutor::global_state_index_ + iChunk) -
            (irow << ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)));
        if (irow == icol)
          BaseExecutor::states_[iChunk].qreg().initialize();
        else
          BaseExecutor::states_[iChunk].qreg().zero();
      }
    }
  } else {
    for (iChunk = 0; iChunk < BaseExecutor::states_.size(); iChunk++) {
      uint_t irow, icol;
      irow = (BaseExecutor::global_state_index_ + iChunk) >>
             ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_));
      icol =
          (BaseExecutor::global_state_index_ + iChunk) -
          (irow << ((BaseExecutor::num_qubits_ - BaseExecutor::chunk_bits_)));
      if (irow == icol)
        BaseExecutor::states_[iChunk].qreg().initialize();
      else
        BaseExecutor::states_[iChunk].qreg().zero();
    }
  }

  BaseExecutor::apply_global_phase();
}

template <class unitary_matrix_t>
void ParallelExecutor<unitary_matrix_t>::apply_parallel_op(
    const Operations::Op &op, ExperimentResult &result, RngEngine &rng,
    bool final_op) {
  // temporary : this is for statevector
  if (BaseExecutor::states_[0].creg().check_conditional(op)) {
    switch (op.type) {
    case Operations::OpType::barrier:
    case Operations::OpType::nop:
    case Operations::OpType::qerror_loc:
      break;
    case Operations::OpType::bfunc:
      BaseExecutor::states_[0].creg().apply_bfunc(op);
      break;
    case Operations::OpType::roerror:
      BaseExecutor::states_[0].creg().apply_roerror(op, rng);
      break;
    case Operations::OpType::set_unitary:
      BaseExecutor::initialize_from_matrix(op.mats[0]);
      break;
    case Operations::OpType::save_state:
    case Operations::OpType::save_unitary:
      apply_save_unitary(op, result, final_op);
      break;
    default:
      throw std::invalid_argument("ParallelExecutor::invalid instruction \'" +
                                  op.name + "\'.");
    }
  }
}

template <class unitary_matrix_t>
auto ParallelExecutor<unitary_matrix_t>::move_to_matrix(void) {
  return BaseExecutor::apply_to_matrix(false);
}

template <class unitary_matrix_t>
auto ParallelExecutor<unitary_matrix_t>::copy_to_matrix(void) {
  return BaseExecutor::apply_to_matrix(true);
}

template <class unitary_matrix_t>
void ParallelExecutor<unitary_matrix_t>::apply_save_unitary(
    const Operations::Op &op, ExperimentResult &result, bool last_op) {
  if (op.qubits.size() != BaseExecutor::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full unitary can be saved.");
  }
  std::string key =
      (op.string_params[0] == "_method_") ? "unitary" : op.string_params[0];

  if (last_op) {
    result.save_data_pershot(BaseExecutor::states_[0].creg(), key,
                             move_to_matrix(), Operations::OpType::save_unitary,
                             op.save_type);
  } else {
    result.save_data_pershot(BaseExecutor::states_[0].creg(), key,
                             copy_to_matrix(), Operations::OpType::save_unitary,
                             op.save_type);
  }
}

template <class unitary_matrix_t>
double
ParallelExecutor<unitary_matrix_t>::expval_pauli(const reg_t &qubits,
                                                 const std::string &pauli) {
  throw std::runtime_error(
      "Unitary simulator does not support Pauli expectation values.");
}

// swap between chunks
template <class unitary_matrix_t>
void ParallelExecutor<unitary_matrix_t>::apply_chunk_swap(const reg_t &qubits) {
  uint_t q0, q1;
  q0 = qubits[0];
  q1 = qubits[1];

  std::swap(BaseExecutor::qubit_map_[q0], BaseExecutor::qubit_map_[q1]);

  if (qubits[0] >= BaseExecutor::chunk_bits_) {
    q0 += BaseExecutor::chunk_bits_;
  }
  if (qubits[1] >= BaseExecutor::chunk_bits_) {
    q1 += BaseExecutor::chunk_bits_;
  }
  reg_t qs0 = {{q0, q1}};
  BaseExecutor::apply_chunk_swap(qs0);
}

//------------------------------------------------------------------------------
} // namespace QubitUnitary
} // end namespace AER
//------------------------------------------------------------------------------
#endif
