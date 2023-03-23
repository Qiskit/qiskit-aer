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

#ifndef _densitymatrix_batch_executor_hpp_
#define _densitymatrix_batch_executor_hpp_

#include "simulators/batch_shots_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace DensityMatrix {

//-------------------------------------------------------------------------
// batched-shots executor for density matrix
//-------------------------------------------------------------------------
template <class state_t>
class BatchShotsExecutor : public Executor::BatchShotsExecutor<state_t> {
  using BaseExecutor = Executor::BatchShotsExecutor<state_t>;

protected:
public:
  BatchShotsExecutor() {}
  virtual ~BatchShotsExecutor() {}

protected:
  void set_config(const json_t &config) override;

  uint_t qubit_scale(void) override { return 2; }

  // apply op to multiple shots , return flase if op is not supported to execute
  // in a batch
  bool apply_batched_op(const int_t istate, const Operations::Op &op,
                        ExperimentResult &result, std::vector<RngEngine> &rng,
                        bool final_op = false) override;
};

template <class state_t>
void BatchShotsExecutor<state_t>::set_config(const json_t &config) {
  BaseExecutor::set_config(config);
}

template <class state_t>
bool BatchShotsExecutor<state_t>::apply_batched_op(const int_t istate,
                                                   const Operations::Op &op,
                                                   ExperimentResult &result,
                                                   std::vector<RngEngine> &rng,
                                                   bool final_op) {
  if (op.conditional) {
    BaseExecutor::states_[istate].qreg().set_conditional(op.conditional_reg);
  }

  switch (op.type) {
  case Operations::OpType::barrier:
  case Operations::OpType::nop:
  case Operations::OpType::qerror_loc:
    break;
  case Operations::OpType::reset:
    BaseExecutor::states_[istate].qreg().apply_reset(op.qubits);
    break;
  case Operations::OpType::measure:
    BaseExecutor::states_[istate].qreg().apply_batched_measure(
        op.qubits, rng, op.memory, op.registers);
    break;
  case Operations::OpType::bfunc:
    BaseExecutor::states_[istate].qreg().apply_bfunc(op);
    break;
  case Operations::OpType::roerror:
    BaseExecutor::states_[istate].qreg().apply_roerror(op, rng);
    break;
  case Operations::OpType::gate:
    BaseExecutor::states_[istate].apply_gate(op);
    break;
  case Operations::OpType::matrix:
    BaseExecutor::states_[istate].apply_matrix(op.qubits, op.mats[0]);
    break;
  case Operations::OpType::diagonal_matrix:
    BaseExecutor::states_[istate].apply_diagonal_unitary_matrix(op.qubits,
                                                                op.params);
    break;
  case Operations::OpType::superop:
    BaseExecutor::states_[istate].qreg().apply_superop_matrix(
        op.qubits, Utils::vectorize_matrix(op.mats[0]));
    break;
  case Operations::OpType::kraus:
    BaseExecutor::states_[istate].apply_kraus(op.qubits, op.mats);
    break;
  default:
    // other operations should be called to indivisual chunks by apply_op
    return false;
  }
  return true;
}

//-------------------------------------------------------------------------
} // namespace DensityMatrix
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
