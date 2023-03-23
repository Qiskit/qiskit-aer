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

#ifndef _statevector_batch_executor_hpp_
#define _statevector_batch_executor_hpp_

#include "simulators/batch_shots_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace Statevector {

//-------------------------------------------------------------------------
// Batched-shots executor for statevector
//-------------------------------------------------------------------------
template <class statevec_t>
class BatchShotsExecutor : public Executor::BatchShotsExecutor<statevec_t> {
  using BaseExecutor = Executor::BatchShotsExecutor<statevec_t>;

protected:
public:
  BatchShotsExecutor() {}
  virtual ~BatchShotsExecutor() {}

protected:
  void set_config(const json_t &config) override;

  void apply_global_phase() override;

  // apply op to multiple shots , return flase if op is not supported to execute
  // in a batch
  bool apply_batched_op(const int_t istate, const Operations::Op &op,
                        ExperimentResult &result, std::vector<RngEngine> &rng,
                        bool final_op = false) override;
};

template <class statevec_t>
void BatchShotsExecutor<statevec_t>::set_config(const json_t &config) {
  BaseExecutor::set_config(config);
}

template <class statevec_t>
void BatchShotsExecutor<statevec_t>::apply_global_phase() {
  if (BaseExecutor::has_global_phase_) {
    int_t i;
    if (BaseExecutor::shot_omp_parallel_ && BaseExecutor::num_groups_ > 0) {
#pragma omp parallel for
      for (int_t ig = 0; ig < BaseExecutor::num_groups_; ig++) {
        for (int_t iChunk = BaseExecutor::top_shot_of_group_[ig];
             iChunk < BaseExecutor::top_shot_of_group_[ig + 1]; iChunk++)
          BaseExecutor::states_[iChunk].apply_diagonal_matrix(
              {0}, {BaseExecutor::global_phase_, BaseExecutor::global_phase_});
      }
    } else {
      for (i = 0; i < BaseExecutor::states_.size(); i++)
        BaseExecutor::states_[i].apply_diagonal_matrix(
            {0}, {BaseExecutor::global_phase_, BaseExecutor::global_phase_});
    }
  }
}

template <class statevec_t>
bool BatchShotsExecutor<statevec_t>::apply_batched_op(
    const int_t istate, const Operations::Op &op, ExperimentResult &result,
    std::vector<RngEngine> &rng, bool final_op) {
  if (op.conditional) {
    BaseExecutor::states_[istate].qreg().set_conditional(op.conditional_reg);
  }

  switch (op.type) {
  case Operations::OpType::barrier:
  case Operations::OpType::nop:
  case Operations::OpType::qerror_loc:
    break;
  case Operations::OpType::reset:
    BaseExecutor::states_[istate].qreg().apply_batched_reset(op.qubits, rng);
    break;
  case Operations::OpType::initialize:
    BaseExecutor::states_[istate].qreg().apply_batched_reset(op.qubits, rng);
    BaseExecutor::states_[istate].qreg().initialize_component(op.qubits,
                                                              op.params);
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
    BaseExecutor::states_[istate].apply_matrix(op);
    break;
  case Operations::OpType::diagonal_matrix:
    BaseExecutor::states_[istate].qreg().apply_diagonal_matrix(op.qubits,
                                                               op.params);
    break;
  case Operations::OpType::multiplexer:
    BaseExecutor::states_[istate].apply_multiplexer(
        op.regs[0], op.regs[1],
        op.mats); // control qubits ([0]) & target qubits([1])
    break;
  case Operations::OpType::kraus:
    BaseExecutor::states_[istate].qreg().apply_batched_kraus(op.qubits, op.mats,
                                                             rng);
    break;
  case Operations::OpType::sim_op:
    if (op.name == "begin_register_blocking") {
      BaseExecutor::states_[istate].qreg().enter_register_blocking(op.qubits);
    } else if (op.name == "end_register_blocking") {
      BaseExecutor::states_[istate].qreg().leave_register_blocking();
    } else {
      return false;
    }
    break;
  case Operations::OpType::set_statevec:
    BaseExecutor::states_[istate].qreg().initialize_from_vector(op.params);
    break;
  default:
    // other operations should be called to indivisual chunks by apply_op
    return false;
  }
  return true;
}

//-------------------------------------------------------------------------
} // end namespace Statevector
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
