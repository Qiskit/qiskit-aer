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

#ifndef _densitymatrix_multi_shots_executor_hpp_
#define _densitymatrix_multi_shots_executor_hpp_

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
//batched-shots executor for density matrix
//-------------------------------------------------------------------------
template <class state_t>
class MultiShotsExecutor : public Executor::BatchShotsExecutor<state_t> {
  using BaseExecutor = Executor::BatchShotsExecutor<state_t>;
protected:
public:
  MultiShotsExecutor(){}
  virtual ~MultiShotsExecutor(){}

protected:
  void set_config(const Config &config) override;

  uint_t qubit_scale(void) override
  {
    return 2;
  }
  bool shot_branching_supported(void) override
  {
    return true;
  }

  //apply op to multiple shots , return flase if op is not supported to execute in a batch
  bool apply_batched_op(const int_t istate, const Operations::Op &op,
                                ExperimentResult &result,
                                std::vector<RngEngine> &rng,
                                bool final_op = false) override;

  bool apply_branching_op(Executor::Branch& root,
                                  const Operations::Op &op,
                                  ExperimentResult &result,
                                  bool final_op) override;

  rvector_t sample_measure_with_prob(Executor::Branch& root, const reg_t &qubits);
  void measure_reset_update(Executor::Branch& root, 
                             const std::vector<uint_t> &qubits,
                             const int_t final_state,
                             const rvector_t& meas_probs);
  void apply_measure(Executor::Branch& root, 
                                      const reg_t &qubits, const reg_t &cmemory,
                                      const reg_t &cregister);

  std::vector<reg_t> sample_measure(state_t& state, const reg_t &qubits,
                                    uint_t shots, std::vector<RngEngine> &rng) const override;

};


template <class state_t>
void MultiShotsExecutor<state_t>::set_config(const Config &config) 
{
  BaseExecutor::set_config(config);
}

template <class state_t>
bool MultiShotsExecutor<state_t>::apply_batched_op(const int_t istate, 
                                  const Operations::Op &op,
                                  ExperimentResult &result,
                                  std::vector<RngEngine> &rng,
                                  bool final_op) 
{
  if(op.conditional){
    BaseExecutor::states_[istate].qreg().set_conditional(op.conditional_reg);
  }

  switch (op.type) {
    case Operations::OpType::barrier:
    case Operations::OpType::nop:
    case Operations::OpType::qerror_loc:
      break;
    case Operations::OpType::reset:
      BaseExecutor::states_[istate].apply_reset(op.qubits);
      break;
    case Operations::OpType::measure:
      BaseExecutor::states_[istate].qreg().apply_batched_measure(op.qubits,rng,op.memory,op.registers);
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
      BaseExecutor::states_[istate].apply_diagonal_unitary_matrix(op.qubits, op.params);
      break;
    case Operations::OpType::superop:
      BaseExecutor::states_[istate].qreg().apply_superop_matrix(op.qubits, Utils::vectorize_matrix(op.mats[0]));
      break;
    case Operations::OpType::kraus:
      BaseExecutor::states_[istate].apply_kraus(op.qubits, op.mats);
      break;
    default:
      //other operations should be called to indivisual chunks by apply_op
      return false;
  }
  return true;
}

template <class state_t>
bool MultiShotsExecutor<state_t>::apply_branching_op(
                                  Executor::Branch& root,
                                  const Operations::Op &op,
                                  ExperimentResult &result,
                                  bool final_op) 
{
  RngEngine dummy;
  if(BaseExecutor::states_[root.state_index()].creg().check_conditional(op)){
    switch (op.type) {
      //ops with branching
//      case Operations::OpType::reset:
//        apply_reset(root, op.qubits);
//        break;
      case Operations::OpType::measure:
        apply_measure(root, op.qubits, op.memory, op.registers);
        break;
      //save ops
      case Operations::OpType::save_expval:
      case Operations::OpType::save_expval_var:
      case Operations::OpType::save_state:
      case Operations::OpType::save_densmat:
      case Operations::OpType::save_probs:
      case Operations::OpType::save_probs_ket:
      case Operations::OpType::save_amps_sq:
        //call save functions in state class
        BaseExecutor::states_[root.state_index()].apply_op(op, result, dummy, final_op);
        break;
      default:
        return false;
    }
  }
  return true;
}

template <class state_t>
rvector_t MultiShotsExecutor<state_t>::sample_measure_with_prob(Executor::Branch& root, const reg_t &qubits)
{
  rvector_t probs = BaseExecutor::states_[root.state_index()].qreg().probabilities(qubits);
  uint_t nshots = root.num_shots();
  reg_t shot_branch(nshots);

  for(int_t i=0;i<nshots;i++){
    shot_branch[i] = root.rng_shots()[i].rand_int(probs);
  }

  //branch shots
  root.creg() = BaseExecutor::states_[root.state_index()].creg();
  root.branch_shots(shot_branch, probs.size());

  return probs;
}

template <class state_t>
void MultiShotsExecutor<state_t>::measure_reset_update(
                                             Executor::Branch& root, 
                                             const std::vector<uint_t> &qubits,
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
      root.branches()[i]->add_op_after_branch(op);

      if(final_state >= 0 && final_state != i) {
        Operations::Op op;
        op.type = OpType::gate;
        op.name = "x";
        op.qubits = qubits;
        root.branches()[i]->add_op_after_branch(op);
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
      root.branches()[i]->add_op_after_branch(op);

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
        root.branches()[i]->add_op_after_branch(op);
      }
    }
  }
}

template <class state_t>
void MultiShotsExecutor<state_t>::apply_measure(
                                      Executor::Branch& root,
                                      const reg_t &qubits, const reg_t &cmemory,
                                      const reg_t &cregister)
{
  rvector_t probs = sample_measure_with_prob(root, qubits);

  //save result to cregs
  for(int_t i=0;i<probs.size();i++){
    const reg_t outcome = Utils::int2reg(i, 2, qubits.size());
    root.branches()[i]->creg().store_measure(outcome, cmemory, cregister);
  }

  measure_reset_update(root, qubits, -1, probs);
}
/*
template <class state_t>
void MultiShotsExecutor<state_t>::apply_reset(Executor::Branch& root, const reg_t &qubits) 
{
  rvector_t probs = sample_measure_with_prob(root, qubits);

  measure_reset_update(root, qubits, 0, probs);
}
*/

template <class state_t>
std::vector<reg_t> MultiShotsExecutor<state_t>::sample_measure(state_t& state, const reg_t &qubits,
                                            uint_t shots, std::vector<RngEngine> &rng) const
{
  int_t i,j;
  std::vector<double> rnds;
  rnds.reserve(shots);

  /*
  double norm = std::real( state.qreg().trace() );
  std::cout << "   trace = " << norm<<std::endl;

  for (i = 0; i < shots; ++i)
    rnds.push_back(rng[i].rand(0, norm));
  */

  for (i = 0; i < shots; ++i)
    rnds.push_back(rng[i].rand(0, 1));

  bool flg = state.qreg().enable_batch(false);
  auto allbit_samples = state.qreg().sample_measure(rnds);
  state.qreg().enable_batch(flg);

  // Convert to reg_t format
  std::vector<reg_t> all_samples;
  all_samples.reserve(shots);
  for (int_t val : allbit_samples) {
    reg_t allbit_sample = Utils::int2reg(val, 2, BaseExecutor::num_qubits_);
    reg_t sample;
    sample.reserve(qubits.size());
    for (uint_t qubit : qubits) {
      sample.push_back(allbit_sample[qubit]);
    }
    all_samples.push_back(sample);
  }
  return all_samples;
}

//-------------------------------------------------------------------------
} // end namespace DensityMatrix
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif



