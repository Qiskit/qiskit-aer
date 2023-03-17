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

#ifndef _statevector_multi_shots_executor_hpp_
#define _statevector_multi_shots_executor_hpp_

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
//Batched-shots executor for statevector
//-------------------------------------------------------------------------
template <class statevec_t>
class MultiShotsExecutor : public Executor::BatchShotsExecutor<statevec_t> {
  using BaseExecutor = Executor::BatchShotsExecutor<statevec_t>;
protected:
public:
  MultiShotsExecutor(){}
  virtual ~MultiShotsExecutor(){}

protected:
  void set_config(const json_t &config) override;

  void apply_global_phase() override;

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
  void apply_reset(Executor::Branch& root, const reg_t &qubits);
  void apply_initialize(Executor::Branch& root,
                                         const reg_t &qubits,
                                         const cvector_t &params);
  void apply_kraus(Executor::Branch& root,
                                    const reg_t &qubits, const std::vector<cmatrix_t> &kmats);

  std::vector<reg_t> sample_measure(statevec_t& state, const reg_t &qubits,
                                    uint_t shots, std::vector<RngEngine> &rng) const override;

  void apply_save_statevector(Executor::Branch& root, const Operations::Op &op,
                                               ExperimentResult &result,
                                               bool last_op);
  void apply_save_statevector_dict(Executor::Branch& root, const Operations::Op &op,
                                                   ExperimentResult &result);
  void apply_save_amplitudes(Executor::Branch& root, const Operations::Op &op,
                                              ExperimentResult &result);

};


template <class statevec_t>
void MultiShotsExecutor<statevec_t>::set_config(const json_t &config) 
{
  BaseExecutor::set_config(config);
}

template <class statevec_t>
void MultiShotsExecutor<statevec_t>::apply_global_phase() 
{
  if (BaseExecutor::has_global_phase_) {
    int_t i;
    if(BaseExecutor::shot_omp_parallel_ && BaseExecutor::num_groups_ > 0){
#pragma omp parallel for 
      for(int_t ig=0;ig<BaseExecutor::num_groups_;ig++){
        for(int_t iChunk = BaseExecutor::top_shot_of_group_[ig];iChunk < BaseExecutor::top_shot_of_group_[ig + 1];iChunk++)
        BaseExecutor::states_[iChunk].apply_diagonal_matrix({0}, {BaseExecutor::global_phase_, BaseExecutor::global_phase_});
      }
    }
    else{
      for(i=0;i<BaseExecutor::states_.size();i++)
        BaseExecutor::states_[i].apply_diagonal_matrix({0}, {BaseExecutor::global_phase_, BaseExecutor::global_phase_});
    }
  }
}

template <class statevec_t>
bool MultiShotsExecutor<statevec_t>::apply_batched_op(const int_t istate, 
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
      BaseExecutor::states_[istate].qreg().apply_batched_reset(op.qubits,rng);
      break;
    case Operations::OpType::initialize:
      BaseExecutor::states_[istate].qreg().apply_batched_reset(op.qubits,rng);
      BaseExecutor::states_[istate].qreg().initialize_component(op.qubits, op.params);
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
      BaseExecutor::states_[istate].apply_matrix(op);
      break;
    case Operations::OpType::diagonal_matrix:
      BaseExecutor::states_[istate].qreg().apply_diagonal_matrix(op.qubits, op.params);
      break;
    case Operations::OpType::multiplexer:
      BaseExecutor::states_[istate].apply_multiplexer(op.regs[0], op.regs[1],
                        op.mats); // control qubits ([0]) & target qubits([1])
      break;
    case Operations::OpType::kraus:
      BaseExecutor::states_[istate].qreg().apply_batched_kraus(op.qubits, op.mats,rng);
      break;
    case Operations::OpType::sim_op:
      if(op.name == "begin_register_blocking"){
        BaseExecutor::states_[istate].qreg().enter_register_blocking(op.qubits);
      }
      else if(op.name == "end_register_blocking"){
        BaseExecutor::states_[istate].qreg().leave_register_blocking();
      }
      else{
        return false;
      }
      break;
    case Operations::OpType::set_statevec:
      BaseExecutor::states_[istate].qreg().initialize_from_vector(op.params);
      break;
    default:
      //other operations should be called to indivisual chunks by apply_op
      return false;
  }
  return true;
}

template <class statevec_t>
bool MultiShotsExecutor<statevec_t>::apply_branching_op(
                                  Executor::Branch& root,
                                  const Operations::Op &op,
                                  ExperimentResult &result,
                                  bool final_op) 
{
  RngEngine dummy;
  if(BaseExecutor::states_[root.state_index()].creg().check_conditional(op)){
    switch (op.type) {
      //ops with branching
      case Operations::OpType::reset:
        apply_reset(root, op.qubits);
        break;
      case Operations::OpType::initialize:
        apply_initialize(root, op.qubits, op.params);
        break;
      case Operations::OpType::measure:
        apply_measure(root, op.qubits, op.memory, op.registers);
        break;
      case Operations::OpType::kraus:
        apply_kraus(root, op.qubits, op.mats);
        break;
      //save ops
      case Operations::OpType::save_expval:
      case Operations::OpType::save_expval_var:
      case Operations::OpType::save_densmat:
      case Operations::OpType::save_probs:
      case Operations::OpType::save_probs_ket:
        //call save functions in state class
        BaseExecutor::states_[root.state_index()].apply_op(op, result, dummy, final_op);
        break;
      case Operations::OpType::save_state:
      case Operations::OpType::save_statevec:
        apply_save_statevector(root, op, result, final_op);
        break;
      case Operations::OpType::save_statevec_dict:
        apply_save_statevector_dict(root, op, result);
        break;
      case Operations::OpType::save_amps:
      case Operations::OpType::save_amps_sq:
        apply_save_amplitudes(root, op, result);
        break;
      default:
        return false;
    }
  }
  return true;
}

template <class statevec_t>
rvector_t MultiShotsExecutor<statevec_t>::sample_measure_with_prob(Executor::Branch& root, const reg_t &qubits)
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

template <class statevec_t>
void MultiShotsExecutor<statevec_t>::measure_reset_update(
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
        op.name = "mcx";
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

template <class statevec_t>
void MultiShotsExecutor<statevec_t>::apply_measure(
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

template <class statevec_t>
void MultiShotsExecutor<statevec_t>::apply_reset(Executor::Branch& root, const reg_t &qubits) 
{
  rvector_t probs = sample_measure_with_prob(root, qubits);

  measure_reset_update(root, qubits, 0, probs);
}

template <class statevec_t>
void MultiShotsExecutor<statevec_t>::apply_initialize(Executor::Branch& root, 
                                         const reg_t &qubits,
                                         const cvector_t &params)
{
  if (qubits.size() == BaseExecutor::num_qubits_) {
    auto sorted_qubits = qubits;
    std::sort(sorted_qubits.begin(), sorted_qubits.end());
    // If qubits is all ordered qubits in the statevector
    // we can just initialize the whole state directly
    if (qubits == sorted_qubits) {
      BaseExecutor::states_[root.state_index()].initialize_from_vector(params);
      return;
    }
  }

  if(root.additional_ops().size() == 0){
    apply_reset(root, qubits);

    Operations::Op op;
    op.type = OpType::initialize;
    op.name = "initialize";
    op.qubits = qubits;
    op.params = params;
    for(int_t i=0;i<root.num_branches();i++){
      root.branches()[i]->add_op_after_branch(op);
    }
    return; //initialization will be done in next call because of shot branching in reset
  }

  BaseExecutor::states_[root.state_index()].qreg().initialize_component(qubits, params);
}

template <class statevec_t>
void MultiShotsExecutor<statevec_t>::apply_kraus(
                                    Executor::Branch& root,
                                    const reg_t &qubits, const std::vector<cmatrix_t> &kmats)
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

  nshots = root.num_shots();
  shot_branch.resize(nshots);
  rshots.resize(nshots);
  for(int_t i=0;i<nshots;i++){
    shot_branch[i] = kmats.size() - 1;
    rshots[i] = root.rng_shots()[i].rand(0., 1.);
  }
  pmats.resize(kmats.size());

  // Loop through N-1 kraus operators
  for (size_t j = 0; j < kmats.size() - 1; j++) {
    // Calculate probability
    cvector_t vmat = Utils::vectorize_matrix(kmats[j]);

    p = BaseExecutor::states_[root.state_index()].qreg().norm(qubits, vmat);
    accum += p;

    // check if we need to apply this operator
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

  // check if we haven't applied a kraus operator yet
  pmats[pmats.size()-1] = 1. - accum;

  root.creg() = BaseExecutor::states_[root.state_index()].creg();
  root.branch_shots(shot_branch, kmats.size());
  for(int_t i=0;i<kmats.size();i++){
    Operations::Op op;
    op.type = OpType::matrix;
    op.qubits = qubits;
    op.mats.push_back(kmats[i]);
    p = 1/std::sqrt(pmats[i]);
    for(int_t j=0;j<op.mats[0].size();j++)
      op.mats[0][j] *= p;
    root.branches()[i]->add_op_after_branch(op);
  }
}

template <class statevec_t>
void MultiShotsExecutor<statevec_t>::apply_save_statevector(Executor::Branch& root, const Operations::Op &op,
                                               ExperimentResult &result,
                                               bool last_op) 
{
  if (op.qubits.size() != BaseExecutor::num_qubits_) {
    throw std::invalid_argument(
        op.name + " was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  std::string key = (op.string_params[0] == "_method_")
                      ? "statevector"
                      : op.string_params[0];

  if (last_op) {
    auto v = BaseExecutor::states_[root.state_index()].move_to_vector();
    result.save_data_pershot(BaseExecutor::states_[root.state_index()].creg(), key, std::move(v),
                                  OpType::save_statevec, op.save_type, root.num_shots());
  } else {
    auto v = BaseExecutor::states_[root.state_index()].copy_to_vector();
    result.save_data_pershot(BaseExecutor::states_[root.state_index()].creg(), key, v,
                                OpType::save_statevec, op.save_type, root.num_shots());
  }
}

template <class statevec_t>
void MultiShotsExecutor<statevec_t>::apply_save_statevector_dict(Executor::Branch& root, const Operations::Op &op,
                                                   ExperimentResult &result) 
{
  if (op.qubits.size() != BaseExecutor::num_qubits_) {
    throw std::invalid_argument(
        op.name + " was not applied to all qubits."
        " Only the full statevector can be saved.");
  }
  auto state_ket = BaseExecutor::states_[root.state_index()].qreg().vector_ket(BaseExecutor::json_chop_threshold_);
  std::map<std::string, complex_t> result_state_ket;
  for (auto const& it : state_ket){
    result_state_ket[it.first] = it.second;
  }
  result.save_data_pershot(BaseExecutor::states_[root.state_index()].creg(), op.string_params[0],
                               result_state_ket, op.type, op.save_type, root.num_shots());
}

template <class statevec_t>
void MultiShotsExecutor<statevec_t>::apply_save_amplitudes(Executor::Branch& root, const Operations::Op &op,
                                              ExperimentResult &result) 
{
  if (op.int_params.empty()) {
    throw std::invalid_argument("Invalid save_amplitudes instructions (empty params).");
  }
  const int_t size = op.int_params.size();
  if (op.type == Operations::OpType::save_amps) {
    Vector<complex_t> amps(size, false);
    for (int_t i = 0; i < size; ++i) {
      amps[i] = BaseExecutor::states_[root.state_index()].qreg().get_state(op.int_params[i]);
    }
    result.save_data_pershot(BaseExecutor::states_[root.state_index()].creg(), op.string_params[0],
                               amps, op.type, op.save_type, root.num_shots());
  }
  else{
    rvector_t amps_sq(size,0);
    for (int_t i = 0; i < size; ++i) {
      amps_sq[i] = BaseExecutor::states_[root.state_index()].qreg().probability(op.int_params[i]);
    }
    result.save_data_average(BaseExecutor::states_[root.state_index()].creg(), op.string_params[0],
                          amps_sq, op.type, op.save_type);
  }
}

template <class statevec_t>
std::vector<reg_t> MultiShotsExecutor<statevec_t>::sample_measure(statevec_t& state, const reg_t &qubits,
                                            uint_t shots, std::vector<RngEngine> &rng) const
{
  int_t i,j;
  std::vector<double> rnds;
  rnds.reserve(shots);

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
} // end namespace Statevector
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif



