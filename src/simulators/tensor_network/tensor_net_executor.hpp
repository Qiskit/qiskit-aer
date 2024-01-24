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

#ifndef _tensor_network_executor_hpp_
#define _tensor_network_executor_hpp_

#include "simulators/multi_state_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace TensorNetwork {

using ResultItr = std::vector<ExperimentResult>::iterator;

//-------------------------------------------------------------------------
// Batched-shots executor for statevector
//-------------------------------------------------------------------------
template <class state_t>
class Executor : public CircuitExecutor::MultiStateExecutor<state_t> {
  using Base = CircuitExecutor::MultiStateExecutor<state_t>;
  using Base::sample_measure;

protected:
public:
  Executor() {}
  virtual ~Executor() {}

protected:
  void set_config(const Config &config) override;

  bool shot_branching_supported(void) override { return true; }

  bool apply_branching_op(CircuitExecutor::Branch &root,
                          const Operations::Op &op, ResultItr result,
                          bool final_op) override;

  rvector_t sample_measure_with_prob(CircuitExecutor::Branch &root,
                                     const reg_t &qubits);
  void measure_reset_update(CircuitExecutor::Branch &root,
                            const std::vector<uint_t> &qubits,
                            const int_t final_state,
                            const rvector_t &meas_probs);
  void apply_measure(CircuitExecutor::Branch &root, const reg_t &qubits,
                     const reg_t &cmemory, const reg_t &cregister);
  void apply_reset(CircuitExecutor::Branch &root, const reg_t &qubits);
  void apply_initialize(CircuitExecutor::Branch &root, const reg_t &qubits,
                        const cvector_t<double> &params);
  void apply_kraus(CircuitExecutor::Branch &root, const reg_t &qubits,
                   const std::vector<cmatrix_t> &kmats);

  std::vector<reg_t> sample_measure(state_t &state, const reg_t &qubits,
                                    uint_t shots,
                                    std::vector<RngEngine> &rng) const override;

  // Helper functions for shot-branching
  void apply_save_density_matrix(CircuitExecutor::Branch &root,
                                 const Operations::Op &op, ResultItr result);
  void apply_save_probs(CircuitExecutor::Branch &root, const Operations::Op &op,
                        ResultItr result);
  void apply_save_statevector(CircuitExecutor::Branch &root,
                              const Operations::Op &op, ResultItr result,
                              bool last_op);
  void apply_save_statevector_dict(CircuitExecutor::Branch &root,
                                   const Operations::Op &op, ResultItr result);
  void apply_save_amplitudes(CircuitExecutor::Branch &root,
                             const Operations::Op &op, ResultItr result);
};

template <class state_t>
void Executor<state_t>::set_config(const Config &config) {
  Base::set_config(config);
}

template <class state_t>
bool Executor<state_t>::apply_branching_op(CircuitExecutor::Branch &root,
                                           const Operations::Op &op,
                                           ResultItr result, bool final_op) {
  RngEngine dummy;
  if (Base::states_[root.state_index()].creg().check_conditional(op)) {
    switch (op.type) {
    case OpType::reset:
      apply_reset(root, op.qubits);
      break;
    case OpType::initialize:
      apply_initialize(root, op.qubits, op.params);
      break;
    case OpType::measure:
      apply_measure(root, op.qubits, op.memory, op.registers);
      break;
    case OpType::kraus:
      if (!Base::has_statevector_ops_)
        return false;
      apply_kraus(root, op.qubits, op.mats);
      break;
    case OpType::save_expval:
    case OpType::save_expval_var:
      Base::apply_save_expval(root, op, result);
      break;
    case OpType::save_densmat:
      apply_save_density_matrix(root, op, result);
      break;
    case OpType::save_probs:
    case OpType::save_probs_ket:
      apply_save_probs(root, op, result);
      break;
    case OpType::save_state:
    case OpType::save_statevec:
      apply_save_statevector(root, op, result, final_op);
      break;
    case OpType::save_statevec_dict:
      apply_save_statevector_dict(root, op, result);
      break;
    case OpType::save_amps:
    case OpType::save_amps_sq:
      apply_save_amplitudes(root, op, result);
      break;
    default:
      return false;
    }
  }
  return true;
}

template <class state_t>
rvector_t
Executor<state_t>::sample_measure_with_prob(CircuitExecutor::Branch &root,
                                            const reg_t &qubits) {
  rvector_t probs =
      Base::states_[root.state_index()].qreg().probabilities(qubits);
  uint_t nshots = root.num_shots();
  reg_t shot_branch(nshots);

  for (uint_t i = 0; i < nshots; i++) {
    shot_branch[i] = root.rng_shots()[i].rand_int(probs);
  }

  // branch shots
  root.creg() = Base::states_[root.state_index()].creg();
  root.branch_shots(shot_branch, probs.size());

  return probs;
}

template <class state_t>
void Executor<state_t>::measure_reset_update(CircuitExecutor::Branch &root,
                                             const std::vector<uint_t> &qubits,
                                             const int_t final_state,
                                             const rvector_t &meas_probs) {
  // Update a state vector based on an outcome pair [m, p] from
  // sample_measure_with_prob function, and a desired post-measurement
  // final_state

  // Single-qubit case
  if (qubits.size() == 1) {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    for (int_t i = 0; i < 2; i++) {
      cvector_t<double> mdiag(2, 0.);
      mdiag[i] = 1. / std::sqrt(meas_probs[i]);

      Operations::Op op;
      op.type = OpType::diagonal_matrix;
      op.qubits = qubits;
      op.params = mdiag;
      root.branches()[i]->add_op_after_branch(op);

      if (final_state >= 0 && final_state != i) {
        Operations::Op op2;
        op2.type = OpType::gate;
        op2.name = "mcx";
        op2.qubits = qubits;
        root.branches()[i]->add_op_after_branch(op2);
      }
    }
  }
  // Multi qubit case
  else {
    // Diagonal matrix for projecting and renormalizing to measurement outcome
    const size_t dim = 1ULL << qubits.size();
    for (uint_t i = 0; i < dim; i++) {
      cvector_t<double> mdiag(dim, 0.);
      mdiag[i] = 1. / std::sqrt(meas_probs[i]);

      Operations::Op op;
      op.type = OpType::diagonal_matrix;
      op.qubits = qubits;
      op.params = mdiag;
      root.branches()[i]->add_op_after_branch(op);

      if (final_state >= 0 && final_state != (int_t)i) {
        // build vectorized permutation matrix
        cvector_t<double> perm(dim * dim, 0.);
        perm[final_state * dim + i] = 1.;
        perm[i * dim + final_state] = 1.;
        for (size_t j = 0; j < dim; j++) {
          if (j != (size_t)final_state && j != i)
            perm[j * dim + j] = 1.;
        }
        Operations::Op op2;
        op2.type = OpType::matrix;
        op2.qubits = qubits;
        op2.mats.push_back(Utils::devectorize_matrix(perm));
        root.branches()[i]->add_op_after_branch(op2);
      }
    }
  }
}

template <class state_t>
void Executor<state_t>::apply_measure(CircuitExecutor::Branch &root,
                                      const reg_t &qubits, const reg_t &cmemory,
                                      const reg_t &cregister) {
  rvector_t probs = sample_measure_with_prob(root, qubits);

  // save result to cregs
  for (uint_t i = 0; i < probs.size(); i++) {
    const reg_t outcome = Utils::int2reg(i, 2, qubits.size());
    root.branches()[i]->creg().store_measure(outcome, cmemory, cregister);
  }

  measure_reset_update(root, qubits, -1, probs);
}

template <class state_t>
void Executor<state_t>::apply_reset(CircuitExecutor::Branch &root,
                                    const reg_t &qubits) {
  rvector_t probs = sample_measure_with_prob(root, qubits);

  measure_reset_update(root, qubits, 0, probs);
}

template <class state_t>
void Executor<state_t>::apply_initialize(CircuitExecutor::Branch &root,
                                         const reg_t &qubits,
                                         const cvector_t<double> &params_in) {
  // apply global phase here
  cvector_t<double> tmp;
  if (Base::states_[root.state_index()].has_global_phase()) {
    tmp.resize(params_in.size());
    std::complex<double> global_phase =
        Base::states_[root.state_index()].global_phase();
    auto apply_global_phase = [&tmp, params_in, global_phase](int_t i) {
      tmp[i] = params_in[i] * global_phase;
    };
    Utils::apply_omp_parallel_for(
        (qubits.size() > (uint_t)Base::omp_qubit_threshold_), 0,
        params_in.size(), apply_global_phase, Base::parallel_state_update_);
  }
  const cvector_t<double> &params = tmp.empty() ? params_in : tmp;
  if (qubits.size() == Base::num_qubits_) {
    auto sorted_qubits = qubits;
    std::sort(sorted_qubits.begin(), sorted_qubits.end());
    // If qubits is all ordered qubits in the statevector
    // we can just initialize the whole state directly
    if (qubits == sorted_qubits) {
      Base::states_[root.state_index()].initialize_from_vector(params);
      return;
    }
  }

  if (root.additional_ops().size() == 0) {
    apply_reset(root, qubits);

    Operations::Op op;
    op.type = OpType::initialize;
    op.name = "initialize";
    op.qubits = qubits;
    op.params = params;
    for (uint_t i = 0; i < root.num_branches(); i++) {
      root.branches()[i]->add_op_after_branch(op);
    }
    return; // initialization will be done in next call because of shot
            // branching in reset
  }

  Base::states_[root.state_index()].qreg().initialize_component(qubits, params);
}

template <class state_t>
void Executor<state_t>::apply_kraus(CircuitExecutor::Branch &root,
                                    const reg_t &qubits,
                                    const std::vector<cmatrix_t> &kmats) {
  // Check edge case for empty Kraus set (this shouldn't happen)
  if (kmats.empty())
    return; // end function early

  // Choose a real in [0, 1) to choose the applied kraus operator once
  // the accumulated probability is greater than r.
  // We know that the Kraus noise must be normalized
  // So we only compute probabilities for the first N-1 kraus operators
  // and infer the probability of the last one from 1 - sum of the previous

  double accum = 0.;
  double p;

  reg_t shot_branch;
  uint_t nshots;
  rvector_t rshots, pmats;
  uint_t nshots_multiplied = 0;

  nshots = root.num_shots();
  shot_branch.resize(nshots);
  rshots.resize(nshots);
  for (uint_t i = 0; i < nshots; i++) {
    shot_branch[i] = kmats.size() - 1;
    rshots[i] = root.rng_shots()[i].rand(0., 1.);
  }
  pmats.resize(kmats.size());

  // Loop through N-1 kraus operators
  for (size_t j = 0; j < kmats.size() - 1; j++) {
    // Calculate probability
    cvector_t<double> vmat = Utils::vectorize_matrix(kmats[j]);

    p = Base::states_[root.state_index()].qreg().norm(qubits, vmat);
    accum += p;

    // check if we need to apply this operator
    pmats[j] = p;
    for (uint_t i = 0; i < nshots; i++) {
      if (shot_branch[i] >= kmats.size() - 1) {
        if (accum > rshots[i]) {
          shot_branch[i] = j;
          nshots_multiplied++;
        }
      }
    }
    if (nshots_multiplied >= nshots) {
      break;
    }
  }

  // check if we haven't applied a kraus operator yet
  pmats[pmats.size() - 1] = 1. - accum;

  root.creg() = Base::states_[root.state_index()].creg();
  root.branch_shots(shot_branch, kmats.size());
  for (uint_t i = 0; i < kmats.size(); i++) {
    Operations::Op op;
    op.type = OpType::matrix;
    op.qubits = qubits;
    op.mats.push_back(kmats[i]);
    p = 1 / std::sqrt(pmats[i]);
    for (uint_t j = 0; j < op.mats[0].size(); j++)
      op.mats[0][j] *= p;
    root.branches()[i]->add_op_after_branch(op);
  }
}

template <class state_t>
void Executor<state_t>::apply_save_density_matrix(CircuitExecutor::Branch &root,
                                                  const Operations::Op &op,
                                                  ResultItr result) {
  cmatrix_t reduced_state;

  // Check if tracing over all qubits
  if (op.qubits.empty()) {
    reduced_state = cmatrix_t(1, 1);

    reduced_state[0] = Base::states_[root.state_index()].qreg().norm();
  } else {
    reduced_state =
        Base::states_[root.state_index()].qreg().reduced_density_matrix(
            op.qubits);
  }

  std::vector<bool> copied(Base::num_bind_params_, false);
  for (uint_t i = 0; i < root.num_shots(); i++) {
    uint_t ip = root.param_index(i);
    if (!copied[ip]) {
      (result + ip)
          ->save_data_average(Base::states_[root.state_index()].creg(),
                              op.string_params[0], reduced_state, op.type,
                              op.save_type);
      copied[ip] = true;
    }
  }
}

template <class state_t>
void Executor<state_t>::apply_save_probs(CircuitExecutor::Branch &root,
                                         const Operations::Op &op,
                                         ResultItr result) {
  // get probs as hexadecimal
  auto probs =
      Base::states_[root.state_index()].qreg().probabilities(op.qubits);

  std::vector<bool> copied(Base::num_bind_params_, false);
  if (op.type == Operations::OpType::save_probs_ket) {
    // Convert to ket dict
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      if (!copied[ip]) {
        (result + ip)
            ->save_data_average(
                Base::states_[root.state_index()].creg(), op.string_params[0],
                Utils::vec2ket(probs, Base::json_chop_threshold_, 16), op.type,
                op.save_type);
        copied[ip] = true;
      }
    }
  } else {
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      if (!copied[ip]) {
        (result + ip)
            ->save_data_average(Base::states_[root.state_index()].creg(),
                                op.string_params[0], probs, op.type,
                                op.save_type);
        copied[ip] = true;
      }
    }
  }
}

template <class state_t>
void Executor<state_t>::apply_save_statevector(CircuitExecutor::Branch &root,
                                               const Operations::Op &op,
                                               ResultItr result, bool last_op) {
  if (op.qubits.size() != Base::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  std::string key =
      (op.string_params[0] == "_method_") ? "statevector" : op.string_params[0];

  if (last_op) {
    const auto v = Base::states_[root.state_index()].move_to_vector();
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      (result + ip)
          ->save_data_pershot(Base::states_[root.state_index()].creg(), key, v,
                              OpType::save_statevec, op.save_type);
    }
  } else {
    const auto v = Base::states_[root.state_index()].copy_to_vector();
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      (result + ip)
          ->save_data_pershot(Base::states_[root.state_index()].creg(), key, v,
                              OpType::save_statevec, op.save_type);
    }
  }
}

template <class state_t>
void Executor<state_t>::apply_save_statevector_dict(
    CircuitExecutor::Branch &root, const Operations::Op &op, ResultItr result) {
  if (op.qubits.size() != Base::num_qubits_) {
    throw std::invalid_argument(op.name +
                                " was not applied to all qubits."
                                " Only the full statevector can be saved.");
  }
  auto state_ket = Base::states_[root.state_index()].qreg().vector_ket(
      Base::json_chop_threshold_);
  std::map<std::string, complex_t> result_state_ket;
  for (auto const &it : state_ket) {
    result_state_ket[it.first] = it.second;
  }
  for (uint_t i = 0; i < root.num_shots(); i++) {
    uint_t ip = root.param_index(i);
    (result + ip)
        ->save_data_pershot(
            Base::states_[root.state_index()].creg(), op.string_params[0],
            (const std::map<std::string, complex_t> &)result_state_ket, op.type,
            op.save_type);
  }
}

template <class state_t>
void Executor<state_t>::apply_save_amplitudes(CircuitExecutor::Branch &root,
                                              const Operations::Op &op,
                                              ResultItr result) {
  if (op.int_params.empty()) {
    throw std::invalid_argument(
        "Invalid save_amplitudes instructions (empty params).");
  }
  const uint_t size = op.int_params.size();
  if (op.type == Operations::OpType::save_amps) {
    Vector<complex_t> amps(size, false);
    for (uint_t i = 0; i < size; ++i) {
      amps[i] =
          Base::states_[root.state_index()].qreg().get_state(op.int_params[i]);
    }
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      (result + ip)
          ->save_data_pershot(
              Base::states_[root.state_index()].creg(), op.string_params[0],
              (const Vector<complex_t> &)amps, op.type, op.save_type);
    }
  } else {
    rvector_t amps_sq(size, 0);
    for (uint_t i = 0; i < size; ++i) {
      amps_sq[i] = Base::states_[root.state_index()].qreg().probability(
          op.int_params[i]);
    }
    std::vector<bool> copied(Base::num_bind_params_, false);
    for (uint_t i = 0; i < root.num_shots(); i++) {
      uint_t ip = root.param_index(i);
      if (!copied[ip]) {
        (result + ip)
            ->save_data_average(Base::states_[root.state_index()].creg(),
                                op.string_params[0], amps_sq, op.type,
                                op.save_type);
        copied[ip] = true;
      }
    }
  }
}

template <class state_t>
std::vector<reg_t>
Executor<state_t>::sample_measure(state_t &state, const reg_t &qubits,
                                  uint_t shots,
                                  std::vector<RngEngine> &rng) const {
  int_t i, j;
  std::vector<double> rnds;
  rnds.reserve(shots);

  for (i = 0; i < (int_t)shots; ++i)
    rnds.push_back(rng[i].rand(0, 1));

  std::vector<reg_t> samples = state.qreg().sample_measure(rnds);
  std::vector<reg_t> ret(shots);

  if (omp_get_num_threads() > 1) {
    for (i = 0; i < (int_t)shots; ++i) {
      ret[i].resize(qubits.size());
      for (j = 0; j < (int_t)qubits.size(); j++)
        ret[i][j] = samples[i][qubits[j]];
    }
  } else {
#pragma omp parallel for private(j)
    for (i = 0; i < (int_t)shots; ++i) {
      ret[i].resize(qubits.size());
      for (j = 0; j < (int_t)qubits.size(); j++)
        ret[i][j] = samples[i][qubits[j]];
    }
  }
  return ret;
}

//-------------------------------------------------------------------------
} // namespace TensorNetwork
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
