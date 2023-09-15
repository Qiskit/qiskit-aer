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

#ifndef _batch_shots_executor_hpp_
#define _batch_shots_executor_hpp_

#include "simulators/parallel_state_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace CircuitExecutor {

//-------------------------------------------------------------------------
// batched-shots executor class implementation
//-------------------------------------------------------------------------
template <class state_t>
class BatchShotsExecutor : public virtual MultiStateExecutor<state_t> {
  using Base = MultiStateExecutor<state_t>;

protected:
  // config setting for multi-shot parallelization
  bool batched_shots_gpu_ = true;
  int_t batched_shots_gpu_max_qubits_ =
      16; // multi-shot parallelization is applied if qubits is less than max
          // qubits
  bool enable_batch_multi_shots_ =
      false;                 // multi-shot parallelization can be applied
  uint_t local_state_index_; // local shot ID of current loop
public:
  BatchShotsExecutor();
  virtual ~BatchShotsExecutor();

protected:
  void set_config(const Config &config) override;
  void set_parallelization(const Circuit &circ,
                           const Noise::NoiseModel &noise) override;

  void run_circuit_shots(Circuit &circ, const Noise::NoiseModel &noise,
                         const Config &config, RngEngine &init_rng,
                         ExperimentResult &result, bool sample_noise) override;

  // apply ops for multi-shots to one group
  template <typename InputIterator>
  void apply_ops_batched_shots_for_group(int_t i_group, InputIterator first,
                                         InputIterator last,
                                         const Noise::NoiseModel &noise,
                                         ExperimentResult &result,
                                         RngEngine &init_rng, uint_t rng_seed,
                                         bool final_ops);

  // apply op to multiple shots , return flase if op is not supported to execute
  // in a batch
  virtual bool apply_batched_op(const int_t istate, const Operations::Op &op,
                                ExperimentResult &result,
                                std::vector<RngEngine> &rng,
                                bool final_op = false) {
    return false;
  }

  // apply sampled noise to multiple-shots (this is used for ops contains
  // non-Pauli operators)
  void apply_batched_noise_ops(
      const int_t i_group, const std::vector<std::vector<Operations::Op>> &ops,
      ExperimentResult &result, std::vector<RngEngine> &rng);
};

template <class state_t>
BatchShotsExecutor<state_t>::BatchShotsExecutor() {}

template <class state_t>
BatchShotsExecutor<state_t>::~BatchShotsExecutor() {}

template <class state_t>
void BatchShotsExecutor<state_t>::set_config(const Config &config) {
  Base::set_config(config);

  // enable batched multi-shots/experiments optimization
  batched_shots_gpu_ = config.batched_shots_gpu;

  batched_shots_gpu_max_qubits_ = config.batched_shots_gpu_max_qubits;
  if (Base::method_ == Method::density_matrix ||
      Base::method_ == Method::unitary)
    batched_shots_gpu_max_qubits_ /= 2;
}

template <class state_t>
void BatchShotsExecutor<state_t>::set_parallelization(
    const Circuit &circ, const Noise::NoiseModel &noise) {
  Base::set_parallelization(circ, noise);

  enable_batch_multi_shots_ = false;
  if (batched_shots_gpu_ && Base::sim_device_ != Device::CPU) {
    enable_batch_multi_shots_ = true;
    if (circ.num_qubits >= batched_shots_gpu_max_qubits_)
      enable_batch_multi_shots_ = false;
    else if (circ.shots == 1)
      enable_batch_multi_shots_ = false;
    //    else if (Base::multiple_chunk_required(circ, noise))
    //      enable_batch_multi_shots_ = false;
  }

#ifdef AER_CUSTATEVEC
  // disable cuStateVec for batch-shots optimization
  if (enable_batch_multi_shots_ && Base::cuStateVec_enable_)
    Base::cuStateVec_enable_ = false;
#endif
}

template <class state_t>
void BatchShotsExecutor<state_t>::run_circuit_shots(
    Circuit &circ, const Noise::NoiseModel &noise, const Config &config,
    RngEngine &init_rng, ExperimentResult &result, bool sample_noise) {
  state_t dummy_state;
  // if batched-shot is not applicable, use base multi-shots executor
  if (!enable_batch_multi_shots_) {
    return Base::run_circuit_shots(circ, noise, config, init_rng, result,
                                   sample_noise);
  }

  Noise::NoiseModel dummy_noise;

  Base::num_qubits_ = circ.num_qubits;
  Base::num_creg_memory_ = circ.num_memory;
  Base::num_creg_registers_ = circ.num_registers;

  if (Base::sim_device_ == Device::GPU) {
#ifdef _OPENMP
    if (omp_get_num_threads() == 1)
      Base::shot_omp_parallel_ = true;
#endif
  } else if (Base::sim_device_ == Device::ThrustCPU) {
    Base::shot_omp_parallel_ = false;
  }

  Base::set_distribution(circ.shots);
  Base::num_max_shots_ = Base::get_max_parallel_shots(circ, noise);
  if (Base::num_max_shots_ == 0)
    Base::num_max_shots_ = 1;

  RngEngine rng = init_rng;

  Circuit circ_opt;
  if (sample_noise)
    circ_opt =
        noise.sample_noise(circ, rng, Noise::NoiseModel::Method::circuit, true);
  else
    circ_opt = circ;
  auto fusion_pass = Base::transpile_fusion(circ_opt.opset(), config);

  fusion_pass.optimize_circuit(circ_opt, dummy_noise, dummy_state.opset(),
                               result);
  Base::max_matrix_qubits_ = Base::get_max_matrix_qubits(circ_opt);

  // Add batched multi-shots optimizaiton metadata
  result.metadata.add(true, "batched_shots_optimization");

  int_t i;
  int_t i_begin, n_shots;

#ifdef AER_MPI
  // if shots are distributed to MPI processes, allocate cregs to be gathered
  if (Base::num_process_per_experiment_ > 1)
    Base::cregs_.resize(circ_opt.shots);
#endif

  i_begin = 0;
  while (i_begin < Base::num_local_states_) {
    local_state_index_ = Base::global_state_index_ + i_begin;

    // loop for states can be stored in available memory
    n_shots = std::min(Base::num_local_states_, Base::num_max_shots_);
    if (i_begin + n_shots > Base::num_local_states_) {
      n_shots = Base::num_local_states_ - i_begin;
    }

    // allocate shots
    this->allocate_states(n_shots, config);

    // Set state config
    for (i = 0; i < n_shots; i++) {
      Base::states_[i].set_parallelization(Base::parallel_state_update_);
      Base::states_[i].set_global_phase(circ.global_phase_angle);
    }
    this->set_global_phase(circ_opt.global_phase_angle);

    // initialization (equivalent to initialize_qreg + initialize_creg)
    auto init_group = [this](int_t ig) {
      for (uint_t j = Base::top_state_of_group_[ig];
           j < Base::top_state_of_group_[ig + 1]; j++) {
        // enabling batch shots optimization
        Base::states_[j].qreg().enable_batch(true);

        // initialize qreg here
        Base::states_[j].qreg().set_num_qubits(Base::num_qubits_);
        Base::states_[j].qreg().initialize();

        // initialize creg here
        Base::states_[j].qreg().initialize_creg(Base::num_creg_memory_,
                                                Base::num_creg_registers_);
      }
    };
    Utils::apply_omp_parallel_for(
        (Base::num_groups_ > 1 && Base::shot_omp_parallel_), 0,
        Base::num_groups_, init_group);

    this->apply_global_phase(); // this is parallelized in sub-classes

    // apply ops to multiple-shots
    if (Base::num_groups_ > 1 && Base::shot_omp_parallel_) {
      std::vector<ExperimentResult> par_results(Base::num_groups_);
#pragma omp parallel for num_threads(Base::num_groups_)
      for (i = 0; i < Base::num_groups_; i++)
        apply_ops_batched_shots_for_group(
            i, circ_opt.ops.cbegin(), circ_opt.ops.cend(), noise,
            par_results[i], rng, circ_opt.seed, true);

      for (auto &res : par_results)
        result.combine(std::move(res));
    } else {
      for (i = 0; i < Base::num_groups_; i++)
        apply_ops_batched_shots_for_group(i, circ_opt.ops.cbegin(),
                                          circ_opt.ops.cend(), noise, result,
                                          rng, circ_opt.seed, true);
    }

    // collect measured bits and copy memory
    for (i = 0; i < n_shots; i++) {
      if (Base::num_process_per_experiment_ > 1) {
        Base::states_[i].qreg().read_measured_data(
            Base::cregs_[local_state_index_ + i]);
      } else {
        Base::states_[i].qreg().read_measured_data(Base::states_[i].creg());
        result.save_count_data(Base::states_[i].creg(),
                               Base::save_creg_memory_);
      }
    }

    i_begin += n_shots;
  }

  // gather cregs on MPI processes and save to result
#ifdef AER_MPI
  if (Base::num_process_per_experiment_ > 1) {
    Base::gather_creg_memory(Base::cregs_, Base::state_index_begin_);

    for (i = 0; i < circ_opt.shots; i++)
      result.save_count_data(Base::cregs_[i], Base::save_creg_memory_);
    Base::cregs_.clear();
  }
#endif

#ifdef AER_THRUST_GPU
  if (Base::sim_device_ == Device::GPU) {
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      nDev = 0;
    }
    if (nDev > Base::num_groups_)
      nDev = Base::num_groups_;
    result.metadata.add(nDev, "batched_shots_optimization_parallel_gpus");
  }
#endif
}

template <class state_t>
template <typename InputIterator>
void BatchShotsExecutor<state_t>::apply_ops_batched_shots_for_group(
    int_t i_group, InputIterator first, InputIterator last,
    const Noise::NoiseModel &noise, ExperimentResult &result,
    RngEngine &init_rng, uint_t rng_seed, bool final_ops) {
  uint_t istate = Base::top_state_of_group_[i_group];
  std::vector<RngEngine> rng(Base::num_states_in_group_[i_group]);
#ifdef _OPENMP
  int num_inner_threads = omp_get_max_threads() / omp_get_num_threads();
#else
  int num_inner_threads = 1;
#endif

  for (uint_t j = Base::top_state_of_group_[i_group];
       j < Base::top_state_of_group_[i_group + 1]; j++)
    if (local_state_index_ + j == 0)
      rng[j - Base::top_state_of_group_[i_group]] = init_rng;
    else {
      rng[j - Base::top_state_of_group_[i_group]].set_seed(
          rng_seed + local_state_index_ + j);
    }

  for (auto op = first; op != last; ++op) {
    if (op->type == Operations::OpType::sample_noise) {
      if (op->expr) {
        for (uint_t j = Base::top_state_of_group_[i_group];
             j < Base::top_state_of_group_[i_group + 1]; j++) {
          Base::states_[j].qreg().enable_batch(false);
          Base::states_[j].qreg().read_measured_data(Base::states_[j].creg());
          std::vector<Operations::Op> nops = noise.sample_noise_loc(
              *op, rng[j - Base::top_state_of_group_[i_group]]);
          for (int_t k = 0; k < nops.size(); k++) {
            Base::states_[j].apply_op(
                nops[k], result, rng[j - Base::top_state_of_group_[i_group]],
                false);
          }
          Base::states_[j].qreg().enable_batch(true);
        }
        continue;
      }

      // sample error here
      uint_t count = Base::num_states_in_group_[i_group];
      std::vector<std::vector<Operations::Op>> noise_ops(count);

      uint_t count_ops = 0;
      uint_t non_pauli_gate_count = 0;
      if (num_inner_threads > 1) {
#pragma omp parallel for reduction(+: count_ops,non_pauli_gate_count) num_threads(num_inner_threads)
        for (int_t j = 0; j < count; j++) {
          noise_ops[j] = noise.sample_noise_loc(*op, rng[j]);

          if (!(noise_ops[j].size() == 0 ||
                (noise_ops[j].size() == 1 && noise_ops[j][0].name == "id"))) {
            count_ops++;
            for (int_t k = 0; k < noise_ops[j].size(); k++) {
              if (noise_ops[j][k].name != "id" && noise_ops[j][k].name != "x" &&
                  noise_ops[j][k].name != "y" && noise_ops[j][k].name != "z" &&
                  noise_ops[j][k].name != "pauli") {
                non_pauli_gate_count++;
                break;
              }
            }
          }
        }
      } else {
        for (int_t j = 0; j < count; j++) {
          noise_ops[j] = noise.sample_noise_loc(*op, rng[j]);

          if (!(noise_ops[j].size() == 0 ||
                (noise_ops[j].size() == 1 && noise_ops[j][0].name == "id"))) {
            count_ops++;
            for (int_t k = 0; k < noise_ops[j].size(); k++) {
              if (noise_ops[j][k].name != "id" && noise_ops[j][k].name != "x" &&
                  noise_ops[j][k].name != "y" && noise_ops[j][k].name != "z" &&
                  noise_ops[j][k].name != "pauli") {
                non_pauli_gate_count++;
                break;
              }
            }
          }
        }
      }

      if (count_ops == 0) {
        continue; // do nothing
      }
      if (non_pauli_gate_count == 0) { // ptimization for Pauli error
        Base::states_[istate].qreg().apply_batched_pauli_ops(noise_ops);
      } else {
        // otherwise execute each circuit
        apply_batched_noise_ops(i_group, noise_ops, result, rng);
      }
    } else {
      if (!op->expr) {
        if (apply_batched_op(istate, *op, result, rng,
                             final_ops && (op + 1 == last))) {
          continue;
        }
      }
      // call apply_op for each state
      for (uint_t j = Base::top_state_of_group_[i_group];
           j < Base::top_state_of_group_[i_group + 1]; j++) {
        Base::states_[j].qreg().enable_batch(false);
        Base::states_[j].qreg().read_measured_data(Base::states_[j].creg());
        Base::states_[j].apply_op(*op, result,
                                  rng[j - Base::top_state_of_group_[i_group]],
                                  final_ops && (op + 1 == last));
        Base::states_[j].qreg().enable_batch(true);
      }
    }
  }
}

template <class state_t>
void BatchShotsExecutor<state_t>::apply_batched_noise_ops(
    const int_t i_group, const std::vector<std::vector<Operations::Op>> &ops,
    ExperimentResult &result, std::vector<RngEngine> &rng) {
  int_t i, j, k, count, nop, pos = 0;
  uint_t istate = Base::top_state_of_group_[i_group];
  count = ops.size();

  reg_t mask(count);
  std::vector<bool> finished(count, false);
  for (i = 0; i < count; i++) {
    int_t cond_reg = -1;

    if (finished[i])
      continue;
    if (ops[i].size() == 0 || (ops[i].size() == 1 && ops[i][0].name == "id")) {
      finished[i] = true;
      continue;
    }
    mask[i] = 1;

    // find same ops to be exectuted in a batch
    for (j = i + 1; j < count; j++) {
      if (finished[j]) {
        mask[j] = 0;
        continue;
      }
      if (ops[j].size() == 0 ||
          (ops[j].size() == 1 && ops[j][0].name == "id")) {
        mask[j] = 0;
        finished[j] = true;
        continue;
      }

      if (ops[i].size() != ops[j].size()) {
        mask[j] = 0;
        continue;
      }

      mask[j] = true;
      for (k = 0; k < ops[i].size(); k++) {
        if (ops[i][k].conditional) {
          cond_reg = ops[i][k].conditional_reg;
        }
        if (ops[i][k].type != ops[j][k].type ||
            ops[i][k].name != ops[j][k].name) {
          mask[j] = false;
          break;
        }
      }
      if (mask[j])
        finished[j] = true;
    }

    // mask conditional register
    int_t sys_reg = Base::states_[istate].qreg().set_batched_system_conditional(
        cond_reg, mask);

    // batched execution on same ops
    for (k = 0; k < ops[i].size(); k++) {
      Operations::Op cop = ops[i][k];

      // mark op conditional to mask shots
      cop.conditional = true;
      cop.conditional_reg = sys_reg;

      if (!apply_batched_op(istate, cop, result, rng, false)) {
        // call apply_op for each state
        /*if(cop.conditional){
          //copy creg to local state
          reg_t reg_pos(1);
          reg_t mem_pos;
          int bit =
        Base::states_[j].qreg().measured_cregister(cop.conditional_reg);
          const reg_t reg = Utils::int2reg(bit, 2, 1);
          reg_pos[0] = cop.conditional_reg;
          Base::states_[j].creg().store_measure(reg, mem_pos, reg_pos);
        }*/
        for (uint_t j = Base::top_state_of_group_[i_group];
             j < Base::top_state_of_group_[i_group + 1]; j++) {
          Base::states_[j].qreg().enable_batch(false);
          Base::states_[j].apply_op(
              cop, result, rng[j - Base::top_state_of_group_[i_group]], false);
          Base::states_[j].qreg().enable_batch(true);
        }
      }
    }
    mask[i] = 0;
    finished[i] = true;
  }
}

//-------------------------------------------------------------------------
} // end namespace CircuitExecutor
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
