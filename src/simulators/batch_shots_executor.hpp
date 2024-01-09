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
#include "transpile/batch_converter.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace CircuitExecutor {

using OpItr = std::vector<Operations::Op>::const_iterator;
using ResultItr = std::vector<ExperimentResult>::iterator;

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
      false; // multi-shot parallelization can be applied
public:
  BatchShotsExecutor();
  virtual ~BatchShotsExecutor();

protected:
  void set_config(const Config &config) override;
  void set_parallelization(const Config &config, const Circuit &circ,
                           const Noise::NoiseModel &noise) override;

  void run_circuit_with_sampling(Circuit &circ, const Config &config,
                                 RngEngine &init_rng,
                                 ResultItr result) override;

  void run_circuit_shots(Circuit &circ, const Noise::NoiseModel &noise,
                         const Config &config, RngEngine &init_rng,
                         ResultItr result_it, bool sample_noise) override;

  // apply ops for multi-shots to one group
  template <typename InputIterator>
  void apply_ops_batched_shots_for_group(int_t i_group, InputIterator first,
                                         InputIterator last,
                                         const Noise::NoiseModel &noise,
                                         ResultItr result,
                                         std::vector<RngEngine> &rng,
                                         bool final_ops);

  // apply op to multiple shots , return flase if op is not supported to execute
  // in a batch
  virtual bool apply_batched_op(const int_t istate, const Operations::Op &op,
                                ResultItr result, std::vector<RngEngine> &rng,
                                bool final_op = false) {
    return false;
  }

  // apply sampled noise to multiple-shots (this is used for ops contains
  // non-Pauli operators)
  void
  apply_batched_noise_ops(const int_t i_group,
                          const std::vector<std::vector<Operations::Op>> &ops,
                          ResultItr result, std::vector<RngEngine> &rng);

  // batched expval Pauli
  void apply_batched_expval(const int_t istate, const Operations::Op &op,
                            ResultItr result);

  // sample measure for runtime parameter binding
  template <typename InputIterator>
  void batched_measure_sampler(InputIterator first_meas,
                               InputIterator last_meas, uint_t shots,
                               uint_t i_group, ResultItr result,
                               std::vector<RngEngine> &rng);
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

  // enable batch execution for runtime parameter binding
  if (Base::num_bind_params_ > 1 && Base::sim_device_ == Device::GPU) {
    batched_shots_gpu_ = true;
  }

  batched_shots_gpu_max_qubits_ = config.batched_shots_gpu_max_qubits;
  if (Base::method_ == Method::density_matrix ||
      Base::method_ == Method::unitary)
    batched_shots_gpu_max_qubits_ /= 2;
}

template <class state_t>
void BatchShotsExecutor<state_t>::set_parallelization(
    const Config &config, const Circuit &circ, const Noise::NoiseModel &noise) {
  Base::set_parallelization(config, circ, noise);

  enable_batch_multi_shots_ = false;
  if (batched_shots_gpu_ && Base::sim_device_ != Device::CPU) {
    enable_batch_multi_shots_ = true;
    if (circ.num_qubits > (uint_t)batched_shots_gpu_max_qubits_)
      enable_batch_multi_shots_ = false;
    else if (circ.shots == 1 && circ.num_bind_params == 1)
      enable_batch_multi_shots_ = false;
  }

#ifdef AER_CUSTATEVEC
  // disable cuStateVec for batch-shots optimization
  if (enable_batch_multi_shots_ && Base::cuStateVec_enable_)
    Base::cuStateVec_enable_ = false;
#endif
}

template <class state_t>
void BatchShotsExecutor<state_t>::run_circuit_with_sampling(
    Circuit &circ, const Config &config, RngEngine &init_rng,
    ResultItr result_it) {
  if (!enable_batch_multi_shots_) {
    return Executor<state_t>::run_circuit_with_sampling(circ, config, init_rng,
                                                        result_it);
  }
  Noise::NoiseModel dummy_noise;
  state_t dummy_state;
  uint_t i_begin, n_shots;

  Base::num_qubits_ = circ.num_qubits;
  Base::num_creg_memory_ = circ.num_memory;
  Base::num_creg_registers_ = circ.num_registers;
  Base::num_bind_params_ = circ.num_bind_params;

  if (Base::sim_device_ == Device::GPU) {
#ifdef _OPENMP
    if (omp_get_num_threads() == 1)
      Base::shot_omp_parallel_ = true;
#endif
  } else if (Base::sim_device_ == Device::ThrustCPU) {
    Base::shot_omp_parallel_ = false;
  }

  // distribute parameters
  Base::set_distribution(circ.num_bind_params);
  uint_t mem = Base::required_memory_mb(config, circ, dummy_noise);
  if (Base::sim_device_ == Device::GPU && Base::num_gpus_ > 0)
    Base::num_max_shots_ = Base::max_gpu_memory_mb_ * 8 / 10 / mem;
  else
    Base::num_max_shots_ = Base::max_memory_mb_ / mem;
  if (Base::num_max_shots_ == 0)
    Base::num_max_shots_ = 1;

  auto fusion_pass = Base::transpile_fusion(circ.opset(), config);
  ExperimentResult fusion_result;
  fusion_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(),
                               fusion_result);
  // convert parameters into matrix in cvector_t format
  auto timer_start = myclock_t::now();
  Transpile::BatchConverter batch_converter;
  batch_converter.set_config(config);
  batch_converter.optimize_circuit(circ, dummy_noise, dummy_state.opset(),
                                   fusion_result);
  auto time_taken =
      std::chrono::duration<double>(myclock_t::now() - timer_start).count();
  for (uint_t i = 0; i < circ.num_bind_params; i++) {
    ExperimentResult &result = *(result_it + i);
    result.metadata.copy(fusion_result.metadata);
    // Add batched multi-shots optimizaiton metadata
    result.metadata.add(true, "batched_shots_optimization");
    result.metadata.add(time_taken, "parameter_bind_batch_converter_time");
  }

  Base::max_matrix_qubits_ = Base::get_max_matrix_qubits(circ);

#ifdef AER_MPI
  // if shots are distributed to MPI processes, allocate cregs to be gathered
  if (Base::num_process_per_experiment_ > 1)
    Base::cregs_.resize(circ.num_bind_params * circ.shots);
#endif

  auto first_meas = circ.first_measure_pos; // Position of first measurement op
  bool final_ops = (first_meas == circ.ops.size());

  // adjust max_matrix_qubits_ so that all shots can be stored on GPU
  if (circ.ops.begin() + first_meas != circ.ops.end())
    Base::max_sampling_shots_ = circ.shots;

  i_begin = 0;
  while (i_begin < Base::num_local_states_) {
    // loop for states can be stored in available memory
    n_shots = Base::num_local_states_ - i_begin;
    n_shots = std::min(n_shots, Base::num_max_shots_);

    // allocate shots
    this->allocate_states(n_shots, config);

    // Set state config
    for (uint_t i = 0; i < n_shots; i++) {
      Base::states_[i].set_parallelization(Base::parallel_state_update_);
    }

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
        Base::num_groups_, init_group, Base::num_groups_);

    // apply ops to multiple-shots
    auto apply_ops_lambda = [this, circ, init_rng, first_meas, final_ops,
                             dummy_noise, &result_it](int_t i) {
      std::vector<RngEngine> rng(Base::num_states_in_group_[i]);
      for (uint_t j = 0; j < Base::num_states_in_group_[i]; j++) {
        uint_t iparam =
            Base::global_state_index_ + Base::top_state_of_group_[i] + j;
        if (iparam == 0)
          rng[j] = init_rng;
        else
          rng[j].set_seed(circ.seed_for_params[iparam]);
      }
      apply_ops_batched_shots_for_group(i, circ.ops.cbegin(),
                                        circ.ops.cbegin() + first_meas,
                                        dummy_noise, result_it, rng, final_ops);

      batched_measure_sampler(circ.ops.begin() + first_meas, circ.ops.end(),
                              circ.shots, i, result_it, rng);
    };
    Utils::apply_omp_parallel_for(
        (Base::num_groups_ > 1 && Base::shot_omp_parallel_), 0,
        Base::num_groups_, apply_ops_lambda, Base::num_groups_);

    Base::global_state_index_ += n_shots;
    i_begin += n_shots;
  }

  // gather cregs on MPI processes and save to result
#ifdef AER_MPI
  if (Base::num_process_per_experiment_ > 1) {
    Base::gather_creg_memory(Base::cregs_, Base::state_index_begin_);

    for (uint_t i = 0; i < circ.num_bind_params; i++) {
      for (uint_t j = 0; j < circ.shots; j++) {
        (result_it + i)
            ->save_count_data(Base::cregs_[i * circ.shots + j],
                              Base::save_creg_memory_);
      }
    }
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
    for (uint_t i = 0; i < circ.num_bind_params; i++)
      (result_it + i)
          ->metadata.add(nDev, "batched_shots_optimization_parallel_gpus");
  }
#endif
}

template <class state_t>
void BatchShotsExecutor<state_t>::run_circuit_shots(
    Circuit &circ, const Noise::NoiseModel &noise, const Config &config,
    RngEngine &init_rng, ResultItr result_it, bool sample_noise) {
  state_t dummy_state;
  // if batched-shot is not applicable, use base multi-shots executor
  if (!enable_batch_multi_shots_) {
    return Base::run_circuit_shots(circ, noise, config, init_rng, result_it,
                                   sample_noise);
  }

  Noise::NoiseModel dummy_noise;

  Base::num_qubits_ = circ.num_qubits;
  Base::num_creg_memory_ = circ.num_memory;
  Base::num_creg_registers_ = circ.num_registers;
  Base::num_bind_params_ = circ.num_bind_params;
  Base::num_shots_per_bind_param_ = circ.shots;

  if (Base::sim_device_ == Device::GPU) {
#ifdef _OPENMP
    if (omp_get_num_threads() == 1)
      Base::shot_omp_parallel_ = true;
#endif
  } else if (Base::sim_device_ == Device::ThrustCPU) {
    Base::shot_omp_parallel_ = false;
  }

  Base::set_distribution(circ.shots * Base::num_bind_params_);
  Base::num_max_shots_ = Base::get_max_parallel_shots(config, circ, noise);
  if (Base::num_max_shots_ == 0)
    Base::num_max_shots_ = 1;

  Circuit circ_opt;
  if (sample_noise)
    circ_opt = noise.sample_noise(circ, init_rng,
                                  Noise::NoiseModel::Method::circuit, true);
  else
    circ_opt = circ;
  auto fusion_pass = Base::transpile_fusion(circ_opt.opset(), config);
  ExperimentResult fusion_result;
  fusion_pass.optimize_circuit(circ_opt, dummy_noise, dummy_state.opset(),
                               fusion_result);
  // convert parameters into matrix in cvector_t format
  Transpile::BatchConverter batch_converter;
  batch_converter.set_config(config);
  batch_converter.optimize_circuit(circ_opt, dummy_noise, dummy_state.opset(),
                                   fusion_result);

  Base::max_matrix_qubits_ = Base::get_max_matrix_qubits(circ_opt);

  uint_t i_begin, n_shots;

  for (uint_t i = 0; i < Base::num_bind_params_; i++) {
    ExperimentResult &result = *(result_it + i);
    result.metadata.copy(fusion_result.metadata);
    // Add batched multi-shots optimizaiton metadata
    result.metadata.add(true, "batched_shots_optimization");
  }

#ifdef AER_MPI
  // if shots are distributed to MPI processes, allocate cregs to be gathered
  if (Base::num_process_per_experiment_ > 1)
    Base::cregs_.resize(circ_opt.shots * Base::num_bind_params_);
#endif

  i_begin = 0;
  while (i_begin < Base::num_local_states_) {
    // loop for states can be stored in available memory
    n_shots = Base::num_local_states_ - i_begin;
    n_shots = std::min(n_shots, Base::num_max_shots_);

    // allocate shots
    this->allocate_states(n_shots, config);

    // Set state config
    for (uint_t i = 0; i < n_shots; i++) {
      Base::states_[i].set_parallelization(Base::parallel_state_update_);
    }

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
        Base::num_groups_, init_group, Base::num_groups_);

    // apply ops to multiple-shots
    std::vector<std::vector<ExperimentResult>> par_results(Base::num_groups_);
    auto apply_ops_lambda = [this, circ, circ_opt, &par_results, init_rng,
                             noise](int_t i) {
      par_results[i].resize(circ.num_bind_params);
      std::vector<RngEngine> rng(Base::num_states_in_group_[i]);
      for (uint_t j = 0; j < Base::num_states_in_group_[i]; j++) {
        uint_t ishot =
            Base::global_state_index_ + Base::top_state_of_group_[i] + j;
        uint_t iparam = ishot / Base::num_shots_per_bind_param_;
        if (ishot == 0)
          rng[j] = init_rng;
        else {
          if (Base::num_bind_params_ > 1)
            rng[j].set_seed(circ.seed_for_params[iparam] +
                            (ishot % Base::num_shots_per_bind_param_));
          else
            rng[j].set_seed(circ_opt.seed + ishot);
        }
      }
      apply_ops_batched_shots_for_group(i, circ_opt.ops.cbegin(),
                                        circ_opt.ops.cend(), noise,
                                        par_results[i].begin(), rng, true);
    };
    Utils::apply_omp_parallel_for(
        (Base::num_groups_ > 1 && Base::shot_omp_parallel_), 0,
        Base::num_groups_, apply_ops_lambda, Base::num_groups_);

    for (auto &res : par_results) {
      for (uint_t i = 0; i < Base::num_bind_params_; i++) {
        (result_it + i)->combine(std::move(res[i]));
      }
    }

    // collect measured bits and copy memory
    for (uint_t i = 0; i < n_shots; i++) {
      if (Base::num_process_per_experiment_ > 1) {
        Base::states_[i].qreg().read_measured_data(
            Base::cregs_[Base::global_state_index_ + i_begin + i]);
      } else {
        uint_t ishot = Base::global_state_index_ + i;
        uint_t iparam = ishot / Base::num_shots_per_bind_param_;
        Base::states_[i].qreg().read_measured_data(Base::states_[i].creg());
        (result_it + iparam)
            ->save_count_data(Base::states_[i].creg(), Base::save_creg_memory_);
      }
    }

    Base::global_state_index_ += n_shots;
    i_begin += n_shots;
  }

  // gather cregs on MPI processes and save to result
#ifdef AER_MPI
  if (Base::num_process_per_experiment_ > 1) {
    Base::gather_creg_memory(Base::cregs_, Base::state_index_begin_);

    for (uint_t i = 0; i < circ_opt.shots; i++) {
      uint_t iparam = i / Base::num_shots_per_bind_param_;
      (result_it + iparam)
          ->save_count_data(Base::cregs_[i], Base::save_creg_memory_);
    }
    Base::cregs_.clear();
  }
#endif

#ifdef AER_THRUST_CUDA
  if (Base::sim_device_ == Device::GPU) {
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      nDev = 0;
    }
    if (nDev > Base::num_groups_)
      nDev = Base::num_groups_;
    for (uint_t i = 0; i < Base::num_bind_params_; i++)
      (result_it + i)
          ->metadata.add(nDev, "batched_shots_optimization_parallel_gpus");
  }
#endif
}

template <class state_t>
template <typename InputIterator>
void BatchShotsExecutor<state_t>::apply_ops_batched_shots_for_group(
    int_t i_group, InputIterator first, InputIterator last,
    const Noise::NoiseModel &noise, ResultItr result_it,
    std::vector<RngEngine> &rng, bool final_ops) {
  uint_t istate = Base::top_state_of_group_[i_group];
#ifdef _OPENMP
  int num_inner_threads = omp_get_max_threads() / omp_get_num_threads();
#else
  int num_inner_threads = 1;
#endif

  for (auto op = first; op != last; ++op) {
    if (op->type == Operations::OpType::sample_noise) {
      if (op->expr) {
        for (uint_t j = Base::top_state_of_group_[i_group];
             j < Base::top_state_of_group_[i_group + 1]; j++) {
          Base::states_[j].qreg().enable_batch(false);
          Base::states_[j].qreg().read_measured_data(Base::states_[j].creg());
          std::vector<Operations::Op> nops = noise.sample_noise_loc(
              *op, rng[j - Base::top_state_of_group_[i_group]]);
          for (uint_t k = 0; k < nops.size(); k++) {
            Base::states_[j].apply_op(
                nops[k], *result_it,
                rng[j - Base::top_state_of_group_[i_group]], false);
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
        for (int_t j = 0; j < (int_t)count; j++) {
          noise_ops[j] = noise.sample_noise_loc(*op, rng[j]);

          if (!(noise_ops[j].size() == 0 ||
                (noise_ops[j].size() == 1 && noise_ops[j][0].name == "id"))) {
            count_ops++;
            for (uint_t k = 0; k < noise_ops[j].size(); k++) {
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
        for (uint_t j = 0; j < count; j++) {
          noise_ops[j] = noise.sample_noise_loc(*op, rng[j]);

          if (!(noise_ops[j].size() == 0 ||
                (noise_ops[j].size() == 1 && noise_ops[j][0].name == "id"))) {
            count_ops++;
            for (uint_t k = 0; k < noise_ops[j].size(); k++) {
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
      if (non_pauli_gate_count == 0) { // optimization for Pauli error
        Base::states_[istate].qreg().apply_batched_pauli_ops(noise_ops);
      } else {
        // otherwise execute each circuit
        apply_batched_noise_ops(i_group, noise_ops, result_it, rng);
      }
    } else {
      if (!op->expr && apply_batched_op(istate, *op, result_it, rng,
                                        final_ops && (op + 1 == last))) {
        continue;
      }
      // call apply_op for each state
      for (uint_t j = 0; j < Base::num_states_in_group_[i_group]; j++) {
        uint_t is = Base::top_state_of_group_[i_group] + j;
        uint_t ip =
            (Base::global_state_index_ + is) / Base::num_shots_per_bind_param_;
        Base::states_[is].qreg().enable_batch(false);
        Base::states_[is].qreg().read_measured_data(Base::states_[is].creg());
        Base::states_[is].apply_op(*op, *(result_it + ip), rng[j],
                                   final_ops && (op + 1 == last));
        Base::states_[is].qreg().enable_batch(true);
      }
    }
  }
}

template <class state_t>
void BatchShotsExecutor<state_t>::apply_batched_noise_ops(
    const int_t i_group, const std::vector<std::vector<Operations::Op>> &ops,
    ResultItr result_it, std::vector<RngEngine> &rng) {
  uint_t count;
  uint_t istate = Base::top_state_of_group_[i_group];
  count = ops.size();

  reg_t mask(count);
  std::vector<bool> finished(count, false);
  for (uint_t i = 0; i < count; i++) {
    int_t cond_reg = -1;

    if (finished[i])
      continue;
    if (ops[i].size() == 0 || (ops[i].size() == 1 && ops[i][0].name == "id")) {
      finished[i] = true;
      continue;
    }
    mask[i] = 1;

    // find same ops to be exectuted in a batch
    for (uint_t j = i + 1; j < count; j++) {
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
      for (uint_t k = 0; k < ops[i].size(); k++) {
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
    for (uint_t k = 0; k < ops[i].size(); k++) {
      Operations::Op cop = ops[i][k];

      // mark op conditional to mask shots
      cop.conditional = true;
      cop.conditional_reg = sys_reg;

      if (!apply_batched_op(istate, cop, result_it, rng, false)) {
        // call apply_op for each state
        for (uint_t j = 0; j < Base::num_states_in_group_[i_group]; j++) {
          uint_t is = Base::top_state_of_group_[i_group] + j;
          uint_t ip = (Base::global_state_index_ + is) /
                      Base::num_shots_per_bind_param_;
          Base::states_[is].qreg().enable_batch(false);
          Base::states_[is].qreg().read_measured_data(Base::states_[is].creg());
          Base::states_[is].apply_op(cop, *(result_it + ip), rng[j], false);
          Base::states_[is].qreg().enable_batch(true);
        }
      }
    }
    mask[i] = 0;
    finished[i] = true;
  }
}

template <class state_t>
void BatchShotsExecutor<state_t>::apply_batched_expval(const int_t istate,
                                                       const Operations::Op &op,
                                                       ResultItr result) {
  std::vector<double> val;
  bool variance = (op.type == Operations::OpType::save_expval_var);
  for (uint_t i = 0; i < op.expval_params.size(); i++) {
    std::complex<double> cprm;

    if (variance)
      cprm = std::complex<double>(std::get<1>(op.expval_params[i]),
                                  std::get<2>(op.expval_params[i]));
    else
      cprm = std::get<1>(op.expval_params[i]);
    bool last = (i == op.expval_params.size() - 1);

    Base::states_[istate].qreg().batched_expval_pauli(
        val, op.qubits, std::get<0>(op.expval_params[i]), variance, cprm, last);
  }

  if (val.size() == 0)
    return;

  if (variance) {
    for (uint_t i = 0; i < val.size() / 2; i++) {
      uint_t ip = (Base::global_state_index_ + istate + i) /
                  Base::num_shots_per_bind_param_;

      std::vector<double> expval_var(2);
      expval_var[0] = val[i * 2];                               // mean
      expval_var[1] = val[i * 2 + 1] - val[i * 2] * val[i * 2]; // variance
      (result + ip)
          ->save_data_average(Base::states_[istate + i].creg(),
                              op.string_params[0], expval_var, op.type,
                              op.save_type);
    }
  } else {
    for (uint_t i = 0; i < val.size(); i++) {
      uint_t ip = (Base::global_state_index_ + istate + i) /
                  Base::num_shots_per_bind_param_;

      (result + ip)
          ->save_data_average(Base::states_[istate + i].creg(),
                              op.string_params[0], val[i], op.type,
                              op.save_type);
    }
  }
}

template <class state_t>
template <typename InputIterator>
void BatchShotsExecutor<state_t>::batched_measure_sampler(
    InputIterator first_meas, InputIterator last_meas, uint_t shots,
    uint_t i_group, ResultItr result, std::vector<RngEngine> &rng) {
  uint_t par_states = 1;
  if ((uint_t)Base::max_parallel_threads_ >= Base::num_groups_ * 2) {
    par_states =
        std::min((uint_t)(Base::max_parallel_threads_ / Base::num_groups_),
                 Base::num_states_in_group_[i_group]);
  }

  // Check if meas_circ is empty, and if so return initial creg
  if (first_meas == last_meas) {
    return;
  }

  std::vector<Operations::Op> meas_ops;
  std::vector<Operations::Op> roerror_ops;
  for (auto op = first_meas; op != last_meas; op++) {
    if (op->type == Operations::OpType::roerror) {
      roerror_ops.push_back(*op);
    } else { /*(op.type == Operations::OpType::measure) */
      meas_ops.push_back(*op);
    }
  }

  // Get measured qubits from circuit sort and delete duplicates
  std::vector<uint_t> meas_qubits; // measured qubits
  for (const auto &op : meas_ops) {
    for (size_t j = 0; j < op.qubits.size(); ++j)
      meas_qubits.push_back(op.qubits[j]);
  }
  sort(meas_qubits.begin(), meas_qubits.end());
  meas_qubits.erase(unique(meas_qubits.begin(), meas_qubits.end()),
                    meas_qubits.end());

  // Make qubit map of position in vector of measured qubits
  std::unordered_map<uint_t, uint_t> qubit_map;
  for (uint_t j = 0; j < meas_qubits.size(); ++j) {
    qubit_map[meas_qubits[j]] = j;
  }

  // Maps of memory and register to qubit position
  std::map<uint_t, uint_t> memory_map;
  std::map<uint_t, uint_t> register_map;
  for (const auto &op : meas_ops) {
    for (size_t j = 0; j < op.qubits.size(); ++j) {
      auto pos = qubit_map[op.qubits[j]];
      if (!op.memory.empty())
        memory_map[op.memory[j]] = pos;
      if (!op.registers.empty())
        register_map[op.registers[j]] = pos;
    }
  }

  // Generate the samples
  auto timer_start = myclock_t::now();
  std::vector<double> rnd_shots(Base::num_states_in_group_[i_group] * shots);

  auto make_random_proc = [this, shots, &rnd_shots, par_states, i_group,
                           &rng](int_t i) {
    uint_t i_state, state_end;
    i_state = Base::num_states_in_group_[i_group] * i / par_states;
    state_end = Base::num_states_in_group_[i_group] * (i + 1) / par_states;

    for (; i_state < state_end; i_state++) {
      for (uint_t j = 0; j < shots; j++)
        rnd_shots[i_state * shots + j] =
            rng[i_state].rand(0, 1) + (double)i_state;
    }
  };
  Utils::apply_omp_parallel_for((par_states > 1), 0, par_states,
                                make_random_proc, par_states);

  reg_t allbit_samples =
      Base::states_[Base::top_state_of_group_[i_group]].qreg().sample_measure(
          rnd_shots);

  uint_t mask = (1ull << Base::num_qubits_) - 1;

  // Process samples
  uint_t num_memory =
      (memory_map.empty()) ? 0ULL : 1 + memory_map.rbegin()->first;
  uint_t num_registers =
      (register_map.empty()) ? 0ULL : 1 + register_map.rbegin()->first;

  auto save_counts_proc = [this, shots, par_states, i_group, num_memory,
                           num_registers, &result, &allbit_samples, memory_map,
                           register_map, &rng, mask, meas_qubits,
                           roerror_ops](int_t j) {
    uint_t i_state, state_end;
    i_state = Base::num_states_in_group_[i_group] * j / par_states;
    state_end = Base::num_states_in_group_[i_group] * (j + 1) / par_states;

    for (; i_state < state_end; i_state++) {
      uint_t is = Base::top_state_of_group_[i_group] + i_state;
      uint_t ip = (Base::global_state_index_ + is);

      for (uint_t i = 0; i < shots; i++) {
        ClassicalRegister creg;
        creg.initialize(num_memory, num_registers);
        reg_t all_samples(meas_qubits.size());

        uint_t val = allbit_samples[i_state * shots + i] & mask;
        reg_t allbit_sample = Utils::int2reg(val, 2, Base::num_qubits_);
        for (uint_t mq = 0; mq < meas_qubits.size(); mq++) {
          all_samples[mq] = allbit_sample[meas_qubits[mq]];
        }

        // process memory bit measurements
        for (const auto &pair : memory_map) {
          creg.store_measure(reg_t({all_samples[pair.second]}),
                             reg_t({pair.first}), reg_t());
        }
        // process register bit measurements
        for (const auto &pair : register_map) {
          creg.store_measure(reg_t({all_samples[pair.second]}), reg_t(),
                             reg_t({pair.first}));
        }

        // process read out errors for memory and registers
        for (const Operations::Op &roerror : roerror_ops)
          creg.apply_roerror(roerror, rng[i_state]);

        // Save count data
        if (Base::num_process_per_experiment_ > 1)
          Base::cregs_[ip * shots + i] = creg;
        else
          (result + ip)->save_count_data(creg, Base::save_creg_memory_);
      }
    }
  };
  Utils::apply_omp_parallel_for((par_states > 1), 0, par_states,
                                save_counts_proc, par_states);

  auto time_taken =
      std::chrono::duration<double>(myclock_t::now() - timer_start).count();

  for (uint_t i_state = 0; i_state < Base::num_states_in_group_[i_group];
       i_state++) {
    uint_t ip = Base::global_state_index_ + Base::top_state_of_group_[i_group] +
                i_state;
    (result + ip)->metadata.add(time_taken, "sample_measure_time");
    (result + ip)->metadata.add(true, "measure_sampling");
  }
}

//-------------------------------------------------------------------------
} // end namespace CircuitExecutor
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
