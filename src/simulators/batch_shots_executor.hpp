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

#include "simulators/multi_shots_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace Executor {

//-------------------------------------------------------------------------
// batched-shots executor class implementation
//-------------------------------------------------------------------------
template <class state_t>
class BatchShotsExecutor : public MultiShotsExecutor<state_t> {
  using BaseExecutor = MultiShotsExecutor<state_t>;
protected:
   //config setting for multi-shot parallelization
  bool batched_shots_gpu_ = true;
  int_t batched_shots_gpu_max_qubits_ = 16;   //multi-shot parallelization is applied if qubits is less than max qubits
  bool enable_batch_multi_shots_ = false;   //multi-shot parallelization can be applied
public:
  BatchShotsExecutor();
  virtual ~BatchShotsExecutor();

protected:
  void set_config(const json_t &config) override;
  void set_parallelization(const Circuit &circ,
                           const Noise::NoiseModel &noise) override;

  bool allocate_states(uint_t num_shots) override;

  void run_circuit_shots(
            Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
            ExperimentResult &result, bool sample_noise) override;

  //run circuit by using batched shots optimization (on GPU)
  void run_circuit_with_batched_multi_shots(Circuit &circ,
                                         const Noise::NoiseModel &noise,
                                         const json_t &config,
                                         ExperimentResult &result);

  //apply ops for multi-shots to one group
  template <typename InputIterator>
  void apply_ops_batched_shots_for_group(int_t i_group, 
                               InputIterator first, InputIterator last,
                               const Noise::NoiseModel &noise,
                               ExperimentResult &result,
                               uint_t rng_seed,
                               bool final_ops);

  //apply op to multiple shots , return flase if op is not supported to execute in a batch
  virtual bool apply_batched_op(const int_t istate, const Operations::Op &op,
                                ExperimentResult &result,
                                std::vector<RngEngine> &rng,
                                bool final_op = false){return false;}

  //apply sampled noise to multiple-shots (this is used for ops contains non-Pauli operators)
  void apply_batched_noise_ops(const int_t i_group, const std::vector<std::vector<Operations::Op>> &ops, 
                               ExperimentResult &result,
                               std::vector<RngEngine> &rng);
};


template <class state_t>
BatchShotsExecutor<state_t>::BatchShotsExecutor()
{
}

template <class state_t>
BatchShotsExecutor<state_t>::~BatchShotsExecutor()
{
}

template <class state_t>
void BatchShotsExecutor<state_t>::set_config(const json_t &config) 
{
  BaseExecutor::set_config(config);

  //enable batched multi-shots/experiments optimization
  if(JSON::check_key("batched_shots_gpu", config)) {
    JSON::get_value(batched_shots_gpu_, "batched_shots_gpu", config);
  }

  if(JSON::check_key("batched_shots_gpu_max_qubits", config)) {
    JSON::get_value(batched_shots_gpu_max_qubits_, "batched_shots_gpu_max_qubits", config);
  }
  if(BaseExecutor::method_ == Method::density_matrix || BaseExecutor::method_ == Method::unitary)
    batched_shots_gpu_max_qubits_ /= 2;
}


template <class state_t>
void BatchShotsExecutor<state_t>::set_parallelization(const Circuit &circ,
                                                      const Noise::NoiseModel &noise)
{
  enable_batch_multi_shots_ = false;
  if(batched_shots_gpu_ && BaseExecutor::sim_device_ != Device::CPU){
    enable_batch_multi_shots_ = true;
    if(BaseExecutor::cuStateVec_enable_)
      enable_batch_multi_shots_ = false;
    if(circ.num_qubits >= batched_shots_gpu_max_qubits_)
      enable_batch_multi_shots_ = false;
    if(circ.shots == 1)
      enable_batch_multi_shots_ = false;
  }

  if(!enable_batch_multi_shots_)
    BaseExecutor::set_parallelization(circ, noise);
}

template <class state_t>
bool BatchShotsExecutor<state_t>::allocate_states(uint_t num_shots)
{
  int_t i;
  bool init_states = true;
  bool ret = true;
  //deallocate qregs before reallocation
  if(BaseExecutor::states_.size() > 0){
    if(BaseExecutor::states_.size() == num_shots)
      init_states = false;  //can reuse allocated chunks
    else
      BaseExecutor::states_.clear();
  }
  if(init_states){
    BaseExecutor::states_.resize(num_shots);

    if(BaseExecutor::num_creg_memory_ !=0 || BaseExecutor::num_creg_registers_ !=0){
      for(i=0;i<num_shots;i++){
        //set number of creg bits before actual initialization
        BaseExecutor::states_[i].initialize_creg(BaseExecutor::num_creg_memory_, BaseExecutor::num_creg_registers_);
      }
    }

    //allocate qregs
    BaseExecutor::states_[0].qreg().set_max_matrix_bits(BaseExecutor::max_matrix_qubits_);
    BaseExecutor::states_[0].qreg().set_num_threads_per_group(BaseExecutor::num_threads_per_group_);
    BaseExecutor::states_[0].qreg().cuStateVec_enable(BaseExecutor::cuStateVec_enable_);
    BaseExecutor::states_[0].set_num_global_qubits(BaseExecutor::num_qubits_);
    ret &= BaseExecutor::states_[0].qreg().chunk_setup(BaseExecutor::num_qubits_*this->qubit_scale(), BaseExecutor::num_qubits_*this->qubit_scale(), 0, num_shots);
    for(i=1;i<num_shots;i++){
      ret &= BaseExecutor::states_[i].qreg().chunk_setup(BaseExecutor::states_[0].qreg(),0);
      BaseExecutor::states_[i].qreg().set_num_threads_per_group(BaseExecutor::num_threads_per_group_);
      BaseExecutor::states_[i].set_num_global_qubits(BaseExecutor::num_qubits_);
    }
  }
  BaseExecutor::num_active_shots_ = num_shots;

  //initialize groups
  BaseExecutor::top_shot_of_group_.clear();
  BaseExecutor::num_groups_ = 0;
  for(i=0;i<num_shots;i++){
    if(BaseExecutor::states_[i].qreg().top_of_group()){
      BaseExecutor::top_shot_of_group_.push_back(i);
      BaseExecutor::num_groups_++;
    }
  }
  BaseExecutor::top_shot_of_group_.push_back(num_shots);
  BaseExecutor::num_shots_in_group_.resize(BaseExecutor::num_groups_);
  for(i=0;i<BaseExecutor::num_groups_;i++){
    BaseExecutor::num_shots_in_group_[i] = BaseExecutor::top_shot_of_group_[i+1] - BaseExecutor::top_shot_of_group_[i];
  }

  return ret;
}

template <class state_t>
void BatchShotsExecutor<state_t>::run_circuit_shots(
    Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
    ExperimentResult &result, bool sample_noise) 
{
  state_t dummy_state;
  if(!enable_batch_multi_shots_){   //if batched-shot is not applicable, use default shot distribution
    return BaseExecutor::run_circuit_shots(circ, noise, config, result, sample_noise);
  }

  Noise::NoiseModel dummy_noise;

  BaseExecutor::num_qubits_ = circ.num_qubits;
  BaseExecutor::num_creg_memory_ = circ.num_memory;
  BaseExecutor::num_creg_registers_ = circ.num_registers;

  if(BaseExecutor::sim_device_ == Device::GPU){
#ifdef _OPENMP
    if(omp_get_num_threads() == 1)
      BaseExecutor::shot_omp_parallel_ = true;
#endif
  }
  else if(BaseExecutor::sim_device_ == Device::ThrustCPU){
    BaseExecutor::shot_omp_parallel_ = false;
  }

  BaseExecutor::set_distribution(BaseExecutor::num_process_per_experiment_, circ.shots);
  BaseExecutor::num_max_shots_ = BaseExecutor::get_max_parallel_shots(circ, noise);
  if(BaseExecutor::num_max_shots_ == 0)
    BaseExecutor::num_max_shots_ = 1;

  RngEngine rng;
  rng.set_seed(circ.seed);

  Circuit circ_opt;
  if(sample_noise)
    circ_opt = noise.sample_noise(circ, rng, Noise::NoiseModel::Method::circuit, true);
  else
    circ_opt = circ;
  auto fusion_pass = Base<state_t>::transpile_fusion( circ_opt.opset(), config);

  fusion_pass.optimize_circuit(circ_opt, dummy_noise, dummy_state.opset(), result);
  BaseExecutor::max_matrix_qubits_ = BaseExecutor::get_max_matrix_qubits(circ_opt);

  run_circuit_with_batched_multi_shots(circ_opt, noise, config, result);

  // Add batched multi-shots optimizaiton metadata
  result.metadata.add(true, "batched_shots_optimization");
}

template <class state_t>
void BatchShotsExecutor<state_t>::run_circuit_with_batched_multi_shots(Circuit &circ,
                                       const Noise::NoiseModel &noise,
                                       const json_t &config,
                                       ExperimentResult &result)
{
  int_t i;
  int_t i_begin,n_shots;

  i_begin = 0;
  while(i_begin < BaseExecutor::num_local_shots_){
    BaseExecutor::local_shot_index_ = i_begin;

    //loop for states can be stored in available memory
    n_shots = std::min(BaseExecutor::num_local_shots_, BaseExecutor::num_max_shots_);
    if(i_begin+n_shots > BaseExecutor::num_local_shots_){
      n_shots = BaseExecutor::num_local_shots_ - i_begin;
    }

    //allocate shots
    allocate_states(n_shots);

    // Set state config
    for(i=0;i<n_shots;i++){
      BaseExecutor::states_[i].set_config(config);
      BaseExecutor::states_[i].set_parallelization(BaseExecutor::parallel_state_update_);
      BaseExecutor::states_[i].set_global_phase(circ.global_phase_angle);
    }
    this->set_global_phase(circ.global_phase_angle);

    //initialization (equivalent to initialize_qreg + initialize_creg)
    auto init_group = [this](int_t ig){
      for(uint_t j=BaseExecutor::top_shot_of_group_[ig];j<BaseExecutor::top_shot_of_group_[ig+1];j++){
        //enabling batch shots optimization
        BaseExecutor::states_[j].qreg().enable_batch(true);

        //initialize qreg here
        BaseExecutor::states_[j].qreg().set_num_qubits(BaseExecutor::num_qubits_);
        BaseExecutor::states_[j].qreg().initialize();

        //initialize creg here
        BaseExecutor::states_[j].qreg().initialize_creg(BaseExecutor::num_creg_memory_, BaseExecutor::num_creg_registers_);
      }
    };
    Utils::apply_omp_parallel_for((BaseExecutor::num_groups_ > 1 && BaseExecutor::shot_omp_parallel_),0,BaseExecutor::num_groups_,init_group);

    this->apply_global_phase(); //this is parallelized in sub-classes

    //apply ops to multiple-shots
    if(BaseExecutor::num_groups_ > 1 && BaseExecutor::shot_omp_parallel_){
      std::vector<ExperimentResult> par_results(BaseExecutor::num_groups_);
#pragma omp parallel for num_threads(BaseExecutor::num_groups_)
      for(i=0;i<BaseExecutor::num_groups_;i++)
        apply_ops_batched_shots_for_group(i, circ.ops.cbegin(), circ.ops.cend(), noise, par_results[i], circ.seed, true);

      for (auto &res : par_results)
        result.combine(std::move(res));
    }
    else{
      for(i=0;i<BaseExecutor::num_groups_;i++)
        apply_ops_batched_shots_for_group(i, circ.ops.cbegin(), circ.ops.cend(), noise, result, circ.seed, true);
    }

    //collect measured bits and copy memory
    for(i=0;i<n_shots;i++){
      BaseExecutor::states_[i].qreg().read_measured_data(BaseExecutor::states_[i].creg());
      result.save_count_data(BaseExecutor::states_[i].creg(), BaseExecutor::save_creg_memory_);
    }

    i_begin += n_shots;
  }
  /*
  BaseExecutor::gather_creg_memory();

  for(i=0;i<BaseExecutor::num_global_shots_;i++)
    result.save_count_data(BaseExecutor::states_[i].creg(), BaseExecutor::save_creg_memory_);
  */

#ifdef AER_THRUST_CUDA
  if(BaseExecutor::sim_device_name_ == "GPU"){
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      nDev = 0;
    }
    if(nDev > BaseExecutor::num_groups_)
      nDev = BaseExecutor::num_groups_;
    result.metadata.add(nDev,"batched_shots_optimization_parallel_gpus");
  }
#endif
}

template <class state_t>
template <typename InputIterator>
void BatchShotsExecutor<state_t>::apply_ops_batched_shots_for_group(int_t i_group, 
                               InputIterator first, InputIterator last,
                               const Noise::NoiseModel &noise,
                               ExperimentResult &result,
                               uint_t rng_seed,
                               bool final_ops)
{
  uint_t istate = BaseExecutor::top_shot_of_group_[i_group];
  std::vector<RngEngine> rng(BaseExecutor::num_shots_in_group_[i_group]);
#ifdef _OPENMP
  int num_inner_threads = omp_get_max_threads() / omp_get_num_threads();
#else
  int num_inner_threads = 1;
#endif

  for(uint_t j=BaseExecutor::top_shot_of_group_[i_group];j<BaseExecutor::top_shot_of_group_[i_group+1];j++)
    rng[j-BaseExecutor::top_shot_of_group_[i_group]].set_seed(rng_seed + BaseExecutor::global_shot_index_ + BaseExecutor::local_shot_index_ + j);

  for (auto op = first; op != last; ++op) {
    if(op->type == Operations::OpType::sample_noise){
      //sample error here
      uint_t count = BaseExecutor::num_shots_in_group_[i_group];
      std::vector<std::vector<Operations::Op>> noise_ops(count);

      uint_t count_ops = 0;
      uint_t non_pauli_gate_count = 0;
      if(num_inner_threads > 1){
#pragma omp parallel for reduction(+: count_ops,non_pauli_gate_count) num_threads(num_inner_threads)
        for(int_t j=0;j<count;j++){
          noise_ops[j] = noise.sample_noise_loc(*op,rng[j]);

          if(!(noise_ops[j].size() == 0 || (noise_ops[j].size() == 1 && noise_ops[j][0].name == "id"))){
            count_ops++;
            for(int_t k=0;k<noise_ops[j].size();k++){
              if(noise_ops[j][k].name != "id" && noise_ops[j][k].name != "x" && noise_ops[j][k].name != "y" && noise_ops[j][k].name != "z" && noise_ops[j][k].name != "pauli"){
                non_pauli_gate_count++;
                break;
              }
            }
          }
        }
      }
      else{
        for(int_t j=0;j<count;j++){
          noise_ops[j] = noise.sample_noise_loc(*op,rng[j]);

          if(!(noise_ops[j].size() == 0 || (noise_ops[j].size() == 1 && noise_ops[j][0].name == "id"))){
            count_ops++;
            for(int_t k=0;k<noise_ops[j].size();k++){
              if(noise_ops[j][k].name != "id" && noise_ops[j][k].name != "x" && noise_ops[j][k].name != "y" && noise_ops[j][k].name != "z" && noise_ops[j][k].name != "pauli"){
                non_pauli_gate_count++;
                break;
              }
            }
          }
        }
      }

      if(count_ops == 0){
        continue;   //do nothing
      }
      if(non_pauli_gate_count == 0){   //ptimization for Pauli error
        BaseExecutor::states_[istate].qreg().apply_batched_pauli_ops(noise_ops);
      }
      else{
        //otherwise execute each circuit
        apply_batched_noise_ops(i_group, noise_ops,result, rng);
      }
    }
    else{
      if(!apply_batched_op(istate, *op, result, rng, final_ops && (op + 1 == last))){
        //call apply_op for each state
        for(uint_t j=BaseExecutor::top_shot_of_group_[i_group];j<BaseExecutor::top_shot_of_group_[i_group+1];j++){
          BaseExecutor::states_[j].qreg().enable_batch(false);
          BaseExecutor::states_[j].qreg().read_measured_data(BaseExecutor::states_[j].creg());
          BaseExecutor::states_[j].apply_op( *op, result, rng[j-BaseExecutor::top_shot_of_group_[i_group]], final_ops && (op + 1 == last) );
          BaseExecutor::states_[j].qreg().enable_batch(true);
        }
      }
    }
  }
}

template <class state_t>
void BatchShotsExecutor<state_t>::apply_batched_noise_ops(const int_t i_group, const std::vector<std::vector<Operations::Op>> &ops, 
                             ExperimentResult &result,
                             std::vector<RngEngine> &rng)
{
  int_t i,j,k,count,nop,pos = 0;
  uint_t istate = BaseExecutor::top_shot_of_group_[i_group];
  count = ops.size();

  reg_t mask(count);
  std::vector<bool> finished(count,false);
  for(i=0;i<count;i++){
    int_t cond_reg = -1;

    if(finished[i])
      continue;
    if(ops[i].size() == 0 || (ops[i].size() == 1 && ops[i][0].name == "id")){
      finished[i] = true;
      continue;
    }
    mask[i] = 1;

    //find same ops to be exectuted in a batch
    for(j=i+1;j<count;j++){
      if(finished[j]){
        mask[j] = 0;
        continue;
      }
      if(ops[j].size() == 0 || (ops[j].size() == 1 && ops[j][0].name == "id")){
        mask[j] = 0;
        finished[j] = true;
        continue;
      }

      if(ops[i].size() != ops[j].size()){
        mask[j] = 0;
        continue;
      }

      mask[j] = true;
      for(k=0;k<ops[i].size();k++){
        if(ops[i][k].conditional){
          cond_reg = ops[i][k].conditional_reg;
        }
        if(ops[i][k].type != ops[j][k].type || ops[i][k].name != ops[j][k].name){
          mask[j] = false;
          break;
        }
      }
      if(mask[j])
        finished[j] = true;
    }

    //mask conditional register
    int_t sys_reg = BaseExecutor::states_[istate].qreg().set_batched_system_conditional(cond_reg, mask);

    //batched execution on same ops
    for(k=0;k<ops[i].size();k++){
      Operations::Op cop = ops[i][k];

      //mark op conditional to mask shots
      cop.conditional = true;
      cop.conditional_reg = sys_reg;

      if(!apply_batched_op(istate, cop, result,rng, false)){
        //call apply_op for each state
        /*if(cop.conditional){
          //copy creg to local state
          reg_t reg_pos(1);
          reg_t mem_pos;
          int bit = BaseExecutor::states_[j].qreg().measured_cregister(cop.conditional_reg);
          const reg_t reg = Utils::int2reg(bit, 2, 1);
          reg_pos[0] = cop.conditional_reg;
          BaseExecutor::states_[j].creg().store_measure(reg, mem_pos, reg_pos);
        }*/
        for(uint_t j=BaseExecutor::top_shot_of_group_[i_group];j<BaseExecutor::top_shot_of_group_[i_group+1];j++){
          BaseExecutor::states_[j].qreg().enable_batch(false);
          BaseExecutor::states_[j].apply_op(cop, result ,rng[j-BaseExecutor::top_shot_of_group_[i_group]],false);
          BaseExecutor::states_[j].qreg().enable_batch(true);
        }
      }
    }
    mask[i] = 0;
    finished[i] = true;
  }
}


//-------------------------------------------------------------------------
} // end namespace Executor
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif



