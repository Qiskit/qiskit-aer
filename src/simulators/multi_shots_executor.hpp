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

#ifndef _multi_shots_executor_hpp_
#define _multi_shots_executor_hpp_

#include "simulators/aer_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

#include "simulators/shot_branching.hpp"

namespace AER {

namespace Executor {


//-------------------------------------------------------------------------
// Multiple-shots executor class implementation
//-------------------------------------------------------------------------
template <class state_t>
class MultiShotsExecutor : public Base<state_t> {
  using BaseExecutor = Base<state_t>;
protected:
  std::vector<state_t> states_;
  std::vector<ClassicalRegister> cregs_;      //classical registers for all shots

  //number of qubits for the circuit
  uint_t num_qubits_;

  uint_t num_global_shots_;    //number of total shots
  uint_t num_local_shots_;     //number of local shots

  uint_t global_shot_index_;   //beginning chunk index for this process
  reg_t shot_index_begin_;     //beginning chunk index for each process
  reg_t shot_index_end_;       //ending chunk index for each process
  uint_t local_shot_index_;    //local shot ID of current loop
  uint_t num_active_shots_;    //number of active shots in current loop

  uint_t myrank_;               //process ID
  uint_t nprocs_;               //number of processes
  uint_t distributed_rank_;     //process ID in communicator group
  uint_t distributed_procs_;    //number of processes in communicator group
  uint_t distributed_group_;    //group id of distribution
  int_t distributed_proc_bits_; //distributed_procs_=2^distributed_proc_bits_  (if nprocs != power of 2, set -1)

  bool shot_omp_parallel_;     //using thread parallel to process loop of chunks or not

  bool set_parallelization_called_ = false;    //this flag is used to check set_parallelization is already called, if yes the call sets max_batched_shots_
  uint_t num_max_shots_ = 1;    //max number of shots can be stored on available memory

  int max_matrix_qubits_;       //max qubits for matrix

  //shot branching
  bool shot_branching_enable_ = true;
  bool runtime_noise_sampling_enable_ = false;

  //group of states (GPU devices)
  uint_t num_groups_;            //number of groups of chunks
  reg_t top_shot_of_group_;
  reg_t num_shots_in_group_;
  int num_threads_per_group_;   //number of outer threads per group

  uint_t num_creg_memory_ = 0;    //number of total bits for creg (reserve for multi-shots)
  uint_t num_creg_registers_ = 0;

  //cuStateVec settings
  bool cuStateVec_enable_ = false;

  #ifdef AER_MPI
  //communicator group to simulate a circuit (for multi-experiments)
  MPI_Comm distributed_comm_;
#endif

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 14;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  // Set a global phase exp(1j * theta) for the state
  bool has_global_phase_ = false;
  complex_t global_phase_ = 1;
public:
  MultiShotsExecutor();
  virtual ~MultiShotsExecutor();

  size_t required_memory_mb(const Circuit &circuit,
                            const Noise::NoiseModel &noise) const override
  {
    state_t tmp;
    return tmp.required_memory_mb(circuit.num_qubits, circuit.ops);
  }

  uint_t get_process_by_chunk(uint_t cid);
protected:
  void set_config(const json_t &config) override;

  void set_distribution(uint_t nprocs, uint_t num_parallel_shots);

  virtual uint_t qubit_scale(void)
  {
    return 1;
  }

  virtual bool allocate_states(uint_t num_shots);

  void run_circuit_shots(
            Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
            ExperimentResult &result, bool sample_noise) override;

  void run_circuit_with_shot_branching(
            Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
            ExperimentResult &result, bool sample_noise);

  //apply op for shot-branching, return false if op is not applied in sub-class
  virtual bool apply_branching_op(Executor::Branch& root, 
                                  const Operations::Op &op,
                                  ExperimentResult &result,
                                  bool final_op)
  {
    std::cout << "  base is called, implement for each method" << std::endl;
    return false;
  }

  // Apply the global phase
  virtual void apply_global_phase(){}
  void set_global_phase(double theta);

  //gather cregs over processes
  void gather_creg_memory(void);

  void set_parallelization(const Circuit &circ,
                           const Noise::NoiseModel &noise) override;

  virtual bool shot_branching_supported(void)
  {
    return false;   //return true in the sub-class if supports shot-branching
  }
};

template <class state_t>
MultiShotsExecutor<state_t>::MultiShotsExecutor()
{
  num_global_shots_ = 0;
  num_local_shots_ = 0;

  myrank_ = 0;
  nprocs_ = 1;

  distributed_procs_ = 1;
  distributed_rank_ = 0;
  distributed_group_ = 0;
  distributed_proc_bits_ = 0;

  shot_omp_parallel_ = false;

  shot_branching_enable_ = false;

#ifdef AER_MPI
  distributed_comm_ = MPI_COMM_WORLD;
#endif
}

template <class state_t>
MultiShotsExecutor<state_t>::~MultiShotsExecutor()
{
  states_.clear();
  cregs_.clear();
}

template <class state_t>
void MultiShotsExecutor<state_t>::set_config(const json_t &config) 
{
  Base<state_t>::set_config(config);

  // Set threshold for truncating states to be saved
  JSON::get_value(json_chop_threshold_, "zero_threshold", config);

  // Set OMP threshold for state update functions
  JSON::get_value(omp_qubit_threshold_, "statevector_parallel_threshold", config);

  num_threads_per_group_ = 1;
  if(JSON::check_key("num_threads_per_device", config)) {
    JSON::get_value(num_threads_per_group_, "num_threads_per_device", config);
  }

  //shot branching optimization
  if(JSON::check_key("shot_branching_enable", config)) {
    JSON::get_value(shot_branching_enable_, "shot_branching_enable", config);
  }
  if(JSON::check_key("runtime_noise_sampling_enable", config)) {
    JSON::get_value(runtime_noise_sampling_enable_, "runtime_noise_sampling_enable", config);
  }

}

template <class state_t>
void MultiShotsExecutor<state_t>::set_global_phase(double theta) {
  if (Linalg::almost_equal(theta, 0.0)) {
    has_global_phase_ = false;
    global_phase_ = 1;
  }
  else {
    has_global_phase_ = true;
    global_phase_ = std::exp(complex_t(0.0, theta));
  }
}

template <class state_t>
void MultiShotsExecutor<state_t>::set_distribution(uint_t nprocs, uint_t num_parallel_shots)
{
  myrank_ = 0;
  nprocs_ = 1;
#ifdef AER_MPI
  int t;
  MPI_Comm_size(MPI_COMM_WORLD,&t);
  nprocs_ = t;
  MPI_Comm_rank(MPI_COMM_WORLD,&t);
  myrank_ = t;
#endif

  distributed_procs_ = nprocs;
  distributed_rank_ = myrank_ % nprocs;
  distributed_group_ = myrank_ / nprocs;

  distributed_proc_bits_ = 0;
  int proc_bits = 0;
  uint_t p = distributed_procs_;
  while(p > 1){
    if((p & 1) != 0){   //procs is not power of 2
      distributed_proc_bits_ = -1;
      break;
    }
    distributed_proc_bits_++;
    p >>= 1;
  }

#ifdef AER_MPI
  if(nprocs != nprocs_){
    MPI_Comm_split(MPI_COMM_WORLD,(int)distributed_group_,(int)distributed_rank_,&distributed_comm_);
  }
  else{
    distributed_comm_ = MPI_COMM_WORLD;
  }
#endif

  num_global_shots_ = num_parallel_shots;

  shot_index_begin_.resize(distributed_procs_);
  shot_index_end_.resize(distributed_procs_);
  for(int_t i=0;i<distributed_procs_;i++){
    shot_index_begin_[i] = num_global_shots_*i / distributed_procs_;
    shot_index_end_[i] = num_global_shots_*(i+1) / distributed_procs_;
  }

  num_local_shots_ = shot_index_end_[distributed_rank_] - shot_index_begin_[distributed_rank_];
  global_shot_index_ = shot_index_begin_[distributed_rank_];
  local_shot_index_ = 0;
}

template <class state_t>
void MultiShotsExecutor<state_t>::set_parallelization(const Circuit &circ,
                                                      const Noise::NoiseModel &noise)
{
  BaseExecutor::set_parallelization(circ, noise);
}

template <class state_t>
bool MultiShotsExecutor<state_t>::allocate_states(uint_t num_shots)
{
  int_t i;
  bool ret = true;

  states_.resize(num_shots);

  num_active_shots_ = num_shots;

  //initialize groups
  top_shot_of_group_.resize(num_shots);
  num_shots_in_group_.resize(num_shots);
  num_groups_ = num_shots;
  for(i=0;i<num_shots;i++){
    top_shot_of_group_[i] = i;
    num_shots_in_group_[i] = 1;
  }

  return ret;
}

template <class state_t>
void MultiShotsExecutor<state_t>::run_circuit_shots(
    Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
    ExperimentResult &result, bool sample_noise)
{
  num_qubits_ = circ.num_qubits;
  num_creg_memory_ = circ.num_memory;
  num_creg_registers_ = circ.num_registers;

  if(this->sim_device_ == Device::GPU){
#ifdef _OPENMP
    if(omp_get_num_threads() == 1)
      shot_omp_parallel_ = true;
#endif
  }
  else if(this->sim_device_ == Device::ThrustCPU){
    shot_omp_parallel_ = false;
  }

  set_distribution(BaseExecutor::num_process_per_experiment_, circ.shots);
  num_max_shots_ = BaseExecutor::get_max_parallel_shots(circ, noise);
  if(num_max_shots_ == 0)
    num_max_shots_ = 1;

  bool shot_branching = false;
  if(shot_branching_enable_ && num_local_shots_ > 1 && shot_branching_supported()){
    shot_branching = true;
    if(sample_noise){
      if(!runtime_noise_sampling_enable_)
        shot_branching = false;
    }
  }
  else
    shot_branching = false;

  if(shot_branching){
    return run_circuit_with_shot_branching(circ, noise, config, result, sample_noise);
  }

  //insert runtime noise sample ops here
  RngEngine rng;
  rng.set_seed(circ.seed);

  int_t par_shots = std::min((int_t)Base<state_t>::parallel_shots_, (int_t)num_max_shots_);
  std::vector<ExperimentResult> par_results(par_shots);

  auto run_circuit_lambda = [this,&par_results,circ,noise,config,par_shots,sample_noise](int_t i){
    Noise::NoiseModel dummy_noise;
    state_t state;
    uint_t i_shot,shot_end;
    i_shot = num_local_shots_*i/par_shots;
    shot_end = num_local_shots_*(i+1)/par_shots;

    auto fusion_pass = Base<state_t>::transpile_fusion( circ.opset(), config);

    // Set state config
    state.set_config(config);
    state.set_parallelization(this->parallel_state_update_);
    state.set_global_phase(circ.global_phase_angle);

    state.set_distribution(this->num_process_per_experiment_);
    state.set_num_global_qubits(num_qubits_);

    for(;i_shot<shot_end;i_shot++){
      RngEngine rng;
      rng.set_seed(circ.seed + global_shot_index_ + i_shot);

      int max_matrix_qubits;
      Circuit circ_opt;
      if(sample_noise)
        circ_opt = noise.sample_noise(circ, rng);
      else
        circ_opt = circ;
      fusion_pass.optimize_circuit(circ_opt, dummy_noise, state.opset(), par_results[i]);
      max_matrix_qubits = BaseExecutor::get_max_matrix_qubits(circ_opt);
      state.set_max_matrix_qubits(max_matrix_qubits);

      state.allocate(num_qubits_, num_qubits_);
      state.initialize_qreg(num_qubits_);
      state.initialize_creg(circ.num_memory, circ.num_registers);

      state.apply_ops(circ_opt.ops.cbegin(), circ_opt.ops.cend(), par_results[i], rng, true);
      par_results[i].save_count_data(state.creg(), BaseExecutor::save_creg_memory_);
    }
    state.add_metadata(par_results[i]);
  };
  Utils::apply_omp_parallel_for((par_shots > 1),0,par_shots,run_circuit_lambda);

  for (auto &res : par_results) {
    result.combine(std::move(res));
  }
  if (BaseExecutor::sim_device_ == Device::GPU){
    if(par_shots >= BaseExecutor::num_gpus_)
      result.metadata.add(BaseExecutor::num_gpus_, "gpu_parallel_shots_");
    else
      result.metadata.add(par_shots, "gpu_parallel_shots_");
  }
}

template <class state_t>
void MultiShotsExecutor<state_t>::run_circuit_with_shot_branching(
          Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
          ExperimentResult &result, bool sample_noise)
{
  std::vector<std::shared_ptr<Branch>> branches;
  Noise::NoiseModel dummy_noise;
  state_t dummy_state;
  RngEngine dummy_rng;
  dummy_rng.set_seed(circ.seed);   //this is not used actually
  OpItr first;
  OpItr last;

  Circuit circ_opt;
  if(sample_noise){
    circ_opt = noise.sample_noise(circ, dummy_rng, Noise::NoiseModel::Method::circuit, true);
    auto fusion_pass = Base<state_t>::transpile_fusion( circ_opt.opset(), config);
    fusion_pass.optimize_circuit(circ_opt, dummy_noise, dummy_state.opset(), result);
    first = circ_opt.ops.cbegin();
    last = circ_opt.ops.cend();
    max_matrix_qubits_ = BaseExecutor::get_max_matrix_qubits(circ_opt);
  }
  else{
    auto fusion_pass = Base<state_t>::transpile_fusion( circ.opset(), config);
    fusion_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(), result);
    first = circ.ops.cbegin();
    last = circ.ops.cend();
    max_matrix_qubits_ = BaseExecutor::get_max_matrix_qubits(circ);
  }

  //check if there is sequence of measure at the end of operations
  bool can_sample = false;
  OpItr measure_seq = last;
  OpItr it = last - 1;
  int_t num_measure = 0;

  do{
    if(it->type != Operations::OpType::measure){
      measure_seq = it + 1;
      break;
    }
    num_measure += it->qubits.size();
    it--;
  }while(it != first);

  if(num_measure >= num_qubits_ && measure_seq != last){
    can_sample = true;
  }
  else{
    measure_seq = last;
  }

  cregs_.resize(circ.shots);

  //reserve states
  allocate_states(num_max_shots_);

  //initialize local shots
  std::vector<RngEngine> shots_storage(num_local_shots_);
  for(int_t i=0;i<num_local_shots_;i++){
    shots_storage[i].set_seed(circ.seed + global_shot_index_ + i);
  }

  int_t par_shots = std::min((int_t)Base<state_t>::parallel_shots_, (int_t)num_max_shots_);
  std::vector<ExperimentResult> par_results(par_shots);

  uint_t num_shots_saved = 0;

  //loop until all local shots are simulated
  while(shots_storage.size() > 0){
    uint_t num_active_states = 1;

    //initial state
    branches.push_back(std::make_shared<Branch>());
    branches[0]->state_index() = 0;
    branches[0]->set_shots(shots_storage);
    branches[0]->op_iterator() = first;
    shots_storage.clear();

    //initialize initial state
    states_[0].initialize_qreg(num_qubits_);
    states_[0].initialize_creg(num_creg_memory_, num_creg_registers_);

    //functor for ops execution
    auto apply_ops_func = [this, &branches, &dummy_rng, &noise, &par_results, measure_seq, par_shots, num_active_states](int_t i)
    {
      uint_t istate,state_end;
      istate = branches.size()*i/par_shots;
      state_end = branches.size()*(i+1)/par_shots;
      uint_t nbranch = 0;

      for(;istate<state_end;istate++){
        while(branches[istate]->op_iterator() != measure_seq){
          if(!branches[istate]->apply_control_flow(states_[branches[istate]->state_index()].creg(), measure_seq)){
            if(!branches[istate]->apply_runtime_noise_sampling(states_[branches[istate]->state_index()].creg(), *branches[istate]->op_iterator(), noise)){
              if(!apply_branching_op(*branches[istate], *branches[istate]->op_iterator(), par_results[i], true)){
                states_[branches[istate]->state_index()].apply_op(*branches[istate]->op_iterator(), par_results[i], dummy_rng, true);
              }
            }
            branches[istate]->advance_iterator();
            if(branches[istate]->num_branches() > 0){
              nbranch += branches[istate]->num_branches();
              break;
            }
          }
        }
      }
      return nbranch;
    };

    while(num_active_states > 0){   //loop until all branches execute all ops
      uint_t nbranch = 0;

      //apply ops until some branch operations are executed in some branches
      nbranch = Utils::apply_omp_parallel_for_reduction_int((par_shots > 1 && branches.size() > 1), 0, par_shots, apply_ops_func, par_shots);

      //repeat until new branch is available
      while(nbranch > 0){
        uint_t num_states_prev = branches.size();
        for(int_t i=0;i<num_states_prev;i++){
          //add new branches
          if(branches[i]->num_branches() > 0){
            int_t istart = 0;
            for(int_t j=0;j<branches[i]->num_branches();j++){
              if(branches[i]->branches()[j]->num_shots() > 0){
                //copy shots to the root
                branches[i]->set_shots(branches[i]->branches()[j]->rng_shots());
                branches[i]->additional_ops() = branches[i]->branches()[j]->additional_ops();
                states_[branches[i]->state_index()].creg() = branches[i]->branches()[j]->creg(); //save measured bits to state
                branches[i]->branches()[j].reset();
                istart = j+1;
                break;
              }
              branches[i]->branches()[j].reset();
            }

            for(int_t j=istart;j<branches[i]->num_branches();j++){
              if(branches[i]->branches()[j]->num_shots() > 0){
                //add new branched state
                uint_t pos = branches.size();
                if(pos >= num_max_shots_){  //if there is not enough memory to allocate copied state, shots are reserved to the next iteration
                  //reset seed to reproduce same results
                  for(int_t k=0;k<branches[i]->branches()[j]->num_shots();k++){
                    branches[i]->branches()[j]->rng_shots()[k].set_seed(branches[i]->branches()[j]->rng_shots()[k].initial_seed());
                  }
                  shots_storage.insert(shots_storage.end(), branches[i]->branches()[j]->rng_shots().begin(), branches[i]->branches()[j]->rng_shots().end());
                }
                else{
                  branches.push_back(branches[i]->branches()[j]);
                  branches[pos]->state_index() = pos;
                  //copy state to new branch
                  states_[pos].qreg().initialize(states_[branches[i]->state_index()].qreg());
                  states_[pos].creg() = branches[pos]->creg();
                }
              }
              else{
                branches[i]->branches()[j].reset();
              }
            }
          }
        }

        //then execute ops applied after branching (reset, Kraus, noises, etc.)
        auto apply_additional_ops_func = [this, &branches, &dummy_rng, &noise, &par_results, par_shots](int_t i)
        {
          uint_t istate,state_end;
          istate = branches.size()*i/par_shots;
          state_end = branches.size()*(i+1)/par_shots;
          uint_t nbranch = 0;

          for(;istate<state_end;istate++){
            branches[istate]->clear_branch();
            for(int_t j=0;j<branches[istate]->additional_ops().size();j++){
              if(apply_branching_op(*branches[istate], branches[istate]->additional_ops()[j], par_results[i], false)){
                if(branches[istate]->num_branches() > 0){  //check if there are new branches
                  //if there are additional ops remaining, queue them on new branches
                  for(int_t k=j+1;k<branches[istate]->additional_ops().size();k++){
                    for(int_t l=0;l<branches[istate]->num_branches();l++)
                      branches[istate]->branches()[l]->add_op_after_branch(branches[istate]->additional_ops()[k]);
                  }
                  nbranch += branches[istate]->num_branches();
                  break;
                }
              }
              else{
                states_[branches[istate]->state_index()].apply_op(branches[istate]->additional_ops()[j], par_results[i], dummy_rng, false);
              }
            }
            branches[istate]->clear_additional_ops();
          }
          return nbranch;
        };
        nbranch = Utils::apply_omp_parallel_for_reduction_int((par_shots > 1 && branches.size() > 1), 0, par_shots, apply_additional_ops_func, par_shots);
      }

      //check if there are remaining ops
      num_active_states = 0;
      for(int_t i=0;i<branches.size();i++){
        if(branches[i]->op_iterator() != measure_seq)
          num_active_states++;
      }
    }

    if(can_sample){
      //apply sampling measure for each branch
      auto sampling_measure_func = [this, &branches, &par_results, measure_seq, last, par_shots](int_t i)
      {
        uint_t istate,state_end;
        istate = branches.size()*i/par_shots;
        state_end = branches.size()*(i+1)/par_shots;

        for(;istate<state_end;istate++){
          BaseExecutor::measure_sampler(measure_seq, last, branches[istate]->num_shots(), states_[branches[istate]->state_index()],
                                        par_results[i], branches[istate]->rng_shots(), true);
        }
      };
      Utils::apply_omp_parallel_for((par_shots > 1 && branches.size() > 1), 0, par_shots, sampling_measure_func, par_shots);
    }
    else{
      //save cregs to result
      auto save_cregs = [this, &branches, &par_results, par_shots](int_t i){
        uint_t istate,state_end;
        istate = branches.size()*i/par_shots;
        state_end = branches.size()*(i+1)/par_shots;

        for(;istate<state_end;istate++){
          std::string memory_hex = states_[branches[istate]->state_index()].creg().memory_hex();
          for(int_t j=0;j<branches[istate]->num_shots();j++)
            par_results[i].data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
          if (BaseExecutor::save_creg_memory_) {
            for(int_t j=0;j<branches[istate]->num_shots();j++)
              par_results[i].data.add_list(memory_hex, "memory");
          }
        }
      };
      Utils::apply_omp_parallel_for((par_shots > 1),0,par_shots, save_cregs, par_shots);

      /*
      reg_t creg_pos(branches.size());
      for(int_t i=0;i<branches.size();i++){
        creg_pos[i] = num_shots_saved;
        num_shots_saved += branches[i]->num_shots();
      }

      //save cregs to array
      auto save_creg_func = [this, &branches, &creg_pos](int_t i)
      {
        for(int_t j=0;j<branches[i]->num_shots();j++){
          cregs_[creg_pos[i] + j] = states_[branches[i]->state_index()].creg();
        }
      };
      Utils::apply_omp_parallel_for((par_shots > 1 && branches.size() > 1), 0, branches.size(), save_creg_func, par_shots);
      */
    }

    //clear
    for(int_t i=0;i<branches.size();i++){
      branches[i].reset();
    }
    branches.clear();
  }

  /*
  if(!can_sample){
    gather_creg_memory();

    //save cregs to result
    auto save_cregs = [this,&par_results, par_shots](int_t i){
      uint_t i_shot,shot_end;
      i_shot = num_global_shots_*i/par_shots;
      shot_end = num_global_shots_*(i+1)/par_shots;

      for(;i_shot<shot_end;i_shot++){
        if(cregs_[i_shot].memory_size() > 0) {
          std::string memory_hex = cregs_[i_shot].memory_hex();
          par_results[i].data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
          if (BaseExecutor::save_creg_memory_) {
            par_results[i].data.add_list(std::move(memory_hex), "memory");
          }
        }
      }
    };
    Utils::apply_omp_parallel_for((par_shots > 1),0,par_shots, save_cregs, par_shots);
  }
  */

  for (auto &res : par_results) {
    result.combine(std::move(res));
  }

  result.metadata.add(true, "shot_branching_enabled");
  result.metadata.add(sample_noise, "runtime_noise_sampling_enabled");
}


template <class state_t>
void MultiShotsExecutor<state_t>::gather_creg_memory(void)
{
#ifdef AER_MPI
  int_t i,j;
  uint_t n64,i64,ibit;

  if(distributed_procs_ == 1)
    return;
  if(states_[0].creg()..memory_size() == 0)
    return;

  //number of 64-bit integers per memory
  n64 = (states_[0].creg().memory_size() + 63) >> 6;

  reg_t bin_memory(n64*num_local_shots_,0);
  //compress memory string to binary
#pragma omp parallel for private(i,j,i64,ibit)
  for(i=0;i<num_local_shotss_;i++){
    for(j=0;j<states_[0].creg().memory_size();j++){
      i64 = j >> 6;
      ibit = j & 63;
      if(states_[global_shot_index_ + i].creg().creg_memory()[j] == '1'){
        bin_memory[i*n64 + i64] |= (1ull << ibit);
      }
    }
  }

  reg_t recv(n64*num_global_shots_);
  std::vector<int> recv_counts(distributed_procs_);
  std::vector<int> recv_offset(distributed_procs_);

  for(i=0;i<distributed_procs_;i++){
    recv_offset[i] = num_global_shots_ * i / distributed_procs_;
    recv_counts[i] = (num_global_shots_ * (i+1) / distributed_procs_) - recv_offset[i];
  }

  MPI_Allgatherv(&bin_memory[0],n64*num_local_shots_,MPI_UINT64_T,
                 &recv[0],&recv_counts[0],&recv_offset[0],MPI_UINT64_T,distributed_comm_);

  //store gathered memory
#pragma omp parallel for private(i,j,i64,ibit)
  for(i=0;i<num_global_shots_;i++){
    for(j=0;j<states_[0].creg().memory_size();j++){
      i64 = j >> 6;
      ibit = j & 63;
      if(((recv[i*n64 + i64] >> ibit) & 1) == 1)
        states_[i].creg().creg_memory()[j] = '1';
      else
        states_[i].creg().creg_memory()[j] = '0';
    }
  }
#endif
}

  
//-------------------------------------------------------------------------
} // end namespace Executor
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif



