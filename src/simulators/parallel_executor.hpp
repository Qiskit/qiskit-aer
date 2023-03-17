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

#ifndef _parallel_executor_hpp_
#define _parallel_executor_hpp_

#include "simulators/aer_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace Executor {

//-------------------------------------------------------------------------
//Parallel executor class implementation
//-------------------------------------------------------------------------
template <class state_t>
class ParallelExecutor : public Base<state_t> {
protected:
  std::vector<state_t> states_;

  //number of qubits for the circuit
  uint_t num_qubits_;

  //extra parameters for parallel simulations
  uint_t num_global_chunks_;    //number of total chunks 
  uint_t num_local_chunks_;     //number of local chunks
  uint_t chunk_bits_;           //number of qubits per chunk
  uint_t block_bits_;           //number of cache blocked qubits

  uint_t global_chunk_index_;   //beginning chunk index for this process
  reg_t chunk_index_begin_;     //beginning chunk index for each process
  reg_t chunk_index_end_;       //ending chunk index for each process
  uint_t local_shot_index_;    //local shot ID of current batch loop

  uint_t myrank_;               //process ID
  uint_t nprocs_;               //number of processes
  uint_t distributed_rank_;     //process ID in communicator group
  uint_t distributed_procs_;    //number of processes in communicator group
  uint_t distributed_group_;    //group id of distribution
  int_t distributed_proc_bits_; //distributed_procs_=2^distributed_proc_bits_  (if nprocs != power of 2, set -1)

  bool chunk_omp_parallel_;     //using thread parallel to process loop of chunks or not
  bool global_chunk_indexing_;  //using global index for control qubits and diagonal matrix

  bool multi_chunk_distribution_ = false; //distributing chunks to apply cache blocking parallelization
  bool multi_shots_parallelization_ = false; //using chunks as multiple shots parallelization
  bool set_parallelization_called_ = false;    //this flag is used to check set_parallelization is already called, if yes the call sets max_batched_shots_
  uint_t max_batched_shots_ = 1;    //max number of shots can be stored on available memory

  reg_t qubit_map_;             //qubit map to restore swapped qubits

  bool multi_chunk_swap_enable_ = true;     //enable multi-chunk swaps
  uint_t chunk_swap_buffer_qubits_ = 15;    //maximum buffer size in qubits for chunk swap
  uint_t max_multi_swap_;                 //maximum swaps can be applied at a time, calculated by chunk_swap_buffer_bits_

  int max_matrix_qubits_;       //max qubits for matrix

  //group of states (GPU devices)
  uint_t num_groups_;            //number of groups of chunks
  reg_t top_chunk_of_group_;
  reg_t num_chunks_in_group_;
  int num_threads_per_group_;   //number of outer threads per group

  //cuStateVec settings
  bool cuStateVec_enable_ = false;

  uint_t num_creg_memory_ = 0;    //number of total bits for creg (reserve for multi-shots)
  uint_t num_creg_registers_ = 0;

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
  ParallelExecutor();
  virtual ~ParallelExecutor();

  size_t required_memory_mb(const Circuit &circuit,
                            const Noise::NoiseModel &noise) const override
  {
    state_t tmp;
    return tmp.required_memory_mb(circuit.num_qubits, circuit.ops);
  }

  void set_distribution(uint_t nprocs);

  bool allocate(uint_t num_qubits, uint_t block_bits, uint_t num_parallel_shots);

  uint_t get_process_by_chunk(uint_t cid);
protected:
  void set_config(const json_t &config) override;

  virtual uint_t qubit_scale(void)
  {
    return 1;
  }

  bool allocate_states(uint_t num_chunks);

  void run_circuit_with_sampling(Circuit &circ,
                                         const json_t &config,
                                         ExperimentResult &result) override;

  void run_circuit_shots(
            Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
            ExperimentResult &result, bool sample_noise) override;

  template <typename InputIterator>
  void measure_sampler(
      InputIterator first_meas, InputIterator last_meas, uint_t shots,
      ExperimentResult &result, RngEngine &rng) const;

  //apply operations for multi-chunk simulator
  template <typename InputIterator>
  void apply_ops_chunks(InputIterator first, InputIterator last,
                               ExperimentResult &result,
                               RngEngine &rng,
                               bool final_ops);

  //apply ops on cache memory
  template <typename InputIterator>
  void apply_cache_blocking_ops(const int_t iGroup, InputIterator first,
                 InputIterator last,
                 ExperimentResult &result,
                 RngEngine &rng);

  //apply parallel operations (implement for each simulation method)
  virtual void apply_parallel_op(const Operations::Op &op,
                                 ExperimentResult &result,
                                 RngEngine &rng, bool final_op) = 0;

  //store measure to cregs
  void store_measure(const reg_t &outcome, const reg_t &memory, const reg_t &registers);

  void apply_bfunc(const Operations::Op &op);
  void apply_roerror(const Operations::Op &op, RngEngine &rng);

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------
  template <typename list_t>
  void initialize_from_vector(const list_t &vec);

  template <typename list_t>
  void initialize_from_matrix(const list_t &mat);

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) = 0;


//-----------------------------------------------------------------------
  //Functions for multi-chunk distribution
  //-----------------------------------------------------------------------
  //utilities for chunk simulation
  virtual bool is_parallel_op(const Operations::Op &op);
  void block_diagonal_matrix(const int_t iChunk, reg_t &qubits, cvector_t &diag);

  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string& pauli) = 0;

  // Apply a save expectation value instruction
  void apply_save_expval(const Operations::Op &op, ExperimentResult &result);


  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                            RngEngine &rng) const 
  {
    std::vector<reg_t> ret;
    return ret;
  };

  //swap between chunks
  virtual void apply_chunk_swap(const reg_t &qubits);

  //apply multiple swaps between chunks
  virtual void apply_multi_chunk_swap(const reg_t &qubits);

  //apply X gate over chunks
  virtual void apply_chunk_x(const uint_t qubit);

  //send/receive chunk in receive buffer
  void send_chunk(uint_t local_chunk_index, uint_t global_chunk_index);
  void recv_chunk(uint_t local_chunk_index, uint_t global_chunk_index);

  template <class data_t>
  void send_data(data_t* pSend, uint_t size, uint_t myid,uint_t pairid);
  template <class data_t>
  void recv_data(data_t* pRecv, uint_t size, uint_t myid,uint_t pairid);

  //reduce values over processes
  void reduce_sum(reg_t& sum) const;
  void reduce_sum(rvector_t& sum) const;
  void reduce_sum(complex_t& sum) const;
  void reduce_sum(double& sum) const;

  //gather values on each process
  void gather_value(rvector_t& val) const;

  //gather cregs 
//  void gather_creg_memory(void);

  //barrier all processes
  void sync_process(void) const;

  //gather distributed state into vector (if memory is enough)
  template <class data_t>
  void gather_state(std::vector<std::complex<data_t>>& state);

  template <class data_t>
  void gather_state(AER::Vector<std::complex<data_t>>& state);

  //collect matrix over multiple chunks
  auto apply_to_matrix(bool copy = false);

  // Apply the global phase
  virtual void apply_global_phase();

  //return global shot index for the chunk
  inline int_t get_global_shot_index(const int_t iChunk) const
  {
    return multi_shots_parallelization_ ? (iChunk + local_shot_index_ + global_chunk_index_) : 0;
  }

  uint_t mapped_index(const uint_t idx);

  void set_global_phase(double theta);
};


template <class state_t>
ParallelExecutor<state_t>::ParallelExecutor()
{
  num_global_chunks_ = 0;
  num_local_chunks_ = 0;

  myrank_ = 0;
  nprocs_ = 1;

  distributed_procs_ = 1;
  distributed_rank_ = 0;
  distributed_group_ = 0;
  distributed_proc_bits_ = 0;

  chunk_omp_parallel_ = false;
  global_chunk_indexing_ = false;

#ifdef AER_MPI
  distributed_comm_ = MPI_COMM_WORLD;
#endif
}

template <class state_t>
ParallelExecutor<state_t>::~ParallelExecutor()
{
  states_.clear();
}

template <class state_t>
void ParallelExecutor<state_t>::set_config(const json_t &config) 
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

  if(JSON::check_key("chunk_swap_buffer_qubits", config)) {
    JSON::get_value(chunk_swap_buffer_qubits_, "chunk_swap_buffer_qubits", config);
  }
}

template <class state_t>
void ParallelExecutor<state_t>::set_global_phase(double theta) {
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
void ParallelExecutor<state_t>::set_distribution(uint_t nprocs)
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
}


template <class state_t>
bool ParallelExecutor<state_t>::allocate(uint_t num_qubits, uint_t block_bits, uint_t num_parallel_shots)
{
  int_t i;
  num_qubits_ = num_qubits;
  block_bits_ = block_bits;

  if(block_bits_ > 0){
    chunk_bits_ = block_bits_;
    if(chunk_bits_ > num_qubits_){
      chunk_bits_ = num_qubits_;
    }
  }
  else{
    chunk_bits_ = num_qubits_;
  }

  if(chunk_bits_ < num_qubits_){
    //multi-chunk distribution with cache blocking transpiler
    multi_chunk_distribution_ = true;
    multi_shots_parallelization_ = false;
    num_global_chunks_ = 1ull << ((num_qubits_ - chunk_bits_)*qubit_scale());
  }
  else{
    //multi-shots parallelization
    multi_chunk_distribution_ = false;
    if(num_parallel_shots > 1)
      multi_shots_parallelization_ = true;
    else
      multi_shots_parallelization_ = false;
    num_global_chunks_ = num_parallel_shots;
  }

  chunk_index_begin_.resize(distributed_procs_);
  chunk_index_end_.resize(distributed_procs_);
  for(i=0;i<distributed_procs_;i++){
    chunk_index_begin_[i] = num_global_chunks_*i / distributed_procs_;
    chunk_index_end_[i] = num_global_chunks_*(i+1) / distributed_procs_;
  }

  num_local_chunks_ = chunk_index_end_[distributed_rank_] - chunk_index_begin_[distributed_rank_];
  global_chunk_index_ = chunk_index_begin_[distributed_rank_];
  local_shot_index_ = 0;

  global_chunk_indexing_ = false;
  chunk_omp_parallel_ = false;
  if(this->sim_device_ == Device::GPU){
#ifdef _OPENMP
    if(omp_get_num_threads() == 1)
      chunk_omp_parallel_ = true;
#endif

    //set cuStateVec_enable_ 
    if(cuStateVec_enable_){
      if(multi_shots_parallelization_)
        cuStateVec_enable_ = false;   //multi-shots parallelization is not supported for cuStateVec
    }

    if(!cuStateVec_enable_)
      global_chunk_indexing_ = true;    //cuStateVec does not handle global chunk index for diagonal matrix
  }
  else if(this->sim_device_ == Device::ThrustCPU){
    global_chunk_indexing_ = true;
    chunk_omp_parallel_ = false;
  }

  if(multi_shots_parallelization_){
    allocate_states(std::min(num_local_chunks_, max_batched_shots_));
  }
  else{
    allocate_states(num_local_chunks_);
  }

  //initialize qubit map
  qubit_map_.resize(num_qubits_);
  for(i=0;i<num_qubits_;i++){
    qubit_map_[i] = i;
  }

  if(chunk_bits_ <= chunk_swap_buffer_qubits_ + 1)
    multi_chunk_swap_enable_ = false;
  else
    max_multi_swap_ = chunk_bits_ - chunk_swap_buffer_qubits_;

  return true;
}

template <class state_t>
bool ParallelExecutor<state_t>::allocate_states(uint_t num_chunks)
{
  int_t i;
  //deallocate qregs before reallocation
  if(states_.size() > 0){
    if(states_.size() == num_chunks)
      return true;  //can reuse allocated chunks

    states_.clear();
  }

  //allocate states
  states_.resize(num_chunks);

  uint_t chunk_id = multi_chunk_distribution_ ? global_chunk_index_ : 0;
  bool ret = true;
  states_[0].qreg().set_max_matrix_bits(max_matrix_qubits_);
  states_[0].qreg().set_num_threads_per_group(num_threads_per_group_);
  states_[0].qreg().cuStateVec_enable(cuStateVec_enable_);
  ret &= states_[0].qreg().chunk_setup(chunk_bits_*qubit_scale(), num_qubits_*qubit_scale(), chunk_id, num_chunks);
  states_[0].set_num_global_qubits(num_qubits_);
  for(i=1;i<num_chunks;i++){
    uint_t gid = i + chunk_id;
    ret &= states_[i].qreg().chunk_setup(states_[0].qreg() ,gid);
    states_[i].qreg().set_num_threads_per_group(num_threads_per_group_);
    states_[i].set_num_global_qubits(num_qubits_);
  }

  //initialize groups
  top_chunk_of_group_.clear();
  num_groups_ = 0;
  for(i=0;i<states_.size();i++){
    if(states_[i].qreg().top_of_group()){
      top_chunk_of_group_.push_back(i);
      num_groups_++;
    }
  }
  top_chunk_of_group_.push_back(states_.size());
  num_chunks_in_group_.resize(num_groups_);
  for(i=0;i<num_groups_;i++){
    num_chunks_in_group_[i] = top_chunk_of_group_[i+1] - top_chunk_of_group_[i];
  }

  return ret;
}

template <class state_t>
uint_t ParallelExecutor<state_t>::get_process_by_chunk(uint_t cid)
{
  uint_t i;
  for(i=0;i<distributed_procs_;i++){
    if(cid >= chunk_index_begin_[i] && cid < chunk_index_end_[i]){
      return i;
    }
  }
  return distributed_procs_;
}

template <class state_t>
uint_t ParallelExecutor<state_t>::mapped_index(const uint_t idx)
{
  uint_t i,ret = 0;
  uint_t t = idx;

  for(i=0;i<num_qubits_;i++){
    if(t & 1){
      ret |= (1ull << qubit_map_[i]);
    }
    t >>= 1;
  }
  return ret;
}

template <class state_t>
void ParallelExecutor<state_t>::run_circuit_with_sampling(Circuit &circ,
                                                   const json_t &config,
                                                   ExperimentResult &result)
{
  max_matrix_qubits_ = Base<state_t>::get_max_matrix_qubits(circ);

  // Optimize circuit
  Noise::NoiseModel dummy_noise;
  state_t dummy_state;

  auto fusion_pass = Base<state_t>::transpile_fusion( circ.opset(), config);
  fusion_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(), result);

  // Cache blocking pass
  uint_t block_bits = circ.num_qubits;
  auto cache_block_pass = Base<state_t>::transpile_cache_blocking(circ, dummy_noise, config);
  cache_block_pass.set_sample_measure(true);
  cache_block_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(), result);
  if(cache_block_pass.enabled()){
    block_bits = cache_block_pass.block_bits();
  }

  set_distribution(Base<state_t>::num_process_per_experiment_);
  allocate(circ.num_qubits, block_bits, 1);

  // Set state config
  for(uint_t i=0;i<states_.size();i++){
    states_[i].set_config(config);
    states_[i].set_parallelization(Base<state_t>::parallel_state_update_);
    states_[i].set_global_phase(circ.global_phase_angle);
  }
  set_global_phase(circ.global_phase_angle);

  //run with multi-chunks
  RngEngine rng;
  rng.set_seed(circ.seed);

  auto& ops = circ.ops;
  auto first_meas = circ.first_measure_pos; // Position of first measurement op
  bool final_ops = (first_meas == ops.size());

  initialize_qreg(circ.num_qubits);
  for(uint_t i=0;i<states_.size();i++){
    states_[i].initialize_creg(circ.num_memory, circ.num_registers);
  }

  // Run circuit instructions before first measure
  apply_ops_chunks(ops.cbegin(), ops.cbegin() + first_meas, result, rng, final_ops);

  // Get measurement operations and set of measured qubits
  measure_sampler(circ.ops.begin() + first_meas, circ.ops.end(), circ.shots, result, rng);

  // Add measure sampling metadata
  result.metadata.add(true, "measure_sampling");
  states_[0].add_metadata(result);
}

template <class state_t>
void ParallelExecutor<state_t>::run_circuit_shots(
    Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
    ExperimentResult &result, bool sample_noise)
{
  set_distribution(Base<state_t>::num_process_per_experiment_);

  auto fusion_pass = Base<state_t>::transpile_fusion( circ.opset(), config);
  auto cache_block_pass = Base<state_t>::transpile_cache_blocking(circ, noise, config);

  for(int_t ishot=0;ishot<circ.shots;ishot++){
    RngEngine rng;
    rng.set_seed(circ.seed + ishot);

    // Optimize circuit
    Noise::NoiseModel dummy_noise;
    state_t dummy_state;

    Circuit circ_opt;
    if(sample_noise){
      circ_opt = noise.sample_noise(circ, rng);
    }
    else{
      circ_opt = circ;
    }
    fusion_pass.optimize_circuit(circ_opt, dummy_noise, dummy_state.opset(), result);
    max_matrix_qubits_ = Base<state_t>::get_max_matrix_qubits(circ_opt);

    // Cache blocking pass
    uint_t block_bits = circ.num_qubits;
    cache_block_pass.set_sample_measure(false);
    cache_block_pass.optimize_circuit(circ_opt, dummy_noise, dummy_state.opset(), result);
    if(cache_block_pass.enabled()){
      block_bits = cache_block_pass.block_bits();
    }
    allocate(circ.num_qubits, block_bits, 1);

    // Set state config
    for(uint_t i=0;i<states_.size();i++){
      states_[i].set_config(config);
      states_[i].set_parallelization(Base<state_t>::parallel_state_update_);
      states_[i].set_global_phase(circ.global_phase_angle);
    }
    set_global_phase(circ.global_phase_angle);

    initialize_qreg(circ.num_qubits);
    for(uint_t i=0;i<states_.size();i++){
      states_[i].initialize_creg(circ.num_memory, circ.num_registers);
    }

    apply_ops_chunks(circ_opt.ops.cbegin(), circ_opt.ops.cend(), result, rng, true);
    result.save_count_data(states_[0].creg(), Base<state_t>::save_creg_memory_);
  }
  states_[0].add_metadata(result);
}


template <class state_t>
template <typename InputIterator>
void ParallelExecutor<state_t>::measure_sampler(
    InputIterator first_meas, InputIterator last_meas, uint_t shots,
    ExperimentResult &result, RngEngine &rng) const 
{
  // Check if meas_circ is empty, and if so return initial creg
  if (first_meas == last_meas) {
    while (shots-- > 0) {
      result.save_count_data(states_[0].creg(), Base<state_t>::save_creg_memory_);
    }
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

  // Generate the samples
  auto timer_start = myclock_t::now();
  auto all_samples = sample_measure(meas_qubits, shots, rng);
  auto time_taken =
      std::chrono::duration<double>(myclock_t::now() - timer_start).count();
  result.metadata.add(time_taken, "sample_measure_time");

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

  // Process samples
  uint_t num_memory = (memory_map.empty()) ? 0ULL : 1 + memory_map.rbegin()->first;
  uint_t num_registers = (register_map.empty()) ? 0ULL : 1 + register_map.rbegin()->first;
  ClassicalRegister creg;
  while (!all_samples.empty()) {
    auto sample = all_samples.back();
    creg.initialize(num_memory, num_registers);

    // process memory bit measurements
    for (const auto &pair : memory_map) {
      creg.store_measure(reg_t({sample[pair.second]}), reg_t({pair.first}),
                         reg_t());
    }
    // process register bit measurements
    for (const auto &pair : register_map) {
      creg.store_measure(reg_t({sample[pair.second]}), reg_t(),
                         reg_t({pair.first}));
    }

    // process read out errors for memory and registers
    for (const Operations::Op &roerror : roerror_ops) {
      creg.apply_roerror(roerror, rng);
    }

    // Save count data
      result.save_count_data(creg, Base<state_t>::save_creg_memory_);

    // pop off processed sample
    all_samples.pop_back();
  }
}

template <class state_t>
void ParallelExecutor<state_t>::store_measure(const reg_t &outcome, const reg_t &memory, const reg_t &registers)
{
  auto apply_store_measure = [this, outcome, memory, registers](int_t iGroup)
  {
    //store creg to the all top states of groups so that conditional ops can be applied correctly
    states_[top_chunk_of_group_[iGroup]].creg().store_measure(outcome, memory, registers);
  };
  Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1), 0, num_groups_, apply_store_measure);
}

template <class state_t>
void ParallelExecutor<state_t>::apply_bfunc(const Operations::Op &op)
{
  auto bfunc_kernel = [this, op](int_t iGroup)
  {
    //store creg to the all top states of groups so that conditional ops can be applied correctly
    states_[top_chunk_of_group_[iGroup]].creg().apply_bfunc(op);
  };
  Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1), 0, num_groups_, bfunc_kernel);
}

template <class state_t>
void ParallelExecutor<state_t>::apply_roerror(const Operations::Op &op, RngEngine &rng)
{
  auto roerror_kernel = [this, op, &rng](int_t iGroup)
  {
    //store creg to the all top states of groups so that conditional ops can be applied correctly
    states_[top_chunk_of_group_[iGroup]].creg().apply_roerror(op, rng);
  };
  Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1), 0, num_groups_, roerror_kernel);
}

template <class state_t>
template <typename InputIterator>
void ParallelExecutor<state_t>::apply_ops_chunks(InputIterator first, InputIterator last,
                               ExperimentResult &result,
                               RngEngine &rng,
                               bool final_ops) 
{
  uint_t iOp,nOp;
  reg_t multi_swap;

  nOp = std::distance(first, last);
  iOp = 0;

  while(iOp < nOp){
    const Operations::Op op_iOp = *(first + iOp);
    if(op_iOp.type == Operations::OpType::gate && op_iOp.name == "swap_chunk"){
      //apply swap between chunks
      if(multi_chunk_swap_enable_ && op_iOp.qubits[0] < chunk_bits_ && op_iOp.qubits[1] >= chunk_bits_){
        if(distributed_proc_bits_ < 0 || (op_iOp.qubits[1] >= (num_qubits_*qubit_scale() - distributed_proc_bits_))){   //apply multi-swap when swap is cross qubits
          multi_swap.push_back(op_iOp.qubits[0]);
          multi_swap.push_back(op_iOp.qubits[1]);
          if(multi_swap.size() >= max_multi_swap_*2){
            apply_multi_chunk_swap(multi_swap);
            multi_swap.clear();
          }
        }
        else
          apply_chunk_swap(op_iOp.qubits);
      }
      else{
        if(multi_swap.size() > 0){
          apply_multi_chunk_swap(multi_swap);
          multi_swap.clear();
        }
        apply_chunk_swap(op_iOp.qubits);
      }
      iOp++;
      continue;
    }
    else if(multi_swap.size() > 0){
      apply_multi_chunk_swap(multi_swap);
      multi_swap.clear();
    }

    if(op_iOp.type == Operations::OpType::sim_op && op_iOp.name == "begin_blocking"){
      //applying sequence of gates inside each chunk

      uint_t iOpEnd = iOp;
      while(iOpEnd < nOp){
        const Operations::Op op_iOpEnd = *(first + iOpEnd);
        if(op_iOpEnd.type == Operations::OpType::sim_op && op_iOpEnd.name == "end_blocking"){
          break;
        }
        iOpEnd++;
      }

      uint_t iOpBegin = iOp + 1;
      if(num_groups_ > 1 && chunk_omp_parallel_){
#pragma omp parallel for  num_threads(num_groups_)
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(ig, first + iOpBegin, first + iOpEnd, result, rng);
      }
      else{
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(ig, first + iOpBegin, first + iOpEnd, result, rng);
      }
      iOp = iOpEnd;
    }
    else if(is_parallel_op(op_iOp)){
      apply_parallel_op(op_iOp, result, rng, final_ops && nOp == iOp + 1);
    }
    else{
      if(num_groups_ > 1 && chunk_omp_parallel_){
#pragma omp parallel for num_threads(num_groups_)
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(ig, first + iOp, first + iOp+1, result, rng);
      }
      else{
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(ig, first + iOp, first + iOp+1, result, rng);
      }
    }
    iOp++;
  }

  if(multi_swap.size() > 0)
    apply_multi_chunk_swap(multi_swap);

  if(num_groups_ > 1 && chunk_omp_parallel_){
#pragma omp parallel for  num_threads(num_groups_)
    for(int_t ig=0;ig<num_groups_;ig++)
      states_[top_chunk_of_group_[ig]].qreg().synchronize();
  }
  else{
    for(int_t ig=0;ig<num_groups_;ig++)
      states_[top_chunk_of_group_[ig]].qreg().synchronize();
  }

  if(Base<state_t>::sim_device_name_ == "GPU"){
#ifdef AER_THRUST_CUDA
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      nDev = 0;
    }
    if(nDev > num_groups_)
      nDev = num_groups_;
    result.metadata.add(nDev,"cacheblocking", "chunk_parallel_gpus");
#endif

#ifdef AER_CUSTATEVEC
    result.metadata.add(cuStateVec_enable_, "cuStateVec_enable");
#endif
  }

#ifdef AER_MPI
  result.metadata.add(multi_chunk_swap_enable_,"cacheblocking", "multiple_chunk_swaps_enable");
  if(multi_chunk_swap_enable_){
    result.metadata.add(chunk_swap_buffer_qubits_,"cacheblocking", "multiple_chunk_swaps_buffer_qubits");
    result.metadata.add(max_multi_swap_,"cacheblocking", "max_multiple_chunk_swaps");
  }
#endif
}

template <class state_t>
template <typename InputIterator>
void ParallelExecutor<state_t>::apply_cache_blocking_ops(const int_t iGroup, InputIterator first,
               InputIterator last,
               ExperimentResult &result,
               RngEngine &rng)
{
  //for each chunk in group
  for(int_t iChunk = top_chunk_of_group_[iGroup];iChunk < top_chunk_of_group_[iGroup + 1];iChunk++){
    //fecth chunk in cache
    if(states_[iChunk].qreg().fetch_chunk()){
      states_[iChunk].apply_ops( first, last, result, rng, false);

      //release chunk from cache
      states_[iChunk].qreg().release_chunk();
    }
  }
}

template <class state_t>
bool ParallelExecutor<state_t>::is_parallel_op(const Operations::Op &op)
{
  if(op.type == Operations::OpType::gate || op.type == Operations::OpType::matrix || 
            op.type == Operations::OpType::diagonal_matrix || op.type == Operations::OpType::multiplexer ||
            op.type == Operations::OpType::superop){
    return false;     //these ops are cache blocked
  }
  return true;
}


template <class state_t>
void ParallelExecutor<state_t>::block_diagonal_matrix(const int_t iChunk, reg_t &qubits, cvector_t &diag)
{
  uint_t gid = global_chunk_index_ + iChunk;
  uint_t i;
  uint_t mask_out = 0;
  uint_t mask_id = 0;

  reg_t qubits_in;
  cvector_t diag_in;

  for(i=0;i<qubits.size();i++){
    if(qubits[i] < chunk_bits_){ //in chunk
      qubits_in.push_back(qubits[i]);
    }
    else{
      mask_out |= (1ull << i);
      if((gid >> (qubits[i] - chunk_bits_)) & 1)
        mask_id |= (1ull << i);
    }
  }

  if(qubits_in.size() < qubits.size()){
    for(i=0;i<diag.size();i++){
      if((i & mask_out) == mask_id)
        diag_in.push_back(diag[i]);
    }

    if(qubits_in.size() == 0){
      qubits_in.push_back(0);
      diag_in.resize(2);
      diag_in[1] = diag_in[0];
    }
    qubits = qubits_in;
    diag = diag_in;
  }
}

template <class state_t>
template <typename list_t>
void ParallelExecutor<state_t>::initialize_from_vector(const list_t &vec)
{
  int_t iChunk;

  if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for private(iChunk)
    for(int_t ig=0;ig<num_groups_;ig++){
      for(iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++){
        list_t tmp(1ull << (chunk_bits_*qubit_scale()));
        for(int_t i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
          tmp[i] = vec[((global_chunk_index_ + iChunk) << (chunk_bits_*qubit_scale())) + i];
        }
        states_[iChunk].qreg().initialize_from_vector(tmp);
      }
    }
  }
  else{
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      list_t tmp(1ull << (chunk_bits_*qubit_scale()));
      for(int_t i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
        tmp[i] = vec[((global_chunk_index_ + iChunk) << (chunk_bits_*qubit_scale())) + i];
      }
      states_[iChunk].qreg().initialize_from_vector(tmp);
    }
  }
}

template <class state_t>
template <typename list_t>
void ParallelExecutor<state_t>::initialize_from_matrix(const list_t &mat)
{
  int_t iChunk;
  if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for private(iChunk)
    for(int_t ig=0;ig<num_groups_;ig++){
      for(iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++){
        list_t tmp(1ull << (chunk_bits_),1ull << (chunk_bits_));
        uint_t irow_chunk = ((iChunk + global_chunk_index_) >> ((num_qubits_ - chunk_bits_))) << (chunk_bits_);
        uint_t icol_chunk = ((iChunk + global_chunk_index_) & ((1ull << ((num_qubits_ - chunk_bits_)))-1)) << (chunk_bits_);

        //copy part of state for this chunk
        uint_t i,row,col;
        for(i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
          uint_t icol = i & ((1ull << chunk_bits_)-1);
          uint_t irow = i >> chunk_bits_;
          tmp[i] = mat[icol_chunk + icol + ((irow_chunk + irow) << num_qubits_)];
        }
        states_[iChunk].qreg().initialize_from_matrix(tmp);
      }
    }
  }
  else{
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      list_t tmp(1ull << (chunk_bits_),1ull << (chunk_bits_));
      uint_t irow_chunk = ((iChunk + global_chunk_index_) >> ((num_qubits_ - chunk_bits_))) << (chunk_bits_);
      uint_t icol_chunk = ((iChunk + global_chunk_index_) & ((1ull << ((num_qubits_ - chunk_bits_)))-1)) << (chunk_bits_);

      //copy part of state for this chunk
      uint_t i,row,col;
      for(i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
        uint_t icol = i & ((1ull << chunk_bits_)-1);
        uint_t irow = i >> chunk_bits_;
        tmp[i] = mat[icol_chunk + icol + ((irow_chunk + irow) << num_qubits_)];
      }
      states_[iChunk].qreg().initialize_from_matrix(tmp);
    }
  }
}

template <class state_t>
auto ParallelExecutor<state_t>::apply_to_matrix(bool copy)
{
  //this function is used to collect states over chunks
  int_t iChunk;
  uint_t size = 1ull << (chunk_bits_*qubit_scale());
  uint_t mask = (1ull << (chunk_bits_)) - 1;
  uint_t num_threads = states_[0].qreg().get_omp_threads();

  size_t size_required = 2*(sizeof(std::complex<double>) << (num_qubits_*2)) + (sizeof(std::complex<double>) << (chunk_bits_*2))*num_local_chunks_;
  if((size_required>>20) > Utils::get_system_memory_mb()){
    throw std::runtime_error(std::string("There is not enough memory to store states as matrix"));
  }

  auto matrix = states_[0].qreg().copy_to_matrix();

  if(distributed_rank_ == 0){
    matrix.resize(1ull << (num_qubits_),1ull << (num_qubits_));

    auto tmp = states_[0].qreg().copy_to_matrix();
    for(iChunk=0;iChunk<num_global_chunks_;iChunk++){
      int_t i;
      uint_t irow_chunk = (iChunk >> ((num_qubits_ - chunk_bits_))) << chunk_bits_;
      uint_t icol_chunk = (iChunk & ((1ull << ((num_qubits_ - chunk_bits_)))-1)) << chunk_bits_;

      if(iChunk < num_local_chunks_){
        if(copy)
          tmp = states_[iChunk].qreg().copy_to_matrix();
        else
          tmp = states_[iChunk].qreg().move_to_matrix();
      }
#ifdef AER_MPI
      else
        recv_data(tmp.data(),size,0,iChunk);
#endif
#pragma omp parallel for if(num_threads > 1) num_threads(num_threads)
      for(i=0;i<size;i++){
        uint_t irow = i >> (chunk_bits_);
        uint_t icol = i & mask;
        uint_t idx = ((irow+irow_chunk) << (num_qubits_)) + icol_chunk + icol;
        matrix[idx] = tmp[i];
      }
    }
  }
  else{
#ifdef AER_MPI
    //send matrices to process 0
    for(iChunk=0;iChunk<num_global_chunks_;iChunk++){
      uint_t iProc = get_process_by_chunk(iChunk);
      if(iProc == distributed_rank_){
        if(copy){
          auto tmp = states_[iChunk-global_chunk_index_].qreg().copy_to_matrix();
          send_data(tmp.data(),size,iChunk,0);
        }
        else{
          auto tmp = states_[iChunk-global_chunk_index_].qreg().move_to_matrix();
          send_data(tmp.data(),size,iChunk,0);
        }
      }
    }
#endif
  }

  return matrix;
}

template <class state_t>
void ParallelExecutor<state_t>::apply_save_expval(const Operations::Op &op,
                                       ExperimentResult &result){
  // Check empty edge case
  if (op.expval_params.empty()) {
    throw std::invalid_argument(
        "Invalid save expval instruction (Pauli components are empty).");
  }
  bool variance = (op.type == Operations::OpType::save_expval_var);

  // Accumulate expval components
  double expval(0.);
  double sq_expval(0.);

  for (const auto &param : op.expval_params) {
    // param is tuple (pauli, coeff, sq_coeff)
    const auto val = expval_pauli(op.qubits, std::get<0>(param));
    expval += std::get<1>(param) * val;
    if (variance) {
      sq_expval += std::get<2>(param) * val;
    }
  }
  if (variance) {
    std::vector<double> expval_var(2);
    expval_var[0] = expval;  // mean
    expval_var[1] = sq_expval - expval * expval;  // variance
    result.save_data_average(states_[0].creg(), op.string_params[0], expval_var, op.type, op.save_type);
  } else {
    result.save_data_average(states_[0].creg(), op.string_params[0], expval, op.type, op.save_type);
  }
}

template <class state_t>
void ParallelExecutor<state_t>::apply_global_phase() 
{
  if (has_global_phase_) {
    if(chunk_omp_parallel_ && num_groups_ > 0){
#pragma omp parallel for 
      for(int_t ig=0;ig<num_groups_;ig++){
        for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
        states_[iChunk].qreg().apply_diagonal_matrix({0}, {global_phase_, global_phase_});
      }
    }
    else{
      for(int_t i=0;i<states_.size();i++)
        states_[i].qreg().apply_diagonal_matrix({0}, {global_phase_, global_phase_});
    }
  }
}

template <class state_t>
void ParallelExecutor<state_t>::apply_chunk_swap(const reg_t &qubits)
{
  uint_t nLarge = 1;
  uint_t q0,q1;
  int_t iChunk;

  q0 = qubits[qubits.size() - 2];
  q1 = qubits[qubits.size() - 1];

  if(qubit_scale() == 1){
    std::swap(qubit_map_[q0],qubit_map_[q1]);
  }

  if(q0 > q1){
    std::swap(q0,q1);
  }

  if(q1 < chunk_bits_*qubit_scale()){
    //inside chunk
    if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for num_threads(num_groups_) 
      for(int_t ig=0;ig<num_groups_;ig++){
        for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
        states_[iChunk].qreg().apply_mcswap(qubits);
      }
    }
    else{
      for(int_t ig=0;ig<num_groups_;ig++){
        for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
        states_[iChunk].qreg().apply_mcswap(qubits);
      }
    }
  }
  else{ //swap over chunks
    uint_t mask0,mask1;

    mask0 = (1ull << q0);
    mask1 = (1ull << q1);
    mask0 >>= (chunk_bits_*qubit_scale());
    mask1 >>= (chunk_bits_*qubit_scale());

    if(distributed_procs_ == 1 || (distributed_proc_bits_ >= 0 && q1 < (num_qubits_*qubit_scale() - distributed_proc_bits_))){   //no data transfer between processes is needed
      auto apply_chunk_swap_1qubit = [this, mask1, qubits](int_t iGroup)
      {
        for(int_t ic = top_chunk_of_group_[iGroup];ic < top_chunk_of_group_[iGroup + 1];ic++){
          uint_t baseChunk;
          baseChunk = ic & (~mask1);
          if(ic == baseChunk)
          states_[ic].qreg().apply_chunk_swap(qubits,states_[ic | mask1].qreg(),true);
        }
      };
      auto apply_chunk_swap_2qubits = [this, mask0, mask1, qubits](int_t iGroup)
      {
        for(int_t ic = top_chunk_of_group_[iGroup];ic < top_chunk_of_group_[iGroup + 1];ic++){
          uint_t baseChunk;
          baseChunk = ic & (~(mask0 | mask1));
          uint_t iChunk1 = baseChunk | mask0;
          uint_t iChunk2 = baseChunk | mask1;
          if(ic == iChunk1)
          states_[iChunk1].qreg().apply_chunk_swap(qubits,states_[iChunk2].qreg(),true);
        }
      };
      if(q0 < chunk_bits_*qubit_scale())
        Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1), 0, num_groups_, apply_chunk_swap_1qubit);
      else
        Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1), 0, num_groups_, apply_chunk_swap_2qubits);
    }
#ifdef AER_MPI
    else{
      int_t iPair;
      uint_t nPair;
      uint_t baseChunk,iChunk1,iChunk2;

      if(q0 < chunk_bits_*qubit_scale())
        nLarge = 1;
      else
        nLarge = 2;

      //chunk scheduler that supports any number of processes
      uint_t nu[3];
      uint_t ub[3];
      uint_t iu[3];
      uint_t add;
      uint_t iLocalChunk,iRemoteChunk,iProc;
      int i;

      if(q0 < chunk_bits_*qubit_scale()){
        nLarge = 1;
        nu[0] = 1ull << (q1 - chunk_bits_*qubit_scale());
        ub[0] = 0;
        iu[0] = 0;

        nu[1] = 1ull << (num_qubits_*qubit_scale() - q1 - 1);
        ub[1] = (q1 - chunk_bits_*qubit_scale()) + 1;
        iu[1] = 0;
      }
      else{
        nLarge = 2;
        nu[0] = 1ull << (q0 - chunk_bits_*qubit_scale());
        ub[0] = 0;
        iu[0] = 0;

        nu[1] = 1ull << (q1 - q0 - 1);
        ub[1] = (q0 - chunk_bits_*qubit_scale()) + 1;
        iu[1] = 0;

        nu[2] = 1ull << (num_qubits_*qubit_scale() - q1 - 1);
        ub[2] = (q1 - chunk_bits_*qubit_scale()) + 1;
        iu[2] = 0;
      }
      nPair = 1ull << (num_qubits_*qubit_scale() - chunk_bits_*qubit_scale() - nLarge);

      for(iPair=0;iPair<nPair;iPair++){
        //calculate index of pair of chunks
        baseChunk = 0;
        add = 1;
        for(i=nLarge;i>=0;i--){
          baseChunk += (iu[i] << ub[i]);
          //update for next
          iu[i] += add;
          add = 0;
          if(iu[i] >= nu[i]){
            iu[i] = 0;
            add = 1;
          }
        }

        iChunk1 = baseChunk | mask0;
        iChunk2 = baseChunk | mask1;

        if(iChunk1 >= chunk_index_begin_[distributed_rank_] && iChunk1 < chunk_index_end_[distributed_rank_]){    //chunk1 is on this process
          if(iChunk2 >= chunk_index_begin_[distributed_rank_] && iChunk2 < chunk_index_end_[distributed_rank_]){    //chunk2 is on this process
            states_[iChunk1 - global_chunk_index_].qreg().apply_chunk_swap(qubits,states_[iChunk2 - global_chunk_index_].qreg(),true);
            continue;
          }
          else{
            iLocalChunk = iChunk1;
            iRemoteChunk = iChunk2;
            iProc = get_process_by_chunk(iChunk2);
          }
        }
        else{
          if(iChunk2 >= chunk_index_begin_[distributed_rank_] && iChunk2 < chunk_index_end_[distributed_rank_]){    //chunk2 is on this process
            iLocalChunk = iChunk2;
            iRemoteChunk = iChunk1;
            iProc = get_process_by_chunk(iChunk1);
          }
          else{
            continue;   //there is no chunk for this pair on this process
          }
        }

        MPI_Request reqSend,reqRecv;
        MPI_Status st;
        uint_t sizeRecv,sizeSend;

        auto pRecv = states_[iLocalChunk - global_chunk_index_].qreg().recv_buffer(sizeRecv);
        MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,iPair,distributed_comm_,&reqRecv);

        auto pSend = states_[iLocalChunk - global_chunk_index_].qreg().send_buffer(sizeSend);
        MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,iPair,distributed_comm_,&reqSend);

        MPI_Wait(&reqSend,&st);
        MPI_Wait(&reqRecv,&st);

        states_[iLocalChunk - global_chunk_index_].qreg().apply_chunk_swap(qubits,iRemoteChunk);
      }
    }
#endif
  }
}

template <class state_t>
void ParallelExecutor<state_t>::apply_multi_chunk_swap(const reg_t &qubits)
{
  int_t nswap = qubits.size()/2;
  reg_t chunk_shuffle_qubits(nswap,0);
  reg_t local_swaps;
  uint_t baseChunk = 0;
  uint_t nchunk = 1ull << nswap;
  reg_t chunk_procs(nchunk);
  reg_t chunk_offset(nchunk);

  if(qubit_scale() == 1){
    for(int_t i=0;i<nswap;i++)
      std::swap(qubit_map_[qubits[i*2]],qubit_map_[qubits[i*2]+1]);
  }

  //define local swaps
  for(int_t i=0;i<nswap;i++){
    if(qubits[i*2] >= chunk_bits_*qubit_scale() - nswap)  //no swap required
      chunk_shuffle_qubits[qubits[i*2] + nswap - chunk_bits_*qubit_scale()] = qubits[i*2 + 1];
  }
  int_t pos = 0;
  for(int_t i=0;i<nswap;i++){
    if(qubits[i*2] < chunk_bits_*qubit_scale() - nswap){  //local swap required
      //find empty position
      while(pos < nswap){
        if(chunk_shuffle_qubits[pos] < chunk_bits_*qubit_scale()){
          chunk_shuffle_qubits[pos] = qubits[i*2 + 1];
          local_swaps.push_back(qubits[i*2]);
          local_swaps.push_back(chunk_bits_*qubit_scale() - nswap + pos);
          pos++;
          break;
        }
        pos++;
      }
    }
  }
  for(int_t i=0;i<nswap;i++)
    chunk_shuffle_qubits[i] -= chunk_bits_*qubit_scale();

  //swap inside chunks to prepare for all-to-all shuffle
  if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for 
    for(int_t ig=0;ig<num_groups_;ig++){
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
      states_[iChunk].qreg().apply_multi_swaps(local_swaps);
    }
  }
  else{
    for(int_t ig=0;ig<num_groups_;ig++){
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
      states_[iChunk].qreg().apply_multi_swaps(local_swaps);
    }
  }

  //apply all-to-all chunk shuffle
  int_t nPair;
  reg_t chunk_shuffle_qubits_sorted = chunk_shuffle_qubits;
  std::sort(chunk_shuffle_qubits_sorted.begin(), chunk_shuffle_qubits_sorted.end());

  nPair = num_global_chunks_ >> nswap;

  for(uint_t i=0;i<nchunk;i++){
    chunk_offset[i] = 0;
    for(uint_t k=0;k<nswap;k++){
      if(((i >> k) & 1) != 0)
        chunk_offset[i] += (1ull << chunk_shuffle_qubits[k]);
    }
  }

#ifdef AER_MPI
  std::vector<MPI_Request> reqSend(nchunk);
  std::vector<MPI_Request> reqRecv(nchunk);
#endif

  for(int_t iPair=0;iPair<nPair;iPair++){
    uint_t i1,i2,k,ii,t;
    baseChunk = 0;
    ii = iPair;
    for(k=0;k<nswap;k++){
      t = ii & ((1ull << chunk_shuffle_qubits_sorted[k]) - 1);
      baseChunk += t;
      ii = (ii - t) << 1;
    }
    baseChunk += ii;

    for(i1=0;i1<nchunk;i1++){
      chunk_procs[i1] = get_process_by_chunk(baseChunk + chunk_offset[i1]);
    }

    //all-to-all
    //send data
    for(uint_t iswap=1;iswap<nchunk;iswap++){
      uint_t sizeRecv,sizeSend;
      uint_t num_local_swap = 0;
      for(i1=0;i1<nchunk;i1++){
        i2 = i1 ^ iswap;
        if(i1 >= i2)
          continue;

        uint_t iProc1 = chunk_procs[i1];
        uint_t iProc2 = chunk_procs[i2];
        if(iProc1 != distributed_rank_ && iProc2 != distributed_rank_)
          continue;
        if(iProc1 == iProc2){  //on the same process
          num_local_swap++;
          continue;   //swap while data is exchanged between processes
        }
#ifdef AER_MPI
        uint_t offset1 = i1 << (chunk_bits_*qubit_scale() - nswap);
        uint_t offset2 = i2 << (chunk_bits_*qubit_scale() - nswap);
        uint_t iChunk1 = baseChunk + chunk_offset[i1] - global_chunk_index_;
        uint_t iChunk2 = baseChunk + chunk_offset[i2] - global_chunk_index_;

        int_t tid = (iPair << nswap) + iswap;

        if(iProc1 == distributed_rank_){
          auto pRecv = states_[iChunk1].qreg().recv_buffer(sizeRecv);
          MPI_Irecv(pRecv + offset2,(sizeRecv >> nswap),MPI_BYTE,iProc2,tid,distributed_comm_,&reqRecv[i2]);

          auto pSend = states_[iChunk1].qreg().send_buffer(sizeSend);
          MPI_Isend(pSend + offset2,(sizeSend >> nswap),MPI_BYTE,iProc2,tid,distributed_comm_,&reqSend[i2]);
        }
        else{
          auto pRecv = states_[iChunk2].qreg().recv_buffer(sizeRecv);
          MPI_Irecv(pRecv + offset1,(sizeRecv >> nswap),MPI_BYTE,iProc1,tid,distributed_comm_,&reqRecv[i1]);

          auto pSend = states_[iChunk2].qreg().send_buffer(sizeSend);
          MPI_Isend(pSend + offset1,(sizeSend >> nswap),MPI_BYTE,iProc1,tid,distributed_comm_,&reqSend[i1]);
        }
#endif
      }

      //swaps inside process
      if(num_local_swap > 0){
        for(i1=0;i1<nchunk;i1++){
          i2 = i1 ^ iswap;
          if(i1 > i2)
            continue;

          uint_t iProc1 = chunk_procs[i1];
          uint_t iProc2 = chunk_procs[i2];
          if(iProc1 != distributed_rank_ && iProc2 != distributed_rank_)
            continue;
          if(iProc1 == iProc2){  //on the same process
            uint_t offset1 = i1 << (chunk_bits_*qubit_scale() - nswap);
            uint_t offset2 = i2 << (chunk_bits_*qubit_scale() - nswap);
            uint_t iChunk1 = baseChunk + chunk_offset[i1] - global_chunk_index_;
            uint_t iChunk2 = baseChunk + chunk_offset[i2] - global_chunk_index_;
            states_[iChunk1].qreg().apply_chunk_swap(states_[iChunk2].qreg(),offset2,offset1,(1ull << (chunk_bits_*qubit_scale() - nswap)) );
          }
        }
      }

#ifdef AER_MPI
      //recv data
      for(i1=0;i1<nchunk;i1++){
        i2 = i1 ^ iswap;

        uint_t iProc1 = chunk_procs[i1];
        uint_t iProc2 = chunk_procs[i2];
        if(iProc1 != distributed_rank_)
          continue;
        if(iProc1 == iProc2){  //on the same process
          continue;
        }
        uint_t iChunk1 = baseChunk + chunk_offset[i1] - global_chunk_index_;
        uint_t offset2 = i2 << (chunk_bits_*qubit_scale() - nswap);

        MPI_Status st;
        MPI_Wait(&reqSend[i2],&st);
        MPI_Wait(&reqRecv[i2],&st);

        //copy states from recv buffer to chunk
        states_[iChunk1].qreg().apply_chunk_swap(states_[iChunk1].qreg(),offset2,offset2,(1ull << (chunk_bits_*qubit_scale() - nswap)) );
      }
#endif
    }
  }

  //restore qubits order
  if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for 
    for(int_t ig=0;ig<num_groups_;ig++){
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
      states_[iChunk].qreg().apply_multi_swaps(local_swaps);
    }
  }
  else{
    for(int_t ig=0;ig<num_groups_;ig++){
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
      states_[iChunk].qreg().apply_multi_swaps(local_swaps);
    }
  }
}


template <class state_t>
void ParallelExecutor<state_t>::apply_chunk_x(const uint_t qubit)
{
  int_t iChunk;
  uint_t nLarge = 1;


  if(qubit < chunk_bits_*qubit_scale()){
    auto apply_mcx = [this, qubit](int_t ig)
    {
      reg_t qubits(1,qubit);
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
        states_[iChunk].qreg().apply_mcx(qubits);
    };
    Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1),0,num_groups_,apply_mcx);
  }
  else{ //exchange over chunks
    int_t iPair;
    uint_t nPair,mask;
    uint_t baseChunk,iChunk1,iChunk2;
    reg_t qubits(2);
    qubits[0] = qubit;
    qubits[1] = qubit;

    mask = (1ull << qubit);
    mask >>= (chunk_bits_*qubit_scale());

    if(distributed_procs_ == 1 || (distributed_proc_bits_ >= 0 && qubit < (num_qubits_*qubit_scale() - distributed_proc_bits_))){   //no data transfer between processes is needed
      nPair = num_local_chunks_ >> 1;

      auto apply_chunk_swap = [this, mask, qubits](int_t iGroup)
      {
        for(int_t ic = top_chunk_of_group_[iGroup];ic < top_chunk_of_group_[iGroup + 1];ic++){
          uint_t pairChunk;
          pairChunk = ic ^ mask;
          if(ic < pairChunk)
            states_[ic].qreg().apply_chunk_swap(qubits,states_[pairChunk].qreg(),true);
        }
      };
      Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1),0, nPair, apply_chunk_swap);
    }
#ifdef AER_MPI
    else{
      //chunk scheduler that supports any number of processes
      uint_t nu[3];
      uint_t ub[3];
      uint_t iu[3];
      uint_t add;
      uint_t iLocalChunk,iRemoteChunk,iProc;
      int i;

      nLarge = 1;
      nu[0] = 1ull << (qubit - chunk_bits_*qubit_scale());
      ub[0] = 0;
      iu[0] = 0;

      nu[1] = 1ull << (num_qubits_*qubit_scale() - qubit - 1);
      ub[1] = (qubit - chunk_bits_*qubit_scale()) + 1;
      iu[1] = 0;
      nPair = 1ull << (num_qubits_*qubit_scale() - chunk_bits_*qubit_scale() - 1);

      for(iPair=0;iPair<nPair;iPair++){
        //calculate index of pair of chunks
        baseChunk = 0;
        add = 1;
        for(i=1;i>=0;i--){
          baseChunk += (iu[i] << ub[i]);
          //update for next
          iu[i] += add;
          add = 0;
          if(iu[i] >= nu[i]){
            iu[i] = 0;
            add = 1;
          }
        }

        iChunk1 = baseChunk;
        iChunk2 = baseChunk | mask;

        if(iChunk1 >= chunk_index_begin_[distributed_rank_] && iChunk1 < chunk_index_end_[distributed_rank_]){    //chunk1 is on this process
          if(iChunk2 >= chunk_index_begin_[distributed_rank_] && iChunk2 < chunk_index_end_[distributed_rank_]){    //chunk2 is on this process
            states_[iChunk1 - global_chunk_index_].qreg().apply_chunk_swap(qubits,states_[iChunk2 - global_chunk_index_].qreg(),true);
            continue;
          }
          else{
            iLocalChunk = iChunk1;
            iRemoteChunk = iChunk2;
            iProc = get_process_by_chunk(iChunk2);
          }
        }
        else{
          if(iChunk2 >= chunk_index_begin_[distributed_rank_] && iChunk2 < chunk_index_end_[distributed_rank_]){    //chunk2 is on this process
            iLocalChunk = iChunk2;
            iRemoteChunk = iChunk1;
            iProc = get_process_by_chunk(iChunk1);
          }
          else{
            continue;   //there is no chunk for this pair on this process
          }
        }

        MPI_Request reqSend,reqRecv;
        MPI_Status st;
        uint_t sizeRecv,sizeSend;

        auto pSend = states_[iLocalChunk - global_chunk_index_].qreg().send_buffer(sizeSend);
        MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,iPair,distributed_comm_,&reqSend);

        auto pRecv = states_[iLocalChunk - global_chunk_index_].qreg().recv_buffer(sizeRecv);
        MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,iPair,distributed_comm_,&reqRecv);

        MPI_Wait(&reqSend,&st);
        MPI_Wait(&reqRecv,&st);

        states_[iLocalChunk - global_chunk_index_].qreg().apply_chunk_swap(qubits,iRemoteChunk);
      }
    }
#endif

  }
}

template <class state_t>
void ParallelExecutor<state_t>::send_chunk(uint_t local_chunk_index, uint_t global_pair_index)
{
#ifdef AER_MPI
  MPI_Request reqSend;
  MPI_Status st;
  uint_t sizeSend;
  uint_t iProc;

  iProc = get_process_by_chunk(global_pair_index);

  auto pSend = states_[local_chunk_index].qreg().send_buffer(sizeSend);
  MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,local_chunk_index + global_chunk_index_,distributed_comm_,&reqSend);

  MPI_Wait(&reqSend,&st);

  states_[local_chunk_index].qreg().release_send_buffer();
#endif
}

template <class state_t>
void ParallelExecutor<state_t>::recv_chunk(uint_t local_chunk_index, uint_t global_pair_index)
{
#ifdef AER_MPI
  MPI_Request reqRecv;
  MPI_Status st;
  uint_t sizeRecv;
  uint_t iProc;

  iProc = get_process_by_chunk(global_pair_index);

  auto pRecv = states_[local_chunk_index].qreg().recv_buffer(sizeRecv);
  MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,global_pair_index,distributed_comm_,&reqRecv);

  MPI_Wait(&reqRecv,&st);
#endif
}

template <class state_t>
template <class data_t>
void ParallelExecutor<state_t>::send_data(data_t* pSend, uint_t size, uint_t myid,uint_t pairid)
{
#ifdef AER_MPI
  MPI_Request reqSend;
  MPI_Status st;
  uint_t iProc;

  iProc = get_process_by_chunk(pairid);

  MPI_Isend(pSend,size*sizeof(data_t),MPI_BYTE,iProc,myid,distributed_comm_,&reqSend);

  MPI_Wait(&reqSend,&st);
#endif
}

template <class state_t>
template <class data_t>
void ParallelExecutor<state_t>::recv_data(data_t* pRecv, uint_t size, uint_t myid,uint_t pairid)
{
#ifdef AER_MPI
  MPI_Request reqRecv;
  MPI_Status st;
  uint_t iProc;

  iProc = get_process_by_chunk(pairid);

  MPI_Irecv(pRecv,size*sizeof(data_t),MPI_BYTE,iProc,pairid,distributed_comm_,&reqRecv);

  MPI_Wait(&reqRecv,&st);
#endif
}

template <class state_t>
void ParallelExecutor<state_t>::reduce_sum(reg_t& sum) const
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    uint_t i,n = sum.size();
    reg_t tmp(n);
    MPI_Allreduce(&sum[0],&tmp[0],n,MPI_UINT64_T,MPI_SUM,distributed_comm_);
    for(i=0;i<n;i++){
      sum[i] = tmp[i];
    }
  }
#endif
}

template <class state_t>
void ParallelExecutor<state_t>::reduce_sum(rvector_t& sum) const
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    uint_t i,n = sum.size();
    rvector_t tmp(n);
    MPI_Allreduce(&sum[0],&tmp[0],n,MPI_DOUBLE_PRECISION,MPI_SUM,distributed_comm_);
    for(i=0;i<n;i++){
      sum[i] = tmp[i];
    }
  }
#endif
}

template <class state_t>
void ParallelExecutor<state_t>::reduce_sum(complex_t& sum) const
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    complex_t tmp;
    MPI_Allreduce(&sum,&tmp,2,MPI_DOUBLE_PRECISION,MPI_SUM,distributed_comm_);
    sum = tmp;
  }
#endif
}

template <class state_t>
void ParallelExecutor<state_t>::reduce_sum(double& sum) const
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    double tmp;
    MPI_Allreduce(&sum,&tmp,1,MPI_DOUBLE_PRECISION,MPI_SUM,distributed_comm_);
    sum = tmp;
  }
#endif
}

template <class state_t>
void ParallelExecutor<state_t>::gather_value(rvector_t& val) const
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    rvector_t tmp = val;
    MPI_Alltoall(&tmp[0],1,MPI_DOUBLE_PRECISION,&val[0],1,MPI_DOUBLE_PRECISION,distributed_comm_);
  }
#endif
}

template <class state_t>
void ParallelExecutor<state_t>::sync_process(void) const
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    MPI_Barrier(distributed_comm_);
  }
#endif
}

//gather distributed state into vector (if memory is enough)
template <class state_t>
template <class data_t>
void ParallelExecutor<state_t>::gather_state(std::vector<std::complex<data_t>>& state)
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    uint_t size,local_size,global_size,offset;
    int i;
    std::vector<int> recv_counts(distributed_procs_);
    std::vector<int> recv_offset(distributed_procs_);

    global_size = 0;
    for(i=0;i<distributed_procs_;i++){
      recv_offset[i] = (int)(chunk_index_begin_[i] << (chunk_bits_*qubit_scale()))*2;
      recv_counts[i] = (int)((chunk_index_end_[i] - chunk_index_begin_[i]) << (chunk_bits_*qubit_scale()));
      global_size += recv_counts[i];
      recv_counts[i] *= 2;
    }
    if((global_size >> 21) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to gather state"));
    }
    std::vector<std::complex<data_t>> local_state = state;
    state.resize(global_size);

    if(sizeof(std::complex<data_t>) == 16){
      MPI_Allgatherv(local_state.data(),recv_counts[distributed_rank_],MPI_DOUBLE_PRECISION,
                     state.data(),&recv_counts[0],&recv_offset[0],MPI_DOUBLE_PRECISION,distributed_comm_);
    }
    else{
      MPI_Allgatherv(local_state.data(),recv_counts[distributed_rank_],MPI_FLOAT,
                     state.data(),&recv_counts[0],&recv_offset[0],MPI_FLOAT,distributed_comm_);
    }
  }
#endif
}

template <class state_t>
template <class data_t>
void ParallelExecutor<state_t>::gather_state(AER::Vector<std::complex<data_t>>& state)
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    uint_t size,local_size,global_size,offset;
    int i;

    std::vector<int> recv_counts(distributed_procs_);
    std::vector<int> recv_offset(distributed_procs_);

    global_size = 0;
    for(i=0;i<distributed_procs_;i++){
      recv_offset[i] = (int)(chunk_index_begin_[i] << (chunk_bits_*qubit_scale()))*2;
      recv_counts[i] = (int)((chunk_index_end_[i] - chunk_index_begin_[i]) << (chunk_bits_*qubit_scale()));
      global_size += recv_counts[i];
      recv_counts[i] *= 2;
    }
    if((global_size >> 21) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to gather state"));
    }
    AER::Vector<std::complex<data_t>> local_state = state;
    state.resize(global_size);

    if(sizeof(std::complex<data_t>) == 16){
      MPI_Allgatherv(local_state.data(),recv_counts[distributed_rank_],MPI_DOUBLE_PRECISION,
                     state.data(),&recv_counts[0],&recv_offset[0],MPI_DOUBLE_PRECISION,distributed_comm_);
    }
    else{
      MPI_Allgatherv(local_state.data(),recv_counts[distributed_rank_],MPI_FLOAT,
                     state.data(),&recv_counts[0],&recv_offset[0],MPI_FLOAT,distributed_comm_);
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



