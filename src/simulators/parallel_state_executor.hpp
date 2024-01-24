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

#include "simulators/multi_state_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace CircuitExecutor {

//-------------------------------------------------------------------------
// Parallel executor class implementation
//-------------------------------------------------------------------------
template <class state_t>
class ParallelStateExecutor : public virtual MultiStateExecutor<state_t> {
  using Base = MultiStateExecutor<state_t>;

protected:
  // extra parameters for parallel simulations
  uint_t chunk_bits_; // number of qubits per chunk

  bool chunk_omp_parallel_; // using thread parallel to process loop of chunks
                            // or not
  bool global_chunk_indexing_; // using global index for control qubits and
                               // diagonal matrix

  reg_t qubit_map_; // qubit map to restore swapped qubits

  bool multi_chunk_swap_enable_ = true; // enable multi-chunk swaps
  // maximum buffer size in qubits for chunk swap
  uint_t chunk_swap_buffer_qubits_ = 15;
  uint_t max_multi_swap_; // maximum swaps can be applied at a time, calculated
                          // by chunk_swap_buffer_bits_

  uint_t cache_block_qubit_ = 0;

public:
  ParallelStateExecutor();
  virtual ~ParallelStateExecutor();

  uint_t get_process_by_chunk(uint_t cid);

protected:
  void set_config(const Config &config) override;

  virtual uint_t qubit_scale(void) { return 1; }

  bool multiple_chunk_required(const Config &config, const Circuit &circuit,
                               const Noise::NoiseModel &noise) const;

  // Return cache blocking transpiler pass
  Transpile::CacheBlocking
  transpile_cache_blocking(const Circuit &circ, const Noise::NoiseModel &noise,
                           const Config &config) const;

  bool allocate(uint_t num_qubits, const Config &config);
  bool allocate_states(uint_t num_shots, const Config &config) override;

  void run_circuit_with_sampling(Circuit &circ, const Config &config,
                                 RngEngine &init_rng,
                                 ResultItr result_it) override;

  void run_circuit_shots(Circuit &circ, const Noise::NoiseModel &noise,
                         const Config &config, RngEngine &init_rng,
                         ResultItr result_it, bool sample_noise) override;

  template <typename InputIterator>
  void measure_sampler(InputIterator first_meas, InputIterator last_meas,
                       uint_t shots, ExperimentResult &result,
                       RngEngine &rng) const;

  // apply operations for multi-chunk simulator
  template <typename InputIterator>
  void apply_ops_chunks(InputIterator first, InputIterator last,
                        ExperimentResult &result, RngEngine &rng, uint_t iparam,
                        bool final_ops);

  // apply ops on cache memory
  template <typename InputIterator>
  void apply_cache_blocking_ops(const int_t iGroup, InputIterator first,
                                InputIterator last, ExperimentResult &result,
                                RngEngine &rng, uint_t iparam);

  // apply parallel operations (implement for each simulation method)
  virtual bool apply_parallel_op(const Operations::Op &op,
                                 ExperimentResult &result, RngEngine &rng,
                                 bool final_op) = 0;

  // store measure to cregs
  void store_measure(const reg_t &outcome, const reg_t &memory,
                     const reg_t &registers);

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
  // Functions for multi-chunk distribution
  //-----------------------------------------------------------------------
  // Helper function for computing expectation value
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) = 0;

  // Apply a save expectation value instruction
  void apply_save_expval(const Operations::Op &op, ExperimentResult &result);

  // swap between chunks
  virtual void apply_chunk_swap(const reg_t &qubits);

  // apply multiple swaps between chunks
  virtual void apply_multi_chunk_swap(const reg_t &qubits);

  // apply X gate over chunks
  virtual void apply_chunk_x(const uint_t qubit);

  // send/receive chunk in receive buffer
  void send_chunk(uint_t local_chunk_index, uint_t global_chunk_index);
  void recv_chunk(uint_t local_chunk_index, uint_t global_chunk_index);

  template <class data_t>
  void send_data(data_t *pSend, uint_t size, uint_t myid, uint_t pairid);
  template <class data_t>
  void recv_data(data_t *pRecv, uint_t size, uint_t myid, uint_t pairid);

  // reduce values over processes
  void reduce_sum(reg_t &sum) const;
  void reduce_sum(rvector_t &sum) const;
  void reduce_sum(complex_t &sum) const;
  void reduce_sum(double &sum) const;

  // gather values on each process
  void gather_value(rvector_t &val) const;

  // barrier all processes
  void sync_process(void) const;

  // gather distributed state into vector (if memory is enough)
  template <class data_t>
  void gather_state(std::vector<std::complex<data_t>> &state);

  template <class data_t>
  void gather_state(AER::Vector<std::complex<data_t>> &state);

  // collect matrix over multiple chunks
  auto apply_to_matrix(bool copy = false);

  uint_t mapped_index(const uint_t idx);
};

template <class state_t>
ParallelStateExecutor<state_t>::ParallelStateExecutor() {
  chunk_omp_parallel_ = false;
  global_chunk_indexing_ = false;
  chunk_bits_ = 0;
  cache_block_qubit_ = 0;
}

template <class state_t>
ParallelStateExecutor<state_t>::~ParallelStateExecutor() {}

template <class state_t>
void ParallelStateExecutor<state_t>::set_config(const Config &config) {
  Base::set_config(config);

  if (config.chunk_swap_buffer_qubits.has_value())
    chunk_swap_buffer_qubits_ = config.chunk_swap_buffer_qubits.value();

  // enable multiple qregs if cache blocking is enabled
  cache_block_qubit_ = 0;
  if (config.blocking_qubits.has_value())
    cache_block_qubit_ = config.blocking_qubits.value();
}

template <class state_t>
bool ParallelStateExecutor<state_t>::multiple_chunk_required(
    const Config &config, const Circuit &circ,
    const Noise::NoiseModel &noise) const {
  if (circ.num_qubits < 3)
    return false;
  if (cache_block_qubit_ >= 2 && cache_block_qubit_ < circ.num_qubits)
    return true;

  if (Base::num_process_per_experiment_ == 1 &&
      Base::sim_device_ == Device::GPU && Base::num_gpus_ > 0) {
    return (Base::max_gpu_memory_mb_ / Base::num_gpus_ <
            Base::required_memory_mb(config, circ, noise));
  }
  if (Base::num_process_per_experiment_ > 1) {
    size_t total_mem = Base::max_memory_mb_;
    if (Base::sim_device_ == Device::GPU)
      total_mem += Base::max_gpu_memory_mb_;
    if (total_mem * Base::num_process_per_experiment_ >
        Base::required_memory_mb(config, circ, noise))
      return true;
  }

  return false;
}

template <class state_t>
Transpile::CacheBlocking
ParallelStateExecutor<state_t>::transpile_cache_blocking(
    const Circuit &circ, const Noise::NoiseModel &noise,
    const Config &config) const {
  Transpile::CacheBlocking cache_block_pass;

  const bool is_matrix = (Base::method_ == Method::density_matrix ||
                          Base::method_ == Method::unitary);
  const auto complex_size = (Base::sim_precision_ == Precision::Single)
                                ? sizeof(std::complex<float>)
                                : sizeof(std::complex<double>);

  cache_block_pass.set_num_processes(Base::num_process_per_experiment_);
  cache_block_pass.set_config(config);

  if (!cache_block_pass.enabled()) {
    // if blocking is not set by config, automatically set if required
    if (multiple_chunk_required(config, circ, noise)) {
      int nplace = Base::num_process_per_experiment_;
      if (Base::sim_device_ == Device::GPU && Base::num_gpus_ > 0)
        nplace *= Base::num_gpus_;
      cache_block_pass.set_blocking(circ.num_qubits,
                                    Base::get_min_memory_mb() << 20, nplace,
                                    complex_size, is_matrix);
    }
  }
  return cache_block_pass;
}

template <class state_t>
bool ParallelStateExecutor<state_t>::allocate(uint_t num_qubits,
                                              const Config &config) {
  uint_t i;
  Base::num_qubits_ = num_qubits;
  chunk_bits_ = cache_block_qubit_;

  global_chunk_indexing_ = false;
  chunk_omp_parallel_ = false;
  if (Base::sim_device_ == Device::GPU) {
#ifdef _OPENMP
    if (omp_get_num_threads() == 1)
      chunk_omp_parallel_ = true;
#endif

    global_chunk_indexing_ = true; // cuStateVec does not handle global chunk
                                   // index for diagonal matrix
#ifdef AER_CUSTATEVEC
    if (!Base::cuStateVec_enable_)
      global_chunk_indexing_ = false;
#endif
  } else if (Base::sim_device_ == Device::ThrustCPU) {
    global_chunk_indexing_ = true;
    chunk_omp_parallel_ = false;
  }

  allocate_states(Base::num_local_states_, config);

  // initialize qubit map
  qubit_map_.resize(Base::num_qubits_);
  for (i = 0; i < Base::num_qubits_; i++) {
    qubit_map_[i] = i;
  }

  if (chunk_bits_ <= chunk_swap_buffer_qubits_ + 1)
    multi_chunk_swap_enable_ = false;
  else
    max_multi_swap_ = chunk_bits_ - chunk_swap_buffer_qubits_;

  return true;
}

template <class state_t>
bool ParallelStateExecutor<state_t>::allocate_states(uint_t num_states,
                                                     const Config &config) {
  uint_t i;
  bool init_states = true;
  uint_t num_states_allocated = num_states;
  // deallocate qregs before reallocation
  if (Base::states_.size() > 0) {
    if (Base::states_.size() == num_states)
      init_states = false; // can reuse allocated chunks
    else
      Base::states_.clear();
  }
  if (init_states) {
    Base::states_.resize(num_states);

    if (Base::num_creg_memory_ != 0 || Base::num_creg_registers_ != 0) {
      for (i = 0; i < num_states; i++) {
        // set number of creg bits before actual initialization
        Base::states_[i].initialize_creg(Base::num_creg_memory_,
                                         Base::num_creg_registers_);
      }
    }
    uint_t gqubits = Base::num_qubits_ * this->qubit_scale();
    uint_t squbits;
    if (chunk_bits_ == 0)
      squbits = Base::num_qubits_ * this->qubit_scale();
    else
      squbits = chunk_bits_ * this->qubit_scale();

    // allocate qregs
    Base::states_[0].set_config(config);
    Base::states_[0].qreg().set_max_matrix_bits(Base::max_matrix_qubits_);
    Base::states_[0].qreg().set_max_sampling_shots(Base::max_sampling_shots_);
    Base::states_[0].qreg().set_num_threads_per_group(
        Base::num_threads_per_group_);
    Base::states_[0].set_num_global_qubits(Base::num_qubits_);
#ifdef AER_CUSTATEVEC
    Base::states_[0].qreg().cuStateVec_enable(Base::cuStateVec_enable_);
#endif
    Base::states_[0].qreg().set_target_gpus(Base::target_gpus_);
    num_states_allocated = Base::states_[0].qreg().chunk_setup(
        squbits, gqubits, Base::global_state_index_, num_states);
    for (i = 1; i < num_states_allocated; i++) {
      Base::states_[i].set_config(config);
      Base::states_[i].qreg().chunk_setup(Base::states_[0].qreg(),
                                          Base::global_state_index_ + i);
      Base::states_[i].qreg().set_num_threads_per_group(
          Base::num_threads_per_group_);
      Base::states_[i].set_num_global_qubits(Base::num_qubits_);
    }
  }
  Base::num_active_states_ = num_states_allocated;

  // initialize groups
  Base::top_state_of_group_.clear();
  Base::num_groups_ = 0;
  for (i = 0; i < num_states_allocated; i++) {
    if (Base::states_[i].qreg().top_of_group()) {
      Base::top_state_of_group_.push_back(i);
      Base::num_groups_++;
    }
  }
  Base::top_state_of_group_.push_back(num_states_allocated);
  Base::num_states_in_group_.resize(Base::num_groups_);
  for (i = 0; i < Base::num_groups_; i++) {
    Base::num_states_in_group_[i] =
        Base::top_state_of_group_[i + 1] - Base::top_state_of_group_[i];
  }
  return (num_states_allocated == num_states);
}

template <class state_t>
uint_t ParallelStateExecutor<state_t>::get_process_by_chunk(uint_t cid) {
  uint_t i;
  for (i = 0; i < Base::distributed_procs_; i++) {
    if (cid >= Base::state_index_begin_[i] && cid < Base::state_index_end_[i]) {
      return i;
    }
  }
  return Base::distributed_procs_;
}

template <class state_t>
uint_t ParallelStateExecutor<state_t>::mapped_index(const uint_t idx) {
  uint_t i, ret = 0;
  uint_t t = idx;

  for (i = 0; i < Base::num_qubits_; i++) {
    if (t & 1) {
      ret |= (1ull << qubit_map_[i]);
    }
    t >>= 1;
  }
  return ret;
}

template <class state_t>
void ParallelStateExecutor<state_t>::run_circuit_with_sampling(
    Circuit &circ, const Config &config, RngEngine &init_rng,
    ResultItr result_it) {

  // Optimize circuit
  Noise::NoiseModel dummy_noise;
  state_t dummy_state;
  ExperimentResult fusion_result;

  // optimize circuit
  bool cache_block = false;
  if (multiple_chunk_required(config, circ, dummy_noise)) {
    auto fusion_pass = Base::transpile_fusion(circ.opset(), config);
    fusion_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(),
                                 fusion_result);

    // Cache blocking pass
    auto cache_block_pass = transpile_cache_blocking(circ, dummy_noise, config);
    cache_block_pass.set_sample_measure(true);
    cache_block_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(),
                                      fusion_result);
    cache_block = cache_block_pass.enabled();
  }
  if (!cache_block) {
    return Executor<state_t>::run_circuit_with_sampling(circ, config, init_rng,
                                                        result_it);
  }
  Base::max_matrix_qubits_ = Base::get_max_matrix_qubits(circ);
  Base::num_bind_params_ = circ.num_bind_params;

  uint_t nchunks =
      1ull << ((circ.num_qubits - cache_block_qubit_) * qubit_scale());

  Base::set_distribution(nchunks);
  allocate(circ.num_qubits, config);

  for (uint_t iparam = 0; iparam < Base::num_bind_params_; iparam++) {
    ExperimentResult &result = *(result_it + iparam);
    result.metadata.copy(fusion_result.metadata);

    // Set state config
    for (uint_t i = 0; i < Base::states_.size(); i++) {
      Base::states_[i].set_parallelization(Base::parallel_state_update_);
      if (circ.global_phase_for_params.size() == circ.num_bind_params)
        Base::states_[i].set_global_phase(circ.global_phase_for_params[iparam]);
      else
        Base::states_[i].set_global_phase(circ.global_phase_angle);
    }

    // run with multi-chunks
    RngEngine rng;
    if (iparam == 0)
      rng = init_rng;
    else if (Base::num_bind_params_ > 1)
      rng.set_seed(circ.seed_for_params[iparam]);
    else
      rng.set_seed(circ.seed);

    auto &ops = circ.ops;
    auto first_meas =
        circ.first_measure_pos; // Position of first measurement op
    bool final_ops = (first_meas == ops.size());

    initialize_qreg(circ.num_qubits);
    for (uint_t i = 0; i < Base::states_.size(); i++) {
      Base::states_[i].initialize_creg(circ.num_memory, circ.num_registers);
    }

    // Run circuit instructions before first measure
    apply_ops_chunks(ops.cbegin(), ops.cbegin() + first_meas, result, rng,
                     iparam, final_ops);

    // Get measurement operations and set of measured qubits
    measure_sampler(circ.ops.begin() + first_meas, circ.ops.end(), circ.shots,
                    result, rng);

    // Add measure sampling metadata
    result.metadata.add(true, "measure_sampling");
    Base::states_[0].add_metadata(result);
  }
}

template <class state_t>
void ParallelStateExecutor<state_t>::run_circuit_shots(
    Circuit &circ, const Noise::NoiseModel &noise, const Config &config,
    RngEngine &init_rng, ResultItr result_it, bool sample_noise) {

  if (!multiple_chunk_required(config, circ, noise)) {
    return Base::run_circuit_shots(circ, noise, config, init_rng, result_it,
                                   sample_noise);
  }

  uint_t nchunks =
      1ull << ((circ.num_qubits - cache_block_qubit_) * qubit_scale());
  Base::num_bind_params_ = circ.num_bind_params;

  // Optimize circuit
  Noise::NoiseModel dummy_noise;
  state_t dummy_state;
  auto fusion_pass = Base::transpile_fusion(circ.opset(), config);
  auto cache_block_pass = transpile_cache_blocking(circ, noise, config);
  ExperimentResult fusion_result;
  if (!sample_noise) {
    fusion_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(),
                                 fusion_result);
    // Cache blocking pass
    cache_block_pass.set_sample_measure(false);
    cache_block_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(),
                                      fusion_result);
    Base::max_matrix_qubits_ = Base::get_max_matrix_qubits(circ);
  } else {
    Base::max_matrix_qubits_ = Base::get_max_matrix_qubits(circ);
    Base::max_matrix_qubits_ =
        std::max(Base::max_matrix_qubits_, (int)fusion_pass.max_qubit);
  }

  Base::set_distribution(nchunks);
  allocate(circ.num_qubits, config);

  for (uint_t iparam = 0; iparam < Base::num_bind_params_; iparam++) {
    if (!sample_noise) {
      ExperimentResult &result = *(result_it + iparam);
      result.metadata.copy(fusion_result.metadata);
    }

    for (uint_t ishot = 0; ishot < circ.shots; ishot++) {
      RngEngine rng;
      if (iparam == 0 && ishot == 0)
        rng = init_rng;
      else if (Base::num_bind_params_ > 1)
        rng.set_seed(circ.seed_for_params[iparam] + ishot);
      else
        rng.set_seed(circ.seed + ishot);

      // Set state config and global phase
      for (uint_t i = 0; i < Base::states_.size(); i++) {
        Base::states_[i].set_parallelization(Base::parallel_state_update_);
        if (circ.global_phase_for_params.size() == circ.num_bind_params)
          Base::states_[i].set_global_phase(
              circ.global_phase_for_params[iparam]);
        else
          Base::states_[i].set_global_phase(circ.global_phase_angle);
      }

      // initialize
      initialize_qreg(circ.num_qubits);
      for (uint_t i = 0; i < Base::states_.size(); i++) {
        Base::states_[i].initialize_creg(circ.num_memory, circ.num_registers);
      }

      if (sample_noise) {
        Circuit circ_opt = noise.sample_noise(circ, rng);
        fusion_pass.optimize_circuit(circ_opt, dummy_noise, dummy_state.opset(),
                                     *(result_it + iparam));
        // Cache blocking pass
        cache_block_pass.set_sample_measure(false);
        cache_block_pass.optimize_circuit(
            circ_opt, dummy_noise, dummy_state.opset(), *(result_it + iparam));

        apply_ops_chunks(circ_opt.ops.cbegin(), circ_opt.ops.cend(),
                         *(result_it + iparam), rng, iparam, true);
      } else {
        apply_ops_chunks(circ.ops.cbegin(), circ.ops.cend(),
                         *(result_it + iparam), rng, iparam, true);
      }
      (result_it + iparam)
          ->save_count_data(Base::states_[0].creg(), Base::save_creg_memory_);
    }
    Base::states_[0].add_metadata(*(result_it + iparam));
  }
}

template <class state_t>
template <typename InputIterator>
void ParallelStateExecutor<state_t>::measure_sampler(InputIterator first_meas,
                                                     InputIterator last_meas,
                                                     uint_t shots,
                                                     ExperimentResult &result,
                                                     RngEngine &rng) const {
  // Check if meas_circ is empty, and if so return initial creg
  if (first_meas == last_meas) {
    while (shots-- > 0) {
      result.save_count_data(Base::states_[0].creg(), Base::save_creg_memory_);
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
  auto all_samples = this->sample_measure(meas_qubits, shots, rng);
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
  uint_t num_memory =
      (memory_map.empty()) ? 0ULL : 1 + memory_map.rbegin()->first;
  uint_t num_registers =
      (register_map.empty()) ? 0ULL : 1 + register_map.rbegin()->first;
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
    result.save_count_data(creg, Base::save_creg_memory_);

    // pop off processed sample
    all_samples.pop_back();
  }
}

template <class state_t>
void ParallelStateExecutor<state_t>::store_measure(const reg_t &outcome,
                                                   const reg_t &memory,
                                                   const reg_t &registers) {
  auto apply_store_measure = [this, outcome, memory, registers](int_t iGroup) {
    int_t iChunk = Base::top_state_of_group_[iGroup];
    int_t nChunk = 1;
#ifdef AER_CUSTATEVEC
    if (Base::cuStateVec_enable_) {
      nChunk = Base::num_states_in_group_[iGroup];
    }
#endif
    for (int_t i = 0; i < nChunk; i++)
      Base::states_[iChunk + i].creg().store_measure(outcome, memory,
                                                     registers);
  };
  Utils::apply_omp_parallel_for((chunk_omp_parallel_ && Base::num_groups_ > 1),
                                0, Base::num_groups_, apply_store_measure);
}

template <class state_t>
void ParallelStateExecutor<state_t>::apply_bfunc(const Operations::Op &op) {
  auto bfunc_kernel = [this, op](int_t iGroup) {
    int_t iChunk = Base::top_state_of_group_[iGroup];
    int_t nChunk = 1;
#ifdef AER_CUSTATEVEC
    if (Base::cuStateVec_enable_) {
      nChunk = Base::num_states_in_group_[iGroup];
    }
#endif
    for (int_t i = 0; i < nChunk; i++)
      Base::states_[iChunk + i].creg().apply_bfunc(op);
  };
  Utils::apply_omp_parallel_for((chunk_omp_parallel_ && Base::num_groups_ > 1),
                                0, Base::num_groups_, bfunc_kernel);
}

template <class state_t>
void ParallelStateExecutor<state_t>::apply_roerror(const Operations::Op &op,
                                                   RngEngine &rng) {
  auto roerror_kernel = [this, op, &rng](int_t iGroup) {
    int_t iChunk = Base::top_state_of_group_[iGroup];
    int_t nChunk = 1;
#ifdef AER_CUSTATEVEC
    if (Base::cuStateVec_enable_) {
      nChunk = Base::num_states_in_group_[iGroup];
    }
#endif
    for (int_t i = 0; i < nChunk; i++)
      Base::states_[iChunk + i].creg().apply_roerror(op, rng);
  };
  Utils::apply_omp_parallel_for((chunk_omp_parallel_ && Base::num_groups_ > 1),
                                0, Base::num_groups_, roerror_kernel);
}

template <class state_t>
template <typename InputIterator>
void ParallelStateExecutor<state_t>::apply_ops_chunks(
    InputIterator first, InputIterator last, ExperimentResult &result,
    RngEngine &rng, uint_t iparam, bool final_ops) {
  uint_t iOp, nOp;
  reg_t multi_swap;

  nOp = std::distance(first, last);
  iOp = 0;

  while (iOp < nOp) {
    const Operations::Op &op_iOp = *(first + iOp);
    if (op_iOp.type == Operations::OpType::gate &&
        op_iOp.name == "swap_chunk") {
      // apply swap between chunks
      if (multi_chunk_swap_enable_ && op_iOp.qubits[0] < chunk_bits_ &&
          op_iOp.qubits[1] >= chunk_bits_) {
        if (Base::distributed_proc_bits_ < 0 ||
            (op_iOp.qubits[1] >=
             (Base::num_qubits_ * qubit_scale() -
              Base::distributed_proc_bits_))) { // apply multi-swap when swap is
                                                // cross
                                                // qubits
          multi_swap.push_back(op_iOp.qubits[0]);
          multi_swap.push_back(op_iOp.qubits[1]);
          if (multi_swap.size() >= max_multi_swap_ * 2) {
            apply_multi_chunk_swap(multi_swap);
            multi_swap.clear();
          }
        } else
          apply_chunk_swap(op_iOp.qubits);
      } else {
        if (multi_swap.size() > 0) {
          apply_multi_chunk_swap(multi_swap);
          multi_swap.clear();
        }
        apply_chunk_swap(op_iOp.qubits);
      }
      iOp++;
      continue;
    } else if (multi_swap.size() > 0) {
      apply_multi_chunk_swap(multi_swap);
      multi_swap.clear();
    }

    if (op_iOp.type == Operations::OpType::sim_op &&
        op_iOp.name == "begin_blocking") {
      // applying sequence of gates inside each chunk

      uint_t iOpEnd = iOp;
      while (iOpEnd < nOp) {
        const Operations::Op op_iOpEnd = *(first + iOpEnd);
        if (op_iOpEnd.type == Operations::OpType::sim_op &&
            op_iOpEnd.name == "end_blocking") {
          break;
        }
        iOpEnd++;
      }

      uint_t iOpBegin = iOp + 1;
      if (Base::num_groups_ > 1 && chunk_omp_parallel_) {
#pragma omp parallel for num_threads(Base::num_groups_)
        for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++)
          apply_cache_blocking_ops(ig, first + iOpBegin, first + iOpEnd, result,
                                   rng, iparam);
      } else {
        for (uint_t ig = 0; ig < Base::num_groups_; ig++)
          apply_cache_blocking_ops(ig, first + iOpBegin, first + iOpEnd, result,
                                   rng, iparam);
      }
      iOp = iOpEnd;
    } else {
      if (op_iOp.has_bind_params) {
        std::vector<Operations::Op> bind_op(1);
        bind_op[0] =
            Operations::bind_parameter(op_iOp, iparam, Base::num_bind_params_);
        if (!apply_parallel_op(bind_op[0], result, rng,
                               final_ops && nOp == iOp + 1)) {
          if (Base::num_groups_ > 1 && chunk_omp_parallel_) {
#pragma omp parallel for num_threads(Base::num_groups_)
            for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++)
              apply_cache_blocking_ops(ig, bind_op.cbegin(), bind_op.cend(),
                                       result, rng, iparam);
          } else {
            for (uint_t ig = 0; ig < Base::num_groups_; ig++)
              apply_cache_blocking_ops(ig, bind_op.cbegin(), bind_op.cend(),
                                       result, rng, iparam);
          }
        }
      } else {
        if (!apply_parallel_op(op_iOp, result, rng,
                               final_ops && nOp == iOp + 1)) {
          if (Base::num_groups_ > 1 && chunk_omp_parallel_) {
#pragma omp parallel for num_threads(Base::num_groups_)
            for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++)
              apply_cache_blocking_ops(ig, first + iOp, first + iOp + 1, result,
                                       rng, iparam);
          } else {
            for (uint_t ig = 0; ig < Base::num_groups_; ig++)
              apply_cache_blocking_ops(ig, first + iOp, first + iOp + 1, result,
                                       rng, iparam);
          }
        }
      }
    }
    iOp++;
  }

  if (multi_swap.size() > 0)
    apply_multi_chunk_swap(multi_swap);

  if (Base::num_groups_ > 1 && chunk_omp_parallel_) {
#pragma omp parallel for num_threads(Base::num_groups_)
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++)
      Base::states_[Base::top_state_of_group_[ig]].qreg().synchronize();
  } else {
    for (uint_t ig = 0; ig < Base::num_groups_; ig++)
      Base::states_[Base::top_state_of_group_[ig]].qreg().synchronize();
  }

  if (Base::sim_device_ == Device::GPU) {
#ifdef AER_THRUST_GPU
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      nDev = 0;
    }
    if (nDev > Base::num_groups_)
      nDev = Base::num_groups_;
    result.metadata.add(nDev, "cacheblocking", "chunk_parallel_gpus");
#endif
  }

#ifdef AER_MPI
  result.metadata.add(multi_chunk_swap_enable_, "cacheblocking",
                      "multiple_chunk_swaps_enable");
  if (multi_chunk_swap_enable_) {
    result.metadata.add(chunk_swap_buffer_qubits_, "cacheblocking",
                        "multiple_chunk_swaps_buffer_qubits");
    result.metadata.add(max_multi_swap_, "cacheblocking",
                        "max_multiple_chunk_swaps");
  }
#endif
}

template <class state_t>
template <typename InputIterator>
void ParallelStateExecutor<state_t>::apply_cache_blocking_ops(
    const int_t iGroup, InputIterator first, InputIterator last,
    ExperimentResult &result, RngEngine &rng, uint_t iparam) {
  // for each chunk in group
  for (uint_t iChunk = Base::top_state_of_group_[iGroup];
       iChunk < Base::top_state_of_group_[iGroup + 1]; iChunk++) {
    // fecth chunk in cache
    if (Base::states_[iChunk].qreg().fetch_chunk()) {
      if (Base::num_bind_params_ > 1) {
        Base::run_circuit_with_parameter_binding(
            Base::states_[iChunk], first, last, result, rng, iparam, false);
      } else {
        Base::states_[iChunk].apply_ops(first, last, result, rng, false);
      }

      // release chunk from cache
      Base::states_[iChunk].qreg().release_chunk();
    }
  }
}

template <class state_t>
template <typename list_t>
void ParallelStateExecutor<state_t>::initialize_from_vector(const list_t &vec) {
  uint_t iChunk;

  if (chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for private(iChunk)
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++) {
        list_t tmp(1ull << (chunk_bits_ * qubit_scale()));
        for (uint_t i = 0; i < (1ull << (chunk_bits_ * qubit_scale())); i++) {
          tmp[i] = vec[((Base::global_state_index_ + iChunk)
                        << (chunk_bits_ * qubit_scale())) +
                       i];
        }
        Base::states_[iChunk].qreg().initialize_from_vector(tmp);
      }
    }
  } else {
    for (iChunk = 0; iChunk < Base::num_local_states_; iChunk++) {
      list_t tmp(1ull << (chunk_bits_ * qubit_scale()));
      for (uint_t i = 0; i < (1ull << (chunk_bits_ * qubit_scale())); i++) {
        tmp[i] = vec[((Base::global_state_index_ + iChunk)
                      << (chunk_bits_ * qubit_scale())) +
                     i];
      }
      Base::states_[iChunk].qreg().initialize_from_vector(tmp);
    }
  }
}

template <class state_t>
template <typename list_t>
void ParallelStateExecutor<state_t>::initialize_from_matrix(const list_t &mat) {
  uint_t iChunk;
  if (chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for private(iChunk)
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++) {
        list_t tmp(1ull << (chunk_bits_), 1ull << (chunk_bits_));
        uint_t irow_chunk = ((iChunk + Base::global_state_index_) >>
                             ((Base::num_qubits_ - chunk_bits_)))
                            << (chunk_bits_);
        uint_t icol_chunk =
            ((iChunk + Base::global_state_index_) &
             ((1ull << ((Base::num_qubits_ - chunk_bits_))) - 1))
            << (chunk_bits_);

        // copy part of state for this chunk
        uint_t i;
        for (i = 0; i < (1ull << (chunk_bits_ * qubit_scale())); i++) {
          uint_t icol = i & ((1ull << chunk_bits_) - 1);
          uint_t irow = i >> chunk_bits_;
          tmp[i] = mat[icol_chunk + icol +
                       ((irow_chunk + irow) << Base::num_qubits_)];
        }
        Base::states_[iChunk].qreg().initialize_from_matrix(tmp);
      }
    }
  } else {
    for (iChunk = 0; iChunk < Base::num_local_states_; iChunk++) {
      list_t tmp(1ull << (chunk_bits_), 1ull << (chunk_bits_));
      uint_t irow_chunk = ((iChunk + Base::global_state_index_) >>
                           ((Base::num_qubits_ - chunk_bits_)))
                          << (chunk_bits_);
      uint_t icol_chunk = ((iChunk + Base::global_state_index_) &
                           ((1ull << ((Base::num_qubits_ - chunk_bits_))) - 1))
                          << (chunk_bits_);

      // copy part of state for this chunk
      uint_t i;
      for (i = 0; i < (1ull << (chunk_bits_ * qubit_scale())); i++) {
        uint_t icol = i & ((1ull << chunk_bits_) - 1);
        uint_t irow = i >> chunk_bits_;
        tmp[i] =
            mat[icol_chunk + icol + ((irow_chunk + irow) << Base::num_qubits_)];
      }
      Base::states_[iChunk].qreg().initialize_from_matrix(tmp);
    }
  }
}

template <class state_t>
auto ParallelStateExecutor<state_t>::apply_to_matrix(bool copy) {
  // this function is used to collect states over chunks
  uint_t iChunk;
  uint_t size = 1ull << (chunk_bits_ * qubit_scale());
  uint_t mask = (1ull << (chunk_bits_)) - 1;
  uint_t num_threads = Base::states_[0].qreg().get_omp_threads();

  size_t size_required =
      2 * (sizeof(std::complex<double>) << (Base::num_qubits_ * 2)) +
      (sizeof(std::complex<double>) << (chunk_bits_ * 2)) *
          Base::num_local_states_;
  if ((size_required >> 20) > Utils::get_system_memory_mb()) {
    throw std::runtime_error(
        std::string("There is not enough memory to store states as matrix"));
  }

  auto matrix = Base::states_[0].qreg().copy_to_matrix();

  if (Base::distributed_rank_ == 0) {
    matrix.resize(1ull << (Base::num_qubits_), 1ull << (Base::num_qubits_));

    auto tmp = Base::states_[0].qreg().copy_to_matrix();
    for (iChunk = 0; iChunk < Base::num_global_states_; iChunk++) {
      int_t i;
      uint_t irow_chunk = (iChunk >> ((Base::num_qubits_ - chunk_bits_)))
                          << chunk_bits_;
      uint_t icol_chunk =
          (iChunk & ((1ull << ((Base::num_qubits_ - chunk_bits_))) - 1))
          << chunk_bits_;

      if (iChunk < Base::num_local_states_) {
        if (copy)
          tmp = Base::states_[iChunk].qreg().copy_to_matrix();
        else
          tmp = Base::states_[iChunk].qreg().move_to_matrix();
      }
#ifdef AER_MPI
      else
        recv_data(tmp.data(), size, 0, iChunk);
#endif
#pragma omp parallel for if (num_threads > 1) num_threads(num_threads)
      for (i = 0; i < (int_t)size; i++) {
        uint_t irow = i >> (chunk_bits_);
        uint_t icol = i & mask;
        uint_t idx =
            ((irow + irow_chunk) << (Base::num_qubits_)) + icol_chunk + icol;
        matrix[idx] = tmp[i];
      }
    }
  } else {
#ifdef AER_MPI
    // send matrices to process 0
    for (iChunk = 0; iChunk < Base::num_global_states_; iChunk++) {
      uint_t iProc = get_process_by_chunk(iChunk);
      if (iProc == Base::distributed_rank_) {
        if (copy) {
          auto tmp = Base::states_[iChunk - Base::global_state_index_]
                         .qreg()
                         .copy_to_matrix();
          send_data(tmp.data(), size, iChunk, 0);
        } else {
          auto tmp = Base::states_[iChunk - Base::global_state_index_]
                         .qreg()
                         .move_to_matrix();
          send_data(tmp.data(), size, iChunk, 0);
        }
      }
    }
#endif
  }

  return matrix;
}

template <class state_t>
void ParallelStateExecutor<state_t>::apply_save_expval(
    const Operations::Op &op, ExperimentResult &result) {
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
    expval_var[0] = expval;                      // mean
    expval_var[1] = sq_expval - expval * expval; // variance
    result.save_data_average(Base::states_[0].creg(), op.string_params[0],
                             expval_var, op.type, op.save_type);
  } else {
    result.save_data_average(Base::states_[0].creg(), op.string_params[0],
                             expval, op.type, op.save_type);
  }
}

template <class state_t>
void ParallelStateExecutor<state_t>::apply_chunk_swap(const reg_t &qubits) {
  uint_t q0, q1;

  q0 = qubits[qubits.size() - 2];
  q1 = qubits[qubits.size() - 1];

  if (qubit_scale() == 1) {
    std::swap(qubit_map_[q0], qubit_map_[q1]);
  }

  if (q0 > q1) {
    std::swap(q0, q1);
  }

  if (q1 < chunk_bits_ * qubit_scale()) {
    // inside chunk
    if (chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for num_threads(Base::num_groups_)
      for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
        for (uint_t iChunk = Base::top_state_of_group_[ig];
             iChunk < Base::top_state_of_group_[ig + 1]; iChunk++)
          Base::states_[iChunk].qreg().apply_mcswap(qubits);
      }
    } else {
      for (uint_t ig = 0; ig < Base::num_groups_; ig++) {
        for (uint_t iChunk = Base::top_state_of_group_[ig];
             iChunk < Base::top_state_of_group_[ig + 1]; iChunk++)
          Base::states_[iChunk].qreg().apply_mcswap(qubits);
      }
    }
  } else { // swap over chunks
    uint_t mask0, mask1;

    mask0 = (1ull << q0);
    mask1 = (1ull << q1);
    mask0 >>= (chunk_bits_ * qubit_scale());
    mask1 >>= (chunk_bits_ * qubit_scale());

    if (Base::distributed_procs_ == 1 ||
        (Base::distributed_proc_bits_ >= 0 &&
         q1 < (Base::num_qubits_ * qubit_scale() -
               Base::distributed_proc_bits_))) { // no data transfer between
                                                 // processes
                                                 // is needed
      auto apply_chunk_swap_1qubit = [this, mask1, qubits](int_t iGroup) {
        for (uint_t ic = Base::top_state_of_group_[iGroup];
             ic < Base::top_state_of_group_[iGroup + 1]; ic++) {
          uint_t baseChunk;
          baseChunk = ic & (~mask1);
          if (ic == baseChunk)
            Base::states_[ic].qreg().apply_chunk_swap(
                qubits, Base::states_[ic | mask1].qreg(), true);
        }
      };
      auto apply_chunk_swap_2qubits = [this, mask0, mask1,
                                       qubits](int_t iGroup) {
        for (uint_t ic = Base::top_state_of_group_[iGroup];
             ic < Base::top_state_of_group_[iGroup + 1]; ic++) {
          uint_t baseChunk;
          baseChunk = ic & (~(mask0 | mask1));
          uint_t iChunk1 = baseChunk | mask0;
          uint_t iChunk2 = baseChunk | mask1;
          if (ic == iChunk1)
            Base::states_[iChunk1].qreg().apply_chunk_swap(
                qubits, Base::states_[iChunk2].qreg(), true);
        }
      };
      if (q0 < chunk_bits_ * qubit_scale())
        Utils::apply_omp_parallel_for(
            (chunk_omp_parallel_ && Base::num_groups_ > 1), 0,
            Base::num_groups_, apply_chunk_swap_1qubit);
      else
        Utils::apply_omp_parallel_for(
            (chunk_omp_parallel_ && Base::num_groups_ > 1), 0,
            Base::num_groups_, apply_chunk_swap_2qubits);
    }
#ifdef AER_MPI
    else {
      uint_t nLarge = 1;
      uint_t iPair;
      uint_t nPair;
      uint_t baseChunk, iChunk1, iChunk2;

      if (q0 < chunk_bits_ * qubit_scale())
        nLarge = 1;
      else
        nLarge = 2;

      // chunk scheduler that supports any number of processes
      uint_t nu[3];
      uint_t ub[3];
      uint_t iu[3];
      uint_t add;
      uint_t iLocalChunk, iRemoteChunk, iProc;
      int i;

      if (q0 < chunk_bits_ * qubit_scale()) {
        nLarge = 1;
        nu[0] = 1ull << (q1 - chunk_bits_ * qubit_scale());
        ub[0] = 0;
        iu[0] = 0;

        nu[1] = 1ull << (Base::num_qubits_ * qubit_scale() - q1 - 1);
        ub[1] = (q1 - chunk_bits_ * qubit_scale()) + 1;
        iu[1] = 0;
      } else {
        nLarge = 2;
        nu[0] = 1ull << (q0 - chunk_bits_ * qubit_scale());
        ub[0] = 0;
        iu[0] = 0;

        nu[1] = 1ull << (q1 - q0 - 1);
        ub[1] = (q0 - chunk_bits_ * qubit_scale()) + 1;
        iu[1] = 0;

        nu[2] = 1ull << (Base::num_qubits_ * qubit_scale() - q1 - 1);
        ub[2] = (q1 - chunk_bits_ * qubit_scale()) + 1;
        iu[2] = 0;
      }
      nPair = 1ull << (Base::num_qubits_ * qubit_scale() -
                       chunk_bits_ * qubit_scale() - nLarge);

      for (iPair = 0; iPair < nPair; iPair++) {
        // calculate index of pair of chunks
        baseChunk = 0;
        add = 1;
        for (i = nLarge; i >= 0; i--) {
          baseChunk += (iu[i] << ub[i]);
          // update for next
          iu[i] += add;
          add = 0;
          if (iu[i] >= nu[i]) {
            iu[i] = 0;
            add = 1;
          }
        }

        iChunk1 = baseChunk | mask0;
        iChunk2 = baseChunk | mask1;

        if (iChunk1 >= Base::state_index_begin_[Base::distributed_rank_] &&
            iChunk1 <
                Base::state_index_end_[Base::distributed_rank_]) { // chunk1 is
                                                                   // on
          // this process
          if (iChunk2 >= Base::state_index_begin_[Base::distributed_rank_] &&
              iChunk2 <
                  Base::state_index_end_[Base::distributed_rank_]) { // chunk2
                                                                     // is on
            // this process
            Base::states_[iChunk1 - Base::global_state_index_]
                .qreg()
                .apply_chunk_swap(
                    qubits,
                    Base::states_[iChunk2 - Base::global_state_index_].qreg(),
                    true);
            continue;
          } else {
            iLocalChunk = iChunk1;
            iRemoteChunk = iChunk2;
            iProc = get_process_by_chunk(iChunk2);
          }
        } else {
          if (iChunk2 >= Base::state_index_begin_[Base::distributed_rank_] &&
              iChunk2 <
                  Base::state_index_end_[Base::distributed_rank_]) { // chunk2
                                                                     // is on
            // this process
            iLocalChunk = iChunk2;
            iRemoteChunk = iChunk1;
            iProc = get_process_by_chunk(iChunk1);
          } else {
            continue; // there is no chunk for this pair on this process
          }
        }

        MPI_Request reqSend, reqRecv;
        MPI_Status st;
        uint_t sizeRecv, sizeSend;

        auto pRecv = Base::states_[iLocalChunk - Base::global_state_index_]
                         .qreg()
                         .recv_buffer(sizeRecv);
        MPI_Irecv(pRecv, sizeRecv, MPI_BYTE, iProc, iPair,
                  Base::distributed_comm_, &reqRecv);

        auto pSend = Base::states_[iLocalChunk - Base::global_state_index_]
                         .qreg()
                         .send_buffer(sizeSend);
        MPI_Isend(pSend, sizeSend, MPI_BYTE, iProc, iPair,
                  Base::distributed_comm_, &reqSend);

        MPI_Wait(&reqSend, &st);
        MPI_Wait(&reqRecv, &st);

        Base::states_[iLocalChunk - Base::global_state_index_]
            .qreg()
            .apply_chunk_swap(qubits, iRemoteChunk);
      }
    }
#endif
  }
}

template <class state_t>
void ParallelStateExecutor<state_t>::apply_multi_chunk_swap(
    const reg_t &qubits) {
  int_t nswap = qubits.size() / 2;
  reg_t chunk_shuffle_qubits(nswap, 0);
  reg_t local_swaps;
  uint_t baseChunk = 0;
  uint_t nchunk = 1ull << nswap;
  reg_t chunk_procs(nchunk);
  reg_t chunk_offset(nchunk);

  if (qubit_scale() == 1) {
    for (int_t i = 0; i < nswap; i++)
      std::swap(qubit_map_[qubits[i * 2]], qubit_map_[qubits[i * 2] + 1]);
  }

  // define local swaps
  for (int_t i = 0; i < nswap; i++) {
    if (qubits[i * 2] >= chunk_bits_ * qubit_scale() - nswap) // no swap
                                                              // required
      chunk_shuffle_qubits[qubits[i * 2] + nswap -
                           chunk_bits_ * qubit_scale()] = qubits[i * 2 + 1];
  }
  int_t pos = 0;
  for (int_t i = 0; i < nswap; i++) {
    if (qubits[i * 2] <
        chunk_bits_ * qubit_scale() - nswap) { // local swap required
      // find empty position
      while (pos < nswap) {
        if (chunk_shuffle_qubits[pos] < chunk_bits_ * qubit_scale()) {
          chunk_shuffle_qubits[pos] = qubits[i * 2 + 1];
          local_swaps.push_back(qubits[i * 2]);
          local_swaps.push_back(chunk_bits_ * qubit_scale() - nswap + pos);
          pos++;
          break;
        }
        pos++;
      }
    }
  }
  for (int_t i = 0; i < nswap; i++)
    chunk_shuffle_qubits[i] -= chunk_bits_ * qubit_scale();

  // swap inside chunks to prepare for all-to-all shuffle
  if (chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (uint_t iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++)
        Base::states_[iChunk].qreg().apply_multi_swaps(local_swaps);
    }
  } else {
    for (uint_t ig = 0; ig < Base::num_groups_; ig++) {
      for (uint_t iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++)
        Base::states_[iChunk].qreg().apply_multi_swaps(local_swaps);
    }
  }

  // apply all-to-all chunk shuffle
  int_t nPair;
  reg_t chunk_shuffle_qubits_sorted = chunk_shuffle_qubits;
  std::sort(chunk_shuffle_qubits_sorted.begin(),
            chunk_shuffle_qubits_sorted.end());

  nPair = Base::num_global_states_ >> nswap;

  for (uint_t i = 0; i < nchunk; i++) {
    chunk_offset[i] = 0;
    for (int_t k = 0; k < nswap; k++) {
      if (((i >> k) & 1) != 0)
        chunk_offset[i] += (1ull << chunk_shuffle_qubits[k]);
    }
  }

#ifdef AER_MPI
  std::vector<MPI_Request> reqSend(nchunk);
  std::vector<MPI_Request> reqRecv(nchunk);
#endif

  for (int_t iPair = 0; iPair < nPair; iPair++) {
    uint_t i1, i2, k, ii, t;
    baseChunk = 0;
    ii = iPair;
    for (k = 0; k < (uint_t)nswap; k++) {
      t = ii & ((1ull << chunk_shuffle_qubits_sorted[k]) - 1);
      baseChunk += t;
      ii = (ii - t) << 1;
    }
    baseChunk += ii;

    for (i1 = 0; i1 < nchunk; i1++) {
      chunk_procs[i1] = get_process_by_chunk(baseChunk + chunk_offset[i1]);
    }

    // all-to-all
    // send data
    for (uint_t iswap = 1; iswap < nchunk; iswap++) {
      uint_t num_local_swap = 0;
      for (i1 = 0; i1 < nchunk; i1++) {
        i2 = i1 ^ iswap;
        if (i1 >= i2)
          continue;

        uint_t iProc1 = chunk_procs[i1];
        uint_t iProc2 = chunk_procs[i2];
        if (iProc1 != Base::distributed_rank_ &&
            iProc2 != Base::distributed_rank_)
          continue;
        if (iProc1 == iProc2) { // on the same process
          num_local_swap++;
          continue; // swap while data is exchanged between processes
        }
#ifdef AER_MPI
        uint_t sizeRecv, sizeSend;
        uint_t offset1 = i1 << (chunk_bits_ * qubit_scale() - nswap);
        uint_t offset2 = i2 << (chunk_bits_ * qubit_scale() - nswap);
        uint_t iChunk1 =
            baseChunk + chunk_offset[i1] - Base::global_state_index_;
        uint_t iChunk2 =
            baseChunk + chunk_offset[i2] - Base::global_state_index_;

        uint_t tid = (iPair << nswap) + iswap;

        if (iProc1 == Base::distributed_rank_) {
          auto pRecv = Base::states_[iChunk1].qreg().recv_buffer(sizeRecv);
          MPI_Irecv(pRecv + offset2, (sizeRecv >> nswap), MPI_BYTE, iProc2, tid,
                    Base::distributed_comm_, &reqRecv[i2]);

          auto pSend = Base::states_[iChunk1].qreg().send_buffer(sizeSend);
          MPI_Isend(pSend + offset2, (sizeSend >> nswap), MPI_BYTE, iProc2, tid,
                    Base::distributed_comm_, &reqSend[i2]);
        } else {
          auto pRecv = Base::states_[iChunk2].qreg().recv_buffer(sizeRecv);
          MPI_Irecv(pRecv + offset1, (sizeRecv >> nswap), MPI_BYTE, iProc1, tid,
                    Base::distributed_comm_, &reqRecv[i1]);

          auto pSend = Base::states_[iChunk2].qreg().send_buffer(sizeSend);
          MPI_Isend(pSend + offset1, (sizeSend >> nswap), MPI_BYTE, iProc1, tid,
                    Base::distributed_comm_, &reqSend[i1]);
        }
#endif
      }

      // swaps inside process
      if (num_local_swap > 0) {
        for (i1 = 0; i1 < nchunk; i1++) {
          i2 = i1 ^ iswap;
          if (i1 > i2)
            continue;

          uint_t iProc1 = chunk_procs[i1];
          uint_t iProc2 = chunk_procs[i2];
          if (iProc1 != Base::distributed_rank_ &&
              iProc2 != Base::distributed_rank_)
            continue;
          if (iProc1 == iProc2) { // on the same process
            uint_t offset1 = i1 << (chunk_bits_ * qubit_scale() - nswap);
            uint_t offset2 = i2 << (chunk_bits_ * qubit_scale() - nswap);
            uint_t iChunk1 =
                baseChunk + chunk_offset[i1] - Base::global_state_index_;
            uint_t iChunk2 =
                baseChunk + chunk_offset[i2] - Base::global_state_index_;
            Base::states_[iChunk1].qreg().apply_chunk_swap(
                Base::states_[iChunk2].qreg(), offset2, offset1,
                (1ull << (chunk_bits_ * qubit_scale() - nswap)));
          }
        }
      }

#ifdef AER_MPI
      // recv data
      for (i1 = 0; i1 < nchunk; i1++) {
        i2 = i1 ^ iswap;

        uint_t iProc1 = chunk_procs[i1];
        uint_t iProc2 = chunk_procs[i2];
        if (iProc1 != Base::distributed_rank_)
          continue;
        if (iProc1 == iProc2) { // on the same process
          continue;
        }
        uint_t iChunk1 =
            baseChunk + chunk_offset[i1] - Base::global_state_index_;
        uint_t offset2 = i2 << (chunk_bits_ * qubit_scale() - nswap);

        MPI_Status st;
        MPI_Wait(&reqSend[i2], &st);
        MPI_Wait(&reqRecv[i2], &st);

        // copy states from recv buffer to chunk
        Base::states_[iChunk1].qreg().apply_chunk_swap(
            Base::states_[iChunk1].qreg(), offset2, offset2,
            (1ull << (chunk_bits_ * qubit_scale() - nswap)));
      }
#endif
    }
  }

  // restore qubits order
  if (chunk_omp_parallel_ && Base::num_groups_ > 1) {
#pragma omp parallel for
    for (int_t ig = 0; ig < (int_t)Base::num_groups_; ig++) {
      for (uint_t iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++)
        Base::states_[iChunk].qreg().apply_multi_swaps(local_swaps);
    }
  } else {
    for (uint_t ig = 0; ig < Base::num_groups_; ig++) {
      for (uint_t iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++)
        Base::states_[iChunk].qreg().apply_multi_swaps(local_swaps);
    }
  }
}

template <class state_t>
void ParallelStateExecutor<state_t>::apply_chunk_x(const uint_t qubit) {
  if (qubit < chunk_bits_ * qubit_scale()) {
    auto apply_mcx = [this, qubit](int_t ig) {
      reg_t qubits(1, qubit);
      for (uint_t iChunk = Base::top_state_of_group_[ig];
           iChunk < Base::top_state_of_group_[ig + 1]; iChunk++)
        Base::states_[iChunk].qreg().apply_mcx(qubits);
    };
    Utils::apply_omp_parallel_for(
        (chunk_omp_parallel_ && Base::num_groups_ > 1), 0, Base::num_groups_,
        apply_mcx);
  } else { // exchange over chunks
    uint_t nPair, mask;
    reg_t qubits(2);
    qubits[0] = qubit;
    qubits[1] = qubit;

    mask = (1ull << qubit);
    mask >>= (chunk_bits_ * qubit_scale());

    if (Base::distributed_procs_ == 1 ||
        (Base::distributed_proc_bits_ >= 0 &&
         qubit < (Base::num_qubits_ * qubit_scale() -
                  Base::distributed_proc_bits_))) { // no data transfer between
                                                    // processes is needed
      nPair = Base::num_local_states_ >> 1;

      auto apply_chunk_swap = [this, mask, qubits](int_t iGroup) {
        for (uint_t ic = Base::top_state_of_group_[iGroup];
             ic < Base::top_state_of_group_[iGroup + 1]; ic++) {
          uint_t pairChunk;
          pairChunk = ic ^ mask;
          if (ic < pairChunk)
            Base::states_[ic].qreg().apply_chunk_swap(
                qubits, Base::states_[pairChunk].qreg(), true);
        }
      };
      Utils::apply_omp_parallel_for(
          (chunk_omp_parallel_ && Base::num_groups_ > 1), 0, nPair,
          apply_chunk_swap);
    }
#ifdef AER_MPI
    else {
      uint_t iPair;
      uint_t baseChunk, iChunk1, iChunk2;

      // chunk scheduler that supports any number of processes
      uint_t nu[3];
      uint_t ub[3];
      uint_t iu[3];
      uint_t add;
      uint_t iLocalChunk, iRemoteChunk, iProc;
      int i;

      nu[0] = 1ull << (qubit - chunk_bits_ * qubit_scale());
      ub[0] = 0;
      iu[0] = 0;

      nu[1] = 1ull << (Base::num_qubits_ * qubit_scale() - qubit - 1);
      ub[1] = (qubit - chunk_bits_ * qubit_scale()) + 1;
      iu[1] = 0;
      nPair = 1ull << (Base::num_qubits_ * qubit_scale() -
                       chunk_bits_ * qubit_scale() - 1);

      for (iPair = 0; iPair < nPair; iPair++) {
        // calculate index of pair of chunks
        baseChunk = 0;
        add = 1;
        for (i = 1; i >= 0; i--) {
          baseChunk += (iu[i] << ub[i]);
          // update for next
          iu[i] += add;
          add = 0;
          if (iu[i] >= nu[i]) {
            iu[i] = 0;
            add = 1;
          }
        }

        iChunk1 = baseChunk;
        iChunk2 = baseChunk | mask;

        if (iChunk1 >= Base::state_index_begin_[Base::distributed_rank_] &&
            iChunk1 <
                Base::state_index_end_[Base::distributed_rank_]) { // chunk1 is
                                                                   // on
          // this process
          if (iChunk2 >= Base::state_index_begin_[Base::distributed_rank_] &&
              iChunk2 <
                  Base::state_index_end_[Base::distributed_rank_]) { // chunk2
                                                                     // is on
            // this process
            Base::states_[iChunk1 - Base::global_state_index_]
                .qreg()
                .apply_chunk_swap(
                    qubits,
                    Base::states_[iChunk2 - Base::global_state_index_].qreg(),
                    true);
            continue;
          } else {
            iLocalChunk = iChunk1;
            iRemoteChunk = iChunk2;
            iProc = get_process_by_chunk(iChunk2);
          }
        } else {
          if (iChunk2 >= Base::state_index_begin_[Base::distributed_rank_] &&
              iChunk2 <
                  Base::state_index_end_[Base::distributed_rank_]) { // chunk2
                                                                     // is on
            // this process
            iLocalChunk = iChunk2;
            iRemoteChunk = iChunk1;
            iProc = get_process_by_chunk(iChunk1);
          } else {
            continue; // there is no chunk for this pair on this process
          }
        }

        MPI_Request reqSend, reqRecv;
        MPI_Status st;
        uint_t sizeRecv, sizeSend;

        auto pSend = Base::states_[iLocalChunk - Base::global_state_index_]
                         .qreg()
                         .send_buffer(sizeSend);
        MPI_Isend(pSend, sizeSend, MPI_BYTE, iProc, iPair,
                  Base::distributed_comm_, &reqSend);

        auto pRecv = Base::states_[iLocalChunk - Base::global_state_index_]
                         .qreg()
                         .recv_buffer(sizeRecv);
        MPI_Irecv(pRecv, sizeRecv, MPI_BYTE, iProc, iPair,
                  Base::distributed_comm_, &reqRecv);

        MPI_Wait(&reqSend, &st);
        MPI_Wait(&reqRecv, &st);

        Base::states_[iLocalChunk - Base::global_state_index_]
            .qreg()
            .apply_chunk_swap(qubits, iRemoteChunk);
      }
    }
#endif
  }
}

template <class state_t>
void ParallelStateExecutor<state_t>::send_chunk(uint_t local_chunk_index,
                                                uint_t global_pair_index) {
#ifdef AER_MPI
  MPI_Request reqSend;
  MPI_Status st;
  uint_t sizeSend;
  uint_t iProc;

  iProc = get_process_by_chunk(global_pair_index);

  auto pSend = Base::states_[local_chunk_index].qreg().send_buffer(sizeSend);
  MPI_Isend(pSend, sizeSend, MPI_BYTE, iProc,
            local_chunk_index + Base::global_state_index_,
            Base::distributed_comm_, &reqSend);

  MPI_Wait(&reqSend, &st);

  Base::states_[local_chunk_index].qreg().release_send_buffer();
#endif
}

template <class state_t>
void ParallelStateExecutor<state_t>::recv_chunk(uint_t local_chunk_index,
                                                uint_t global_pair_index) {
#ifdef AER_MPI
  MPI_Request reqRecv;
  MPI_Status st;
  uint_t sizeRecv;
  uint_t iProc;

  iProc = get_process_by_chunk(global_pair_index);

  auto pRecv = Base::states_[local_chunk_index].qreg().recv_buffer(sizeRecv);
  MPI_Irecv(pRecv, sizeRecv, MPI_BYTE, iProc, global_pair_index,
            Base::distributed_comm_, &reqRecv);

  MPI_Wait(&reqRecv, &st);
#endif
}

template <class state_t>
template <class data_t>
void ParallelStateExecutor<state_t>::send_data(data_t *pSend, uint_t size,
                                               uint_t myid, uint_t pairid) {
#ifdef AER_MPI
  MPI_Request reqSend;
  MPI_Status st;
  uint_t iProc;

  iProc = get_process_by_chunk(pairid);

  MPI_Isend(pSend, size * sizeof(data_t), MPI_BYTE, iProc, myid,
            Base::distributed_comm_, &reqSend);

  MPI_Wait(&reqSend, &st);
#endif
}

template <class state_t>
template <class data_t>
void ParallelStateExecutor<state_t>::recv_data(data_t *pRecv, uint_t size,
                                               uint_t myid, uint_t pairid) {
#ifdef AER_MPI
  MPI_Request reqRecv;
  MPI_Status st;
  uint_t iProc;

  iProc = get_process_by_chunk(pairid);

  MPI_Irecv(pRecv, size * sizeof(data_t), MPI_BYTE, iProc, pairid,
            Base::distributed_comm_, &reqRecv);

  MPI_Wait(&reqRecv, &st);
#endif
}

template <class state_t>
void ParallelStateExecutor<state_t>::reduce_sum(reg_t &sum) const {
#ifdef AER_MPI
  if (Base::distributed_procs_ > 1) {
    uint_t i, n = sum.size();
    reg_t tmp(n);
    MPI_Allreduce(&sum[0], &tmp[0], n, MPI_UINT64_T, MPI_SUM,
                  Base::distributed_comm_);
    for (i = 0; i < n; i++) {
      sum[i] = tmp[i];
    }
  }
#endif
}

template <class state_t>
void ParallelStateExecutor<state_t>::reduce_sum(rvector_t &sum) const {
#ifdef AER_MPI
  if (Base::distributed_procs_ > 1) {
    uint_t i, n = sum.size();
    rvector_t tmp(n);
    MPI_Allreduce(&sum[0], &tmp[0], n, MPI_DOUBLE_PRECISION, MPI_SUM,
                  Base::distributed_comm_);
    for (i = 0; i < n; i++) {
      sum[i] = tmp[i];
    }
  }
#endif
}

template <class state_t>
void ParallelStateExecutor<state_t>::reduce_sum(complex_t &sum) const {
#ifdef AER_MPI
  if (Base::distributed_procs_ > 1) {
    complex_t tmp;
    MPI_Allreduce(&sum, &tmp, 2, MPI_DOUBLE_PRECISION, MPI_SUM,
                  Base::distributed_comm_);
    sum = tmp;
  }
#endif
}

template <class state_t>
void ParallelStateExecutor<state_t>::reduce_sum(double &sum) const {
#ifdef AER_MPI
  if (Base::distributed_procs_ > 1) {
    double tmp;
    MPI_Allreduce(&sum, &tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM,
                  Base::distributed_comm_);
    sum = tmp;
  }
#endif
}

template <class state_t>
void ParallelStateExecutor<state_t>::gather_value(rvector_t &val) const {
#ifdef AER_MPI
  if (Base::distributed_procs_ > 1) {
    rvector_t tmp = val;
    MPI_Alltoall(&tmp[0], 1, MPI_DOUBLE_PRECISION, &val[0], 1,
                 MPI_DOUBLE_PRECISION, Base::distributed_comm_);
  }
#endif
}

template <class state_t>
void ParallelStateExecutor<state_t>::sync_process(void) const {
#ifdef AER_MPI
  if (Base::distributed_procs_ > 1) {
    MPI_Barrier(Base::distributed_comm_);
  }
#endif
}

// gather distributed state into vector (if memory is enough)
template <class state_t>
template <class data_t>
void ParallelStateExecutor<state_t>::gather_state(
    std::vector<std::complex<data_t>> &state) {
#ifdef AER_MPI
  if (Base::distributed_procs_ > 1) {
    uint_t size, local_size, global_size, offset;
    int i;
    std::vector<int> recv_counts(Base::distributed_procs_);
    std::vector<int> recv_offset(Base::distributed_procs_);

    global_size = 0;
    for (i = 0; i < Base::distributed_procs_; i++) {
      recv_offset[i] =
          (int)(Base::state_index_begin_[i] << (chunk_bits_ * qubit_scale())) *
          2;
      recv_counts[i] =
          (int)((Base::state_index_end_[i] - Base::state_index_begin_[i])
                << (chunk_bits_ * qubit_scale()));
      global_size += recv_counts[i];
      recv_counts[i] *= 2;
    }
    if ((global_size >> 21) > Utils::get_system_memory_mb()) {
      throw std::runtime_error(
          std::string("There is not enough memory to gather state"));
    }
    std::vector<std::complex<data_t>> local_state = state;
    state.resize(global_size);

    if (sizeof(std::complex<data_t>) == 16) {
      MPI_Allgatherv(local_state.data(), recv_counts[Base::distributed_rank_],
                     MPI_DOUBLE_PRECISION, state.data(), &recv_counts[0],
                     &recv_offset[0], MPI_DOUBLE_PRECISION,
                     Base::distributed_comm_);
    } else {
      MPI_Allgatherv(local_state.data(), recv_counts[Base::distributed_rank_],
                     MPI_FLOAT, state.data(), &recv_counts[0], &recv_offset[0],
                     MPI_FLOAT, Base::distributed_comm_);
    }
  }
#endif
}

template <class state_t>
template <class data_t>
void ParallelStateExecutor<state_t>::gather_state(
    AER::Vector<std::complex<data_t>> &state) {
#ifdef AER_MPI
  if (Base::distributed_procs_ > 1) {
    uint_t global_size;
    uint_t i;

    std::vector<int> recv_counts(Base::distributed_procs_);
    std::vector<int> recv_offset(Base::distributed_procs_);

    global_size = 0;
    for (i = 0; i < Base::distributed_procs_; i++) {
      recv_offset[i] =
          (int)(Base::state_index_begin_[i] << (chunk_bits_ * qubit_scale())) *
          2;
      recv_counts[i] =
          (int)((Base::state_index_end_[i] - Base::state_index_begin_[i])
                << (chunk_bits_ * qubit_scale()));
      global_size += recv_counts[i];
      recv_counts[i] *= 2;
    }
    if ((global_size >> 21) > Utils::get_system_memory_mb()) {
      throw std::runtime_error(
          std::string("There is not enough memory to gather state"));
    }
    AER::Vector<std::complex<data_t>> local_state = state;
    state.resize(global_size);

    if (sizeof(std::complex<data_t>) == 16) {
      MPI_Allgatherv(local_state.data(), recv_counts[Base::distributed_rank_],
                     MPI_DOUBLE_PRECISION, state.data(), &recv_counts[0],
                     &recv_offset[0], MPI_DOUBLE_PRECISION,
                     Base::distributed_comm_);
    } else {
      MPI_Allgatherv(local_state.data(), recv_counts[Base::distributed_rank_],
                     MPI_FLOAT, state.data(), &recv_counts[0], &recv_offset[0],
                     MPI_FLOAT, Base::distributed_comm_);
    }
  }
#endif
}

//-------------------------------------------------------------------------
} // end namespace CircuitExecutor
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
