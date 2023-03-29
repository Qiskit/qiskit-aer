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

#ifndef _multi_state_executor_hpp_
#define _multi_state_executor_hpp_

#include "simulators/circuit_executor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

#include "simulators/shot_branching.hpp"

namespace AER {

namespace CircuitExecutor {

//-------------------------------------------------------------------------
// Multiple-shots executor class implementation
//-------------------------------------------------------------------------
template <class state_t>
class MultiStateExecutor : public Executor<state_t> {
  using Base = Executor<state_t>;

protected:
  std::vector<state_t> states_;
  std::vector<ClassicalRegister> cregs_; // classical registers for all shots

  // number of qubits for the circuit
  uint_t num_qubits_;

  uint_t num_global_states_; // number of total shots
  uint_t num_local_states_;  // number of local shots

  uint_t global_state_index_; // beginning chunk index for this process
  reg_t state_index_begin_;   // beginning chunk index for each process
  reg_t state_index_end_;     // ending chunk index for each process
  uint_t num_active_states_;  // number of active shots in current loop

  bool shot_omp_parallel_; // using thread parallel to process loop of chunks or
                           // not

  bool set_parallelization_called_ =
      false; // this flag is used to check set_parallelization is already
             // called, if yes the call sets max_batched_shots_
  uint_t num_max_shots_ =
      1; // max number of shots can be stored on available memory

  int max_matrix_qubits_; // max qubits for matrix

  // shot branching
  bool shot_branching_enable_ = true;
  bool runtime_noise_sampling_enable_ = false;

  // group of states (GPU devices)
  uint_t num_groups_; // number of groups of chunks
  reg_t top_state_of_group_;
  reg_t num_states_in_group_;
  int num_threads_per_group_; // number of outer threads per group

  uint_t num_creg_memory_ =
      0; // number of total bits for creg (reserve for multi-shots)
  uint_t num_creg_registers_ = 0;

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 14;

  // Threshold for chopping small values to zero in JSON
  double json_chop_threshold_ = 1e-10;

  // Set a global phase exp(1j * theta) for the state
  bool has_global_phase_ = false;
  complex_t global_phase_ = 1;

public:
  MultiStateExecutor();
  virtual ~MultiStateExecutor();

  size_t required_memory_mb(const Circuit &circuit,
                            const Noise::NoiseModel &noise) const override {
    state_t tmp;
    return tmp.required_memory_mb(circuit.num_qubits, circuit.ops);
  }

  uint_t get_process_by_chunk(uint_t cid);

protected:
  void set_config(const Config &config) override;

  void set_distribution(uint_t num_states);

  virtual uint_t qubit_scale(void) { return 1; }

  virtual bool allocate_states(uint_t num_shots, const Config &config);

  void run_circuit_shots(Circuit &circ, const Noise::NoiseModel &noise,
                         const Config &config, ExperimentResult &result,
                         bool sample_noise) override;

  void run_circuit_with_shot_branching(Circuit &circ,
                                       const Noise::NoiseModel &noise,
                                       const Config &config,
                                       ExperimentResult &result,
                                       bool sample_noise);

  // apply op for shot-branching, return false if op is not applied in sub-class
  virtual bool apply_branching_op(Branch &root, const Operations::Op &op,
                                  ExperimentResult &result, bool final_op) {
    std::cout << "  base is called, implement for each method" << std::endl;
    return false;
  }

  // Apply the global phase
  virtual void apply_global_phase() {}
  void set_global_phase(double theta);

  void set_parallelization(const Circuit &circ,
                           const Noise::NoiseModel &noise) override;

  virtual bool shot_branching_supported(void) {
    return false; // return true in the sub-class if supports shot-branching
  }

  template <typename InputIterator>
  void measure_sampler(InputIterator first_meas, InputIterator last_meas,
                       uint_t shots, Branch &branch, ExperimentResult &result,
                       std::vector<RngEngine> &rng);

  // sampling measure
  virtual std::vector<reg_t> sample_measure(state_t &state, const reg_t &qubits,
                                            uint_t shots,
                                            std::vector<RngEngine> &rng) const {
    // this is for single rng, impement in sub-class for multi-shots case
    return state.sample_measure(qubits, shots, rng[0]);
  }
};

template <class state_t>
MultiStateExecutor<state_t>::MultiStateExecutor() {
  num_global_states_ = 0;
  num_local_states_ = 0;

  shot_omp_parallel_ = false;

  shot_branching_enable_ = false;
}

template <class state_t>
MultiStateExecutor<state_t>::~MultiStateExecutor() {
  states_.clear();
  cregs_.clear();
}

template <class state_t>
void MultiStateExecutor<state_t>::set_config(const Config &config) {
  Base::set_config(config);

  // Set threshold for truncating states to be saved
  json_chop_threshold_ = config.zero_threshold;

  // Set OMP threshold for state update functions
  omp_qubit_threshold_ = config.statevector_parallel_threshold;

  // shot branching optimization
  shot_branching_enable_ = config.shot_branching_enable;
  runtime_noise_sampling_enable_ = config.runtime_noise_sampling_enable;

  if (config.num_threads_per_device.has_value())
    num_threads_per_group_ = config.num_threads_per_device.value();
}

template <class state_t>
void MultiStateExecutor<state_t>::set_global_phase(double theta) {
  if (Linalg::almost_equal(theta, 0.0)) {
    has_global_phase_ = false;
    global_phase_ = 1;
  } else {
    has_global_phase_ = true;
    global_phase_ = std::exp(complex_t(0.0, theta));
  }
}

template <class state_t>
void MultiStateExecutor<state_t>::set_distribution(uint_t num_states) {

  num_global_states_ = num_states;

  state_index_begin_.resize(Base::distributed_procs_);
  state_index_end_.resize(Base::distributed_procs_);
  for (int_t i = 0; i < Base::distributed_procs_; i++) {
    state_index_begin_[i] = num_global_states_ * i / Base::distributed_procs_;
    state_index_end_[i] =
        num_global_states_ * (i + 1) / Base::distributed_procs_;
  }

  num_local_states_ = state_index_end_[Base::distributed_rank_] -
                      state_index_begin_[Base::distributed_rank_];
  global_state_index_ = state_index_begin_[Base::distributed_rank_];
}

template <class state_t>
void MultiStateExecutor<state_t>::set_parallelization(
    const Circuit &circ, const Noise::NoiseModel &noise) {
  Base::set_parallelization(circ, noise);
}

template <class state_t>
bool MultiStateExecutor<state_t>::allocate_states(uint_t num_shots,
                                                  const Config &config) {
  int_t i;
  bool ret = true;

  states_.resize(num_shots);

  num_active_states_ = num_shots;

  // initialize groups
  top_state_of_group_.resize(num_shots);
  num_states_in_group_.resize(num_shots);
  num_groups_ = num_shots;
  for (i = 0; i < num_shots; i++) {
    states_[i].set_config(config);
    states_[i].set_num_global_qubits(num_qubits_);

    top_state_of_group_[i] = i;
    num_states_in_group_[i] = 1;
  }

  return ret;
}

template <class state_t>
void MultiStateExecutor<state_t>::run_circuit_shots(
    Circuit &circ, const Noise::NoiseModel &noise, const Config &config,
    ExperimentResult &result, bool sample_noise) {
  num_qubits_ = circ.num_qubits;
  num_creg_memory_ = circ.num_memory;
  num_creg_registers_ = circ.num_registers;

  if (this->sim_device_ == Device::GPU) {
#ifdef _OPENMP
    if (omp_get_num_threads() == 1)
      shot_omp_parallel_ = true;
#endif
  } else if (this->sim_device_ == Device::ThrustCPU) {
    shot_omp_parallel_ = false;
  }

  set_distribution(circ.shots);
  num_max_shots_ = Base::get_max_parallel_shots(circ, noise);
  if (num_max_shots_ == 0)
    num_max_shots_ = 1;

  bool shot_branching = false;
  if (shot_branching_enable_ && num_local_states_ > 1 &&
      shot_branching_supported()) {
    shot_branching = true;
    if (sample_noise) {
      if (!runtime_noise_sampling_enable_)
        shot_branching = false;
    }
  } else
    shot_branching = false;

  if (shot_branching) {
    // disable cuStateVec if shot-branching is enabled
    if (Base::cuStateVec_enable_)
      Base::cuStateVec_enable_ = false;
    return run_circuit_with_shot_branching(circ, noise, config, result,
                                           sample_noise);
  } else {
    return Base::run_circuit_shots(circ, noise, config, result, sample_noise);
  }
}

template <class state_t>
void MultiStateExecutor<state_t>::run_circuit_with_shot_branching(
    Circuit &circ, const Noise::NoiseModel &noise, const Config &config,
    ExperimentResult &result, bool sample_noise) {
  std::vector<std::shared_ptr<Branch>> branches;
  Noise::NoiseModel dummy_noise;
  state_t dummy_state;
  RngEngine dummy_rng;
  dummy_rng.set_seed(circ.seed); // this is not used actually
  OpItr first;
  OpItr last;

  Circuit circ_opt;
  if (sample_noise) {
    circ_opt = noise.sample_noise(circ, dummy_rng,
                                  Noise::NoiseModel::Method::circuit, true);
    auto fusion_pass = Base::transpile_fusion(circ_opt.opset(), config);
    fusion_pass.optimize_circuit(circ_opt, dummy_noise, dummy_state.opset(),
                                 result);
    first = circ_opt.ops.cbegin();
    last = circ_opt.ops.cend();
    max_matrix_qubits_ = Base::get_max_matrix_qubits(circ_opt);
  } else {
    auto fusion_pass = Base::transpile_fusion(circ.opset(), config);
    fusion_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(),
                                 result);
    first = circ.ops.cbegin();
    last = circ.ops.cend();
    max_matrix_qubits_ = Base::get_max_matrix_qubits(circ);
  }

  // check if there is sequence of measure at the end of operations
  bool can_sample = false;
  OpItr measure_seq = last;
  OpItr it = last - 1;
  int_t num_measure = 0;

  do {
    if (it->type != Operations::OpType::measure) {
      measure_seq = it + 1;
      break;
    }
    num_measure += it->qubits.size();
    it--;
  } while (it != first);

  if (num_measure >= num_qubits_ && measure_seq != last) {
    can_sample = true;
  } else {
    measure_seq = last;
  }

#ifdef AER_MPI
  // if shots are distributed to MPI processes, allocate cregs to be gathered
  if (Base::num_process_per_experiment_ > 1)
    cregs_.resize(circ.shots);
#endif

  // reserve states
  allocate_states(num_max_shots_, config);

  // initialize local shots
  std::vector<RngEngine> shots_storage(num_local_states_);
  for (int_t i = 0; i < num_local_states_; i++) {
    shots_storage[i].set_seed(circ.seed + global_state_index_ + i);
  }

  int_t par_shots =
      std::min((int_t)Base::parallel_shots_, (int_t)num_max_shots_);
  std::vector<ExperimentResult> par_results(par_shots);

  uint_t num_shots_saved = 0;

  // loop until all local shots are simulated
  while (shots_storage.size() > 0) {
    uint_t num_active_states = 1;

    // initial state
    branches.push_back(std::make_shared<Branch>());
    branches[0]->state_index() = 0;
    branches[0]->set_shots(shots_storage);
    branches[0]->op_iterator() = first;
    branches[0]->shot_index() =
        global_state_index_ + num_local_states_ - shots_storage.size();
    shots_storage.clear();

    // initialize initial state
    states_[0].set_parallelization(this->parallel_state_update_);
    states_[0].set_global_phase(circ.global_phase_angle);
    states_[0].enable_density_matrix(!Base::has_statevector_ops_);
    states_[0].initialize_qreg(num_qubits_);
    states_[0].initialize_creg(num_creg_memory_, num_creg_registers_);

    // functor for ops execution
    auto apply_ops_func = [this, &branches, &dummy_rng, &noise, &par_results,
                           measure_seq, par_shots, num_active_states](int_t i) {
      uint_t istate, state_end;
      istate = branches.size() * i / par_shots;
      state_end = branches.size() * (i + 1) / par_shots;
      uint_t nbranch = 0;

      for (; istate < state_end; istate++) {
        while (branches[istate]->op_iterator() != measure_seq) {
          if (!branches[istate]->apply_control_flow(
                  states_[branches[istate]->state_index()].creg(),
                  measure_seq)) {
            if (!branches[istate]->apply_runtime_noise_sampling(
                    states_[branches[istate]->state_index()].creg(),
                    *branches[istate]->op_iterator(), noise)) {
              if (!apply_branching_op(*branches[istate],
                                      *branches[istate]->op_iterator(),
                                      par_results[i], true)) {
                states_[branches[istate]->state_index()].apply_op(
                    *branches[istate]->op_iterator(), par_results[i], dummy_rng,
                    true);
              }
            }
            branches[istate]->advance_iterator();
            if (branches[istate]->num_branches() > 0) {
              nbranch += branches[istate]->num_branches();
              break;
            }
          }
        }
      }
      return nbranch;
    };

    while (num_active_states > 0) { // loop until all branches execute all ops
      uint_t nbranch = 0;

      // apply ops until some branch operations are executed in some branches
      nbranch = Utils::apply_omp_parallel_for_reduction_int(
          (par_shots > 1 && branches.size() > 1), 0, par_shots, apply_ops_func,
          par_shots);

      // repeat until new branch is available
      while (nbranch > 0) {
        uint_t num_states_prev = branches.size();
        for (int_t i = 0; i < num_states_prev; i++) {
          // add new branches
          if (branches[i]->num_branches() > 0) {
            int_t istart = 0;
            for (int_t j = 0; j < branches[i]->num_branches(); j++) {
              if (branches[i]->branches()[j]->num_shots() > 0) {
                // copy shots to the root
                branches[i]->set_shots(branches[i]->branches()[j]->rng_shots());
                branches[i]->additional_ops() =
                    branches[i]->branches()[j]->additional_ops();
                branches[i]->shot_index() =
                    branches[i]->branches()[j]->shot_index();
                // save measured bits to state
                states_[branches[i]->state_index()].creg() =
                    branches[i]->branches()[j]->creg();
                branches[i]->branches()[j].reset();
                istart = j + 1;
                break;
              }
              branches[i]->branches()[j].reset();
            }

            for (int_t j = istart; j < branches[i]->num_branches(); j++) {
              if (branches[i]->branches()[j]->num_shots() > 0) {
                // add new branched state
                uint_t pos = branches.size();
                if (pos >= num_max_shots_) { // if there is not enough memory to
                                             // allocate copied state, shots are
                                             // reserved to the next iteration
                  // reset seed to reproduce same results
                  for (int_t k = 0; k < branches[i]->branches()[j]->num_shots();
                       k++) {
                    branches[i]->branches()[j]->rng_shots()[k].set_seed(
                        branches[i]
                            ->branches()[j]
                            ->rng_shots()[k]
                            .initial_seed());
                  }
                  shots_storage.insert(
                      shots_storage.end(),
                      branches[i]->branches()[j]->rng_shots().begin(),
                      branches[i]->branches()[j]->rng_shots().end());
                } else {
                  branches.push_back(branches[i]->branches()[j]);
                  branches[pos]->state_index() = pos;
                  // copy state to new branch
                  states_[pos].set_parallelization(
                      this->parallel_state_update_);
                  states_[pos].set_global_phase(circ.global_phase_angle);
                  states_[pos].enable_density_matrix(
                      !Base::has_statevector_ops_);
                  states_[pos].qreg().initialize(
                      states_[branches[i]->state_index()].qreg());
                  states_[pos].creg() = branches[pos]->creg();
                }
              } else {
                branches[i]->branches()[j].reset();
              }
            }
          }
        }

        // then execute ops applied after branching (reset, Kraus, noises, etc.)
        auto apply_additional_ops_func = [this, &branches, &dummy_rng, &noise,
                                          &par_results, par_shots](int_t i) {
          uint_t istate, state_end;
          istate = branches.size() * i / par_shots;
          state_end = branches.size() * (i + 1) / par_shots;
          uint_t nbranch = 0;

          for (; istate < state_end; istate++) {
            branches[istate]->clear_branch();
            for (int_t j = 0; j < branches[istate]->additional_ops().size();
                 j++) {
              if (apply_branching_op(*branches[istate],
                                     branches[istate]->additional_ops()[j],
                                     par_results[i], false)) {
                // check if there are new branches
                if (branches[istate]->num_branches() > 0) {
                  // if there are additional ops remaining, queue them on new
                  // branches
                  for (int_t k = j + 1;
                       k < branches[istate]->additional_ops().size(); k++) {
                    for (int_t l = 0; l < branches[istate]->num_branches(); l++)
                      branches[istate]->branches()[l]->add_op_after_branch(
                          branches[istate]->additional_ops()[k]);
                  }
                  nbranch += branches[istate]->num_branches();
                  break;
                }
              } else {
                states_[branches[istate]->state_index()].apply_op(
                    branches[istate]->additional_ops()[j], par_results[i],
                    dummy_rng, false);
              }
            }
            branches[istate]->clear_additional_ops();
          }
          return nbranch;
        };
        nbranch = Utils::apply_omp_parallel_for_reduction_int(
            (par_shots > 1 && branches.size() > 1), 0, par_shots,
            apply_additional_ops_func, par_shots);
      }

      // check if there are remaining ops
      num_active_states = 0;
      for (int_t i = 0; i < branches.size(); i++) {
        if (branches[i]->op_iterator() != measure_seq)
          num_active_states++;
      }
    }

    if (can_sample) {
      // apply sampling measure for each branch
      auto sampling_measure_func = [this, &branches, &par_results, measure_seq,
                                    last, par_shots](int_t i) {
        uint_t istate, state_end;
        istate = branches.size() * i / par_shots;
        state_end = branches.size() * (i + 1) / par_shots;

        for (; istate < state_end; istate++) {
          measure_sampler(measure_seq, last, branches[istate]->num_shots(),
                          *branches[istate], par_results[i],
                          branches[istate]->rng_shots());
        }
      };
      bool can_parallel =
          par_shots > 1 && branches.size() > 1 && !Base::cuStateVec_enable_;
      Utils::apply_omp_parallel_for(can_parallel, 0, par_shots,
                                    sampling_measure_func, par_shots);
    } else {
      // save cregs to result
      auto save_cregs = [this, &branches, &par_results, par_shots](int_t i) {
        uint_t istate, state_end;
        istate = branches.size() * i / par_shots;
        state_end = branches.size() * (i + 1) / par_shots;

        for (; istate < state_end; istate++) {
          if (Base::num_process_per_experiment_ > 1) {
            for (int_t j = 0; j < branches[istate]->num_shots(); j++) {
              cregs_[branches[istate]->shot_index() + j] =
                  states_[branches[istate]->state_index()].creg();
            }
          } else {
            std::string memory_hex =
                states_[branches[istate]->state_index()].creg().memory_hex();
            for (int_t j = 0; j < branches[istate]->num_shots(); j++)
              par_results[i].data.add_accum(static_cast<uint_t>(1ULL), "counts",
                                            memory_hex);
            if (Base::save_creg_memory_) {
              for (int_t j = 0; j < branches[istate]->num_shots(); j++)
                par_results[i].data.add_list(memory_hex, "memory");
            }
          }
        }
      };
      Utils::apply_omp_parallel_for((par_shots > 1), 0, par_shots, save_cregs,
                                    par_shots);
    }

    // clear
    for (int_t i = 0; i < branches.size(); i++) {
      branches[i].reset();
    }
    branches.clear();
  }

  // gather cregs on MPI processes and save to result
  if (Base::num_process_per_experiment_ > 1) {
    Base::gather_creg_memory(cregs_, state_index_begin_);

    // save cregs to result
    auto save_cregs = [this, &par_results, par_shots](int_t i) {
      uint_t i_shot, shot_end;
      i_shot = num_global_states_ * i / par_shots;
      shot_end = num_global_states_ * (i + 1) / par_shots;

      for (; i_shot < shot_end; i_shot++) {
        if (cregs_[i_shot].memory_size() > 0) {
          std::string memory_hex = cregs_[i_shot].memory_hex();
          par_results[i].data.add_accum(static_cast<uint_t>(1ULL), "counts",
                                        memory_hex);
          if (Base::save_creg_memory_) {
            par_results[i].data.add_list(std::move(memory_hex), "memory");
          }
        }
      }
    };
    Utils::apply_omp_parallel_for((par_shots > 1), 0, par_shots, save_cregs,
                                  par_shots);
    cregs_.clear();
  }

  for (auto &res : par_results) {
    result.combine(std::move(res));
  }

  result.metadata.add(true, "shot_branching_enabled");
  result.metadata.add(sample_noise, "runtime_noise_sampling_enabled");
}

template <class state_t>
template <typename InputIterator>
void MultiStateExecutor<state_t>::measure_sampler(InputIterator first_meas,
                                                  InputIterator last_meas,
                                                  uint_t shots, Branch &branch,
                                                  ExperimentResult &result,
                                                  std::vector<RngEngine> &rng) {
  state_t &state = states_[branch.state_index()];
  // Check if meas_circ is empty, and if so return initial creg
  if (first_meas == last_meas) {
    for (int_t i = 0; i < shots; i++) {
      if (Base::num_process_per_experiment_ > 1) {
        cregs_[branch.shot_index() + i] = state.creg();
      } else {
        result.save_count_data(state.creg(), Base::save_creg_memory_);
      }
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
  std::vector<reg_t> all_samples;
  all_samples = sample_measure(state, meas_qubits, shots, rng);
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
  for (int_t i = 0; i < all_samples.size(); i++) {
    creg = state.creg();

    // process memory bit measurements
    for (const auto &pair : memory_map) {
      creg.store_measure(reg_t({all_samples[i][pair.second]}),
                         reg_t({pair.first}), reg_t());
    }
    // process register bit measurements
    for (const auto &pair : register_map) {
      creg.store_measure(reg_t({all_samples[i][pair.second]}), reg_t(),
                         reg_t({pair.first}));
    }

    // process read out errors for memory and registers
    for (const Operations::Op &roerror : roerror_ops)
      creg.apply_roerror(roerror, rng[i]);

    // save creg to gather
    if (Base::num_process_per_experiment_ > 1) {
      for (int_t j = 0; j < shots; j++)
        cregs_[branch.shot_index() + j] = creg;
    } else {
      std::string memory_hex = creg.memory_hex();
      result.data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
      if (Base::save_creg_memory_)
        result.data.add_list(memory_hex, "memory");
    }
  }
}

//-------------------------------------------------------------------------
} // end namespace CircuitExecutor
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
