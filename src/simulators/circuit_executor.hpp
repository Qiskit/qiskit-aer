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

#ifndef _circuit_executor_hpp_
#define _circuit_executor_hpp_

#include "framework/config.hpp"
#include "framework/creg.hpp"
#include "framework/json.hpp"
#include "framework/opset.hpp"
#include "framework/results/experiment_result.hpp"
#include "framework/results/result.hpp"
#include "framework/rng.hpp"
#include "framework/types.hpp"
#include "noise/noise_model.hpp"

#include "transpile/cacheblocking.hpp"
#include "transpile/fusion.hpp"

#include "simulators/state.hpp"

namespace AER {

namespace CircuitExecutor {

using OpItr = std::vector<Operations::Op>::const_iterator;
using ResultItr = std::vector<ExperimentResult>::iterator;

// Timer type
using myclock_t = std::chrono::high_resolution_clock;

//-------------------------------------------------------------------------
// Executor base class
//-------------------------------------------------------------------------
class Base {
protected:
public:
  Base() {}
  virtual ~Base() {}

  virtual void run_circuit(Circuit &circ, const Noise::NoiseModel &noise,
                           const Config &config, const Method method,
                           const Device device, ResultItr result) = 0;

  // Return an estimate of the required memory for a circuit.
  virtual size_t required_memory_mb(const Config &config,
                                    const Circuit &circuit,
                                    const Noise::NoiseModel &noise) const = 0;
  virtual size_t max_memory_mb(void) = 0;

  virtual bool validate_state(const Config &config, const Circuit &circ,
                              const Noise::NoiseModel &noise,
                              bool throw_except) const = 0;
};

//-------------------------------------------------------------------------
// Simple Executor
//-------------------------------------------------------------------------
template <class state_t>
class Executor : public Base {
protected:
  // Simulation method
  Method method_;

  // Simulation device
  Device sim_device_ = Device::CPU;

  // Simulation precision
  Precision sim_precision_ = Precision::Double;

  // Save counts as memory list
  bool save_creg_memory_ = false;

  // The maximum number of threads to use for various levels of parallelization
  int max_parallel_threads_;

  // Parameters for parallelization management in configuration
  int max_parallel_shots_;
  size_t max_memory_mb_;
  size_t max_gpu_memory_mb_;
  size_t min_gpu_memory_mb_; // minimum size per GPU
  int num_gpus_;             // max number of GPU per process
  reg_t target_gpus_;        // GPUs to be used

  // use explicit parallelization
  bool explicit_parallelization_;

  // Parameters for parallelization management for experiments
  int parallel_experiments_;
  int parallel_shots_;
  int parallel_state_update_;

  // OpenMP qubit threshold
  int omp_qubit_threshold_ = 14;

  // results are stored independently in each process if true
  bool accept_distributed_results_ = true;

  uint_t myrank_;               // process ID
  uint_t nprocs_;               // number of processes
  uint_t distributed_rank_;     // process ID in communicator group
  uint_t distributed_procs_;    // number of processes in communicator group
  uint_t distributed_group_;    // group id of distribution
  int_t distributed_proc_bits_; // distributed_procs_=2^distributed_proc_bits_
                                // (if nprocs != power of 2, set -1)
  uint_t num_process_per_experiment_ = 1;

#ifdef AER_MPI
  // communicator group to simulate a circuit (for multi-experiments)
  MPI_Comm distributed_comm_;
#endif

#ifdef AER_CUSTATEVEC
  // settings for cuStateVec
  bool cuStateVec_enable_ = false;
#endif

  // if circuit has statevector operations or not
  bool has_statevector_ops_;

  // runtime parameter binding
  uint_t num_bind_params_ = 1;
  uint_t num_shots_per_bind_param_ = 1;

public:
  Executor();
  virtual ~Executor() {}

  void run_circuit(Circuit &circ, const Noise::NoiseModel &noise,
                   const Config &config, const Method method,
                   const Device device, ResultItr result) override;

  // Return an estimate of the required memory for a circuit.
  size_t required_memory_mb(const Config &config, const Circuit &circuit,
                            const Noise::NoiseModel &noise) const override {
    state_t tmp;
    tmp.set_config(config);
    uint_t ret = tmp.required_memory_mb(circuit.num_qubits, circuit.ops);
    return ret;
  }
  size_t max_memory_mb(void) override { return max_memory_mb_; }

  bool validate_state(const Config &config, const Circuit &circ,
                      const Noise::NoiseModel &noise,
                      bool throw_except) const override;

protected:
  // Return a fusion transpilation pass configured for the current
  // method, circuit and config
  Transpile::Fusion transpile_fusion(const Operations::OpSet &opset,
                                     const Config &config) const;

  // return maximum number of qubits for matrix
  int_t get_max_matrix_qubits(const Circuit &circ) const;
  int_t get_matrix_bits(const Operations::Op &op) const;

  // Get system memory size
  size_t get_system_memory_mb();
  size_t get_gpu_memory_mb();

  size_t get_min_memory_mb() const {
    if (sim_device_ == Device::GPU && num_gpus_ > 0) {
      return max_gpu_memory_mb_ / num_gpus_; // return per GPU memory size
    }
    return max_memory_mb_;
  }

  // get max shots stored on memory
  uint_t get_max_parallel_shots(const Config &config, const Circuit &circuit,
                                const Noise::NoiseModel &noise) const;

  bool multiple_shots_required(const Circuit &circuit,
                               const Noise::NoiseModel &noise) const;

  // Check if measure sampling optimization is valid for the input circuit
  // for the given method. This checks if operation types before
  // the first measurement in the circuit prevent sampling
  bool check_measure_sampling_opt(const Circuit &circ) const;

  bool has_statevector_ops(const Circuit &circ) const;

  virtual void set_config(const Config &config);
  virtual void set_parallelization(const Config &config, const Circuit &circ,
                                   const Noise::NoiseModel &noise);

  virtual void run_circuit_with_sampling(Circuit &circ, const Config &config,
                                         RngEngine &init_rng, ResultItr result);

  virtual void run_circuit_shots(Circuit &circ, const Noise::NoiseModel &noise,
                                 const Config &config, RngEngine &init_rng,
                                 ResultItr result, bool sample_noise);

  void run_circuit_with_parameter_binding(state_t &state, OpItr first,
                                          OpItr last, ExperimentResult &result,
                                          RngEngine &rng, const uint_t iparam,
                                          bool final_op);

  template <typename InputIterator>
  void measure_sampler(InputIterator first_meas, InputIterator last_meas,
                       uint_t shots, state_t &state, ExperimentResult &result,
                       RngEngine &rng, bool save_creg_to_state = false) const;

#ifdef AER_MPI
  void gather_creg_memory(std::vector<ClassicalRegister> &cregs,
                          reg_t &shot_index);
#endif

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                            RngEngine &rng) const {
    std::vector<reg_t> ret;
    return ret;
  };
  virtual std::vector<reg_t> sample_measure(state_t &state, const reg_t &qubits,
                                            uint_t shots,
                                            std::vector<RngEngine> &rng) const {
    // this is for single rng, impement in sub-class for multi-shots case
    return state.sample_measure(qubits, shots, rng[0]);
  }
};

template <class state_t>
Executor<state_t>::Executor() {
  max_memory_mb_ = 0;
  max_gpu_memory_mb_ = 0;
  max_parallel_threads_ = 0;
  max_parallel_shots_ = 0;

  parallel_shots_ = 1;
  parallel_state_update_ = 1;

  num_process_per_experiment_ = 0;

  num_gpus_ = 0;

  explicit_parallelization_ = false;

  has_statevector_ops_ = false;

  myrank_ = 0;
  nprocs_ = 1;

  distributed_procs_ = 1;
  distributed_rank_ = 0;
  distributed_group_ = 0;
  distributed_proc_bits_ = 0;

#ifdef AER_MPI
  distributed_comm_ = MPI_COMM_WORLD;
#endif
}

template <class state_t>
void Executor<state_t>::set_config(const Config &config) {
  // Load config for memory (creg list data)
  if (config.memory.has_value())
    save_creg_memory_ = config.memory.value();

#ifdef _OPENMP
  // Load OpenMP maximum thread settings
  if (config.max_parallel_threads.has_value())
    max_parallel_threads_ = config.max_parallel_threads.value();
  if (config.max_parallel_shots.has_value())
    max_parallel_shots_ = config.max_parallel_shots.value();
  // Limit max threads based on number of available OpenMP threads
  auto omp_threads = omp_get_max_threads();
  max_parallel_threads_ = (max_parallel_threads_ > 0)
                              ? std::min(max_parallel_threads_, omp_threads)
                              : std::max(1, omp_threads);

  // Set OMP threshold for state update functions
  omp_qubit_threshold_ = config.statevector_parallel_threshold;
#else
  // No OpenMP so we disable parallelization
  max_parallel_threads_ = 1;
  max_parallel_shots_ = 1;
#endif

  // Load configurations for parallelization

  if (config.max_memory_mb.has_value())
    max_memory_mb_ = config.max_memory_mb.value();

  // for debugging
  if (config._parallel_shots.has_value()) {
    parallel_shots_ = config._parallel_shots.value();
    explicit_parallelization_ = true;
  }

  // for debugging
  if (config._parallel_state_update.has_value()) {
    parallel_state_update_ = config._parallel_state_update.value();
    explicit_parallelization_ = true;
  }

  if (explicit_parallelization_) {
    parallel_shots_ = std::max<int>({parallel_shots_, 1});
    parallel_state_update_ = std::max<int>({parallel_state_update_, 1});
  }

  if (config.accept_distributed_results.has_value())
    accept_distributed_results_ = config.accept_distributed_results.value();

#ifdef AER_CUSTATEVEC
  // cuStateVec configs
  cuStateVec_enable_ = false;
  if (config.cuStateVec_enable.has_value())
    cuStateVec_enable_ = config.cuStateVec_enable.value();
#endif

  std::string precision = config.precision;
  if (precision == "double") {
    sim_precision_ = Precision::Double;
  } else if (precision == "single") {
    sim_precision_ = Precision::Single;
  }
  // set target GPUs
#ifdef AER_THRUST_GPU
  int nDev = 0;
  if (sim_device_ == Device::GPU) {
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      nDev = 0;
    }
  }
  if (config.target_gpus.has_value()) {
    target_gpus_ = config.target_gpus.value();
    if (nDev < target_gpus_.size()) {
      throw std::invalid_argument("target_gpus has more GPUs than available.");
    }
    num_gpus_ = target_gpus_.size();
  } else {
    num_gpus_ = nDev;
    target_gpus_.resize(num_gpus_);
    for (int_t i = 0; i < num_gpus_; i++)
      target_gpus_[i] = i;
  }
#endif
}

template <class state_t>
size_t Executor<state_t>::get_system_memory_mb() {
  size_t total_physical_memory = Utils::get_system_memory_mb();
#ifdef AER_MPI
  // get minimum memory size per process
  uint64_t locMem, minMem;
  locMem = total_physical_memory;
  MPI_Allreduce(&locMem, &minMem, 1, MPI_UINT64_T, MPI_MIN, distributed_comm_);
  total_physical_memory = minMem;
#endif

  return total_physical_memory;
}

template <class state_t>
size_t Executor<state_t>::get_gpu_memory_mb() {
  size_t total_physical_memory = 0;
#ifdef AER_THRUST_GPU
  for (int_t iDev = 0; iDev < target_gpus_.size(); iDev++) {
    size_t freeMem, totalMem;
    cudaSetDevice(target_gpus_[iDev]);
    cudaMemGetInfo(&freeMem, &totalMem);
    if (iDev == 0)
      min_gpu_memory_mb_ = totalMem;
    else if (totalMem < min_gpu_memory_mb_)
      min_gpu_memory_mb_ = totalMem;
    total_physical_memory += totalMem;
  }

  min_gpu_memory_mb_ >>= 20;
#endif

#ifdef AER_MPI
  // get minimum memory size per process
  uint64_t locMem, minMem;
  locMem = total_physical_memory;
  MPI_Allreduce(&locMem, &minMem, 1, MPI_UINT64_T, MPI_MIN, distributed_comm_);
  total_physical_memory = minMem;

#endif

  return total_physical_memory >> 20;
}

template <class state_t>
bool Executor<state_t>::multiple_shots_required(
    const Circuit &circ, const Noise::NoiseModel &noise) const {
  if (circ.shots < 2)
    return false;
  if (method_ == Method::density_matrix || method_ == Method::superop ||
      method_ == Method::unitary) {
    return false;
  }

  bool can_sample = check_measure_sampling_opt(circ);

  if (noise.is_ideal()) {
    return !can_sample;
  }

  return true;
}

template <class state_t>
uint_t Executor<state_t>::get_max_parallel_shots(
    const Config &config, const Circuit &circ,
    const Noise::NoiseModel &noise) const {
  uint_t mem = required_memory_mb(config, circ, noise);
  if (mem == 0)
    return circ.shots * circ.num_bind_params;

  if (sim_device_ == Device::GPU && num_gpus_ > 0) {
    return std::min(circ.shots * circ.num_bind_params,
                    (max_gpu_memory_mb_ * 8 / 10 / mem));
  } else {
    return std::min(circ.shots * circ.num_bind_params, (max_memory_mb_ / mem));
  }
}

template <class state_t>
void Executor<state_t>::set_parallelization(const Config &config,
                                            const Circuit &circ,
                                            const Noise::NoiseModel &noise) {
  // MPI setting
  myrank_ = 0;
  nprocs_ = 1;
#ifdef AER_MPI
  int t;
  MPI_Comm_size(MPI_COMM_WORLD, &t);
  nprocs_ = t;
  MPI_Comm_rank(MPI_COMM_WORLD, &t);
  myrank_ = t;
#endif
  if (num_process_per_experiment_ == 0)
    num_process_per_experiment_ = nprocs_;

  distributed_procs_ = num_process_per_experiment_;
  distributed_rank_ = myrank_ % distributed_procs_;
  distributed_group_ = myrank_ / distributed_procs_;

  distributed_proc_bits_ = 0;
  uint_t p = distributed_procs_;
  while (p > 1) {
    if ((p & 1) != 0) { // procs is not power of 2
      distributed_proc_bits_ = -1;
      break;
    }
    distributed_proc_bits_++;
    p >>= 1;
  }

#ifdef AER_MPI
  if (num_process_per_experiment_ != nprocs_) {
    MPI_Comm_split(MPI_COMM_WORLD, (int)distributed_group_,
                   (int)distributed_rank_, &distributed_comm_);
  } else {
    distributed_comm_ = MPI_COMM_WORLD;
  }
#endif

  if (max_memory_mb_ == 0)
    max_memory_mb_ = get_system_memory_mb();
  if (sim_device_ == Device::GPU && num_gpus_ > 0)
    max_gpu_memory_mb_ = get_gpu_memory_mb();

  // number of threads for parallel loop of experiments
  parallel_experiments_ = omp_get_num_threads();

  if (explicit_parallelization_)
    return;

  // Check for trivial parallelization conditions
  switch (method_) {
  case Method::statevector:
  case Method::stabilizer:
  case Method::unitary:
  case Method::matrix_product_state: {
    if (circ.shots == 1 || num_process_per_experiment_ > 1 ||
        (!noise.has_quantum_errors() && check_measure_sampling_opt(circ) &&
         circ.num_bind_params == 1)) {
      parallel_shots_ = 1;
      parallel_state_update_ =
          std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
      return;
    }
    break;
  }
  case Method::density_matrix:
  case Method::superop:
  case Method::tensor_network: {
    if (circ.shots == 1 || num_process_per_experiment_ > 1 ||
        check_measure_sampling_opt(circ)) {
      parallel_shots_ = 1;
      parallel_state_update_ =
          std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
      return;
    }
    break;
  }
  case Method::extended_stabilizer:
    break;
  default:
    throw std::invalid_argument(
        "Cannot set parallelization for unresolved method.");
  }

  // Use a local variable to not override stored maximum based
  // on currently executed circuits
  const auto max_shots =
      (max_parallel_shots_ > 0)
          ? std::min({max_parallel_shots_, max_parallel_threads_})
          : max_parallel_threads_;

  // If we are executing circuits in parallel we disable
  // parallel shots
  if (max_shots == 1 || parallel_experiments_ > 1) {
    parallel_shots_ = 1;
  } else {
    // Parallel shots is > 1
    // Limit parallel shots by available memory and number of shots
    // And assign the remaining threads to state update
    int circ_memory_mb = (int)(required_memory_mb(config, circ, noise) /
                               num_process_per_experiment_);
    size_t mem_size =
        (sim_device_ == Device::GPU) ? max_gpu_memory_mb_ : max_memory_mb_;
    if (mem_size < (size_t)circ_memory_mb)
      throw std::runtime_error(
          "a circuit requires more memory than max_memory_mb.");
    // If circ memory is 0, set it to 1 so that we don't divide by zero
    circ_memory_mb = std::max<int>({1, circ_memory_mb});

    int shots = circ.shots;
    parallel_shots_ = std::min<int>(
        {static_cast<int>(mem_size / (circ_memory_mb * 2)), max_shots, shots});
  }
  parallel_state_update_ =
      (parallel_shots_ > 1)
          ? std::max<int>({1, max_parallel_threads_ / parallel_shots_})
          : std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
}

template <class state_t>
void Executor<state_t>::run_circuit(Circuit &circ,
                                    const Noise::NoiseModel &noise,
                                    const Config &config, const Method method,
                                    const Device device, ResultItr result_it) {
  // Start individual circuit timer
  auto timer_start = myclock_t::now(); // state circuit timer

  // Execute in try block so we can catch errors and return the error message
  // for individual circuit failures.
  try {
    // set configuration
    method_ = method;
    sim_device_ = device;

    set_config(config);
    set_parallelization(config, circ, noise);

    // Rng engine (this one is used to add noise on circuit)
    RngEngine rng;
    rng.set_seed(circ.seed);

    // Output data container
    for (uint_t i = 0; i < circ.num_bind_params; i++) {
      ExperimentResult &result = *(result_it + i);
      result.set_config(config);
      result.metadata.add(method_names_.at(method), "method");
      if (sim_device_ == Device::GPU)
        result.metadata.add("GPU", "device");
      else if (sim_device_ == Device::ThrustCPU)
        result.metadata.add("Thrust", "device");
      else
        result.metadata.add("CPU", "device");

      // Circuit qubit metadata
      result.metadata.add(circ.num_qubits, "num_qubits");
      result.metadata.add(circ.num_memory, "num_clbits");
      result.metadata.add(circ.qubits(), "active_input_qubits");
      result.metadata.add(circ.qubit_map(), "input_qubit_map");
      result.metadata.add(circ.remapped_qubits, "remapped_qubits");
      result.metadata.add(max_memory_mb_, "max_memory_mb");
      if (sim_device_ == Device::GPU)
        result.metadata.add(max_gpu_memory_mb_, "max_gpu_memory_mb");

      // Add measure sampling to metadata
      // Note: this will set to `true` if sampling is enabled for the circuit
      result.metadata.add(false, "measure_sampling");
      result.metadata.add(false, "batched_shots_optimization");
    }

    // Validate gateset and memory requirements, raise exception if they're
    // exceeded
    validate_state(config, circ, noise, true);

    has_statevector_ops_ = has_statevector_ops(circ);

    if (circ.num_qubits > 0) { // do nothing for query steps
      // Choose execution method based on noise and method
      Circuit opt_circ;
      bool noise_sampling = false;

      // Ideal circuit
      if (noise.is_ideal()) {
        opt_circ = circ;
        for (uint_t i = 0; i < circ.num_bind_params; i++) {
          ExperimentResult &result = *(result_it + i);
          result.metadata.add("ideal", "noise");
        }
      }
      // Readout error only
      else if (noise.has_quantum_errors() == false) {
        opt_circ = noise.sample_noise(circ, rng);
        for (uint_t i = 0; i < circ.num_bind_params; i++) {
          ExperimentResult &result = *(result_it + i);
          result.metadata.add("readout", "noise");
        }
      }
      // Superop noise sampling
      else if (method == Method::density_matrix || method == Method::superop ||
               (method == Method::tensor_network && !has_statevector_ops_)) {
        // Sample noise using SuperOp method
        opt_circ =
            noise.sample_noise(circ, rng, Noise::NoiseModel::Method::superop);
        for (uint_t i = 0; i < circ.num_bind_params; i++) {
          ExperimentResult &result = *(result_it + i);
          result.metadata.add("superop", "noise");
        }
      }
      // Kraus noise sampling
      else if (noise.opset().contains(Operations::OpType::kraus) ||
               noise.opset().contains(Operations::OpType::superop)) {
        opt_circ =
            noise.sample_noise(circ, rng, Noise::NoiseModel::Method::kraus);
        for (uint_t i = 0; i < circ.num_bind_params; i++) {
          ExperimentResult &result = *(result_it + i);
          result.metadata.add("kraus", "noise");
        }
      }
      // General circuit noise sampling
      else {
        noise_sampling = true;
        for (uint_t i = 0; i < circ.num_bind_params; i++) {
          ExperimentResult &result = *(result_it + i);
          result.metadata.add("circuit", "noise");
        }
      }

      if (noise_sampling) {
        run_circuit_shots(circ, noise, config, rng, result_it, true);
      } else {
        // Run multishot simulation without noise sampling
        bool can_sample = opt_circ.can_sample;
        can_sample &= check_measure_sampling_opt(opt_circ);

        if (can_sample)
          run_circuit_with_sampling(opt_circ, config, rng, result_it);
        else
          run_circuit_shots(opt_circ, noise, config, rng, result_it, false);
      }
    }
    for (uint_t i = 0; i < circ.num_bind_params; i++) {
      ExperimentResult &result = *(result_it + i);
      // Report success
      result.status = ExperimentResult::Status::completed;

      // Pass through circuit header and add metadata
      result.header = circ.header;
      result.shots = circ.shots;
      if (circ.num_bind_params > 1)
        result.seed = circ.seed_for_params[i];
      else
        result.seed = circ.seed;
      result.metadata.add(parallel_shots_, "parallel_shots");
      result.metadata.add(parallel_state_update_, "parallel_state_update");
      if (circ.num_bind_params > 1) {
        result.metadata.add(true, "runtime_parameter_bind");
        result.metadata.add(circ.num_bind_params, "num_bind_params");
        result.metadata.add(i, "bind_param_index");
      } else {
        result.metadata.add(false, "runtime_parameter_bind");
        result.metadata.add(1, "num_bind_params");
      }
      if (sim_device_ == Device::GPU) {
#ifdef AER_CUSTATEVEC
        result.metadata.add(cuStateVec_enable_, "cuStateVec_enable");
#endif
        result.metadata.add(target_gpus_, "target_gpus");
      }
    }

    // Add timer data
    auto timer_stop = myclock_t::now(); // stop timer
    double time_taken =
        std::chrono::duration<double>(timer_stop - timer_start).count();
    for (uint_t i = 0; i < circ.num_bind_params; i++) {
      ExperimentResult &result = *(result_it + i);
      result.time_taken = time_taken;
      // save time also to metadata to pick time in primitive result
      result.metadata.add(time_taken, "time_taken");
    }
  }
  // If an exception occurs during execution, catch it and pass it to the output
  catch (std::exception &e) {
    for (uint_t i = 0; i < circ.num_bind_params; i++) {
      ExperimentResult &result = *(result_it + i);
      result.status = ExperimentResult::Status::error;
      result.message = e.what();
    }
  }
}

template <class state_t>
void Executor<state_t>::run_circuit_with_sampling(Circuit &circ,
                                                  const Config &config,
                                                  RngEngine &init_rng,
                                                  ResultItr result_it) {
  // Optimize circuit
  Noise::NoiseModel dummy_noise;
  state_t dummy_state;

  auto fusion_pass = transpile_fusion(circ.opset(), config);
  ExperimentResult fusion_result;
  fusion_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(),
                               fusion_result);
  auto max_bits = get_max_matrix_qubits(circ);

  auto first_meas = circ.first_measure_pos; // Position of first measurement op
  bool final_ops = (first_meas == circ.ops.size());

  auto circ_shots = circ.shots;
  circ.shots = 1;
  int_t par_shots = (int_t)get_max_parallel_shots(config, circ, dummy_noise);
  par_shots = std::min((int_t)parallel_shots_, par_shots);
  circ.shots = circ_shots;

  num_bind_params_ = circ.num_bind_params;

  auto run_circuit_lambda = [this, circ, &result_it, &fusion_result, config,
                             init_rng, max_bits, first_meas, final_ops,
                             par_shots](int_t i) {
    uint_t iparam, param_end;
    iparam = circ.num_bind_params * i / par_shots;
    param_end = circ.num_bind_params * (i + 1) / par_shots;

    for (; iparam < param_end; iparam++) {
      ExperimentResult &result = *(result_it + iparam);
      result.metadata.copy(fusion_result.metadata);
      RngEngine rng;
      if (iparam == 0)
        rng = init_rng;
      else
        rng.set_seed(circ.seed_for_params[iparam]);

      // Set state config
      state_t state;
      state.set_config(config);
      state.set_parallelization(parallel_state_update_);

      state.set_distribution(1);
      state.set_max_matrix_qubits(max_bits);
      if (circ.ops.begin() + first_meas != circ.ops.end()) {
        state.set_max_sampling_shots(circ.shots);
      }

      if (circ.global_phase_for_params.size() == circ.num_bind_params)
        state.set_global_phase(circ.global_phase_for_params[iparam]);
      else
        state.set_global_phase(circ.global_phase_angle);

        // allocate qubit register
#ifdef AER_CUSTATEVEC
      state.enable_cuStateVec(cuStateVec_enable_);
#endif
      state.allocate(circ.num_qubits, circ.num_qubits);
      state.set_num_global_qubits(circ.num_qubits);
      state.enable_density_matrix(!has_statevector_ops_);

      // Run circuit instructions before first measure
      state.initialize_qreg(circ.num_qubits);
      state.initialize_creg(circ.num_memory, circ.num_registers);

      if (circ.num_bind_params > 1) {
        run_circuit_with_parameter_binding(state, circ.ops.cbegin(),
                                           circ.ops.cbegin() + first_meas,
                                           result, rng, iparam, final_ops);
      } else {
        state.apply_ops(circ.ops.cbegin(), circ.ops.cbegin() + first_meas,
                        result, rng, final_ops);
      }

      // Get measurement operations and set of measured qubits
      measure_sampler(circ.ops.begin() + first_meas, circ.ops.end(), circ.shots,
                      state, result, rng);
      // Add measure sampling metadata
      result.metadata.add(true, "measure_sampling");

      state.add_metadata(result);
    }
  };
  Utils::apply_omp_parallel_for((par_shots > 1), 0, par_shots,
                                run_circuit_lambda, par_shots);
}

template <class state_t>
void Executor<state_t>::run_circuit_shots(
    Circuit &circ, const Noise::NoiseModel &noise, const Config &config,
    RngEngine &init_rng, ResultItr result_it, bool sample_noise) {

  // insert runtime noise sample ops here
  int_t par_shots = (int_t)get_max_parallel_shots(config, circ, noise);
  par_shots = std::min((int_t)parallel_shots_, par_shots);

  uint_t num_shots = circ.shots * circ.num_bind_params;

  // MPI distribution settings
  std::vector<ClassicalRegister> cregs;
  reg_t shot_begin(distributed_procs_);
  reg_t shot_end(distributed_procs_);
  for (uint_t i = 0; i < distributed_procs_; i++) {
    shot_begin[i] = num_shots * i / distributed_procs_;
    shot_end[i] = num_shots * (i + 1) / distributed_procs_;
  }
  uint_t num_local_shots =
      shot_end[distributed_rank_] - shot_begin[distributed_rank_];

  int max_matrix_qubits = 1;
  if (!sample_noise) {
    Noise::NoiseModel dummy_noise;
    state_t dummy_state;
    ExperimentResult fusion_result;
    auto fusion_pass = transpile_fusion(circ.opset(), config);
    fusion_pass.optimize_circuit(circ, dummy_noise, dummy_state.opset(),
                                 fusion_result);
    for (uint_t i = 0; i < circ.num_bind_params; i++) {
      ExperimentResult &result = *(result_it + i);
      result.metadata.copy(fusion_result.metadata);
    }
    max_matrix_qubits = get_max_matrix_qubits(circ);
  }
  num_bind_params_ = circ.num_bind_params;

  std::vector<std::vector<ExperimentResult>> par_results(par_shots);
  for (int_t i = 0; i < par_shots; i++) {
    par_results[i].resize(num_bind_params_);
  }

  if (distributed_procs_ > 1)
    cregs.resize(num_shots);

  // run each shot
  auto run_circuit_lambda = [this, &par_results, circ, noise, config, par_shots,
                             sample_noise, num_shots, shot_begin, &cregs,
                             init_rng, max_matrix_qubits,
                             num_local_shots](int_t i) {
    state_t state;
    uint_t i_shot, e_shot;
    i_shot = num_local_shots * i / par_shots;
    e_shot = num_local_shots * (i + 1) / par_shots;

    auto fusion_pass = transpile_fusion(circ.opset(), config);

    // Set state config
    state.set_config(config);
    state.set_parallelization(this->parallel_state_update_);
    state.enable_density_matrix(!has_statevector_ops_);

    state.set_distribution(this->num_process_per_experiment_);
    state.set_num_global_qubits(circ.num_qubits);

    for (; i_shot < e_shot; i_shot++) {
      RngEngine rng;
      uint_t shot_index = shot_begin[distributed_rank_] + i_shot;
      uint_t iparam = shot_index / circ.shots;
      if (shot_index == 0 && iparam == 0)
        rng = init_rng;
      else {
        if (circ.num_bind_params > 1) {
          uint_t lid = shot_index % circ.shots;
          rng.set_seed(circ.seed_for_params[iparam] + lid);
        } else
          rng.set_seed(circ.seed + shot_index);
      }
      ExperimentResult &result = par_results[i][iparam];

      Circuit circ_opt;
      if (sample_noise) {
        Noise::NoiseModel dummy_noise;
        circ_opt = noise.sample_noise(circ, rng);
        fusion_pass.optimize_circuit(circ_opt, dummy_noise, state.opset(),
                                     result);
        int max_bits = get_max_matrix_qubits(circ_opt);
        state.set_max_matrix_qubits(
            std::max(max_bits, (int)fusion_pass.max_qubit));
      } else
        state.set_max_matrix_qubits(max_matrix_qubits);

      if (circ.global_phase_for_params.size() == circ.num_bind_params)
        state.set_global_phase(circ.global_phase_for_params[iparam]);
      else
        state.set_global_phase(circ.global_phase_angle);
#ifdef AER_CUSTATEVEC
      state.enable_cuStateVec(cuStateVec_enable_);
#endif
      state.allocate(circ.num_qubits, circ.num_qubits);
      state.initialize_qreg(circ.num_qubits);
      state.initialize_creg(circ.num_memory, circ.num_registers);

      if (sample_noise) {
        if (circ.num_bind_params > 1) {
          run_circuit_with_parameter_binding(state, circ_opt.ops.cbegin(),
                                             circ_opt.ops.cend(), result, rng,
                                             iparam, true);
        } else {
          state.apply_ops(circ_opt.ops.cbegin(), circ_opt.ops.cend(), result,
                          rng, true);
        }
      } else {
        if (circ.num_bind_params > 1) {
          run_circuit_with_parameter_binding(state, circ.ops.cbegin(),
                                             circ.ops.cend(), result, rng,
                                             iparam, true);
        } else {
          state.apply_ops(circ.ops.cbegin(), circ.ops.cend(), result, rng,
                          true);
        }
      }
      if (distributed_procs_ > 1) {
        // save creg to be gathered
        cregs[shot_index] = state.creg();
      } else {
        result.save_count_data(state.creg(), save_creg_memory_);
      }
      state.add_metadata(result);
    }
  };
  Utils::apply_omp_parallel_for((par_shots > 1), 0, par_shots,
                                run_circuit_lambda, par_shots);

  // gather cregs on MPI processes and save to result
#ifdef AER_MPI
  if (num_process_per_experiment_ > 1) {
    gather_creg_memory(cregs, shot_begin);

    // save cregs to result
    num_shots = circ.shots * circ.num_bind_params;
    auto save_cregs = [this, &par_results, par_shots, num_shots, circ,
                       cregs](int_t i) {
      uint_t i_shot, e_shot;
      i_shot = num_shots * i / par_shots;
      e_shot = num_shots * (i + 1) / par_shots;

      for (; i_shot < e_shot; i_shot++) {
        uint_t ip = i_shot / circ.shots;
        par_results[i][ip].save_count_data(cregs[i_shot], save_creg_memory_);
      }
    };
    Utils::apply_omp_parallel_for((par_shots > 1), 0, par_shots, save_cregs,
                                  par_shots);
  }
#endif

  for (auto &res : par_results) {
    for (uint_t i = 0; i < circ.num_bind_params; i++) {
      (result_it + i)->combine(std::move(res[i]));
    }
  }
  if (sim_device_ == Device::GPU) {
    for (uint_t i = 0; i < circ.num_bind_params; i++) {
#ifdef AER_CUSTATEVEC
      (result_it + i)->metadata.add(cuStateVec_enable_, "cuStateVec_enable");
#endif
      if (par_shots >= num_gpus_)
        (result_it + i)->metadata.add(num_gpus_, "gpu_parallel_shots_");
      else
        (result_it + i)->metadata.add(par_shots, "gpu_parallel_shots_");
    }
  }
}

template <class state_t>
void Executor<state_t>::run_circuit_with_parameter_binding(
    state_t &state, OpItr first, OpItr last, ExperimentResult &result,
    RngEngine &rng, const uint_t iparam, bool final_op) {
  OpItr op_begin = first;
  OpItr op = first;

  while (op != last) {
    // run with parameter bind
    if (op->has_bind_params) {
      if (op_begin != op) {
        // run ops before this
        state.apply_ops(op_begin, op, result, rng, false);
      }

      std::vector<Operations::Op> binded_op(1);
      binded_op[0] = Operations::bind_parameter(*op, iparam, num_bind_params_);
      state.apply_ops(binded_op.cbegin(), binded_op.cend(), result, rng,
                      final_op && (op == last - 1));
      op_begin = op + 1;
    }
    op++;
  }
  if (op_begin != last) {
    state.apply_ops(op_begin, last, result, rng, final_op);
  }
}

template <class state_t>
template <typename InputIterator>
void Executor<state_t>::measure_sampler(InputIterator first_meas,
                                        InputIterator last_meas, uint_t shots,
                                        state_t &state,
                                        ExperimentResult &result,
                                        RngEngine &rng,
                                        bool save_creg_to_state) const {
  // Check if meas_circ is empty, and if so return initial creg
  if (first_meas == last_meas) {
    while (shots-- > 0) {
      result.save_count_data(state.creg(), save_creg_memory_);
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
  all_samples = state.sample_measure(meas_qubits, shots, rng);
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
  for (int_t i = all_samples.size() - 1; i >= 0; i--) {
    creg.initialize(num_memory, num_registers);

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
      creg.apply_roerror(roerror, rng);

    // Save count data
    if (save_creg_to_state)
      state.creg() = creg;
    else
      result.save_count_data(creg, save_creg_memory_);
  }
}

template <class state_t>
bool Executor<state_t>::validate_state(const Config &config,
                                       const Circuit &circ,
                                       const Noise::NoiseModel &noise,
                                       bool throw_except) const {
  std::stringstream error_msg;
  std::string circ_name;
  state_t state;

  JSON::get_value(circ_name, "name", circ.header);

  state.set_config(config);
  // Check if a circuit is valid for state ops
  bool circ_valid = state.opset().contains(circ.opset());
  if (throw_except && !circ_valid) {
    error_msg << "Circuit " << circ_name << " contains invalid instructions ";
    error_msg << state.opset().difference(circ.opset());
    error_msg << " for \"" << state.name() << "\" method.";
  }
  // check parameters set inf ops
  circ_valid &= state.validate_parameters(circ.ops);
  if (throw_except && !circ_valid) {
    error_msg << "Circuit " << circ_name << " contains invalid parameters ";
    error_msg << " for \"" << state.name() << "\" method.";
  }

  // Check if a noise model valid for state ops
  bool noise_valid = noise.is_ideal() || state.opset().contains(noise.opset());
  if (throw_except && !noise_valid) {
    error_msg << "Noise model contains invalid instructions ";
    error_msg << state.opset().difference(noise.opset());
    error_msg << " for \"" << state.name() << "\" method.";
  }

  // Validate memory requirements
  bool memory_valid = true;
  if (max_memory_mb_ > 0) {
    size_t required_mb = state.required_memory_mb(circ.num_qubits, circ.ops) /
                         num_process_per_experiment_;
    size_t mem_size = (sim_device_ == Device::GPU)
                          ? max_memory_mb_ + max_gpu_memory_mb_
                          : max_memory_mb_;
    memory_valid = (required_mb <= mem_size);
    if (throw_except && !memory_valid) {
      error_msg << "Insufficient memory to run circuit " << circ_name;
      error_msg << " using the " << state.name() << " simulator.";
      error_msg << " Required memory: " << required_mb
                << "M, max memory: " << max_memory_mb_ << "M";
      if (sim_device_ == Device::GPU) {
        error_msg << " (Host) + " << max_gpu_memory_mb_ << "M (GPU)";
      }
    }
  }

  if (noise_valid && circ_valid && memory_valid) {
    return true;
  }

  // One of the validation checks failed for the current state
  if (throw_except) {
    throw std::runtime_error(error_msg.str());
  }
  return false;
}

template <class state_t>
Transpile::Fusion
Executor<state_t>::transpile_fusion(const Operations::OpSet &opset,
                                    const Config &config) const {
  Transpile::Fusion fusion_pass;
  fusion_pass.set_parallelization(parallel_state_update_);

  if (opset.contains(Operations::OpType::superop)) {
    fusion_pass.allow_superop = true;
  }
  if (opset.contains(Operations::OpType::kraus)) {
    fusion_pass.allow_kraus = true;
  }
  switch (method_) {
  case Method::density_matrix:
  case Method::superop: {
    // Halve the default threshold and max fused qubits for density matrix
    fusion_pass.threshold /= 2;
    fusion_pass.max_qubit /= 2;
    break;
  }
  case Method::matrix_product_state: {
    fusion_pass.active = false;
    return fusion_pass; // Do not allow the config to set active for MPS
  }
  case Method::statevector: {
    if (fusion_pass.allow_kraus) {
      // Halve default max fused qubits for Kraus noise fusion
      fusion_pass.max_qubit /= 2;
    }
    break;
  }
  case Method::unitary: {
    // max_qubit is the same with statevector
    fusion_pass.threshold /= 2;
    break;
  }
  case Method::tensor_network: {
    if (opset.contains(Operations::OpType::save_statevec) ||
        opset.contains(Operations::OpType::save_statevec_dict)) {
      if (fusion_pass.allow_kraus) {
        // Halve default max fused qubits for Kraus noise fusion
        fusion_pass.max_qubit /= 2;
      }
    } else {
      // Halve the default threshold and max fused qubits for density matrix
      fusion_pass.threshold /= 2;
      fusion_pass.max_qubit /= 2;
    }
    break;
  }
  default: {
    fusion_pass.active = false;
    return fusion_pass;
  }
  }
  // Override default fusion settings with custom config
  fusion_pass.set_config(config);
  return fusion_pass;
}

template <class state_t>
bool Executor<state_t>::check_measure_sampling_opt(const Circuit &circ) const {
  // Check if circuit has sampling flag disabled
  if (circ.can_sample == false) {
    return false;
  }

  // If density matrix, unitary, superop method all supported instructions
  // allow sampling
  if (method_ == Method::density_matrix || method_ == Method::superop ||
      method_ == Method::unitary) {
    return true;
  }
  if (method_ == Method::tensor_network) {
    // if there are no save statevec ops, tensor network simulator runs as
    // density matrix simulator
    if ((!circ.opset().contains(Operations::OpType::save_statevec)) &&
        (!circ.opset().contains(Operations::OpType::save_statevec_dict))) {
      return true;
    }
  }

  // If circuit contains a non-initial initialize that is not a full width
  // instruction we can't sample
  if (circ.can_sample_initialize == false) {
    return false;
  }

  // Check if non-density matrix simulation and circuit contains
  // a stochastic instruction before measurement
  // ie. reset, kraus, superop
  // TODO:
  // * Resets should be allowed if applied to |0> state (no gates before).
  if (circ.opset().contains(Operations::OpType::reset) ||
      circ.opset().contains(Operations::OpType::kraus) ||
      circ.opset().contains(Operations::OpType::superop) ||
      circ.opset().contains(Operations::OpType::jump) ||
      circ.opset().contains(Operations::OpType::mark)) {
    return false;
  }
  // Otherwise true
  return true;
}

template <class state_t>
int_t Executor<state_t>::get_matrix_bits(const Operations::Op &op) const {
  int_t bit = 1;
  if (op.type == Operations::OpType::matrix ||
      op.type == Operations::OpType::diagonal_matrix ||
      op.type == Operations::OpType::initialize)
    bit = op.qubits.size();
  else if (op.type == Operations::OpType::kraus ||
           op.type == Operations::OpType::superop) {
    if (method_ == Method::density_matrix)
      bit = op.qubits.size() * 2;
    else
      bit = op.qubits.size();
  }
  return bit;
}

template <class state_t>
int_t Executor<state_t>::get_max_matrix_qubits(const Circuit &circ) const {
  int_t max_bits = 0;
  uint_t i;

  if (sim_device_ != Device::CPU) { // Only applicable for GPU (and Thrust)
    for (i = 0; i < circ.ops.size(); i++) {
      int_t bit = 1;
      bit = get_matrix_bits(circ.ops[i]);
      max_bits = std::max(max_bits, bit);
    }
  }
  return max_bits;
}

template <class state_t>
bool Executor<state_t>::has_statevector_ops(const Circuit &circ) const {
  return circ.opset().contains(Operations::OpType::save_statevec) ||
         circ.opset().contains(Operations::OpType::save_statevec_dict) ||
         circ.opset().contains(Operations::OpType::save_amps);
}

#ifdef AER_MPI
template <class state_t>
void Executor<state_t>::gather_creg_memory(
    std::vector<ClassicalRegister> &cregs, reg_t &shot_index) {
  uint_t n64, i64, ibit, num_local_shots;

  if (distributed_procs_ == 0)
    return;
  if (cregs.size() == 0)
    return;
  int_t size = cregs[0].memory_size();
  if (size == 0)
    return;

  if (distributed_rank_ == distributed_procs_ - 1)
    num_local_shots = cregs.size() - shot_index[distributed_rank_];
  else
    num_local_shots =
        shot_index[distributed_rank_ + 1] - shot_index[distributed_rank_];

  // number of 64-bit integers per memory
  n64 = (size + 63) >> 6;

  reg_t bin_memory(n64 * num_local_shots, 0);
  // compress memory string to binary
#pragma omp parallel for private(i64, ibit)
  for (int_t i = 0; i < (int_t)num_local_shots; i++) {
    for (int_t j = 0; j < size; j++) {
      i64 = j >> 6;
      ibit = j & 63;
      if (cregs[shot_index[distributed_rank_] + i].creg_memory()[j] == '1') {
        bin_memory[i * n64 + i64] |= (1ull << ibit);
      }
    }
  }

  reg_t recv(n64 * cregs.size());
  std::vector<int> recv_counts(distributed_procs_);
  std::vector<int> recv_offset(distributed_procs_);

  for (uint_t i = 0; i < distributed_procs_ - 1; i++) {
    recv_offset[i] = shot_index[i];
    recv_counts[i] = shot_index[i + 1] - shot_index[i];
  }
  recv_offset[distributed_procs_ - 1] = shot_index[distributed_procs_ - 1];
  recv_counts[distributed_procs_ - 1] =
      cregs.size() - shot_index[distributed_procs_ - 1];

  MPI_Allgatherv(&bin_memory[0], n64 * num_local_shots, MPI_UINT64_T, &recv[0],
                 &recv_counts[0], &recv_offset[0], MPI_UINT64_T,
                 distributed_comm_);

  // store gathered memory
#pragma omp parallel for private(i64, ibit)
  for (int_t i = 0; i < (int_t)cregs.size(); i++) {
    for (int_t j = 0; j < size; j++) {
      i64 = j >> 6;
      ibit = j & 63;
      if (((recv[i * n64 + i64] >> ibit) & 1) == 1)
        cregs[i].creg_memory()[j] = '1';
      else
        cregs[i].creg_memory()[j] = '0';
    }
  }
}
#endif

//-------------------------------------------------------------------------
} // end namespace CircuitExecutor
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
