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

#ifndef _aer_executor_hpp_
#define _aer_executor_hpp_

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

#include "simulators/simulators.hpp"
#include "simulators/state.hpp"

namespace AER {

namespace Executor {

using OpItr = std::vector<Operations::Op>::const_iterator;

// Timer type
using myclock_t = std::chrono::high_resolution_clock;

//-------------------------------------------------------------------------
// Executor base class
//-------------------------------------------------------------------------
template <class state_t>
class Base {
protected:
  // Simulation method
  Method method_;

  // Simulation device
  Device sim_device_ = Device::CPU;
  std::string sim_device_name_ = "CPU";

  // Simulation precision
  Precision sim_precision_ = Precision::Double;

  // Save counts as memory list
  bool save_creg_memory_ = false;

  // The maximum number of threads to use for various levels of parallelization
  int max_parallel_threads_;

  // Parameters for parallelization management in configuration
  int max_parallel_experiments_;
  int max_parallel_shots_;
  size_t max_memory_mb_;
  size_t max_gpu_memory_mb_;
  int num_gpus_; // max number of GPU per process

  // use explicit parallelization
  bool explicit_parallelization_;

  // Parameters for parallelization management for experiments
  int parallel_experiments_;
  int parallel_shots_;
  int parallel_state_update_;

  bool parallel_nested_ = false;

  // max number of states can be stored on memory for batched
  // multi-shots/experiments optimization
  int max_batched_states_;

  // max number of qubits in given circuits
  int max_qubits_;

  // results are stored independently in each process if true
  bool accept_distributed_results_ = true;

  // process information (MPI)
  int myrank_ = 0;
  int num_processes_ = 1;
  int num_process_per_experiment_ = 1;

  uint_t cache_block_qubit_ = 0;

  // multi-chunks are required to simulate circuits
  bool multi_chunk_required_ = false;

  // settings for cuStateVec
  bool cuStateVec_enable_ = false;

  //if circuit has statevector operations or not
  bool has_statevector_ops_;
public:
  Base();
  virtual ~Base() {}

  void run_circuit(Circuit &circ, const Noise::NoiseModel &noise,
                   const Config &config, const Method method,
                   const Device device, ExperimentResult &result);

protected:
  // Return a fusion transpilation pass configured for the current
  // method, circuit and config
  Transpile::Fusion transpile_fusion(const Operations::OpSet &opset,
                                     const Config &config) const;

  // Return cache blocking transpiler pass
  Transpile::CacheBlocking
  transpile_cache_blocking(const Circuit &circ, const Noise::NoiseModel &noise,
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
  // Return an estimate of the required memory for a circuit.
  virtual size_t required_memory_mb(const Circuit &circuit,
                                    const Noise::NoiseModel &noise) const {
    state_t tmp;
    return tmp.required_memory_mb(circuit.num_qubits, circuit.ops);
  }

  // get max shots stored on memory
  uint_t get_max_parallel_shots(const Circuit &circuit,
                                const Noise::NoiseModel &noise) const;

  bool multiple_chunk_required(const Circuit &circuit,
                               const Noise::NoiseModel &noise) const;

  bool multiple_shots_required(const Circuit &circuit,
                               const Noise::NoiseModel &noise) const;

  // Check if measure sampling optimization is valid for the input circuit
  // for the given method. This checks if operation types before
  // the first measurement in the circuit prevent sampling
  bool check_measure_sampling_opt(const Circuit &circ) const;

  bool has_statevector_ops(const Circuit &circ) const;

  virtual void set_config(const Config &config);
  virtual void set_parallelization(const Circuit &circ,
                                   const Noise::NoiseModel &noise);

  virtual void run_circuit_with_sampling(Circuit &circ, const Config &config,
                                         ExperimentResult &result);

  virtual void run_circuit_shots(Circuit &circ, const Noise::NoiseModel &noise,
                                 const Config &config, ExperimentResult &result,
                                 bool sample_noise) = 0;

  bool validate_state(const Circuit &circ, const Noise::NoiseModel &noise,
                      bool throw_except) const;

  void run_with_sampling(const Circuit &circ, state_t &state,
                         ExperimentResult &result, RngEngine &rng,
                         const uint_t shots);

  template <typename InputIterator>
  void measure_sampler(InputIterator first_meas, InputIterator last_meas,
                       uint_t shots, state_t &state, ExperimentResult &result,
                       std::vector<RngEngine> &rng,
                       bool shot_branching = false) const;

  // sampling measure
  virtual std::vector<reg_t> sample_measure(state_t &state, const reg_t &qubits,
                                            uint_t shots,
                                            std::vector<RngEngine> &rng) const {
    // this is for single rng, impement in sub-class for multi-shots case
    return state.sample_measure(qubits, shots, rng[0]);
  }
};

template <class state_t>
Base<state_t>::Base() {
  max_parallel_threads_ = 0;
  max_parallel_experiments_ = 1;
  max_parallel_shots_ = 0;
  max_batched_states_ = 1;

  parallel_experiments_ = 1;
  parallel_shots_ = 1;
  parallel_state_update_ = 1;
  parallel_nested_ = false;

  num_process_per_experiment_ = 1;

  num_gpus_ = 0;

  explicit_parallelization_ = false;

  max_memory_mb_ = get_system_memory_mb();
  max_gpu_memory_mb_ = get_gpu_memory_mb();

  has_statevector_ops_ = false;
}

template <class state_t>
void Base<state_t>::set_config(const Config &config) {
  // Load config for memory (creg list data)
  if (config.memory.has_value())
    save_creg_memory_ = config.memory.value();

#ifdef _OPENMP
  // Load OpenMP maximum thread settings
  if (config.max_parallel_threads.has_value())
    max_parallel_threads_ = config.max_parallel_threads.value();
  if (config.max_parallel_experiments.has_value())
    max_parallel_experiments_ = config.max_parallel_experiments.value();
  if (config.max_parallel_shots.has_value())
    max_parallel_shots_ = config.max_parallel_shots.value();
  // Limit max threads based on number of available OpenMP threads
  auto omp_threads = omp_get_max_threads();
  max_parallel_threads_ = (max_parallel_threads_ > 0)
                              ? std::min(max_parallel_threads_, omp_threads)
                              : std::max(1, omp_threads);
#else
  // No OpenMP so we disable parallelization
  max_parallel_threads_ = 1;
  max_parallel_shots_ = 1;
  max_parallel_experiments_ = 1;
  parallel_nested_ = false;
#endif

  // Load configurations for parallelization

  if (config.max_memory_mb.has_value())
    max_memory_mb_ = config.max_memory_mb.value();

  // for debugging
  if (config._parallel_experiments.has_value()) {
    parallel_experiments_ = config._parallel_experiments.value();
    explicit_parallelization_ = true;
  }

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
    parallel_experiments_ = std::max<int>({parallel_experiments_, 1});
    parallel_shots_ = std::max<int>({parallel_shots_, 1});
    parallel_state_update_ = std::max<int>({parallel_state_update_, 1});
  }

  if (config.accept_distributed_results.has_value())
    accept_distributed_results_ = config.accept_distributed_results.value();

  // enable multiple qregs if cache blocking is enabled
  cache_block_qubit_ = 0;
  if (config.blocking_qubits.has_value())
    cache_block_qubit_ = config.blocking_qubits.value();

  // cuStateVec configs
  cuStateVec_enable_ = false;
  if (config.cuStateVec_enable.has_value())
    cuStateVec_enable_ = config.cuStateVec_enable.value();

  // Override automatic simulation method with a fixed method
  sim_device_name_ = config.device;
  if (sim_device_name_ == "CPU") {
    sim_device_ = Device::CPU;
  } else if (sim_device_name_ == "Thrust") {
#ifndef AER_THRUST_CPU
    throw std::runtime_error(
        "Simulation device \"Thrust\" is not supported on this system");
#else
    sim_device_ = Device::ThrustCPU;
#endif
  } else if (sim_device_name_ == "GPU") {
#ifndef AER_THRUST_CUDA
    throw std::runtime_error(
        "Simulation device \"GPU\" is not supported on this system");
#else

#ifndef AER_CUSTATEVEC
    if (cuStateVec_enable_) {
      // Aer is not built for cuStateVec
      throw std::runtime_error("Simulation device \"GPU\" does not support "
                               "cuStateVec on this system");
    }
#endif
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      throw std::runtime_error("No CUDA device available!");
    }
    num_gpus_ = nDev;
    sim_device_ = Device::GPU;
#endif
  } else {
    throw std::runtime_error(std::string("Invalid simulation device (\"") +
                             sim_device_name_ + std::string("\")."));
  }

  if (method_ == Method::tensor_network) {
#if defined(AER_THRUST_CUDA) && defined(AER_CUTENSORNET)
    if (sim_device_ != Device::GPU)
#endif
      throw std::runtime_error(
          "Invalid combination of simulation method and device, "
          "\"tensor_network\" only supports \"device=GPU\"");
  }

  std::string precision = config.precision;
  if (precision == "double") {
    sim_precision_ = Precision::Double;
  } else if (precision == "single") {
    sim_precision_ = Precision::Single;
  } else {
    throw std::runtime_error(std::string("Invalid simulation precision (") +
                             precision + std::string(")."));
  }
}

template <class state_t>
size_t Base<state_t>::get_system_memory_mb() {
  size_t total_physical_memory = Utils::get_system_memory_mb();
#ifdef AER_MPI
  // get minimum memory size per process
  uint64_t locMem, minMem;
  locMem = total_physical_memory;
  MPI_Allreduce(&locMem, &minMem, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  total_physical_memory = minMem;
#endif

  return total_physical_memory;
}

template <class state_t>
size_t Base<state_t>::get_gpu_memory_mb() {
  size_t total_physical_memory = 0;
#ifdef AER_THRUST_CUDA
  int iDev, nDev, j;
  if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
    cudaGetLastError();
    nDev = 0;
  }
  for (iDev = 0; iDev < nDev; iDev++) {
    size_t freeMem, totalMem;
    cudaSetDevice(iDev);
    cudaMemGetInfo(&freeMem, &totalMem);
    total_physical_memory += totalMem;
  }
  num_gpus_ = nDev;
#endif

#ifdef AER_MPI
  // get minimum memory size per process
  uint64_t locMem, minMem;
  locMem = total_physical_memory;
  MPI_Allreduce(&locMem, &minMem, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  total_physical_memory = minMem;

  int t = num_gpus_;
  MPI_Allreduce(&t, &num_gpus_, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif

  return total_physical_memory >> 20;
}

template <class state_t>
bool Base<state_t>::multiple_chunk_required(
    const Circuit &circ, const Noise::NoiseModel &noise) const {
  if (circ.num_qubits < 3)
    return false;
  if (cache_block_qubit_ >= 2 && cache_block_qubit_ < circ.num_qubits)
    return true;

  if (num_process_per_experiment_ == 1 && sim_device_ == Device::GPU &&
      num_gpus_ > 0) {
    return (max_gpu_memory_mb_ / num_gpus_ < required_memory_mb(circ, noise));
  }
  if (num_process_per_experiment_ > 1) {
    size_t total_mem = max_memory_mb_;
    if (sim_device_ == Device::GPU)
      total_mem += max_gpu_memory_mb_;
    if (total_mem * num_process_per_experiment_ >
        required_memory_mb(circ, noise))
      return true;
  }

  return false;
}

template <class state_t>
bool Base<state_t>::multiple_shots_required(
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
uint_t
Base<state_t>::get_max_parallel_shots(const Circuit &circ,
                                      const Noise::NoiseModel &noise) const {
  uint_t mem = required_memory_mb(circ, noise);
  if (mem == 0)
    return circ.shots;

  if (sim_device_ == Device::GPU && num_gpus_ > 0) {
    return std::min(circ.shots, (max_gpu_memory_mb_ * 8 / 10 / mem));
  } else {
    return std::min(circ.shots, (max_memory_mb_ / mem));
  }
}

template <class state_t>
void Base<state_t>::set_parallelization(const Circuit &circ,
                                        const Noise::NoiseModel &noise) {
  if (explicit_parallelization_)
    return;

  // number of threads for parallel loop of experiments
  parallel_experiments_ = omp_get_num_threads();

  // Check for trivial parallelization conditions
  switch (method_) {
  case Method::statevector:
  case Method::stabilizer:
  case Method::unitary:
  case Method::matrix_product_state: {
    if (circ.shots == 1 || num_process_per_experiment_ > 1 ||
        (!noise.has_quantum_errors() && check_measure_sampling_opt(circ))) {
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
    int circ_memory_mb =
        required_memory_mb(circ, noise) / num_process_per_experiment_;
    size_t mem_size =
        (sim_device_ == Device::GPU) ? max_gpu_memory_mb_ : max_memory_mb_;
    if (mem_size < circ_memory_mb)
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
void Base<state_t>::run_circuit(Circuit &circ, const Noise::NoiseModel &noise,
                                const Config &config, const Method method,
                                const Device device, ExperimentResult &result) {
  // Start individual circuit timer
  auto timer_start = myclock_t::now(); // state circuit timer

  // Execute in try block so we can catch errors and return the error message
  // for individual circuit failures.
  try {
    // set configuration
    method_ = method;
    sim_device_ = device;
    set_config(config);
    set_parallelization(circ, noise);

    // Rng engine (this one is used to add noise on circuit)
    RngEngine rng;
    rng.set_seed(circ.seed);

    // Output data container
    result.set_config(config);
    result.metadata.add(method_names_.at(method), "method");
    if (method == Method::statevector || method == Method::density_matrix ||
        method == Method::unitary || method == Method::tensor_network) {
      result.metadata.add(sim_device_name_, "device");
    } else {
      result.metadata.add("CPU", "device");
    }

    // Circuit qubit metadata
    result.metadata.add(circ.num_qubits, "num_qubits");
    result.metadata.add(circ.num_memory, "num_clbits");
    result.metadata.add(circ.qubits(), "active_input_qubits");
    result.metadata.add(circ.qubit_map(), "input_qubit_map");
    result.metadata.add(circ.remapped_qubits, "remapped_qubits");

    // Add measure sampling to metadata
    // Note: this will set to `true` if sampling is enabled for the circuit
    result.metadata.add(false, "measure_sampling");
    result.metadata.add(false, "batched_shots_optimization");

    // Validate gateset and memory requirements, raise exception if they're
    // exceeded
    validate_state(circ, noise, true);

    has_statevector_ops_ = has_statevector_ops(circ);

    if (circ.num_qubits > 0) { // do nothing for query steps
      // Choose execution method based on noise and method
      Circuit opt_circ;
      bool noise_sampling = false;

      // Ideal circuit
      if (noise.is_ideal()) {
        opt_circ = circ;
        result.metadata.add("ideal", "noise");
      }
      // Readout error only
      else if (noise.has_quantum_errors() == false) {
        opt_circ = noise.sample_noise(circ, rng);
        result.metadata.add("readout", "noise");
      }
      // Superop noise sampling
      else if (method == Method::density_matrix || method == Method::superop ||
               (method == Method::tensor_network && !has_statevector_ops_)) {
        // Sample noise using SuperOp method
        opt_circ =
            noise.sample_noise(circ, rng, Noise::NoiseModel::Method::superop);
        result.metadata.add("superop", "noise");
      }
      // Kraus noise sampling
      else if (noise.opset().contains(Operations::OpType::kraus) ||
               noise.opset().contains(Operations::OpType::superop)) {
        opt_circ =
            noise.sample_noise(circ, rng, Noise::NoiseModel::Method::kraus);
        result.metadata.add("kraus", "noise");
      }
      // General circuit noise sampling
      else {
        noise_sampling = true;
        result.metadata.add("circuit", "noise");
      }

      if (noise_sampling) {
        run_circuit_shots(circ, noise, config, result, true);
      } else {
        // Run multishot simulation without noise sampling
        bool can_sample = opt_circ.can_sample;
        can_sample &= check_measure_sampling_opt(opt_circ);

        if (can_sample)
          run_circuit_with_sampling(opt_circ, config, result);
        else
          run_circuit_shots(opt_circ, noise, config, result, false);
      }
    }
    // Report success
    result.status = ExperimentResult::Status::completed;

    // Pass through circuit header and add metadata
    result.header = circ.header;
    result.shots = circ.shots;
    result.seed = circ.seed;
    result.metadata.add(parallel_shots_, "parallel_shots");
    result.metadata.add(parallel_state_update_, "parallel_state_update");
#ifdef AER_CUSTATEVEC
    result.metadata.add(cuStateVec_enable_, "cuStateVec_enable");
#endif

    // Add timer data
    auto timer_stop = myclock_t::now(); // stop timer
    double time_taken =
        std::chrono::duration<double>(timer_stop - timer_start).count();
    result.time_taken = time_taken;
  }
  // If an exception occurs during execution, catch it and pass it to the output
  catch (std::exception &e) {
    result.status = ExperimentResult::Status::error;
    result.message = e.what();
  }
}

template <class state_t>
void Base<state_t>::run_with_sampling(const Circuit &circ, state_t &state,
                                      ExperimentResult &result, RngEngine &rng,
                                      const uint_t shots) {
  auto &ops = circ.ops;
  auto first_meas = circ.first_measure_pos; // Position of first measurement op
  bool final_ops = (first_meas == ops.size());

  // allocate qubit register
  state.enable_cuStateVec(cuStateVec_enable_);
  state.allocate(circ.num_qubits, circ.num_qubits);
  state.set_num_global_qubits(circ.num_qubits);
  state.enable_density_matrix(!has_statevector_ops_);

  // Run circuit instructions before first measure
  state.initialize_qreg(circ.num_qubits);
  state.initialize_creg(circ.num_memory, circ.num_registers);

  state.apply_ops(ops.cbegin(), ops.cbegin() + first_meas, result, rng,
                  final_ops);

  // Get measurement operations and set of measured qubits
  std::vector<RngEngine> rngs;
  rngs.push_back(rng);
  measure_sampler(circ.ops.begin() + first_meas, circ.ops.end(), shots, state,
                  result, rngs);
}

template <class state_t>
void Base<state_t>::run_circuit_with_sampling(Circuit &circ,
                                              const Config &config,
                                              ExperimentResult &result) {
  state_t state;

  // Optimize circuit
  Noise::NoiseModel dummy_noise;

  auto fusion_pass = transpile_fusion(circ.opset(), config);
  fusion_pass.optimize_circuit(circ, dummy_noise, state.opset(), result);

  auto max_bits = get_max_matrix_qubits(circ);

  // Implement measure sampler
  if (parallel_shots_ <= 1) {
    // Set state config
    state.set_config(config);
    state.set_parallelization(parallel_state_update_);
    state.set_global_phase(circ.global_phase_angle);

    state.set_distribution(1);
    state.set_max_matrix_qubits(max_bits);

    RngEngine rng;
    rng.set_seed(circ.seed);
    run_with_sampling(circ, state, result, rng, circ.shots);
  } else {
    // Vector to store parallel thread output data
    std::vector<ExperimentResult> par_results(parallel_shots_);

#pragma omp parallel for num_threads(parallel_shots_)
    for (int i = 0; i < parallel_shots_; i++) {
      uint_t i_shot = circ.shots * i / parallel_shots_;
      uint_t shot_end = circ.shots * (i + 1) / parallel_shots_;
      uint_t this_shot = shot_end - i_shot;

      state_t shot_state;
      // Set state config
      shot_state.set_config(config);
      shot_state.set_parallelization(parallel_state_update_);
      shot_state.set_global_phase(circ.global_phase_angle);
      shot_state.enable_density_matrix(!has_statevector_ops_);

      shot_state.set_max_matrix_qubits(max_bits);

      RngEngine rng;
      rng.set_seed(circ.seed + i);

      run_with_sampling(circ, shot_state, par_results[i], rng, this_shot);

      shot_state.add_metadata(par_results[i]);
    }
    for (auto &res : par_results) {
      result.combine(std::move(res));
    }

    if (sim_device_ == Device::GPU) {
      if (parallel_shots_ >= num_gpus_)
        result.metadata.add(num_gpus_, "gpu_parallel_shots");
      else
        result.metadata.add(parallel_shots_, "gpu_parallel_shots");
    }
  }
  // Add measure sampling metadata
  result.metadata.add(true, "measure_sampling");

  state.add_metadata(result);
}

template <class state_t>
template <typename InputIterator>
void Base<state_t>::measure_sampler(InputIterator first_meas,
                                    InputIterator last_meas, uint_t shots,
                                    state_t &state, ExperimentResult &result,
                                    std::vector<RngEngine> &rng,
                                    bool shot_branching) const {
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
  if (shot_branching)
    all_samples = sample_measure(state, meas_qubits, shots, rng);
  else
    all_samples = state.sample_measure(meas_qubits, shots, rng[0]);
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
    if (shot_branching)
      creg = state.creg();
    else
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

    if (shot_branching) {
      // for shot-branching

      // process read out errors for memory and registers
      for (const Operations::Op &roerror : roerror_ops)
        creg.apply_roerror(roerror, rng[i]);

      std::string memory_hex = creg.memory_hex();
      result.data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
      if (save_creg_memory_)
        result.data.add_list(memory_hex, "memory");
    } else {
      // process read out errors for memory and registers
      for (const Operations::Op &roerror : roerror_ops)
        creg.apply_roerror(roerror, rng[0]);

      // Save count data
      result.save_count_data(creg, save_creg_memory_);
    }
  }
}

template <class state_t>
bool Base<state_t>::validate_state(const Circuit &circ,
                                   const Noise::NoiseModel &noise,
                                   bool throw_except) const {
  std::stringstream error_msg;
  std::string circ_name;
  state_t state;

  JSON::get_value(circ_name, "name", circ.header);

  // Check if a circuit is valid for state ops
  bool circ_valid = state.opset().contains(circ.opset());
  if (throw_except && !circ_valid) {
    error_msg << "Circuit " << circ_name << " contains invalid instructions ";
    error_msg << state.opset().difference(circ.opset());
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
Base<state_t>::transpile_fusion(const Operations::OpSet &opset,
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
Transpile::CacheBlocking
Base<state_t>::transpile_cache_blocking(const Circuit &circ,
                                        const Noise::NoiseModel &noise,
                                        const Config &config) const {
  Transpile::CacheBlocking cache_block_pass;

  const bool is_matrix =
      (method_ == Method::density_matrix || method_ == Method::unitary);
  const auto complex_size = (sim_precision_ == Precision::Single)
                                ? sizeof(std::complex<float>)
                                : sizeof(std::complex<double>);

  cache_block_pass.set_num_processes(num_process_per_experiment_);
  cache_block_pass.set_config(config);

  if (!cache_block_pass.enabled()) {
    // if blocking is not set by config, automatically set if required
    if (multiple_chunk_required(circ, noise)) {
      int nplace = num_process_per_experiment_;
      if (sim_device_ == Device::GPU && num_gpus_ > 0)
        nplace *= num_gpus_;
      cache_block_pass.set_blocking(circ.num_qubits, get_min_memory_mb() << 20,
                                    nplace, complex_size, is_matrix);
    }
  }
  return cache_block_pass;
}

template <class state_t>
bool Base<state_t>::check_measure_sampling_opt(const Circuit &circ) const {
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
int_t Base<state_t>::get_matrix_bits(const Operations::Op &op) const {
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
int_t Base<state_t>::get_max_matrix_qubits(const Circuit &circ) const {
  int_t max_bits = 0;
  int_t i;

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
bool Base<state_t>::has_statevector_ops(const Circuit &circ) const {
  return circ.opset().contains(Operations::OpType::save_statevec) ||
         circ.opset().contains(Operations::OpType::save_statevec_dict) ||
         circ.opset().contains(Operations::OpType::save_amps);
}

//-------------------------------------------------------------------------
} // end namespace Executor
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
