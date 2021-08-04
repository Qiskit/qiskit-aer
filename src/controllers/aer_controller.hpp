/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_controller_hpp_
#define _aer_controller_hpp_

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#elif defined(_WIN64) || defined(_WIN32)
// This is needed because windows.h redefine min()/max() so interferes with
// std::min/max
#define NOMINMAX
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

#include "framework/creg.hpp"
#include "framework/qobj.hpp"
#include "framework/results/experiment_result.hpp"
#include "framework/results/result.hpp"
#include "framework/rng.hpp"
#include "noise/noise_model.hpp"

#include "transpile/basic_opts.hpp"
#include "transpile/cacheblocking.hpp"
#include "transpile/delay_measure.hpp"
#include "transpile/fusion.hpp"
#include "transpile/truncate_qubits.hpp"

#include "simulators/density_matrix/densitymatrix_state.hpp"
#include "simulators/density_matrix/densitymatrix_state_chunk.hpp"
#include "simulators/extended_stabilizer/extended_stabilizer_state.hpp"
#include "simulators/matrix_product_state/matrix_product_state.hpp"
#include "simulators/stabilizer/stabilizer_state.hpp"
#include "simulators/statevector/qubitvector.hpp"
#include "simulators/statevector/statevector_state.hpp"
#include "simulators/statevector/statevector_state_chunk.hpp"
#include "simulators/superoperator/superoperator_state.hpp"
#include "simulators/unitary/unitary_state.hpp"
#include "simulators/unitary/unitary_state_chunk.hpp"

namespace AER {

//=========================================================================
// AER::Controller class
//=========================================================================

// This is the top level controller for the Qiskit-Aer simulator
// It manages execution of all the circuits in a QOBJ, parallelization,
// noise sampling from a noise model, and circuit optimizations.

class Controller {
public:
  Controller() { clear_parallelization(); }

  //-----------------------------------------------------------------------
  // Execute qobj
  //-----------------------------------------------------------------------

  // Load a QOBJ from a JSON file and execute on the State type
  // class.
  template <typename inputdata_t>
  Result execute(const inputdata_t &qobj);

  Result execute(std::vector<Circuit> &circuits,
                 const Noise::NoiseModel &noise_model, const json_t &config);

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  void set_config(const json_t &config);

  // Clear the current config
  void clear_config();

protected:
  //-----------------------------------------------------------------------
  // Simulation types
  //-----------------------------------------------------------------------

  // Simulation methods for the Qasm Controller
  enum class Method {
    automatic,
    statevector,
    density_matrix,
    matrix_product_state,
    stabilizer,
    extended_stabilizer,
    unitary,
    superop
  };

  enum class Device { CPU, GPU, ThrustCPU };

  // Simulation precision
  enum class Precision { Double, Single };

  const std::unordered_map<Method, std::string> method_names_ = {
    {Method::automatic, "automatic"},
    {Method::statevector, "statevector"},
    {Method::density_matrix, "density_matrix"},
    {Method::matrix_product_state, "matrix_product_state"},
    {Method::stabilizer, "stabilizer"},
    {Method::extended_stabilizer, "extended_stabilizer"},
    {Method::unitary, "unitary"},
    {Method::superop, "superop"}
  };

  //-----------------------------------------------------------------------
  // Config
  //-----------------------------------------------------------------------

  // Timer type
  using myclock_t = std::chrono::high_resolution_clock;

  // Transpile pass override flags
  bool truncate_qubits_ = true;

  // Validation threshold for validating states and operators
  double validation_threshold_ = 1e-8;

  // Save counts as memory list
  bool save_creg_memory_ = false;

  // Simulation method
  Method sim_method_ = Method::automatic;

  // Simulation device
  Device sim_device_ = Device::CPU;
  std::string sim_device_name_ = "CPU";

  // Simulation precision
  Precision sim_precision_ = Precision::Double;

  // Controller-level parameter for CH method
  bool extended_stabilizer_measure_sampling_ = false;

  //-----------------------------------------------------------------------
  // Circuit Execution
  //-----------------------------------------------------------------------

  // Parallel execution of a circuit
  // This function manages parallel shot configuration and internally calls
  // the `run_circuit` method for each shot thread
  void execute_circuit(Circuit &circ, Noise::NoiseModel &noise,
                       const json_t &config, ExperimentResult &result);

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  void run_circuit(const Circuit &circ, const Noise::NoiseModel &noise,
                   const json_t &config, uint_t shots, uint_t rng_seed,
                   ExperimentResult &result) const;

  //----------------------------------------------------------------
  // Run circuit helpers
  //----------------------------------------------------------------

  // Execute n-shots of a circuit on the input state
  template <class State_t>
  void run_circuit_helper(const Circuit &circ, const Noise::NoiseModel &noise,
                          const json_t &config, uint_t shots, uint_t rng_seed,
                          const Method method, bool cache_block,
                          ExperimentResult &result) const;

  // Execute a single shot a of circuit by initializing the state vector,
  // running all ops in circ, and updating data with
  // simulation output.
  template <class State_t>
  void run_single_shot(const Circuit &circ, State_t &state,
                       ExperimentResult &result, RngEngine &rng) const;

  // Execute multiple shots a of circuit by initializing the state vector,
  // running all ops in circ, and updating data with
  // simulation output. Will use measurement sampling if possible
  template <class State_t>
  void run_circuit_without_sampled_noise(Circuit &circ,
                                         const json_t &config,
                                         uint_t shots,
                                         State_t &state,
                                         const Method method,
                                         bool cache_blocking,
                                         ExperimentResult &result,
                                         RngEngine &rng) const;

  template <class State_t>
  void run_circuit_with_sampled_noise(const Circuit &circ,
                                      const Noise::NoiseModel &noise,
                                      const json_t &config,
                                      uint_t shots,
                                      State_t &state,
                                      const Method method,
                                      bool cache_blocking,
                                      ExperimentResult &result,
                                      RngEngine &rng) const;

  //----------------------------------------------------------------
  // Measurement
  //----------------------------------------------------------------

  // Sample measurement outcomes for the input measure ops from the
  // current state of the input State_t
  template <class State_t>
  void measure_sampler(const std::vector<Operations::Op> &meas_ops,
                       uint_t shots, State_t &state, ExperimentResult &result,
                       RngEngine &rng) const;

  // Check if measure sampling optimization is valid for the input circuit
  // for the given method. This checks if operation types before
  // the first measurement in the circuit prevent sampling
  bool check_measure_sampling_opt(const Circuit &circ,
                                  const Method method) const;

  // Save count data
  void save_count_data(ExperimentResult &result,
                       const ClassicalRegister &creg) const;

  //-------------------------------------------------------------------------
  // State validation
  //-------------------------------------------------------------------------

  // Return True if a given circuit (and internal noise model) are valid for
  // execution on the given state. Otherwise return false.
  // If throw_except is true an exception will be thrown on the return false
  // case listing the invalid instructions in the circuit or noise model.
  template <class state_t>
  static bool validate_state(const state_t &state, const Circuit &circ,
                             const Noise::NoiseModel &noise,
                             bool throw_except = false);

  // Return True if a given circuit are valid for execution on the given state.
  // Otherwise return false.
  // If throw_except is true an exception will be thrown directly.
  template <class state_t>
  bool validate_memory_requirements(const state_t &state, const Circuit &circ,
                                    bool throw_except = false) const;

  // Return an estimate of the required memory for a circuit.
  size_t required_memory_mb(const Circuit &circuit,
                            const Noise::NoiseModel &noise) const;

  //----------------------------------------------------------------
  // Utility functions
  //----------------------------------------------------------------

  // Return the simulation method to use for the input circuit
  // If a custom method is specified in the config this will be
  // used. If the default automatic method is set this will choose
  // the appropriate method based on the input circuit.
  Method simulation_method(const Circuit &circ, const Noise::NoiseModel &noise,
                           bool validate = false) const;

  // Return a fusion transpilation pass configured for the current
  // method, circuit and config
  Transpile::Fusion transpile_fusion(Method method,
                                     const Operations::OpSet &opset,
                                     const json_t &config) const;

  // Return cache blocking transpiler pass
  Transpile::CacheBlocking
  transpile_cache_blocking(Controller::Method method,
                           const Circuit &circ,
                           const Noise::NoiseModel &noise,
                           const json_t &config) const;

  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Set OpenMP thread settings to default values
  void clear_parallelization();

  // Set parallelization for experiments
  void
  set_parallelization_experiments(const std::vector<Circuit> &circuits,
                                  const std::vector<Noise::NoiseModel> &noise);

  // Set circuit parallelization
  void set_parallelization_circuit(const Circuit &circ,
                                   const Noise::NoiseModel &noise);

  void set_parallelization_circuit_method(const Circuit &circ,
                                          const Noise::NoiseModel &noise);


  bool multiple_chunk_required(const Circuit &circuit,
                               const Noise::NoiseModel &noise) const;

  void save_exception_to_results(Result &result, const std::exception &e);

  // Get system memory size
  size_t get_system_memory_mb();
  size_t get_gpu_memory_mb();

  size_t get_min_memory_mb() const {
    if (sim_device_ == Device::GPU && num_gpus_ > 0) {
      return max_gpu_memory_mb_ / num_gpus_; // return per GPU memory size
    }
    return max_memory_mb_;
  }

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

  // max number of qubits in given circuits
  int max_qubits_;

  // results are stored independently in each process if true
  bool accept_distributed_results_ = true;

  // process information (MPI)
  int myrank_ = 0;
  int num_processes_ = 1;
  int num_process_per_experiment_ = 1;

  uint_t cache_block_qubit_ = 0;
};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Config settings
//-------------------------------------------------------------------------

void Controller::set_config(const json_t &config) {

  // Load validation threshold
  JSON::get_value(validation_threshold_, "validation_threshold", config);

  // Load config for memory (creg list data)
  JSON::get_value(save_creg_memory_, "memory", config);

#ifdef _OPENMP
  // Load OpenMP maximum thread settings
  if (JSON::check_key("max_parallel_threads", config))
    JSON::get_value(max_parallel_threads_, "max_parallel_threads", config);
  if (JSON::check_key("max_parallel_experiments", config))
    JSON::get_value(max_parallel_experiments_, "max_parallel_experiments",
                    config);
  if (JSON::check_key("max_parallel_shots", config))
    JSON::get_value(max_parallel_shots_, "max_parallel_shots", config);
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

  if (JSON::check_key("max_memory_mb", config)) {
    JSON::get_value(max_memory_mb_, "max_memory_mb", config);
  }

  // for debugging
  if (JSON::check_key("_parallel_experiments", config)) {
    JSON::get_value(parallel_experiments_, "_parallel_experiments", config);
    explicit_parallelization_ = true;
  }

  // for debugging
  if (JSON::check_key("_parallel_shots", config)) {
    JSON::get_value(parallel_shots_, "_parallel_shots", config);
    explicit_parallelization_ = true;
  }

  // for debugging
  if (JSON::check_key("_parallel_state_update", config)) {
    JSON::get_value(parallel_state_update_, "_parallel_state_update", config);
    explicit_parallelization_ = true;
  }

  if (explicit_parallelization_) {
    parallel_experiments_ = std::max<int>({parallel_experiments_, 1});
    parallel_shots_ = std::max<int>({parallel_shots_, 1});
    parallel_state_update_ = std::max<int>({parallel_state_update_, 1});
  }

  if (JSON::check_key("accept_distributed_results", config)) {
    JSON::get_value(accept_distributed_results_, "accept_distributed_results",
                    config);
  }

  // enable multiple qregs if cache blocking is enabled
  cache_block_qubit_ = 0;
  if (JSON::check_key("blocking_qubits", config)) {
    JSON::get_value(cache_block_qubit_, "blocking_qubits", config);
  }

  // Override automatic simulation method with a fixed method
  std::string method;
  if (JSON::get_value(method, "method", config)) {
    if (method == "statevector") {
      sim_method_ = Method::statevector;
    } else if (method == "density_matrix") {
      sim_method_ = Method::density_matrix;
    } else if (method == "stabilizer") {
      sim_method_ = Method::stabilizer;
    } else if (method == "extended_stabilizer") {
      sim_method_ = Method::extended_stabilizer;
    } else if (method == "matrix_product_state") {
      sim_method_ = Method::matrix_product_state;
    } else if (method == "unitary") {
      sim_method_ = Method::unitary;
    } else if (method == "superop") {
      sim_method_ = Method::superop;
    } else if (method != "automatic") {
      throw std::runtime_error(std::string("Invalid simulation method (") +
                               method + std::string(")."));
    }
  }

  // Override automatic simulation method with a fixed method
  if (JSON::get_value(sim_device_name_, "device", config)) {
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
        int nDev;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error("No CUDA device available!");
        }

        sim_device_ = Device::GPU;
#endif
      }
    else {
      throw std::runtime_error(std::string("Invalid simulation device (\"") +
                               sim_device_name_ + std::string("\")."));
    }
  }

  std::string precision;
  if (JSON::get_value(precision, "precision", config)) {
    if (precision == "double") {
      sim_precision_ = Precision::Double;
    } else if (precision == "single") {
      sim_precision_ = Precision::Single;
    } else {
      throw std::runtime_error(std::string("Invalid simulation precision (") +
                               precision + std::string(")."));
    }
  }
}

void Controller::clear_config() {
  clear_parallelization();
  validation_threshold_ = 1e-8;
  sim_method_ = Method::automatic;
  sim_device_ = Device::CPU;
  sim_precision_ = Precision::Double;
}

void Controller::clear_parallelization() {
  max_parallel_threads_ = 0;
  max_parallel_experiments_ = 1;
  max_parallel_shots_ = 0;

  parallel_experiments_ = 1;
  parallel_shots_ = 1;
  parallel_state_update_ = 1;
  parallel_nested_ = false;

  num_process_per_experiment_ = 1;

  num_gpus_ = 0;

  explicit_parallelization_ = false;
  max_memory_mb_ = get_system_memory_mb();
  max_gpu_memory_mb_ = get_gpu_memory_mb();
}

void Controller::set_parallelization_experiments(
    const std::vector<Circuit> &circuits,
    const std::vector<Noise::NoiseModel> &noise) 
{
  if(circuits.size() == 1){
    parallel_experiments_ = 1;
    return;
  }

  // Use a local variable to not override stored maximum based
  // on currently executed circuits
  const auto max_experiments =
      (max_parallel_experiments_ > 1)
          ? std::min({max_parallel_experiments_, max_parallel_threads_})
          : max_parallel_threads_;

  if (max_experiments == 1) {
    // No parallel experiment execution
    parallel_experiments_ = 1;
    return;
  }

  // If memory allows, execute experiments in parallel
  std::vector<size_t> required_memory_mb_list(circuits.size());
  for (size_t j = 0; j < circuits.size(); j++) {
    required_memory_mb_list[j] = required_memory_mb(circuits[j], noise[j]);
  }
  std::sort(required_memory_mb_list.begin(), required_memory_mb_list.end(),
            std::greater<>());
  size_t total_memory = 0;
  parallel_experiments_ = 0;
  for (size_t required_memory_mb : required_memory_mb_list) {
    total_memory += required_memory_mb;
    if (total_memory > max_memory_mb_)
      break;
    ++parallel_experiments_;
  }

  if (parallel_experiments_ <= 0)
    throw std::runtime_error(
        "a circuit requires more memory than max_memory_mb.");
  parallel_experiments_ =
      std::min<int>({parallel_experiments_, max_experiments,
                     max_parallel_threads_, static_cast<int>(circuits.size())});
}

void Controller::set_parallelization_circuit(const Circuit &circ,
                                             const Noise::NoiseModel &noise) 
{
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
    size_t mem_size = (sim_device_ == Device::GPU) ? max_memory_mb_ + max_gpu_memory_mb_ : max_memory_mb_;
    if (mem_size < circ_memory_mb)
      throw std::runtime_error(
          "a circuit requires more memory than max_memory_mb.");
    // If circ memory is 0, set it to 1 so that we don't divide by zero
    circ_memory_mb = std::max<int>({1, circ_memory_mb});

    int shots = circ.shots;
    parallel_shots_ = std::min<int>(
        {static_cast<int>(max_memory_mb_ / circ_memory_mb), max_shots, shots});
  }
  parallel_state_update_ =
      (parallel_shots_ > 1)
          ? std::max<int>({1, max_parallel_threads_ / parallel_shots_})
          : std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
}

bool Controller::multiple_chunk_required(const Circuit &circ,
                                         const Noise::NoiseModel &noise) const 
{
  if (circ.num_qubits < 3)
    return false;
  if (cache_block_qubit_ >= 2 && cache_block_qubit_ < circ.num_qubits)
    return true;

  if(num_process_per_experiment_ == 1 && sim_device_ == Device::GPU && num_gpus_ > 0){
    return (max_gpu_memory_mb_ / num_gpus_ < required_memory_mb(circ, noise));
  }
  if(num_process_per_experiment_ > 1){
    size_t total_mem = max_memory_mb_;
    if(sim_device_ == Device::GPU)
      total_mem += max_gpu_memory_mb_;
    if(total_mem*num_process_per_experiment_ > required_memory_mb(circ, noise))
      return true;
  }

  return false;
}

size_t Controller::get_system_memory_mb() 
{
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

size_t Controller::get_gpu_memory_mb() {
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

//-------------------------------------------------------------------------
// State validation
//-------------------------------------------------------------------------

template <class state_t>
bool Controller::validate_state(const state_t &state, const Circuit &circ,
                                const Noise::NoiseModel &noise,
                                bool throw_except) {
  // First check if a noise model is valid for a given state
  bool noise_valid = noise.is_ideal() || state.opset().contains(noise.opset());
  bool circ_valid = state.opset().contains(circ.opset());
  if (noise_valid && circ_valid) {
    return true;
  }

  // If we didn't return true then either noise model or circ has
  // invalid instructions.
  if (throw_except == false)
    return false;

  // If we are throwing an exception we include information
  // about the invalid operations
  std::stringstream msg;
  if (!noise_valid) {
    msg << "Noise model contains invalid instructions ";
    msg << state.opset().difference(noise.opset());
    msg << " for \"" << state.name() << "\" method";
  }
  if (!circ_valid) {
    msg << "Circuit contains invalid instructions ";
    msg << state.opset().difference(circ.opset());
    msg << " for \"" << state.name() << "\" method";
  }
  throw std::runtime_error(msg.str());
}

template <class state_t>
bool Controller::validate_memory_requirements(const state_t &state,
                                              const Circuit &circ,
                                              bool throw_except) const {
  if (max_memory_mb_ == 0)
    return true;

  size_t required_mb = state.required_memory_mb(circ.num_qubits, circ.ops) /
                       num_process_per_experiment_;
                                                
  size_t mem_size = (sim_device_ == Device::GPU) ? max_memory_mb_ + max_gpu_memory_mb_ : max_memory_mb_;
  if (mem_size < required_mb) {
    if (throw_except) {
      std::string name = "";
      JSON::get_value(name, "name", circ.header);
      throw std::runtime_error("Insufficient memory to run circuit \"" + name +
                               "\" using the " + state.name() + " simulator.");
    }
    return false;
  }
  return true;
}

void Controller::save_exception_to_results(Result &result,
                                           const std::exception &e) {
  result.status = Result::Status::error;
  result.message = e.what();
  for (auto &res : result.results) {
    res.status = ExperimentResult::Status::error;
    res.message = e.what();
  }
}

Transpile::CacheBlocking
Controller::transpile_cache_blocking(Controller::Method method, const Circuit &circ,
                                     const Noise::NoiseModel &noise,
                                     const json_t &config) const {
  Transpile::CacheBlocking cache_block_pass;

  const bool is_matrix = (method == Method::density_matrix
                          || method == Method::unitary);
  const auto complex_size = (sim_precision_ == Precision::Single)
                              ? sizeof(std::complex<float>)
                              : sizeof(std::complex<double>);

  cache_block_pass.set_config(config);
  if (!cache_block_pass.enabled()) {
    // if blocking is not set by config, automatically set if required
    if (multiple_chunk_required(circ, noise)) {
      int nplace = num_process_per_experiment_;
      if(sim_device_ == Device::GPU && num_gpus_ > 0)
        nplace *= num_gpus_;
      cache_block_pass.set_blocking(circ.num_qubits, get_min_memory_mb() << 20,
                                    nplace, complex_size, is_matrix);
    }
  }
  return cache_block_pass;
}

//-------------------------------------------------------------------------
// Qobj execution
//-------------------------------------------------------------------------
template <typename inputdata_t>
Result Controller::execute(const inputdata_t &input_qobj) {
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
#endif

  // Load QOBJ in a try block so we can catch parsing errors and still return
  // a valid JSON output containing the error message.
  try {
    // Start QOBJ timer
    auto timer_start = myclock_t::now();

    Qobj qobj(input_qobj);
    Noise::NoiseModel noise_model;
    json_t config;
    // Check for config
    if (Parser<inputdata_t>::get_value(config, "config", input_qobj)) {
      // Set config
      set_config(config);
      // Load noise model
      Parser<json_t>::get_value(noise_model, "noise_model", config);
    }
    auto result = execute(qobj.circuits, noise_model, config);
    // Get QOBJ id and pass through header to result
    result.qobj_id = qobj.id;
    if (!qobj.header.empty()) {
      result.header = qobj.header;
    }
    // Stop the timer and add total timing data including qobj parsing
    auto timer_stop = myclock_t::now();
    auto time_taken =
        std::chrono::duration<double>(timer_stop - timer_start).count();
    result.metadata.add(time_taken, "time_taken");
    return result;
  } catch (std::exception &e) {
    // qobj was invalid, return valid output containing error message
    Result result;

    result.status = Result::Status::error;
    result.message = std::string("Failed to load qobj: ") + e.what();
    return result;
  }
}

//-------------------------------------------------------------------------
// Experiment execution
//-------------------------------------------------------------------------

Result Controller::execute(std::vector<Circuit> &circuits,
                           const Noise::NoiseModel &noise_model,
                           const json_t &config) {
  // Start QOBJ timer
  auto timer_start = myclock_t::now();

  // Initialize Result object for the given number of experiments
  Result result(circuits.size());
  // Make a copy of the noise model for each circuit execution
  // so that it can be modified if required
  std::vector<Noise::NoiseModel> circ_noise_models(circuits.size(),
                                                   noise_model);

  // Execute each circuit in a try block
  try {
    // truncate circuits before experiment settings (to get correct
    // required_memory_mb value)
    if (truncate_qubits_) {
      for (size_t j = 0; j < circuits.size(); j++) {
        // Truncate unused qubits from circuit and noise model
        Transpile::TruncateQubits truncate_pass;
        truncate_pass.set_config(config);
        truncate_pass.optimize_circuit(circuits[j], circ_noise_models[j],
                                       circuits[j].opset(), result.results[j]);
      }
    }

    // get max qubits for this process (to allocate qubit register at once)
    max_qubits_ = 0;
    for (size_t j = 0; j < circuits.size(); j++) {
      if (circuits[j].num_qubits > max_qubits_) {
        max_qubits_ = circuits[j].num_qubits;
      }
    }
    num_process_per_experiment_ = num_processes_;

    if (!explicit_parallelization_) {
      // set parallelization for experiments
      try {
        // catch exception raised by required_memory_mb because of invalid
        // simulation method
        set_parallelization_experiments(circuits, circ_noise_models);
      } catch (std::exception &e) {
        save_exception_to_results(result, e);
      }
    }

#ifdef _OPENMP
    result.metadata.add(true, "omp_enabled");
#else
    result.metadata.add(false, "omp_enabled");
#endif
    result.metadata.add(parallel_experiments_, "parallel_experiments");
    result.metadata.add(max_memory_mb_, "max_memory_mb");
    result.metadata.add(max_gpu_memory_mb_, "max_gpu_memory_mb");

    // store rank and number of processes, if no distribution rank=0 procs=1 is
    // set
    result.metadata.add(num_processes_, "num_mpi_processes");
    result.metadata.add(myrank_, "mpi_rank");

#ifdef _OPENMP
    // Check if circuit parallelism is nested with one of the others
    if (parallel_experiments_ > 1 &&
        parallel_experiments_ < max_parallel_threads_) {
      // Nested parallel experiments
      parallel_nested_ = true;
#ifdef _WIN32
      omp_set_nested(1);
#else
      omp_set_max_active_levels(3);
#endif
      result.metadata.add(parallel_nested_, "omp_nested");
    } else {
      parallel_nested_ = false;
#ifdef _WIN32
      omp_set_nested(0);
#else
      omp_set_max_active_levels(1);
#endif
    }
#endif
    // then- and else-blocks have intentionally duplication.
    // Nested omp has significant overheads even though a guard condition
    // exists.
    const int NUM_RESULTS = result.results.size();
    if (parallel_experiments_ > 1) {
#pragma omp parallel for num_threads(parallel_experiments_)
      for (int j = 0; j < result.results.size(); ++j) {
        execute_circuit(circuits[j], circ_noise_models[j],
                        config, result.results[j]);
      }
    } else {
      for (int j = 0; j < result.results.size(); ++j) {
        execute_circuit(circuits[j], circ_noise_models[j],
                        config, result.results[j]);
      }
    }

    // Check each experiment result for completed status.
    // If only some experiments completed return partial completed status.

    bool all_failed = true;
    result.status = Result::Status::completed;
    for (int i = 0; i < NUM_RESULTS; ++i) {
      auto &experiment = result.results[i];
      if (experiment.status == ExperimentResult::Status::completed) {
        all_failed = false;
      } else {
        result.status = Result::Status::partial_completed;
        result.message += std::string(" [Experiment ") + std::to_string(i) +
                          std::string("] ") + experiment.message;
      }
    }
    if (all_failed) {
      result.status = Result::Status::error;
    }

    // Stop the timer and add total timing data
    auto timer_stop = myclock_t::now();
    auto time_taken =
        std::chrono::duration<double>(timer_stop - timer_start).count();
    result.metadata.add(time_taken, "time_taken");
  }
  // If execution failed return valid output reporting error
  catch (std::exception &e) {
    result.status = Result::Status::error;
    result.message = e.what();
  }
  return result;
}

void Controller::execute_circuit(Circuit &circ, Noise::NoiseModel &noise,
                                 const json_t &config,
                                 ExperimentResult &result) {
  // Start individual circuit timer
  auto timer_start = myclock_t::now(); // state circuit timer

  // Initialize circuit json return
  result.legacy_data.set_config(config);

  // Execute in try block so we can catch errors and return the error message
  // for individual circuit failures.
  try {
    // Remove barriers from circuit
    Transpile::ReduceBarrier barrier_pass;
    barrier_pass.optimize_circuit(circ, noise, circ.opset(), result);

    // Truncate unused qubits from circuit and noise model
    if (truncate_qubits_) {
      Transpile::TruncateQubits truncate_pass;
      truncate_pass.set_config(config);
      truncate_pass.optimize_circuit(circ, noise, circ.opset(), result);
    }

    // set parallelization for this circuit
    if (!explicit_parallelization_) {
      set_parallelization_circuit_method(circ, noise);
    }

    int shots = circ.shots;

    // Single shot thread execution
    if (parallel_shots_ <= 1) {
      run_circuit(circ, noise, config, shots, circ.seed, result);
      // Parallel shot thread execution
    } else {
      // Calculate shots per thread
      std::vector<unsigned int> subshots;
      for (int j = 0; j < parallel_shots_; ++j) {
        subshots.push_back(shots / parallel_shots_);
      }
      // If shots is not perfectly divisible by threads, assign the remainder
      for (int j = 0; j < int(shots % parallel_shots_); ++j) {
        subshots[j] += 1;
      }

      // Vector to store parallel thread output data
      std::vector<ExperimentResult> par_results(parallel_shots_);
      std::vector<std::string> error_msgs(parallel_shots_);

#ifdef _OPENMP
      if (!parallel_nested_) {
        if (parallel_shots_ > 1 && parallel_state_update_ > 1) {
// Nested parallel shots + state update
#ifdef _WIN32
          omp_set_nested(1);
#else
          omp_set_max_active_levels(2);
#endif
          result.metadata.add(true, "omp_nested");
        } else {
#ifdef _WIN32
          omp_set_nested(0);
#else
          omp_set_max_active_levels(1);
#endif
        }
      }
#endif

#pragma omp parallel for if (parallel_shots_ > 1) num_threads(parallel_shots_)
      for (int i = 0; i < parallel_shots_; i++) {
        try {
          run_circuit(circ, noise, config, subshots[i], circ.seed + i,
                      par_results[i]);
        } catch (std::runtime_error &error) {
          error_msgs[i] = error.what();
        }
      }

      for (std::string error_msg : error_msgs)
        if (error_msg != "")
          throw std::runtime_error(error_msg);

      // Accumulate results across shots
      // Use move semantics to avoid copying data
      for (auto &res : par_results) {
        result.combine(std::move(res));
      }
    }
    // Report success
    result.status = ExperimentResult::Status::completed;

    // Pass through circuit header and add metadata
    result.header = circ.header;
    result.shots = shots;
    result.seed = circ.seed;
    result.metadata.add(parallel_shots_, "parallel_shots");
    result.metadata.add(parallel_state_update_, "parallel_state_update");
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

void Controller::save_count_data(ExperimentResult &result,
                                 const ClassicalRegister &creg) const {
  if (creg.memory_size() > 0) {
    std::string memory_hex = creg.memory_hex();
    result.data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
    if (save_creg_memory_) {
      result.data.add_list(std::move(memory_hex), "memory");
    }
  }
}

//-------------------------------------------------------------------------
// Base class override
//-------------------------------------------------------------------------

void Controller::run_circuit(const Circuit &circ,
                             const Noise::NoiseModel &noise,
                             const json_t &config, uint_t shots,
                             uint_t rng_seed, ExperimentResult &result) const {
  // Validate circuit for simulation method
  switch (simulation_method(circ, noise, true)) {
  case Method::statevector: {
    if (sim_device_ == Device::CPU) {
      if (multiple_chunk_required(circ, noise)) {
        // Chunk based simualtion
        if (sim_precision_ == Precision::Double) {
          // Double-precision Statevector simulation
          return run_circuit_helper<
              StatevectorChunk::State<QV::QubitVector<double>>>(
              circ, noise, config, shots, rng_seed, Method::statevector,
              true, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<
              StatevectorChunk::State<QV::QubitVector<float>>>(
              circ, noise, config, shots, rng_seed, Method::statevector,
              true, result);
        }
      } else {
        // Non-chunk based simulation
        if (sim_precision_ == Precision::Double) {
          // Double-precision Statevector simulation
          return run_circuit_helper<
              Statevector::State<QV::QubitVector<double>>>(
              circ, noise, config, shots, rng_seed, Method::statevector,
              false, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<Statevector::State<QV::QubitVector<float>>>(
              circ, noise, config, shots, rng_seed, Method::statevector,
              false, result);
        }
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      if (multiple_chunk_required(circ, noise) ||
          (parallel_shots_ > 1 || parallel_experiments_ > 1)) {
        // Chunk based simulation
        if (sim_precision_ == Precision::Double) {
          // Double-precision Statevector simulation
          return run_circuit_helper<
              StatevectorChunk::State<QV::QubitVectorThrust<double>>>(
              circ, noise, config, shots, rng_seed, Method::statevector,
              true, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<
              StatevectorChunk::State<QV::QubitVectorThrust<float>>>(
              circ, noise, config, shots, rng_seed, Method::statevector,
              true, result);
        }
      } else {
        // Non-chunk based simulation
        if (sim_precision_ == Precision::Double) {
          // Double-precision Statevector simulation
          return run_circuit_helper<
              Statevector::State<QV::QubitVectorThrust<double>>>(
              circ, noise, config, shots, rng_seed, Method::statevector,
              false, result);
        } else {
          // Single-precision Statevector simulation
          return run_circuit_helper<
              Statevector::State<QV::QubitVectorThrust<float>>>(
              circ, noise, config, shots, rng_seed, Method::statevector,
              false, result);
        }
      }
#endif
    }
  }
  case Method::density_matrix: {
    if (sim_device_ == Device::CPU) {
      if (multiple_chunk_required(circ, noise)) {
        if (sim_precision_ == Precision::Double) {
          // Double-precision density matrix simulation
          return run_circuit_helper<
              DensityMatrixChunk::State<QV::DensityMatrix<double>>>(
              circ, noise, config, shots, rng_seed, Method::density_matrix,
              true, result);
        } else {
          // Single-precision density matrix simulation
          return run_circuit_helper<
              DensityMatrixChunk::State<QV::DensityMatrix<float>>>(
              circ, noise, config, shots, rng_seed, Method::density_matrix,
              true, result);
        }
      } else {
        if (sim_precision_ == Precision::Double) {
          // Double-precision density matrix simulation
          return run_circuit_helper<
              DensityMatrix::State<QV::DensityMatrix<double>>>(
              circ, noise, config, shots, rng_seed, Method::density_matrix,
              false, result);
        } else {
          // Single-precision density matrix simulation
          return run_circuit_helper<
              DensityMatrix::State<QV::DensityMatrix<float>>>(
              circ, noise, config, shots, rng_seed, Method::density_matrix,
              false, result);
        }
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      if (multiple_chunk_required(circ, noise) ||
          (parallel_shots_ > 1 || parallel_experiments_ > 1)) {
        if (sim_precision_ == Precision::Double) {
          // Double-precision density matrix simulation
          return run_circuit_helper<
              DensityMatrixChunk::State<QV::DensityMatrixThrust<double>>>(
              circ, noise, config, shots, rng_seed, Method::density_matrix,
              true, result);
        } else {
          // Single-precision density matrix simulation
          return run_circuit_helper<
              DensityMatrixChunk::State<QV::DensityMatrixThrust<float>>>(
              circ, noise, config, shots, rng_seed, Method::density_matrix,
              true, result);
        }
      } else {
        if (sim_precision_ == Precision::Double) {
          // Double-precision density matrix simulation
          return run_circuit_helper<
              DensityMatrix::State<QV::DensityMatrixThrust<double>>>(
              circ, noise, config, shots, rng_seed, Method::density_matrix,
              false, result);
        } else {
          // Single-precision density matrix simulation
          return run_circuit_helper<
              DensityMatrix::State<QV::DensityMatrixThrust<float>>>(
              circ, noise, config, shots, rng_seed, Method::density_matrix,
              false, result);
        }
      }
#endif
    }
  }
  case Method::unitary: {
    if (sim_device_ == Device::CPU) {
      if (multiple_chunk_required(circ, noise)) {
        if (sim_precision_ == Precision::Double) {
          // Double-precision unitary simulation
          return run_circuit_helper<
              QubitUnitaryChunk::State<QV::UnitaryMatrix<double>>>(
              circ, noise, config, shots, rng_seed, Method::unitary,
              false, result);
        } else {
          // Single-precision unitary simulation
          return run_circuit_helper<
              QubitUnitaryChunk::State<QV::UnitaryMatrix<float>>>(
              circ, noise, config, shots, rng_seed, Method::unitary,
              false, result);
        }
      }
      else{
        if (sim_precision_ == Precision::Double) {
          // Double-precision unitary simulation
          return run_circuit_helper<
              QubitUnitary::State<QV::UnitaryMatrix<double>>>(
              circ, noise, config, shots, rng_seed, Method::unitary,
              false, result);
        } else {
          // Single-precision unitary simulation
          return run_circuit_helper<
              QubitUnitary::State<QV::UnitaryMatrix<float>>>(
              circ, noise, config, shots, rng_seed, Method::unitary,
              false, result);
        }
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      if (multiple_chunk_required(circ, noise)) {
        if (sim_precision_ == Precision::Double) {
          // Double-precision unitary simulation
          return run_circuit_helper<
              QubitUnitaryChunk::State<QV::UnitaryMatrixThrust<double>>>(
              circ, noise, config, shots, rng_seed, Method::unitary,
              false, result);
        } else {
          // Single-precision unitary simulation
          return run_circuit_helper<
              QubitUnitaryChunk::State<QV::UnitaryMatrixThrust<float>>>(
              circ, noise, config, shots, rng_seed, Method::unitary,
              false, result);
        }
      }
      else{
        if (sim_precision_ == Precision::Double) {
          // Double-precision unitary simulation
          return run_circuit_helper<
              QubitUnitary::State<QV::UnitaryMatrixThrust<double>>>(
              circ, noise, config, shots, rng_seed, Method::unitary,
              false, result);
        } else {
          // Single-precision unitary simulation
          return run_circuit_helper<
              QubitUnitary::State<QV::UnitaryMatrixThrust<float>>>(
              circ, noise, config, shots, rng_seed, Method::unitary,
              false, result);
        }
      }
#endif
    }
  }
  case Method::superop: {
    if (sim_precision_ == Precision::Double) {
      return run_circuit_helper<
          QubitSuperoperator::State<QV::Superoperator<double>>>(
          circ, noise, config, shots, rng_seed, Method::superop,
          false, result);
    } else {
      return run_circuit_helper<
          QubitSuperoperator::State<QV::Superoperator<float>>>(
          circ, noise, config, shots, rng_seed, Method::superop,
          false, result);
    }
  }
  case Method::stabilizer:
    // Stabilizer simulation
    // TODO: Stabilizer doesn't yet support custom state initialization
    return run_circuit_helper<Stabilizer::State>(
        circ, noise, config, shots, rng_seed, Method::stabilizer,
        false, result);
  case Method::extended_stabilizer:
    return run_circuit_helper<ExtendedStabilizer::State>(
        circ, noise, config, shots, rng_seed, Method::extended_stabilizer,
        false, result);
  case Method::matrix_product_state:
    return run_circuit_helper<MatrixProductState::State>(
        circ, noise, config, shots, rng_seed, Method::matrix_product_state,
        false, result);
  default:
    throw std::runtime_error("Controller:Invalid simulation method");
  }
}

//-------------------------------------------------------------------------
// Utility methods
//-------------------------------------------------------------------------
Controller::Method
Controller::simulation_method(const Circuit &circ,
                              const Noise::NoiseModel &noise_model,
                              bool validate) const {
  // Check simulation method and validate state
  switch (sim_method_) {
  case Method::statevector: {
    if (validate) {
      if (sim_device_ == Device::CPU) {
        if (sim_precision_ == Precision::Single) {
          Statevector::State<QV::QubitVector<float>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        } else {
          Statevector::State<QV::QubitVector<double>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        }
      } else {
#ifdef AER_THRUST_SUPPORTED
        if (sim_precision_ == Precision::Single) {
          Statevector::State<QV::QubitVectorThrust<float>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        } else {
          Statevector::State<QV::QubitVectorThrust<>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        }
#endif
      }
    }
    return Method::statevector;
  }
  case Method::density_matrix: {
    if (validate) {
      if (sim_device_ == Device::CPU) {
        if (sim_precision_ == Precision::Single) {
          DensityMatrix::State<QV::DensityMatrix<float>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        } else {
          DensityMatrix::State<QV::DensityMatrix<double>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        }
      } else {
#ifdef AER_THRUST_SUPPORTED
        if (sim_precision_ == Precision::Single) {
          DensityMatrix::State<QV::DensityMatrixThrust<float>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        } else {
          DensityMatrix::State<QV::DensityMatrixThrust<double>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        }
#endif
      }
    }
    return Method::density_matrix;
  }
  case Method::unitary: {
    if (validate) {
      if (sim_device_ == Device::CPU) {
        if (sim_precision_ == Precision::Single) {
          QubitUnitary::State<QV::UnitaryMatrix<float>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        } else {
          QubitUnitary::State<QV::UnitaryMatrix<double>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        }
      } else {
#ifdef AER_THRUST_SUPPORTED
        if (sim_precision_ == Precision::Single) {
          QubitUnitary::State<QV::UnitaryMatrixThrust<float>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        } else {
          QubitUnitary::State<QV::UnitaryMatrixThrust<double>> state;
          validate_state(state, circ, noise_model, true);
          validate_memory_requirements(state, circ, true);
        }
#endif
      }
    }
    return Method::unitary;
  }
  case Method::superop: {
    if (validate) {
      if (sim_precision_ == Precision::Single) {
        QubitSuperoperator::State<QV::Superoperator<float>> state;
        validate_state(state, circ, noise_model, true);
        validate_memory_requirements(state, circ, true);
      } else {
        QubitSuperoperator::State<QV::Superoperator<double>> state;
        validate_state(state, circ, noise_model, true);
        validate_memory_requirements(state, circ, true);
      }
    }
    return Method::superop;
  }
  case Method::stabilizer: {
    if (validate) {
      Stabilizer::State state;
      validate_state(Stabilizer::State(), circ, noise_model, true);
      validate_memory_requirements(state, circ, true);
    }
    return Method::stabilizer;
  }
  case Method::extended_stabilizer: {
    if (validate) {
      ExtendedStabilizer::State state;
      validate_state(state, circ, noise_model, true);
      validate_memory_requirements(ExtendedStabilizer::State(), circ, true);
    }
    return Method::extended_stabilizer;
  }
  case Method::matrix_product_state: {
    if (validate) {
      MatrixProductState::State state;
      validate_state(state, circ, noise_model, true);
      validate_memory_requirements(state, circ, true);
    }
    return Method::matrix_product_state;
  }
  case Method::automatic: {
    // If circuit and noise model are Clifford run on Stabilizer simulator
    if (validate_state(Stabilizer::State(), circ, noise_model, false)) {
      return Method::stabilizer;
    }
    // For noisy simulations we enable the density matrix method if
    // shots > 2 ** num_qubits. This is based on a rough estimate that
    // a single shot of the density matrix simulator is approx 2 ** nq
    // times slower than a single shot of statevector due the increased
    // dimension
    if (noise_model.has_quantum_errors() &&
        circ.shots > (1ULL << circ.num_qubits) &&
        validate_memory_requirements(DensityMatrix::State<>(), circ, false) &&
        validate_state(DensityMatrix::State<>(), circ, noise_model, false) &&
        check_measure_sampling_opt(circ, Method::density_matrix)) {
      return Method::density_matrix;
    }
  
    // If the special conditions for stabilizer or density matrix are
    // not satisfied we choose simulation method based on supported
    // operations only with preference given by memory requirements
    // statevector > density matrix > matrix product state > unitary > superop
    // typically any save state instructions will decide the method.
    if (validate_state(Statevector::State<>(), circ, noise_model, false)) {
      return Method::statevector;
    }
    if (validate_state(DensityMatrix::State<>(), circ, noise_model, false)) {
      return Method::density_matrix;
    }
    if (validate_state(MatrixProductState::State(), circ, noise_model, false)) {
      return Method::matrix_product_state;
    }
    if (validate_state(QubitUnitary::State<>(), circ, noise_model, false)) {
      return Method::unitary;
    }
    if (validate_state(QubitSuperoperator::State<>(), circ, noise_model, false)) {
      return Method::superop;
    }
    // If we got here, circuit isn't compatible with any of the simulation
    // methods
    std::stringstream msg;
    msg << "AerSimulator: ";
    if (noise_model.is_ideal()) {
      msg << "circuit with instructions " << circ.opset();
    } else {
      auto opset = circ.opset();
      opset.insert(noise_model.opset());
      msg << "circuit and noise model with instructions" << opset;
    }
    msg << " is not compatible with any of the automatic simulation methods";
    throw std::runtime_error(msg.str());
  }}
}

size_t Controller::required_memory_mb(const Circuit &circ,
                                      const Noise::NoiseModel &noise) const {
  switch (simulation_method(circ, noise, false)) {
  case Method::statevector: {
    if (sim_precision_ == Precision::Single) {
      Statevector::State<QV::QubitVector<float>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    } else {
      Statevector::State<QV::QubitVector<double>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
  }
  case Method::density_matrix: {
    if (sim_precision_ == Precision::Single) {
      DensityMatrix::State<QV::DensityMatrix<float>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    } else {
      DensityMatrix::State<QV::DensityMatrix<double>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
  }
  case Method::unitary: {
    if (sim_precision_ == Precision::Single) {
      QubitUnitary::State<QV::UnitaryMatrix<float>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    } else {
      QubitUnitary::State<QV::UnitaryMatrix<double>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
  }
  case Method::superop: {
    if (sim_precision_ == Precision::Single) {
      QubitSuperoperator::State<QV::Superoperator<float>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    } else {
      QubitSuperoperator::State<QV::Superoperator<double>> state;
      return state.required_memory_mb(circ.num_qubits, circ.ops);
    }
  }
  case Method::stabilizer: {
    Stabilizer::State state;
    return state.required_memory_mb(circ.num_qubits, circ.ops);
  }
  case Method::extended_stabilizer: {
    ExtendedStabilizer::State state;
    return state.required_memory_mb(circ.num_qubits, circ.ops);
  }
  case Method::matrix_product_state: {
    MatrixProductState::State state;
    return state.required_memory_mb(circ.num_qubits, circ.ops);
  }
  default:
    // We shouldn't get here, so throw an exception if we do
    throw std::runtime_error("Controller: Invalid simulation method");
  }
}

Transpile::Fusion Controller::transpile_fusion(Method method,
                                               const Operations::OpSet &opset,
                                               const json_t &config) const {
  Transpile::Fusion fusion_pass;
  fusion_pass.set_parallelization(parallel_state_update_);

  if (opset.contains(Operations::OpType::superop)) {
    fusion_pass.allow_superop = true;
  }
  if (opset.contains(Operations::OpType::kraus)) {
    fusion_pass.allow_kraus = true;
  }
  switch (method) {
  case Method::density_matrix:
  case Method::unitary:
  case Method::superop: {
    // Halve the default threshold and max fused qubits for density matrix
    fusion_pass.threshold /= 2;
    fusion_pass.max_qubit /= 2;
    break;
  }
  case Method::matrix_product_state: {
    fusion_pass.active = false;
    return fusion_pass;  // Do not allow the config to set active for MPS
  }
  case Method::statevector: {
    if (fusion_pass.allow_kraus) {
      // Halve default max fused qubits for Kraus noise fusion
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

void Controller::set_parallelization_circuit_method(
    const Circuit &circ, const Noise::NoiseModel &noise_model) {
  const auto method = simulation_method(circ, noise_model, false);
  switch (method) {
  case Method::statevector:
  case Method::stabilizer:
  case Method::unitary:
  case Method::matrix_product_state: {
    if (circ.shots == 1 ||
        (!noise_model.has_quantum_errors() &&
         check_measure_sampling_opt(circ, method))) {
      parallel_shots_ = 1;
      parallel_state_update_ =
          std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
      return;
    }
    set_parallelization_circuit(circ, noise_model);
    break;
  }
  case Method::density_matrix:
  case Method::superop: {
    if (circ.shots == 1 ||
        check_measure_sampling_opt(circ, method)) {
      parallel_shots_ = 1;
      parallel_state_update_ =
          std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
      return;
    }
    set_parallelization_circuit(circ, noise_model);
    break;
  }
  default: {
    set_parallelization_circuit(circ, noise_model);
  }
  }
}

//-------------------------------------------------------------------------
// Run circuit helpers
//-------------------------------------------------------------------------

template <class State_t>
void Controller::run_circuit_helper(const Circuit &circ,
                                    const Noise::NoiseModel &noise,
                                    const json_t &config, uint_t shots,
                                    uint_t rng_seed, const Method method,
                                    bool cache_blocking,
                                    ExperimentResult &result) const {
  // Initialize new state object
  State_t state;

  // Check memory requirements, raise exception if they're exceeded
  validate_memory_requirements(state, circ, true);

  // Set state config
  state.set_config(config);
  state.set_parallalization(parallel_state_update_);
  state.set_distribution(num_processes_);
  state.set_global_phase(circ.global_phase_angle);

  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  result.set_config(config);
  result.metadata.add(method_names_.at(method), "method");
  if (method == Method::statevector || method == Method::density_matrix ||
      method == Method::unitary) {
    result.metadata.add(sim_device_name_, "device");
  } else {
    result.metadata.add("CPU", "device");
  }

  // Add measure sampling to metadata
  // Note: this will set to `true` if sampling is enabled for the circuit
  result.metadata.add(false, "measure_sampling");
  // Choose execution method based on noise and method
  Circuit opt_circ;

  // Ideal circuit
  if (noise.is_ideal()) {
    opt_circ = circ;
  }
  // Readout error only
  else if (noise.has_quantum_errors() == false) {
    opt_circ = noise.sample_noise(circ, rng);
  }
  // Superop noise sampling
  else if (method == Method::density_matrix || method == Method::superop) {
    // Sample noise using SuperOp method
    auto noise_superop = noise;
    noise_superop.activate_superop_method();
    opt_circ = noise_superop.sample_noise(circ, rng);
  }
  // Kraus noise sampling
  else if (noise.opset().contains(Operations::OpType::kraus) ||
           noise.opset().contains(Operations::OpType::superop)) {
    auto noise_kraus = noise;
    noise_kraus.activate_kraus_method();
    opt_circ = noise_kraus.sample_noise(circ, rng);
  }
  // General circuit noise sampling
  else {
    run_circuit_with_sampled_noise(circ, noise, config, shots, state, method,
                                   cache_blocking, result, rng);
    state.add_metadata(result);
    return;
  }

  // Run multishot simulation without noise sampling
  run_circuit_without_sampled_noise(opt_circ, config, shots, state,
                                    method, cache_blocking, result, rng);
  state.add_metadata(result);
}

template <class State_t>
void Controller::run_single_shot(const Circuit &circ, State_t &state,
                                 ExperimentResult &result,
                                 RngEngine &rng) const {
  state.initialize_qreg(circ.num_qubits);
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(circ.ops, result, rng, true);
  save_count_data(result, state.creg());
}

template <class State_t>
void Controller::run_circuit_without_sampled_noise(Circuit &circ,
                                                   const json_t &config,
                                                   uint_t shots,
                                                   State_t &state,
                                                   const Method method,
                                                   bool cache_blocking,
                                                   ExperimentResult &result,
                                                   RngEngine &rng) const {
  // Optimize circuit
  Noise::NoiseModel dummy_noise;
  Transpile::DelayMeasure measure_pass;
  measure_pass.set_config(config);
  measure_pass.optimize_circuit(circ, dummy_noise, state.opset(), result);

  auto fusion_pass = transpile_fusion(method, circ.opset(), config);
  fusion_pass.optimize_circuit(circ, dummy_noise, state.opset(), result);

  // Check if measure sampling supported
  const bool can_sample = check_measure_sampling_opt(circ, method);
  
  // Cache blocking pass
  uint_t block_bits = 0;
  if (cache_blocking) {
    auto cache_block_pass = transpile_cache_blocking(method, circ, dummy_noise, config);
    cache_block_pass.set_sample_measure(can_sample);
    cache_block_pass.optimize_circuit(circ, dummy_noise, state.opset(), result);
    if (cache_block_pass.enabled()) {
      block_bits = cache_block_pass.block_bits();
    }
  }

  // allocate qubit register
  state.allocate(max_qubits_, block_bits);

    // Check if measure sampler and optimization are valid
  if (can_sample) {
    // Implement measure sampler
    auto& ops = circ.ops;
    auto pos =circ.first_measure_pos; // Position of first measurement op
    auto it_pos = std::next(ops.begin(), pos);
    bool final_ops = (pos == ops.size());

    // Get measurement opts
    std::vector<Operations::Op> meas_ops;
    std::move(it_pos, ops.end(), std::back_inserter(meas_ops));
    ops.resize(pos);

    // Run circuit instructions before first measure
    state.initialize_qreg(circ.num_qubits);
    state.initialize_creg(circ.num_memory, circ.num_registers);
    state.apply_ops(ops, result, rng, final_ops);

    // Get measurement operations and set of measured qubits
    measure_sampler(meas_ops, shots, state, result, rng);

    // Add measure sampling metadata
    result.metadata.add(true, "measure_sampling");
  } else {
    // Perform standard execution if we cannot apply the
    // measurement sampling optimization
    while (shots-- > 0) {
      run_single_shot(circ, state, result, rng);
    }
  }
}                                  

template <class State_t>
void Controller::run_circuit_with_sampled_noise(
    const Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
    uint_t shots, State_t &state, const Method method, bool cache_blocking,
    ExperimentResult &result, RngEngine &rng) const {
  // Transpilation for circuit noise method
  auto fusion_pass = transpile_fusion(method, circ.opset(), config);
  auto cache_block_pass = transpile_cache_blocking(method, circ, noise, config);
  Transpile::DelayMeasure measure_pass;
  measure_pass.set_config(config);
  Noise::NoiseModel dummy_noise;

  // Sample noise using circuit method
  while (shots-- > 0) {
    Circuit noise_circ = noise.sample_noise(circ, rng);
    noise_circ.shots = 1;
    measure_pass.optimize_circuit(noise_circ, dummy_noise, state.opset(),
                                  result);
    fusion_pass.optimize_circuit(noise_circ, dummy_noise, state.opset(),
				 result);
    uint_t block_bits = 0;
    if (cache_blocking) {
      cache_block_pass.optimize_circuit(noise_circ, dummy_noise, state.opset(),
                                        result);
      if (cache_block_pass.enabled()) {
        block_bits = cache_block_pass.block_bits();
      }
    }

    // allocate qubit register
    state.allocate(max_qubits_, block_bits);

    run_single_shot(noise_circ, state, result, rng);
  }
}

//-------------------------------------------------------------------------
// Measure sampling optimization
//-------------------------------------------------------------------------

bool Controller::check_measure_sampling_opt(const Circuit &circ,
                                            const Method method) const {
  // Check if circuit has sampling flag disabled
  if (circ.can_sample == false) {
    return false;
  }

  // If density matrix, unitary, superop method all supported instructions
  // allow sampling
  if (method == Method::density_matrix ||
      method == Method::superop ||
      method == Method::unitary) {
    return true;
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
      circ.opset().contains(Operations::OpType::superop)) {
    return false;
  }
  // Otherwise true
  return true;
}

template <class State_t>
void Controller::measure_sampler(
    const std::vector<Operations::Op> &meas_roerror_ops, uint_t shots,
    State_t &state, ExperimentResult &result, RngEngine &rng) const {
  // Check if meas_circ is empty, and if so return initial creg
  if (meas_roerror_ops.empty()) {
    while (shots-- > 0) {
      save_count_data(result, state.creg());
    }
    return;
  }

  std::vector<Operations::Op> meas_ops;
  std::vector<Operations::Op> roerror_ops;
  for (const Operations::Op &op : meas_roerror_ops)
    if (op.type == Operations::OpType::roerror)
      roerror_ops.push_back(op);
    else /*(op.type == Operations::OpType::measure) */
      meas_ops.push_back(op);

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
  auto all_samples = state.sample_measure(meas_qubits, shots, rng);

  // Make qubit map of position in vector of measured qubits
  std::unordered_map<uint_t, uint_t> qubit_map;
  for (uint_t j = 0; j < meas_qubits.size(); ++j) {
    qubit_map[meas_qubits[j]] = j;
  }

  // Maps of memory and register to qubit position
  std::unordered_map<uint_t, uint_t> memory_map;
  std::unordered_map<uint_t, uint_t> register_map;
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
  // Convert opts to circuit so we can get the needed creg sizes
  // NB: this function could probably be moved somewhere else like Utils or Ops
  Circuit meas_circ(meas_roerror_ops);
  ClassicalRegister creg;
  while (!all_samples.empty()) {
    auto sample = all_samples.back();
    creg.initialize(meas_circ.num_memory, meas_circ.num_registers);

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
    save_count_data(result, creg);

    // pop off processed sample
    all_samples.pop_back();
  }
}

//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
