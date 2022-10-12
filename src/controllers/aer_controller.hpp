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

#include "transpile/cacheblocking.hpp"
#include "transpile/fusion.hpp"

#include "simulators/density_matrix/densitymatrix_state.hpp"
#include "simulators/extended_stabilizer/extended_stabilizer_state.hpp"
#include "simulators/matrix_product_state/matrix_product_state.hpp"
#include "simulators/stabilizer/stabilizer_state.hpp"
#include "simulators/statevector/qubitvector.hpp"
#include "simulators/statevector/statevector_state.hpp"
#include "simulators/superoperator/superoperator_state.hpp"
#include "simulators/unitary/unitary_state.hpp"

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
                 Noise::NoiseModel &noise_model,
                 const json_t &config);

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

  // Validation threshold for validating states and operators
  double validation_threshold_ = 1e-8;

  // Save counts as memory list
  bool save_creg_memory_ = false;

  // Simulation method
  Method method_ = Method::automatic;

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

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  void run_circuit(const Circuit &circ, const Noise::NoiseModel &noise,
                   const Method method,const json_t &config, ExperimentResult &result) const;

  //----------------------------------------------------------------
  // Run circuit helpers
  //----------------------------------------------------------------

  // Execute n-shots of a circuit on the input state
  template <class State_t>
  void run_circuit_helper(const Circuit &circ, const Noise::NoiseModel &noise,
                          const json_t &config, const Method method, 
                          ExperimentResult &result) const;

  // Execute a single shot a of circuit by initializing the state vector,
  // running all ops in circ, and updating data with
  // simulation output.
  template <class State_t>
  void run_single_shot(const Circuit &circ, State_t &state,
                       ExperimentResult &result, RngEngine &rng) const;

  // Execute a single shot a of circuit by initializing the state vector,
  // running all ops in circ, and updating data with
  // simulation output.
  template <class State_t>
  void run_with_sampling(const Circuit &circ,
                         State_t &state,
                         ExperimentResult &result,
                         RngEngine &rng,
                         const uint_t block_bits,
                         const uint_t shots) const;

  // Execute multiple shots a of circuit by initializing the state vector,
  // running all ops in circ, and updating data with
  // simulation output. Will use measurement sampling if possible
  template <class State_t>
  void run_circuit_without_sampled_noise(Circuit &circ,
                                         const Noise::NoiseModel &noise,
                                         const json_t &config,
                                         const Method method,
                                         ExperimentResult &result) const;

  template <class State_t>
  void run_circuit_with_sampled_noise(const Circuit &circ,
                                      const Noise::NoiseModel &noise,
                                      const json_t &config,
                                      const Method method,
                                      ExperimentResult &result) const;

  //----------------------------------------------------------------
  // Measurement
  //----------------------------------------------------------------

  // Sample measurement outcomes for the input measure ops from the
  // current state of the input State_t
  template <typename InputIterator, class State_t>
  void measure_sampler(InputIterator first_meas, InputIterator last_meas,
                       uint_t shots, State_t &state, ExperimentResult &result,
                       RngEngine &rng , int_t shot_index = -1) const;

  // Check if measure sampling optimization is valid for the input circuit
  // for the given method. This checks if operation types before
  // the first measurement in the circuit prevent sampling
  bool check_measure_sampling_opt(const Circuit &circ,
                                  const Method method) const;

  //-------------------------------------------------------------------------
  // State validation
  //-------------------------------------------------------------------------

  // Return True if the operations in the circuit and noise model are valid
  // for execution on the given method, and that the required memory is less
  // than the maximum allowed memory, otherwise return false.
  // If `throw_except` is true an exception will be thrown on the return false
  // case listing the invalid instructions in the circuit or noise model, or
  // the required memory.
  bool validate_method(Method method,
                       const Circuit &circ,
                       const Noise::NoiseModel &noise,
                       bool throw_except = false) const;
                            
  template <class state_t>
  bool validate_state(const state_t &state, const Circuit &circ,
                      const Noise::NoiseModel &noise,
                      bool throw_except = false) const;

  // Return an estimate of the required memory for a circuit.
  size_t required_memory_mb(const Circuit &circuit,
                            const Noise::NoiseModel &noise,
                            const Method method) const;

  //----------------------------------------------------------------
  // Utility functions
  //----------------------------------------------------------------
  
  // Return a vector of simulation methods for each circuit.
  // If the default method is automatic this will be computed based on the
  // circuit and noise model.
  // The noise model will be modified to enable superop or kraus sampling
  // methods if required by the chosen methods.
  std::vector<Controller::Method>
  simulation_methods(std::vector<Circuit> &circuits,
                     Noise::NoiseModel &noise_model) const;

  // Return the simulation method to use based on the input circuit
  // and noise model
  Controller::Method
  automatic_simulation_method(const Circuit &circ,
                              const Noise::NoiseModel &noise_model) const;

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

  //return maximum number of qubits for matrix
  int_t get_max_matrix_qubits(const Circuit &circ) const;
  int_t get_matrix_bits(const Operations::Op& op) const;

  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Set OpenMP thread settings to default values
  void clear_parallelization();

  // Set parallelization for experiments
  void
  set_parallelization_experiments(const std::vector<Circuit> &circuits,
                                  const Noise::NoiseModel &noise,
                                  const std::vector<Method> &methods);

  // Set circuit parallelization
  void set_parallelization_circuit(const Circuit &circ,
                                   const Noise::NoiseModel &noise,
                                   const Method method);

  bool multiple_chunk_required(const Circuit &circuit,
                               const Noise::NoiseModel &noise,
                               const Method method) const;

  bool multiple_shots_required(const Circuit &circuit,
                               const Noise::NoiseModel &noise,
                               const Method method) const;

  void save_exception_to_results(Result &result, const std::exception &e) const;

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

  //max number of states can be stored on memory for batched multi-shots/experiments optimization
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

  //multi-chunks are required to simulate circuits
  bool multi_chunk_required_ = false;

  //config setting for multi-shot parallelization
  bool batched_shots_gpu_ = true;
  int_t batched_shots_gpu_max_qubits_ = 16;   //multi-shot parallelization is applied if qubits is less than max qubits
  bool enable_batch_multi_shots_ = false;   //multi-shot parallelization can be applied

  //settings for cuStateVec
  bool cuStateVec_enable_ = false;
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

  //enable batched multi-shots/experiments optimization
  if(JSON::check_key("batched_shots_gpu", config)) {
    JSON::get_value(batched_shots_gpu_, "batched_shots_gpu", config);
  }
  if(JSON::check_key("batched_shots_gpu_max_qubits", config)) {
    JSON::get_value(batched_shots_gpu_max_qubits_, "batched_shots_gpu_max_qubits", config);
  }

  //cuStateVec configs
  cuStateVec_enable_ = false;
  if(JSON::check_key("cuStateVec_enable", config)) {
    JSON::get_value(cuStateVec_enable_, "cuStateVec_enable", config);
  }

  // Override automatic simulation method with a fixed method
  std::string method;
  if (JSON::get_value(method, "method", config)) {
    if (method == "statevector") {
      method_ = Method::statevector;
    } else if (method == "density_matrix") {
      method_ = Method::density_matrix;
    } else if (method == "stabilizer") {
      method_ = Method::stabilizer;
    } else if (method == "extended_stabilizer") {
      method_ = Method::extended_stabilizer;
    } else if (method == "matrix_product_state") {
      method_ = Method::matrix_product_state;
    } else if (method == "unitary") {
      method_ = Method::unitary;
    } else if (method == "superop") {
      method_ = Method::superop;
    } else if (method != "automatic") {
      throw std::runtime_error(std::string("Invalid simulation method (") +
                               method + std::string(")."));
    }
  }

  if(method_ == Method::density_matrix || method_ == Method::unitary)
    batched_shots_gpu_max_qubits_ /= 2;

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

#ifndef AER_CUSTATEVEC
      if(cuStateVec_enable_){
        //Aer is not built for cuStateVec
        throw std::runtime_error(
            "Simulation device \"GPU\" does not support cuStateVec on this system");
      }
#endif
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
  method_ = Method::automatic;
  sim_device_ = Device::CPU;
  sim_precision_ = Precision::Double;
}

void Controller::clear_parallelization() {
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
}

void Controller::set_parallelization_experiments(
    const std::vector<Circuit> &circuits,
    const Noise::NoiseModel &noise,
    const std::vector<Method> &methods) 
{
  std::vector<size_t> required_memory_mb_list(circuits.size());
  max_qubits_ = 0;
  for (size_t j = 0; j < circuits.size(); j++) {
    if(circuits[j].num_qubits > max_qubits_)
      max_qubits_ = circuits[j].num_qubits;
    required_memory_mb_list[j] = required_memory_mb(circuits[j], noise, methods[j]);
  }
  std::sort(required_memory_mb_list.begin(), required_memory_mb_list.end(),
            std::greater<>());

  //set max batchable number of circuits
  if(batched_shots_gpu_){
    if(required_memory_mb_list[0] == 0 || max_qubits_ == 0)
      max_batched_states_ = 1;
    else{
      if(sim_device_ == Device::GPU){
        max_batched_states_ = ((max_gpu_memory_mb_/num_gpus_*8/10) / required_memory_mb_list[0])*num_gpus_;
      }
      else{
        max_batched_states_ = (max_memory_mb_*8/10) / required_memory_mb_list[0];
      }
    }
  }
  if(max_qubits_ == 0)
    max_qubits_ = 1;

  if(explicit_parallelization_ )
    return;

  if(circuits.size() == 1){
    parallel_experiments_ = 1;
    return;
  }

  // Use a local variable to not override stored maximum based
  // on currently executed circuits
  const auto max_experiments =
      (max_parallel_experiments_ > 0)
          ? std::min({max_parallel_experiments_, max_parallel_threads_})
          : max_parallel_threads_;

  if (max_experiments == 1) {
    // No parallel experiment execution
    parallel_experiments_ = 1;
    return;
  }

  // If memory allows, execute experiments in parallel
  size_t total_memory = 0;
  int parallel_experiments = 0;
  for (size_t required_memory_mb : required_memory_mb_list) {
    total_memory += required_memory_mb;
    if (total_memory > max_memory_mb_)
      break;
    ++parallel_experiments;
  }

  if (parallel_experiments <= 0)
    throw std::runtime_error(
        "a circuit requires more memory than max_memory_mb.");
  parallel_experiments_ =
      std::min<int>({parallel_experiments, max_experiments,
                     max_parallel_threads_, static_cast<int>(circuits.size())});
}

void Controller::set_parallelization_circuit(const Circuit &circ,
                                             const Noise::NoiseModel &noise,
                                             const Method method)  
{
  enable_batch_multi_shots_ = false;
  if(batched_shots_gpu_ && sim_device_ == Device::GPU && 
     circ.shots > 1 && max_batched_states_ >= num_gpus_ && 
     batched_shots_gpu_max_qubits_ >= circ.num_qubits ){
      enable_batch_multi_shots_ = true;
  }

  if(sim_device_ == Device::GPU && cuStateVec_enable_){
    enable_batch_multi_shots_ = false;    //cuStateVec does not support batch execution of multi-shots
    return;
  }

  if(explicit_parallelization_)
    return;

  // Check for trivial parallelization conditions
  switch (method) {
    case Method::statevector:
    case Method::stabilizer:
    case Method::unitary:
    case Method::matrix_product_state: {
      if (circ.shots == 1 || num_process_per_experiment_ > 1 ||
          (!noise.has_quantum_errors() &&
          check_measure_sampling_opt(circ, method))) {
        parallel_shots_ = 1;
        parallel_state_update_ =
            std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
        return;
      }
      break;
    }
    case Method::density_matrix:
    case Method::superop: {
      if (circ.shots == 1 || num_process_per_experiment_ > 1 ||
          check_measure_sampling_opt(circ, method)) {
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
      throw std::invalid_argument("Cannot set parallelization for unresolved method.");
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
        required_memory_mb(circ, noise, method) / num_process_per_experiment_;
    size_t mem_size = (sim_device_ == Device::GPU) ? max_gpu_memory_mb_ : max_memory_mb_;
    if (mem_size < circ_memory_mb)
      throw std::runtime_error(
          "a circuit requires more memory than max_memory_mb.");
    // If circ memory is 0, set it to 1 so that we don't divide by zero
    circ_memory_mb = std::max<int>({1, circ_memory_mb});

    int shots = circ.shots;
    parallel_shots_ = std::min<int>(
        {static_cast<int>(mem_size/(circ_memory_mb*2)), max_shots, shots});
  }
  parallel_state_update_ =
      (parallel_shots_ > 1)
          ? std::max<int>({1, max_parallel_threads_ / parallel_shots_})
          : std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
}

bool Controller::multiple_chunk_required(const Circuit &circ,
                                         const Noise::NoiseModel &noise,
                                         const Method method) const 
{
  if (circ.num_qubits < 3)
    return false;
  if (cache_block_qubit_ >= 2 && cache_block_qubit_ < circ.num_qubits)
    return true;

  if(num_process_per_experiment_ == 1 && sim_device_ == Device::GPU && num_gpus_ > 0){
    return (max_gpu_memory_mb_ / num_gpus_ < required_memory_mb(circ, noise, method));
  }
  if(num_process_per_experiment_ > 1){
    size_t total_mem = max_memory_mb_;
    if(sim_device_ == Device::GPU)
      total_mem += max_gpu_memory_mb_;
    if(total_mem*num_process_per_experiment_ > required_memory_mb(circ, noise, method))
      return true;
  }

  return false;
}

bool Controller::multiple_shots_required(const Circuit &circ,
                                         const Noise::NoiseModel &noise,
                                         const Method method) const 
{
  if (circ.shots < 2)
    return false;
  if (method == Method::density_matrix ||
      method == Method::superop ||
      method == Method::unitary) {
    return false;
  }

  bool can_sample = check_measure_sampling_opt(circ, method);

  if (noise.is_ideal()){
   return !can_sample;
  }

  return true;
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


Transpile::CacheBlocking
Controller::transpile_cache_blocking(Controller::Method method, const Circuit &circ,
                                     const Noise::NoiseModel &noise,
                                     const json_t &config) const 
{
  Transpile::CacheBlocking cache_block_pass;

  const bool is_matrix = (method == Method::density_matrix
                          || method == Method::unitary);
  const auto complex_size = (sim_precision_ == Precision::Single)
                              ? sizeof(std::complex<float>)
                              : sizeof(std::complex<double>);

  cache_block_pass.set_num_processes(num_process_per_experiment_);
  cache_block_pass.set_config(config);

  if (!cache_block_pass.enabled()) {
    // if blocking is not set by config, automatically set if required
    if (multiple_chunk_required(circ, noise, method)) {
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

    // Initialize QOBJ
    Qobj qobj(input_qobj);
    auto qobj_time_taken =
        std::chrono::duration<double>(myclock_t::now() - timer_start).count();

    // Set config
    set_config(qobj.config);

    // Run qobj circuits
    auto result = execute(qobj.circuits, qobj.noise_model, qobj.config);

    // Add QOBJ loading time
    result.metadata.add(qobj_time_taken, "time_taken_load_qobj");

    // Get QOBJ id and pass through header to result
    result.qobj_id = qobj.id;
    if (!qobj.header.empty()) {
      result.header = qobj.header;
    }

    // Stop the timer and add total timing data including qobj parsing
    auto time_taken =
        std::chrono::duration<double>(myclock_t::now() - timer_start).count();
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
                           Noise::NoiseModel &noise_model,
                           const json_t &config) 
{
  // Start QOBJ timer
  auto timer_start = myclock_t::now();

  // Determine simulation method for each circuit
  // and enable required noise sampling methods
  auto methods = simulation_methods(circuits, noise_model);

  // Initialize Result object for the given number of experiments
  Result result(circuits.size());

  // Execute each circuit in a try block
  try {
    //check if multi-chunk distribution is required
    bool multi_chunk_required_ = false;
    for (size_t j = 0; j < circuits.size(); j++){
      if(circuits[j].num_qubits > 0){
        if(multiple_chunk_required(circuits[j], noise_model, methods[j]))
          multi_chunk_required_ = true;
      }
    }
    if(multi_chunk_required_)
      num_process_per_experiment_ = num_processes_;
    else
      num_process_per_experiment_ = 1;

    // set parallelization for experiments
    try {
      // catch exception raised by required_memory_mb because of invalid
      // simulation method
      set_parallelization_experiments(circuits, noise_model, methods);
    } catch (std::exception &e) {
      save_exception_to_results(result, e);
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
    result.metadata.add(num_process_per_experiment_, "num_processes_per_experiments");
    result.metadata.add(num_processes_, "num_mpi_processes");
    result.metadata.add(myrank_, "mpi_rank");

#ifdef _OPENMP
    // Check if circuit parallelism is nested with one of the others
    if (parallel_experiments_ > 1 &&
        parallel_experiments_ < max_parallel_threads_) {
      // Nested parallel experiments
      parallel_nested_ = true;

      //nested should be set to zero if num_threads clause will be used
      omp_set_nested(0);

      result.metadata.add(parallel_nested_, "omp_nested");
    } else {
      parallel_nested_ = false;
    }
#endif

#ifdef AER_MPI
    //average random seed to set the same seed to each process (when seed_simulator is not set)
    if(num_processes_ > 1){
      reg_t seeds(circuits.size());
      reg_t avg_seeds(circuits.size());
      for(int_t i=0;i<circuits.size();i++)
        seeds[i] = circuits[i].seed;
      MPI_Allreduce(seeds.data(), avg_seeds.data(), circuits.size(), MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
      for(int_t i=0;i<circuits.size();i++)
        circuits[i].seed = avg_seeds[i]/num_processes_;
    }
#endif

    const int NUM_RESULTS = result.results.size();
    //following looks very similar but we have to separate them to avoid omp nested loops that causes performance degradation
    //(DO NOT use if statement in #pragma omp)
    if (parallel_experiments_ == 1) {
      for (int j = 0; j < NUM_RESULTS; ++j) {
        set_parallelization_circuit(circuits[j], noise_model, methods[j]);
        run_circuit(circuits[j], noise_model,methods[j],
                    config, result.results[j]);
      }
    }
    else{
#pragma omp parallel for num_threads(parallel_experiments_)
      for (int j = 0; j < NUM_RESULTS; ++j) {
        run_circuit(circuits[j], noise_model,methods[j],
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
    result.metadata.add(time_taken, "time_taken_execute");
  }
  // If execution failed return valid output reporting error
  catch (std::exception &e) {
    result.status = Result::Status::error;
    result.message = e.what();
  }
  return result;
}

//-------------------------------------------------------------------------
// Base class override
//-------------------------------------------------------------------------
void Controller::run_circuit(const Circuit &circ, const Noise::NoiseModel &noise,
                 const Method method,const json_t &config, ExperimentResult &result) const
{
  // Run the circuit
  switch (method) {
  case Method::statevector: {
    if (sim_device_ == Device::CPU) {
      // Chunk based simualtion
      if (sim_precision_ == Precision::Double) {
        // Double-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVector<double>>>(
            circ, noise, config, Method::statevector, result);
      } else {
        // Single-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVector<float>>>(
            circ, noise, config, Method::statevector, result);
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      // Chunk based simulation
      if (sim_precision_ == Precision::Double) {
        // Double-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<double>>>(
            circ, noise, config, Method::statevector, result);
      } else {
        // Single-precision Statevector simulation
        return run_circuit_helper<
            Statevector::State<QV::QubitVectorThrust<float>>>(
            circ, noise, config, Method::statevector, result);
      }
#endif
    }
  }
  case Method::density_matrix: {
    if (sim_device_ == Device::CPU) {
      if (sim_precision_ == Precision::Double) {
        // Double-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrix<double>>>(
            circ, noise, config, Method::density_matrix, result);
      } else {
        // Single-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrix<float>>>(
            circ, noise, config, Method::density_matrix, result);
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      if (sim_precision_ == Precision::Double) {
        // Double-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrixThrust<double>>>(
            circ, noise, config, Method::density_matrix, result);
      } else {
        // Single-precision density matrix simulation
        return run_circuit_helper<
            DensityMatrix::State<QV::DensityMatrixThrust<float>>>(
            circ, noise, config, Method::density_matrix, result);
      }
#endif
    }
  }
  case Method::unitary: {
    if (sim_device_ == Device::CPU) {
      if (sim_precision_ == Precision::Double) {
        // Double-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrix<double>>>(
            circ, noise, config, Method::unitary, result);
      } else {
        // Single-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrix<float>>>(
            circ, noise, config, Method::unitary, result);
      }
    } else {
#ifdef AER_THRUST_SUPPORTED
      if (sim_precision_ == Precision::Double) {
        // Double-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrixThrust<double>>>(
            circ, noise, config, Method::unitary, result);
      } else {
        // Single-precision unitary simulation
        return run_circuit_helper<
            QubitUnitary::State<QV::UnitaryMatrixThrust<float>>>(
            circ, noise, config, Method::unitary, result);
      }
#endif
    }
  }
  case Method::superop: {
    if (sim_precision_ == Precision::Double) {
      return run_circuit_helper<
          QubitSuperoperator::State<QV::Superoperator<double>>>(
          circ, noise, config, Method::superop, result);
    } else {
      return run_circuit_helper<
          QubitSuperoperator::State<QV::Superoperator<float>>>(
          circ, noise, config, Method::superop, result);
    }
  }
  case Method::stabilizer:
    // Stabilizer simulation
    // TODO: Stabilizer doesn't yet support custom state initialization
    return run_circuit_helper<Stabilizer::State>(
        circ, noise, config, Method::stabilizer, result);
  case Method::extended_stabilizer:
    return run_circuit_helper<ExtendedStabilizer::State>(
        circ, noise, config, Method::extended_stabilizer, result);
  case Method::matrix_product_state:
    return run_circuit_helper<MatrixProductState::State>(
        circ, noise, config, Method::matrix_product_state, result);
  default:
    throw std::runtime_error("Controller:Invalid simulation method");
  }
}

//-------------------------------------------------------------------------
// Utility methods
//-------------------------------------------------------------------------

size_t Controller::required_memory_mb(const Circuit &circ,
                                      const Noise::NoiseModel &noise,
                                      const Method method) const {
  switch (method) {
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
  case Method::unitary: {
    // max_qubit is the same with statevector
    fusion_pass.threshold /= 2;
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

//-------------------------------------------------------------------------
// Run circuit helpers
//-------------------------------------------------------------------------

template <class State_t>
void Controller::run_circuit_helper(const Circuit &circ,
                                    const Noise::NoiseModel &noise,
                                    const json_t &config,
                                    const Method method,
                                    ExperimentResult &result) const
{
  // Start individual circuit timer
  auto timer_start = myclock_t::now(); // state circuit timer

  // Initialize circuit json return
  result.legacy_data.set_config(config);

  // Execute in try block so we can catch errors and return the error message
  // for individual circuit failures.
  try {
    // Rng engine (this one is used to add noise on circuit)
    RngEngine rng;
    rng.set_seed(circ.seed);

    // Output data container
    result.set_config(config);
    result.metadata.add(method_names_.at(method), "method");
    if (method == Method::statevector || method == Method::density_matrix ||
        method == Method::unitary) {
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

    if(circ.num_qubits > 0){  //do nothing for query steps
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
      else if (method == Method::density_matrix || method == Method::superop) {
        // Sample noise using SuperOp method
        opt_circ = noise.sample_noise(circ, rng, Noise::NoiseModel::Method::superop);
        result.metadata.add("superop", "noise");
      }
      // Kraus noise sampling
      else if (noise.opset().contains(Operations::OpType::kraus) ||
               noise.opset().contains(Operations::OpType::superop)) {
        opt_circ = noise.sample_noise(circ, rng, Noise::NoiseModel::Method::kraus);
        result.metadata.add("kraus", "noise");
      }
      // General circuit noise sampling
      else {
        if(enable_batch_multi_shots_ && !multi_chunk_required_){
          //batched optimization samples noise at runtime
          opt_circ = noise.sample_noise(circ, rng, Noise::NoiseModel::Method::circuit, true);
        }
        else{
          noise_sampling = true;
        }
        result.metadata.add("circuit", "noise");
      }

      if(noise_sampling){
        run_circuit_with_sampled_noise<State_t>(circ, noise, config, method, result);
      }
      else{
        // Run multishot simulation without noise sampling
        run_circuit_without_sampled_noise<State_t>(opt_circ, noise, config, method, result);
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

template <class State_t>
void Controller::run_single_shot(const Circuit &circ, State_t &state,
                                 ExperimentResult &result,
                                 RngEngine &rng) const {
  state.initialize_qreg(circ.num_qubits);
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(circ.ops.cbegin(), circ.ops.cend(), result, rng, true);
  result.save_count_data(state.cregs(), save_creg_memory_);
}

template <class State_t>
void Controller::run_with_sampling(const Circuit &circ,
                                   State_t &state,
                                   ExperimentResult &result,
                                   RngEngine &rng,
                                   const uint_t block_bits,
                                   const uint_t shots) const {
  auto& ops = circ.ops;
  auto first_meas = circ.first_measure_pos; // Position of first measurement op
  bool final_ops = (first_meas == ops.size());

  // allocate qubit register
  state.allocate(circ.num_qubits, block_bits);

  // Run circuit instructions before first measure
  state.initialize_qreg(circ.num_qubits);
  state.initialize_creg(circ.num_memory, circ.num_registers);

  state.apply_ops(ops.cbegin(), ops.cbegin() + first_meas, result, rng, final_ops);

  // Get measurement operations and set of measured qubits
  measure_sampler(circ.ops.begin() + first_meas, circ.ops.end(), shots, state, result, rng);
}

template <class State_t>
void Controller::run_circuit_without_sampled_noise(Circuit &circ,
                                                   const Noise::NoiseModel &noise,
                                                   const json_t &config,
                                                   const Method method,
                                                   ExperimentResult &result) const 
{
  State_t state;

  // Validate gateset and memory requirements, raise exception if they're exceeded
  validate_state(state, circ, noise, true);

  // Set state config
  state.set_config(config);
  state.set_parallelization(parallel_state_update_);
  state.set_global_phase(circ.global_phase_angle);

  bool can_sample = circ.can_sample;

  // Optimize circuit
  Noise::NoiseModel dummy_noise;

  auto fusion_pass = transpile_fusion(method, circ.opset(), config);
  fusion_pass.optimize_circuit(circ, dummy_noise, state.opset(), result);

  // Cache blocking pass
  uint_t block_bits = circ.num_qubits;
  if(state.multi_chunk_distribution_supported()){
    auto cache_block_pass = transpile_cache_blocking(method, circ, dummy_noise, config);
    cache_block_pass.set_sample_measure(can_sample);
    cache_block_pass.optimize_circuit(circ, dummy_noise, state.opset(), result);
    if (cache_block_pass.enabled()) {
      block_bits = cache_block_pass.block_bits();
    }
  }
  // Check if measure sampling supported
  can_sample &= check_measure_sampling_opt(circ, method);
  auto max_bits = get_max_matrix_qubits(circ);

  // Check if measure sampler and optimization are valid
  if (can_sample) {
    // Implement measure sampler
    if (parallel_shots_ <= 1) {
      state.set_distribution(num_process_per_experiment_);
      state.set_max_matrix_qubits(max_bits);
      RngEngine rng;
      rng.set_seed(circ.seed);
      run_with_sampling(circ, state, result, rng, block_bits, circ.shots);
    } else {
      // Vector to store parallel thread output data
      std::vector<ExperimentResult> par_results(parallel_shots_);

#pragma omp parallel for num_threads(parallel_shots_)
      for (int i = 0; i < parallel_shots_; i++) {
        uint_t i_shot = circ.shots*i/parallel_shots_;
        uint_t shot_end = circ.shots*(i+1)/parallel_shots_;
        uint_t this_shot = shot_end - i_shot;

        State_t shot_state;
        // Set state config
        shot_state.set_config(config);
        shot_state.set_parallelization(parallel_state_update_);
        shot_state.set_global_phase(circ.global_phase_angle);

        shot_state.set_max_matrix_qubits(max_bits);

        RngEngine rng;
        rng.set_seed(circ.seed + i);

        run_with_sampling(circ, shot_state, par_results[i], rng, block_bits, this_shot);

        shot_state.add_metadata(par_results[i]);
      }
      for (auto &res : par_results) {
        result.combine(std::move(res));
      }

      if (sim_device_name_ == "GPU"){
        if(parallel_shots_ >= num_gpus_)
          result.metadata.add(num_gpus_, "gpu_parallel_shots_");
        else
          result.metadata.add(parallel_shots_, "gpu_parallel_shots_");
      }
    }
    // Add measure sampling metadata
    result.metadata.add(true, "measure_sampling");

  }
  else{
    // Perform standard execution if we cannot apply the
    // measurement sampling optimization

    if(block_bits == circ.num_qubits && enable_batch_multi_shots_ && state.multi_shot_parallelization_supported()){
      //apply batched multi-shots optimization (currenly only on GPU)
      state.set_max_bached_shots(max_batched_states_);
      state.set_distribution(num_processes_);
      state.set_max_matrix_qubits(max_bits);
      state.allocate(circ.num_qubits, circ.num_qubits, circ.shots);    //allocate multiple-shots

      //qreg is initialized inside state class
      state.initialize_creg(circ.num_memory, circ.num_registers);

      state.apply_ops_multi_shots(circ.ops.cbegin(), circ.ops.cend(), noise, result, circ.seed, true);

      result.save_count_data(state.cregs(), save_creg_memory_);

      // Add batched multi-shots optimizaiton metadata
      result.metadata.add(true, "batched_shots_optimization");
    }
    else{
      std::vector<ExperimentResult> par_results(parallel_shots_);
      int_t par_shots = parallel_shots_;
      if(block_bits != circ.num_qubits)
        par_shots = 1;

      auto run_circuit_without_sampled_noise_lambda = [this,&par_results,circ,noise,config,method,block_bits,max_bits,par_shots](int_t i){
        uint_t i_shot,shot_end;
        i_shot = circ.shots*i/par_shots;
        shot_end = circ.shots*(i+1)/par_shots;

        State_t par_state;
        // Set state config
        par_state.set_config(config);
        par_state.set_parallelization(parallel_state_update_);
        par_state.set_global_phase(circ.global_phase_angle);

        par_state.set_distribution(num_process_per_experiment_);
        par_state.set_max_matrix_qubits(max_bits );

        // allocate qubit register
        par_state.allocate(circ.num_qubits, block_bits);

        for(;i_shot<shot_end;i_shot++){
          RngEngine rng;
          rng.set_seed(circ.seed + i_shot);
          run_single_shot(circ, par_state, par_results[i], rng);
        }
        par_state.add_metadata(par_results[i]);
      };
      Utils::apply_omp_parallel_for((par_shots > 1),0,par_shots,run_circuit_without_sampled_noise_lambda);

      for (auto &res : par_results) {
        result.combine(std::move(res));
      }
      if (sim_device_name_ == "GPU"){
        if(par_shots >= num_gpus_)
          result.metadata.add(num_gpus_, "gpu_parallel_shots_");
        else
          result.metadata.add(par_shots, "gpu_parallel_shots_");
      }
    }
  }
  state.add_metadata(result);
}

template <class State_t>
void Controller::run_circuit_with_sampled_noise(
    const Circuit &circ, const Noise::NoiseModel &noise, const json_t &config,
    const Method method, ExperimentResult &result) const 
{
  std::vector<ExperimentResult> par_results(parallel_shots_);

  auto run_circuit_with_sampled_noise_lambda = [this,&par_results,circ,noise,config,method](int_t i){
    State_t state;
    uint_t i_shot,shot_end;
    Noise::NoiseModel dummy_noise;

    // Validate gateset and memory requirements, raise exception if they're exceeded
    validate_state(state, circ, noise, true);

    // Set state config
    state.set_config(config);
    state.set_parallelization(parallel_state_update_);
    state.set_global_phase(circ.global_phase_angle);

    // Transpilation for circuit noise method
    auto fusion_pass = transpile_fusion(method, circ.opset(), config);
    auto cache_block_pass = transpile_cache_blocking(method, circ, noise, config);

    i_shot = circ.shots*i/parallel_shots_;
    shot_end = circ.shots*(i+1)/parallel_shots_;

    for(;i_shot<shot_end;i_shot++){
      RngEngine rng;
      rng.set_seed(circ.seed + i_shot);

      // Sample noise using circuit method
      Circuit noise_circ = noise.sample_noise(circ, rng);

      noise_circ.shots = 1;
      fusion_pass.optimize_circuit(noise_circ, dummy_noise, state.opset(),
                                   par_results[i]);
      uint_t block_bits = circ.num_qubits;
      if(state.multi_chunk_distribution_supported()){
        cache_block_pass.optimize_circuit(noise_circ, dummy_noise, state.opset(),
                                          par_results[i]);
       if (cache_block_pass.enabled()) {
         block_bits = cache_block_pass.block_bits();
        }
      }

      state.set_distribution(num_process_per_experiment_);
      state.set_max_matrix_qubits(get_max_matrix_qubits(circ) );
      // allocate qubit register
      state.allocate(noise_circ.num_qubits, block_bits);

      run_single_shot(noise_circ, state, par_results[i], rng);
    }
    state.add_metadata(par_results[i]);
  };
  Utils::apply_omp_parallel_for((parallel_shots_ > 1),0,parallel_shots_,run_circuit_with_sampled_noise_lambda);

  for (auto &res : par_results) {
    result.combine(std::move(res));
  }

  if (sim_device_name_ == "GPU"){
    if(parallel_shots_ >= num_gpus_)
      result.metadata.add(num_gpus_, "gpu_parallel_shots_");
    else
      result.metadata.add(parallel_shots_, "gpu_parallel_shots_");
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
      circ.opset().contains(Operations::OpType::superop) ||
      circ.opset().contains(Operations::OpType::jump) ||
      circ.opset().contains(Operations::OpType::mark )) {
    return false;
  }
  // Otherwise true
  return true;
}

template <typename InputIterator, class State_t>
void Controller::measure_sampler(
    InputIterator first_meas, InputIterator last_meas, uint_t shots,
    State_t &state, ExperimentResult &result, RngEngine &rng, int_t shot_index) const 
{
  // Check if meas_circ is empty, and if so return initial creg
  if (first_meas == last_meas) {
    while (shots-- > 0) {
      result.save_count_data(state.cregs(), save_creg_memory_);
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
  uint_t shots_or_index;
  if(shot_index < 0)
    shots_or_index = shots;
  else
    shots_or_index = shot_index;

  auto timer_start = myclock_t::now();
  auto all_samples = state.sample_measure(meas_qubits, shots_or_index, rng);
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
      result.save_count_data(creg, save_creg_memory_);

    // pop off processed sample
    all_samples.pop_back();
  }
}


//-------------------------------------------------------------------------
// Validation
//-------------------------------------------------------------------------

std::vector<Controller::Method>
Controller::simulation_methods(std::vector<Circuit> &circuits,
                               Noise::NoiseModel &noise_model) const {
  // Does noise model contain kraus noise
  bool kraus_noise = (noise_model.opset().contains(Operations::OpType::kraus) ||
                      noise_model.opset().contains(Operations::OpType::superop));

  if (method_ == Method::automatic) {
    // Determine simulation methods for each circuit and noise model
    std::vector<Method> sim_methods;
    bool superop_enabled = false;
    bool kraus_enabled = false;
    for (const auto& circ: circuits) {
      auto method = automatic_simulation_method(circ, noise_model);
      sim_methods.push_back(method);
      if (!superop_enabled && (method == Method::density_matrix || method == Method::superop)) {
        noise_model.enable_superop_method(max_parallel_threads_);
        superop_enabled = true;
      } else if (kraus_noise && !kraus_enabled &&
                 (method == Method::statevector || method == Method::matrix_product_state)) {
        noise_model.enable_kraus_method(max_parallel_threads_);
        kraus_enabled = true;
      }
    }
    return sim_methods;
  }

  // Use non-automatic default method for all circuits
  std::vector<Method> sim_methods(circuits.size(), method_);
  if (method_ == Method::density_matrix || method_ == Method::superop) {
    noise_model.enable_superop_method(max_parallel_threads_);
  } else if (kraus_noise && (
              method_ == Method::statevector
              || method_ == Method::matrix_product_state)) {
    noise_model.enable_kraus_method(max_parallel_threads_);
  }
  return sim_methods;
}

Controller::Method
Controller::automatic_simulation_method(const Circuit &circ,
                                        const Noise::NoiseModel &noise_model) const {
  // If circuit and noise model are Clifford run on Stabilizer simulator
  if (validate_method(Method::stabilizer, circ, noise_model, false)) {
    return Method::stabilizer;
  }
  // For noisy simulations we enable the density matrix method if
  // shots > 2 ** num_qubits. This is based on a rough estimate that
  // a single shot of the density matrix simulator is approx 2 ** nq
  // times slower than a single shot of statevector due the increased
  // dimension
  if (noise_model.has_quantum_errors() && circ.num_qubits < 64 &&
      circ.shots > (1ULL << circ.num_qubits) &&
      validate_method(Method::density_matrix, circ, noise_model, false) &&
      check_measure_sampling_opt(circ, Method::density_matrix)) {
    return Method::density_matrix;
  }

  // If the special conditions for stabilizer or density matrix are
  // not satisfied we choose simulation method based on supported
  // operations only with preference given by memory requirements
  // statevector > density matrix > matrix product state > unitary > superop
  // typically any save state instructions will decide the method.
  const std::vector<Method> methods({Method::statevector,
                                     Method::density_matrix,
                                     Method::matrix_product_state,
                                     Method::unitary,
                                     Method::superop});
  for (const auto& method : methods) {
    if (validate_method(method, circ, noise_model, false))
      return method;
  }

  // If we got here, circuit isn't compatible with any of the simulation
  // method so fallback to a default method of statevector. The execution will
  // fail but we will get partial result generation and generate a user facing
  // error message
  return Method::statevector;
}

bool Controller::validate_method(Method method,
                                 const Circuit &circ, 
                                 const Noise::NoiseModel &noise_model,
                                 bool throw_except) const {
  // Switch wrapper for templated function validate_state
  switch (method) {
    case Method::stabilizer:
      return validate_state(Stabilizer::State(), circ, noise_model, throw_except);
    case Method::extended_stabilizer:
      return validate_state(ExtendedStabilizer::State(), circ, noise_model, throw_except);
    case Method::matrix_product_state:
      return validate_state(MatrixProductState::State(), circ, noise_model, throw_except);
    case Method::statevector:
      return validate_state(Statevector::State<>(), circ, noise_model,  throw_except);
    case Method::density_matrix:
      return validate_state(DensityMatrix::State<>(), circ, noise_model, throw_except);
    case Method::unitary:
      return validate_state(QubitUnitary::State<>(), circ, noise_model, throw_except);
    case Method::superop:
      return validate_state(QubitSuperoperator::State<>(), circ, noise_model, throw_except);
    case Method::automatic:
      throw std::runtime_error("Cannot validate circuit for unresolved simulation method.");
  }
}


template <class state_t>
bool Controller::validate_state(const state_t &state, const Circuit &circ,
                                const Noise::NoiseModel &noise,
                                bool throw_except) const {
  std::stringstream error_msg;
  std::string circ_name;
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
    size_t required_mb = state.required_memory_mb(circ.num_qubits, circ.ops) / num_process_per_experiment_;                                        
    size_t mem_size = (sim_device_ == Device::GPU) ? max_memory_mb_ + max_gpu_memory_mb_ : max_memory_mb_;
    memory_valid = (required_mb <= mem_size);
    if (throw_except && !memory_valid) {
      error_msg << "Insufficient memory to run circuit " << circ_name;
      error_msg << " using the " << state.name() << " simulator.";
      error_msg << " Required memory: " << required_mb << "M, max memory: " << max_memory_mb_ << "M";
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

void Controller::save_exception_to_results(Result &result,
                                           const std::exception &e) const {
  result.status = Result::Status::error;
  result.message = e.what();
  for (auto &res : result.results) {
    res.status = ExperimentResult::Status::error;
    res.message = e.what();
  }
}

int_t Controller::get_matrix_bits(const Operations::Op& op) const
{
  int_t bit = 1;
  if(op.type == Operations::OpType::matrix || op.type == Operations::OpType::diagonal_matrix || op.type == Operations::OpType::initialize)
    bit = op.qubits.size();
  else if(op.type == Operations::OpType::kraus || op.type == Operations::OpType::superop){
    if(method_ == Method::density_matrix)
      bit = op.qubits.size() * 2;
    else
      bit = op.qubits.size();
  }
  return bit;
}

int_t Controller::get_max_matrix_qubits(const Circuit &circ) const
{
  int_t max_bits = 0;
  int_t i;

  for(i=0;i<circ.ops.size();i++){
    int_t bit = 1;
    bit = get_matrix_bits(circ.ops[i]);
    max_bits = std::max(max_bits,bit);
  }
  return max_bits;
}

//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
