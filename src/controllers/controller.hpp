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

#ifndef _aer_base_controller_hpp_
#define _aer_base_controller_hpp_

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


// Base Controller
#include "framework/creg.hpp"
#include "framework/qobj.hpp"
#include "framework/results/experiment_result.hpp"
#include "framework/results/result.hpp"
#include "framework/rng.hpp"
#include "noise/noise_model.hpp"
#include "transpile/basic_opts.hpp"
#include "transpile/truncate_qubits.hpp"
#include "transpile/cacheblocking.hpp"

namespace AER {
namespace Base {

//=========================================================================
// Controller base class
//=========================================================================

// This is the top level controller for the Qiskit-Aer simulator
// It manages execution of all the circuits in a QOBJ, parallelization,
// noise sampling from a noise model, and circuit optimizations.

/**************************************************************************
 * ---------------
 * Parallelization
 * ---------------
 * Parallel execution uses the OpenMP library. It may happen at three levels:
 *
 *  1. Parallel execution of circuits in a QOBJ
 *  2. Parallel execution of shots in a Circuit
 *  3. Parallelization used by the State class for performing gates.
 *
 * Options 1 and 2 are mutually exclusive: enabling circuit parallelization
 * disables shot parallelization. Option 3 is available for both cases but
 * conservatively limits the number of threads since these are subthreads
 * spawned by the higher level threads. If no parallelization is used for
 * 1 and 2, all available threads will be used for 3.
 *
 * -------------------------
 * Config settings:
 *
 * - "noise_model" (json): A noise model to use for simulation [Default: null]
 * - "max_parallel_threads" (int): Set the maximum OpenMP threads that may
 *      be used across all levels of parallelization. Set to 0 for maximum
 *      available. [Default : 0]
 * - "max_parallel_experiments" (int): Set number of circuits that may be
 *      executed in parallel. Set to 0 to automatically select a number of
 *      parallel threads. [Default: 1]
 * - "max_parallel_shots" (int): Set number of shots that maybe be executed
 *      in parallel for each circuit. Set to 0 to automatically select a
 *      number of parallel threads. [Default: 0].
 * - "max_memory_mb" (int): Sets the maximum size of memory for a store.
 *      If a state needs more, an error is thrown. If set to 0, the maximum
 *      will be automatically set to the system memory size [Default: 0].
 *
 * Config settings from Data class:
 *
 * - "counts" (bool): Return counts object in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 * - "register" (bool): Return register array in circuit data [Default: False]
 **************************************************************************/

class Controller {
public:
  Controller() { clear_parallelization(); }

  //-----------------------------------------------------------------------
  // Execute qobj
  //-----------------------------------------------------------------------

  // Load a QOBJ from a JSON file and execute on the State type
  // class.
  template <typename inputdata_t>
  Result execute(const inputdata_t& qobj);

  virtual Result execute(std::vector<Circuit> &circuits,
                         const Noise::NoiseModel &noise_model,
                         const json_t &config);

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  // Load Controller, State and Data config from a JSON
  // config settings will be passed to the State and Data classes
  virtual void set_config(const json_t &config);

  // Clear the current config
  void virtual clear_config();

protected:
  //-----------------------------------------------------------------------
  // Circuit Execution
  //-----------------------------------------------------------------------

  // Parallel execution of a circuit
  // This function manages parallel shot configuration and internally calls
  // the `run_circuit` method for each shot thread
  virtual void execute_circuit(Circuit &circ,
                               Noise::NoiseModel &noise,
                               const json_t &config,
                               ExperimentResult &result);

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  virtual void run_circuit(const Circuit &circ, const Noise::NoiseModel &noise,
                           const json_t &config, uint_t shots, uint_t rng_seed,
                           ExperimentResult &result) const = 0;

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

  // Save count data
  void save_count_data(ExperimentResult &result,
                       const ClassicalRegister &creg) const;

  //-----------------------------------------------------------------------
  // Parallelization Config
  //-----------------------------------------------------------------------

  // Set OpenMP thread settings to default values
  void clear_parallelization();

  // Set parallelization for experiments
  virtual void
  set_parallelization_experiments(const std::vector<Circuit> &circuits,
                                  const std::vector<Noise::NoiseModel> &noise);

  // Set parallelization for a circuit
  virtual void set_parallelization_circuit(const Circuit &circuit,
                                           const Noise::NoiseModel &noise);

  // Return an estimate of the required memory for a circuit.
  virtual size_t required_memory_mb(const Circuit &circuit,
                                    const Noise::NoiseModel &noise) const = 0;

  // Set distributed parallelization
  virtual void
  set_distributed_parallelization(const std::vector<Circuit> &circuits,
                                  const std::vector<Noise::NoiseModel> &noise);

  virtual bool multiple_chunk_required(const Circuit &circuit,
                                  const Noise::NoiseModel &noise) const;

  void save_exception_to_results(Result &result,const std::exception &e);


  //setting cache blocking transpiler
  Transpile::CacheBlocking transpile_cache_blocking(const Circuit& circ,
                                     const Noise::NoiseModel& noise,
                                     const json_t& config,
                                     const size_t complex_size,bool is_matrix) const;


  // Get system memory size
  size_t get_system_memory_mb();
  size_t get_gpu_memory_mb();

  uint_t get_distributed_num_processes(bool par_shots) const;

  size_t get_min_memory_mb() const
  {
    if(num_gpus_ > 0){
      return max_gpu_memory_mb_ / num_gpus_;  //return per GPU memory size
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
  int num_gpus_;    //max number of GPU per process

  // use explicit parallelization
  bool explicit_parallelization_;

  // Parameters for parallelization management for experiments
  int parallel_experiments_;
  int parallel_shots_;
  int parallel_state_update_;

  bool parallel_nested_ = false;

  //max number of qubits in given circuits
  int max_qubits_;

  //results are stored independently in each process if true
  bool accept_distributed_results_ = true;

  //distributed experiments (MPI)
  int distributed_experiments_rank_ = 0;
  int distributed_experiments_group_id_ = 0;
  uint_t distributed_experiments_num_processes_ = 1;
  int distributed_experiments_ = 1;
  uint_t num_process_per_experiment_;
  uint_t distributed_experiments_begin_;
  uint_t distributed_experiments_end_;

  //distributed shots (MPI)
  int distributed_shots_rank_ = 0;
  int distributed_shots_ = 1;

  //process information (MPI)
  int myrank_ = 0;
  int num_processes_ = 1;

  uint_t cache_block_qubit_ = 0;
};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Config settings
//-------------------------------------------------------------------------

void Controller::set_config(const json_t &config) {
#ifdef AER_THRUST_CUDA
  {
    std::string method;
    if (JSON::get_value(method, "method", config)) {
      if(method.find("gpu") != std::string::npos){
        int nDev;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
          cudaGetLastError();
          throw std::runtime_error("No CUDA device available!");
        }
      }
    }
  }
#endif

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
    JSON::get_value(accept_distributed_results_, "accept_distributed_results", config);
  }

  //enable multiple qregs if cache blocking is enabled
  cache_block_qubit_ = 0;
  if(JSON::check_key("blocking_qubits", config)){
    JSON::get_value(cache_block_qubit_,"blocking_qubits", config);
  }
}

void Controller::clear_config() {
  clear_parallelization();
  validation_threshold_ = 1e-8;
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
  distributed_experiments_ = 1;
  distributed_shots_ = 1;

  num_gpus_ = 0;

  explicit_parallelization_ = false;
  max_memory_mb_ = get_system_memory_mb();
  max_gpu_memory_mb_ = get_gpu_memory_mb();
}

void Controller::set_parallelization_experiments(
    const std::vector<Circuit> &circuits, const std::vector<Noise::NoiseModel> &noise) 
{
  // Use a local variable to not override stored maximum based
  // on currently executed circuits
  const auto max_experiments =
      (max_parallel_experiments_ > 0)
          ? std::min({max_parallel_experiments_, max_parallel_threads_})
          : max_parallel_threads_;

  if (max_experiments == 1 && num_processes_ == 1) {
    // No parallel experiment execution
    parallel_experiments_ = 1;
    return;
  }

  // If memory allows, execute experiments in parallel
#ifdef AER_MPI
  std::vector<size_t> required_memory_mb_list(distributed_experiments_end_ - distributed_experiments_begin_);
  for (size_t j = 0; j < distributed_experiments_end_-distributed_experiments_begin_; j++) {
    required_memory_mb_list[j] = required_memory_mb(circuits[j+distributed_experiments_begin_], noise[j+distributed_experiments_begin_]) / num_process_per_experiment_;
  }
#else
  std::vector<size_t> required_memory_mb_list(circuits.size());
  for (size_t j = 0; j < circuits.size(); j++) {
    required_memory_mb_list[j] = required_memory_mb(circuits[j], noise[j]);
  }
#endif
  std::sort(required_memory_mb_list.begin(), required_memory_mb_list.end(),
            std::greater<>());
  size_t total_memory = 0;
  parallel_experiments_ = 0;
  for (size_t required_memory_mb : required_memory_mb_list) {
    total_memory += required_memory_mb;
    if (total_memory > max_memory_mb_*num_process_per_experiment_)
      break;
    ++parallel_experiments_;
  }

  if (parallel_experiments_ <= 0)
    throw std::runtime_error(
        "a circuit requires more memory than max_memory_mb.");
#ifdef AER_MPI
  parallel_experiments_ =
      std::min<int>({parallel_experiments_, max_experiments,
                     max_parallel_threads_, static_cast<int>(distributed_experiments_end_ - distributed_experiments_begin_)});
#else
  parallel_experiments_ =
      std::min<int>({parallel_experiments_, max_experiments,
                     max_parallel_threads_, static_cast<int>(circuits.size())});
#endif
}

void Controller::set_parallelization_circuit(const Circuit &circ,
                                             const Noise::NoiseModel &noise) {

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
    int circ_memory_mb = required_memory_mb(circ, noise) / num_process_per_experiment_;
    if (max_memory_mb_ + max_gpu_memory_mb_ < circ_memory_mb)
      throw std::runtime_error(
          "a circuit requires more memory than max_memory_mb.");
    // If circ memory is 0, set it to 1 so that we don't divide by zero
    circ_memory_mb = std::max<int>({1, circ_memory_mb});

#ifdef AER_MPI
    int shots = (circ.shots * (distributed_shots_rank_ + 1)/distributed_shots_) - (circ.shots * distributed_shots_rank_ /distributed_shots_);
#else
    int shots = circ.shots;
#endif
    parallel_shots_ =
        std::min<int>({static_cast<int>(max_memory_mb_ / circ_memory_mb),
                       max_shots, shots});
  }
  parallel_state_update_ =
      (parallel_shots_ > 1)
          ? std::max<int>({1, max_parallel_threads_ / parallel_shots_})
          : std::max<int>({1, max_parallel_threads_ / parallel_experiments_});
}

void Controller::set_distributed_parallelization(const std::vector<Circuit> &circuits,
                                  const std::vector<Noise::NoiseModel> &noise)
{
  std::vector<size_t> required_memory_mb_list(circuits.size());
  num_process_per_experiment_ = 1;
  for (size_t j = 0; j < circuits.size(); j++) {
    size_t size = required_memory_mb(circuits[j], noise[j]);
    if(size > max_memory_mb_ + max_gpu_memory_mb_){
      num_process_per_experiment_ = std::max<int>(num_process_per_experiment_,(size + (max_memory_mb_+max_gpu_memory_mb_) - 1) / (max_memory_mb_+max_gpu_memory_mb_));
    }
  }
  while((num_processes_ % num_process_per_experiment_) != 0){
    num_process_per_experiment_++;
  }

  distributed_experiments_ = num_processes_ / num_process_per_experiment_;

  if(circuits.size() < distributed_experiments_){
    // e.g. np = 8, circuits = 3, npe = 2,  de = 4 -> 3 , then np_in_group = [3,3,2]
    //      np = 4, circuits = 1, npe = 2,  de = 2 -> 1 , then np_in_group = [4]
    distributed_experiments_ = circuits.size();

    distributed_experiments_num_processes_ = (num_processes_ + distributed_experiments_ - 1)/distributed_experiments_;
    distributed_experiments_group_id_ = myrank_ / distributed_experiments_num_processes_;
    if((distributed_experiments_group_id_+1)*distributed_experiments_num_processes_ > num_processes_){
      distributed_experiments_num_processes_ = num_processes_ - distributed_experiments_group_id_*distributed_experiments_num_processes_;
    }

    if(distributed_experiments_num_processes_ > num_process_per_experiment_ && (distributed_experiments_num_processes_ % num_process_per_experiment_) == 0){
      distributed_shots_ = distributed_experiments_num_processes_ / num_process_per_experiment_;
      distributed_shots_rank_ = 0;
    }
    else{
      //shots are not distributed
      distributed_shots_ = 1;
      distributed_shots_rank_ = 0;
    }
    distributed_experiments_rank_ = myrank_ % distributed_experiments_;

    distributed_experiments_begin_ = distributed_experiments_group_id_;
    distributed_experiments_end_ = distributed_experiments_begin_ + 1;
  }
  else{
    distributed_experiments_group_id_ = myrank_ / num_process_per_experiment_;
    distributed_experiments_rank_ = myrank_ % num_process_per_experiment_;
    distributed_experiments_num_processes_ = num_process_per_experiment_;

    distributed_experiments_begin_ = circuits.size() * distributed_experiments_group_id_ / distributed_experiments_;
    distributed_experiments_end_ = circuits.size() * (distributed_experiments_group_id_ + 1) / distributed_experiments_;

    //shots are not distributed
    distributed_shots_ = 1;
    distributed_shots_rank_ = 0;
  }
}

uint_t Controller::get_distributed_num_processes(bool par_shots) const
{
  if(par_shots){
    return num_process_per_experiment_;
  }
  else{
    return distributed_experiments_num_processes_;    //no shot distribution, parallelize this experiment by processes in group
  }
}

bool Controller::multiple_chunk_required(const Circuit &circ,
                                const Noise::NoiseModel &noise) const
{
  if(circ.num_qubits < 3)
    return false;

  if(num_process_per_experiment_ > 1 || Controller::get_min_memory_mb() < required_memory_mb(circ, noise))
    return true;

  if(cache_block_qubit_ >= 2 && cache_block_qubit_ < circ.num_qubits)
    return true;

  return false;
}

size_t Controller::get_system_memory_mb() {
  size_t total_physical_memory = 0;
#if defined(__linux__) || defined(__APPLE__)
  auto pages = sysconf(_SC_PHYS_PAGES);
  auto page_size = sysconf(_SC_PAGE_SIZE);
  total_physical_memory = pages * page_size;
#elif defined(_WIN64)  || defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  total_physical_memory = status.ullTotalPhys;
#endif
#ifdef AER_MPI
  //get minimum memory size per process
  uint64_t locMem,minMem;
  locMem = total_physical_memory;
  MPI_Allreduce(&locMem,&minMem,1,MPI_UINT64_T,MPI_MIN,MPI_COMM_WORLD);
  total_physical_memory = minMem;
#endif

  return total_physical_memory >> 20;
}

size_t Controller::get_gpu_memory_mb() {
  size_t total_physical_memory = 0;
#ifdef AER_THRUST_CUDA
  int iDev,nDev,j;
  if(cudaGetDeviceCount(&nDev) != cudaSuccess){
    cudaGetLastError();
    nDev = 0;
  }
  for(iDev=0;iDev<nDev;iDev++){
    size_t freeMem,totalMem;
    cudaSetDevice(iDev);
    cudaMemGetInfo(&freeMem,&totalMem);
    total_physical_memory += totalMem;
  }
  num_gpus_ = nDev;
#endif
#ifdef AER_MPI
  //get minimum memory size per process
  uint64_t locMem,minMem;
  locMem = total_physical_memory;
  MPI_Allreduce(&locMem,&minMem,1,MPI_UINT64_T,MPI_MIN,MPI_COMM_WORLD);
  total_physical_memory = minMem;

  int t = num_gpus_;
  MPI_Allreduce(&t,&num_gpus_,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
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

  size_t required_mb = state.required_memory_mb(circ.num_qubits, circ.ops) / num_process_per_experiment_;
  if (max_memory_mb_+max_gpu_memory_mb_ < required_mb) {
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

void Controller::save_exception_to_results(Result &result,const std::exception &e)
{
  result.status = Result::Status::error;
  result.message = e.what();
  for(auto& res : result.results){
    res.status = ExperimentResult::Status::error;
    res.message = e.what();
  }
}

Transpile::CacheBlocking Controller::transpile_cache_blocking(const Circuit& circ,
                                     const Noise::NoiseModel& noise,
                                     const json_t& config,
                                     const size_t complex_size,bool is_matrix) const
{
  Transpile::CacheBlocking cache_block_pass;

  cache_block_pass.set_config(config);
  if(!cache_block_pass.enabled()){
    //if blocking is not set by config, automatically set if required
    if(multiple_chunk_required(circ,noise)){
      int nplace = num_process_per_experiment_;
      if(num_gpus_ > 0)
        nplace *= num_gpus_;
      cache_block_pass.set_blocking(circ.num_qubits, get_min_memory_mb() << 20, nplace, complex_size,is_matrix);
    }
  }

  return cache_block_pass;
}

//-------------------------------------------------------------------------
// Qobj execution
//-------------------------------------------------------------------------
template <typename inputdata_t>
Result Controller::execute(const inputdata_t& input_qobj)
{
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD,&num_processes_);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank_);
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
      // Load noise model (from json config)
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
    auto time_taken = std::chrono::duration<double>(timer_stop - timer_start).count();
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
                           const json_t &config) 
{
  // Start QOBJ timer
  auto timer_start = myclock_t::now();

  // Initialize Result object for the given number of experiments
  Result result(circuits.size());
  // Make a copy of the noise model for each circuit execution
  // so that it can be modified if required
  std::vector<Noise::NoiseModel> circ_noise_models(circuits.size(),noise_model);

  // Execute each circuit in a try block
  try {
    //truncate circuits before experiment settings (to get correct required_memory_mb value)
    if (truncate_qubits_) {
      for(size_t j = 0; j < circuits.size(); j++) {
        // Truncate unused qubits from circuit and noise model
        Transpile::TruncateQubits truncate_pass;
        truncate_pass.set_config(config);
        truncate_pass.optimize_circuit(circuits[j], circ_noise_models[j], circuits[j].opset(),
                                       result.results[j]);
      }
    }

#ifdef AER_MPI
    try{
      //catch exception raised by required_memory_mb because of invalid simulation method
      set_distributed_parallelization(circuits, circ_noise_models);
    }
    catch (std::exception &e) {
      save_exception_to_results(result,e);
    }

    const auto num_circuits = distributed_experiments_end_ - distributed_experiments_begin_;
    result.resize(num_circuits);
#endif

    //get max qubits for this process (to allocate qubit register at once)
    max_qubits_ = 0;
#ifdef AER_MPI
    for (size_t j = distributed_experiments_begin_; j < distributed_experiments_end_; j++) {
#else
    for (size_t j = 0; j < circuits.size(); j++) {
#endif
      if(circuits[j].num_qubits > max_qubits_){
        max_qubits_ = circuits[j].num_qubits;
      }
    }

    if (!explicit_parallelization_) {
      // set parallelization for experiments
      try{
        //catch exception raised by required_memory_mb because of invalid simulation method
        set_parallelization_experiments(circuits, circ_noise_models);
      }
      catch (std::exception &e) {
        save_exception_to_results(result,e);
      }
    }

#ifdef _OPENMP
    result.metadata.add(true, "omp_enabled");
#else
    result.metadata.add(false, "omp_enabled");
#endif
    result.metadata.add(parallel_experiments_, "parallel_experiments");
    result.metadata.add(max_memory_mb_, "max_memory_mb");
    result.metadata.add(max_gpu_memory_mb_,"max_gpu_memory_mb");

    //store rank and number of processes, if no distribution rank=0 procs=1 is set
    result.metadata.add(num_processes_,"num_mpi_processes");
    result.metadata.add(myrank_,"mpi_rank");
#ifdef AER_MPI
    result.metadata.add(distributed_experiments_,"distributed_experiments");
    result.metadata.add(distributed_experiments_group_id_,"distributed_experiments_group_id");
    result.metadata.add(distributed_experiments_rank_,"distributed_experiments_rank_in_group");
#endif

#ifdef _OPENMP
    // Check if circuit parallelism is nested with one of the others
    if (parallel_experiments_ > 1 && parallel_experiments_ < max_parallel_threads_) {
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
    uint_t offset = 0;
#ifdef AER_MPI
    offset = distributed_experiments_begin_;
#endif
    // then- and else-blocks have intentionally duplication.
    // Nested omp has significant overheads even though a guard condition exists.
    const int NUM_RESULTS = result.results.size();
    if (parallel_experiments_ > 1) {
      #pragma omp parallel for num_threads(parallel_experiments_)
      for (int j = 0; j < result.results.size(); ++j) {
        execute_circuit(circuits[j + offset], circ_noise_models[j + offset], config, result.results[j]);
      }
    } else {
      for (int j = 0; j < result.results.size(); ++j) {
        execute_circuit(circuits[j + offset], circ_noise_models[j + offset], config, result.results[j]);
      }
    }

    // Check each experiment result for completed status.
    // If only some experiments completed return partial completed status.

    bool all_failed = true;
    result.status = Result::Status::completed;
    for (int i = 0; i < NUM_RESULTS; ++i) {
      auto& experiment = result.results[i];
      if (experiment.status == ExperimentResult::Status::completed) {
        all_failed = false;
      } else {
        result.status = Result::Status::partial_completed;
        result.message += std::string(" [Experiment ") + std::to_string(i)
                          + std::string("] ") + experiment.message;
      }
    }
    if (all_failed) {
      result.status = Result::Status::error;
    }

    // Stop the timer and add total timing data
    auto timer_stop = myclock_t::now();
    auto time_taken = std::chrono::duration<double>(timer_stop - timer_start).count();
    result.metadata.add(time_taken, "time_taken");
  }
  // If execution failed return valid output reporting error
  catch (std::exception &e) {
    result.status = Result::Status::error;
    result.message = e.what();
  }
  return result;
}

void Controller::execute_circuit(Circuit &circ,
                                 Noise::NoiseModel &noise,
                                 const json_t &config,
                                 ExperimentResult &result) 
{
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
      truncate_pass.optimize_circuit(circ, noise, circ.opset(),
                                     result);
    }

    // set parallelization for this circuit
    if (!explicit_parallelization_) {
      set_parallelization_circuit(circ, noise);
    }

    int shots = circ.shots;
#ifdef AER_MPI
    if(parallel_shots_ > 1 && distributed_shots_ > 1){   //if shots can be distributed
      shots = (circ.shots * (distributed_shots_rank_ + 1)/distributed_shots_) - (circ.shots * distributed_shots_rank_ /distributed_shots_);
    }
#endif

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
#ifdef AER_MPI
    if(parallel_shots_ > 1 && distributed_shots_ > 1){
      result.metadata.add(distributed_shots_,"distributed_shots");
    }
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
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
