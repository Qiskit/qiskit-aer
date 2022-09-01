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

#ifndef _aer_base_state_hpp_
#define _aer_base_state_hpp_

#include "framework/json.hpp"
#include "framework/opset.hpp"
#include "framework/types.hpp"
#include "framework/creg.hpp"
#include "framework/results/experiment_result.hpp"

#include "noise/noise_model.hpp"

#include "simulators/registers.hpp"

namespace AER {
namespace Base {

using OpItr = std::vector<Operations::Op>::const_iterator;

//=========================================================================
// State interface base class for Qiskit-Aer
//=========================================================================

//TO DO : State will be moved to Simulator
template <class state_t>
class State {

public:
  using ignore_argument = void;
  using DataSubType = Operations::DataSubType;
  using OpType = Operations::OpType;

  //-----------------------------------------------------------------------
  // Constructors
  //-----------------------------------------------------------------------

  // The constructor arguments are used to initialize the OpSet
  // for the State class for checking supported simulator Operations
  //
  // Standard OpTypes that can be included here are:
  // - `OpType::gate` if gates are supported
  // - `OpType::measure` if measure is supported
  // - `OpType::reset` if reset is supported
  // - `OpType::snapshot` if any snapshots are supported
  // - `OpType::barrier` if barrier is supported
  // - `OpType::matrix` if arbitrary unitary matrices are supported
  // - `OpType::kraus` if general Kraus noise channels are supported
  //
  // For gate ops allowed gates are specified by a set of string names,
  // for example this could include {"u1", "u2", "u3", "U", "cx", "CX"}
  //
  // For snapshot ops allowed snapshots are specified by a set of string names,
  // For example this could include {"probabilities", "pauli_observable"}

  State(const Operations::OpSet &opset) : opset_(opset) 
  {
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

  State(const Operations::OpSet::optypeset_t &optypes,
        const stringset_t &gates,
        const stringset_t &snapshots)
    : State(Operations::OpSet(optypes, gates, snapshots))
  {
  }

  virtual ~State();

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

  // Return the state qreg object
  auto &qreg() { return state_.qreg(); }
  const auto &qreg() const { return state_.qreg(); }

  // Return the state creg object
  auto &creg() { return state_.creg(); }
  const auto &creg() const { return state_.creg(); }

  // Return the state opset object
  auto &opset() { return opset_; }
  const auto &opset() const { return opset_; }

  //=======================================================================
  // Subclass Override Methods
  //
  // The following methods should be implemented by any State subclasses.
  // Abstract methods are required, while some methods are optional for
  // State classes that support measurement to be compatible with a general
  // QasmController.
  //=======================================================================

  //-----------------------------------------------------------------------
  // Abstract methods
  //
  // The implementation of these methods must be defined in all subclasses
  //-----------------------------------------------------------------------
  
  // Return a string name for the State type
  virtual std::string name() const = 0;

  // Initializes the State to the default state.
  // Typically this is the n-qubit all |0> state
  void initialize_qreg(uint_t num_qubits);

  // Initializes the State to a specific state.
  void initialize_qreg(uint_t num_qubits, const state_t &state);

  // Return an estimate of the required memory for implementing the
  // specified sequence of operations on a `num_qubit` sized State.
  virtual size_t required_memory_mb(uint_t num_qubits,
                                    OpItr first,
                                    OpItr last)
                                    const = 0;

  //memory allocation (previously called before inisitalize_qreg)
  virtual bool allocate(uint_t num_qubits,uint_t block_bits,uint_t num_initial_states = 1);

  virtual bool allocate_state(RegistersBase& state, uint_t num_max_shots = 1){return true;}

  // Return the expectation value of a N-qubit Pauli operator
  // If the simulator does not support Pauli expectation value this should
  // raise an exception.
  virtual double expval_pauli(RegistersBase& state, const reg_t &qubits,
                              const std::string& pauli) = 0;

  //-----------------------------------------------------------------------
  // Optional: Load config settings
  //-----------------------------------------------------------------------

  // Load any settings for the State class from a config JSON
  virtual void set_config(const json_t &config);

  //-----------------------------------------------------------------------
  // Optional: Add information to metadata 
  //-----------------------------------------------------------------------

  // Every state can add information to the metadata structure
  virtual void add_metadata(ExperimentResult &result) const {
  }

  //-----------------------------------------------------------------------
  // Optional: measurement sampling
  //
  // This method is only required for a State subclass to be compatible with
  // the measurement sampling optimization of a general the QasmController
  //-----------------------------------------------------------------------

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state. Even though this method is not marked as const
  // at the end of sample the system should be left in the same state
  // as before sampling
  virtual std::vector<reg_t> sample_measure(const reg_t &qubits,
                                            uint_t shots,
                                            RngEngine &rng);

  //=======================================================================
  // Standard non-virtual methods
  //
  // These methods should not be modified in any State subclasses
  //=======================================================================

  //-----------------------------------------------------------------------
  // Apply circuits and ops
  //-----------------------------------------------------------------------

  // Apply a single operation
  // The `final_op` flag indicates no more instructions will be applied
  // to the state after this sequence, so the state can be modified at the
  // end of the instructions.
  virtual void apply_op(RegistersBase& state,
                        const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine& rng,
                        bool final_op = false) = 0;

  // Apply a sequence of operations to the current state of the State class.
  // It is up to the State subclass to decide how this sequence should be
  // executed (ie in sequence, or some other execution strategy.)
  // If this sequence contains operations not in the supported opset
  // an exeption will be thrown.
  // The `final_ops` flag indicates no more instructions will be applied
  // to the state after this sequence, so the state can be modified at the
  // end of the instructions.
  void apply_ops(OpItr first,
                 OpItr last,
                 ExperimentResult &result,
                 RngEngine &rng,
                 bool final_ops = false);

  //apply ops to multiple shots
  //this function should be separately defined since apply_ops is called in quantum_error
  template <typename InputIterator>
  void apply_ops_multi_shots(InputIterator first,
                 InputIterator last,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 uint_t rng_seed,
                 bool final_ops = false)
  {
    throw std::invalid_argument("apply_ops_multi_shots is not supported in State " + name());
  }

  //run multiple shots
  void run_shots(OpItr first,
                 OpItr last,
                 const json_t &config,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 const uint_t rng_seed,
                 const uint_t num_shots,
                 const bool can_sample = false);

  //-----------------------------------------------------------------------
  // ClassicalRegister methods
  //-----------------------------------------------------------------------

  // Initialize classical memory and register to default value (all-0)
  virtual void initialize_creg(uint_t num_memory, uint_t num_register);

  // Initialize classical memory and register to specific values
  virtual void initialize_creg(uint_t num_memory,
                       uint_t num_register,
                       const std::string &memory_hex,
                       const std::string &register_hex);

  //-----------------------------------------------------------------------
  // Save result data
  //-----------------------------------------------------------------------

  // Save current value of all classical registers to result
  // This supports DataSubTypes: c_accum (counts), list (memory)
  // TODO: Make classical data allow saving only subset of specified clbit values
  void save_creg(Registers<state_t>& state,ExperimentResult &result,
                 const std::string &key,
                 DataSubType subtype = DataSubType::c_accum) const;
              
  // Save single shot data type. Typically this will be the value for the
  // last shot of the simulation
  template <class T>
  void save_data_single(ExperimentResult &result,
                        const std::string &key, const T& datum, OpType type) const;

  template <class T>
  void save_data_single(ExperimentResult &result,
                        const std::string &key, T&& datum, OpType type) const;

  // Save data type which can be averaged over all shots.
  // This supports DataSubTypes: list, c_list, accum, c_accum, average, c_average
  template <class T>
  void save_data_average(Registers<state_t>& state, ExperimentResult &result,
                         const std::string &key, const T& datum, OpType type,
                         DataSubType subtype = DataSubType::average) const;

  template <class T>
  void save_data_average(Registers<state_t>& state, ExperimentResult &result,
                         const std::string &key, T&& datum, OpType type,
                         DataSubType subtype = DataSubType::average) const;
  
  // Save data type which is pershot and does not support accumulator or average
  // This supports DataSubTypes: single, c_single, list, c_list
  template <class T>
  void save_data_pershot(Registers<state_t>& state, ExperimentResult &result,
                         const std::string &key, const T& datum, OpType type,
                         DataSubType subtype = DataSubType::list) const;

  template <class T>
  void save_data_pershot(Registers<state_t>& state, ExperimentResult &result,
                         const std::string &key, T&& datum, OpType type,
                         DataSubType subtype = DataSubType::list) const;


  //save creg as count data 
  virtual void save_count_data(ExperimentResult& result,bool save_memory);

  //-----------------------------------------------------------------------
  // Common instructions
  //-----------------------------------------------------------------------
 
  // Apply a save expectation value instruction
  void apply_save_expval(Registers<state_t>& state, const Operations::Op &op, ExperimentResult &result);

  //-----------------------------------------------------------------------
  // Standard snapshots
  //-----------------------------------------------------------------------

  // Snapshot the current statevector (single-shot)
  // if type_label is the empty string the operation type will be used for the type
  virtual void snapshot_state(Registers<state_t>& state, const Operations::Op &op, ExperimentResult &result,
                      std::string name = "") const;

  // Snapshot the classical memory bits state (single-shot)
  void snapshot_creg_memory(Registers<state_t>& state, const Operations::Op &op, ExperimentResult &result,
                            std::string name = "memory") const;

  // Snapshot the classical register bits state (single-shot)
  void snapshot_creg_register(Registers<state_t>& state, const Operations::Op &op, ExperimentResult &result,
                              std::string name = "register") const;


  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Sets the number of threads available to the State implementation
  // If negative there is no restriction on the backend
  virtual inline void set_parallelization(int n) {threads_ = n;}

  virtual void set_parallel_shots(int shots)
  {
    parallel_shots_ = shots;
  }

  // Set a complex global phase value exp(1j * theta) for the state
  void set_global_phase(double theta);

  // Set a complex global phase value exp(1j * theta) for the state
  void add_global_phase(double theta);

  //set number of processes to be distributed
  virtual void set_distribution(uint_t nprocs);

  //set maximum number of qubits for matrix multiplication
  virtual void set_max_matrix_qubits(int_t bits)
  {
    max_matrix_qubits_ = bits;
  }

  //set max number of shots to execute in a batch (used in StateChunk class)
  virtual void set_max_bached_shots(uint_t shots){}

  //Does this state support multi-chunk distribution?
  virtual bool multi_chunk_distribution_supported(void){return false;}
  //Does this state support multi-shot parallelization?
  virtual bool multi_shot_parallelization_supported(void){return false;}
  //Does this state support runtime noise sampling?
  virtual bool runtime_noise_sampling_supported(void){return false;}

  void enable_shot_branching(bool flg)
  {
    enable_shot_branching_ = flg;
  }
  void enable_batch_execution(bool flg)
  {
    enable_batch_execution_ = flg;
  }

protected:

  // Initializes the State to the default state.
  // Typically this is the n-qubit all |0> state
  virtual void initialize_state(RegistersBase& state, uint_t num_qubits){}

  // Initializes the State to a specific state.
  virtual void initialize_state(RegistersBase& state, uint_t num_qubits, const state_t &src_state){}

  // Initialize classical memory and register to default value (all-0)
  virtual void initialize_cregister(RegistersBase& state, uint_t num_memory, uint_t num_register);

  // Initialize classical memory and register to specific values
  virtual void initialize_cregister(RegistersBase& state, 
                       uint_t num_memory,
                       uint_t num_register,
                       const std::string &memory_hex,
                       const std::string &register_hex);

  virtual void initialize_cregister(RegistersBase& state, const ClassicalRegister& creg);

  // Load any settings for the State class from a config JSON
  virtual void set_state_config(RegistersBase& state, const json_t &config){}

  virtual void apply_ops_state(RegistersBase& state,
                 OpItr first,
                 OpItr last,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 RngEngine &rng,
                 const bool final_ops = false);

  void run_shots_with_branching(OpItr first,
                 OpItr last,
                 const json_t &config,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 const uint_t rng_seed,
                 const uint_t num_shots,
                 const bool can_sample = false);

  virtual void run_shots_with_bathed_execution((RegistersBase& state,
                 OpItr first,
                 OpItr last,
                 const json_t &config,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 const uint_t rng_seed,
                 const uint_t num_shots);

  //runtime noise sampling for shot branching
  void apply_runtime_noise_sampling(RegistersBase& state, const Operations::Op &op, const Noise::NoiseModel &noise);


  // The quantum state and Classical register data structure for single shot execution
  Registers<state_t> state_;

  // Opset of instructions supported by the state
  Operations::OpSet opset_;

  // Maximum threads which may be used by the backend for OpenMP multithreading
  // Default value is single-threaded unless overridden
  int threads_ = 1;

  // Set a global phase exp(1j * theta) for the state
  bool has_global_phase_ = false;
  complex_t global_phase_ = 1;

  int_t max_matrix_qubits_ = 0;

  std::string sim_device_name_ = "CPU";

  // Save counts as memory list
  bool save_creg_memory_ = false;

  //OMP parallel shots 
  int parallel_shots_ = 1;

  //max allocatable shots
  uint_t num_max_shots_ = 1;

  //number of places for shot distribution
  uint_t num_distributed_places_ = 1;

  //MPI settings
  uint_t myrank_;               //process ID
  uint_t nprocs_;               //number of processes
  uint_t distributed_rank_;     //process ID in communicator group
  uint_t distributed_procs_;    //number of processes in communicator group
  uint_t distributed_group_;    //group id of distribution
  int_t distributed_proc_bits_; //distributed_procs_=2^distributed_proc_bits_  (if nprocs != power of 2, set -1)
#ifdef AER_MPI
  //communicator group to simulate a circuit (for multi-experiments)
  MPI_Comm distributed_comm_;
#endif

  //number of qubits for the circuit
  uint_t num_qubits_;

  //creg initialization
  uint_t num_creg_memory_;
  uint_t num_creg_registers_;

  //shot branching
  bool enable_shot_branching_ = false;
  bool enable_batch_execution_ = false;  //apply the same op to multiple states, if enable_shot_branching_ is false this flag is used for batched execution on GPU

  bool runtime_noise_sampled_ = false;  //true when runtime noise sampling is done

  virtual bool shot_branching_supported(void)
  {
    return false;   //return true if simulation method supports
  }

  uint_t get_max_allocatable_shots(const uint_t num_qubits, OpItr first, OpItr last);

};


//=========================================================================
// Implementations
//=========================================================================
template <class state_t>
State<state_t>::~State(void)
{

#ifdef AER_MPI
  if(distributed_comm_ != MPI_COMM_WORLD){
    MPI_Comm_free(&distributed_comm_);
  }
#endif
}

template <class state_t>
void State<state_t>::set_config(const json_t &config) 
{
  JSON::get_value(sim_device_name_, "device", config);

  // Load config for memory (creg list data)
  JSON::get_value(save_creg_memory_, "memory", config);

  set_state_config(state_, config);
}

template <class state_t>
uint_t State<state_t>::get_max_allocatable_shots(const uint_t num_qubits,  OpItr first, OpItr last)
{
  state_t t;
  uint_t size_per_shot_mb = required_memory_mb(num_qubits, first, last);
  uint_t free_mem_mb = 0;

  if(size_per_shot_mb == 0)
    size_per_shot_mb = 1;

  if(sim_device_name_ == "GPU"){
#ifdef AER_THRUST_CUDA
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      nDev = 0;
    }
    
    for(int iDev=0;iDev<nDev;iDev++){
      size_t freeMem, totalMem;
      cudaSetDevice(iDev);
      cudaMemGetInfo(&freeMem, &totalMem);
      free_mem_mb += freeMem;
    }
    free_mem_mb >>= 20;
#endif
  }
  else{
    free_mem_mb = Utils::get_free_memory_mb();
  }

  free_mem_mb = free_mem_mb*8/10;
  if(free_mem_mb < size_per_shot_mb)
    return 1;
  return free_mem_mb/size_per_shot_mb;
}

template <class state_t>
void State<state_t>::set_distribution(uint_t nprocs)
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

#ifdef AER_THRUST_CUDA
  int nDev;
  if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
    cudaGetLastError();
    nDev = 0;
  }
  num_distributed_places_ = nDev;
#else
  num_distributed_places_ = 1;
#endif
}

template <class state_t>
bool State<state_t>::allocate(uint_t num_qubits,uint_t block_bits,uint_t num_initial_states)
{
  num_qubits_ = num_qubits;

  state_.allocate(num_initial_states);
  return true;
}

template <class state_t>
void State<state_t>::set_global_phase(double theta) {
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
void State<state_t>::add_global_phase(double theta) {
  if (Linalg::almost_equal(theta, 0.0)) 
    return;
  
  has_global_phase_ = true;
  global_phase_ *= std::exp(complex_t(0.0, theta));
}

template <class state_t>
void State<state_t>::initialize_qreg(uint_t num_qubits)
{
  initialize_state(state_, num_qubits);
}

template <class state_t>
void State<state_t>::initialize_qreg(uint_t num_qubits, const state_t &state)
{
  initialize_state(state_, num_qubits, state);
}

template <class state_t>
void State<state_t>::apply_ops(OpItr first, OpItr last,
                               ExperimentResult &result,
                               RngEngine &rng,
                               bool final_ops) 
{
  apply_op_state(state_, first, last, result, rng, final_ops);
}

template <class state_t>
void State<state_t>::apply_ops_state(RegistersBase& state_in,
                               OpItr first, OpItr last,
                               const Noise::NoiseModel &noise,
                               ExperimentResult &result,
                               RngEngine &rng,
                               const bool final_ops) 
{
  Registers<state_t>& state = dynamic_cast<Registers<state_t>&>(state_in);

  // Simple loop over vector of input operations
  for (auto it = first; it != last; ++it) {
    switch (it->type) {
      case Operations::OpType::mark: {
        state.marks()[it->string_params[0]] = it;
        break;
      }
      case Operations::OpType::jump: {
        if (state.creg().check_conditional(*it)) {
          const auto& mark_name = it->string_params[0];
          auto mark_it = state.marks().find(mark_name);
          if (mark_it != state.marks().end()) {
            it = mark_it->second;
          } else {
            for (++it; it != last; ++it) {
              if (it->type == Operations::OpType::mark) {
                state.marks()[it->string_params[0]] = it;
                if (it->string_params[0] == mark_name) {
                  break;
                }
              }
            }
            if (it == last) {
              std::stringstream msg;
              msg << "Invalid jump destination:\"" << mark_name << "\"." << std::endl;
              throw std::runtime_error(msg.str());
            }
          }
        }
        break;
      }
      case Operations::OpType::sample_noise: {
        //runtime noise sampling
        apply_runtime_noise_sampling(state, *it, noise);
        state.next_iter() = it + 1;
        return;
      }
      default: {
        apply_op(state, *it, result, rng, final_ops && (it + 1 == last));
        if(enable_shot_branching_ && state.num_branch() > 0){
          state.next_iter() = it + 1;
          return;
        }
      }
    }
  }
  state.next_iter() = last;
}

template <class state_t>
void State<state_t>::run_shots(OpItr first,
               OpItr last,
               const json_t &config,
               const Noise::NoiseModel &noise,
               ExperimentResult &result,
               const uint_t rng_seed,
               const uint_t num_shots,
               const bool can_sample)
{
  num_max_shots_ = get_max_allocatable_shots(num_qubits_, first, last);

  enable_shot_branching_ &= shot_branching_supported();
  if(enable_shot_branching_ && num_max_shots_ > 1){
    return run_shots_with_branching(first, last, config, noise, result, rng_seed, num_shots, can_sample);
  }
  else if(enable_batch_execution_){
    return run_shots_with_batched_execution(first, last, config, noise, result, rng_seed, num_shots);
  }

  bool batch_shots_tmp = enable_batch_execution_; //save this option and disable for single shot execution
  enable_batch_execution_ = false;

  int_t par_shots = parallel_shots_;
  if(par_shots > num_max_shots_)
    par_shots = num_max_shots_;

  std::vector<ExperimentResult> par_results(par_shots);
  std::vector<Registers<state_t>> states(par_shots);

  for(int_t i=0;i<par_shots;i++){
    states[i].allocate(1);  //allocate single chunk

    // allocate qubit register
    allocate_state(states[i], 1);
  }

  auto run_single_shot = [this,&par_results,config,num_shots,par_shots,rng_seed, first, last, &states, &noise](int_t i){
    uint_t i_shot,shot_end;
    i_shot = num_shots*i/par_shots;
    shot_end = num_shots*(i+1)/par_shots;

    // Set state config
    set_state_config(states[i], config);

    for(;i_shot<shot_end;i_shot++){
      RngEngine rng;
      rng.set_seed(rng_seed + i_shot);

      initialize_state(states[i], num_qubits_);
      initialize_cregister(states[i], num_creg_memory_, num_creg_registers_);

      apply_ops_state(states[i], first,last,noise, par_results[i], rng, true);

      if (states[i].creg().memory_size() > 0) {
        std::string memory_hex = states[i].creg().memory_hex();
        par_results[i].data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
        if (save_creg_memory_) {
          par_results[i].data.add_list(std::move(memory_hex), "memory");
        }
      }
    }
  };
  Utils::apply_omp_parallel_for((par_shots > 1),0,par_shots,run_single_shot);

  for (auto &res : par_results) {
    result.combine(std::move(res));
  }
  add_metadata(result);

  enable_batch_execution_ = batch_shots_tmp;

  result.metadata.add(false, "shot_blanching_enabled");
  result.metadata.add(false, "runtime_noise_sampling_enabled");
}

template <class state_t>
void State<state_t>::run_shots_with_branching(OpItr first,
                 OpItr last,
                 const json_t &config,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 const uint_t rng_seed,
                 const uint_t num_shots,
                 const bool can_sample)
{
  RngEngine rng;
  rng.set_seed(rng_seed);   //this is not used actually

  std::vector<RngEngine> reserved_shots(num_shots);
  for(int_t i=0;i<num_shots;i++){
    reserved_shots[i].set_seed(rng_seed + i);
  }

  while(reserved_shots.size() > 0){
    std::vector<ExperimentResult> par_results(parallel_shots_);

    std::vector<std::shared_ptr<Registers<state_t>>> states;
    std::shared_ptr<Registers<state_t>> initial_state = std::make_shared<Registers<state_t>>();

    allocate_state(*initial_state, std::min(reserved_shots.size(),num_max_shots_) );
    initial_state->set_shots(reserved_shots);
    reserved_shots.clear();

    set_state_config(*initial_state, config);
    initialize_state(*initial_state, num_qubits_);
    initialize_cregister(*initial_state, num_creg_memory_, num_creg_registers_);

    states.push_back(initial_state);

    //functor for ops execution
    auto apply_ops_func = [this, &states, &rng, &noise, &par_results, last](int_t i)
    {
      int_t ires = omp_get_thread_num() % par_results.size();
      apply_ops_state(*states[i], states[i]->next_iter(),last,noise,par_results[ires], rng, true);

      if(states[i]->num_branch() > 0)  //check if there are new branches
        return 1;
      return 0;
    };

    initial_state->next_iter() = first;
    uint_t nactive = 1;
    while(nactive > 0){   //loop until all states execute all ops
      uint_t nbranch = 0;

      //apply ops until a branch operation comes (reset, measure, kraus, initialize, noises)
      nbranch = Utils::apply_omp_parallel_for_reduction_int((parallel_shots_ > 1 && states.size() > 1), 0, states.size(), apply_ops_func, parallel_shots_);

      while(nbranch > 0){
        uint_t num_states_prev = states.size();

        for(int_t i=0;i<num_states_prev;i++){
          if(states[i]->num_branch() > 0){
            int_t istart = 1;
            if(states[i]->branch(0).shots_.size() == 0){   //if first state has no shots after branch, copy other shots to the first
              for(int_t j=1;j<states[i]->num_branch();j++){
                if(states[i]->branch(j).shots_.size() > 0){
                  states[i]->set_shots(states[i]->branch(j).shots_);
                  states[i]->additional_ops() = states[i]->branch(j).additional_ops_;
                  initialize_cregister(*states[i], states[i]->branch(j).creg_);
                  istart = j+1;
                  break;
                }
              }
            }
            else{ //otherwise set branched shots 
              states[i]->set_shots(states[i]->branch(0).shots_);
              states[i]->additional_ops() = states[i]->branch(0).additional_ops_;
              initialize_cregister(*states[i], states[i]->branch(0).creg_);
            }
            for(int_t j=istart;j<states[i]->num_branch();j++){
              if(states[i]->branch(j).shots_.size() > 0){  //copy state and set branched shots
                uint_t pos = states.size();
                if(pos >= num_max_shots_){  //if there is not enough memory to allocate copied state, shots are reserved to the next iteration
                  //reset seed to reproduce same results
                  for(int_t k=0;k<states[i]->branch(j).shots_.size();k++){
                    states[i]->branch(j).shots_[k].set_seed(states[i]->branch(j).shots_[k].initial_seed());
                  }
                  reserved_shots.insert(reserved_shots.end(), states[i]->branch(j).shots_.begin(), states[i]->branch(j).shots_.end());
                }
                else{
                  states.push_back(std::make_shared<Registers<state_t>>(*states[i]));
                  states[pos]->set_shots(states[i]->branch(j).shots_);
                  states[pos]->additional_ops() = states[i]->branch(j).additional_ops_;
                  initialize_cregister(*states[pos], states[i]->branch(j).creg_);
                }
              }
            }
          }
        }

        //then execute ops applied after branching (reset, Kraus, noises, etc.)
        auto apply_additional_ops_func = [this, &states, &rng, &noise, &par_results](int_t i)
        {
          int_t ires = omp_get_thread_num() % par_results.size();
          int ret = 0;
          states[i]->clear_branch();
          for(int_t j=0;j<states[i]->additional_ops().size();j++){
            apply_op(*states[i], states[i]->additional_ops()[j], par_results[ires], rng, false );

            if(states[i]->num_branch() > 0){  //check if there are new branches
              //if there are additional ops remaining, queue them on new branches
              for(int_t k=j+1;k<states[i]->additional_ops().size();k++){
                for(int_t l=0;l<states[i]->num_branch();l++)
                  states[i]->add_op_after_branch(l,states[i]->additional_ops()[k]);
              }
              ret = 1;
              break;
            }
          }
          states[i]->clear_additional_ops();
          return ret;
        };
        nbranch = Utils::apply_omp_parallel_for_reduction_int((parallel_shots_ > 1), 0, states.size(), apply_additional_ops_func, parallel_shots_);
      }

      nactive = 0;
      for(int_t i=0;i<states.size();i++){
        if(states[i]->next_iter() != last)
          nactive++;
      }
    }

    for (auto &res : par_results) {
      result.combine(std::move(res));
    }

    //TO DO: gather cregs among MPI processes here for MPI shots distribution
    for(int_t i=0;i<states.size();i++){
      if(states[i]->creg().memory_size() > 0) {
        std::string memory_hex = states[i]->creg().memory_hex();
        for(int_t j=0;j<states[i]->num_shots();j++){
          result.data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
          if (save_creg_memory_) {
            result.data.add_list(memory_hex, "memory");
          }
        }
      }
      states[i].reset();
    }
    states.clear();
    initial_state.reset();
  }

  result.metadata.add(true, "shot_blanching_enabled");
  result.metadata.add(runtime_noise_sampled_, "runtime_noise_sampling_enabled");
}

template <class state_t>
void State<state_t>::run_shots_with_batched_execution(OpItr first,
                 OpItr last,
                 const json_t &config,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 const uint_t rng_seed,
                 const uint_t num_shots)
{
  RngEngine rng;
  rng.set_seed(rng_seed);   //this is not used actually

  std::vector<RngEngine> reserved_shots(num_shots);
  for(int_t i=0;i<num_shots;i++){
    reserved_shots[i].set_seed(rng_seed + i);
  }

  while(reserved_shots.size() > 0){
    std::vector<ExperimentResult> par_results(parallel_shots_);

    std::vector<std::shared_ptr<Registers<state_t>>> states;
    std::shared_ptr<Registers<state_t>> initial_state = std::make_shared<Registers<state_t>>();

    allocate_state(*initial_state, std::min(reserved_shots.size(),num_max_shots_) );
    initial_state->set_shots(reserved_shots);
    reserved_shots.clear();

    set_state_config(*initial_state, config);
    initialize_state(*initial_state, num_qubits_);
    initialize_cregister(*initial_state, num_creg_memory_, num_creg_registers_);

    states.push_back(initial_state);

    //functor for ops execution
    auto apply_ops_func = [this, &states, &rng, &noise, &par_results, last](int_t i)
    {
      int_t ires = omp_get_thread_num() % par_results.size();
      apply_ops_state(*states[i], states[i]->next_iter(),last,noise,par_results[ires], rng, true);

      if(states[i]->num_branch() > 0)  //check if there are new branches
        return 1;
      return 0;
    };

    initial_state->next_iter() = first;
    uint_t nactive = 1;
    while(nactive > 0){   //loop until all states execute all ops
      uint_t nbranch = 0;

      //apply ops until a branch operation comes (reset, measure, kraus, initialize, noises)
      nbranch = Utils::apply_omp_parallel_for_reduction_int((parallel_shots_ > 1 && states.size() > 1), 0, states.size(), apply_ops_func, parallel_shots_);

      while(nbranch > 0){
        uint_t num_states_prev = states.size();

        for(int_t i=0;i<num_states_prev;i++){
          if(states[i]->num_branch() > 0){
            int_t istart = 1;
            if(states[i]->branch(0).shots_.size() == 0){   //if first state has no shots after branch, copy other shots to the first
              for(int_t j=1;j<states[i]->num_branch();j++){
                if(states[i]->branch(j).shots_.size() > 0){
                  states[i]->set_shots(states[i]->branch(j).shots_);
                  states[i]->additional_ops() = states[i]->branch(j).additional_ops_;
                  initialize_cregister(*states[i], states[i]->branch(j).creg_);
                  istart = j+1;
                  break;
                }
              }
            }
            else{ //otherwise set branched shots 
              states[i]->set_shots(states[i]->branch(0).shots_);
              states[i]->additional_ops() = states[i]->branch(0).additional_ops_;
              initialize_cregister(*states[i], states[i]->branch(0).creg_);
            }
            for(int_t j=istart;j<states[i]->num_branch();j++){
              if(states[i]->branch(j).shots_.size() > 0){  //copy state and set branched shots
                uint_t pos = states.size();
                if(pos >= num_max_shots_){  //if there is not enough memory to allocate copied state, shots are reserved to the next iteration
                  //reset seed to reproduce same results
                  for(int_t k=0;k<states[i]->branch(j).shots_.size();k++){
                    states[i]->branch(j).shots_[k].set_seed(states[i]->branch(j).shots_[k].initial_seed());
                  }
                  reserved_shots.insert(reserved_shots.end(), states[i]->branch(j).shots_.begin(), states[i]->branch(j).shots_.end());
                }
                else{
                  states.push_back(std::make_shared<Registers<state_t>>(*states[i]));
                  states[pos]->set_shots(states[i]->branch(j).shots_);
                  states[pos]->additional_ops() = states[i]->branch(j).additional_ops_;
                  initialize_cregister(*states[pos], states[i]->branch(j).creg_);
                }
              }
            }
          }
        }

        //then execute ops applied after branching (reset, Kraus, noises, etc.)
        auto apply_additional_ops_func = [this, &states, &rng, &noise, &par_results](int_t i)
        {
          int_t ires = omp_get_thread_num() % par_results.size();
          int ret = 0;
          states[i]->clear_branch();
          for(int_t j=0;j<states[i]->additional_ops().size();j++){
            apply_op(*states[i], states[i]->additional_ops()[j], par_results[ires], rng, false );

            if(states[i]->num_branch() > 0){  //check if there are new branches
              //if there are additional ops remaining, queue them on new branches
              for(int_t k=j+1;k<states[i]->additional_ops().size();k++){
                for(int_t l=0;l<states[i]->num_branch();l++)
                  states[i]->add_op_after_branch(l,states[i]->additional_ops()[k]);
              }
              ret = 1;
              break;
            }
          }
          states[i]->clear_additional_ops();
          return ret;
        };
        nbranch = Utils::apply_omp_parallel_for_reduction_int((parallel_shots_ > 1), 0, states.size(), apply_additional_ops_func, parallel_shots_);
      }

      nactive = 0;
      for(int_t i=0;i<states.size();i++){
        if(states[i]->next_iter() != last)
          nactive++;
      }
    }

    for (auto &res : par_results) {
      result.combine(std::move(res));
    }

    //TO DO: gather cregs among MPI processes here for MPI shots distribution
    for(int_t i=0;i<states.size();i++){
      if(states[i]->creg().memory_size() > 0) {
        std::string memory_hex = states[i]->creg().memory_hex();
        for(int_t j=0;j<states[i]->num_shots();j++){
          result.data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
          if (save_creg_memory_) {
            result.data.add_list(memory_hex, "memory");
          }
        }
      }
      states[i].reset();
    }
    states.clear();
    initial_state.reset();
  }

  result.metadata.add(true, "shot_blanching_enabled");
  result.metadata.add(runtime_noise_sampled_, "runtime_noise_sampling_enabled");
}


template <class state_t>
void State<state_t>::apply_runtime_noise_sampling(RegistersBase& state_in, const Operations::Op &op, const Noise::NoiseModel &noise)
{
  Registers<state_t>& state = dynamic_cast<Registers<state_t>&>(state_in);
  uint_t nshots = state.num_shots();
  reg_t shot_map(nshots);
  std::vector<std::vector<Operations::Op>> noises;

  for(int_t i=0;i<nshots;i++){
    std::vector<Operations::Op> noise_ops = noise.sample_noise_loc(op, state.rng_shots(i));

    //search same noise ops 
    int_t pos = -1;
    for(int_t j=0;j<noises.size();j++){
      if(noise_ops.size() != noises[j].size())
        continue;
      bool same = true;
      for(int_t k=0;k<noise_ops.size();k++){
        if(noise_ops[k].type != noises[j][k].type || noise_ops[k].name != noises[j][k].name)
          same = false;
        else if(noise_ops[k].qubits.size() != noises[j][k].qubits.size())
          same = false;
        else{
          for(int_t l=0;l<noise_ops[k].qubits.size();l++){
            if(noise_ops[k].qubits[l] != noises[j][k].qubits[l]){
              same = false;
              break;
            }
          }
        }
        if(!same)
          break;
        if(noise_ops[k].type == OpType::gate){
          if(noise_ops[k].name == "pauli"){
            if(noise_ops[k].string_params[0] != noises[j][k].string_params[0])
              same = false;
          }
          else if(noise_ops[k].params.size() != noises[j][k].params.size())
            same = false;
          else{
            for(int_t l=0;l<noise_ops[k].params.size();l++){
              if(noise_ops[k].params[l] != noises[j][k].params[l]){
                same = false;
                break;
              }
            }
          }
        }
        else if(noise_ops[k].type == OpType::matrix || noise_ops[k].type == OpType::diagonal_matrix){
          if(noise_ops[k].mats.size() != noises[j][k].mats.size())
            same = false;
          else{
            for(int_t l=0;l<noise_ops[k].mats.size();l++){
              if(noise_ops[k].mats[l].size() != noises[j][k].mats[l].size()){
                same = false;
                break;
              }
              for(int_t m=0;m<noise_ops[k].mats[l].size();m++){
                if(noise_ops[k].mats[l][m] != noises[j][k].mats[l][m]){
                  same = false;
                  break;
                }
              }
              if(!same)
                break;
            }
          }
        }
        if(!same)
          break;
      }
      if(same){
        pos = j;
        break;
      }
    }

    if(pos < 0){  //if not found, add noise ops to the list
      shot_map[i] = noises.size();
      noises.push_back(noise_ops);
    }
    else{   //if found, add shot
      shot_map[i] = pos;
    }
  }

  state.branch_shots(shot_map, noises.size());
  for(int_t i=0;i<noises.size();i++){
    state.copy_ops_after_branch(i,noises[i]);
  }
  runtime_noise_sampled_ = true;
}

template <class state_t>
std::vector<reg_t> State<state_t>::sample_measure(
                                                  const reg_t &qubits,
                                                  uint_t shots,
                                                  RngEngine &rng) {
  (ignore_argument)qubits;
  (ignore_argument)shots;
  return std::vector<reg_t>();
}


template <class state_t>
void State<state_t>::initialize_creg(uint_t num_memory, uint_t num_register) 
{
  num_creg_memory_ = num_memory;
  num_creg_registers_ = num_register;

  initialize_cregister(state_, num_memory, num_register);
}


template <class state_t>
void State<state_t>::initialize_creg(uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) 
{
  num_creg_memory_ = num_memory;
  num_creg_registers_ = num_register;

  initialize_cregister(state_, num_memory, num_register, memory_hex, register_hex);
}

template <class state_t>
void State<state_t>::initialize_cregister(RegistersBase& state, uint_t num_memory, uint_t num_register) 
{
  num_creg_memory_ = num_memory;
  num_creg_registers_ = num_register;

  state.creg().initialize(num_memory, num_register);
}


template <class state_t>
void State<state_t>::initialize_cregister(RegistersBase& state, 
                                     uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) 
{
  num_creg_memory_ = num_memory;
  num_creg_registers_ = num_register;

  state.creg().initialize(num_memory, num_register, memory_hex, register_hex);
}

template <class state_t>
void State<state_t>::initialize_cregister(RegistersBase& state, const ClassicalRegister& creg)
{
  state.creg() = creg;
}

template <class state_t>
void State<state_t>::save_creg(Registers<state_t>& state,ExperimentResult &result,
                               const std::string &key,
                               DataSubType subtype) const {
  if (state.creg().memory_size() == 0)
    return;
  switch (subtype) {
    case DataSubType::list:
      result.data.add_list(state.creg().memory_hex(), key);
      result.metadata.add("creg", "result_types", key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(1ULL, key, state.creg().memory_hex());
      result.metadata.add("creg", "result_types", key);
      break;
    default:
      throw std::runtime_error("Invalid creg data subtype for data key: " + key);
  }
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void State<state_t>::save_data_average(Registers<state_t>& state,
                                       ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum, OpType type,
                                       DataSubType subtype) const 
{
  switch (subtype) {
    case DataSubType::list:
      result.data.add_list(datum, key);
      break;
    case DataSubType::c_list:
      result.data.add_list(datum, key, state.creg().memory_hex());
      break;
    case DataSubType::accum:
      result.data.add_accum(datum, key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(datum, key, state.creg().memory_hex());
      break;
    case DataSubType::average:
      result.data.add_average(datum, key);
      break;
    case DataSubType::c_average:
      result.data.add_average(datum, key, state.creg().memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid average data subtype for data key: " + key);
  }
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void State<state_t>::save_data_average(Registers<state_t>& state,
                                       ExperimentResult &result,
                                       const std::string &key,
                                       T&& datum, OpType type,
                                       DataSubType subtype) const 
{
  switch (subtype) {
    case DataSubType::list:
      result.data.add_list(std::move(datum), key);
      break;
    case DataSubType::c_list:
      result.data.add_list(std::move(datum), key, state.creg().memory_hex());
      break;
    case DataSubType::accum:
      result.data.add_accum(std::move(datum), key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(std::move(datum), key, state.creg().memory_hex());
      break;
    case DataSubType::average:
      result.data.add_average(std::move(datum), key);
      break;
    case DataSubType::c_average:
      result.data.add_average(std::move(datum), key, state.creg().memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid average data subtype for data key: " + key);
  }
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void State<state_t>::save_data_pershot(Registers<state_t>& state,
                                       ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum, OpType type,
                                       DataSubType subtype) const 
{
  uint_t nshots = state.num_shots();

  for(int_t i=0;i<nshots;i++){
    switch (subtype) {
    case DataSubType::single:
      result.data.add_single(datum, key);
      break;
    case DataSubType::c_single:
      result.data.add_single(datum, key, state.creg().memory_hex());
      break;
    case DataSubType::list:
      result.data.add_list(datum, key);
      break;
    case DataSubType::c_list:
      result.data.add_list(datum, key, state.creg().memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid pershot data subtype for data key: " + key);
    }
    result.metadata.add(type, "result_types", key);
    result.metadata.add(subtype, "result_subtypes", key);
  }
}

template <class state_t>
template <class T>
void State<state_t>::save_data_pershot(Registers<state_t>& state,
                                       ExperimentResult &result, 
                                       const std::string &key,
                                       T&& datum, OpType type,
                                       DataSubType subtype) const 
{
  uint_t nshots = state.num_shots();

  for(int_t i=0;i<nshots;i++){
    switch (subtype) {
      case DataSubType::single:
        if(i == nshots-1)
          result.data.add_single(std::move(datum), key);
        else
          result.data.add_single(datum, key);
        break;
      case DataSubType::c_single:
        if(i == nshots-1)
          result.data.add_single(std::move(datum), key, state.creg().memory_hex());
        else
          result.data.add_single(datum, key, state.creg().memory_hex());
        break;
      case DataSubType::list:
        if(i == nshots-1)
          result.data.add_list(std::move(datum), key);
        else
          result.data.add_list(datum, key);
        break;
      case DataSubType::c_list:
        if(i == nshots-1)
          result.data.add_list(std::move(datum), key, state.creg().memory_hex());
        else
          result.data.add_list(datum, key, state.creg().memory_hex());
        break;
      default:
        throw std::runtime_error("Invalid pershot data subtype for data key: " + key);
    }
    result.metadata.add(type, "result_types", key);
    result.metadata.add(subtype, "result_subtypes", key);
  }
}

template <class state_t>
template <class T>
void State<state_t>::save_data_single(ExperimentResult &result,
                                      const std::string &key,
                                      const T& datum, OpType type) const {
  result.data.add_single(datum, key);
  result.metadata.add(type, "result_types", key);
  result.metadata.add(DataSubType::single, "result_subtypes", key);
}

template <class state_t>
template <class T>
void State<state_t>::save_data_single(ExperimentResult &result,
                                      const std::string &key,
                                      T&& datum, OpType type) const {
  result.data.add_single(std::move(datum), key);
  result.metadata.add(type, "result_types", key);
  result.metadata.add(DataSubType::single, "result_subtypes", key);
}

template <class state_t>
void State<state_t>::snapshot_state(Registers<state_t>& state, const Operations::Op &op,
                                    ExperimentResult &result,
                                    std::string name) const {
  name = (name.empty()) ? op.name : name;
  result.legacy_data.add_pershot_snapshot(name, op.string_params[0], state.qreg());
}


template <class state_t>
void State<state_t>::snapshot_creg_memory(Registers<state_t>& state,const Operations::Op &op,
                                          ExperimentResult &result,
                                          std::string name) const 
{
  result.legacy_data.add_pershot_snapshot(name,
                               op.string_params[0],
                               state.creg().memory_hex());
}


template <class state_t>
void State<state_t>::snapshot_creg_register(Registers<state_t>& state, const Operations::Op &op,
                                            ExperimentResult &result,
                                            std::string name) const 
{
  result.legacy_data.add_pershot_snapshot(name,
                               op.string_params[0],
                               state.creg().register_hex());
}


template <class state_t>
void State<state_t>::apply_save_expval(Registers<state_t>& state, 
                                       const Operations::Op &op,
                                       ExperimentResult &result)
{
  // Check empty edge case
  if (op.expval_params.empty()) {
    throw std::invalid_argument(
        "Invalid save expval instruction (Pauli components are empty).");
  }
  bool variance = (op.type == OpType::save_expval_var);

  // Accumulate expval components
  double expval(0.);
  double sq_expval(0.);

  for (const auto &param : op.expval_params) {
    // param is tuple (pauli, coeff, sq_coeff)
    const auto val = expval_pauli(state, op.qubits, std::get<0>(param));
    expval += std::get<1>(param) * val;
    if (variance) {
      sq_expval += std::get<2>(param) * val;
    }
  }
  if (variance) {
    std::vector<double> expval_var(2);
    expval_var[0] = expval;  // mean
    expval_var[1] = sq_expval - expval * expval;  // variance
    save_data_average(state, result, op.string_params[0], expval_var, op.type, op.save_type);
  } else {
    save_data_average(state, result, op.string_params[0], expval, op.type, op.save_type);
  }
}

template <class state_t>
void State<state_t>::save_count_data(ExperimentResult& result,bool save_memory)
{
  if (state_.creg().memory_size() > 0) {
    std::string memory_hex = state_.creg().memory_hex();
    result.data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
    if(save_memory) {
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
