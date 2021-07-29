/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_base_multi_states_hpp_
#define _aer_base_multi_states_hpp_

#include "framework/json.hpp"
#include "framework/opset.hpp"
#include "framework/types.hpp"
#include "framework/creg.hpp"

#include "noise/noise_model.hpp"


#ifdef AER_MPI
#include <mpi.h>
#endif

#include <omp.h>

namespace AER {
namespace Multi {

//=========================================================================
// State interface base class for Qiskit-Aer
//=========================================================================

template <class state_t>
class States {

public:
  using ignore_argument = void;
  using DataSubType = Operations::DataSubType;

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

  States();

  virtual ~States();

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

  // Returns a const reference to the states data structure
  // Return the state qreg object
  auto &qreg(uint_t idx=0) { return states_[idx].qreg(); }
  const auto &qreg(uint_t idx=0) const { return states_[idx].qreg(); }

  // Return the state creg object
  auto &creg(uint_t idx=0) { return states_[idx].creg(); }
  const auto &creg(uint_t idx=0) const { return states_[idx].creg(); }

  // Return the state opset object
  auto &opset() { return states_[0].opset(); }
  const auto &opset() const { return states_[0].opset(); }

  auto &state(uint_t idx = 0){ return states_[idx];}

  uint_t num_qubits(void)
  {
    return num_qubits_;
  }

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
  virtual std::string name()
  {
    return states_[0].name();
  }

  // Apply a sequence of operations to the current state of the State class.
  // It is up to the State subclass to decide how this sequence should be
  // executed (ie in sequence, or some other execution strategy.)
  // If this sequence contains operations not in the supported opset
  // an exeption will be thrown.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentResult &result,
                         RngEngine& rng,
                         bool final_ops = false)
  {
  }

  virtual void apply_single_ops(const std::vector<Operations::Op> &ops,
                         ExperimentResult &result,
                         std::vector<RngEngine>& rng,
                         const AER::Noise::NoiseModel &noise,
                         bool final_ops = false);

  virtual void apply_multi_ops(const std::vector<std::vector<Operations::Op>> &ops,
                         std::vector<ExperimentResult> &result,
                         std::vector<RngEngine>& rng,
                         const std::vector<AER::Noise::NoiseModel> &noise,
                         bool final_ops = false);

  virtual void end_of_circuit();

  //memory allocation (previously called before inisitalize_qreg)
  virtual void allocate(uint_t num_qubits,uint_t block_bits,uint_t num_parallel_shots = 1);
  virtual void bind_state(States<state_t>& state,uint_t ishot,bool batch_enable);

  // Initializes the State to the default state.
  // Typically this is the n-qubit all |0> state
  virtual void initialize_qreg(uint_t num_qubits);

  // Initializes the State to a specific state.
  virtual void initialize_qreg(uint_t num_qubits, const state_t &state);

  // Return an estimate of the required memory for implementing the
  // specified sequence of operations on a `num_qubit` sized State.
  virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops)
                                    const;

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

  virtual std::vector<reg_t> batched_sample_measure(const reg_t &qubits,
                                            reg_t& shots,
                                            std::vector<RngEngine> &rng);

  //=======================================================================
  // Standard non-virtual methods
  //
  // These methods should not be modified in any State subclasses
  //=======================================================================

  //-----------------------------------------------------------------------
  // ClassicalRegister methods
  //-----------------------------------------------------------------------

  // Initialize classical memory and register to default value (all-0)
  void initialize_creg(uint_t num_memory, uint_t num_register);

  // Initialize classical memory and register to specific values
  void initialize_creg(uint_t num_memory,
                       uint_t num_register,
                       const std::string &memory_hex,
                       const std::string &register_hex);

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------
  template <typename list_t>
  void initialize_from_vector(const list_t &vec);

  template <typename list_t>
  void initialize_from_matrix(const list_t &mat);

  //-----------------------------------------------------------------------
  // Save result data
  //-----------------------------------------------------------------------

  // Save current value of all classical registers to result
  // This supports DataSubTypes: c_accum (counts), list (memory)
  // TODO: Make classical data allow saving only subset of specified clbit values
  void save_creg(ExperimentResult &result,
                 const std::string &key,
                 DataSubType type = DataSubType::c_accum) const;
              
  // Save single shot data type. Typically this will be the value for the
  // last shot of the simulation
  template <class T>
  void save_data_single(ExperimentResult &result,
                        const std::string &key, const T& datum) const;

  template <class T>
  void save_data_single(ExperimentResult &result,
                        const std::string &key, T&& datum) const;

  // Save data type which can be averaged over all shots.
  // This supports DataSubTypes: list, c_list, accum, c_accum, average, c_average
  template <class T>
  void save_data_average(ExperimentResult &result,
                         const std::string &key, const T& datum,
                         DataSubType type = DataSubType::average) const;

  template <class T>
  void save_data_average(ExperimentResult &result,
                         const std::string &key, T&& datum,
                         DataSubType type = DataSubType::average) const;
  
  // Save data type which is pershot and does not support accumulator or average
  // This supports DataSubTypes: single, list, c_list
  template <class T>
  void save_data_pershot(ExperimentResult &result,
                         const std::string &key, const T& datum,
                         DataSubType type = DataSubType::list) const;

  template <class T>
  void save_data_pershot(ExperimentResult &result,
                         const std::string &key, T&& datum,
                         DataSubType type = DataSubType::list) const;

  //-----------------------------------------------------------------------
  // Common instructions
  //-----------------------------------------------------------------------
  
  // Apply a save expectation value instruction
  void apply_save_expval(const Operations::Op &op, ExperimentResult &result);

  //-----------------------------------------------------------------------
  // Standard snapshots
  //-----------------------------------------------------------------------

  // Snapshot the current statevector (single-shot)
  // if type_label is the empty string the operation type will be used for the type
  void snapshot_state(const Operations::Op &op, ExperimentResult &result,
                      std::string name = "") const;

  // Snapshot the classical memory bits state (single-shot)
  void snapshot_creg_memory(const Operations::Op &op, ExperimentResult &result,
                            std::string name = "memory") const;

  // Snapshot the classical register bits state (single-shot)
  void snapshot_creg_register(const Operations::Op &op, ExperimentResult &result,
                              std::string name = "register") const;

  //-----------------------------------------------------------------------
  // OpenMP thread settings
  //-----------------------------------------------------------------------

  // Sets the number of threads available to the State implementation
  // If negative there is no restriction on the backend
  inline void set_parallalization(int n) {threads_ = n;}

  // Set a complex global phase value exp(1j * theta) for the state
  void set_global_phase(const double &phase);

  //set number of processes to be distributed
  void set_distribution(uint_t nprocs);

protected:
  std::string method_;

  // multiple states
  std::vector<state_t> states_;

  uint_t num_qubits_;           //number of qubits

  uint_t num_global_states_;
  uint_t num_local_states_;
  uint_t global_state_index_;

  uint_t myrank_;               //process ID
  uint_t nprocs_;               //number of processes
  uint_t distributed_rank_;     //process ID in communicator group
  uint_t distributed_procs_;    //number of processes in communicator group
  uint_t distributed_group_;    //group id of distribution

  bool thrust_optimization_;       //optimization for Thrust implementation

  reg_t top_states_;
  reg_t num_states_on_device_;

  virtual int qubit_scale(void)
  {
    return 1;     //scale of qubit number (x2 for density and unitary matrices)
  }

  // Maximum threads which may be used by the backend for OpenMP multithreading
  // Default value is single-threaded unless overridden
  int threads_ = 1;

  // Set a global phase exp(1j * theta) for the state
  bool has_global_phase_ = false;
  complex_t global_phase_ = 1;

#ifdef AER_MPI
  //communicator group to simulate a circuit (for multi-experiments)
  MPI_Comm distributed_comm_;
#endif

  void apply_runtime_error(int_t istate,const Operations::Op& op,
                          ExperimentResult &result,
                          std::vector<RngEngine>& rng,
                          const AER::Noise::NoiseModel &noise);
};

template <class state_t>
States<state_t>::States()
{
  num_global_states_ = 0;
  num_local_states_ = 0;
  global_state_index_ = 0;

  myrank_ = 0;
  nprocs_ = 1;

  distributed_procs_ = 1;
  distributed_rank_ = 0;
  distributed_group_ = 0;

  num_local_states_ = 0;

  thrust_optimization_ = false;

#ifdef AER_MPI
  distributed_comm_ = MPI_COMM_WORLD;
#endif
}

template <class state_t>
States<state_t>::~States(void)
{
#ifdef AER_MPI
  if(distributed_comm_ != MPI_COMM_WORLD){
    MPI_Comm_free(&distributed_comm_);
  }
#endif
}

//=========================================================================
// Implementations
//=========================================================================
template <class state_t>
void States<state_t>::set_global_phase(const double &phase_angle) 
{
  int_t i;

  if (Linalg::almost_equal(phase_angle, 0.0)) {
    has_global_phase_ = false;
    global_phase_ = 1;
  }
  else {
    has_global_phase_ = true;
    global_phase_ = std::exp(complex_t(0.0, phase_angle));
  }

  for(i=0;i<num_local_states_;i++){
    states_[i].set_global_phase(phase_angle);
  }
}


template <class state_t>
void States<state_t>::set_distribution(uint_t nprocs)
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
void States<state_t>::allocate(uint_t num_qubits,uint_t block_bits,uint_t num_parallel_states)
{
  int_t i;
  uint_t n_states;

  global_state_index_ = (num_parallel_states*(distributed_rank_) / distributed_procs_);
  n_states = (num_parallel_states*(distributed_rank_+1) / distributed_procs_) - global_state_index_;

  //reuse allocated states if enough states are already allocated
  if(num_qubits_ != num_qubits || n_states > num_local_states_){
    states_.resize(n_states);

    states_[0].allocate(num_qubits,block_bits,n_states);
    for(i=1;i<n_states;i++){
      states_[i].allocate(num_qubits,block_bits,0);
      states_[i].bind_state(states_[0],global_state_index_+i,true);
    }
  }

  num_qubits_ = num_qubits;
  num_global_states_ = num_parallel_states;
  num_local_states_ = n_states;
}

template <class state_t>
void States<state_t>::bind_state(States<state_t>& state,uint_t ishot,bool batch_enable)
{
}

template <class state_t>
void States<state_t>::set_config(const json_t &config) 
{
  int_t i;

  for(i=0;i<num_local_states_;i++){
    states_[i].set_config(config);
  }

  JSON::get_value(method_, "method", config);
}


template <class state_t>
void States<state_t>::initialize_qreg(uint_t num_qubits)
{
  int_t i;

  for(i=0;i<num_local_states_;i++){
    states_[i].initialize_qreg(num_qubits);
  }
}

template <class state_t>
void States<state_t>::initialize_qreg(uint_t num_qubits, const state_t &state)
{
  int_t i;

  for(i=0;i<num_local_states_;i++){
    states_[i].initialize_qreg(num_qubits,state.qreg());
  }
}

template <class state_t>
size_t States<state_t>::required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops) const
{
  int_t i;
  size_t size = 0;
  for(i=0;i<num_local_states_;i++){
    size = std::max(size,states_[i].required_memory_mb(num_qubits,ops));
  }
  return size;
}

template <class state_t>
void States<state_t>::apply_single_ops(const std::vector<Operations::Op> &ops,
                         ExperimentResult &result,
                         std::vector<RngEngine>& rng,
                         const Noise::NoiseModel &noise,
                         bool final_ops)
{
  int_t i,iOp,nOp;
  int_t num_top_states = 0;

  //get top states on arrays on each device
  for(i=0;i<num_local_states_;i++){
    if(states_[i].top_of_array()){
      top_states_.push_back(i);
      num_top_states++;
    }
  }
  top_states_.push_back(num_local_states_);

  num_states_on_device_.resize(num_top_states);
  for(i=0;i<num_top_states-1;i++){
    num_states_on_device_[i] = top_states_[i+1] - top_states_[i];
  }
  num_states_on_device_[num_top_states-1] = num_local_states_ - top_states_[num_top_states-1];

  nOp = ops.size();

  std::vector<ExperimentResult> par_results(num_top_states);

#pragma omp parallel for if(num_top_states > 1) num_threads(num_top_states) private(iOp)
  for(i=0;i<num_top_states;i++){
    uint_t istate = top_states_[i];
    for(iOp=0;iOp<nOp;iOp++){
      if(ops[iOp].type == Operations::OpType::runtime_error){
        apply_runtime_error(i,ops[iOp],par_results[i],rng,noise);
      }
      else if(states_[istate].batchable_op(ops[iOp])){
        states_[istate].apply_op(0,ops[iOp],par_results[i],rng,final_ops && nOp == iOp + 1);
      }
      else{
        //call apply_op for each state
        uint_t j;
        for(j=top_states_[i];j<top_states_[i+1];j++){
          states_[j].apply_op(0,ops[iOp],par_results[i],rng,final_ops && nOp == iOp + 1);
        }
      }
    }
  }
  for (auto &res : par_results) {
    result.combine(std::move(res));
  }

#pragma omp parallel for if(num_top_states > 1) num_threads(num_top_states) 
  for(i=0;i<num_top_states;i++){
    states_[i].end_of_circuit();
  }

  //collect measured bits
  for(iOp=0;iOp<nOp;iOp++){
    if(ops[iOp].type == Operations::OpType::measure){
      for(i=0;i<num_local_states_;i++){
        states_[i].store_measured_cbits(ops[iOp]);
      }
    }
  }
}

template <class state_t>
void States<state_t>::apply_multi_ops(const std::vector<std::vector<Operations::Op>> &ops,
                         std::vector<ExperimentResult> &result,
                         std::vector<RngEngine>& rng,
                         const std::vector<AER::Noise::NoiseModel> &noise,
                         bool final_ops)
{
  int_t i;
  int_t num_top_states = 0;

  top_states_.clear();
  //get top states on arrays on each device
  for(i=0;i<num_local_states_;i++){
    if(states_[i].top_of_array()){
      top_states_.push_back(i);
      num_top_states++;
    }
  }
  top_states_.push_back(num_local_states_);

  num_states_on_device_.resize(num_top_states);
  for(i=0;i<num_top_states;i++){
    num_states_on_device_[i] = top_states_[i+1] - top_states_[i];
  }

  bool density_matrix = false;
  if(method_.find("density_matrix") != std::string::npos)
    density_matrix = true;

#pragma omp parallel for if(num_top_states > 1) num_threads(num_top_states) 
  for(i=0;i<num_top_states;i++){
    uint_t j,iOp,istate = top_states_[i];
    uint_t num_active = num_states_on_device_[i];
    std::vector<Operations::Op> batched_ops(top_states_[i+1] - top_states_[i]);
    uint_t n_states;

    iOp = 0;
    while(num_active > 0){
      n_states = 0;
      num_active = 0;
      for(j=top_states_[i];j<top_states_[i+1];j++){
        if(iOp < ops[j].size()){
          bool batch_enable = false;
          batched_ops[j - top_states_[i]] = ops[j][iOp];

          if(density_matrix){ //for density matrix
            if(ops[j][iOp].type == Operations::OpType::gate || ops[j][iOp].type == Operations::OpType::matrix || ops[j][iOp].type == Operations::OpType::diagonal_matrix ||
               ops[j][iOp].type == Operations::OpType::superop || ops[j][iOp].type == Operations::OpType::kraus)
              batch_enable = true;
          }
          else{ //for statevector
            if(ops[j][iOp].type == Operations::OpType::gate || ops[j][iOp].type == Operations::OpType::matrix || ops[j][iOp].type == Operations::OpType::diagonal_matrix)
              batch_enable = true;
          }

          if(batch_enable){
            n_states++;
          }
          else{
            states_[j].enable_batch(false);
            states_[j].apply_op(0,ops[j][iOp],result[j],rng,final_ops && ops[j].size() == iOp + 1);
            states_[j].enable_batch(true);
          }
          num_active++;
        }
        else{
          batched_ops[j - top_states_[i]].type = Operations::OpType::nop;
        }
      }

      if(n_states > 0){
        states_[istate].apply_batched_ops(batched_ops);

        /*
        //for test
        std::cout << " ops[" << iOp << "] ===================" << std::endl;
        for(j=top_states_[i];j<top_states_[i+1];j++){
          std::cout << "    (" << j << ") " << batched_ops[j - top_states_[i]] << std::endl;
          if(batched_ops[j - top_states_[i]].type != Operations::OpType::nop){
            states_[j].enable_batch(false);
            states_[j].apply_op(0,ops[j][iOp],result[j],rng,final_ops && ops[j].size() == iOp + 1);
            states_[j].enable_batch(true);
          }
        }
        std::cout << "=========================================" << std::endl;
        */
      }

      iOp++;
    }
  }

#pragma omp parallel for if(num_top_states > 1) num_threads(num_top_states) 
  for(i=0;i<num_top_states;i++){
    states_[i].end_of_circuit();
  }
}

template <class state_t>
void States<state_t>::apply_runtime_error(int_t istate,const Operations::Op& op,
                          ExperimentResult &result,
                          std::vector<RngEngine>& rng,
                          const AER::Noise::NoiseModel &noise)
{
  uint_t count_i = 0;
  int_t i;
  int_t num_top_states = top_states_.size();
  reg_t params(4*num_states_on_device_[istate]);

  for(i=0;i<num_states_on_device_[istate];i++){
    uint_t x_max = 0;
    uint_t num_y = 0;
    uint_t x_mask = 0;
    uint_t z_mask = 0;
    auto ops = noise.sample_noise_at_runtime(op,rng[i+top_states_[istate]]);
    if(ops.size() == 0)
      count_i++;
    else if(ops.size() == 1 && ops[0].name == "id")
      count_i++;
    else{
      uint_t j;
      for(j=0;j<ops.size();j++){
        if(ops[j].name == "x"){
          x_mask ^= (1ull << ops[j].qubits[0]);
          x_max = std::max<uint_t>(x_max, (ops[j].qubits[0]));
        }
        else if(ops[j].name == "z"){
          z_mask ^= (1ull << ops[j].qubits[0]);
        }
        else if(ops[j].name == "y"){
          x_mask ^= (1ull << ops[j].qubits[0]);
          z_mask ^= (1ull << ops[j].qubits[0]);
          x_max = std::max<uint_t>(x_max, (ops[j].qubits[0]));
          num_y++;
        }
      }
    }
    params[i*4] = x_max;
    params[i*4+1] = num_y % 4;
    params[i*4+2] = x_mask;
    params[i*4+3] = z_mask;
  }

  if(count_i < num_local_states_){
    states_[top_states_[istate]].apply_batched_pauli(params);
  }

}

template <class state_t>
void States<state_t>::end_of_circuit()
{
  int_t i;

  for(i=0;i<num_local_states_;i++){
    states_[i].end_of_circuit();
  }
}

template <class state_t>
std::vector<reg_t> States<state_t>::sample_measure(const reg_t &qubits,
                                                  uint_t shots,
                                                  RngEngine &rng) {
  (ignore_argument)qubits;
  (ignore_argument)shots;
  return std::vector<reg_t>();
}

template <class state_t>
std::vector<reg_t> States<state_t>::batched_sample_measure(const reg_t &qubits,
                                                        reg_t& shots,
                                                        std::vector<RngEngine> &rng)
{
  int_t i,j,k;
//  int_t num_top_states = top_states_.size() - 1;
  std::vector<reg_t> ret;
  uint_t total_shots = 0;
  reg_t shot_offset(shots.size());

  int_t num_top_states = 0;

  top_states_.clear();
  //get top states on arrays on each device
  for(i=0;i<num_local_states_;i++){
    if(states_[i].top_of_array()){
      top_states_.push_back(i);
      num_top_states++;
    }
  }
  top_states_.push_back(num_local_states_);

  for(i=0;i<shots.size();i++){
    shot_offset[i] = total_shots;
    total_shots += shots[i];
  }
  ret.resize(total_shots);

#pragma omp parallel for if(num_top_states > 1) num_threads(num_top_states) private(i,j,k)
  for(i=0;i<num_top_states;i++){
    uint_t istate = top_states_[i];
    reg_t local_shots(shots.begin() + top_states_[i],shots.begin() + top_states_[i+1]);
    std::vector<RngEngine> local_rng(rng.begin() + top_states_[i],rng.begin() + top_states_[i+1]);

    auto samples = states_[istate].batched_sample_measure(qubits,local_shots,local_rng);

    uint_t sample_idx = 0;
    for(j=top_states_[i];j<top_states_[i+1];j++){
      for(k=0;k<shots[j];k++){
        ret[shot_offset[j]+k] = samples[sample_idx++];
      }
    }
  }
  return ret;
}

template <class state_t>
void States<state_t>::initialize_creg(uint_t num_memory, uint_t num_register) 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].initialize_creg(num_memory, num_register);
  }
}


template <class state_t>
void States<state_t>::initialize_creg(uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].initialize_creg(num_memory, num_register, memory_hex, register_hex);
  }
}

template <class state_t>
template <typename list_t>
void States<state_t>::initialize_from_vector(const list_t &vec)
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].initialize_from_vector(vec);
  }
}

template <class state_t>
template <typename list_t>
void States<state_t>::initialize_from_matrix(const list_t &mat)
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].initialize_from_matrix(mat);
  }
}

template <class state_t>
void States<state_t>::save_creg(ExperimentResult &result,
                               const std::string &key,
                               DataSubType type) const 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].save_creg(result,key,type);
  }
}

template <class state_t>
template <class T>
void States<state_t>::save_data_average(ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum,
                                       DataSubType type) const 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].save_data_average(result,key,datum,type);
  }
}

template <class state_t>
template <class T>
void States<state_t>::save_data_average(ExperimentResult &result,
                                       const std::string &key,
                                       T&& datum,
                                       DataSubType type) const 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].save_data_average(result,key,datum,type);
  }
}

template <class state_t>
template <class T>
void States<state_t>::save_data_pershot(ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum,
                                       DataSubType type) const 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].save_data_pershot(result,key,datum,type);
  }
}

template <class state_t>
template <class T>
void States<state_t>::save_data_pershot(ExperimentResult &result, 
                                       const std::string &key,
                                       T&& datum,
                                       DataSubType type) const 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].save_data_pershot(result,key,datum,type);
  }
}

template <class state_t>
template <class T>
void States<state_t>::save_data_single(ExperimentResult &result,
                                      const std::string &key,
                                      const T& datum) const {
  result.data.add_single(datum, key);
}

template <class state_t>
template <class T>
void States<state_t>::save_data_single(ExperimentResult &result,
                                      const std::string &key,
                                      T&& datum) const {
  result.data.add_single(std::move(datum), key);
}

template <class state_t>
void States<state_t>::snapshot_state(const Operations::Op &op,
                                    ExperimentResult &result,
                                    std::string name) const 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].snapshot_state(op,result,name);
  }
}


template <class state_t>
void States<state_t>::snapshot_creg_memory(const Operations::Op &op,
                                          ExperimentResult &result,
                                          std::string name) const 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].snapshot_creg_memory(op,result,name);
  }
}


template <class state_t>
void States<state_t>::snapshot_creg_register(const Operations::Op &op,
                                            ExperimentResult &result,
                                            std::string name) const 
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].snapshot_creg_register(op,result,name);
  }
}


template <class state_t>
void States<state_t>::apply_save_expval(const Operations::Op &op,
                                            ExperimentResult &result)
{
  int_t i;
  for(i=0;i<num_local_states_;i++){
    states_[i].apply_save_expval(op,result);
  }
}


//-------------------------------------------------------------------------
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
