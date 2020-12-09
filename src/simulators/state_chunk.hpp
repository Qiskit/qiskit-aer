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

#ifndef _aer_base_state_chunk_hpp_
#define _aer_base_state_chunk_hpp_

#include "framework/json.hpp"
#include "framework/opset.hpp"
#include "framework/types.hpp"
#include "framework/creg.hpp"
#include "framework/results/experiment_data.hpp"

#ifdef AER_MPI
#include <mpi.h>
#endif

#include <omp.h>

namespace AER {
namespace Base {

//=========================================================================
// State interface base class for Qiskit-Aer
//=========================================================================

template <class state_t>
class StateChunk {

public:
  using ignore_argument = void;

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

  StateChunk(const Operations::OpSet &opset);

  StateChunk(const Operations::OpSet::optypeset_t &optypes,
        const stringset_t &gates,
        const stringset_t &snapshots)
    : StateChunk(Operations::OpSet(optypes, gates, snapshots)) {};

  virtual ~StateChunk();

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

  // Returns a const reference to the states data structure
  // Return the state qreg object
  auto &qreg(uint_t idx=0) { return qregs_[idx]; }
  const auto &qreg(uint_t idx=0) const { return qregs_[idx]; }

  // Return the state creg object
  auto &creg() { return cregs_; }
  const auto &creg() const { return cregs_; }

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

  // Apply a sequence of operations to the current state of the State class.
  // It is up to the State subclass to decide how this sequence should be
  // executed (ie in sequence, or some other execution strategy.)
  // If this sequence contains operations not in the supported opset
  // an exeption will be thrown.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops = false);

  //memory allocation (previously called before inisitalize_qreg)
  virtual void allocate(uint_t num_qubits,uint_t shots)
  {
  }

  // Initializes the State to the default state.
  // Typically this is the n-qubit all |0> state
  virtual void initialize_qreg(uint_t num_qubits) = 0;

  // Initializes the State to a specific state.
  virtual void initialize_qreg(uint_t num_qubits, const state_t &state) = 0;

  // Return an estimate of the required memory for implementing the
  // specified sequence of operations on a `num_qubit` sized State.
  virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops)
                                    const = 0;

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
  // ClassicalRegister methods
  //-----------------------------------------------------------------------

  // Initialize classical memory and register to default value (all-0)
  void initialize_creg(uint_t num_memory, uint_t num_register);

  // Initialize classical memory and register to specific values
  void initialize_creg(uint_t num_memory,
                       uint_t num_register,
                       const std::string &memory_hex,
                       const std::string &register_hex);

  // Add current creg classical bit values to a ExperimentResult container
  void add_creg_to_data(ExperimentResult &result) const;

  //-----------------------------------------------------------------------
  // Standard snapshots
  //-----------------------------------------------------------------------

  // Snapshot the current statevector (single-shot)
  // if type_label is the empty string the operation type will be used for the type
  void snapshot_state(const int_t ireg, const Operations::Op &op, ExperimentResult &result,
                      std::string name = "") const;

  // Snapshot the classical memory bits state (single-shot)
  void snapshot_creg_memory(const int_t ireg, const Operations::Op &op, ExperimentResult &result,
                            std::string name = "memory") const;

  // Snapshot the classical register bits state (single-shot)
  void snapshot_creg_register(const int_t ireg, const Operations::Op &op, ExperimentResult &result,
                              std::string name = "register") const;

  //add final state to result
  virtual void add_state_to_data(ExperimentResult &result)
  {
    ;
  }

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

  // The quantum state data structure
  std::vector<state_t> qregs_;

  // Classical register data
  std::vector<ClassicalRegister> cregs_;

  // Opset of instructions supported by the state
  Operations::OpSet opset_;

  // Maximum threads which may be used by the backend for OpenMP multithreading
  // Default value is single-threaded unless overridden
  int threads_ = 1;

  uint_t num_shots_;            //number of shots to be parallelized
  uint_t shot_count_;           //shot counter

  uint_t num_qubits_;           //number of qubits

  uint_t num_global_chunks_;    //number of total chunks 
  uint_t num_local_chunks_;     //number of local chunks
  uint_t chunk_bits_;           //number of qubits per chunk
  uint_t block_bits_;           //number of cache blocked qubits

  uint_t global_chunk_index_;   //beginning chunk index for this process
  reg_t chunk_index_begin_;     //beginning chunk index for each process
  reg_t chunk_index_end_;       //ending chunk index for each process

  uint_t myrank_;               //process ID
  uint_t nprocs_;               //number of processes
  uint_t distributed_rank_;     //process ID in communicator group
  uint_t distributed_procs_;    //number of processes in communicator group
  uint_t distributed_group_;    //group id of distribution

  bool chunk_omp_parallel_;     //using thread parallel to process loop of chunks or not

  bool multi_shot_parallelization_;

  void setup_chunk_bits(uint_t num_qubits, int scale = 1);
  uint_t get_process_by_chunk(uint_t cid);

  //swap between chunks
  virtual void apply_chunk_swap(const reg_t &qubits);

  //reduce values over processes
  void reduce_sum(rvector_t& sum) const;
  void reduce_sum(complex_t& sum) const;
  void reduce_sum(double& sum) const;

  //gather values on each process
  void gather_value(rvector_t& val) const;

  //barrier all processes
  void sync_process(void) const;

  //gather distributed state into vector (if memory is enough)
  void gather_state(std::vector<std::complex<double>>& state);
  void gather_state(std::vector<std::complex<float>>& state);

  //apply one operator
  //implement this function instead of apply_ops in the sub classes for simulation methods
  virtual void apply_op(const int_t iChunk,const Operations::Op &op,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops = false)  = 0;

  // Set a global phase exp(1j * theta) for the state
  bool has_global_phase_ = false;
  complex_t global_phase_ = 1;

#ifdef AER_MPI
  //communicator group to simulate a circuit (for multi-experiments)
  MPI_Comm distributed_comm_;
#endif

};

template <class state_t>
StateChunk<state_t>::StateChunk(const Operations::OpSet &opset) : opset_(opset)
{
  num_global_chunks_ = 0;
  num_local_chunks_ = 0;

  myrank_ = 0;
  nprocs_ = 1;

  distributed_procs_ = 1;
  distributed_rank_ = 0;
  distributed_group_ = 0;

  chunk_omp_parallel_ = false;
  multi_shot_parallelization_ = false;

  num_shots_ = 1;
  shot_count_ = 0;

#ifdef AER_MPI
  distributed_comm_ = MPI_COMM_WORLD;
#endif
}

template <class state_t>
StateChunk<state_t>::~StateChunk(void)
{
  qregs_.clear();
  cregs_.clear();

  chunk_index_begin_.clear();
  chunk_index_end_.clear();

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
void StateChunk<state_t>::set_global_phase(const double &phase_angle) {
  if (Linalg::almost_equal(phase_angle, 0.0)) {
    has_global_phase_ = false;
    global_phase_ = 1;
  }
  else {
    has_global_phase_ = true;
    global_phase_ = std::exp(complex_t(0.0, phase_angle));
  }
}


template <class state_t>
void StateChunk<state_t>::set_distribution(uint_t nprocs)
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
void StateChunk<state_t>::setup_chunk_bits(uint_t num_qubits,int scale)
{
  int max_bits = num_qubits;
  uint_t i;

  num_qubits_ = num_qubits;

  if(block_bits_ > 0){
    chunk_bits_ = block_bits_;
    if(chunk_bits_ > num_qubits_){
      chunk_bits_ = num_qubits_;
    }
  }
  else{
    if(omp_get_num_threads() > 1){
      multi_shot_parallelization_ = true;
    }
    chunk_bits_ = num_qubits_;
  }

  //scale for density matrix
  chunk_bits_ *= scale;
  num_qubits_ *= scale;

  num_global_chunks_ = num_shots_ << (num_qubits_ - chunk_bits_);


  chunk_index_begin_.resize(distributed_procs_);
  chunk_index_end_.resize(distributed_procs_);
  for(i=0;i<distributed_procs_;i++){
    chunk_index_begin_[i] = num_global_chunks_*i / distributed_procs_;
    chunk_index_end_[i] = num_global_chunks_*(i+1) / distributed_procs_;
  }

  num_local_chunks_ = chunk_index_end_[distributed_rank_] - chunk_index_begin_[distributed_rank_];
  global_chunk_index_ = chunk_index_begin_[distributed_rank_];

  if(num_shots_ > 1){
    num_shots_ = num_local_chunks_;
  }

  qregs_.resize(num_local_chunks_);
  cregs_.resize(num_shots_);
}

template <class state_t>
uint_t StateChunk<state_t>::get_process_by_chunk(uint_t cid)
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
void StateChunk<state_t>::set_config(const json_t &config) 
{
  block_bits_ = 0;
  if (JSON::check_key("blocking_qubits", config))
    JSON::get_value(block_bits_, "blocking_qubits", config);
}


template <class state_t>
void StateChunk<state_t>::apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops)
{
  int_t iChunk;
  uint_t iOp,nOp;

  if(multi_shot_parallelization_){
    //for multi-shot distribution mode

    //apply_ops is called shots times from controller class, so apply only the first call
    if(shot_count_++ > 0){
      if(shot_count_ >= num_shots_)
        shot_count_ = 0;
      return;
    }

    nOp = ops.size();
    iOp = 0;
    while(iOp < nOp){
      for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
        apply_op(iChunk,ops[iOp],result,rng,final_ops && nOp == iOp + 1);
      }
      iOp++;
    }
  }
  else{
    //multi-chunk parallelization mode

    nOp = ops.size();
    iOp = 0;
    while(iOp < nOp){
      if(ops[iOp].type == Operations::OpType::gate && ops[iOp].name == "swap_chunk"){
        //apply swap between chunks
        apply_chunk_swap(ops[iOp].qubits);
      }
      else if(ops[iOp].type == Operations::OpType::sim_op && ops[iOp].name == "begin_blocking"){
        //applying sequence of gates inside each chunk
        uint_t iOpBegin = iOp + 1;
#pragma omp parallel for if(chunk_omp_parallel_) private(iChunk) 
        for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
          uint_t iOpBlock;
          //fecth chunk in cache
          qregs_[iChunk].fetch_chunk();

          iOpBlock = iOpBegin;
          while(iOpBlock < nOp){
            if(ops[iOpBlock].type == Operations::OpType::sim_op && ops[iOpBlock].name == "end_blocking"){
              //end of sequence of blocking
              break;
            }
            apply_op(iChunk,ops[iOpBlock],result,rng,final_ops);
            iOpBlock++;
          }

#ifdef _MSC_VER
#pragma omp critical
                {
#else
#pragma omp atomic write
#endif
                  iOp = iOpBlock;
#ifdef _MSC_VER
                }
#endif
          //release chunk from cache
          qregs_[iChunk].release_chunk();
        }
      }
      else if(ops[iOp].type == Operations::OpType::measure || ops[iOp].type == Operations::OpType::snapshot || ops[iOp].type == Operations::OpType::kraus ||
              ops[iOp].type == Operations::OpType::bfunc || ops[iOp].type == Operations::OpType::roerror){
                //for these operations, parallelize inside state implementations
        apply_op(-1,ops[iOp],result,rng,final_ops && nOp == iOp + 1);
      }
      else{
#pragma omp parallel for if(chunk_omp_parallel_) private(iChunk) 
        for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
          apply_op(iChunk,ops[iOp],result,rng,final_ops && nOp == iOp + 1);
        }
      }
      iOp++;
    }
  }
}

template <class state_t>
std::vector<reg_t> StateChunk<state_t>::sample_measure(const reg_t &qubits,
                                                  uint_t shots,
                                                  RngEngine &rng) {
  (ignore_argument)qubits;
  (ignore_argument)shots;
  return std::vector<reg_t>();
}


template <class state_t>
void StateChunk<state_t>::initialize_creg(uint_t num_memory, uint_t num_register) 
{
  int_t ireg,nreg;
  nreg = cregs_.size();
  for(ireg=0;ireg<nreg;ireg++)
    cregs_[ireg].initialize(num_memory, num_register);
}


template <class state_t>
void StateChunk<state_t>::initialize_creg(uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) 
{
  int_t ireg,nreg;
  nreg = cregs_.size();
  for(ireg=0;ireg<nreg;ireg++)
    cregs_[ireg].initialize(num_memory, num_register, memory_hex, register_hex);
}


template <class state_t>
void StateChunk<state_t>::snapshot_state(const int_t ireg, const Operations::Op &op,
                                    ExperimentResult &result,
                                    std::string name) const 
{
  name = (name.empty()) ? op.name : name;

  if(ireg < 0){
    int_t i;
    for(i=0;i<qregs_.size();i++){
      result.data.add_pershot_snapshot(name, op.string_params[0], qregs_[i]);
    }
  }
  else if(ireg < qregs_.size()){
    result.data.add_pershot_snapshot(name, op.string_params[0], qregs_[ireg]);
  }
}


template <class state_t>
void StateChunk<state_t>::snapshot_creg_memory(const int_t ireg, const Operations::Op &op,
                                          ExperimentResult &result,
                                          std::string name) const 
{
  if(ireg < 0){
    int_t i;
    for(i=0;i<cregs_.size();i++){
      result.data.add_pershot_snapshot(name,
                                   op.string_params[0],
                                   cregs_[i].memory_hex());
    }
  }
  else if(ireg < cregs_.size()){
    result.data.add_pershot_snapshot(name,
                                 op.string_params[0],
                                 cregs_[ireg].memory_hex());
  }
}


template <class state_t>
void StateChunk<state_t>::snapshot_creg_register(const int_t ireg, const Operations::Op &op,
                                            ExperimentResult &result,
                                            std::string name) const 
{
  if(ireg < 0){
    int_t i;
    for(i=0;i<cregs_.size();i++){
      result.data.add_pershot_snapshot(name,
                               op.string_params[0],
                               cregs_[i].register_hex());
    }
  }
  else if(ireg < cregs_.size()){
    result.data.add_pershot_snapshot(name,
                               op.string_params[0],
                               cregs_[ireg].register_hex());
  }
}


template <class state_t>
void StateChunk<state_t>::add_creg_to_data(ExperimentResult &result) const 
{
  int_t ireg,nreg;
  nreg = cregs_.size();

  for(ireg=0;ireg<nreg;ireg++){
    if (cregs_[ireg].memory_size() > 0) {
      std::string memory_hex = cregs_[ireg].memory_hex();
      result.data.add_memory_count(memory_hex);
      result.data.add_pershot_memory(memory_hex);
    }
    // Register bits value
    if (cregs_[ireg].register_size() > 0) {
      result.data.add_pershot_register(cregs_[ireg].register_hex());
    }
  }
}


template <class state_t>
void StateChunk<state_t>::apply_chunk_swap(const reg_t &qubits)
{
  uint_t q0,q1,t,nLarge;
  int_t iChunk;

  q0 = qubits[qubits.size() - 2];
  q1 = qubits[qubits.size() - 1];

  if(q0 > q1){
    t = q0;
    q0 = q1;
    q1 = t;
  }

  if(q1 < chunk_bits_){
    //device
#pragma omp parallel for if(chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      qregs_[iChunk].apply_mcswap(qubits);
    }
  }
  else{ //swap over chunks
    int_t iPair;
    uint_t nPair,mask0,mask1;
    uint_t baseChunk,iChunk1,iChunk2;

    if(q0 < chunk_bits_)
      nLarge = 1;
    else
      nLarge = 2;

    mask0 = (1ull << q0);
    mask1 = (1ull << q1);
    mask0 >>= chunk_bits_;
    mask1 >>= chunk_bits_;

    int proc_bits = 0;
    uint_t procs = distributed_procs_;
    while(procs > 1){
      if((procs & 1) != 0){
        proc_bits = -1;
        break;
      }
      proc_bits++;
      procs >>= 1;
    }

    if(distributed_procs_ == 1 || (proc_bits >= 0 && q1 < (num_qubits_ - proc_bits))){   //no data transfer between processes is needed
      if(q0 < chunk_bits_){
        nPair = num_local_chunks_ >> 1;
      }
      else{
        nPair = num_local_chunks_ >> 2;
      }

#pragma omp parallel for if(chunk_omp_parallel_) private(iPair,baseChunk,iChunk1,iChunk2)
      for(iPair=0;iPair<nPair;iPair++){
        if(q0 < chunk_bits_){
          baseChunk = iPair & (mask1-1);
          baseChunk += ((iPair - baseChunk) << 1);
        }
        else{
          uint_t t0,t1;
          t0 = iPair & (mask0-1);
          baseChunk = (iPair - t0) << 1;
          t1 = baseChunk & (mask1-1);
          baseChunk = (baseChunk - t1) << 1;
          baseChunk += t0 + t1;
        }

        iChunk1 = baseChunk | mask0;
        iChunk2 = baseChunk | mask1;

        qregs_[iChunk1].apply_chunk_swap(qubits,qregs_[iChunk2],true);
      }
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

      if(q0 < chunk_bits_){
        nLarge = 1;
        nu[0] = 1ull << (q1 - chunk_bits_);
        ub[0] = 0;
        iu[0] = 0;

        nu[1] = 1ull << (num_qubits_ - q1 - 1);
        ub[1] = (q1 - chunk_bits_) + 1;
        iu[1] = 0;
      }
      else{
        nLarge = 2;
        nu[0] = 1ull << (q0 - chunk_bits_);
        ub[0] = 0;
        iu[0] = 0;

        nu[1] = 1ull << (q1 - q0 - 1);
        ub[1] = (q0 - chunk_bits_) + 1;
        iu[1] = 0;

        nu[2] = 1ull << (num_qubits_ - q1 - 1);
        ub[2] = (q1 - chunk_bits_) + 1;
        iu[2] = 0;
      }
      nPair = 1ull << (num_qubits_ - chunk_bits_ - nLarge);

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
            qregs_[iChunk1 - global_chunk_index_].apply_chunk_swap(qubits,qregs_[iChunk2 - global_chunk_index_],true);
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

        void* pSend = qregs_[iLocalChunk - global_chunk_index_].send_buffer(sizeSend);
        MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,iPair,distributed_comm_,&reqSend);

        void* pRecv = qregs_[iLocalChunk - global_chunk_index_].recv_buffer(sizeRecv);
        MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,iPair,distributed_comm_,&reqRecv);

        MPI_Wait(&reqSend,&st);
        MPI_Wait(&reqRecv,&st);
        //MPI_Sendrecv can be used if number of processes = 2^m
        //MPI_Sendrecv(pSend,sizeSend,MPI_BYTE,iProc,distributed_rank_,pRecv,sizeRecv,MPI_BYTE,iProc,iProc,MPI_COMM_WORLD,&st);

        qregs_[iLocalChunk - global_chunk_index_].apply_chunk_swap(qubits,iRemoteChunk);
      }
    }
#endif

  }
}

template <class state_t>
void StateChunk<state_t>::reduce_sum(rvector_t& sum) const
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
void StateChunk<state_t>::reduce_sum(complex_t& sum) const
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
void StateChunk<state_t>::reduce_sum(double& sum) const
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
void StateChunk<state_t>::gather_value(rvector_t& val) const
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    MPI_Alltoall(&val[0],1,MPI_DOUBLE_PRECISION,&val[0],1,MPI_DOUBLE_PRECISION,distributed_comm_);
  }
#endif
}

template <class state_t>
void StateChunk<state_t>::sync_process(void) const
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    MPI_Barrier(distributed_comm_);
  }
#endif
}

//gather distributed state into vector (if memory is enough)
template <class state_t>
void StateChunk<state_t>::gather_state(std::vector<std::complex<double>>& state)
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    uint_t size,local_size,global_size,offset;
    int i;
    MPI_Status st;
    MPI_Request reqSend,reqRecv;

    local_size = state.size();
    MPI_Allreduce(&local_size,&global_size,1,MPI_UINT64_T,MPI_SUM,distributed_comm_);

    //TO DO check memory availability

    if(distributed_rank_ == 0){
      state.resize(global_size);

      offset = 0;
      for(i=1;i<distributed_procs_;i++){
        MPI_Irecv(&size,1,MPI_UINT64_T,i,i*2,distributed_comm_,&reqRecv);
        MPI_Wait(&reqRecv,&st);
        MPI_Irecv(&state[offset],size*2,MPI_DOUBLE_PRECISION,i,i*2+1,distributed_comm_,&reqRecv);
        MPI_Wait(&reqRecv,&st);
        offset += size;
      }
    }
    else{
      MPI_Isend(&local_size,1,MPI_UINT64_T,0,i*2,distributed_comm_,&reqSend);
      MPI_Wait(&reqSend,&st);
      MPI_Isend(&state[0],local_size*2,MPI_DOUBLE_PRECISION,0,i*2+1,distributed_comm_,&reqSend);
      MPI_Wait(&reqSend,&st);
    }
  }
#endif
}

template <class state_t>
void StateChunk<state_t>::gather_state(std::vector<std::complex<float>>& state)
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    uint_t size,local_size,global_size,offset;
    int i;
    MPI_Status st;
    MPI_Request reqSend,reqRecv;

    local_size = state.size();
    MPI_Allreduce(&local_size,&global_size,1,MPI_UINT64_T,MPI_SUM,distributed_comm_);

    //TO DO check memory availability

    if(distributed_rank_ == 0){
      state.resize(global_size);

      offset = 0;
      for(i=1;i<distributed_procs_;i++){
        MPI_Irecv(&size,1,MPI_UINT64_T,i,i*2,distributed_comm_,&reqRecv);
        MPI_Wait(&reqRecv,&st);
        MPI_Irecv(&state[offset],size*2,MPI_FLOAT,i,i*2+1,distributed_comm_,&reqRecv);
        MPI_Wait(&reqRecv,&st);
        offset += size;
      }
    }
    else{
      MPI_Isend(&local_size,1,MPI_UINT64_T,0,i*2,distributed_comm_,&reqSend);
      MPI_Wait(&reqSend,&st);
      MPI_Isend(&state[0],local_size*2,MPI_FLOAT,0,i*2+1,distributed_comm_,&reqSend);
      MPI_Wait(&reqSend,&st);
    }
  }
#endif
}

//-------------------------------------------------------------------------
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
