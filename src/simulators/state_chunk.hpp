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
  const auto &qreg(uint_t idx=0) const {return qregs_[idx];}
  const auto &creg() const {return creg_;}
  const auto &opset() const {return opset_;}

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
                         ExperimentData &data,
                         RngEngine &rng)  = 0;

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
  virtual void add_metadata(ExperimentData &data) const {
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

  // Add current creg classical bit values to a ExperimentData container
  void add_creg_to_data(ExperimentData &data) const;

  //-----------------------------------------------------------------------
  // Standard snapshots
  //-----------------------------------------------------------------------

  // Snapshot the current statevector (single-shot)
  // if type_label is the empty string the operation type will be used for the type
  void snapshot_state(const Operations::Op &op, ExperimentData &data,
                      std::string name = "") const;

  // Snapshot the classical memory bits state (single-shot)
  void snapshot_creg_memory(const Operations::Op &op, ExperimentData &data,
                            std::string name = "memory") const;

  // Snapshot the classical register bits state (single-shot)
  void snapshot_creg_register(const Operations::Op &op, ExperimentData &data,
                              std::string name = "register") const;

  //-----------------------------------------------------------------------
  // OpenMP thread settings
  //-----------------------------------------------------------------------

  // Sets the number of threads available to the State implementation
  // If negative there is no restriction on the backend
  inline void set_parallalization(int n) {threads_ = n;}

protected:

  // The quantum state data structure
  std::vector<state_t> qregs_;

  // Classical register data
  ClassicalRegister creg_;

  // Opset of instructions supported by the state
  Operations::OpSet opset_;

  // Maximum threads which may be used by the backend for OpenMP multithreading
  // Default value is single-threaded unless overridden
  int threads_ = 1;


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

  bool chunk_omp_parallel_;     //using thread parallel to process loop of chunks or not

  void setup_chunk_bits(uint_t num_qubits, int scale = 1);
  uint_t get_process_by_chunk(uint_t cid);

  //swap between chunks
  void apply_chunk_swap(const reg_t &qubits);
};

template <class state_t>
StateChunk<state_t>::StateChunk(const Operations::OpSet &opset) : opset_(opset)
{
  num_global_chunks_ = 0;
  num_local_chunks_ = 0;

  myrank_ = 0;
  nprocs_ = 1;

  chunk_omp_parallel_ = false;
}

template <class state_t>
StateChunk<state_t>::~StateChunk(void)
{
  chunk_index_begin_.clear();
  chunk_index_end_.clear();
}

//=========================================================================
// Implementations
//=========================================================================
template <class state_t>
void StateChunk<state_t>::setup_chunk_bits(uint_t num_qubits,int scale)
{
  int max_bits = num_qubits;
  uint_t i;

  myrank_ = 0;
  nprocs_ = 1;
#ifdef AER_MPI
  int t;
	MPI_Comm_size(MPI_COMM_WORLD,&t);
  nprocs_ = t;
	MPI_Comm_rank(MPI_COMM_WORLD,&t);
  myrank_ = t;
#endif

  num_qubits_ = num_qubits;

  if(block_bits_ > 0){
    chunk_bits_ = block_bits_;
    if(chunk_bits_ > num_qubits_){
      chunk_bits_ = num_qubits_;
    }
  }
  else{
    chunk_bits_ = num_qubits_;
  }

  //scale for density matrix
  chunk_bits_ *= scale;
  num_qubits_ *= scale;

  num_global_chunks_ = 1ull << (num_qubits_ - chunk_bits_);

  chunk_index_begin_.resize(nprocs_);
  chunk_index_end_.resize(nprocs_);
  for(i=0;i<nprocs_;i++){
    chunk_index_begin_[i] = num_global_chunks_*i / nprocs_;
    chunk_index_end_[i] = num_global_chunks_*(i+1) / nprocs_;
  }

  num_local_chunks_ = chunk_index_end_[myrank_] - chunk_index_begin_[myrank_];
  global_chunk_index_ = chunk_index_begin_[myrank_];

}

template <class state_t>
uint_t StateChunk<state_t>::get_process_by_chunk(uint_t cid)
{
  uint_t i;
  for(i=0;i<nprocs_;i++){
    if(cid >= chunk_index_begin_[i] && cid < chunk_index_end_[i]){
      return i;
    }
  }
  return nprocs_;
}

template <class state_t>
void StateChunk<state_t>::set_config(const json_t &config) 
{
  block_bits_ = 0;
  if (JSON::check_key("blocking_qubits", config))
    JSON::get_value(block_bits_, "blocking_qubits", config);
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
void StateChunk<state_t>::initialize_creg(uint_t num_memory, uint_t num_register) {
  creg_.initialize(num_memory, num_register);
}


template <class state_t>
void StateChunk<state_t>::initialize_creg(uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) {
  creg_.initialize(num_memory, num_register, memory_hex, register_hex);
}


template <class state_t>
void StateChunk<state_t>::snapshot_state(const Operations::Op &op,
                                    ExperimentData &data,
                                    std::string name) const 
{
  name = (name.empty()) ? op.name : name;

  uint_t i;
  for(i=0;i<qregs_.size();i++){
    data.add_pershot_snapshot(name, op.string_params[0], qregs_[i]);
  }
}


template <class state_t>
void StateChunk<state_t>::snapshot_creg_memory(const Operations::Op &op,
                                          ExperimentData &data,
                                          std::string name) const {
  data.add_pershot_snapshot(name,
                               op.string_params[0],
                               creg_.memory_hex());
}


template <class state_t>
void StateChunk<state_t>::snapshot_creg_register(const Operations::Op &op,
                                            ExperimentData &data,
                                            std::string name) const {
  data.add_pershot_snapshot(name,
                               op.string_params[0],
                               creg_.register_hex());
}


template <class state_t>
void StateChunk<state_t>::add_creg_to_data(ExperimentData &data) const {
  if (creg_.memory_size() > 0) {
    std::string memory_hex = creg_.memory_hex();
    data.add_memory_count(memory_hex);
    data. add_pershot_memory(memory_hex);
  }
  // Register bits value
  if (creg_.register_size() > 0) {
    data. add_pershot_register(creg_.register_hex());
  }
}


template <class state_t>
void StateChunk<state_t>::apply_chunk_swap(const reg_t &qubits)
{
  uint_t q0,q1,t;
  uint_t iChunk,nLarge;

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
    uint_t iPair,nPair,mask0,mask1;
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
    uint_t procs = nprocs_;
    while(procs > 1){
      if((procs & 1) != 0){
        proc_bits = -1;
        break;
      }
      proc_bits++;
      procs >>= 1;
    }

    if(nprocs_ == 1 || (proc_bits >= 0 && q1 < (num_qubits_ - proc_bits))){   //no data transfer between processes is needed
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

        if(iChunk1 >= chunk_index_begin_[myrank_] && iChunk1 < chunk_index_end_[myrank_]){    //chunk1 is on this process
          if(iChunk2 >= chunk_index_begin_[myrank_] && iChunk2 < chunk_index_end_[myrank_]){    //chunk2 is on this process
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
          if(iChunk2 >= chunk_index_begin_[myrank_] && iChunk2 < chunk_index_end_[myrank_]){    //chunk2 is on this process
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

        void* pRecv = qregs_[iLocalChunk - global_chunk_index_].recv_buffer(sizeRecv);
        MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,iPair % 256,MPI_COMM_WORLD,&reqRecv);

        void* pSend = qregs_[iLocalChunk - global_chunk_index_].send_buffer(sizeSend);
        MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,iPair % 256,MPI_COMM_WORLD,&reqSend);

        MPI_Wait(&reqSend,&st);
        MPI_Wait(&reqRecv,&st);
        //MPI_Sendrecv can be used if number of processes = 2^m
        //MPI_Sendrecv(pSend,sizeSend,MPI_BYTE,iProc,myrank_,pRecv,sizeRecv,MPI_BYTE,iProc,iProc,MPI_COMM_WORLD,&st);

        qregs_[iLocalChunk - global_chunk_index_].apply_chunk_swap(qubits,iRemoteChunk);
      }
    }
#endif

  }
}

  
//-------------------------------------------------------------------------
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
