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
  auto &creg() { return creg_; }
  const auto &creg() const { return creg_; }

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
  virtual void allocate(uint_t num_qubits,uint_t block_bits);

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

  // Return the expectation value of a N-qubit Pauli operator
  // If the simulator does not support Pauli expectation value this should
  // raise an exception.
  virtual double expval_pauli(const reg_t &qubits,
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
  uint_t distributed_rank_;     //process ID in communicator group
  uint_t distributed_procs_;    //number of processes in communicator group
  uint_t distributed_group_;    //group id of distribution

  bool chunk_omp_parallel_;     //using thread parallel to process loop of chunks or not
  bool gpu_optimization_;       //optimization for GPU

  reg_t qubit_map_;             //qubit map to restore swapped qubits

  virtual int qubit_scale(void)
  {
    return 1;     //scale of qubit number (x2 for density and unitary matrices)
  }
  uint_t get_process_by_chunk(uint_t cid);

  //swap between chunks
  virtual void apply_chunk_swap(const reg_t &qubits);

  virtual void apply_chunk_x(const uint_t qubit);

  //send/receive chunk in receive buffer
  void send_chunk(uint_t local_chunk_index, uint_t global_chunk_index);
  void recv_chunk(uint_t local_chunk_index, uint_t global_chunk_index);

  template <class data_t>
  void send_data(data_t* pSend, uint_t size, uint_t myid,uint_t pairid);
  template <class data_t>
  void recv_data(data_t* pRecv, uint_t size, uint_t myid,uint_t pairid);

  //reduce values over processes
  void reduce_sum(rvector_t& sum) const;
  void reduce_sum(complex_t& sum) const;
  void reduce_sum(double& sum) const;

  //gather values on each process
  void gather_value(rvector_t& val) const;

  //barrier all processes
  void sync_process(void) const;

  //gather distributed state into vector (if memory is enough)
  template <class data_t>
  void gather_state(std::vector<std::complex<data_t>>& state);

  template <class data_t>
  void gather_state(AER::Vector<std::complex<data_t>>& state);

  //apply one operator
  //implement this function instead of apply_ops in the sub classes for simulation methods
  virtual void apply_op(const int_t iChunk,const Operations::Op &op,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops = false)  = 0;
  // block diagonal matrix in chunk
  void block_diagonal_matrix(const int_t iChunk, reg_t &qubits, cvector_t &diag);

  void qubits_inout(const reg_t& qubits, reg_t& qubits_in,reg_t& qubits_out) const;

  auto apply_to_matrix(bool copy = false);

  virtual bool is_applied_to_each_chunk(const Operations::Op &op);

  // Set a global phase exp(1j * theta) for the state
  bool has_global_phase_ = false;
  complex_t global_phase_ = 1;

#ifdef AER_MPI
  //communicator group to simulate a circuit (for multi-experiments)
  MPI_Comm distributed_comm_;
#endif

  uint_t mapped_index(const uint_t idx);
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
  gpu_optimization_ = false;

#ifdef AER_MPI
  distributed_comm_ = MPI_COMM_WORLD;
#endif
}

template <class state_t>
StateChunk<state_t>::~StateChunk(void)
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
void StateChunk<state_t>::allocate(uint_t num_qubits,uint_t block_bits)
{
  int_t i;
  uint_t nchunks;

  num_qubits_ = num_qubits;
  block_bits_ = block_bits;

  if(block_bits_ > 0){
    chunk_bits_ = block_bits_;
    if(chunk_bits_ > num_qubits_){
      chunk_bits_ = num_qubits_;
    }
  }
  else{
    chunk_bits_ = num_qubits_;
  }

  num_global_chunks_ = 1ull << ((num_qubits_ - chunk_bits_)*qubit_scale());

  chunk_index_begin_.resize(distributed_procs_);
  chunk_index_end_.resize(distributed_procs_);
  for(i=0;i<distributed_procs_;i++){
    chunk_index_begin_[i] = num_global_chunks_*i / distributed_procs_;
    chunk_index_end_[i] = num_global_chunks_*(i+1) / distributed_procs_;
  }

  num_local_chunks_ = chunk_index_end_[distributed_rank_] - chunk_index_begin_[distributed_rank_];
  global_chunk_index_ = chunk_index_begin_[distributed_rank_];

  qregs_.resize(num_local_chunks_);

  gpu_optimization_ = false;
  chunk_omp_parallel_ = false;
  if(qregs_[0].name().find("gpu") != std::string::npos){
    if(chunk_bits_ < num_qubits_){
      chunk_omp_parallel_ = true;   //CUDA backend requires thread parallelization of chunk loop
    }
    gpu_optimization_ = true;
  }

  nchunks = num_local_chunks_;
  for(i=0;i<num_local_chunks_;i++){
    uint_t gid = i + global_chunk_index_;
    qregs_[i].chunk_setup(chunk_bits_*qubit_scale(),num_qubits_*qubit_scale(),gid,nchunks);

    //only first one allocates chunks, others only set chunk index
    nchunks = 0;
  }

  //initialize qubit map
  qubit_map_.resize(num_qubits_);
  for(i=0;i<num_qubits_;i++){
    qubit_map_[i] = i;
  }
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
bool StateChunk<state_t>::is_applied_to_each_chunk(const Operations::Op &op)
{
  if(op.type == Operations::OpType::gate || op.type == Operations::OpType::matrix || 
            op.type == Operations::OpType::diagonal_matrix || op.type == Operations::OpType::multiplexer ||
            op.type == Operations::OpType::superop){
    return true;
  }
  return false;
}

template <class state_t>
void StateChunk<state_t>::apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentResult &result,
                         RngEngine &rng,
                         bool final_ops)
{
  int_t iChunk;
  uint_t iOp,nOp;

  nOp = ops.size();
  iOp = 0;
  while(iOp < nOp){
    if(ops[iOp].type == Operations::OpType::gate && ops[iOp].name == "swap_chunk"){
      //apply swap between chunks
      apply_chunk_swap(ops[iOp].qubits);
    }
    else if(ops[iOp].type == Operations::OpType::sim_op && ops[iOp].name == "begin_blocking"){
      //applying sequence of gates inside each chunk

      uint_t iOpEnd = iOp;
      while(iOpEnd < nOp){
        if(ops[iOpEnd].type == Operations::OpType::sim_op && ops[iOpEnd].name == "end_blocking"){
          break;
        }
        iOpEnd++;
      }

      uint_t iOpBegin = iOp + 1;
#pragma omp parallel for if(chunk_omp_parallel_) private(iChunk) 
      for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
        uint_t iOpBlock = iOpBegin;
        //fecth chunk in cache
        if(qregs_[iChunk].fetch_chunk()){
          while(iOpBlock < iOpEnd){
            apply_op(iChunk,ops[iOpBlock],result,rng,final_ops);
            iOpBlock++;
          }

          //release chunk from cache
          qregs_[iChunk].release_chunk();
        }
      }

      iOp = iOpEnd;
    }
    else if(is_applied_to_each_chunk(ops[iOp])){
#pragma omp parallel for if(chunk_omp_parallel_) private(iChunk) 
      for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
        apply_op(iChunk,ops[iOp],result,rng,final_ops && nOp == iOp + 1);
      }
    }
    else{
      //parallelize inside state implementations
      apply_op(-1,ops[iOp],result,rng,final_ops && nOp == iOp + 1);
    }
    iOp++;
  }
}

template <class state_t>
void StateChunk<state_t>::block_diagonal_matrix(const int_t iChunk, reg_t &qubits, cvector_t &diag)
{
  uint_t gid = global_chunk_index_ + iChunk;
  uint_t i;
  uint_t mask_out = 0;
  uint_t mask_id = 0;

  reg_t qubits_in;
  cvector_t diag_in;

  for(i=0;i<qubits.size();i++){
    if(qubits[i] < chunk_bits_){ //in chunk
      qubits_in.push_back(qubits[i]);
    }
    else{
      mask_out |= (1ull << i);
      if((gid >> (qubits[i] - chunk_bits_)) & 1)
        mask_id |= (1ull << i);
    }
  }

  if(qubits_in.size() < qubits.size()){
    for(i=0;i<diag.size();i++){
      if((i & mask_out) == mask_id)
        diag_in.push_back(diag[i]);
    }

    if(qubits_in.size() == 0){
      qubits_in.push_back(0);
      diag_in.resize(2);
      diag_in[1] = diag_in[0];
    }
    qubits = qubits_in;
    diag = diag_in;
  }
}

template <class state_t>
void StateChunk<state_t>::qubits_inout(const reg_t& qubits, reg_t& qubits_in,reg_t& qubits_out) const
{
  int_t i;
  qubits_in.clear();
  qubits_out.clear();
  for(i=0;i<qubits.size();i++){
    if(qubits[i] < chunk_bits_){ //in chunk
      qubits_in.push_back(qubits[i]);
    }
    else{
      qubits_out.push_back(qubits[i]);
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
  creg_.initialize(num_memory, num_register);
}


template <class state_t>
void StateChunk<state_t>::initialize_creg(uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) 
{
  creg_.initialize(num_memory, num_register, memory_hex, register_hex);
}

template <class state_t>
template <typename list_t>
void StateChunk<state_t>::initialize_from_vector(const list_t &vec)
{
  int_t iChunk;
  if(chunk_bits_ == num_qubits_){
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      qregs_[iChunk].initialize_from_vector(vec);
    }
  }
  else{   //multi-chunk distribution
#pragma omp parallel for if(chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      list_t tmp(1ull << (chunk_bits_*qubit_scale()));
      for(int_t i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
        tmp[i] = vec[((global_chunk_index_ + iChunk) << (chunk_bits_*qubit_scale())) + i];
      }
      qregs_[iChunk].initialize_from_vector(tmp);
    }
  }
}

template <class state_t>
template <typename list_t>
void StateChunk<state_t>::initialize_from_matrix(const list_t &mat)
{
  int_t iChunk;
  if(chunk_bits_ == num_qubits_){
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      qregs_[iChunk].initialize_from_matrix(mat);
    }
  }
  else{   //multi-chunk distribution
#pragma omp parallel for if(chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      list_t tmp(1ull << (chunk_bits_),1ull << (chunk_bits_));
      uint_t irow_chunk = ((iChunk + global_chunk_index_) >> ((num_qubits_ - chunk_bits_))) << (chunk_bits_);
      uint_t icol_chunk = ((iChunk + global_chunk_index_) & ((1ull << ((num_qubits_ - chunk_bits_)))-1)) << (chunk_bits_);

      //copy part of state for this chunk
      uint_t i,row,col;
      for(i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
        uint_t icol = i & ((1ull << chunk_bits_)-1);
        uint_t irow = i >> chunk_bits_;
        tmp[i] = mat[icol_chunk + icol + ((irow_chunk + irow) << num_qubits_)];
      }
      qregs_[iChunk].initialize_from_matrix(tmp);
    }
  }
}

template <class state_t>
auto StateChunk<state_t>::apply_to_matrix(bool copy)
{
  int_t iChunk;
  uint_t size = 1ull << (chunk_bits_*qubit_scale());
  uint_t mask = (1ull << (chunk_bits_)) - 1;
  uint_t num_threads = qregs_[0].get_omp_threads();

  auto matrix = qregs_[0].copy_to_matrix();

  if(distributed_rank_ == 0){
    //TO DO check memory availability
    matrix.resize(1ull << (num_qubits_),1ull << (num_qubits_));

    auto tmp = qregs_[0].copy_to_matrix();
    for(iChunk=0;iChunk<num_global_chunks_;iChunk++){
      int_t i;
      uint_t irow_chunk = (iChunk >> ((num_qubits_ - chunk_bits_))) << chunk_bits_;
      uint_t icol_chunk = (iChunk & ((1ull << ((num_qubits_ - chunk_bits_)))-1)) << chunk_bits_;

      if(iChunk < num_local_chunks_){
        if(copy)
          tmp = qregs_[iChunk].copy_to_matrix();
        else
          tmp = qregs_[iChunk].move_to_matrix();
      }
#ifdef AER_MPI
      else
        recv_data(tmp.data(),size,0,iChunk);
#endif
#pragma omp parallel for if(num_threads > 1) num_threads(num_threads)
      for(i=0;i<size;i++){
        uint_t irow = i >> (chunk_bits_);
        uint_t icol = i & mask;
        uint_t idx = ((irow+irow_chunk) << (num_qubits_)) + icol_chunk + icol;
        matrix[idx] = tmp[i];
      }
    }
  }
  else{
#ifdef AER_MPI
    //send matrices to process 0
    for(iChunk=0;iChunk<num_global_chunks_;iChunk++){
      uint_t iProc = get_process_by_chunk(iChunk);
      if(iProc == distributed_rank_){
        if(copy){
          auto tmp = qregs_[iChunk-global_chunk_index_].copy_to_matrix();
          send_data(tmp.data(),size,iChunk,0);
        }
        else{
          auto tmp = qregs_[iChunk-global_chunk_index_].move_to_matrix();
          send_data(tmp.data(),size,iChunk,0);
        }
      }
    }
#endif
  }

  return matrix;
}

template <class state_t>
void StateChunk<state_t>::save_creg(ExperimentResult &result,
                               const std::string &key,
                               DataSubType type) const {
  if (creg_.memory_size() == 0)
    return;
  switch (type) {
    case DataSubType::list:
      result.data.add_list(creg_.memory_hex(), key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(1ULL, key, creg_.memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid creg data subtype for data key: " + key);
  }
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_average(ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum,
                                       DataSubType type) const {
  switch (type) {
    case DataSubType::list:
      result.data.add_list(datum, key);
      break;
    case DataSubType::c_list:
      result.data.add_list(datum, key, creg_.memory_hex());
      break;
    case DataSubType::accum:
      result.data.add_accum(datum, key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(datum, key, creg_.memory_hex());
      break;
    case DataSubType::average:
      result.data.add_average(datum, key);
      break;
    case DataSubType::c_average:
      result.data.add_average(datum, key, creg_.memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid average data subtype for data key: " + key);
  }
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_average(ExperimentResult &result,
                                       const std::string &key,
                                       T&& datum,
                                       DataSubType type) const {
  switch (type) {
    case DataSubType::list:
      result.data.add_list(std::move(datum), key);
      break;
    case DataSubType::c_list:
      result.data.add_list(std::move(datum), key, creg_.memory_hex());
      break;
    case DataSubType::accum:
      result.data.add_accum(std::move(datum), key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(std::move(datum), key, creg_.memory_hex());
      break;
    case DataSubType::average:
      result.data.add_average(std::move(datum), key);
      break;
    case DataSubType::c_average:
      result.data.add_average(std::move(datum), key, creg_.memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid average data subtype for data key: " + key);
  }
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_pershot(ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum,
                                       DataSubType type) const {
  switch (type) {
  case DataSubType::single:
    result.data.add_single(datum, key);
    break;
  case DataSubType::c_single:
    result.data.add_single(datum, key, creg_.memory_hex());
    break;
  case DataSubType::list:
    result.data.add_list(datum, key);
    break;
  case DataSubType::c_list:
    result.data.add_list(datum, key, creg_.memory_hex());
    break;
  default:
    throw std::runtime_error("Invalid pershot data subtype for data key: " + key);
  }
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_pershot(ExperimentResult &result, 
                                       const std::string &key,
                                       T&& datum,
                                       DataSubType type) const {
  switch (type) {
    case DataSubType::single:
      result.data.add_single(std::move(datum), key);
      break;
    case DataSubType::c_single:
      result.data.add_single(datum, key, creg_.memory_hex());
      break;
    case DataSubType::list:
      result.data.add_list(std::move(datum), key);
      break;
    case DataSubType::c_list:
      result.data.add_list(std::move(datum), key, creg_.memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid pershot data subtype for data key: " + key);
  }
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_single(ExperimentResult &result,
                                      const std::string &key,
                                      const T& datum) const {
  result.data.add_single(datum, key);
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_single(ExperimentResult &result,
                                      const std::string &key,
                                      T&& datum) const {
  result.data.add_single(std::move(datum), key);
}

template <class state_t>
void StateChunk<state_t>::snapshot_state(const Operations::Op &op,
                                    ExperimentResult &result,
                                    std::string name) const 
{
  name = (name.empty()) ? op.name : name;

  //TO DO : gather qregs over processes
  int_t i;
  for(i=0;i<qregs_.size();i++){
    result.legacy_data.add_pershot_snapshot(name, op.string_params[0], qregs_[i]);
  }
}


template <class state_t>
void StateChunk<state_t>::snapshot_creg_memory(const Operations::Op &op,
                                          ExperimentResult &result,
                                          std::string name) const 
{
  result.legacy_data.add_pershot_snapshot(name,
                               op.string_params[0],
                               creg_.memory_hex());
}


template <class state_t>
void StateChunk<state_t>::snapshot_creg_register(const Operations::Op &op,
                                            ExperimentResult &result,
                                            std::string name) const 
{
  result.legacy_data.add_pershot_snapshot(name,
                               op.string_params[0],
                               creg_.register_hex());
}


template <class state_t>
void StateChunk<state_t>::apply_save_expval(const Operations::Op &op,
                                            ExperimentResult &result){
  // Check empty edge case
  if (op.expval_params.empty()) {
    throw std::invalid_argument(
        "Invalid save expval instruction (Pauli components are empty).");
  }
  bool variance = (op.type == Operations::OpType::save_expval_var);

  // Accumulate expval components
  double expval(0.);
  double sq_expval(0.);

  for (const auto &param : op.expval_params) {
    // param is tuple (pauli, coeff, sq_coeff)
    const auto val = expval_pauli(op.qubits, std::get<0>(param));
    expval += std::get<1>(param) * val;
    if (variance) {
      sq_expval += std::get<2>(param) * val;
    }
  }
  if (variance) {
    std::vector<double> expval_var(2);
    expval_var[0] = expval;  // mean
    expval_var[1] = sq_expval - expval * expval;  // variance
    save_data_average(result, op.string_params[0], expval_var, op.save_type);
  } else {
    save_data_average(result, op.string_params[0], expval, op.save_type);
  }
}

template <class state_t>
uint_t StateChunk<state_t>::mapped_index(const uint_t idx)
{
  uint_t i,ret = 0;
  uint_t t = idx;

  for(i=0;i<num_qubits_;i++){
    if(t & 1){
      ret |= (1ull << qubit_map_[i]);
    }
    t >>= 1;
  }
  return ret;
}

template <class state_t>
void StateChunk<state_t>::apply_chunk_swap(const reg_t &qubits)
{
  uint_t nLarge = 1;
  uint_t q0,q1;
  int_t iChunk;

  q0 = qubits[qubits.size() - 2];
  q1 = qubits[qubits.size() - 1];

  if(qubit_scale() == 1){
    std::swap(qubit_map_[q0],qubit_map_[q1]);
  }
    
  if(q0 > q1){
    std::swap(q0,q1);
  }

  if(q1 < chunk_bits_*qubit_scale()){
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

    if(q0 < chunk_bits_*qubit_scale())
      nLarge = 1;
    else
      nLarge = 2;

    mask0 = (1ull << q0);
    mask1 = (1ull << q1);
    mask0 >>= (chunk_bits_*qubit_scale());
    mask1 >>= (chunk_bits_*qubit_scale());

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

    if(distributed_procs_ == 1 || (proc_bits >= 0 && q1 < (num_qubits_*qubit_scale() - proc_bits))){   //no data transfer between processes is needed
      if(q0 < chunk_bits_*qubit_scale()){
        nPair = num_local_chunks_ >> 1;
      }
      else{
        nPair = num_local_chunks_ >> 2;
      }

#pragma omp parallel for if(chunk_omp_parallel_) private(iPair,baseChunk,iChunk1,iChunk2)
      for(iPair=0;iPair<nPair;iPair++){
        if(q0 < chunk_bits_*qubit_scale()){
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

      if(q0 < chunk_bits_*qubit_scale()){
        nLarge = 1;
        nu[0] = 1ull << (q1 - chunk_bits_*qubit_scale());
        ub[0] = 0;
        iu[0] = 0;

        nu[1] = 1ull << (num_qubits_*qubit_scale() - q1 - 1);
        ub[1] = (q1 - chunk_bits_*qubit_scale()) + 1;
        iu[1] = 0;
      }
      else{
        nLarge = 2;
        nu[0] = 1ull << (q0 - chunk_bits_*qubit_scale());
        ub[0] = 0;
        iu[0] = 0;

        nu[1] = 1ull << (q1 - q0 - 1);
        ub[1] = (q0 - chunk_bits_*qubit_scale()) + 1;
        iu[1] = 0;

        nu[2] = 1ull << (num_qubits_*qubit_scale() - q1 - 1);
        ub[2] = (q1 - chunk_bits_*qubit_scale()) + 1;
        iu[2] = 0;
      }
      nPair = 1ull << (num_qubits_*qubit_scale() - chunk_bits_*qubit_scale() - nLarge);

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

        auto pSend = qregs_[iLocalChunk - global_chunk_index_].send_buffer(sizeSend);
        MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,iPair,distributed_comm_,&reqSend);

        auto pRecv = qregs_[iLocalChunk - global_chunk_index_].recv_buffer(sizeRecv);
        MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,iPair,distributed_comm_,&reqRecv);

        MPI_Wait(&reqSend,&st);
        MPI_Wait(&reqRecv,&st);

        qregs_[iLocalChunk - global_chunk_index_].apply_chunk_swap(qubits,iRemoteChunk);
      }
    }
#endif

  }
}

template <class state_t>
void StateChunk<state_t>::apply_chunk_x(const uint_t qubit)
{
  int_t iChunk;
  uint_t nLarge = 1;


  if(qubit < chunk_bits_*qubit_scale()){
    reg_t qubits(1,qubit);
#pragma omp parallel for if(chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      qregs_[iChunk].apply_mcx(qubits);
    }
  }
  else{ //exchange over chunks
    int_t iPair;
    uint_t nPair,mask;
    uint_t baseChunk,iChunk1,iChunk2;
    reg_t qubits(2);
    qubits[0] = qubit;
    qubits[1] = qubit;

    mask = (1ull << qubit);
    mask >>= (chunk_bits_*qubit_scale());

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

    if(distributed_procs_ == 1 || (proc_bits >= 0 && qubit < (num_qubits_*qubit_scale() - proc_bits))){   //no data transfer between processes is needed
      nPair = num_local_chunks_ >> 1;

#pragma omp parallel for if(chunk_omp_parallel_) private(iPair,baseChunk,iChunk1,iChunk2)
      for(iPair=0;iPair<nPair;iPair++){
        baseChunk = iPair & (mask-1);
        baseChunk += ((iPair - baseChunk) << 1);

        iChunk1 = baseChunk;
        iChunk2 = baseChunk | mask;

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

      nLarge = 1;
      nu[0] = 1ull << (qubit - chunk_bits_*qubit_scale());
      ub[0] = 0;
      iu[0] = 0;

      nu[1] = 1ull << (num_qubits_*qubit_scale() - qubit - 1);
      ub[1] = (qubit - chunk_bits_*qubit_scale()) + 1;
      iu[1] = 0;
      nPair = 1ull << (num_qubits_*qubit_scale() - chunk_bits_*qubit_scale() - 1);

      for(iPair=0;iPair<nPair;iPair++){
        //calculate index of pair of chunks
        baseChunk = 0;
        add = 1;
        for(i=1;i>=0;i--){
          baseChunk += (iu[i] << ub[i]);
          //update for next
          iu[i] += add;
          add = 0;
          if(iu[i] >= nu[i]){
            iu[i] = 0;
            add = 1;
          }
        }

        iChunk1 = baseChunk;
        iChunk2 = baseChunk | mask;

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

        auto pSend = qregs_[iLocalChunk - global_chunk_index_].send_buffer(sizeSend);
        MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,iPair,distributed_comm_,&reqSend);

        auto pRecv = qregs_[iLocalChunk - global_chunk_index_].recv_buffer(sizeRecv);
        MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,iPair,distributed_comm_,&reqRecv);

        MPI_Wait(&reqSend,&st);
        MPI_Wait(&reqRecv,&st);

        qregs_[iLocalChunk - global_chunk_index_].apply_chunk_swap(qubits,iRemoteChunk);
      }
    }
#endif

  }
}

template <class state_t>
void StateChunk<state_t>::send_chunk(uint_t local_chunk_index, uint_t global_pair_index)
{
#ifdef AER_MPI
  MPI_Request reqSend;
  MPI_Status st;
  uint_t sizeSend;
  uint_t iProc;

  iProc = get_process_by_chunk(global_pair_index);

  auto pSend = qregs_[local_chunk_index].send_buffer(sizeSend);
  MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,local_chunk_index + global_chunk_index_,distributed_comm_,&reqSend);

  MPI_Wait(&reqSend,&st);

  qregs_[local_chunk_index].release_send_buffer();
#endif
}

template <class state_t>
void StateChunk<state_t>::recv_chunk(uint_t local_chunk_index, uint_t global_pair_index)
{
#ifdef AER_MPI
  MPI_Request reqRecv;
  MPI_Status st;
  uint_t sizeRecv;
  uint_t iProc;

  iProc = get_process_by_chunk(global_pair_index);

  auto pRecv = qregs_[local_chunk_index].recv_buffer(sizeRecv);
  MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,global_pair_index,distributed_comm_,&reqRecv);

  MPI_Wait(&reqRecv,&st);
#endif
}

template <class state_t>
template <class data_t>
void StateChunk<state_t>::send_data(data_t* pSend, uint_t size, uint_t myid,uint_t pairid)
{
#ifdef AER_MPI
  MPI_Request reqSend;
  MPI_Status st;
  uint_t iProc;

  iProc = get_process_by_chunk(pairid);

  MPI_Isend(pSend,size*sizeof(data_t),MPI_BYTE,iProc,myid,distributed_comm_,&reqSend);

  MPI_Wait(&reqSend,&st);
#endif
}

template <class state_t>
template <class data_t>
void StateChunk<state_t>::recv_data(data_t* pRecv, uint_t size, uint_t myid,uint_t pairid)
{
#ifdef AER_MPI
  MPI_Request reqRecv;
  MPI_Status st;
  uint_t iProc;

  iProc = get_process_by_chunk(pairid);

  MPI_Irecv(pRecv,size*sizeof(data_t),MPI_BYTE,iProc,pairid,distributed_comm_,&reqRecv);

  MPI_Wait(&reqRecv,&st);
#endif
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
template <class data_t>
void StateChunk<state_t>::gather_state(std::vector<std::complex<data_t>>& state)
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
        MPI_Irecv(&state[offset],size*sizeof(std::complex<data_t>),MPI_BYTE,i,i*2+1,distributed_comm_,&reqRecv);
        MPI_Wait(&reqRecv,&st);
        offset += size;
      }
    }
    else{
      MPI_Isend(&local_size,1,MPI_UINT64_T,0,i*2,distributed_comm_,&reqSend);
      MPI_Wait(&reqSend,&st);
      MPI_Isend(&state[0],local_size*sizeof(std::complex<data_t>),MPI_BYTE,0,i*2+1,distributed_comm_,&reqSend);
      MPI_Wait(&reqSend,&st);
    }
  }
#endif
}

template <class state_t>
template <class data_t>
void StateChunk<state_t>::gather_state(AER::Vector<std::complex<data_t>>& state)
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
        MPI_Irecv(state.data() + offset,size*sizeof(std::complex<data_t>),MPI_BYTE,i,i*2+1,distributed_comm_,&reqRecv);
        MPI_Wait(&reqRecv,&st);
        offset += size;
      }
    }
    else{
      MPI_Isend(&local_size,1,MPI_UINT64_T,0,i*2,distributed_comm_,&reqSend);
      MPI_Wait(&reqSend,&st);
      MPI_Isend(state.data(),local_size*sizeof(std::complex<data_t>),MPI_BYTE,0,i*2+1,distributed_comm_,&reqSend);
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
