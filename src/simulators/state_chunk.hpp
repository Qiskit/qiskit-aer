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

#ifndef _aer_base_state_chunk_hpp_
#define _aer_base_state_chunk_hpp_

#include "framework/json.hpp"
#include "framework/opset.hpp"
#include "framework/types.hpp"
#include "framework/creg.hpp"
#include "framework/results/experiment_result.hpp"

#include "noise/noise_model.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

namespace AER {

namespace Base {

#define STATE_APPLY_TO_ALL_CHUNKS     0

//=========================================================================
// StateChunk interface base class with multiple chunks for Qiskit-Aer
// The base state class that supports multi-chunk distribution/ multi-shot parallelization 
//=========================================================================

template <class state_t>
class StateChunk : public State<state_t> {

public:
  using ignore_argument = void;
  using BaseState = State<state_t>;
  using DataSubType = Operations::DataSubType;
  using OpType = Operations::OpType;

  //-----------------------------------------------------------------------
  // Constructors
  //-----------------------------------------------------------------------

  // The constructor arguments are used to initialize the OpSet
  // for the StateChunk class for checking supported simulator Operations
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

  StateChunk(const Operations::OpSet &opset) : BaseState(opset)
  {
    num_global_chunks_ = 0;
    num_local_chunks_ = 0;

    myrank_ = 0;
    nprocs_ = 1;

    distributed_procs_ = 1;
    distributed_rank_ = 0;
    distributed_group_ = 0;

    chunk_omp_parallel_ = false;
    thrust_optimization_ = false;

#ifdef AER_MPI
    distributed_comm_ = MPI_COMM_WORLD;
#endif
  }

  virtual ~StateChunk();

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

  // Return the state qreg object
  auto &qreg(int_t idx=0) { return qregs_[idx]; }
  const auto &qreg(int_t idx=0) const { return qregs_[idx]; }

  // Return the state creg object
  auto &creg(uint_t idx=0) { return cregs_[idx]; }
  const auto &creg(uint_t idx=0) const { return cregs_[idx]; }

  //=======================================================================
  // Subclass Override Methods
  //
  // The following methods should be implemented by any StateChunk subclasses.
  // Abstract methods are required, while some methods are optional for
  // StateChunk classes that support measurement to be compatible with a general
  // QasmController.
  //=======================================================================

  //-----------------------------------------------------------------------
  // Abstract methods
  //
  // The implementation of these methods must be defined in all subclasses
  //-----------------------------------------------------------------------
  
  // Return a string name for the StateChunk type
  virtual std::string name() const = 0;

  // Initializes the StateChunk to the default state.
  // Typically this is the n-qubit all |0> state
  virtual void initialize_qreg(uint_t num_qubits) = 0;

  // Initializes the StateChunk to a specific state.
  virtual void initialize_qreg(uint_t num_qubits, const state_t &state) = 0;

  // Return an estimate of the required memory for implementing the
  // specified sequence of operations on a `num_qubit` sized StateChunk.
  virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops)
                                    const = 0;

  //memory allocation (previously called before inisitalize_qreg)
  virtual bool allocate(uint_t num_qubits,uint_t block_bits,uint_t num_parallel_shots = 1);

  // Return the expectation value of a N-qubit Pauli operator
  // If the simulator does not support Pauli expectation value this should
  // raise an exception.
  double expval_pauli(const reg_t &qubits,const std::string& pauli) override final {return 0.0;}

  virtual double expval_pauli(const int_t iChunk, const reg_t &qubits,
                              const std::string& pauli) = 0;

  //-----------------------------------------------------------------------
  // Optional: Load config settings
  //-----------------------------------------------------------------------

  // Load any settings for the StateChunk class from a config JSON
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
  // This method is only required for a StateChunk subclass to be compatible with
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
  // These methods should not be modified in any StateChunk subclasses
  //=======================================================================

  //-----------------------------------------------------------------------
  // Apply circuits and ops
  //-----------------------------------------------------------------------

  // Apply a single operation
  // The `final_op` flag indicates no more instructions will be applied
  // to the state after this sequence, so the state can be modified at the
  // end of the instructions.

  //this is not used for StateChunk
  void apply_op(const Operations::Op &op,
                      ExperimentResult &result,
                      RngEngine& rng,
                      bool final_op = false) override final {}

  //so this one is used
  virtual void apply_op(const int_t iChunk, const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine& rng,
                        bool final_op = false) = 0;


  // Apply a sequence of operations to the current state of the StateChunk class.
  // It is up to the StateChunk subclass to decide how this sequence should be
  // executed (ie in sequence, or some other execution strategy.)
  // If this sequence contains operations not in the supported opset
  // an exeption will be thrown.
  // The `final_ops` flag indicates no more instructions will be applied
  // to the state after this sequence, so the state can be modified at the
  // end of the instructions.
  template <typename InputIterator>
  void apply_ops(InputIterator first,
                 InputIterator last,
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
                 bool final_ops = false);

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------
  template <typename list_t>
  void initialize_from_vector(const int_t iChunk, const list_t &vec);

  template <typename list_t>
  void initialize_from_matrix(const int_t iChunk, const list_t &mat);

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
  void save_creg(const int_t iChunk, ExperimentResult &result,
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
  void save_data_average(const int_t iChunk, ExperimentResult &result,
                         const std::string &key, const T& datum,
                         OpType type, DataSubType subtype = DataSubType::average) const;

  template <class T>
  void save_data_average(const int_t iChunk, ExperimentResult &result,
                         const std::string &key, T&& datum,
                         OpType type, DataSubType subtype = DataSubType::average) const;
  
  // Save data type which is pershot and does not support accumulator or average
  // This supports DataSubTypes: single, c_single, list, c_list
  template <class T>
  void save_data_pershot(const int_t iChunk, ExperimentResult &result,
                         const std::string &key, const T& datum,
                         OpType type, DataSubType subtype = DataSubType::list) const;

  template <class T>
  void save_data_pershot(const int_t iChunk, ExperimentResult &result,
                         const std::string &key, T&& datum,
                         OpType type, DataSubType subtype = DataSubType::list) const;


  //save creg as count data 
  virtual void save_count_data(ExperimentResult& result,bool save_memory);

  //-----------------------------------------------------------------------
  // Common instructions
  //-----------------------------------------------------------------------
 
  // Apply a save expectation value instruction
  void apply_save_expval(const int_t iChunk, const Operations::Op &op, ExperimentResult &result);

  //-----------------------------------------------------------------------
  // Standard snapshots
  //-----------------------------------------------------------------------

  // Snapshot the current statevector (single-shot)
  // if type_label is the empty string the operation type will be used for the type
  virtual void snapshot_state(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                      std::string name = "") const;

  // Snapshot the classical memory bits state (single-shot)
  void snapshot_creg_memory(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                            std::string name = "memory") const;

  // Snapshot the classical register bits state (single-shot)
  void snapshot_creg_register(const int_t iChunk, const Operations::Op &op, ExperimentResult &result,
                              std::string name = "register") const;


  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------


  //set number of processes to be distributed
  virtual void set_distribution(uint_t nprocs);

  //set max number of shots to execute in a batch
  void set_max_bached_shots(uint_t shots)
  {
    max_batched_shots_ = shots;
  }


  //Does this state support multi-chunk distribution?
  virtual bool multi_chunk_distribution_supported(void){return true;}
  //Does this state support multi-shot parallelization?
  virtual bool multi_shot_parallelization_supported(void){return true;}

protected:

  // The array of the quantum state data structure
  std::vector<state_t> qregs_;

  // The array of classical register data
  std::vector<ClassicalRegister> cregs_;

  //number of qubits for the circuit
  uint_t num_qubits_;

  //extra parameters for parallel simulations
  uint_t num_global_chunks_;    //number of total chunks 
  uint_t num_local_chunks_;     //number of local chunks
  uint_t chunk_bits_;           //number of qubits per chunk
  uint_t block_bits_;           //number of cache blocked qubits

  uint_t global_chunk_index_;   //beginning chunk index for this process
  reg_t chunk_index_begin_;     //beginning chunk index for each process
  reg_t chunk_index_end_;       //ending chunk index for each process
  uint_t local_shot_index_;    //local shot ID of current batch loop

  uint_t myrank_;               //process ID
  uint_t nprocs_;               //number of processes
  uint_t distributed_rank_;     //process ID in communicator group
  uint_t distributed_procs_;    //number of processes in communicator group
  uint_t distributed_group_;    //group id of distribution

  bool chunk_omp_parallel_;     //using thread parallel to process loop of chunks or not
  bool thrust_optimization_;       //optimization for Thrust implementation

  bool multi_chunk_distribution_ = false; //distributing chunks to apply cache blocking parallelization
  bool multi_shots_parallelization_ = false; //using chunks as multiple shots parallelization
  bool set_parallelization_called_ = false;    //this flag is used to check set_parallelization is already called, if yes the call sets max_batched_shots_
  uint_t max_batched_shots_ = 1;    //max number of shots can be stored on available memory

  reg_t qubit_map_;             //qubit map to restore swapped qubits

  //group of states (GPU devices)
  uint_t num_groups_;            //number of groups of chunks
  reg_t top_chunk_of_group_;
  reg_t num_chunks_in_group_;

  //-----------------------------------------------------------------------
  // Apply circuits and ops
  //-----------------------------------------------------------------------
  //apply ops for multi-chunk distribution
  template <typename InputIterator>
  void apply_ops_chunks(InputIterator first,
                 InputIterator last,
                 ExperimentResult &result,
                 RngEngine &rng,
                 bool final_ops = false);

  //apply cache blocked ops in each chunk
  template <typename InputIterator>
  void apply_cache_blocking_ops(const int_t iChunk, InputIterator first,
                 InputIterator last,
                 ExperimentResult &result,
                 RngEngine &rng);

  //apply ops for multi-shots to one group
  template <typename InputIterator>
  void apply_ops_multi_shots_for_group(int_t i_group, 
                               InputIterator first, InputIterator last,
                               const Noise::NoiseModel &noise,
                               ExperimentResult &result,
                               uint_t rng_seed,
                               bool final_ops);

  //apply op to multiple shots , return flase if op is not supported to execute in a batch
  virtual bool apply_batched_op(const int_t iChunk, const Operations::Op &op,
                                ExperimentResult &result,
                                std::vector<RngEngine> &rng,
                                bool final_op = false){return false;}

  //apply sampled noise to multiple-shots (this is used for ops contains non-Pauli operators)
  void apply_batched_noise_ops(const int_t i_group, const std::vector<std::vector<Operations::Op>> &ops, 
                               ExperimentResult &result,
                               std::vector<RngEngine> &rng);

  //check conditional
  bool check_conditional(const int_t iChunk, const Operations::Op &op);

  //this function is used to scale chunk qubits for multi-chunk distribution
  virtual int qubit_scale(void)
  {
    return 1;     //scale of qubit number (x2 for density and unitary matrices)
  }
  uint_t get_process_by_chunk(uint_t cid);

  //allocate qregs
  bool allocate_qregs(uint_t num_chunks);


  //-----------------------------------------------------------------------
  //Functions for multi-chunk distribution
  //-----------------------------------------------------------------------
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
  void reduce_sum(reg_t& sum) const;
  void reduce_sum(rvector_t& sum) const;
  void reduce_sum(complex_t& sum) const;
  void reduce_sum(double& sum) const;

  //gather values on each process
  void gather_value(rvector_t& val) const;

  //gather cregs 
  void gather_creg_memory(void);

  //barrier all processes
  void sync_process(void) const;

  //gather distributed state into vector (if memory is enough)
  template <class data_t>
  void gather_state(std::vector<std::complex<data_t>>& state);

  template <class data_t>
  void gather_state(AER::Vector<std::complex<data_t>>& state);

  // block diagonal matrix in chunk
  void block_diagonal_matrix(const int_t iChunk, reg_t &qubits, cvector_t &diag);
  void qubits_inout(const reg_t& qubits, reg_t& qubits_in,reg_t& qubits_out) const;

  //collect matrix over multiple chunks
  auto apply_to_matrix(bool copy = false);

  // Apply the global phase
  virtual void apply_global_phase(){}

  //check if the operator should be applied to each chunk
  virtual bool is_applied_to_each_chunk(const Operations::Op &op);

  //return global shot index for the chunk
  inline int_t get_global_shot_index(const int_t iChunk) const
  {
    return multi_shots_parallelization_ ? (iChunk + local_shot_index_ + global_chunk_index_) : 0;
  }

#ifdef AER_MPI
  //communicator group to simulate a circuit (for multi-experiments)
  MPI_Comm distributed_comm_;
#endif

  uint_t mapped_index(const uint_t idx);

};


//=========================================================================
// Implementations
//=========================================================================

template <class state_t>
StateChunk<state_t>::~StateChunk(void)
{
#ifdef AER_MPI
  if(distributed_comm_ != MPI_COMM_WORLD){
    MPI_Comm_free(&distributed_comm_);
  }
#endif
}

template <class state_t>
void StateChunk<state_t>::set_config(const json_t &config) {
  (ignore_argument)config;
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
bool StateChunk<state_t>::allocate(uint_t num_qubits,uint_t block_bits,uint_t num_parallel_shots)
{
  int_t i;
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

  if(chunk_bits_ < num_qubits_){
    //multi-chunk distribution with cache blocking transpiler
    multi_chunk_distribution_ = true;
    multi_shots_parallelization_ = false;
    num_global_chunks_ = 1ull << ((num_qubits_ - chunk_bits_)*qubit_scale());

    cregs_.resize(1);
  }
  else{
    //multi-shots parallelization
    multi_chunk_distribution_ = false;
    if(num_parallel_shots > 1)
      multi_shots_parallelization_ = true;
    else
      multi_shots_parallelization_ = false;
    num_global_chunks_ = num_parallel_shots;

    //classical registers for all shots
    cregs_.resize(num_parallel_shots);
  }

  chunk_index_begin_.resize(distributed_procs_);
  chunk_index_end_.resize(distributed_procs_);
  for(i=0;i<distributed_procs_;i++){
    chunk_index_begin_[i] = num_global_chunks_*i / distributed_procs_;
    chunk_index_end_[i] = num_global_chunks_*(i+1) / distributed_procs_;
  }

  num_local_chunks_ = chunk_index_end_[distributed_rank_] - chunk_index_begin_[distributed_rank_];
  global_chunk_index_ = chunk_index_begin_[distributed_rank_];
  local_shot_index_ = 0;

  if(multi_shots_parallelization_){
    allocate_qregs(std::min(num_local_chunks_,max_batched_shots_));
  }
  else{
    allocate_qregs(num_local_chunks_);
  }

  thrust_optimization_ = false;
  chunk_omp_parallel_ = false;
  if(qregs_[0].name().find("gpu") != std::string::npos){
#ifdef _OPENMP
    if(multi_chunk_distribution_){
      if(omp_get_num_threads() == 1)
        chunk_omp_parallel_ = true;
    }
#endif
    thrust_optimization_ = true;
  }
  else if(qregs_[0].name().find("thrust") != std::string::npos){
    thrust_optimization_ = true;
  }


  //initialize qubit map
  qubit_map_.resize(num_qubits_);
  for(i=0;i<num_qubits_;i++){
    qubit_map_[i] = i;
  }

  return true;
}

template <class state_t>
bool StateChunk<state_t>::allocate_qregs(uint_t num_chunks)
{
  int_t i;
  //deallocate qregs before reallocation
  if(qregs_.size() > 0){
    if(qregs_.size() == num_chunks)
      return true;  //can reuse allocated chunks

    qregs_.clear();
  }

  qregs_.resize(num_chunks);

  //allocate qregs
  uint_t chunk_id = multi_chunk_distribution_ ? global_chunk_index_ : 0;
  bool ret = true;
  qregs_[0].set_max_matrix_bits(BaseState::max_matrix_qubits_);
  ret &= qregs_[0].chunk_setup(chunk_bits_*qubit_scale(),num_qubits_*qubit_scale(),chunk_id,num_chunks);
  for(i=1;i<num_chunks;i++){
    uint_t gid = i + chunk_id;
    ret &= qregs_[i].chunk_setup(qregs_[0],gid);
  }

  //initialize groups
  top_chunk_of_group_.clear();
  num_groups_ = 0;
  for(i=0;i<qregs_.size();i++){
    if(qregs_[i].top_of_group()){
      top_chunk_of_group_.push_back(i);
      num_groups_++;
    }
  }
  top_chunk_of_group_.push_back(qregs_.size());
  num_chunks_in_group_.resize(num_groups_);
  for(i=0;i<num_groups_;i++){
    num_chunks_in_group_[i] = top_chunk_of_group_[i+1] - top_chunk_of_group_[i];
  }

  return ret;
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
template <typename InputIterator>
void StateChunk<state_t>::apply_ops(InputIterator first, InputIterator last,
                               ExperimentResult &result,
                               RngEngine &rng,
                               bool final_ops) 
{
  if(multi_chunk_distribution_){
    return apply_ops_chunks(first,last,result,rng,final_ops);
  }

  std::unordered_map<std::string, InputIterator> marks;
  // Simple loop over vector of input operations
  for (auto it = first; it != last; ++it) {
    switch (it->type) {
    case Operations::OpType::mark: {
      marks[it->string_params[0]] = it;
      break;
    }
    case Operations::OpType::jump: {
      if (check_conditional(0, *it)) {
        const auto& mark_name = it->string_params[0];
        auto mark_it = marks.find(mark_name);
        if (mark_it != marks.end()) {
          it = mark_it->second;
        } else {
          for (++it; it != last; ++it) {
            if (it->type == Operations::OpType::mark) {
              marks[it->string_params[0]] = it;
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
    default: {
    apply_op(0, *it, result, rng, final_ops && (it + 1 == last) );
    }
    }
  }
}

template <class state_t>
template <typename InputIterator>
void StateChunk<state_t>::apply_ops_chunks(InputIterator first, InputIterator last,
                               ExperimentResult &result,
                               RngEngine &rng,
                               bool final_ops) 
{
  uint_t iOp,nOp;

  nOp = std::distance(first, last);
  iOp = 0;
  while(iOp < nOp){
    const Operations::Op op_iOp = *(first + iOp);

    if(op_iOp.type == Operations::OpType::gate && op_iOp.name == "swap_chunk"){
      //apply swap between chunks
      apply_chunk_swap(op_iOp.qubits);
    }
    else if(op_iOp.type == Operations::OpType::sim_op && op_iOp.name == "begin_blocking"){
      //applying sequence of gates inside each chunk

      uint_t iOpEnd = iOp;
      while(iOpEnd < nOp){
        const Operations::Op op_iOpEnd = *(first + iOpEnd);
        if(op_iOpEnd.type == Operations::OpType::sim_op && op_iOpEnd.name == "end_blocking"){
          break;
        }
        iOpEnd++;
      }

      uint_t iOpBegin = iOp + 1;
      if(num_groups_ > 1 && chunk_omp_parallel_){
#pragma omp parallel for  num_threads(num_groups_)
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(top_chunk_of_group_[ig], first + iOpBegin, first + iOpEnd, result, rng);
      }
      else{
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(top_chunk_of_group_[ig], first + iOpBegin, first + iOpEnd, result, rng);
      }
      iOp = iOpEnd;
    }
    else if(is_applied_to_each_chunk(op_iOp)){
      if(num_groups_ > 1 && chunk_omp_parallel_){
#pragma omp parallel for num_threads(num_groups_)
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(top_chunk_of_group_[ig], first + iOp, first + iOp+1, result, rng);
      }
      else{
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(top_chunk_of_group_[ig], first + iOp, first + iOp+1, result, rng);
      }
    }
    else{
      //parallelize inside state implementations
      apply_op(STATE_APPLY_TO_ALL_CHUNKS, op_iOp,result,rng,final_ops && nOp == iOp + 1);
    }
    iOp++;
  }
}

template <class state_t>
template <typename InputIterator>
void StateChunk<state_t>::apply_cache_blocking_ops(const int_t iChunk, InputIterator first,
               InputIterator last,
               ExperimentResult &result,
               RngEngine &rng)
{
  //fecth chunk in cache
  if(qregs_[iChunk].fetch_chunk()){
    for (auto it = first; it != last; ++it) {
      apply_op(iChunk, *it, result, rng, false);
    }
    //release chunk from cache
    qregs_[iChunk].release_chunk();
  }
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
bool StateChunk<state_t>::check_conditional(const int_t iChunk, const Operations::Op &op)
{
  if(multi_shots_parallelization_){
    //multi-shots parallelization
    if(op.conditional){
      qregs_[iChunk].set_conditional(op.conditional_reg);
    }
    return true;
  }
  else{
    return cregs_[0].check_conditional(op);
  }
}

template <class state_t>
template <typename InputIterator>
void StateChunk<state_t>::apply_ops_multi_shots(InputIterator first, InputIterator last,
                               const Noise::NoiseModel &noise,
                               ExperimentResult &result,
                               uint_t rng_seed,
                               bool final_ops) 
{
  int_t i;
  int_t i_begin,n_shots;

  i_begin = 0;
  while(i_begin<num_local_chunks_){
    local_shot_index_ = i_begin;

    //loop for states can be stored in available memory
    n_shots = qregs_.size();
    if(i_begin+n_shots > num_local_chunks_){
      n_shots = num_local_chunks_ - i_begin;
      //resize qregs
      allocate_qregs(n_shots);
    }
    //initialization (equivalent to initialize_qreg + initialize_creg)
    if(num_groups_ > 1 && chunk_omp_parallel_){
#pragma omp parallel for 
      for(i=0;i<num_groups_;i++){
        uint_t istate = top_chunk_of_group_[i];

        for(uint_t j=top_chunk_of_group_[i];j<top_chunk_of_group_[i+1];j++){
          //enabling batch shots optimization
          qregs_[j].enable_batch(true);

          //initialize qreg here
          qregs_[j].set_num_qubits(chunk_bits_);
          qregs_[j].initialize();

          //initialize creg here
          qregs_[j].initialize_creg(cregs_[0].memory_size(), cregs_[0].register_size());
        }
      }
    }
    else{
      for(i=0;i<num_groups_;i++){
        uint_t istate = top_chunk_of_group_[i];

        for(uint_t j=top_chunk_of_group_[i];j<top_chunk_of_group_[i+1];j++){
          //enabling batch shots optimization
          qregs_[j].enable_batch(true);

          //initialize qreg here
          qregs_[j].set_num_qubits(chunk_bits_);
          qregs_[j].initialize();

          //initialize creg here
          qregs_[j].initialize_creg(cregs_[0].memory_size(), cregs_[0].register_size());
        }
      }
    }
    apply_global_phase(); //this is parallelized in StateChunk sub-classes

    //apply ops to multiple-shots
    if(num_groups_ > 1 && chunk_omp_parallel_){
      std::vector<ExperimentResult> par_results(num_groups_);
#pragma omp parallel for
      for(i=0;i<num_groups_;i++)
        apply_ops_multi_shots_for_group(i, first, last, noise, par_results[i], rng_seed, final_ops);

      for (auto &res : par_results)
        result.combine(std::move(res));
    }
    else{
      for(i=0;i<num_groups_;i++)
        apply_ops_multi_shots_for_group(i, first, last, noise, result, rng_seed, final_ops);
    }

    //collect measured bits and copy memory
    for(i=0;i<n_shots;i++){
      qregs_[i].get_creg(cregs_[global_chunk_index_ + i_begin + i]);
    }

    i_begin += n_shots;
  }

  gather_creg_memory();
}

template <class state_t>
template <typename InputIterator>
void StateChunk<state_t>::apply_ops_multi_shots_for_group(int_t i_group, 
                               InputIterator first, InputIterator last,
                               const Noise::NoiseModel &noise,
                               ExperimentResult &result,
                               uint_t rng_seed,
                               bool final_ops) 
{
  uint_t istate = top_chunk_of_group_[i_group];
  std::vector<RngEngine> rng(num_chunks_in_group_[i_group]);

  for(uint_t j=top_chunk_of_group_[i_group];j<top_chunk_of_group_[i_group+1];j++)
    rng[j-top_chunk_of_group_[i_group]].set_seed(rng_seed + global_chunk_index_ + local_shot_index_ + j);

  for (auto op = first; op != last; ++op) {
    if(op->type == Operations::OpType::qerror_loc){
      //sample error here
      uint_t count = num_chunks_in_group_[i_group];
      uint_t max_ops = 0;
      bool pauli_only = true;
      std::vector<std::vector<Operations::Op>> noise_ops(count);
      for(uint_t j=0;j<count;j++){
        noise_ops[j] = noise.sample_noise_loc(*op,rng[j]);

        if(noise_ops[j].size() == 0 || (noise_ops[j].size() == 1 && noise_ops[j][0].name == "id"))
          continue;
        else{
          if(max_ops < noise_ops[j].size())
            max_ops = noise_ops[j].size();
          if(pauli_only){
            for(int_t k=0;k<noise_ops[j].size();k++){
              if(noise_ops[j][k].name != "x" && noise_ops[j][k].name != "y" && noise_ops[j][k].name != "z" 
                                             && noise_ops[j][k].name != "pauli" && noise_ops[j][k].name != "id"){
                pauli_only = false;
              }
            }
          }
        }
      }

      if(max_ops == 0){
        continue;   //do nothing
      }
      if(pauli_only){   //batched Pauli can be applied (optimization for Pauli error)
        qregs_[istate].apply_batched_pauli_ops(noise_ops);
      }
      else{
        //otherwise execute each circuit
        apply_batched_noise_ops(i_group, noise_ops,result, rng);
      }
    }
    else{
      if(!apply_batched_op(istate, *op, result, rng, final_ops && (op + 1 == last))){
        //call apply_op for each state
        for(uint_t j=top_chunk_of_group_[i_group];j<top_chunk_of_group_[i_group+1];j++){
          qregs_[j].enable_batch(false);
          apply_op(j, *op, result, rng[j-top_chunk_of_group_[i_group]], final_ops && (op + 1 == last) );
          qregs_[j].enable_batch(true);
        }
      }
    }
  }
}

template <class state_t>
void StateChunk<state_t>::apply_batched_noise_ops(const int_t i_group, const std::vector<std::vector<Operations::Op>> &ops, 
                             ExperimentResult &result,
                             std::vector<RngEngine> &rng)
{
  int_t i,j,k,count,nop,pos = 0;
  uint_t istate = top_chunk_of_group_[i_group];
  count = ops.size();

  reg_t mask(count);
  std::vector<bool> finished(count,false);
  for(i=0;i<count;i++){
    int_t cond_reg = -1;

    if(finished[i])
      continue;
    if(ops[i].size() == 0 || (ops[i].size() == 1 && ops[i][0].name == "id")){
      finished[i] = true;
      continue;
    }
    mask[i] = 1;

    //find same ops to be exectuted in a batch
    for(j=i+1;j<count;j++){
      if(finished[j]){
        mask[j] = 0;
        continue;
      }
      if(ops[j].size() == 0 || (ops[j].size() == 1 && ops[j][0].name == "id")){
        mask[j] = 0;
        finished[j] = true;
        continue;
      }

      if(ops[i].size() != ops[j].size()){
        mask[j] = 0;
        continue;
      }

      mask[j] = true;
      for(k=0;k<ops[i].size();k++){
        if(ops[i][k].conditional){
          cond_reg = ops[i][k].conditional_reg;
        }
        if(ops[i][k].type != ops[j][k].type || ops[i][k].name != ops[j][k].name){
          mask[j] = false;
          break;
        }
      }
      if(mask[j])
        finished[j] = true;
    }

    //mask conditional register
    int_t sys_reg = qregs_[istate].set_batched_system_conditional(cond_reg, mask);

    //batched execution on same ops
    for(k=0;k<ops[i].size();k++){
      Operations::Op cop = ops[i][k];

      //mark op conditional to mask shots
      cop.conditional = true;
      cop.conditional_reg = sys_reg;

      if(!apply_batched_op(istate, cop, result,rng, false)){
        //call apply_op for each state
        for(uint_t j=top_chunk_of_group_[i_group];j<top_chunk_of_group_[i_group+1];j++){
          qregs_[j].enable_batch(false);
          apply_op(j, cop, result ,rng[j-top_chunk_of_group_[i_group]],false);
          qregs_[j].enable_batch(true);
        }
      }
    }
    mask[i] = 0;
    finished[i] = true;
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
  for(int_t i=0;i<cregs_.size();i++){
    cregs_[i].initialize(num_memory, num_register);
  }
}


template <class state_t>
void StateChunk<state_t>::initialize_creg(uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) {
  for(int_t i=0;i<cregs_.size();i++){
    cregs_[i].initialize(num_memory, num_register, memory_hex, register_hex);
  }
}

template <class state_t>
void StateChunk<state_t>::save_creg(const int_t iChunk, ExperimentResult &result,
                               const std::string &key,
                               DataSubType subtype) const 
{
  int_t ishot = get_global_shot_index(iChunk);
  if (cregs_[ishot].memory_size() == 0)
    return;
  switch (subtype) {
    case DataSubType::list:
      result.data.add_list(cregs_[ishot].memory_hex(), key);
      result.metadata.add("creg", "result_types", key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(1ULL, key, cregs_[ishot].memory_hex());
      result.metadata.add("creg", "result_types", key);
      break;
    default:
      throw std::runtime_error("Invalid creg data subtype for data key: " + key);
  }
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_average(const int_t iChunk, ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum, OpType type,
                                       DataSubType subtype) const {
  int_t ishot = get_global_shot_index(iChunk);
  switch (subtype) {
    case DataSubType::list:
      result.data.add_list(datum, key);
      break;
    case DataSubType::c_list:
      result.data.add_list(datum, key, cregs_[ishot].memory_hex());
      break;
    case DataSubType::accum:
      result.data.add_accum(datum, key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(datum, key, cregs_[ishot].memory_hex());
      break;
    case DataSubType::average:
      result.data.add_average(datum, key);
      break;
    case DataSubType::c_average:
      result.data.add_average(datum, key, cregs_[ishot].memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid average data subtype for data key: " + key);
  }
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_average(const int_t iChunk, ExperimentResult &result,
                                       const std::string &key,
                                       T&& datum, OpType type,
                                       DataSubType subtype) const {
  int_t ishot = get_global_shot_index(iChunk);
  switch (subtype) {
    case DataSubType::list:
      result.data.add_list(std::move(datum), key);
      break;
    case DataSubType::c_list:
      result.data.add_list(std::move(datum), key, cregs_[ishot].memory_hex());
      break;
    case DataSubType::accum:
      result.data.add_accum(std::move(datum), key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(std::move(datum), key, cregs_[ishot].memory_hex());
      break;
    case DataSubType::average:
      result.data.add_average(std::move(datum), key);
      break;
    case DataSubType::c_average:
      result.data.add_average(std::move(datum), key, cregs_[ishot].memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid average data subtype for data key: " + key);
  }
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_pershot(const int_t iChunk, ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum, OpType type,
                                       DataSubType subtype) const {
  int_t ishot = get_global_shot_index(iChunk);
  switch (subtype) {
  case DataSubType::single:
    result.data.add_single(datum, key);
    break;
  case DataSubType::c_single:
    result.data.add_single(datum, key, cregs_[ishot].memory_hex());
    break;
  case DataSubType::list:
    result.data.add_list(datum, key);
    break;
  case DataSubType::c_list:
    result.data.add_list(datum, key, cregs_[ishot].memory_hex());
    break;
  default:
    throw std::runtime_error("Invalid pershot data subtype for data key: " + key);
  }
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_pershot(const int_t iChunk, ExperimentResult &result, 
                                       const std::string &key,
                                       T&& datum, OpType type,
                                       DataSubType subtype) const {
  int_t ishot = get_global_shot_index(iChunk);
  switch (subtype) {
    case DataSubType::single:
      result.data.add_single(std::move(datum), key);
      break;
    case DataSubType::c_single:
      result.data.add_single(std::move(datum), key, cregs_[ishot].memory_hex());
      break;
    case DataSubType::list:
      result.data.add_list(std::move(datum), key);
      break;
    case DataSubType::c_list:
      result.data.add_list(std::move(datum), key, cregs_[ishot].memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid pershot data subtype for data key: " + key);
  }
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_single(ExperimentResult &result,
                                      const std::string &key,
                                      const T& datum, OpType type) const {
  result.data.add_single(datum, key);
  result.metadata.add(type, "result_types", key);
  result.metadata.add(DataSubType::single, "result_subtypes", key);
}

template <class state_t>
template <class T>
void StateChunk<state_t>::save_data_single(ExperimentResult &result,
                                      const std::string &key,
                                      T&& datum, OpType type) const {
  result.data.add_single(std::move(datum), key);
  result.metadata.add(type, "result_types", key);
  result.metadata.add(DataSubType::single, "result_subtypes", key);
}

template <class state_t>
void StateChunk<state_t>::snapshot_state(const int_t iChunk, const Operations::Op &op,
                                    ExperimentResult &result,
                                    std::string name) const 
{
  name = (name.empty()) ? op.name : name;
  result.legacy_data.add_pershot_snapshot(name, op.string_params[0], qregs_[iChunk]);
}


template <class state_t>
void StateChunk<state_t>::snapshot_creg_memory(const int_t iChunk, const Operations::Op &op,
                                          ExperimentResult &result,
                                          std::string name) const 
{
  int_t ishot = get_global_shot_index(iChunk);
  result.legacy_data.add_pershot_snapshot(name,
                               op.string_params[0],
                               cregs_[ishot].memory_hex());
}


template <class state_t>
void StateChunk<state_t>::snapshot_creg_register(const int_t iChunk, const Operations::Op &op,
                                            ExperimentResult &result,
                                            std::string name) const 
{
  int_t ishot = get_global_shot_index(iChunk);
  result.legacy_data.add_pershot_snapshot(name,
                               op.string_params[0],
                               cregs_[ishot].register_hex());
}


template <class state_t>
void StateChunk<state_t>::apply_save_expval(const int_t iChunk, const Operations::Op &op,
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
    const auto val = expval_pauli(iChunk, op.qubits, std::get<0>(param));
    expval += std::get<1>(param) * val;
    if (variance) {
      sq_expval += std::get<2>(param) * val;
    }
  }
  if (variance) {
    std::vector<double> expval_var(2);
    expval_var[0] = expval;  // mean
    expval_var[1] = sq_expval - expval * expval;  // variance
    save_data_average(iChunk, result, op.string_params[0], expval_var, op.type, op.save_type);
  } else {
    save_data_average(iChunk, result, op.string_params[0], expval, op.type, op.save_type);
  }
}

template <class state_t>
void StateChunk<state_t>::save_count_data(ExperimentResult& result,bool save_memory)
{
  for(int_t i=0;i<cregs_.size();i++){
    if (cregs_[i].memory_size() > 0) {
      std::string memory_hex = cregs_[i].memory_hex();
      result.data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
      if(save_memory) {
        result.data.add_list(std::move(memory_hex), "memory");
      }
    }
  }
}

//-------------------------------------------------------------------------
// functions for multi-chunk distribution
//-------------------------------------------------------------------------
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
template <typename list_t>
void StateChunk<state_t>::initialize_from_vector(const int_t iChunkIn, const list_t &vec)
{
  int_t iChunk;

  if(multi_chunk_distribution_){
#pragma omp parallel for if(chunk_omp_parallel_) private(iChunk) 
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      list_t tmp(1ull << (chunk_bits_*qubit_scale()));
      for(int_t i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
        tmp[i] = vec[((global_chunk_index_ + iChunk) << (chunk_bits_*qubit_scale())) + i];
      }
      qregs_[iChunk].initialize_from_vector(tmp);
    }
  }
  else{
    if(iChunkIn == STATE_APPLY_TO_ALL_CHUNKS){
      for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
        qregs_[iChunk].initialize_from_vector(vec);
      }
    }
    else
      qregs_[iChunkIn].initialize_from_vector(vec);
  }
}

template <class state_t>
template <typename list_t>
void StateChunk<state_t>::initialize_from_matrix(const int_t iChunkIn, const list_t &mat)
{
  int_t iChunk;
  if(multi_chunk_distribution_){
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
  else{
    if(iChunkIn == STATE_APPLY_TO_ALL_CHUNKS){
      for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
        qregs_[iChunk].initialize_from_matrix(mat);
      }
    }
    else
      qregs_[iChunkIn].initialize_from_matrix(mat);
  }
}

template <class state_t>
auto StateChunk<state_t>::apply_to_matrix(bool copy)
{
  //this function is used to collect states over chunks
  int_t iChunk;
  uint_t size = 1ull << (chunk_bits_*qubit_scale());
  uint_t mask = (1ull << (chunk_bits_)) - 1;
  uint_t num_threads = qregs_[0].get_omp_threads();

  size_t size_required = 2*(sizeof(std::complex<double>) << (num_qubits_*2)) + (sizeof(std::complex<double>) << (chunk_bits_*2))*num_local_chunks_;
  if((size_required>>20) > Utils::get_system_memory_mb()){
    throw std::runtime_error(std::string("There is not enough memory to store states as matrix"));
  }

  auto matrix = qregs_[0].copy_to_matrix();

  if(distributed_rank_ == 0){
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
    //inside chunk
    if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for num_threads(num_groups_) 
      for(int_t ig=0;ig<num_groups_;ig++)
        qregs_[top_chunk_of_group_[ig]].apply_mcswap(qubits);
    }
    else{
      for(int_t ig=0;ig<num_groups_;ig++)
        qregs_[top_chunk_of_group_[ig]].apply_mcswap(qubits);
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

      if(chunk_omp_parallel_){
#pragma omp parallel for private(iPair,baseChunk,iChunk1,iChunk2)
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
      else{
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
#pragma omp parallel for if(chunk_omp_parallel_ && num_groups_ > 1) 
    for(int_t ig=0;ig<num_groups_;ig++){
      uint_t istate = top_chunk_of_group_[ig];
      qregs_[istate].apply_mcx(qubits);
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
void StateChunk<state_t>::reduce_sum(reg_t& sum) const
{
#ifdef AER_MPI
  if(distributed_procs_ > 1){
    uint_t i,n = sum.size();
    reg_t tmp(n);
    MPI_Allreduce(&sum[0],&tmp[0],n,MPI_UINT64_T,MPI_SUM,distributed_comm_);
    for(i=0;i<n;i++){
      sum[i] = tmp[i];
    }
  }
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

    if((global_size >> 21) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to gather state"));
    }

    if(distributed_rank_ == 0){
      if((global_size >> 21) > Utils::get_system_memory_mb()){
        throw std::runtime_error(std::string("There is not enough memory to gather state"));
      }

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

    if((global_size >> 21) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to gather state"));
    }

    if(distributed_rank_ == 0){
      if((global_size >> 21) > Utils::get_system_memory_mb()){
        throw std::runtime_error(std::string("There is not enough memory to gather state"));
      }

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

template <class state_t>
void StateChunk<state_t>::gather_creg_memory(void)
{
#ifdef AER_MPI
  int_t i,j;
  uint_t n64,i64,ibit;

  if(distributed_procs_ == 1)
    return;
  if(cregs_[0].memory_size() == 0)
    return;

  //number of 64-bit integers per memory
  n64 = (cregs_[0].memory_size() + 63) >> 6;

  reg_t bin_memory(n64*num_local_chunks_,0);
  //compress memory string to binary
#pragma omp parallel for private(i,j,i64,ibit)
  for(i=0;i<num_local_chunks_;i++){
    for(j=0;j<cregs_[0].memory_size();j++){
      i64 = j >> 6;
      ibit = j & 63;
      if(cregs_[global_chunk_index_ + i].creg_memory()[j] == '1'){
        bin_memory[i*n64 + i64] |= (1ull << ibit);
      }
    }
  }

  reg_t recv(n64*num_global_chunks_);
  std::vector<int> recv_counts(distributed_procs_);
  std::vector<int> recv_offset(distributed_procs_);

  for(i=0;i<distributed_procs_;i++){
    recv_offset[i] = num_global_chunks_ * i / distributed_procs_;
    recv_counts[i] = (num_global_chunks_ * (i+1) / distributed_procs_) - recv_offset[i];
  }

  MPI_Allgatherv(&bin_memory[0],n64*num_local_chunks_,MPI_UINT64_T,
                 &recv[0],&recv_counts[0],&recv_offset[0],MPI_UINT64_T,distributed_comm_);

  //store gathered memory
#pragma omp parallel for private(i,j,i64,ibit)
  for(i=0;i<num_global_chunks_;i++){
    for(j=0;j<cregs_[0].memory_size();j++){
      i64 = j >> 6;
      ibit = j & 63;
      if(((recv[i*n64 + i64] >> ibit) & 1) == 1)
        cregs_[i].creg_memory()[j] = '1';
      else
        cregs_[i].creg_memory()[j] = '0';
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
