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

namespace QuantumState {

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
  using OpItr = std::vector<Operations::Op>::const_iterator;

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
  // - `OpType::barrier` if barrier is supported
  // - `OpType::matrix` if arbitrary unitary matrices are supported
  // - `OpType::kraus` if general Kraus noise channels are supported
  //
  // For gate ops allowed gates are specified by a set of string names,
  // for example this could include {"u1", "u2", "u3", "U", "cx", "CX"}
  //

  StateChunk(const Operations::OpSet &opset) : BaseState(opset)
  {
    num_global_chunks_ = 0;
    num_local_chunks_ = 0;

    chunk_omp_parallel_ = false;
    global_chunk_indexing_ = false;

  }

  virtual ~StateChunk();

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

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

  //memory allocation (previously called before inisitalize_qreg)
  virtual bool allocate(uint_t num_qubits,uint_t block_bits,uint_t num_parallel_shots = 1);

  bool allocate_state(RegistersBase& state, uint_t num_max_shots = 1) override;

  // Return the expectation value of a N-qubit Pauli operator
  // If the simulator does not support Pauli expectation value this should
  // raise an exception.
  virtual double expval_pauli(RegistersBase& state, const reg_t &qubits,
                              const std::string& pauli) = 0;

  //-----------------------------------------------------------------------
  // Optional: Load config settings
  //-----------------------------------------------------------------------

  // Load any settings for the StateChunk class from a config JSON
  void set_config(const json_t &config) override final;

  //=======================================================================
  // Standard non-virtual methods
  //
  // These methods should not be modified in any StateChunk subclasses
  //=======================================================================

  //-----------------------------------------------------------------------
  // Apply circuits and ops
  //-----------------------------------------------------------------------

  // Apply a sequence of operations to the current state of the StateChunk class.
  // It is up to the StateChunk subclass to decide how this sequence should be
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
                 bool final_ops = false) override;

  //apply_op for specific chunk
  virtual void apply_op_chunk(uint_t iChunk, RegistersBase& state, 
                        const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine& rng,
                        bool final_op = false) = 0;

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------
  template <typename list_t>
  void initialize_from_vector(Registers<state_t>& state, const list_t &vec);

  template <typename list_t>
  void initialize_from_matrix(Registers<state_t>& state, const list_t &mat);

  //-----------------------------------------------------------------------
  // Common instructions
  //-----------------------------------------------------------------------
 
  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  //set max number of shots to execute in a batch
  void set_max_bached_shots(uint_t shots)
  {
    max_batched_shots_ = shots;
  }

  //Does this state support multi-chunk distribution?
  bool multi_chunk_distribution_supported(void) override
  {return true;}
  //Does this state support multi-shot parallelization?
  virtual bool multi_shot_parallelization_supported(void)
  {
    return true;
  }

  //set creg bit counts before initialize creg
  void set_num_creg_bits(uint_t num_memory, uint_t num_register) override
  {
    num_creg_memory_ = num_memory;
    num_creg_registers_ = num_register;
  }

protected:

  //extra parameters for parallel simulations
  uint_t num_global_chunks_;    //number of total chunks 
  uint_t num_local_chunks_;     //number of local chunks
  uint_t chunk_bits_;           //number of qubits per chunk
  uint_t block_bits_;           //number of cache blocked qubits

  uint_t global_chunk_index_;   //beginning chunk index for this process
  reg_t chunk_index_begin_;     //beginning chunk index for each process
  reg_t chunk_index_end_;       //ending chunk index for each process
  uint_t local_shot_index_;    //local shot ID of current batch loop


  bool chunk_omp_parallel_;     //using thread parallel to process loop of chunks or not
  bool global_chunk_indexing_;  //using global index for control qubits and diagonal matrix

  bool multi_chunk_distribution_ = false; //distributing chunks to apply cache blocking parallelization
  bool multi_shots_parallelization_ = false; //using chunks as multiple shots parallelization
  bool set_parallelization_called_ = false;    //this flag is used to check set_parallelization is already called, if yes the call sets max_batched_shots_
  uint_t max_batched_shots_ = 1;    //max number of shots can be stored on available memory

  reg_t qubit_map_;             //qubit map to restore swapped qubits

  bool multi_chunk_swap_enable_ = true;     //enable multi-chunk swaps
  uint_t chunk_swap_buffer_qubits_ = 15;    //maximum buffer size in qubits for chunk swap
  uint_t max_multi_swap_;                 //maximum swaps can be applied at a time, calculated by chunk_swap_buffer_bits_

  //group of states (GPU devices)
  uint_t num_groups_;            //number of groups of chunks
  reg_t top_chunk_of_group_;
  reg_t num_chunks_in_group_;
  int num_threads_per_group_;   //number of outer threads per group


  uint_t num_creg_memory_ = 0;    //number of total bits for creg (reserve for multi-shots)
  uint_t num_creg_registers_ = 0;

  //-----------------------------------------------------------------------
  // Apply circuits and ops
  //-----------------------------------------------------------------------
  //apply ops for multi-chunk distribution
  void apply_ops_chunks(Registers<state_t>& state, 
                 OpItr first,
                 OpItr last,
                 ExperimentResult &result,
                 RngEngine &rng,
                 bool final_ops = false);

  void apply_ops(RegistersBase& state,
                 OpItr first,
                 OpItr last,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 RngEngine &rng,
                 const bool final_ops = false) override;

  //apply cache blocked ops in each chunk
  void apply_cache_blocking_ops(Registers<state_t>& state, const int_t iGroup,
                 OpItr first,
                 OpItr last,
                 ExperimentResult &result,
                 RngEngine &rng);


  //apply ops to multiple shots
  //this function should be separately defined since apply_ops is called in quantum_error
  bool run_shots_with_batched_execution(OpItr first,
                 OpItr last,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 const uint_t rng_seed,
                 const uint_t num_shots) override;

  //apply ops for multi-shots to one group
  void apply_ops_multi_shots_for_group(Registers<state_t>& state, int_t i_group, 
                               OpItr first, OpItr last,
                               const Noise::NoiseModel &noise,
                               ExperimentResult &result,
                               const uint_t rng_seed,
                               bool final_ops);

  //apply op to multiple shots , return flase if op is not supported to execute in a batch
  virtual bool apply_batched_op(const int_t iChunk, RegistersBase& state, const Operations::Op &op,
                                ExperimentResult &result,
                                std::vector<RngEngine> &rng,
                                bool final_op = false){return false;}

  //apply sampled noise to multiple-shots (this is used for ops contains non-Pauli operators)
  void apply_batched_noise_ops(Registers<state_t>& state,const int_t i_group, const std::vector<std::vector<Operations::Op>> &ops, 
                               ExperimentResult &result,
                               std::vector<RngEngine> &rng);

  //check conditional
  bool check_conditional(Registers<state_t>& state, const Operations::Op &op);

  //this function is used to scale chunk qubits for multi-chunk distribution
  virtual int qubit_scale(void)
  {
    return 1;     //scale of qubit number (x2 for density and unitary matrices)
  }
  uint_t get_process_by_chunk(uint_t cid);

  //allocate qregs
  bool allocate_qregs(Registers<state_t>& state, uint_t num_chunks);


  //-----------------------------------------------------------------------
  //Functions for multi-chunk distribution
  //-----------------------------------------------------------------------
  //swap between chunks
  virtual void apply_chunk_swap(RegistersBase& state, const reg_t &qubits);

  //apply multiple swaps between chunks
  virtual void apply_multi_chunk_swap(RegistersBase& state, const reg_t &qubits);

  //apply X gate over chunks
  virtual void apply_chunk_x(RegistersBase& state, const uint_t qubit);

  //send/receive chunk in receive buffer
  void send_chunk(Registers<state_t>& state, uint_t local_chunk_index, uint_t global_chunk_index);
  void recv_chunk(Registers<state_t>& state, uint_t local_chunk_index, uint_t global_chunk_index);

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
  auto apply_to_matrix(Registers<state_t>& state, bool copy = false);

  // Apply the global phase
  virtual void apply_global_phase(RegistersBase& state){}

  //check if the operator should be applied to each chunk
  virtual bool is_applied_to_each_chunk(const Operations::Op &op);

  //return global shot index for the chunk
  inline int_t get_global_shot_index(const int_t iChunk) const
  {
    return multi_shots_parallelization_ ? (iChunk + local_shot_index_ + global_chunk_index_) : 0;
  }

  //separate inside and outside qubits for (multi) control gates
  void get_inout_ctrl_qubits(const Operations::Op &op, reg_t& qubits_out, reg_t& qubits_in);

  std::vector<Operations::Op> sample_noise(const Noise::NoiseModel &noise, const Operations::Op &op, RngEngine &rng) override
  {
    return noise.sample_noise_loc(op, rng);
  }

  //remake gate operation by qubits inside chunk
  Operations::Op remake_gate_in_chunk_qubits(const Operations::Op &op, reg_t& qubits_in);
};


//=========================================================================
// Implementations
//=========================================================================

template <class state_t>
StateChunk<state_t>::~StateChunk(void)
{
}

template <class state_t>
void StateChunk<state_t>::set_config(const json_t &config) 
{
  BaseState::set_config(config);

  num_threads_per_group_ = 1;
  if(JSON::check_key("num_threads_per_device", config)) {
    JSON::get_value(num_threads_per_group_, "num_threads_per_device", config);
  }

  if(JSON::check_key("chunk_swap_buffer_qubits", config)) {
    JSON::get_value(chunk_swap_buffer_qubits_, "chunk_swap_buffer_qubits", config);
  }

}


template <class state_t>
bool StateChunk<state_t>::allocate(uint_t num_qubits,uint_t block_bits,uint_t num_parallel_shots)
{
  int_t i;
  BaseState::num_qubits_ = num_qubits;
  block_bits_ = block_bits;

  if(block_bits_ > 0){
    chunk_bits_ = block_bits_;
    if(chunk_bits_ > BaseState::num_qubits_){
      chunk_bits_ = BaseState::num_qubits_;
    }
  }
  else{
    chunk_bits_ = BaseState::num_qubits_;
  }

  if(chunk_bits_ < BaseState::num_qubits_){
    //multi-chunk distribution with cache blocking transpiler
    multi_chunk_distribution_ = true;
    multi_shots_parallelization_ = false;
    num_global_chunks_ = 1ull << ((BaseState::num_qubits_ - chunk_bits_)*qubit_scale());
  }
  else{
    //single-shot or multi-shots parallelization
    multi_chunk_distribution_ = false;
    if(num_parallel_shots > 1)
      multi_shots_parallelization_ = true;
    else
      multi_shots_parallelization_ = false;
    num_global_chunks_ = num_parallel_shots;
  }

  chunk_index_begin_.resize(BaseState::distributed_procs_);
  chunk_index_end_.resize(BaseState::distributed_procs_);
  for(i=0;i<BaseState::distributed_procs_;i++){
    chunk_index_begin_[i] = num_global_chunks_*i / BaseState::distributed_procs_;
    chunk_index_end_[i] = num_global_chunks_*(i+1) / BaseState::distributed_procs_;
  }

  num_local_chunks_ = chunk_index_end_[BaseState::distributed_rank_] - chunk_index_begin_[BaseState::distributed_rank_];
  global_chunk_index_ = chunk_index_begin_[BaseState::distributed_rank_];
  local_shot_index_ = 0;

  global_chunk_indexing_ = false;
  chunk_omp_parallel_ = false;
  if(BaseState::sim_device_name_ == "GPU"){
#ifdef _OPENMP
    if(omp_get_num_threads() == 1 && multi_chunk_distribution_)
      chunk_omp_parallel_ = true;
#endif

    //set cuStateVec_enable_ 
    if(BaseState::cuStateVec_enable_){
      if(multi_shots_parallelization_)
        BaseState::cuStateVec_enable_ = false;   //multi-shots parallelization is not supported for cuStateVec
    }

    if(!BaseState::cuStateVec_enable_)
      global_chunk_indexing_ = true;    //cuStateVec does not handle global chunk index for diagonal matrix
  }
  else if(BaseState::sim_device_name_ == "Thrust"){
    global_chunk_indexing_ = true;
    chunk_omp_parallel_ = false;
  }

  allocate_state(BaseState::state_, num_local_chunks_);

  if(chunk_bits_ <= chunk_swap_buffer_qubits_ + 1)
    multi_chunk_swap_enable_ = false;
  else
    max_multi_swap_ = chunk_bits_ - chunk_swap_buffer_qubits_;

  return true;
}

template <class state_t>
bool StateChunk<state_t>::allocate_state(RegistersBase& state, uint_t num_max_shots)
{
  allocate_qregs(dynamic_cast<Registers<state_t>&>(state), std::max(num_local_chunks_, num_max_shots));

  state.initialize_qubit_map(BaseState::num_qubits_);

  return true;
}


template <class state_t>
bool StateChunk<state_t>::allocate_qregs(Registers<state_t>& state, uint_t num_chunks)
{
  int_t i;

  //deallocate qregs before reallocation
  if(state.qregs().size() > 0 && num_local_chunks_ > 1){
    if(state.qregs().size() == num_local_chunks_)
      return true;  //can reuse allocated chunks

    state.qregs().clear();
  }
  state.allocate(num_local_chunks_);   //for multi-shot + multi-chunk, allocate 1st set of chunks only (=num_local_chunks_)

  if(num_creg_memory_ !=0 || num_creg_registers_ !=0){
    for(i=0;i<num_chunks;i++){
      //set number of creg bits before actual initialization
      qregs_[i].initialize_creg(num_creg_memory_, num_creg_registers_);
    }
  }

  //allocate qregs
  uint_t chunk_id = multi_chunk_distribution_ ? global_chunk_index_ : 0;
  bool ret = true;
  state.qreg(0).set_max_matrix_bits(BaseState::max_matrix_qubits_);
  state.qreg(0).set_num_threads_per_group(num_threads_per_group_);
  state.qreg(0).cuStateVec_enable(BaseState::cuStateVec_enable_);
  //reserve num_chunk chunks memory space for multi-shot (num_chunks == num_local_chunks_ for single shot)
  ret &= state.qreg(0).chunk_setup(chunk_bits_*qubit_scale(), BaseState::num_qubits_*qubit_scale(), chunk_id, num_chunks);
  for(i=1;i<num_local_chunks_;i++){
    uint_t gid = i + chunk_id;
    ret &= state.qreg(i).chunk_setup(state.qreg(0),gid);
    state.qreg(i).set_num_threads_per_group(num_threads_per_group_);
  }

  //initialize groups
  top_chunk_of_group_.clear();
  num_groups_ = 0;
  for(i=0;i<state.num_qregs();i++){
    if(state.qreg(i).top_of_group()){
      top_chunk_of_group_.push_back(i);
      num_groups_++;
    }
  }
  top_chunk_of_group_.push_back(state.qregs().size());
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
  for(i=0;i<BaseState::distributed_procs_;i++){
    if(cid >= chunk_index_begin_[i] && cid < chunk_index_end_[i]){
      return i;
    }
  }
  return BaseState::distributed_procs_;
}

template <class state_t>
void StateChunk<state_t>::apply_ops(OpItr first, OpItr last,
                               ExperimentResult &result,
                               RngEngine &rng,
                               bool final_ops) 
{
  if(multi_chunk_distribution_){
    apply_ops_chunks(BaseState::state_, first,last,result,rng, final_ops);
    return;
  }
  BaseState::apply_ops(first, last, result, rng, final_ops);

  BaseState::state_.qreg().synchronize();

#ifdef AER_CUSTATEVEC
  result.metadata.add(BaseState::cuStateVec_enable_, "cuStateVec_enable");
#endif
}

template <class state_t>
void StateChunk<state_t>::apply_ops(RegistersBase& state_in,
               OpItr first,
               OpItr last,
               const Noise::NoiseModel &noise,
               ExperimentResult &result,
               RngEngine &rng,
               const bool final_ops)
{
  Registers<state_t>& state = dynamic_cast<Registers<state_t>&>(state_in);
  if(multi_chunk_distribution_){
    return apply_ops_chunks(state, first,last,result,rng, final_ops);
  }

  // Simple loop over vector of input operations
  for (auto it = first; it != last; ++it) {
    switch (it->type) {
      case Operations::OpType::mark: {
        state.marks()[it->string_params[0]] = it;
        break;
      }
      case Operations::OpType::jump: {
        if(state.creg().check_conditional(*it)) {
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
        BaseState::apply_runtime_noise_sampling(state, *it, noise);
        state.next_iter() = it + 1;
        return;
      }
      default: {
        this->apply_op(state, *it, result, rng, final_ops && (it + 1 == last) );
        if(BaseState::enable_shot_branching_ && state.num_branch() > 0){
          //break loop to branch states
          state.next_iter() = it + 1;
          return;
        }
      }
    }
  }
  state.next_iter() = last;

#ifdef AER_CUSTATEVEC
  result.metadata.add(BaseState::cuStateVec_enable_, "cuStateVec_enable");
#endif
}


template <class state_t>
void StateChunk<state_t>::apply_ops_chunks(Registers<state_t>& state, 
                               OpItr first, OpItr last,
                               ExperimentResult &result,
                               RngEngine &rng,
                               bool final_ops) 
{
  uint_t iOp,nOp;
  reg_t multi_swap;

  nOp = std::distance(first, last);
  iOp = 0;

  while(iOp < nOp){
    const Operations::Op op_iOp = *(first + iOp);

    if(op_iOp.type == Operations::OpType::gate && op_iOp.name == "swap_chunk"){
      //apply swap between chunks
      if(multi_chunk_swap_enable_ && op_iOp.qubits[0] < chunk_bits_ && op_iOp.qubits[1] >= chunk_bits_){
        if(BaseState::distributed_proc_bits_ < 0 || (op_iOp.qubits[1] >= (BaseState::num_qubits_*qubit_scale() - BaseState::distributed_proc_bits_))){   //apply multi-swap when swap is cross qubits
          multi_swap.push_back(op_iOp.qubits[0]);
          multi_swap.push_back(op_iOp.qubits[1]);
          if(multi_swap.size() >= max_multi_swap_*2){
            apply_multi_chunk_swap(state, multi_swap);
            multi_swap.clear();
          }
        }
        else
          apply_chunk_swap(state, op_iOp.qubits);
      }
      else{
        if(multi_swap.size() > 0){
          apply_multi_chunk_swap(state, multi_swap);
          multi_swap.clear();
        }
        apply_chunk_swap(state, op_iOp.qubits);
      }
      iOp++;
      continue;
    }

    if(multi_swap.size() > 0){
      apply_multi_chunk_swap(state, multi_swap);
      multi_swap.clear();
    }

    if(op_iOp.type == Operations::OpType::sim_op && op_iOp.name == "begin_blocking"){
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
          apply_cache_blocking_ops(state, ig, first + iOpBegin, first + iOpEnd, result, rng);
      }
      else{
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(state, ig, first + iOpBegin, first + iOpEnd, result, rng);
      }
      iOp = iOpEnd;
    }
    else if(is_applied_to_each_chunk(op_iOp)){
      if(num_groups_ > 1 && chunk_omp_parallel_){
#pragma omp parallel for num_threads(num_groups_)
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(state, ig, first + iOp, first + iOp+1, result, rng);
      }
      else{
        for(int_t ig=0;ig<num_groups_;ig++)
          apply_cache_blocking_ops(state, ig, first + iOp, first + iOp+1, result, rng);
      }
    }
    else{
      //parallelize inside state implementations
      this->apply_op(state, op_iOp,result,rng,final_ops && nOp == iOp + 1);
    }
    iOp++;

    if(BaseState::enable_shot_branching_ && state.num_branch() > 0){
      state.next_iter() = (first + iOp);
      return;
    }
  }

  if(multi_swap.size() > 0)
    apply_multi_chunk_swap(state, multi_swap);

  if(num_groups_ > 1 && chunk_omp_parallel_){
#pragma omp parallel for  num_threads(num_groups_)
    for(int_t ig=0;ig<num_groups_;ig++)
      state.qreg(top_chunk_of_group_[ig]).synchronize();
  }
  else{
    for(int_t ig=0;ig<num_groups_;ig++)
      state.qreg(top_chunk_of_group_[ig]).synchronize();
  }

  if(BaseState::sim_device_name_ == "GPU"){
#ifdef AER_THRUST_CUDA
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      nDev = 0;
    }
    if(nDev > num_groups_)
      nDev = num_groups_;
    result.metadata.add(nDev,"cacheblocking", "chunk_parallel_gpus");
#endif

#ifdef AER_CUSTATEVEC
    result.metadata.add(BaseState::cuStateVec_enable_, "cuStateVec_enable");
#endif
  }

#ifdef AER_MPI
  result.metadata.add(multi_chunk_swap_enable_,"cacheblocking", "multiple_chunk_swaps_enable");
  if(multi_chunk_swap_enable_){
    result.metadata.add(chunk_swap_buffer_qubits_,"cacheblocking", "multiple_chunk_swaps_buffer_qubits");
    result.metadata.add(max_multi_swap_,"cacheblocking", "max_multiple_chunk_swaps");
  }
#endif

  state.next_iter() = last;
}

template <class state_t>
void StateChunk<state_t>::apply_cache_blocking_ops(Registers<state_t>& state, const int_t iGroup, 
               OpItr first,
               OpItr last,
               ExperimentResult &result,
               RngEngine &rng)
{
  //for each chunk in group
  for(int_t iChunk = top_chunk_of_group_[iGroup];iChunk < top_chunk_of_group_[iGroup + 1];iChunk++){
    //fecth chunk in cache
    if(state.qreg(iChunk).fetch_chunk()){
      for (auto it = first; it != last; ++it) {
        apply_op_chunk(iChunk, state, *it, result, rng, false);
      }
      //release chunk from cache
      state.qreg(iChunk).release_chunk();
    }
  }
}

template <class state_t>
void StateChunk<state_t>::get_inout_ctrl_qubits(const Operations::Op &op, reg_t& qubits_out, reg_t& qubits_in)
{
  if(op.type == Operations::OpType::gate && (op.name[0] == 'c' || op.name.find("mc") == 0)){
    for(int i=0;i<op.qubits.size();i++){
      if(op.qubits[i] < chunk_bits_)
        qubits_in.push_back(op.qubits[i]);
      else
        qubits_out.push_back(op.qubits[i]);
    }
  }
}

template <class state_t>
Operations::Op StateChunk<state_t>::remake_gate_in_chunk_qubits(const Operations::Op &op, reg_t& qubits_in)
{
  Operations::Op new_op = op;
  new_op.qubits = qubits_in;
  //change gate name if there is no control qubits inside chunk
  if(op.name.find("swap") != std::string::npos && qubits_in.size() == 2){
    new_op.name = "swap";
  }
  if(op.name.find("ccx") != std::string::npos){
    if(qubits_in.size() == 1)
      new_op.name = "x";
    else
      new_op.name = "cx";
  }
  else if(qubits_in.size() == 1){
    if(op.name[0] == 'c')
      new_op.name = op.name.substr(1);
    else if(op.name == "mcphase")
      new_op.name = "p";
    else
      new_op.name = op.name.substr(2);  //remove "mc"
  }
  return new_op;
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
bool StateChunk<state_t>::check_conditional(Registers<state_t>& state, const Operations::Op &op)
{
  if(multi_shots_parallelization_){
    //multi-shots parallelization
    if(op.conditional){
      state.qreg().set_conditional(op.conditional_reg);
    }
    return true;
  }
  else{
    return state.creg().check_conditional(op);
  }
}

template <class state_t>
bool StateChunk<state_t>::run_shots_with_batched_execution(
                               OpItr first, OpItr last,
                               const Noise::NoiseModel &noise,
                               ExperimentResult &result,
                               const uint_t rng_seed,
                               const uint_t num_shots)
{
  if(!multi_shots_parallelization_ || BaseState::sim_device_name_ != "GPU" || multi_chunk_distribution_){
    return false;
  }

  Registers<state_t> state;
  int_t i;
  int_t i_begin,n_shots;

  std::vector<ClassicalRegister> cregs(num_local_chunks_);

  if(num_shots != num_global_chunks_)
    allocate(BaseState::num_qubits_, BaseState::num_qubits_, num_shots);

  allocate_state(state, std::min(max_batched_shots_, num_local_chunks_));

  state.creg().initialize(BaseState::num_creg_memory_, BaseState::num_creg_registers_);
  for(int_t i=0;i<cregs.size();i++){
    cregs[i].initialize(BaseState::num_creg_memory_, BaseState::num_creg_registers_);
  }

  i_begin = 0;
  while(i_begin<num_local_chunks_){
    local_shot_index_ = i_begin;

    //loop for states can be stored in available memory
    n_shots = state.qregs().size();
    if(i_begin+n_shots > num_local_chunks_){
      n_shots = num_local_chunks_ - i_begin;
      //resize qregs
      allocate_qregs(state, n_shots);
    }
    //initialization (equivalent to initialize_qreg + initialize_creg)
    auto init_group = [this, &state](int_t ig){
      for(uint_t j=top_chunk_of_group_[ig];j<top_chunk_of_group_[ig+1];j++){
        //enabling batch shots optimization
        state.qreg(j).enable_batch(true);

        //initialize qreg here
        state.qreg(j).set_num_qubits(chunk_bits_);
        state.qreg(j).initialize();

        //initialize creg here
        state.qreg(j).initialize_creg(state.creg().memory_size(), state.creg().register_size());
      }
    };
    Utils::apply_omp_parallel_for((num_groups_ > 1 && chunk_omp_parallel_),0,num_groups_,init_group);

    this->apply_global_phase(state); //this is parallelized in StateChunk sub-classes

    //apply ops to multiple-shots
    if(num_groups_ > 1 && chunk_omp_parallel_){
      std::vector<ExperimentResult> par_results(num_groups_);
#pragma omp parallel for num_threads(num_groups_)
      for(i=0;i<num_groups_;i++)
        apply_ops_multi_shots_for_group(state, i, first, last, noise, par_results[i], rng_seed, true);

      for (auto &res : par_results)
        result.combine(std::move(res));
    }
    else{
      for(i=0;i<num_groups_;i++)
        apply_ops_multi_shots_for_group(state, i, first, last, noise, result, rng_seed, true);
    }

    //collect measured bits and copy memory
    for(i=0;i<n_shots;i++){
      state.qreg(i).read_measured_data(cregs[i_begin + i]);
    }
    i_begin += n_shots;
  }

  BaseState::gather_creg_memory(cregs, num_local_chunks_);

  result.save_count_data(cregs, BaseState::save_creg_memory_);

#ifdef AER_THRUST_CUDA
  if(BaseState::sim_device_name_ == "GPU"){
    int nDev;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      cudaGetLastError();
      nDev = 0;
    }
    if(nDev > num_groups_)
      nDev = num_groups_;
    result.metadata.add(nDev,"batched_shots_optimization_parallel_gpus");
  }
#endif

  result.metadata.add(true, "batched_shots_optimization");

  return true;
}

template <class state_t>
void StateChunk<state_t>::apply_ops_multi_shots_for_group(Registers<state_t>& state,
                               int_t i_group,
                               OpItr first, OpItr last,
                               const Noise::NoiseModel &noise,
                               ExperimentResult &result,
                               const uint_t rng_seed,
                               bool final_ops)
{
  uint_t istate = top_chunk_of_group_[i_group];
  std::vector<RngEngine> rng(num_chunks_in_group_[i_group]);
#ifdef _OPENMP
  int num_inner_threads = omp_get_max_threads() / omp_get_num_threads();
#else
  int num_inner_threads = 1;
#endif

  for(uint_t j=top_chunk_of_group_[i_group];j<top_chunk_of_group_[i_group+1];j++)
    rng[j-top_chunk_of_group_[i_group]].set_seed(rng_seed + global_chunk_index_ + local_shot_index_ + j);

  for (auto op = first; op != last; ++op) {
    if(op->type == Operations::OpType::sample_noise){
      //sample error here
      uint_t count = num_chunks_in_group_[i_group];
      std::vector<std::vector<Operations::Op>> noise_ops(count);

      uint_t count_ops = 0;
      uint_t non_pauli_gate_count = 0;
      if(num_inner_threads > 1){
#pragma omp parallel for reduction(+: count_ops,non_pauli_gate_count) num_threads(num_inner_threads)
        for(int_t j=0;j<count;j++){
          noise_ops[j] = noise.sample_noise_loc(*op,rng[j]);

          if(!(noise_ops[j].size() == 0 || (noise_ops[j].size() == 1 && noise_ops[j][0].name == "id"))){
            count_ops++;
            for(int_t k=0;k<noise_ops[j].size();k++){
              if(noise_ops[j][k].name != "id" && noise_ops[j][k].name != "x" && noise_ops[j][k].name != "y" && noise_ops[j][k].name != "z" && noise_ops[j][k].name != "pauli"){
                non_pauli_gate_count++;
                break;
              }
            }
          }
        }
      }
      else{
        for(int_t j=0;j<count;j++){
          noise_ops[j] = noise.sample_noise_loc(*op,rng[j]);

          if(!(noise_ops[j].size() == 0 || (noise_ops[j].size() == 1 && noise_ops[j][0].name == "id"))){
            count_ops++;
            for(int_t k=0;k<noise_ops[j].size();k++){
              if(noise_ops[j][k].name != "id" && noise_ops[j][k].name != "x" && noise_ops[j][k].name != "y" && noise_ops[j][k].name != "z" && noise_ops[j][k].name != "pauli"){
                non_pauli_gate_count++;
                break;
              }
            }
          }
        }
      }
      if(count_ops == 0){
        continue;   //do nothing
      }
      if(non_pauli_gate_count == 0){   //ptimization for Pauli error
        state.qreg(istate).apply_batched_pauli_ops(noise_ops);
      }
      else{
        //otherwise execute each circuit
        apply_batched_noise_ops(state, i_group, noise_ops,result, rng);
      }
    }
    else{
      if(!apply_batched_op(istate, state, *op, result, rng, final_ops && (op + 1 == last))){
        //call apply_op for each state
        for(uint_t j=top_chunk_of_group_[i_group];j<top_chunk_of_group_[i_group+1];j++)
          state.qreg(j).enable_batch(false);

        this->apply_op(state, *op, result, rng[0], final_ops && (op + 1 == last) );

        for(uint_t j=top_chunk_of_group_[i_group];j<top_chunk_of_group_[i_group+1];j++)
          state.qreg(j).enable_batch(true);
      }
    }
  }

}

template <class state_t>
void StateChunk<state_t>::apply_batched_noise_ops(Registers<state_t>& state, const int_t i_group, const std::vector<std::vector<Operations::Op>> &ops, 
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
    int_t sys_reg = state.qreg(istate).set_batched_system_conditional(cond_reg, mask);

    //batched execution on same ops
    for(k=0;k<ops[i].size();k++){
      Operations::Op cop = ops[i][k];

      //mark op conditional to mask shots
      cop.conditional = true;
      cop.conditional_reg = sys_reg;

      if(!apply_batched_op(istate, state, cop, result,rng, false)){
        //call apply_op for each state
        for(uint_t j=top_chunk_of_group_[i_group];j<top_chunk_of_group_[i_group+1];j++)
          state.qreg(j).enable_batch(false);
        this->apply_op(state, cop, result ,rng[0],false);
        for(uint_t j=top_chunk_of_group_[i_group];j<top_chunk_of_group_[i_group+1];j++)
          state.qreg(j).enable_batch(true);
      }
    }
    mask[i] = 0;
    finished[i] = true;
  }
}

//-------------------------------------------------------------------------
// functions for multi-chunk distribution
//-------------------------------------------------------------------------
template <class state_t>
void StateChunk<state_t>::block_diagonal_matrix(const int_t gid, reg_t &qubits, cvector_t &diag)
{
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
void StateChunk<state_t>::initialize_from_vector(Registers<state_t>& state, const list_t &vec)
{
  int_t iChunk;

  if(multi_chunk_distribution_){
    if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for private(iChunk)
      for(int_t ig=0;ig<num_groups_;ig++){
        for(iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++){
          list_t tmp(1ull << (chunk_bits_*qubit_scale()));
          for(int_t i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
            tmp[i] = vec[((global_chunk_index_ + iChunk) << (chunk_bits_*qubit_scale())) + i];
          }
          state.qreg(iChunk).initialize_from_vector(tmp);
        }
      }
    }
    else{
      for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
        list_t tmp(1ull << (chunk_bits_*qubit_scale()));
        for(int_t i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
          tmp[i] = vec[((global_chunk_index_ + iChunk) << (chunk_bits_*qubit_scale())) + i];
        }
        state.qreg(iChunk).initialize_from_vector(tmp);
      }
    }
  }
  else{
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      state.qreg(iChunk).initialize_from_vector(vec);
    }
  }
}

template <class state_t>
template <typename list_t>
void StateChunk<state_t>::initialize_from_matrix(Registers<state_t>& state, const list_t &mat)
{
  int_t iChunk;
  if(multi_chunk_distribution_){
    if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for private(iChunk)
      for(int_t ig=0;ig<num_groups_;ig++){
        for(iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++){
          list_t tmp(1ull << (chunk_bits_),1ull << (chunk_bits_));
          uint_t irow_chunk = ((iChunk + global_chunk_index_) >> ((BaseState::num_qubits_ - chunk_bits_))) << (chunk_bits_);
          uint_t icol_chunk = ((iChunk + global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - chunk_bits_)))-1)) << (chunk_bits_);

          //copy part of state for this chunk
          uint_t i,row,col;
          for(i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
            uint_t icol = i & ((1ull << chunk_bits_)-1);
            uint_t irow = i >> chunk_bits_;
            tmp[i] = mat[icol_chunk + icol + ((irow_chunk + irow) << BaseState::num_qubits_)];
          }
          state.qreg(iChunk).initialize_from_matrix(tmp);
        }
      }
    }
    else{
      for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
        list_t tmp(1ull << (chunk_bits_),1ull << (chunk_bits_));
        uint_t irow_chunk = ((iChunk + global_chunk_index_) >> ((BaseState::num_qubits_ - chunk_bits_))) << (chunk_bits_);
        uint_t icol_chunk = ((iChunk + global_chunk_index_) & ((1ull << ((BaseState::num_qubits_ - chunk_bits_)))-1)) << (chunk_bits_);

        //copy part of state for this chunk
        uint_t i,row,col;
        for(i=0;i<(1ull << (chunk_bits_*qubit_scale()));i++){
          uint_t icol = i & ((1ull << chunk_bits_)-1);
          uint_t irow = i >> chunk_bits_;
          tmp[i] = mat[icol_chunk + icol + ((irow_chunk + irow) << BaseState::num_qubits_)];
        }
        state.qreg(iChunk).initialize_from_matrix(tmp);
      }
    }
  }
  else{
    for(iChunk=0;iChunk<num_local_chunks_;iChunk++){
      state.qreg(iChunk).initialize_from_matrix(mat);
    }
  }
}

template <class state_t>
auto StateChunk<state_t>::apply_to_matrix(Registers<state_t>& state, bool copy)
{
  //this function is used to collect states over chunks
  int_t iChunk;
  uint_t size = 1ull << (chunk_bits_*qubit_scale());
  uint_t mask = (1ull << (chunk_bits_)) - 1;
  uint_t num_threads = state.qreg().get_omp_threads();

  size_t size_required = 2*(sizeof(std::complex<double>) << (BaseState::num_qubits_*2)) + (sizeof(std::complex<double>) << (chunk_bits_*2))*num_local_chunks_;
  if((size_required>>20) > Utils::get_system_memory_mb()){
    throw std::runtime_error(std::string("There is not enough memory to store states as matrix"));
  }

  auto matrix = state.qreg(0).copy_to_matrix();

  if(BaseState::distributed_rank_ == 0){
    matrix.resize(1ull << (BaseState::num_qubits_),1ull << (BaseState::num_qubits_));

    auto tmp = state.qreg(0).copy_to_matrix();
    for(iChunk=0;iChunk<num_global_chunks_;iChunk++){
      int_t i;
      uint_t irow_chunk = (iChunk >> ((BaseState::num_qubits_ - chunk_bits_))) << chunk_bits_;
      uint_t icol_chunk = (iChunk & ((1ull << ((BaseState::num_qubits_ - chunk_bits_)))-1)) << chunk_bits_;

      if(iChunk < num_local_chunks_){
        if(copy)
          tmp = state.qreg(iChunk).copy_to_matrix();
        else
          tmp = state.qreg(iChunk).move_to_matrix();
      }
#ifdef AER_MPI
      else
        recv_data(tmp.data(),size,0,iChunk);
#endif
#pragma omp parallel for if(num_threads > 1) num_threads(num_threads)
      for(i=0;i<size;i++){
        uint_t irow = i >> (chunk_bits_);
        uint_t icol = i & mask;
        uint_t idx = ((irow+irow_chunk) << (BaseState::num_qubits_)) + icol_chunk + icol;
        matrix[idx] = tmp[i];
      }
    }
  }
  else{
#ifdef AER_MPI
    //send matrices to process 0
    for(iChunk=0;iChunk<num_global_chunks_;iChunk++){
      uint_t iProc = get_process_by_chunk(iChunk);
      if(iProc == BaseState::distributed_rank_){
        if(copy){
          auto tmp = state.qreg(iChunk-global_chunk_index_).copy_to_matrix();
          send_data(tmp.data(),size,iChunk,0);
        }
        else{
          auto tmp = state.qreg(iChunk-global_chunk_index_).move_to_matrix();
          send_data(tmp.data(),size,iChunk,0);
        }
      }
    }
#endif
  }

  return matrix;
}

template <class state_t>
void StateChunk<state_t>::apply_chunk_swap(RegistersBase& state_in, const reg_t &qubits)
{
  uint_t nLarge = 1;
  uint_t q0,q1;
  int_t iChunk;

  Registers<state_t>& state = dynamic_cast<Registers<state_t>&>(state_in);

  q0 = qubits[qubits.size() - 2];
  q1 = qubits[qubits.size() - 1];

  if(qubit_scale() == 1){
    state.swap_qubit_map(q0,q1);
  }

  if(q0 > q1){
    std::swap(q0,q1);
  }

  if(q1 < chunk_bits_*qubit_scale()){
    //inside chunk
    if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for num_threads(num_groups_) 
      for(int_t ig=0;ig<num_groups_;ig++){
        for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
          state.qreg(iChunk).apply_mcswap(qubits);
      }
    }
    else{
      for(int_t ig=0;ig<num_groups_;ig++){
        for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
          state.qreg(iChunk).apply_mcswap(qubits);
      }
    }
  }
  else{ //swap over chunks
    uint_t mask0,mask1;

    mask0 = (1ull << q0);
    mask1 = (1ull << q1);
    mask0 >>= (chunk_bits_*qubit_scale());
    mask1 >>= (chunk_bits_*qubit_scale());

    if(BaseState::distributed_procs_ == 1 || (BaseState::distributed_proc_bits_ >= 0 && q1 < (BaseState::num_qubits_*qubit_scale() - BaseState::distributed_proc_bits_))){   //no data transfer between processes is needed
      auto apply_chunk_swap_1qubit = [this, mask1, &qubits, &state](int_t iGroup)
      {
        for(int_t ic = top_chunk_of_group_[iGroup];ic < top_chunk_of_group_[iGroup + 1];ic++){
          uint_t baseChunk;
          baseChunk = ic & (~mask1);
          if(ic == baseChunk)
            state.qreg(ic).apply_chunk_swap(qubits,state.qreg(ic | mask1),true);
        }
      };
      auto apply_chunk_swap_2qubits = [this, mask0, mask1, &qubits, &state](int_t iGroup)
      {
        for(int_t ic = top_chunk_of_group_[iGroup];ic < top_chunk_of_group_[iGroup + 1];ic++){
          uint_t baseChunk;
          baseChunk = ic & (~(mask0 | mask1));
          uint_t iChunk1 = baseChunk | mask0;
          uint_t iChunk2 = baseChunk | mask1;
          if(ic == iChunk1)
            state.qreg(iChunk1).apply_chunk_swap(qubits,state.qreg(iChunk2),true);
        }
      };
      if(q0 < chunk_bits_*qubit_scale())
        Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1), 0, num_groups_, apply_chunk_swap_1qubit);
      else
        Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1), 0, num_groups_, apply_chunk_swap_2qubits);
    }
#ifdef AER_MPI
    else{
      int_t iPair;
      uint_t nPair;
      uint_t baseChunk,iChunk1,iChunk2;

      if(q0 < chunk_bits_*qubit_scale())
        nLarge = 1;
      else
        nLarge = 2;

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

        nu[1] = 1ull << (BaseState::num_qubits_*qubit_scale() - q1 - 1);
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

        nu[2] = 1ull << (BaseState::num_qubits_*qubit_scale() - q1 - 1);
        ub[2] = (q1 - chunk_bits_*qubit_scale()) + 1;
        iu[2] = 0;
      }
      nPair = 1ull << (BaseState::num_qubits_*qubit_scale() - chunk_bits_*qubit_scale() - nLarge);

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

        if(iChunk1 >= chunk_index_begin_[BaseState::distributed_rank_] && iChunk1 < chunk_index_end_[BaseState::distributed_rank_]){    //chunk1 is on this process
          if(iChunk2 >= chunk_index_begin_[BaseState::distributed_rank_] && iChunk2 < chunk_index_end_[BaseState::distributed_rank_]){    //chunk2 is on this process
            state.qreg(iChunk1 - global_chunk_index_).apply_chunk_swap(qubits,state.qreg(iChunk2 - global_chunk_index_),true);
            continue;
          }
          else{
            iLocalChunk = iChunk1;
            iRemoteChunk = iChunk2;
            iProc = get_process_by_chunk(iChunk2);
          }
        }
        else{
          if(iChunk2 >= chunk_index_begin_[BaseState::distributed_rank_] && iChunk2 < chunk_index_end_[BaseState::distributed_rank_]){    //chunk2 is on this process
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

        auto pRecv = state.qreg(iLocalChunk - global_chunk_index_).recv_buffer(sizeRecv);
        MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,iPair,BaseState::distributed_comm_,&reqRecv);

        auto pSend = state.qreg(iLocalChunk - global_chunk_index_).send_buffer(sizeSend);
        MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,iPair,BaseState::distributed_comm_,&reqSend);

        MPI_Wait(&reqSend,&st);
        MPI_Wait(&reqRecv,&st);

        state.qreg(iLocalChunk - global_chunk_index_).apply_chunk_swap(qubits,iRemoteChunk);
      }
    }
#endif
  }
}

template <class state_t>
void StateChunk<state_t>::apply_multi_chunk_swap(RegistersBase& state_in, const reg_t &qubits)
{
  int_t nswap = qubits.size()/2;
  reg_t chunk_shuffle_qubits(nswap,0);
  reg_t local_swaps;
  uint_t baseChunk = 0;
  uint_t nchunk = 1ull << nswap;
  reg_t chunk_procs(nchunk);
  reg_t chunk_offset(nchunk);

  Registers<state_t>& state = dynamic_cast<Registers<state_t>&>(state_in);

  if(qubit_scale() == 1){
    for(int_t i=0;i<nswap;i++)
      state.swap_qubit_map(qubits[i*2],qubits[i*2+1]);
  }

  //define local swaps
  for(int_t i=0;i<nswap;i++){
    if(qubits[i*2] >= chunk_bits_*qubit_scale() - nswap)  //no swap required
      chunk_shuffle_qubits[qubits[i*2] + nswap - chunk_bits_*qubit_scale()] = qubits[i*2 + 1];
  }
  int_t pos = 0;
  for(int_t i=0;i<nswap;i++){
    if(qubits[i*2] < chunk_bits_*qubit_scale() - nswap){  //local swap required
      //find empty position
      while(pos < nswap){
        if(chunk_shuffle_qubits[pos] < chunk_bits_*qubit_scale()){
          chunk_shuffle_qubits[pos] = qubits[i*2 + 1];
          local_swaps.push_back(qubits[i*2]);
          local_swaps.push_back(chunk_bits_*qubit_scale() - nswap + pos);
          pos++;
          break;
        }
        pos++;
      }
    }
  }
  for(int_t i=0;i<nswap;i++)
    chunk_shuffle_qubits[i] -= chunk_bits_*qubit_scale();

  //swap inside chunks to prepare for all-to-all shuffle
  if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for 
    for(int_t ig=0;ig<num_groups_;ig++){
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
        state.qreg(iChunk).apply_multi_swaps(local_swaps);
    }
  }
  else{
    for(int_t ig=0;ig<num_groups_;ig++){
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
        state.qreg(iChunk).apply_multi_swaps(local_swaps);
    }
  }

  //apply all-to-all chunk shuffle
  int_t nPair;
  reg_t chunk_shuffle_qubits_sorted = chunk_shuffle_qubits;
  std::sort(chunk_shuffle_qubits_sorted.begin(), chunk_shuffle_qubits_sorted.end());

  nPair = num_global_chunks_ >> nswap;

  for(uint_t i=0;i<nchunk;i++){
    chunk_offset[i] = 0;
    for(uint_t k=0;k<nswap;k++){
      if(((i >> k) & 1) != 0)
        chunk_offset[i] += (1ull << chunk_shuffle_qubits[k]);
    }
  }

#ifdef AER_MPI
  std::vector<MPI_Request> reqSend(nchunk);
  std::vector<MPI_Request> reqRecv(nchunk);
#endif

  for(int_t iPair=0;iPair<nPair;iPair++){
    uint_t i1,i2,k,ii,t;
    baseChunk = 0;
    ii = iPair;
    for(k=0;k<nswap;k++){
      t = ii & ((1ull << chunk_shuffle_qubits_sorted[k]) - 1);
      baseChunk += t;
      ii = (ii - t) << 1;
    }
    baseChunk += ii;

    for(i1=0;i1<nchunk;i1++){
      chunk_procs[i1] = get_process_by_chunk(baseChunk + chunk_offset[i1]);
    }

    //all-to-all
    //send data
    for(uint_t iswap=1;iswap<nchunk;iswap++){
      uint_t sizeRecv,sizeSend;
      uint_t num_local_swap = 0;
      for(i1=0;i1<nchunk;i1++){
        i2 = i1 ^ iswap;
        if(i1 >= i2)
          continue;

        uint_t iProc1 = chunk_procs[i1];
        uint_t iProc2 = chunk_procs[i2];
        if(iProc1 != BaseState::distributed_rank_ && iProc2 != BaseState::distributed_rank_)
          continue;
        if(iProc1 == iProc2){  //on the same process
          num_local_swap++;
          continue;   //swap while data is exchanged between processes
        }
#ifdef AER_MPI
        uint_t offset1 = i1 << (chunk_bits_*qubit_scale() - nswap);
        uint_t offset2 = i2 << (chunk_bits_*qubit_scale() - nswap);
        uint_t iChunk1 = baseChunk + chunk_offset[i1] - global_chunk_index_;
        uint_t iChunk2 = baseChunk + chunk_offset[i2] - global_chunk_index_;

        int_t tid = (iPair << nswap) + iswap;

        if(iProc1 == BaseState::distributed_rank_){
          auto pRecv = state.qreg(iChunk1).recv_buffer(sizeRecv);
          MPI_Irecv(pRecv + offset2,(sizeRecv >> nswap),MPI_BYTE,iProc2,tid,BaseState::distributed_comm_,&reqRecv[i2]);

          auto pSend = state.qreg(iChunk1).send_buffer(sizeSend);
          MPI_Isend(pSend + offset2,(sizeSend >> nswap),MPI_BYTE,iProc2,tid,BaseState::distributed_comm_,&reqSend[i2]);
        }
        else{
          auto pRecv = state.qreg(iChunk2).recv_buffer(sizeRecv);
          MPI_Irecv(pRecv + offset1,(sizeRecv >> nswap),MPI_BYTE,iProc1,tid,BaseState::distributed_comm_,&reqRecv[i1]);

          auto pSend = state.qreg(iChunk2).send_buffer(sizeSend);
          MPI_Isend(pSend + offset1,(sizeSend >> nswap),MPI_BYTE,iProc1,tid,BaseState::distributed_comm_,&reqSend[i1]);
        }
#endif
      }

      //swaps inside process
      if(num_local_swap > 0){
        for(i1=0;i1<nchunk;i1++){
          i2 = i1 ^ iswap;
          if(i1 > i2)
            continue;

          uint_t iProc1 = chunk_procs[i1];
          uint_t iProc2 = chunk_procs[i2];
          if(iProc1 != BaseState::distributed_rank_ && iProc2 != BaseState::distributed_rank_)
            continue;
          if(iProc1 == iProc2){  //on the same process
            uint_t offset1 = i1 << (chunk_bits_*qubit_scale() - nswap);
            uint_t offset2 = i2 << (chunk_bits_*qubit_scale() - nswap);
            uint_t iChunk1 = baseChunk + chunk_offset[i1] - global_chunk_index_;
            uint_t iChunk2 = baseChunk + chunk_offset[i2] - global_chunk_index_;
            state.qreg(iChunk1).apply_chunk_swap(state.qreg(iChunk2),offset2,offset1,(1ull << (chunk_bits_*qubit_scale() - nswap)) );
          }
        }
      }

#ifdef AER_MPI
      //recv data
      for(i1=0;i1<nchunk;i1++){
        i2 = i1 ^ iswap;

        uint_t iProc1 = chunk_procs[i1];
        uint_t iProc2 = chunk_procs[i2];
        if(iProc1 != BaseState::distributed_rank_)
          continue;
        if(iProc1 == iProc2){  //on the same process
          continue;
        }
        uint_t iChunk1 = baseChunk + chunk_offset[i1] - global_chunk_index_;
        uint_t offset2 = i2 << (chunk_bits_*qubit_scale() - nswap);

        MPI_Status st;
        MPI_Wait(&reqSend[i2],&st);
        MPI_Wait(&reqRecv[i2],&st);

        //copy states from recv buffer to chunk
        state.qreg(iChunk1).apply_chunk_swap(state.qreg(iChunk1),offset2,offset2,(1ull << (chunk_bits_*qubit_scale() - nswap)) );
      }
#endif
    }
  }

  //restore qubits order
  if(chunk_omp_parallel_ && num_groups_ > 1){
#pragma omp parallel for 
    for(int_t ig=0;ig<num_groups_;ig++){
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
        state.qreg(iChunk).apply_multi_swaps(local_swaps);
    }
  }
  else{
    for(int_t ig=0;ig<num_groups_;ig++){
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
        state.qreg(iChunk).apply_multi_swaps(local_swaps);
    }
  }
}


template <class state_t>
void StateChunk<state_t>::apply_chunk_x(RegistersBase& state_in, const uint_t qubit)
{
  int_t iChunk;
  uint_t nLarge = 1;

  Registers<state_t>& state = dynamic_cast<Registers<state_t>&>(state_in);

  if(qubit < chunk_bits_*qubit_scale()){
    auto apply_par_mcx = [this, qubit, &state](int_t ig)
    {
      reg_t qubits(1,qubit);
      for(int_t iChunk = top_chunk_of_group_[ig];iChunk < top_chunk_of_group_[ig + 1];iChunk++)
        state.qreg(iChunk).apply_mcx(qubits);
    };
    Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1),0,num_groups_,apply_par_mcx);
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

    if(BaseState::distributed_procs_ == 1 || (BaseState::distributed_proc_bits_ >= 0 && qubit < (BaseState::num_qubits_*qubit_scale() - BaseState::distributed_proc_bits_))){   //no data transfer between processes is needed
      nPair = num_local_chunks_ >> 1;

      auto apply_par_chunk_swap = [this, mask, &qubits,&state](int_t iGroup)
      {
        for(int_t ic = top_chunk_of_group_[iGroup];ic < top_chunk_of_group_[iGroup + 1];ic++){
          uint_t pairChunk;
          pairChunk = ic ^ mask;
          if(ic < pairChunk)
            state.qreg(ic).apply_chunk_swap(qubits,state.qreg(pairChunk),true);
        }
      };
      Utils::apply_omp_parallel_for((chunk_omp_parallel_ && num_groups_ > 1),0, nPair, apply_par_chunk_swap);
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

      nu[1] = 1ull << (BaseState::num_qubits_*qubit_scale() - qubit - 1);
      ub[1] = (qubit - chunk_bits_*qubit_scale()) + 1;
      iu[1] = 0;
      nPair = 1ull << (BaseState::num_qubits_*qubit_scale() - chunk_bits_*qubit_scale() - 1);

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

        if(iChunk1 >= chunk_index_begin_[BaseState::distributed_rank_] && iChunk1 < chunk_index_end_[BaseState::distributed_rank_]){    //chunk1 is on this process
          if(iChunk2 >= chunk_index_begin_[BaseState::distributed_rank_] && iChunk2 < chunk_index_end_[BaseState::distributed_rank_]){    //chunk2 is on this process
            state.qreg(iChunk1 - global_chunk_index_).apply_chunk_swap(qubits,state.qreg(iChunk2 - global_chunk_index_),true);
            continue;
          }
          else{
            iLocalChunk = iChunk1;
            iRemoteChunk = iChunk2;
            iProc = get_process_by_chunk(iChunk2);
          }
        }
        else{
          if(iChunk2 >= chunk_index_begin_[BaseState::distributed_rank_] && iChunk2 < chunk_index_end_[BaseState::distributed_rank_]){    //chunk2 is on this process
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

        auto pSend = state.qreg(iLocalChunk - global_chunk_index_).send_buffer(sizeSend);
        MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,iPair,BaseState::distributed_comm_,&reqSend);

        auto pRecv = state.qreg(iLocalChunk - global_chunk_index_).recv_buffer(sizeRecv);
        MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,iPair,BaseState::distributed_comm_,&reqRecv);

        MPI_Wait(&reqSend,&st);
        MPI_Wait(&reqRecv,&st);

        state.qreg(iLocalChunk - global_chunk_index_).apply_chunk_swap(qubits,iRemoteChunk);
      }
    }
#endif

  }
}

template <class state_t>
void StateChunk<state_t>::send_chunk(Registers<state_t>& state, uint_t local_chunk_index, uint_t global_pair_index)
{
#ifdef AER_MPI
  MPI_Request reqSend;
  MPI_Status st;
  uint_t sizeSend;
  uint_t iProc;

  iProc = get_process_by_chunk(global_pair_index);

  auto pSend = state.qreg(local_chunk_index).send_buffer(sizeSend);
  MPI_Isend(pSend,sizeSend,MPI_BYTE,iProc,local_chunk_index + global_chunk_index_,BaseState::distributed_comm_,&reqSend);

  MPI_Wait(&reqSend,&st);

  state.qreg(local_chunk_index).release_send_buffer();
#endif
}

template <class state_t>
void StateChunk<state_t>::recv_chunk(Registers<state_t>& state, uint_t local_chunk_index, uint_t global_pair_index)
{
#ifdef AER_MPI
  MPI_Request reqRecv;
  MPI_Status st;
  uint_t sizeRecv;
  uint_t iProc;

  iProc = get_process_by_chunk(global_pair_index);

  auto pRecv = state.qreg(local_chunk_index).recv_buffer(sizeRecv);
  MPI_Irecv(pRecv,sizeRecv,MPI_BYTE,iProc,global_pair_index,BaseState::distributed_comm_,&reqRecv);

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

  MPI_Isend(pSend,size*sizeof(data_t),MPI_BYTE,iProc,myid,BaseState::distributed_comm_,&reqSend);

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

  MPI_Irecv(pRecv,size*sizeof(data_t),MPI_BYTE,iProc,pairid,BaseState::distributed_comm_,&reqRecv);

  MPI_Wait(&reqRecv,&st);
#endif
}

template <class state_t>
void StateChunk<state_t>::reduce_sum(reg_t& sum) const
{
#ifdef AER_MPI
  if(BaseState::distributed_procs_ > 1){
    uint_t i,n = sum.size();
    reg_t tmp(n);
    MPI_Allreduce(&sum[0],&tmp[0],n,MPI_UINT64_T,MPI_SUM,BaseState::distributed_comm_);
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
  if(BaseState::distributed_procs_ > 1){
    uint_t i,n = sum.size();
    rvector_t tmp(n);
    MPI_Allreduce(&sum[0],&tmp[0],n,MPI_DOUBLE_PRECISION,MPI_SUM,BaseState::distributed_comm_);
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
  if(BaseState::distributed_procs_ > 1){
    complex_t tmp;
    MPI_Allreduce(&sum,&tmp,2,MPI_DOUBLE_PRECISION,MPI_SUM,BaseState::distributed_comm_);
    sum = tmp;
  }
#endif
}

template <class state_t>
void StateChunk<state_t>::reduce_sum(double& sum) const
{
#ifdef AER_MPI
  if(BaseState::distributed_procs_ > 1){
    double tmp;
    MPI_Allreduce(&sum,&tmp,1,MPI_DOUBLE_PRECISION,MPI_SUM,BaseState::distributed_comm_);
    sum = tmp;
  }
#endif
}

template <class state_t>
void StateChunk<state_t>::gather_value(rvector_t& val) const
{
#ifdef AER_MPI
  if(BaseState::distributed_procs_ > 1){
    rvector_t tmp = val;
    MPI_Alltoall(&tmp[0],1,MPI_DOUBLE_PRECISION,&val[0],1,MPI_DOUBLE_PRECISION,BaseState::distributed_comm_);
  }
#endif
}

template <class state_t>
void StateChunk<state_t>::sync_process(void) const
{
#ifdef AER_MPI
  if(BaseState::distributed_procs_ > 1){
    MPI_Barrier(BaseState::distributed_comm_);
  }
#endif
}

//gather distributed state into vector (if memory is enough)
template <class state_t>
template <class data_t>
void StateChunk<state_t>::gather_state(std::vector<std::complex<data_t>>& state)
{
#ifdef AER_MPI
  if(BaseState::distributed_procs_ > 1){
    uint_t size,local_size,global_size,offset;
    int i;
    std::vector<int> recv_counts(BaseState::distributed_procs_);
    std::vector<int> recv_offset(BaseState::distributed_procs_);

    global_size = 0;
    for(i=0;i<BaseState::distributed_procs_;i++){
      recv_offset[i] = (int)(chunk_index_begin_[i] << (chunk_bits_*qubit_scale()))*2;
      recv_counts[i] = (int)((chunk_index_end_[i] - chunk_index_begin_[i]) << (chunk_bits_*qubit_scale()));
      global_size += recv_counts[i];
      recv_counts[i] *= 2;
    }
    if((global_size >> 21) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to gather state"));
    }
    std::vector<std::complex<data_t>> local_state = state;
    state.resize(global_size);

    if(sizeof(std::complex<data_t>) == 16){
      MPI_Allgatherv(local_state.data(),recv_counts[BaseState::distributed_rank_],MPI_DOUBLE_PRECISION,
                     state.data(),&recv_counts[0],&recv_offset[0],MPI_DOUBLE_PRECISION,BaseState::distributed_comm_);
    }
    else{
      MPI_Allgatherv(local_state.data(),recv_counts[BaseState::distributed_rank_],MPI_FLOAT,
                     state.data(),&recv_counts[0],&recv_offset[0],MPI_FLOAT,BaseState::distributed_comm_);
    }
  }
#endif
}

template <class state_t>
template <class data_t>
void StateChunk<state_t>::gather_state(AER::Vector<std::complex<data_t>>& state)
{
#ifdef AER_MPI
  if(BaseState::distributed_procs_ > 1){
    uint_t size,local_size,global_size,offset;
    int i;

    std::vector<int> recv_counts(BaseState::distributed_procs_);
    std::vector<int> recv_offset(BaseState::distributed_procs_);

    global_size = 0;
    for(i=0;i<BaseState::distributed_procs_;i++){
      recv_offset[i] = (int)(chunk_index_begin_[i] << (chunk_bits_*qubit_scale()))*2;
      recv_counts[i] = (int)((chunk_index_end_[i] - chunk_index_begin_[i]) << (chunk_bits_*qubit_scale()));
      global_size += recv_counts[i];
      recv_counts[i] *= 2;
    }
    if((global_size >> 21) > Utils::get_system_memory_mb()){
      throw std::runtime_error(std::string("There is not enough memory to gather state"));
    }
    AER::Vector<std::complex<data_t>> local_state = state;
    state.resize(global_size);

    if(sizeof(std::complex<data_t>) == 16){
      MPI_Allgatherv(local_state.data(),recv_counts[BaseState::distributed_rank_],MPI_DOUBLE_PRECISION,
                     state.data(),&recv_counts[0],&recv_offset[0],MPI_DOUBLE_PRECISION,BaseState::distributed_comm_);
    }
    else{
      MPI_Allgatherv(local_state.data(),recv_counts[BaseState::distributed_rank_],MPI_FLOAT,
                     state.data(),&recv_counts[0],&recv_offset[0],MPI_FLOAT,BaseState::distributed_comm_);
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
