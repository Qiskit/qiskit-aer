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

namespace QuantumState {

using OpItr = std::vector<Operations::Op>::const_iterator;

//=========================================================================
// State interface base class for Qiskit-Aer
//=========================================================================

class Base {
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

  Base(const Operations::OpSet &opset) : opset_(opset) 
  {
  }

  virtual ~Base() = default;

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

  // Return the state creg object
  virtual ClassicalRegister& creg(void) = 0;
  virtual const ClassicalRegister& creg(void) const = 0;

  // Return the state opset object
  Operations::OpSet &opset() { return opset_; }
  const Operations::OpSet &opset() const { return opset_; }

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

  // Return an estimate of the required memory for implementing the
  // specified sequence of operations on a `num_qubit` sized State.
  virtual size_t required_memory_mb(uint_t num_qubits,
                                    OpItr first, OpItr last)
                                    const = 0;

  //memory allocation (previously called before inisitalize_qreg)
  virtual bool allocate(uint_t num_qubits,uint_t block_bits,uint_t num_parallel_shots = 1){return true;}

  // Return the expectation value of a N-qubit Pauli operator
  // If the simulator does not support Pauli expectation value this should
  // raise an exception.
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string& pauli) = 0;

  // Initializes the State to the default state.
  // Typically this is the n-qubit all |0> state
  virtual void initialize_qreg(const uint_t num_qubits) = 0;

  //-----------------------------------------------------------------------
  // ClassicalRegister methods
  //-----------------------------------------------------------------------

  // Initialize classical memory and register to default value (all-0)
  virtual void initialize_creg(uint_t num_memory, uint_t num_register) = 0;

  // Initialize classical memory and register to specific values
  virtual void initialize_creg(uint_t num_memory,
                               uint_t num_register,
                               const std::string &memory_hex,
                               const std::string &register_hex) = 0;

  //-----------------------------------------------------------------------
  // Apply circuits and ops
  //-----------------------------------------------------------------------

  // Apply the global phase
  virtual void apply_global_phase() {};

  // Apply a single operation
  // The `final_op` flag indicates no more instructions will be applied
  // to the state after this sequence, so the state can be modified at the
  // end of the instructions.
  virtual void apply_op(const Operations::Op &op,
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
  virtual void apply_ops(OpItr first, OpItr last,
                         ExperimentResult &result, RngEngine &rng, bool final_ops = false) = 0;

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
                                            RngEngine &rng) = 0;

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Sets the number of threads available to the State implementation
  // If negative there is no restriction on the backend
  virtual inline void set_parallelization(int n) {threads_ = n;}

  // Set a complex global phase value exp(1j * theta) for the state
  void set_global_phase(double theta);

  // Set a complex global phase value exp(1j * theta) for the state
  void add_global_phase(double theta);

  //set number of processes to be distributed
  virtual void set_distribution(uint_t nprocs) = 0;

  //set maximum number of qubits for matrix multiplication
  virtual void set_max_matrix_qubits(int_t bits)
  {
    max_matrix_qubits_ = bits;
  }

  virtual void set_parallel_shots(int shots)
  {
    parallel_shots_ = shots;
  }

  void enable_shot_branching(bool flg)
  {
    enable_shot_branching_ = flg;
  }
  void enable_batch_execution(bool flg)
  {
    enable_batch_execution_ = flg;
  }

  //set max number of shots to execute in a batch (used in StateChunk class)
  virtual void set_max_bached_shots(uint_t shots){}

  //Does this state support multi-chunk distribution?
  virtual bool multi_chunk_distribution_supported(void){return false;}

  //Does this state support multi-shot parallelization?
  virtual bool multi_shot_parallelization_supported(void){return false;}

  //Does this state support runtime noise sampling?
  virtual bool runtime_noise_sampling_supported(void){return false;}

  //-----------------------------------------------------------------------
  // Common instructions
  //-----------------------------------------------------------------------
 
  // Apply a save expectation value instruction
  void apply_save_expval(const Operations::Op &op, ExperimentResult &result);

protected:
  // Opset of instructions supported by the state
  Operations::OpSet opset_;

  // Maximum threads which may be used by the backend for OpenMP multithreading
  // Default value is single-threaded unless overridden
  int threads_ = 1;

  // Save counts as memory list
  bool save_creg_memory_ = false;

  // Set a global phase exp(1j * theta) for the state
  bool has_global_phase_ = false;
  complex_t global_phase_ = 1;

  int_t max_matrix_qubits_ = 0;

  //OMP parallel shots 
  int parallel_shots_ = 1;

  //shot branching
  bool enable_shot_branching_ = false;
  bool enable_batch_execution_ = false;  //apply the same op to multiple states, if enable_shot_branching_ is false this flag is used for batched execution on GPU

  std::string sim_device_name_ = "CPU";
};

void Base::set_config(const json_t &config) 
{
  JSON::get_value(sim_device_name_, "device", config);

  // Load config for memory (creg list data)
  JSON::get_value(save_creg_memory_, "memory", config);

#ifdef AER_CUSTATEVEC
  //cuStateVec configs
  if(JSON::check_key("cuStateVec_enable", config)) {
    JSON::get_value(cuStateVec_enable_, "cuStateVec_enable", config);
  }
#endif
}

void Base::set_global_phase(double theta) 
{
  if (Linalg::almost_equal(theta, 0.0)) {
    has_global_phase_ = false;
    global_phase_ = 1;
  }
  else {
    has_global_phase_ = true;
    global_phase_ = std::exp(complex_t(0.0, theta));
  }
}

void Base::add_global_phase(double theta) 
{
  if (Linalg::almost_equal(theta, 0.0)) 
    return;
  
  has_global_phase_ = true;
  global_phase_ *= std::exp(complex_t(0.0, theta));
}

//=========================================================================
// State interface base class for Qiskit-Aer
//=========================================================================

template <class state_t>
class State : public Base {
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

  State(const Operations::OpSet &opset) : Base(opset) 
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
  ClassicalRegister& creg() override final { return state_.creg(); }
  const ClassicalRegister& creg() const override final { return state_.creg(); }

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

  // Initializes the State to the default state.
  // Typically this is the n-qubit all |0> state
  void initialize_qreg(const uint_t num_qubits) override;

  // Initializes the State to a specific state.
  virtual void initialize_qreg(const state_t &state);

  //memory allocation (previously called before inisitalize_qreg)
  virtual bool allocate(uint_t num_qubits,uint_t block_bits,uint_t num_initial_states = 1);

  virtual bool allocate_state(RegistersBase& state, uint_t num_max_shots = 1){return true;}

  // Return the expectation value of a N-qubit Pauli operator
  // If the simulator does not support Pauli expectation value this should
  // raise an exception.
  double expval_pauli(const reg_t &qubits,
                              const std::string& pauli) override final
  {
    return expval_pauli(state_, qubits, pauli);
  }

  virtual double expval_pauli(RegistersBase& state, const reg_t &qubits,
                              const std::string& pauli) = 0;

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
  // Apply circuits and ops
  //-----------------------------------------------------------------------

  // Apply a single operation
  // The `final_op` flag indicates no more instructions will be applied
  // to the state after this sequence, so the state can be modified at the
  // end of the instructions.
  void apply_op(
                        const Operations::Op &op,
                        ExperimentResult &result,
                        RngEngine& rng,
                        bool final_op = false) override final
  {
    apply_op(state_, op, result, rng, final_op);
  }

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
                 bool final_ops = false) override;

  //run multiple shots
  void run_shots(OpItr first,
                 OpItr last,
                 const json_t &config,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 const uint_t rng_seed,
                 const uint_t num_shots);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Load any settings for the State class from a config JSON
  void set_config(const json_t &config) override;

  //set number of processes to be distributed
  void set_distribution(uint_t nprocs) override final;


  //-----------------------------------------------------------------------
  // Common instructions
  //-----------------------------------------------------------------------
 
  // Apply a save expectation value instruction
  void apply_save_expval(Registers<state_t>& state, const Operations::Op &op, ExperimentResult &result);

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
  std::vector<reg_t> sample_measure(const reg_t &qubits,
                                            uint_t shots,
                                            RngEngine &rng) override final
  {
    return sample_measure(state_, qubits, shots, rng);
  }
  virtual std::vector<reg_t> sample_measure(RegistersBase& state, const reg_t &qubits,
                                            uint_t shots,
                                            RngEngine &rng);

  void measure_sampler(OpItr first_meas, OpItr last_meas, uint_t num_shots, 
                       ExperimentResult &result, RngEngine& rng)
  {
    std::vector<ClassicalRegister> cregs(1);  //this is not used for this call
    measure_sampler(state_, first_meas, last_meas, num_shots, result, rng, true, cregs.begin());
  }

  void measure_sampler(Registers<state_t>& state, OpItr first_meas, OpItr last_meas, uint_t num_shots, 
                       ExperimentResult &result, RngEngine& rng, bool save_results, std::vector<ClassicalRegister>::iterator creg_save);

  //-----------------------------------------------------------------------
  // Standard snapshots
  //-----------------------------------------------------------------------

  // Snapshot the classical memory bits state (single-shot)
  void snapshot_creg_memory(Registers<state_t>& state, const Operations::Op &op, ExperimentResult &result,
                            std::string name = "memory") const;

  // Snapshot the classical register bits state (single-shot)
  void snapshot_creg_register(Registers<state_t>& state, const Operations::Op &op, ExperimentResult &result,
                              std::string name = "register") const;
  // Snapshot the current statevector (single-shot)
  // if type_label is the empty string the operation type will be used for the type
  virtual void snapshot_state(Registers<state_t>& state, const Operations::Op &op, ExperimentResult &result,
                      std::string name = "") const;

protected:
  // Initializes the State to the default state.
  // Typically this is the n-qubit all |0> state
  virtual void initialize_qreg_state(RegistersBase& state, const uint_t num_qubits) = 0;

  // Initializes the State to a specific state.
  virtual void initialize_qreg_state(RegistersBase& state, const state_t &src_state) = 0;

  // Initialize classical memory and register to default value (all-0)
  virtual void initialize_creg_state(RegistersBase& state, uint_t num_memory, uint_t num_register);

  // Initialize classical memory and register to specific values
  virtual void initialize_creg_state(RegistersBase& state, 
                       uint_t num_memory,
                       uint_t num_register,
                       const std::string &memory_hex,
                       const std::string &register_hex);

  virtual void initialize_creg_state(RegistersBase& state, const ClassicalRegister& creg);

  // Load any settings for the State class from a config JSON
  virtual void set_state_config(RegistersBase& state, const json_t &config){}

  virtual void apply_ops(RegistersBase& state,
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
                 const uint_t num_shots);

  virtual bool run_shots_with_batched_execution(
                 OpItr first,
                 OpItr last,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 const uint_t rng_seed,
                 const uint_t num_shots)
  {
    return false;   //return true if this method is supported
  }

  //sample noise function, this is used to avoid compile error for Superoperator::State referred from noise/quantum_error.h
  virtual std::vector<Operations::Op> sample_noise(const Noise::NoiseModel &noise, const Operations::Op &op, RngEngine &rng)
  {
    return std::vector<Operations::Op>();
  }

  //runtime noise sampling for shot branching
  void apply_runtime_noise_sampling(RegistersBase& state, const Operations::Op &op, const Noise::NoiseModel &noise);

  // The quantum state and Classical register data structure for single shot execution
  Registers<state_t> state_;

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

  bool runtime_noise_sampled_ = false;  //true when runtime noise sampling is done

  //cuStateVec settings
  bool cuStateVec_enable_ = false;

  virtual bool shot_branching_supported(void)
  {
    return false;   //return true if simulation method supports
  }

  uint_t get_max_allocatable_shots(const uint_t num_qubits, OpItr first, OpItr last);

  //gather cregs 
  void gather_creg_memory(std::vector<ClassicalRegister>& cregs, uint_t num_local);
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
  Base::set_config(config);

#ifdef AER_CUSTATEVEC
  //cuStateVec configs
  if(JSON::check_key("cuStateVec_enable", config)) {
    JSON::get_value(cuStateVec_enable_, "cuStateVec_enable", config);
  }
#endif

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
void State<state_t>::initialize_qreg(const uint_t num_qubits)
{
  initialize_qreg_state(state_, num_qubits);
}

template <class state_t>
void State<state_t>::initialize_qreg(const state_t &state)
{
  initialize_qreg_state(state_, state);
}

template <class state_t>
void State<state_t>::apply_ops(OpItr first, OpItr last,
                               ExperimentResult &result,
                               RngEngine &rng,
                               bool final_ops) 
{
  // Simple loop over vector of input operations
  for (auto it = first; it != last; ++it) {
    switch (it->type) {
      case Operations::OpType::mark: {
        state_.marks()[it->string_params[0]] = it;
        break;
      }
      case Operations::OpType::jump: {
        if (state_.creg().check_conditional(*it)) {
          const auto& mark_name = it->string_params[0];
          auto mark_it = state_.marks().find(mark_name);
          if (mark_it != state_.marks().end()) {
            it = mark_it->second;
          } else {
            for (++it; it != last; ++it) {
              if (it->type == Operations::OpType::mark) {
                state_.marks()[it->string_params[0]] = it;
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
        apply_op(state_, *it, result, rng, final_ops && (it + 1 == last));
      }
    }
  }
}

template <class state_t>
std::vector<reg_t> State<state_t>::sample_measure(RegistersBase& state, const reg_t &qubits,
                                             uint_t shots,
                                             RngEngine &rng) {
  (ignore_argument)qubits;
  (ignore_argument)shots;
  return std::vector<reg_t>();
}

template <class state_t>
void State<state_t>::apply_ops(RegistersBase& state_in,
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
        if(Base::enable_shot_branching_ && state.num_branch() > 0){
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
               const uint_t num_shots)
{
  num_max_shots_ = get_max_allocatable_shots(num_qubits_, first, last);

  Base::enable_shot_branching_ &= shot_branching_supported();
  if(Base::enable_shot_branching_ && num_max_shots_ > 1){
    return run_shots_with_branching(first, last, config, noise, result, rng_seed, num_shots);
  }
  else if(Base::enable_batch_execution_ && multi_shot_parallelization_supported()){
    if(run_shots_with_batched_execution(first, last, noise, result, rng_seed, num_shots))
      return;
  }

  bool batch_shots_tmp = Base::enable_batch_execution_; //save this option and disable for single shot execution
  Base::enable_batch_execution_ = false;

  uint_t shot_index = num_shots*distributed_rank_/distributed_procs_;
  uint_t num_local_shots = (num_shots*(distributed_rank_+1)/distributed_procs_) - shot_index;
  std::vector<ClassicalRegister> cregs;   //storage for cregs for shots
  if(num_shots != num_local_shots){
    cregs.resize(num_local_shots);
  }

  int_t par_shots = Base::parallel_shots_;
  if(par_shots > num_max_shots_)
    par_shots = num_max_shots_;

  std::vector<ExperimentResult> par_results(par_shots);
  std::vector<Registers<state_t>> states(par_shots);

  for(int_t i=0;i<par_shots;i++){
    states[i].allocate(1);  //allocate single chunk

    // allocate qubit register
    allocate_state(states[i], 1);
  }

  auto run_single_shot = [this,&par_results,config,shot_index, num_shots, num_local_shots,par_shots,rng_seed, first, last, &states, &noise, &cregs](int_t i){
    uint_t i_shot,shot_end;
    i_shot = num_local_shots*i/par_shots;
    shot_end = num_local_shots*(i+1)/par_shots;

    // Set state config
    set_state_config(states[i], config);

    for(;i_shot<shot_end;i_shot++){
      RngEngine rng;
      rng.set_seed(rng_seed + shot_index + i_shot);

      initialize_qreg_state(states[i], num_qubits_);
      initialize_creg_state(states[i], num_creg_memory_, num_creg_registers_);

      apply_ops(states[i], first,last,noise, par_results[i], rng, true);

      if(num_shots != num_local_shots){
        //store cregs into array if shots are distributed on MPI processes
        cregs[i_shot] = states[i].creg();
      }
      else{
        //otherwise save to result
        if(states[i].creg().memory_size() > 0) {
          std::string memory_hex = states[i].creg().memory_hex();
          par_results[i].data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
          if (Base::save_creg_memory_) {
            par_results[i].data.add_list(std::move(memory_hex), "memory");
          }
        }
      }
    }
  };
  Utils::apply_omp_parallel_for((par_shots > 1), 0, par_shots, run_single_shot, par_shots);

  if(num_shots != num_local_shots){
    //gather cregs among MPI processes
    gather_creg_memory(cregs, num_local_shots);

    //save cregs to result
    auto save_cregs = [this,&par_results, &cregs, num_shots, par_shots](int_t i){
      uint_t i_shot,shot_end;
      i_shot = num_shots*i/par_shots;
      shot_end = num_shots*(i+1)/par_shots;

      for(;i_shot<shot_end;i_shot++){
        if(cregs[i_shot].memory_size() > 0) {
          std::string memory_hex = cregs[i_shot].memory_hex();
          par_results[i].data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
          if (Base::save_creg_memory_) {
            par_results[i].data.add_list(std::move(memory_hex), "memory");
          }
        }
      }
    };
    Utils::apply_omp_parallel_for((par_shots > 1),0,par_shots,save_cregs, par_shots);
  }

  for (auto &res : par_results) {
    result.combine(std::move(res));
  }
  add_metadata(result);

  Base::enable_batch_execution_ = batch_shots_tmp;

  result.metadata.add(false, "shot_branching_enabled");
  result.metadata.add(false, "runtime_noise_sampling_enabled");
}

template <class state_t>
void State<state_t>::run_shots_with_branching(OpItr first,
                 OpItr last,
                 const json_t &config,
                 const Noise::NoiseModel &noise,
                 ExperimentResult &result,
                 const uint_t rng_seed,
                 const uint_t num_shots)
{
  RngEngine rng;
  rng.set_seed(rng_seed);   //this is not used actually

  //check if there is sequence of measure at the end of operations
  bool can_sample = false;
  OpItr measure_seq = last;
  OpItr it = last - 1;
  int_t num_measure = 0;

  do{
    if(it->type != OpType::measure){
      measure_seq = it + 1;
      break;
    }
    num_measure += it->qubits.size();
    it--;
  }while(it != first);

  if(num_measure >= num_qubits_ && measure_seq != last){
    can_sample = true;
  }
  else{
    measure_seq = last;
  }

  uint_t shot_index = num_shots*distributed_rank_/distributed_procs_;
  uint_t num_local_shots = (num_shots*(distributed_rank_+1)/distributed_procs_) - shot_index;

  std::vector<ClassicalRegister> cregs(num_local_shots);   //storage for cregs for local shots

  std::vector<RngEngine> reserved_shots(num_local_shots);
  for(int_t i=0;i<num_local_shots;i++){
    reserved_shots[i].set_seed(rng_seed + shot_index + i);
  }

  uint_t num_shots_saved = 0;

  std::vector<ExperimentResult> par_results(Base::parallel_shots_);

  //cuStateVec is not supported
  cuStateVec_enable_ = false;

  while(reserved_shots.size() > 0){
    std::vector<std::shared_ptr<Registers<state_t>>> states;
    std::shared_ptr<Registers<state_t>> initial_state = std::make_shared<Registers<state_t>>();

    allocate_state(*initial_state, std::min((uint_t)reserved_shots.size(),num_max_shots_) );
    initial_state->set_shots(reserved_shots);
    reserved_shots.clear();

    set_state_config(*initial_state, config);
    initialize_qreg_state(*initial_state, num_qubits_);
    initialize_creg_state(*initial_state, num_creg_memory_, num_creg_registers_);

    states.push_back(initial_state);

    //functor for ops execution
    auto apply_ops_func = [this, &states, &rng, &noise, &par_results, measure_seq](int_t i)
    {
      uint_t istate,state_end;
      istate = states.size()*i/Base::parallel_shots_;
      state_end = states.size()*(i+1)/Base::parallel_shots_;
      uint_t nbranch = 0;

      for(;istate<state_end;istate++){
        apply_ops(*states[istate], states[istate]->next_iter(),measure_seq ,noise,par_results[i], rng, true);
        nbranch += states[istate]->num_branch();
      }
      return nbranch;
    };

    initial_state->next_iter() = first;
    uint_t nactive = 1;
    while(nactive > 0){   //loop until all states execute all ops
      uint_t nbranch = 0;

      //apply ops until a branch operation comes (reset, measure, kraus, initialize, noises)
      nbranch = Utils::apply_omp_parallel_for_reduction_int((Base::parallel_shots_ > 1 && states.size() > 1), 0, Base::parallel_shots_, apply_ops_func, Base::parallel_shots_);

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
                  initialize_creg_state(*states[i], states[i]->branch(j).creg_);
                  istart = j+1;
                  break;
                }
              }
            }
            else{ //otherwise set branched shots 
              states[i]->set_shots(states[i]->branch(0).shots_);
              states[i]->additional_ops() = states[i]->branch(0).additional_ops_;
              initialize_creg_state(*states[i], states[i]->branch(0).creg_);
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
                  initialize_creg_state(*states[pos], states[i]->branch(j).creg_);
                }
              }
            }
          }
        }

        //then execute ops applied after branching (reset, Kraus, noises, etc.)
        auto apply_additional_ops_func = [this, &states, &rng, &noise, &par_results](int_t i)
        {
          uint_t istate,state_end;
          istate = states.size()*i/Base::parallel_shots_;
          state_end = states.size()*(i+1)/Base::parallel_shots_;
          uint_t nbranch = 0;

          for(;istate<state_end;istate++){
            states[istate]->clear_branch();
            for(int_t j=0;j<states[istate]->additional_ops().size();j++){
              apply_op(*states[istate], states[istate]->additional_ops()[j], par_results[i], rng, false );

              if(states[istate]->num_branch() > 0){  //check if there are new branches
                //if there are additional ops remaining, queue them on new branches
                for(int_t k=j+1;k<states[istate]->additional_ops().size();k++){
                  for(int_t l=0;l<states[istate]->num_branch();l++)
                    states[istate]->add_op_after_branch(l,states[istate]->additional_ops()[k]);
                }
                nbranch += states[istate]->num_branch();
                break;
              }
            }
            states[istate]->clear_additional_ops();
          }
          return nbranch;
        };
        nbranch = Utils::apply_omp_parallel_for_reduction_int((Base::parallel_shots_ > 1 && states.size() > 1), 0, Base::parallel_shots_, apply_additional_ops_func, Base::parallel_shots_);
      }

      nactive = 0;
      for(int_t i=0;i<states.size();i++){
        if(states[i]->next_iter() != measure_seq)
          nactive++;
      }
    }

    reg_t creg_pos(states.size());
    for(int_t i=0;i<states.size();i++){
      creg_pos[i] = num_shots_saved;
      num_shots_saved += states[i]->num_shots();
    }

    //save cregs to array
    auto save_creg_func = [this, &states, &cregs, &creg_pos](int_t i)
    {
      for(int_t j=0;j<states[i]->num_shots();j++){
        cregs[creg_pos[i] + j] = states[i]->creg();
      }
    };
    Utils::apply_omp_parallel_for((Base::parallel_shots_ > 1 && states.size() > 1), 0, states.size(), save_creg_func, Base::parallel_shots_);

    //apply sampling measure for each branch
    if(can_sample){
      auto sampling_measure_func = [this, &states, &cregs, &creg_pos, &par_results, &rng, measure_seq, last](int_t i)
      {
        uint_t istate,state_end;
        istate = states.size()*i/Base::parallel_shots_;
        state_end = states.size()*(i+1)/Base::parallel_shots_;

        for(;istate<state_end;istate++)
          measure_sampler(*states[istate], measure_seq, last, states[istate]->num_shots(), par_results[i], rng, false, cregs.begin() + creg_pos[istate]);
      };
      Utils::apply_omp_parallel_for((Base::parallel_shots_ > 1 && states.size() > 1), 0, Base::parallel_shots_, sampling_measure_func, Base::parallel_shots_);
    }

    //clear
    for(int_t i=0;i<states.size();i++){
      states[i].reset();
    }
    states.clear();
    initial_state.reset();
  }

  gather_creg_memory(cregs, num_local_shots);

  //save cregs to result
  auto save_cregs = [this,&par_results, &cregs, num_shots](int_t i){
    uint_t i_shot,shot_end;
    i_shot = num_shots*i/Base::parallel_shots_;
    shot_end = num_shots*(i+1)/Base::parallel_shots_;

    for(;i_shot<shot_end;i_shot++){
      if(cregs[i_shot].memory_size() > 0) {
        std::string memory_hex = cregs[i_shot].memory_hex();
        par_results[i].data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
        if (Base::save_creg_memory_) {
          par_results[i].data.add_list(std::move(memory_hex), "memory");
        }
      }
    }
  };
  Utils::apply_omp_parallel_for((Base::parallel_shots_ > 1),0,Base::parallel_shots_,save_cregs, Base::parallel_shots_);

  for (auto &res : par_results) {
    result.combine(std::move(res));
  }

  result.metadata.add(true, "shot_branching_enabled");
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
    std::vector<Operations::Op> noise_ops = sample_noise(noise, op, state.rng_shots(i));

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
void State<state_t>::initialize_creg(uint_t num_memory, uint_t num_register) 
{
  num_creg_memory_ = num_memory;
  num_creg_registers_ = num_register;

  initialize_creg_state(state_, num_memory, num_register);
}
template <class state_t>
void State<state_t>::initialize_creg(uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) 
{
  num_creg_memory_ = num_memory;
  num_creg_registers_ = num_register;

  initialize_creg_state(state_, num_memory, num_register, memory_hex, register_hex);
}

template <class state_t>
void State<state_t>::initialize_creg_state(RegistersBase& state, uint_t num_memory, uint_t num_register) 
{
  num_creg_memory_ = num_memory;
  num_creg_registers_ = num_register;

  state.creg().initialize(num_memory, num_register);
}

template <class state_t>
void State<state_t>::initialize_creg_state(RegistersBase& state, 
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
void State<state_t>::initialize_creg_state(RegistersBase& state, const ClassicalRegister& creg)
{
  state.creg() = creg;
}


template <class state_t>
void State<state_t>::snapshot_state(Registers<state_t>& state, const Operations::Op &op,
                                    ExperimentResult &result,
                                    std::string name) const 
{
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
    result.save_data_average(state.creg(), op.string_params[0], expval_var, op.type, op.save_type);
  } else {
    result.save_data_average(state.creg(), op.string_params[0], expval, op.type, op.save_type);
  }
}

template <class state_t>
void State<state_t>::gather_creg_memory(std::vector<ClassicalRegister>& cregs, uint_t num_local)
{
#ifdef AER_MPI
  int_t i,j;
  uint_t n64,i64,ibit,mem_size;
  uint_t num_global = 0;

  if(distributed_procs_ == 1)
    return;
  mem_size = cregs[0].memory_size();
  if(mem_size == 0)
    return;

  std::vector<int> recv_counts(distributed_procs_);
  std::vector<int> recv_offset(distributed_procs_);
  std::vector<int> tmp(distributed_procs_);

  for(i=0;i<distributed_procs_;i++){
    recv_counts[i] = num_local;
  }

  tmp = recv_counts;
  MPI_Alltoall(&tmp[0],1,MPI_INTEGER,&recv_counts[0],1,MPI_INTEGER,distributed_comm_);

  for(i=0;i<distributed_procs_;i++){
    recv_offset[i] = num_global;
    num_global += recv_counts[i];
  }
  uint_t global_id = recv_offset[distributed_rank_];

  //number of 64-bit integers per memory
  n64 = (mem_size + 63) >> 6;

  reg_t bin_memory(n64*num_local,0);
  //compress memory string to binary
#pragma omp parallel for private(i,j,i64,ibit)
  for(i=0;i<num_local;i++){
    for(j=0;j<mem_size;j++){
      i64 = j >> 6;
      ibit = j & 63;
      if(cregs[i].creg_memory()[j] == '1'){
        bin_memory[i*n64 + i64] |= (1ull << ibit);
      }
    }
  }

  reg_t recv(n64*num_global);

  MPI_Allgatherv(&bin_memory[0],n64*num_local,MPI_UINT64_T,
                 &recv[0],&recv_counts[0],&recv_offset[0],MPI_UINT64_T, distributed_comm_);

  cregs.clear();
  cregs.resize(num_global);

  //store gathered memory
#pragma omp parallel for private(i,j,i64,ibit)
  for(i=0;i<num_global;i++){
    cregs[i].initialize(mem_size,mem_size);
    for(j=0;j<mem_size;j++){
      i64 = j >> 6;
      ibit = j & 63;
      if(((recv[i*n64 + i64] >> ibit) & 1) == 1)
        cregs[i].creg_memory()[j] = '1';
      else
        cregs[i].creg_memory()[j] = '0';
    }
  }
#endif
}

template <class state_t>
void State<state_t>::measure_sampler(Registers<state_t>& state, OpItr first_meas, OpItr last_meas, uint_t num_shots, 
                       ExperimentResult &result, RngEngine& rng, bool save_results, std::vector<ClassicalRegister>::iterator creg_save)
{
    using myclock_t = std::chrono::high_resolution_clock;

  // Check if meas_circ is empty, and if so return initial creg
  if (first_meas == last_meas) {
    if(save_results){
      while (num_shots-- > 0) {
        result.save_count_data(state.creg(), Base::save_creg_memory_);
      }
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
  auto timer_start = myclock_t::now();
  auto all_samples = sample_measure(state, meas_qubits, num_shots, rng);
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
  if(save_results){
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
        result.save_count_data(creg, Base::save_creg_memory_);

      // pop off processed sample
      all_samples.pop_back();
    }
  }
  else{
    for(int_t i=0;i<all_samples.size();i++){
      // process memory bit measurements
      for (const auto &pair : memory_map) {
        (creg_save + i)->store_measure(reg_t({all_samples[i][pair.second]}), reg_t({pair.first}),
                           reg_t());
      }
      // process register bit measurements
      for (const auto &pair : register_map) {
        (creg_save + i)->store_measure(reg_t({all_samples[i][pair.second]}), reg_t(),
                           reg_t({pair.first}));
      }

      // process read out errors for memory and registers
      for (const Operations::Op &roerror : roerror_ops) {
        (creg_save + i)->apply_roerror(roerror, rng);
      }
    }
  }
}

//-------------------------------------------------------------------------
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
