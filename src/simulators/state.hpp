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

namespace AER {

namespace Base {

//=========================================================================
// State interface base class for Qiskit-Aer
//=========================================================================

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

  State(const Operations::OpSet &opset) : opset_(opset) {}

  State(const Operations::OpSet::optypeset_t &optypes,
        const stringset_t &gates,
        const stringset_t &snapshots)
    : State(Operations::OpSet(optypes, gates, snapshots)) {};

  virtual ~State();

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

  // Return the state qreg object
  auto &qreg() { return qreg_; }
  const auto &qreg() const { return qreg_; }

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

  //memory allocation (previously called before inisitalize_qreg)
  virtual bool allocate(uint_t num_qubits,uint_t block_bits,uint_t num_parallel_shots = 1){return true;}

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
  // Apply circuits and ops
  //-----------------------------------------------------------------------

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
                 bool final_ops = false)
  {
    throw std::invalid_argument("apply_ops_multi_shots is not supported in State " + name());
  }

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
  void save_creg(ExperimentResult &result,
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
  void save_data_average(ExperimentResult &result,
                         const std::string &key, const T& datum, OpType type,
                         DataSubType subtype = DataSubType::average) const;

  template <class T>
  void save_data_average(ExperimentResult &result,
                         const std::string &key, T&& datum, OpType type,
                         DataSubType subtype = DataSubType::average) const;
  
  // Save data type which is pershot and does not support accumulator or average
  // This supports DataSubTypes: single, c_single, list, c_list
  template <class T>
  void save_data_pershot(ExperimentResult &result,
                         const std::string &key, const T& datum, OpType type,
                         DataSubType subtype = DataSubType::list) const;

  template <class T>
  void save_data_pershot(ExperimentResult &result,
                         const std::string &key, T&& datum, OpType type,
                         DataSubType subtype = DataSubType::list) const;


  //save creg as count data 
  virtual void save_count_data(ExperimentResult& result,bool save_memory);

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
  virtual void snapshot_state(const Operations::Op &op, ExperimentResult &result,
                      std::string name = "") const;

  // Snapshot the classical memory bits state (single-shot)
  void snapshot_creg_memory(const Operations::Op &op, ExperimentResult &result,
                            std::string name = "memory") const;

  // Snapshot the classical register bits state (single-shot)
  void snapshot_creg_register(const Operations::Op &op, ExperimentResult &result,
                              std::string name = "register") const;


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
  virtual void set_distribution(uint_t nprocs){}

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

protected:

  // The quantum state data structure
  state_t qreg_;

  // Classical register data
  ClassicalRegister creg_;

  // Opset of instructions supported by the state
  Operations::OpSet opset_;

  // Maximum threads which may be used by the backend for OpenMP multithreading
  // Default value is single-threaded unless overridden
  int threads_ = 1;

  // Set a global phase exp(1j * theta) for the state
  bool has_global_phase_ = false;
  complex_t global_phase_ = 1;

  int_t max_matrix_qubits_ = 0;
};


//=========================================================================
// Implementations
//=========================================================================

template <class state_t>
State<state_t>::~State(void)
{
}

template <class state_t>
void State<state_t>::set_config(const json_t &config) {
  (ignore_argument)config;
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
template <typename InputIterator>
void State<state_t>::apply_ops(InputIterator first, InputIterator last,
                               ExperimentResult &result,
                               RngEngine &rng,
                               bool final_ops) {

  std::unordered_map<std::string, InputIterator> marks;
  // Simple loop over vector of input operations
  for (auto it = first; it != last; ++it) {
    switch (it->type) {
    case Operations::OpType::mark: {
      marks[it->string_params[0]] = it;
      break;
    }
    case Operations::OpType::jump: {
      if (creg_.check_conditional(*it)) {
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
      apply_op(*it, result, rng, final_ops && (it + 1 == last));
    }
    }
  }
}

template <class state_t>
std::vector<reg_t> State<state_t>::sample_measure(const reg_t &qubits,
                                                  uint_t shots,
                                                  RngEngine &rng) {
  (ignore_argument)qubits;
  (ignore_argument)shots;
  return std::vector<reg_t>();
}


template <class state_t>
void State<state_t>::initialize_creg(uint_t num_memory, uint_t num_register) 
{
  creg_.initialize(num_memory, num_register);
}


template <class state_t>
void State<state_t>::initialize_creg(uint_t num_memory,
                                     uint_t num_register,
                                     const std::string &memory_hex,
                                     const std::string &register_hex) {
  creg_.initialize(num_memory, num_register, memory_hex, register_hex);
}

template <class state_t>
void State<state_t>::save_creg(ExperimentResult &result,
                               const std::string &key,
                               DataSubType subtype) const {
  if (creg_.memory_size() == 0)
    return;
  switch (subtype) {
    case DataSubType::list:
      result.data.add_list(creg_.memory_hex(), key);
      result.metadata.add("creg", "result_types", key);
      break;
    case DataSubType::c_accum:
      result.data.add_accum(1ULL, key, creg_.memory_hex());
      result.metadata.add("creg", "result_types", key);
      break;
    default:
      throw std::runtime_error("Invalid creg data subtype for data key: " + key);
  }
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void State<state_t>::save_data_average(ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum, OpType type,
                                       DataSubType subtype) const {
  switch (subtype) {
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
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void State<state_t>::save_data_average(ExperimentResult &result,
                                       const std::string &key,
                                       T&& datum, OpType type,
                                       DataSubType subtype) const {
  switch (subtype) {
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
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void State<state_t>::save_data_pershot(ExperimentResult &result,
                                       const std::string &key,
                                       const T& datum, OpType type,
                                       DataSubType subtype) const {
  switch (subtype) {
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
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
}

template <class state_t>
template <class T>
void State<state_t>::save_data_pershot(ExperimentResult &result, 
                                       const std::string &key,
                                       T&& datum, OpType type,
                                       DataSubType subtype) const {
  switch (subtype) {
    case DataSubType::single:
      result.data.add_single(std::move(datum), key);
      break;
    case DataSubType::c_single:
      result.data.add_single(std::move(datum), key, creg_.memory_hex());
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
  result.metadata.add(type, "result_types", key);
  result.metadata.add(subtype, "result_subtypes", key);
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
void State<state_t>::snapshot_state(const Operations::Op &op,
                                    ExperimentResult &result,
                                    std::string name) const {
  name = (name.empty()) ? op.name : name;
  result.legacy_data.add_pershot_snapshot(name, op.string_params[0], qreg_);
}


template <class state_t>
void State<state_t>::snapshot_creg_memory(const Operations::Op &op,
                                          ExperimentResult &result,
                                          std::string name) const {
  result.legacy_data.add_pershot_snapshot(name,
                               op.string_params[0],
                               creg_.memory_hex());
}


template <class state_t>
void State<state_t>::snapshot_creg_register(const Operations::Op &op,
                                            ExperimentResult &result,
                                            std::string name) const {
  result.legacy_data.add_pershot_snapshot(name,
                               op.string_params[0],
                               creg_.register_hex());
}


template <class state_t>
void State<state_t>::apply_save_expval(const Operations::Op &op,
                                       ExperimentResult &result){
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
    save_data_average(result, op.string_params[0], expval_var, op.type, op.save_type);
  } else {
    save_data_average(result, op.string_params[0], expval, op.type, op.save_type);
  }
}

template <class state_t>
void State<state_t>::save_count_data(ExperimentResult& result,bool save_memory)
{
  if (creg_.memory_size() > 0) {
    std::string memory_hex = creg_.memory_hex();
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
