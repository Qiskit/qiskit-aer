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
#include "framework/results/experiment_data.hpp"

namespace AER {
namespace Base {

//=========================================================================
// State interface base class for Qiskit-Aer
//=========================================================================

template <class state_t>
class State {

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

  State(const Operations::OpSet &opset) : opset_(opset) {}

  State(const Operations::OpSet::optypeset_t &optypes,
        const stringset_t &gates,
        const stringset_t &snapshots)
    : State(Operations::OpSet(optypes, gates, snapshots)) {};

  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

  // Returns a const reference to the states data structure
  const auto &qreg() const {return qreg_;}
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
  state_t qreg_;

  // Classical register data
  ClassicalRegister creg_;

  // Opset of instructions supported by the state
  Operations::OpSet opset_;

  // Maximum threads which may be used by the backend for OpenMP multithreading
  // Default value is single-threaded unless overridden
  int threads_ = 1;
};


//=========================================================================
// Implementations
//=========================================================================

template <class state_t>
void State<state_t>::set_config(const json_t &config) {
  (ignore_argument)config;
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
void State<state_t>::initialize_creg(uint_t num_memory, uint_t num_register) {
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
void State<state_t>::snapshot_state(const Operations::Op &op,
                                    ExperimentData &data,
                                    std::string name) const {
  name = (name.empty()) ? op.name : name;
  data.add_pershot_snapshot(name, op.string_params[0], qreg_);
}


template <class state_t>
void State<state_t>::snapshot_creg_memory(const Operations::Op &op,
                                          ExperimentData &data,
                                          std::string name) const {
  data.add_pershot_snapshot(name,
                               op.string_params[0],
                               creg_.memory_hex());
}


template <class state_t>
void State<state_t>::snapshot_creg_register(const Operations::Op &op,
                                            ExperimentData &data,
                                            std::string name) const {
  data.add_pershot_snapshot(name,
                               op.string_params[0],
                               creg_.register_hex());
}


template <class state_t>
void State<state_t>::add_creg_to_data(ExperimentData &data) const {
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
//-------------------------------------------------------------------------
} // end namespace Base
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
