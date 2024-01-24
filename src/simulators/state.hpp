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

#include "framework/config.hpp"
#include "framework/creg.hpp"
#include "framework/json.hpp"
#include "framework/opset.hpp"
#include "framework/results/experiment_result.hpp"
#include "framework/types.hpp"

#include "noise/noise_model.hpp"

namespace AER {

namespace QuantumState {

//=========================================================================
// State interface base class for Qiskit-Aer
//=========================================================================

class Base {
public:
  using ignore_argument = void;
  using DataSubType = Operations::DataSubType;
  using OpType = Operations::OpType;
  using OpItr = std::vector<Operations::Op>::const_iterator;

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
  // - `OpType::barrier` if barrier is supported
  // - `OpType::matrix` if arbitrary unitary matrices are supported
  // - `OpType::kraus` if general Kraus noise channels are supported
  //
  // For gate ops allowed gates are specified by a set of string names,
  // for example this could include {"u1", "u2", "u3", "U", "cx", "CX"}

  Base(const Operations::OpSet &opset) : opset_(opset) { cregs_.resize(1); }

  virtual ~Base() = default;

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

  // Return the state creg object
  auto &creg(uint_t idx = 0) { return cregs_[idx]; }
  const auto &creg(uint_t idx = 0) const { return cregs_[idx]; }
  std::vector<ClassicalRegister> &cregs() { return cregs_; }
  const std::vector<ClassicalRegister> &cregs() const { return cregs_; }

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
  virtual size_t
  required_memory_mb(uint_t num_qubits,
                     const std::vector<Operations::Op> &ops) const = 0;

  // memory allocation (previously called before inisitalize_qreg)
  virtual bool allocate(uint_t num_qubits, uint_t block_bits,
                        uint_t num_parallel_shots = 1) {
    return true;
  }

  // Return the expectation value of a N-qubit Pauli operator
  // If the simulator does not support Pauli expectation value this should
  // raise an exception.
  virtual double expval_pauli(const reg_t &qubits,
                              const std::string &pauli) = 0;

  // Initializes the State to the default state.
  // Typically this is the n-qubit all |0> state
  virtual void initialize_qreg(uint_t num_qubits) = 0;

  // validate parameters in input operations
  virtual bool
  validate_parameters(const std::vector<Operations::Op> &ops) const {
    return true;
  }

  //-----------------------------------------------------------------------
  // ClassicalRegister methods
  //-----------------------------------------------------------------------

  // Initialize classical memory and register to default value (all-0)
  virtual void initialize_creg(uint_t num_memory, uint_t num_register);

  // Initialize classical memory and register to specific values
  virtual void initialize_creg(uint_t num_memory, uint_t num_register,
                               const std::string &memory_hex,
                               const std::string &register_hex);

  //-----------------------------------------------------------------------
  // Apply circuits and ops
  //-----------------------------------------------------------------------

  // Apply the global phase
  virtual void apply_global_phase(){};

  // Apply a single operation
  // The `final_op` flag indicates no more instructions will be applied
  // to the state after this sequence, so the state can be modified at the
  // end of the instructions.
  virtual void apply_op(const Operations::Op &op, ExperimentResult &result,
                        RngEngine &rng, bool final_op = false) = 0;

  // Apply a sequence of operations to the current state of the State class.
  // It is up to the State subclass to decide how this sequence should be
  // executed (ie in sequence, or some other execution strategy.)
  // If this sequence contains operations not in the supported opset
  // an exeption will be thrown.
  // The `final_ops` flag indicates no more instructions will be applied
  // to the state after this sequence, so the state can be modified at the
  // end of the instructions.
  virtual void apply_ops(OpItr first, OpItr last, ExperimentResult &result,
                         RngEngine &rng, bool final_ops = false);

  // apply ops to multiple shots
  // this function should be separately defined since apply_ops is called in
  // quantum_error
  void apply_ops_multi_shots(OpItr first, OpItr last,
                             const Noise::NoiseModel &noise,
                             ExperimentResult &result, uint_t rng_seed,
                             bool final_ops = false) {
    throw std::invalid_argument(
        "apply_ops_multi_shots is not supported in State " + name());
  }

  //-----------------------------------------------------------------------
  // Optional: Load config settings
  //-----------------------------------------------------------------------

  // Load any settings for the State class from a config
  virtual void set_config(const Config &config);

  //-----------------------------------------------------------------------
  // Optional: Add information to metadata
  //-----------------------------------------------------------------------

  // Every state can add information to the metadata structure
  virtual void add_metadata(ExperimentResult &result) const {}

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
  virtual std::vector<reg_t> sample_measure(const reg_t &qubits, uint_t shots,
                                            RngEngine &rng);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Sets the number of threads available to the State implementation
  // If negative there is no restriction on the backend
  virtual inline void set_parallelization(int n) { threads_ = n; }

  // Set a complex global phase value exp(1j * theta) for the state
  void set_global_phase(double theta);
  bool has_global_phase() { return has_global_phase_; }
  complex_t global_phase() { return global_phase_; }

  // Set a complex global phase value exp(1j * theta) for the state
  void add_global_phase(double theta);

  // set number of processes to be distributed
  virtual void set_distribution(uint_t nprocs) {}

  // set maximum number of qubits for matrix multiplication
  virtual void set_max_matrix_qubits(int_t bits) { max_matrix_qubits_ = bits; }

  // set max sampling shots
  void set_max_sampling_shots(int_t shots) { max_sampling_shots_ = shots; }

  // set max number of shots to execute in a batch (used in StateChunk class)
  virtual void set_max_bached_shots(uint_t shots) {}

  // Does this state support multi-chunk distribution?
  virtual bool multi_chunk_distribution_supported(void) { return false; }

  // Does this state support multi-shot parallelization?
  virtual bool multi_shot_parallelization_supported(void) { return false; }

  // set creg bit counts before initialize creg
  virtual void set_num_creg_bits(uint_t num_memory, uint_t num_register) {}

  // can apply density matrix (without statevector output required)
  virtual void enable_density_matrix(bool flg) {}

  void set_num_global_qubits(uint_t qubits) { num_global_qubits_ = qubits; }

  void enable_cuStateVec(bool flg) { cuStateVec_enable_ = flg; }

  //-----------------------------------------------------------------------
  // Common instructions
  //-----------------------------------------------------------------------

  // Apply a save expectation value instruction
  void apply_save_expval(const Operations::Op &op, ExperimentResult &result);

protected:
  // Classical register data
  std::vector<ClassicalRegister> cregs_;

  // Opset of instructions supported by the state
  Operations::OpSet opset_;

  // Maximum threads which may be used by the backend for OpenMP multithreading
  // Default value is single-threaded unless overridden
  int threads_ = 1;

  // Set a global phase exp(1j * theta) for the state
  bool has_global_phase_ = false;
  complex_t global_phase_ = 1;

  int_t max_matrix_qubits_ = 0;
  int_t max_sampling_shots_ = 0;

  std::string sim_device_name_ = "CPU";

  uint_t num_global_qubits_; // used for chunk parallelization

  bool cuStateVec_enable_ = false;

  reg_t target_gpus_;
};

void Base::set_config(const Config &config) {
  sim_device_name_ = config.device;

  if (config.target_gpus.has_value()) {
    target_gpus_ = config.target_gpus.value();
  }
}

std::vector<reg_t> Base::sample_measure(const reg_t &qubits, uint_t shots,
                                        RngEngine &rng) {
  (ignore_argument) qubits;
  (ignore_argument) shots;
  return std::vector<reg_t>();
}

void Base::apply_ops(const OpItr first, const OpItr last,
                     ExperimentResult &result, RngEngine &rng, bool final_ops) {

  std::unordered_map<std::string, OpItr> marks;
  // Simple loop over vector of input operations
  for (auto it = first; it != last; ++it) {
    switch (it->type) {
    case Operations::OpType::mark: {
      marks[it->string_params[0]] = it;
      break;
    }
    case Operations::OpType::jump: {
      if (creg().check_conditional(*it)) {
        const auto &mark_name = it->string_params[0];
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
            msg << "Invalid jump destination:\"" << mark_name << "\"."
                << std::endl;
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
};

void Base::initialize_creg(uint_t num_memory, uint_t num_register) {
  creg().initialize(num_memory, num_register);
}

void Base::initialize_creg(uint_t num_memory, uint_t num_register,
                           const std::string &memory_hex,
                           const std::string &register_hex) {
  creg().initialize(num_memory, num_register, memory_hex, register_hex);
}

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
  // - `OpType::barrier` if barrier is supported
  // - `OpType::matrix` if arbitrary unitary matrices are supported
  // - `OpType::kraus` if general Kraus noise channels are supported
  //
  // For gate ops allowed gates are specified by a set of string names,
  // for example this could include {"u1", "u2", "u3", "U", "cx", "CX"}

  State(const Operations::OpSet &opset) : Base(opset) {}

  State(const Operations::OpSet::optypeset_t &optypes, const stringset_t &gates)
      : State(Operations::OpSet(optypes, gates)){};

  virtual ~State(){};

  //-----------------------------------------------------------------------
  // Data accessors
  //-----------------------------------------------------------------------

  // Return the state qreg object
  auto &qreg() { return qreg_; }
  const auto &qreg() const { return qreg_; }

protected:
  // The quantum state data structure
  state_t qreg_;
};

void Base::set_global_phase(double theta) {
  if (Linalg::almost_equal(theta, 0.0)) {
    has_global_phase_ = false;
    global_phase_ = 1;
  } else {
    has_global_phase_ = true;
    global_phase_ = std::exp(complex_t(0.0, theta));
  }
}

void Base::add_global_phase(double theta) {
  if (Linalg::almost_equal(theta, 0.0))
    return;

  has_global_phase_ = true;
  global_phase_ *= std::exp(complex_t(0.0, theta));
}

void Base::apply_save_expval(const Operations::Op &op,
                             ExperimentResult &result) {
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
    expval_var[0] = expval;                      // mean
    expval_var[1] = sq_expval - expval * expval; // variance
    result.save_data_average(creg(), op.string_params[0], expval_var, op.type,
                             op.save_type);
  } else {
    result.save_data_average(creg(), op.string_params[0], expval, op.type,
                             op.save_type);
  }
}

//-------------------------------------------------------------------------
} // namespace QuantumState
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
