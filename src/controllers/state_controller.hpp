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

#ifndef _aer_state_hpp_
#define _aer_state_hpp_

#include <cstdint>
#include <complex>
#include <string>
#include <vector>

#include "framework/rng.hpp"
#include "misc/warnings.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AER_MPI
#include <mpi.h>
#endif

DISABLE_WARNING_PUSH
#include <nlohmann/json.hpp>
DISABLE_WARNING_POP

#include "framework/creg.hpp"
#include "framework/qobj.hpp"
#include "framework/results/experiment_result.hpp"
#include "framework/results/result.hpp"
#include "framework/rng.hpp"
#include "framework/linalg/vector.hpp"

#include "noise/noise_model.hpp"

#include "transpile/cacheblocking.hpp"
#include "transpile/fusion.hpp"

#include "simulators/density_matrix/densitymatrix_state.hpp"
#include "simulators/extended_stabilizer/extended_stabilizer_state.hpp"
#include "simulators/matrix_product_state/matrix_product_state.hpp"
#include "simulators/stabilizer/stabilizer_state.hpp"
#include "simulators/statevector/qubitvector.hpp"
#include "simulators/statevector/qubitvector.hpp"
#include "simulators/statevector/statevector_state.hpp"
#include "simulators/superoperator/superoperator_state.hpp"
#include "simulators/unitary/unitary_state.hpp"

using int_t = int_fast64_t;
using uint_t = uint_fast64_t; 
using complex_t = std::complex<double>;
using complexf_t = std::complex<float>;
using cvector_t = std::vector<complex_t>;
using cvectorf_t = std::vector<complexf_t>;
using cmatrix_t = matrix<complex_t>;
using cmatrixf_t = matrix<complexf_t>;
using reg_t = std::vector<uint_t>;
using json_t = nlohmann::json;

namespace AER {

class AerState {
public:
  //-----------------------------------------------------------------------
  // Constructors
  //-----------------------------------------------------------------------
  AerState(int seed = 0) : seed_(seed){ rng_.set_seed(seed); }

  virtual ~AerState() { };

  //-----------------------------------------------------------------------
  // Configuration
  //-----------------------------------------------------------------------
  
  // Set configuration. All of the configuration must be done before calling
  // any gate operations.
  virtual void configure(const std::string& key, const std::string& value);

  // Return true if gate operations have been performed and no configuration
  // is permitted.
  virtual bool is_initialized() const { return initialized_; };

  // Allocate qubits and return a list of qubit identifiers, which start
  // `0` with incrementation `+1`.
  virtual reg_t allocate_qubits(uint_t num_qubits);

  // Return a number of qubits.
  virtual uint_t num_of_qubits() const { return num_of_qubits_; };

  // Allocate qubits with inputted complex array
  // method must be statevector and the length of the array must be 2^{num_qubits}
  virtual reg_t initialize_statevector(uint_t num_qubits, complex_t* data);

  // Allocate qubits with qubit vector
  virtual reg_t initialize_statevector(uint_t num_of_qubits, QV::QubitVector<double> qv);

  // Clear qubits
  virtual void clear();

  // Release internal statevector
  // The caller must free the returned pointer
  virtual AER::Vector<complex_t> release_statevector();

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  virtual void apply_unitary(const reg_t &qubits, const cmatrix_t &mat);

  // Apply a stacked set of 2^control_count target_count--qubit matrix to the state vector.
  virtual void apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const std::vector<cmatrix_t> &mats);

  // Apply a N-qubit diagonal matrix to the state vector.
  virtual void apply_diagonal_matrix(const reg_t &qubits, const cvector_t &mat);

  //-----------------------------------------------------------------------
  // Apply Specialized Gates
  //-----------------------------------------------------------------------

  // Apply a general N-qubit multi-controlled X-gate
  // If N=1 this implements an optimized X gate
  // If N=2 this implements an optimized CX gate
  // If N=3 this implements an optimized Toffoli gate
  virtual void apply_mcx(const reg_t &qubits);

  // Apply a general multi-controlled Y-gate
  // If N=1 this implements an optimized Y gate
  // If N=2 this implements an optimized CY gate
  // If N=3 this implements an optimized CCY gate
  virtual void apply_mcy(const reg_t &qubits);

  // Apply a general multi-controlled Z-gate
  // If N=1 this implements an optimized Z gate
  // If N=2 this implements an optimized CZ gate
  // If N=3 this implements an optimized CCZ gate
  virtual void apply_mcz(const reg_t &qubits);

  // Apply a general multi-controlled single-qubit phase gate
  // with diagonal [1, ..., 1, std::exp(complex_t(0, 1) * phase]
  // If N=1 this implements an optimized single-qubit phase gate
  // If N=2 this implements an optimized CPhase gate
  // If N=3 this implements an optimized CCPhase gate
  virtual void apply_mcphase(const reg_t &qubits, const std::complex<double> phase);

  // Apply a general multi-controlled single-qubit unitary gate
  // If N=1 this implements an optimized single-qubit U gate
  // If N=2 this implements an optimized CU gate
  // If N=3 this implements an optimized CCU gate
  virtual void apply_mcu(const reg_t &qubits, const double theta, const double phi, const double lambda);

  // Apply a general multi-controlled SWAP gate
  // If N=2 this implements an optimized SWAP  gate
  // If N=3 this implements an optimized Fredkin gate
  virtual void apply_mcswap(const reg_t &qubits);

  // Apply a general N-qubit multi-controlled RX-gate
  // If N=1 this implements an optimized RX gate
  // If N=2 this implements an optimized CRX gate
  // If N=3 this implements an optimized CCRX gate
  virtual void apply_mcrx(const reg_t &qubits, const double theta);

  // Apply a general N-qubit multi-controlled RY-gate
  // If N=1 this implements an optimized RY gate
  // If N=2 this implements an optimized CRY gate
  // If N=3 this implements an optimized CCRY gate
  virtual void apply_mcry(const reg_t &qubits, const double theta);

  // Apply a general N-qubit multi-controlled RZ-gate
  // If N=1 this implements an optimized RZ gate
  // If N=2 this implements an optimized CRZ gate
  // If N=3 this implements an optimized CCRZ gate
  virtual void apply_mcrz(const reg_t &qubits, const double theta);

  //-----------------------------------------------------------------------
  // Apply Non-Unitary Gates
  //-----------------------------------------------------------------------

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual uint_t apply_measure(const reg_t &qubits);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits);

  //-----------------------------------------------------------------------
  // Z-measurement outcome probabilities
  //-----------------------------------------------------------------------

  // Return the Z-basis measurement outcome probability P(outcome) for
  // outcome in [0, 2^num_qubits - 1]
  virtual double probability(const uint_t outcome);

  // Return the probability amplitude for outcome in [0, 2^num_qubits - 1]
  virtual complex_t amplitude(const uint_t outcome);

  // // Return the probabilities for all measurement outcomes in the current vector
  // // This is equivalent to returning a new vector with  new[i]=|orig[i]|^2.
  // // Eg. For 2-qubits this is [P(00), P(01), P(010), P(11)]
  virtual std::vector<double> probabilities();

  // // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // // for measurement of N-qubits.
  virtual std::vector<double> probabilities(const reg_t &qubits);

  // // Return M sampled outcomes for Z-basis measurement of all qubits
  // // The input is a length M list of random reals between [0, 1) used for
  // // generating samples.
  virtual std::unordered_map<uint_t, uint_t> sample_measure(uint_t shots);

private:
  void assert_initialized() const;
  void assert_not_initialized() const;
  void initialize_if_necessary();

private:
  bool initialized_ = false;
  uint_t num_of_qubits_ = 0;
  RngEngine rng_;
  const int seed_;
  std::shared_ptr<Base::StateBase> state_;
  json_t configs_;
};

void AerState::configure(const std::string& key, const std::string& value) {
  configs_[key] = value;
};

void AerState::assert_initialized() const {
  if (!initialized_) {
    std::stringstream msg;
    msg << "this AerState has not initialized." << std::endl;
    throw std::runtime_error(msg.str());
  }
};

void AerState::assert_not_initialized() const {
  if (initialized_) {
    std::stringstream msg;
    msg << "this AerState has already initialized." << std::endl;
    throw std::runtime_error(msg.str());
  }
};

void AerState::initialize_if_necessary() {
  if (initialized_) return;
  state_ = std::make_shared<Statevector::State<QV::QubitVector<double>>>();
  state_->initialize_qreg(num_of_qubits_);
  state_->initialize_creg(num_of_qubits_, num_of_qubits_);
  state_->set_config(configs_);
  rng_.set_seed(seed_);
  initialized_ = true;
};

reg_t AerState::allocate_qubits(uint_t num_qubits) {
  assert_not_initialized();
  reg_t ret;
  for (auto i = 0; i < num_qubits; ++i)
      ret.push_back(num_of_qubits_++);
  return ret;
};

reg_t AerState::initialize_statevector(uint_t num_of_qubits, complex_t* data) {
  num_of_qubits_ = num_of_qubits;
  auto qv = QV::QubitVector<double>(num_of_qubits_, data);
  auto state = std::make_shared<Statevector::State<QV::QubitVector<double>>>();
  state->initialize_qreg(num_of_qubits_, qv);
  state->initialize_creg(num_of_qubits_, num_of_qubits_);
  state->set_config(configs_);
  state_ = state;
  rng_.set_seed(seed_);
  initialized_ = true;
  reg_t ret;
  ret.reserve(num_of_qubits);
  for (auto i = 0; i < num_of_qubits; ++i)
    ret.push_back(i);
  return ret;
};

reg_t AerState::initialize_statevector(uint_t num_of_qubits, QV::QubitVector<double> qv) {
  num_of_qubits_ = num_of_qubits;
  auto state = std::make_shared<Statevector::State<QV::QubitVector<double>>>();
  state->initialize_qreg(num_of_qubits_, qv);
  state->initialize_creg(num_of_qubits_, num_of_qubits_);
  state->set_config(configs_);
  state_ = state;
  rng_.set_seed(seed_);
  initialized_ = true;
  reg_t ret;
  ret.reserve(num_of_qubits);
  for (auto i = 0; i < num_of_qubits; ++i)
    ret.push_back(i);
  return ret;
};

void AerState::clear() {
  if (initialized_) {
    state_.reset();
    initialized_ = false;
  }
  num_of_qubits_ = 0;
};

AER::Vector<complex_t> AerState::release_statevector() {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::save_statevec;
  op.name = "save_statevec";
  op.qubits.reserve(num_of_qubits_);
  for (auto i = 0; i < num_of_qubits_; ++i)
    op.qubits.push_back(i);
  op.string_params.push_back("s");
  op.save_type = Operations::DataSubType::list;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);

  AER::Vector<complex_t> sv = std::move(std::move(((DataMap<ListData, Vector<complex_t>>)ret.data).value()["s"].value())[0]);

  this->clear();

  return std::move(sv);
};


//-----------------------------------------------------------------------
// Apply Matrices
//-----------------------------------------------------------------------

void AerState::apply_unitary(const reg_t &qubits, const cmatrix_t &mat) {
  initialize_if_necessary();
  Operations::Op op;
  op.type = Operations::OpType::matrix;
  op.name = "unitary";
  op.qubits = qubits;
  op.mats.push_back(mat);

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_diagonal_matrix(const reg_t &qubits, const cvector_t &mat) {
  initialize_if_necessary();
  Operations::Op op;
  op.type = Operations::OpType::diagonal_matrix;
  op.name = "diagonal";
  op.qubits = qubits;
  op.params = mat;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const std::vector<cmatrix_t> &mats) {
  initialize_if_necessary();

  if (mats.empty())
    throw std::invalid_argument("no matrix input.");

  // Check matrices are N-qubit
  auto dim = mats[0].GetRows();
  auto num_targets = static_cast<uint_t>(std::log2(dim));
  if (1ULL << num_targets != dim || num_targets != target_qubits.size())
    throw std::invalid_argument("invalid multiplexer matrix dimension.");

  size_t num_mats = control_qubits.size();
  auto num_controls = static_cast<uint_t>(std::log2(num_mats));
  if (1ULL << num_controls != num_mats)
    throw std::invalid_argument("invalid number of multiplexer matrices.");

  if (num_controls == 0) // mats.size() must be 1
    return this->apply_unitary(target_qubits, mats[0]);

  // Get lists of controls and targets
  reg_t qubits(num_controls + num_targets);
  std::copy_n(control_qubits.begin(), num_controls, qubits.begin());
  std::copy_n(target_qubits.begin(), num_targets, qubits.begin());

  Operations::Op op;
  op.type = Operations::OpType::multiplexer;
  op.name = "multiplexer";
  op.qubits = qubits;
  op.regs = std::vector<reg_t>({control_qubits, target_qubits});
  op.mats = mats;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

//-----------------------------------------------------------------------
// Apply Specialized Gates
//-----------------------------------------------------------------------

void AerState::apply_mcx(const reg_t &qubits) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcx";
  op.qubits = qubits;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_mcy(const reg_t &qubits) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcy";
  op.qubits = qubits;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_mcz(const reg_t &qubits) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcz";
  op.qubits = qubits;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_mcphase(const reg_t &qubits, const std::complex<double> phase) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcp";
  op.qubits = qubits;
  op.params.push_back(phase);

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_mcu(const reg_t &qubits, const double theta, const double phi, const double lambda) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcu";
  op.qubits = qubits;
  op.params = {theta, phi, lambda};

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_mcswap(const reg_t &qubits) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcswap";
  op.qubits = qubits;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_mcrx(const reg_t &qubits, const double theta) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcrx";
  op.qubits = qubits;
  op.params = {theta};

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_mcry(const reg_t &qubits, const double theta) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcry";
  op.qubits = qubits;
  op.params = {theta};

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

void AerState::apply_mcrz(const reg_t &qubits, const double theta) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcrz";
  op.qubits = qubits;
  op.params = {theta};

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

//-----------------------------------------------------------------------
// Apply Non-Unitary Gates
//-----------------------------------------------------------------------

uint_t AerState::apply_measure(const reg_t &qubits) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::measure;
  op.name = "measure";
  op.qubits = qubits;
  op.memory = qubits;
  op.registers = qubits;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);

  uint_t bitstring = 0;
  uint_t bit = 1;
  for (const auto& qubit: qubits) {
    if (state_->creg().creg_memory()[qubit] == '1')
      bitstring |= bit;
    bit <<= 1;
  }
  return bitstring;
}

void AerState::apply_reset(const reg_t &qubits) {
  initialize_if_necessary();

  Operations::Op op;
  op.type = Operations::OpType::reset;
  op.name = "reset";
  op.qubits = qubits;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);
}

//-----------------------------------------------------------------------
// Z-measurement outcome probabilities
//-----------------------------------------------------------------------

double AerState::probability(const uint_t outcome) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::save_amps_sq;
  op.name = "save_amplitudes_sq";
  op.string_params.push_back("s");
  op.int_params.push_back(outcome);
  op.save_type = Operations::DataSubType::list;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);

  return ((DataMap<ListData, rvector_t>)ret.data).value()["s"].value()[0][0];
}

// Return the probability amplitude for outcome in [0, 2^num_qubits - 1]
complex_t AerState::amplitude(const uint_t outcome) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::save_amps;
  op.name = "save_amplitudes";
  op.string_params.push_back("s");
  op.int_params.push_back(outcome);
  op.save_type = Operations::DataSubType::list;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);

  return ((DataMap<ListData, Vector<complex_t>>)ret.data).value()["s"].value()[0][0];
};

std::vector<double> AerState::probabilities() {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::save_probs;
  op.name = "save_probs";
  op.string_params.push_back("s");
  op.save_type = Operations::DataSubType::list;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);

  return ((DataMap<ListData, rvector_t>)ret.data).value()["s"].value()[0];
}

std::vector<double> AerState::probabilities(const reg_t &qubits) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::save_probs;
  op.name = "save_probs";
  op.string_params.push_back("s");
  op.qubits = qubits;
  op.save_type = Operations::DataSubType::list;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_);

  return ((DataMap<ListData, rvector_t>)ret.data).value()["s"].value()[0];
}

std::unordered_map<uint_t, uint_t> AerState::sample_measure(uint_t shots) {
  assert_initialized();
  reg_t qubits;
  qubits.reserve(num_of_qubits_);
  for (uint_t i = 0; i < num_of_qubits_; ++i)
    qubits.push_back(i);
  std::vector<reg_t> samples = state_->sample_measure(qubits, shots, rng_);
  std::unordered_map<uint_t, uint_t> ret;
  for(const auto sample: samples) {
    uint_t sample_u = 0ULL;
    uint_t mask = 1ULL;
    for (const auto b: sample) {
      if (b) sample_u |= mask;
      mask <<= 1;
    }
    if (ret.find(sample_u) == ret.end())
      ret[sample_u] = 1ULL;
    else
      ++ret[sample_u];
  }
  return ret;
}

//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif


