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
#include <chrono>

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

#include "simulators/density_matrix/densitymatrix.hpp"
#include "simulators/density_matrix/densitymatrix_state.hpp"
#include "simulators/extended_stabilizer/extended_stabilizer_state.hpp"
#include "simulators/matrix_product_state/matrix_product_state.hpp"
#include "simulators/stabilizer/stabilizer_state.hpp"
#include "simulators/statevector/qubitvector.hpp"
#include "simulators/statevector/statevector_state.hpp"
#include "simulators/superoperator/superoperator_state.hpp"
#include "simulators/unitary/unitarymatrix.hpp"
#include "simulators/unitary/unitary_state.hpp"

#ifdef AER_THRUST_SUPPORTED
#include "simulators/density_matrix/densitymatrix_thrust.hpp"
#include "simulators/statevector/qubitvector_thrust.hpp"
#include "simulators/unitary/unitarymatrix_thrust.hpp"
#endif

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
using myclock_t = std::chrono::high_resolution_clock;

namespace AER {

class AerState {
public:
  // Simulation methods for the Qasm Controller
  enum class Method {
    statevector,
    density_matrix,
    matrix_product_state,
    stabilizer,
    extended_stabilizer,
    unitary,
    superop
  };

  // Simulation devices
  enum class Device { CPU, GPU, ThrustCPU };

  // Simulation precision
  enum class Precision { Double, Single };

  //-----------------------------------------------------------------------
  // Constructors
  //-----------------------------------------------------------------------
  AerState() = default;

  virtual ~AerState() { };

  //-----------------------------------------------------------------------
  // Configuration
  //-----------------------------------------------------------------------
  
  // set configuration. 
  // All of the configuration must be done before calling any gate operations.
  virtual void configure(const std::string& key, const std::string& value);

  // configure a method.
  virtual bool set_method(const std::string& name);

  // configure a device.
  virtual bool set_device(const std::string& name);

  // configure a precision.
  virtual bool set_precision(const std::string& name);

  // configure custatevec enabled or not
  virtual bool set_custatevec(const bool& enabled);

  // configure seed
  virtual bool set_seed_simulator(const int& seed);

  // configure number of threads to update state
  virtual bool set_parallel_state_update(const uint_t& parallel_state_update);

  // configure max number of qubits for a gate
  virtual bool set_max_gate_qubits(const uint_t& max_gate_qubits);

  // configure cache blocking qubits
  virtual bool set_blocking_qubits(const uint_t& blocking_qubits);

  // Return true if gate operations have been performed and no configuration
  // is permitted.
  virtual bool is_initialized() const { return initialized_; };

  // Allocate qubits and return a list of qubit identifiers, which start
  // `0` with incrementation `+1`.
  virtual reg_t allocate_qubits(uint_t num_qubits);

  // Reallocate qubits and return a list of qubit identifiers, which start
  // `0` with incrementation `+1`.
  virtual reg_t reallocate_qubits(uint_t num_qubits);

  // Return a number of qubits.
  virtual uint_t num_of_qubits() const { return num_of_qubits_; };

  // Clear all the configurations
  virtual void clear();

  virtual ExperimentResult& last_result() { return last_result_; };

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Initialize state with given configuration
  void initialize();

  // Allocate qubits with inputted complex array
  // method must be statevector and the length of the array must be 2^{num_qubits}
  // given data will not be freed in this class
  virtual reg_t initialize_statevector(uint_t num_qubits, complex_t* data, bool copy);

  // Release internal statevector
  // The caller must free the returned pointer
  virtual AER::Vector<complex_t> move_to_vector();

  //-----------------------------------------------------------------------
  // Apply initialization
  //-----------------------------------------------------------------------

  // Apply an initialization op
  void apply_initialize(const reg_t &qubits, cvector_t &&mat);

  // Apply global phase
  void apply_global_phase(double phase);

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

  // Return the probabilities for all measurement outcomes in the current vector
  // This is equivalent to returning a new vector with  new[i]=|orig[i]|^2.
  // Eg. For 2-qubits this is [P(00), P(01), P(010), P(11)]
  virtual std::vector<double> probabilities();

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  virtual std::vector<double> probabilities(const reg_t &qubits);

  // Return M sampled outcomes for Z-basis measurement of specified qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  // The returned value is unordered sampled outcomes
  virtual std::vector<std::string> sample_memory(const reg_t &qubits, uint_t shots);

  // Return M sampled outcomes for Z-basis measurement of specified qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  // The returned value is a map from outcome to its number of samples.
  virtual std::unordered_map<uint_t, uint_t> sample_counts(const reg_t &qubits, uint_t shots);

  //-----------------------------------------------------------------------
  // Operation management
  //-----------------------------------------------------------------------
  // Buffer Operations::Op
  virtual void buffer_op(const Operations::Op&& op);

  // Flush buffered Operations::Op
  virtual void flush_ops();

  // Clear buffered Operations::Op
  virtual void clear_ops();

  // Transpile
  virtual void transpile_ops();

private:
  void assert_initialized() const;
  void assert_not_initialized() const;
  bool is_gpu(bool raise_error) const;
  void initialize_experiment_result();
  void finalize_experiment_result(bool success, double time_taken);

  bool initialized_ = false;
  uint_t num_of_qubits_ = 0;
  RngEngine rng_;
  int seed_;
  std::shared_ptr<QuantumState::Base> state_;
  json_t configs_;
  ExperimentResult last_result_;

  Method method_ = Method::statevector;

  const std::unordered_map<Method, std::string> method_names_ = {
    {Method::statevector, "statevector"},
    {Method::density_matrix, "density_matrix"},
    {Method::matrix_product_state, "matrix_product_state"},
    {Method::stabilizer, "stabilizer"},
    {Method::extended_stabilizer, "extended_stabilizer"},
    {Method::unitary, "unitary"},
    {Method::superop, "superop"}
  };

  Device device_ = Device::CPU;

  const std::unordered_map<Device, std::string> device_names_ = {
    {Device::CPU, "CPU"},
    {Device::GPU, "GPU"},
    {Device::ThrustCPU, "ThrustCPU"}
  };

  Precision precision_ = Precision::Double;

  bool cuStateVec_enable_ = false;

  uint_t parallel_state_update_ = 0;

  uint_t max_gate_qubits_ = 5;

  Circuit buffer_;

  Noise::NoiseModel noise_model_;

  Transpile::Fusion fusion_pass_;

  // process information (MPI)
  int myrank_ = 0;
  int num_processes_ = 1;
  int num_process_per_experiment_ = 1;

  uint_t cache_block_qubits_ = 0;

  Transpile::CacheBlocking cache_block_pass_;
};

bool AerState::is_gpu(bool raise_error) const {
#ifndef AER_THRUST_CUDA
  if (raise_error)
    throw std::runtime_error("Simulation device \"GPU\" is not supported on this system");
  else
    return false;
#else
  int nDev;
  if (cudaGetDeviceCount(&nDev) != cudaSuccess) {
      if (raise_error) {
        cudaGetLastError();
        throw std::runtime_error("No CUDA device available!");
      } else return false;
  }
#endif
  return true;
}

void AerState::configure(const std::string& _key, const std::string& _value) {

  std::string key = _key;
  std::transform(key.begin(), key.end(), key.begin(), ::tolower);  
  std::string value = _value;
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);  

  bool error = false;
  if (key == "method") {
    error = !set_method(value);
  } else if (key == "device") {
    error = !set_device(value);
  } else if (key == "precision") {
    error = !set_precision(value);
  } else if (key == "custatevec_enable") {
    error = !set_custatevec("true" == value);
  } else if (key == "seed_simulator") {
    error = !set_seed_simulator(std::stoi(value));
  } else if (key == "parallel_state_update") {
    error = !set_parallel_state_update(std::stoul(value));
  } else if (key == "fusion_max_qubit") {
    error = !set_max_gate_qubits(std::stoul(value));
  } else if (key == "blocking_qubits") {
    error = !set_blocking_qubits(std::stoul(value));
  }

  if (error) {
    std::stringstream msg;
    msg << "invalid configuration: " << key << "=" << value << std::endl;
    throw std::runtime_error(msg.str());
  }

  static std::unordered_set<std::string> str_config = { "method", "device", "precision", "extended_stabilizer_sampling_method",
                                                        "mps_sample_measure_algorithm", "mps_log_data", "mps_swap_direction"};
  static std::unordered_set<std::string> int_config = { "seed_simulator", "max_parallel_threads", "max_memory_mb", "parallel_state_update",
                                                        "blocking_qubits", "batched_shots_gpu_max_qubits", "statevector_parallel_threshold", 
                                                        "statevector_sample_measure_opt", "stabilizer_max_snapshot_probabilities",
                                                        "extended_stabilizer_metropolis_mixing_time", "extended_stabilizer_norm_estimation_samples",
                                                        "extended_stabilizer_norm_estimation_repetitions", "extended_stabilizer_parallel_threshold",
                                                        "extended_stabilizer_probabilities_snapshot_samples", "matrix_product_state_max_bond_dimension",
                                                        "fusion_max_qubit", "fusion_threshold"};
  static std::unordered_set<std::string> double_config = { "extended_stabilizer_approximation_error", "matrix_product_state_truncation_threshold",
                                                           };
  static std::unordered_set<std::string> bool_config = { "custatevec_enable", "blocking_enable", "batched_shots_gpu", "fusion_enable", "fusion_verbose"};

  if (str_config.find(key) != str_config.end() ) {
    configs_[_key] = _value;
  } else if (int_config.find(key) != int_config.end() ) {
    configs_[_key] = std::stoi(value);
  } else if (bool_config.find(key) != bool_config.end() ) {
    configs_[_key] = "true" == value;
  } else if (double_config.find(key) != double_config.end() ) {
    configs_[_key] = std::stod(value);
  } else {
    std::stringstream msg;
    msg << "not supported configuration: " << key << "=" << value << std::endl;
    throw std::runtime_error(msg.str());
  }

};

bool AerState::set_method(const std::string& method_name) {
  assert_not_initialized();
  auto it = find_if(method_names_.begin(), method_names_.end(), [method_name](const auto& vt) { return vt.second == method_name; });
  if (it == method_names_.end()) return false;
  method_ = it->first;
  return true;
};

bool AerState::set_device(const std::string& device_name) {
  assert_not_initialized();
  if (device_name == "cpu")
    device_ = Device::CPU;
  else if (device_name == "gpu" && is_gpu(true))
    device_ = Device::GPU;
  else if (device_name == "thrustcpu")
    device_ = Device::ThrustCPU;
  else
    return false;
  return true;
};

bool AerState::set_precision(const std::string& precision_name) {
  assert_not_initialized();
  if (precision_name == "single")
    precision_ = Precision::Single;
  else if (precision_name == "double")
    precision_ = Precision::Double;
  else
    return false;
  return true;
};

bool AerState::set_custatevec(const bool& enabled) {
  assert_not_initialized();
  cuStateVec_enable_ = enabled;
  return true;
};

bool AerState::set_seed_simulator(const int& seed) {
  assert_not_initialized();
  seed_ = seed;
  return true;
};

bool AerState::set_parallel_state_update(const uint_t& parallel_state_update) {
  assert_not_initialized();
  parallel_state_update_ = parallel_state_update;
  return true;
};

bool AerState::set_max_gate_qubits(const uint_t& max_gate_qubits) {
  assert_not_initialized();
  max_gate_qubits_ = max_gate_qubits;
  return true;
};

bool AerState::set_blocking_qubits(const uint_t& blocking_qubits)
{
  assert_not_initialized();
  cache_block_qubits_ = blocking_qubits;
  return true;
}

void AerState::assert_initialized() const {
  if (!initialized_) {
    std::stringstream msg;
    msg << "this AerState has not been initialized." << std::endl;
    throw std::runtime_error(msg.str());
  }
};

void AerState::assert_not_initialized() const {
  if (initialized_) {
    std::stringstream msg;
    msg << "this AerState has already been initialized." << std::endl;
    throw std::runtime_error(msg.str());
  }
};

void AerState::initialize() {
  assert_not_initialized();

#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
  num_process_per_experiment_ = num_processes_;
#endif

  if (method_ == Method::statevector) {
    if (device_ == Device::CPU)
      if (precision_ == Precision::Double)
        state_ = std::make_shared<Statevector::State<QV::QubitVector<double>>>();
      else
        state_ = std::make_shared<Statevector::State<QV::QubitVector<float>>>();
    else // if (device_ == Device::GPU)
#ifdef AER_THRUST_SUPPORTED
      if (precision_ == Precision::Double)
        state_ = std::make_shared<Statevector::State<QV::QubitVectorThrust<double>>>();
      else
        state_ = std::make_shared<Statevector::State<QV::QubitVectorThrust<float>>>();
#else
      throw std::runtime_error("specified method does not support non-CPU device: method=statevector");
#endif
  } else if (method_ == Method::density_matrix) {
    if (device_ == Device::CPU)
      if (precision_ == Precision::Double)
        state_ = std::make_shared<DensityMatrix::State<QV::DensityMatrix<double>>>();
      else
        state_ = std::make_shared<DensityMatrix::State<QV::DensityMatrix<float>>>();
    else // if (device_ == Device::GPU)
#ifdef AER_THRUST_SUPPORTED
      if (precision_ == Precision::Double)
        state_ = std::make_shared<DensityMatrix::State<QV::DensityMatrixThrust<double>>>();
      else
        state_ = std::make_shared<DensityMatrix::State<QV::DensityMatrixThrust<float>>>();
#else
      throw std::runtime_error("specified method does not support non-CPU device: method=density_matrix");
#endif
  } else if (method_ == Method::unitary) {
    if (device_ == Device::CPU)
      if (precision_ == Precision::Double)
        state_ = std::make_shared<QubitUnitary::State<QV::UnitaryMatrix<double>>>();
      else
        state_ = std::make_shared<QubitUnitary::State<QV::UnitaryMatrix<float>>>();
    else // if (device_ == Device::GPU)
#ifdef AER_THRUST_SUPPORTED
      if (precision_ == Precision::Double)
        state_ = std::make_shared<QubitUnitary::State<QV::UnitaryMatrixThrust<double>>>();
      else
        state_ = std::make_shared<QubitUnitary::State<QV::UnitaryMatrixThrust<float>>>();
#else
      throw std::runtime_error("specified method does not support non-CPU device: method=unitary");
#endif
  } else if (method_ == Method::matrix_product_state) {
    if (device_ == Device::CPU)
      state_ = std::make_shared<MatrixProductState::State>();
    else // if (device_ == Device::GPU)
        throw std::runtime_error("specified method does not support non-CPU device: method=matrix_product_state");
  } else if (method_ == Method::stabilizer) {
    if (device_ == Device::CPU)
      state_ = std::make_shared<Stabilizer::State>();
    else // if (device_ == Device::GPU)
        throw std::runtime_error("specified method does not support non-CPU device: method=stabilizer");
  } else if (method_ == Method::extended_stabilizer) {
    if (device_ == Device::CPU)
      state_ = std::make_shared<ExtendedStabilizer::State>();
    else // if (device_ == Device::GPU)
        throw std::runtime_error("specified method does not support non-CPU device: method=extended_stabilizer");
  } else if (method_ == Method::superop) {
    if (device_ == Device::CPU)
      if (precision_ == Precision::Double)
        state_ = std::make_shared<QubitSuperoperator::State<QV::Superoperator<double>>>();
      else
        state_ = std::make_shared<QubitSuperoperator::State<QV::Superoperator<float>>>();
    else // if (device_ == Device::GPU)
        throw std::runtime_error("specified method does not support non-CPU device: method=superop");
  } else {
      throw std::runtime_error("not supported method.");
  }

#ifdef _OPENMP
  if (parallel_state_update_ == 0) {
    parallel_state_update_ = omp_get_max_threads();
  }
#endif

  uint_t block_qubits = cache_block_qubits_;
  cache_block_pass_.set_num_processes(num_process_per_experiment_);
  cache_block_pass_.set_config(configs_);
  if(!cache_block_pass_.enabled() || !state_->multi_chunk_distribution_supported())
    block_qubits = num_of_qubits_;

  state_->set_config(configs_);
  state_->set_distribution(num_process_per_experiment_);
  state_->set_max_matrix_qubits(max_gate_qubits_);
  state_->set_parallelization(parallel_state_update_);
  state_->allocate(num_of_qubits_, block_qubits);

  state_->initialize_qreg(num_of_qubits_);
  state_->initialize_creg(num_of_qubits_, num_of_qubits_);
  rng_.set_seed(seed_);

  clear_ops();

  initialized_ = true;
};

reg_t AerState::allocate_qubits(uint_t num_qubits) {
  assert_not_initialized();
  reg_t ret;
  for (auto i = 0; i < num_qubits; ++i)
      ret.push_back(num_of_qubits_++);
  return ret;
};

reg_t AerState::reallocate_qubits(uint_t num_qubits) {
  assert_not_initialized();
  num_of_qubits_ = 0;
  return allocate_qubits(num_qubits);
};

reg_t AerState::initialize_statevector(uint_t num_of_qubits, complex_t* data, bool copy) {
  assert_not_initialized();
#ifdef AER_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank_);
  num_process_per_experiment_ = num_processes_;
#endif
  uint_t block_qubits = cache_block_qubits_;
  cache_block_pass_.set_num_processes(num_process_per_experiment_);
  cache_block_pass_.set_config(configs_);

  if (device_ != Device::CPU)
    throw std::runtime_error("only CPU device supports initialize_statevector()");
  if (precision_ != Precision::Double)
    throw std::runtime_error("only Double precision supports initialize_statevector()");
  num_of_qubits_ = num_of_qubits;
  auto state = std::make_shared<Statevector::State<QV::QubitVector<double>>>();
  state->set_config(configs_);
  state->set_distribution(num_process_per_experiment_);
  state->set_max_matrix_qubits(max_gate_qubits_);
  
  if(!cache_block_pass_.enabled() || !state->multi_chunk_distribution_supported())
    block_qubits = num_of_qubits_;
  
  state->allocate(num_of_qubits_, block_qubits);
  auto qv = QV::QubitVector<double>(num_of_qubits_, data, copy);
  state->initialize_qreg(num_of_qubits_);
  state->initialize_creg(num_of_qubits_, num_of_qubits_);
  state->initialize_statevector(num_of_qubits_, std::move(qv));
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
    clear_ops();
    initialized_ = false;
  }
  num_of_qubits_ = 0;
};

AER::Vector<complex_t> AerState::move_to_vector() {
  assert_initialized();

  flush_ops();

  Operations::Op op;
  op.type = Operations::OpType::save_statevec;
  op.name = "save_statevec";
  op.qubits.reserve(num_of_qubits_);
  for (auto i = 0; i < num_of_qubits_; ++i)
    op.qubits.push_back(i);
  op.string_params.push_back("s");
  op.save_type = Operations::DataSubType::single;

  ExperimentResult ret;
  state_->apply_op(op, ret, rng_, true);

  auto sv = std::move(static_cast<DataMap<SingleData, Vector<complex_t>>>(std::move(ret).data).value()["s"].value());
  clear();

  return std::move(sv);
};

//-----------------------------------------------------------------------
// Apply Initialization
//-----------------------------------------------------------------------

void AerState::apply_initialize(const reg_t &qubits, cvector_t && vec) {
  assert_initialized();
  Operations::Op op;
  op.type = Operations::OpType::initialize;
  op.name = "initialize";
  op.qubits = qubits;
  op.params = std::move(vec);

  last_result_ = ExperimentResult();
  state_->apply_op(op, last_result_, rng_);
};

void AerState::apply_global_phase(double phase) {
  assert_initialized();
  state_->set_global_phase(phase);
  state_->apply_global_phase();
};

//-----------------------------------------------------------------------
// Apply Matrices
//-----------------------------------------------------------------------

void AerState::apply_unitary(const reg_t &qubits, const cmatrix_t &mat) {
  assert_initialized();
  Operations::Op op;
  op.type = Operations::OpType::matrix;
  op.name = "unitary";
  op.qubits = qubits;
  op.mats.push_back(mat);

  buffer_op(std::move(op));
}

void AerState::apply_diagonal_matrix(const reg_t &qubits, const cvector_t &mat) {
  assert_initialized();
  Operations::Op op;
  op.type = Operations::OpType::diagonal_matrix;
  op.name = "diagonal";
  op.qubits = qubits;
  op.params = mat;

  buffer_op(std::move(op));
}

void AerState::apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const std::vector<cmatrix_t> &mats) {
  assert_initialized();

  if (mats.empty())
    throw std::invalid_argument("no matrix input.");

  // Check matrices are N-qubit
  auto dim = mats[0].GetRows();
  auto num_targets = static_cast<uint_t>(std::log2(dim));
  if (1ULL << num_targets != dim || num_targets != target_qubits.size())
    throw std::invalid_argument("invalid multiplexer matrix dimension.");

  size_t num_mats = mats.size();
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

  buffer_op(std::move(op));
}

//-----------------------------------------------------------------------
// Apply Specialized Gates
//-----------------------------------------------------------------------

void AerState::apply_mcx(const reg_t &qubits) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcx";
  op.qubits = qubits;

  buffer_op(std::move(op));
}

void AerState::apply_mcy(const reg_t &qubits) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcy";
  op.qubits = qubits;

  buffer_op(std::move(op));
}

void AerState::apply_mcz(const reg_t &qubits) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcz";
  op.qubits = qubits;

  buffer_op(std::move(op));
}

void AerState::apply_mcphase(const reg_t &qubits, const std::complex<double> phase) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcp";
  op.qubits = qubits;
  op.params.push_back(phase);

  buffer_op(std::move(op));
}

void AerState::apply_mcu(const reg_t &qubits, const double theta, const double phi, const double lambda) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcu";
  op.qubits = qubits;
  op.params = {theta, phi, lambda, 0.0};

  buffer_op(std::move(op));
}

void AerState::apply_mcswap(const reg_t &qubits) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcswap";
  op.qubits = qubits;

  buffer_op(std::move(op));
}

void AerState::apply_mcrx(const reg_t &qubits, const double theta) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcrx";
  op.qubits = qubits;
  op.params = {theta};

  buffer_op(std::move(op));
}

void AerState::apply_mcry(const reg_t &qubits, const double theta) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcry";
  op.qubits = qubits;
  op.params = {theta};

  buffer_op(std::move(op));
}

void AerState::apply_mcrz(const reg_t &qubits, const double theta) {
  assert_initialized();

  Operations::Op op;
  op.type = Operations::OpType::gate;
  op.name = "mcrz";
  op.qubits = qubits;
  op.params = {theta};

  buffer_op(std::move(op));
}

//-----------------------------------------------------------------------
// Apply Non-Unitary Gates
//-----------------------------------------------------------------------

uint_t AerState::apply_measure(const reg_t &qubits) {
  assert_initialized();

  flush_ops();

  Operations::Op op;
  op.type = Operations::OpType::measure;
  op.name = "measure";
  op.qubits = qubits;
  op.memory = qubits;
  op.registers = qubits;

  last_result_ = ExperimentResult();
  state_->apply_op(op, last_result_, rng_);

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
  assert_initialized();

  flush_ops();

  Operations::Op op;
  op.type = Operations::OpType::reset;
  op.name = "reset";
  op.qubits = qubits;

  last_result_ = ExperimentResult();
  state_->apply_op(op, last_result_, rng_);
}

//-----------------------------------------------------------------------
// Z-measurement outcome probabilities
//-----------------------------------------------------------------------

double AerState::probability(const uint_t outcome) {
  assert_initialized();

  flush_ops();

  Operations::Op op;
  op.type = Operations::OpType::save_amps_sq;
  op.name = "save_amplitudes_sq";
  op.string_params.push_back("s");
  op.int_params.push_back(outcome);
  op.save_type = Operations::DataSubType::list;

  last_result_ = ExperimentResult();
  state_->apply_op(op, last_result_, rng_);

  return ((DataMap<ListData, rvector_t>)last_result_.data).value()["s"].value()[0][0];
}

// Return the probability amplitude for outcome in [0, 2^num_qubits - 1]
complex_t AerState::amplitude(const uint_t outcome) {
  assert_initialized();

  flush_ops();

  Operations::Op op;
  op.type = Operations::OpType::save_amps;
  op.name = "save_amplitudes";
  op.string_params.push_back("s");
  op.int_params.push_back(outcome);
  op.save_type = Operations::DataSubType::list;

  last_result_ = ExperimentResult();
  state_->apply_op(op, last_result_, rng_);

  return ((DataMap<ListData, Vector<complex_t>>)last_result_.data).value()["s"].value()[0][0];
};

std::vector<double> AerState::probabilities() {
  assert_initialized();

  flush_ops();

  Operations::Op op;
  op.type = Operations::OpType::save_probs;
  op.name = "save_probs";
  op.string_params.push_back("s");
  op.save_type = Operations::DataSubType::list;

  last_result_ = ExperimentResult();
  state_->apply_op(op, last_result_, rng_);

  return ((DataMap<ListData, rvector_t>)last_result_.data).value()["s"].value()[0];
}

std::vector<double> AerState::probabilities(const reg_t &qubits) {
  assert_initialized();

  flush_ops();

  Operations::Op op;
  op.type = Operations::OpType::save_probs;
  op.name = "save_probs";
  op.string_params.push_back("s");
  op.qubits = qubits;
  op.save_type = Operations::DataSubType::list;

  last_result_ = ExperimentResult();
  state_->apply_op(op, last_result_, rng_);

  return ((DataMap<ListData, rvector_t>)last_result_.data).value()["s"].value()[0];
}

std::vector<std::string> AerState::sample_memory(const reg_t &qubits, uint_t shots) {
  assert_initialized();

  flush_ops();

  std::vector<std::string> ret;
  ret.reserve(shots);
  std::vector<reg_t> samples = state_->sample_measure(qubits, shots, rng_);
  for (auto& sample : samples) {
    ret.push_back(Utils::int2string(Utils::reg2int(sample, 2), 2, qubits.size()));
  }
  return ret;
}

std::unordered_map<uint_t, uint_t> AerState::sample_counts(const reg_t &qubits, uint_t shots) {
  assert_initialized();

  flush_ops();

  std::vector<reg_t> samples = state_->sample_measure(qubits, shots, rng_);
  std::unordered_map<uint_t, uint_t> ret;
  for(const auto & sample: samples) {
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

//-----------------------------------------------------------------------
// Operation management
//-----------------------------------------------------------------------
void AerState::buffer_op(const Operations::Op&& op) {
  assert_initialized();
  buffer_.ops.push_back(std::move(op));
};

void AerState::initialize_experiment_result() {
  last_result_ = ExperimentResult();
  last_result_.legacy_data.set_config(configs_);
  last_result_.set_config(configs_);
  last_result_.metadata.add(method_names_.at(method_), "method");
  if (method_ == Method::statevector || method_ == Method::density_matrix || method_ == Method::unitary)
    last_result_.metadata.add(device_names_.at(device_), "device");
  else
    last_result_.metadata.add("CPU", "device");
  
  last_result_.metadata.add(num_of_qubits_, "num_qubits");
  last_result_.header = buffer_.header;
  last_result_.shots = 1;
  last_result_.seed = buffer_.seed;
  last_result_.metadata.add(parallel_state_update_, "parallel_state_update");
};

void AerState::finalize_experiment_result(bool success, double time_taken) {
  last_result_.status = success? ExperimentResult::Status::completed : ExperimentResult::Status::error;
  last_result_.time_taken = time_taken;
};

void AerState::flush_ops() {
  assert_initialized();

  if (buffer_.ops.empty()) return;

  auto timer_start = myclock_t::now();

  initialize_experiment_result();

  buffer_.set_params(false);
  transpile_ops();
  state_->apply_ops(buffer_.ops.begin(), buffer_.ops.end(), last_result_, rng_);
  
  finalize_experiment_result(true, std::chrono::duration<double>(myclock_t::now() - timer_start).count());
  clear_ops();
};

void AerState::clear_ops() {
  buffer_ = Circuit();
  buffer_.seed = seed_;
};

void AerState::transpile_ops() {
  fusion_pass_ = Transpile::Fusion();
  
  fusion_pass_.set_parallelization(parallel_state_update_);

  if (buffer_.opset().contains(Operations::OpType::superop))
    fusion_pass_.allow_superop = true;
  if (buffer_.opset().contains(Operations::OpType::kraus))
    fusion_pass_.allow_kraus = true;

  switch (method_) {
  case Method::density_matrix:
  case Method::superop: {
    // Halve the default threshold and max fused qubits for density matrix
    fusion_pass_.threshold /= 2;
    fusion_pass_.max_qubit /= 2;
    break;
  }
  case Method::matrix_product_state: {
    fusion_pass_.active = false;
  }
  case Method::statevector: {
    if (fusion_pass_.allow_kraus) {
      // Halve default max fused qubits for Kraus noise fusion
      fusion_pass_.max_qubit /= 2;
    }
    break;
  }
  case Method::unitary: {
    // max_qubit is the same with statevector
    fusion_pass_.threshold /= 2;
    break;
  }
  default: {
    fusion_pass_.active = false;
  }
  }
  // Override default fusion settings with custom config
  fusion_pass_.set_config(configs_);
  fusion_pass_.optimize_circuit(buffer_, noise_model_, state_->opset(), last_result_);

  //cache blocking
  if(cache_block_pass_.enabled() && state_->multi_chunk_distribution_supported()){
    cache_block_pass_.optimize_circuit(buffer_, noise_model_, state_->opset(), last_result_);
  }
}

//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif


