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

#ifndef _aer_framework_config_hpp_
#define _aer_framework_config_hpp_

#include "json.hpp"
#include "types.hpp"
#include <optional>
#include <string>

namespace AER {

// very simple mimic of std::optional of C++-17
template <typename T>
struct optional {
  T val;
  bool exist = false;

  T value() const {
    if (!exist)
      throw std::runtime_error("value does not exist.");
    return val;
  }

  void value(const T &input) {
    exist = true;
    val = input;
  }

  void clear() { exist = false; }

  // operator bool() const {
  //   return exist;
  // }

  bool has_value() const { return exist; }
};

template <typename T>
bool get_value(optional<T> &var, const std::string &key, const json_t &js) {
  if (JSON::check_key(key, js)) {
    var.value(js[key].get<T>());
    return true;
  } else {
    var.exist = false;
    return false;
  }
}

template <typename T>
bool get_value(T &var, const std::string &key, const json_t &js) {
  return JSON::get_value(var, key, js);
}

// Configuration of Aer simulation
// This class is corresponding to `AerSimulator._default_options()`.
struct Config {
  // # Global options
  uint_t shots = 1024;
  std::string method = "automatic";
  std::string device = "CPU";
  std::string precision = "double";
  // executor=None,
  // max_job_size=None,
  // max_shot_size=None,
  bool enable_truncation = true;
  double zero_threshold = 1e-10;
  double validation_threshold = 1e-8;
  optional<uint_t> max_parallel_threads;
  optional<uint_t> max_parallel_experiments;
  optional<uint_t> max_parallel_shots;
  optional<uint_t> max_memory_mb;
  bool fusion_enable = true;
  bool fusion_verbose = false;
  optional<uint_t> fusion_max_qubit;
  optional<uint_t> fusion_threshold;
  optional<bool> accept_distributed_results;
  optional<bool> memory;
  // noise_model=None,
  optional<int_t> seed_simulator;
  // # cuStateVec (cuQuantum) option
  optional<bool> cuStateVec_enable;
  // # cache blocking for multi-GPUs/MPI options
  optional<uint_t> blocking_qubits;
  bool blocking_enable = false;
  optional<uint_t> chunk_swap_buffer_qubits;
  // # multi-shots optimization options (GPU only)
  bool batched_shots_gpu = false;
  uint_t batched_shots_gpu_max_qubits = 16;
  optional<uint_t> num_threads_per_device;
  // # multi-shot branching
  bool shot_branching_enable = false;
  bool shot_branching_sampling_enable = false;
  // # statevector options
  uint_t statevector_parallel_threshold = 14;
  uint_t statevector_sample_measure_opt = 10;
  // # stabilizer options
  uint_t stabilizer_max_snapshot_probabilities = 32;
  // # extended stabilizer options
  std::string extended_stabilizer_sampling_method = "resampled_metropolis";
  uint_t extended_stabilizer_metropolis_mixing_time = 5000;
  double extended_stabilizer_approximation_error = 0.05;
  uint_t extended_stabilizer_norm_estimation_samples = 100;
  uint_t extended_stabilizer_norm_estimation_repetitions = 3;
  uint_t extended_stabilizer_parallel_threshold = 100;
  uint_t extended_stabilizer_probabilities_snapshot_samples = 3000;
  // # MPS options
  double matrix_product_state_truncation_threshold = 1e-16;
  optional<uint_t> matrix_product_state_max_bond_dimension;
  std::string mps_sample_measure_algorithm = "mps_heuristic";
  bool mps_log_data = false;
  std::string mps_swap_direction = "mps_swap_left";
  double chop_threshold = 1e-8;
  uint_t mps_parallel_threshold = 14;
  uint_t mps_omp_threads = 1;
  // # tensor network options
  uint_t tensor_network_num_sampling_qubits = 10;
  bool use_cuTensorNet_autotuning = false;

  // system configurations
  std::string library_dir = "";
  const static int_t GLOBAL_PHASE_POS =
      -1; // special param position for global phase
  using pos_t = std::pair<int_t, int_t>;
  using exp_params_t = std::vector<std::pair<pos_t, std::vector<double>>>;
  std::vector<exp_params_t> param_table;
  optional<uint_t> n_qubits;
  double global_phase = 0.0;
  uint_t memory_slots = 0;
  optional<uint_t> _parallel_experiments;
  optional<uint_t> _parallel_shots;
  optional<uint_t> _parallel_state_update;
  optional<bool> fusion_allow_kraus;
  optional<bool> fusion_allow_superop;
  optional<uint_t> fusion_parallelization_threshold;
  optional<bool> _fusion_enable_n_qubits;
  optional<uint_t> _fusion_enable_n_qubits_1;
  optional<uint_t> _fusion_enable_n_qubits_2;
  optional<uint_t> _fusion_enable_n_qubits_3;
  optional<uint_t> _fusion_enable_n_qubits_4;
  optional<uint_t> _fusion_enable_n_qubits_5;
  optional<bool> _fusion_enable_diagonal;
  optional<uint_t> _fusion_min_qubit;
  optional<double> fusion_cost_factor;
  optional<bool> _fusion_enable_cost_based;
  optional<uint_t> _fusion_cost_1;
  optional<uint_t> _fusion_cost_2;
  optional<uint_t> _fusion_cost_3;
  optional<uint_t> _fusion_cost_4;
  optional<uint_t> _fusion_cost_5;
  optional<uint_t> _fusion_cost_6;
  optional<uint_t> _fusion_cost_7;
  optional<uint_t> _fusion_cost_8;
  optional<uint_t> _fusion_cost_9;
  optional<uint_t> _fusion_cost_10;

  optional<uint_t> superoperator_parallel_threshold;
  optional<uint_t> unitary_parallel_threshold;
  optional<uint_t> memory_blocking_bits;
  optional<uint_t> extended_stabilizer_norm_estimation_default_samples;
  optional<reg_t> target_gpus;
  optional<bool> runtime_parameter_bind_enable;

  void clear() {
    shots = 1024;
    method = "automatic";
    device = "CPU";
    precision = "double";
    // executor=None,
    // max_job_size=None,
    // max_shot_size=None,
    enable_truncation = true;
    zero_threshold = 1e-10;
    validation_threshold = 1e-8;
    max_parallel_threads.clear();
    max_parallel_experiments.clear();
    max_parallel_shots.clear();
    max_memory_mb.clear();
    fusion_enable = true;
    fusion_verbose = false;
    fusion_max_qubit.clear();
    fusion_threshold.clear();
    accept_distributed_results.clear();
    memory.clear();
    // noise_model=None,
    seed_simulator.clear();
    // # cuStateVec (cuQuantum) option
    cuStateVec_enable.clear();
    // # cache blocking for multi-GPUs/MPI options
    blocking_qubits.clear();
    blocking_enable = false;
    chunk_swap_buffer_qubits.clear();
    // # multi-shots optimization options (GPU only)
    batched_shots_gpu = false;
    batched_shots_gpu_max_qubits = 16;
    num_threads_per_device.clear();
    // # multi-shot branching
    shot_branching_enable = false;
    shot_branching_sampling_enable = false;
    // # statevector options
    statevector_parallel_threshold = 14;
    statevector_sample_measure_opt = 10;
    // # stabilizer options
    stabilizer_max_snapshot_probabilities = 32;
    // # extended stabilizer options
    extended_stabilizer_sampling_method = "resampled_metropolis";
    extended_stabilizer_metropolis_mixing_time = 5000;
    extended_stabilizer_approximation_error = 0.05;
    extended_stabilizer_norm_estimation_samples = 100;
    extended_stabilizer_norm_estimation_repetitions = 3;
    extended_stabilizer_parallel_threshold = 100;
    extended_stabilizer_probabilities_snapshot_samples = 3000;
    // # MPS options
    matrix_product_state_truncation_threshold = 1e-16;
    matrix_product_state_max_bond_dimension.clear();
    mps_sample_measure_algorithm = "mps_heuristic";
    mps_log_data = false;
    mps_swap_direction = "mps_swap_left";
    chop_threshold = 1e-8;
    mps_parallel_threshold = 14;
    mps_omp_threads = 1;
    // # tensor network options
    tensor_network_num_sampling_qubits = 10;
    use_cuTensorNet_autotuning = false;

    // system configurations
    param_table.clear();
    library_dir = "";
    n_qubits.clear();
    global_phase = 0.0;
    memory_slots = 0;
    _parallel_experiments.clear();
    _parallel_shots.clear();
    _parallel_state_update.clear();
    fusion_allow_kraus.clear();
    fusion_allow_superop.clear();
    fusion_parallelization_threshold.clear();
    _fusion_enable_n_qubits.clear();
    _fusion_enable_n_qubits_1.clear();
    _fusion_enable_n_qubits_2.clear();
    _fusion_enable_n_qubits_3.clear();
    _fusion_enable_n_qubits_4.clear();
    _fusion_enable_n_qubits_5.clear();
    _fusion_min_qubit.clear();
    fusion_cost_factor.clear();
    _fusion_enable_cost_based.clear();
    _fusion_cost_1.clear();
    _fusion_cost_2.clear();
    _fusion_cost_3.clear();
    _fusion_cost_4.clear();
    _fusion_cost_5.clear();
    _fusion_cost_6.clear();
    _fusion_cost_7.clear();
    _fusion_cost_8.clear();
    _fusion_cost_9.clear();
    _fusion_cost_10.clear();

    superoperator_parallel_threshold.clear();
    unitary_parallel_threshold.clear();
    memory_blocking_bits.clear();
    extended_stabilizer_norm_estimation_default_samples.clear();

    target_gpus.clear();
    runtime_parameter_bind_enable.clear();
  }

  void merge(const Config &other) {
    shots = other.shots;
    method = other.method;
    device = other.device;
    precision = other.precision;
    // executor=None,
    // max_job_size=None,
    // max_shot_size=None,
    enable_truncation = other.enable_truncation;
    zero_threshold = other.zero_threshold;
    validation_threshold = other.validation_threshold;
    if (other.max_parallel_threads.has_value())
      max_parallel_threads.value(other.max_parallel_threads.value());
    if (other.max_parallel_experiments.has_value())
      max_parallel_experiments.value(other.max_parallel_experiments.value());
    if (other.max_parallel_shots.has_value())
      max_parallel_shots.value(other.max_parallel_shots.value());
    if (other.max_memory_mb.has_value())
      max_memory_mb.value(other.max_memory_mb.value());
    fusion_enable = other.fusion_enable;
    fusion_verbose = other.fusion_verbose;
    if (other.fusion_max_qubit.has_value())
      fusion_max_qubit.value(other.fusion_max_qubit.value());
    if (other.fusion_threshold.has_value())
      fusion_threshold.value(other.fusion_threshold.value());
    if (other.accept_distributed_results.has_value())
      accept_distributed_results.value(
          other.accept_distributed_results.value());
    if (other.memory.has_value())
      memory.value(other.memory.value());
    // noise_model=None,
    if (other.seed_simulator.has_value())
      seed_simulator.value(other.seed_simulator.value());
    // # cuStateVec (cuQuantum) option
    if (other.cuStateVec_enable.has_value())
      cuStateVec_enable.value(other.cuStateVec_enable.value());
    // # cache blocking for multi-GPUs/MPI options
    if (other.blocking_qubits.has_value())
      blocking_qubits.value(other.blocking_qubits.value());
    blocking_enable = other.blocking_enable;
    if (other.chunk_swap_buffer_qubits.has_value())
      chunk_swap_buffer_qubits.value(other.chunk_swap_buffer_qubits.value());
    // # multi-shots optimization options (GPU only)
    batched_shots_gpu = other.batched_shots_gpu;
    batched_shots_gpu_max_qubits = other.batched_shots_gpu_max_qubits;
    if (other.num_threads_per_device.has_value())
      num_threads_per_device.value(other.num_threads_per_device.value());
    // # multi-shot branching
    shot_branching_enable = other.shot_branching_enable;
    shot_branching_sampling_enable = other.shot_branching_sampling_enable;
    // # statevector options
    statevector_parallel_threshold = other.statevector_parallel_threshold;
    statevector_sample_measure_opt = other.statevector_sample_measure_opt;
    // # stabilizer options
    stabilizer_max_snapshot_probabilities =
        other.stabilizer_max_snapshot_probabilities;
    // # extended stabilizer options
    extended_stabilizer_sampling_method =
        other.extended_stabilizer_sampling_method;
    extended_stabilizer_metropolis_mixing_time =
        other.extended_stabilizer_metropolis_mixing_time;
    extended_stabilizer_approximation_error =
        other.extended_stabilizer_approximation_error;
    extended_stabilizer_norm_estimation_samples =
        other.extended_stabilizer_norm_estimation_samples;
    extended_stabilizer_norm_estimation_repetitions =
        other.extended_stabilizer_norm_estimation_repetitions;
    extended_stabilizer_parallel_threshold =
        other.extended_stabilizer_parallel_threshold;
    extended_stabilizer_probabilities_snapshot_samples =
        other.extended_stabilizer_probabilities_snapshot_samples;
    // # MPS options
    matrix_product_state_truncation_threshold =
        other.matrix_product_state_truncation_threshold;
    if (other.matrix_product_state_max_bond_dimension.has_value())
      matrix_product_state_max_bond_dimension.value(
          other.matrix_product_state_max_bond_dimension.value());
    mps_sample_measure_algorithm = other.mps_sample_measure_algorithm;
    mps_log_data = other.mps_log_data;
    mps_swap_direction = other.mps_swap_direction;
    chop_threshold = other.chop_threshold;
    mps_parallel_threshold = other.mps_parallel_threshold;
    mps_omp_threads = other.mps_omp_threads;
    // # tensor network options
    tensor_network_num_sampling_qubits =
        other.tensor_network_num_sampling_qubits;
    use_cuTensorNet_autotuning = other.use_cuTensorNet_autotuning;
    // system configurations
    param_table = other.param_table;
    library_dir = other.library_dir;
    if (other.n_qubits.has_value())
      n_qubits.value(other.n_qubits.value());
    global_phase = other.global_phase;
    memory_slots = other.memory_slots;
    if (other._parallel_experiments.has_value())
      _parallel_experiments.value(other._parallel_experiments.value());
    if (other._parallel_shots.has_value())
      _parallel_shots.value(other._parallel_shots.value());
    if (other._parallel_state_update.has_value())
      _parallel_state_update.value(other._parallel_state_update.value());
    if (other._parallel_experiments.has_value())
      _parallel_experiments.value(other._parallel_experiments.value());
    if (other.fusion_allow_kraus.has_value())
      fusion_allow_kraus.value(other.fusion_allow_kraus.value());
    if (other.fusion_allow_superop.has_value())
      fusion_allow_superop.value(other.fusion_allow_superop.value());
    if (other.fusion_parallelization_threshold.has_value())
      fusion_parallelization_threshold.value(
          other.fusion_parallelization_threshold.value());
    if (other._fusion_enable_n_qubits.has_value())
      _fusion_enable_n_qubits.value(other._fusion_enable_n_qubits.value());
    if (other._fusion_enable_n_qubits_1.has_value())
      _fusion_enable_n_qubits_1.value(other._fusion_enable_n_qubits_1.value());
    if (other._fusion_enable_n_qubits_2.has_value())
      _fusion_enable_n_qubits_2.value(other._fusion_enable_n_qubits_2.value());
    if (other._fusion_enable_n_qubits_3.has_value())
      _fusion_enable_n_qubits_3.value(other._fusion_enable_n_qubits_3.value());
    if (other._fusion_enable_n_qubits_4.has_value())
      _fusion_enable_n_qubits_4.value(other._fusion_enable_n_qubits_4.value());
    if (other._fusion_enable_n_qubits_5.has_value())
      _fusion_enable_n_qubits_5.value(other._fusion_enable_n_qubits_5.value());
    if (other._fusion_enable_diagonal.has_value())
      _fusion_enable_diagonal.value(other._fusion_enable_diagonal.value());
    if (other._fusion_min_qubit.has_value())
      _fusion_min_qubit.value(other._fusion_min_qubit.value());
    if (other.fusion_cost_factor.has_value())
      fusion_cost_factor.value(other.fusion_cost_factor.value());

    if (other.superoperator_parallel_threshold.has_value())
      superoperator_parallel_threshold.value(
          other.superoperator_parallel_threshold.value());
    if (other.unitary_parallel_threshold.has_value())
      unitary_parallel_threshold.value(
          other.unitary_parallel_threshold.value());
    if (other.memory_blocking_bits.has_value())
      memory_blocking_bits.value(other.memory_blocking_bits.value());
    if (other.extended_stabilizer_norm_estimation_default_samples.has_value())
      extended_stabilizer_norm_estimation_default_samples.value(
          other.extended_stabilizer_norm_estimation_default_samples.value());

    if (other.target_gpus.has_value())
      target_gpus.value(other.target_gpus.value());
    if (other.runtime_parameter_bind_enable.has_value())
      runtime_parameter_bind_enable.value(
          other.runtime_parameter_bind_enable.value());
  }
};

// Json conversion function
inline void from_json(const json_t &js, Config &config) {
  get_value(config.shots, "shots", js);
  get_value(config.method, "method", js);
  get_value(config.device, "device", js);
  get_value(config.precision, "precision", js);
  // executor=None,
  // max_job_size=None,
  // max_shot_size=None,
  get_value(config.enable_truncation, "enable_truncation", js);
  get_value(config.zero_threshold, "zero_threshold", js);
  get_value(config.validation_threshold, "validation_threshold", js);
  get_value(config.max_parallel_threads, "max_parallel_threads", js);
  get_value(config.max_parallel_experiments, "max_parallel_experiments", js);
  get_value(config.max_parallel_shots, "max_parallel_shots", js);
  get_value(config.max_parallel_shots, "max_parallel_shots", js);
  get_value(config.fusion_enable, "fusion_enable", js);
  get_value(config.fusion_verbose, "fusion_verbose", js);
  get_value(config.fusion_max_qubit, "fusion_max_qubit", js);
  get_value(config.fusion_threshold, "fusion_threshold", js);
  get_value(config.accept_distributed_results, "accept_distributed_results",
            js);
  get_value(config.memory, "memory", js);
  // noise_model=None,
  get_value(config.seed_simulator, "seed_simulator", js);
  // # cuStateVec (cuQuantum) option
  get_value(config.cuStateVec_enable, "cuStateVec_enable", js);
  // # cache blocking for multi-GPUs/MPI options
  get_value(config.blocking_qubits, "blocking_qubits", js);
  get_value(config.blocking_enable, "blocking_enable", js);
  get_value(config.chunk_swap_buffer_qubits, "chunk_swap_buffer_qubits", js);
  // # multi-shots optimization options (GPU only)
  get_value(config.batched_shots_gpu, "batched_shots_gpu", js);
  get_value(config.batched_shots_gpu_max_qubits, "batched_shots_gpu_max_qubits",
            js);
  get_value(config.num_threads_per_device, "num_threads_per_device", js);
  // # multi-shot branching
  get_value(config.shot_branching_enable, "shot_branching_enable", js);
  get_value(config.shot_branching_sampling_enable,
            "shot_branching_sampling_enable", js);
  // # statevector options
  get_value(config.statevector_parallel_threshold,
            "statevector_parallel_threshold", js);
  get_value(config.statevector_sample_measure_opt,
            "statevector_sample_measure_opt", js);
  // # stabilizer options
  get_value(config.stabilizer_max_snapshot_probabilities,
            "stabilizer_max_snapshot_probabilities", js);
  // # extended stabilizer options
  get_value(config.extended_stabilizer_sampling_method,
            "extended_stabilizer_sampling_method", js);
  get_value(config.extended_stabilizer_metropolis_mixing_time,
            "extended_stabilizer_metropolis_mixing_time", js);
  get_value(config.extended_stabilizer_approximation_error,
            "extended_stabilizer_approximation_error", js);
  get_value(config.extended_stabilizer_norm_estimation_samples,
            "extended_stabilizer_norm_estimation_samples", js);
  get_value(config.extended_stabilizer_norm_estimation_repetitions,
            "extended_stabilizer_norm_estimation_repetitions", js);
  get_value(config.extended_stabilizer_parallel_threshold,
            "extended_stabilizer_parallel_threshold", js);
  get_value(config.extended_stabilizer_probabilities_snapshot_samples,
            "extended_stabilizer_probabilities_snapshot_samples", js);
  // # MPS options
  get_value(config.matrix_product_state_truncation_threshold,
            "matrix_product_state_truncation_threshold", js);
  get_value(config.matrix_product_state_max_bond_dimension,
            "matrix_product_state_max_bond_dimension", js);
  get_value(config.mps_sample_measure_algorithm, "mps_sample_measure_algorithm",
            js);
  get_value(config.mps_log_data, "mps_log_data", js);
  get_value(config.mps_swap_direction, "mps_swap_direction", js);
  get_value(config.chop_threshold, "chop_threshold", js);
  get_value(config.mps_parallel_threshold, "mps_parallel_threshold", js);
  get_value(config.mps_omp_threads, "mps_omp_threads", js);
  // # tensor network options
  get_value(config.tensor_network_num_sampling_qubits,
            "tensor_network_num_sampling_qubits", js);
  get_value(config.use_cuTensorNet_autotuning, "use_cuTensorNet_autotuning",
            js);
  // system configurations
  get_value(config.param_table, "parameterizations", js);
  get_value(config.library_dir, "library_dir", js);
  get_value(config.global_phase, "global_phase", js);
  get_value(config.memory_slots, "memory_slots", js);

  get_value(config.memory_slots, "_parallel_experiments", js);
  get_value(config.memory_slots, "_parallel_shots", js);
  get_value(config.memory_slots, "_parallel_state_update", js);

  get_value(config.fusion_allow_kraus, "fusion_allow_kraus", js);
  get_value(config.fusion_allow_superop, "fusion_allow_superop", js);
  get_value(config.fusion_parallelization_threshold,
            "fusion_parallelization_threshold", js);
  get_value(config._fusion_enable_n_qubits, "_fusion_enable_n_qubits", js);
  get_value(config._fusion_enable_n_qubits_1, "_fusion_enable_n_qubits_1", js);
  get_value(config._fusion_enable_n_qubits_2, "_fusion_enable_n_qubits_2", js);
  get_value(config._fusion_enable_n_qubits_3, "_fusion_enable_n_qubits_3", js);
  get_value(config._fusion_enable_n_qubits_4, "_fusion_enable_n_qubits_4", js);
  get_value(config._fusion_enable_n_qubits_5, "_fusion_enable_n_qubits_5", js);
  get_value(config._fusion_enable_diagonal, "_fusion_enable_diagonal", js);
  get_value(config._fusion_min_qubit, "_fusion_min_qubit", js);
  get_value(config.fusion_cost_factor, "fusion_cost_factor", js);

  get_value(config.superoperator_parallel_threshold,
            "superoperator_parallel_threshold", js);
  get_value(config.unitary_parallel_threshold, "unitary_parallel_threshold",
            js);
  get_value(config.memory_blocking_bits, "memory_blocking_bits", js);
  get_value(config.extended_stabilizer_norm_estimation_default_samples,
            "extended_stabilizer_norm_estimation_default_samples", js);
  get_value(config.target_gpus, "target_gpus", js);
  get_value(config.runtime_parameter_bind_enable,
            "runtime_parameter_bind_enable", js);
}

} // namespace AER

#endif