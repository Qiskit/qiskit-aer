/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_controller_binding_hpp_
#define _aer_controller_binding_hpp_

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/pybind11.h>
DISABLE_WARNING_POP
#if defined(_MSC_VER)
#undef snprintf
#endif

#include <vector>

#include "framework/matrix.hpp"
#include "framework/pybind_casts.hpp"
#include "framework/python_parser.hpp"
#include "framework/results/pybind_result.hpp"
#include "framework/types.hpp"

#include "controllers/aer_controller.hpp"

#include "controllers/controller_execute.hpp"

namespace py = pybind11;
using namespace AER;

template <typename T>
class ControllerExecutor {
public:
  ControllerExecutor() = default;
  py::object operator()(const py::handle &qobj) {
#ifdef TEST_JSON // Convert input qobj to json to test standalone data reading
    return AerToPy::to_python(controller_execute<T>(json_t(qobj)));
#else
    return AerToPy::to_python(controller_execute<T>(qobj));
#endif
  }

  py::object execute(std::vector<std::shared_ptr<Circuit>> &circuits,
                     Noise::NoiseModel &noise_model,
                     AER::Config &config) const {
    return AerToPy::to_python(
        controller_execute<T>(circuits, noise_model, config));
  }

  py::object available_devices() {
    T controller;
    return AerToPy::to_python(controller.available_devices());
  }
};

template <typename T>
py::tuple write_value(size_t index, const optional<T> &v) {
  return py::make_tuple(v.has_value(), v.value());
}

template <typename T>
T write_value(size_t index, const T &v) {
  return v;
}

template <typename T>
void read_value(const py::tuple &t, size_t index, optional<T> &v) {
  if (t[index].cast<py::tuple>()[0].cast<bool>())
    v.value(t[index].cast<py::tuple>()[1].cast<T>());
}

template <typename T>
void read_value(const py::tuple &t, size_t index, T &v) {
  v = t[index].cast<T>();
}

template <typename MODULE>
void bind_aer_controller(MODULE m) {
  py::class_<ControllerExecutor<Controller>> aer_ctrl(m,
                                                      "aer_controller_execute");
  aer_ctrl.def(py::init<>());
  aer_ctrl.def("__call__", &ControllerExecutor<Controller>::operator());
  aer_ctrl.def("__reduce__",
               [aer_ctrl](const ControllerExecutor<Controller> &self) {
                 return py::make_tuple(aer_ctrl, py::tuple());
               });
  aer_ctrl.def("execute",
               [aer_ctrl](ControllerExecutor<Controller> &self,
                          std::vector<std::shared_ptr<Circuit>> &circuits,
                          py::object noise_model, AER::Config &config) {
                 Noise::NoiseModel noise_model_native;
                 if (noise_model)
                   noise_model_native.load_from_json(noise_model);

                 return self.execute(circuits, noise_model_native, config);
               });

  aer_ctrl.def("available_devices",
               [aer_ctrl](ControllerExecutor<Controller> &self) {
                 return self.available_devices();
               });

  py::class_<Config> aer_config(m, "AerConfig");
  aer_config.def(py::init());
  aer_config.def_readwrite("shots", &Config::shots);
  aer_config.def_readwrite("method", &Config::method);
  aer_config.def_readwrite("device", &Config::device);
  aer_config.def_readwrite("precision", &Config::precision);
  aer_config.def_readwrite("enable_truncation", &Config::enable_truncation);
  aer_config.def_readwrite("zero_threshold", &Config::zero_threshold);
  aer_config.def_readwrite("validation_threshold",
                           &Config::validation_threshold);
  aer_config.def_property(
      "max_parallel_threads",
      [](const Config &config) { return config.max_parallel_threads.val; },
      [](Config &config, uint_t val) {
        config.max_parallel_threads.value(val);
      });
  aer_config.def_property(
      "max_parallel_experiments",
      [](const Config &config) { return config.max_parallel_experiments.val; },
      [](Config &config, uint_t val) {
        config.max_parallel_experiments.value(val);
      });
  aer_config.def_property(
      "max_parallel_shots",
      [](const Config &config) { return config.max_parallel_shots.val; },
      [](Config &config, uint_t val) { config.max_parallel_shots.value(val); });
  aer_config.def_property(
      "max_memory_mb",
      [](const Config &config) { return config.max_memory_mb.val; },
      [](Config &config, uint_t val) { config.max_memory_mb.value(val); });
  aer_config.def_readwrite("fusion_enable", &Config::fusion_enable);
  aer_config.def_readwrite("fusion_verbose", &Config::fusion_verbose);
  aer_config.def_property(
      "fusion_max_qubit",
      [](const Config &config) { return config.fusion_max_qubit.val; },
      [](Config &config, uint_t val) { config.fusion_max_qubit.value(val); });
  aer_config.def_property(
      "fusion_threshold",
      [](const Config &config) { return config.fusion_threshold.val; },
      [](Config &config, uint_t val) { config.fusion_threshold.value(val); });
  aer_config.def_property(
      "accept_distributed_results",
      [](const Config &config) {
        return config.accept_distributed_results.val;
      },
      [](Config &config, bool val) {
        config.accept_distributed_results.value(val);
      });
  aer_config.def_property(
      "memory", [](const Config &config) { return config.memory.val; },
      [](Config &config, bool val) { config.memory.value(val); });
  aer_config.def_property(
      "seed_simulator",
      [](const Config &config) { return config.seed_simulator.val; },
      [](Config &config, int_t val) { config.seed_simulator.value(val); });
  // # cuStateVec (cuQuantum) option
  aer_config.def_property(
      "cuStateVec_enable",
      [](const Config &config) { return config.cuStateVec_enable.val; },
      [](Config &config, bool val) { config.cuStateVec_enable.value(val); });
  // # cache blocking for multi-GPUs/MPI options
  aer_config.def_property(
      "blocking_qubits",
      [](const Config &config) { return config.blocking_qubits.val; },
      [](Config &config, uint_t val) { config.blocking_qubits.value(val); });
  aer_config.def_readwrite("blocking_enable", &Config::blocking_enable);
  aer_config.def_property(
      "chunk_swap_buffer_qubits",
      [](const Config &config) { return config.chunk_swap_buffer_qubits.val; },
      [](Config &config, uint_t val) {
        config.chunk_swap_buffer_qubits.value(val);
      });
  // # multi-shots optimization options (GPU only)
  aer_config.def_readwrite("batched_shots_gpu", &Config::batched_shots_gpu);
  aer_config.def_readwrite("batched_shots_gpu_max_qubits",
                           &Config::batched_shots_gpu_max_qubits);
  aer_config.def_property(
      "num_threads_per_device",
      [](const Config &config) { return config.num_threads_per_device.val; },
      [](Config &config, uint_t val) {
        config.num_threads_per_device.value(val);
      });
  // # multi-shot branching
  aer_config.def_readwrite("shot_branching_enable",
                           &Config::shot_branching_enable);
  aer_config.def_readwrite("shot_branching_sampling_enable",
                           &Config::shot_branching_sampling_enable);
  // # statevector options
  aer_config.def_readwrite("statevector_parallel_threshold",
                           &Config::statevector_parallel_threshold);
  aer_config.def_readwrite("statevector_sample_measure_opt",
                           &Config::statevector_sample_measure_opt);
  // # stabilizer options
  aer_config.def_readwrite("stabilizer_max_snapshot_probabilities",
                           &Config::stabilizer_max_snapshot_probabilities);
  // # extended stabilizer options
  aer_config.def_readwrite("extended_stabilizer_sampling_method",
                           &Config::extended_stabilizer_sampling_method);
  aer_config.def_readwrite("extended_stabilizer_metropolis_mixing_time",
                           &Config::extended_stabilizer_metropolis_mixing_time);
  aer_config.def_readwrite("extended_stabilizer_approximation_error",
                           &Config::extended_stabilizer_approximation_error);
  aer_config.def_readwrite(
      "extended_stabilizer_norm_estimation_samples",
      &Config::extended_stabilizer_norm_estimation_samples);
  aer_config.def_readwrite(
      "extended_stabilizer_norm_estimation_repetitions",
      &Config::extended_stabilizer_norm_estimation_repetitions);
  aer_config.def_readwrite("extended_stabilizer_parallel_threshold",
                           &Config::extended_stabilizer_parallel_threshold);
  aer_config.def_readwrite(
      "extended_stabilizer_probabilities_snapshot_samples",
      &Config::extended_stabilizer_probabilities_snapshot_samples);
  // # MPS options
  aer_config.def_readwrite("matrix_product_state_truncation_threshold",
                           &Config::matrix_product_state_truncation_threshold);
  aer_config.def_property(
      "matrix_product_state_max_bond_dimension",
      [](const Config &config) {
        return config.matrix_product_state_max_bond_dimension.val;
      },
      [](Config &config, uint_t val) {
        config.matrix_product_state_max_bond_dimension.value(val);
      });
  aer_config.def_readwrite("mps_sample_measure_algorithm",
                           &Config::mps_sample_measure_algorithm);
  aer_config.def_readwrite("mps_log_data", &Config::mps_log_data);
  aer_config.def_readwrite("mps_swap_direction", &Config::mps_swap_direction);
  aer_config.def_readwrite("chop_threshold", &Config::chop_threshold);
  aer_config.def_readwrite("mps_parallel_threshold",
                           &Config::mps_parallel_threshold);
  aer_config.def_readwrite("mps_omp_threads", &Config::mps_omp_threads);
  // # tensor network options
  aer_config.def_readwrite("tensor_network_num_sampling_qubits",
                           &Config::tensor_network_num_sampling_qubits);
  aer_config.def_readwrite("use_cuTensorNet_autotuning",
                           &Config::use_cuTensorNet_autotuning);

  // system configurations
  aer_config.def_readwrite("library_dir", &Config::library_dir);
  aer_config.def_property_readonly_static(
      "GLOBAL_PHASE_POS",
      [](const py::object &) { return Config::GLOBAL_PHASE_POS; });
  aer_config.def_readwrite("parameterizations", &Config::param_table);
  aer_config.def_property(
      "n_qubits", [](const Config &config) { return config.n_qubits.val; },
      [](Config &config, uint_t val) { config.n_qubits.value(val); });
  aer_config.def_readwrite("global_phase", &Config::global_phase);
  aer_config.def_readwrite("memory_slots", &Config::memory_slots);
  aer_config.def_property(
      "_parallel_experiments",
      [](const Config &config) { return config._parallel_experiments.val; },
      [](Config &config, uint_t val) {
        config._parallel_experiments.value(val);
      });
  aer_config.def_property(
      "_parallel_shots",
      [](const Config &config) { return config._parallel_shots.val; },
      [](Config &config, uint_t val) { config._parallel_shots.value(val); });
  aer_config.def_property(
      "_parallel_state_update",
      [](const Config &config) { return config._parallel_state_update.val; },
      [](Config &config, uint_t val) {
        config._parallel_state_update.value(val);
      });
  aer_config.def_property(
      "fusion_allow_kraus",
      [](const Config &config) { return config.fusion_allow_kraus.val; },
      [](Config &config, bool val) { config.fusion_allow_kraus.value(val); });
  aer_config.def_property(
      "fusion_allow_superop",
      [](const Config &config) { return config.fusion_allow_superop.val; },
      [](Config &config, bool val) { config.fusion_allow_superop.value(val); });
  aer_config.def_property(
      "fusion_parallelization_threshold",
      [](const Config &config) {
        return config.fusion_parallelization_threshold.val;
      },
      [](Config &config, uint_t val) {
        config.fusion_parallelization_threshold.value(val);
      });
  aer_config.def_property(
      "_fusion_enable_n_qubits",
      [](const Config &config) { return config._fusion_enable_n_qubits.val; },
      [](Config &config, bool val) {
        config._fusion_enable_n_qubits.value(val);
      });
  aer_config.def_property(
      "_fusion_enable_n_qubits_1",
      [](const Config &config) { return config._fusion_enable_n_qubits_1.val; },
      [](Config &config, uint_t val) {
        config._fusion_enable_n_qubits_1.value(val);
      });
  aer_config.def_property(
      "_fusion_enable_n_qubits_2",
      [](const Config &config) { return config._fusion_enable_n_qubits_2.val; },
      [](Config &config, uint_t val) {
        config._fusion_enable_n_qubits_2.value(val);
      });
  aer_config.def_property(
      "_fusion_enable_n_qubits_3",
      [](const Config &config) { return config._fusion_enable_n_qubits_3.val; },
      [](Config &config, uint_t val) {
        config._fusion_enable_n_qubits_3.value(val);
      });
  aer_config.def_property(
      "_fusion_enable_n_qubits_4",
      [](const Config &config) { return config._fusion_enable_n_qubits_4.val; },
      [](Config &config, uint_t val) {
        config._fusion_enable_n_qubits_4.value(val);
      });
  aer_config.def_property(
      "_fusion_enable_n_qubits_5",
      [](const Config &config) { return config._fusion_enable_n_qubits_5.val; },
      [](Config &config, uint_t val) {
        config._fusion_enable_n_qubits_5.value(val);
      });
  aer_config.def_property(
      "_fusion_enable_diagonal",
      [](const Config &config) { return config._fusion_enable_diagonal.val; },
      [](Config &config, uint_t val) {
        config._fusion_enable_diagonal.value(val);
      });
  aer_config.def_property(
      "_fusion_min_qubit",
      [](const Config &config) { return config._fusion_min_qubit.val; },
      [](Config &config, uint_t val) { config._fusion_min_qubit.value(val); });
  aer_config.def_property(
      "fusion_cost_factor",
      [](const Config &config) { return config.fusion_cost_factor.val; },
      [](Config &config, double val) { config.fusion_cost_factor.value(val); });
  aer_config.def_property(
      "_fusion_enable_cost_based",
      [](const Config &config) { return config._fusion_enable_cost_based.val; },
      [](Config &config, bool val) {
        config._fusion_enable_cost_based.value(val);
      });
  aer_config.def_property(
      "_fusion_cost_1",
      [](const Config &config) { return config._fusion_cost_1.val; },
      [](Config &config, uint_t val) { config._fusion_cost_1.value(val); });
  aer_config.def_property(
      "_fusion_cost_2",
      [](const Config &config) { return config._fusion_cost_2.val; },
      [](Config &config, uint_t val) { config._fusion_cost_2.value(val); });
  aer_config.def_property(
      "_fusion_cost_3",
      [](const Config &config) { return config._fusion_cost_3.val; },
      [](Config &config, uint_t val) { config._fusion_cost_3.value(val); });
  aer_config.def_property(
      "_fusion_cost_4",
      [](const Config &config) { return config._fusion_cost_4.val; },
      [](Config &config, uint_t val) { config._fusion_cost_4.value(val); });
  aer_config.def_property(
      "_fusion_cost_5",
      [](const Config &config) { return config._fusion_cost_5.val; },
      [](Config &config, uint_t val) { config._fusion_cost_5.value(val); });
  aer_config.def_property(
      "_fusion_cost_6",
      [](const Config &config) { return config._fusion_cost_6.val; },
      [](Config &config, uint_t val) { config._fusion_cost_6.value(val); });
  aer_config.def_property(
      "_fusion_cost_7",
      [](const Config &config) { return config._fusion_cost_7.val; },
      [](Config &config, uint_t val) { config._fusion_cost_7.value(val); });
  aer_config.def_property(
      "_fusion_cost_8",
      [](const Config &config) { return config._fusion_cost_8.val; },
      [](Config &config, uint_t val) { config._fusion_cost_8.value(val); });
  aer_config.def_property(
      "_fusion_cost_9",
      [](const Config &config) { return config._fusion_cost_9.val; },
      [](Config &config, uint_t val) { config._fusion_cost_9.value(val); });
  aer_config.def_property(
      "_fusion_cost_10",
      [](const Config &config) { return config._fusion_cost_10.val; },
      [](Config &config, uint_t val) { config._fusion_cost_10.value(val); });

  aer_config.def_property(
      "superoperator_parallel_threshold",
      [](const Config &config) {
        return config.superoperator_parallel_threshold.val;
      },
      [](Config &config, uint_t val) {
        config.superoperator_parallel_threshold.value(val);
      });
  aer_config.def_property(
      "unitary_parallel_threshold",
      [](const Config &config) {
        return config.unitary_parallel_threshold.val;
      },
      [](Config &config, uint_t val) {
        config.unitary_parallel_threshold.value(val);
      });
  aer_config.def_property(
      "memory_blocking_bits",
      [](const Config &config) { return config.memory_blocking_bits.val; },
      [](Config &config, uint_t val) {
        config.memory_blocking_bits.value(val);
      });
  aer_config.def_property(
      "extended_stabilizer_norm_estimation_default_samples",
      [](const Config &config) {
        return config.extended_stabilizer_norm_estimation_default_samples.val;
      },
      [](Config &config, uint_t val) {
        config.extended_stabilizer_norm_estimation_default_samples.value(val);
      });
  aer_config.def_property(
      "target_gpus",
      [](const Config &config) { return config.target_gpus.val; },
      [](Config &config, reg_t val) { config.target_gpus.value(val); });
  aer_config.def_property(
      "runtime_parameter_bind_enable",
      [](const Config &config) {
        return config.runtime_parameter_bind_enable.val;
      },
      [](Config &config, bool val) {
        config.runtime_parameter_bind_enable.value(val);
      });

  aer_config.def(py::pickle(
      [](const AER::Config &config) {
        return py::make_tuple(
            write_value(0, config.shots), write_value(1, config.method),
            write_value(2, config.device), write_value(3, config.precision),
            write_value(4, config.enable_truncation),
            write_value(5, config.zero_threshold),
            write_value(6, config.validation_threshold),
            write_value(7, config.max_parallel_threads),
            write_value(8, config.max_parallel_experiments),
            write_value(9, config.max_parallel_shots),
            write_value(10, config.max_memory_mb),
            write_value(11, config.fusion_enable),
            write_value(12, config.fusion_verbose),
            write_value(13, config.fusion_max_qubit),
            write_value(14, config.fusion_threshold),
            write_value(15, config.accept_distributed_results),
            write_value(16, config.memory),
            write_value(17, config.seed_simulator),
            write_value(18, config.cuStateVec_enable),
            write_value(19, config.blocking_qubits),
            write_value(20, config.blocking_enable),
            write_value(21, config.chunk_swap_buffer_qubits),
            write_value(22, config.batched_shots_gpu),
            write_value(23, config.batched_shots_gpu_max_qubits),
            write_value(24, config.num_threads_per_device),
            write_value(25, config.statevector_parallel_threshold),
            write_value(26, config.statevector_sample_measure_opt),
            write_value(27, config.stabilizer_max_snapshot_probabilities),
            write_value(28, config.extended_stabilizer_sampling_method),
            write_value(29, config.extended_stabilizer_metropolis_mixing_time),
            write_value(20, config.extended_stabilizer_approximation_error),
            write_value(31, config.extended_stabilizer_norm_estimation_samples),
            write_value(32,
                        config.extended_stabilizer_norm_estimation_repetitions),
            write_value(33, config.extended_stabilizer_parallel_threshold),
            write_value(
                34, config.extended_stabilizer_probabilities_snapshot_samples),
            write_value(35, config.matrix_product_state_truncation_threshold),
            write_value(36, config.matrix_product_state_max_bond_dimension),
            write_value(37, config.mps_sample_measure_algorithm),
            write_value(38, config.mps_log_data),
            write_value(39, config.mps_swap_direction),
            write_value(30, config.chop_threshold),
            write_value(41, config.mps_parallel_threshold),
            write_value(42, config.mps_omp_threads),
            write_value(43, config.tensor_network_num_sampling_qubits),
            write_value(44, config.use_cuTensorNet_autotuning),
            write_value(45, config.library_dir),
            write_value(46, config.param_table),
            write_value(47, config.n_qubits),
            write_value(48, config.global_phase),
            write_value(49, config.memory_slots),
            write_value(50, config._parallel_experiments),
            write_value(51, config._parallel_shots),
            write_value(52, config._parallel_state_update),
            write_value(53, config.fusion_allow_kraus),
            write_value(54, config.fusion_allow_superop),
            write_value(55, config.fusion_parallelization_threshold),
            write_value(56, config._fusion_enable_n_qubits),
            write_value(57, config._fusion_enable_n_qubits_1),
            write_value(58, config._fusion_enable_n_qubits_2),
            write_value(59, config._fusion_enable_n_qubits_3),
            write_value(60, config._fusion_enable_n_qubits_4),
            write_value(61, config._fusion_enable_n_qubits_5),
            write_value(62, config._fusion_enable_diagonal),
            write_value(63, config._fusion_min_qubit),
            write_value(64, config.fusion_cost_factor),
            write_value(65, config._fusion_enable_cost_based),
            write_value(66, config._fusion_cost_1),
            write_value(67, config._fusion_cost_2),
            write_value(68, config._fusion_cost_3),
            write_value(69, config._fusion_cost_4),
            write_value(70, config._fusion_cost_5),
            write_value(71, config._fusion_cost_6),
            write_value(72, config._fusion_cost_7),
            write_value(73, config._fusion_cost_8),
            write_value(74, config._fusion_cost_9),
            write_value(75, config._fusion_cost_10),

            write_value(76, config.superoperator_parallel_threshold),
            write_value(77, config.unitary_parallel_threshold),
            write_value(78, config.memory_blocking_bits),
            write_value(
                79, config.extended_stabilizer_norm_estimation_default_samples),
            write_value(80, config.shot_branching_enable),
            write_value(81, config.shot_branching_sampling_enable),
            write_value(82, config.target_gpus),
            write_value(83, config.runtime_parameter_bind_enable));
      },
      [](py::tuple t) {
        AER::Config config;
        if (t.size() != 84)
          throw std::runtime_error("Invalid serialization format.");

        read_value(t, 0, config.shots);
        read_value(t, 1, config.method);
        read_value(t, 2, config.device);
        read_value(t, 3, config.precision);
        read_value(t, 4, config.enable_truncation);
        read_value(t, 5, config.zero_threshold);
        read_value(t, 6, config.validation_threshold);
        read_value(t, 7, config.max_parallel_threads);
        read_value(t, 8, config.max_parallel_experiments);
        read_value(t, 9, config.max_parallel_shots);
        read_value(t, 10, config.max_memory_mb);
        read_value(t, 11, config.fusion_enable);
        read_value(t, 12, config.fusion_verbose);
        read_value(t, 13, config.fusion_max_qubit);
        read_value(t, 14, config.fusion_threshold);
        read_value(t, 15, config.accept_distributed_results);
        read_value(t, 16, config.memory);
        read_value(t, 17, config.seed_simulator);
        read_value(t, 18, config.cuStateVec_enable);
        read_value(t, 19, config.blocking_qubits);
        read_value(t, 20, config.blocking_enable);
        read_value(t, 21, config.chunk_swap_buffer_qubits);
        read_value(t, 22, config.batched_shots_gpu);
        read_value(t, 23, config.batched_shots_gpu_max_qubits);
        read_value(t, 24, config.num_threads_per_device);
        read_value(t, 25, config.statevector_parallel_threshold);
        read_value(t, 26, config.statevector_sample_measure_opt);
        read_value(t, 27, config.stabilizer_max_snapshot_probabilities);
        read_value(t, 28, config.extended_stabilizer_sampling_method);
        read_value(t, 29, config.extended_stabilizer_metropolis_mixing_time);
        read_value(t, 20, config.extended_stabilizer_approximation_error);
        read_value(t, 31, config.extended_stabilizer_norm_estimation_samples);
        read_value(t, 32,
                   config.extended_stabilizer_norm_estimation_repetitions);
        read_value(t, 33, config.extended_stabilizer_parallel_threshold);
        read_value(t, 34,
                   config.extended_stabilizer_probabilities_snapshot_samples);
        read_value(t, 35, config.matrix_product_state_truncation_threshold);
        read_value(t, 36, config.matrix_product_state_max_bond_dimension);
        read_value(t, 37, config.mps_sample_measure_algorithm);
        read_value(t, 38, config.mps_log_data);
        read_value(t, 39, config.mps_swap_direction);
        read_value(t, 30, config.chop_threshold);
        read_value(t, 41, config.mps_parallel_threshold);
        read_value(t, 42, config.mps_omp_threads);
        read_value(t, 43, config.tensor_network_num_sampling_qubits);
        read_value(t, 44, config.use_cuTensorNet_autotuning);
        read_value(t, 45, config.library_dir);
        read_value(t, 46, config.param_table);
        read_value(t, 47, config.n_qubits);
        read_value(t, 48, config.global_phase);
        read_value(t, 49, config.memory_slots);
        read_value(t, 50, config._parallel_experiments);
        read_value(t, 51, config._parallel_shots);
        read_value(t, 52, config._parallel_state_update);
        read_value(t, 53, config.fusion_allow_kraus);
        read_value(t, 54, config.fusion_allow_superop);
        read_value(t, 55, config.fusion_parallelization_threshold);
        read_value(t, 56, config._fusion_enable_n_qubits);
        read_value(t, 57, config._fusion_enable_n_qubits_1);
        read_value(t, 58, config._fusion_enable_n_qubits_2);
        read_value(t, 59, config._fusion_enable_n_qubits_3);
        read_value(t, 60, config._fusion_enable_n_qubits_4);
        read_value(t, 61, config._fusion_enable_n_qubits_5);
        read_value(t, 62, config._fusion_enable_diagonal);
        read_value(t, 63, config._fusion_min_qubit);
        read_value(t, 64, config.fusion_cost_factor);
        read_value(t, 65, config._fusion_enable_cost_based);
        read_value(t, 66, config._fusion_cost_1);
        read_value(t, 67, config._fusion_cost_2);
        read_value(t, 68, config._fusion_cost_3);
        read_value(t, 69, config._fusion_cost_4);
        read_value(t, 70, config._fusion_cost_5);
        read_value(t, 71, config._fusion_cost_6);
        read_value(t, 72, config._fusion_cost_7);
        read_value(t, 73, config._fusion_cost_8);
        read_value(t, 74, config._fusion_cost_9);
        read_value(t, 75, config._fusion_cost_10);

        read_value(t, 76, config.superoperator_parallel_threshold);
        read_value(t, 77, config.unitary_parallel_threshold);
        read_value(t, 78, config.memory_blocking_bits);
        read_value(t, 79,
                   config.extended_stabilizer_norm_estimation_default_samples);
        read_value(t, 80, config.shot_branching_enable);
        read_value(t, 81, config.shot_branching_sampling_enable);
        read_value(t, 82, config.target_gpus);
        read_value(t, 83, config.runtime_parameter_bind_enable);
        return config;
      }));
}
#endif
