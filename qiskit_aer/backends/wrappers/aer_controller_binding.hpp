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
#include "framework/python_parser.hpp"
#include "framework/pybind_casts.hpp"
#include "framework/types.hpp"
#include "framework/results/pybind_result.hpp"

#include "controllers/aer_controller.hpp"

#include "controllers/controller_execute.hpp"

namespace py = pybind11;
using namespace AER;

template<typename T>
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

    py::object execute(std::vector<Circuit> &circuits, Noise::NoiseModel &noise_model, AER::Config& config) const {
        return AerToPy::to_python(controller_execute<T>(circuits, noise_model, config));
    }
};

template<typename MODULE>
void bind_aer_controller(MODULE m) {
    py::class_<ControllerExecutor<Controller> > aer_ctrl (m, "aer_controller_execute");
    aer_ctrl.def(py::init<>());
    aer_ctrl.def("__call__", &ControllerExecutor<Controller>::operator());
    aer_ctrl.def("__reduce__", [aer_ctrl](const ControllerExecutor<Controller> &self) {
        return py::make_tuple(aer_ctrl, py::tuple());
    });
    aer_ctrl.def("execute", [aer_ctrl](ControllerExecutor<Controller> &self,
                                       std::vector<Circuit> &circuits,
                                       py::object noise_model,
                                       AER::Config &config) {

        Noise::NoiseModel noise_model_native;
        if (noise_model)
          noise_model_native.load_from_json(noise_model);

        return self.execute(circuits, noise_model_native, config);
    });

    py::class_<Config> aer_config(m, "AerConfig");
    aer_config.def(py::init());
    aer_config.def_readwrite("shots", &Config::shots);
    aer_config.def_readwrite("method", &Config::method);
    aer_config.def_readwrite("device", &Config::device);
    aer_config.def_readwrite("precision", &Config::precision);
    // executor=None,
    // max_job_size=None,
    // max_shot_size=None,
    aer_config.def_readwrite("enable_truncation", &Config::enable_truncation);
    aer_config.def_readwrite("zero_threshold", &Config::zero_threshold);
    aer_config.def_readwrite("validation_threshold", &Config::validation_threshold);
    aer_config.def_property("max_parallel_threads",
                            [](const Config &config) { return config.max_parallel_threads.value();},
                            [](Config &config, uint_t val) { config.max_parallel_threads.value(val);});
    aer_config.def_property("max_parallel_experiments",
                            [](const Config &config) { return config.max_parallel_experiments.value();},
                            [](Config &config, uint_t val) { config.max_parallel_experiments.value(val);});
    aer_config.def_property("max_parallel_shots",
                            [](const Config &config) { return config.max_parallel_shots.value();},
                            [](Config &config, uint_t val) { config.max_parallel_shots.value(val);});
    aer_config.def_property("max_memory_mb",
                            [](const Config &config) { return config.max_memory_mb.value();},
                            [](Config &config, uint_t val) { config.max_memory_mb.value(val);});
    aer_config.def_readwrite("fusion_enable", &Config::fusion_enable);
    aer_config.def_readwrite("fusion_verbose", &Config::fusion_verbose);
    aer_config.def_property("fusion_max_qubit",
                            [](const Config &config) { return config.fusion_max_qubit.value();},
                            [](Config &config, uint_t val) { config.fusion_max_qubit.value(val);});
    aer_config.def_property("fusion_threshold",
                            [](const Config &config) { return config.fusion_threshold.value();},
                            [](Config &config, uint_t val) { config.fusion_threshold.value(val);});
    aer_config.def_property("accept_distributed_results",
                            [](const Config &config) { return config.accept_distributed_results.value();},
                            [](Config &config, bool val) { config.accept_distributed_results.value(val);});
    aer_config.def_property("memory",
                            [](const Config &config) { return config.memory.value();},
                            [](Config &config, bool val) { config.memory.value(val);});
//   // noise_model=None,
    aer_config.def_property("seed_simulator",
                            [](const Config &config) { return config.seed_simulator.value();},
                            [](Config &config, int_t val) { config.seed_simulator.value(val);});
//   // # cuStateVec (cuQuantum) option
    aer_config.def_property("cuStateVec_enable",
                            [](const Config &config) { return config.cuStateVec_enable.value();},
                            [](Config &config, bool val) { config.cuStateVec_enable.value(val);});
//   // # cache blocking for multi-GPUs/MPI options
    aer_config.def_property("blocking_qubits",
                            [](const Config &config) { return config.blocking_qubits.value();},
                            [](Config &config, uint_t val) { config.blocking_qubits.value(val);});
    aer_config.def_readwrite("blocking_enable", &Config::blocking_enable);
    aer_config.def_property("chunk_swap_buffer_qubits",
                            [](const Config &config) { return config.chunk_swap_buffer_qubits.value();},
                            [](Config &config, uint_t val) { config.chunk_swap_buffer_qubits.value(val);});
//   // # multi-shots optimization options (GPU only)
    aer_config.def_readwrite("batched_shots_gpu", &Config::batched_shots_gpu);
    aer_config.def_readwrite("batched_shots_gpu_max_qubits", &Config::batched_shots_gpu_max_qubits);
    aer_config.def_readwrite("num_threads_per_device", &Config::num_threads_per_device);
//   // # statevector options
    aer_config.def_readwrite("statevector_parallel_threshold", &Config::statevector_parallel_threshold);
    aer_config.def_readwrite("statevector_sample_measure_opt", &Config::statevector_sample_measure_opt);
//   // # stabilizer options
    aer_config.def_readwrite("stabilizer_max_snapshot_probabilities", &Config::stabilizer_max_snapshot_probabilities);
//   // # extended stabilizer options
    aer_config.def_readwrite("extended_stabilizer_sampling_method", &Config::extended_stabilizer_sampling_method);
    aer_config.def_readwrite("extended_stabilizer_metropolis_mixing_time", &Config::extended_stabilizer_metropolis_mixing_time);
    aer_config.def_readwrite("extended_stabilizer_approximation_error", &Config::extended_stabilizer_approximation_error);
    aer_config.def_readwrite("extended_stabilizer_norm_estimation_samples", &Config::extended_stabilizer_norm_estimation_samples);
    aer_config.def_readwrite("extended_stabilizer_norm_estimation_repetitions", &Config::extended_stabilizer_norm_estimation_repetitions);
    aer_config.def_readwrite("extended_stabilizer_parallel_threshold", &Config::extended_stabilizer_parallel_threshold);
    aer_config.def_readwrite("extended_stabilizer_probabilities_snapshot_samples", &Config::extended_stabilizer_probabilities_snapshot_samples);
//   // # MPS options
    aer_config.def_readwrite("matrix_product_state_truncation_threshold", &Config::matrix_product_state_truncation_threshold);
    aer_config.def_property("matrix_product_state_max_bond_dimension",
                            [](const Config &config) { return config.matrix_product_state_max_bond_dimension.value();},
                            [](Config &config, uint_t val) { config.matrix_product_state_max_bond_dimension.value(val);});
    aer_config.def_readwrite("mps_sample_measure_algorithm", &Config::mps_sample_measure_algorithm);
    aer_config.def_readwrite("mps_log_data", &Config::mps_log_data);
    aer_config.def_readwrite("mps_swap_direction", &Config::mps_swap_direction);
    aer_config.def_readwrite("chop_threshold", &Config::chop_threshold);
    aer_config.def_readwrite("mps_parallel_threshold", &Config::mps_parallel_threshold);
    aer_config.def_readwrite("mps_omp_threads", &Config::mps_omp_threads);
//   // # tensor network options
    aer_config.def_readwrite("tensor_network_num_sampling_qubits", &Config::tensor_network_num_sampling_qubits);
    aer_config.def_readwrite("use_cuTensorNet_autotuning", &Config::use_cuTensorNet_autotuning);

    aer_config.def_readwrite("param_table", &Config::param_table);
    aer_config.def_readwrite("library_dir", &Config::library_dir);
}
#endif
