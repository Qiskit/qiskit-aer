# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Basic Circuit Benchmarking with 5 qubits
"""
from qiskit.circuit.library import IntegerComparator, WeightedAdder, QuadraticForm

from benchmark.simulator_benchmark import SimulatorBenchmarkSuite
from benchmark.basic import BasicSimulatorBenchmarkSuite

DEFAULT_QUBITS = [5]

DEFAULT_RUNTIME = [
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_CPU,
    SimulatorBenchmarkSuite.RUNTIME_MPS_CPU,
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_GPU,
]

DEFAULT_MEASUREMENT_METHODS = [SimulatorBenchmarkSuite.MEASUREMENT_SAMPLING]

DEFAULT_MEASUREMENT_COUNTS = SimulatorBenchmarkSuite.DEFAULT_MEASUREMENT_COUNTS

DEFAULT_NOISE_MODELS = [SimulatorBenchmarkSuite.NOISE_IDEAL]


class ArithmeticCircuits(BasicSimulatorBenchmarkSuite):
    def __init__(
        self,
        apps={"integer_comparator": 10, "weighted_adder": 1, "quadratic_form": 10},
        qubits=DEFAULT_QUBITS,
        runtime_names=DEFAULT_RUNTIME,
        measures=DEFAULT_MEASUREMENT_METHODS,
        measure_counts=DEFAULT_MEASUREMENT_COUNTS,
        noise_model_names=DEFAULT_NOISE_MODELS,
    ):
        super().__init__(
            "arithmetic_circuits",
            apps,
            qubits=qubits,
            runtime_names=runtime_names,
            measures=measures,
            measure_counts=measure_counts,
            noise_model_names=noise_model_names,
        )


class BasicChangeCircuits(BasicSimulatorBenchmarkSuite):
    def __init__(
        self,
        apps={"qft": 1},
        qubits=DEFAULT_QUBITS,
        runtime_names=DEFAULT_RUNTIME,
        measures=DEFAULT_MEASUREMENT_METHODS,
        measure_counts=DEFAULT_MEASUREMENT_COUNTS,
        noise_model_names=DEFAULT_NOISE_MODELS,
    ):
        super().__init__(
            "basic_change_circuits",
            apps,
            qubits=qubits,
            runtime_names=runtime_names,
            measures=measures,
            measure_counts=measure_counts,
            noise_model_names=noise_model_names,
        )


class NLocalCircuits(BasicSimulatorBenchmarkSuite):
    def __init__(
        self,
        apps={
            "real_amplitudes": 10,
            "real_amplitudes_linear": 10,
            "efficient_su2": 10,
            "efficient_su2_linear": 10,
            #'excitation_preserving': 10,
            #'excitation_preserving_linear': 10
        },
        qubits=DEFAULT_QUBITS,
        runtime_names=DEFAULT_RUNTIME,
        measures=DEFAULT_MEASUREMENT_METHODS,
        measure_counts=DEFAULT_MEASUREMENT_COUNTS,
        noise_model_names=DEFAULT_NOISE_MODELS,
    ):
        super().__init__(
            "n_local_circuits",
            apps,
            qubits=qubits,
            runtime_names=runtime_names,
            measures=measures,
            measure_counts=measure_counts,
            noise_model_names=noise_model_names,
        )


class ParticularQuantumCircuits(BasicSimulatorBenchmarkSuite):
    def __init__(
        self,
        apps={
            "fourier_checking": 10,
            "graph_state": 10,
            "hidden_linear_function": 10,
            "iqp": 10,
            "quantum_volume": 1,
            "phase_estimation": 1,
        },
        qubits=DEFAULT_QUBITS,
        runtime_names=DEFAULT_RUNTIME,
        measures=DEFAULT_MEASUREMENT_METHODS,
        measure_counts=DEFAULT_MEASUREMENT_COUNTS,
        noise_model_names=DEFAULT_NOISE_MODELS,
    ):
        super().__init__(
            "particular_quantum_circuits",
            apps,
            qubits=qubits,
            runtime_names=runtime_names,
            measures=measures,
            measure_counts=measure_counts,
            noise_model_names=noise_model_names,
        )


if __name__ == "__main__":
    benrhmarks = [
        ArithmeticCircuits(),
        BasicChangeCircuits(),
        NLocalCircuits(),
        ParticularQuantumCircuits(),
    ]
    for benrhmark in benrhmarks:
        benrhmark.run_manual()
