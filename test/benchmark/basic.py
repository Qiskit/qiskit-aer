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
Base Class of Basic Circuit Benchmarking
"""
from qiskit.circuit.library import IntegerComparator, WeightedAdder, QuadraticForm

from benchmark.simulator_benchmark import SimulatorBenchmarkSuite


class BasicSimulatorBenchmarkSuite(SimulatorBenchmarkSuite):
    def __init__(
        self,
        name="basic",
        apps=[],
        qubits=[],
        runtime_names=[],
        measures=[],
        measure_counts=[],
        noise_model_names=[],
    ):
        super().__init__(
            name,
            apps,
            qubits=qubits,
            runtime_names=runtime_names,
            measures=measures,
            measure_counts=measure_counts,
            noise_model_names=noise_model_names,
        )

    def track_statevector(self, app, measure, measure_count, noise_name, qubit):
        return self._run(
            self.RUNTIME_STATEVECTOR_CPU, app, measure, measure_count, noise_name, qubit
        )

    def track_statevector_gpu(self, app, measure, measure_count, noise_name, qubit):
        return self._run(
            self.RUNTIME_STATEVECTOR_GPU, app, measure, measure_count, noise_name, qubit
        )

    def track_matrix_product_state(self, app, measure, measure_count, noise_name, qubit):
        return self._run(self.RUNTIME_MPS_CPU, app, measure, measure_count, noise_name, qubit)
