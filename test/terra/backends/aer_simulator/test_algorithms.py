# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests
"""
from math import sqrt
from ddt import ddt
from qiskit import transpile, QuantumCircuit
from test.terra.reference import ref_algorithms

from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


@ddt
class TestAlgorithms(SimulatorTestCase):
    """AerSimulator algorithm tests in the default basis"""

    def _test_grovers(self, **options):
        shots = 2000
        backend = self.backend(**options)

        circuits = ref_algorithms.grovers_circuit(final_measure=True, allow_sampling=True)

        targets = ref_algorithms.grovers_counts(shots)
        circuits = transpile(circuits, backend)
        job = backend.run(circuits, shots=shots)
        result = job.result()

        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    def _test_teleport(self, **options):
        """Test teleport circuits."""
        shots = 1000
        for key, val in options.items():
            if "method" == key and "tensor_network" in val:
                shots = 100

        backend = self.backend(**options)

        circuits = ref_algorithms.teleport_circuit()
        targets = ref_algorithms.teleport_counts(shots)
        circuits = transpile(circuits, backend)
        job = backend.run(circuits, shots=shots)
        result = job.result()

        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_grovers(self, method, device):
        """Test grovers circuits execute."""

        opts = {
            "method": method,
            "device": device,
            # ops only for extended stabilizer method
            "extended_stabilizer_sampling_method": "metropolis",
            "extended_stabilizer_metropolis_mixing_time": 100,
        }
        self._test_grovers(**opts)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_teleport(self, method, device):
        """Test teleport circuits."""
        self._test_teleport(method=method, device=device)

    @supported_methods(["statevector", "density_matrix"])
    def test_grovers_cache_blocking(self, method, device):
        """Test grovers circuits execute."""
        self._test_grovers(method=method, device=device, blocking_qubits=2, max_parallel_threads=1)

    @supported_methods(["statevector", "density_matrix"])
    def test_teleport_cache_blocking(self, method, device):
        """Test teleport circuits."""
        self._test_teleport(method=method, device=device, blocking_qubits=2, max_parallel_threads=1)

    def test_extended_stabilizer_sparse_output_probs(self):
        """
        Test a circuit for which the metropolis method fails.
        See Issue #306 for details.
        """
        backend = self.backend(
            method="extended_stabilizer",
            extended_stabilizer_sampling_method="norm_estimation",
            extended_stabilizer_norm_estimation_samples=100,
            extended_stabilizer_norm_estimation_repetitions=3,
        )

        shots = 100
        nqubits = 2
        circ = QuantumCircuit(nqubits)
        circ.h(0)
        circ.t(0)
        circ.h(0)
        for i in range(1, nqubits):
            circ.cx(i - 1, i)
        circ.measure_all()

        # circ = transpile(circ, backend)

        target = {
            nqubits * "0": shots * (0.5 + sqrt(2) / 4.0),
            nqubits * "1": shots * (0.5 - sqrt(2) / 4.0),
        }
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circ], [target], hex_counts=False, delta=0.1 * shots)
