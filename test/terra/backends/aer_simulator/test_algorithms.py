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
QasmSimulator Integration Tests
"""

from ddt import ddt
from test.terra.reference import ref_algorithms
from qiskit import transpile
from .aer_simulator_test_case import (
    AerSimulatorTestCase, supported_methods)


@ddt
class TestAlgorithms(AerSimulatorTestCase):
    """AerSimulator algorithm tests in the default basis"""

    def _test_grovers(self, **options):
        shots = 4000
        backend = self.backend(**options)

        circuits = ref_algorithms.grovers_circuit(
            final_measure=True, allow_sampling=True)

        targets = ref_algorithms.grovers_counts(shots)
        circuits = transpile(circuits, backend)
        job = backend.run(circuits, shots=shots)
        result = job.result()

        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def _test_teleport(self, **options):
        """Test teleport circuits."""
        shots = 4000
        backend = self.backend(**options)

        circuits = ref_algorithms.teleport_circuit()
        targets = ref_algorithms.teleport_counts(shots)
        circuits = transpile(circuits, backend)
        job = backend.run(circuits, shots=shots)
        result = job.result()

        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(
        ['automatic', 'statevector', 'density_matrix', 'matrix_product_state'])
    def test_grovers(self, method, device):
        """Test grovers circuits execute."""
        self._test_grovers(method=method, device=device)

    @supported_methods(
        ['automatic', 'statevector', 'density_matrix', 'matrix_product_state'])
    def test_teleport(self, method, device):
        """Test teleport circuits."""
        self._test_teleport(method=method, device=device)

    @supported_methods(['statevector', 'density_matrix'])
    def test_grovers_cache_blocking(self, method, device):
        """Test grovers circuits execute."""
        self._test_grovers(
            method=method, device=device,
            blocking_qubits=2, max_parallel_threads=1)

    @supported_methods(['statevector', 'density_matrix'])
    def test_teleport_cache_blocking(self, method, device):
        """Test teleport circuits."""
        self._test_teleport(
             method=method, device=device,
             blocking_qubits=2, max_parallel_threads=1)
