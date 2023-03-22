# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Integration Tests for SaveExpval instruction
"""

from ddt import ddt
import numpy as np
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.result import Counts


@ddt
class TestSaveProbabilities(SimulatorTestCase):
    """Test SaveProbabilities instruction."""

    def _test_save_probabilities(self, qubits, **options):
        """Test save probabilities instruction"""
        backend = self.backend(**options)

        circ = QuantumCircuit(3)
        circ.x(0)
        circ.h(1)
        circ.cx(1, 2)

        # Target probabilities
        state = qi.Statevector(circ)
        target = state.probabilities(qubits)

        label = "probs"
        circ.save_probabilities(qubits, label=label)
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertTrue(np.allclose(value, target))

    def _test_save_probabilities_dict(self, qubits, **options):
        """Test save probabilities dict instruction"""
        backend = self.backend(**options)

        circ = QuantumCircuit(3)
        circ.x(0)
        circ.h(1)
        circ.cx(1, 2)

        # Target probabilities
        state = qi.Statevector(circ)
        target = state.probabilities_dict(qubits)

        # Snapshot circuit
        label = "probs"
        circ.save_probabilities_dict(qubits, label=label)
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = Counts(result.data(0)[label], memory_slots=len(qubits))
        self.assertDictAlmostEqual(value, target)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "stabilizer",
            "tensor_network",
        ],
        [[0, 1], [1, 0], [0], [1]],
    )
    def test_save_probabilities(self, method, device, qubits):
        """Test save probabilities instruction"""
        self._test_save_probabilities(qubits, method=method, device=device)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "stabilizer",
            "tensor_network",
        ],
        [[0, 1], [1, 0], [0], [1]],
    )
    def test_save_probabilities_dict(self, method, device, qubits):
        """Test save probabilities dict instruction"""
        self._test_save_probabilities_dict(qubits, method=method, device=device)

    @supported_methods(["statevector", "density_matrix"], [[0, 1], [1, 0], [0], [1]])
    def test_save_probabilities_cache_blocking(self, method, device, qubits):
        """Test save probabilities instruction"""
        self._test_save_probabilities(
            qubits, method=method, device=device, blocking_qubits=2, max_parallel_threads=1
        )

    @supported_methods(["statevector", "density_matrix"], [[0, 1], [1, 0], [0], [1]])
    def test_save_probabilities_dict_cache_blocking(self, method, device, qubits):
        """Test save probabilities dict instruction"""
        self._test_save_probabilities_dict(
            qubits, method=method, device=device, blocking_qubits=2, max_parallel_threads=1
        )
