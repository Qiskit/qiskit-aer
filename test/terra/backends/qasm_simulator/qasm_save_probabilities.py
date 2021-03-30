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
QasmSimulator Integration Tests for SaveExpval instruction
"""

from ddt import ddt, data
import numpy as np
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.compiler import transpile, assemble
from qiskit.result import Counts
from qiskit.providers.aer import QasmSimulator


@ddt
class QasmSaveProbabilitiesTests:
    """QasmSimulator SaveProbabilities instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    @data([0, 1], [1, 0], [0], [1])
    def test_save_probabilities(self, qubits):
        """Test save probabilities instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'stabilizer'
        ]

        circ = QuantumCircuit(3)
        circ.x(0)
        circ.h(1)
        circ.cx(1, 2)

        # Target probabilities
        state = qi.Statevector(circ)
        target = state.probabilities(qubits)

        # Snapshot circuit
        label = 'probs'
        opts = self.BACKEND_OPTS.copy()
        circ = transpile(circ, self.SIMULATOR)
        circ.save_probabilities(qubits, label)
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)[label]
            self.assertTrue(np.allclose(value, target))

    @data([0, 1], [1, 0], [0], [1])
    def test_save_probabilities_dict(self, qubits):
        """Test save probabilities dict instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'stabilizer'
        ]

        circ = QuantumCircuit(3)
        circ.x(0)
        circ.h(1)
        circ.cx(1, 2)

        # Target probabilities
        state = qi.Statevector(circ)
        target = state.probabilities_dict(qubits)

        # Snapshot circuit
        label = 'probs'
        opts = self.BACKEND_OPTS.copy()
        circ = transpile(circ, self.SIMULATOR)
        circ.save_probabilities_dict(qubits, label)
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = Counts(result.data(0)[label], memory_slots=len(qubits))
            self.assertDictAlmostEqual(value, target)
