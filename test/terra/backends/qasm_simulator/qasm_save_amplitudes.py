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
QasmSimulator Integration Tests for SaveAmplitudes instruction
"""

from ddt import ddt, data
import numpy as np

import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, assemble
from qiskit.circuit.library import QFT
from qiskit.providers.aer import QasmSimulator


@ddt
class QasmSaveAmplitudesTests:
    """QasmSimulator SaveAmplitudes instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    @data([0, 1, 2, 3, 4, 5, 6, 7],
          [7, 6, 5, 4, 3, 2, 1, 0],
          [5, 3, 0, 2],
          [0],
          [5, 2],
          [7, 0])
    def test_save_amplitudes(self, params):
        """Test save_amplitudes instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'matrix_product_state'
        ]

        # Stabilizer test circuit
        circ = QFT(3)

        # Target statevector
        target = qi.Statevector(circ).data[params]

        # Add save to circuit
        label = 'amps'
        circ.save_amplitudes(params, label=label)

        # Run
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = result.data(0)[label]
            self.assertTrue(np.allclose(value, target))

    @data([0, 1, 2, 3, 4, 5, 6, 7],
          [7, 6, 5, 4, 3, 2, 1, 0],
          [5, 3, 0, 2],
          [0],
          [5, 2],
          [7, 0])
    def test_save_amplitudes_squared_nonclifford(self, params):
        """Test save_amplitudes_squared instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state'
        ]

        # Stabilizer test circuit
        circ = QFT(3)

        # Target statevector
        target = np.abs(qi.Statevector(circ).data[params]) ** 2

        # Add save to circuit
        label = 'amps'
        circ.save_amplitudes_squared(params, label=label)

        # Run
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = result.data(0)[label]
            self.assertTrue(np.allclose(value, target))

    @data([0, 1, 2, 3, 4, 5, 6, 7],
          [7, 6, 5, 4, 3, 2, 1, 0],
          [5, 3, 0, 2],
          [0],
          [5, 2],
          [7, 0])
    def test_save_amplitudes_squared_clifford(self, params):
        """Test save_amplitudes_squared instruction for Clifford circuit"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'stabilizer'
        ]

        # Stabilizer test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.cx(0, 1)
        circ.x(2)
        circ.sdg(1)

        # Target statevector
        target = np.abs(qi.Statevector(circ).data[params]) ** 2

        # Add save to circuit
        label = 'amps'
        circ.save_amplitudes_squared(params, label=label)

        # Run
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = result.data(0)[label]
            self.assertTrue(np.allclose(value, target))
