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
QasmSimulator Integration Tests for SaveMatrixProductState instruction
"""

import qiskit
import numpy as np
from numpy import array
from qiskit import QuantumCircuit, assemble
from qiskit.providers.aer import QasmSimulator
from qiskit import execute

class QasmSaveMatrixProductStateTests:
    """QasmSimulator MatrixProductState instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    def test_save_matrix_product_state(self):
        """Test save matrix_product_state for instruction"""

        SUPPORTED_METHODS = ['matrix_product_state', 'automatic']

        # Target mps structure
        target_qreg = []
        target_qreg.append((np.array([[1, 0]], dtype=complex), np.array([[0, 1]], dtype=complex)))
        target_qreg.append((np.array([[1], [0]], dtype=complex), np.array([[0], [1]], dtype=complex)))
        target_qreg.append((np.array([[1]], dtype=complex), np.array([[0]], dtype=complex)))

        target_lambda_reg = []
        target_lambda_reg.append(np.array([1 / np.math.sqrt(2)], dtype=float))
        target_lambda_reg.append(np.array([1], dtype=float))

        # Matrix product state test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.cx(0, 1)

        # Add save to circuit
        label = 'mps'
        circ.save_matrix_product_state(label=label)

        # Run
        shots = 10
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'matrix_product_state')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = result.data(0)[label]
            for val, target in zip(value[0], target_qreg):
                self.assertTrue(np.allclose(val, target))
            for val, target in zip(value[1], target_lambda_reg):
                self.assertTrue(np.allclose(val, target))
