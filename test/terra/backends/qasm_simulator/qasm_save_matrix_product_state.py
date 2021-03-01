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
from qiskit import QuantumCircuit, assemble
from qiskit.providers.aer import QasmSimulator
from qiskit import execute

class QasmSaveMatrixProductStateTests:
    """QasmSimulator MatrixProductState instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    def test_save_matrix_product_state(self):
        """Test save matrix_product_state for instruction"""

        SUPPORTED_METHODS = ['matrix_product_state']

        # Target mps structure
        target = []
        q_vec = [([[(1-0j), -0j]],
           [[-0j, (1-0j)]]),
          ([[(1-0j)], [-0j]],
           [[-0j], [(1-0j)]]),
          ([[(1+0j)]], [[0j]])],
        lambda_vec = [[0.7071067811865475, 0.7071067811865475], [1.0]]
        target.append(q_vec)
        target.append(lambda_vec)

        # Matrix product state test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.cx(0, 1)

        # Add save to circuit
        save_key = 'mps'
        circ.save_matrix_product_state(key=save_key)

        # Run
        shots = 1
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'matrix_product_state')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(save_key, data)
            value = result.data(0)[save_key]
            print(value)
            self.assertAlmostEqual(value, target)
