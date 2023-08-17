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
AerSimulator Integration Tests for SaveMatrixProductState instruction
"""
from ddt import ddt
import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


@ddt
class TestSaveMatrixProductStateTests(SimulatorTestCase):
    """SaveMatrixProductState instruction tests."""

    @supported_methods(["automatic", "matrix_product_state"])
    def test_save_matrix_product_state(self, method, device):
        """Test save matrix_product_state instruction"""
        backend = self.backend(method=method, device=device)

        # Target mps structure
        target_qreg = []
        target_qreg.append((np.array([[1, 0]], dtype=complex), np.array([[0, 1]], dtype=complex)))
        target_qreg.append(
            (np.array([[1], [0]], dtype=complex), np.array([[0], [1]], dtype=complex))
        )
        target_qreg.append((np.array([[1]], dtype=complex), np.array([[0]], dtype=complex)))

        target_lambda_reg = []
        target_lambda_reg.append(np.array([1 / math.sqrt(2)], dtype=float))
        target_lambda_reg.append(np.array([1], dtype=float))

        # Matrix product state test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.cx(0, 1)

        # Add save to circuit
        label = "mps"
        circ.save_matrix_product_state(label=label)

        # Run
        shots = 10
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=shots).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        for val, target in zip(value[0], target_qreg):
            self.assertTrue(np.allclose(val, target))
        for val, target in zip(value[1], target_lambda_reg):
            self.assertTrue(np.allclose(val, target))
