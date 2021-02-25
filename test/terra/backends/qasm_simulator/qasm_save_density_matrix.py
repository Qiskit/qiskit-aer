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
QasmSimulator Integration Tests for SaveDensityMatrix instruction
"""

import numpy as np
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, assemble
from qiskit.providers.aer import QasmSimulator


class QasmSaveDensityMatrixTests:
    """QasmSimulator SaveDensityMatrix instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    def test_save_density_matrix(self):
        """Test save density matrix for instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state'
        ]

        # Stabilizer test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.sdg(0)
        circ.cx(0, 1)
        circ.cx(0, 2)

        # Target statevector
        target = qi.DensityMatrix(circ)

        # Add save to circuit
        label = 'state'
        circ.save_density_matrix(label=label)

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
            value = qi.DensityMatrix(result.data(0)[label])
            self.assertAlmostEqual(value, target)

    def test_save_density_matrix_conditional(self):
        """Test conditional save density matrix instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state'
        ]

        # Stabilizer test circuit
        label = 'state'
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.sdg(0)
        circ.cx(0, 1)
        circ.measure_all()
        circ.save_density_matrix(label=label, conditional=True)

        # Target statevector
        target = {'0x0': qi.DensityMatrix(np.diag([1, 0, 0, 0])),
                  '0x3': qi.DensityMatrix(np.diag([0, 0, 0, 1]))}

        # Run
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR, shots=10)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            for key, state in data[label].items():
                self.assertIn(key, target)
                self.assertAlmostEqual(qi.DensityMatrix(state), target[key])

    def test_save_density_matrix_pershot(self):
        """Test pershot save density matrix instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state'
        ]

        # Stabilizer test circuit
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.reset(0)
        circ.h(0)
        circ.sdg(0)

        # Target statevector
        target = qi.DensityMatrix(circ)

        # Add save
        label = 'state'
        circ.save_density_matrix(label=label, pershot=True)

        # Run
        shots = 10
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = result.data(0)[label]
            for state in value:
                self.assertAlmostEqual(qi.DensityMatrix(state), target)

    def test_save_density_matrix_pershot_conditional(self):
        """Test pershot conditional save density matrix instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state'
        ]

        # Stabilizer test circuit
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.reset(0)
        circ.h(0)
        circ.sdg(0)

        # Target statevector
        target = qi.DensityMatrix(circ)

        # Add save
        label = 'state'
        circ.save_density_matrix(label=label, pershot=True, conditional=True)
        circ.measure_all()

        # Run
        shots = 10
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = result.data(0)[label]
            self.assertIn('0x0', value)
            for state in value['0x0']:
                self.assertAlmostEqual(qi.DensityMatrix(state), target)
