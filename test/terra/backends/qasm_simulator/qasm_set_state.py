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
QasmSimulator Integration Tests for set state instructions
"""

from ddt import ddt, data
import numpy as np
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, assemble
from qiskit.providers.aer import QasmSimulator


@ddt
class QasmSetStateTests:
    """QasmSimulator set state instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    @data(1, 2, 3, 4, 5)
    def test_set_statevector(self, num_qubits):
        """Test SetStatevector for instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu',
            'statevector_thrust'
        ]

        seed = 100
        save_label = 'state'

        target = qi.random_statevector(2 ** num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_statevector(target)
        circ.save_statevector(label=save_label)

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
            self.assertIn(save_label, data)
            value = qi.Statevector(result.data(0)[save_label])
            self.assertAlmostEqual(value, target)

    @data(1, 2, 3)
    def test_set_density_matrix(self, num_qubits):
        """Test SetDensityMatrix instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'density_matrix', 'density_matrix_gpu',
            'density_matrix_thrust'
        ]

        seed = 100
        save_label = 'state'

        target = qi.random_density_matrix(2 ** num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_density_matrix(target)
        circ.save_density_matrix(label=save_label)

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
            self.assertIn(save_label, data)
            value = qi.DensityMatrix(result.data(0)[save_label])
            self.assertAlmostEqual(value, target)

    @data(1, 2, 3)
    def test_set_stabilizer(self, num_qubits):
        """Test SetStabilizer instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'stabilizer'
        ]

        seed = 100
        save_label = 'state'

        target = qi.random_clifford(num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_stabilizer(target)
        circ.save_stabilizer(label=save_label)

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
            self.assertIn(save_label, data)
            value = qi.Clifford.from_dict(result.data(0)[save_label])
            self.assertEqual(value, target)

class QasmSetMPSTests:
    """QasmSimulator set mps instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}
    def test_set_matrix_product_state(self):
        """Test SetMatrixProductState instruction"""

        SUPPORTED_METHODS = [
            'matrix_product_state'
        ]

        seed = 100
        tests = []
        shots = 100
        # circuit1 - |11>
        num_qubits = 2
        circ1 = QuantumCircuit(num_qubits)
        state1 = ([ ([[0]],[[1]]),
                   ([[0]],[[1]]) ],
                 [ [1.] ]
                 )
        target1 = {'0x3':shots}
        tests.append((circ1, state1, target1))

        # circuit2 - |000>+|111>
        num_qubits = 3
        circ2 = QuantumCircuit(num_qubits)

        state2 = ([([[1, 0]], [[0, 1]]),
                   ([[1.41421356, 0], [0, 0]], [[0, 0], [0, 1.41421356]]),
                   ([[1.-0.j], [0.-0.j]], [[0.-0.j], [1.-0.j]])],
        [[0.70710678, 0.70710678], [0.70710678, 0.70710678]])
        target2 = {'0x0':shots/2, '0x7':shots/2}
        
        tests.append((circ2, state2, target2))
        for circ, state, target in tests:
            circ.set_matrix_product_state(state)
            circ.measure_all()

            # Run
            opts = self.BACKEND_OPTS.copy()
            qobj = assemble(circ, self.SIMULATOR)
            result = self.SIMULATOR.run(qobj, **opts, shots=shots).result()
            method = opts.get('method', 'automatic')
            data = result.get_counts()
            self.compare_counts(result, [circ], [target], delta=0.1*shots)
