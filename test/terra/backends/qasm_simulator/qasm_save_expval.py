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
QasmSimulator Integration Tests for Snapshot instructions
"""

from ddt import ddt, data

import qiskit.quantum_info as qi
from qiskit.circuit.library import QuantumVolume
from qiskit.compiler import transpile, assemble

from qiskit.providers.aer import QasmSimulator


@ddt
class QasmSaveExpvalTests:
    """QasmSimulator SaveExpval instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    @data('II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
          'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ')
    def test_save_expval_stabilizer_pauli(self, pauli):
        """Test Pauli expval for stabilizer circuit"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'stabilizer'
        ]
        SEED = 5832

        # Stabilizer test circuit
        state = qi.random_clifford(2, seed=SEED).to_circuit()
        oper = qi.Pauli(pauli)
        target = qi.Statevector(state).expectation_value(oper).real.round(10)

        # Snapshot circuit
        circ = transpile(state, self.SIMULATOR)
        circ.save_expval('expval', oper, [0, 1])
        qobj = assemble(circ, **self.BACKEND_OPTS)
        result = self.SIMULATOR.run(qobj).result()
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertAlmostEqual(value, target)

    @data([0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1])
    def test_save_expval_stabilizer_hermitian(self, qubits):
        """Test expval for stabilizer circuit and Hermitian operator"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'stabilizer'
        ]
        SEED = 7123

        # Stabilizer test circuit
        state = qi.random_clifford(3, seed=SEED).to_circuit()
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        target = qi.Statevector(state).expectation_value(oper, qubits).real

        # Snapshot circuit
        circ = transpile(state, self.SIMULATOR)
        circ.save_expval('expval', oper, qubits)
        qobj = assemble(circ, **self.BACKEND_OPTS)
        result = self.SIMULATOR.run(qobj).result()
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertAlmostEqual(value, target)

    @data('II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
          'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ')
    def test_save_expval_nonstabilizer_pauli(self, pauli):
        """Test Pauli expval for non-stabilizer circuit"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state'
        ]
        SEED = 7382

        # Stabilizer test circuit
        state = QuantumVolume(2, 1, seed=SEED)
        oper = qi.Pauli(pauli)
        target = qi.Statevector(state).expectation_value(oper).real

        # Snapshot circuit
        circ = transpile(state, self.SIMULATOR)
        circ.save_expval('expval', oper, [0, 1])
        qobj = assemble(circ, **self.BACKEND_OPTS)
        result = self.SIMULATOR.run(qobj).result()
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertAlmostEqual(value, target)

    @data([0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1])
    def test_save_expval_nonstabilizer_hermitian(self, qubits):
        """Test expval for non-stabilizer circuit and Hermitian operator"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state'
        ]
        SEED = 8124

        # Stabilizer test circuit
        state = QuantumVolume(3, 1, seed=SEED)
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        target = qi.Statevector(state).expectation_value(oper, qubits).real

        # Snapshot circuit
        circ = transpile(state, self.SIMULATOR)
        circ.save_expval('expval', oper, qubits)
        qobj = assemble(circ, **self.BACKEND_OPTS)
        result = self.SIMULATOR.run(qobj).result()
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertAlmostEqual(value, target)
