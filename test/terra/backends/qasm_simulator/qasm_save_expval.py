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
from numpy import allclose
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.compiler import transpile, assemble

from qiskit.providers.aer import QasmSimulator

PAULI2 = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
          'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']


@ddt
class QasmSaveExpectationValueTests:
    """QasmSimulator SaveExpectationValue instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    @data(*PAULI2)
    def test_save_expval_stabilizer_pauli(self, pauli):
        """Test Pauli expval for stabilizer circuit"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'stabilizer', 'extended_stabilizer'
        ]
        SEED = 5832

        # Stabilizer test circuit
        state_circ = qi.random_clifford(2, seed=SEED).to_circuit()
        oper = qi.Operator(qi.Pauli(pauli))
        state = qi.Statevector(state_circ)
        target = state.expectation_value(oper).real

        # Snapshot circuit
        opts = self.BACKEND_OPTS.copy()
        method = opts.get('method', 'automatic')
        circ = transpile(state_circ, basis_gates=[
            'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'cx', 'cz', 'swap'])
        circ.save_expectation_value(oper, [0, 1], label='expval')
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertAlmostEqual(value, target)

    @data(*PAULI2)
    def test_save_expval_var_stabilizer_pauli(self, pauli):
        """Test Pauli expval_var for stabilizer circuit"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'stabilizer'
        ]
        SEED = 5832

        # Stabilizer test circuit
        state_circ = qi.random_clifford(2, seed=SEED).to_circuit()
        oper = qi.Operator(qi.Pauli(pauli))
        state = qi.Statevector(state_circ)
        expval = state.expectation_value(oper).real
        variance = state.expectation_value(oper ** 2).real - expval ** 2
        target = [expval, variance]

        # Snapshot circuit
        opts = self.BACKEND_OPTS.copy()
        method = opts.get('method', 'automatic')
        circ = transpile(state_circ, basis_gates=[
            'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'cx', 'cz', 'swap'])
        circ.save_expectation_value_variance(oper, [0, 1], label='expval')
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertTrue(allclose(value, target))

    @data([0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1])
    def test_save_expval_stabilizer_hermitian(self, qubits):
        """Test expval for stabilizer circuit and Hermitian operator"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'stabilizer', 'extended_stabilizer'
        ]
        SEED = 7123

        # Stabilizer test circuit
        state_circ = qi.random_clifford(3, seed=SEED).to_circuit()
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        state = qi.Statevector(state_circ)
        target = state.expectation_value(oper, qubits).real

        # Snapshot circuit
        opts = self.BACKEND_OPTS.copy()
        method = opts.get('method', 'automatic')
        circ = transpile(state_circ, basis_gates=[
            'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'cx', 'cz', 'swap'])
        circ.save_expectation_value(oper, qubits, label='expval')
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertAlmostEqual(value, target)

    @data([0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1])
    def test_save_expval_var_stabilizer_hermitian(self, qubits):
        """Test expval_var for stabilizer circuit and Hermitian operator"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'stabilizer', 'extended_stabilizer'
        ]
        SEED = 7123

        # Stabilizer test circuit
        state_circ = qi.random_clifford(3, seed=SEED).to_circuit()
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        state = qi.Statevector(state_circ)
        expval = state.expectation_value(oper, qubits).real
        variance = state.expectation_value(oper ** 2, qubits).real - expval ** 2
        target = [expval, variance]

        # Snapshot circuit
        opts = self.BACKEND_OPTS.copy()
        method = opts.get('method', 'automatic')
        circ = transpile(state_circ, basis_gates=[
            'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'cx', 'cz', 'swap'])
        circ.save_expectation_value_variance(oper, qubits, label='expval')
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertTrue(allclose(value, target))

    @data(*PAULI2)
    def test_save_expval_nonstabilizer_pauli(self, pauli):
        """Test Pauli expval for non-stabilizer circuit"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state'
        ]
        SEED = 7382

        # Stabilizer test circuit
        state_circ = QuantumVolume(2, 1, seed=SEED)
        oper = qi.Operator(qi.Pauli(pauli))
        state = qi.Statevector(state_circ)
        target = state.expectation_value(oper).real

        # Snapshot circuit
        opts = self.BACKEND_OPTS.copy()
        method = opts.get('method', 'automatic')
        circ = transpile(state_circ, basis_gates=['u1', 'u2', 'u3', 'cx', 'swap'])
        circ.save_expectation_value(oper, [0, 1], label='expval')
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertAlmostEqual(value, target)

    @data(*PAULI2)
    def test_save_expval_var_nonstabilizer_pauli(self, pauli):
        """Test Pauli expval_var for non-stabilizer circuit"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'extended_stabilizer'
        ]
        SEED = 7382

        # Stabilizer test circuit
        state_circ = QuantumVolume(2, 1, seed=SEED)
        oper = qi.Operator(qi.Pauli(pauli))
        state = qi.Statevector(state_circ)
        expval = state.expectation_value(oper).real
        variance = state.expectation_value(oper ** 2).real - expval ** 2
        target = [expval, variance]

        # Snapshot circuit
        opts = self.BACKEND_OPTS.copy()
        method = opts.get('method', 'automatic')
        circ = transpile(state_circ, basis_gates=['u1', 'u2', 'u3', 'cx', 'swap'])
        circ.save_expectation_value_variance(oper, [0, 1], label='expval')
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertTrue(allclose(value, target))

    @data([0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1])
    def test_save_expval_nonstabilizer_hermitian(self, qubits):
        """Test expval for non-stabilizer circuit and Hermitian operator"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'extended_stabilizer'
        ]
        SEED = 8124

        # Stabilizer test circuit
        state_circ = QuantumVolume(3, 1, seed=SEED)
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        state = qi.Statevector(state_circ)
        target = state.expectation_value(oper, qubits).real

        # Snapshot circuit
        opts = self.BACKEND_OPTS.copy()
        method = opts.get('method', 'automatic')
        circ = transpile(state_circ, basis_gates=['u1', 'u2', 'u3', 'cx', 'swap'])
        circ.save_expectation_value(oper, qubits, label='expval')
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertAlmostEqual(value, target)

    @data([0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1])
    def test_save_expval_var_nonstabilizer_hermitian(self, qubits):
        """Test expval_var for non-stabilizer circuit and Hermitian operator"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust',
            'matrix_product_state', 'extended_stabilizer'
        ]
        SEED = 8124

        # Stabilizer test circuit
        state_circ = QuantumVolume(3, 1, seed=SEED)
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        state = qi.Statevector(state_circ)
        expval = state.expectation_value(oper, qubits).real
        variance = state.expectation_value(oper ** 2, qubits).real - expval ** 2
        target = [expval, variance]

        # Snapshot circuit
        opts = self.BACKEND_OPTS.copy()
        method = opts.get('method', 'automatic')
        circ = transpile(state_circ, basis_gates=['u1', 'u2', 'u3', 'cx', 'swap'])
        circ.save_expectation_value_variance(oper, qubits, label='expval')
        qobj = assemble(circ)
        result = self.SIMULATOR.run(qobj, **opts).result()
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertTrue(allclose(value, target))

    @data(*PAULI2)
    def test_save_expval_cptp_pauli(self, pauli):
        """Test Pauli expval for stabilizer circuit"""

        SUPPORTED_METHODS = [
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust'
        ]
        SEED = 5832

        opts = self.BACKEND_OPTS.copy()
        if opts.get('method') in SUPPORTED_METHODS:

            oper = qi.Operator(qi.Pauli(pauli))

            # CPTP channel test circuit
            channel = qi.random_quantum_channel(4, seed=SEED)
            state_circ = QuantumCircuit(2)
            state_circ.append(channel, range(2))

            state = qi.DensityMatrix(state_circ)
            target = state.expectation_value(oper).real

            # Snapshot circuit
            circ = transpile(state_circ, self.SIMULATOR)
            circ.save_expectation_value(oper, [0, 1], label='expval')
            qobj = assemble(circ)
            result = self.SIMULATOR.run(qobj, **opts).result()

            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertAlmostEqual(value, target)

    @data(*PAULI2)
    def test_save_expval_var_cptp_pauli(self, pauli):
        """Test Pauli expval_var for stabilizer circuit"""

        SUPPORTED_METHODS = [
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust'
        ]
        SEED = 5832

        opts = self.BACKEND_OPTS.copy()
        if opts.get('method') in SUPPORTED_METHODS:

            oper = qi.Operator(qi.Operator(qi.Pauli(pauli)))

            # CPTP channel test circuit
            channel = qi.random_quantum_channel(4, seed=SEED)
            state_circ = QuantumCircuit(2)
            state_circ.append(channel, range(2))

            state = qi.DensityMatrix(state_circ)
            expval = state.expectation_value(oper).real
            variance = state.expectation_value(oper ** 2).real - expval ** 2
            target = [expval, variance]

            # Snapshot circuit
            circ = transpile(state_circ, self.SIMULATOR)
            circ.save_expectation_value_variance(oper, [0, 1], label='expval')
            qobj = assemble(circ)
            result = self.SIMULATOR.run(qobj, **opts).result()

            self.assertTrue(result.success)
            value = result.data(0)['expval']
            self.assertTrue(allclose(value, target))
