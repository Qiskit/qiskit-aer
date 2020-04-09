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
NoiseModel class integration tests
"""

import unittest
from test.terra import common
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import assemble, transpile
from qiskit.providers.aer.backends import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error
from qiskit.providers.aer.noise.errors.standard_errors import reset_error
from qiskit.providers.aer.noise.errors.standard_errors import amplitude_damping_error
from qiskit.test import mock

# Backwards compatibility for Terra <= 0.13
if not hasattr(QuantumCircuit, 'i'):
    QuantumCircuit.i = QuantumCircuit.iden


class TestNoise(common.QiskitAerTestCase):
    """Testing noise model"""

    def test_amplitude_damping_error(self):
        """Test amplitude damping error damps to correct state"""
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)  # prepare + state
        for _ in range(30):
            # Add noisy identities
            circuit.barrier(qr)
            circuit.i(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        shots = 2000
        backend = QasmSimulator()
        # test noise model
        error = amplitude_damping_error(0.75, 0.25)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')
        # Execute
        target = {'0x0': 3 * shots / 4, '0x1': shots / 4}
        circuit = transpile(circuit, basis_gates=noise_model.basis_gates)
        qobj = assemble([circuit], backend, shots=shots)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_noise_model_basis_gates(self):
        """Test noise model basis_gates"""
        basis_gates = ['u1', 'u2', 'u3', 'cx']
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates)
        self.assertEqual(model.basis_gates, target)

        # Check adding readout errors doesn't add to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates)
        model.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]], False)
        self.assertEqual(model.basis_gates, target)
        model.add_readout_error([[0.9, 0.1], [0, 1]], [2], False)
        self.assertEqual(model.basis_gates, target)

        # Check a reset instruction error isn't added to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates)
        model.add_all_qubit_quantum_error(reset_error(0.2), ['reset'], False)
        self.assertEqual(model.basis_gates, target)

        # Check a non-standard gate isn't added to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates)
        model.add_all_qubit_quantum_error(reset_error(0.2), ['label'], False)
        self.assertEqual(model.basis_gates, target)

        # Check a standard gate is added to basis gates
        model = NoiseModel(basis_gates)
        target = sorted(basis_gates + ['h'])
        model.add_all_qubit_quantum_error(reset_error(0.2), ['h'], False)
        self.assertEqual(model.basis_gates, target)

    def test_noise_model_noise_instructions(self):
        """Test noise instructions"""
        model = NoiseModel()
        target = []
        self.assertEqual(model.noise_instructions, target)

        # Check a non-standard gate is added to noise instructions
        model = NoiseModel()
        model.add_all_qubit_quantum_error(reset_error(0.2), ['label'], False)
        target = ['label']
        self.assertEqual(model.noise_instructions, target)

        # Check a standard gate is added to noise instructions
        model = NoiseModel()
        model.add_all_qubit_quantum_error(reset_error(0.2), ['h'], False)
        target = ['h']
        self.assertEqual(model.noise_instructions, target)

        # Check a reset is added to noise instructions
        model = NoiseModel()
        model.add_all_qubit_quantum_error(reset_error(0.2), ['reset'], False)
        target = ['reset']
        self.assertEqual(model.noise_instructions, target)

        # Check a measure is added to noise instructions for readout error
        model = NoiseModel()
        model.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]], False)
        target = ['measure']
        self.assertEqual(model.noise_instructions, target)

    def test_noise_model_noise_qubits(self):
        """Test noise instructions"""
        model = NoiseModel()
        target = []
        self.assertEqual(model.noise_qubits, target)

        # Check adding a default error isn't added to noise qubits
        model = NoiseModel()
        model.add_all_qubit_quantum_error(pauli_error([['XX', 1]]), ['label'], False)
        target = []
        self.assertEqual(model.noise_qubits, target)

        # Check adding a local error adds to noise qubits
        model = NoiseModel()
        model.add_quantum_error(pauli_error([['XX', 1]]), ['label'], [1, 0], False)
        target = sorted([0, 1])
        self.assertEqual(model.noise_qubits, target)

        # Check adding a non-local error adds to noise qubits
        model = NoiseModel()
        model.add_nonlocal_quantum_error(pauli_error([['XX', 1]]), ['label'], [0], [1, 2], False)
        target = sorted([0, 1, 2])
        self.assertEqual(model.noise_qubits, target)

        # Check adding a default error isn't added to noise qubits
        model = NoiseModel()
        model.add_all_qubit_readout_error([[0.9, 0.1], [0, 1]], False)
        target = []
        self.assertEqual(model.noise_qubits, target)

        # Check adding a local error adds to noise qubits
        model = NoiseModel()
        model.add_readout_error([[0.9, 0.1], [0, 1]], [2], False)
        target = [2]
        self.assertEqual(model.noise_qubits, target)

    def test_noise_models_equal(self):
        """Test two noise models are Equal"""
        roerror = [[0.9, 0.1], [0.5, 0.5]]
        error1 = pauli_error([['X', 1]], standard_gates=False)
        error2 = pauli_error([['X', 1]], standard_gates=True)

        model1 = NoiseModel()
        model1.add_all_qubit_quantum_error(error1, ['u3'], False)
        model1.add_quantum_error(error1, ['u3'], [2], False)
        model1.add_nonlocal_quantum_error(error1, ['cx'], [0, 1], [3], False)
        model1.add_all_qubit_readout_error(roerror, False)
        model1.add_readout_error(roerror, [0], False)

        model2 = NoiseModel()
        model2.add_all_qubit_quantum_error(error2, ['u3'], False)
        model2.add_quantum_error(error2, ['u3'], [2], False)
        model2.add_nonlocal_quantum_error(error2, ['cx'], [0, 1], [3], False)
        model2.add_all_qubit_readout_error(roerror, False)
        model2.add_readout_error(roerror, [0], False)
        self.assertEqual(model1, model2)

    def test_noise_models_not_equal(self):
        """Test two noise models are not equal"""
        error = pauli_error([['X', 1]])

        model1 = NoiseModel()
        model1.add_all_qubit_quantum_error(error, ['u3'], False)

        model2 = NoiseModel(basis_gates=['u3', 'cx'])
        model2.add_all_qubit_quantum_error(error, ['u3'], False)

    def test_noise_model_from_backend_singapore(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = mock.FakeSingapore()
        noise_model = NoiseModel.from_backend(backend)
        qobj = assemble(transpile(circ, backend), backend)
        sim = QasmSimulator()
        result = sim.run(qobj, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_backend_almaden(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = mock.FakeAlmaden()
        noise_model = NoiseModel.from_backend(backend)
        qobj = assemble(transpile(circ, backend), backend)
        sim = QasmSimulator()
        result = sim.run(qobj, noise_model=noise_model).result()
        self.assertTrue(result.success)

    def test_noise_model_from_rochester(self):
        circ = QuantumCircuit(2)
        circ.x(0)
        circ.x(1)
        circ.measure_all()

        backend = mock.FakeRochester()
        noise_model = NoiseModel.from_backend(backend)
        qobj = assemble(transpile(circ, backend), backend)
        sim = QasmSimulator()
        result = sim.run(qobj, noise_model=noise_model).result()
        self.assertTrue(result.success)


if __name__ == '__main__':
    unittest.main()
