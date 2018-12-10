# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import test.terra.utils.common as common
import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import compile
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError
from qiskit_aer.backends import QasmSimulator
from qiskit_aer.noise.errors.standard_errors import pauli_error
from qiskit_aer.noise.errors.standard_errors import amplitude_damping_error


class TestNoise(common.QiskitAerTestCase):
    """Testing noise model"""

    def test_readout_error_specific_qubit_50percent(self):
        """Test 50% readout error on qubit 0"""

        # Test circuit: ideal outcome "01"
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.measure(qr, cr)
        backend = QasmSimulator()

        # 25% readout error on qubit-1 only.
        error = ReadoutError([[0.75, 0.25], [0.25, 0.75]])
        noise_model = NoiseModel()
        noise_model.add_readout_error(error, [1])

        shots = 1000
        # target = {'01': 3 * shots / 4, '11': shots / 4}
        target = {'0x1': 3 * shots / 4, '0x3': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_readout_error_specific_qubit_25percent(self):
        """Test 50% readout error on qubit 1"""

        # Test circuit: ideal outcome "01"
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.measure(qr, cr)
        backend = QasmSimulator()

        # 25% readout error on qubit-1 only.
        error = ReadoutError([[0.75, 0.25], [0.25, 0.75]])
        noise_model = NoiseModel()
        noise_model.add_readout_error(error, [1])

        shots = 1000
        # target = {'01': 3 * shots / 4, '11': shots / 4}
        target = {'0x1': 3 * shots / 4, '0x3': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_readout_error_all_qubit(self):
        """Test 100% readout error on all qubits"""

        # Test circuit: ideal outcome "01"
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.measure(qr, cr)
        backend = QasmSimulator()

        # 100% readout error on both qubits
        error = ReadoutError([[0, 1], [1, 0]])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_readout_error(error)

        shots = 100
        # target = {'10': shots}
        target = {'0x2': shots}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0)

        # asymetric readout error on both qubits (always readout 0)
        error = ReadoutError([[1, 0], [1, 0]])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_readout_error(error)

        shots = 100
        # target = {'00': shots}
        target = {'0x0': shots}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0)

    def test_reset_error_specific_qubit_50percent(self):
        """Test 50% perecent reset error on qubit-0"""

        # Test circuit: ideal outcome "11"
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        noise_circs = [[{"name": "reset", "qubits": [0]}],
                       [{"name": "id", "qubits": [0]}]]

        # 50% reset noise on qubit-0 "u3" only.
        noise_probs = [0.5, 0.5]
        error = QuantumError(zip(noise_circs, noise_probs))
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error, "u3", [0])
        shots = 1000
        target = {'0x2': shots / 2, '0x3': shots / 2}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_reset_error_specific_qubit_25percent(self):
        """Test 25% percent reset error on qubit-1"""

        # Test circuit: ideal outcome "11"
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        noise_circs = [[{"name": "reset", "qubits": [0]}],
                       [{"name": "id", "qubits": [0]}]]

        # 25% reset noise on qubit-1 "u3" only.
        noise_probs = [0.25, 0.75]
        error = QuantumError(zip(noise_circs, noise_probs))
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error, "u3", [1])
        shots = 1000
        # target = {'01': shots / 4, '11': 3 * shots / 4}
        target = {'0x1': shots / 4, '0x3': 3 * shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_reset_error_all_qubit_100percent(self):
        """Test 100% precent reset error on all qubits"""

        # Test circuit: ideal outcome "11"
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        noise_circs = [[{"name": "reset", "qubits": [0]}],
                       [{"name": "id", "qubits": [0]}]]

        # 100% reset noise on all qubit "u3".
        noise_probs = [1, 0]
        error = QuantumError(zip(noise_circs, noise_probs))
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, "u3")
        shots = 100
        # target = {'00': shots}
        target = {'0x0': shots}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0)

    def test_all_qubit_pauli_error_gate_100percent(self):
        """Test 100% Pauli error on id gates"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.iden(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 100
        # test noise model
        error = pauli_error([('X', 1)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')
        # Execute
        target = {'0x3': shots}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0)

    def test_all_qubit_pauli_error_gate_25percent(self):
        """Test 100% Pauli error on id gates"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.iden(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 2000
        # test noise model
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')
        # Execute
        target = {'0x0': 9 * shots / 16, '0x1': 3 * shots / 16,
                  '0x2': 3 * shots / 16, '0x3': shots / 16}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_specific_qubit_pauli_error_gate_100percent(self):
        """Test 100% Pauli error on id gates on qubit-1"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.iden(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 100
        # test noise model
        error = pauli_error([('X', 1)])
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error, 'id', [1])
        # Execute
        target = {'0x2': shots}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0)

    def test_specific_qubit_pauli_error_gate_25percent(self):
        """Test 100% Pauli error on id gates qubit-0"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.iden(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 1000
        # test noise model
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error, 'id', [0])
        # Execute
        target = {'0x0': 3 * shots / 4, '0x1': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_nonlocal_pauli_error_gate_25percent(self):
        """Test 100% non-local Pauli error on cx(0, 1) gate"""
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(3, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.barrier(qr)
        circuit.cx(qr[1], qr[0])
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 1000
        # test noise model
        error = pauli_error([('XII', 0.25), ('III', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_nonlocal_quantum_error(error, 'cx', [0, 1], [0, 1, 2])
        # Execute
        target = {'0x0': 3 * shots / 4, '0x4': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_all_qubit_pauli_error_measure_25percent(self):
        """Test 25% Pauli-X error on measure"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 2000
        # test noise model
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'measure')
        # Execute
        target = {'0x0': 9 * shots / 16, '0x1': 3 * shots / 16,
                  '0x2': 3 * shots / 16, '0x3': shots / 16}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_specific_qubit_pauli_error_measure_25percent(self):
        """Test 25% Pauli-X error on measure of qubit-1"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 1000
        # test noise model
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error, 'measure', [1])
        # Execute
        target = {'0x0': 3 * shots / 4, '0x2': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_nonlocal_pauli_error_measure_25percent(self):
        """Test 25% Pauli-X error on qubit-1 when measuring qubit 0"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        # use barrier to ensure measure qubit 0 is before qubit 1
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[1])
        backend = QasmSimulator()
        shots = 1000
        # test noise model
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_nonlocal_quantum_error(error, 'measure', [0], [1])
        # Execute
        target = {'0x0': 3 * shots / 4, '0x2': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_all_qubit_pauli_error_reset_25percent(self):
        """Test 25% Pauli-X error on reset"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.reset(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 2000
        # test noise model
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'measure')
        # Execute
        target = {'0x0': 9 * shots / 16, '0x1': 3 * shots / 16,
                  '0x2': 3 * shots / 16, '0x3': shots / 16}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_specific_qubit_pauli_error_reset_25percent(self):
        """Test 25% Pauli-X error on reset of qubit-1"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.reset(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 1000
        # test noise model
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error, 'measure', [1])
        # Execute
        target = {'0x0': 3 * shots / 4, '0x2': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_nonlocal_pauli_error_reset_25percent(self):
        """Test 25% Pauli-X error on qubit-1 when reseting qubit 0"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.reset(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        backend = QasmSimulator()
        shots = 1000
        # test noise model
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_nonlocal_quantum_error(error, 'measure', [0], [1])
        # Execute
        target = {'0x0': 3 * shots / 4, '0x2': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_amplitude_damping_error(self):
        """Test amplitude damping error damps to correct state"""
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)  # prepare + state
        for _ in range(30):
            # Add noisy identities
            circuit.barrier(qr)
            circuit.iden(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        shots = 1000
        backend = QasmSimulator()
        # test noise model
        error = amplitude_damping_error(0.75, 0.25)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, 'id')
        # Execute
        target = {'0x0': 3 * shots / 4, '0x1': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)


if __name__ == '__main__':
    unittest.main()
