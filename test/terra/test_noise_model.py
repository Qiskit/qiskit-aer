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


class TestNoise(common.QiskitAerTestCase):
    """Testing noise model"""

    def test_reset_error_specific_qubit(self):
        """Test reset error noise model"""

        # Test circuit: ideal outcome "11"
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
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
        shots = 2000
        target = {'0x2': shots / 2, '0x3': shots / 2}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

        # 25% reset noise on qubit-1 "u3" only.
        noise_probs = [0.25, 0.75]
        error = QuantumError(zip(noise_circs, noise_probs))
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error, "u3", [1])
        shots = 2000
        # target = {'01': shots / 4, '11': 3 * shots / 4}
        target = {'0x1': shots / 4, '0x3': 3 * shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_reset_error_all_qubit(self):
        """Test reset error noise model"""

        # Test circuit: ideal outcome "11"
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
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
        shots = 2000
        # target = {'00': shots}
        target = {'0x0': shots}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0)

    def test_readout_error_specific_qubit(self):
        """Test specific qubit readout error noise model"""

        # Test circuit: ideal outcome "01"
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.measure(qr, cr)
        backend = QasmSimulator()

        # 50% readout error on qubit-0 only.
        error = ReadoutError([[0.5, 0.5], [0.5, 0.5]])
        noise_model = NoiseModel()
        noise_model.add_readout_error(error, [0])

        shots = 2000
        # target = {'00': shots / 2, '01': shots / 2}
        target = {'0x0': shots / 2, '0x1': shots / 2}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

        # 25% readout error on qubit-1 only.
        error = ReadoutError([[0.75, 0.25], [0.25, 0.75]])
        noise_model = NoiseModel()
        noise_model.add_readout_error(error, [1])

        shots = 2000
        # target = {'01': 3 * shots / 4, '11': shots / 4}
        target = {'0x1': 3 * shots / 4, '0x3': shots / 4}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_readout_error_all_qubit(self):
        """Test specific qubit readout error noise model"""

        # Test circuit: ideal outcome "01"
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.measure(qr, cr)
        backend = QasmSimulator()

        # 100% readout error on both qubits
        error = ReadoutError([[0, 1], [1, 0]])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_readout_error(error)

        shots = 2000
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

        shots = 2000
        # target = {'00': shots}
        target = {'0x0': shots}
        qobj = compile([circuit], backend, shots=shots,
                       basis_gates=noise_model.basis_gates)
        result = backend.run(qobj, noise_model=noise_model).result()
        self.is_completed(result)
        self.compare_counts(result, [circuit], [target], delta=0)


if __name__ == '__main__':
    unittest.main()
