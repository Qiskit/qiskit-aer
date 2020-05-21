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
noise_model_inserter module tests
"""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.providers.aer.utils import insert_noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error
from qiskit.qasm import pi
import unittest

class TestNoiseInserter(unittest.TestCase):
    def test_no_noise(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[1])
        circuit.z(qr[2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.y(qr[1])
        target_circuit.z(qr[2])

        noise_model = NoiseModel() #empty

        result_circuit = insert_noise(circuit, noise_model)

        self.assertEqual(target_circuit, result_circuit)

    def test_all_qubit_quantum_errors(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[1])
        circuit.z(qr[2])

        error_x = pauli_error([('Y', 0.25), ('I', 0.75)])
        error_y = pauli_error([('X', 0.35), ('Z', 0.65)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_x, 'x')
        noise_model.add_all_qubit_quantum_error(error_y, 'y')

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.append(error_x.to_instruction(), [qr[0]])
        target_circuit.y(qr[1])
        target_circuit.append(error_y.to_instruction(), [qr[1]])
        target_circuit.z(qr[2])

        result_circuit = insert_noise(circuit, noise_model)

        self.assertEqual(target_circuit, result_circuit)

    def test_local_quantum_errors(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.x(qr[1])
        circuit.y(qr[2])

        error_x = pauli_error([('Y', 0.25), ('I', 0.75)])
        error_y = pauli_error([('X', 0.35), ('Z', 0.65)])
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error_x, 'x', [0])
        noise_model.add_quantum_error(error_y, 'y', [2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.append(error_x.to_instruction(), [qr[0]])
        target_circuit.x(qr[1])
        target_circuit.y(qr[2])
        target_circuit.append(error_y.to_instruction(), [qr[2]])

        result_circuit = insert_noise(circuit, noise_model)

        self.assertEqual(target_circuit, result_circuit)

    def test_nonlocal_quantum_errors(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.x(qr[2])

        error_x = pauli_error([('Y', 0.25), ('I', 0.75)])
        noise_model = NoiseModel()
        noise_model.add_nonlocal_quantum_error(error_x, 'x', [0], [1])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.append(error_x.to_instruction(), [qr[1]])
        target_circuit.x(qr[2])

        result_circuit = insert_noise(circuit, noise_model)

        self.assertEqual(target_circuit, result_circuit)

    def test_transpiling(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[1])
        circuit.z(qr[2])

        error_x = pauli_error([('Y', 0.25), ('I', 0.75)])
        error_y = pauli_error([('X', 0.35), ('Z', 0.65)])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_x, 'x')
        noise_model.add_all_qubit_quantum_error(error_y, 'u1')

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.append(error_x.to_instruction(), [qr[0]])
        target_circuit.u3(pi, pi / 2, pi / 2, qr[1])
        target_circuit.u1(pi, qr[2])
        target_circuit.append(error_y.to_instruction(), [qr[2]])

        result_circuit = insert_noise(circuit, noise_model, transpile=True)
        self.assertEqual(target_circuit, result_circuit)

    def test_multiple_inputs(self):
        qr = QuantumRegister(1, 'qr')
        circuit1 = QuantumCircuit(qr)
        circuit1.x(qr[0])

        circuit2 = QuantumCircuit(qr)
        circuit2.y(qr[0])

        circuits_list = [circuit1, circuit2]
        circuits_tuple = (circuit1, circuit2)

        noise_model = NoiseModel()
        error_x = pauli_error([('Y', 0.25), ('I', 0.75)])
        error_y = pauli_error([('X', 0.35), ('Z', 0.65)])
        noise_model.add_all_qubit_quantum_error(error_x, 'x')
        noise_model.add_all_qubit_quantum_error(error_y, 'y')

        target_circuit1 = QuantumCircuit(qr)
        target_circuit1.x(qr[0])
        target_circuit1.append(error_x.to_instruction(), [qr[0]])

        target_circuit2 = QuantumCircuit(qr)
        target_circuit2.y(qr[0])
        target_circuit2.append(error_y.to_instruction(), [qr[0]])

        target_circuits = [target_circuit1, target_circuit2]
        result_circuits = insert_noise(circuits_list, noise_model)
        self.assertEqual(target_circuits, result_circuits)

        target_circuits = [target_circuit1, target_circuit2]
        result_circuits = insert_noise(circuits_tuple, noise_model)
        self.assertEqual(target_circuits, result_circuits)


if __name__ == '__main__':
    unittest.main()
