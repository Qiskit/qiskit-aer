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
from qiskit.providers.aer.noise.utils import add_errors
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error
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

        result_circuit = add_errors(circuit, noise_model)

        self.assertEqual(target_circuit, result_circuit)

    def test_simple_noise(self):
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

        result_circuit = add_errors(circuit, noise_model)

        self.assertEqual(target_circuit, result_circuit)

if __name__ == '__main__':
    qr = QuantumRegister(1, 'qr')
    circuit = QuantumCircuit(qr)
    error = pauli_error([('X', 0.25), ('I', 0.75)])
    circuit.append(error.to_instruction(), [qr[0]])
    circuit == circuit
    #unittest.main()