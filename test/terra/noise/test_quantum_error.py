# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
QuantumError class tests
"""

import unittest
from test.terra import common
import numpy as np

from qiskit.quantum_info.operators import SuperOp, Kraus
from qiskit.providers.aer.noise import QuantumError
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.providers.aer.noise.errors.errorutils import standard_gate_unitary


class TestQuantumError(common.QiskitAerTestCase):
    """Testing QuantumError class"""

    def kraus_error(self, param):
        """Return a Kraus error list"""
        return [
            np.array([[1, 0], [0, np.sqrt(1 - param)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(param)]], dtype=complex)
        ]

    def mixed_unitary_error(self, probs, labels):
        """Return a mixed unitary error list"""
        return [np.sqrt(prob) * standard_gate_unitary(label)
                for prob, label in zip(probs, labels)]

    def depol_error(self, param):
        """Return depol error unitary list"""
        return self.mixed_unitary_error(
            [1 - param * 0.75, param * 0.25, param * 0.25, param * 0.25],
            ['id', 'x', 'y', 'z'])

    def test_standard_gate_unitary(self):
        """Test standard gates are correct"""
        def norm(op_a, op_b):
            return round(np.linalg.norm(op_a - op_b), 15)

        self.assertEqual(norm(standard_gate_unitary('id'), np.eye(2)),
                         0,
                         msg="identity matrix")
        self.assertEqual(norm(standard_gate_unitary('x'),
                              np.array([[0, 1], [1, 0]])),
                         0,
                         msg="Pauli-X matrix")
        self.assertEqual(norm(standard_gate_unitary('y'),
                              np.array([[0, -1j], [1j, 0]])),
                         0,
                         msg="Pauli-Y matrix")
        self.assertEqual(norm(standard_gate_unitary('z'), np.diag([1, -1])),
                         0,
                         msg="Pauli-Z matrix")
        self.assertEqual(norm(standard_gate_unitary('h'),
                              np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
                         0,
                         msg="Hadamard gate matrix")
        self.assertEqual(norm(standard_gate_unitary('s'), np.diag([1, 1j])),
                         0,
                         msg="Phase gate matrix")
        self.assertEqual(norm(standard_gate_unitary('sdg'), np.diag([1, -1j])),
                         0,
                         msg="Adjoint phase gate matrix")
        self.assertEqual(norm(standard_gate_unitary('t'),
                              np.diag([1, (1 + 1j) / np.sqrt(2)])),
                         0,
                         msg="T gate matrix")
        self.assertEqual(norm(standard_gate_unitary('tdg'),
                              np.diag([1, (1 - 1j) / np.sqrt(2)])),
                         0,
                         msg="Adjoint T gate matrix")
        self.assertEqual(norm(
            standard_gate_unitary('cx'),
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0,
                                                                 0]])),
                         0,
                         msg="Controlled-NOT gate matrix")
        self.assertEqual(norm(standard_gate_unitary('cz'),
                              np.diag([1, 1, 1, -1])),
                         0,
                         msg="Controlled-Z gate matrix")
        self.assertEqual(norm(
            standard_gate_unitary('swap'),
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0,
                                                                 1]])),
                         0,
                         msg="SWAP matrix")
        self.assertEqual(norm(
            standard_gate_unitary('ccx'),
            np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0]])),
                         0,
                         msg="Toffoli gate matrix")

    def test_raise_probabilities_negative(self):
        """Test exception is raised for negative probabilities."""
        noise_ops = [([{
            "name": "id",
            "qubits": [0]
        }], 1.1), ([{
            "name": "x",
            "qubits": [0]
        }], -0.1)]
        self.assertRaises(NoiseError, lambda: QuantumError(noise_ops))

    def test_raise_probabilities_normalized_qobj(self):
        """Test exception is raised for qobj probabilities greater than 1."""
        noise_ops = [([{
            "name": "id",
            "qubits": [0]
        }], 0.9), ([{
            "name": "x",
            "qubits": [0]
        }], 0.2)]
        self.assertRaises(NoiseError, lambda: QuantumError(noise_ops))

    def test_raise_probabilities_normalized_unitary_kraus(self):
        """Test exception is raised for unitary kraus probs greater than 1."""
        a_0 = np.sqrt(0.9) * np.eye(2)
        a_1 = np.sqrt(0.2) * np.diag([1, -1])
        self.assertRaises(NoiseError, lambda: QuantumError([a_0, a_1]))

    def test_raise_probabilities_normalized_nonunitary_kraus(self):
        """Test exception is raised for non-unitary kraus probs greater than 1."""
        a_0 = np.sqrt(0.9) * np.array([[1, 0], [0, np.sqrt(1 - 0.3)]],
                                      dtype=complex)
        a_1 = np.sqrt(0.2) * np.array([[0, np.sqrt(0.3)], [0, 0]],
                                      dtype=complex)
        self.assertRaises(NoiseError, lambda: QuantumError([a_0, a_1]))

    def test_raise_non_cptp_kraus(self):
        """Test exception is raised for non-CPTP input."""
        a_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.array([[0, 0], [np.sqrt(0.3), 0]], dtype=complex)
        self.assertRaises(NoiseError, lambda: QuantumError([a_0, a_1]))
        self.assertRaises(NoiseError, lambda: QuantumError([a_0]))

    def test_raise_non_multiqubit_kraus(self):
        """Test exception is raised for non-multiqubit input."""
        a_0 = np.sqrt(0.5) * np.diag([1, 1, 1])
        a_1 = np.sqrt(0.5) * np.diag([1, 1, -1])
        self.assertRaises(NoiseError, lambda: QuantumError([a_0, a_1]))

    def test_pauli_conversion_standard_gates(self):
        """Test conversion of Pauli channel kraus to gates"""
        error = QuantumError(self.depol_error(1), standard_gates=True)
        for j in range(4):
            instr, _ = error.error_term(j)
            self.assertEqual(len(instr), 1)
            self.assertIn(instr[0]['name'], ['x', 'y', 'z', 'id'])
            self.assertEqual(instr[0]['qubits'], [0])
        target = SuperOp(Kraus(self.depol_error(1)))
        self.assertEqual(target, SuperOp(error))

    def test_pauli_conversion_unitary(self):
        """Test conversion of Pauli channel kraus to unitary qobj"""
        error = QuantumError(self.depol_error(1), standard_gates=False)
        for j in range(4):
            instr, _ = error.error_term(j)
            self.assertEqual(len(instr), 1)
            self.assertIn(instr[0]['name'], ['unitary', 'id'])
            self.assertEqual(instr[0]['qubits'], [0])
        target = SuperOp(Kraus(self.depol_error(1)))
        self.assertEqual(target, SuperOp(error))

    def test_tensor_both_kraus(self):
        """Test tensor of two kraus errors"""
        kraus0 = self.kraus_error(0.3)
        kraus1 = self.kraus_error(0.5)
        error = QuantumError(kraus0).tensor(QuantumError(kraus1))
        target = SuperOp(Kraus(kraus0)).tensor(Kraus(kraus1))

        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0, 1])
        self.assertEqual(target, SuperOp(error), msg="Incorrect tensor kraus")

    def test_tensor_both_unitary_instruction(self):
        """Test tensor of two unitary instruction errors."""
        unitaries0 = self.mixed_unitary_error([0.9, 0.1], ['z', 's'])
        unitaries1 = self.mixed_unitary_error([0.6, 0.4], ['x', 'y'])

        error0 = QuantumError(unitaries0, standard_gates=False)
        error1 = QuantumError(unitaries1, standard_gates=False)
        error = error0.tensor(error1)
        target = SuperOp(Kraus(unitaries0)).tensor(Kraus(unitaries1))

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertEqual(len(circ), 1)
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0, 1])
        self.assertEqual(SuperOp(error), target)

    def test_tensor_both_unitary_standard_gates(self):
        """Test tensor of two unitary standard gate errors"""
        unitaries0 = self.mixed_unitary_error([0.9, 0.1], ['z', 's'])
        unitaries1 = self.mixed_unitary_error([0.6, 0.4], ['x', 'y'])
        error0 = QuantumError(unitaries0, standard_gates=True)
        error1 = QuantumError(unitaries1, standard_gates=True)
        error = error0.tensor(error1)
        target = SuperOp(Kraus(unitaries0)).tensor(Kraus(unitaries1))

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertEqual(len(circ), 2)
            for instr in circ:
                self.assertIn(instr['name'], ['s', 'x', 'y', 'z'])
                self.assertIn(instr['qubits'], [[0], [1]])
        self.assertEqual(SuperOp(error), target)

    def test_tensor_kraus_and_unitary(self):
        """Test tensor of a kraus and unitary error."""
        kraus = self.kraus_error(0.4)
        unitaries = self.depol_error(0.1)
        error = QuantumError(kraus).tensor(QuantumError(unitaries))
        target = SuperOp(Kraus(kraus)).tensor(Kraus(unitaries))

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0, 1])
        self.assertEqual(target, SuperOp(error))

    def test_tensor_unitary_and_kraus(self):
        """Test tensor of a unitary and kraus error."""
        kraus = self.kraus_error(0.4)
        unitaries = self.depol_error(0.1)
        error = QuantumError(unitaries).tensor(QuantumError(kraus))
        target = SuperOp(Kraus(unitaries)).tensor(Kraus(kraus))

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0, 1])
        self.assertEqual(target, SuperOp(error))

    def test_expand_both_kraus(self):
        """Test expand of two kraus errors"""
        kraus0 = self.kraus_error(0.3)
        kraus1 = self.kraus_error(0.5)
        error = QuantumError(kraus0).expand(QuantumError(kraus1))
        target = SuperOp(Kraus(kraus0)).expand(Kraus(kraus1))

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0, 1])
        self.assertEqual(target, SuperOp(error))

    def test_expand_both_unitary_instruction(self):
        """Test expand of two unitary instruction errors."""
        unitaries0 = self.mixed_unitary_error([0.9, 0.1], ['z', 's'])
        unitaries1 = self.mixed_unitary_error([0.6, 0.4], ['x', 'y'])

        error0 = QuantumError(unitaries0, standard_gates=False)
        error1 = QuantumError(unitaries1, standard_gates=False)
        error = error0.expand(error1)
        target = SuperOp(Kraus(unitaries0)).expand(Kraus(unitaries1))

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0, 1])
        self.assertEqual(SuperOp(error), target)

    def test_expand_both_unitary_standard_gates(self):
        """Test expand of two unitary standard gate errors"""
        unitaries0 = self.mixed_unitary_error([0.9, 0.1], ['z', 's'])
        unitaries1 = self.mixed_unitary_error([0.6, 0.4], ['x', 'y'])

        error0 = QuantumError(unitaries0, standard_gates=True)
        error1 = QuantumError(unitaries1, standard_gates=True)
        error = error0.expand(error1)
        target = SuperOp(Kraus(unitaries0)).expand(Kraus(unitaries1))

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertEqual(len(circ), 2)
            for instr in circ:
                self.assertIn(instr['name'], ['s', 'x', 'y', 'z'])
                self.assertIn(instr['qubits'], [[0], [1]])
        self.assertEqual(SuperOp(error), target)

    def test_expand_kraus_and_unitary(self):
        """Test expand of a kraus and unitary error."""
        kraus = self.kraus_error(0.4)
        unitaries = self.depol_error(0.1)
        error = QuantumError(kraus).expand(QuantumError(unitaries))
        target = SuperOp(Kraus(kraus)).expand(Kraus(unitaries))

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0, 1])
        self.assertEqual(target, SuperOp(error))

    def test_expand_unitary_and_kraus(self):
        """Test expand of a unitary and kraus error."""
        kraus = self.kraus_error(0.4)
        unitaries = self.depol_error(0.1)
        error = QuantumError(unitaries).expand(QuantumError(kraus))
        target = SuperOp(Kraus(unitaries)).expand(Kraus(kraus))

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0, 1])
        self.assertEqual(target, SuperOp(error))

    def test_raise_compose_different_dim(self):
        """Test composing incompatible errors raises exception"""
        error0 = QuantumError([np.diag([1, 1, 1,
                                        -1])])  # 2-qubit coherent error
        error1 = QuantumError([np.diag([1, -1])])  # 1-qubit coherent error
        self.assertRaises(NoiseError, lambda: error0.compose(error1))
        self.assertRaises(NoiseError, lambda: error1.compose(error0))

    def test_compose_both_kraus(self):
        """Test compose of two kraus errors"""
        kraus0 = self.kraus_error(0.3)
        kraus1 = self.kraus_error(0.5)
        error = QuantumError(kraus0).compose(QuantumError(kraus1))
        target = SuperOp(Kraus(kraus0)).compose(Kraus(kraus1))

        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0])
        self.assertEqual(target, SuperOp(error), msg="Incorrect tensor kraus")

    def test_compose_both_unitary_instruction(self):
        """Test compose of two unitary instruction errors."""
        unitaries0 = self.mixed_unitary_error([0.9, 0.1], ['z', 's'])
        unitaries1 = self.mixed_unitary_error([0.6, 0.4], ['x', 'y'])

        error0 = QuantumError(unitaries0, standard_gates=False)
        error1 = QuantumError(unitaries1, standard_gates=False)
        error = error0.compose(error1)
        target = SuperOp(Kraus(unitaries0)).compose(Kraus(unitaries1))

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0])
        self.assertEqual(SuperOp(error), target)

    def test_compose_both_unitary_standard_gates(self):
        """Test compose of two unitary standard gate errors"""
        unitaries0 = self.mixed_unitary_error([0.9, 0.1], ['z', 's'])
        unitaries1 = self.mixed_unitary_error([0.6, 0.4], ['x', 'y'])
        error0 = QuantumError(unitaries0, standard_gates=True)
        error1 = QuantumError(unitaries1, standard_gates=True)
        error = error0.compose(error1)
        target = SuperOp(Kraus(unitaries0)).compose(Kraus(unitaries1))

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertIn(circ[0]['name'], ['s', 'x', 'y', 'z'])
            self.assertEqual(circ[0]['qubits'], [0])
        self.assertEqual(SuperOp(error), target)

    def test_compose_kraus_and_unitary(self):
        """Test compose of a kraus and unitary error."""
        kraus = self.kraus_error(0.4)
        unitaries = self.depol_error(0.1)
        error = QuantumError(kraus).compose(QuantumError(unitaries))
        target = SuperOp(Kraus(kraus)).compose(Kraus(unitaries))

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0])
        self.assertEqual(target, SuperOp(error))

    def test_compose_unitary_and_kraus(self):
        """Test compose of a unitary and kraus error."""
        kraus = self.kraus_error(0.4)
        unitaries = self.depol_error(0.1)
        error = QuantumError(unitaries).compose(QuantumError(kraus))
        target = SuperOp(Kraus(unitaries)).compose(Kraus(kraus))

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0])
        self.assertEqual(target, SuperOp(error))

    def test_dot_both_kraus(self):
        """Test dot of two kraus errors"""
        kraus0 = self.kraus_error(0.3)
        kraus1 = self.kraus_error(0.5)
        error = QuantumError(kraus0).dot(QuantumError(kraus1))
        target = SuperOp(Kraus(kraus0)).dot(Kraus(kraus1))

        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0])
        self.assertEqual(target, SuperOp(error), msg="Incorrect dot kraus")

    def test_dot_both_unitary_instruction(self):
        """Test dot of two unitary instruction errors."""
        unitaries0 = self.mixed_unitary_error([0.9, 0.1], ['z', 's'])
        unitaries1 = self.mixed_unitary_error([0.6, 0.4], ['x', 'y'])

        error0 = QuantumError(unitaries0, standard_gates=False)
        error1 = QuantumError(unitaries1, standard_gates=False)
        error = error0.dot(error1)
        target = SuperOp(Kraus(unitaries0)).dot(Kraus(unitaries1))

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0])
        self.assertEqual(SuperOp(error), target)

    def test_dot_both_unitary_standard_gates(self):
        """Test dot of two unitary standard gate errors"""
        unitaries0 = self.mixed_unitary_error([0.9, 0.1], ['z', 's'])
        unitaries1 = self.mixed_unitary_error([0.6, 0.4], ['x', 'y'])
        error0 = QuantumError(unitaries0, standard_gates=True)
        error1 = QuantumError(unitaries1, standard_gates=True)
        error = error0.dot(error1)
        target = SuperOp(Kraus(unitaries0)).dot(Kraus(unitaries1))

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertIn(circ[0]['name'], ['s', 'x', 'y', 'z'])
            self.assertEqual(circ[0]['qubits'], [0])
        self.assertEqual(SuperOp(error), target)

    def test_dot_kraus_and_unitary(self):
        """Test dot of a kraus and unitary error."""
        kraus = self.kraus_error(0.4)
        unitaries = self.depol_error(0.1)
        error = QuantumError(kraus).dot(QuantumError(unitaries))
        target = SuperOp(Kraus(kraus)).dot(Kraus(unitaries))

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0])
        self.assertEqual(target, SuperOp(error))

    def test_dot_unitary_and_kraus(self):
        """Test dot of a unitary and kraus error."""
        kraus = self.kraus_error(0.4)
        unitaries = self.depol_error(0.1)
        error = QuantumError(unitaries).dot(QuantumError(kraus))
        target = SuperOp(Kraus(unitaries)).dot(Kraus(kraus))

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0])
        self.assertEqual(target, SuperOp(error))

    def test_to_quantumchannel_kraus(self):
        """Test to_quantumchannel for Kraus inputs."""
        a_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.array([[0, 0], [0, np.sqrt(0.3)]], dtype=complex)
        b_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.5)]], dtype=complex)
        b_1 = np.array([[0, 0], [0, np.sqrt(0.5)]], dtype=complex)
        target = SuperOp(Kraus([a_0, a_1])).tensor(SuperOp(Kraus([b_0, b_1])))
        error = QuantumError([a_0, a_1]).tensor(QuantumError([b_0, b_1]))
        self.assertEqual(target, error.to_quantumchannel())

    def test_to_quantumchannel_circuit(self):
        """Test to_quantumchannel for circuit inputs."""
        noise_ops = [([{
            'name': 'reset',
            'qubits': [0]
        }], 0.2), ([{
            'name': 'reset',
            'qubits': [1]
        }], 0.3), ([{
            'name': 'id',
            'qubits': [0]
        }], 0.5)]
        error = QuantumError(noise_ops)
        reset = SuperOp(
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
        iden = SuperOp(np.eye(4))
        target = 0.2 * iden.tensor(reset) + 0.3 * reset.tensor(
            iden) + 0.5 * iden.tensor(iden)
        self.assertEqual(target, error.to_quantumchannel())

    def test_equal(self):
        """Test two quantum errors are equal"""
        a_i = np.sqrt(0.25) * standard_gate_unitary('id')
        a_x = np.sqrt(0.25) * standard_gate_unitary('x')
        a_y = np.sqrt(0.25) * standard_gate_unitary('y')
        a_z = np.sqrt(0.25) * standard_gate_unitary('z')
        error1 = QuantumError([a_i, a_x, a_y, a_z], standard_gates=True)
        error2 = QuantumError([a_i, a_x, a_y, a_z], standard_gates=False)
        self.assertEqual(error1, error2)


if __name__ == '__main__':
    unittest.main()
