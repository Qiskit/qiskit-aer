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
QuantumError class tests
"""

import unittest
from test.terra import common
import numpy as np

from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.providers.aer.noise.errors.quantum_error import QuantumError
from qiskit.providers.aer.noise.errors.errorutils import standard_gate_unitary


class TestQuantumError(common.QiskitAerTestCase):
    """Testing QuantumError class"""
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
        a_i = np.sqrt(0.25) * standard_gate_unitary('id')
        a_x = np.sqrt(0.25) * standard_gate_unitary('x')
        a_y = np.sqrt(0.25) * standard_gate_unitary('y')
        a_z = np.sqrt(0.25) * standard_gate_unitary('z')
        error_dict = QuantumError([a_i, a_x, a_y, a_z],
                                  standard_gates=True).to_dict()
        self.assertEqual(error_dict['type'], 'qerror')
        self.assertAlmostEqual(
            np.linalg.norm(
                np.array(4 * [0.25]) - np.array(error_dict['probabilities'])),
            0.0)
        for instr in error_dict['instructions']:
            self.assertEqual(len(instr), 1)
            self.assertIn(instr[0]['name'], ['x', 'y', 'z', 'id'])
            self.assertEqual(instr[0]['qubits'], [0])

    def test_pauli_conversion_unitary(self):
        """Test conversion of Pauli channel kraus to unitary qobj"""
        a_i = np.sqrt(0.25) * standard_gate_unitary('id')
        a_x = np.sqrt(0.25) * standard_gate_unitary('x')
        a_y = np.sqrt(0.25) * standard_gate_unitary('y')
        a_z = np.sqrt(0.25) * standard_gate_unitary('z')
        error_dict = QuantumError([a_i, a_x, a_y, a_z],
                                  standard_gates=False).to_dict()
        self.assertEqual(error_dict['type'], 'qerror')
        self.assertAlmostEqual(
            np.linalg.norm(
                np.array(4 * [0.25]) - np.array(error_dict['probabilities'])),
            0.0)
        for instr in error_dict['instructions']:
            self.assertEqual(len(instr), 1)
            self.assertIn(instr[0]['name'], ['unitary', 'id'])
            self.assertEqual(instr[0]['qubits'], [0])

    def test_tensor_both_kraus(self):
        """Test tensor of two kraus errors"""
        a_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.array([[0, 0], [0, np.sqrt(0.3)]], dtype=complex)
        b_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.5)]], dtype=complex)
        b_1 = np.array([[0, 0], [0, np.sqrt(0.5)]], dtype=complex)
        # Use quantum channels for reference
        target = SuperOp(Kraus([a_0, a_1]).tensor(Kraus([b_0, b_1])))
        error = QuantumError([a_0, a_1]).tensor(QuantumError([b_0, b_1]))
        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0, 1])
        error_superop = SuperOp(Kraus(kraus[0]['params']))
        self.assertEqual(target, error_superop, msg="Incorrect tensor kraus")

    def test_tensor_both_unitary_instruction(self):
        """Test tensor of two unitary instruction errors."""
        unitaries0 = [standard_gate_unitary('z'), standard_gate_unitary('s')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'), standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([
            np.sqrt(probs0[0]) * unitaries0[0],
            np.sqrt(probs0[1]) * unitaries0[1]
        ],
                              standard_gates=False)
        error1 = QuantumError([
            np.sqrt(probs1[0]) * unitaries1[0],
            np.sqrt(probs1[1]) * unitaries1[1]
        ],
                              standard_gates=False)
        error = error1.tensor(error0)
        # Kronecker product unitaries
        target_unitaries = [
            np.kron(unitaries1[0], unitaries0[0]),
            np.kron(unitaries1[0], unitaries0[1]),
            np.kron(unitaries1[1], unitaries0[0]),
            np.kron(unitaries1[1], unitaries0[1])
        ]
        # Kronecker product probabilities
        target_probs = [
            probs1[0] * probs0[0], probs1[0] * probs0[1],
            probs1[1] * probs0[0], probs1[1] * probs0[1]
        ]

        for j in range(4):
            circ, prob = error.error_term(j)
            unitary = circ[0]['params'][0]
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0, 1])
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(prob, target_probs)
            self.remove_if_found(unitary, target_unitaries)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [],
                         msg="Incorrect tensor probabilities")
        self.assertEqual(target_unitaries, [],
                         msg="Incorrect tensor unitaries")

    def test_tensor_both_unitary_standard_gates(self):
        """Test tensor of two unitary standard gate errors"""
        unitaries0 = [standard_gate_unitary('id'), standard_gate_unitary('z')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'), standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([
            np.sqrt(probs0[0]) * unitaries0[0],
            np.sqrt(probs0[1]) * unitaries0[1]
        ],
                              standard_gates=True)
        error1 = QuantumError([
            np.sqrt(probs1[0]) * unitaries1[0],
            np.sqrt(probs1[1]) * unitaries1[1]
        ],
                              standard_gates=True)
        error = error1.tensor(error0)
        # Kronecker product probabilities
        target_probs = [
            probs1[0] * probs0[0], probs1[0] * probs0[1],
            probs1[1] * probs0[0], probs1[1] * probs0[1]
        ]
        # Target circuits
        target_circs = [[{
            'name': 'x',
            'qubits': [1]
        }], [{
            'name': 'y',
            'qubits': [1]
        }], [{
            'name': 'z',
            'qubits': [0]
        }, {
            'name': 'x',
            'qubits': [1]
        }], [{
            'name': 'z',
            'qubits': [0]
        }, {
            'name': 'y',
            'qubits': [1]
        }]]
        for j in range(4):
            circ, prob = error.error_term(j)
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(prob, target_probs)
            self.remove_if_found(circ, target_circs)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [],
                         msg="Incorrect tensor probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect tensor circuits")

    def test_expand_both_kraus(self):
        """Test expand of two kraus errors"""
        a_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.array([[0, 0], [0, np.sqrt(0.3)]], dtype=complex)
        b_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.5)]], dtype=complex)
        b_1 = np.array([[0, 0], [0, np.sqrt(0.5)]], dtype=complex)
        # Use quantum channels for reference
        target = SuperOp(Kraus([a_0, a_1]).expand(Kraus([b_0, b_1])))
        error = QuantumError([a_0, a_1]).expand(QuantumError([b_0, b_1]))
        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0, 1])
        error_superop = SuperOp(Kraus(kraus[0]['params']))
        self.assertEqual(target, error_superop, msg="Incorrect expand kraus")

    def test_expand_both_unitary_instruction(self):
        """Test expand of two unitary instruction errors."""
        unitaries0 = [standard_gate_unitary('z'), standard_gate_unitary('s')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'), standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([
            np.sqrt(probs0[0]) * unitaries0[0],
            np.sqrt(probs0[1]) * unitaries0[1]
        ],
                              standard_gates=False)
        error1 = QuantumError([
            np.sqrt(probs1[0]) * unitaries1[0],
            np.sqrt(probs1[1]) * unitaries1[1]
        ],
                              standard_gates=False)
        error = error0.expand(error1)
        # Kronecker product unitaries
        target_unitaries = [
            np.kron(unitaries1[0], unitaries0[0]),
            np.kron(unitaries1[0], unitaries0[1]),
            np.kron(unitaries1[1], unitaries0[0]),
            np.kron(unitaries1[1], unitaries0[1])
        ]
        # Kronecker product probabilities
        target_probs = [
            probs1[0] * probs0[0], probs1[0] * probs0[1],
            probs1[1] * probs0[0], probs1[1] * probs0[1]
        ]

        for j in range(4):
            circ, prob = error.error_term(j)
            unitary = circ[0]['params'][0]
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0, 1])
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(prob, target_probs)
            self.remove_if_found(unitary, target_unitaries)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [],
                         msg="Incorrect expand probabilities")
        self.assertEqual(target_unitaries, [],
                         msg="Incorrect expand unitaries")

    def test_expand_both_unitary_standard_gates(self):
        """Test expand of two unitary standard gate errors"""
        unitaries0 = [standard_gate_unitary('id'), standard_gate_unitary('z')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'), standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([
            np.sqrt(probs0[0]) * unitaries0[0],
            np.sqrt(probs0[1]) * unitaries0[1]
        ],
                              standard_gates=True)
        error1 = QuantumError([
            np.sqrt(probs1[0]) * unitaries1[0],
            np.sqrt(probs1[1]) * unitaries1[1]
        ],
                              standard_gates=True)
        error = error0.expand(error1)
        # Kronecker product probabilities
        target_probs = [
            probs1[0] * probs0[0], probs1[0] * probs0[1],
            probs1[1] * probs0[0], probs1[1] * probs0[1]
        ]
        # Target circuits
        target_circs = [[{
            'name': 'x',
            'qubits': [1]
        }], [{
            'name': 'y',
            'qubits': [1]
        }], [{
            'name': 'z',
            'qubits': [0]
        }, {
            'name': 'x',
            'qubits': [1]
        }], [{
            'name': 'z',
            'qubits': [0]
        }, {
            'name': 'y',
            'qubits': [1]
        }]]
        for j in range(4):
            circ, prob = error.error_term(j)
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(prob, target_probs)
            self.remove_if_found(circ, target_circs)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [],
                         msg="Incorrect expand probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect expand circuits")

    def test_raise_compose_different_dim(self):
        """Test composing incompatible errors raises exception"""
        error0 = QuantumError([np.diag([1, 1, 1,
                                        -1])])  # 2-qubit coherent error
        error1 = QuantumError([np.diag([1, -1])])  # 1-qubit coherent error
        self.assertRaises(NoiseError, lambda: error0.compose(error1))
        self.assertRaises(NoiseError, lambda: error1.compose(error0))

    def test_compose_both_kraus(self):
        """Test composition of two kraus errors"""
        a_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.array([[0, 0], [0, np.sqrt(0.3)]], dtype=complex)
        b_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.5)]], dtype=complex)
        b_1 = np.array([[0, 0], [0, np.sqrt(0.5)]], dtype=complex)
        # Use quantum channels for reference
        target = SuperOp(Kraus([a_0, a_1]).compose(Kraus([b_0, b_1])))
        # Compose method
        error = QuantumError([a_0, a_1]).compose(QuantumError([b_0, b_1]))
        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0])
        error_superop = SuperOp(Kraus(kraus[0]['params']))
        self.assertEqual(target, error_superop, msg="Incorrect compose kraus")
        # @ method
        error = QuantumError([a_0, a_1]) @ QuantumError([b_0, b_1])
        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0])
        error_superop = SuperOp(Kraus(kraus[0]['params']))
        self.assertEqual(target, error_superop, msg="Incorrect compose kraus")

    def test_compose_both_unitary(self):
        """Test composition of two unitary errors."""
        unitaries0 = [standard_gate_unitary('z'), standard_gate_unitary('s')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'), standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([
            np.sqrt(probs0[0]) * unitaries0[0],
            np.sqrt(probs0[1]) * unitaries0[1]
        ],
                              standard_gates=False)
        error1 = QuantumError([
            np.sqrt(probs1[0]) * unitaries1[0],
            np.sqrt(probs1[1]) * unitaries1[1]
        ],
                              standard_gates=False)
        error = error0.compose(error1)
        # Kronecker product unitaries
        target_unitaries = [
            np.dot(unitaries1[0], unitaries0[0]),
            np.dot(unitaries1[0], unitaries0[1]),
            np.dot(unitaries1[1], unitaries0[0]),
            np.dot(unitaries1[1], unitaries0[1])
        ]
        # Kronecker product probabilities
        target_probs = [
            probs1[0] * probs0[0], probs1[0] * probs0[1],
            probs1[1] * probs0[0], probs1[1] * probs0[1]
        ]

        for j in range(4):
            circ, prob = error.error_term(j)
            unitary = circ[0]['params'][0]
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0])
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(prob, target_probs)
            self.remove_if_found(unitary, target_unitaries)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [],
                         msg="Incorrect compose probabilities")
        self.assertEqual(target_unitaries, [],
                         msg="Incorrect compose unitaries")

    def test_compose_both_qobj(self):
        """Test composition of two circuit errors"""
        unitaries0 = [standard_gate_unitary('id'), standard_gate_unitary('z')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'), standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([
            np.sqrt(probs0[0]) * unitaries0[0],
            np.sqrt(probs0[1]) * unitaries0[1]
        ],
                              standard_gates=True)
        error1 = QuantumError([
            np.sqrt(probs1[0]) * unitaries1[0],
            np.sqrt(probs1[1]) * unitaries1[1]
        ],
                              standard_gates=True)
        error = error0.compose(error1)
        # Kronecker product probabilities
        target_probs = [
            probs1[0] * probs0[0], probs1[0] * probs0[1],
            probs1[1] * probs0[0], probs1[1] * probs0[1]
        ]
        # Target circuits
        target_circs = [[{
            'name': 'x',
            'qubits': [0]
        }], [{
            'name': 'y',
            'qubits': [0]
        }], [{
            'name': 'z',
            'qubits': [0]
        }, {
            'name': 'x',
            'qubits': [0]
        }], [{
            'name': 'z',
            'qubits': [0]
        }, {
            'name': 'y',
            'qubits': [0]
        }]]
        for j in range(4):
            circ, prob = error.error_term(j)
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(prob, target_probs)
            self.remove_if_found(circ, target_circs)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [],
                         msg="Incorrect compose probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect compose circuits")

    def test_dot_both_kraus(self):
        """Test dot of two kraus errors"""
        a_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.array([[0, 0], [0, np.sqrt(0.3)]], dtype=complex)
        b_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.5)]], dtype=complex)
        b_1 = np.array([[0, 0], [0, np.sqrt(0.5)]], dtype=complex)
        # Use quantum channels for reference
        target = SuperOp(Kraus([b_0, b_1]).compose(Kraus([a_0, a_1])))
        # dot method
        error = QuantumError([a_0, a_1]).dot(QuantumError([b_0, b_1]))
        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0])
        error_superop = SuperOp(Kraus(kraus[0]['params']))
        self.assertEqual(target,
                         error_superop,
                         msg="Incorrect kraus dot method")
        # * method
        error = QuantumError([a_0, a_1]) * QuantumError([b_0, b_1])
        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0])
        error_superop = SuperOp(Kraus(kraus[0]['params']))
        self.assertEqual(target,
                         error_superop,
                         msg="Incorrect kraus dot method")

    def test_dot_both_unitary(self):
        """Test dot of two unitary errors."""
        unitaries0 = [standard_gate_unitary('z'), standard_gate_unitary('s')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'), standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([
            np.sqrt(probs0[0]) * unitaries0[0],
            np.sqrt(probs0[1]) * unitaries0[1]
        ],
                              standard_gates=False)
        error1 = QuantumError([
            np.sqrt(probs1[0]) * unitaries1[0],
            np.sqrt(probs1[1]) * unitaries1[1]
        ],
                              standard_gates=False)
        error = error1.dot(error0)
        # Kronecker product unitaries
        target_unitaries = [
            np.dot(unitaries1[0], unitaries0[0]),
            np.dot(unitaries1[0], unitaries0[1]),
            np.dot(unitaries1[1], unitaries0[0]),
            np.dot(unitaries1[1], unitaries0[1])
        ]
        # Kronecker product probabilities
        target_probs = [
            probs1[0] * probs0[0], probs1[0] * probs0[1],
            probs1[1] * probs0[0], probs1[1] * probs0[1]
        ]

        for j in range(4):
            circ, prob = error.error_term(j)
            unitary = circ[0]['params'][0]
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0])
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(prob, target_probs)
            self.remove_if_found(unitary, target_unitaries)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [],
                         msg="Incorrect compose probabilities")
        self.assertEqual(target_unitaries, [],
                         msg="Incorrect compose unitaries")

    def test_dot_both_qobj(self):
        """Test dot of two circuit errors"""
        unitaries0 = [standard_gate_unitary('id'), standard_gate_unitary('z')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'), standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([
            np.sqrt(probs0[0]) * unitaries0[0],
            np.sqrt(probs0[1]) * unitaries0[1]
        ],
                              standard_gates=True)
        error1 = QuantumError([
            np.sqrt(probs1[0]) * unitaries1[0],
            np.sqrt(probs1[1]) * unitaries1[1]
        ],
                              standard_gates=True)
        error = error1.dot(error0)
        # Kronecker product probabilities
        target_probs = [
            probs1[0] * probs0[0], probs1[0] * probs0[1],
            probs1[1] * probs0[0], probs1[1] * probs0[1]
        ]
        # Target circuits
        target_circs = [[{
            'name': 'x',
            'qubits': [0]
        }], [{
            'name': 'y',
            'qubits': [0]
        }], [{
            'name': 'z',
            'qubits': [0]
        }, {
            'name': 'x',
            'qubits': [0]
        }], [{
            'name': 'z',
            'qubits': [0]
        }, {
            'name': 'y',
            'qubits': [0]
        }]]
        for j in range(4):
            circ, prob = error.error_term(j)
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(prob, target_probs)
            self.remove_if_found(circ, target_circs)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [],
                         msg="Incorrect compose probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect compose circuits")

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
