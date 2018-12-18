# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumError class tests
"""

import unittest
from test.terra.utils import common
import numpy as np
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.providers.aer.noise.errors.quantum_error import QuantumError
from qiskit.providers.aer.noise.errors.errorutils import standard_gate_unitary


class TestQuantumError(common.QiskitAerTestCase):
    """Testing QuantumError class"""

    def test_standard_gate_unitary(self):
        """Test standard gates are correct"""
        def norm(a, b):
            return round(np.linalg.norm(a - b), 15)

        self.assertEqual(norm(standard_gate_unitary('id'), np.eye(2)), 0,
                         msg="identity matrix")
        self.assertEqual(norm(standard_gate_unitary('x'),
                              np.array([[0, 1], [1, 0]])), 0,
                         msg="Pauli-X matrix")
        self.assertEqual(norm(standard_gate_unitary('y'),
                              np.array([[0, -1j], [1j, 0]])), 0,
                         msg="Pauli-Y matrix")
        self.assertEqual(norm(standard_gate_unitary('z'),
                              np.diag([1, -1])), 0,
                         msg="Pauli-Z matrix")
        self.assertEqual(norm(standard_gate_unitary('h'),
                              np.array([[1, 1], [1, -1]]) / np.sqrt(2)), 0,
                         msg="Hadamard gate matrix")
        self.assertEqual(norm(standard_gate_unitary('s'),
                              np.diag([1, 1j])), 0,
                         msg="Phase gate matrix")
        self.assertEqual(norm(standard_gate_unitary('sdg'),
                              np.diag([1, -1j])), 0,
                         msg="Adjoint phase gate matrix")
        self.assertEqual(norm(standard_gate_unitary('t'),
                              np.diag([1, (1 + 1j) / np.sqrt(2)])), 0,
                         msg="T gate matrix")
        self.assertEqual(norm(standard_gate_unitary('tdg'),
                              np.diag([1, (1 - 1j) / np.sqrt(2)])), 0,
                         msg="Adjoint T gate matrix")
        self.assertEqual(norm(standard_gate_unitary('cx'),
                              np.array([[1, 0, 0, 0],
                                        [0, 0, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 1, 0, 0]])), 0,
                         msg="Controlled-NOT gate matrix")
        self.assertEqual(norm(standard_gate_unitary('cz'),
                              np.diag([1, 1, 1, -1])), 0,
                         msg="Controlled-Z gate matrix")
        self.assertEqual(norm(standard_gate_unitary('swap'),
                              np.array([[1, 0, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]])), 0,
                         msg="SWAP matrix")
        self.assertEqual(norm(standard_gate_unitary('ccx'),
                              np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 1, 0, 0, 0, 0]])), 0,
                         msg="Toffoli gate matrix")

    def test_raise_probabilities_negative(self):
        """Test exception is raised for negative probabilities."""
        noise_ops = [([{"name": "id", "qubits": [0]}], 1.1),
                     ([{"name": "x", "qubits": [0]}], -0.1)]
        self.assertRaises(NoiseError, lambda: QuantumError(noise_ops))

    def test_raise_probabilities_normalized_qobj(self):
        """Test exception is raised for qobj probabilities greater than 1."""
        noise_ops = [([{"name": "id", "qubits": [0]}], 0.9),
                     ([{"name": "x", "qubits": [0]}], 0.2)]
        self.assertRaises(NoiseError, lambda: QuantumError(noise_ops))

    def test_raise_probabilities_normalized_unitary_kraus(self):
        """Test exception is raised for unitary kraus probs greater than 1."""
        A0 = np.sqrt(0.9) * np.eye(2)
        A1 = np.sqrt(0.2) * np.diag([1, -1])
        self.assertRaises(NoiseError, lambda: QuantumError([A0, A1]))

    def test_raise_probabilities_normalized_nonunitary_kraus(self):
        """Test exception is raised for non-unitary kraus probs greater than 1."""
        A0 = np.sqrt(0.9) * np.array([[1, 0], [0, np.sqrt(1 - 0.3)]])
        A1 = np.sqrt(0.2) * np.array([[0, np.sqrt(0.3)], [0, 0]])
        self.assertRaises(NoiseError, lambda: QuantumError([A0, A1]))

    def test_raise_non_cptp_kraus(self):
        """Test exception is raised for non-CPTP input."""
        A0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]])
        A1 = np.array([[0, 0], [np.sqrt(0.3), 0]])
        self.assertRaises(NoiseError, lambda: QuantumError([A0, A1]))
        self.assertRaises(NoiseError, lambda: QuantumError([A0]))

    def test_raise_non_multiqubit_kraus(self):
        """Test exception is raised for non-multiqubit input."""
        A0 = np.sqrt(0.5) * np.diag([1, 1, 1])
        A1 = np.sqrt(0.5) * np.diag([1, 1, -1])
        self.assertRaises(NoiseError, lambda: QuantumError([A0, A1]))

    def test_pauli_conversion_standard_gates(self):
        """Test conversion of Pauli channel kraus to gates"""
        Ai = np.sqrt(0.25) * standard_gate_unitary('id')
        Ax = np.sqrt(0.25) * standard_gate_unitary('x')
        Ay = np.sqrt(0.25) * standard_gate_unitary('y')
        Az = np.sqrt(0.25) * standard_gate_unitary('z')
        error_dict = QuantumError([Ai, Ax, Ay, Az], standard_gates=True).as_dict()
        self.assertEqual(error_dict['type'], 'qerror')
        self.assertAlmostEqual(np.linalg.norm(np.array(4 * [0.25]) -
                                              np.array(error_dict['probabilities'])), 0.0)
        for instr in error_dict['instructions']:
            self.assertEqual(len(instr), 1)
            self.assertIn(instr[0]['name'], ['x', 'y', 'z', 'id'])
            self.assertEqual(instr[0]['qubits'], [0])

    def test_pauli_conversion_unitary(self):
        """Test conversion of Pauli channel kraus to unitary qobj"""
        Ai = np.sqrt(0.25) * standard_gate_unitary('id')
        Ax = np.sqrt(0.25) * standard_gate_unitary('x')
        Ay = np.sqrt(0.25) * standard_gate_unitary('y')
        Az = np.sqrt(0.25) * standard_gate_unitary('z')
        error_dict = QuantumError([Ai, Ax, Ay, Az],
                                  standard_gates=False).as_dict()
        self.assertEqual(error_dict['type'], 'qerror')
        self.assertAlmostEqual(np.linalg.norm(np.array(4 * [0.25]) -
                                              np.array(error_dict['probabilities'])), 0.0)
        for instr in error_dict['instructions']:
            self.assertEqual(len(instr), 1)
            self.assertIn(instr[0]['name'], ['unitary', 'id'])
            self.assertEqual(instr[0]['qubits'], [0])

    def test_kron_both_kraus(self):
        """Test kronecker product of two kraus errors"""
        A0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]])
        A1 = np.array([[0, 0], [0, np.sqrt(0.3)]])
        B0 = np.array([[1, 0], [0, np.sqrt(1 - 0.5)]])
        B1 = np.array([[0, 0], [0, np.sqrt(0.5)]])
        error = QuantumError([B0, B1]).kron(QuantumError([A0, A1]))
        kraus, p = error.error_term(0)
        targets = [np.kron(B0, A0), np.kron(B0, A1),
                   np.kron(B1, A0), np.kron(B1, A1)]
        self.assertEqual(p, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0, 1])
        for op in kraus[0]['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect kron kraus")

    def test_kron_both_unitary(self):
        """Test kronecker product of two unitary qobj errors."""
        unitaries0 = [standard_gate_unitary('z'),
                      standard_gate_unitary('s')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'),
                      standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([np.sqrt(probs0[0]) * unitaries0[0],
                               np.sqrt(probs0[1]) * unitaries0[1]],
                              standard_gates=False)
        error1 = QuantumError([np.sqrt(probs1[0]) * unitaries1[0],
                               np.sqrt(probs1[1]) * unitaries1[1]],
                              standard_gates=False)
        error = error1.kron(error0)
        # Kronecker product unitaries
        target_unitaries = [np.kron(unitaries1[0], unitaries0[0]),
                            np.kron(unitaries1[0], unitaries0[1]),
                            np.kron(unitaries1[1], unitaries0[0]),
                            np.kron(unitaries1[1], unitaries0[1])]
        # Kronecker product probabilities
        target_probs = [probs1[0] * probs0[0], probs1[0] * probs0[1],
                        probs1[1] * probs0[0], probs1[1] * probs0[1]]

        for j in range(4):
            circ, p = error.error_term(j)
            unitary = circ[0]['params']
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0, 1])
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(p, target_probs)
            self.remove_if_found(unitary, target_unitaries)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [], msg="Incorrect kron probabilities")
        self.assertEqual(target_unitaries, [], msg="Incorrect kron unitaries")

    def test_kron_both_qobj(self):
        """Test kronecker product of two unitary gate errors"""
        unitaries0 = [standard_gate_unitary('id'),
                      standard_gate_unitary('z')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'),
                      standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([np.sqrt(probs0[0]) * unitaries0[0],
                               np.sqrt(probs0[1]) * unitaries0[1]],
                              standard_gates=True)
        error1 = QuantumError([np.sqrt(probs1[0]) * unitaries1[0],
                               np.sqrt(probs1[1]) * unitaries1[1]],
                              standard_gates=True)
        error = error1.kron(error0)
        # Kronecker product probabilities
        target_probs = [probs1[0] * probs0[0],
                        probs1[0] * probs0[1],
                        probs1[1] * probs0[0],
                        probs1[1] * probs0[1]]
        # Target circuits
        target_circs = [[{'name': 'id', 'qubits': [0]}, {'name': 'x', 'qubits': [1]}],
                        [{'name': 'id', 'qubits': [0]}, {'name': 'y', 'qubits': [1]}],
                        [{'name': 'z', 'qubits': [0]}, {'name': 'x', 'qubits': [1]}],
                        [{'name': 'z', 'qubits': [0]}, {'name': 'y', 'qubits': [1]}]]
        for j in range(4):
            circ, p = error.error_term(j)
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ, target_circs)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [], msg="Incorrect kron probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect kron circuits")

    def test_raise_compose_different_dim(self):
        """Test composing incompatible errors raises exception"""
        error0 = QuantumError([np.diag([1, 1, 1, -1])])  # 2-qubit coherent error
        error1 = QuantumError([np.diag([1, -1])])  # 1-qubit coherent error
        self.assertRaises(NoiseError, lambda: error0.compose(error1))
        self.assertRaises(NoiseError, lambda: error1.compose(error0))

    def test_compose_both_kraus(self):
        """Test composition of two kraus errors"""
        A0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]])
        A1 = np.array([[0, 0], [0, np.sqrt(0.3)]])
        B0 = np.array([[1, 0], [0, np.sqrt(1 - 0.5)]])
        B1 = np.array([[0, 0], [0, np.sqrt(0.5)]])
        error = QuantumError([A0, A1]).compose(QuantumError([B0, B1]))
        kraus, p = error.error_term(0)
        targets = [np.dot(B0, A0), np.dot(B0, A1),
                   np.dot(B1, A0), np.dot(B1, A1)]
        self.assertEqual(p, 1)
        self.assertEqual(kraus[0]['name'], 'kraus')
        self.assertEqual(kraus[0]['qubits'], [0])
        for op in kraus[0]['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect compose kraus")

    def test_compose_both_unitary(self):
        """Test composition of two unitary errors."""
        unitaries0 = [standard_gate_unitary('z'),
                      standard_gate_unitary('s')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'),
                      standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([np.sqrt(probs0[0]) * unitaries0[0],
                               np.sqrt(probs0[1]) * unitaries0[1]],
                              standard_gates=False)
        error1 = QuantumError([np.sqrt(probs1[0]) * unitaries1[0],
                               np.sqrt(probs1[1]) * unitaries1[1]],
                              standard_gates=False)
        error = error0.compose(error1)
        # Kronecker product unitaries
        target_unitaries = [np.dot(unitaries1[0], unitaries0[0]),
                            np.dot(unitaries1[0], unitaries0[1]),
                            np.dot(unitaries1[1], unitaries0[0]),
                            np.dot(unitaries1[1], unitaries0[1])]
        # Kronecker product probabilities
        target_probs = [probs1[0] * probs0[0],
                        probs1[0] * probs0[1],
                        probs1[1] * probs0[0],
                        probs1[1] * probs0[1]]

        for j in range(4):
            circ, p = error.error_term(j)
            unitary = circ[0]['params']
            self.assertEqual(circ[0]['name'], 'unitary')
            self.assertEqual(circ[0]['qubits'], [0])
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(p, target_probs)
            self.remove_if_found(unitary, target_unitaries)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [], msg="Incorrect compose probabilities")
        self.assertEqual(target_unitaries, [], msg="Incorrect compose unitaries")

    def test_compose_both_qobj(self):
        """Test composition of two circuit errors"""
        unitaries0 = [standard_gate_unitary('id'),
                      standard_gate_unitary('z')]
        probs0 = [0.9, 0.1]
        unitaries1 = [standard_gate_unitary('x'),
                      standard_gate_unitary('y')]
        probs1 = [0.6, 0.4]
        error0 = QuantumError([np.sqrt(probs0[0]) * unitaries0[0],
                               np.sqrt(probs0[1]) * unitaries0[1]],
                              standard_gates=True)
        error1 = QuantumError([np.sqrt(probs1[0]) * unitaries1[0],
                               np.sqrt(probs1[1]) * unitaries1[1]],
                              standard_gates=True)
        error = error0.compose(error1)
        # Kronecker product probabilities
        target_probs = [probs1[0] * probs0[0],
                        probs1[0] * probs0[1],
                        probs1[1] * probs0[0],
                        probs1[1] * probs0[1]]
        # Target circuits
        target_circs = [[{'name': 'id', 'qubits': [0]}, {'name': 'x', 'qubits': [0]}],
                        [{'name': 'id', 'qubits': [0]}, {'name': 'y', 'qubits': [0]}],
                        [{'name': 'z', 'qubits': [0]}, {'name': 'x', 'qubits': [0]}],
                        [{'name': 'z', 'qubits': [0]}, {'name': 'y', 'qubits': [0]}]]
        for j in range(4):
            circ, p = error.error_term(j)
            # Remove prob from target if it is found
            # later we will check that target_probs is empty so all
            # the required ones have been removed
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ, target_circs)
        # Check we had all the correct target probs and unitaries
        # by seeing if these lists are empty
        # Note that this doesn't actually check that the correct
        # prob was assigned to the correct unitary.
        self.assertEqual(target_probs, [], msg="Incorrect compose probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect compose circuits")


if __name__ == '__main__':
    unittest.main()
