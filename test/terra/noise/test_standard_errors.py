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
Standard error function tests
"""

import unittest
from test.terra import common
import numpy as np

import qiskit.quantum_info as qi
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.providers.aer.noise.errors.errorutils import standard_gate_unitary
from qiskit.providers.aer.noise.errors.standard_errors import kraus_error
from qiskit.providers.aer.noise.errors.standard_errors import mixed_unitary_error
from qiskit.providers.aer.noise.errors.standard_errors import coherent_unitary_error
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error
from qiskit.providers.aer.noise.errors.standard_errors import thermal_relaxation_error
from qiskit.providers.aer.noise.errors.standard_errors import phase_amplitude_damping_error
from qiskit.providers.aer.noise.errors.standard_errors import amplitude_damping_error
from qiskit.providers.aer.noise.errors.standard_errors import phase_damping_error

# TODO: Test Kraus thermal relaxation error by comparing to amplitude damping channel


class TestNoise(common.QiskitAerTestCase):
    """Testing Standard Errors package"""

    def test_kraus_error(self):
        """Test Kraus error when input is list instead of numpy array"""
        A0 = [[1, 0], [0, np.sqrt(1 - 0.3)]]
        A1 = [[0, 0], [0, np.sqrt(0.3)]]
        targets = [A0, A1]
        error = kraus_error(targets)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1)
        kraus = circ[0]
        self.assertEqual(kraus['name'], 'kraus')
        self.assertEqual(kraus['qubits'], [0])
        for op in kraus['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect kraus QuantumError")

    def test_mixed_unitary_error_raise_nonunitary(self):
        """Test error is raised if input is not unitary."""
        A0 = [[1, 0], [0, np.sqrt(1 - 0.3)]]
        A1 = [[0, 0], [0, np.sqrt(0.3)]]
        noise_ops = [(A0, 0.5), (A1, 0.5)]
        self.assertRaises(NoiseError, lambda: mixed_unitary_error(noise_ops))

    def test_mixed_unitary_error_raise_differnt_shape(self):
        """Test error is raised if input matrices different size"""
        unitaries = [np.eye(4), np.eye(2)]
        probs = [0.7, 0.4]
        noise_ops = [(unitaries[0], probs[0]), (unitaries[1], probs[1])]
        self.assertRaises(NoiseError, lambda: mixed_unitary_error(noise_ops))

    def test_mixed_unitary_error(self):
        """Test construction of mixed unitary error"""
        unitaries = [np.eye(2), np.diag([1, -1])]
        probs = [0.7, 0.3]
        error = mixed_unitary_error([(unitaries[0], probs[0]),
                                     (unitaries[1], probs[1])],
                                    standard_gates=True)
        (op0, p0) = error.error_term(0)
        (op1, p1) = error.error_term(1)
        self.assertEqual(op0[0], {"name": "z", "qubits": [0]})
        self.assertEqual(op1[0], {"name": "id", "qubits": [0]})
        self.assertEqual(p0, 0.3)
        self.assertEqual(p1, 0.7)

    def test_coherent_unitary_error(self):
        """Test coherent unitary error"""
        unitary = np.diag([1, -1, 1, -1])
        error = coherent_unitary_error(unitary)
        ref = mixed_unitary_error([(unitary, 1)])
        self.assertEqual(qi.SuperOp(error), qi.SuperOp(ref))

    def test_pauli_error_raise_invalid(self):
        """Test exception for invalid Pauli string"""
        self.assertRaises(NoiseError, lambda: pauli_error([('S', 1)]))

    def test_pauli_error_1q_unitary_from_string(self):
        """Test single-qubit pauli error as unitary qobj from string label"""
        paulis = ['I', 'X', 'Y', 'Z']
        probs = [0.4, 0.3, 0.2, 0.1]
        error = pauli_error(zip(paulis, probs), standard_gates=False)

        target_unitaries = [standard_gate_unitary('x'),
                            standard_gate_unitary('y'),
                            standard_gate_unitary('z')]
        target_probs = probs.copy()
        target_identity_count = 0
        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            name = circ[0]['name']
            self.assertIn(name, ('unitary', 'id'))
            self.assertEqual(circ[0]['qubits'], [0])
            self.remove_if_found(p, target_probs)
            if name == "unitary":
                self.remove_if_found(circ[0]['params'][0], target_unitaries)
            else:
                target_identity_count += 1
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_unitaries, [], msg="Incorrect unitaries")
        self.assertEqual(target_identity_count, 1, msg="Incorrect identities")

    def test_pauli_error_1q_gate_from_string(self):
        """Test single-qubit pauli error as gate qobj from string label"""
        paulis = ['I', 'X', 'Y', 'Z']
        probs = [0.4, 0.3, 0.2, 0.1]
        error = pauli_error(zip(paulis, probs), standard_gates=True)

        target_circs = [[{"name": "id", "qubits": [0]}],
                        [{"name": "x", "qubits": [0]}],
                        [{"name": "y", "qubits": [0]}],
                        [{"name": "z", "qubits": [0]}]]
        target_probs = probs.copy()

        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ, target_circs)
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect circuits")

    def test_pauli_error_1q_unitary_from_pauli(self):
        """Test single-qubit pauli error as unitary qobj from Pauli obj"""
        paulis = [qi.Pauli(s) for s in ['I', 'X', 'Y', 'Z']]
        probs = [0.4, 0.3, 0.2, 0.1]
        error = pauli_error(zip(paulis, probs), standard_gates=False)

        target_unitaries = [standard_gate_unitary('x'),
                            standard_gate_unitary('y'),
                            standard_gate_unitary('z')]
        target_probs = probs.copy()
        target_identity_count = 0
        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            name = circ[0]['name']
            self.assertIn(name, ('unitary', 'id'))
            self.assertEqual(circ[0]['qubits'], [0])
            self.remove_if_found(p, target_probs)
            if name == "unitary":
                self.remove_if_found(circ[0]['params'][0], target_unitaries)
            else:
                target_identity_count += 1
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_unitaries, [], msg="Incorrect unitaries")
        self.assertEqual(target_identity_count, 1, msg="Incorrect identities")

    def test_pauli_error_1q_gate_from_pauli(self):
        """Test single-qubit pauli error as gate qobj from Pauli obj"""
        paulis = [qi.Pauli(s) for s in ['I', 'X', 'Y', 'Z']]
        probs = [0.4, 0.3, 0.2, 0.1]
        error = pauli_error(zip(paulis, probs), standard_gates=True)

        target_circs = [[{"name": "id", "qubits": [0]}],
                        [{"name": "x", "qubits": [0]}],
                        [{"name": "y", "qubits": [0]}],
                        [{"name": "z", "qubits": [0]}]]
        target_probs = probs.copy()

        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ, target_circs)
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect circuits")

    def test_pauli_error_2q_unitary_from_string(self):
        """Test two-qubit pauli error as unitary qobj from string label"""
        paulis = ['XY', 'YZ', 'ZX']
        probs = [0.5, 0.3, 0.2]
        error = pauli_error(zip(paulis, probs), standard_gates=False)

        X = standard_gate_unitary('x')
        Y = standard_gate_unitary('y')
        Z = standard_gate_unitary('z')
        target_unitaries = [np.kron(X, Y), np.kron(Y, Z), np.kron(Z, X)]
        target_probs = probs.copy()
        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            name = circ[0]['name']
            self.assertIn(name, 'unitary')
            self.assertEqual(circ[0]['qubits'], [0, 1])
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ[0]['params'][0], target_unitaries)
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_unitaries, [], msg="Incorrect unitaries")

    def test_pauli_error_2q_unitary_from_string_1q_only(self):
        """Test two-qubit pauli error as unitary qobj from string label"""
        paulis = ['XI', 'YI', 'ZI']
        probs = [0.5, 0.3, 0.2]
        error = pauli_error(zip(paulis, probs), standard_gates=False)

        target_unitaries = [standard_gate_unitary('x'),
                            standard_gate_unitary('y'),
                            standard_gate_unitary('z')]
        target_probs = probs.copy()
        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            name = circ[0]['name']
            self.assertIn(name, 'unitary')
            self.assertEqual(circ[0]['qubits'], [1])
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ[0]['params'][0], target_unitaries)
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_unitaries, [], msg="Incorrect unitaries")

    def test_pauli_error_2q_gate_from_string(self):
        """Test two-qubit pauli error as gate qobj from string label"""
        paulis = ['XZ', 'YX', 'ZY']
        probs = [0.5, 0.3, 0.2]
        error = pauli_error(zip(paulis, probs), standard_gates=True)

        target_circs = [[{"name": "z", "qubits": [0]}, {"name": "x", "qubits": [1]}],
                        [{"name": "x", "qubits": [0]}, {"name": "y", "qubits": [1]}],
                        [{"name": "y", "qubits": [0]}, {"name": "z", "qubits": [1]}]]
        target_probs = probs.copy()

        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ, target_circs)
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect circuits")

    def test_pauli_error_2q_gate_from_string_1qonly(self):
        """Test two-qubit pauli error as gate qobj from string label"""
        paulis = ['XI', 'YI', 'ZI']
        probs = [0.5, 0.3, 0.2]
        error = pauli_error(zip(paulis, probs), standard_gates=True)

        target_circs = [[{"name": "x", "qubits": [1]}],
                        [{"name": "y", "qubits": [1]}],
                        [{"name": "z", "qubits": [1]}]]
        target_probs = probs.copy()

        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ, target_circs)
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect circuits")

    def test_pauli_error_2q_unitary_from_pauli(self):
        """Test two-qubit pauli error as unitary qobj from Pauli obj"""
        paulis = [qi.Pauli(s) for s in ['XY', 'YZ', 'ZX']]
        probs = [0.5, 0.3, 0.2]
        error = pauli_error(zip(paulis, probs), standard_gates=False)

        X = standard_gate_unitary('x')
        Y = standard_gate_unitary('y')
        Z = standard_gate_unitary('z')
        target_unitaries = [np.kron(X, Y), np.kron(Y, Z), np.kron(Z, X)]
        target_probs = probs.copy()
        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            name = circ[0]['name']
            self.assertIn(name, 'unitary')
            self.assertEqual(circ[0]['qubits'], [0, 1])
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ[0]['params'][0], target_unitaries)
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_unitaries, [], msg="Incorrect unitaries")

    def test_pauli_error_2q_gate_from_pauli(self):
        """Test two-qubit pauli error as gate qobj from Pauli obj"""
        paulis = [qi.Pauli(s) for s in ['XZ', 'YX', 'ZY']]
        probs = [0.5, 0.3, 0.2]
        error = pauli_error(zip(paulis, probs), standard_gates=True)

        target_circs = [[{"name": "z", "qubits": [0]}, {"name": "x", "qubits": [1]}],
                        [{"name": "x", "qubits": [0]}, {"name": "y", "qubits": [1]}],
                        [{"name": "y", "qubits": [0]}, {"name": "z", "qubits": [1]}]]
        target_probs = probs.copy()

        for j in range(len(paulis)):
            circ, p = error.error_term(j)
            self.remove_if_found(p, target_probs)
            self.remove_if_found(circ, target_circs)
        self.assertEqual(target_probs, [], msg="Incorrect probabilities")
        self.assertEqual(target_circs, [], msg="Incorrect circuits")

    def test_depolarizing_error_identity_unitary(self):
        """Test depolarizing error with p=0 (ideal) as unitary qobj"""
        # 1 qubit
        error = depolarizing_error(0, 1, standard_gates=False)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="ideal probability")
        self.assertEqual(circ[0], {"name": "id", "qubits": [0]}, msg="ideal circuit")
        # 2-qubit
        error = depolarizing_error(0, 2, standard_gates=False)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="ideal probability")
        self.assertEqual(circ[0], {"name": "id", "qubits": [0]}, msg="ideal circuit")

    def test_depolarizing_error_ideal(self):
        """Test depolarizing error with p=0 (ideal) as gate qobj"""
        # 1 qubit
        error = depolarizing_error(0, 1, standard_gates=True)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="ideal probability")
        self.assertEqual(circ[0], {"name": "id", "qubits": [0]}, msg="ideal circuit")
        # 2-qubit
        error = depolarizing_error(0, 2, standard_gates=True)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="ideal probability")
        self.assertEqual(circ[0], {"name": "id", "qubits": [0]}, msg="ideal circuit")

    def test_depolarizing_error_1q_unitary(self):
        """Test 1-qubit depolarizing error as unitary qobj"""
        p_depol = 0.3
        error = depolarizing_error(p_depol, 1, standard_gates=False)
        target_unitaries = [standard_gate_unitary('x'),
                            standard_gate_unitary('y'),
                            standard_gate_unitary('z')]
        for j in range(4):
            circ, p = error.error_term(j)
            name = circ[0]['name']
            self.assertIn(name, ('unitary', "id"))
            self.assertEqual(circ[0]['qubits'], [0])
            if name == "unitary":
                self.assertAlmostEqual(p, p_depol / 4, msg="Incorrect Pauli probability")
                self.remove_if_found(circ[0]['params'][0], target_unitaries)
            else:
                self.assertAlmostEqual(p, 1 - p_depol + p_depol / 4,
                                       msg="Incorrect identity probability")
        self.assertEqual(target_unitaries, [], msg="Incorrect unitaries")

    def test_depolarizing_error_1q_gate(self):
        """Test 1-qubit depolarizing error as gate qobj"""
        p_depol = 0.3
        error = depolarizing_error(p_depol, 1, standard_gates=True)
        target_circs = [[{"name": "id", "qubits": [0]}],
                        [{"name": "x", "qubits": [0]}],
                        [{"name": "y", "qubits": [0]}],
                        [{"name": "z", "qubits": [0]}]]
        for j in range(4):
            circ, p = error.error_term(j)
            self.assertEqual(circ[0]['qubits'], [0])
            if circ[0]['name'] == "id":
                self.assertAlmostEqual(p, 1 - p_depol + p_depol / 4,
                                       msg="Incorrect identity probability")
            else:
                self.assertAlmostEqual(p, p_depol / 4, msg="Incorrect Pauli probability")
            self.remove_if_found(circ, target_circs)
        self.assertEqual(target_circs, [], msg="Incorrect unitaries")

    def test_depolarizing_error_2q_unitary(self):
        """Test 2-qubit depolarizing error as unitary qobj"""
        p_depol = 0.3
        error = depolarizing_error(p_depol, 2, standard_gates=False)
        X = standard_gate_unitary('x')
        Y = standard_gate_unitary('y')
        Z = standard_gate_unitary('z')
        target_unitaries = [X, Y, Z,  # on qubit 0
                            X, Y, Z,  # on qubit 1
                            np.kron(X, X), np.kron(X, Y), np.kron(X, Z),
                            np.kron(Y, X), np.kron(Y, Y), np.kron(Y, Z),
                            np.kron(Z, X), np.kron(Z, Y), np.kron(Z, Z)]
        for j in range(16):
            circ, p = error.error_term(j)
            name = circ[0]['name']
            self.assertIn(name, ('unitary', "id"))
            if name == "unitary":
                self.assertAlmostEqual(p, p_depol / 16)
                op = circ[0]['params'][0]
                qubits = circ[0]['qubits']
                if len(op) == 2:
                    self.assertIn(qubits, [[0], [1]])
                else:
                    self.assertEqual(qubits, [0, 1])
                self.remove_if_found(op, target_unitaries)
            else:
                self.assertAlmostEqual(p, 1 - p_depol + p_depol / 16)
                self.assertEqual(circ[0]['qubits'], [0])
        self.assertEqual(target_unitaries, [], msg="Incorrect unitaries")

    def test_depolarizing_error_2q_gate(self):
        """Test 2-qubit depolarizing error as gate qobj"""
        p_depol = 0.3
        error = depolarizing_error(p_depol, 2, standard_gates=True)
        target_circs = [[{"name": "id", "qubits": [0]}],
                        [{"name": "x", "qubits": [0]}],
                        [{"name": "y", "qubits": [0]}],
                        [{"name": "z", "qubits": [0]}],
                        [{"name": "x", "qubits": [1]}],
                        [{"name": "y", "qubits": [1]}],
                        [{"name": "z", "qubits": [1]}],
                        [{"name": "x", "qubits": [0]}, {"name": "x", "qubits": [1]}],
                        [{"name": "x", "qubits": [0]}, {"name": "y", "qubits": [1]}],
                        [{"name": "x", "qubits": [0]}, {"name": "z", "qubits": [1]}],
                        [{"name": "y", "qubits": [0]}, {"name": "x", "qubits": [1]}],
                        [{"name": "y", "qubits": [0]}, {"name": "y", "qubits": [1]}],
                        [{"name": "y", "qubits": [0]}, {"name": "z", "qubits": [1]}],
                        [{"name": "z", "qubits": [0]}, {"name": "x", "qubits": [1]}],
                        [{"name": "z", "qubits": [0]}, {"name": "y", "qubits": [1]}],
                        [{"name": "z", "qubits": [0]}, {"name": "z", "qubits": [1]}]]
        for j in range(16):
            circ, p = error.error_term(j)
            self.remove_if_found(circ, target_circs)
            if circ == [{"name": "id", "qubits": [0]}]:
                self.assertAlmostEqual(p, 1 - p_depol + p_depol / 16,
                                       msg="Incorrect identity probability")
            else:
                self.assertAlmostEqual(p, p_depol / 16, msg="Incorrect Pauli probability")
        self.assertEqual(target_circs, [], msg="Incorrect unitaries")

    def test_amplitude_damping_error_raises_invalid_amp_param(self):
        """Test phase and amplitude damping error raises for invalid amp_param"""
        self.assertRaises(NoiseError,
                          lambda: phase_amplitude_damping_error(-0.5, 0, 0))
        self.assertRaises(NoiseError,
                          lambda: phase_amplitude_damping_error(1.1, 0, 0))

    def test_amplitude_damping_error_raises_invalid_phase_param(self):
        """Test phase and amplitude damping error raises for invalid amp_param"""
        self.assertRaises(NoiseError,
                          lambda: phase_amplitude_damping_error(0, -0.5, 0))
        self.assertRaises(NoiseError,
                          lambda: phase_amplitude_damping_error(0, 1.1, 0))

    def test_amplitude_damping_error_raises_invalid_excited_state_pop(self):
        """Test phase and amplitude damping error raises for invalid pop"""
        self.assertRaises(NoiseError,
                          lambda: phase_amplitude_damping_error(0, 0, -0.5))
        self.assertRaises(NoiseError,
                          lambda: phase_amplitude_damping_error(0, 0, 1.1))

    def test_amplitude_damping_error_raises_invalid_combined_params(self):
        """Test phase and amplitude damping error raises for invalid pop"""
        self.assertRaises(NoiseError,
                          lambda: phase_amplitude_damping_error(0.5, 0.6, 0))

    def test_phase_amplitude_damping_error_noncanonical(self):
        """Test phase maplitude damping channel has correct number of ops"""
        error = phase_amplitude_damping_error(0.25, 0.5, 0.3, canonical_kraus=False)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="Kraus probability")
        self.assertEqual(circ[0]["qubits"], [0])
        self.assertEqual(len(circ[0]['params']), 6,
                         msg="Incorrect number of kraus matrices")

    def test_phase_amplitude_damping_error_canonical(self):
        """Test phase maplitude damping channel has correct number of ops"""
        error = phase_amplitude_damping_error(0.25, 0.5, 0.3, canonical_kraus=True)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="Kraus probability")
        self.assertEqual(circ[0]["qubits"], [0])
        self.assertEqual(len(circ[0]['params']), 4,
                         msg="Incorrect number of kraus matrices")

    def test_amplitude_damping_error_ideal_noncanonical(self):
        """Test amplitude damping error with param=0 and noncanonical kraus"""
        error = amplitude_damping_error(0, excited_state_population=0.5,
                                        canonical_kraus=False)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="ideal probability")
        self.assertEqual(circ[0], {"name": "id", "qubits": [0]},
                         msg="ideal circuit")

    def test_amplitude_damping_error_full_0state_canonical(self):
        """Test amplitude damping error with param=1 and canonical kraus"""
        error = amplitude_damping_error(1, excited_state_population=0,
                                        canonical_kraus=True)
        targets = [np.diag([1, 0]), np.array([[0, 1], [0, 0]])]
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="Kraus probability")
        self.assertEqual(circ[0]["qubits"], [0])
        for op in circ[0]['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect kraus matrices")

    def test_amplitude_damping_error_full_1state_canonical(self):
        """Test amplitude damping error with param=1 and canonical kraus"""
        error = amplitude_damping_error(1, excited_state_population=1,
                                        canonical_kraus=True)
        targets = [np.diag([0, 1]), np.array([[0, 0], [1, 0]])]
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="Kraus probability")
        self.assertEqual(circ[0]["qubits"], [0])
        for op in circ[0]['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect kraus matrices")

    def test_amplitude_damping_error_full_0state_noncanonical(self):
        """Test amplitude damping error with param=1 and canonical kraus"""
        error = amplitude_damping_error(1, excited_state_population=0,
                                        canonical_kraus=False)
        targets = [np.diag([1, 0]), np.array([[0, 1], [0, 0]])]
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="Kraus probability")
        self.assertEqual(circ[0]["qubits"], [0])
        for op in circ[0]['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect kraus matrices")

    def test_amplitude_damping_error_full_1state_noncanonical(self):
        """Test amplitude damping error with param=1 and canonical kraus"""
        error = amplitude_damping_error(1, excited_state_population=1,
                                        canonical_kraus=False)
        targets = [np.diag([0, 1]), np.array([[0, 0], [1, 0]])]
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="Kraus probability")
        self.assertEqual(circ[0]["qubits"], [0])
        for op in circ[0]['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect kraus matrices")

    def test_phase_damping_error_ideal(self):
        """Test phase damping error with param=0 (ideal)"""
        error = phase_damping_error(0)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="ideal probability")
        self.assertEqual(circ[0], {"name": "id", "qubits": [0]},
                         msg="ideal circuit")

    def test_phase_damping_error_full_canonical(self):
        """Test phase damping error with param=1 and canonical kraus"""
        error = phase_damping_error(1, canonical_kraus=True)
        circ, p = error.error_term(0)
        targets = [np.diag([1, 0]), np.diag([0, 1])]
        self.assertEqual(p, 1, msg="Kraus probability")
        self.assertEqual(circ[0]["qubits"], [0])
        for op in circ[0]['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect kraus matrices")

    def test_phase_damping_error_full_noncanonical(self):
        """Test phase damping error with param=1 and non-canonical kraus"""
        error = phase_damping_error(1, canonical_kraus=False)
        circ, p = error.error_term(0)
        targets = [np.diag([1, 0]), np.diag([0, 1])]
        self.assertEqual(p, 1, msg="Kraus probability")
        self.assertEqual(circ[0]["qubits"], [0])
        for op in circ[0]['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect kraus matrices")

    def test_phase_damping_error_canonical(self):
        """Test phase damping error with canonical kraus"""
        p_phase = 0.3
        error = phase_damping_error(p_phase, canonical_kraus=True)
        # The canonical form of this channel should be a mixed
        # unitary dephasing channel
        targets = [[{'name': 'z', 'qubits': [0]}],
                   [{'name': 'id', 'qubits': [0]}]]
        for j in range(2):
            circ, p = error.error_term(j)
            self.assertEqual(circ[0]["qubits"], [0])
            self.remove_if_found(circ, targets)
        self.assertEqual(targets, [], msg="Incorrect canonical circuits")

    def test_phase_damping_error_noncanonical(self):
        """Test phase damping error with non-canonical kraus"""
        p_phase = 0.3
        error = phase_damping_error(0.3, canonical_kraus=False)
        circ, p = error.error_term(0)
        targets = [np.array([[1, 0], [0, np.sqrt(1 - p_phase)]]),
                   np.array([[0, 0], [0, np.sqrt(p_phase)]])]
        self.assertEqual(p, 1, msg="Kraus probability")
        self.assertEqual(circ[0]["qubits"], [0])
        for op in circ[0]['params']:
            self.remove_if_found(op, targets)
        self.assertEqual(targets, [], msg="Incorrect kraus matrices")

    def test_thermal_relaxation_error_raises_invalid_t2(self):
        """Test raises error for invalid t2 parameters"""
        # T2 == 0
        self.assertRaises(NoiseError, lambda: thermal_relaxation_error(1, 0, 0))
        # T2 < 0
        self.assertRaises(NoiseError, lambda: thermal_relaxation_error(1, -1, 0))

    def test_thermal_relaxation_error_raises_invalid_t1(self):
        """Test raises error for invalid t1 parameters"""
        # T1 == 0
        self.assertRaises(NoiseError, lambda: thermal_relaxation_error(0, 0, 0))
        # T1 < 0
        self.assertRaises(NoiseError, lambda: thermal_relaxation_error(-0.1, 0.1, 0))

    def test_thermal_relaxation_error_raises_invalid_t1_t2(self):
        """Test raises error for invalid t2 > 2 * t1 parameters"""
        # T2 > 2 * T1
        self.assertRaises(NoiseError, lambda: thermal_relaxation_error(1, 2.1, 0))

    def test_thermal_relaxation_error_t1_t2_inf_ideal(self):
        """Test t1 = t2 = inf returns identity channel"""
        error = thermal_relaxation_error(np.inf, np.inf, 0)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="ideal probability")
        self.assertEqual(circ[0], {"name": "id", "qubits": [0]},
                         msg="ideal circuit")

    def test_thermal_relaxation_error_zero_time_ideal(self):
        """Test gate_time = 0 returns identity channel"""
        error = thermal_relaxation_error(2, 3, 0)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1, msg="ideal probability")
        self.assertEqual(circ[0], {"name": "id", "qubits": [0]},
                         msg="ideal circuit")

    def test_thermal_relaxation_error_t1_equal_t2_0state(self):
        """Test qobj instructions return for t1=t2"""
        error = thermal_relaxation_error(1, 1, 1)
        targets = [[{'name': 'id', 'qubits': [0]}],
                   [{'name': 'reset', 'qubits': [0]}]]
        probs = [np.exp(-1), 1 - np.exp(-1)]
        for j in range(2):
            circ, p = error.error_term(j)
            self.remove_if_found(circ, targets)
            if circ[0]['name'] == 'id':
                self.assertAlmostEqual(p, probs[0], msg="identity probability")
            else:
                self.assertAlmostEqual(p, probs[1], msg="reset probability")
        self.assertEqual(targets, [], msg="relaxation circuits")

    def test_thermal_relaxation_error_t1_equal_t2_1state(self):
        """Test qobj instructions return for t1=t2"""
        error = thermal_relaxation_error(1, 1, 1, 1)
        targets = [[{'name': 'id', 'qubits': [0]}],
                   [{'name': 'reset', 'qubits': [0]}, {'name': 'x', 'qubits': [0]}]]
        probs = [np.exp(-1), 1 - np.exp(-1)]
        for j in range(2):
            circ, p = error.error_term(j)
            self.remove_if_found(circ, targets)
            if circ[0]['name'] == 'id':
                self.assertAlmostEqual(p, probs[0], msg="identity probability")
            else:
                self.assertAlmostEqual(p, probs[1], msg="reset probability")
        self.assertEqual(targets, [], msg="relaxation circuits")

    def test_thermal_relaxation_error_gate(self):
        """Test qobj instructions return for t2 < t1"""
        t1, t2, time, p1 = (2, 1, 1, 0.3)
        error = thermal_relaxation_error(t1, t2, time, p1)
        targets = [[{'name': 'id', 'qubits': [0]}],
                   [{'name': 'z', 'qubits': [0]}],
                   [{'name': 'reset', 'qubits': [0]}],
                   [{'name': 'reset', 'qubits': [0]}, {'name': 'x', 'qubits': [0]}]]
        p_reset0 = (1 - p1) * (1 - np.exp(-1 / t1))
        p_reset1 = p1 * (1 - np.exp(-1 / t1))
        p_z = 0.5 * np.exp(-1 / t1) * (1 - np.exp(-(1 / t2 - 1 / t1) * time))
        p_id = 1 - p_z - p_reset0 - p_reset1
        for j in range(4):
            circ, p = error.error_term(j)
            self.remove_if_found(circ, targets)
            name = circ[0]['name']
            if circ[0]['name'] == 'id':
                self.assertAlmostEqual(p, p_id, msg="identity probability")
            elif name == 'z':
                self.assertAlmostEqual(p, p_z, msg="Z error probability")
            elif len(circ) == 1:
                self.assertAlmostEqual(p, p_reset0, msg="reset-0 probability")
            else:
                self.assertAlmostEqual(p, p_reset1, msg="reset-1 probability")
        self.assertEqual(targets, [], msg="relaxation circuits")

    def test_thermal_relaxation_error_kraus(self):
        """Test non-kraus instructions return for t2 < t1"""
        t1, t2, time, p1 = (1, 2, 1, 0.3)
        error = thermal_relaxation_error(t1, t2, time, p1)
        circ, p = error.error_term(0)
        self.assertEqual(p, 1)
        self.assertEqual(circ[0]['name'], 'kraus')
        self.assertEqual(circ[0]['qubits'], [0])


if __name__ == '__main__':
    unittest.main()
