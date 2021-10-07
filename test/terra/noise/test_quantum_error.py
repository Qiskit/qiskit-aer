# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021.
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

import ddt
import numpy as np
from qiskit.circuit import QuantumCircuit, Reset, Measure
from qiskit.circuit.library.standard_gates import *
from qiskit.extensions import UnitaryGate
from qiskit.providers.aer.noise import QuantumError
from qiskit.providers.aer.noise.errors.errorutils import standard_gate_unitary
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.quantum_info.operators import SuperOp, Kraus, Pauli
from test.terra import common


class TestQuantumError(common.QiskitAerTestCase):
    """Testing QuantumError class"""

    def test_empty(self):
        """Test construction with empty noise_ops."""
        with self.assertRaises(TypeError):
            QuantumError()

        with self.assertRaises(NoiseError):
            QuantumError([])

    def test_init_with_circuits(self):
        """Test construction with mixture of quantum circuits."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        self.assertEqual(QuantumError(qc).size, 1)
        self.assertEqual(QuantumError([(qc, 0.7), (qc, 0.3)]).size, 2)

    def test_init_with_operators(self):
        """Test construction with mixture of operators."""
        kraus = Kraus([np.sqrt(0.7) * np.eye(2),
                       np.sqrt(0.3) * np.diag([1, -1])])
        self.assertEqual(QuantumError(kraus).size, 1)
        self.assertEqual(QuantumError([(kraus, 0.7), (kraus, 0.3)]).size, 2)

        self.assertEqual(QuantumError(Pauli("X")).probabilities, [1.0])
        self.assertEqual(QuantumError([(Pauli("I"), 0.7), (Pauli("Z"), 0.3)]).size, 2)

        self.assertEqual(QuantumError([(Pauli("I"), 0.7), (kraus, 0.3)]).size, 2)

    def test_init_with_instructions(self):
        """Test construction with mixture of instructions."""
        self.assertEqual(QuantumError(Reset()).size, 1)
        self.assertEqual(QuantumError([(IGate(), 0.7), (Reset(), 0.3)]).size, 2)
        mixed_insts = QuantumError([((IGate(), [1]), 0.4),
                                   (ZGate(), 0.3),
                                   ([(Reset(), [0])], 0.2),
                                   ([(Reset(), [0]), (XGate(), [0])], 0.1)])
        self.assertEqual(mixed_insts.size, 4)

        self.assertEqual(QuantumError(XGate()).size, 1)
        self.assertEqual(QuantumError([(IGate(), 0.7), (ZGate(), 0.3)]).size, 2)
        mixed_gates = QuantumError([((IGate(), [0]), 0.6),
                                    ((XGate(), [1]), 0.4)])
        self.assertEqual(mixed_gates.size, 2)

        mixed_ops = QuantumError([(IGate(), 0.7),          # Gate
                                  (Pauli("Z"), 0.3)])      # Operator
        self.assertEqual(mixed_ops.size, 2)
        mixed_ops = QuantumError([(IGate(), 0.7),          # Instruction
                                  ((Reset(), [1]), 0.3)])  # Tuple[Instruction, List[int]
        self.assertEqual(mixed_ops.size, 2)

    def test_raise_if_invalid_op_type_for_init(self):
        """Test exception is raised when input with invalid type are supplied."""
        with self.assertRaises(NoiseError):
            QuantumError(Measure())  # instruction with clbits

        with self.assertRaises(NoiseError):
            QuantumError([Reset(), XGate()])  # list of instructions expecting default qubits

        with self.assertRaises(NoiseError):
            QuantumError([(Reset(), [0]), XGate()])  # partially supplied

    def test_raise_negative_probabilities(self):
        """Test exception is raised for negative probabilities."""
        noise_ops = [((IGate(), [0]), 1.1), ((XGate(), [0]), -0.1)]
        with self.assertRaises(NoiseError):
            QuantumError(noise_ops)

    def test_raise_unnormalized_probabilities(self):
        """Test exception is raised for probabilities greater than 1."""
        noise_ops = [((IGate(), [0]), 0.9), ((XGate(), [0]), 0.2)]
        with self.assertRaises(NoiseError):
            QuantumError(noise_ops)

    def test_raise_unnormalized_probabilities_unitary_kraus(self):
        """Test exception is raised for unitary kraus probs greater than 1."""
        a_0 = np.sqrt(0.9) * np.eye(2)
        a_1 = np.sqrt(0.2) * np.diag([1, -1])
        with self.assertRaises(NoiseError):
            QuantumError(Kraus([a_0, a_1]))

    def test_raise_unnormalized_probabilities_nonunitary_kraus(self):
        """Test exception is raised for non-unitary kraus probs greater than 1."""
        a_0 = np.sqrt(0.9) * np.array([[1, 0], [0, np.sqrt(1 - 0.3)]],
                                      dtype=complex)
        a_1 = np.sqrt(0.2) * np.array([[0, np.sqrt(0.3)], [0, 0]],
                                      dtype=complex)
        with self.assertRaises(NoiseError):
            QuantumError(Kraus([a_0, a_1]))

    def test_raise_if_construct_with_non_cptp_kraus(self):
        """Test exception is raised for non-CPTP input."""
        a_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.array([[0, 0], [np.sqrt(0.3), 0]], dtype=complex)
        with self.assertRaises(NoiseError):
            QuantumError(Kraus([a_0, a_1]))
        with self.assertRaises(NoiseError):
            QuantumError(Kraus([a_0]))

    def test_raise_if_construct_with_non_multiqubit_kraus(self):
        """Test exception is raised for non-multiqubit input."""
        a_0 = np.sqrt(0.5) * np.diag([1, 1, 1])
        a_1 = np.sqrt(0.5) * np.diag([1, 1, -1])
        with self.assertRaises(NoiseError):
            QuantumError(Kraus([a_0, a_1]))

    def test_ideal(self):
        """Test ideal gates are identified correctly."""
        self.assertTrue(QuantumError(IGate()).ideal())
        self.assertTrue(QuantumError(UnitaryGate(np.eye(2))).ideal())
        self.assertTrue(QuantumError([(IGate(), 0.7), (IGate(), 0.3)]).ideal())

        # up to global phase
        qc = QuantumCircuit(1, global_phase=0.5)
        qc.i(0)
        self.assertTrue(QuantumError(qc).ideal())
        self.assertTrue(QuantumError(UnitaryGate(-1.0 * np.eye(2))).ideal())

    def test_to_quantum_channel(self):
        """Test conversion into quantum channel."""
        meas_kraus = Kraus([np.diag([1, 0]),
                            np.diag([0, 1])])
        actual = QuantumError(meas_kraus).to_quantumchannel()
        expected = SuperOp(np.diag([1, 0, 0, 1]))
        self.assertEqual(actual, expected)

    def test_compose(self):
        """Test compose two quantum errors."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_y = QuantumError([((IGate(), [0]), 0.8), ((YGate(), [0]), 0.2)])
        actual = noise_x.compose(noise_y)
        expected = QuantumError([([(IGate(), [0]), (IGate(), [0])], 0.9 * 0.8),
                                 ([(IGate(), [0]), (YGate(), [0])], 0.9 * 0.2),
                                 ([(XGate(), [0]), (IGate(), [0])], 0.1 * 0.8),
                                 ([(XGate(), [0]), (YGate(), [0])], 0.1 * 0.2)])
        self.assertEqual(actual, expected)

    def test_dot(self):
        """Test dot two quantum errors."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_y = QuantumError([((IGate(), [0]), 0.8), ((YGate(), [0]), 0.2)])
        actual = noise_y.dot(noise_x)  # reversed order of compose
        expected = QuantumError([([(IGate(), [0]), (IGate(), [0])], 0.9 * 0.8),
                                 ([(IGate(), [0]), (YGate(), [0])], 0.9 * 0.2),
                                 ([(XGate(), [0]), (IGate(), [0])], 0.1 * 0.8),
                                 ([(XGate(), [0]), (YGate(), [0])], 0.1 * 0.2)])
        self.assertEqual(actual, expected)

    def test_compose_one_with_different_num_qubits(self):
        """Test compose errors with different number of qubits."""
        noise_1q = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_2q = QuantumError([((IGate(), [0]), 0.8), ((XGate(), [1]), 0.2)])
        actual = noise_1q.compose(noise_2q)
        expected = QuantumError([([(IGate(), [0]), (IGate(), [0])], 0.9 * 0.8),
                                 ([(IGate(), [0]), (XGate(), [1])], 0.9 * 0.2),
                                 ([(XGate(), [0]), (IGate(), [0])], 0.1 * 0.8),
                                 ([(XGate(), [0]), (XGate(), [1])], 0.1 * 0.2)])
        self.assertEqual(actual, expected)

    def test_compose_with_different_type_of_operator(self):
        """Test compose with Kraus operator."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        meas_kraus = Kraus([np.diag([1, 0]),
                            np.diag([0, 1])])
        actual = noise_x.compose(meas_kraus)
        expected = QuantumError([([(IGate(), [0]), (meas_kraus.to_instruction(), [0])], 0.9),
                                 ([(XGate(), [0]), (meas_kraus.to_instruction(), [0])], 0.1)])
        self.assertEqual(actual, expected)

    def test_tensor(self):
        """Test tensor two quantum errors."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_y = QuantumError([((IGate(), [0]), 0.8), ((YGate(), [0]), 0.2)])
        actual = noise_x.tensor(noise_y)
        expected = QuantumError([([(IGate(), [1]), (IGate(), [0])], 0.9 * 0.8),
                                 ([(IGate(), [1]), (YGate(), [0])], 0.9 * 0.2),
                                 ([(XGate(), [1]), (IGate(), [0])], 0.1 * 0.8),
                                 ([(XGate(), [1]), (YGate(), [0])], 0.1 * 0.2)])
        self.assertEqual(actual, expected)

    def test_expand(self):
        """Test tensor two quantum errors."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_y = QuantumError([((IGate(), [0]), 0.8), ((YGate(), [0]), 0.2)])
        actual = noise_y.expand(noise_x)  # reversed order of expand
        expected = QuantumError([([(IGate(), [1]), (IGate(), [0])], 0.9 * 0.8),
                                 ([(IGate(), [1]), (YGate(), [0])], 0.9 * 0.2),
                                 ([(XGate(), [1]), (IGate(), [0])], 0.1 * 0.8),
                                 ([(XGate(), [1]), (YGate(), [0])], 0.1 * 0.2)])
        self.assertEqual(actual, expected)

    def test_tensor_with_different_type_of_operator(self):
        """Test tensor with Kraus operator."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        meas_kraus = Kraus([np.diag([1, 0]),
                            np.diag([0, 1])])
        actual = noise_x.tensor(meas_kraus)
        expected = QuantumError([([(IGate(), [1]), (meas_kraus.to_instruction(), [0])], 0.9),
                                 ([(XGate(), [1]), (meas_kraus.to_instruction(), [0])], 0.1)])
        self.assertEqual(actual, expected)


# ================== Tests for old interfaces ================== #
# TODO: remove after deprecation period
@ddt.ddt
class TestQuantumErrorOldInterface(common.QiskitAerTestCase):
    """Testing the deprecating interface of QuantumError class"""

    def assertKrausWarning(self):
        return self.assertWarns(
            DeprecationWarning,
            msg=r"Constructing QuantumError with list of arrays representing a Kraus channel .* qiskit-aer 0\.10\.0.*",
        )

    def kraus_error(self, param):
        """Return a Kraus class, and the associated QuantumError, constructed by
        passing the list of operators directly, without first wrapping in the
        ``Kraus()`` class."""
        krauses = [
            np.array([[1, 0], [0, np.sqrt(1 - param)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(param)]], dtype=complex)
        ]
        with self.assertKrausWarning():
            # The point here is to test the path where you construct
            # QuantumError using a list of Kraus operators, _without_ wrapping
            # it in the Kraus class first, despite the deprecation.
            error = QuantumError(krauses)
        return Kraus(krauses), error

    def mixed_unitary_error(self, probs, labels, **kwargs):
        """Return a Kraus class with the given unitaries (represented by the
        labels) at the given probabilities, and the same result unitary error list"""
        with self.assertWarns(
            DeprecationWarning,
            msg=r"standard_gate_unitary has been deprecated as of qiskit-aer 0\.10\.0 .*",
        ):
            unitaries = [
                np.sqrt(prob) * standard_gate_unitary(label)
                for prob, label in zip(probs, labels)
            ]
        with self.assertKrausWarning():
            error = QuantumError(unitaries, **kwargs)
        return Kraus(unitaries), error

    def depol_error(self, param, **kwargs):
        """Return depol error unitary list"""
        return self.mixed_unitary_error(
            [1 - param * 0.75, param * 0.25, param * 0.25, param * 0.25],
            ['id', 'x', 'y', 'z'],
            **kwargs,
        )

    @staticmethod
    def aslist(qargs):
        return [q.index for q in qargs]

    @ddt.data(
        ('id', np.eye(2)),
        ('x', np.array([[0, 1], [1, 0]])),
        ('y', np.array([[0, -1j], [1j, 0]])),
        ('z', np.array([[1, 0], [0, -1]])),
        ('h', np.sqrt(0.5) * np.array([[1, 1], [1, -1]])),
        ('s', np.array([[1, 0], [0, 1j]])),
        ('sdg', np.array([[1, 0], [0, -1j]])),
        ('t', np.array([[1, 0], [0, np.sqrt(0.5) * (1 + 1j)]])),
        ('tdg', np.array([[1, 0], [0, np.sqrt(0.5) * (1 - 1j)]])),
        (
            'cx',
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
        ),
        ('cz', np.diag([1, 1, 1, -1])),
        (
            'swap',
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        ),
        (
            'ccx',
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]),
        ),
    )
    @ddt.unpack
    def test_standard_gate_unitary(self, label, matrix):
        """Test standard gates are correct"""
        with self.assertWarns(
            DeprecationWarning,
            msg=r"standard_gate_unitary has been deprecated as of qiskit-aer 0\.10\.0.*",
        ):
            created = standard_gate_unitary(label)
        self.assertLess(np.linalg.norm(created - matrix), 1e-15)

    def test_raise_probabilities_negative(self):
        """Test exception is raised for negative probabilities."""
        noise_ops = [
            ([{"name": "id", "qubits": [0]}], 1.1),
            ([{"name": "x", "qubits": [0]}], -0.1),
        ]
        with self.assertRaises(NoiseError):
            QuantumError(noise_ops)

    def test_raise_probabilities_normalized_qobj(self):
        """Test exception is raised for qobj probabilities greater than 1."""
        noise_ops = [
            ([{"name": "id", "qubits": [0]}], 0.9),
            ([{"name": "x", "qubits": [0]}], 0.2),
        ]
        with self.assertRaises(NoiseError):
            QuantumError(noise_ops)

    def test_raise_probabilities_normalized_unitary_kraus(self):
        """Test exception is raised for unitary kraus probs greater than 1."""
        a_0 = np.sqrt(0.9) * np.eye(2)
        a_1 = np.sqrt(0.2) * np.diag([1, -1])
        with self.assertRaises(NoiseError), self.assertKrausWarning():
                QuantumError([a_0, a_1])

    def test_raise_probabilities_normalized_nonunitary_kraus(self):
        """Test exception is raised for non-unitary kraus probs greater than 1."""
        a_0 = np.sqrt(0.9) * np.array([[1, 0], [0, np.sqrt(1 - 0.3)]])
        a_1 = np.sqrt(0.2) * np.array([[0, np.sqrt(0.3)], [0, 0]])
        with self.assertRaises(NoiseError), self.assertKrausWarning():
            QuantumError([a_0, a_1])

    def test_raise_non_cptp_kraus(self):
        """Test exception is raised for non-CPTP input."""
        a_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.array([[0, 0], [np.sqrt(0.3), 0]], dtype=complex)
        with self.assertRaises(NoiseError), self.assertKrausWarning():
            QuantumError([a_0, a_1])
        with self.assertRaises(NoiseError), self.assertKrausWarning():
            QuantumError([a_0])

    def test_raise_non_multiqubit_kraus(self):
        """Test exception is raised for non-multiqubit input."""
        a_0 = np.sqrt(0.5) * np.diag([1, 1, 1])
        a_1 = np.sqrt(0.5) * np.diag([1, 1, -1])
        with self.assertRaises(NoiseError), self.assertKrausWarning():
            QuantumError([a_0, a_1])

    def test_pauli_conversion_standard_gates(self):
        """Test conversion of Pauli channel kraus to gates"""
        kraus, error = self.depol_error(1, standard_gates=True)
        for j in range(4):
            instr, _ = error.error_term(j)
            self.assertEqual(len(instr), 1)
            self.assertIn(instr[0][0].name, ['x', 'y', 'z', 'id'])
            self.assertEqual(self.aslist(instr[0][1]), [0])
        target = SuperOp(kraus)
        self.assertEqual(target, SuperOp(error))

    def test_tensor_both_kraus(self):
        """Test tensor of two kraus errors"""
        kraus0, error_kraus0 = self.kraus_error(0.3)
        kraus1, error_kraus1 = self.kraus_error(0.5)
        error = error_kraus0.tensor(error_kraus1)
        target = SuperOp(kraus0).tensor(kraus1)

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0][0].name, 'kraus')
        self.assertEqual(self.aslist(circ.qubits), [0, 1])
        self.assertEqual(target, SuperOp(error), msg="Incorrect tensor kraus")

    def test_tensor_both_unitary_standard_gates(self):
        """Test tensor of two unitary standard gate errors"""
        kraus0, error0 = self.mixed_unitary_error(
            [0.9, 0.1], ['z', 's'], standard_gates=True,
        )
        kraus1, error1 = self.mixed_unitary_error(
            [0.6, 0.4], ['x', 'y'], standard_gates=True,
        )
        error = error0.tensor(error1)
        target = SuperOp(kraus0).tensor(kraus1)

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertEqual(len(circ), 2)
            for instr, qargs, _ in circ:
                self.assertIn(instr.name, ['s', 'x', 'y', 'z'])
                self.assertIn(self.aslist(qargs), [[0], [1]])
        self.assertEqual(SuperOp(error), target)

    def test_tensor_kraus_and_unitary(self):
        """Test tensor of a kraus and unitary error."""
        kraus, error_kraus = self.kraus_error(0.4)
        kraus_unitaries, error_unitaries = self.depol_error(0.1)
        error = error_kraus.tensor(error_unitaries)
        target = SuperOp(kraus).tensor(kraus_unitaries)

        circ, prob = error.error_term(0)
        self.assertEqual(self.aslist(circ.qubits), [0, 1])
        self.assertEqual(target, SuperOp(error))

    def test_tensor_unitary_and_kraus(self):
        """Test tensor of a unitary and kraus error."""
        kraus, error_kraus = self.kraus_error(0.4)
        kraus_unitaries, error_unitaries = self.depol_error(0.1)
        error = error_unitaries.tensor(error_kraus)
        target = SuperOp(kraus_unitaries).tensor(kraus)

        circ, prob = error.error_term(0)
        self.assertEqual(self.aslist(circ.qubits), [0, 1])
        self.assertEqual(target, SuperOp(error))

    def test_expand_both_kraus(self):
        """Test expand of two kraus errors"""
        kraus0, error0 = self.kraus_error(0.3)
        kraus1, error1 = self.kraus_error(0.5)
        error = error0.expand(error1)
        target = SuperOp(kraus0).expand(kraus1)

        circ, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(circ[0][0].name, 'kraus')
        self.assertEqual(circ[1][0].name, 'kraus')
        self.assertEqual(self.aslist(circ[0][1]), [0])
        self.assertEqual(self.aslist(circ[1][1]), [1])
        self.assertEqual(target, SuperOp(error))

    def test_expand_both_unitary_standard_gates(self):
        """Test expand of two unitary standard gate errors"""
        kraus0, error0 = self.mixed_unitary_error(
            [0.9, 0.1], ['z', 's'], standard_gates=True,
        )
        kraus1, error1 = self.mixed_unitary_error(
            [0.6, 0.4], ['x', 'y'], standard_gates=True,
        )
        error = error0.expand(error1)
        target = SuperOp(kraus0).expand(kraus1)

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertEqual(len(circ), 2)
            for instr, qargs, _ in circ:
                self.assertIn(instr.name, ['s', 'x', 'y', 'z'])
                self.assertIn(self.aslist(qargs), [[0], [1]])
        self.assertEqual(SuperOp(error), target)

    def test_expand_kraus_and_unitary(self):
        """Test expand of a kraus and unitary error."""
        kraus, error_kraus = self.kraus_error(0.4)
        kraus_unitaries, error_unitaries = self.depol_error(0.1)
        error = error_kraus.expand(error_unitaries)
        target = SuperOp(kraus).expand(kraus_unitaries)

        circ, prob = error.error_term(0)
        # self.assertEqual(prob, 1)
        self.assertEqual(circ[0][0].name, 'kraus')
        self.assertEqual(self.aslist(circ.qubits), [0, 1])
        self.assertEqual(target, SuperOp(error))

    def test_expand_unitary_and_kraus(self):
        """Test expand of a unitary and kraus error."""
        kraus, error_kraus = self.kraus_error(0.4)
        kraus_unitaries, error_unitaries = self.depol_error(0.1)
        error = error_unitaries.expand(error_kraus)
        target = SuperOp(kraus_unitaries).expand(kraus)

        circ, prob = error.error_term(0)
        self.assertEqual(self.aslist(circ.qubits), [0, 1])
        self.assertEqual(target, SuperOp(error))

    def test_compose_both_kraus(self):
        """Test compose of two kraus errors"""
        kraus0, error0 = self.kraus_error(0.3)
        kraus1, error1 = self.kraus_error(0.5)
        error = error0.compose(error1)
        target = SuperOp(kraus0).compose(kraus1)

        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0][0].name, 'kraus')
        self.assertEqual(self.aslist(kraus[0][1]), [0])
        self.assertEqual(target, SuperOp(error), msg="Incorrect tensor kraus")

    def test_compose_both_unitary_standard_gates(self):
        """Test compose of two unitary standard gate errors"""
        kraus0, error0 = self.mixed_unitary_error(
            [0.9, 0.1], ['z', 's'], standard_gates=True,
        )
        kraus1, error1 = self.mixed_unitary_error(
            [0.6, 0.4], ['x', 'y'], standard_gates=True,
        )
        error = error0.compose(error1)
        target = SuperOp(kraus0).compose(kraus1)

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertIn(circ[0][0].name, ['s', 'x', 'y', 'z'])
            self.assertEqual(self.aslist(circ[0][1]), [0])
        self.assertEqual(SuperOp(error), target)

    def test_compose_kraus_and_unitary(self):
        """Test compose of a kraus and unitary error."""
        kraus, error_kraus = self.kraus_error(0.4)
        kraus_unitaries, error_unitaries = self.depol_error(0.1)
        error = error_kraus.compose(error_unitaries)
        target = SuperOp(kraus).compose(kraus_unitaries)

        circ, prob = error.error_term(0)
        # self.assertEqual(prob, 1)
        self.assertEqual(circ[0][0].name, 'kraus')
        self.assertEqual(self.aslist(circ[0][1]), [0])
        self.assertEqual(target, SuperOp(error))

    def test_compose_unitary_and_kraus(self):
        """Test compose of a unitary and kraus error."""
        kraus, error_kraus = self.kraus_error(0.4)
        kraus_unitaries, error_unitaries = self.depol_error(0.1)
        error = error_unitaries.compose(error_kraus)
        target = SuperOp(kraus_unitaries).compose(kraus)

        circ, prob = error.error_term(0)
        self.assertEqual(self.aslist(circ[0][1]), [0])
        self.assertEqual(target, SuperOp(error))

    def test_dot_both_kraus(self):
        """Test dot of two kraus errors"""
        kraus0, error0 = self.kraus_error(0.3)
        kraus1, error1 = self.kraus_error(0.5)
        error = error0.dot(error1)
        target = SuperOp(kraus0).dot(kraus1)

        kraus, prob = error.error_term(0)
        self.assertEqual(prob, 1)
        self.assertEqual(kraus[0][0].name, 'kraus')
        self.assertEqual(self.aslist(kraus[0][1]), [0])
        self.assertEqual(target, SuperOp(error), msg="Incorrect dot kraus")

    def test_dot_both_unitary_standard_gates(self):
        """Test dot of two unitary standard gate errors"""
        kraus0, error0 = self.mixed_unitary_error(
            [0.9, 0.1], ['z', 's'], standard_gates=True,
        )
        kraus1, error1 = self.mixed_unitary_error(
            [0.6, 0.4], ['x', 'y'], standard_gates=True,
        )
        error = error0.dot(error1)
        target = SuperOp(kraus0).dot(kraus1)

        for j in range(4):
            circ, _ = error.error_term(j)
            self.assertIn(circ[0][0].name, ['s', 'x', 'y', 'z'])
            self.assertEqual(self.aslist(circ[0][1]), [0])
        self.assertEqual(SuperOp(error), target)

    def test_dot_kraus_and_unitary(self):
        """Test dot of a kraus and unitary error."""
        kraus, error_kraus = self.kraus_error(0.4)
        kraus_unitaries, error_unitaries = self.depol_error(0.1)
        error = error_kraus.dot(error_unitaries)
        target = SuperOp(kraus).dot(kraus_unitaries)

        circ, prob = error.error_term(0)
        self.assertEqual(self.aslist(circ[0][1]), [0])
        self.assertEqual(target, SuperOp(error))

    def test_dot_unitary_and_kraus(self):
        """Test dot of a unitary and kraus error."""
        kraus, error_kraus = self.kraus_error(0.4)
        kraus_unitaries, error_unitaries = self.depol_error(0.1)
        error = error_unitaries.dot(error_kraus)
        target = SuperOp(kraus_unitaries).dot(kraus)

        circ, prob = error.error_term(0)
        # self.assertEqual(prob, 1)
        self.assertEqual(circ[0][0].name, 'kraus')
        self.assertEqual(self.aslist(circ[0][1]), [0])
        self.assertEqual(target, SuperOp(error))

    def test_to_quantumchannel_kraus(self):
        """Test to_quantumchannel for Kraus inputs."""
        a_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.array([[0, 0], [0, np.sqrt(0.3)]], dtype=complex)
        b_0 = np.array([[1, 0], [0, np.sqrt(1 - 0.5)]], dtype=complex)
        b_1 = np.array([[0, 0], [0, np.sqrt(0.5)]], dtype=complex)
        target = SuperOp(Kraus([a_0, a_1])).tensor(SuperOp(Kraus([b_0, b_1])))
        with self.assertWarns(
            DeprecationWarning,
            msg=r"Constructing QuantumError .* Kraus channel .* qiskit-aer 0\.10\.0 .*",
        ):
            error = QuantumError([a_0, a_1]).tensor(QuantumError([b_0, b_1]))
        self.assertEqual(target, error.to_quantumchannel())

    def test_to_quantumchannel_circuit(self):
        """Test to_quantumchannel for circuit inputs."""
        noise_ops = [
            ([{'name': 'reset', 'qubits': [0]}], 0.2),
            ([{'name': 'reset', 'qubits': [1]}], 0.3),
            ([{'name': 'id', 'qubits': [0]}], 0.5),
        ]
        with self.assertWarns(
            DeprecationWarning,
            msg=r"Constructing QuantumError .* list of dict .* qiskit-aer 0\.10\.0 .*",
        ):
            error = QuantumError(noise_ops)
        reset = SuperOp(
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        )
        iden = SuperOp(np.eye(4))
        target = (
            0.2 * iden.tensor(reset)
            + 0.3 * reset.tensor(iden)
            + 0.5 * iden.tensor(iden)
        )
        self.assertEqual(target, error.to_quantumchannel())

    def test_equal(self):
        """Test two quantum errors are equal"""
        with self.assertWarns(
            DeprecationWarning,
            msg=r"standard_gate_unitary is deprecated as of qiskit-aer 0\.10\.0.*",
        ):
            a_i = np.sqrt(0.25) * standard_gate_unitary('id')
            a_x = np.sqrt(0.25) * standard_gate_unitary('x')
            a_y = np.sqrt(0.25) * standard_gate_unitary('y')
            a_z = np.sqrt(0.25) * standard_gate_unitary('z')
        with self.assertKrausWarning():
            error1 = QuantumError([a_i, a_x, a_y, a_z], standard_gates=True)
            error2 = QuantumError([a_i, a_x, a_y, a_z], standard_gates=False)
        self.assertEqual(error1, error2)


if __name__ == '__main__':
    unittest.main()
