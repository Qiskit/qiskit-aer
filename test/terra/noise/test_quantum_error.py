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
from test.terra.common import QiskitAerTestCase

import unittest

import numpy as np

from qiskit.circuit import QuantumCircuit, Reset, Measure
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators import SuperOp, Kraus, Pauli
from qiskit_aer.noise import QuantumError
from qiskit_aer.noise.noiseerror import NoiseError


class TestQuantumError(QiskitAerTestCase):
    """Testing QuantumError class"""

    def test_empty(self):
        """Test construction with empty noise_ops."""
        with self.assertRaises(TypeError):
            QuantumError()  # pylint: disable=no-value-for-parameter

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
        kraus = Kraus([np.sqrt(0.7) * np.eye(2), np.sqrt(0.3) * np.diag([1, -1])])
        self.assertEqual(QuantumError(kraus).size, 1)
        self.assertEqual(QuantumError([(kraus, 0.7), (kraus, 0.3)]).size, 2)

        self.assertEqual(QuantumError(Pauli("X")).probabilities, [1.0])
        self.assertEqual(QuantumError([(Pauli("I"), 0.7), (Pauli("Z"), 0.3)]).size, 2)

        self.assertEqual(QuantumError([(Pauli("I"), 0.7), (kraus, 0.3)]).size, 2)

    def test_init_with_instructions(self):
        """Test construction with mixture of instructions."""
        self.assertEqual(QuantumError(Reset()).size, 1)
        self.assertEqual(QuantumError([(IGate(), 0.7), (Reset(), 0.3)]).size, 2)
        mixed_insts = QuantumError(
            [
                ((IGate(), [1]), 0.4),
                (ZGate(), 0.3),
                ([(Reset(), [0])], 0.2),
                ([(Reset(), [0]), (XGate(), [0])], 0.1),
            ]
        )
        self.assertEqual(mixed_insts.size, 4)

        self.assertEqual(QuantumError(XGate()).size, 1)
        self.assertEqual(QuantumError([(IGate(), 0.7), (ZGate(), 0.3)]).size, 2)
        mixed_gates = QuantumError([((IGate(), [0]), 0.6), ((XGate(), [1]), 0.4)])
        self.assertEqual(mixed_gates.size, 2)

        mixed_ops = QuantumError([(IGate(), 0.7), (Pauli("Z"), 0.3)])  # Gate  # Operator
        self.assertEqual(mixed_ops.size, 2)
        mixed_ops = QuantumError(
            [(IGate(), 0.7), ((Reset(), [1]), 0.3)]  # Instruction
        )  # Tuple[Instruction, List[int]
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
        a_0 = np.sqrt(0.9) * np.array([[1, 0], [0, np.sqrt(1 - 0.3)]], dtype=complex)
        a_1 = np.sqrt(0.2) * np.array([[0, np.sqrt(0.3)], [0, 0]], dtype=complex)
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
        meas_kraus = Kraus([np.diag([1, 0]), np.diag([0, 1])])
        actual = QuantumError(meas_kraus).to_quantumchannel()
        expected = SuperOp(np.diag([1, 0, 0, 1]))
        self.assertEqual(actual, expected)

    def test_compose(self):
        """Test compose two quantum errors."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_y = QuantumError([((IGate(), [0]), 0.8), ((YGate(), [0]), 0.2)])
        actual = noise_x.compose(noise_y)
        expected = QuantumError(
            [
                ([(IGate(), [0]), (IGate(), [0])], 0.9 * 0.8),
                ([(IGate(), [0]), (YGate(), [0])], 0.9 * 0.2),
                ([(XGate(), [0]), (IGate(), [0])], 0.1 * 0.8),
                ([(XGate(), [0]), (YGate(), [0])], 0.1 * 0.2),
            ]
        )
        self.assertEqual(actual, expected)

    def test_dot(self):
        """Test dot two quantum errors."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_y = QuantumError([((IGate(), [0]), 0.8), ((YGate(), [0]), 0.2)])
        actual = noise_y.dot(noise_x)  # reversed order of compose
        expected = QuantumError(
            [
                ([(IGate(), [0]), (IGate(), [0])], 0.9 * 0.8),
                ([(IGate(), [0]), (YGate(), [0])], 0.9 * 0.2),
                ([(XGate(), [0]), (IGate(), [0])], 0.1 * 0.8),
                ([(XGate(), [0]), (YGate(), [0])], 0.1 * 0.2),
            ]
        )
        self.assertEqual(actual, expected)

    def test_compose_one_with_different_num_qubits(self):
        """Test compose errors with different number of qubits."""
        noise_1q = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_2q = QuantumError([((IGate(), [0]), 0.8), ((XGate(), [1]), 0.2)])
        actual = noise_1q.compose(noise_2q)
        expected = QuantumError(
            [
                ([(IGate(), [0]), (IGate(), [0])], 0.9 * 0.8),
                ([(IGate(), [0]), (XGate(), [1])], 0.9 * 0.2),
                ([(XGate(), [0]), (IGate(), [0])], 0.1 * 0.8),
                ([(XGate(), [0]), (XGate(), [1])], 0.1 * 0.2),
            ]
        )
        self.assertEqual(actual, expected)

    def test_compose_with_different_type_of_operator(self):
        """Test compose with Kraus operator."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        meas_kraus = Kraus([np.diag([1, 0]), np.diag([0, 1])])
        actual = noise_x.compose(meas_kraus)
        expected = QuantumError(
            [
                ([(IGate(), [0]), (meas_kraus.to_instruction(), [0])], 0.9),
                ([(XGate(), [0]), (meas_kraus.to_instruction(), [0])], 0.1),
            ]
        )
        self.assertEqual(actual, expected)

    def test_tensor(self):
        """Test tensor two quantum errors."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_y = QuantumError([((IGate(), [0]), 0.8), ((YGate(), [0]), 0.2)])
        actual = noise_x.tensor(noise_y)
        expected = QuantumError(
            [
                ([(IGate(), [1]), (IGate(), [0])], 0.9 * 0.8),
                ([(IGate(), [1]), (YGate(), [0])], 0.9 * 0.2),
                ([(XGate(), [1]), (IGate(), [0])], 0.1 * 0.8),
                ([(XGate(), [1]), (YGate(), [0])], 0.1 * 0.2),
            ]
        )
        self.assertEqual(actual, expected)

    def test_expand(self):
        """Test tensor two quantum errors."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        noise_y = QuantumError([((IGate(), [0]), 0.8), ((YGate(), [0]), 0.2)])
        actual = noise_y.expand(noise_x)  # reversed order of expand
        expected = QuantumError(
            [
                ([(IGate(), [1]), (IGate(), [0])], 0.9 * 0.8),
                ([(IGate(), [1]), (YGate(), [0])], 0.9 * 0.2),
                ([(XGate(), [1]), (IGate(), [0])], 0.1 * 0.8),
                ([(XGate(), [1]), (YGate(), [0])], 0.1 * 0.2),
            ]
        )
        self.assertEqual(actual, expected)

    def test_tensor_with_different_type_of_operator(self):
        """Test tensor with Kraus operator."""
        noise_x = QuantumError([((IGate(), [0]), 0.9), ((XGate(), [0]), 0.1)])
        meas_kraus = Kraus([np.diag([1, 0]), np.diag([0, 1])])
        actual = noise_x.tensor(meas_kraus)
        expected = QuantumError(
            [
                ([(IGate(), [1]), (meas_kraus.to_instruction(), [0])], 0.9),
                ([(XGate(), [1]), (meas_kraus.to_instruction(), [0])], 0.1),
            ]
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
