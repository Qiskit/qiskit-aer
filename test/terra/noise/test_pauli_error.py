# This code is part of Qiskit.
#
# (C) Copyright IBM 2018-2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
PauliError class tests
"""
from test.terra.common import QiskitAerTestCase

import unittest
from itertools import combinations

import ddt
import numpy as np

import qiskit.quantum_info as qi
from qiskit_aer.noise import PauliError, QuantumError
from qiskit_aer.noise.noiseerror import NoiseError


@ddt.ddt
class TestPauliError(QiskitAerTestCase):
    """Testing QuantumError class"""

    @ddt.data(str, qi.Pauli, qi.PauliList)
    def test_init_with_paulis(self, cls):
        """Test construction with different inputs."""
        paulis = ["I", "X"]
        if cls == qi.Pauli:
            paulis = [qi.Pauli(i) for i in paulis]
        elif cls == qi.PauliList:
            paulis = qi.PauliList(paulis)
        perr = PauliError(paulis, np.ones(len(paulis)) / len(paulis))
        self.assertEqual(perr.size, 2)

    def test_invalid_inputs(self):
        """Test inputs with different lengths raises"""
        with self.assertRaises(NoiseError):
            PauliError(["X", "I"], [1])
        with self.assertRaises(NoiseError):
            PauliError(["X", "I"], np.eye(2))
        with self.assertRaises(NoiseError):
            PauliError(["X"], 1)

    def test_quasi_dist_probabilities(self):
        """Test initalizing with qasui-dist probabilities."""
        perr = PauliError(["I", "X"], [1.1, -0.1])
        self.assertTrue(perr.is_tp())
        self.assertFalse(perr.is_cp())
        self.assertFalse(perr.is_cptp())
        with self.assertRaises(NoiseError):
            perr.to_quantum_error()

    def test_non_normalized_probabilities(self):
        """Test initalizing with qasui-dist probabilities."""
        perr = PauliError(["I", "X"], [1.1, 0.1])
        self.assertFalse(perr.is_tp())
        self.assertTrue(perr.is_cp())
        self.assertFalse(perr.is_cptp())
        with self.assertRaises(NoiseError):
            perr.to_quantum_error()

    def test_attributes(self):
        """Test basic attributes"""
        rng = np.random.default_rng(555)
        paulis = qi.random_pauli_list(3, 8, phase=False, seed=rng)
        probs = rng.random(len(paulis))
        probs /= sum(probs)
        perr = PauliError(paulis, probs)
        self.assertEqual(perr.size, len(paulis))
        self.assertEqual(perr.paulis, paulis)
        np.testing.assert_allclose(perr.probabilities, probs)

    @ddt.data(1, 10, 100, 1000)
    def test_ideal_single_iden(self, num_qubits):
        """Test ideal gates are identified correctly."""
        self.assertTrue(PauliError([num_qubits * "I"], [1.0]).ideal())
        self.assertFalse(PauliError([num_qubits * "I"], [0.9]).ideal())  # non-TP
        self.assertFalse(PauliError([num_qubits * "I"], [-1]).ideal())  # non-CP

    @ddt.data(1, 10, 100, 1000)
    def test_ideal_single_zero_probs(self, num_qubits):
        """Test ideal gates are identified correctly."""
        paulis = [num_qubits * s for s in ["I", "X", "Y", "Z"]]
        probs = [1, 0, 0, 0]
        self.assertTrue(PauliError(paulis, probs).ideal())

    @ddt.data(1, 10, 100, 1000)
    def test_ideal_single_multi_iden(self, num_qubits):
        """Test ideal gates are identified correctly."""
        self.assertTrue(PauliError(4 * [num_qubits * "I"], 4 * [0.25]).ideal())

    def test_to_quantumchannel(self):
        """Test conversion into quantum channel."""
        rng = np.random.default_rng(1234)
        paulis = qi.random_pauli_list(3, 8, phase=False, seed=rng)
        probs = rng.random(len(paulis))
        probs /= sum(probs)
        target = sum((prob * qi.SuperOp(op) for op, prob in zip(paulis, probs)))
        perr = PauliError(paulis, probs)
        self.assertEqual(perr.to_quantumchannel(), target)

    def test_to_quantum_error(self):
        """Test conversion into quantum error."""
        rng = np.random.default_rng(4444)
        paulis = qi.random_pauli_list(4, 5, phase=False, seed=rng)
        probs = rng.random(len(paulis))
        probs /= sum(probs)
        target = QuantumError(zip(paulis, probs))
        perr = PauliError(paulis, probs)
        self.assertEqual(perr.to_quantum_error(), target)

    def test_dict_round_trip(self):
        """Test to_dict and from_dict round trip."""
        rng = np.random.default_rng(55)
        paulis = qi.random_pauli_list(5, 6, phase=False, seed=rng)
        probs = rng.random(len(paulis))
        probs /= sum(probs)
        target = PauliError(paulis, probs)
        value = PauliError.from_dict(target.to_dict())
        self.assertDictAlmostEqual(value, target)

    def test_quantum_error_dict_from_dict_equiv(self):
        """Test PauliError.to_dict -> QuantumError.from_dict."""
        rng = np.random.default_rng(33)
        paulis = qi.random_pauli_list(4, 6, phase=False, seed=rng)
        probs = rng.random(len(paulis))
        probs /= sum(probs)
        target = QuantumError(zip(paulis, probs))
        perr = PauliError(paulis, probs)
        value = QuantumError.from_dict(perr.to_dict())
        self.assertEqual(value, target)

    def test_quantum_error_dict_to_dict_equiv(self):
        """Test QuantumError.to_dict -> PauliError.from_dict."""
        rng = np.random.default_rng(33)
        paulis = qi.random_pauli_list(4, 6, phase=False, seed=rng)
        probs = rng.random(len(paulis))
        probs /= sum(probs)
        target = PauliError(paulis, probs)
        qerr = QuantumError(zip(paulis, probs))
        value = PauliError.from_dict(qerr.to_dict())
        self.assertEqual(value, target)

    def test_simplify_zero(self):
        """Test simplify removes zeros"""
        paulis = ["XI", "IX", "YI", "IY"]
        p1 = 1e-1
        p2 = 1e-5
        p3 = 1e-9
        probs = [1 - p1 - p2 - p3, p1, p2, p3]
        perr = PauliError(paulis, probs)

        value1 = perr.simplify()
        target1 = PauliError(paulis[:3], probs[:3])
        self.assertEqual(value1, target1)

        value2 = perr.simplify(atol=1e-4)
        target2 = PauliError(paulis[:2], probs[:2])
        self.assertEqual(value2, target2)

        value3 = perr.simplify(atol=0)
        target3 = perr
        self.assertEqual(value3, target3)

    def test_simplify_duplicates(self):
        """Test simplify combines duplicate terms"""
        paulis = ["XX", "ZZ"]
        probs = [0.8, 0.2]
        target = PauliError(paulis, probs)
        value = PauliError(4 * paulis, np.array(4 * probs) / 4).simplify()
        self.assertEqual(value, target)

    def test_equal_diff_order(self):
        rng = np.random.default_rng(33)
        paulis = qi.random_pauli_list(5, 10, phase=False, seed=rng)
        probs = rng.random(len(paulis))
        probs /= sum(probs)
        perr1 = PauliError(paulis, probs)
        perr2 = PauliError(paulis[::-1], probs[::-1])
        self.assertEqual(perr1, perr2)

    def test_not_equal_type(self):
        perr = PauliError(["II", "XX"], [0.9, 0.1])
        qerr = perr.to_quantum_error()
        chan = perr.to_quantumchannel()
        self.assertNotEqual(perr, qerr)
        self.assertNotEqual(perr, chan)

    def test_not_equal_shape(self):
        perr1 = PauliError(["II", "XX"], [0.9, 0.1])
        perr2 = PauliError(["II", "XX", "ZZ"], [0.9, 0.05, 0.05])
        self.assertNotEqual(perr1, perr2)

    @ddt.data(1, 2, 3, 4)
    def test_compose(self, qarg_qubits):
        """Test compose two quantum errors."""
        rng = np.random.default_rng(33)
        paulis1 = qi.random_pauli_list(4, 3, phase=False, seed=rng)
        probs1 = rng.random(len(paulis1))
        probs1 /= sum(probs1)
        perr1 = PauliError(paulis1, probs1)
        paulis2 = qi.random_pauli_list(qarg_qubits, 3, phase=False, seed=rng)
        probs2 = rng.random(len(paulis2))
        probs2 /= sum(probs2)
        perr2 = PauliError(paulis2, probs2)
        target = perr1.to_quantumchannel().compose(perr2.to_quantumchannel(), range(qarg_qubits))
        value = (perr1.compose(perr2, range(qarg_qubits))).to_quantumchannel()
        self.assertEqual(value, target)

    @ddt.idata(list(combinations(range(4), 1)) + list(combinations(range(4), 2)))
    def test_compose_subsystem(self, qargs):
        """Test compose with 1 and 2-qubit subsystem permutations"""
        rng = np.random.default_rng(123)
        paulis1 = qi.random_pauli_list(4, 3, phase=False, seed=rng)
        probs1 = rng.random(len(paulis1))
        probs1 /= sum(probs1)
        perr1 = PauliError(paulis1, probs1)
        paulis2 = qi.random_pauli_list(len(qargs), 3, phase=False, seed=rng)
        probs2 = rng.random(len(paulis2))
        probs2 /= sum(probs2)
        perr2 = PauliError(paulis2, probs2)
        target = perr1.to_quantumchannel().compose(perr2.to_quantumchannel(), qargs)
        value = (perr1.compose(perr2, qargs)).to_quantumchannel()
        self.assertEqual(value, target)

    @ddt.data(1, 10, 100)
    def test_dot_equals_compose(self, num_qubits):
        """Test dot is equal to compose."""
        rng = np.random.default_rng(999)

        for _ in range(20):
            paulis1 = qi.random_pauli_list(num_qubits, 3, phase=False, seed=rng)
            coeffs1 = rng.random(len(paulis1))
            perr1 = PauliError(paulis1, coeffs1)
            paulis2 = qi.random_pauli_list(num_qubits, 5, phase=False, seed=rng)
            coeffs2 = rng.random(len(paulis2))
            perr2 = PauliError(paulis2, coeffs2)
            self.assertEqual(perr1.dot(perr2), perr1.compose(perr2))

    def test_tensor(self):
        """Test tensor two quantum errors."""
        rng = np.random.default_rng(99)
        paulis1 = qi.random_pauli_list(2, 4, phase=False, seed=rng)
        probs1 = rng.random(len(paulis1))
        probs1 /= sum(probs1)
        perr1 = PauliError(paulis1, probs1)
        paulis2 = qi.random_pauli_list(2, 3, phase=False, seed=rng)
        probs2 = rng.random(len(paulis2))
        probs2 /= sum(probs2)
        perr2 = PauliError(paulis2, probs2)
        value = perr1.tensor(perr2).to_quantumchannel()
        target = perr1.to_quantumchannel().tensor(perr2.to_quantumchannel())
        self.assertEqual(value, target)

    def test_expand(self):
        """Test tensor two quantum errors."""
        rng = np.random.default_rng(99)
        paulis1 = qi.random_pauli_list(2, 4, phase=False, seed=rng)
        probs1 = rng.random(len(paulis1))
        probs1 /= sum(probs1)
        perr1 = PauliError(paulis1, probs1)
        paulis2 = qi.random_pauli_list(2, 3, phase=False, seed=rng)
        probs2 = rng.random(len(paulis2))
        probs2 /= sum(probs2)
        perr2 = PauliError(paulis2, probs2)
        value = perr1.expand(perr2).to_quantumchannel()
        target = perr1.to_quantumchannel().expand(perr2.to_quantumchannel())
        self.assertEqual(value, target)


if __name__ == "__main__":
    unittest.main()
