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
PauliLindbladError class tests
"""
from test.terra.common import QiskitAerTestCase

import unittest
from itertools import combinations, product

import ddt
import numpy as np
import scipy.linalg as la

import qiskit.quantum_info as qi
from qiskit_aer.noise import PauliLindbladError, PauliError, QuantumError
from qiskit_aer.noise.noiseerror import NoiseError


@ddt.ddt
class TestPauliLindbladError(QiskitAerTestCase):
    """Testing PauliLindbladError class"""

    @ddt.data(str, qi.Pauli, qi.PauliList)
    def test_init_with_paulis(self, cls):
        """Test construction with different inputs."""
        gens = ["I", "X"]
        if cls == qi.Pauli:
            gens = [qi.Pauli(i) for i in gens]
        elif cls == qi.PauliList:
            gens = qi.PauliList(gens)
        perr = PauliLindbladError(gens, np.ones(len(gens)) / len(gens))
        self.assertEqual(perr.size, 2)

    def test_invalid_inputs(self):
        """Test inputs with different lengths raises"""
        with self.assertRaises(NoiseError):
            PauliLindbladError(["X", "Z"], [1])
        with self.assertRaises(NoiseError):
            PauliLindbladError(["X", "Z"], np.eye(2))
        with self.assertRaises(NoiseError):
            PauliLindbladError(["X"], 1)

    def test_negative_rates(self):
        """Test initalizing with negative rate."""
        perr = PauliLindbladError(["X", "Z"], [0.1, -0.1])
        self.assertTrue(perr.is_tp())
        self.assertFalse(perr.is_cp())
        self.assertFalse(perr.is_cptp())
        with self.assertRaises(NoiseError):
            perr.to_quantum_error()

    def test_attributes(self):
        """Test basic attributes"""
        rng = np.random.default_rng(555)
        gens = qi.random_pauli_list(3, 8, phase=False, seed=rng)
        rates = rng.random(len(gens))
        perr = PauliLindbladError(gens, rates)
        self.assertEqual(perr.size, len(gens))
        self.assertEqual(perr.generators, gens)
        np.testing.assert_allclose(perr.rates, rates)

    @ddt.data(1, 10, 100, 1000)
    def test_ideal_single_iden(self, num_qubits):
        """Test ideal gates are identified correctly."""
        self.assertTrue(PauliLindbladError([num_qubits * "I"], [0.1]).ideal())
        self.assertTrue(PauliLindbladError([num_qubits * "I"], [-0.1]).ideal())  # non-CP

    @ddt.data(1, 10, 100, 1000)
    def test_ideal_single_zero_rates(self, num_qubits):
        """Test ideal gates are identified correctly."""
        gens = [num_qubits * s for s in ["X", "Y", "Z"]]
        rates = [0, 0, 0]
        self.assertTrue(PauliLindbladError(gens, rates).ideal())

    @ddt.data(1, 2, 3)
    def test_to_quantum_channel_iden(self, num_qubits):
        """Test ideal gates are identified correctly."""
        gens = [num_qubits * s for s in ["X", "Y", "Z"]]
        rates = [0, 0, 0]
        plerr = PauliLindbladError(gens, rates)
        chan = plerr.to_quantumchannel()
        self.assertEqual(chan, qi.SuperOp(np.eye(4**num_qubits)))

    @ddt.idata(product(["pauli_error", "quantum_error", "quantumchannel"], [0, 0.1, 0.25, 0.5]))
    @ddt.unpack
    def test_conversion_1q_single(self, output, prob):
        """Test conversion into quantum channel."""
        if prob == 0.5:
            rate = np.inf
        else:
            rate = -0.5 * np.log(1 - 2 * prob)
        plerr = PauliLindbladError(["Y"], [rate])
        target = PauliError(["I", "Y"], [1 - prob, prob])
        if output == "pauli_error":
            value = plerr.to_pauli_error()
        elif output == "quantum_error":
            value = plerr.to_quantum_error()
            target = target.to_quantum_error()
        elif output == "quantumchannel":
            value = plerr.to_quantumchannel()
            target = target.to_quantumchannel()
        self.assertEqual(value, target)

    @ddt.idata(product(["pauli_error", "quantum_error", "quantumchannel"], [0, 0.1, 0.25, 0.5]))
    @ddt.unpack
    def test_conversion_1q_multi(self, output, prob):
        """Test conversion into quantum channel."""
        if prob == 0.5:
            rate = np.inf
        else:
            rate = -0.5 * np.log(1 - 2 * prob)
        plerr = PauliLindbladError(["X", "Y", "Z"], 3 * [rate])
        plerr = plerr.simplify()
        target = PauliError(["I"], [1])
        for pauli in ["X", "Y", "Z"]:
            target = target.compose(PauliError(["I", pauli], [1 - prob, prob]))
        target = target.simplify()
        if output == "pauli_error":
            value = plerr.to_pauli_error()
        elif output == "quantum_error":
            value = plerr.to_quantum_error()
            target = target.to_quantum_error()
        elif output == "quantumchannel":
            value = plerr.to_quantumchannel()
            target = target.to_quantumchannel()
        self.assertEqual(value, target)

    @ddt.data(1, 2, 3, 4)
    def test_to_quantumchannel_rand(self, num_qubits):
        """Test conversion into quantum channel."""
        rng = np.random.default_rng(55)
        gens = qi.random_pauli_list(num_qubits, 3 * num_qubits, phase=False, seed=rng)
        rates = rng.random(len(gens))
        plerr = PauliLindbladError(gens, rates)
        target = plerr.to_pauli_error().to_quantumchannel()
        self.assertEqual(plerr.to_quantumchannel(), target)

    def test_dict_round_trip(self):
        """Test to_dict and from_dict round trip."""
        rng = np.random.default_rng(55)
        gens = qi.random_pauli_list(5, 6, phase=False, seed=rng)
        rates = rng.random(len(gens))
        target = PauliLindbladError(gens, rates)
        value = PauliLindbladError.from_dict(target.to_dict())
        self.assertDictAlmostEqual(value, target)

    def test_simplify_zero_rates(self):
        """Test simplify removes zero rates"""
        gens = ["XI", "IX", "YI", "IY"]
        rates = [1e-1, 1e-2, 1e-5, 1e-9]
        perr = PauliLindbladError(gens, rates)

        value1 = perr.simplify()
        target1 = PauliLindbladError(gens[:3], rates[:3])
        self.assertEqual(value1, target1)

        value2 = perr.simplify(atol=1e-4)
        target2 = PauliLindbladError(gens[:2], rates[:2])
        self.assertEqual(value2, target2)

        value3 = perr.simplify(atol=0)
        target3 = perr
        self.assertEqual(value3, target3)

    def test_simplify_duplicates(self):
        """Test simplify combines duplicate terms"""
        gens = ["XX", "ZZ"]
        rates = [0.1, 0.2]
        target = PauliLindbladError(gens, rates)
        value = PauliLindbladError(4 * gens, np.array(4 * rates) / 4).simplify()
        self.assertEqual(value, target)

    def test_equal_diff_order(self):
        rng = np.random.default_rng(33)
        gens = qi.random_pauli_list(5, 10, phase=False, seed=rng)
        rates = rng.random(len(gens))
        perr1 = PauliLindbladError(gens, rates)
        perr2 = PauliLindbladError(gens[::-1], rates[::-1])
        self.assertEqual(perr1, perr2)

    def test_not_equal_type(self):
        perr = PauliLindbladError(["ZZ", "XX"], [0.2, 0.1])
        qerr = perr.to_quantum_error()
        chan = perr.to_quantumchannel()
        self.assertNotEqual(perr, qerr)
        self.assertNotEqual(perr, chan)

    def test_not_equal_shape(self):
        perr1 = PauliLindbladError(["YY", "XX"], [0.9, 0.1])
        perr2 = PauliLindbladError(["YY", "XX", "ZZ"], [0.9, 0.05, 0.05])
        self.assertNotEqual(perr1, perr2)

    @ddt.data(1, 2, 3, 4)
    def test_compose(self, qarg_qubits):
        """Test compose two quantum errors."""
        rng = np.random.default_rng(33)
        gens1 = qi.random_pauli_list(4, 3, phase=False, seed=rng)
        rates1 = rng.random(len(gens1))
        perr1 = PauliLindbladError(gens1, rates1)
        gens2 = qi.random_pauli_list(qarg_qubits, 3, phase=False, seed=rng)
        rates2 = rng.random(len(gens2))
        perr2 = PauliLindbladError(gens2, rates2)
        target = perr1.to_quantumchannel().compose(perr2.to_quantumchannel(), range(qarg_qubits))
        value = (perr1.compose(perr2, range(qarg_qubits))).to_quantumchannel()
        self.assertEqual(value, target)

    @ddt.idata(list(combinations(range(4), 1)) + list(combinations(range(4), 2)))
    def test_compose_subsystem(self, qargs):
        """Test compose with 1 and 2-qubit subsystem permutations"""
        rng = np.random.default_rng(123)
        gens1 = qi.random_pauli_list(4, 3, phase=False, seed=rng)
        rates1 = rng.random(len(gens1))
        perr1 = PauliLindbladError(gens1, rates1)
        gens2 = qi.random_pauli_list(len(qargs), 3, phase=False, seed=rng)
        rates2 = rng.random(len(gens2))
        perr2 = PauliLindbladError(gens2, rates2)
        target = perr1.to_quantumchannel().compose(perr2.to_quantumchannel(), qargs)
        value = (perr1.compose(perr2, qargs)).to_quantumchannel()
        self.assertEqual(value, target)

    @ddt.data(1, 10, 100)
    def test_dot_equals_compose(self, num_qubits):
        """Test dot is equal to compose."""
        rng = np.random.default_rng(999)

        for _ in range(20):
            gens1 = qi.random_pauli_list(num_qubits, 3, phase=False, seed=rng)
            coeffs1 = rng.random(len(gens1))
            perr1 = PauliLindbladError(gens1, coeffs1)
            gens2 = qi.random_pauli_list(num_qubits, 5, phase=False, seed=rng)
            coeffs2 = rng.random(len(gens2))
            perr2 = PauliLindbladError(gens2, coeffs2)
            self.assertEqual(perr1.dot(perr2), perr1.compose(perr2))

    def test_tensor(self):
        """Test tensor two quantum errors."""
        rng = np.random.default_rng(99)
        gens1 = qi.random_pauli_list(2, 4, phase=False, seed=rng)
        rates1 = rng.random(len(gens1))
        rates1 /= sum(rates1)
        perr1 = PauliLindbladError(gens1, rates1)
        gens2 = qi.random_pauli_list(2, 3, phase=False, seed=rng)
        rates2 = rng.random(len(gens2))
        rates2 /= sum(rates2)
        perr2 = PauliLindbladError(gens2, rates2)
        value = perr1.tensor(perr2).to_quantumchannel()
        target = perr1.to_quantumchannel().tensor(perr2.to_quantumchannel())
        self.assertEqual(value, target)

    def test_expand(self):
        """Test tensor two quantum errors."""
        rng = np.random.default_rng(99)
        gens1 = qi.random_pauli_list(2, 4, phase=False, seed=rng)
        rates1 = rng.random(len(gens1))
        perr1 = PauliLindbladError(gens1, rates1)
        gens2 = qi.random_pauli_list(2, 3, phase=False, seed=rng)
        rates2 = rng.random(len(gens2))
        perr2 = PauliLindbladError(gens2, rates2)
        value = perr1.expand(perr2).to_quantumchannel()
        target = perr1.to_quantumchannel().expand(perr2.to_quantumchannel())
        self.assertEqual(value, target)

    @ddt.data(0, 1, -1, 0.5, 2)
    def test_power(self, exponent):
        """Test power method"""
        rng = np.random.default_rng(seed=1234)
        gens = qi.random_pauli_list(3, 5, phase=False)
        rates = rng.random(5)
        plerr = PauliLindbladError(gens, rates)
        plpow = plerr.power(exponent)
        supmat = qi.SuperOp(plerr).data
        if exponent == 0.5:
            suppow = la.sqrtm(supmat)
        elif exponent == -1:
            suppow = la.inv(supmat)
        else:
            suppow = la.fractional_matrix_power(supmat, exponent)
        supop_pow = qi.SuperOp(suppow)
        self.assertEqual(plpow, PauliLindbladError(gens, exponent * rates))
        self.assertEqual(plpow.to_quantumchannel(), supop_pow)

    def test_inverse(self):
        """Test inverse"""
        rng = np.random.default_rng(seed=1234)
        gens = qi.random_pauli_list(3, 5, phase=False)
        rates = rng.random(5)
        plerr = PauliLindbladError(gens, rates)
        inv = plerr.inverse()
        supop = qi.SuperOp(plerr)
        supop_inv = qi.SuperOp(np.linalg.inv(supop.data))
        self.assertEqual(inv, PauliLindbladError(gens, -rates))
        self.assertEqual(inv.to_quantumchannel(), supop_inv)

    def test_subsystem_errors(self):
        """Test subsystem errors method"""
        rng = np.random.default_rng(seed=1234)
        gens = qi.PauliList(["XX", "YY", "ZZ"])
        plerr1 = PauliLindbladError(gens, rng.random(3))
        plerr2 = PauliLindbladError(gens, rng.random(3))
        plerr3 = PauliLindbladError(gens, rng.random(3))
        plerr4 = PauliLindbladError(gens, rng.random(3))

        error = plerr1.expand(PauliLindbladError(["III"], [0]))
        error = error.compose(plerr2, [1, 2])
        error = error.compose(plerr3, [2, 3])
        error = error.compose(plerr4, [3, 4])

        targets = {
            (0, 1): plerr1,
            (1, 2): plerr2,
            (2, 3): plerr3,
            (3, 4): plerr4,
        }
        sub_errors = error.subsystem_errors()
        self.assertEqual(len(sub_errors), 4)
        for suberr, subsys in sub_errors:
            self.assertIn(subsys, targets)
            self.assertEqual(suberr, targets[subsys])


if __name__ == "__main__":
    unittest.main()
