# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest


from qiskit_aer.library import SaveExpectationValue, SaveExpectationValueVariance
from qiskit.quantum_info.operators import Pauli

from ..common import QiskitAerTestCase


class TestSaveExpectationValue(QiskitAerTestCase):
    """SaveExpectationValue instruction tests"""

    def test_invalid_key_raises(self):
        """Test save instruction key is str"""
        self.assertRaises(TypeError, lambda: SaveExpectationValue(Pauli("Z"), 1))

    def test_nonhermitian_raises(self):
        """Test non-Hermitian op raises exception."""
        op = [[0, 1j], [1j, 0]]
        self.assertRaises(ValueError, lambda: SaveExpectationValue(op, "expval"))

    def test_default_kwarg(self):
        """Test default kwargs"""
        key = "test_key"
        instr = SaveExpectationValue(Pauli("X"), key)
        self.assertEqual(instr.name, "save_expval")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "average")

    def test_cond_kwarg(self):
        """Test conditional kwarg"""
        key = "test_key"
        instr = SaveExpectationValue(Pauli("X"), key, conditional=True)
        self.assertEqual(instr.name, "save_expval")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "c_average")

    def test_unnorm_kwarg(self):
        """Test unnormalized kwarg"""
        key = "test_key"
        instr = SaveExpectationValue(Pauli("X"), key, unnormalized=True)
        self.assertEqual(instr.name, "save_expval")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "accum")

    def test_unnorm_cond_kwarg(self):
        """Test unnormalized, conditonal kwargs"""
        key = "test_key"
        instr = SaveExpectationValue(Pauli("X"), key, conditional=True, unnormalized=True)
        self.assertEqual(instr.name, "save_expval")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "c_accum")

    def test_pershot_kwarg(self):
        """Test pershot kwarg"""
        key = "test_key"
        instr = SaveExpectationValue(Pauli("X"), key, pershot=True)
        self.assertEqual(instr.name, "save_expval")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "list")

    def test_pershot_cond_kwarg(self):
        """Test pershot, conditonal kwargs"""
        key = "test_key"
        instr = SaveExpectationValue(Pauli("X"), key, conditional=True, pershot=True)
        self.assertEqual(instr.name, "save_expval")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "c_list")


class TestSaveExpectationValueVariance(QiskitAerTestCase):
    """SaveExpectationValue instruction tests"""

    def test_invalid_key_raises(self):
        """Test save instruction key is str"""
        self.assertRaises(TypeError, lambda: SaveExpectationValueVariance(Pauli("Z"), 1))

    def test_nonhermitian_raises(self):
        """Test non-Hermitian op raises exception."""
        op = [[0, 1j], [1j, 0]]
        self.assertRaises(ValueError, lambda: SaveExpectationValueVariance(op, "expval"))

    def test_default_kwarg(self):
        """Test default kwargs"""
        key = "test_key"
        instr = SaveExpectationValueVariance(Pauli("X"), key)
        self.assertEqual(instr.name, "save_expval_var")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "average")

    def test_cond_kwarg(self):
        """Test conditional kwarg"""
        key = "test_key"
        instr = SaveExpectationValueVariance(Pauli("X"), key, conditional=True)
        self.assertEqual(instr.name, "save_expval_var")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "c_average")

    def test_unnorm_kwarg(self):
        """Test unnormalized kwarg"""
        key = "test_key"
        instr = SaveExpectationValueVariance(Pauli("X"), key, unnormalized=True)
        self.assertEqual(instr.name, "save_expval_var")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "accum")

    def test_unnorm_cond_kwarg(self):
        """Test unnormalized, conditonal kwargs"""
        key = "test_key"
        instr = SaveExpectationValueVariance(Pauli("X"), key, conditional=True, unnormalized=True)
        self.assertEqual(instr.name, "save_expval_var")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "c_accum")

    def test_pershot_kwarg(self):
        """Test pershot kwarg"""
        key = "test_key"
        instr = SaveExpectationValueVariance(Pauli("X"), key, pershot=True)
        self.assertEqual(instr.name, "save_expval_var")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "list")

    def test_pershot_cond_kwarg(self):
        """Test pershot, conditonal kwargs"""
        key = "test_key"
        instr = SaveExpectationValueVariance(Pauli("X"), key, conditional=True, pershot=True)
        self.assertEqual(instr.name, "save_expval_var")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "c_list")


if __name__ == "__main__":
    unittest.main()
