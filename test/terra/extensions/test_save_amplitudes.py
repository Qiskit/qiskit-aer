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


import unittest

from qiskit_aer.library import SaveAmplitudes
from ..common import QiskitAerTestCase


class TestSaveAmplitudes(QiskitAerTestCase):
    """SaveAmplitudes instruction tests"""

    def test_invalid_key_raises(self):
        """Test save instruction key is str"""
        self.assertRaises(TypeError, lambda: SaveAmplitudes(1, [0], 1))

    def test_invalid_state_raises(self):
        """Test non-Hermitian op raises exception."""
        self.assertRaises(ValueError, lambda: SaveAmplitudes(2, [4], "key"))

    def test_default_kwarg(self):
        """Test default kwargs"""
        key = "test_key"
        instr = SaveAmplitudes(2, [0], key)
        self.assertEqual(instr.name, "save_amplitudes")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "single")

    def test_cond_kwarg(self):
        """Test conditional kwarg"""
        key = "test_key"
        instr = SaveAmplitudes(2, [0], key, conditional=True)
        self.assertEqual(instr.name, "save_amplitudes")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "c_single")

    def test_pershot_kwarg(self):
        """Test pershot kwarg"""
        key = "test_key"
        instr = SaveAmplitudes(2, [0], key, pershot=True)
        self.assertEqual(instr.name, "save_amplitudes")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "list")

    def test_pershot_cond_kwarg(self):
        """Test pershot, conditonal kwargs"""
        key = "test_key"
        instr = SaveAmplitudes(2, [0], key, conditional=True, pershot=True)
        self.assertEqual(instr.name, "save_amplitudes")
        self.assertEqual(instr._label, key)
        self.assertEqual(instr._subtype, "c_list")


if __name__ == "__main__":
    unittest.main()
