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


from qiskit.extensions.exceptions import ExtensionError
from qiskit.providers.aer.library import SaveExpval
from qiskit.quantum_info.operators import Pauli

from ..common import QiskitAerTestCase


class TestSaveExpval(QiskitAerTestCase):
    """SaveExpval instruction tests"""

    def test_invalid_key_raises(self):
        """Test save instruction key is str"""
        self.assertRaises(ExtensionError, lambda: SaveExpval(1, Pauli('Z')))

    def test_nonhermitian_raises(self):
        """Test non-Hermitian op raises exception."""
        op = [[0, 1j], [1j, 0]]
        self.assertRaises(ExtensionError, lambda: SaveExpval('expval', op))

    def test_default_kwarg(self):
        """Test default kwargs"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'))
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertFalse(instr._variance)
        self.assertEqual(instr._subtype, 'average')

    def test_cond_kwarg(self):
        """Test conditional kwarg"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), conditional=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertFalse(instr._variance)
        self.assertEqual(instr._subtype, 'c_average')

    def test_var_kwarg(self):
        """Test variance kwarg"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), variance=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertTrue(instr._variance)
        self.assertEqual(instr._subtype, 'average')

    def test_var_cond_kwarg(self):
        """Test variance, conditional kwargs"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), variance=True, conditional=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertTrue(instr._variance)
        self.assertEqual(instr._subtype, 'c_average')

    def test_unnorm_kwarg(self):
        """Test unnormalized kwarg"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), unnormalized=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertFalse(instr._variance)
        self.assertEqual(instr._subtype, 'accum')

    def test_unnorm_cond_kwarg(self):
        """Test unnormalized, conditonal kwargs"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), conditional=True, unnormalized=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertFalse(instr._variance)
        self.assertEqual(instr._subtype, 'c_accum')

    def test_unnorm_var_kwarg(self):
        """Test unnormalized, variance kwargs"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), variance=True, unnormalized=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertTrue(instr._variance)
        self.assertEqual(instr._subtype, 'accum')

    def test_unnorm_var_cond_kwarg(self):
        """Test unnormalized, conditional kwargs"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), variance=True, conditional=True,
                           unnormalized=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertTrue(instr._variance)
        self.assertEqual(instr._subtype, 'c_accum')

    def test_pershot_kwarg(self):
        """Test pershot kwarg"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), pershot=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertFalse(instr._variance)
        self.assertEqual(instr._subtype, 'list')

    def test_pershot_cond_kwarg(self):
        """Test pershot, conditonal kwargs"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), conditional=True, pershot=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertFalse(instr._variance)
        self.assertEqual(instr._subtype, 'c_list')

    def test_pershot_var_kwarg(self):
        """Test pershot, variance kwargs"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), variance=True, pershot=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertTrue(instr._variance)
        self.assertEqual(instr._subtype, 'list')

    def test_pershot_var_cond_kwarg(self):
        """Test pershot, conditional kwargs"""
        key = 'test_key'
        instr = SaveExpval(key, Pauli('X'), variance=True, conditional=True,
                           pershot=True)
        self.assertEqual(instr.name, 'save_expval')
        self.assertEqual(instr._key, key)
        self.assertTrue(instr._variance)
        self.assertEqual(instr._subtype, 'c_list')


if __name__ == '__main__':
    unittest.main()
