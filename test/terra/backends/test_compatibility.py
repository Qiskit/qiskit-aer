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
"""
Tests if quantum info result compatibility classes.
These can be removed when this deprecation period is finished.
"""

from test.terra.common import QiskitAerTestCase

import numpy as np
import qiskit.quantum_info as qi
import qiskit.providers.aer.backends.compatibility as cqi


class TestResultCompatibility(QiskitAerTestCase):
    """Result compatiblity class tests."""

    def test_statevector_eq(self):
        orig = qi.random_statevector(4, seed=10)
        compat = cqi.Statevector(orig.data)
        self.assertEqual(compat, orig)
        self.assertEqual(orig, compat)

    def test_statevector_getattr(self):
        compat = cqi.Statevector([1, 1e-10])
        with self.assertWarns(DeprecationWarning):
            value = compat.round(5)
        self.assertEqual(type(value), np.ndarray)
        self.assertTrue(np.all(value == np.array([1, 0])))

    def test_density_matrix_eq(self):
        orig = qi.random_density_matrix(4, seed=10)
        compat = cqi.DensityMatrix(orig.data)
        self.assertEqual(compat, orig)
        self.assertEqual(orig, compat)

    def test_density_matrix_getattr(self):
        compat = cqi.DensityMatrix([[1, 0], [0, 1e-10]])
        with self.assertWarns(DeprecationWarning):
            value = compat.round(5)
        self.assertEqual(type(value), np.ndarray)
        self.assertTrue(np.all(value == np.array([[1, 0], [0, 0]])))

    def test_operator_eq(self):
        orig = qi.random_unitary(4, seed=10)
        compat = cqi.Operator(orig.data)
        self.assertEqual(compat, orig)
        self.assertEqual(orig, compat)

    def test_operator_getattr(self):
        compat = cqi.Operator([[1, 0], [1e-10, 1]])
        with self.assertWarns(DeprecationWarning):
            value = compat.round(5)
        self.assertEqual(type(value), np.ndarray)
        self.assertTrue(np.all(value == np.eye(2)))

    def test_superop_eq(self):
        orig = qi.SuperOp(qi.random_quantum_channel(4, seed=10))
        compat = cqi.SuperOp(orig.data)
        self.assertEqual(compat, orig)
        self.assertEqual(orig, compat)

    def test_superop_getattr(self):
        compat = cqi.SuperOp(np.eye(4))
        with self.assertWarns(DeprecationWarning):
            value = compat.round(5)
        self.assertEqual(type(value), np.ndarray)
        self.assertTrue(np.all(value == np.eye(4)))

    def test_stabilizer_eq(self):
        orig = qi.StabilizerState(qi.random_clifford(4, seed=10))
        compat = cqi.StabilizerState(orig.clifford)
        self.assertEqual(compat, orig)
        self.assertEqual(orig, compat)

    def test_stabilizer_getattr(self):
        clifford = qi.random_clifford(4, seed=10)
        compat = cqi.StabilizerState(clifford)
        with self.assertWarns(DeprecationWarning):
            value = compat.keys()
        self.assertEqual(value, clifford.to_dict().keys())

    def test_stabilizer_getitem(self):
        clifford = qi.random_clifford(4, seed=10)
        cliff_dict = clifford.to_dict()
        compat = cqi.StabilizerState(clifford)
        with self.assertWarns(DeprecationWarning):
            stabs = compat['stabilizer']
        self.assertEqual(stabs, cliff_dict['stabilizer'])
        with self.assertWarns(DeprecationWarning):
            destabs = compat['destabilizer']
        self.assertEqual(destabs, cliff_dict['destabilizer'])
