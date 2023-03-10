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

import copy
from test.terra.common import QiskitAerTestCase

import numpy as np
import qiskit.quantum_info as qi
import qiskit_aer.backends.compatibility as cqi


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

    def test_statevector_copy(self):
        compat = cqi.Statevector([1, 1e-10])
        cpy = copy.copy(compat)
        self.assertEqual(cpy, compat)

    def test_statevector_linop(self):
        orig = qi.random_statevector(4, seed=10)
        compat = cqi.Statevector(orig.data)
        self.assertEqual(2 * compat - orig, orig)
        self.assertEqual(2 * orig - compat, orig)

    def test_statevector_tensor(self):
        orig = qi.random_statevector(2, seed=10)
        compat = cqi.Statevector(orig.data)
        target = orig.tensor(orig)
        self.assertEqual(compat.tensor(orig), target)
        self.assertEqual(orig.tensor(compat), target)

    def test_statevector_evolve(self):
        orig = qi.random_statevector(2, seed=10)
        compat = cqi.Statevector(orig.data)
        orig_op = qi.random_unitary(2, seed=10)
        compat_op = cqi.Operator(orig_op.data)
        target = orig.evolve(orig_op)
        self.assertEqual(orig.evolve(compat_op), target)
        self.assertEqual(compat.evolve(orig_op), target)
        self.assertEqual(compat.evolve(compat_op), target)

    def test_statevector_iterable_methods(self):
        """Test that the iterable magic methods and related Numpy properties
        work on the compatibility classes."""
        compat = cqi.Statevector([0.5, 0.5j, -0.5, 0.5j])
        compat_data = compat.data

        with self.assertWarns(DeprecationWarning):
            compat_len = len(compat)
        self.assertEqual(compat_len, len(compat_data))
        with self.assertWarns(DeprecationWarning):
            compat_shape = compat.shape
        self.assertEqual(compat_shape, compat_data.shape)
        with self.assertWarns(DeprecationWarning):
            compat_iter = tuple(compat)
        self.assertEqual(compat_iter, tuple(compat.data))

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

    def test_density_matrix_copy(self):
        compat = cqi.DensityMatrix([[1, 0], [0, 1e-10]])
        cpy = copy.copy(compat)
        self.assertEqual(cpy, compat)

    def test_density_matrix_linop(self):
        orig = qi.random_density_matrix(4, seed=10)
        compat = cqi.DensityMatrix(orig.data)
        self.assertEqual(2 * compat - orig, orig)
        self.assertEqual(2 * orig - compat, orig)

    def test_density_matrix_tensor(self):
        orig = qi.random_density_matrix(2, seed=10)
        compat = cqi.DensityMatrix(orig.data)
        target = orig.tensor(orig)
        self.assertEqual(compat.tensor(orig), target)
        self.assertEqual(orig.tensor(compat), target)

    def test_density_matrix_evolve(self):
        orig = qi.random_density_matrix(2, seed=10)
        compat = cqi.DensityMatrix(orig.data)
        orig_op = qi.random_unitary(2, seed=10)
        compat_op = cqi.Operator(orig_op.data)
        target = orig.evolve(orig_op)
        self.assertEqual(orig.evolve(compat_op), target)
        self.assertEqual(compat.evolve(orig_op), target)
        self.assertEqual(compat.evolve(compat_op), target)

    def test_density_matrix_iterable_methods(self):
        """Test that the iterable magic methods and related Numpy properties
        work on the compatibility classes."""
        compat = cqi.DensityMatrix([[0.5, 0.5j], [-0.5j, 0.5]])
        compat_data = compat.data

        with self.assertWarns(DeprecationWarning):
            compat_len = len(compat)
        self.assertEqual(compat_len, len(compat_data))
        with self.assertWarns(DeprecationWarning):
            compat_shape = compat.shape
        self.assertEqual(compat_shape, compat_data.shape)
        with self.assertWarns(DeprecationWarning):
            compat_iter = tuple(compat)
        np.testing.assert_array_equal(compat_iter, compat.data)

    def test_unitary_eq(self):
        orig = qi.random_unitary(4, seed=10)
        compat = cqi.Operator(orig.data)
        self.assertEqual(compat, orig)
        self.assertEqual(orig, compat)

    def test_unitary_getattr(self):
        compat = cqi.Operator([[1, 0], [1e-10, 1]])
        with self.assertWarns(DeprecationWarning):
            value = compat.round(5)
        self.assertEqual(type(value), np.ndarray)
        self.assertTrue(np.all(value == np.eye(2)))

    def test_unitary_copy(self):
        compat = cqi.Operator([[1, 0], [1e-10, 1]])
        cpy = copy.copy(compat)
        self.assertEqual(cpy, compat)

    def test_unitary_linop(self):
        orig = qi.random_unitary(4, seed=10)
        compat = cqi.Operator(orig.data)
        self.assertEqual(2 * compat - orig, orig)
        self.assertEqual(2 * orig - compat, orig)

    def test_unitary_tensor(self):
        orig = qi.random_unitary(2, seed=10)
        compat = cqi.Operator(orig.data)
        target = orig.tensor(orig)
        self.assertEqual(compat.tensor(orig), target)
        self.assertEqual(orig.tensor(compat), target)

    def test_unitary_compose(self):
        orig = qi.random_unitary(2, seed=10)
        compat = cqi.Operator(orig.data)
        target = orig.compose(orig)
        self.assertEqual(compat.compose(orig), target)
        self.assertEqual(orig.compose(compat), target)

    def test_unitary_evolve(self):
        orig = qi.random_unitary(2, seed=10)
        compat = cqi.Operator(orig.data)
        state = qi.random_statevector(2, seed=10)
        target = state.evolve(orig)
        self.assertEqual(state.evolve(compat), target)

    def test_unitary_iterable_methods(self):
        """Test that the iterable magic methods and related Numpy properties
        work on the compatibility classes."""
        compat = cqi.Operator(qi.random_unitary(2, seed=10))
        compat_data = compat.data

        with self.assertWarns(DeprecationWarning):
            compat_len = len(compat)
        self.assertEqual(compat_len, len(compat_data))
        with self.assertWarns(DeprecationWarning):
            compat_shape = compat.shape
        self.assertEqual(compat_shape, compat_data.shape)
        with self.assertWarns(DeprecationWarning):
            compat_iter = tuple(compat)
        np.testing.assert_array_equal(compat_iter, compat.data)

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

    def test_superop_copy(self):
        compat = cqi.SuperOp(np.eye(4))
        cpy = copy.copy(compat)
        self.assertEqual(cpy, compat)

    def test_superop_linop(self):
        orig = qi.SuperOp(qi.random_quantum_channel(4, seed=10))
        compat = cqi.SuperOp(orig.data)
        self.assertEqual(2 * compat - orig, orig)
        self.assertEqual(2 * orig - compat, orig)

    def test_superop_iterable_methods(self):
        """Test that the iterable magic methods and related Numpy properties
        work on the compatibility classes."""
        compat = cqi.SuperOp(np.eye(4))
        compat_data = compat.data

        with self.assertWarns(DeprecationWarning):
            compat_len = len(compat)
        self.assertEqual(compat_len, len(compat_data))
        with self.assertWarns(DeprecationWarning):
            compat_shape = compat.shape
        self.assertEqual(compat_shape, compat_data.shape)
        with self.assertWarns(DeprecationWarning):
            compat_iter = tuple(compat)
        np.testing.assert_array_equal(compat_iter, compat.data)

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
            stabs = compat["stabilizer"]
        self.assertEqual(stabs, cliff_dict["stabilizer"])
        with self.assertWarns(DeprecationWarning):
            destabs = compat["destabilizer"]
        self.assertEqual(destabs, cliff_dict["destabilizer"])

    def test_stabilizer_copy(self):
        clifford = qi.random_clifford(4, seed=10)
        compat = cqi.StabilizerState(clifford)
        cpy = copy.copy(compat)
        self.assertEqual(cpy, compat)

    def test_stabilizer_iterable_methods(self):
        """Test that the iterable magic methods and related dict properties
        work on the compatibility classes."""
        clifford = qi.random_clifford(4, seed=10)
        cliff_dict = clifford.to_dict()
        compat = cqi.StabilizerState(clifford)

        with self.assertWarns(DeprecationWarning):
            compat_keys = compat.keys()
        self.assertEqual(compat_keys, cliff_dict.keys())

        with self.assertWarns(DeprecationWarning):
            compat_iter = set(compat)
        self.assertEqual(compat_iter, set(cliff_dict))

        with self.assertWarns(DeprecationWarning):
            compat_items = compat.items()
        self.assertEqual(sorted(compat_items), sorted(cliff_dict.items()))

        with self.assertWarns(DeprecationWarning):
            compat_len = len(compat)
        self.assertEqual(compat_len, len(cliff_dict))
