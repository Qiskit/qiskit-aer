# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""tests for type_utils.py"""

import unittest
import numpy as np

from qiskit.providers.aer.pulse_new.de.type_utils import (convert_state,
                                                      type_spec_from_instance,
                                                      StateTypeConverter)

class TestTypeUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_convert_state(self):
        """Test convert_state"""

        type_spec = {'type': 'array', 'shape': (4,)}
        y = np.array([[1, 2],[3, 4]])
        expected = np.array([1, 2, 3, 4])

        self.assertAlmostEqual(convert_state(y, type_spec), expected)

        type_spec = {'type': 'array'}
        y = [[1, 2], [3, 4]]
        expected = np.array([[1, 2],[3, 4]])

        self.assertAlmostEqual(convert_state(y, type_spec), expected)

    def test_type_spec_from_instance(self):
        """Test type_spec_from_instance"""

        y = np.array([1, 2, 3, 4])
        type_spec = type_spec_from_instance(y)

        self.assertEqual(type_spec, {'type': 'array', 'shape': (4,)})

        y = np.array([[1, 2], [3, 4], [5, 6]])
        type_spec = type_spec_from_instance(y)

        self.assertEqual(type_spec, {'type': 'array', 'shape': (3, 2)})

    def test_converter_inner_outer(self):
        """Test standard constructor of StateTypeConverter along with basic state conversion
        functions"""

        inner_spec = {'type': 'array', 'shape': (4,)}
        outer_spec = {'type': 'array', 'shape': (2,2)}
        converter = StateTypeConverter(inner_spec, outer_spec)

        y_in = np.array([1,2,3,4])
        y_out = np.array([[1, 2], [3, 4]])

        convert_out = converter.inner_to_outer(y_in)
        convert_in = converter.outer_to_inner(y_out)

        self.assertAlmostEqual(convert_out, y_out)
        self.assertAlmostEqual(convert_in, y_in)

    def test_from_instances(self):
        """Test from_instances constructor"""

        inner_y = np.array([1, 2, 3, 4])
        outer_y = np.array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_instances(inner_y, outer_y)

        self.assertEqual(converter.inner_type_spec, {'type': 'array', 'shape': (4,)})
        self.assertEqual(converter.outer_type_spec, {'type': 'array', 'shape': (2,2)})

        converter = StateTypeConverter.from_instances(inner_y)

        self.assertEqual(converter.inner_type_spec, {'type': 'array', 'shape': (4,)})
        self.assertEqual(converter.outer_type_spec, {'type': 'array', 'shape': (4,)})

    def test_from_outer_instance_inner_type_spec(self):
        """Test from_outer_instance_inner_type_spec constructor"""

        # test case for inner type spec with 1d array
        inner_type_spec = {'type': 'array', 'ndim': 1}
        outer_y = np.array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_outer_instance_inner_type_spec(outer_y, inner_type_spec)

        self.assertEqual(converter.inner_type_spec, {'type': 'array', 'shape': (4,)})
        self.assertEqual(converter.outer_type_spec, {'type': 'array', 'shape': (2,2)})

        # inner type spec is a generic array
        inner_type_spec = {'type': 'array'}
        outer_y = np.array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_outer_instance_inner_type_spec(outer_y, inner_type_spec)

        self.assertEqual(converter.inner_type_spec, {'type': 'array', 'shape': (2,2)})
        self.assertEqual(converter.outer_type_spec, {'type': 'array', 'shape': (2,2)})

    def test_transform_rhs_funcs(self):
        """Test rhs function conversion"""

        inner_spec = {'type': 'array', 'shape': (4,)}
        outer_spec = {'type': 'array', 'shape': (2,2)}
        converter = StateTypeConverter(inner_spec, outer_spec)

        X = np.array([[1., 0.], [0., 1.]])

        # do matrix multiplication (a truly '2d' operation)
        def rhs(t, y):
            return t * (y @ y)

        def generator(t):
            return X

        rhs_funcs = {'rhs': rhs, 'generator': generator}
        new_rhs_funcs = converter.transform_rhs_funcs(rhs_funcs)

        test_t = np.pi
        y_2d = np.array([[1, 2], [3, 4]])
        y_1d = y_2d.flatten()

        expected_output = rhs(test_t, y_2d).flatten()
        output = new_rhs_funcs['rhs'](test_t, y_1d)

        self.assertAlmostEqual(output, expected_output)

        # verify generator vectorization
        expected_output = np.kron(X, np.eye(2))
        output = new_rhs_funcs['generator'](test_t)

        self.assertAlmostEqual(output, expected_output)

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)
