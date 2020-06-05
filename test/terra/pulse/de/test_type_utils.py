"""tests for type_utils.py"""

import unittest
import numpy as np
from qiskit.providers.aer.pulse.de.type_utils import (convert_state,
                                                      type_spec_from_instance,
                                                      StateTypeConverter)

class TestTypeUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_convert_state(self):
        """Test convert_state"""

        type_spec = {'type': 'array', 'shape': (4,)}
        y = np.array([[1, 2],[3, 4]])
        expected = np.array([1,2,3,4])

        self.assertAlmostEqual(convert_state(y, type_spec), expected)

        type_spec = {'type': 'array'}
        y = [[1, 2], [3, 4]]
        expected = np.array([[1, 2],[3, 4]])

        self.assertAlmostEqual(convert_state(y, type_spec), expected)

    def test_type_spec_from_instance(self):
        """Test type_spec_from_instance"""

        y = np.array([1,2,3,4])
        type_spec = type_spec_from_instance(y)

        self.assertEqual(type_spec, {'type': 'array', 'shape': (4,)})


    def test_StateTypeConverter(self):
        """Test methods and standard constructor of StateTypeConverter"""

        inner_spec = {'type': 'array', 'shape': (4,)}
        outer_spec = {'type': 'array', 'shape': (2,2)}
        converter = StateTypeConverter(inner_spec, outer_spec)

        y_in = np.array([1,2,3,4])
        y_out = np.array([[1, 2], [3, 4]])

        convert_out = converter.inner_to_outer(y_in)
        convert_in = converter.outer_to_inner(y_out)

        self.assertAlmostEqual(convert_out, y_out)
        self.assertAlmostEqual(convert_in, y_in)

        def rhs(t, y):
            return t * y

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

    def test_from_instances(self):
        """Test from_outer_instance_inner_type_spec constructor"""

        inner_type_spec = {'type': 'array', 'ndim': 1}
        outer_y = np.array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_outer_instance_inner_type_spec(outer_y, inner_type_spec)

        self.assertEqual(converter.inner_type_spec, {'type': 'array', 'shape': (4,)})
        self.assertEqual(converter.outer_type_spec, {'type': 'array', 'shape': (2,2)})

        inner_type_spec = {'type': 'array'}
        outer_y = np.array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_outer_instance_inner_type_spec(outer_y, inner_type_spec)

        self.assertEqual(converter.inner_type_spec, {'type': 'array', 'shape': (2,2)})
        self.assertEqual(converter.outer_type_spec, {'type': 'array', 'shape': (2,2)})

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)
