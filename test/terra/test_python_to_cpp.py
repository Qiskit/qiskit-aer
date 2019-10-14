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
from qiskit.providers.aer.openpulse.cy.test_py_to_cpp_helpers import \
     get_value_complex, get_value_list_of_doubles, get_value_list_of_list_of_doubles, \
     get_value_map_of_string_and_list_of_list_of_doubles, get_value_string

class TestPythonToCpp(unittest.TestCase):
    """ Test Pyhton C API wrappers we have for dealing with Python data structures
        in C++ code. """
    def setUp(self):
        pass

    def test_py_string_to_cpp_string(self):
        expected = "thestring"
        result = get_value_string(expected)
        self.assertEquals(expected, result)

    def test_py_complex_double_to_cpp_complex_double(self):
        expected = 1. + 1.j
        result = get_value_complex(1.+1.j)
        self.assertEqual(expected, result)

    def test_py_list_to_cpp_vec(self):
        expected = [1., 2., 3.]
        result = get_value_list_of_doubles(expected)
        self.assertEqual(expected, result)

    def test_py_list_of_lists_to_cpp_vector_of_vectors(self):
        expected = [[1., 2., 3.]]
        result = get_value_list_of_list_of_doubles(expected)
        self.assertEqual(expected, result)

    def test_py_map_string_numeric_to_cpp_map_string_numeric(self):
        self.assertEquals(True, False)

    def test_py_map_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(self):
        expected = {"key": [[1., 2., 3.]]}
        result = get_value_map_of_string_and_list_of_list_of_doubles(expected)
        self.assertEqual(expected, result)

    def test_evaluate_hamiltonians(self):
        """ Evaluate different hamiltonina expressions"""
        self.assertEquals(True, False)