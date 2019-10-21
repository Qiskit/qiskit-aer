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
import numpy as np
from qiskit.providers.aer.openpulse.qutip_lite.qobj import Qobj
from qiskit.providers.aer.openpulse.cy.test_py_to_cpp_helpers import \
    test_py_list_to_cpp_vec, test_py_list_of_lists_to_cpp_vector_of_vectors,\
    test_py_list_of_np_arrays,\
    test_py_dict_string_numeric_to_cpp_map_string_numeric,\
    test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles,\
    test_py_dict_string_list_of_np_array_to_cpp_map_string_vec_of_nparrays_of_doubles,\
    test_np_array_of_doubles, test_evaluate_hamiltonians, test_pass_qutip_qobj_to_cpp

class QutipFake:
    def __init__(self, arr):
        self.arr = arr
        self.is_flag = True
        self.value = 10

class TestPythonToCpp(unittest.TestCase):
    """ Test Pyhton C API wrappers we have for dealing with Python data structures
        in C++ code. """
    def setUp(self):
        pass

    def test_py_list_to_cpp_vec(self):
        arg = [1., 2., 3.]
        self.assertTrue(test_py_list_to_cpp_vec(arg))

    def test_py_list_of_lists_to_cpp_vector_of_vectors(self):
        arg = [[1., 2., 3.]]
        self.assertTrue(test_py_list_of_lists_to_cpp_vector_of_vectors(arg))

    def test_py_list_of_np_arrays(self):
        arg = [np.array([1., 2., 3.]), np.array([1., 2., 3.])]
        self.assertTrue(test_py_list_of_np_arrays(arg))

    def test_py_dict_string_numeric_to_cpp_map_string_numeric(self):
        arg = {"key": 1}
        self.assertTrue(test_py_dict_string_numeric_to_cpp_map_string_numeric(arg))

    def test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(self):
        arg = {"key": [[1., 2., 3.]]}
        self.assertTrue(test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(arg))

    def test_py_dict_string_list_of_np_array_to_cpp_map_string_vec_of_nparrays_of_doubles(self):
        arg = {"key": [np.array([0., 1.]), np.array([2., 3.])]}
        self.assertTrue(test_py_dict_string_list_of_np_array_to_cpp_map_string_vec_of_nparrays_of_doubles(arg))

    def test_np_array_of_doubles(self):
        arg = np.array([0., 1., 2., 3.])
        self.assertTrue(test_np_array_of_doubles(arg))

    def test_evaluate_hamiltonians(self):
        """ Evaluate different hamiltonina expressions"""
        self.assertEqual(True, False)

