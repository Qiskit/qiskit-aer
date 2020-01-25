#!python
#cython: language_level=3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have be

cimport cython
from cpython.ref cimport PyObject
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.complex cimport complex
from libcpp cimport bool
import numpy as np
cimport numpy as np

# These definitions are only for testing the C++ wrappers over Python C API
cdef extern from "test_python_to_cpp.hpp":
    cdef bool cpp_test_py_list_to_cpp_vec(list val)
    cdef bool cpp_test_py_list_of_lists_to_cpp_vector_of_vectors(list val)
    cdef bool cpp_test_py_dict_string_numeric_to_cpp_map_string_numeric(dict val)
    cdef bool cpp_test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(dict val)
    cdef bool cpp_test_np_array_of_doubles(np.ndarray val)
    cdef bool cpp_test_evaluate_hamiltonians(list val)
    cdef bool cpp_test_py_ordered_map(dict val)

def test_py_list_to_cpp_vec(val):
    return cpp_test_py_list_to_cpp_vec(val)

def test_py_list_of_lists_to_cpp_vector_of_vectors(val):
    return cpp_test_py_list_of_lists_to_cpp_vector_of_vectors(val)

def test_py_dict_string_numeric_to_cpp_map_string_numeric(val):
    return cpp_test_py_dict_string_numeric_to_cpp_map_string_numeric(val)

def test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(val):
    return cpp_test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(val)

def test_np_array_of_doubles(val):
    return cpp_test_np_array_of_doubles(val)

def test_evaluate_hamiltonians(val):
    return cpp_test_evaluate_hamiltonians(val)

def test_py_ordered_map(val):
    return cpp_test_py_ordered_map(val)
