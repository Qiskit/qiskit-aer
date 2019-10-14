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
import numpy as np
cimport numpy as np

# These definitions are only for testing the C++ wrappers over Python C API
cdef extern from "src/helpers.hpp":
    cdef T get_value[T](PyObject * value) except +


def get_value_string(val):
    return get_value[string](<PyObject *>val)

def get_value_complex(val):
    return get_value[complex](<PyObject *>val)

def get_value_list_of_doubles(val):
    return get_value[vector[double]](<PyObject *>val)

def get_value_list_of_list_of_doubles(val):
    return get_value[vector[vector[double]]](<PyObject *>val)

def get_value_map_of_string_and_list_of_list_of_doubles(val):
    return get_value[unordered_map[string, vector[vector[double]]]](<PyObject *>val)