# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Cython wrapper for Aer QasmController.
"""

from libcpp.string cimport string

cdef extern from "simulators/qasm/qasm_controller.hpp" namespace "AER::Simulator":
    cdef cppclass QasmController:
        QasmController() except +

cdef extern from "base/controller.hpp" namespace "AER":
    cdef string controller_execute[QasmController](string &qobj) except +


def qasm_controller_execute(qobj):
    """Execute qobj on Aer C++ QasmController"""
    return controller_execute[QasmController](qobj)
