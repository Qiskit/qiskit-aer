# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Cython wrapper for Aer CHController.
"""

from libcpp.string cimport string

cdef extern from "simulators/ch/ch_controller.hpp" namespace "AER::Simulator":
    cdef cppclass CHController:
        CHController() except +
    cdef bint validate_memory(string &qobj, unsigned long long max_mb) except +

cdef extern from "base/controller.hpp" namespace "AER":
    cdef string controller_execute[CHController](string &qobj) except +

def ch_controller_execute(qobj):
    """Execute qobj on Aer C++ CHController"""
    return controller_execute[CHController](qobj)

def ch_validate_memory(qobj, max_mem_mb):
    """Check if we have enough memory to run these circuits."""
    return validate_memory(qobj, max_mem_mb)