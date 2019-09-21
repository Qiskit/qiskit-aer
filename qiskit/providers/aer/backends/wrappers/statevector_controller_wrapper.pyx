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

"""
Cython wrapper for Aer StatevectorController.
"""

from libcpp.string cimport string

cdef extern from "simulators/statevector/statevector_controller.hpp" namespace "AER::Simulator":
    cdef cppclass StatevectorController:
        StatevectorController() except +

cdef extern from "simulators/controller_execute.hpp" namespace "AER":
    cdef string controller_execute[StatevectorController](string &qobj) except +


def statevector_controller_execute(qobj):
    """Execute qobj on Aer C++ QasmController"""
    return controller_execute[StatevectorController](qobj)
