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
# that they have been altered from the originals.

cimport cython
from libcpp.vector cimport vector
from libcpp.complex cimport complex
import numpy as np
cimport numpy as np

#from qiskit.providers.aer.openpulse.qobj.op_system import OPSystem

#cdef public class OPSystem [object OP_System, type op_system_t]

# cdef dict global_data
# cdef dict exp
# cdef unsigned char[::1] register


cdef extern from "src/numeric_integrator.hpp":
    cdef void td_ode_rhs(
        double t,
        np.ndarray vec,
        dict global_data,
        dict exp,
        list system,
        register
    ) except +

def td_ode_rhs_static(t, vec, global_data, exp, system, register):
    td_ode_rhs(t, vec, global_data, exp, system, register)
