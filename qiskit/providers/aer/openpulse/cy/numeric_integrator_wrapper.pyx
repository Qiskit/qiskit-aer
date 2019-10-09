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


#from qiskit.providers.aer.openpulse.qobj.op_system import OPSystem

#cdef public class OPSystem [object OP_System, type op_system_t]

# cdef dict global_data
# cdef dict exp
# cdef unsigned char[::1] register


cdef extern from "src/numeric_integrator.hpp":
    cdef void td_ode_rhs(
        dict global_data,
        dict channels,
        dict vars,
        dict freqs,
        dict exp,
        unsigned char register
    ) except +

def td_ode_rhs_static(global_data, channels, vars, freqs, exp, register):
    td_ode_rhs(global_data, channels, vars, freqs, exp, register)
