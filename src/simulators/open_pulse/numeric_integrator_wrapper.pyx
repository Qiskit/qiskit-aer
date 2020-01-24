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
cimport numpy as np

cdef extern from "numeric_integrator.hpp":
    cdef np.ndarray td_ode_rhs(
        double t,
        np.ndarray vec,
        dict global_data,
        dict exp,
        list system,
        dict channels,
        register
    ) except +

def td_ode_rhs_static(t, vec, global_data, exp, system, channels, register):
    return td_ode_rhs(t, vec, global_data, exp, system, channels, register)

