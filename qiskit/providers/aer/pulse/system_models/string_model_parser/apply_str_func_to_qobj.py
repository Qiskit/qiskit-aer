# -*- coding: utf-8 -*-

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
# pylint: disable=invalid-name

"""
Functions for applying scalar functions in __fundict to the operators
represented in qutip Qobj.
"""

#from ...qutip_extra_lite.qobj import Qobj


def sin(val):
    """ Qiskit wrapper of sine function
    """
    if isinstance(val, Qobj):
        return val.sinm()
    else:
        return np.sin(val)


def cos(val):
    """ Qiskit wrapper of cosine function
    """
    if isinstance(val, Qobj):
        return val.cosm()
    else:
        return np.cos(val)


def exp(val):
    """ Qiskit wrapper of exponential function
    """
    if isinstance(val, Qobj):
        return val.expm()
    else:
        return np.exp(val)


def sqrt(val):
    """ Qiskit wrapper of square root
    """
    if isinstance(val, Qobj):
        return val.sqrtm()
    else:
        return np.sqrt(val)


def dag(qobj):
    """ Qiskit wrapper of adjoint
    """
    return qobj.dag()


def conj(val):
    """ Qiskit wrapper of conjugate
    """
    if isinstance(val, Qobj):
        return val.conj()
    else:
        return np.conj(val)


def apply_func(name, qobj):
    """ Apply function of given name, or do nothing if func not found
    """
    return __funcdict.get(name, lambda x: x)(qobj)


__funcdict = {'cos': cos, 'sin': sin, 'exp': exp,
              'sqrt': sqrt, 'conj': conj, 'dag': dag}
