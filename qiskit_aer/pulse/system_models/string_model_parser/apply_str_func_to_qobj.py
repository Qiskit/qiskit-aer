# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Functions for applying scalar functions in __fundict to the operators
represented in qutip Qobj.
"""


def dag(qobj):
    """ Qiskit wrapper of adjoint
    """
    return qobj.dag()


def apply_func(name, qobj):
    """ Apply function of given name, or do nothing if func not found
    """
    return __funcdict.get(name, lambda x: x)(qobj)


# pylint: disable=invalid-name
__funcdict = {'dag': dag}
