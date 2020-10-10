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
Functions to construct terra operators
"""

import numpy as np

from qiskit.quantum_info.operators.operator import Operator


def _create_destroy(dim, create=True):
    a, b = (0, 1) if create else (1, 0)
    data_op = np.zeros((dim,dim))
    for i in range(dim-1):
        data_op[i+a,i+b] = np.sqrt(i+1)
    return Operator(data_op)


def create(dim):
    return _create_destroy(dim, False)

def destroy(dim):
    return _create_destroy(dim, True)

def num(dim):
    return Operator(np.diag(np.arange(dim)))

def sigmax():
    return Operator.from_label('X')

def sigmay():
    return Operator.from_label('Y')

def sigmaz():
    return Operator.from_label('Z')

def qeye(dim):
    return Operator(np.identity(dim, dtype=np.complex128))

def basis(dim, n=0):
    data_op = np.zeros((dim, 1))
    data_op[n] = 1
    return Operator(data_op)

def tensor(op_list):
    ret = op_list[0]
    for op in op_list[1:]:
        ret = ret.tensor(op)
    return ret

def state(state_vector):
    return Operator(state_vector.reshape(1, state_vector.shape[0]))

def fock_dm(N, n=0):
    psi = basis(N, n)
    return psi * psi.adjoint()