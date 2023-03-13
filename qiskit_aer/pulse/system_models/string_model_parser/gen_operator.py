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


def _create_destroy(dim, creation=True):
    a_x, b_x = (1, 0) if creation else (0, 1)
    data_op = np.zeros((dim, dim))
    for i in range(dim - 1):
        data_op[i + a_x, i + b_x] = np.sqrt(i + 1)
    return Operator(data_op)


def create(dim):
    """Creation operator.
    Args:
        dim (int): Dimension of Hilbert space.
    Returns:
        Operator: Operator representation of creation operator.
    """
    return _create_destroy(dim, creation=True)


def destroy(dim):
    """Destruction operator.
    Args:
        dim (int): Dimension of Hilbert space.
    Returns:
        Operator: Operator representation of destruction operator.
    """
    return _create_destroy(dim, creation=False)


def num(dim):
    """Operator representation for number operator.
    Args:
        dim (int): The dimension of the Hilbert space.
    Returns:
        Operator: Operator representation for number operator.
    """
    return Operator(np.diag(np.arange(dim)))


def sigmax():
    """Operator representation for Pauli spin 1/2 sigma-x operator.
    Returns:
        Operator: Operator representation for sigma x.
    """
    return Operator.from_label("X")


def sigmay():
    """Operator representation for Pauli spin 1/2 sigma-y operator.
    Returns:
        Operator: Operator representation for sigma y.
    """
    return Operator.from_label("Y")


def sigmaz():
    """Operator representation for Pauli spin 1/2 sigma-z operator.
    Returns:
        Operator: Operator representation for sigma z.
    """
    return Operator.from_label("Z")


def identity(dim):
    """Identity operator.
    Args:
        dim (int): Dimension of Hilbert space.
    Returns:
        Operator: Operator representation of identity operator.
    """
    return Operator(np.identity(dim, dtype=np.complex128))


def basis(dim, n=0):
    """Vector representation of a Fock state.
    Args:
        dim (int): Number of Fock states in Hilbert space.
        n (int): Integer corresponding to desired number state, default value is 0.
    Returns:
        Operator: Opertor representing the requested number state ``|n>``.
    """
    data_op = np.zeros((dim, 1))
    data_op[n] = 1
    return Operator(data_op)


def tensor(op_list):
    """Tensor product of input operators.
    Args:
        op_list (array_like): List or array of Operators objects for tensor product.
    Returns:
        Operator: Operator representation of tensor product.
    """
    ret = op_list[0]
    for op in op_list[1:]:
        ret = ret.tensor(op)
    return ret


def state(state_vector):
    """Operator representation of state vector.
    Args:
         state_vector (array_like): array representing the state-vector.
    Returns:
         Operator: State vector operator representation.
    """
    return Operator(state_vector.reshape(1, state_vector.shape[0]))


def fock_dm(dim, n=0):
    """Density matrix representation of a Fock state
    Args:
        dim (int): Number of Fock states in Hilbert space.
        n (int): Desired number state, defaults to 0 if omitted.
    Returns:
        Operator: Density matrix Operator representation of Fock state.
    """
    psi = basis(dim, n)
    return psi.adjoint() & psi
