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


# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
# pylint: disable=invalid-name

"""
This module contains functions for generating Qobj representation of a variety
of commonly occuring quantum operators.
"""

import numpy as np
from .fastsparse import fast_csr_matrix, fast_identity
from .qobj import Qobj


# Spin operators
def jmat(j, *args):
    """Higher-order spin operators:

    Args:
        j (float): Spin of operator

        args (str): Which operator to return 'x','y','z','+','-'.
                    If no args given, then output is ['x','y','z']

    Returns:
        Qobj: Requested spin operator(s).

    Raises:
        TypeError: Invalid input.
    """
    if (np.fix(2 * j) != 2 * j) or (j < 0):
        raise TypeError('j must be a non-negative integer or half-integer')

    if not args:
        return jmat(j, 'x'), jmat(j, 'y'), jmat(j, 'z')

    if args[0] == '+':
        A = _jplus(j)
    elif args[0] == '-':
        A = _jplus(j).getH()
    elif args[0] == 'x':
        A = 0.5 * (_jplus(j) + _jplus(j).getH())
    elif args[0] == 'y':
        A = -0.5 * 1j * (_jplus(j) - _jplus(j).getH())
    elif args[0] == 'z':
        A = _jz(j)
    else:
        raise TypeError('Invalid type')

    return Qobj(A)


def _jplus(j):
    """
    Internal functions for generating the data representing the J-plus
    operator.
    """
    m = np.arange(j, -j - 1, -1, dtype=complex)
    data = (np.sqrt(j * (j + 1.0) - (m + 1.0) * m))[1:]
    N = m.shape[0]
    ind = np.arange(1, N, dtype=np.int32)
    ptr = np.array(list(range(N - 1)) + [N - 1] * 2, dtype=np.int32)
    ptr[-1] = N - 1
    return fast_csr_matrix((data, ind, ptr), shape=(N, N))


def _jz(j):
    """
    Internal functions for generating the data representing the J-z operator.
    """
    N = int(2 * j + 1)
    data = np.array([j - k for k in range(N) if (j - k) != 0], dtype=complex)
    # Even shaped matrix
    if N % 2 == 0:
        ind = np.arange(N, dtype=np.int32)
        ptr = np.arange(N + 1, dtype=np.int32)
        ptr[-1] = N
    # Odd shaped matrix
    else:
        j = int(j)
        ind = np.array(list(range(j)) + list(range(j + 1, N)), dtype=np.int32)
        ptr = np.array(list(range(j + 1)) + list(range(j, N)), dtype=np.int32)
        ptr[-1] = N - 1
    return fast_csr_matrix((data, ind, ptr), shape=(N, N))


#
# Spin j operators:
#
def spin_Jx(j):
    """Spin-j x operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'x')


def spin_Jy(j):
    """Spin-j y operator

    Args:
        j (float): Spin of operator

    Returns:
        Qobj: representation of the operator.

    """
    return jmat(j, 'y')


def spin_Jz(j):
    """Spin-j z operator

    Args:
        j (float): Spin of operator

    Returns:
        Qobj: representation of the operator.

    """
    return jmat(j, 'z')


def spin_Jm(j):
    """Spin-j annihilation operator

    Parameters:
        j (float): Spin of operator

    Returns:
        Qobj: representation of the operator.

    """
    return jmat(j, '-')


def spin_Jp(j):
    """Spin-j creation operator

    Args:
        j (float): Spin of operator

    Returns:
        Qobj: representation of the operator.

    """
    return jmat(j, '+')


def spin_J_set(j):
    """Set of spin-j operators (x, y, z)

    Args:
        j (float): Spin of operators

    Returns:
        list: list of ``qobj`` representating of the spin operator.

    """
    return jmat(j)


#
# Pauli spin 1/2 operators:
#
def sigmap():
    """Creation operator for Pauli spins.

    Examples
    --------
    >>> sigmap()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 0.  0.]]

    """
    return jmat(1 / 2., '+')


def sigmam():
    """Annihilation operator for Pauli spins.

    Examples
    --------
    >>> sigmam()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  0.]
     [ 1.  0.]]

    """
    return jmat(1 / 2., '-')


def sigmax():
    """Pauli spin 1/2 sigma-x operator

    Examples
    --------
    >>> sigmax()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 1.  0.]]

    """
    return 2.0 * jmat(1.0 / 2, 'x')


def sigmay():
    """Pauli spin 1/2 sigma-y operator.

    Examples
    --------
    >>> sigmay()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.-1.j]
     [ 0.+1.j  0.+0.j]]

    """
    return 2.0 * jmat(1.0 / 2, 'y')


def sigmaz():
    """Pauli spin 1/2 sigma-z operator.

    Examples
    --------
    >>> sigmaz()
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]

    """
    return 2.0 * jmat(1.0 / 2, 'z')


#
# DESTROY returns annihilation operator for N dimensional Hilbert space
# out = destroy(N), N is integer value &  N>0
#
def destroy(N, offset=0):
    """Destruction (lowering) operator.

    Args:
        N (int): Dimension of Hilbert space.

        offset (int): (default 0) The lowest number state that is included
                      in the finite number state representation of the operator.

    Returns:
        Qobj: Qobj for lowering operator.

    Raises:
        ValueError: Invalid input.

    """
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset + 1, N + offset, dtype=complex))
    ind = np.arange(1, N, dtype=np.int32)
    ptr = np.arange(N + 1, dtype=np.int32)
    ptr[-1] = N - 1
    return Qobj(fast_csr_matrix((data, ind, ptr), shape=(N, N)), isherm=False)


#
# create returns creation operator for N dimensional Hilbert space
# out = create(N), N is integer value &  N>0
#
def create(N, offset=0):
    """Creation (raising) operator.

    Args:
        N (int): Dimension of Hilbert space.

        offset (int): (default 0) The lowest number state that is included
                      in the finite number state representation of the operator.

    Returns:
        Qobj: Qobj for raising operator.

    Raises:
        ValueError: Invalid inputs.

    """
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    qo = destroy(N, offset=offset)  # create operator using destroy function
    return qo.dag()


#
# QEYE returns identity operator for an N dimensional space
# a = qeye(N), N is integer & N>0
#
def qeye(N):
    """
    Identity operator

    Args:
        N (int): Dimension of Hilbert space. If provided as a list of ints,
                 then the dimension is the product over this list, but the
                 ``dims`` property of the new Qobj are set to this list.

    Returns:
        Qobj: Identity operator Qobj.

    Raises:
        ValueError: Invalid input.
    """
    N = int(N)
    if N < 0:
        raise ValueError("N must be integer N>=0")
    return Qobj(fast_identity(N), isherm=True, isunitary=True)


def identity(N):
    """Identity operator. Alternative name to :func:`qeye`.

    Parameters
    ----------
    N : int or list of ints
        Dimension of Hilbert space. If provided as a list of ints,
        then the dimension is the product over this list, but the
        ``dims`` property of the new Qobj are set to this list.

    Returns
    -------
    oper : qobj
        Identity operator Qobj.
    """
    return qeye(N)


def position(N, offset=0):
    """
    Position operator x=1/sqrt(2)*(a+a.dag())

    Args:
        N (int): Number of Fock states in Hilbert space.

        offset (int): (default 0) The lowest number state that is included
                      in the finite number state representation of the operator.
    Returns:
        Qobj: Position operator as Qobj.
    """
    a = destroy(N, offset=offset)
    return 1.0 / np.sqrt(2.0) * (a + a.dag())


def momentum(N, offset=0):
    """
    Momentum operator p=-1j/sqrt(2)*(a-a.dag())

    Args:
        N (int): Number of Fock states in Hilbert space.

        offset (int): (default 0) The lowest number state that is
                      included in the finite number state
                      representation of the operator.
    Returns:
        Qobj: Momentum operator as Qobj.
    """
    a = destroy(N, offset=offset)
    return -1j / np.sqrt(2.0) * (a - a.dag())


# number operator, important!
def num(N, offset=0):
    """Quantum object for number operator.

    Args:
        N (int): The dimension of the Hilbert space.

        offset(int): (default 0) The lowest number state that is included
                     in the finite number state representation of the operator.

    Returns:
        Qobj: Qobj for number operator.

    """
    if offset == 0:
        data = np.arange(1, N, dtype=complex)
        ind = np.arange(1, N, dtype=np.int32)
        ptr = np.array([0] + list(range(0, N)), dtype=np.int32)
        ptr[-1] = N - 1
    else:
        data = np.arange(offset, offset + N, dtype=complex)
        ind = np.arange(N, dtype=np.int32)
        ptr = np.arange(N + 1, dtype=np.int32)
        ptr[-1] = N

    return Qobj(fast_csr_matrix((data, ind, ptr),
                                shape=(N, N)), isherm=True)
