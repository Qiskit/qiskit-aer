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

import numpy as np
from scipy import arange, conj
import scipy.sparse as sp

from .qobj import Qobj
from .operators import destroy
from .fastsparse import fast_csr_matrix


def basis(N, n=0, offset=0):
    """Generates the vector representation of a Fock state.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    n : int
        Integer corresponding to desired number state, defaults
        to 0 if omitted.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state.

    Returns
    -------
    state : :class:`qutip.Qobj`
      Qobj representing the requested number state ``|n>``.

    Examples
    --------
    >>> basis(5,2)
    Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
    Qobj data =
    [[ 0.+0.j]
     [ 0.+0.j]
     [ 1.+0.j]
     [ 0.+0.j]
     [ 0.+0.j]]

    Notes
    -----

    A subtle incompatibility with the quantum optics toolbox: In QuTiP::

        basis(N, 0) = ground state

    but in the qotoolbox::

        basis(N, 1) = ground state

    """
    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N >= 0")

    if (not isinstance(n, (int, np.integer))) or n < offset:
        raise ValueError("n must be integer n >= 0")

    if n - offset > (N - 1):  # check if n is within bounds
        raise ValueError("basis vector index need to be in n <= N-1")

    data = np.array([1], dtype=complex)
    ind = np.array([0], dtype=np.int32)
    ptr = np.array([0]*((n - offset)+1)+[1]*(N-(n-offset)),dtype=np.int32)

    return Qobj(fast_csr_matrix((data,ind,ptr), shape=(N,1)), isherm=False)


def qutrit_basis():
    """Basis states for a three level system (qutrit)

    Returns
    -------
    qstates : array
        Array of qutrit basis vectors

    """
    return np.array([basis(3, 0), basis(3, 1), basis(3, 2)], dtype=object)


def coherent(N, alpha, offset=0, method='operator'):
    """Generates a coherent state with eigenvalue alpha.

    Constructed using displacement operator on vacuum state.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue of coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state. Using a non-zero offset will make the
        default method 'analytic'.

    method : string {'operator', 'analytic'}
        Method for generating coherent state.

    Returns
    -------
    state : qobj
        Qobj quantum object for coherent state

    Examples
    --------
    >>> coherent(5,0.25j)
    Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
    Qobj data =
    [[  9.69233235e-01+0.j        ]
     [  0.00000000e+00+0.24230831j]
     [ -4.28344935e-02+0.j        ]
     [  0.00000000e+00-0.00618204j]
     [  7.80904967e-04+0.j        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent state is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting state is normalized. With 'analytic' method the coherent state
    is generated using the analytical formula for the coherent state
    coefficients in the Fock basis. This method does not guarantee that the
    state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """
    if method == "operator" and offset == 0:

        x = basis(N, 0)
        a = destroy(N)
        D = (alpha * a.dag() - conj(alpha) * a).expm()
        return D * x

    elif method == "analytic" or offset > 0:

        sqrtn = np.sqrt(np.arange(offset, offset+N, dtype=complex))
        sqrtn[0] = 1 # Get rid of divide by zero warning
        data = alpha/sqrtn
        if offset == 0:
            data[0] = np.exp(-abs(alpha)**2 / 2.0)
        else:
            s = np.prod(np.sqrt(np.arange(1, offset + 1))) # sqrt factorial
            data[0] = np.exp(-abs(alpha)**2 / 2.0) * alpha**(offset) / s
        np.cumprod(data, out=sqrtn) # Reuse sqrtn array
        return Qobj(sqrtn)

    else:
        raise TypeError(
            "The method option can only take values 'operator' or 'analytic'")


def coherent_dm(N, alpha, offset=0, method='operator'):
    """Density matrix representation of a coherent state.

    Constructed via outer product of :func:`qutip.states.coherent`

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue for coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state.

    method : string {'operator', 'analytic'}
        Method for generating coherent density matrix.

    Returns
    -------
    dm : qobj
        Density matrix representation of coherent state.

    Examples
    --------
    >>> coherent_dm(3,0.25j)
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.93941695+0.j          0.00000000-0.23480733j -0.04216943+0.j        ]
     [ 0.00000000+0.23480733j  0.05869011+0.j          0.00000000-0.01054025j]
     [-0.04216943+0.j          0.00000000+0.01054025j  0.00189294+0.j\
        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent density matrix is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting density matrix is normalized. With 'analytic' method the coherent
    density matrix is generated using the analytical formula for the coherent
    state coefficients in the Fock basis. This method does not guarantee that
    the state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """
    if method == "operator":
        psi = coherent(N, alpha, offset=offset)
        return psi * psi.dag()

    elif method == "analytic":
        psi = coherent(N, alpha, offset=offset, method='analytic')
        return psi * psi.dag()

    else:
        raise TypeError(
            "The method option can only take values 'operator' or 'analytic'")


def fock_dm(N, n=0, offset=0):
    """Density matrix representation of a Fock state

    Constructed via outer product of :func:`qutip.states.fock`.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    n : int
        ``int`` for desired number state, defaults to 0 if omitted.

    Returns
    -------
    dm : qobj
        Density matrix representation of Fock state.

    Examples
    --------
     >>> fock_dm(3,1)
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
     Qobj data =
     [[ 0.+0.j  0.+0.j  0.+0.j]
      [ 0.+0.j  1.+0.j  0.+0.j]
      [ 0.+0.j  0.+0.j  0.+0.j]]

    """
    psi = basis(N, n, offset=offset)

    return psi * psi.dag()


def fock(N, n=0, offset=0):
    """Bosonic Fock (number) state.

    Same as :func:`qutip.states.basis`.

    Parameters
    ----------
    N : int
        Number of states in the Hilbert space.

    n : int
        ``int`` for desired number state, defaults to 0 if omitted.

    Returns
    -------
        Requested number state :math:`\\left|n\\right>`.

    Examples
    --------
    >>> fock(4,3)
    Quantum object: dims = [[4], [1]], shape = [4, 1], type = ket
    Qobj data =
    [[ 0.+0.j]
     [ 0.+0.j]
     [ 0.+0.j]
     [ 1.+0.j]]

    """
    return basis(N, n, offset=offset)


def thermal_dm(N, n, method='operator'):
    """Density matrix for a thermal state of n particles

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    n : float
        Expectation value for number of particles in thermal state.

    method : string {'operator', 'analytic'}
        ``string`` that sets the method used to generate the
        thermal state probabilities

    Returns
    -------
    dm : qobj
        Thermal state density matrix.

    Examples
    --------
    >>> thermal_dm(5, 1)
    Quantum object: dims = [[5], [5]], \
shape = [5, 5], type = oper, isHerm = True
    Qobj data =
    [[ 0.51612903  0.          0.          0.          0.        ]
     [ 0.          0.25806452  0.          0.          0.        ]
     [ 0.          0.          0.12903226  0.          0.        ]
     [ 0.          0.          0.          0.06451613  0.        ]
     [ 0.          0.          0.          0.          0.03225806]]


    >>> thermal_dm(5, 1, 'analytic')
    Quantum object: dims = [[5], [5]], \
shape = [5, 5], type = oper, isHerm = True
    Qobj data =
    [[ 0.5      0.       0.       0.       0.     ]
     [ 0.       0.25     0.       0.       0.     ]
     [ 0.       0.       0.125    0.       0.     ]
     [ 0.       0.       0.       0.0625   0.     ]
     [ 0.       0.       0.       0.       0.03125]]

    Notes
    -----
    The 'operator' method (default) generates
    the thermal state using the truncated number operator ``num(N)``. This
    is the method that should be used in computations. The
    'analytic' method uses the analytic coefficients derived in
    an infinite Hilbert space. The analytic form is not necessarily normalized,
    if truncated too aggressively.

    """
    if n == 0:
        return fock_dm(N, 0)
    else:
        i = arange(N)
        if method == 'operator':
            beta = np.log(1.0 / n + 1.0)
            diags = np.exp(-beta * i)
            diags = diags / np.sum(diags)
            # populates diagonal terms using truncated operator expression
            rm = sp.spdiags(diags, 0, N, N, format='csr')
        elif method == 'analytic':
            # populates diagonal terms using analytic values
            rm = sp.spdiags((1.0 + n) ** (-1.0) * (n / (1.0 + n)) ** (i),
                            0, N, N, format='csr')
        else:
            raise ValueError(
                "'method' keyword argument must be 'operator' or 'analytic'")
    return Qobj(rm)


def maximally_mixed_dm(N):
    """
    Returns the maximally mixed density matrix for a Hilbert space of
    dimension N.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    Returns
    -------
    dm : qobj
        Thermal state density matrix.
    """
    if (not isinstance(N, (int, np.int64))) or N <= 0:
        raise ValueError("N must be integer N > 0")

    dm = sp.spdiags(np.ones(N, dtype=complex)/float(N), 0, N, N, format='csr')

    return Qobj(dm, isherm=True)


def ket2dm(Q):
    """Takes input ket or bra vector and returns density matrix
    formed by outer product.

    Parameters
    ----------
    Q : qobj
        Ket or bra type quantum object.

    Returns
    -------
    dm : qobj
        Density matrix formed by outer product of `Q`.

    Examples
    --------
    >>> x=basis(3,2)
    >>> ket2dm(x)
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j]]

    """
    if Q.type == 'ket':
        out = Q * Q.dag()
    elif Q.type == 'bra':
        out = Q.dag() * Q
    else:
        raise TypeError("Input is not a ket or bra vector.")
    return Qobj(out)


#
# projection operator
#
def projection(N, n, m, offset=0):
    """The projection operator that projects state :math:`|m>` on state :math:`|n>`.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    n, m : float
        The number states in the projection.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the projector.

    Returns
    -------
    oper : qobj
         Requested projection operator.

    """
    ket1 = basis(N, n, offset=offset)
    ket2 = basis(N, m, offset=offset)

    return ket1 * ket2.dag()


def zero_ket(N, dims=None):
    """
    Creates the zero ket vector with shape Nx1 and
    dimensions `dims`.

    Parameters
    ----------
    N : int
        Hilbert space dimensionality
    dims : list
        Optional dimensions if ket corresponds to
        a composite Hilbert space.

    Returns
    -------
    zero_ket : qobj
        Zero ket on given Hilbert space.

    """
    return Qobj(sp.csr_matrix((N, 1), dtype=complex), dims=dims)
