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

"""States
"""

import numpy as np
from .qobj import Qobj
from .fastsparse import fast_csr_matrix


# used by qobj_generators
def fock_dm(N, n=0, offset=0):
    """Density matrix representation of a Fock state

    Constructed via outer product of :func:`qutip.states.fock`.

    Args:
        N (int): Number of Fock states in Hilbert space.

        n (int): Desired number state, defaults to 0 if omitted.

        offset (int): Energy level offset.

    Returns:
        Qobj: Density matrix representation of Fock state.

    """
    psi = basis(N, n, offset=offset)

    return psi * psi.dag()


def basis(N, n=0, offset=0):
    """Generates the vector representation of a Fock state.

    Args:
        N (int): Number of Fock states in Hilbert space.

        n (int): Integer corresponding to desired number
        state, defaults to 0 if omitted.

        offset (int): The lowest number state that is included
                      in the finite number state representation
                      of the state.

    Returns:
        Qobj: Qobj representing the requested number state ``|n>``.

    Raises:
        ValueError: Invalid input value.

    """
    if (not isinstance(N, (int, np.integer))) or N < 0:
        raise ValueError("N must be integer N >= 0")

    if (not isinstance(n, (int, np.integer))) or n < offset:
        raise ValueError("n must be integer n >= 0")

    if n - offset > (N - 1):  # check if n is within bounds
        raise ValueError("basis vector index need to be in n <= N-1")

    data = np.array([1], dtype=complex)
    ind = np.array([0], dtype=np.int32)
    ptr = np.array([0] * ((n - offset) + 1) + [1] * (N - (n - offset)),
                   dtype=np.int32)

    return Qobj(fast_csr_matrix((data, ind, ptr), shape=(N, 1)), isherm=False)
