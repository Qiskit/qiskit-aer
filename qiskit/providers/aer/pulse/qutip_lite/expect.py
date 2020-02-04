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
# pylint: disable=invalid-name

"""
Module for expectation values.
"""

__all__ = ['expect', 'variance']

import numpy as np
from .qobj import Qobj
# pylint: disable=import-error, no-name-in-module
from .cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi, cy_spmm_tr,
                            expect_csr_ket)

expect_rho_vec = cy_expect_rho_vec
expect_psi = cy_expect_psi


def expect(oper, state):
    """Calculates the expectation value for operator(s) and state(s).

    Args:
        oper (Qobj or list): A single or a `list` or operators
                              for expectation value.

        state (Qobj or list): A single or a `list` of quantum states
                               or density matrices.

    Returns:
        real or complex or ndarray: Expectation value.  ``real`` if `oper` is
                                    Hermitian, ``complex`` otherwise. A (nested)
                                    array of expectaction values of state or
                                    operator are arrays.

    Raises:
        TypeError: Inputs are not quantum objects.
    """
    if isinstance(state, Qobj) and isinstance(oper, Qobj):
        return _single_qobj_expect(oper, state)

    elif isinstance(oper, (list, np.ndarray)):
        if isinstance(state, Qobj):
            if (all([op.isherm for op in oper]) and
                    (state.isket or state.isherm)):
                return np.array([_single_qobj_expect(o, state) for o in oper])
            else:
                return np.array([_single_qobj_expect(o, state) for o in oper],
                                dtype=complex)
        else:
            return [expect(o, state) for o in oper]

    elif isinstance(state, (list, np.ndarray)):
        if oper.isherm and all([(op.isherm or op.type == 'ket')
                                for op in state]):
            return np.array([_single_qobj_expect(oper, x) for x in state])
        else:
            return np.array([_single_qobj_expect(oper, x) for x in state],
                            dtype=complex)
    else:
        raise TypeError('Arguments must be quantum objects')


# pylint: disable=inconsistent-return-statements
def _single_qobj_expect(oper, state):
    """
    Private function used by expect to calculate expectation values of Qobjs.
    """
    if oper.isoper:
        if oper.dims[1] != state.dims[0]:
            raise Exception('Operator and state do not have same tensor ' +
                            'structure: %s and %s' %
                            (oper.dims[1], state.dims[0]))

        if state.type == 'oper':
            # calculates expectation value via TR(op*rho)
            return cy_spmm_tr(oper.data, state.data,
                              oper.isherm and state.isherm)

        elif state.type == 'ket':
            # calculates expectation value via <psi|op|psi>
            return expect_csr_ket(oper.data, state.data,
                                  oper.isherm)
    else:
        raise TypeError('Invalid operand types')


def variance(oper, state):
    """
    Variance of an operator for the given state vector or density matrix.

    Args:
        oper (qobj.Qobj): Operator for expectation value.
        state (qobj.Qobj or list): A single or `list` of quantum states or density matrices..

    Returns:
        float: Variance of operator 'oper' for given state.
    """
    return expect(oper ** 2, state) - expect(oper, state) ** 2
