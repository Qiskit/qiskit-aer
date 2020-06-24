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
Internal use module for manipulating dims specifications.
"""

__all__ = []
# Everything should be explicitly imported, not made available
# by default.

import numpy as np


def flatten(the_list):
    """Flattens a list of lists to the first level.

    Given a list containing a mix of scalars and lists,
    flattens down to a list of the scalars within the original
    list.

    Args:
        the_list (list): Input list

    Returns:
        list: Flattened list.

    """
    if not isinstance(the_list, list):
        return [the_list]
    else:
        return sum(map(flatten, the_list), [])


def is_scalar(dims):
    """
    Returns True if a dims specification is effectively
    a scalar (has dimension 1).
    """
    return np.prod(flatten(dims)) == 1


def is_vector(dims):
    """Is a vector"""
    return (
        isinstance(dims, list) and
        isinstance(dims[0], (int, np.integer))
    )


def is_vectorized_oper(dims):
    """Is a vectorized operator."""
    return (
        isinstance(dims, list) and
        isinstance(dims[0], list)
    )


# pylint: disable=too-many-return-statements
def type_from_dims(dims, enforce_square=True):
    """Get the type of operator from dims structure"""
    bra_like, ket_like = map(is_scalar, dims)

    if bra_like:
        if is_vector(dims[1]):
            return 'bra'
        elif is_vectorized_oper(dims[1]):
            return 'operator-bra'

    if ket_like:
        if is_vector(dims[0]):
            return 'ket'
        elif is_vectorized_oper(dims[0]):
            return 'operator-ket'

    elif is_vector(dims[0]) and (dims[0] == dims[1] or not enforce_square):
        return 'oper'

    elif (
            is_vectorized_oper(dims[0]) and
            (
                (
                    dims[0] == dims[1] and
                    dims[0][0] == dims[1][0]
                ) or not enforce_square
            )
    ):
        return 'super'

    return 'other'
