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
Internal use module for manipulating dims specifications.
"""

__all__ = []
# Everything should be explicitly imported, not made available
# by default.

import numpy as np


def flatten(l):
    """Flattens a list of lists to the first level.

    Given a list containing a mix of scalars and lists,
    flattens down to a list of the scalars within the original
    list.

    Args:
        l (list): Input list

    Returns:
        list: Flattened list.

    """
    if not isinstance(l, list):
        return [l]
    else:
        return sum(map(flatten, l), [])


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


def _enumerate_flat(l, idx=0):
    if not isinstance(l, list):
        # Found a scalar, so return and increment.
        return idx, idx + 1
    else:
        # Found a list, so append all the scalars
        # from it and recurse to keep the increment
        # correct.
        acc = []
        for elem in l:
            labels, idx = _enumerate_flat(elem, idx)
            acc.append(labels)
        return acc, idx


def _collapse_composite_index(dims):
    """
    Given the dimensions specification for a composite index
    (e.g.: [2, 3] for the right index of a ket with dims [[1], [2, 3]]),
    returns a dimensions specification for an index of the same shape,
    but collapsed to a single "leg." In the previous example, [2, 3]
    would collapse to [6].
    """
    return [np.prod(dims)]


def _collapse_dims_to_level(dims, level=1):
    """
    Recursively collapses all indices in a dimensions specification
    appearing at a given level, such that the returned dimensions
    specification does not represent any composite systems.
    """
    if level == 0:
        return _collapse_composite_index(dims)
    else:
        return [_collapse_dims_to_level(index, level=level - 1) for index in dims]


def collapse_dims_super(dims):
    """
    Given the dimensions specifications for an operator-ket-, operator-bra- or
    super-type Qobj, returns a dimensions specification describing the same shape
    by collapsing all composite systems. For instance, the super-type
    dimensions specification ``[[[2, 3], [2, 3]], [[2, 3], [2, 3]]]`` collapses to
    ``[[[6], [6]], [[6], [6]]]``.

    Args:
        dims (list): Dimensions specifications to be collapsed.

    Returns:
        list: Collapsed dimensions specification describing the same shape
              such that ``len(collapsed_dims[i][j]) == 1`` for ``i`` and ``j``
              in ``range(2)``.
    """
    return _collapse_dims_to_level(dims, 2)


def enumerate_flat(l):
    """Labels the indices at which scalars occur in a flattened list.

    Given a list containing a mix of scalars and lists,
    returns a list of the same structure, where each scalar
    has been replaced by an index into the flattened list.

    Examples
    --------

    >>> print(enumerate_flat([[[10], [20, 30]], 40]))
    [[[0], [1, 2]], 3]

    """
    return _enumerate_flat(l)[0]
