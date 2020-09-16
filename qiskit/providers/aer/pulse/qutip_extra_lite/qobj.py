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
# pylint: disable=invalid-name, redefined-outer-name, no-name-in-module
# pylint: disable=import-error, unused-import

"""The Quantum Object (Qobj) class, for representing quantum states and
operators, and related functions.
"""

__all__ = ['Qobj']

import warnings
import builtins

# import math functions from numpy.math: required for td string evaluation
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from qiskit.providers.aer.version import __version__

from .dimensions import type_from_dims

# used in existing functions that break if removed
from .cy.spmath import (zcsr_adjoint, zcsr_isherm)
from .fastsparse import fast_csr_matrix, fast_identity

# general absolute tolerance
atol = 1e-12


class Qobj():
    """A class for representing quantum objects, such as quantum operators
    and states.

    The Qobj class is the QuTiP representation of quantum operators and state
    vectors. This class also implements math operations +,-,* between Qobj
    instances (and / by a C-number), as well as a collection of common
    operator/state operations.  The Qobj constructor optionally takes a
    dimension ``list`` and/or shape ``list`` as arguments.

    Attributes
    ----------
    data : array_like
        Sparse matrix characterizing the quantum object.
    dims : list
        List of dimensions keeping track of the tensor structure.
    shape : list
        Shape of the underlying `data` array.
    type : str
        Type of quantum object: 'bra', 'ket', 'oper', 'operator-ket',
        'operator-bra', or 'super'.
    superrep : str
        Representation used if `type` is 'super'. One of 'super'
        (Liouville form) or 'choi' (Choi matrix with tr = dimension).
    isherm : bool
        Indicates if quantum object represents Hermitian operator.
    isunitary : bool
        Indictaes if quantum object represents unitary operator.
    iscp : bool
        Indicates if the quantum object represents a map, and if that map is
        completely positive (CP).
    ishp : bool
        Indicates if the quantum object represents a map, and if that map is
        hermicity preserving (HP).
    istp : bool
        Indicates if the quantum object represents a map, and if that map is
        trace preserving (TP).
    iscptp : bool
        Indicates if the quantum object represents a map that is completely
        positive and trace preserving (CPTP).
    isket : bool
        Indicates if the quantum object represents a ket.
    isbra : bool
        Indicates if the quantum object represents a bra.
    isoper : bool
        Indicates if the quantum object represents an operator.
    issuper : bool
        Indicates if the quantum object represents a superoperator.
    isoperket : bool
        Indicates if the quantum object represents an operator in column vector
        form.
    isoperbra : bool
        Indicates if the quantum object represents an operator in row vector
        form.

    Methods
    -------
    copy()
        Create copy of Qobj
    conj()
        Conjugate of quantum object.
    cosm()
        Cosine of quantum object.
    dag()
        Adjoint (dagger) of quantum object.
    dnorm()
        Diamond norm of quantum operator.
    dual_chan()
        Dual channel of quantum object representing a CP map.
    eigenenergies(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies (eigenvalues) of a quantum object.
    eigenstates(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies and eigenstates of quantum object.
    expm()
        Matrix exponential of quantum object.
    full(order='C')
        Returns dense array of quantum object `data` attribute.
    groundstate(sparse=False, tol=0, maxiter=100000)
        Returns eigenvalue and eigenket for the groundstate of a quantum
        object.
    matrix_element(bra, ket)
        Returns the matrix element of operator between `bra` and `ket` vectors.
    norm(norm='tr', sparse=False, tol=0, maxiter=100000)
        Returns norm of a ket or an operator.
    proj()
        Computes the projector for a ket or bra vector.
    sinm()
        Sine of quantum object.
    sqrtm()
        Matrix square root of quantum object.
    tidyup(atol=1e-12)
        Removes small elements from quantum object.
    tr()
        Trace of quantum object.
    trans()
        Transpose of quantum object.
    transform(inpt, inverse=False)
        Performs a basis transformation defined by `inpt` matrix.
    trunc_neg(method='clip')
        Removes negative eigenvalues and returns a new Qobj that is
        a valid density operator.
    unit(norm='tr', sparse=False, tol=0, maxiter=100000)
        Returns normalized quantum object.

    """
    __array_priority__ = 100  # sets Qobj priority above numpy arrays

    # pylint: disable=dangerous-default-value, redefined-builtin
    def __init__(self, inpt=None, dims=[[], []], shape=[],
                 type=None, isherm=None, copy=True,
                 fast=False, superrep=None, isunitary=None):
        """
        Qobj constructor.

        Args:
            inpt (ndarray): Input array or matrix data.
            dims (list): List of Qobj dims.
            shape (list):  shape of underlying data.
            type (str): Is object a ket, bra, oper, super.
            isherm (bool): Is object Hermitian.
            copy (bool): Copy input data.
            fast (str or bool): Fast object instantiation.
            superrep (str): Type of super representaiton.
            isunitary (bool): Is object unitary.

        Raises:
            Exception: Something bad happened.
        """

        self._isherm = isherm
        self._type = type
        self.superrep = superrep
        self._isunitary = isunitary

        if fast == 'mc':
            # fast Qobj construction for use in mcsolve with ket output
            self._data = inpt
            self.dims = dims
            self._isherm = False
            return

        if fast == 'mc-dm':
            # fast Qobj construction for use in mcsolve with dm output
            self._data = inpt
            self.dims = dims
            self._isherm = True
            return

        if isinstance(inpt, Qobj):
            # if input is already Qobj then return identical copy

            self._data = fast_csr_matrix((inpt.data.data, inpt.data.indices,
                                          inpt.data.indptr),
                                         shape=inpt.shape, copy=copy)

            if not np.any(dims):
                # Dimensions of quantum object used for keeping track of tensor
                # components
                self.dims = inpt.dims
            else:
                self.dims = dims

            self.superrep = inpt.superrep
            self._isunitary = inpt._isunitary

        elif inpt is None:
            # initialize an empty Qobj with correct dimensions and shape

            if any(dims):
                N, M = np.prod(dims[0]), np.prod(dims[1])
                self.dims = dims

            elif shape:
                N, M = shape
                self.dims = [[N], [M]]

            else:
                N, M = 1, 1
                self.dims = [[N], [M]]

            self._data = fast_csr_matrix(shape=(N, M))

        elif isinstance(inpt, (list, tuple)):
            # case where input is a list
            data = np.array(inpt)
            if len(data.shape) == 1:
                # if list has only one dimension (i.e [5,4])
                data = data.transpose()

            _tmp = sp.csr_matrix(data, dtype=complex)
            self._data = fast_csr_matrix((_tmp.data, _tmp.indices, _tmp.indptr),
                                         shape=_tmp.shape)
            if not np.any(dims):
                self.dims = [[int(data.shape[0])], [int(data.shape[1])]]
            else:
                self.dims = dims

        elif isinstance(inpt, np.ndarray) or sp.issparse(inpt):
            # case where input is array or sparse
            if inpt.ndim == 1:
                inpt = inpt[:, np.newaxis]

            do_copy = copy
            if not isinstance(inpt, fast_csr_matrix):
                _tmp = sp.csr_matrix(inpt, dtype=complex, copy=do_copy)
                _tmp.sort_indices()  # Make sure indices are sorted.
                do_copy = 0
            else:
                _tmp = inpt
            self._data = fast_csr_matrix((_tmp.data, _tmp.indices, _tmp.indptr),
                                         shape=_tmp.shape, copy=do_copy)

            if not np.any(dims):
                self.dims = [[int(inpt.shape[0])], [int(inpt.shape[1])]]
            else:
                self.dims = dims

        elif isinstance(inpt, (int, float, complex,
                               np.integer, np.floating, np.complexfloating)):
            # if input is int, float, or complex then convert to array
            _tmp = sp.csr_matrix([[inpt]], dtype=complex)
            self._data = fast_csr_matrix((_tmp.data, _tmp.indices, _tmp.indptr),
                                         shape=_tmp.shape)
            if not np.any(dims):
                self.dims = [[1], [1]]
            else:
                self.dims = dims

        else:
            warnings.warn("Initializing Qobj from unsupported type: %s" %
                          builtins.type(inpt))
            inpt = np.array([[0]])
            _tmp = sp.csr_matrix(inpt, dtype=complex, copy=copy)
            self._data = fast_csr_matrix((_tmp.data, _tmp.indices, _tmp.indptr),
                                         shape=_tmp.shape)
            self.dims = [[int(inpt.shape[0])], [int(inpt.shape[1])]]

        if type == 'super':
            # Type is not super, i.e. dims not explicitly passed, but oper shape
            if dims == [[], []] and self.shape[0] == self.shape[1]:
                sub_shape = np.sqrt(self.shape[0])
                # check if root of shape is int
                if (sub_shape % 1) != 0:
                    raise Exception('Invalid shape for a super operator.')

                sub_shape = int(sub_shape)
                self.dims = [[[sub_shape], [sub_shape]]] * 2

        if superrep:
            self.superrep = superrep
        else:
            if self.type == 'super' and self.superrep is None:
                self.superrep = 'super'

        # clear type cache
        self._type = None

    def copy(self):
        """Create identical copy"""
        return Qobj(inpt=self)

    def get_data(self):
        """Gets underlying data."""
        return self._data

    # Here we perfrom a check of the csr matrix type during setting of Q.data
    def set_data(self, data):
        """Data setter
        """
        if not isinstance(data, fast_csr_matrix):
            raise TypeError('Qobj data must be in fast_csr format.')

        self._data = data
    data = property(get_data, set_data)

    def __add__(self, other):
        """
        ADDITION with Qobj on LEFT [ ex. Qobj+4 ]
        """
        self._isunitary = None

        if not isinstance(other, Qobj):
            if isinstance(other, (int, float, complex, np.integer,
                                  np.floating, np.complexfloating, np.ndarray,
                                  list, tuple)) or sp.issparse(other):
                other = Qobj(other)
            else:
                return NotImplemented

        if np.prod(other.shape) == 1 and np.prod(self.shape) != 1:
            # case for scalar quantum object
            dat = other.data[0, 0]
            if dat == 0:
                return self

            out = Qobj()

            if self.type in ['oper', 'super']:
                out.data = self.data + dat * fast_identity(
                    self.shape[0])
            else:
                out.data = self.data
                out.data.data = out.data.data + dat

            out.dims = self.dims

            if isinstance(dat, (int, float)):
                out._isherm = self._isherm
            else:
                # We use _isherm here to prevent recalculating on self and
                # other, relying on that bool(None) == False.
                out._isherm = (True if self._isherm and other._isherm
                               else out.isherm)

            out.superrep = self.superrep

            return out

        elif np.prod(self.shape) == 1 and np.prod(other.shape) != 1:
            # case for scalar quantum object
            dat = self.data[0, 0]
            if dat == 0:
                return other

            out = Qobj()
            if other.type in ['oper', 'super']:
                out.data = dat * fast_identity(other.shape[0]) + other.data
            else:
                out.data = other.data
                out.data.data = out.data.data + dat
            out.dims = other.dims

            if isinstance(dat, complex):
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm

            out.superrep = self.superrep

            return out

        elif self.dims != other.dims:
            raise TypeError('Incompatible quantum object dimensions')

        elif self.shape != other.shape:
            raise TypeError('Matrix shapes do not match')

        else:  # case for matching quantum objects
            out = Qobj()
            out.data = self.data + other.data
            out.dims = self.dims

            if self.type in ['ket', 'bra', 'operator-ket', 'operator-bra']:
                out._isherm = False
            elif self._isherm is None or other._isherm is None:
                out._isherm = out.isherm
            elif not self._isherm and not other._isherm:
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm and other._isherm

            if self.superrep and other.superrep:
                if self.superrep != other.superrep:
                    msg = ("Adding superoperators with different " +
                           "representations")
                    warnings.warn(msg)

                out.superrep = self.superrep

            return out

    def __radd__(self, other):
        """
        ADDITION with Qobj on RIGHT [ ex. 4+Qobj ]
        """
        return self + other

    def __sub__(self, other):
        """
        SUBTRACTION with Qobj on LEFT [ ex. Qobj-4 ]
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        SUBTRACTION with Qobj on RIGHT [ ex. 4-Qobj ]
        """
        return (-self) + other

    # used
    # pylint: disable=too-many-return-statements
    def __mul__(self, other):
        """
        MULTIPLICATION with Qobj on LEFT [ ex. Qobj*4 ]
        """
        self._isunitary = None

        if isinstance(other, Qobj):
            if self.dims[1] == other.dims[0]:
                out = Qobj()
                out.data = self.data * other.data
                dims = [self.dims[0], other.dims[1]]
                out.dims = dims
                out.dims = dims

                out._isherm = None

                if self.superrep and other.superrep:
                    if self.superrep != other.superrep:
                        msg = ("Multiplying superoperators with different " +
                               "representations")
                        warnings.warn(msg)

                    out.superrep = self.superrep

                return out

            elif np.prod(self.shape) == 1:
                out = Qobj(other)
                out.data *= self.data[0, 0]
                out.superrep = other.superrep
                return out

            elif np.prod(other.shape) == 1:
                out = Qobj(self)
                out.data *= other.data[0, 0]
                out.superrep = self.superrep
                return out

            else:
                raise TypeError("Incompatible Qobj shapes")

        elif isinstance(other, np.ndarray):
            if other.dtype == 'object':
                return np.array([self * item for item in other],
                                dtype=object)
            else:
                return self.data * other

        elif isinstance(other, list):
            # if other is a list, do element-wise multiplication
            return np.array([self * item for item in other],
                            dtype=object)

        elif isinstance(other, (int, float, complex,
                                np.integer, np.floating, np.complexfloating)):
            out = Qobj()
            out.data = self.data * other
            out.dims = self.dims
            out.superrep = self.superrep
            if isinstance(other, complex):
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm

            return out

        else:
            return NotImplemented

    # keep for now
    def __rmul__(self, other):
        """
        MULTIPLICATION with Qobj on RIGHT [ ex. 4*Qobj ]
        """
        if isinstance(other, np.ndarray):
            if other.dtype == 'object':
                return np.array([item * self for item in other],
                                dtype=object)
            else:
                return other * self.data

        elif isinstance(other, list):
            # if other is a list, do element-wise multiplication
            return np.array([item * self for item in other],
                            dtype=object)

        elif isinstance(other, (int, float, complex,
                                np.integer, np.floating,
                                np.complexfloating)):
            out = Qobj()
            out.data = other * self.data
            out.dims = self.dims
            out.superrep = self.superrep
            if isinstance(other, complex):
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm

            return out

        else:
            raise TypeError("Incompatible object for multiplication")

    # keep for now
    def __truediv__(self, other):
        return self.__div__(other)

    # keep for now
    def __div__(self, other):
        """
        DIVISION (by numbers only)
        """
        if isinstance(other, Qobj):  # if both are quantum objects
            raise TypeError("Incompatible Qobj shapes " +
                            "[division with Qobj not implemented]")

        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            out = Qobj()
            out.data = self.data / other
            out.dims = self.dims
            if isinstance(other, complex):
                out._isherm = out.isherm
            else:
                out._isherm = self._isherm

            out.superrep = self.superrep

            return out

        else:
            raise TypeError("Incompatible object for division")

    # keep for now
    def __neg__(self):
        """
        NEGATION operation.
        """
        out = Qobj()
        out.data = -self.data
        out.dims = self.dims
        out.superrep = self.superrep
        out._isherm = self._isherm
        out._isunitary = self._isunitary
        return out

    # needed by qobj_generators
    def __getitem__(self, ind):
        """
        GET qobj elements.
        """
        out = self.data[ind]
        if sp.issparse(out):
            return np.asarray(out.todense())
        else:
            return out

    # keep for now
    def __eq__(self, other):
        """
        EQUALITY operator.
        """
        return bool(isinstance(other, Qobj) and
                    self.dims == other.dims and
                    not np.any(np.abs((self.data - other.data).data) > atol))

    # keep for now
    def __ne__(self, other):
        """
        INEQUALITY operator.
        """
        return not self == other

    # not needed functionally but is useful to keep for now
    def __str__(self):
        s = ""
        t = self.type
        shape = self.shape
        if self.type in ['oper', 'super']:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t +
                  ", isherm = " + str(self.isherm) +
                  (
                      ", superrep = {0.superrep}".format(self)
                      if t == "super" and self.superrep != "super"
                      else ""
                  ) + "\n")
        else:
            s += ("Quantum object: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(shape) +
                  ", type = " + t + "\n")
        s += "Qobj data =\n"

        if shape[0] > 10000 or shape[1] > 10000:
            # if the system is huge, don't attempt to convert to a
            # dense matrix and then to string, because it is pointless
            # and is likely going to produce memory errors. Instead print the
            # sparse data string representation
            s += str(self.data)

        elif all(np.imag(self.data.data) == 0):
            s += str(np.real(self.full()))

        else:
            s += str(self.full())

        return s

    # used by states
    def dag(self):
        """Adjoint operator of quantum object.
        """
        out = Qobj()
        out.data = zcsr_adjoint(self.data)
        out.dims = [self.dims[1], self.dims[0]]
        out._isherm = self._isherm
        out.superrep = self.superrep
        return out

    # breaks if removed - used in hamiltonian_model
    def full(self, order='C', squeeze=False):
        """Dense array from quantum object.

        Parameters
        ----------
        order : str {'C', 'F'}
            Return array in C (default) or Fortran ordering.
        squeeze : bool {False, True}
            Squeeze output array.

        Returns
        -------
        data : array
            Array of complex data from quantum objects `data` attribute.
        """
        if squeeze:
            return self.data.toarray(order=order).squeeze()
        else:
            return self.data.toarray(order=order)

    # breaks if removed - only ever actually used in tests for duffing_model_generators
    # pylint: disable=unused-argument
    def __array__(self, *arg, **kwarg):
        """Numpy array from Qobj
        For compatibility with np.array
        """
        return self.full()

    # breaks if removed due to tensor
    @property
    def isherm(self):
        """Is operator Hermitian.

        Returns:
            bool: Operator is Hermitian or not.
        """

        if self._isherm is not None:
            # used previously computed value
            return self._isherm

        self._isherm = bool(zcsr_isherm(self.data))

        return self._isherm

    # breaks if removed due to tensor
    @isherm.setter
    def isherm(self, isherm):
        self._isherm = isherm

    @property
    def type(self):
        """Type of Qobj
        """
        if not self._type:
            self._type = type_from_dims(self.dims)

        return self._type

    @property
    def shape(self):
        """Shape of Qobj
        """
        if self.data.shape == (1, 1):
            return tuple([np.prod(self.dims[0]), np.prod(self.dims[1])])
        else:
            return tuple(self.data.shape)

    # breaks if removed - called in pulse_controller
    # when initial state being set
    @property
    def isket(self):
        """Is ket vector"""
        return self.type == 'ket'

    # breaks if removed - called in tensor
    @property
    def issuper(self):
        """Is super operator"""
        return self.type == 'super'
