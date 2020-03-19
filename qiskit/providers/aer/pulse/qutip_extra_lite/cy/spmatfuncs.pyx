#!python
#cython: language_level=3
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
cimport numpy as cnp
cimport cython
cimport libc.math
from libcpp cimport bool

cdef extern from "src/zspmv.hpp" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows)

include "complex_math.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv_csr(complex[::1] data,
            int[::1] ind, int[::1] ptr, complex[::1] vec):
    """
    Sparse matrix, dense vector multiplication.
    Here the vector is assumed to have one-dimension.
    Matrix must be in CSR format and have complex entries.

    Parameters
    ----------
    data : array
        Data for sparse matrix.
    idx : array
        Indices for sparse matrix data.
    ptr : array
        Pointers for sparse matrix data.
    vec : array
        Dense vector for multiplication.  Must be one-dimensional.

    Returns
    -------
    out : array
        Returns dense array.

    """
    cdef unsigned int num_rows = ptr.shape[0] - 1
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = np.zeros((num_rows), dtype=np.complex)
    zspmvpy(&data[0], &ind[0], &ptr[0], &vec[0], 1.0, &out[0], num_rows)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_expect_psi_csr(complex[::1] data,
                        int[::1] ind,
                        int[::1] ptr,
                        complex[::1] vec,
                        bool isherm):

    cdef size_t row, jj
    cdef int nrows = vec.shape[0]
    cdef complex expt = 0, temp, cval

    for row in range(nrows):
        cval = conj(vec[row])
        temp = 0
        for jj in range(ptr[row], ptr[row+1]):
            temp += data[jj]*vec[ind[jj]]
        expt += cval*temp

    if isherm :
        return real(expt)
    else:
        return expt