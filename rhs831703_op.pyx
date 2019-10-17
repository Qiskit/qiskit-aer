#!python
#cython: language_level=3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "<complex>" namespace "std" nogil:
    double complex exp(double complex x)

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

from qiskit.providers.aer.openpulse.qutip_lite.cy.spmatfuncs cimport spmvpy
from qiskit.providers.aer.openpulse.qutip_lite.cy.math cimport erf
from libc.math cimport pi

from qiskit.providers.aer.openpulse.cy.channel_value cimport chan_value

include '/Users/zacharyschoenfeld/Documents/Dev_OpenSource/qiskit-aer/qiskit/providers/aer/openpulse/qutip_lite/cy/complex_math.pxi'

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_td_ode_rhs(
        double t,
        complex[::1] vec,
        double[::1] energ,
        complex[::1] data0, int[::1] idx0, int[::1] ptr0,
        complex[::1] data1, int[::1] idx1, int[::1] ptr1,
        complex[::1] pulse_array,
        unsigned int[::1] pulse_indices,
        double[::1] D0_pulses,
        double[::1] D0_fc,
        complex omega0,
        complex omega1,
        double D0_freq,
        unsigned char[::1] register):

    cdef size_t row, jj
    cdef unsigned int row_start, row_end
    cdef unsigned int num_rows = vec.shape[0]
    cdef double complex dot, osc_term, coef
    cdef double complex * out = <complex *>PyDataMem_NEW_ZEROED(num_rows,sizeof(complex))

    cdef double complex D0
    cdef double complex td0
    cdef double complex td1
    cdef double complex td2

    # Compute complex channel values at time `t`
    D0 = chan_value(t, 0, D0_freq, D0_pulses,  pulse_array, pulse_indices, D0_fc, register)

    # Eval the time-dependent terms and do SPMV.
    td0 = -0.5*omega0
    if abs(td0) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr0[row];
            row_end = ptr0[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx0[jj]])*t)
                if row<idx0[jj]:
                    coef = conj(td0)
                else:
                    coef = td0
                dot += coef*osc_term*data0[jj]*vec[idx0[jj]];
            out[row] += dot;
    td1 = 0.5*omega1*D0
    if abs(td1) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr1[row];
            row_end = ptr1[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx1[jj]])*t)
                if row<idx1[jj]:
                    coef = conj(td1)
                else:
                    coef = td1
                dot += coef*osc_term*data1[jj]*vec[idx1[jj]];
            out[row] += dot;
    for row in range(num_rows):
        out[row] += 1j*energ[row]*vec[row];

    # Convert to NumPy array, grab ownership, and return.
    cdef np.npy_intp dims = num_rows
    cdef np.ndarray[complex, ndim=1, mode='c'] arr_out = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_COMPLEX128, out)
    PyArray_ENABLEFLAGS(arr_out, np.NPY_OWNDATA)
    return arr_out
