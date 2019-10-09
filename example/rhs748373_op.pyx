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

include '/anaconda3/envs/aer37/lib/python3.7/site-packages/qiskit/providers/aer/openpulse/qutip_lite/cy/complex_math.pxi'

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_td_ode_rhs(
        double t,
        complex[::1] vec,
        double[::1] energ,
        complex[::1] data0, int[::1] idx0, int[::1] ptr0,
        complex[::1] data1, int[::1] idx1, int[::1] ptr1,
        complex[::1] data2, int[::1] idx2, int[::1] ptr2,
        complex[::1] data3, int[::1] idx3, int[::1] ptr3,
        complex[::1] data4, int[::1] idx4, int[::1] ptr4,
        complex[::1] data5, int[::1] idx5, int[::1] ptr5,
        complex[::1] data6, int[::1] idx6, int[::1] ptr6,
        complex[::1] data7, int[::1] idx7, int[::1] ptr7,
        complex[::1] data8, int[::1] idx8, int[::1] ptr8,
        complex[::1] data9, int[::1] idx9, int[::1] ptr9,
        complex[::1] pulse_array,
        unsigned int[::1] pulse_indices,
        double[::1] D0_pulses,
        double[::1] D0_fc,
        double[::1] U0_pulses,
        double[::1] U0_fc,
        double[::1] D1_pulses,
        double[::1] D1_fc,
        double[::1] U1_pulses,
        double[::1] U1_fc,
        complex v0,
        complex v1,
        complex j,
        complex r,
        complex alpha0,
        complex alpha1,
        double D0_freq,
        double U0_freq,
        double D1_freq,
        double U1_freq,
        unsigned char[::1] register):
    
    cdef size_t row, jj
    cdef unsigned int row_start, row_end
    cdef unsigned int num_rows = vec.shape[0]
    cdef double complex dot, osc_term, coef
    cdef double complex * out = <complex *>PyDataMem_NEW_ZEROED(num_rows,sizeof(complex))
    
    cdef double complex D0
    cdef double complex U0
    cdef double complex D1
    cdef double complex U1
    cdef double complex td0
    cdef double complex td1
    cdef double complex td2
    cdef double complex td3
    cdef double complex td4
    cdef double complex td5
    cdef double complex td6
    cdef double complex td7
    cdef double complex td8
    cdef double complex td9
    
    # Compute complex channel values at time `t`
    D0 = chan_value(t, 0, D0_freq, D0_pulses,  pulse_array, pulse_indices, D0_fc, register)
    U0 = chan_value(t, 1, U0_freq, U0_pulses,  pulse_array, pulse_indices, U0_fc, register)
    D1 = chan_value(t, 2, D1_freq, D1_pulses,  pulse_array, pulse_indices, D1_fc, register)
    U1 = chan_value(t, 3, U1_freq, U1_pulses,  pulse_array, pulse_indices, U1_fc, register)
    
    # Eval the time-dependent terms and do SPMV.
    td0 = np.pi*(2*v0-alpha0)
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
    td1 = np.pi*alpha0
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
    td2 = np.pi*(2*v1-alpha1)
    if abs(td2) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr2[row];
            row_end = ptr2[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx2[jj]])*t)
                if row<idx2[jj]:
                    coef = conj(td2)
                else:
                    coef = td2
                dot += coef*osc_term*data2[jj]*vec[idx2[jj]];
            out[row] += dot;
    td3 = np.pi*alpha1
    if abs(td3) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr3[row];
            row_end = ptr3[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx3[jj]])*t)
                if row<idx3[jj]:
                    coef = conj(td3)
                else:
                    coef = td3
                dot += coef*osc_term*data3[jj]*vec[idx3[jj]];
            out[row] += dot;
    td4 = 2*np.pi*j
    if abs(td4) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr4[row];
            row_end = ptr4[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx4[jj]])*t)
                if row<idx4[jj]:
                    coef = conj(td4)
                else:
                    coef = td4
                dot += coef*osc_term*data4[jj]*vec[idx4[jj]];
            out[row] += dot;
    td5 = 2*np.pi*r*D0
    if abs(td5) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr5[row];
            row_end = ptr5[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx5[jj]])*t)
                if row<idx5[jj]:
                    coef = conj(td5)
                else:
                    coef = td5
                dot += coef*osc_term*data5[jj]*vec[idx5[jj]];
            out[row] += dot;
    td6 = 2*np.pi*r*U1
    if abs(td6) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr6[row];
            row_end = ptr6[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx6[jj]])*t)
                if row<idx6[jj]:
                    coef = conj(td6)
                else:
                    coef = td6
                dot += coef*osc_term*data6[jj]*vec[idx6[jj]];
            out[row] += dot;
    td7 = 2*np.pi*r*U0
    if abs(td7) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr7[row];
            row_end = ptr7[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx7[jj]])*t)
                if row<idx7[jj]:
                    coef = conj(td7)
                else:
                    coef = td7
                dot += coef*osc_term*data7[jj]*vec[idx7[jj]];
            out[row] += dot;
    td8 = 2*np.pi*r*D1
    if abs(td8) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr8[row];
            row_end = ptr8[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx8[jj]])*t)
                if row<idx8[jj]:
                    coef = conj(td8)
                else:
                    coef = td8
                dot += coef*osc_term*data8[jj]*vec[idx8[jj]];
            out[row] += dot;
    td9 = 1.0
    if abs(td9) > 1e-15:
        for row in range(num_rows):
            dot = 0;
            row_start = ptr9[row];
            row_end = ptr9[row+1];
            for jj in range(row_start,row_end):
                osc_term = exp(1j*(energ[row]-energ[idx9[jj]])*t)
                if row<idx9[jj]:
                    coef = conj(td9)
                else:
                    coef = td9
                dot += coef*osc_term*data9[jj]*vec[idx9[jj]];
            out[row] += dot;
    for row in range(num_rows):
        out[row] += 1j*energ[row]*vec[row];
    
    # Convert to NumPy array, grab ownership, and return.
    cdef np.npy_intp dims = num_rows
    cdef np.ndarray[complex, ndim=1, mode='c'] arr_out = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_COMPLEX128, out)
    PyArray_ENABLEFLAGS(arr_out, np.NPY_OWNDATA)
    return arr_out
