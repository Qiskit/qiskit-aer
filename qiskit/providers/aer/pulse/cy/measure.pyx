# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3
#distutils: language = c++

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

cimport cython
import numpy as np
cimport numpy as np
from qiskit.providers.aer.pulse.qutip_lite.cy.spmatfuncs import cy_expect_psi_csr

@cython.boundscheck(False)
def occ_probabilities(unsigned int[::1] qubits, complex[::1] state, list meas_ops):
    """Computes the occupation probabilities of the specifed qubits for
    the given state.

    Args:
        qubits (int array): Ints labelling which qubits are to be measured.
    """

    cdef unsigned int num_qubits = <unsigned int>qubits.shape[0]
    cdef np.ndarray[double, ndim=1, mode="c"] probs = np.zeros(qubits.shape[0], dtype=float)

    cdef size_t kk
    cdef object oper

    for kk in range(num_qubits):
        oper = meas_ops[qubits[kk]]
        probs[kk] = cy_expect_psi_csr(oper.data.data,
                                      oper.data.indices,
                                      oper.data.indptr,
                                      state, 1)

    return probs

@cython.boundscheck(False)
def write_shots_memory(unsigned char[:, ::1] mem,
                 unsigned int[::1] mem_slots,
                 double[::1] probs,
                 double[::1] rand_vals):

    """Converts probabilities back into shots

    Args:
        mem
        mem_slots
        probs: expectation value
        rand_vals: random values used to convert back into shots
    """

    cdef unsigned int nrows = mem.shape[0]
    cdef unsigned int nprobs = probs.shape[0]

    cdef size_t ii, jj
    cdef unsigned char temp
    for ii in range(nrows):
        for jj in range(nprobs):
            temp = <unsigned char>(probs[jj] > rand_vals[nprobs*ii+jj])
            if temp:
                mem[ii,mem_slots[jj]] = temp