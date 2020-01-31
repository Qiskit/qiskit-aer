# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3
#distutils: language = c++

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

@cython.boundscheck(False)
def write_memory(unsigned char[:, ::1] mem,
                 unsigned int[::1] memory_slots,
                 double[::1] probs,
                 double[::1] rand_vals):
    """Writes the results of measurements to memory
    in place.

    Args:
        mem (int8 array): 2D array of memory of size (shots*memory_slots).
        memory_slots (uint32 array): Array of ints for memory_slots to write too.
        probs (double array): Probability of being in excited state for
            each qubit in `qubits`.
        rand_vals (double array): Random values of len = len(qubits)*shots
    """
    cdef unsigned int nrows = mem.shape[0]
    cdef unsigned int nprobs = probs.shape[0]
    
    cdef size_t ii, jj
    cdef unsigned char temp
    for ii in range(nrows):
        for jj in range(nprobs):
            temp = <unsigned char>(probs[jj] > rand_vals[nprobs*ii+jj])
            if temp:
                mem[ii,memory_slots[jj]] = temp