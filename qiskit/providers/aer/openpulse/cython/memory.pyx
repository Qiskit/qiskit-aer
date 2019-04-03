# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3
#distutils: language = c++

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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
            temp = <unsigned char>(probs[jj] < rand_vals[nprobs*ii+jj])
            if temp:
                mem[ii,memory_slots[jj]] = temp