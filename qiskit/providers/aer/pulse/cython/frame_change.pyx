# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3
#distutils: language = c++

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

cimport cython

cdef extern from "<complex>" namespace "std" nogil:
    double complex exp(double complex x)

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void compute_fc_value(double t, int chan_idx, double[::1] fc_array, 
                 complex[::1] cnt_fc,
                 unsigned int[::1] fc_idx, 
                 unsigned int[::1] register):
    """
    Computes the frame change value at time `t` on the
    channel labeled by `chan_idx`.  The result is stored
    in the current frame change array at `cnt_fc[chan_idx]`.
    
    Args:
        t (double): Current time.
        chan_idx (int): Index of channel.
        fc_array (ndarray): Frame change array of doubles.
        cnt_fc (ndarray): Complex values for current frame change values.
        fc_idx (ndarray): Ints for frame change index.
        register (ndarray): Classical register of ints.
        
    """
    cdef unsigned int arr_len = fc_array.shape[0]
    cdef unsigned int do_fc
    
    if fc_idx[chan_idx] < arr_len:
        while t >= fc_array[fc_idx[chan_idx]]:
            do_fc = 1
            # Check if FC is conditioned on register
            if fc_array[fc_idx[chan_idx]+2] >= 0:
                # If condition not satisfied no do FC
                if not register[<int>fc_array[fc_idx[chan_idx]+2]]:
                    do_fc = 0
            if do_fc:   
                # Update the frame change value
                cnt_fc[chan_idx] *= exp(1j*fc_array[fc_idx[chan_idx]+1])
            # update the index
            fc_idx[chan_idx] += 3
            # check if we hit the end
            if fc_idx[chan_idx] == arr_len:
                break
                
                
def check_fc_compute(double t, int chan_idx, double[::1] fc_array, 
                 complex[::1] cnt_fc,
                 unsigned int[::1] fc_idx, 
                 unsigned int[::1] register):
    """
    Python function to check the compute_fc_value is working.
    """
    compute_fc_value(t, chan_idx, fc_array, cnt_fc, fc_idx, register)