# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3
#distutils: language = c++

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

cimport cython
from libc.math cimport floor

@cython.cdivision(True)
cdef inline int get_arr_idx(double t, double start, double stop, int len_arr):
    """
    Computes the index corresponding to time `t` in [`start`, 'stop') for 
    an array of length `len_arr`.

    Args:
        t (double): Time.
        start (double): Start time.
        stop (double): Stop time.
        len_arr (int): Length of array.
    
    Returns:
        int: Array index.
    """
    return <int>floor(((t-start)/(stop-start)*(len_arr-1)))


def test_get_arr_idx(double t, double start, double stop, int len_arr):
    return get_arr_idx(t, start, stop, len_arr)