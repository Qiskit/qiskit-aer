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
@cython.wraparound(False)
def oplist_to_array(list A, complex[::1] B, 
                    int start_idx=0):
    """Takes a list of complex numbers represented by a list
    of pairs of floats, and inserts them into a complex NumPy
    array at a given starting index.
    
    Parameters:
        A (list): A nested-list of [re, im] pairs.
        B(ndarray): Array for storing complex numbers from list A.
        start_idx (int): The starting index at which to insert elements.
        
    Notes:
        This is ~5x faster than doing it in Python.
    """
    cdef size_t kk
    cdef int lenA = len(A)
    cdef list temp
    
    if (start_idx+lenA) > B.shape[0]:
        raise Exception('Input list does not fit into array if start_idx is {}.'.format(start_idx))
    
    for kk in range(lenA):
        temp = A[kk]
        B[start_idx+kk] = temp[0]+1j*temp[1]