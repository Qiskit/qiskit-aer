#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
from scipy.sparse import coo_matrix
from ..fastsparse import fast_csr_matrix
cimport numpy as np
cimport cython
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from .sparse_structs cimport CSR_Matrix
np.import_array()
from libc.string cimport memset

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW(size_t size)


#Struct used for CSR indices sorting
cdef struct _data_ind_pair:
    double complex data
    int ind

ctypedef _data_ind_pair data_ind_pair
ctypedef int (*cfptr)(data_ind_pair, data_ind_pair)


cdef void raise_error_CSR(int E, CSR_Matrix * C = NULL):
    if not C.numpy_lock and C != NULL:
        free_CSR(C)
    if E == -1:
        raise MemoryError('Could not allocate memory.')
    elif E == -2:
        raise Exception('Error manipulating CSR_Matrix structure.')
    elif E == -3:
        raise Exception('CSR_Matrix is not initialized.')
    elif E == -4:
        raise Exception('NumPy already has lock on data.')
    elif E == -5:
        raise Exception('Cannot expand data structures past max_length.')
    elif E == -6:
        raise Exception('CSR_Matrix cannot be expanded.')
    elif E == -7:
        raise Exception('Data length cannot be larger than max_length')
    else:
        raise Exception('Error in Cython code.')

cdef inline int int_min(int a, int b) nogil:
    return b if b < a else a

cdef inline int int_max(int a, int b) nogil:
    return a if a > b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void init_CSR(CSR_Matrix * mat, int nnz, int nrows, int ncols = 0,
                int max_length = 0, int init_zeros = 1):
    """
    Initialize CSR_Matrix struct. Matrix is assumed to be square with
    shape nrows x nrows.  Manually set mat.ncols otherwise

    Parameters
    ----------
    mat : CSR_Matrix *
        Pointer to struct.
    nnz : int
        Length of data and indices arrays. Also number of nonzero elements
    nrows : int
        Number of rows in matrix. Also gives length
        of indptr array (nrows+1).
    ncols : int (default = 0)
        Number of cols in matrix. Default is ncols = nrows.
    max_length : int (default = 0)
        Maximum length of data and indices arrays.  Used for resizing.
        Default value of zero indicates no resizing.
    """
    if max_length == 0:
        max_length = nnz
    if nnz > max_length:
        raise_error_CSR(-7, mat)
    if init_zeros:
        mat.data = <double complex *>PyDataMem_NEW(nnz * sizeof(double complex))
        memset(&mat.data[0],0,nnz * sizeof(double complex))
    else:
        mat.data = <double complex *>PyDataMem_NEW(nnz * sizeof(double complex))
    if mat.data == NULL:
        raise_error_CSR(-1, mat)
    if init_zeros:
        mat.indices = <int *>PyDataMem_NEW(nnz * sizeof(int))
        mat.indptr = <int *>PyDataMem_NEW((nrows+1) * sizeof(int))
        memset(&mat.indices[0],0,nnz * sizeof(int))
        memset(&mat.indptr[0],0,(nrows+1) * sizeof(int))
    else:
        mat.indices = <int *>PyDataMem_NEW(nnz * sizeof(int))
        mat.indptr = <int *>PyDataMem_NEW((nrows+1) * sizeof(int))
    mat.nnz = nnz
    mat.nrows = nrows
    if ncols == 0:
        mat.ncols = nrows
    else:
        mat.ncols = ncols
    mat.is_set = 1
    mat.max_length = max_length
    mat.numpy_lock = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy_CSR(CSR_Matrix * out, CSR_Matrix * mat):
    """
    Copy a CSR_Matrix.
    """
    cdef size_t kk
    if not mat.is_set:
        raise_error_CSR(-3)
    elif out.is_set:
        raise_error_CSR(-2)
    init_CSR(out, mat.nnz, mat.nrows, mat.nrows, mat.max_length)
    # We cannot use memcpy here since there are issues with
    # doing so on Win with the GCC compiler
    for kk in range(mat.nnz):
        out.data[kk] = mat.data[kk]
        out.indices[kk] = mat.indices[kk]
    for kk in range(mat.nrows+1):
        out.indptr[kk] = mat.indptr[kk]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void free_CSR(CSR_Matrix * mat):
    """
    Manually free CSR_Matrix data structures if
    data is not locked by NumPy.
    """
    if not mat.numpy_lock and mat.is_set:
        if mat.data != NULL:
            PyDataMem_FREE(mat.data)
        if mat.indices != NULL:
            PyDataMem_FREE(mat.indices)
        if mat.indptr != NULL:
            PyDataMem_FREE(mat.indptr)
        mat.is_set = 0
    else:
        raise_error_CSR(-2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void shorten_CSR(CSR_Matrix * mat, int N):
    """
    Shortends the length of CSR data and indices arrays.
    """
    if (not mat.numpy_lock) and mat.is_set:
        mat.data = <double complex *>PyDataMem_RENEW(mat.data, N * sizeof(double complex))
        mat.indices = <int *>PyDataMem_RENEW(mat.indices, N * sizeof(int))
        mat.nnz = N
    else:
        if mat.numpy_lock:
            raise_error_CSR(-4, mat)
        elif not mat.is_set:
            raise_error_CSR(-3, mat)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object CSR_to_scipy(CSR_Matrix * mat):
    """
    Converts a CSR_Matrix struct to a SciPy csr_matrix class object.
    The NumPy arrays are generated from the pointers, and the lifetime
    of the pointer memory is tied to that of the NumPy array
    (i.e. automatic garbage cleanup.)

    Parameters
    ----------
    mat : CSR_Matrix *
        Pointer to CSR_Matrix.
    """
    cdef np.npy_intp dat_len, ptr_len
    cdef np.ndarray[complex, ndim=1] _data
    cdef np.ndarray[int, ndim=1] _ind, _ptr
    if (not mat.numpy_lock) and mat.is_set:
        dat_len = mat.nnz
        ptr_len = mat.nrows+1
        _data = np.PyArray_SimpleNewFromData(1, &dat_len, np.NPY_COMPLEX128, mat.data)
        PyArray_ENABLEFLAGS(_data, np.NPY_OWNDATA)

        _ind = np.PyArray_SimpleNewFromData(1, &dat_len, np.NPY_INT32, mat.indices)
        PyArray_ENABLEFLAGS(_ind, np.NPY_OWNDATA)

        _ptr = np.PyArray_SimpleNewFromData(1, &ptr_len, np.NPY_INT32, mat.indptr)
        PyArray_ENABLEFLAGS(_ptr, np.NPY_OWNDATA)
        mat.numpy_lock = 1
        return fast_csr_matrix((_data, _ind, _ptr), shape=(mat.nrows,mat.ncols))
    else:
        if mat.numpy_lock:
            raise_error_CSR(-4)
        elif not mat.is_set:
            raise_error_CSR(-3)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int ind_sort(data_ind_pair x, data_ind_pair y):
    return x.ind < y.ind


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sort_indices(CSR_Matrix * mat):
    """
    Sorts the indices of a CSR_Matrix inplace.
    """
    cdef size_t ii, jj
    cdef vector[data_ind_pair] pairs
    cdef cfptr cfptr_ = &ind_sort
    cdef int row_start, row_end, length

    for ii in range(mat.nrows):
        row_start = mat.indptr[ii]
        row_end = mat.indptr[ii+1]
        length = row_end - row_start
        pairs.resize(length)

        for jj in range(length):
            pairs[jj].data = mat.data[row_start+jj]
            pairs[jj].ind = mat.indices[row_start+jj]

        sort(pairs.begin(),pairs.end(),cfptr_)

        for jj in range(length):
            mat.data[row_start+jj] = pairs[jj].data
            mat.indices[row_start+jj] = pairs[jj].ind