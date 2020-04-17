/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef PULSE_UTILS_H
#define PULSE_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "types.hpp"

namespace py = pybind11;

py::object expect_psi_csr(py::array_t<complex_t> data,
                          py::array_t<int> ind,
                          py::array_t<int> ptr,
                          py::array_t<complex_t> vec,
                          bool isherm);

//============================================================================
// Computes the occupation probabilities of the specifed qubits for
// the given state.
// Args:
// qubits (int array): Ints labelling which qubits are to be measured.
//============================================================================
py::array_t<double> occ_probabilities(py::array_t<int> qubits,
                                      py::array_t<complex_t> state,
                                      py::list meas_ops);

//============================================================================
// Converts probabilities back into shots
// Args:
// mem
//         mem_slots
// probs: expectation value
// rand_vals: random values used to convert back into shots
//============================================================================
void write_shots_memory(py::array_t<unsigned char> mem,
                        py::array_t<unsigned int> mem_slots,
                        py::array_t<double> probs,
                        py::array_t<double> rand_vals);


//============================================================================
// Takes a list of complex numbers represented by a list
// of pairs of floats, and inserts them into a complex NumPy
//         array at a given starting index.
//
// Parameters:
// A (list): A nested-list of [re, im] pairs.
// B(ndarray): Array for storing complex numbers from list A.
// start_idx (int): The starting index at which to insert elements.
//============================================================================
void oplist_to_array(py::list A, py::array_t<complex_t> B, int start_idx);


//============================================================================
// Sparse matrix, dense vector multiplication.
// Here the vector is assumed to have one-dimension.
// Matrix must be in CSR format and have complex entries.
//
// Parameters
// ----------
// data : array
//         Data for sparse matrix.
// idx : array
//         Indices for sparse matrix data.
// ptr : array
//         Pointers for sparse matrix data.
// vec : array
//         Dense vector for multiplication.  Must be one-dimensional.
//
// Returns
// -------
// out : array
//         Returns dense array.
//============================================================================
py::array_t<complex_t> spmv_csr(py::array_t<complex_t> data,
                                py::array_t<int> ind,
                                py::array_t<int> ptr,
                                py::array_t<complex_t> vec);

#endif //PULSE_UTILS_H
