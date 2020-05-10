/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _TEST_HELPERS_HPP
#define _TEST_HELPERS_HPP

#include <numpy/arrayobject.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// TODO: Test QuantumObj
// TODO: Test Hamiltonian

bool cpp_test_py_list_to_cpp_vec(PyObject * val);
bool cpp_test_py_list_of_lists_to_cpp_vector_of_vectors(PyObject * val);
bool cpp_test_py_dict_string_numeric_to_cpp_map_string_numeric(PyObject * val);
bool cpp_test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(PyObject * val);
bool cpp_test_np_array_of_doubles(PyArrayObject * val);
bool cpp_test_evaluate_hamiltonians(PyObject * val);
bool cpp_test_py_ordered_map(PyObject * val);

PYBIND11_MODULE(test_python_to_cpp, m) {
    m.doc() = "pybind11 test_python_to_cpp"; // optional module docstring

    m.def("test_py_list_to_cpp_vec", [](py::list list) { return cpp_test_py_list_to_cpp_vec(list.ptr()); } , "");
    m.def("test_py_list_of_lists_to_cpp_vector_of_vectors",
            [](py::list list) { return cpp_test_py_list_of_lists_to_cpp_vector_of_vectors(list.ptr()); } , "");
    m.def("test_py_dict_string_numeric_to_cpp_map_string_numeric",
            [](py::dict dict) { return cpp_test_py_dict_string_numeric_to_cpp_map_string_numeric(dict.ptr()); } , "");
    m.def("test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles",
        [](py::dict dict) { return cpp_test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(dict.ptr()); } , "");
    m.def("test_np_array_of_doubles",
            [](py::array_t<double> array_doubles) { return cpp_test_np_array_of_doubles(reinterpret_cast<PyArrayObject *>(array_doubles.ptr())); } , "");
    m.def("test_evaluate_hamiltonians", [](py::list list) { return cpp_test_evaluate_hamiltonians(list.ptr()); } , "");
    m.def("test_py_ordered_map", [](py::dict dict) { return cpp_test_py_ordered_map(dict.ptr()); } , "");
}

#endif // _TEST_HELPERS_HPP