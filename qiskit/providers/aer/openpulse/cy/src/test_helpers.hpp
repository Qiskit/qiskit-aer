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

#include <unordered_map>
#include <vector>
#include <complex>
#include <Python.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "helpers.hpp"

    cdef cpp_test_py_string_to_cpp_string(string val)
    cdef cpp_test_py_complex_double_to_cpp_complex_double(double complex val)
    cdef cpp_test_py_list_to_cpp_vec(list val)
    cdef cpp_test_py_list_of_lists_to_cpp_vector_of_vectors(list val)
    cdef cpp_test_py_dict_string_numeric_to_cpp_map_string_numeric(dict val)
    cdef cpp_test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(dict val)
    cdef cpp_test_py_dict_string_list_of_np_array_to_cpp_map_string_vec_of_nparrays_of_doubles(dict val)
    cdef cpp_test_np_array_of_doubles(np.array val)
    cdef cpp_test_evaluate_hamiltonians(list val)



#endif // _TEST_HELPERS_HPP