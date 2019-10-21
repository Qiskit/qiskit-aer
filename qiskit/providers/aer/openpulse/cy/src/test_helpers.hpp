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

#include <csignal>
#include <unordered_map>
#include <vector>
#include <complex>
#include <Python.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <numpy/arrayobject.h>
#include "helpers.hpp"

bool cpp_test_py_list_to_cpp_vec(PyObject * val){
    // val = [1., 2., 3.]
    auto vec = get_value<std::vector<double>>(val);
    auto expected = std::vector<double>{1., 2., 3.};
    return vec == expected;
}

bool cpp_test_py_list_of_lists_to_cpp_vector_of_vectors(PyObject * val){
    // val = [[1., 2., 3.]]
    auto vec = get_value<std::vector<std::vector<double>>>(val);
    auto expected = std::vector<std::vector<double>>{{1., 2., 3.}};
    return vec == expected;
}

bool cpp_test_py_list_of_np_arrays(PyObject * val){
    // val = = [np.array([1., 2., 3.]), np.array([1., 2., 3.])]
    auto vec = get_value<std::vector<NpArray<double>>>(val);
    auto expected = std::vector<NpArray<double>>{ NpArray<double>{{1., 2., 3.}, {3}}, NpArray<double>{{1., 2., 3.}, {3}} };
    return vec == expected;
}

bool cpp_test_py_dict_string_numeric_to_cpp_map_string_numeric(PyObject * val){
    // val = {"key": 1}
    auto map = get_value<std::unordered_map<std::string, long>>(val);
    auto expected = std::unordered_map<std::string, long>{{"key", 1}};
    return map == expected;

}

bool cpp_test_py_dict_string_list_of_list_of_doubles_to_cpp_map_string_vec_of_vecs_of_doubles(PyObject * val){
    // val = {"key": [[1., 2., 3.]]}
    auto map = get_value<std::unordered_map<std::string, std::vector<std::vector<double>>>>(val);
    auto expected = std::unordered_map<std::string, std::vector<std::vector<double>>>{{"key", {{1., 2., 3.}}}};
    return map == expected;
}

bool cpp_test_py_dict_string_list_of_np_array_to_cpp_map_string_vec_of_nparrays_of_doubles(PyObject * val){
    // {"key": [np.array([0., 1.]), np.array([2., 3.])]}
    const auto map = get_value<std::unordered_map<std::string, std::vector<NpArray<double>>>>(val);
    const auto expected =
        std::unordered_map<std::string, std::vector<NpArray<double>>>{
            {"key", {NpArray<double>{{0.,1.},{2}}, NpArray<double>{{2., 3.},{2}}}}
        };
    return map == expected;
}

bool cpp_test_np_array_of_doubles(PyArrayObject * val){
    // arg = np.array([0., 1., 2., 3.])
    auto vec = get_value<NpArray<double>>(val);
    auto expected = NpArray<double>{{0., 1., 2., 3.}, {4}};
    return vec == expected;
}

bool cpp_test_evaluate_hamiltonians(PyObject * val){
    //std::raise(SIGTRAP);
    return false;
}


#endif // _TEST_HELPERS_HPP