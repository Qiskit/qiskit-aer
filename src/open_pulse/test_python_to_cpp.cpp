#include "test_python_to_cpp.hpp"

#include <csignal>
#include <unordered_map>
#include <vector>
#include <complex>
#include <Python.h>
#include <iostream>
#include "python_to_cpp.hpp"

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

bool cpp_test_np_array_of_doubles(PyArrayObject * val){
    // val = np.array([0., 1., 2., 3.])
    auto vec = get_value<NpArray<double>>(val);
    if(vec[0] != 0. || vec[1] != 1. || vec[2] != 2. || vec[3] != 3.)
        return false;

    return true;
}

bool cpp_test_evaluate_hamiltonians(PyObject * val){
    // TODO: Add tests!
    return false;
}

bool cpp_test_py_ordered_map(PyObject * val){
    // Ordered map should guarantee insertion order.
    // val = {"D0": 1, "U0": 2, "D1": 3, "U1": 4}
    std::vector<std::string> order = {"D0", "U0", "D1", "U1"};
    auto ordered = get_value<ordered_map<std::string, long>>(val);
    size_t i = 0;
    for(const auto& elem: ordered) {
        auto key = elem.first;
        if(key != order[i++])
            return false;
    }
    return true;
}
