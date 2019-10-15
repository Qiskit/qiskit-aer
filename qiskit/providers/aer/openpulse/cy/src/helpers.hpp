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

#ifndef _HELPERS_HPP
#define _HELPERS_HPP

#include <unordered_map>
#include <vector>
#include <complex>
#include <type_traits>
#include <Python.h>
#include <exprtk.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>


/**
 * TODO: There's a lot of copying due to converting from Pyhton C strucutres
 * to C++ stl containers. We can avoid this by wrapping the former and avoiding
 * copies
 **/


/**
 * Helper Types
 **/
using complex_t = std::complex<double>;

template<typename VecType>
class NpArray {
    NpArray() = delete;
    public:
    explicit NpArray(const std::vector<VecType>& _data, const std::vector<int>& _shape)
    : data(_data), shape(_shape)
    {}
    const std::vector<VecType>& data;
    /**
     * The shape of the array: like
     * ```pyhton
     * arr = np.array([0,1,2],[3,4,5])
     * arr.shape
     **/
    const std::vector<int>& shape;

    const VecType& operator[](size_t index) const {
        return data[index];
    }
};


template<typename T>
void jlog(const std::string& msg, const T& values){
    spdlog::debug("{}\n", msg);
    for(const auto& val : values){
        spdlog::debug("{} ", val);
    }
    spdlog::debug("\n");
}

template<>
void jlog(const std::string& msg, const std::vector<complex_t>& values){
    spdlog::debug("{}\n", msg);
    for(const auto& val : values){
        spdlog::debug("{}+{}j", val.real(), val.imag());
    }
    spdlog::debug("\n");
}

template<>
void jlog(const std::string& msg, const std::unordered_map<std::string, std::vector<std::vector<double>>>& values){
    spdlog::debug("{}\n", msg);
    for(const auto& val : values){
        for(const auto& inner: val.second){
            for(const auto& inner2: inner){
                spdlog::debug("{}:{} ", val.first, inner2);
            }
        }
    }
    spdlog::debug("\n");
}

template<>
void jlog(const std::string& msg, const std::unordered_map<std::string, double>& values){
    spdlog::debug("{}\n", msg);
    for(const auto& val : values){
        spdlog::debug("{}:{} ", val.first, val.second);
    }
    spdlog::debug("\n");
}

template<>
void jlog(const std::string& msg, const std::unordered_map<std::string, std::vector<NpArray<double>>>& values){
    spdlog::debug("{}\n", msg);
    for(const auto& val : values){
        for(const auto& inner: val.second){
            jlog("", inner);
        }
    }
    spdlog::debug("\n");
}

template<typename T>
void jlog(const std::string& msg, const NpArray<T>& values){
    spdlog::debug("{}\n", msg);
    spdlog::debug(".shape: ");
    for(const auto& shape : values.shape)
        spdlog::debug("{} ", shape);

    spdlog::debug("\n.data: ");
    for(const auto& val : values.data){
        spdlog::debug("{} ", val);
    }
    spdlog::debug("\n");
}


template <typename T>
struct iterator_extractor { typedef typename T::iterator type; };

template <typename T>
struct iterator_extractor<T const> { typedef typename T::const_iterator type; };

/**
 * Python-like `enumerate()` for C++14 ranged-for
 *
 * I wish I'd had this included in the STL :)
 *
 * Usage:
 * ```c++
 * for(auto& elem: index(vec)){
 *     std::cout << "Index: " << elem.first << " Element: " << elem.second;
 * }
 * ```
 **/
template <typename T>
class Indexer {
public:
    class _Iterator {
        typedef typename iterator_extractor<T>::type inner_iterator;
        typedef typename std::iterator_traits<inner_iterator>::reference inner_reference;
    public:
        typedef std::pair<size_t, inner_reference> reference;

        _Iterator(inner_iterator it): _pos(0), _it(it) {}

        reference operator*() const {
            return reference(_pos, *_it);
        }

        _Iterator& operator++() {
            ++_pos;
            ++_it;
            return *this;
        }

        _Iterator operator++(int) {
            _Iterator tmp(*this);
            ++*this;
            return tmp;
        }

        bool operator==(_Iterator const& it) const {
            return _it == it._it;
        }
        bool operator!=(_Iterator const& it) const {
            return !(*this == it);
        }

    private:
        size_t _pos;
        inner_iterator _it;
    };

    Indexer(T& t): _container(t) {}

    _Iterator begin() const {
        return _Iterator(_container.begin());
    }
    _Iterator end() const {
        return _Iterator(_container.end());
    }

private:
    T& _container;
}; // class Indexer

template <typename T>
Indexer<T> enumerate(T& t) { return Indexer<T>(t); }



bool _check_is_integer(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("PyObject is null!");

    // Seems like this function checks every integer type
    if(!PyLong_Check(value))
        return false;

    return true;
}

bool _check_is_string(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("PyObject is null!");

    if(!PyUnicode_Check(value))
        return false;

    return true;
}

bool _check_is_floating_point(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("PyObject is null!");

    if(!PyFloat_Check(value))
        return false;

    return true;
}

bool _check_is_complex(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("PyObject is null!");

    if(!PyComplex_Check(value))
        return false;

    return true;
}

bool _check_is_list(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("Pyhton list is null!");

    // Check that it's a list
    if(!PyList_Check(value))
        return false;

    return true;
}

bool _check_is_dict(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("Pyhton dict is null!");

    // Check that it's a dict
    if(!PyDict_Check(value))
        return false;

    return true;
}

bool _check_is_np_array(PyArrayObject * value){
    if(value == nullptr)
        throw std::invalid_argument("Numpy ndarray is null!");

    // Check that it's a numpy ndarray
    if(!PyArray_Check(value))
        return false;

    return true;
}

// Simon Brand technique to achive partial specialization on function templates
// https://www.fluentcpp.com/2017/08/15/function-templates-partial-specialization-cpp/
// This "type" struct will carry T, but wil be ignored by the compiler later.
// It's like a help you give to the compiler so it can resolve the specialization
template<typename T>
struct type{};

template<typename T>
T get_value(type<T> _, PyObject * value){
    throw std::invalid_argument("Cannot get value for this type!");
}

// <JUAN> TODO: We might want to expose only these two functions
template<typename T>
T get_value(PyObject * value){
    return get_value(type<T>{}, value);
}

template<typename T>
const T get_value(PyArrayObject * value){
    return get_value(type<T>{}, value);
}
// </JUAN>

template<>
long get_value(type<long> _, PyObject * value){
    if(!_check_is_integer(value))
        throw std::invalid_argument("PyObject is not a long!");

    long c_value = PyLong_AsLong(value);
    auto ex = PyErr_Occurred();
    if(ex)
        throw ex;

    return c_value;
}

template<>
double get_value(type<double> _, PyObject * value){
    if(!_check_is_floating_point(value))
        throw std::invalid_argument("PyObject is not a double!");

    double c_value = PyFloat_AsDouble(value);
    auto ex = PyErr_Occurred();
    if(ex)
        throw ex;

    return c_value;
}

template<>
std::complex<double> get_value(type<std::complex<double>> _, PyObject * value){
    if(!_check_is_complex(value))
        throw std::invalid_argument("PyObject is not a complex number!");

    Py_complex c_value = PyComplex_AsCComplex(value);
    auto ex = PyErr_Occurred();
    if(ex)
        throw ex;

    return std::complex<double>(c_value.real, c_value.imag);
}

template<>
std::string get_value(type<std::string> _, PyObject * value){
    if(!_check_is_string(value))
        throw std::invalid_argument("PyObject is not a string!");

    auto bytes_str = PyUnicode_AsUTF8String(value);
    auto c_str = PyBytes_AsString(bytes_str);

    if(c_str == nullptr)
        throw std::invalid_argument("Conversion to utf-8 has failed!");

    return std::string(c_str);
}

template<typename T>
std::vector<T> get_value(type<std::vector<T>> _, PyObject * value){
    if(!_check_is_list(value))
        throw std::invalid_argument("PyObject is not a List!");

    auto size = PyList_Size(value);
    std::vector<T> vector;
    vector.reserve(size);
    for(auto i=0; i<size; ++i){
        auto py_item = PyList_GetItem(value, i);
        if(py_item == nullptr)
            continue;
        auto item = get_value<T>(py_item);
        vector.emplace_back(item);
    }
    return vector;
}


template<typename ValueType>
std::unordered_map<std::string, ValueType> get_value(type<std::unordered_map<std::string, ValueType>> _, PyObject * value){
    if(!_check_is_dict(value))
        throw std::invalid_argument("PyObject is not a dictonary!!");

    auto size = PyDict_Size(value);
    std::unordered_map<std::string, ValueType> map;
    map.reserve(size);

    PyObject *key, *val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(value, &pos, &key, &val)) {
        auto inner_key = get_value<std::string>(key);
        auto inner_value = get_value<ValueType>(val);
        map.emplace(inner_key, inner_value);
    }
    return map;
}



/**
 * Returns the .shape of the a Numpy array object
 * This is equivalent to:
 * ```pyhton
 * arr = np.array([[1,2],[3,4]])
 * arr.shape
 * ```
 */
std::vector<int> _get_shape_from_np_array(PyArrayObject * np_array){
    if(!_check_is_np_array(np_array))
        throw std::invalid_argument("PyArrayObject is not a numpy array!");

    auto p_dims = PyArray_SHAPE(np_array);
    if(p_dims == nullptr)
        throw std::invalid_argument("Couldn't get the shape of the array!");

    auto num_dims = PyArray_NDIM(np_array);
    std::vector<int> dims;
    dims.reserve(num_dims);
    for(auto i = 0; i < num_dims; ++i){
        dims.emplace_back(p_dims[i]);
    }
    return dims;
}

/**
 * Get a C++ Vector of C++ types from a Numpy ndarray (PyArrayObject) type
 **/
template<typename VecType>
const std::vector<VecType> _get_vec_from_np_array(PyArrayObject * py_array){
    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(py_array) == 0) {
        return {};
    }
    /* TODO This is faster if we deal with PyArrayObject directly */
    PyObject * py_list = PyArray_ToList(py_array);
    return get_value<std::vector<VecType>>(py_list);
}

template<typename T>
const NpArray<T> get_value(type<NpArray<T>> _, PyArrayObject * value){
    if(!_check_is_np_array(value))
        throw std::invalid_argument("PyArrayObject is not a numpy array!");

    const auto arr = _get_vec_from_np_array<T>(value);
    const auto shape = _get_shape_from_np_array(value);
    return NpArray<T>(arr, shape);
}



PyObject * _get_py_value_from_py_dict(PyObject * dict, const std::string& key){
    if(dict == nullptr)
        throw std::invalid_argument("Python dictionary is null!");

    PyObject * tmp_key;
    PyObject * value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &tmp_key, &value)) {
        auto key_str = get_value<std::string>(tmp_key);
        if(key_str == key){
            return value;
        }
    }
    return nullptr;
}


/**
 * Get a C++ Vector of C++ types from a Python List
 **/

// template<typename VecType>
// const std::vector<VecType> get_vec_from_py_list(PyObject * py_list){
//     if(!_check_is_list(py_list))
//         throw std::invalid_argument("PyObject is not a List!");

//     auto size = PyList_Size(py_list);
//     std::vector<VecType> vector;
//     vector.reserve(size);
//     for(auto i=0; i<size; ++i){
//         auto py_item = PyList_GetItem(py_list, i);
//         if(py_item == nullptr)
//             continue;
//         auto cpp_item = get_value<VecType>(py_item);
//         vector.emplace_back(cpp_item);
//     }
//     return vector;
// }


/**
 * Returns a C++ vector from a Python list that is inside a Pyhton dictionary under a key.
 *
 * We assume that the item indexed by the key, it's a list:
 * ```python
 * my_dict = { "key": [1,2,3,4] }
 * ```
 * ```c++
 * auto v = <std::string> get_vec_from_dict_item(pyobj_dict, "key")
 * for(auto& elem: v)
 *     std::cout << elem;
 * ```
 * Output:
 * ```
 * 1234
 * ```
 *
 * @param dict PyObject* A pointer to a PyObject type representing a dictionary
 * @return A vector of type VecType from the Pyhton's dictionary key.
 **/
template<typename VecType>
const std::vector<VecType> get_vec_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    return get_value<std::vector<VecType>>(py_value);
}

/**
 * Returns a C++ unordered map from a Python dictionary that is inside another Pyhton
 * dictionary under a key.
 *
 * We assume that the item indexed by the key, it's a dictionary:
 * ```python
 * my_dict = { "key": {"inner": 1, "renni": 255} }
 * ```
 * ```c++
 * auto m = <std::string, int> get_map_from_dict_item(pyobj_dict, "key");
 * for(auto& item: m)
 *     std::cout << " key:" << item.first << " val:" << item.second;
 * ```
 * Output:
 * ```
 *  key:inner val:1 key:renni val:255
 * ```
 *
 * @param dict PyObject* A pointer to a PyObject type representing a dictionary
 * @return An unordered_map of type <KeyType,ValueType> from the Pyhton's dictionary key.
 **/
template<typename KeyType, typename ValueType>
const std::unordered_map<KeyType, ValueType> get_map_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    return get_value<std::unordered_map<KeyType, ValueType>>(py_value);
}

/**
 * Returns a C++ value of type ValueTyep from a Python numeric that is inside a Pyhton
 * dictionary under a key.
 *
 * We assume that the item indexed by the key, it's a numeric:
 * ```python
 * my_dict = { "key": 255} }
 * ```
 * ```c++
 * auto l = get_value_from_dict_item<long>(pyobj_dict, "key");
 * std::cout << "val: " << l;
 * ```
 * Output:
 * ```
 * 255
 * ```
 *
 * @param dict PyObject* A pointer to a PyObject type representing a dictionary
 * @return A long from the Pyhton's dictionary key.
 **/
template<typename ValueType>
long get_value_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    return get_value<ValueType>(py_value);
}



/**
 * Math expression evaluator for the Hamiltonian terms
 **/
double evaluate_hamiltonian_expression(const std::string& expr_string,
                                  const std::vector<complex_t>& vars,
                                  const std::vector<std::string>& vars_names){
    exprtk::symbol_table<double> symbol_table;
    auto pi = M_PI;
    symbol_table.add_variable("np.pi", pi);

    for(const auto& idx_var : enumerate(vars)){
        auto index = idx_var.first;
        auto var = idx_var.second;
        auto real_part = var.real();
        symbol_table.add_variable(vars_names[index], real_part);
    }

    exprtk::expression<double> expression;
    expression.register_symbol_table(symbol_table);

    exprtk::parser<double> parser;

    if (!parser.compile(expr_string, expression)){
        throw std::invalid_argument("Cannot evaluate hamiltonian expression: " + expr_string);
    }

    return expression.value();
}

#endif // _HELPERS_HPP