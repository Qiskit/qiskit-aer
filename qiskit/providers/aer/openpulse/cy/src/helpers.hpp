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
#include <Python.h>

bool _check_is_integer(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("PyObject is null!");

    // Seems like this function checks every integer type
    if(!PyLong_Check(value))
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

template<typename T>
T _get_value(PyObject * value){
    throw std::invalid_argument("Can't get the value for this type!");
}

template<>
long _get_value(PyObject * value){
    if(!_check_is_integer(value))
        throw std::invalid_argument("PyObject is not a long!");

    long c_value = PyLong_AsLong(value);
    auto ex = PyErr_Occurred();
    if(ex)
        throw ex;

    return c_value;
}

template<>
double _get_value(PyObject * value){
    if(!_check_is_floating_point(value))
        throw std::invalid_argument("PyObject is not a double!");

    double c_value = PyFloat_AsDouble(value);
    auto ex = PyErr_Occurred();
    if(ex)
        throw ex;

    return c_value;
}

template<>
std::complex<double> _get_value(PyObject * value){
    if(!_check_is_complex(value))
        throw std::invalid_argument("PyObject is not a complex number!");

    Py_complex c_value = PyComplex_AsCComplex(value);
    auto ex = PyErr_Occurred();
    if(ex)
        throw ex;

    return std::complex<double>(c_value.real, c_value.imag);
}

template<>
std::string _get_value(PyObject * value){
    PyObject * tmp_py_str = PyUnicode_AsEncodedString(value, "utf-8", "replace");
    auto c_str = PyBytes_AS_STRING(tmp_py_str);
    if(c_str == nullptr)
        throw std::invalid_argument("Conversion to utf-8 has failed!");

    return std::string(c_str);
}


PyObject * _get_py_value_from_py_dict(PyObject * dict, const std::string& key){

    if(dict == nullptr)
        throw std::invalid_argument("Python dictionary is null!");

    PyObject * tmp_key;
    PyObject * value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &tmp_key, &value)) {
        auto key_str = _get_value<std::string>(tmp_key);
        if(key_str == key){
            return value;
        }
    }
    return nullptr;
}

template<typename VecType>
const std::vector<VecType> _get_vec_from_py_list(PyObject * py_list){
    if(py_list == nullptr)
        throw std::invalid_argument("Pyhton list is null!");

    // Check that it's a list
    if(!PyList_Check(py_list))
        throw std::invalid_argument("PyObject is not a list!!");


    auto size = PyList_Size(py_list);
    std::vector<VecType> vector;
    vector.reserve(size);
    for(int i=0; i<size; ++i){
        auto p_item = PyList_GetItem(py_list, i);
        if(p_item == nullptr)
            continue;
        vector.emplace_back(p_item);
    }


    return vector;
}


template<typename KeyType, typename ValueType>
const std::unordered_map<KeyType, ValueType> _get_map_from_py_dict(PyObject * py_dict){
    if(py_dict == nullptr)
        throw std::invalid_argument("Pyhton list is null!");

    // Check that it's a dict
    if(!PyDict_Check(py_dict))
        throw std::invalid_argument("PyObject is not a dictonary!!");

    auto size = PyDict_Size(py_dict);
    std::unordered_map<KeyType, ValueType> map;
    map.reserve(size);

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(py_dict, &pos, &key, &value)) {
        auto c_key = _get_value<KeyType>(key);
        auto c_value = _get_value<ValueType>(value);
        map.emplace(c_key, c_value);
    }
    return map;
}

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
    return _get_vec_from_py_list<VecType>(py_value);
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
    return _get_map_from_py_dict<KeyType, ValueType>(py_value);
}

/**
 * Returns a C++ long from a Python numeric that is inside a Pyhton
 * dictionary under a key.
 *
 * We assume that the item indexed by the key, it's a numeric:
 * ```python
 * my_dict = { "key": 255} }
 * ```
 * ```c++
 * auto l = <long> get_long_from_dict_item(pyobj_dict, "key");
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
long get_long_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    return _get_value<long>(py_value);
}

/**
 * Returns a C++ double from a Python numeric that is inside a Pyhton
 * dictionary under a key.
 *
 * We assume that the item indexed by the key, it's a numeric:
 * ```python
 * my_dict = { "key": 3.141592} }
 * ```
 * ```c++
 * auto d = <double> get_double_from_dict_item(pyobj_dict, "key");
 * std::cout << "val: " << d;
 * ```
 * Output:
 * ```
 * 3.141592
 * ```
 *
 * @param dict PyObject* A pointer to a PyObject type representing a dictionary
 * @return A double from the Pyhton's dictionary key.
 **/
double get_double_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    return _get_value<double>(py_value);
}

/**
 * Returns a C++ std::complex<double> from a Python complex that is inside a Pyhton
 * dictionary under a key.
 *
 * We assume that the item indexed by the key, it's a complex:
 * ```python
 * my_dict = { "key": 0.1+1j} }
 * ```
 * ```c++
 * auto c = <std::complex<double>> get_double_from_dict_item(pyobj_dict, "key");
 * std::cout << "val: " << c;
 * ```
 * Output:
 * ```
 * (0.1,1)
 * ```
 *
 * @param dict PyObject* A pointer to a PyObject type representing a dictionary
 * @return A double from the Pyhton's dictionary key.
 **/
std::complex<double> get_complex_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    return _get_value<std::complex<double>>(py_value);
}


#endif // _HELPERS_HPP