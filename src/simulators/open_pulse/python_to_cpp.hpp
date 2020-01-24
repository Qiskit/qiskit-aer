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

#ifndef _PYTHON_TO_CPP_HPP
#define _PYTHON_TO_CPP_HPP

#include <utility>
#include <unordered_map>
#include <map>
#include <vector>
#include <complex>
#include <type_traits>
#include <Python.h>
#ifdef DEBUG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#endif
#include <numpy/arrayobject.h>
#include "ordered_map.hpp"
#include "types.hpp"
#include "iterators.hpp"
#include "eval_hamiltonian.hpp"

static bool init_numpy(){
    static bool initialized = false;
    if(!initialized){
        import_array();
        initialized = true;
    }
};

bool check_is_integer(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("PyObject is null!");

    // Seems like this function checks every integer type
    if(!PyLong_Check(value))
        return false;

    return true;
}

bool check_is_string(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("PyObject is null!");

    if(!PyUnicode_Check(value))
        return false;

    return true;
}

bool check_is_floating_point(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("PyObject is null!");

    if(!PyFloat_Check(value))
        return false;

    return true;
}

bool check_is_complex(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("PyObject is null!");

    if(!PyComplex_Check(value))
        return false;

    return true;
}

bool check_is_list(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("Pyhton list is null!");

    // Check that it's a list
    if(!PyList_Check(value))
        return false;

    return true;
}

bool check_is_tuple(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("Pyhton tuple is null!");

    // Check that it's a tuple
    if(!PyTuple_Check(value))
        return false;

    return true;
}

bool check_is_dict(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("Pyhton dict is null!");

    // Check that it's a dict
    if(!PyDict_Check(value))
        return false;

    return true;
}

bool check_is_np_array(PyArrayObject * value){
    if(value == nullptr)
        throw std::invalid_argument("Numpy ndarray is null!");
    init_numpy();
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
uint8_t get_value(type<uint8_t> _, PyObject * value){
    return get_value<long>(value);
}

template<>
long get_value(type<long> _, PyObject * value){
    if(!check_is_integer(value))
        throw std::invalid_argument("PyObject is not a long!");

    long c_value = PyLong_AsLong(value);
    auto ex = PyErr_Occurred();
    if(ex)
        throw ex;

    return c_value;
}

template<>
double get_value(type<double> _, PyObject * value){
    if(!check_is_floating_point(value)){
        // it's not a floating point, but maybe an integer?
        if(check_is_integer(value))
            return static_cast<double>(get_value<long>(value));

        throw std::invalid_argument("PyObject is not a double!");
    }

    double c_value = PyFloat_AsDouble(value);
    auto ex = PyErr_Occurred();
    if(ex)
        throw ex;

    return c_value;
}

template<>
std::complex<double> get_value(type<std::complex<double>> _, PyObject * value){
    if(!check_is_complex(value))
        throw std::invalid_argument("PyObject is not a complex number!");

    Py_complex c_value = PyComplex_AsCComplex(value);
    auto ex = PyErr_Occurred();
    if(ex)
        throw ex;

    return std::complex<double>(c_value.real, c_value.imag);
}

template<>
std::string get_value(type<std::string> _, PyObject * value){
    if(!check_is_string(value))
        throw std::invalid_argument("PyObject is not a string!");

    auto bytes_str = PyUnicode_AsUTF8String(value);
    auto c_str = PyBytes_AsString(bytes_str);

    if(c_str == nullptr)
        throw std::invalid_argument("Conversion to utf-8 has failed!");

    return std::string(c_str);
}

template<typename T>
std::vector<T> get_value(type<std::vector<T>> _, PyObject * value){
    if(!check_is_list(value))
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

/* WARNING: There's no support for variadic templates in Cython, so
   we use a std::pair because there's no more than two types in the Python
   tuples so far, so as we are fine for now... */
template<typename T, typename U>
std::pair<T, U> get_value(type<std::pair<T, U>> _, PyObject * value){
    if(!check_is_tuple(value))
        throw std::invalid_argument("PyObject is not a Tuple!");

    if(PyTuple_Size(value) > 2)
        throw std::invalid_argument("Tuples with more than 2 elements are not supported yet!!");

    auto first_py_item = PyTuple_GetItem(value, 0);
    if(first_py_item == nullptr)
        throw std::invalid_argument("The tuple must have a first element");

    auto second_py_item = PyTuple_GetItem(value, 1);
    if(second_py_item == nullptr)
        throw std::invalid_argument("The tuple must have a second element");

    auto first_item = get_value<T>(first_py_item);
    auto second_item = get_value<U>(second_py_item);

    return std::make_pair(first_item, second_item);
}

template<typename ValueType>
std::unordered_map<std::string, ValueType> get_value(type<std::unordered_map<std::string, ValueType>> _, PyObject * value){
    if(!check_is_dict(value))
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

template<typename ValueType>
const ordered_map<std::string, ValueType> get_value(type<ordered_map<std::string, ValueType>> _, PyObject * value){
    if(!check_is_dict(value))
        throw std::invalid_argument("PyObject is not a dictonary!!");

    auto size = PyDict_Size(value);
    ordered_map<std::string, ValueType> map;
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


template<>
TermExpression get_value(type<TermExpression>_, PyObject * value) {
    if(!check_is_tuple(value))
        throw std::invalid_argument("PyObject is not a Tuple!");

    if(PyTuple_Size(value) > 2)
        throw std::invalid_argument("Tuples with more than 2 elements are not supported yet!!");

    auto term = PyTuple_GetItem(value, 1); // 0 is first
    if(term == nullptr)
        throw std::invalid_argument("The tuple must have a second element");

    auto term_expr = get_value<std::string>(term);
    return TermExpression(term_expr);
}


template<typename VecType>
class NpArray {
  public:
	NpArray(){}
	NpArray(PyArrayObject * array){
		_populate_data(array);
		_populate_shape(array);
        size = array->dimensions[0];
	}

    const VecType * data = nullptr;
    size_t size = 0;

    /**
     * The shape of the array: like
     * ```pyhton
     * arr = np.array([0,1,2],[3,4,5])
     * arr.shape
     **/
    std::vector<int> shape;

    const VecType& operator[](size_t index) const {
        return data[index];
    }

    NpArray& operator=(const NpArray<VecType>& other){
		data = reinterpret_cast<VecType *>(other.data);
        size = other.size;
		shape = other.shape;
		return *this;
	}

    bool operator==(const NpArray<VecType>& other) const {
        if(other.size != size ||
           other.shape.size() != shape.size())
           return false;

        for(auto i = 0; i < other.size; ++i){
            if(data[i] != other[i])
                return false;
        }

        for(auto i = 0; i < other.shape.size(); ++i){
            if(shape[i] != other.shape[i])
                return false;
        }

        return true;
    }
  private:

	void _populate_shape(PyArrayObject * array){
		if(!check_is_np_array(array))
			throw std::invalid_argument("PyArrayObject is not a numpy array!");

		auto p_dims = PyArray_SHAPE(array);
		if(p_dims == nullptr)
			throw std::invalid_argument("Couldn't get the shape of the array!");

		auto num_dims = PyArray_NDIM(array);
		shape.reserve(num_dims);
		for(auto i = 0; i < num_dims; ++i){
			shape.emplace_back(p_dims[i]);
		}
	}

    void _populate_data(PyArrayObject * array){
        data = reinterpret_cast<VecType *>(array->data);
	}
};

template<typename T>
const NpArray<T> get_value(type<NpArray<T>> _, PyArrayObject * value){
    if(!check_is_np_array(value))
        throw std::invalid_argument("PyArrayObject is not a numpy array!");

    return NpArray<T>(value);
}

template<typename T>
const NpArray<T> get_value(type<NpArray<T>> _, PyObject * value) {
    PyArrayObject * array = reinterpret_cast<PyArrayObject *>(value);
    return get_value<NpArray<T>>(array);
}


PyObject * _get_py_value_from_py_dict(PyObject * dict, const std::string& key){
    if(!check_is_dict(dict))
        throw std::invalid_argument("Python dictionary is null!");

    // PyObject * tmp_key;
    // PyObject * value;
    // Py_ssize_t pos = 0;
    // while (PyDict_Next(dict, &pos, &tmp_key, &value)) {
    //     auto key_str = get_value<std::string>(tmp_key);
    //     if(key_str == key){
    //         return value;
    //     }
    // }
    return PyDict_GetItemString(dict, key.c_str());
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
    return get_value<std::vector<VecType>>(py_value);
}

/**
 * Returns a C++ unordered_map from a Python dictionary that is inside another Pyhton
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
 * @return An unordered map of type <KeyType,ValueType> from the Pyhton's dictionary key.
 **/
template<typename KeyType, typename ValueType>
const std::unordered_map<KeyType, ValueType> get_map_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    return get_value<std::unordered_map<KeyType, ValueType>>(py_value);
}


template<typename KeyType, typename ValueType>
const ordered_map<KeyType, ValueType> get_ordered_map_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    return get_value<ordered_map<KeyType, ValueType>>(py_value);
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
ValueType get_value_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    if(py_value == Py_None)
        return {};

    return get_value<ValueType>(py_value);
}

#endif //_PYTHON_TO_CPP_HPP
