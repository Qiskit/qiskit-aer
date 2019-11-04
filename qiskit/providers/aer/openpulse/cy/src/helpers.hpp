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

#include <utility>
#include <unordered_map>
#include <map>
#include <vector>
#include <complex>
#include <type_traits>
#include <Python.h>
#include <exprtk.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <numpy/arrayobject.h>
#include <muparserx/mpParser.h>


/**
 * TODO: There's a lot of copying due to converting from Pyhton C strucutres
 * to C++ stl containers. We can avoid this by wrapping the former and avoiding
 * copies
 **/




static bool init_numpy(){
    static bool initialized = false;
    if(!initialized){
        import_array();
        initialized = true;
    }
};

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

bool _check_is_tuple(PyObject * value){
    if(value == nullptr)
        throw std::invalid_argument("Pyhton tuple is null!");

    // Check that it's a tuple
    if(!PyTuple_Check(value))
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


/**
 * Helper Types
 **/
using complex_t = std::complex<double>;

template<typename VecType>
class NpArray {
  public:
	NpArray(){}
    NpArray(const std::vector<VecType>& data, const std::vector<int>& shape) :
        data(data), shape(shape){
    }
	NpArray(PyArrayObject * array){
		_populate_data(array);
		_populate_shape(array);
	}

    std::vector<VecType> data;
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
		data = other.data;
		shape = other.shape;
		return *this;
	}

    bool operator==(const NpArray<VecType>& other) const {
        if(other.data.size() != data.size() ||
           other.shape.size() != shape.size())
           return false;

        for(auto i = 0; i < other.data.size(); ++i){
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
		if(!_check_is_np_array(array))
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
		/* Handle zero-sized arrays specially */
		if (PyArray_SIZE(array) == 0){
			data = {};
			return;
		}
		/* TODO This is faster if we deal with PyObject directly */
		PyObject * py_list = PyArray_ToList(array);
		data = get_value<std::vector<VecType>>(py_list);
	}
};

template<typename T>
void jlog(const std::string& msg, const T& value){
    spdlog::debug("{}: {}", msg, value);
}

template<>
void jlog(const std::string& msg, const complex_t& values){
    spdlog::debug("{}: [{},{}i]", msg, values.real(), values.imag());
}

template<typename T>
void jlog(const std::string& msg, const NpArray<T>& values){
    spdlog::debug("{}", msg);
    spdlog::debug(".shape: ");
    for(const auto& shape : values.shape)
        spdlog::debug("{} ", shape);

    spdlog::debug("\n.data: ");
    for(const auto& val : values.data){
        jlog("", val);
    }
}

template<typename T>
void jlog(const std::string& msg, const std::vector<T>& values){
    spdlog::debug("{}", msg);
    for(const auto& val : values){
        jlog("", val);
    }
}

template<>
void jlog(const std::string& msg, const std::map<std::string, std::vector<std::vector<double>>>& values){
    spdlog::debug("{}", msg);
    for(const auto& val : values){
        for(const auto& inner: val.second){
            for(const auto& inner2: inner){
                spdlog::debug("{}:{} ", val.first, inner2);
            }
        }
    }
}

template<>
void jlog(const std::string& msg, const std::map<std::string, double>& values){
    spdlog::debug("{}", msg);
    for(const auto& val : values){
        spdlog::debug("{}:{} ", val.first, val.second);
    }
}

template<>
void jlog(const std::string& msg, const std::map<std::string, std::vector<NpArray<double>>>& values){
    spdlog::debug("{}", msg);
    for(const auto& val : values){
        for(const auto& inner: val.second){
            jlog(val.first, inner);
        }
    }
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



template<>
uint8_t get_value(type<uint8_t> _, PyObject * value){
    return get_value<long>(value);
}

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

/* WARNING: There's no support for variadic templates in Cython, so
   we use a std::pair because there's no more than two types in the Python
   tuples so far, so as we are fine for now... */
template<typename T, typename U>
std::pair<T, U> get_value(type<std::pair<T, U>> _, PyObject * value){
    if(!_check_is_tuple(value))
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
std::map<std::string, ValueType> get_value(type<std::map<std::string, ValueType>> _, PyObject * value){
    if(!_check_is_dict(value))
        throw std::invalid_argument("PyObject is not a dictonary!!");

    struct channel_order {
        bool operator()(const std::string& a, const std::string& b){


        }
    };

    std::map<std::string, ValueType, channel_order> map;

    PyObject *key, *val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(value, &pos, &key, &val)) {
        auto inner_key = get_value<std::string>(key);
        auto inner_value = get_value<ValueType>(val);
        map.emplace(inner_key, inner_value);
    }
    return map;
}

template<typename T>
const NpArray<T> get_value(type<NpArray<T>> _, PyArrayObject * value){
    if(!_check_is_np_array(value))
        throw std::invalid_argument("PyArrayObject is not a numpy array!");

    return NpArray<T>(value);
}

template<typename T>
const NpArray<T> get_value(type<NpArray<T>> _, PyObject * value){
    PyArrayObject * array = reinterpret_cast<PyArrayObject *>(value);
    return get_value<NpArray<T>>(array);
}

PyObject * _get_py_value_from_py_dict(PyObject * dict, const std::string& key){
    if(!_check_is_dict(dict))
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
 * Returns a C++ map from a Python dictionary that is inside another Pyhton
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
 * @return A map of type <KeyType,ValueType> from the Pyhton's dictionary key.
 **/
template<typename KeyType, typename ValueType>
const std::map<KeyType, ValueType> get_map_from_dict_item(PyObject * dict, const std::string& item_key){
    PyObject * py_value = _get_py_value_from_py_dict(dict, item_key);
    return get_value<std::map<KeyType, ValueType>>(py_value);
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


/**
 * Math expression evaluator for the Hamiltonian terms
 **/
// complex_t evaluate_hamiltonian_expression(const std::string& expr_string,
//                                   const std::vector<double>& vars,
//                                   const std::vector<std::string>& vars_names,
//                                   const std::map<std::string, complex_t>& chan_values){
//     exprtk::symbol_table<complex_t> symbol_table;
//     auto pi = M_PI;
//     auto complex_pi = static_cast<complex_t>(pi);
//     symbol_table.add_variable("np.pi", complex_pi);

//     for(const auto& idx_var : enumerate(vars)){
//         auto index = idx_var.first;
//         auto var = static_cast<complex_t>(idx_var.second);
//         symbol_table.add_variable(vars_names[index], var);
//     }

//     for(const auto& idx_channel : enumerate(chan_values)){
//         auto index = idx_channel.first;
//         auto channel = idx_channel.second.first; // The std::string of the map
//         auto val = idx_channel.second.second; // The complex_t of the map
//         symbol_table.add_variable(channel, val);
//     }

//     exprtk::expression<complex_t> expression;
//     expression.register_symbol_table(symbol_table);

//     exprtk::parser<complex_t> parser;

//     if (!parser.compile(expr_string, expression)){
//         throw std::invalid_argument("Cannot evaluate hamiltonian expression: " + expr_string);
//     }

//     return expression.value();
// }

complex_t evaluate_hamiltonian_expression(const std::string& expr_string,
                                  const std::vector<double>& vars,
                                  const std::vector<std::string>& vars_names,
                                  const std::unordered_map<std::string, complex_t>& chan_values){
    using namespace mup;
    ParserX parser;
    Value pi(M_PI);
    parser.DefineVar("npi", Variable(&pi));

    std::vector<Value> values;
    values.reserve(vars.size() + chan_values.size());
    for(const auto& idx_var : enumerate(vars)){
        auto index = idx_var.first;
        auto var = static_cast<complex_t>(idx_var.second);
        values.emplace_back(Value(var));
        parser.DefineVar(vars_names[index], Variable(&values[values.size()-1]));
    }

    for(const auto& idx_channel : enumerate(chan_values)){
        auto index = idx_channel.first;
        auto channel = idx_channel.second.first; // The std::string of the map
        auto var = idx_channel.second.second; // The complex_t of the map
        values.emplace_back(Value(var));
        parser.DefineVar(channel, Variable(&values[values.size()-1]));
    }

    const auto replace = [](const std::string& from, const std::string& to, std::string& where) -> std::string {
        size_t start_pos = 0;
        while((start_pos = where.find(from, start_pos)) != std::string::npos) {
            where.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
        return where;
    };

    parser.SetExpr(replace("np.pi", "npi", const_cast<std::string&>(expr_string)));
    Value result = parser.Eval();

    return result.GetComplex();
}


/**
 * Fast CSR Matrix representation
 **/
class FastCsrMatrix{
  public:
    FastCsrMatrix(){}
    FastCsrMatrix(PyObject * obj){
        auto dict = PyObject_GenericGetDict(obj, nullptr);
        data = get_value_from_dict_item<NpArray<complex_t>>(dict, "data");
        indices = get_value_from_dict_item<NpArray<long>>(dict, "indices");
        indptr = get_value_from_dict_item<NpArray<long>>(dict, "indptr");

    }
    NpArray<complex_t> data;
    NpArray<long> indices;
    NpArray<long> indptr;
};

template<>
FastCsrMatrix get_value(type<FastCsrMatrix>_, PyObject * value) {
    return FastCsrMatrix(value);
}


/**
 * This is the Qutip Quantum Object repesentation
 **/
struct QuantumObj {
    public:
	QuantumObj(){}
	QuantumObj(PyObject * obj){
        auto dict = PyObject_GenericGetDict(obj, nullptr);
		data = get_value_from_dict_item<FastCsrMatrix>(dict, "_data");
		is_hermitian = static_cast<bool>(get_value_from_dict_item<long>(dict, "_isherm"));
		type = get_value_from_dict_item<std::string>(dict, "_type");
		super_rep = get_value_from_dict_item<std::string>(dict, "superrep");
		is_unitary = static_cast<bool>(get_value_from_dict_item<long>(dict, "_isunitary"));
		dims = get_value_from_dict_item<std::vector<std::vector<long>>>(dict, "dims");
	}

    FastCsrMatrix data;
    bool is_hermitian;
    std::string type;
    std::string super_rep;
    bool is_unitary;
    std::vector<std::vector<long>> dims;
};

template<>
QuantumObj get_value(type<QuantumObj>_, PyObject * value) {
    return QuantumObj(value);
}



#endif // _HELPERS_HPP