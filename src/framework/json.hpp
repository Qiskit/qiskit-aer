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

#ifndef _aer_framework_json_hpp_
#define _aer_framework_json_hpp_

#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <iostream>
#include <type_traits>

#include <nlohmann_json.hpp>
#include "framework/matrix.hpp"

namespace py = pybind11;
namespace nl = nlohmann;
using namespace pybind11::literals;
using json_t = nlohmann::json;

//============================================================================
// JSON Library helper functions
//============================================================================

namespace JSON {

/**
 * Load a json_t from a file. If the file name is 'stdin' or '-' the json_t will
 * be
 * loaded from the standard input stream.
 * @param name: file name to load.
 * @returns: the loaded json.
 */
json_t load(std::string name);

/**
 * Check if a key exists in a json_t object.
 * @param key: key name.
 * @param js: the json_t to search for key.
 * @returns: true if the key exists, false otherwise.
 */
bool check_key(std::string key, const json_t &js);

/**
 * Check if all keys exists in a json_t object.
 * @param keys: vector of key names.
 * @param js: the json_t to search for keys.
 * @returns: true if all keys exists, false otherwise.
 */
bool check_keys(std::vector<std::string> keys, const json_t &js);

/**
 * Load a json_t object value into a variable if the key name exists.
 * @param var: variable to store key value.
 * @param key: key name.
 * @param js: the json_t to search for key.
 * @returns: true if the keys exists and val was set, false otherwise.
 */
template <typename T> bool get_value(T &var, std::string key, const json_t &js);

} // end namespace JSON

//============================================================================
// JSON Conversion for complex STL types
//============================================================================

namespace std {

/**
 * Convert a complex number to a json list z -> [real(z), imag(z)].
 * @param js a json_t object to contain converted type.
 * @param z a complex number to convert.
 */
template <typename T> void to_json(json_t &js, const std::complex<T> &z);

/**
 * Convert a JSON value to a complex number z. If the json value is a float
 * it will be converted to a complex z = (val, 0.). If the json value is a
 * length two list it will be converted to a complex z = (val[0], val[1]).
 * @param js a json_t object to convert.
 * @param z a complex number to contain result.
 */
template <typename T> void from_json(const json_t &js, std::complex<T> &z);

/**
 * Convert a complex vector to a json list
 * v -> [ [real(v[0]), imag(v[0])], ...]
 * @param js a json_t object to contain converted type.
 * @param vec a complex vector to convert.
 */
template <typename RealType>
void to_json(json_t &js, const std::vector<std::complex<RealType>> &vec);

/**
 * Convert a JSON list to a complex vector. The input JSON value may be:
 * - an object with complex pair values: {'00': [re, im], ... }
 * - an object with real pair values: {'00': n, ... }
 * - an list with complex values: [ [a0re, a0im], ...]
 * - an list with real values: [a0, a1, ....]
 * @param js a json_t object to convert.
 * @param vec a complex vector to contain result.
 */
template <typename RealType>
void from_json(const json_t &js, std::vector<std::complex<RealType>> &vec);

/**
 * Convert a map with integer keys to a json. This converts the integer keys
 * to strings in the resulting json object.
 * @param js a json_t object to contain converted type.
 * @param map a map to convert.
 */
template <typename T1, typename T2>
void to_json(json_t &js, const std::map<int64_t, T1, T2> &map);

template <typename T1, typename T2>
void to_json(json_t &js, const std::map<uint64_t, T1, T2> &map);

/**
 * Convert a python object to a json.
 * @param js a json_t object to contain converted type.
 * @param o is a python object to convert.
 */
void to_json(json_t &js, const py::handle &o);

/**
 * Create a python object from a json
 * @param js a json_t object 
 * @param o is a reference to an existing (empty) python object
 */
void from_json(const json_t &js, py::object &o);

} // end namespace std.

/**
 * Convert a numpy array to a json object
 * @param arr is a numpy array 
 * @returns a json list (potentially of lists)
 */
template <typename T>
json_t numpy_to_json(py::array_t<T, py::array::c_style> arr);

/**
 * Convert a 1-d numpy array to a json object
 * @param arr is a numpy array 
 * @returns a json list (potentially of lists)
 */
template <typename T>
json_t numpy_to_json_1d(py::array_t<T, py::array::c_style> arr);

/**
 * Convert a 2-d numpy array to a json object
 * @param arr is a numpy array 
 * @returns a json list (potentially of lists)
 */
template <typename T>
json_t numpy_to_json_2d(py::array_t<T, py::array::c_style> arr);

/**
 * Convert a 3-d numpy array to a json object
 * @param arr is a numpy array 
 * @returns a json list (potentially of lists)
 */
template <typename T>
json_t numpy_to_json_3d(py::array_t<T, py::array::c_style> arr);

/**
 * Convert a matrix to a json.
 * @param js a json_t object to contain converted type.
 * @param mat a matrix to convert.
 */
template<class T> 
void from_json(const json_t &js, matrix<T> &mat);
template<class T>
void to_json(json_t &js, const matrix<T> &mat);

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Helper Functions
//------------------------------------------------------------------------------

json_t JSON::load(std::string name) {
  if (name == "") {
    json_t js;
    return js; // Return empty node if no config file
  }
  json_t js;
  if (name == "stdin" || name == "-") // Load from stdin
    std::cin >> js;
  else { // Load from file
    std::ifstream ifile;
    ifile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      ifile.open(name);
    } catch (std::exception &e) {
      throw std::runtime_error(std::string("no such file or directory"));
    }
    ifile >> js;
  }
  return js;
}

bool JSON::check_key(std::string key, const json_t &js) {
  // returns false if the value is 'null'
  if (js.find(key) != js.end() && !js[key].is_null())
    return true;
  else
    return false;
}

bool JSON::check_keys(std::vector<std::string> keys, const json_t &js) {
  bool pass = true;
  for (auto s : keys)
    pass &= check_key(s, js);
  return pass;
}

template <typename T>
bool JSON::get_value(T &var, std::string key, const json_t &js) {
  if (check_key(key, js)) {
    var = js[key].get<T>();
    return true;
  } else {
    return false;
  }
}

//------------------------------------------------------------------------------
// JSON Conversion
//------------------------------------------------------------------------------

template <typename RealType>
void std::to_json(json_t &js, const std::complex<RealType> &z) {
  js = std::pair<RealType, RealType>{z.real(), z.imag()};
}

template <typename RealType>
void std::from_json(const json_t &js, std::complex<RealType> &z) {
  if (js.is_number())
    z = std::complex<RealType>{js.get<RealType>()};
  else if (js.is_array() && js.size() == 2) {
    z = std::complex<RealType>{js[0].get<RealType>(), js[1].get<RealType>()};
  } else {
    throw std::invalid_argument(
        std::string("JSON: invalid complex number"));
  }
}

template <typename RealType>
void std::to_json(json_t &js, const std::vector<std::complex<RealType>> &vec) {
  std::vector<std::vector<RealType>> out;
  for (auto &z : vec) {
    out.push_back(std::vector<RealType>{real(z), imag(z)});
  }
  js = out;
}

template <typename RealType>
void std::from_json(const json_t &js, std::vector<std::complex<RealType>> &vec) {
  std::vector<std::complex<RealType>> ret;
  if (js.is_array()) {
    for (auto &elt : js)
      ret.push_back(elt);
    vec = ret;
  } 
  else {
    throw std::invalid_argument(
        std::string("JSON: invalid complex vector."));
  }
}

// Int-key maps
template <typename T1, typename T2>
void std::to_json(json_t &js, const std::map<int64_t, T1, T2> &map) {
  js = json_t();
  for (const auto &p : map) {
    std::string key = std::to_string(p.first);
    js[key] = p.second;
  }
}

// Int-key maps
template <typename T1, typename T2>
void std::to_json(json_t &js, const std::map<uint64_t, T1, T2> &map) {
  js = json_t();
  for (const auto &p : map) {
    std::string key = std::to_string(p.first);
    js[key] = p.second;
  }
}

template <typename T>
json_t numpy_to_json_1d(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dims must be 1");
    }

    T *ptr = (T *) buf.ptr;
    size_t D0 = buf.shape[0];

    std::vector<T> tbr; // to be returned 
    for (size_t n0 = 0; n0 < D0; n0++)
        tbr.push_back(ptr[n0]);

    return std::move(tbr);
}

template <typename T>
json_t numpy_to_json_2d(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dims must be 2");
    }

    T *ptr = (T *) buf.ptr;
    size_t D0 = buf.shape[0];
    size_t D1 = buf.shape[1];

    std::vector<std::vector<T > > tbr; // to be returned
    for (size_t n0 = 0; n0 < D0; n0++) {
        std::vector<T> tbr1;
        for (size_t n1 = 0; n1 < D1; n1++) {
            tbr1.push_back(ptr[n1 + D1*n0]);
        }
        tbr.push_back(tbr1);
    }

    return std::move(tbr);

}

template <typename T>
json_t numpy_to_json_3d(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Number of dims must be 3");
    }
    T *ptr = (T *) buf.ptr;
    size_t D0 = buf.shape[0];
    size_t D1 = buf.shape[1];
    size_t D2 = buf.shape[2];

    // to be returned
    std::vector<std::vector<std::vector<T > > > tbr;
    for (size_t n0 = 0; n0 < D0; n0++) {
        std::vector<std::vector<T> > tbr1;
        for (size_t n1 = 0; n1 < D1; n1++) {
            std::vector<T> tbr2;
            for (size_t n2 = 0; n2 < D2; n2++) {
                tbr2.push_back(ptr[n2 + D2*(n1 + D1*n0)]);
            }
            tbr1.push_back(tbr2);
        }
        tbr.push_back(tbr1);
    }

    return std::move(tbr);

}

template <typename T>
json_t numpy_to_json(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    //std::cout << "buff dim: " << buf.ndim << std::endl;

    if (buf.ndim == 1) {
        return numpy_to_json_1d(arr);
    } else if (buf.ndim == 2) {
        return numpy_to_json_2d(arr);
    } else if (buf.ndim == 3) {
        return numpy_to_json_3d(arr);
    } else {
        throw std::runtime_error("Invalid number of dimensions!");
    }
    json_t tbr;
    return tbr;
}

void std::to_json(json_t &js, const py::handle &obj) {
    if (py::isinstance<py::bool_>(obj)) {
        js = obj.cast<nl::json::boolean_t>();
        //js = obj.cast<bool>();
    } else if (py::isinstance<py::int_>(obj)) {
        js = obj.cast<nl::json::number_integer_t>();
    } else if (py::isinstance<py::float_>(obj)) {
        js = obj.cast<nl::json::number_float_t>();
    } else if (py::isinstance<py::str>(obj)) {
        js = obj.cast<nl::json::string_t>();
    } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        js = nl::json::array();
        for (py::handle value: obj)
        {
            js.push_back(value);
        }
    } else if (py::isinstance<py::dict>(obj)) {
        for (auto item : py::cast<py::dict>(obj))
        {
            js[item.first.cast<nl::json::string_t>()] = item.second;
        }
    } else if (py::isinstance<py::array_t<double> >(obj)) {
        js = numpy_to_json(obj.cast<py::array_t<double, py::array::c_style> >());
    } else if (py::isinstance<py::array_t<std::complex<double> > >(obj)) {
        js = numpy_to_json(obj.cast<py::array_t<std::complex<double>, py::array::c_style> >());
    } else if (std::string(py::str(obj.get_type())) == "<class \'complex\'>") {
        auto tmp = obj.cast<std::complex<double>>();
        js.push_back(tmp.real());
        js.push_back(tmp.imag());
    } else if (obj.is_none()) {
        return;
    } else {
        throw std::runtime_error("to_json not implemented for this type of object: " + obj.cast<std::string>());
    }
}

void std::from_json(const json_t &js, py::object &o) {
    if (js.is_boolean()) {
        o = py::bool_(js.get<nl::json::boolean_t>());
    } else if (js.is_number()) {
        if (js.is_number_float()) {
            o = py::float_(js.get<nl::json::number_float_t>());
        } else if (js.is_number_unsigned()) {
            o = py::int_(js.get<nl::json::number_unsigned_t>());
        } else {
            o = py::int_(js.get<nl::json::number_integer_t>());
        }
    } else if (js.is_string()) {
        o = py::str(js.get<nl::json::string_t>());
    } else if (js.is_array()) {
        std::vector<py::object> obj(js.size());
        for (auto i = 0; i < js.size(); i++)
        {
            py::object tmp;
            from_json(js[i], tmp);
            obj[i] = tmp;
        }
        o = py::cast(obj);
    } else if (js.is_object()) {
        py::dict obj;
        for (json_t::const_iterator it = js.cbegin(); it != js.cend(); ++it)
        {
            py::object tmp;
            from_json(it.value(), tmp);
            obj[py::str(it.key())] = tmp;
        }
        o = std::move(obj);
    } else if (js.is_null()) {
        o = py::none();
    } else {
        throw std::runtime_error("from_json not implemented for this json::type: " + js.dump());
    }
}

// Matrices
//------------------------------------------------------------------------------
// Implementation: JSON Conversion
//------------------------------------------------------------------------------

template <typename T> void to_json(json_t &js, const matrix<T> &mat) {
  js = json_t();
  size_t rows = mat.GetRows();
  size_t cols = mat.GetColumns();
  for (size_t r = 0; r < rows; r++) {
    std::vector<T> mrow;
    for (size_t c = 0; c < cols; c++)
      mrow.push_back(mat(r, c));
    js.push_back(mrow);
  }
}


template <typename T> void from_json(const json_t &js, matrix<T> &mat) {
  // Check JSON is an array
  if(!js.is_array()) {
    throw std::invalid_argument(
        std::string("JSON: invalid matrix (not array)."));
  }
  // Check JSON isn't empty
  if(js.empty()) {
    throw std::invalid_argument(
        std::string("JSON: invalid matrix (empty array)."));
  }
  // check rows are all same length
  bool rows_valid = js.is_array() && !js.empty();
  // Check all entries of array are same size
  size_t ncols = js[0].size();
  size_t nrows = js.size();
  for (auto &row : js)
    rows_valid &= (row.is_array() && row.size() == ncols);
  if(!rows_valid) {
    throw std::invalid_argument(
        std::string("JSON: invalid matrix (rows different sizes)."));
  }
  // Matrix looks ok, now we parse it
  mat = matrix<T>(nrows, ncols);
  for (size_t r = 0; r < nrows; r++)
    for (size_t c = 0; c < ncols; c++)
      mat(r, c) = js[r][c].get<T>();
}

//------------------------------------------------------------------------------
#endif
