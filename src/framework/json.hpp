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

#include <iostream>
#include <type_traits>

#include <nlohmann_json.hpp>
#include "framework/matrix.hpp"

namespace nl = nlohmann;
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

} // end namespace std.

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
