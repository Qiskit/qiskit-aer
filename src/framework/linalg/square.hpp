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

#ifndef _aer_framework_linalg_square_hpp_
#define _aer_framework_linalg_square_hpp_

#include <array>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "framework/json.hpp"
#include "framework/matrix.hpp"
#include "framework/types.hpp"
#include "framework/linalg/enable_if_numeric.hpp"

namespace AER {
namespace Linalg {

// This defines functions 'square' for entrywise square of
// numeric types, and 'isquare' for inplace entywise square.

//----------------------------------------------------------------------------
// Entrywise square of std::array
//----------------------------------------------------------------------------

// Return entrywise square of a vector
template <class T, size_t N, typename = enable_if_numeric_t<T>>
std::array<T, N> square(const std::array<T, N>& data) {
  std::array<T, N> result = data;
  return isquare(result);
}

// Return inplace entrywise square of a vector
template <class T, size_t N, typename = enable_if_numeric_t<T>>
std::array<T, N>& isquare(std::array<T, N>& data) {
  std::transform(data.begin(), data.end(), data.begin(), data.begin(),
                 std::multiplies<T>());
  return data;
}

//----------------------------------------------------------------------------
// Entrywise square of std::vector
//----------------------------------------------------------------------------

// Return entrywise square of a vector
template <class T, typename = enable_if_numeric_t<T>>
std::vector<T> square(const std::vector<T>& data) {
  std::vector<T> result;
  result.reserve(data.size());
  std::transform(data.begin(), data.end(), data.begin(), std::back_inserter(result),
                 std::multiplies<T>());
  return result;
}

// Return inplace entrywise square of a vector
template <class T, typename = enable_if_numeric_t<T>>
std::vector<T>& isquare(std::vector<T>& data) {
  std::transform(data.begin(), data.end(), data.begin(), data.begin(),
                 std::multiplies<T>());
  return data;
}

//----------------------------------------------------------------------------
// Entrywise square of std::map
//----------------------------------------------------------------------------

template <class T1, class T2, class T3, class T4, typename = enable_if_numeric_t<T2>>
std::map<T1, T2, T3, T4> square(const std::map<T1, T2, T3, T4>& data) {
  std::map<T1, T2, T3, T4> result;
  for (const auto& pair : data) {
    result[pair.first] = pair.second * pair.second;
  }
  return result;
}

template <class T1, class T2, class T3, class T4, typename = enable_if_numeric_t<T2>>
std::map<T1, T2, T3, T4>& isquare(std::map<T1, T2, T3, T4>& data) {
  for (auto& pair : data) {
    pair.second *= pair.second;
  }
  return data;
}

//----------------------------------------------------------------------------
// Entrywise square of std::unordered_map
//----------------------------------------------------------------------------

template <class T1, class T2, class T3, class T4, class T5, typename = enable_if_numeric_t<T2>>
std::unordered_map<T1, T2, T3, T4, T5> square(
    const std::unordered_map<T1, T2, T3, T4, T5>& data) {
  std::unordered_map<T1, T2, T3, T4, T5> result;
  for (const auto& pair : data) {
    result[pair.first] = pair.second * pair.second;
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class T5, typename = enable_if_numeric_t<T2>>
std::unordered_map<T1, T2, T3, T4, T5>& isquare(std::unordered_map<T1, T2, T3, T4, T5>& data) {
  for (auto& pair : data) {
    pair.second *= pair.second;
  }
  return data;
}

//----------------------------------------------------------------------------
// Entrywise square of matrix
//----------------------------------------------------------------------------

template <class T, typename = enable_if_numeric_t<T>>
matrix<T>& isquare(matrix<T>& data) {
  for (size_t j = 0; j < data.size(); j++) {
    data[j] *= data[j];
  }
  return data;
}

template <class T, typename = enable_if_numeric_t<T>>
matrix<T> square(const matrix<T>& data) {
  matrix<T> result = data;
  return isquare(result);
}

//----------------------------------------------------------------------------
// Entrywise square of JSON
//----------------------------------------------------------------------------

json_t& isquare(json_t& data) {
  // Terminating case
  if (data.is_number()) {
    double val = data;
    data = val * val;
    return data;
  }
  // Recursive cases
  if (data.is_array()) {
    for (size_t pos = 0; pos < data.size(); pos++) {
      isquare(data[pos]);
    }
    return data;
  } 
  if (data.is_object()) {
    for (auto it = data.begin(); it != data.end(); ++it) {
      isquare(data[it.key()]);
    }
    return data;
  }
  throw std::invalid_argument("Input JSONs cannot be squared.");
}

json_t square(const json_t& data) {
  json_t result = data;
  return isquare(result);
}

//----------------------------------------------------------------------------
// Square of general type
//----------------------------------------------------------------------------

template <class T>
T square(const T& data) {
  return data * data;
}

template <class T>
T& isquare(T& data) {
  data *= data;
  return data;
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif