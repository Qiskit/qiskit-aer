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

#ifndef _aer_framework_linalg_linops_vector_hpp_
#define _aer_framework_linalg_linops_vector_hpp_

#include <functional>
#include <vector>

#include "framework/linalg/almost_equal.hpp"
#include "framework/linalg/enable_if_numeric.hpp"

namespace AER {
namespace Linalg {

// This defines functions add, sub, mul, div and iadd, imul, isub, idiv
// that work for numeric vector types

//----------------------------------------------------------------------------
// Linear operations
//----------------------------------------------------------------------------
template <class T, typename = enable_if_numeric_t<T>>
std::vector<T> add(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("Cannot add two vectors of different sizes.");
  }
  std::vector<T> result;
  result.reserve(lhs.size());
  std::transform(lhs.begin(), lhs.end(), rhs.begin(),
                 std::back_inserter(result), std::plus<T>());
  return result;
}

template <class T, typename = enable_if_numeric_t<T>>
std::vector<T>& iadd(std::vector<T>& lhs, const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("Cannot add two vectors of different sizes.");
  }
  std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
                 std::plus<T>());
  return lhs;
}

template <class T, typename = enable_if_numeric_t<T>>
std::vector<T> sub(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("Cannot add two vectors of different sizes.");
  }
  std::vector<T> result;
  result.reserve(lhs.size());
  std::transform(lhs.begin(), lhs.end(), rhs.begin(),
                 std::back_inserter(result), std::minus<T>());
  return result;
}

template <class T, typename = enable_if_numeric_t<T>>
std::vector<T>& isub(std::vector<T>& lhs, const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("Cannot add two vectors of different sizes.");
  }
  std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
                 std::minus<T>());
  return lhs;
}

//----------------------------------------------------------------------------
// Affine operations
//----------------------------------------------------------------------------
template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::vector<T> add(const std::vector<T>& data, const Scalar& val) {
  std::vector<T> result;
  result.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(result),
                 std::bind(std::plus<T>(), std::placeholders::_1, val));
  return result;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::vector<T>& iadd(std::vector<T>& data, const Scalar& val) {
  std::transform(data.begin(), data.end(), data.begin(),
                 std::bind(std::plus<T>(), std::placeholders::_1, val));
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::vector<T> sub(const std::vector<T>& data, const Scalar& val) {
  std::vector<T> result;
  result.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(result),
                 std::bind(std::minus<T>(), std::placeholders::_1, val));
  return result;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::vector<T>& isub(std::vector<T>& data, const Scalar& val) {
  std::transform(data.begin(), data.end(), data.begin(),
                 std::bind(std::minus<T>(), std::placeholders::_1, val));
  return data;
}

//----------------------------------------------------------------------------
// Scalar operations
//----------------------------------------------------------------------------
template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::vector<T> mul(const std::vector<T>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  std::vector<T> result;
  result.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(result),
                 std::bind(std::multiplies<T>(), std::placeholders::_1, val));
  return result;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::vector<T>& imul(std::vector<T>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  std::transform(data.begin(), data.end(), data.begin(),
                 std::bind(std::multiplies<T>(), std::placeholders::_1, val));
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::vector<T> div(const std::vector<T>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  std::vector<T> result;
  result.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(result),
                 std::bind(std::divides<T>(), std::placeholders::_1, val));
  return result;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::vector<T>& idiv(std::vector<T>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  std::transform(data.begin(), data.end(), data.begin(),
                 std::bind(std::divides<T>(), std::placeholders::_1, val));
  return data;
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif