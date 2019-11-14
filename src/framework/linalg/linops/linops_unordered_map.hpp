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

#ifndef _aer_framework_linalg_linops_unordered_map_hpp_
#define _aer_framework_linalg_linops_unordered_map_hpp_

#include <functional>
#include <unordered_map>

#include "framework/linalg/almost_equal.hpp"
#include "framework/linalg/enable_if_numeric.hpp"
namespace AER {
namespace Linalg {

// This defines functions add, sub, mul, div and iadd, imul, isub, idiv
// that work for numeric unordered_map types

//----------------------------------------------------------------------------
// Linear operations
//----------------------------------------------------------------------------
template <class T1, class T2, class T3, class T4, class T5,
          typename = enable_if_numeric_t<T2>>
std::unordered_map<T1, T2, T3, T4, T5> add(
    const std::unordered_map<T1, T2, T3, T4, T5>& lhs,
    const std::unordered_map<T1, T2, T3, T4, T5>& rhs) {
  std::unordered_map<T1, T2, T3, T4, T5> result = lhs;
  for (const auto& pair : rhs) {
    result[pair.first] = std::plus<T2>()(result[pair.first], pair.second);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class T5,
          typename = enable_if_numeric_t<T2>>
std::unordered_map<T1, T2, T3, T4, T5>& iadd(
    std::unordered_map<T1, T2, T3, T4, T5>& lhs,
    const std::unordered_map<T1, T2, T3, T4, T5>& rhs) {
  for (const auto& pair : rhs) {
    lhs[pair.first] = std::plus<T2>()(lhs[pair.first], pair.second);
  }
  return lhs;
}

template <class T1, class T2, class T3, class T4, class T5,
          typename = enable_if_numeric_t<T2>>
std::unordered_map<T1, T2, T3, T4, T5> sub(
    const std::unordered_map<T1, T2, T3, T4, T5>& lhs,
    const std::unordered_map<T1, T2, T3, T4, T5>& rhs) {
  std::unordered_map<T1, T2, T3, T4, T5> result = lhs;
  for (const auto& pair : rhs) {
    result[pair.first] = std::minus<T2>()(result[pair.first], pair.second);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class T5,
          typename = enable_if_numeric_t<T2>>
std::unordered_map<T1, T2, T3, T4, T5>& isub(
    std::unordered_map<T1, T2, T3, T4, T5>& lhs,
    const std::unordered_map<T1, T2, T3, T4, T5>& rhs) {
  for (const auto& pair : rhs) {
    lhs[pair.first] = std::minus<T2>()(lhs[pair.first], pair.second);
  }
  return lhs;
}

//----------------------------------------------------------------------------
// Affine operations
//----------------------------------------------------------------------------
template <class T1, class T2, class T3, class T4, class T5, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::unordered_map<T1, T2, T3, T4, T5> add(
    const std::unordered_map<T1, T2, T3, T4, T5>& data, const Scalar& val) {
  std::unordered_map<T1, T2, T3, T4, T5> result;
  for (const auto& pair : data) {
    result[pair.first] = std::plus<T2>()(pair.second, val);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class T5, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::unordered_map<T1, T2, T3, T4, T5>& iadd(
    std::unordered_map<T1, T2, T3, T4, T5>& data, const Scalar& val) {
  for (const auto& pair : data) {
    data[pair.first] = std::plus<T2>()(data[pair.first], val);
  }
  return data;
}

template <class T1, class T2, class T3, class T4, class T5, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::unordered_map<T1, T2, T3, T4, T5> sub(
    const std::unordered_map<T1, T2, T3, T4, T5>& data, const Scalar& val) {
  std::unordered_map<T1, T2, T3, T4, T5> result;
  for (const auto& pair : data) {
    result[pair.first] = std::minus<T2>()(pair.second, val);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class T5, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::unordered_map<T1, T2, T3, T4, T5>& isub(
    std::unordered_map<T1, T2, T3, T4, T5>& data, const Scalar& val) {
  for (const auto& pair : data) {
    data[pair.first] = std::plus<T2>()(data[pair.first], val);
  }
  return data;
}

//----------------------------------------------------------------------------
// Scalar operations
//----------------------------------------------------------------------------

template <class T1, class T2, class T3, class T4, class T5, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::unordered_map<T1, T2, T3, T4, T5> mul(
    const std::unordered_map<T1, T2, T3, T4, T5>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  std::unordered_map<T1, T2, T3, T4, T5> result;
  for (const auto& pair : data) {
    result[pair.first] = std::multiplies<T2>()(pair.second, val);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class T5, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::unordered_map<T1, T2, T3, T4, T5>& imul(
    std::unordered_map<T1, T2, T3, T4, T5>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  for (const auto& pair : data) {
    data[pair.first] = std::multiplies<T2>()(data[pair.first], val);
  }
  return data;
}

template <class T1, class T2, class T3, class T4, class T5, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::unordered_map<T1, T2, T3, T4, T5> div(
    const std::unordered_map<T1, T2, T3, T4, T5>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  std::unordered_map<T1, T2, T3, T4, T5> result;
  for (const auto& pair : data) {
    result[pair.first] = std::divides<T2>()(pair.second, val);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class T5, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::unordered_map<T1, T2, T3, T4, T5>& idiv(
    std::unordered_map<T1, T2, T3, T4, T5>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  for (const auto& pair : data) {
    data[pair.first] = std::divides<T2>()(data[pair.first], val);
  }
  return data;
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif