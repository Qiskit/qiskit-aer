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

#ifndef _aer_framework_linalg_linops_map_hpp_
#define _aer_framework_linalg_linops_map_hpp_

#include <functional>
#include <map>
#include "framework/linalg/almost_equal.hpp"
#include "framework/linalg/enable_if_numeric.hpp"

namespace AER {
namespace Linalg {

// This defines functions add, sub, mul, div and iadd, imul, isub, idiv
// that work for numeric map types

//----------------------------------------------------------------------------
// Linear operations
//----------------------------------------------------------------------------
template <class T1, class T2, class T3, class T4,
          typename = enable_if_numeric_t<T2>>
std::map<T1, T2, T3, T4> add(const std::map<T1, T2, T3, T4>& lhs,
                             const std::map<T1, T2, T3, T4>& rhs) {
  std::map<T1, T2, T3, T4> result = lhs;
  for (const auto& pair : rhs) {
    result[pair.first] = std::plus<T2>()(result[pair.first], pair.second);
  }
  return result;
}

template <class T1, class T2, class T3, class T4,
          typename = enable_if_numeric_t<T2>>
std::map<T1, T2, T3, T4>& iadd(std::map<T1, T2, T3, T4>& lhs,
                               const std::map<T1, T2, T3, T4>& rhs) {
  for (const auto& pair : rhs) {
    lhs[pair.first] = std::plus<T2>()(lhs[pair.first], pair.second);
  }
  return lhs;
}

template <class T1, class T2, class T3, class T4,
          typename = enable_if_numeric_t<T2>>
std::map<T1, T2, T3, T4> sub(const std::map<T1, T2, T3, T4>& lhs,
                             const std::map<T1, T2, T3, T4>& rhs) {
  std::map<T1, T2, T3, T4> result = lhs;
  for (const auto& pair : rhs) {
    result[pair.first] = std::minus<T2>()(result[pair.first], pair.second);
  }
  return result;
}

template <class T1, class T2, class T3, class T4,
          typename = enable_if_numeric_t<T2>>
std::map<T1, T2, T3, T4>& isub(std::map<T1, T2, T3, T4>& lhs,
                               const std::map<T1, T2, T3, T4>& rhs) {
  for (const auto& pair : rhs) {
    lhs[pair.first] = std::minus<T2>()(lhs[pair.first], pair.second);
  }
  return lhs;
}

//----------------------------------------------------------------------------
// Affine operations
//----------------------------------------------------------------------------
template <class T1, class T2, class T3, class T4, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::map<T1, T2, T3, T4> add(const std::map<T1, T2, T3, T4>& data,
                             const Scalar& val) {
  std::map<T1, T2, T3, T4> result;
  for (const auto& pair : data) {
    result[pair.first] = std::plus<T2>()(pair.second, val);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::map<T1, T2, T3, T4>& iadd(std::map<T1, T2, T3, T4>& data,
                               const Scalar& val) {
  for (const auto& pair : data) {
    data[pair.first] = std::plus<T2>()(data[pair.first], val);
  }
  return data;
}

template <class T1, class T2, class T3, class T4, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::map<T1, T2, T3, T4> sub(const std::map<T1, T2, T3, T4>& data,
                             const Scalar& val) {
  std::map<T1, T2, T3, T4> result;
  for (const auto& pair : data) {
    result[pair.first] = std::minus<T2>()(pair.second, val);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::map<T1, T2, T3, T4>& isub(std::map<T1, T2, T3, T4>& data,
                               const Scalar& val) {
  for (const auto& pair : data) {
    data[pair.first] = std::plus<T2>()(data[pair.first], val);
  }
  return data;
}

//----------------------------------------------------------------------------
// Scalar operations
//----------------------------------------------------------------------------

template <class T1, class T2, class T3, class T4, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::map<T1, T2, T3, T4> mul(const std::map<T1, T2, T3, T4>& data,
                             const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  std::map<T1, T2, T3, T4> result;
  for (const auto& pair : data) {
    result[pair.first] = std::multiplies<T2>()(pair.second, val);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::map<T1, T2, T3, T4>& imul(std::map<T1, T2, T3, T4>& data,
                               const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  for (const auto& pair : data) {
    data[pair.first] = std::multiplies<T2>()(data[pair.first], val);
  }
  return data;
}

template <class T1, class T2, class T3, class T4, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::map<T1, T2, T3, T4> div(const std::map<T1, T2, T3, T4>& data,
                             const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  std::map<T1, T2, T3, T4> result;
  for (const auto& pair : data) {
    result[pair.first] = std::divides<T2>()(pair.second, val);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class Scalar,
          typename = enable_if_numeric_t<T2>,
          typename = enable_if_numeric_t<Scalar>>
std::map<T1, T2, T3, T4>& idiv(std::map<T1, T2, T3, T4>& data,
                               const Scalar& val) {
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