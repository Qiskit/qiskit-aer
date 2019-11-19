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

#ifndef _aer_framework_linalg_linops_array_hpp_
#define _aer_framework_linalg_linops_array_hpp_

#include <array>
#include <functional>

#include "framework/linalg/almost_equal.hpp"
#include "framework/linalg/enable_if_numeric.hpp"

namespace AER {
namespace Linalg {

// This defines functions add, sub, mul, div and iadd, imul, isub, idiv
// that work for numeric vector types

//----------------------------------------------------------------------------
// Linear operations
//----------------------------------------------------------------------------
template <class T, size_t N, typename = enable_if_numeric_t<T>>
std::array<T, N>& iadd(std::array<T, N>& lhs, const std::array<T, N>& rhs) {
  std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
                 std::plus<T>());
  return lhs;
}

template <class T, size_t N, typename = enable_if_numeric_t<T>>
std::array<T, N> add(const std::array<T, N>& lhs, const std::array<T, N>& rhs) {
  std::array<T, N> result = lhs;
  return iadd(result, rhs);
}

template <class T, size_t N, typename = enable_if_numeric_t<T>>
std::array<T, N>& isub(std::array<T, N>& lhs, const std::array<T, N>& rhs) {
  std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
                 std::minus<T>());
  return lhs;
}

template <class T, size_t N, typename = enable_if_numeric_t<T>>
std::array<T, N> sub(const std::array<T, N>& lhs, const std::array<T, N>& rhs) {
  std::array<T, N> result = lhs;
  return isub(result, rhs);
}

//----------------------------------------------------------------------------
// Affine operations
//----------------------------------------------------------------------------
template <class T, size_t N, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::array<T, N>& iadd(std::array<T, N>& data, const Scalar& val) {
  std::transform(data.begin(), data.end(), data.begin(),
                 std::bind(std::plus<T>(), std::placeholders::_1, val));
  return data;
}

template <class T, size_t N, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::array<T, N> add(const std::array<T, N>& data, const Scalar& val) {
  std::array<T, N> result = data;
  return iadd(result, val);
}

template <class T, size_t N, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::array<T, N>& isub(std::array<T, N>& data, const Scalar& val) {
  std::transform(data.begin(), data.end(), data.begin(),
                 std::bind(std::minus<T>(), std::placeholders::_1, val));
  return data;
}

template <class T, size_t N, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::array<T, N> sub(const std::array<T, N>& data, const Scalar& val) {
  std::array<T, N> result = data;
  return isub(result, val);
}

//----------------------------------------------------------------------------
// Scalar operations
//----------------------------------------------------------------------------
template <class T, size_t N, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::array<T, N>& imul(std::array<T, N>& data, const Scalar& val) {
  std::transform(data.begin(), data.end(), data.begin(),
                 std::bind(std::multiplies<T>(), std::placeholders::_1, val));
  return data;
}

template <class T, size_t N, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::array<T, N> mul(const std::array<T, N>& data, const Scalar& val) {
  std::array<T, N> result = data;
  return imul(result, val);
}

template <class T, size_t N, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::array<T, N>& idiv(std::array<T, N>& data, const Scalar& val) {
  std::transform(data.begin(), data.end(), data.begin(),
                 std::bind(std::divides<T>(), std::placeholders::_1, val));
  return data;
}

template <class T, size_t N, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
std::array<T, N> div(const std::array<T, N>& data, const Scalar& val) {
  std::array<T, N> result = data;
  return idiv(result, val);
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif