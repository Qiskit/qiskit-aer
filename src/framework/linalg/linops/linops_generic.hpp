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

#ifndef _aer_framework_linalg_linops_generic_hpp_
#define _aer_framework_linalg_linops_generic_hpp_

#include <functional>

#include "framework/linalg/almost_equal.hpp"
#include "framework/linalg/enable_if_numeric.hpp"

namespace AER {
namespace Linalg {

// This defines functions add, sub, mul, div and iadd, imul, isub, idiv
// that for generic types that support +,-,*,/ and +=, -=, *=, /= overloads

//----------------------------------------------------------------------------
// Linear operations
//----------------------------------------------------------------------------
template <class T>
T add(const T& lhs, const T& rhs) {
  return std::plus<T>()(lhs, rhs);
}

template <class T>
T& iadd(T& lhs, const T& rhs) {
  lhs = std::plus<T>()(lhs, rhs);
  return lhs;
}

template <class T>
T sub(const T& lhs, const T& rhs) {
  return std::minus<T>()(lhs, rhs);
}

template <class T>
T& isub(T& lhs, const T& rhs) {
  lhs = std::minus<T>()(lhs, rhs);
  return lhs;
}

//----------------------------------------------------------------------------
// Affine operations
//----------------------------------------------------------------------------
template <class T, class Scalar, typename = enable_if_numeric_t<Scalar>>
T add(const T& data, const Scalar& val) {
  return std::plus<T>()(data, val);
}

template <class T, class Scalar, typename = enable_if_numeric_t<Scalar>>
T& iadd(T& data, const Scalar& val) {
  data = std::plus<T>()(data, val);
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<Scalar>>
T sub(const T& data, const Scalar& val) {
  return std::minus<T>()(data, val);
}

template <class T, class Scalar, typename = enable_if_numeric_t<Scalar>>
T& isub(T& data, const Scalar& val) {
  data = std::minus<T>()(data, val);
  return data;
}

//----------------------------------------------------------------------------
// Scalar operations
//----------------------------------------------------------------------------
template <class T, class Scalar, typename = enable_if_numeric_t<Scalar>>
T mul(const T& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  return std::multiplies<T>()(data, val);
}

template <class T, class Scalar, typename = enable_if_numeric_t<Scalar>>
T& imul(T& data, const Scalar& val) {
  if (!almost_equal<Scalar>(val, 1)) {
    data = std::multiplies<T>()(data, val);
  }
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<Scalar>>
T div(const T& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  return std::divides<T>()(data, val);
}

template <class T, class Scalar, typename = enable_if_numeric_t<Scalar>>
T& idiv(T& data, const Scalar& val) {
  if (!almost_equal<Scalar>(val, 1)) {
    data = std::divides<T>()(data, val);
  }
  return data;
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif