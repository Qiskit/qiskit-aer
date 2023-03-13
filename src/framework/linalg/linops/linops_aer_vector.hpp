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

#ifndef _aer_framework_linalg_linops_aer_vector_hpp_
#define _aer_framework_linalg_linops_aer_vector_hpp_

#include <functional>
#include <vector>

#include "framework/linalg/almost_equal.hpp"
#include "framework/linalg/vector.hpp"

namespace AER {
namespace Linalg {

// This defines functions add, sub, mul, div and iadd, imul, isub, idiv
// that work for numeric vector types

//----------------------------------------------------------------------------
// Linear operations
//----------------------------------------------------------------------------
template <class T, typename = enable_if_numeric_t<T>>
Vector<T> add(const Vector<T> &lhs, const Vector<T> &rhs) {
  return lhs + rhs;
}

template <class T, typename = enable_if_numeric_t<T>>
Vector<T> &iadd(Vector<T> &lhs, const Vector<T> &rhs) {
  lhs += rhs;
  return lhs;
}

template <class T, typename = enable_if_numeric_t<T>>
Vector<T> sub(const Vector<T> &lhs, const Vector<T> &rhs) {
  return lhs - rhs;
}

template <class T, typename = enable_if_numeric_t<T>>
Vector<T> &isub(Vector<T> &lhs, const Vector<T> &rhs) {
  lhs -= rhs;
  return lhs;
}

//----------------------------------------------------------------------------
// Affine operations
//----------------------------------------------------------------------------
template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
Vector<T> &iadd(Vector<T> &data, const Scalar &val) {
  const T cast_val(val);
  std::for_each(data.data(), data.data() + data.size(),
                [&cast_val](T &a) -> T { a += cast_val; });
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
Vector<T> add(const Vector<T> &data, const Scalar &val) {
  auto ret = data;
  return iadd(data, val);
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
Vector<T> sub(const Vector<T> &data, const Scalar &val) {
  return add(data, -val);
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
Vector<T> &isub(Vector<T> &data, const Scalar &val) {
  return iadd(data, -val);
}

//----------------------------------------------------------------------------
// Scalar operations
//----------------------------------------------------------------------------
template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
Vector<T> &imul(Vector<T> &data, const Scalar &val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  data *= T(val);
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
Vector<T> mul(const Vector<T> &data, const Scalar &val) {
  auto ret = data;
  return imul(ret, val);
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
Vector<T> &idiv(Vector<T> &data, const Scalar &val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  data /= T(val);
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
Vector<T> div(const Vector<T> &data, const Scalar &val) {
  auto ret = data;
  return idiv(ret, val);
}

//------------------------------------------------------------------------------
} // end namespace Linalg
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif