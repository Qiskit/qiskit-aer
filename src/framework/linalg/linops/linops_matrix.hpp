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

#ifndef _aer_framework_linalg_linops_matrix_hpp_
#define _aer_framework_linalg_linops_matrix_hpp_

#include <functional>

#include "framework/linalg/almost_equal.hpp"
#include "framework/linalg/enable_if_numeric.hpp"
#include "framework/matrix.hpp"

namespace AER {
namespace Linalg {

// This defines some missing overloads for matrix class
// It should really be added to the matrix class

//----------------------------------------------------------------------------
// Linear operations
//----------------------------------------------------------------------------
template <class T, typename = enable_if_numeric_t<T>>
matrix<T> add(const matrix<T>& lhs, const matrix<T>& rhs) {
  return lhs + rhs;
}

template <class T, typename = enable_if_numeric_t<T>>
matrix<T>& iadd(matrix<T>& lhs, const matrix<T>& rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template <class T, typename = enable_if_numeric_t<T>>
matrix<T> sub(const matrix<T>& lhs, const matrix<T>& rhs) {
  return lhs - rhs;
}

template <class T, typename = enable_if_numeric_t<T>>
matrix<T>& isub(matrix<T>& lhs, const matrix<T>& rhs) {
  lhs = lhs - rhs;
  return lhs;
}

//----------------------------------------------------------------------------
// Affine operations
//----------------------------------------------------------------------------
template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
matrix<T>& iadd(matrix<T>& data, const Scalar& val) {
  if (val == 0) {
    return data;
  }
  for (size_t j = 0; j < data.size(); j++) {
    data[j] = std::plus<T>()(data[j], val);
  }
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
matrix<T> add(const matrix<T>& data, const Scalar& val) {
  matrix<T> result(data);
  return iadd(result, val);
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
matrix<T> sub(const matrix<T>& data, const Scalar& val) {
  return add(data, -val);
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
matrix<T>& isub(matrix<T>& data, const Scalar& val) {
  return iadd(data, -val);
}

//----------------------------------------------------------------------------
// Scalar operations
//----------------------------------------------------------------------------

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
matrix<T>& imul(matrix<T>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  for (size_t j = 0; j < data.size(); j++) {
    data[j] = std::multiplies<T>()(data[j], val);
  }
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
matrix<T> mul(const matrix<T>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  matrix<T> result = data;
  imul(result, val);
  return result;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
matrix<T>& idiv(matrix<T>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  for (size_t j = 0; j < data.size(); j++) {
    data[j] = std::divides<T>()(data[j], val);
  }
  return data;
}

template <class T, class Scalar, typename = enable_if_numeric_t<T>,
          typename = enable_if_numeric_t<Scalar>>
matrix<T> div(const matrix<T>& data, const Scalar& val) {
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  matrix<T> result = data;
  idiv(result, val);
  return result;
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif