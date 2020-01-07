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

#ifndef _aer_framework_linalg_linops_json_hpp_
#define _aer_framework_linalg_linops_json_hpp_

#include "framework/json.hpp"
#include "framework/linalg/almost_equal.hpp"
#include "framework/linalg/enable_if_numeric.hpp"

namespace AER {
namespace Linalg {

// This defines functions add, sub, mul, div and iadd, imul, isub, idiv
// that for numeric json that support +,-,*,/ and +=, -=, *=, /= overloads

//----------------------------------------------------------------------------
// Linear operations
//----------------------------------------------------------------------------
json_t& iadd(json_t& lhs, const json_t& rhs) {
  // Null case
  if (lhs.is_null()) {
    lhs = rhs;
    return lhs;
  }
  if (rhs.is_null()) {
    return lhs;
  }
  // Terminating case
  if (lhs.is_number() && rhs.is_number()) {
    lhs = double(lhs) + double(rhs);
    return lhs;
  }
  // Recursive cases
  if (lhs.is_array() && rhs.is_array() && lhs.size() == rhs.size()) {
    for (size_t pos = 0; pos < lhs.size(); pos++) {
      iadd(lhs[pos], rhs[pos]);
    }
  } else if (lhs.is_object() && rhs.is_object()) {
    for (auto it = rhs.begin(); it != rhs.end(); ++it) {
      iadd(lhs[it.key()], it.value());
    }
  } else {
    throw std::invalid_argument("Input JSONs cannot be added.");
  }
  return lhs;
}

json_t add(const json_t& lhs, const json_t& rhs) {
  json_t result = lhs;
  return iadd(result, rhs);
}

json_t& isub(json_t& lhs, const json_t& rhs) {
  // Null case
  if (rhs.is_null()) {
    return lhs;
  }
  // Terminating case
  if (lhs.is_number() && rhs.is_number()) {
    lhs = double(lhs) - double(rhs);
    return lhs;
  }
  // Recursive cases
  if (lhs.is_array() && rhs.is_array() && lhs.size() == rhs.size()) {
    for (size_t pos = 0; pos < lhs.size(); pos++) {
      isub(lhs[pos], rhs[pos]);
    }
  } else if (lhs.is_object() && rhs.is_object()) {
    for (auto it = rhs.begin(); it != rhs.end(); ++it) {
      isub(lhs[it.key()], it.value());
    }
  } else {
    throw std::invalid_argument("Input JSONs cannot be subtracted.");
  }
  return lhs;
}

template <class T>
json_t sub(const T& lhs, const json_t& rhs) {
  json_t result = lhs;
  return isub(result, rhs);
}

//----------------------------------------------------------------------------
// Affine operations
//----------------------------------------------------------------------------
template <class Scalar, typename = enable_if_numeric_t<Scalar>>
json_t& iadd(json_t& data, const Scalar& val) {
  // Null case
  if (val == 0) {
    return data;
  }
  // Terminating case
  if (data.is_number()) {
    data = double(data) + val;
    return data;
  }
  // Recursive cases
  if (data.is_array()) {
    for (size_t pos = 0; pos < data.size(); pos++) {
      iadd(data[pos], val);
    }
  } else if (data.is_object()) {
    for (auto it = data.begin(); it != data.end(); ++it) {
      iadd(data[it.key()], val);
    }
  } else {
    throw std::invalid_argument("Input JSON does not support affine addition.");
  }
  return data;
}

template <class Scalar, typename = enable_if_numeric_t<Scalar>>
json_t add(const json_t& data, const Scalar& val) {
  json_t result = data;
  return iadd(result, val);
}

template <class Scalar, typename = enable_if_numeric_t<Scalar>>
json_t& isub(json_t& data, const Scalar& val) {
  // Null case
  if (val == 0) {
    return data;
  }
  // Terminating case
  if (data.is_number()) {
    data = double(data) - val;
    return data;
  }
  // Recursive cases
  if (data.is_array()) {
    for (size_t pos = 0; pos < data.size(); pos++) {
      isub(data[pos], val);
    }
  } else if (data.is_object()) {
    for (auto it = data.begin(); it != data.end(); ++it) {
      isub(data[it.key()], val);
    }
  } else {
    throw std::invalid_argument(
        "Input JSON does not support affine subtraction.");
  }
  return data;
}

template <class Scalar, typename = enable_if_numeric_t<Scalar>>
json_t sub(const json_t& data, const Scalar& val) {
  json_t result = data;
  return isub(result, val);
}

//----------------------------------------------------------------------------
// Scalar operations
//----------------------------------------------------------------------------

template <class Scalar, typename = enable_if_numeric_t<Scalar>>
json_t& imul(json_t& data, const Scalar& val) {
  // Trival case
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  // Terminating case
  if (data.is_number()) {
    data = double(data) * val;
    return data;
  }
  // Recursive cases
  if (data.is_array()) {
    for (size_t pos = 0; pos < data.size(); pos++) {
      imul(data[pos], val);
    }
    return data;
  }
  if (data.is_object()) {
    for (auto it = data.begin(); it != data.end(); ++it) {
      imul(data[it.key()], val);
    }
    return data;
  }
  throw std::invalid_argument(
      "Input JSON does not support scalar multiplication.");
}

template <class Scalar, typename = enable_if_numeric_t<Scalar>>
json_t mul(const json_t& data, const Scalar& val) {
  // Null case
  json_t result = data;
  return imul(result, val);
}

template <class Scalar, typename = enable_if_numeric_t<Scalar>>
json_t& idiv(json_t& data, const Scalar& val) {
  // Trival case
  if (almost_equal<Scalar>(val, 1)) {
    return data;
  }
  // Terminating case
  if (data.is_number()) {
    data = double(data) / val;
    return data;
  }
  // Recursive cases
  if (data.is_array()) {
    for (size_t pos = 0; pos < data.size(); pos++) {
      idiv(data[pos], val);
    }
    return data;
  }
  if (data.is_object()) {
    for (auto it = data.begin(); it != data.end(); ++it) {
      idiv(data[it.key()], val);
    }
    return data;
  }
  throw std::invalid_argument("Input JSON does not support scalar division.");
}

template <class Scalar, typename = enable_if_numeric_t<Scalar>>
json_t div(const json_t& data, const Scalar& val) {
  // Null case
  json_t result = data;
  return idiv(result, val);
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif