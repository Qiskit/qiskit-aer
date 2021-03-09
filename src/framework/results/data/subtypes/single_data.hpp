/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2021.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_results_data_subtypes_single_hpp_
#define _aer_framework_results_data_subtypes_single_hpp_

#include "framework/json.hpp"

namespace AER {

template <typename T>
class SingleData {
public:
  // Access data
  T& value() { return data_; }

  // Add data (copy)
  void add(const T& data);

  // Add data (move)
  void add(T&& data);

  // Combine with another data object
  void combine(SingleData<T>&& other);

  // Clear all stored data
  void clear();

  // Convert to JSON
  json_t to_json();

protected:
  T data_;
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

template <typename T>
void SingleData<T>::add(const T& data) {
  data_ = data;
}

template <typename T>
void SingleData<T>::add(T&& data) {
  data_ = std::move(data);
}

template <typename T>
void SingleData<T>::combine(SingleData<T>&& other) {
  data_ = std::move(other.data_);
}

template <typename T>
void SingleData<T>::clear() {
  data_ = T();
}

template <typename T>
json_t SingleData<T>::to_json() {
  json_t jsdata = value();
  return jsdata;
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
