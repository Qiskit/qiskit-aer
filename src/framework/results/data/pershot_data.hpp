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

#ifndef _aer_framework_results_data_pershot_data_hpp_
#define _aer_framework_results_data_pershot_data_hpp_

#include "framework/json.hpp"
#include "framework/types.hpp"

namespace AER {

template <typename T>
class PershotData {
 public:
  // Constructor
  PershotData(size_t num_shots = 0) { data_.reserve(num_shots); }

  // Add a new shot of data by appending to data vector
  // Uses copy semantics
  void add_data(const T& datum) { data_.push_back(datum); }

  // Add a new shot of data by appending to data vector
  // Uses move semantics
  void add_data(T&& datum) noexcept { data_.push_back(std::move(datum)); }

  // Combine with another pershot data by concatinating
  // Uses copy semantics
  void combine(const PershotData<T>& other);

  // Combine with another pershot data by concatinating
  // Uses move semantics
  void combine(PershotData<T>&& other) noexcept;

  // Access data
  std::vector<T>& data() { return data_; }

  // Const access data
  const std::vector<T>& data() const { return data_; }

  // Get number of datum accumulated
  size_t size() const { return data_.size(); }

  void reserve(size_t sz) { data_.reserve(sz); }

  // Clear all stored data
  void clear() { data_.clear(); }

  // Return true if data is empty
  bool empty() const { return data_.empty(); }

 protected:
  // Internal Storage
  std::vector<T> data_;
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

template <typename T>
void PershotData<T>::combine(const PershotData<T>& other) {
  // Copy data from other container
  data_.reserve(data_.size() + other.data_.size());
  data_.insert(data_.end(), other.data_.begin(), other.data_.end());
}

template <typename T>
void PershotData<T>::combine(PershotData<T>&& other) noexcept {
  // Move data from other container
  data_.insert(data_.end(), std::make_move_iterator(other.data_.begin()),
               std::make_move_iterator(other.data_.end()));
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------
template <typename T>
void to_json(json_t& js, const PershotData<T>& data) {
  js = data.data();
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
