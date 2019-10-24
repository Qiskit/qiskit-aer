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

#ifndef _aer_framework_results_data_pershot_snapshot_hpp_
#define _aer_framework_results_data_pershot_snapshot_hpp_

#include "framework/json.hpp"
#include "framework/results/data/pershot_data.hpp"
#include "framework/types.hpp"

namespace AER {

//------------------------------------------------------------------------------
// Pershot Snapshot data storage class
//------------------------------------------------------------------------------

template <typename T>
class PershotSnapshot {
  // Inner snapshot data map type

 public:
  // Add a new datum to the snapshot at the specified key
  // Uses copy semantics
  void add_data(const std::string &key, const T &datum);

  // Add a new datum to the snapshot at the specified key
  // Uses move semantics
  void add_data(const std::string &key, T &&datum) noexcept;

  // Combine with another pershot snapshot container
  // Uses copy semantics
  void combine(const PershotSnapshot<T> &other);

  // Combine with another pershot snapshot container
  // Uses move semantics
  void combine(PershotSnapshot<T> &&other) noexcept;

  // Clear all data from current snapshot container
  void clear() { data_.clear(); }

  // Clear all snapshot data for a given label
  void erase(const std::string &label) { data_.erase(label); }

  // Return true if snapshot container is empty
  bool empty() const { return data_.empty(); }

  // Return data reference
  stringmap_t<PershotData<T>> data() { return data_; }

  // Return const data reference
  const stringmap_t<PershotData<T>> &data() const { return data_; }

 private:
  // Internal Storage
  // Map key is the snapshot label string
  stringmap_t<PershotData<T>> data_;
};

//------------------------------------------------------------------------------
// Implementation: PershotSnapshot class methods
//------------------------------------------------------------------------------

template <typename T>
void PershotSnapshot<T>::add_data(const std::string &key, const T &datum) {
  data_[key].add_data(datum);
}

template <typename T>
void PershotSnapshot<T>::add_data(const std::string &key, T &&datum) noexcept {
  data_[key].add_data(std::move(datum));
}

template <typename T>
void PershotSnapshot<T>::combine(const PershotSnapshot<T> &other) {
  for (const auto &pair : other.data_) {
    data_[pair.first].combine(pair.second);
  }
}

template <typename T>
void PershotSnapshot<T>::combine(PershotSnapshot<T> &&other) noexcept {
  for (auto &pair : other.data_) {
    data_[pair.first].combine(std::move(pair.second));
  }
  // Clear added snapshot
  other.clear();
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------
template <typename T>
void to_json(json_t &js, const PershotSnapshot<T> &snapshot) {
  js = json_t::object();
  for (const auto &pair : snapshot.data()) {
    js[pair.first] = pair.second;
  }
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
