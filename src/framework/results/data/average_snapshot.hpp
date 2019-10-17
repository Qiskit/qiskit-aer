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

#ifndef _aer_framework_results_data_average_snapshot_hpp_
#define _aer_framework_results_data_average_snapshot_hpp_

#include "framework/json.hpp"
#include "framework/results/data/average_data.hpp"
#include "framework/types.hpp"

namespace AER {

//------------------------------------------------------------------------------
// Average Snapshot data storage class
//------------------------------------------------------------------------------

template <typename T>
class AverageSnapshot {
  // Inner snapshot data map type

 public:
  // Add a new datum to the snapshot at the specified key
  // Uses copy semantics
  void add_data(const std::string &key, const std::string &memory,
                const T &datum, bool variance = false);
  
  // Add a new datum to the snapshot at the specified key
  // Uses move semantics
  void add_data(const std::string &key, const std::string &memory,
                T &&datum, bool variance = false) noexcept;

  // Combine with another average snapshot container
  // Uses copy semantics
  void combine(const AverageSnapshot<T> &other);

  // Combine with another average snapshot container
  // Uses move semantics
  void combine(AverageSnapshot<T> &&other) noexcept;

  // Clear all data from current snapshot
  void clear() { data_.clear(); }

  // Clear all snapshot data for a given label
  void erase(const std::string &label) { data_.erase(label); }

  // Return true if snapshot container is empty
  bool empty() const { return data_.empty(); }

  // Return data reference
  stringmap_t<stringmap_t<AverageData<T>>> data() { return data_; }

  // Return const data reference
  const stringmap_t<stringmap_t<AverageData<T>>> &data() const { return data_; }

 protected:
  // Internal Storage
  // Outer map key is the snapshot label string
  // Inner map key is the memory value string
  stringmap_t<stringmap_t<AverageData<T>>> data_;
};

//------------------------------------------------------------------------------
// Implementation: AverageSnapshot class methods
//------------------------------------------------------------------------------

template <typename T>
void AverageSnapshot<T>::add_data(const std::string &key,
                                  const std::string &memory, const T &datum,
                                  bool variance) {
  data_[key][memory].add_data(datum, variance);
}

template <typename T>
void AverageSnapshot<T>::add_data(const std::string &key,
                                  const std::string &memory, T &&datum,
                                  bool variance) noexcept {
  data_[key][memory].add_data(std::move(datum), variance);
}

template <typename T>
void AverageSnapshot<T>::combine(const AverageSnapshot<T> &other) {
  for (const auto &outer : other.data_) {
    for (const auto &inner : outer.second) {
      data_[outer.first][inner.first].combine(inner.second);
    }
  }
}

template <typename T>
void AverageSnapshot<T>::combine(AverageSnapshot<T> &&other) noexcept {
  for (auto &outer : other.data_) {
    for (auto &inner : outer.second) {
      data_[outer.first][inner.first].combine(std::move(inner.second));
    }
  }
  // Clear moved snapshot
  other.clear();  
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------
template <typename T>
void to_json(json_t &js, const AverageSnapshot<T> &snapshot) {
  js = json_t::object();
  for (const auto &outer_pair : snapshot.data()) {
    for (const auto &inner_pair : outer_pair.second) {
      // Store mean and variance for snapshot
      json_t datum = inner_pair.second;
      // Add memory key if there are classical registers
      auto memory = inner_pair.first;
      if (memory.empty() == false) datum["memory"] = inner_pair.first;
      // Add to list of output
      js[outer_pair.first].push_back(datum);
    }
  }
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
