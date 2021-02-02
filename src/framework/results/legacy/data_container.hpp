/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_results_data_container_hpp_
#define _aer_framework_results_data_container_hpp_

#include "framework/json.hpp"

#include "framework/results/legacy/average_snapshot.hpp"
#include "framework/results/legacy/pershot_snapshot.hpp"

namespace AER {

//============================================================================
// DEPRECATED DataContainer Data class for Qiskit-Aer
//============================================================================


template <typename T>
class DataContainer {
public:

  //----------------------------------------------------------------
  // Snapshot data
  //----------------------------------------------------------------
  
  // Pershot snapshot
  void add_pershot_snapshot(const std::string &type,
                            const std::string &label,
                            T &&datum);
  void add_pershot_snapshot(const std::string &type,
                            const std::string &label,
                            T &datum);
  void add_pershot_snapshot(const std::string &type,
                            const std::string &label,
                            const T &datum);

  // Average snapshot
  void add_average_snapshot(const std::string &type,
                            const std::string &label,
                            const std::string &memory,
                            T &&datum, bool variance);
  void add_average_snapshot(const std::string &type,
                            const std::string &label,
                            const std::string &memory,
                            T &datum, bool variance);
  void add_average_snapshot(const std::string &type,
                            const std::string &label,
                            const std::string &memory,
                            const T &datum, bool variance);

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  // Enable data container
  void enable(bool value) {enabled_ = value;}

  // Empty engine of stored data
  void clear();

  // Convert and add to json
  void add_to_json(json_t &js);

  // Combine engines for accumulating data
  // Second engine should no longer be used after combining
  // as this function should use move semantics to minimize copying
  DataContainer<T> &combine(DataContainer<T> &&eng); // Move semantics
  DataContainer<T> &combine(const DataContainer<T> &eng); // Copy semantics

  // Operator overload for combine
  // Note this operator is not defined to be const on the input argument
  inline DataContainer<T> &operator+=(const DataContainer<T> &data) {
    return combine(data);
  }
  inline DataContainer<T> &operator+=(DataContainer<T> &&data) {
    return combine(std::move(data));
  }

  //----------------------------------------------------------------
  // Data containers
  //----------------------------------------------------------------

  // Pershot snapshots
  stringmap_t<PershotSnapshot<T>> pershot_snapshots_;

  // Average snapshots
  stringmap_t<AverageSnapshot<T>> average_snapshots_;

  // Store whether this data type is enabled or not
  bool enabled_ = true;
};

//============================================================================
// Implementations
//============================================================================

//------------------------------------------------------------------
// Add pershot snapshot data
//------------------------------------------------------------------

template <typename T>
void DataContainer<T>::add_pershot_snapshot(const std::string &type,
                                            const std::string &label,
                                            T &&datum) {
  if (enabled_) {
    pershot_snapshots_[type].add_data(label, std::move(datum));
  }
}

template <typename T>
void DataContainer<T>::add_pershot_snapshot(const std::string &type,
                                            const std::string &label,
                                            const T &datum) {
  if (enabled_) {
    pershot_snapshots_[type].add_data(label, datum);
  }
}

template <typename T>
void DataContainer<T>::add_pershot_snapshot(const std::string &type,
                                            const std::string &label,
                                            T &datum) {
  if (enabled_) {
    pershot_snapshots_[type].add_data(label, datum);
  }
}

//------------------------------------------------------------------
// Add average snapshot data
//------------------------------------------------------------------

template <typename T>
void DataContainer<T>::add_average_snapshot(const std::string &type,
                                            const std::string &label,
                                            const std::string &memory,
                                            T &&datum, bool variance) {
  if (enabled_) {
    average_snapshots_[type].add_data(label, memory, std::move(datum), variance);
  }
}

template <typename T>
void DataContainer<T>::add_average_snapshot(const std::string &type,
                                         const std::string &label,
                                         const std::string &memory,
                                         const T &datum, bool variance) {
  if (enabled_) {
    average_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

template <typename T>
void DataContainer<T>::add_average_snapshot(const std::string &type,
                                            const std::string &label,
                                            const std::string &memory,
                                            T &datum, bool variance) {
  if (enabled_) {
    average_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

//------------------------------------------------------------------
// Clear
//------------------------------------------------------------------

template <typename T>
void DataContainer<T>::clear() {
  average_snapshots_.clear();
  pershot_snapshots_.clear();
}

//------------------------------------------------------------------
// Combine
//------------------------------------------------------------------

template <typename T>
DataContainer<T> &DataContainer<T>::combine(const DataContainer<T> &other) {

  // Pershot snapshots
  for (const auto &pair : other.pershot_snapshots_) {
    pershot_snapshots_[pair.first].combine(pair.second);
  }

  // Average snapshots
  for (const auto &pair : other.average_snapshots_) {
    average_snapshots_[pair.first].combine(pair.second);
  }

  return *this;
}

template <typename T>
DataContainer<T> &DataContainer<T>::combine(DataContainer<T> &&other) {

  // Pershot snapshots
  for (auto &pair : other.pershot_snapshots_) {
    pershot_snapshots_[pair.first].combine(std::move(pair.second));
  }

  // Average snapshots
  for (auto &pair : other.average_snapshots_) {
    average_snapshots_[pair.first].combine(std::move(pair.second));
  }

  // Clear any remaining data from other container
  other.clear();

  return *this;
}

//============================================================================
// JSON Serlialization
//============================================================================

template <typename T>
void DataContainer<T>::add_to_json(json_t &js) {
  // Add data to json
  // Note other data types may also be adding to same
  // JSON so we should not re-initialize it
  if (enabled_) {

    // Average snapshots
    for (auto &pair : average_snapshots_) {
      js[pair.first] = pair.second.to_json();
    }

    // Pershot snapshots
    for (auto &pair : pershot_snapshots_) {
      js[pair.first] = pair.second.to_json();
    }
  }
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
