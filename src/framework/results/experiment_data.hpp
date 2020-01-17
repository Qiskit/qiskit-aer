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

#ifndef _aer_framework_experiment_data_hpp_
#define _aer_framework_experiment_data_hpp_

#include "framework/json.hpp"
#include "framework/results/data/average_snapshot.hpp"
#include "framework/results/data/pershot_snapshot.hpp"
#include "framework/utils.hpp"

namespace AER {

//============================================================================
// Output data class for Qiskit-Aer
//============================================================================

/**************************************************************************
 * Data config options:
 *
 * - "counts" (bool): Return counts object in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 * - "register" (bool): Return register array in circuit data [Default: False]
 **************************************************************************/

class ExperimentData {
 public:
  //----------------------------------------------------------------
  // Measurement
  //----------------------------------------------------------------

  // Add a single memory value to the counts map
  void add_memory_count(const std::string &memory);

  // Add a single memory value to the memory vector
  void add_pershot_memory(const std::string &memory);

  // Add a single register value to the register vector
  void add_pershot_register(const std::string &reg);

  //----------------------------------------------------------------
  // Pershot snapshots
  //----------------------------------------------------------------

  // Add a new datum to the snapshot of the specified type and label
  // This will use the json conversion method `to_json` for
  // data type T unless T is one of the supported basic types:
  // complex, complex vector, complex matrix, map<string, complex>,
  // map<string, double>
  template <typename T>
  void add_pershot_snapshot(const std::string &type, const std::string &label,
                            T &&datum);

  //----------------------------------------------------------------
  // Average snapshots
  //----------------------------------------------------------------

  // Add a new datum to the snapshot of the specified type and label
  // This will use the json conversion method `to_json` for
  // data type T
  // If variance is true the variance of the averaged sample will also
  // be computed
  template <typename T>
  void add_average_snapshot(const std::string &type, const std::string &label,
                            const std::string &memory, T &&datum,
                            bool variance = false);

  //----------------------------------------------------------------
  // Additional data
  //----------------------------------------------------------------

  // Add new data at the specified key.
  // If they key already exists this will override the stored data
  // This will use the json conversion method `to_json` for data type T
  // if it isn't a natively supported data type.
  // Current native types are: cvector_t, cmatrix_t.
  template <typename T>
  void add_additional_data(const std::string &key, T &&data);

  // Erase additional data stored at the provided key.
  // This will erase from all additional data containers
  void erase_additional_data(const std::string &key);

  //----------------------------------------------------------------
  // Metadata
  //----------------------------------------------------------------

  // Access metadata map
  stringmap_t<json_t> &metadata() { return metadata_; }
  const stringmap_t<json_t> &metadata() const { return metadata_; }

  // Add new data to metadata at the specified key.
  // This will use the json conversion method `to_json` for data type T.
  // If they key already exists this will update the current data
  // with the new data.
  template <typename T>
  void add_metadata(const std::string &key, T &&data);

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  // Set the output data config options
  void set_config(const json_t &config);

  // Empty engine of stored data
  void clear();

  // Serialize engine data to JSON
  json_t json() const;

  // Combine engines for accumulating data
  // Second engine should no longer be used after combining
  // as this function should use move semantics to minimize copying
  ExperimentData &combine(ExperimentData &&eng); // Move semantics
  ExperimentData &combine(const ExperimentData &eng); // Copy semantics

  // Operator overload for combine
  // Note this operator is not defined to be const on the input argument
  inline ExperimentData &operator+=(const ExperimentData &eng) {
    return combine(eng);
  }
  inline ExperimentData &operator+=(ExperimentData &&eng) {
    return combine(std::move(eng));
  }

  //----------------------------------------------------------------
  // Measurement data
  //----------------------------------------------------------------

  // Histogram of memory counts over shots
  std::map<std::string, uint_t> counts_;

  // Memory state for each shot as hex string
  std::vector<std::string> memory_;

  // Register state for each shot as hex string
  std::vector<std::string> register_;

  //----------------------------------------------------------------
  // Snapshots
  //----------------------------------------------------------------

  // The outer keys of the snapshot maps are the "snapshot_type".
  // The JSON type snapshot is used as a generic dynamic-type
  // storage since it can store any class that has a `to_json`
  // method implemented.

  //----------------------------------------------------------------
  // Pershot Snapshots
  //----------------------------------------------------------------

  // Generic JSON pershot snapshots
  stringmap_t<PershotSnapshot<json_t>> pershot_json_snapshots_;

  // Complex value pershot snapshots
  stringmap_t<PershotSnapshot<complex_t>> pershot_complex_snapshots_;

  // Complex vector pershot snapshots
  stringmap_t<PershotSnapshot<cvector_t>> pershot_cvector_snapshots_;

  // Complex matrix pershot snapshots
  stringmap_t<PershotSnapshot<cmatrix_t>> pershot_cmatrix_snapshots_;

  // Map<string, complex> pershot snapshots
  stringmap_t<PershotSnapshot<std::map<std::string, complex_t>>>
      pershot_cmap_snapshots_;

  // Map<string, double> pershot snapshots
  stringmap_t<PershotSnapshot<std::map<std::string, double>>>
      pershot_rmap_snapshots_;

  //----------------------------------------------------------------
  // Average Snapshots
  //----------------------------------------------------------------

  // Generic JSON average snapshots
  stringmap_t<AverageSnapshot<json_t>> average_json_snapshots_;

  // Complex value average snapshots
  stringmap_t<AverageSnapshot<complex_t>> average_complex_snapshots_;

  // Complex vector average snapshots
  stringmap_t<AverageSnapshot<cvector_t>> average_cvector_snapshots_;

  // Complex matrix average snapshots
  stringmap_t<AverageSnapshot<cmatrix_t>> average_cmatrix_snapshots_;

  // Map<string, complex> average snapshots
  stringmap_t<AverageSnapshot<std::map<std::string, complex_t>>>
      average_cmap_snapshots_;

  // Map<string, double> average snapshots
  stringmap_t<AverageSnapshot<std::map<std::string, double>>>
      average_rmap_snapshots_;

  //----------------------------------------------------------------
  // Additional data
  //----------------------------------------------------------------

  // Reserved keys that can't be used by additional data
  const static stringset_t reserved_keys_;

  // Miscelaneous data
  stringmap_t<json_t> additional_json_data_;

  // Complex vector data
  stringmap_t<cvector_t> additional_cvector_data_;

  // Complex matrix data
  stringmap_t<cmatrix_t> additional_cmatrix_data_;

  // Check if key name is reserved and if so throw an exception
  void check_reserved_key(const std::string &key);

  //----------------------------------------------------------------
  // Metadata
  //----------------------------------------------------------------

  // This will be passed up to the experiment_result level
  // metadata field
  stringmap_t<json_t> metadata_;

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  bool return_counts_ = true;
  bool return_memory_ = false;
  bool return_register_ = false;
  bool return_snapshots_ = true;
  bool return_additional_data_ = true;
};

//============================================================================
// Implementations
//============================================================================

void ExperimentData::set_config(const json_t &config) {
  JSON::get_value(return_counts_, "counts", config);
  JSON::get_value(return_memory_, "memory", config);
  JSON::get_value(return_register_, "register", config);
  JSON::get_value(return_snapshots_, "snapshots", config);
}

//------------------------------------------------------------------
// Classical data
//------------------------------------------------------------------

void ExperimentData::add_memory_count(const std::string &memory) {
  // Memory bits value
  if (return_counts_ && !memory.empty()) {
    counts_[memory] += 1;
  }
}

void ExperimentData::add_pershot_memory(const std::string &memory) {
  // Memory bits value
  if (return_memory_ && !memory.empty()) {
    memory_.push_back(memory);
  }
}

void ExperimentData::add_pershot_register(const std::string &reg) {
  if (return_register_ && !reg.empty()) {
    register_.push_back(reg);
  }
}

//------------------------------------------------------------------
// Pershot Snapshots
//------------------------------------------------------------------

// Generic
template <typename T>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label, T &&datum) {
  if (return_snapshots_) {
    // use implicit to_json conversion function for T
    json_t tmp = datum;
    add_pershot_snapshot(type, label, std::move(tmp));
  }
}

// JSON
template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          json_t &&datum) {
  if (return_snapshots_) {
    pershot_json_snapshots_[type].add_data(label, std::move(datum));
  }
}

template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          const json_t &datum) {
  if (return_snapshots_) {
    pershot_json_snapshots_[type].add_data(label, datum);
  }
}

template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          json_t &datum) {
  if (return_snapshots_) {
    pershot_json_snapshots_[type].add_data(label, datum);
  }
}

// Complex
template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          complex_t &&datum) {
  if (return_snapshots_) {
    pershot_complex_snapshots_[type].add_data(label, std::move(datum));
  }
}

template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          const complex_t &datum) {
  if (return_snapshots_) {
    pershot_complex_snapshots_[type].add_data(label, datum);
  }
}

template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          complex_t &datum) {
  if (return_snapshots_) {
    pershot_complex_snapshots_[type].add_data(label, datum);
  }
}

// Complex vector
template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          cvector_t &&datum) {
  if (return_snapshots_) {
    pershot_cvector_snapshots_[type].add_data(label, std::move(datum));
  }
}

template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          const cvector_t &datum) {
  if (return_snapshots_) {
    pershot_cvector_snapshots_[type].add_data(label, datum);
  }
}

template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          cvector_t &datum) {
  if (return_snapshots_) {
    pershot_cvector_snapshots_[type].add_data(label, datum);
  }
}

// Complex matrix
template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          cmatrix_t &&datum) {
  if (return_snapshots_) {
    pershot_cmatrix_snapshots_[type].add_data(label, std::move(datum));
  }
}

template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          const cmatrix_t &datum) {
  if (return_snapshots_) {
    pershot_cmatrix_snapshots_[type].add_data(label, datum);
  }
}

template <>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          cmatrix_t &datum) {
  if (return_snapshots_) {
    pershot_cmatrix_snapshots_[type].add_data(label, datum);
  }
}

// Complex map
template <>
void ExperimentData::add_pershot_snapshot(
    const std::string &type, const std::string &label,
    std::map<std::string, complex_t> &&datum) {
  if (return_snapshots_) {
    pershot_cmap_snapshots_[type].add_data(label, std::move(datum));
  }
}

template <>
void ExperimentData::add_pershot_snapshot(
    const std::string &type, const std::string &label,
    const std::map<std::string, complex_t> &datum) {
  if (return_snapshots_) {
    pershot_cmap_snapshots_[type].add_data(label, datum);
  }
}

template <>
void ExperimentData::add_pershot_snapshot(
    const std::string &type, const std::string &label,
    std::map<std::string, complex_t> &datum) {
  if (return_snapshots_) {
    pershot_cmap_snapshots_[type].add_data(label, datum);
  }
}

// Real map
template <>
void ExperimentData::add_pershot_snapshot(
    const std::string &type, const std::string &label,
    std::map<std::string, double> &&datum) {
  if (return_snapshots_) {
    pershot_rmap_snapshots_[type].add_data(label, std::move(datum));
  }
}

template <>
void ExperimentData::add_pershot_snapshot(
    const std::string &type, const std::string &label,
    const std::map<std::string, double> &datum) {
  if (return_snapshots_) {
    pershot_rmap_snapshots_[type].add_data(label, datum);
  }
}

template <>
void ExperimentData::add_pershot_snapshot(
    const std::string &type, const std::string &label,
    std::map<std::string, double> &datum) {
  if (return_snapshots_) {
    pershot_rmap_snapshots_[type].add_data(label, datum);
  }
}

//------------------------------------------------------------------
// Average Snapshots
//------------------------------------------------------------------

template <typename T>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory, T &&datum,
                                          bool variance) {
  if (return_snapshots_) {
    json_t tmp = datum;  // use implicit to_json conversion function for T
    add_average_snapshot(type, label, memory, std::move(tmp), variance);
  }
}

// JSON
template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          json_t &&datum, bool variance) {
  if (return_snapshots_) {
    average_json_snapshots_[type].add_data(label, memory, std::move(datum), variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          const json_t &datum, bool variance) {
  if (return_snapshots_) {
    average_json_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          json_t &datum, bool variance) {
  if (return_snapshots_) {
    average_json_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

// Complex
template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          complex_t &&datum, bool variance) {
  if (return_snapshots_) {
    average_complex_snapshots_[type].add_data(label, memory, std::move(datum), variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          const complex_t &datum, bool variance) {
  if (return_snapshots_) {
    average_complex_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          complex_t &datum, bool variance) {
  if (return_snapshots_) {
    average_complex_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

// Complex vector
template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          cvector_t &&datum, bool variance) {
  if (return_snapshots_) {
    average_cvector_snapshots_[type].add_data(label, memory, std::move(datum), variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          const cvector_t &datum, bool variance) {
  if (return_snapshots_) {
    average_cvector_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          cvector_t &datum, bool variance) {
  if (return_snapshots_) {
    average_cvector_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

// Complex matrix
template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          cmatrix_t &&datum, bool variance) {
  if (return_snapshots_) {
    average_cmatrix_snapshots_[type].add_data(label, memory, std::move(datum), variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          const cmatrix_t &datum, bool variance) {
  if (return_snapshots_) {
    average_cmatrix_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          cmatrix_t &datum, bool variance) {
  if (return_snapshots_) {
    average_cmatrix_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

// Complex map
template <>
void ExperimentData::add_average_snapshot(
    const std::string &type, const std::string &label,
    const std::string &memory, std::map<std::string, complex_t> &&datum,
    bool variance) {
  if (return_snapshots_) {
    average_cmap_snapshots_[type].add_data(label, memory, std::move(datum), variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(
    const std::string &type, const std::string &label,
    const std::string &memory, const std::map<std::string, complex_t> &datum,
    bool variance) {
  if (return_snapshots_) {
    average_cmap_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(
    const std::string &type, const std::string &label,
    const std::string &memory, std::map<std::string, complex_t> &datum,
    bool variance) {
  if (return_snapshots_) {
    average_cmap_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

// Real map
template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          std::map<std::string, double> &&datum,
                                          bool variance) {
  if (return_snapshots_) {
    average_rmap_snapshots_[type].add_data(label, memory, std::move(datum), variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          const std::map<std::string, double> &datum,
                                          bool variance) {
  if (return_snapshots_) {
    average_rmap_snapshots_[type].add_data(label, memory, datum, variance);
  }
}

template <>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          std::map<std::string, double> &datum,
                                          bool variance) {
  if (return_snapshots_) {
    average_rmap_snapshots_[type].add_data(label, memory, datum, variance);
  }
}
//------------------------------------------------------------------
// Additional Data
//------------------------------------------------------------------

const stringset_t ExperimentData::reserved_keys_ = {"counts", "memory",
                                                    "register", "snapshots"};

void ExperimentData::check_reserved_key(const std::string &key) {
  if (reserved_keys_.find(key) != reserved_keys_.end()) {
    throw std::invalid_argument(
        "Cannot add additional data with reserved key name \"" + key + "\".");
  }
}

void ExperimentData::erase_additional_data(const std::string &key) {
  additional_json_data_.erase(key);
  additional_cvector_data_.erase(key);
  additional_cmatrix_data_.erase(key);
}

template <typename T>
void ExperimentData::add_additional_data(const std::string &key, T &&data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    json_t jdata = data;
    add_additional_data(key, std::move(jdata));
  }
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         json_t &&data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    erase_additional_data(key);
    additional_json_data_[key] = std::move(data);
  }
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         const json_t &data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    erase_additional_data(key);
    additional_json_data_[key] = data;
  }
}

template <>
void ExperimentData::add_additional_data(const std::string &key, json_t &data) {
  const json_t &const_data = data;
  add_additional_data(key, const_data);
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         cvector_t &&data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    erase_additional_data(key);
    additional_cvector_data_[key] = std::move(data);
  }
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         const cvector_t &data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    erase_additional_data(key);
    additional_cvector_data_[key] = data;
  }
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         cvector_t &data) {
  const cvector_t &const_data = data;
  add_additional_data(key, const_data);
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         cmatrix_t &&data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    erase_additional_data(key);
    additional_cmatrix_data_[key] = std::move(data);
  }
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         const cmatrix_t &data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    erase_additional_data(key);
    additional_cmatrix_data_[key] = data;
  }
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         cmatrix_t &data) {
  const cmatrix_t &const_data = data;
  add_additional_data(key, const_data);
}

//------------------------------------------------------------------
// Metadata
//------------------------------------------------------------------

template <typename T>
void ExperimentData::add_metadata(const std::string &key, T &&data) {
  // Use implicit to_json conversion function for T
  json_t jdata = data;
  add_metadata(key, std::move(jdata));
}

template <>
void ExperimentData::add_metadata(const std::string &key, json_t &&data) {
  auto elt = metadata_.find("key");
  if (elt == metadata_.end()) {
    // If key doesn't already exist add new data
    metadata_[key] = std::move(data);
  } else {
    // If key already exists append with additional data
    elt->second.update(data.begin(), data.end());
  }
}

template <>
void ExperimentData::add_metadata(const std::string &key, const json_t &data) {
  auto elt = metadata_.find("key");
  if (elt == metadata_.end()) {
    // If key doesn't already exist add new data
    metadata_[key] = data;
  } else {
    // If key already exists append with additional data
    elt->second.update(data.begin(), data.end());
  }
}

template <>
void ExperimentData::add_metadata(const std::string &key, json_t &data) {
  const json_t &const_data = data;
  add_metadata(key, const_data);
}

//------------------------------------------------------------------
// Clear and combine
//------------------------------------------------------------------

void ExperimentData::clear() {
  // Clear measurement data
  counts_.clear();
  memory_.clear();
  register_.clear();

  // Clear pershot snapshots
  pershot_json_snapshots_.clear();
  pershot_complex_snapshots_.clear();
  pershot_cvector_snapshots_.clear();
  pershot_cmatrix_snapshots_.clear();
  pershot_cmap_snapshots_.clear();
  pershot_rmap_snapshots_.clear();
  // Clear average snapshots
  average_json_snapshots_.clear();
  average_complex_snapshots_.clear();
  average_cvector_snapshots_.clear();
  average_cmatrix_snapshots_.clear();
  average_cmap_snapshots_.clear();
  average_rmap_snapshots_.clear();
  // Clear additional data
  additional_json_data_.clear();
  additional_cvector_data_.clear();
  additional_cmatrix_data_.clear();

  // Clear metadata
  metadata_.clear();
}

ExperimentData &ExperimentData::combine(const ExperimentData &other) {
  // Combine measure
  std::copy(other.memory_.begin(), other.memory_.end(),
            std::back_inserter(memory_));
  std::copy(other.register_.begin(), other.register_.end(),
            std::back_inserter(register_));

  // Combine counts
  for (auto pair : other.counts_) {
    counts_[pair.first] += pair.second;
  }

  // Combine pershot snapshots
  for (const auto &pair : other.pershot_json_snapshots_) {
    pershot_json_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.pershot_complex_snapshots_) {
    pershot_complex_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.pershot_cvector_snapshots_) {
    pershot_cvector_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.pershot_cmatrix_snapshots_) {
    pershot_cmatrix_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.pershot_cmap_snapshots_) {
    pershot_cmap_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.pershot_rmap_snapshots_) {
    pershot_rmap_snapshots_[pair.first].combine(pair.second);
  }

  // Combine average snapshots
  for (const auto &pair : other.average_json_snapshots_) {
    average_json_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.average_complex_snapshots_) {
    average_complex_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.average_cvector_snapshots_) {
    average_cvector_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.average_cmatrix_snapshots_) {
    average_cmatrix_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.average_cmap_snapshots_) {
    average_cmap_snapshots_[pair.first].combine(pair.second);
  }
  for (const auto &pair : other.average_rmap_snapshots_) {
    average_rmap_snapshots_[pair.first].combine(pair.second);
  }

  // Combine metadata
  for (const auto &pair : other.metadata_) {
    metadata_[pair.first] = pair.second;
  }

  // Combine additional data
  for (const auto &pair : other.additional_json_data_) {
    const auto &key = pair.first;
    erase_additional_data(key);
    additional_json_data_[key] = pair.second;
  }
  for (const auto &pair : other.additional_cvector_data_) {
    const auto &key = pair.first;
    erase_additional_data(key);
    additional_cvector_data_[key] = pair.second;
  }
  for (const auto &pair : other.additional_cmatrix_data_) {
    const auto &key = pair.first;
    erase_additional_data(key);
    additional_cmatrix_data_[key] = pair.second;
  }

  return *this;
}

ExperimentData &ExperimentData::combine(ExperimentData &&other) {
  // Combine measure
  std::move(other.memory_.begin(), other.memory_.end(),
            std::back_inserter(memory_));
  std::move(other.register_.begin(), other.register_.end(),
            std::back_inserter(register_));

  // Combine counts
  for (auto pair : other.counts_) {
    counts_[pair.first] += pair.second;
  }

  // Combine pershot snapshots
  for (auto &pair : other.pershot_json_snapshots_) {
    pershot_json_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.pershot_complex_snapshots_) {
    pershot_complex_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.pershot_cvector_snapshots_) {
    pershot_cvector_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.pershot_cmatrix_snapshots_) {
    pershot_cmatrix_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.pershot_cmap_snapshots_) {
    pershot_cmap_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.pershot_rmap_snapshots_) {
    pershot_rmap_snapshots_[pair.first].combine(std::move(pair.second));
  }

  // Combine average snapshots
  for (auto &pair : other.average_json_snapshots_) {
    average_json_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.average_complex_snapshots_) {
    average_complex_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.average_cvector_snapshots_) {
    average_cvector_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.average_cmatrix_snapshots_) {
    average_cmatrix_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.average_cmap_snapshots_) {
    average_cmap_snapshots_[pair.first].combine(std::move(pair.second));
  }
  for (auto &pair : other.average_rmap_snapshots_) {
    average_rmap_snapshots_[pair.first].combine(std::move(pair.second));
  }

  // Combine metadata
  for (auto &pair : other.metadata_) {
    metadata_[pair.first] = std::move(pair.second);
  }

  // Combine additional data
  for (auto &pair : other.additional_json_data_) {
    const auto &key = pair.first;
    erase_additional_data(key);
    additional_json_data_[key] = std::move(pair.second);
  }
  for (auto &pair : other.additional_cvector_data_) {
    const auto &key = pair.first;
    erase_additional_data(key);
    additional_cvector_data_[key] = std::move(pair.second);
  }
  for (auto &pair : other.additional_cmatrix_data_) {
    const auto &key = pair.first;
    erase_additional_data(key);
    additional_cmatrix_data_[key] = std::move(pair.second);
  }

  // Clear any remaining data from other container
  other.clear();

  return *this;
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------

json_t ExperimentData::json() const {
  // Initialize output as additional data JSON
  json_t tmp;

  // Measure data
  if (return_counts_ && counts_.empty() == false) tmp["counts"] = counts_;
  if (return_memory_ && memory_.empty() == false) tmp["memory"] = memory_;
  if (return_register_ && register_.empty() == false)
    tmp["register"] = register_;

  // Add additional data
  for (const auto &pair : additional_json_data_) {
    tmp[pair.first] = pair.second;
  }
  for (const auto &pair : additional_cvector_data_) {
    tmp[pair.first] = pair.second;
  }
  for (const auto &pair : additional_cmatrix_data_) {
    tmp[pair.first] = pair.second;
  }

  // Snapshot data
  if (return_snapshots_) {
    // Average snapshots
    for (const auto &pair : average_json_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : average_complex_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : average_cvector_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : average_cmatrix_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : average_cmap_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : average_rmap_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    // Singleshot snapshot data
    // Note these will override the average snapshots
    // if they share the same type string
    for (const auto &pair : pershot_json_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : pershot_complex_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : pershot_cvector_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : pershot_cmatrix_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : pershot_cmap_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    for (auto &pair : pershot_rmap_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
  }

  // Check if data is null (empty) and if so return an empty JSON object
  if (tmp.is_null()) return json_t::object();
  return tmp;
}

inline void to_json(json_t &js, const ExperimentData &data) {
  js = data.json();
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
