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
#include "framework/results/snapshot.hpp"
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
  // data type T
  template <typename T>
  void add_pershot_snapshot(const std::string &type, const std::string &label,
                            const T &datum);

  //----------------------------------------------------------------
  // Average snapshots
  //----------------------------------------------------------------

  // Add a new datum to the snapshot of the specified type and label
  // This will use the json conversion method `to_json` for
  // data type T
  // if variance is true the variance of the averaged sample will also
  // be computed
  template <typename T>
  void add_average_snapshot(const std::string &type, const std::string &label,
                            const std::string &memory, const T &datum,
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
  ExperimentData &combine(ExperimentData &eng);

  // Operator overload for combine
  // Note this operator is not defined to be const on the input argument
  inline ExperimentData &operator+=(ExperimentData &eng) {
    return combine(eng);
  }

 protected:
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
  // Pershot Snapshots
  //----------------------------------------------------------------
  // The outer keys of the snapshot maps are the "snapshot_type".
  // The JSON type snapshot is used as a generic dynamic-type
  // storage since it can store any class that has a `to_json`
  // method implemented.

  // Generic JSON pershot snapshots
  stringmap_t<SingleShotSnapshot> pershot_json_snapshots_;

  //----------------------------------------------------------------
  // Average Snapshots
  //----------------------------------------------------------------
  // The outer keys of the snapshot maps are the "snapshot_type"
  // The JSON type snapshot is used as a generic dynamic-type
  // storage since it can store any class that has a `to_json`
  // method implemented.

  // Generic JSON average snapshots
  stringmap_t<AverageSnapshot> average_json_snapshots_;

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

template <typename T>
void ExperimentData::add_pershot_snapshot(const std::string &type,
                                          const std::string &label,
                                          const T &datum) {
  if (return_snapshots_) {
    // use implicit to_json conversion function for T
    json_t tmp = datum;
    pershot_json_snapshots_[type].add_data(label, tmp);
  }
}

//------------------------------------------------------------------
// Average Snapshots
//------------------------------------------------------------------

template <typename T>
void ExperimentData::add_average_snapshot(const std::string &type,
                                          const std::string &label,
                                          const std::string &memory,
                                          const T &datum, bool variance) {
  if (return_snapshots_) {
    json_t tmp = datum;  // use implicit to_json conversion function for T
    average_json_snapshots_[type].add_data(label, memory, tmp, variance);
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
void ExperimentData::add_additional_data(const std::string &key, json_t &&data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    erase_additional_data(key);
    additional_json_data_[key] = data;
  }
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         cvector_t &&data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    erase_additional_data(key);
    additional_cvector_data_[key] = data;
  }
}

template <>
void ExperimentData::add_additional_data(const std::string &key,
                                         cmatrix_t &&data) {
  check_reserved_key(key);
  if (return_additional_data_) {
    erase_additional_data(key);
    additional_cmatrix_data_[key] = data;
  }
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

template<>
void ExperimentData::add_metadata(const std::string &key, json_t &&data) {
  auto elt = metadata_.find("key");
  if (elt == metadata_.end()) {
    // If key doesn't already exist add new data
    metadata_[key] = data;
  } else {
    // If key already exists append with additional data
    elt->second.update(data.begin(), data.end());
  }
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
  // Clear average snapshots
  average_json_snapshots_.clear();
  // Clear additional data
  additional_json_data_.clear();
  additional_cvector_data_.clear();
  additional_cmatrix_data_.clear();
  // Clear metadata
  metadata_.clear();
}

ExperimentData &ExperimentData::combine(ExperimentData &data) {
  // Combine measure
  std::move(data.memory_.begin(), data.memory_.end(),
            std::back_inserter(memory_));
  std::move(data.register_.begin(), data.register_.end(),
            std::back_inserter(register_));
  // Combine counts
  for (auto pair : data.counts_) {
    counts_[pair.first] += pair.second;
  }
  // Combine snapshots
  for (auto &pair : data.pershot_json_snapshots_) {
    pershot_json_snapshots_[pair.first].combine(pair.second);
  }
  for (auto &pair : data.average_json_snapshots_) {
    average_json_snapshots_[pair.first].combine(pair.second);
  }
  // Combine metadata
  for (auto &pair : data.metadata_) {
    metadata_[pair.first] = pair.second;
  }

  // Combine additional data
  for (auto &pair : data.additional_json_data_) {
    const auto &key = pair.first;
    erase_additional_data(key);
    additional_json_data_[key] = std::move(pair.second);
  }
  for (auto &pair : data.additional_cvector_data_) {
    const auto &key = pair.first;
    erase_additional_data(key);
    additional_cvector_data_[key] = std::move(pair.second);
  }
  for (auto &pair : data.additional_cmatrix_data_) {
    const auto &key = pair.first;
    erase_additional_data(key);
    additional_cmatrix_data_[key] = std::move(pair.second);
  }
  // Clear any remaining data from other container
  data.clear();

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
    for (auto &pair : average_json_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second;
    }
    // Singleshot snapshot data
    // Note these will override the average snapshots
    // if they share the same type string
    for (auto &pair : pershot_json_snapshots_) {
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
