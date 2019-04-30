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

#ifndef _aer_framework_data_hpp_
#define _aer_framework_data_hpp_

#include "framework/json.hpp"
#include "framework/snapshot.hpp"
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

class OutputData {
public:
  //----------------------------------------------------------------
  // Measurement
  //----------------------------------------------------------------

  // Add a single memory value to the counts map
  void add_memory_count(const std::string &memory);

  // Add a single memory value to the memory vector
  void add_memory_singleshot(const std::string &memory);

  // Add a single register value to the register vector
  void add_register_singleshot(const std::string &reg);

  //----------------------------------------------------------------
  // Snapshots
  //----------------------------------------------------------------

  // Add a new datum to the snapshot of the specified type and label
  // This will use the json conversion method `to_json` for
  // data type T
  template <typename T>
  void add_singleshot_snapshot(const std::string &type,
                               const std::string &label,
                               const T &datum);

  // Add a new datum to the snapshot of the specified type and label
  // This will use the json conversion method `to_json` for
  // data type T
  // if variance is true the variance of the averaged sample will also
  // be computed
  template <typename T>
  void add_average_snapshot(const std::string &type,
                            const std::string &label,
                            const std::string &memory,
                            const T &datum,
                            bool variance = false);

  // Delete all singleshot snapshots of a given type
  void clear_singleshot_snapshot(const std::string &type);

  // Delete all singleshot snapshots of a given type and label
  void clear_singleshot_snapshot(const std::string &type,
                                 const std::string &label);

  // Delete all singleshot snapshots of a given type
  void clear_average_snapshot(const std::string &type);

  // Delete all singleshot snapshots of a given type and label
  void clear_average_snapshot(const std::string &type,
                              const std::string &label);

  //----------------------------------------------------------------
  // Additional data
  //----------------------------------------------------------------

  template <typename T>
  void add_additional_data(const std::string &key, const T &data);

  void clear_additional_data(const std::string &key);

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
  OutputData& combine(OutputData &eng);

  // Operator overload for combine
  // Note this operator is not defined to be const on the input argument
  inline OutputData& operator+=(OutputData &eng) {return combine(eng);}

protected:

  //----------------------------------------------------------------
  // OutputData
  //----------------------------------------------------------------

  // Measure outcomes
  std::map<std::string, uint_t> counts_; // histogram of memory counts over shots
  std::vector<std::string> memory_;      // memory state for each shot as hex string
  std::vector<std::string> register_;   // register state for each shot as hex string

  // Snapshots
  stringmap_t<SingleShotSnapshot> singleshot_snapshots_;
  stringmap_t<AverageSnapshot> average_snapshots_;

  // Miscelaneous data
  json_t additional_data_;

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

void OutputData::set_config(const json_t &config) {
  JSON::get_value(return_counts_, "counts", config);
  JSON::get_value(return_memory_, "memory", config);
  JSON::get_value(return_register_, "register", config);
  JSON::get_value(return_snapshots_, "snapshots", config);
}


void OutputData::add_memory_count(const std::string &memory) {
  // Memory bits value
  if (return_counts_ && !memory.empty()) {
    counts_[memory] += 1;
  }
}

void OutputData::add_memory_singleshot(const std::string &memory) {
  // Memory bits value
  if (return_memory_ && !memory.empty()) {
    memory_.push_back(memory);
  }
}

void OutputData::add_register_singleshot(const std::string &reg) {
  if (return_register_ && !reg.empty()) {
      register_.push_back(reg);
  }
}


template <typename T>
void OutputData::add_singleshot_snapshot(const std::string &type,
                                          const std::string &label,
                                          const T &datum) {
  if (return_snapshots_) {                                              
    json_t js = datum; // use implicit to_json conversion function for T
    singleshot_snapshots_[type].add_data(label, js);
  }
}


template <typename T>
void OutputData::add_average_snapshot(const std::string &type,
                                      const std::string &label,
                                      const std::string &memory,
                                      const T &datum,
                                      bool variance) {
  if (return_snapshots_) {   
    json_t js = datum; // use implicit to_json conversion function for T
    average_snapshots_[type].add_data(label, memory, js, variance);
  }
}


void OutputData::clear_singleshot_snapshot(const std::string &type) {
  singleshot_snapshots_.erase(type);
}


void OutputData::clear_singleshot_snapshot(const std::string &type,
                                            const std::string &label) {
  if (singleshot_snapshots_.find(type) != singleshot_snapshots_.end()) {
    singleshot_snapshots_[type].erase(label);
  }
}


void OutputData::clear_average_snapshot(const std::string &type) {
  average_snapshots_.erase(type);
}


void OutputData::clear_average_snapshot(const std::string &type,
                                             const std::string &label) {
  if (average_snapshots_.find(type) != average_snapshots_.end()) {
    average_snapshots_[type].erase(label);
  }
}


template <typename T>
void OutputData::add_additional_data(const std::string &key, const T &data) {
  if (return_additional_data_) {
    json_t js = data; // use implicit to_json conversion function for T
    if (JSON::check_key(key, additional_data_))
      additional_data_[key].update(js.begin(), js.end());
    else
      additional_data_[key] = js;
  }
}


void OutputData::clear_additional_data(const std::string &key) {
  additional_data_.erase(key);
}


void OutputData::clear() {
  // Clear measure and counts
  counts_.clear();
  memory_.clear();
  register_.clear();
  // Clear snapshots
  singleshot_snapshots_.clear();
  average_snapshots_.clear();
  // Clear additional data
  additional_data_.clear();
}


OutputData& OutputData::combine(OutputData &data) {
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
  for (auto &pair : data.singleshot_snapshots_) {
    singleshot_snapshots_[pair.first].combine(pair.second);
  }
  for (auto &pair : data.average_snapshots_) {
    average_snapshots_[pair.first].combine(pair.second);
  }
  // Combine additional data
  // Note that this will override any fields that have the same value
  for (auto it = data.additional_data_.begin();
       it != data.additional_data_.end(); ++it) {
    additional_data_[it.key()] = it.value();
  }

  // Clear any remaining data from other container
  data.clear();

  return *this;
}


json_t OutputData::json() const {

  // Initialize output as additional data JSON
  json_t tmp = additional_data_;

  // Add standard data
  // This will override any additional data fields if they use keys:
  // "counts", "memory", "register", "snapshots"

  // Measure data
  if (return_counts_ && counts_.empty() == false)
    tmp["counts"] = counts_;
  if (return_memory_ && memory_.empty() == false)
    tmp["memory"] = memory_;
  if (return_register_ && register_.empty() == false)
    tmp["register"] = register_;
  // Average snapshot data
  if (return_snapshots_) {
    for (auto &pair : average_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second.json();
    }
    // Singleshot snapshot data
    // Note these will override the average snapshots
    // if they share the same type string
    for (auto &pair : singleshot_snapshots_) {
      tmp["snapshots"][pair.first] = pair.second.json();
    }
  }
  // Check if data is null (empty) and if so return an empty JSON object
  if (tmp.is_null())
    return json_t::object();
  return tmp;
}


//------------------------------------------------------------------------------
// Implicit JSON conversion function
//------------------------------------------------------------------------------
inline void to_json(json_t &js, const OutputData &data) {
  js = data.json();
}


//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
