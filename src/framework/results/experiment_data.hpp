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
#include "framework/linalg/vector.hpp"
#include "framework/linalg/vector_json.hpp"
#include "framework/results/data/data_container.hpp"
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

class ExperimentData : public DataContainer<json_t>,
                       public DataContainer<complex_t>,
                       public DataContainer<std::vector<std::complex<float>>>,
                       public DataContainer<std::vector<std::complex<double>>>,
                       public DataContainer<Vector<std::complex<float>>>,
                       public DataContainer<Vector<std::complex<double>>>,
                       public DataContainer<matrix<std::complex<float>>>,
                       public DataContainer<matrix<std::complex<double>>>,
                       public DataContainer<std::map<std::string, complex_t>>,
                       public DataContainer<std::map<std::string, double>> {
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
  // data type T unless T is one of the parent container types.
  template <typename T>
  void add_pershot_snapshot(const std::string &type, const std::string &label,
                            T &&datum);

  // Using aliases so we don't shadow parent class methods
  using DataContainer<json_t>::add_pershot_snapshot;
  using DataContainer<complex_t>::add_pershot_snapshot;
  using DataContainer<std::vector<std::complex<float>>>::add_pershot_snapshot;
  using DataContainer<std::vector<std::complex<double>>>::add_pershot_snapshot;
  using DataContainer<Vector<std::complex<float>>>::add_pershot_snapshot;
  using DataContainer<Vector<std::complex<double>>>::add_pershot_snapshot;
  using DataContainer<matrix<std::complex<float>>>::add_pershot_snapshot;
  using DataContainer<matrix<std::complex<double>>>::add_pershot_snapshot;
  using DataContainer<std::map<std::string, complex_t>>::add_pershot_snapshot;
  using DataContainer<std::map<std::string, double>>::add_pershot_snapshot;
  
  //----------------------------------------------------------------
  // Average snapshots
  //----------------------------------------------------------------

  // Add a new datum to the snapshot of the specified type and label
  // This will use the json conversion method `to_json` for
  // data type T unless T is one of the parent container types.
  // If variance is true the variance of the averaged sample will also
  // be computed
  template <typename T>
  void add_average_snapshot(const std::string &type, const std::string &label,
                            const std::string &memory, T &&datum,
                            bool variance = false);

  // Using aliases so we don't shadow parent class methods
  using DataContainer<json_t>::add_average_snapshot;
  using DataContainer<complex_t>::add_average_snapshot;
  using DataContainer<std::vector<std::complex<float>>>::add_average_snapshot;
  using DataContainer<std::vector<std::complex<double>>>::add_average_snapshot;
  using DataContainer<Vector<std::complex<float>>>::add_average_snapshot;
  using DataContainer<Vector<std::complex<double>>>::add_average_snapshot;
  using DataContainer<matrix<std::complex<float>>>::add_average_snapshot;
  using DataContainer<matrix<std::complex<double>>>::add_average_snapshot;
  using DataContainer<std::map<std::string, complex_t>>::add_average_snapshot;
  using DataContainer<std::map<std::string, double>>::add_average_snapshot;

  //----------------------------------------------------------------
  // Additional data
  //----------------------------------------------------------------

  // Add new data at the specified key.
  // If they key already exists this will override the stored data
  // This will use the json conversion method `to_json` for data type T
  // data type T unless T is one of the parent container types.
  template <typename T>
  void add_additional_data(const std::string &key, T &&data);

  // Using aliases so we don't shadow parent class methods
  using DataContainer<json_t>::add_additional_data;
  using DataContainer<complex_t>::add_additional_data;
  using DataContainer<std::vector<std::complex<float>>>::add_additional_data;
  using DataContainer<std::vector<std::complex<double>>>::add_additional_data;
  using DataContainer<Vector<std::complex<float>>>::add_additional_data;
  using DataContainer<Vector<std::complex<double>>>::add_additional_data;
  using DataContainer<matrix<std::complex<float>>>::add_additional_data;
  using DataContainer<matrix<std::complex<double>>>::add_additional_data;
  using DataContainer<std::map<std::string, complex_t>>::add_additional_data;
  using DataContainer<std::map<std::string, double>>::add_additional_data;

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  // Set the output data config options
  void set_config(const json_t &config);

  // Empty engine of stored data
  void clear();

  // Serialize engine data to JSON
  json_t to_json();

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
  // Access Templated DataContainers
  //----------------------------------------------------------------

  template <typename T>
  stringmap_t<T>& additional_data();

  template <typename T>
  const stringmap_t<T>& additional_data() const;

  template <typename T>
  stringmap_t<PershotSnapshot<T>>& pershot_snapshots();

  template <typename T>
  const stringmap_t<PershotSnapshot<T>>& pershot_snapshots() const;

  template <typename T>
  stringmap_t<AverageSnapshot<T>>& average_snapshots();

  template <typename T>
  const stringmap_t<AverageSnapshot<T>>& average_snapshots() const;


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

  // Snapshots enabled
  bool enabled = true;
  JSON::get_value(enabled, "snapshots", config);
  DataContainer<json_t>::enable(enabled);
  DataContainer<complex_t>::enable(enabled);
  DataContainer<std::vector<std::complex<float>>>::enable(enabled);
  DataContainer<std::vector<std::complex<double>>>::enable(enabled);
  DataContainer<Vector<std::complex<float>>>::enable(enabled);
  DataContainer<Vector<std::complex<double>>>::enable(enabled);
  DataContainer<matrix<std::complex<float>>>::enable(enabled);
  DataContainer<matrix<std::complex<double>>>::enable(enabled);
  DataContainer<std::map<std::string, complex_t>>::enable(enabled);
  DataContainer<std::map<std::string, double>>::enable(enabled);
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
    DataContainer<json_t>::add_pershot_snapshot(type, label, std::move(tmp));
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
    DataContainer<json_t>::add_average_snapshot(type, label, memory, std::move(tmp), variance);
  }
}

//------------------------------------------------------------------
// Additional Data
//------------------------------------------------------------------

template <typename T>
void ExperimentData::add_additional_data(const std::string &key, T &&data) {
  if (return_additional_data_) {
    json_t jdata = data;
    DataContainer<json_t>::add_additional_data(key, std::move(jdata));
  }
}

//------------------------------------------------------------------
// Access Data
//------------------------------------------------------------------
template <typename T>
stringmap_t<T>& ExperimentData::additional_data() {
  return DataContainer<T>::additional_data_;
}

template <typename T>
const stringmap_t<T>& ExperimentData::additional_data() const {
  return DataContainer<T>::additional_data_;
}

template <typename T>
stringmap_t<PershotSnapshot<T>>& ExperimentData::pershot_snapshots() {
  return DataContainer<T>::pershot_snapshots_;
}

template <typename T>
const stringmap_t<PershotSnapshot<T>>& ExperimentData::pershot_snapshots() const {
  return DataContainer<T>::pershot_snapshots_;
}

template <typename T>
stringmap_t<AverageSnapshot<T>>& ExperimentData::average_snapshots() {
  return DataContainer<T>::average_snapshots_;
}

template <typename T>
const stringmap_t<AverageSnapshot<T>>& ExperimentData::average_snapshots() const {
  return DataContainer<T>::average_snapshots_;
}

//------------------------------------------------------------------
// Clear and combine
//------------------------------------------------------------------

void ExperimentData::clear() {

  DataContainer<json_t>::clear();
  DataContainer<complex_t>::clear();
  DataContainer<std::vector<std::complex<float>>>::clear();
  DataContainer<std::vector<std::complex<double>>>::clear();
  DataContainer<Vector<std::complex<float>>>::clear();
  DataContainer<Vector<std::complex<double>>>::clear();
  DataContainer<matrix<std::complex<float>>>::clear();
  DataContainer<matrix<std::complex<double>>>::clear();
  DataContainer<std::map<std::string, complex_t>>::clear();
  DataContainer<std::map<std::string, double>>::clear();

  // Clear measurement data
  counts_.clear();
  memory_.clear();
  register_.clear();
}

ExperimentData &ExperimentData::combine(const ExperimentData &other) {

  // Combine containers
  DataContainer<json_t>::combine(other);
  DataContainer<complex_t>::combine(other);
  DataContainer<std::vector<std::complex<float>>>::combine(other);
  DataContainer<std::vector<std::complex<double>>>::combine(other);
  DataContainer<Vector<std::complex<float>>>::combine(other);
  DataContainer<Vector<std::complex<double>>>::combine(other);
  DataContainer<matrix<std::complex<float>>>::combine(other);
  DataContainer<matrix<std::complex<double>>>::combine(other);
  DataContainer<std::map<std::string, complex_t>>::combine(other);
  DataContainer<std::map<std::string, double>>::combine(other);

  // Combine measure
  std::copy(other.memory_.begin(), other.memory_.end(),
            std::back_inserter(memory_));
  std::copy(other.register_.begin(), other.register_.end(),
            std::back_inserter(register_));

  // Combine counts
  for (auto pair : other.counts_) {
    counts_[pair.first] += pair.second;
  }

  return *this;
}

ExperimentData &ExperimentData::combine(ExperimentData &&other) {

  DataContainer<json_t>::combine(std::move(other));
  DataContainer<complex_t>::combine(std::move(other));
  DataContainer<std::vector<std::complex<float>>>::combine(std::move(other));
  DataContainer<std::vector<std::complex<double>>>::combine(std::move(other));
  DataContainer<Vector<std::complex<float>>>::combine(std::move(other));
  DataContainer<Vector<std::complex<double>>>::combine(std::move(other));
  DataContainer<matrix<std::complex<float>>>::combine(std::move(other));
  DataContainer<matrix<std::complex<double>>>::combine(std::move(other));
  DataContainer<std::map<std::string, complex_t>>::combine(std::move(other));
  DataContainer<std::map<std::string, double>>::combine(std::move(other));

  // Combine measure
  std::move(other.memory_.begin(), other.memory_.end(),
            std::back_inserter(memory_));
  std::move(other.register_.begin(), other.register_.end(),
            std::back_inserter(register_));

  // Combine counts
  for (auto pair : other.counts_) {
    counts_[pair.first] += pair.second;
  }

  // Clear any remaining data from other container
  other.clear();

  return *this;
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------

json_t ExperimentData::to_json() {
  // Initialize output as additional data JSON
  json_t js;

  // Add all container data
  DataContainer<json_t>::add_to_json(js);
  DataContainer<complex_t>::add_to_json(js);
  DataContainer<std::vector<std::complex<float>>>::add_to_json(js);
  DataContainer<std::vector<std::complex<double>>>::add_to_json(js);
  DataContainer<Vector<std::complex<float>>>::add_to_json(js);
  DataContainer<Vector<std::complex<double>>>::add_to_json(js);
  DataContainer<matrix<std::complex<float>>>::add_to_json(js);
  DataContainer<matrix<std::complex<double>>>::add_to_json(js);
  DataContainer<std::map<std::string, complex_t>>::add_to_json(js);
  DataContainer<std::map<std::string, double>>::add_to_json(js);

  // Measure data
  if (return_counts_ && counts_.empty() == false) js["counts"] = counts_;
  if (return_memory_ && memory_.empty() == false) js["memory"] = memory_;
  if (return_register_ && register_.empty() == false)
    js["register"] = register_;

  // Check if data is null (empty) and if so return an empty JSON object
  if (js.is_null()) return json_t::object();
  return js;
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
