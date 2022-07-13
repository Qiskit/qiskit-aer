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

#ifndef _aer_framework_results_experiment_result_hpp_
#define _aer_framework_results_experiment_result_hpp_

#include "framework/results/legacy/snapshot_data.hpp"
#include "framework/results/data/data.hpp"
#include "framework/results/data/metadata.hpp"
#include "framework/creg.hpp"
#include "framework/opset.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct ExperimentResult {
  using OpType = Operations::OpType;
  using DataSubType = Operations::DataSubType;

public:

  // Status 
  enum class Status {empty, completed, error};

  // Experiment data
  Data data;
  SnapshotData legacy_data; // Legacy snapshot data
  uint_t shots;
  uint_t seed;
  double time_taken;

  // Success and status
  Status status = Status::empty;
  std::string message; // error message

  // Metadata
  json_t header;
  Metadata metadata;

  // Serialize engine data to JSON
  json_t to_json();

  // Set the output data config options
  void set_config(const json_t &config);

  // Combine stored data
  ExperimentResult& combine(ExperimentResult &&other);

  //save creg as count data 
  void save_count_data(const ClassicalRegister& creg, bool save_memory);
  void save_count_data(const std::vector<ClassicalRegister>& cregs, bool save_memory);

  // Save data type which can be averaged over all shots.
  // This supports DataSubTypes: list, c_list, accum, c_accum, average, c_average
  template <class T>
  void save_data_average(const ClassicalRegister& creg,
                         const std::string &key, const T& datum, OpType type,
                         DataSubType subtype = DataSubType::average);

  template <class T>
  void save_data_average(const ClassicalRegister& creg,
                         const std::string &key, T&& datum, OpType type,
                         DataSubType subtype = DataSubType::average);

  // Save single shot data type. Typically this will be the value for the
  // last shot of the simulation
  template <class T>
  void save_data_single(const ClassicalRegister& creg,
                        const std::string &key, const T& datum, OpType type);

  template <class T>
  void save_data_single(const ClassicalRegister& creg,
                        const std::string &key, T&& datum, OpType type);

  // Save data type which is pershot and does not support accumulator or average
  // This supports DataSubTypes: single, c_single, list, c_list
  template <class T>
  void save_data_pershot(const ClassicalRegister& creg,
                         const std::string &key, const T& datum, OpType type,
                         DataSubType subtype = DataSubType::list);

  template <class T>
  void save_data_pershot(const ClassicalRegister& creg,
                         const std::string &key, T&& datum, OpType type,
                         DataSubType subtype = DataSubType::list);


 };

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

ExperimentResult& ExperimentResult::combine(ExperimentResult &&other) {
  legacy_data.combine(std::move(other.legacy_data));
  data.combine(std::move(other.data));
  metadata.combine(std::move(other.metadata));
  return *this;
}

void ExperimentResult::set_config(const json_t &config) {
  legacy_data.set_config(config);
}

json_t ExperimentResult::to_json() {
  // Initialize output as additional data JSON
  json_t result;
  result["data"] = data.to_json();
  json_t legacy_snapshots = legacy_data.to_json();
  if (!legacy_snapshots.empty()) {
    result["data"]["snapshots"] = std::move(legacy_snapshots);
  }
  result["shots"] = shots;
  result["seed_simulator"] = seed;
  result["success"] = (status == Status::completed);
  switch (status) {
    case Status::completed:
      result["status"] = std::string("DONE");
      break;
    case Status::error:
      result["status"] = std::string("ERROR: ") + message;
      break;
    case Status::empty:
      result["status"] = std::string("EMPTY");
  }
  result["time_taken"] = time_taken;
  if (header.empty() == false)
    result["header"] = header;
  result["metadata"] = metadata.to_json();
  return result;
}

void ExperimentResult::save_count_data(const ClassicalRegister& creg, bool save_memory) {
  if (creg.memory_size() > 0) {
    std::string memory_hex = creg.memory_hex();
    data.add_accum(static_cast<uint_t>(1ULL), "counts", memory_hex);
    if(save_memory) {
      data.add_list(std::move(memory_hex), "memory");
    }
  }
}

void ExperimentResult::save_count_data(const std::vector<ClassicalRegister>& cregs, bool save_memory) {
  for(int_t i=0;i<cregs.size();i++)
    save_count_data(cregs[i], save_memory);
}

template <class T>
void ExperimentResult::save_data_average(const ClassicalRegister& creg,
                                         const std::string &key,
                                         const T& datum, OpType type,
                                         DataSubType subtype) {
  switch (subtype) {
    case DataSubType::list:
      data.add_list(datum, key);
      break;
    case DataSubType::c_list:
      data.add_list(datum, key, creg.memory_hex());
      break;
    case DataSubType::accum:
      data.add_accum(datum, key);
      break;
    case DataSubType::c_accum:
      data.add_accum(datum, key, creg.memory_hex());
      break;
    case DataSubType::average:
      data.add_average(datum, key);
      break;
    case DataSubType::c_average:
      data.add_average(datum, key, creg.memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid average data subtype for data key: " + key);
  }
  metadata.add(type, "result_types", key);
  metadata.add(subtype, "result_subtypes", key);
}

template <class T>
void ExperimentResult::save_data_average(const ClassicalRegister& creg,
                                         const std::string &key,
                                         T&& datum, OpType type,
                                         DataSubType subtype) {
  switch (subtype) {
    case DataSubType::list:
      data.add_list(std::move(datum), key);
      break;
    case DataSubType::c_list:
      data.add_list(std::move(datum), key, creg.memory_hex());
      break;
    case DataSubType::accum:
      data.add_accum(std::move(datum), key);
      break;
    case DataSubType::c_accum:
      data.add_accum(std::move(datum), key, creg.memory_hex());
      break;
    case DataSubType::average:
      data.add_average(std::move(datum), key);
      break;
    case DataSubType::c_average:
      data.add_average(std::move(datum), key, creg.memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid average data subtype for data key: " + key);
  }
  metadata.add(type, "result_types", key);
  metadata.add(subtype, "result_subtypes", key);
}

template <class T>
void ExperimentResult::save_data_pershot(const ClassicalRegister& creg,
                                       const std::string &key,
                                       const T& datum, OpType type,
                                       DataSubType subtype) {
  switch (subtype) {
  case DataSubType::single:
    data.add_single(datum, key);
    break;
  case DataSubType::c_single:
    data.add_single(datum, key, creg.memory_hex());
    break;
  case DataSubType::list:
    data.add_list(datum, key);
    break;
  case DataSubType::c_list:
    data.add_list(datum, key, creg.memory_hex());
    break;
  default:
    throw std::runtime_error("Invalid pershot data subtype for data key: " + key);
  }
  metadata.add(type, "result_types", key);
  metadata.add(subtype, "result_subtypes", key);
}

template <class T>
void ExperimentResult::save_data_pershot(const ClassicalRegister& creg, 
                                         const std::string &key,
                                         T&& datum, OpType type,
                                         DataSubType subtype) {
  switch (subtype) {
    case DataSubType::single:
      data.add_single(std::move(datum), key);
      break;
    case DataSubType::c_single:
      data.add_single(std::move(datum), key, creg.memory_hex());
      break;
    case DataSubType::list:
      data.add_list(std::move(datum), key);
      break;
    case DataSubType::c_list:
      data.add_list(std::move(datum), key, creg.memory_hex());
      break;
    default:
      throw std::runtime_error("Invalid pershot data subtype for data key: " + key);
  }
  metadata.add(type, "result_types", key);
  metadata.add(subtype, "result_subtypes", key);
}

template <class T>
void ExperimentResult::save_data_single(const ClassicalRegister& creg,
                                        const std::string &key,
                                        const T& datum, OpType type) {
  data.add_single(datum, key);
  metadata.add(type, "result_types", key);
  metadata.add(DataSubType::single, "result_subtypes", key);
}

template <class T>
void ExperimentResult::save_data_single(const ClassicalRegister& creg,
                                        const std::string &key,
                                        T&& datum, OpType type) {
  data.add_single(std::move(datum), key);
  metadata.add(type, "result_types", key);
  metadata.add(DataSubType::single, "result_subtypes", key);
}



//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
