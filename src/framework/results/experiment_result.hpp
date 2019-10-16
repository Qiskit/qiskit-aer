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

#include "framework/results/experiment_data.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct ExperimentResult {
public:

  // Status 
  enum class Status {empty, completed, error};

  // Experiment data
  ExperimentData data;
  uint_t shots;
  uint_t seed;
  double time_taken;

  // Success and status
  Status status = Status::empty;
  std::string message; // error message

  // Metadata
  json_t header;
  json_t metadata;

  // Clear all metadata for given key
  void clear_metadata(const std::string &key);
  
  // Append metadata for a given key.
  // This assumes the metadata value is a dictionary and appends
  // any new values
  template <typename T>
  void add_metadata(const std::string &key, const T &data);

  // Serialize engine data to JSON
  json_t json() const;
};


//------------------------------------------------------------------------------
// Add metadata
//------------------------------------------------------------------------------
template <typename T>
void ExperimentResult::add_metadata(const std::string &key, const T &meta) {
  json_t js = meta; // use implicit to_json conversion function for T
  if (JSON::check_key(key, metadata))
    metadata[key].update(js.begin(), js.end());
  else
    metadata[key] = js;
}

void ExperimentResult::clear_metadata(const std::string &key) {
  metadata.erase(key);
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------
json_t ExperimentResult::json() const {
  // Initialize output as additional data JSON
  json_t result;
  result["data"] = data;
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
  if (metadata.empty() == false)
    result["metadata"] = metadata;
  return result;
}

inline void to_json(json_t &js, const ExperimentResult &result) {
  js = result.json();
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
