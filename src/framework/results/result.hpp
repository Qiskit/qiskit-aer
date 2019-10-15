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

#ifndef _aer_framework_results_result_hpp_
#define _aer_framework_results_result_hpp_

#include "framework/results/experiment_result.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct Result {
public:

  // Result status:
  // completed: all experiments were executed successfully
  // partial: only some experiments were executed succesfully
  enum class Status {empty, completed, partial_completed, error};

  // Constructor
  Result(size_t num_exp = 0) {results.resize(num_exp);}

  // Experiment results
  std::vector<ExperimentResult> results;

  // Job metadata
  std::string backend_name;
  std::string backend_version;
  std::string qobj_id;
  std::string job_id;
  std::string date;
  
  // Success and status
  Status status = Status::empty; // Result status
  std::string message; // error message

  // Metadata
  json_t header;
  json_t metadata;

  // Size and resize
  auto size() {return results.size();}
  void resize(size_t size) {results.resize(size);}

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
void Result::add_metadata(const std::string &key, const T &data) {
  json_t js = data; // use implicit to_json conversion function for T
  if (JSON::check_key(key, metadata))
    metadata[key].update(js.begin(), js.end());
  else
    metadata[key] = js;
}

void Result::clear_metadata(const std::string &key) {
  metadata.erase(key);
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------
json_t Result::json() const {
  // Initialize output as additional data JSON
  json_t result;
  result["qobj_id"] = qobj_id;
  
  result["backend_name"] = backend_name;
  result["backend_version"] = backend_version;
  result["date"] = date;
  result["job_id"] = job_id;
  result["results"] = results;
  if (header.empty() == false)
    result["header"] = header;
  if (metadata.empty() == false)
    result["metadata"] = metadata;
  result["success"] = (status == Status::completed);
  switch (status) {
    case Status::completed:
      result["status"] = std::string("COMPLETED");
      break;
    case Status:: partial_completed:
      result["status"] = std::string("PARTIAL COMPLETED");
      break;
    case Status::error:
      result["status"] = std::string("ERROR: ") + message;
      break;
    case Status::empty:
      result["status"] = std::string("EMPTY");
  }
  return result;
}

inline void to_json(json_t &js, const Result &result) {
  js = result.json();
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
