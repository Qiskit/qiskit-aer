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
  json_t to_json();
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
json_t Result::to_json() {
  // Initialize output as additional data JSON
  json_t js;
  js["qobj_id"] = qobj_id;
  
  js["backend_name"] = backend_name;
  js["backend_version"] = backend_version;
  js["date"] = date;
  js["job_id"] = job_id;
  for (auto& res : results) {
    js["results"].push_back(res.to_json());
  }
  if (header.empty() == false)
    js["header"] = header;
  if (metadata.empty() == false)
    js["metadata"] = metadata;
  js["success"] = (status == Status::completed);
  switch (status) {
    case Status::completed:
      js["status"] = std::string("COMPLETED");
      break;
    case Status:: partial_completed:
      js["status"] = std::string("PARTIAL COMPLETED");
      break;
    case Status::error:
      js["status"] = std::string("ERROR: ") + message;
      break;
    case Status::empty:
      js["status"] = std::string("EMPTY");
  }
  return js;
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
