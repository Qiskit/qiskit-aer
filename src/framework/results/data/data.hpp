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

#ifndef _aer_framework_results_data_hpp_
#define _aer_framework_results_data_hpp_

#include "framework/results/data/data_map.hpp"
#include "framework/results/data/subtypes/accum_data.hpp"
#include "framework/results/data/subtypes/average_data.hpp"
#include "framework/results/data/subtypes/list_data.hpp"
#include "framework/results/data/subtypes/single_data.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct Data : public DataMap<SingleData, Vector<complex_t>, 1>,
              public DataMap<SingleData, Vector<complexf_t>, 1>,
              public DataMap<SingleData, matrix<complex_t>, 1>,
              public DataMap<SingleData, matrix<complexf_t>, 1> {

  //----------------------------------------------------------------
  // Measurement data
  //----------------------------------------------------------------

  // Count data
  DataMap<AccumData, uint_t> counts;

  // Memory data
  ListData<std::string> memory;

  // Add outcome to count dictionary
  void add_count(const std::string& outcome);

  // Add outcome to memory list
  void add_memory(const std::string& outcome);
  void add_memory(std::string&& outcome);

  //----------------------------------------------------------------
  // Add single data
  //----------------------------------------------------------------
  template <class T, typename... Args>
  void add_single(const T &data, const std::string &outer_key,
                  const Args &... inner_keys);

  template <class T, typename... Args>
  void add_single(T &data, const std::string &outer_key,
                  const Args &... inner_keys);

  template <class T, typename... Args>
  void add_single(T &&data, const std::string &outer_key,
                  const Args &... inner_keys);

  //----------------------------------------------------------------
  // Add list data
  //----------------------------------------------------------------
  template <class T, typename... Args>
  void add_list(const T &data, const std::string &outer_key,
                const Args &... inner_keys);

  template <class T, typename... Args>
  void add_list(T &data, const std::string &outer_key,
                const Args &... inner_keys);

  template <class T, typename... Args>
  void add_list(T &&data, const std::string &outer_key,
                const Args &... inner_keys);

  //----------------------------------------------------------------
  // Add accum data
  //----------------------------------------------------------------
  template <class T, typename... Args>
  void add_accum(const T &data, const std::string &outer_key,
                 const Args &... inner_keys);

  template <class T, typename... Args>
  void add_accum(T &data, const std::string &outer_key,
                 const Args &... inner_keys);

  template <class T, typename... Args>
  void add_accum(T &&data, const std::string &outer_key,
                 const Args &... inner_keys);

  //----------------------------------------------------------------
  // Add average data
  //----------------------------------------------------------------
  template <class T, typename... Args>
  void add_average(const T &data, const std::string &outer_key,
                   const Args &... inner_keys);

  template <class T, typename... Args>
  void add_average(T &data, const std::string &outer_key,
                   const Args &... inner_keys);

  template <class T, typename... Args>
  void add_average(T &&data, const std::string &outer_key,
                   const Args &... inner_keys);

  //----------------------------------------------------------------
  // Utility and config
  //----------------------------------------------------------------

  // Serialize engine data to JSON
  json_t to_json();

  // Combine stored data
  Data &combine(Data &&other);

  // Set the output data config options
  void set_config(const json_t &config);
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

Data &Data::combine(Data &&other) {
  // Measurement data
  counts.combine(std::move(other.counts));
  memory.combine(std::move(other.memory));
  // General data
  DataMap<SingleData, Vector<complex_t>, 1>::combine(std::move(other));
  DataMap<SingleData, Vector<complexf_t>, 1>::combine(std::move(other));
  DataMap<SingleData, matrix<complex_t>, 1>::combine(std::move(other));
  DataMap<SingleData, matrix<complexf_t>, 1>::combine(std::move(other));
  return *this;
}

json_t Data::to_json() {
  json_t result;
  // General data
  DataMap<SingleData, Vector<complex_t>, 1>::add_to_json(result);
  DataMap<SingleData, Vector<complexf_t>, 1>::add_to_json(result);
  DataMap<SingleData, matrix<complex_t>, 1>::add_to_json(result);
  DataMap<SingleData, matrix<complexf_t>, 1>::add_to_json(result);

  // Measurement data. This should be last to ensure it overrides
  // any other data that might have used the "count" keys
  result["counts"] = counts.to_json();
  if (memory.enabled) {
    result["memory"] = memory.to_json();
  }
  return result;
}

void Data::set_config(const json_t &config) {
  JSON::get_value(memory.enabled, "memory", config);
}

void Data::add_count(const std::string& outcome) {
  if (!outcome.empty()) {
    counts.add(1, outcome);
  }
}

void Data::add_memory(const std::string& outcome) {
  if (memory.enabled && !outcome.empty()) {
    memory.add(outcome);
  }
}
void Data::add_memory(std::string&& outcome) {
  if (memory.enabled && !outcome.empty()) {
    memory.add(std::move(outcome));
  }
}

template <class T, typename... Args>
void Data::add_single(const T &data, const std::string &outer_key,
                      const Args &... inner_keys) {
  DataMap<SingleData, T, sizeof...(Args) + 1>::add(data, outer_key,
                                                   inner_keys...);
}

template <class T, typename... Args>
void Data::add_single(T &data, const std::string &outer_key,
                      const Args &... inner_keys) {
  DataMap<SingleData, T, sizeof...(Args) + 1>::add(data, outer_key,
                                                   inner_keys...);
}

template <class T, typename... Args>
void Data::add_single(T &&data, const std::string &outer_key,
                      const Args &... inner_keys) {
  DataMap<SingleData, T, sizeof...(Args) + 1>::add(std::move(data), outer_key,
                                                   inner_keys...);
}

template <class T, typename... Args>
void Data::add_list(const T &data, const std::string &outer_key,
                    const Args &... inner_keys) {
  DataMap<ListData, T, sizeof...(Args) + 1>::add(data, outer_key,
                                                 inner_keys...);
}

template <class T, typename... Args>
void Data::add_list(T &data, const std::string &outer_key,
                    const Args &... inner_keys) {
  DataMap<ListData, T, sizeof...(Args) + 1>::add(data, outer_key,
                                                 inner_keys...);
}

template <class T, typename... Args>
void Data::add_list(T &&data, const std::string &outer_key,
                    const Args &... inner_keys) {
  DataMap<ListData, T, sizeof...(Args) + 1>::add(std::move(data), outer_key,
                                                 inner_keys...);
}

template <class T, typename... Args>
void Data::add_accum(const T &data, const std::string &outer_key,
                     const Args &... inner_keys) {
  DataMap<AccumData, T, sizeof...(Args) + 1>::add(data, outer_key,
                                                  inner_keys...);
}

template <class T, typename... Args>
void Data::add_accum(T &data, const std::string &outer_key,
                     const Args &... inner_keys) {
  DataMap<AccumData, T, sizeof...(Args) + 1>::add(data, outer_key,
                                                  inner_keys...);
}

template <class T, typename... Args>
void Data::add_accum(T &&data, const std::string &outer_key,
                     const Args &... inner_keys) {
  DataMap<AccumData, T, sizeof...(Args) + 1>::add(std::move(data), outer_key,
                                                  inner_keys...);
}

template <class T, typename... Args>
void Data::add_average(const T &data, const std::string &outer_key,
                       const Args &... inner_keys) {
  DataMap<AverageData, T, sizeof...(Args) + 1>::add(data, outer_key,
                                                       inner_keys...);
}

template <class T, typename... Args>
void Data::add_average(T &data, const std::string &outer_key,
                       const Args &... inner_keys) {
  DataMap<AverageData, T, sizeof...(Args) + 1>::add(data, outer_key,
                                                       inner_keys...);
}

template <class T, typename... Args>
void Data::add_average(T &&data, const std::string &outer_key,
                       const Args &... inner_keys) {
  DataMap<AverageData, T, sizeof...(Args) + 1>::add(
      std::move(data), outer_key, inner_keys...);
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
