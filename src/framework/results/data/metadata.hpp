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

#ifndef _aer_framework_results_data_metadata_hpp_
#define _aer_framework_results_data_metadata_hpp_

#include "framework/results/data/subtypes/data_map.hpp"
#include "framework/results/data/subtypes/single_data.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct Metadata : public DataMap<SingleData, json_t, 1>,
                  public DataMap<SingleData, json_t, 2>,
                  public DataMap<SingleData, json_t, 3> {

  //----------------------------------------------------------------
  // Add JSON metadata
  //----------------------------------------------------------------
  template <typename... Args>
  void add(const json_t &data, const std::string &outer_key,
           const Args &... inner_keys);

  template <typename... Args>
  void add(json_t &data, const std::string &outer_key,
           const Args &... inner_keys);

  template <typename... Args>
  void add(json_t &&data, const std::string &outer_key,
           const Args &... inner_keys);

  //----------------------------------------------------------------
  // Add general metadata
  //
  // These functions allow adding general data types via conversion
  // to json_t.
  //----------------------------------------------------------------
  template <typename T, typename... Args>
  void add(const T &data, const std::string &outer_key,
           const Args &... inner_keys);

  template <typename T, typename... Args>
  void add(T &data, const std::string &outer_key, const Args &... inner_keys);

  template <typename T, typename... Args>
  void add(T &&data, const std::string &outer_key, const Args &... inner_keys);

  // Serialize engine data to JSON
  json_t to_json();

  // Combine stored data
  Metadata &combine(Metadata &&other);
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

Metadata &Metadata::combine(Metadata &&other) {
  DataMap<SingleData, json_t, 1>::combine(std::move(other));
  DataMap<SingleData, json_t, 2>::combine(std::move(other));
  DataMap<SingleData, json_t, 3>::combine(std::move(other));
  return *this;
}

json_t Metadata::to_json() {
  json_t result = json_t::object();
  DataMap<SingleData, json_t, 1>::add_to_json(result);
  DataMap<SingleData, json_t, 2>::add_to_json(result);
  DataMap<SingleData, json_t, 3>::add_to_json(result);
  return result;
}

template <typename T, typename... Args>
void Metadata::add(const T &data, const std::string &outer_key,
                   const Args &... inner_keys) {
  json_t tmp = data;
  DataMap<SingleData, json_t, sizeof...(Args) + 1>::add(
      std::move(tmp), outer_key, inner_keys...);
}

template <typename T, typename... Args>
void Metadata::add(T &data, const std::string &outer_key,
                   const Args &... inner_keys) {
  json_t tmp = data;
  DataMap<SingleData, json_t, sizeof...(Args) + 1>::add(
      std::move(tmp), outer_key, inner_keys...);
}

template <typename T, typename... Args>
void Metadata::add(T &&data, const std::string &outer_key,
                   const Args &... inner_keys) {
  json_t tmp = data;
  DataMap<SingleData, json_t, sizeof...(Args) + 1>::add(
      std::move(tmp), outer_key, inner_keys...);
}

template <typename... Args>
void Metadata::add(const json_t &data, const std::string &outer_key,
                   const Args &... inner_keys) {
  DataMap<SingleData, json_t, sizeof...(Args) + 1>::add(data, outer_key,
                                                        inner_keys...);
}

template <typename... Args>
void Metadata::add(json_t &data, const std::string &outer_key,
                   const Args &... inner_keys) {
  DataMap<SingleData, json_t, sizeof...(Args) + 1>::add(data, outer_key,
                                                        inner_keys...);
}

template <typename... Args>
void Metadata::add(json_t &&data, const std::string &outer_key,
                   const Args &... inner_keys) {
  DataMap<SingleData, json_t, sizeof...(Args) + 1>::add(
      std::move(data), outer_key, inner_keys...);
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
