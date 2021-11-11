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

#ifndef _aer_framework_results_data_hpp_
#define _aer_framework_results_data_hpp_

// Data primatives
#include "framework/results/data/subtypes/data_map.hpp"
#include "framework/results/data/subtypes/accum_data.hpp"
#include "framework/results/data/subtypes/average_data.hpp"
#include "framework/results/data/subtypes/list_data.hpp"
#include "framework/results/data/subtypes/single_data.hpp"

// Data Containers
#include "framework/results/data/mixins/data_creg.hpp"
#include "framework/results/data/mixins/data_rvalue.hpp"
#include "framework/results/data/mixins/data_rvector.hpp"
#include "framework/results/data/mixins/data_rdict.hpp"
#include "framework/results/data/mixins/data_cmatrix.hpp"
#include "framework/results/data/mixins/data_cvector.hpp"
#include "framework/results/data/mixins/data_cdict.hpp"
#include "framework/results/data/mixins/data_json.hpp"
#include "framework/results/data/mixins/data_mps.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct Data : public DataCreg,
              public DataRValue,
              public DataRVector,
              public DataRDict,
              public DataCVector,
              public DataCMatrix,
              public DataCDict,
              public DataJSON,
              public DataMPS {

  //----------------------------------------------------------------
  // Measurement data
  //----------------------------------------------------------------

  // Add outcome to count dictionary
  void add_count(const std::string &outcome);

  // Add outcome to memory list
  void add_memory(const std::string &outcome);
  void add_memory(std::string &&outcome);

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
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

Data &Data::combine(Data &&other) {
  DataRValue::combine(std::move(other));
  DataRVector::combine(std::move(other));
  DataRDict::combine(std::move(other));
  DataCVector::combine(std::move(other));
  DataCMatrix::combine(std::move(other));
  DataCDict::combine(std::move(other));
  DataJSON::combine(std::move(other));
  DataMPS::combine(std::move(other));
  DataCreg::combine(std::move(other));
  return *this;
}

json_t Data::to_json() {
  json_t result = json_t::object();
  DataRValue::add_to_json(result);
  DataRVector::add_to_json(result);
  DataRDict::add_to_json(result);
  DataCVector::add_to_json(result);
  DataCMatrix::add_to_json(result);
  DataCDict::add_to_json(result);
  DataJSON::add_to_json(result);
  DataMPS::add_to_json(result);
  DataCreg::add_to_json(result);
  return result;
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
  DataMap<AverageData, T, sizeof...(Args) + 1>::add(std::move(data), outer_key,
                                                    inner_keys...);
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
