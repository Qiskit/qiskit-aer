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
#include "framework/results/data/mixins/data_cmatrix.hpp"
#include "framework/results/data/mixins/data_cvector.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct Data : public DataCReg,
              public DataCVector,
              public DataCMatrix {

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
  DataCVector::combine(std::move(other));
  DataCMatrix::combine(std::move(other));
  DataCReg::combine(std::move(other));
  return *this;
}

json_t Data::to_json() {
  json_t result;
  DataCVector::add_to_json(result);
  DataCMatrix::add_to_json(result);
  DataCReg::add_to_json(result);
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
