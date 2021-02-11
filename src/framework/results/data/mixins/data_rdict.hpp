/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2021.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_results_data_rdict_hpp_
#define _aer_framework_results_data_rdict_hpp_

#include <map>
#include "framework/results/data/subtypes/data_map.hpp"
#include "framework/results/data/subtypes/list_data.hpp"
#include "framework/results/data/subtypes/accum_data.hpp"
#include "framework/results/data/subtypes/average_data.hpp"
#include "framework/types.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct DataRDict : public DataMap<ListData, std::map<std::string, double>, 1>,
                   public DataMap<ListData, std::map<std::string, double>, 2>,
                   public DataMap<AccumData, std::map<std::string, double>, 1>,
                   public DataMap<AccumData, std::map<std::string, double>, 2>,
                   public DataMap<AverageData, std::map<std::string, double>, 1>,
                   public DataMap<AverageData, std::map<std::string, double>, 2> {

  // Serialize engine data to JSON
  void add_to_json(json_t &result);

  // Combine stored data
  DataRDict &combine(DataRDict &&other);
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

DataRDict &DataRDict::combine(DataRDict &&other) {
  DataMap<ListData, std::map<std::string, double>, 1>::combine(std::move(other));
  DataMap<ListData, std::map<std::string, double>, 2>::combine(std::move(other));
  DataMap<AccumData, std::map<std::string, double>, 1>::combine(std::move(other));
  DataMap<AccumData, std::map<std::string, double>, 2>::combine(std::move(other));
  DataMap<AverageData, std::map<std::string, double>, 1>::combine(std::move(other));
  DataMap<AverageData, std::map<std::string, double>, 2>::combine(std::move(other));
  return *this;
}

void DataRDict::add_to_json(json_t &result) {
  DataMap<ListData, std::map<std::string, double>, 1>::add_to_json(result);
  DataMap<ListData, std::map<std::string, double>, 2>::add_to_json(result);
  DataMap<AccumData, std::map<std::string, double>, 1>::add_to_json(result);
  DataMap<AccumData, std::map<std::string, double>, 2>::add_to_json(result);
  DataMap<AverageData, std::map<std::string, double>, 1>::add_to_json(result);
  DataMap<AverageData, std::map<std::string, double>, 2>::add_to_json(result);
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
