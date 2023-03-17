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

#ifndef _aer_framework_results_data_mps_hpp_
#define _aer_framework_results_data_mps_hpp_

#include "framework/results/data/subtypes/data_map.hpp"
#include "framework/results/data/subtypes/list_data.hpp"
#include "framework/results/data/subtypes/single_data.hpp"
#include "framework/types.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================
// using cmat = std::vector<std::vector<complex_t>>;
struct DataMPS : public DataMap<SingleData, mps_container_t, 1>,
                 public DataMap<SingleData, mps_container_t, 2>,
                 public DataMap<ListData, mps_container_t, 1>,
                 public DataMap<ListData, mps_container_t, 2> {

  // Serialize engine data to JSON
  void add_to_json(json_t &result);

  // Combine stored data
  DataMPS &combine(DataMPS &&other);
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

DataMPS &DataMPS::combine(DataMPS &&other) {
  DataMap<SingleData, mps_container_t, 1>::combine(std::move(other));
  DataMap<SingleData, mps_container_t, 2>::combine(std::move(other));
  DataMap<ListData, mps_container_t, 1>::combine(std::move(other));
  DataMap<ListData, mps_container_t, 2>::combine(std::move(other));
  return *this;
}

void DataMPS::add_to_json(json_t &result) {
  DataMap<SingleData, mps_container_t, 1>::add_to_json(result);
  DataMap<SingleData, mps_container_t, 2>::add_to_json(result);
  DataMap<ListData, mps_container_t, 1>::add_to_json(result);
  DataMap<ListData, mps_container_t, 2>::add_to_json(result);
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
