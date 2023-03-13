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

#ifndef _aer_framework_results_data_cmatrix_hpp_
#define _aer_framework_results_data_cmatrix_hpp_

#include "framework/results/data/subtypes/accum_data.hpp"
#include "framework/results/data/subtypes/average_data.hpp"
#include "framework/results/data/subtypes/data_map.hpp"
#include "framework/results/data/subtypes/list_data.hpp"
#include "framework/results/data/subtypes/single_data.hpp"
#include "framework/types.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct DataCMatrix : public DataMap<SingleData, matrix<complex_t>, 1>,
                     public DataMap<SingleData, matrix<complexf_t>, 1>,
                     public DataMap<SingleData, matrix<complex_t>, 2>,
                     public DataMap<SingleData, matrix<complexf_t>, 2>,
                     public DataMap<ListData, matrix<complex_t>, 1>,
                     public DataMap<ListData, matrix<complexf_t>, 1>,
                     public DataMap<ListData, matrix<complex_t>, 2>,
                     public DataMap<ListData, matrix<complexf_t>, 2>,
                     public DataMap<AccumData, matrix<complex_t>, 1>,
                     public DataMap<AccumData, matrix<complexf_t>, 1>,
                     public DataMap<AccumData, matrix<complex_t>, 2>,
                     public DataMap<AccumData, matrix<complexf_t>, 2>,
                     public DataMap<AverageData, matrix<complex_t>, 1>,
                     public DataMap<AverageData, matrix<complexf_t>, 1>,
                     public DataMap<AverageData, matrix<complex_t>, 2>,
                     public DataMap<AverageData, matrix<complexf_t>, 2> {

  // Serialize engine data to JSON
  void add_to_json(json_t &result);

  // Combine stored data
  DataCMatrix &combine(DataCMatrix &&other);
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

DataCMatrix &DataCMatrix::combine(DataCMatrix &&other) {
  DataMap<SingleData, matrix<complex_t>, 1>::combine(std::move(other));
  DataMap<SingleData, matrix<complexf_t>, 1>::combine(std::move(other));
  DataMap<SingleData, matrix<complex_t>, 2>::combine(std::move(other));
  DataMap<SingleData, matrix<complexf_t>, 2>::combine(std::move(other));
  DataMap<ListData, matrix<complex_t>, 1>::combine(std::move(other));
  DataMap<ListData, matrix<complexf_t>, 1>::combine(std::move(other));
  DataMap<ListData, matrix<complex_t>, 2>::combine(std::move(other));
  DataMap<ListData, matrix<complexf_t>, 2>::combine(std::move(other));
  DataMap<AccumData, matrix<complex_t>, 1>::combine(std::move(other));
  DataMap<AccumData, matrix<complexf_t>, 1>::combine(std::move(other));
  DataMap<AccumData, matrix<complex_t>, 2>::combine(std::move(other));
  DataMap<AccumData, matrix<complexf_t>, 2>::combine(std::move(other));
  DataMap<AverageData, matrix<complex_t>, 1>::combine(std::move(other));
  DataMap<AverageData, matrix<complexf_t>, 1>::combine(std::move(other));
  DataMap<AverageData, matrix<complex_t>, 2>::combine(std::move(other));
  DataMap<AverageData, matrix<complexf_t>, 2>::combine(std::move(other));
  return *this;
}

void DataCMatrix::add_to_json(json_t &result) {

  DataMap<SingleData, matrix<complex_t>, 1>::add_to_json(result);
  DataMap<SingleData, matrix<complexf_t>, 1>::add_to_json(result);
  DataMap<SingleData, matrix<complex_t>, 2>::add_to_json(result);
  DataMap<SingleData, matrix<complexf_t>, 2>::add_to_json(result);
  DataMap<ListData, matrix<complex_t>, 1>::add_to_json(result);
  DataMap<ListData, matrix<complexf_t>, 1>::add_to_json(result);
  DataMap<ListData, matrix<complex_t>, 2>::add_to_json(result);
  DataMap<ListData, matrix<complexf_t>, 2>::add_to_json(result);
  DataMap<AccumData, matrix<complex_t>, 1>::add_to_json(result);
  DataMap<AccumData, matrix<complexf_t>, 1>::add_to_json(result);
  DataMap<AccumData, matrix<complex_t>, 2>::add_to_json(result);
  DataMap<AccumData, matrix<complexf_t>, 2>::add_to_json(result);
  DataMap<AverageData, matrix<complex_t>, 1>::add_to_json(result);
  DataMap<AverageData, matrix<complexf_t>, 1>::add_to_json(result);
  DataMap<AverageData, matrix<complex_t>, 2>::add_to_json(result);
  DataMap<AverageData, matrix<complexf_t>, 2>::add_to_json(result);
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
