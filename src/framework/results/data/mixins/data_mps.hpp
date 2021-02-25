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
//  using MPSContainer = std::pair<std::vector<std::pair<cmatrix_t, cmatrix_t>>, 
  //			 std::vector<rvector_t>>;

class MPSContainer {
public:
  MPSContainer() {}

private:
  size_t num_qubits_=0;
  // lambda_size[i] is the length of lambda_struct[i];
  std::vector<size_t> lambda_size_;  
  // lambda_reg_[i] contains the Schmidt coefficients for qubit[i+1]
  std::vector<std::vector<double>> lambda_reg_;
  // q_reg_size_[i] contains the num_rows and num_columns for qreg[i] 
  std::vector<std::pair<size_t, size_t>> q_reg_size_;
  // q_reg_[i] contains 2 complex matrices that make the tensor for qubit i
  std::vector<std::pair<cmatrix_t, cmatrix_t>> q_reg_;
  // The internal ordering of the qubits
  std::vector<uint_t> order_;

};

struct DataMPS : public DataMap<SingleData, MPSContainer, 1>,
                 public DataMap<SingleData, MPSContainer, 2>,
                 public DataMap<ListData, MPSContainer, 1>,
                 public DataMap<ListData, MPSContainer, 2> {

  // Serialize engine data to JSON
  void add_to_json(json_t &result);

  // Combine stored data
  DataMPS &combine(DataMPS &&other);
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

DataMPS &DataMPS::combine(DataMPS &&other) {
  DataMap<SingleData, MPSContainer, 1>::combine(std::move(other));
  DataMap<SingleData, MPSContainer, 2>::combine(std::move(other));
  DataMap<ListData, MPSContainer, 1>::combine(std::move(other));
  DataMap<ListData, MPSContainer, 2>::combine(std::move(other));
  return *this;
}

void DataMPS::add_to_json(json_t &result) {
  DataMap<SingleData, MPSContainer, 1>::add_to_json(result);
  DataMap<SingleData, MPSContainer, 2>::add_to_json(result);
  DataMap<ListData, MPSContainer, 1>::add_to_json(result);
  DataMap<ListData, MPSContainer, 2>::add_to_json(result);
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
