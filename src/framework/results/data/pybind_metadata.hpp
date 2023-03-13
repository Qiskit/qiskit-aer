/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_result_data_pybind_metadata_hpp_
#define _aer_framework_result_data_pybind_metadata_hpp_

#include "framework/results/data/metadata.hpp"
#include "framework/results/data/subtypes/pybind_data_map.hpp"

namespace AerToPy {

// Move an ExperimentResult metdata object to a Python dict
template <>
py::object to_python(AER::Metadata &&metadata);

} // end namespace AerToPy

//============================================================================
// Implementations
//============================================================================

template <>
py::object AerToPy::to_python(AER::Metadata &&metadata) {
  py::dict pydata;
  add_to_python(
      pydata,
      static_cast<AER::DataMap<AER::SingleData, json_t, 1> &&>(metadata));
  add_to_python(
      pydata,
      static_cast<AER::DataMap<AER::SingleData, json_t, 2> &&>(metadata));
  add_to_python(
      pydata,
      static_cast<AER::DataMap<AER::SingleData, json_t, 3> &&>(metadata));
  return std::move(pydata);
}

#endif
