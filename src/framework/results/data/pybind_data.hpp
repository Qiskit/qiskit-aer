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

#ifndef _aer_framework_result_data_pybind_data_hpp_
#define _aer_framework_result_data_pybind_data_hpp_

#include "framework/results/data/data.hpp"
#include "framework/results/data/mixins/pybind_data_creg.hpp"
#include "framework/results/data/mixins/pybind_data_cmatrix.hpp"
#include "framework/results/data/mixins/pybind_data_cvector.hpp"

namespace AerToPy {

// Move an ExperimentResult data object to a Python dict
template <> py::object to_python(AER::Data &&data);

} //end namespace AerToPy


//============================================================================
// Implementations
//============================================================================

template <>
py::object AerToPy::to_python(AER::Data &&data) {
  py::dict pydata;
  AerToPy::add_to_python(pydata, static_cast<AER::DataCVector&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataCMatrix&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataCReg&&>(data));
  return std::move(pydata);
}

#endif
