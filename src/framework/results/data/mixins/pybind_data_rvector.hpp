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

#ifndef _aer_framework_result_data_pybind_data_rvector_hpp_
#define _aer_framework_result_data_pybind_data_rvector_hpp_

#include "framework/results/data/mixins/data_rvector.hpp"
#include "framework/results/data/subtypes/pybind_data_map.hpp"

//------------------------------------------------------------------------------
// Aer C++ -> Python Conversion
//------------------------------------------------------------------------------

namespace AerToPy {

// Move an DataRVector container object to a new Python dict
py::object to_python(AER::DataRVector &&data);

// Move an DataRVector container object to an existing new Python dict
void add_to_python(py::dict &pydata, AER::DataRVector &&data);

} //end namespace AerToPy


//============================================================================
// Implementations
//============================================================================

py::object AerToPy::to_python(AER::DataRVector &&data) {
  py::dict pydata;
  AerToPy::add_to_python(pydata, std::move(data));
  return std::move(pydata);
}

void AerToPy::add_to_python(py::dict &pydata, AER::DataRVector &&data) {
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::ListData, std::vector<double>, 1>&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::ListData, std::vector<double>, 2>&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::AccumData, std::vector<double>, 1>&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::AccumData, std::vector<double>, 2>&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::AverageData, std::vector<double>, 1>&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::AverageData, std::vector<double>, 2>&&>(data));
}

#endif
