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

#ifndef _aer_framework_result_data_pybind_data_mps_hpp_
#define _aer_framework_result_data_pybind_data_mps_hpp_

#include "framework/results/data/mixins/data_mps.hpp"
#include "framework/results/data/subtypes/pybind_data_map.hpp"

//------------------------------------------------------------------------------
// Aer C++ -> Python Conversion
//------------------------------------------------------------------------------

namespace AerToPy {

// Move mps_container_t to python object
template <> py::object to_python(AER::mps_container_t &&mps);

// Move an DataMPS container object to a new Python dict
py::object to_python(AER::DataMPS &&data);

// Move an DataMPS container object to an existing new Python dict
void add_to_python(py::dict &pydata, AER::DataMPS &&data);

} //end namespace AerToPy


//============================================================================
// Implementations
//============================================================================

template <> py::object AerToPy::to_python(AER::mps_container_t &&data) {
  py::list mats;
  for (auto& pair: data.first) {
    mats.append(py::make_tuple(AerToPy::to_python(std::move(pair.first)),
                               AerToPy::to_python(std::move(pair.second))));
  }
  py::list vecs;
  for (auto&& vec: data.second) {
    vecs.append(AerToPy::to_python(std::move(vec)));
  }
  return py::make_tuple(std::move(mats), std::move(vecs));
}

py::object AerToPy::to_python(AER::DataMPS &&data) {
  py::dict pydata;
  AerToPy::add_to_python(pydata, std::move(data));
  return std::move(pydata);
}

void AerToPy::add_to_python(py::dict &pydata, AER::DataMPS &&data) {
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::SingleData, AER::mps_container_t, 1>&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::SingleData, AER::mps_container_t, 2>&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::ListData, AER::mps_container_t, 1>&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::ListData, AER::mps_container_t, 2>&&>(data));
}

#endif
