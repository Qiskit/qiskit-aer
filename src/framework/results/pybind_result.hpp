/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_result_pybind_result_hpp_
#define _aer_framework_result_pybind_result_hpp_

#include "framework/pybind_basics.hpp"
#include "framework/results/pybind_data.hpp"
#include "framework/results/experiment_result.hpp"
#include "framework/results/result.hpp"

//------------------------------------------------------------------------------
// Aer C++ -> Python Conversion
//------------------------------------------------------------------------------

namespace AerToPy {

// Move an ExperimentData object to a Python dict
template <> py::object to_python(AER::ExperimentData &&result);

// Move an ExperimentResult object to a Python dict
template <> py::object to_python(AER::ExperimentResult &&result);

// Move a Result object to a Python dict
template <> py::object to_python(AER::Result &&result);

} //end namespace AerToPy


//============================================================================
// Implementations
//============================================================================

template <>
py::object AerToPy::to_python(AER::ExperimentData &&datum) {
  return AerToPy::from_data(std::move(datum));
}


template <>
py::object AerToPy::to_python(AER::ExperimentResult &&result) {
  py::dict pyexperiment;

  pyexperiment["shots"] = result.shots;
  pyexperiment["seed_simulator"] = result.seed;

  pyexperiment["data"] = AerToPy::to_python(std::move(result.data));

  pyexperiment["success"] = (result.status == AER::ExperimentResult::Status::completed);
  switch (result.status) {
    case AER::ExperimentResult::Status::completed:
      pyexperiment["status"] = "DONE";
      break;
    case AER::ExperimentResult::Status::error:
      pyexperiment["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::ExperimentResult::Status::empty:
      pyexperiment["status"] = "EMPTY";
  }
  pyexperiment["time_taken"] = result.time_taken;
  if (result.header.empty() == false) {
    py::object tmp;
    from_json(result.header, tmp);
    pyexperiment["header"] = std::move(tmp);
  }
  if (result.metadata.empty() == false) {
    py::object tmp;
    from_json(result.metadata, tmp);
    pyexperiment["metadata"] = std::move(tmp);
  }
  return std::move(pyexperiment);
}


template <>
py::object AerToPy::to_python(AER::Result &&result) {
  py::dict pyresult;
  pyresult["qobj_id"] = result.qobj_id;

  pyresult["backend_name"] = result.backend_name;
  pyresult["backend_version"] = result.backend_version;
  pyresult["date"] = result.date;
  pyresult["job_id"] = result.job_id;

  py::list exp_results;
  for(AER::ExperimentResult& exp : result.results)
    exp_results.append(AerToPy::to_python(std::move(exp)));
  pyresult["results"] = std::move(exp_results);

  // For header and metadata we continue using the json->pyobject casting
  //   bc these are assumed to be small relative to the ExperimentResults
  if (result.header.empty() == false) {
    py::object tmp;
    from_json(result.header, tmp);
    pyresult["header"] = std::move(tmp);
  }
  if (result.metadata.empty() == false) {
    py::object tmp;
    from_json(result.metadata, tmp);
    pyresult["metadata"] = std::move(tmp);
  }
  pyresult["success"] = (result.status == AER::Result::Status::completed);
  switch (result.status) {
    case AER::Result::Status::completed:
      pyresult["status"] = "COMPLETED";
      break;
    case AER::Result::Status::partial_completed:
      pyresult["status"] = "PARTIAL COMPLETED";
      break;
    case AER::Result::Status::error:
      pyresult["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::Result::Status::empty:
      pyresult["status"] = "EMPTY";
  }
  return std::move(pyresult);
}

#endif
