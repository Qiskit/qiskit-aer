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

#include "framework/results/data/pybind_data.hpp"
#include "framework/results/data/pybind_metadata.hpp"
#include "framework/results/legacy/pybind_data.hpp"
#include "framework/results/result.hpp"

//------------------------------------------------------------------------------
// Aer C++ -> Python Conversion
//------------------------------------------------------------------------------

namespace AerToPy {

// Move an ExperimentResult object to a Python dict
template <> py::object to_python(AER::ExperimentResult &&result);

// Move a Result object to a Python dict
template <> py::object to_python(AER::Result &&result);

//============================================================================
// Implementations
//============================================================================


std::string get_status(const AER::ExperimentResult& result){
    switch (result.status) {
        case AER::ExperimentResult::Status::completed:
            return "DONE";
        case AER::ExperimentResult::Status::error:
            return std::string("ERROR: ") + result.message;
        case AER::ExperimentResult::Status::empty:
            return "EMPTY";
        default:
            return "No STATUS info provided";
    }
}

std::string get_status(const AER::Result& result){
    switch (result.status) {
        case AER::Result::Status::completed:
            return "COMPLETED";
        case AER::Result::Status::partial_completed:
            return "PARTIAL COMPLETED";
        case AER::Result::Status::error:
            return std::string("ERROR: ") + result.message;
        case AER::Result::Status::empty:
            return "EMPTY";
        default:
            return "No STATUS info provided";
    }
}

py::object to_python(AER::ExperimentResult &&exp_result){
    static py::object PyExpResult = py::module::import("qiskit.result.models").attr("ExperimentResult");
    static py::object PyQobjExperimentHeader = py::module::import("qiskit.qobj.common").attr("QobjExperimentHeader");
    static py::object PyExpResultData = py::module::import("qiskit.result.models").attr("ExperimentResultData");

    py::dict py_data = AerToPy::to_python(std::move(exp_result.data));
    py::dict legacy_snapshots = AerToPy::from_snapshot(std::move(exp_result.legacy_data));
    if (!legacy_snapshots.empty()) {
        py_data["snapshots"] = std::move(legacy_snapshots);
    }
    py::object py_exp_data = PyExpResultData.attr("from_dict")(py_data);

    py::object py_exp_result = PyExpResult(exp_result.shots,
                                           exp_result.status == AER::ExperimentResult::Status::completed,
                                           py_exp_data);

    py::dict tmp_dict;
    tmp_dict["status"] = get_status(exp_result);
    tmp_dict["seed_simulator"] = exp_result.seed;
    tmp_dict["time_taken"] = exp_result.time_taken;

    if(!exp_result.header.empty()){
        py::object tmp;
        from_json(exp_result.header, tmp);
        py_exp_result.attr("header") = PyQobjExperimentHeader(**tmp);
    }

    tmp_dict["metadata"] = AerToPy::to_python(std::move(exp_result.metadata));
    py_exp_result.attr("_metadata") = tmp_dict;

    return py_exp_result;
}

py::list get_exp_results(AER::Result&& result){
    py::list py_exp_result_list;

    for(AER::ExperimentResult& exp_result : result.results){
        py_exp_result_list.append(to_python(std::move(exp_result)));
    }
    return py_exp_result_list;
}

py::object to_python(AER::Result &&result) {
    static py::object PyResult = py::module::import("qiskit.result").attr("Result");
    static py::object PyQobjHeader = py::module::import("qiskit.qobj.common").attr("QobjHeader");
    py::object py_result = PyResult(result.backend_name, result.backend_version, result.qobj_id, result.job_id,
                                    result.status == AER::Result::Status::completed, get_exp_results(std::move(result)),
                                    result.date, get_status(std::move(result)));

    if(!result.header.empty()){
        py::object tmp;
        from_json(result.header, tmp);
        py_result.attr("header") = PyQobjHeader(**tmp);
    }
    py::dict tmp_dict;
    tmp_dict["metadata"] = AerToPy::to_python(std::move(result.metadata));
    py_result.attr("_metadata") = tmp_dict;
    return py_result;
}
} //end namespace AerToPy

#endif
