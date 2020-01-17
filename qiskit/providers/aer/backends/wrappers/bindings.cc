#include <iostream>

#include <pybind11/pybind11.h>

#include "framework/pybind_json.hpp"

#include "simulators/qasm/qasm_controller.hpp"
#include "simulators/statevector/statevector_controller.hpp"
#include "simulators/unitary/unitary_controller.hpp"

#include "simulators/controller_execute.hpp"
#include "framework/results/result.hpp"

py::object from_exp_data(const AER::ExperimentData &result) {
  py::dict pyresult;

  // Measure data
  if (result.return_counts_ && ! result.counts_.empty()) 
    pyresult["counts"] = result.counts_;
  if (result.return_memory_ && ! result.memory_.empty())
    pyresult["memory"] = result.memory_;
  if (result.return_register_ && ! result.register_.empty())
    pyresult["register"] = result.register_;

  // Add additional data
  for (const auto &pair : result.additional_json_data_) {
    py::object tmp;
    from_json(pair.second, tmp);
    pyresult[pair.first.data()] = tmp;
  }
  for (const auto &pair : result.additional_cvector_data_) {
    py::object tmp;
    from_json(pair.second, tmp);
    pyresult[pair.first.data()] = tmp;
  }
  for (const auto &pair : result.additional_cmatrix_data_) {
    py::object tmp;
    from_json(pair.second, tmp);
    pyresult[pair.first.data()] = tmp;
  }

  // Snapshot data
  if (result.return_snapshots_) {
    py::dict snapshots;
    // Average snapshots
    for (const auto &pair : result.average_json_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.average_complex_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.average_cvector_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.average_cmatrix_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.average_cmap_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.average_rmap_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    // Singleshot snapshot data
    // Note these will override the average snapshots
    // if they share the same type string
    for (const auto &pair : result.pershot_json_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.pershot_complex_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.pershot_cvector_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.pershot_cmatrix_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.pershot_cmap_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    for (auto &pair : result.pershot_rmap_snapshots_) {
      py::object tmp;
      from_json(pair.second, tmp);
      snapshots[pair.first.data()] = tmp;
    }
    pyresult["snapshots"] = snapshots;
  }

  return pyresult;
}

py::object from_exp_result(const AER::ExperimentResult &result) {
  py::dict pyresult;
  
  pyresult["shots"] = result.shots;
  pyresult["seed_simulator"] = result.seed;

  pyresult["data"] = from_exp_data(result.data);

  pyresult["success"] = (result.status == AER::ExperimentResult::Status::completed);
  switch (result.status) {
    case AER::ExperimentResult::Status::completed:
      pyresult["status"] = std::string("DONE");
      break;
    case AER::ExperimentResult::Status::error:
      pyresult["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::ExperimentResult::Status::empty:
      pyresult["status"] = std::string("EMPTY");
  }
  pyresult["time_taken"] = result.time_taken;
  if (result.header.empty() == false) {
    py::object tmp;
    from_json(result.header, tmp);
    pyresult["header"] = tmp;
  }
  if (result.metadata.empty() == false) {
    py::object tmp;
    from_json(result.metadata, tmp);
    pyresult["metadata"] = tmp;
  }
  return pyresult;

}
 
py::object from_result(const AER::Result &result) {
  py::dict pyresult;
  pyresult["qobj_id"] = result.qobj_id;
  
  pyresult["backend_name"] = result.backend_name;
  pyresult["backend_version"] = result.backend_version;
  pyresult["date"] = result.date;
  pyresult["job_id"] = result.job_id;

  py::list exp_results;
  for( const AER::ExperimentResult& exp : result.results)
    exp_results.append(from_exp_result(exp));
  pyresult["results"] = exp_results;
 
  // For header and metadata we continue using the json->pyobject casting
  //   bc these are assumed to be small relative to the ExperimentResults
  if (result.header.empty() == false) {
    py::object tmp;
    from_json(result.header, tmp);
    pyresult["header"] = tmp;
  }
  if (result.metadata.empty() == false) {
    py::object tmp;
    from_json(result.metadata, tmp);
    pyresult["metadata"] = tmp;
  }
  pyresult["success"] = (result.status == AER::Result::Status::completed);
  switch (result.status) {
    case AER::Result::Status::completed:
      pyresult["status"] = std::string("COMPLETED");
      break;
    case AER::Result::Status::partial_completed:
      pyresult["status"] = std::string("PARTIAL COMPLETED");
      break;
    case AER::Result::Status::error:
      pyresult["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::Result::Status::empty:
      pyresult["status"] = std::string("EMPTY");
  }
  return pyresult;

}

PYBIND11_MODULE(controller_wrappers, m) {
    /*py::class_<AER::ExperimentData>(m, "AerExperimentData")
        .def("json", &AER::ExperimentData::json)
    ;
    py::class_<AER::ExperimentResult>(m, "AerExperimentResult")
        .def_readonly("data", &AER::ExperimentResult::data)
    ;
    py::class_<AER::Result>(m, "AerResult")
        .def(py::init<const size_t &>())
        .def_readonly("backend_name", &AER::Result::backend_name)
        .def_readonly("results", &AER::Result::results)
    ;*/

    m.def("qasm_controller_execute_json", &AER::controller_execute_json<AER::Simulator::QasmController>, "instance of controller_execute for QasmController");
    m.def("qasm_controller_execute", [](const py::object &qobj) -> py::object {
        return AER::controller_execute<AER::Simulator::QasmController>(qobj);
    });

    m.def("statevector_controller_execute_json", &AER::controller_execute_json<AER::Simulator::StatevectorController>, "instance of controller_execute for StatevectorController");
    m.def("statevector_controller_execute", [](const py::object &qobj) -> py::object {
        return AER::controller_execute<AER::Simulator::StatevectorController>(qobj);
    });

    m.def("unitary_controller_execute_json", &AER::controller_execute_json<AER::Simulator::UnitaryController>, "instance of controller_execute for UnitaryController");
    m.def("unitary_controller_execute", [](const py::object &qobj) -> py::object {
        return AER::controller_execute<AER::Simulator::UnitaryController>(qobj);
    });
 
}
