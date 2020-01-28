#include <iostream>

#include <pybind11/pybind11.h>

#include "framework/matrix.hpp"
#include "framework/types.hpp"
#include "framework/pybind_json.hpp"

#include "simulators/qasm/qasm_controller.hpp"
#include "simulators/statevector/statevector_controller.hpp"
#include "simulators/unitary/unitary_controller.hpp"
#include "simulators/controller_execute.hpp"

PYBIND11_MODULE(controller_wrappers, m) {
    m.def("qasm_controller_execute_json", &AER::controller_execute_json<AER::Simulator::QasmController>, "instance of controller_execute for QasmController");
    m.def("qasm_controller_execute", [](const py::object &qobj) -> py::object {
        return AERPy::from_result(AER::controller_execute<AER::Simulator::QasmController>(qobj));
    });

    m.def("statevector_controller_execute_json", &AER::controller_execute_json<AER::Simulator::StatevectorController>, "instance of controller_execute for StatevectorController");
    m.def("statevector_controller_execute", [](const py::object &qobj) -> py::object {
        return AERPy::from_result(AER::controller_execute<AER::Simulator::StatevectorController>(qobj));
    });

    m.def("unitary_controller_execute_json", &AER::controller_execute_json<AER::Simulator::UnitaryController>, "instance of controller_execute for UnitaryController");
    m.def("unitary_controller_execute", [](const py::object &qobj) -> py::object {
        return AERPy::from_result(AER::controller_execute<AER::Simulator::UnitaryController>(qobj));
    });
}
