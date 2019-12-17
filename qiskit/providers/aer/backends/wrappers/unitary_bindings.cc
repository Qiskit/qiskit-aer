#include <pybind11/pybind11.h>

#include "simulators/unitary/unitary_controller.hpp"
#include "simulators/controller_execute.hpp"
#include "framework/pybind_json.hpp"

PYBIND11_MODULE(unitary_controller_wrapper, m) {
    m.def("unitary_controller_execute_json", &AER::controller_execute_json<AER::Simulator::UnitaryController>, "instance of controller_execute for UnitaryController");
    m.def("unitary_controller_execute", [](const py::object &qobj) -> py::object {
        return AER::controller_execute<AER::Simulator::UnitaryController>(qobj);
    });
}
