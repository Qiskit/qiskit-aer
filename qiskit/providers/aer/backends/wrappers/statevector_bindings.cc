#include <pybind11/pybind11.h>

#include "simulators/statevector/statevector_controller.hpp"
#include "simulators/controller_execute.hpp"
#include "framework/pybind_json.hpp"

PYBIND11_MODULE(statevector_controller_wrapper, m) {
    m.def("statevector_controller_execute_json", &AER::controller_execute_json<AER::Simulator::StatevectorController>, "instance of controller_execute for StatevectorController");
    m.def("statevector_controller_execute", [](const py::object &qobj) -> py::object {
        return AER::controller_execute<AER::Simulator::StatevectorController>(qobj);
    });
}
