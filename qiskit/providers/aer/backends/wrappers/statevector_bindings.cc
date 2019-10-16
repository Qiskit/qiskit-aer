#include <pybind11/pybind11.h>

#include "simulators/statevector/statevector_controller.hpp"
#include "simulators/controller_execute.hpp"

#include "nlohmann_pybind.hpp"

PYBIND11_MODULE(statevector_controller_wrapper, m) {
    m.def("statevector_controller_execute", &AER::controller_execute<AER::Simulator::StatevectorController>, "instance of controller_execute for StatevectorController");
    m.def("statevector_controller_execute_new", [](const py::object &qobj) -> py::object {
        json_t qobj_js = qobj;
        return AER::controller_execute_new<AER::Simulator::StatevectorController>(qobj_js);
    });
}
