#include <pybind11/pybind11.h>

#include "simulators/unitary/unitary_controller.hpp"
#include "simulators/controller_execute.hpp"

PYBIND11_MODULE(unitary_controller_wrapper, m) {
    m.def("unitary_controller_execute", &AER::controller_execute<AER::Simulator::UnitaryController>, "instance of controller_execute for UnitaryController");
}
