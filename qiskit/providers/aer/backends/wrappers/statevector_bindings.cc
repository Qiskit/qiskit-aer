#include <pybind11/pybind11.h>

#include "simulators/statevector/statevector_controller.hpp"
#include "simulators/controller_execute.hpp"

PYBIND11_MODULE(statevector_controller_wrapper, m) {
    m.def("statevector_controller_execute", &AER::controller_execute<AER::Simulator::StatevectorController>, "instance of controller_execute for StatevectorController");
}
