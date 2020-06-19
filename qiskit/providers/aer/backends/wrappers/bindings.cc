/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "framework/results/pybind_result.hpp"
#include "controllers/qasm_controller.hpp"
#include "controllers/statevector_controller.hpp"
#include "controllers/unitary_controller.hpp"
#include "controllers/controller_execute.hpp"

PYBIND11_MODULE(controller_wrappers, m) {

    m.def("qasm_controller_execute_json", &AER::controller_execute_json<AER::Simulator::QasmController>, "instance of controller_execute for QasmController");
    m.def("qasm_controller_execute", [](const py::object &qobj) -> py::object {
        return AerToPy::from_result(AER::controller_execute<AER::Simulator::QasmController>(qobj));
    }, pybind11::return_value_policy::take_ownership);

    m.def("statevector_controller_execute_json", &AER::controller_execute_json<AER::Simulator::StatevectorController>, "instance of controller_execute for StatevectorController");
    m.def("statevector_controller_execute", [](const py::object &qobj) -> py::object {
        return AerToPy::from_result(AER::controller_execute<AER::Simulator::StatevectorController>(qobj));
    }, pybind11::return_value_policy::take_ownership);

    m.def("unitary_controller_execute_json", &AER::controller_execute_json<AER::Simulator::UnitaryController>, "instance of controller_execute for UnitaryController");
    m.def("unitary_controller_execute", [](const py::object &qobj) -> py::object {
        return AerToPy::from_result(AER::controller_execute<AER::Simulator::UnitaryController>(qobj));
    }, pybind11::return_value_policy::take_ownership);

}
