#include <iostream>

#include <pybind11/pybind11.h>

#include "framework/matrix.hpp"
#include "framework/types.hpp"
#include "framework/pybind_json.hpp"

#include "controllers/qasm_controller.hpp"
#include "controllers/statevector_controller.hpp"
#include "controllers/unitary_controller.hpp"
#include "controllers/controller_execute.hpp"

PYBIND11_MODULE(controller_wrappers, m) {
    py::class_<matrix<std::complex<double>>>(m, "Matrix_zd", py::buffer_protocol())
        .def_buffer([](matrix<std::complex<double>> &mat) -> py::buffer_info {
            return py::buffer_info(
                mat.GetMat(),
                sizeof(std::complex<double>),
                py::format_descriptor<std::complex<double>>::format(),
                2,
                { mat.GetRows(), mat.GetColumns() },
                { sizeof(std::complex<double>),
                  sizeof(std::complex<double>) * mat.GetRows() }
        );
    });
    py::class_<matrix<double>>(m, "Matrix_d", py::buffer_protocol())
        .def_buffer([](matrix<double> &mat) -> py::buffer_info {
            return py::buffer_info(
                mat.GetMat(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                2,
                { mat.GetRows(), mat.GetColumns() },
                { sizeof(double),
                  sizeof(double) * mat.GetRows() }
        );
    });


    m.def("qasm_controller_execute_json", &AER::controller_execute_json<AER::Simulator::QasmController>, "instance of controller_execute for QasmController");
    m.def("qasm_controller_execute", [](const py::object &qobj) -> py::object {
        return AerToPy::from_result(AER::controller_execute<AER::Simulator::QasmController>(qobj));
    });

    m.def("statevector_controller_execute_json", &AER::controller_execute_json<AER::Simulator::StatevectorController>, "instance of controller_execute for StatevectorController");
    m.def("statevector_controller_execute", [](const py::object &qobj) -> py::object {
        return AerToPy::from_result(AER::controller_execute<AER::Simulator::StatevectorController>(qobj));
    });

    m.def("unitary_controller_execute_json", &AER::controller_execute_json<AER::Simulator::UnitaryController>, "instance of controller_execute for UnitaryController");
    m.def("unitary_controller_execute", [](const py::object &qobj) -> py::object {
        return AerToPy::from_result(AER::controller_execute<AER::Simulator::UnitaryController>(qobj));
    });
}
