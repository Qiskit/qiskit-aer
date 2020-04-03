#include <iostream>

#include <pybind11/pybind11.h>

#include "framework/matrix.hpp"
#include "framework/types.hpp"
#include "framework/pybind_json.hpp"
#include "framework/utils.hpp"

#include "controllers/qasm_controller.hpp"
#include "controllers/statevector_controller.hpp"
#include "controllers/unitary_controller.hpp"
#include "controllers/controller_execute.hpp"

PYBIND11_MODULE(controller_wrappers, m) {
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

/*
    py::class_<Matrix_f>(m, "Matrix", py::buffer_protocol())
        .def_buffer([](matrix<float> &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                               
                sizeof(float),                          
                py::format_descriptor<float>::format(), 
                2,                                      
                { m.getRows(), m.getColumns() },        
                { sizeof(float) * m.getColumns(),       
                  sizeof(float) }
            );
    });
*/
    m.def("eig_psd_tester", []() {
        matrix<std::complex<float>> m(2, 2);
        m(0,0) = std::complex<float>{0.0,0.0};
        m(1,0) = std::complex<float>{1.0,0.0};
        m(0,1) = std::complex<float>{1.0,0.0};
        m(1,1) = std::complex<float>{0.0,0.0};
        std::vector<float> evals;
        // copy constructor
        matrix<std::complex<float>> evecs = m;

        std::cout << "original: " << std::endl;
        std::cout << m << std::endl;
        eig_psd(m, evals, evecs);
        std::cout << "eigenvals: " << std::endl;
        std::cout << evals << std::endl;
        std::cout << "eigenvectors: " << std::endl;
        std::cout << evecs << std::endl;
        
        std::cout << std::endl << 
                     "original: " << std::endl;
        std::cout << m << std::endl;
        matrix<std::complex<float>> value(m.size());
        for (size_t j=0; j < evals.size(); j++) {
            value += evals[j] * AER::Utils::projector(evecs.row_index(j));
        }
        std::cout << "test: " << std::endl;
        std::cout << value << std::endl;
    });
    m.def("eig_psd_tester1", []() {
        matrix<std::complex<float>> m(4, 4);
        m(0,0) = std::complex<float>{2.0,0.0};
        m(1,0) = std::complex<float>{2.0,0.0};
        m(2,0) = std::complex<float>{2.0,0.0};
        m(3,0) = std::complex<float>{2.0,0.0};
        m(0,1) = std::complex<float>{2.0,0.0};
        m(1,1) = std::complex<float>{2.0,0.0};
        m(2,1) = std::complex<float>{2.0,0.0};
        m(3,1) = std::complex<float>{2.0,0.0};
        m(0,2) = std::complex<float>{2.0,0.0};
        m(1,2) = std::complex<float>{2.0,0.0};
        m(2,2) = std::complex<float>{2.0,0.0};
        m(3,2) = std::complex<float>{2.0,0.0};
        m(0,3) = std::complex<float>{2.0,0.0};
        m(1,3) = std::complex<float>{2.0,0.0};
        m(2,3) = std::complex<float>{2.0,0.0};
        m(3,3) = std::complex<float>{2.0,0.0};
        m(0,3) = std::complex<float>{2.0,0.0};
        std::vector<float> evals;
        matrix<std::complex<float>> evecs(4,4);

        std::cout << "original: " << std::endl;
        std::cout << m << std::endl;
        eig_psd(m, evals, evecs);
        std::cout << "eigenvals: " << std::endl;
        std::cout << evals << std::endl;
        std::cout << "eigenvectors: " << std::endl;
        std::cout << evecs << std::endl;
        std::cout << "original: " << std::endl;
        std::cout << m << std::endl;

        std::cout << "test: " << std::endl;
        matrix<std::complex<float>> value(m.size());
        for (size_t j=0; j < evals.size(); j++) {
            value += evals[j] * AER::Utils::projector(evecs.row_index(j));
        }
        std::cout << value << std::endl;
    });
}
