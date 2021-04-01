#include <iostream>

#ifdef AER_MPI
#include <mpi.h>
#endif

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/pybind11.h>
DISABLE_WARNING_POP
#if defined(_MSC_VER)
    #undef snprintf
#endif

#include "framework/matrix.hpp"
#include "framework/python_parser.hpp"
#include "framework/pybind_casts.hpp"
#include "framework/types.hpp"
#include "framework/results/pybind_result.hpp"

#include "controllers/aer_controller.hpp"
#include "controllers/qasm_controller.hpp"
#include "controllers/statevector_controller.hpp"
#include "controllers/unitary_controller.hpp"
#include "controllers/controller_execute.hpp"

template<typename T>
class ControllerExecutor {
public:
    ControllerExecutor() = default;
    py::object operator()(const py::handle &qobj) {
#ifdef TEST_JSON // Convert input qobj to json to test standalone data reading
        return AerToPy::to_python(AER::controller_execute<T>(json_t(qobj)));
#else
        return AerToPy::to_python(AER::controller_execute<T>(qobj));
#endif
    }
};

PYBIND11_MODULE(controller_wrappers, m) {

#ifdef AER_MPI
  int prov;
  MPI_Init_thread(nullptr,nullptr,MPI_THREAD_MULTIPLE,&prov);
#endif

    py::class_<ControllerExecutor<AER::Controller> > aer_ctrl (m, "aer_controller_execute");
    aer_ctrl.def(py::init<>());
    aer_ctrl.def("__call__", &ControllerExecutor<AER::Controller>::operator());
    aer_ctrl.def("__reduce__", [aer_ctrl](const ControllerExecutor<AER::Controller> &self) {
        return py::make_tuple(aer_ctrl, py::tuple());
    });

    py::class_<ControllerExecutor<AER::Simulator::QasmController> > qasm_ctrl (m, "qasm_controller_execute");
    qasm_ctrl.def(py::init<>());
    qasm_ctrl.def("__call__", &ControllerExecutor<AER::Simulator::QasmController>::operator());
    qasm_ctrl.def("__reduce__", [qasm_ctrl](const ControllerExecutor<AER::Simulator::QasmController> &self) {
        return py::make_tuple(qasm_ctrl, py::tuple());
    });

    py::class_<ControllerExecutor<AER::Simulator::StatevectorController> > statevec_ctrl (m, "statevector_controller_execute");
    statevec_ctrl.def(py::init<>());
    statevec_ctrl.def("__call__", &ControllerExecutor<AER::Simulator::StatevectorController>::operator());
    statevec_ctrl.def("__reduce__", [statevec_ctrl](const ControllerExecutor<AER::Simulator::StatevectorController> &self) {
        return py::make_tuple(statevec_ctrl, py::tuple());
    });

    py::class_<ControllerExecutor<AER::Simulator::UnitaryController> > unitary_ctrl (m, "unitary_controller_execute");
    unitary_ctrl.def(py::init<>());
    unitary_ctrl.def("__call__", &ControllerExecutor<AER::Simulator::UnitaryController>::operator());
    unitary_ctrl.def("__reduce__", [unitary_ctrl](const ControllerExecutor<AER::Simulator::UnitaryController> &self) {
        return py::make_tuple(unitary_ctrl, py::tuple());
    });

}
