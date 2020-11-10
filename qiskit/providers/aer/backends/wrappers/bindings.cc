#include <iostream>
#include "misc/common_macros.hpp"
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(GNUC_AVX2)
#include <cpuid.h>
#endif

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/pybind11.h>
DISABLE_WARNING_POP

#include "framework/matrix.hpp"
#include "framework/types.hpp"
#include "framework/results/pybind_result.hpp"

#include "controllers/qasm_controller.hpp"
#include "controllers/statevector_controller.hpp"
#include "controllers/unitary_controller.hpp"
#include "controllers/controller_execute.hpp"

template<typename T>
class ControllerExecutor {
public:
    ControllerExecutor() = default;
    py::object operator()(const py::object &qobj) { return AerToPy::to_python(AER::controller_execute<T>(qobj)); }
};

PYBIND11_MODULE(controller_wrappers, m) {

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
