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
#include "framework/circuit.hpp"
#include "framework/operations.hpp"

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

    py::class_<AER::Simulator::QasmController> qasm_controller(m, "QasmController");
    qasm_controller.def(py::init<>());
    qasm_controller.def("execute", [](AER::Simulator::QasmController &controller, std::vector<AER::Circuit> circuits) {
      std::cout << ">> execute" << std::endl;
      AER::Noise::NoiseModel noise_model;
      json_t config;
      controller.set_config(config);
      auto ret = AerToPy::to_python(controller.execute(circuits, noise_model, config));
      std::cout << "<< execute" << std::endl;
      return ret;
    });

    py::class_<AER::cmatrix_t>(m, "cmatrix")//
      .def(py::init<unsigned int, unsigned int>(), "constructor", py::arg("rows"), py::arg("cols"))//
      .def("__repr__", [](const AER::cmatrix_t &mat) { std::stringstream ss; ss << mat; return ss.str(); })//
      .def("get", [](const AER::cmatrix_t &mat, unsigned int row, unsigned int col) { return mat(row, col); })//
      .def("set", [](AER::cmatrix_t &mat, unsigned int row, unsigned int col, std::complex<double> &c) { mat(row, col) = c; })//
      ;

    py::class_<AER::Operations::Op>(m, "AerOp")//
      .def(py::init(), "constructor")//
      .def("__repr__", [](const AER::Operations::Op &op) { std::stringstream ss; ss << op; return ss.str(); });
      ;

    py::class_<AER::Circuit>(m, "AerCircuit")//
      .def(py::init<std::vector<AER::Operations::Op>>(), "constructor", py::arg("ops"))//
      .def("__repr__", [](const AER::Circuit &circ) {
        std::stringstream ss;
        ss << "AerCircuit("
            << "qubit=" << circ.num_qubits
            << ", num_memory=" << circ.num_memory
            << ", num_registers=" << circ.num_registers;
        ss << ", ops={";
        for (auto i = 0; i < circ.ops.size(); ++i)
          if (i == 0)
            ss << circ.ops[i];
          else
            ss << "," << circ.ops[i];

        ss << "}"
            << ", shots=" << circ.shots
            << ", seed=" << circ.seed
            << ", global_phase_angle=" << circ.global_phase_angle;
        ss << ")";
        return ss.str(); })
      .def("set_shots", &AER::Circuit::set_shots, "set_shots", py::arg("shots"))
      .def("set_seed", &AER::Circuit::set_seed, "set_seed", py::arg("seed"))
      .def("set_golbal_phase", &AER::Circuit::set_golbal_phase, "set_golbal_phase", py::arg("golbal_phase"))
    ;

    m.def("make_unitary", [](const AER::reg_t &qubits, const AER::cmatrix_t &mat) {
      return AER::Operations::make_unitary(qubits, mat);
    }, "return unitary op", py::arg("qubits"), py::arg("mat"));

    m.def("make_diagonal", [](const AER::reg_t &qubits, AER::cvector_t &vec) {
      auto copy = vec;
      return AER::Operations::make_diagonal(qubits, std::move(copy));
    }, "return diagonal op", py::arg("qubits"), py::arg("vec"));

    m.def("make_superop", [](const AER::reg_t &qubits, const AER::cmatrix_t &mat) {
      return AER::Operations::make_superop(qubits, mat);
    }, "return superop op", py::arg("qubits"), py::arg("mat"));

    m.def("make_kraus", [](const AER::reg_t &qubits, const std::vector<AER::cmatrix_t> &mats) {
      return AER::Operations::make_kraus(qubits, mats);
    }, "return kraus op", py::arg("qubits"), py::arg("mats"));

    m.def("make_roerror", [](const AER::reg_t &memory, const std::vector<AER::rvector_t> &probs) {
      return AER::Operations::make_roerror(memory, probs);
    }, "return roerror op", py::arg("memory"), py::arg("probs"));

    m.def("make_u1", [](unsigned int qubit, double theta) {
      return AER::Operations::make_u1(qubit, theta);
    }, "return u1 op", py::arg("qubit"), py::arg("theta"));

    m.def("make_u2", [](unsigned int qubit, double phi, double lam) {
      return AER::Operations::make_u2(qubit, phi, lam);
    }, "return u2 op", py::arg("qubit"), py::arg("phi"), py::arg("lam"));

    m.def("make_u3", [](unsigned int qubit, double theta, double phi, double lam) {
      return AER::Operations::make_u3(qubit, theta, phi, lam);
    }, "return u3 op", py::arg("qubit"), py::arg("theta"), py::arg("phi"), py::arg("lam"));

    m.def("make_cx", [](unsigned int control_qubit, unsigned int target_qubit) {
      return AER::Operations::make_cx(control_qubit, target_qubit);
    }, "return cx op", py::arg("control_qubit"), py::arg("target_qubit"));

    m.def("make_cz", [](unsigned int control_qubit, unsigned int target_qubit) {
      return AER::Operations::make_cz(control_qubit, target_qubit);
    }, "return cz op",  py::arg("control_qubit"), py::arg("target_qubit"));

    m.def("make_cy", [](unsigned int control_qubit, unsigned int target_qubit) {
      return AER::Operations::make_cy(control_qubit, target_qubit);
    }, "return cy op", py::arg("control_qubit"), py::arg("target_qubit"));

    m.def("make_cp", [](unsigned int control_qubit, unsigned int target_qubit, double theta) {
      return AER::Operations::make_cp(control_qubit, target_qubit, theta);
    }, "return cp op", py::arg("control_qubit"), py::arg("target_qubit"), py::arg("theta"));

    m.def("make_cu1", [](unsigned int control_qubit, unsigned int target_qubit, double theta) {
      return AER::Operations::make_cu1(control_qubit, target_qubit, theta);
    }, "return cu1 op", py::arg("control_qubit"), py::arg("target_qubit"), py::arg("theta"));

    m.def("make_cu2", [](unsigned int control_qubit, unsigned int target_qubit, double phi, double lam) {
      return AER::Operations::make_cu2(control_qubit, target_qubit, phi, lam);
    }, "return cu2 op", py::arg("control_qubit"), py::arg("target_qubit"), py::arg("phi"), py::arg("lam"));

    m.def("make_cu3", [](unsigned int control_qubit, unsigned int target_qubit, double theta, double phi, double lam) {
      return AER::Operations::make_cu3(control_qubit, target_qubit, theta, phi, lam);
    }, "return cu3 op", py::arg("control_qubit"), py::arg("target_qubit"), py::arg("theta"), py::arg("phi"), py::arg("lam"));

    m.def("make_swap", [](unsigned int qubit1, unsigned int qubit2) {
      return AER::Operations::make_swap(qubit1, qubit2);
    }, "return swap op", py::arg("qubit1"), py::arg("qubit2"));

    m.def("make_id", [](unsigned int qubit) {
      return AER::Operations::make_id(qubit);
    }, "return id op", py::arg("qubit"));

    m.def("make_p", [](unsigned int qubit, double theta) {
      return AER::Operations::make_p(qubit, theta);
    }, "return p op", py::arg("qubit"), py::arg("theta"));

    m.def("make_x", [](unsigned int qubit) {
      return AER::Operations::make_x(qubit);
    }, "return x op", py::arg("qubit"));

    m.def("make_y", [](unsigned int qubit) {
      return AER::Operations::make_y(qubit);
    }, "return y op", py::arg("qubit"));

    m.def("make_z", [](unsigned int qubit) {
      return AER::Operations::make_z(qubit);
    }, "return z op", py::arg("qubit"));

    m.def("make_h", [](unsigned int qubit) {
      return AER::Operations::make_h(qubit);
    }, "return h op", py::arg("qubit"));

    m.def("make_s", [](unsigned int qubit) {
      return AER::Operations::make_s(qubit);
    }, "return s op", py::arg("qubit"));

    m.def("make_sdg", [](unsigned int qubit) {
      return AER::Operations::make_sdg(qubit);
    }, "return sdg op", py::arg("qubit"));

    m.def("make_t", [](unsigned int qubit) {
      return AER::Operations::make_t(qubit);
    }, "return t op", py::arg("qubit"));

    m.def("make_tdg", [](unsigned int qubit) {
      return AER::Operations::make_tdg(qubit);
    }, "return tdg op", py::arg("qubit"));

    m.def("make_r", [](unsigned int qubit, double theta, double phi) {
      return AER::Operations::make_r(qubit, theta, phi);
    }, "return r op", py::arg("qubit"), py::arg("theta"), py::arg("phi"));

    m.def("make_rx", [](unsigned int qubit, double theta) {
      return AER::Operations::make_rx(qubit, theta);
    }, "return rx op", py::arg("qubit"), py::arg("theta"));

    m.def("make_ry", [](unsigned int qubit, double theta) {
      return AER::Operations::make_ry(qubit, theta);
    }, "return ry op", py::arg("qubit"), py::arg("theta"));

    m.def("make_rz", [](unsigned int qubit, double phi) {
      return AER::Operations::make_rz(qubit, phi);
    }, "return rz op", py::arg("qubit"), py::arg("phi"));

    m.def("make_rxx", [](unsigned int qubit1, unsigned int qubit2, double theta) {
      return AER::Operations::make_rxx(qubit1, qubit2, theta);
    }, "return rxx op", py::arg("qubit1"), py::arg("qubit2"), py::arg("theta"));

    m.def("make_ryy", [](unsigned int qubit1, unsigned int qubit2, double theta) {
      return AER::Operations::make_ryy(qubit1, qubit2, theta);
    }, "return ryy op", py::arg("qubit1"), py::arg("qubit2"), py::arg("theta"));

    m.def("make_rzz", [](unsigned int qubit1, unsigned int qubit2, double theta) {
      return AER::Operations::make_rzz(qubit1, qubit2, theta);
    }, "return rzz op", py::arg("qubit1"), py::arg("qubit2"), py::arg("theta"));

    m.def("make_rzx", [](unsigned int qubit1, unsigned int qubit2, double theta) {
      return AER::Operations::make_rzx(qubit1, qubit2, theta);
    }, "return rzx op", py::arg("qubit1"), py::arg("qubit2"), py::arg("theta"));

    m.def("make_ccx", [](unsigned int control_qubit1, unsigned int control_qubit2, unsigned int target_qubit) {
      return AER::Operations::make_ccx(control_qubit1, control_qubit2, target_qubit);
    }, "return ccx op", py::arg("control_qubit1"), py::arg("control_qubit2"), py::arg("target_qubit"));

    m.def("make_cswap", [](unsigned int control_qubit, unsigned int target_qubit1, unsigned int target_qubit2) {
      return AER::Operations::make_cswap(control_qubit, target_qubit1, target_qubit2);
    }, "return cswap op", py::arg("control_qubit"), py::arg("target_qubit1"), py::arg("target_qubit2"));

    m.def("make_mcx", [](const AER::reg_t& control_qubits, unsigned int target_qubit) {
      return AER::Operations::make_mcx(control_qubits, target_qubit);
    }, "return mcx op", py::arg("control_qubits"), py::arg("target_qubit"));

    m.def("make_mcy", [](const AER::reg_t& control_qubits, unsigned int target_qubit) {
      return AER::Operations::make_mcy(control_qubits, target_qubit);
    }, "return mcy op", py::arg("control_qubits"), py::arg("target_qubit"));

    m.def("make_mcz", [](const AER::reg_t& control_qubits, unsigned int target_qubit) {
      return AER::Operations::make_mcz(control_qubits, target_qubit);
    }, "return mcz op", py::arg("control_qubits"), py::arg("target_qubit"));

    m.def("make_mcu1", [](const AER::reg_t& control_qubits, unsigned int target_qubit, double lam) {
      return AER::Operations::make_mcu1(control_qubits, target_qubit, lam);
    }, "return mcu1 op", py::arg("control_qubits"), py::arg("target_qubit"), py::arg("lam"));

    m.def("make_mcu2", [](const AER::reg_t& control_qubits, unsigned int target_qubit, double phi, double lam) {
      return AER::Operations::make_mcu2(control_qubits, target_qubit, phi, lam);
    }, "return mcu2 op", py::arg("control_qubits"), py::arg("target_qubit"), py::arg("phi"), py::arg("lam"));

    m.def("make_mcu3", [](const AER::reg_t& control_qubits, unsigned int target_qubit, double theta, double phi, double lam) {
      return AER::Operations::make_mcu3(control_qubits, target_qubit, theta, phi, lam);
    }, "return mcu3 op", py::arg("control_qubits"), py::arg("target_qubit"), py::arg("theta"), py::arg("phi"), py::arg("lam"));

    m.def("make_mcswap", [](const AER::reg_t& control_qubits, unsigned int target_qubit1, unsigned int target_qubit2) {
      return AER::Operations::make_mcswap(control_qubits, target_qubit1, target_qubit2);
    }, "return mcswap op", py::arg("control_qubits"), py::arg("target_qubit1"), py::arg("target_qubit2"));

    m.def("make_mcphase", [](const AER::reg_t& control_qubits, unsigned int target_qubit) {
      return AER::Operations::make_mcphase(control_qubits, target_qubit);
    }, "return unitary op", py::arg("control_qubits"), py::arg("target_qubit"));

    m.def("make_mcr", [](const AER::reg_t& control_qubits, unsigned int target_qubit, double theta, double phi) {
      return AER::Operations::make_mcr(control_qubits, target_qubit, theta, phi);
    }, "return mcr op", py::arg("control_qubits"), py::arg("target_qubit"), py::arg("theta"), py::arg("phi"));

    m.def("make_mcrx", [](const AER::reg_t& control_qubits, unsigned int target_qubit, double theta) {
      return AER::Operations::make_mcrx(control_qubits, target_qubit, theta);
    }, "return mcrx op", py::arg("control_qubits"), py::arg("target_qubit"), py::arg("theta"));

    m.def("make_mcry", [](const AER::reg_t& control_qubits, unsigned int target_qubit, double theta) {
      return AER::Operations::make_mcry(control_qubits, target_qubit, theta);
    }, "return mcry op", py::arg("control_qubits"), py::arg("target_qubit"), py::arg("theta"));

    m.def("make_sx", [](unsigned int qubit) {
      return AER::Operations::make_sx(qubit);
    }, "return sx op", py::arg("qubit"));

    m.def("make_csx", [](unsigned int control_qubit, unsigned int target_qubit) {
      return AER::Operations::make_csx(control_qubit, target_qubit);
    }, "return csx op", py::arg("control_qubit"), py::arg("target_qubit"));

    m.def("make_mcsx", [](const AER::reg_t& control_qubits, unsigned int target_qubit) {
      return AER::Operations::make_mcsx(control_qubits, target_qubit);
    }, "return mcsx op", py::arg("control_qubits"), py::arg("target_qubit"));

    m.def("make_delay", [](unsigned int qubit) {
      return AER::Operations::make_delay(qubit);
    }, "return delay op", py::arg("qubit"));

    m.def("make_pauli", [](const AER::reg_t& qubits, std::string pauli_string) {
      return AER::Operations::make_pauli(qubits, pauli_string);
    }, "return pauli op", py::arg("qubits"), py::arg("pauli_string"));

    m.def("make_mcx_gray", [](const AER::reg_t& control_qubits, unsigned int target_qubit) {
      return AER::Operations::make_mcx_gray(control_qubits, target_qubit);
    }, "return mcx_gray op", py::arg("control_qubits"), py::arg("target_qubit"));

    m.def("make_reset", [](const AER::reg_t & qubits) {
      return AER::Operations::make_reset(qubits);
    }, "return reset op", py::arg("qubits"));

    m.def("make_multiplexer", [](const AER::reg_t &qubits, const std::vector<AER::cmatrix_t> &mats) {
      return AER::Operations::make_multiplexer(qubits, mats);
    }, "return multiplexer op", py::arg("qubits"), py::arg("mats"));

}
