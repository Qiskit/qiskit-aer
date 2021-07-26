#include <iostream>
#include <string>
#include <sstream>

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

using namespace AER;

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

    py::class_<AER::Controller> aer_controller(m, "AerController");
    aer_controller.def(py::init<>());
    aer_controller.def("execute", [](AER::Controller &controller,
                                     std::vector<AER::Circuit> circuits,
                                     const uint_t shots,
                                     const py::handle &py_config
                                     ) {
      json_t config;
      Parser<py::handle>::get_value(config, "config", py_config);
      controller.set_config(config);
      AER::Noise::NoiseModel noise_model;
      Parser<json_t>::get_value(noise_model, "noise_model", config);
      for (auto &circuit: circuits)
        circuit.shots = shots;
      auto ret = AerToPy::to_python(controller.execute(circuits, noise_model, config));
      return ret;
    });

    py::enum_<Operations::OpType> optype(m, "OpType") ;
    optype.value("gate", Operations::OpType::gate);
    optype.value("measure", Operations::OpType::measure);
    optype.value("reset", Operations::OpType::reset);
    optype.value("bfunc", Operations::OpType::bfunc);
    optype.value("barrier", Operations::OpType::barrier);
    optype.value("matrix", Operations::OpType::matrix);
    optype.value("diagonal_matrix", Operations::OpType::diagonal_matrix);
    optype.value("multiplexer", Operations::OpType::multiplexer);
    optype.value("initialize", Operations::OpType::initialize);
    optype.value("sim_op", Operations::OpType::sim_op);
    optype.value("nop", Operations::OpType::nop);
    optype.value("kraus", Operations::OpType::kraus);
    optype.value("superop", Operations::OpType::superop);
    optype.value("roerror", Operations::OpType::roerror);
    optype.value("noise_switch", Operations::OpType::noise_switch);
    optype.value("save_state", Operations::OpType::save_state);
    optype.value("save_expval", Operations::OpType::save_expval);
    optype.value("save_expval_var", Operations::OpType::save_expval_var);
    optype.value("save_statevec", Operations::OpType::save_statevec);
    optype.value("save_statevec_dict", Operations::OpType::save_statevec_dict);
    optype.value("save_densmat", Operations::OpType::save_densmat);
    optype.value("save_probs", Operations::OpType::save_probs);
    optype.value("save_probs_ket", Operations::OpType::save_probs_ket);
    optype.value("save_amps", Operations::OpType::save_amps);
    optype.value("save_amps_sq", Operations::OpType::save_amps_sq);
    optype.value("save_stabilizer", Operations::OpType::save_stabilizer);
    optype.value("save_unitary", Operations::OpType::save_unitary);
    optype.export_values();

    py::enum_<Operations::RegComparison> reg_comparison(m, "RegComparison");
    reg_comparison.value("equal", Operations::RegComparison::Equal);
    reg_comparison.value("NotEqual", Operations::RegComparison::NotEqual);
    reg_comparison.value("Less", Operations::RegComparison::Less);
    reg_comparison.value("LessEqual", Operations::RegComparison::LessEqual);
    reg_comparison.value("Greater", Operations::RegComparison::Greater);
    reg_comparison.value("GreaterEqual", Operations::RegComparison::GreaterEqual);
    reg_comparison.export_values();

    py::class_<Operations::Op> aer_op(m, "AerOp");
    aer_op.def(py::init(), "constructor");
    aer_op.def("__repr__", [](const Operations::Op &op) { std::stringstream ss; ss << op; return ss.str(); });
    aer_op.def_readwrite("type", &Operations::Op::type);
    aer_op.def_readwrite("name", &Operations::Op::name);
    aer_op.def_readwrite("qubits", &Operations::Op::qubits);
    aer_op.def_readwrite("regs", &Operations::Op::regs);
    aer_op.def_readwrite("params", &Operations::Op::params);
    aer_op.def_readwrite("string_params", &Operations::Op::string_params);
    aer_op.def_readwrite("conditional", &Operations::Op::conditional);
    aer_op.def_readwrite("conditional_reg", &Operations::Op::conditional_reg);
    aer_op.def_readwrite("bfunc", &Operations::Op::bfunc);
    aer_op.def_readwrite("memory", &Operations::Op::memory);
    aer_op.def_readwrite("registers", &Operations::Op::registers);
    aer_op.def_readwrite("mats", &Operations::Op::mats);
    aer_op.def_readwrite("probs", &Operations::Op::probs);
    aer_op.def_readwrite("expval_params", &Operations::Op::expval_params);
    aer_op.def_readwrite("params_expval_pauli", &Operations::Op::params_expval_pauli);
    aer_op.def_readwrite("params_expval_matrix", &Operations::Op::params_expval_matrix);
    aer_op.def_readwrite("mats", &Operations::Op::mats);

    py::class_<Circuit> aer_circuit(m, "AerCircuit");
    aer_circuit.def(py::init<std::vector<Operations::Op>>(), "constructor", py::arg("ops"));
    aer_circuit.def("__repr__", [](const Circuit &circ) {
      std::stringstream ss;
      ss << "Circuit("
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
          << ", global_phase_angle=" << circ.global_phase_angle
          ;
      ss << ")";
      return ss.str();
    });
    aer_circuit.def_readwrite("shots", &Circuit::shots);
    aer_circuit.def_readwrite("seed", &Circuit::seed);
    aer_circuit.def_readwrite("global_phase_angle", &Circuit::global_phase_angle);
}
