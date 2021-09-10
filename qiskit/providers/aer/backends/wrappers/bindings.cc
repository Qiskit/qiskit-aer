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
#include "controllers/controller_execute.hpp"

using namespace AER;

template<typename T>
class ControllerExecutor {
public:
    ControllerExecutor() = default;
    py::object operator()(const py::handle &qobj) {
#ifdef TEST_JSON // Convert input qobj to json to test standalone data reading
        return AerToPy::to_python(controller_execute<T>(json_t(qobj)));
#else
        return AerToPy::to_python(controller_execute<T>(qobj));
#endif
    }
};

Operations::Op make_unitary(const reg_t &qubits, const py::array_t<std::complex<double>> &values, const bool carray) {
  size_t mat_len = (1UL << qubits.size());
  auto ptr = values.unchecked<2>();
  cmatrix_t mat(mat_len, mat_len);
    for (auto i = 0; i < mat_len; ++i)
      for (auto j = 0; j < mat_len; ++j)
        mat(i, j) = ptr(i, j);
  return Operations::make_unitary(qubits, mat);
};

PYBIND11_MODULE(controller_wrappers, m) {

#ifdef AER_MPI
  int prov;
  MPI_Init_thread(nullptr,nullptr,MPI_THREAD_MULTIPLE,&prov);
#endif

    py::class_<ControllerExecutor<Controller> > aer_ctrl (m, "aer_controller_execute");
    aer_ctrl.def(py::init<>());
    aer_ctrl.def("__call__", &ControllerExecutor<Controller>::operator());
    aer_ctrl.def("__reduce__", [aer_ctrl](const ControllerExecutor<Controller> &self) {
        return py::make_tuple(aer_ctrl, py::tuple());
    });

    py::class_<Controller> aer_controller(m, "AerController");
    aer_controller.def(py::init<>());
    aer_controller.def("execute", [aer_controller](Controller &controller,
                                     std::vector<Circuit> circuits,
                                     const py::handle &py_config,
                                     const py::handle &py_noise_model
                                     ) {
      auto config = json_t(py_config);
      auto noise_model = Noise::NoiseModel(json_t(py_noise_model));
      return AerToPy::to_python(controller.execute(circuits, noise_model, config));
    }, py::arg("circuits"), py::arg("config") = py::none(), py::arg("noise_model") = py::none());

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
    optype.value("save_mps", Operations::OpType::save_mps);
    optype.value("save_superop", Operations::OpType::save_superop);
    optype.value("save_unitary", Operations::OpType::save_unitary);
    optype.value("set_statevec", Operations::OpType::set_statevec);
    optype.value("set_unitary", Operations::OpType::set_unitary);
    optype.value("set_densmat", Operations::OpType::set_densmat);
    optype.value("set_superop", Operations::OpType::set_superop);
    optype.value("set_stabilizer", Operations::OpType::set_stabilizer);
    optype.value("set_mps", Operations::OpType::set_mps);
    optype.export_values();

    py::enum_<Operations::RegComparison> reg_comparison(m, "RegComparison");
    reg_comparison.value("equal", Operations::RegComparison::Equal);
    reg_comparison.value("NotEqual", Operations::RegComparison::NotEqual);
    reg_comparison.value("Less", Operations::RegComparison::Less);
    reg_comparison.value("LessEqual", Operations::RegComparison::LessEqual);
    reg_comparison.value("Greater", Operations::RegComparison::Greater);
    reg_comparison.value("GreaterEqual", Operations::RegComparison::GreaterEqual);
    reg_comparison.export_values();

    py::enum_<Operations::DataSubType> data_sub_type(m, "DataSubType");
    data_sub_type.value("single", Operations::DataSubType::single);
    data_sub_type.value("c_single", Operations::DataSubType::c_single);
    data_sub_type.value("list", Operations::DataSubType::list);
    data_sub_type.value("c_list", Operations::DataSubType::c_list);
    data_sub_type.value("accum", Operations::DataSubType::accum);
    data_sub_type.value("c_accum", Operations::DataSubType::c_accum);
    data_sub_type.value("average", Operations::DataSubType::average);
    data_sub_type.value("c_average", Operations::DataSubType::c_average);
    data_sub_type.export_values();

    py::class_<Operations::Op> aer_op(m, "AerOp");
    aer_op.def(py::init(), "constructor");
    aer_op.def("__repr__", [](const Operations::Op &op) { std::stringstream ss; ss << op; return ss.str(); });
    aer_op.def_readwrite("type", &Operations::Op::type);
    aer_op.def_readwrite("name", &Operations::Op::name);
    aer_op.def_readwrite("qubits", &Operations::Op::qubits);
    aer_op.def_readwrite("regs", &Operations::Op::regs);
    aer_op.def_readwrite("params", &Operations::Op::params);
    aer_op.def_readwrite("int_params", &Operations::Op::int_params);
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
    aer_op.def_readwrite("save_type", &Operations::Op::save_type);
    aer_op.def_readwrite("mps", &Operations::Op::mps);

    py::class_<Circuit> aer_circuit(m, "AerCircuit");
    aer_circuit.def(py::init<std::string>(), "constructor",
        py::arg("name") = "circuit");
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
    aer_circuit.def_readwrite("num_qubits", &Circuit::num_qubits);
    aer_circuit.def_readwrite("num_memory", &Circuit::num_memory);
    aer_circuit.def_readwrite("seed", &Circuit::seed);
    aer_circuit.def_readwrite("ops", &Circuit::ops);
    aer_circuit.def_readwrite("global_phase_angle", &Circuit::global_phase_angle);
    aer_circuit.def("append_op", [aer_circuit](Circuit &circ, const Operations::Op &op) {
      circ.ops.push_back(op);
    });
    aer_circuit.def("initialize", [aer_circuit](Circuit &circ, bool truncation) {
      circ.set_params(truncation);
    });
    aer_circuit.def("unitary", [aer_circuit](Circuit &circ,
                                             const reg_t &qubits,
                                             const py::array_t<std::complex<double>> &values,
                                             const bool carray) {
      circ.ops.push_back(make_unitary(qubits, values, carray));
    });

    aer_circuit.def("measure", [aer_circuit](Circuit &circ, const reg_t &qubits, const reg_t &memory, const reg_t &clbits) {
      circ.ops.push_back(Operations::make_measure(qubits, memory, clbits));
    });


    m.def("make_unitary", &make_unitary, "return unitary op");
    m.def("make_measure", &Operations::make_measure, "return measure op");

    m.def("make_multiplexer", [](const reg_t &qubits, const std::vector<cmatrix_t> &mats, const std::string label) {
      return Operations::make_multiplexer(qubits, mats, label);
    }, "return multiplexer", py::arg("qubits"), py::arg("mats"), py::arg("label") = "");


    m.def("make_set_clifford", [](const reg_t &qubits, const std::vector<std::string> &stab, const std::vector<std::string> &destab) {
      return Operations::make_set_clifford(qubits, stab, destab);
    }, "return set_clifford", py::arg("qubits"), py::arg("stab"), py::arg("destab"));


}
