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
#include "controllers/controller_execute.hpp"

#include "controllers/state_controller.hpp"

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

    py::class_<AER::AerState> aer_state(m, "AerStateWrapper");

    aer_state.def(py::init<>(), "constructor");

    aer_state.def("__repr__", [](const AER::AerState &state) {
      std::stringstream ss;
      ss << "AerStateWrapper("
          << "initialized=" << state.is_initialized()
          << ", num_of_qubits=" << state.num_of_qubits();
      ss << ")";
      return ss.str();
    });

    aer_state.def("configure",  &AER::AerState::configure);
    aer_state.def("allocate_qubits",  &AER::AerState::allocate_qubits);
    aer_state.def("reallocate_qubits",  &AER::AerState::reallocate_qubits);
    aer_state.def("clear",  &AER::AerState::clear);
    aer_state.def("num_of_qubits",  &AER::AerState::num_of_qubits);

    aer_state.def("initialize",  &AER::AerState::initialize);
    aer_state.def("initialize_statevector", [aer_state](AER::AerState &state,
                                                        int num_of_qubits,
                                                        py::array_t<std::complex<double>> &values,
                                                        bool copy) {
      std::complex<double>* data_ptr = reinterpret_cast<std::complex<double>*>(values.mutable_data(0));
      state.configure("method", "statevector");
      state.initialize_statevector(num_of_qubits, data_ptr, copy);
      return true;
    });

    aer_state.def("move_to_buffer",  [aer_state](AER::AerState &state) {
      return state.move_to_vector().move_to_buffer();
    });

    aer_state.def("move_to_ndarray", [aer_state](AER::AerState &state) {
      auto vec = state.move_to_vector();

      std::complex<double>* data_ptr = vec.data();
      auto ret = AerToPy::to_numpy(std::move(vec));
      return ret;
    });

    aer_state.def("flush",  &AER::AerState::flush_ops);

    aer_state.def("last_result",  [aer_state](AER::AerState &state) {
      return AerToPy::to_python(state.last_result().to_json());
    });


    aer_state.def("apply_initialize",  &AER::AerState::apply_initialize);
    aer_state.def("apply_global_phase",  &AER::AerState::apply_global_phase);
    aer_state.def("apply_unitary", [aer_state](AER::AerState &state,
                                                   const reg_t &qubits,
                                                   const py::array_t<std::complex<double>> &values) {
      size_t mat_len = (1UL << qubits.size());
      auto ptr = values.unchecked<2>();
      AER::cmatrix_t mat(mat_len, mat_len);
      for (auto i = 0; i < mat_len; ++i)
        for (auto j = 0; j < mat_len; ++j)
          mat(i, j) = ptr(i, j);
      state.apply_unitary(qubits, mat);
    });

    aer_state.def("apply_multiplexer", [aer_state](AER::AerState &state,
                                                       const reg_t &control_qubits,
                                                       const reg_t &target_qubits,
                                                       const py::array_t<std::complex<double>> &values) {
      size_t mat_len = (1UL << target_qubits.size());
      size_t mat_size = (1UL << control_qubits.size());
      auto ptr = values.unchecked<3>();
      std::vector<AER::cmatrix_t> mats;
      for (auto i = 0; i < mat_size; ++i) {
        AER::cmatrix_t mat(mat_len, mat_len);
        for (auto j = 0; j < mat_len; ++j)
          for (auto k = 0; k < mat_len; ++k)
            mat(j, k) = ptr(i, j, k);
        mats.push_back(mat);
      }
      state.apply_multiplexer(control_qubits, target_qubits, mats);
    });

    aer_state.def("apply_diagonal",  &AER::AerState::apply_diagonal_matrix);
    aer_state.def("apply_mcx",  &AER::AerState::apply_mcx);
    aer_state.def("apply_mcy",  &AER::AerState::apply_mcy);
    aer_state.def("apply_mcz",  &AER::AerState::apply_mcz);
    aer_state.def("apply_mcphase",  &AER::AerState::apply_mcphase);
    aer_state.def("apply_mcu",  &AER::AerState::apply_mcu);
    aer_state.def("apply_mcswap",  &AER::AerState::apply_mcswap);
    aer_state.def("apply_measure",  &AER::AerState::apply_measure);
    aer_state.def("apply_reset",  &AER::AerState::apply_reset);
    aer_state.def("probability",  &AER::AerState::probability);
    aer_state.def("probabilities", [aer_state](AER::AerState &state,
                                                   const reg_t qubits) {
      if (qubits.empty())
        return state.probabilities();
      else
        return state.probabilities(qubits);
    }, py::arg("qubits") = reg_t());
    aer_state.def("sample_memory",  &AER::AerState::sample_memory);
    aer_state.def("sample_counts",  &AER::AerState::sample_counts);

}
