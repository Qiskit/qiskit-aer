/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "numeric_integrator.hpp"
#include "pulse_utils.hpp"

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/functional.h>
DISABLE_WARNING_POP

RhsFunctor get_ode_rhs_functor(py::object the_global_data, py::object the_exp,
                               py::object the_system, py::object the_channels, py::object the_reg) {
  return RhsFunctor(the_global_data, the_exp, the_system, the_channels, the_reg);
}


class OccProbabilitiesFunctor {
public:
    OccProbabilitiesFunctor() = default;
    py::array_t<double> operator()(py::array_t<int> qubits,
                                   py::array_t<complex_t> state,
                                   py::list meas_ops) { return occ_probabilities(qubits, state, meas_ops); }
};

PYBIND11_MODULE(pulse_utils, m) {
    m.doc() = "Utility functions for pulse simulator"; // optional module docstring

    m.def("td_ode_rhs_static", &td_ode_rhs, "Compute rhs for ODE");
    m.def("cy_expect_psi_csr", &expect_psi_csr, "Expected value for a operator");
    m.def("cy_expect_psi", &expect_psi, "Expected value for a operator");
    m.def("occ_probabilities", &occ_probabilities, "Computes the occupation probabilities of the specifed qubits for the given state");
    m.def("write_shots_memory", &write_shots_memory, "Converts probabilities back into shots");
    m.def("oplist_to_array", &oplist_to_array, "Insert list of complex numbers into numpy complex array");
    m.def("spmv_csr", &spmv_csr, "Sparse matrix, dense vector multiplication.");
    m.def("spmv", &spmv, "Matrix vector multiplication.");

    py::class_<RhsFunctor> ode_rhs_func(m, "OdeRhsFunctor");
    ode_rhs_func.def("__call__", &RhsFunctor::operator());
    ode_rhs_func.def("__reduce__", [ode_rhs_func](const RhsFunctor& self) { return py::make_tuple(ode_rhs_func, py::tuple());});

    m.def("get_ode_rhs_functor", &get_ode_rhs_functor, "Get ode_rhs functor to allow caching of parameters");
}
