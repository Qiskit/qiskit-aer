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
#include "ode/sundials_wrapper/sundials_cvode_wrapper.hpp"

namespace{
  using Cvode_Wrapper_t = AER::CvodeWrapper<std::vector<complex_t>>;

  Cvode_Wrapper_t create_sundials_integrator(double t0,
                                             std::vector<complex_t> y0,
                                             py::object global_data,
                                             py::object exp,
                                             py::object system,
                                             py::object channels,
                                             py::object reg){
    auto func = std::bind(td_ode_rhs_vec, std::placeholders::_1, std::placeholders::_2,std::placeholders::_3,
        global_data, exp, system, channels, reg);

    return AER::CvodeWrapper<std::vector<complex_t>>(AER::OdeMethod::ADAMS, func, y0, t0, 1e-6, 1e-8);
  }
}


#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/functional.h>
DISABLE_WARNING_POP

RhsFunctor get_ode_rhs_functor(py::object the_global_data, py::object the_exp,
                               py::object the_system, py::object the_channels, py::object the_reg) {
  return RhsFunctor(the_global_data, the_exp, the_system, the_channels, the_reg);
}

PYBIND11_MODULE(pulse_utils, m) {
    m.doc() = "Utility functions for pulse simulator"; // optional module docstring

    m.def("td_ode_rhs_static", &td_ode_rhs, "Compute rhs for ODE");
    m.def("cy_expect_psi_csr", &expect_psi_csr, "Expected value for a operator");
    m.def("occ_probabilities", &occ_probabilities, "Computes the occupation probabilities of the specifed qubits for the given state");
    m.def("write_shots_memory", &write_shots_memory, "Converts probabilities back into shots");
    m.def("oplist_to_array", &oplist_to_array, "Insert list of complex numbers into numpy complex array");
    m.def("spmv_csr", &spmv_csr, "Sparse matrix, dense vector multiplication.");

    py::class_<Cvode_Wrapper_t>(m, "CvodeWrapper")
      .def("integrate", [](Cvode_Wrapper_t &cvode, double time, py::kwargs kwargs){
        bool step = false;
        if(kwargs && kwargs.contains("step")){
          step = kwargs["step"].cast<bool>();
        }
        return cvode.integrate(time, step);})
      .def("successful", [](const Cvode_Wrapper_t &a){ return true;})
      .def_readwrite("t", &Cvode_Wrapper_t::t_)
      .def_property("_y", [](const Cvode_Wrapper_t &cvode){return py::array(py::cast(cvode.get_solution()));},
          &Cvode_Wrapper_t::set_solution)
      .def_property_readonly("y", [](const Cvode_Wrapper_t &cvode){return py::array(py::cast(cvode.get_solution()));})
      .def("set_intial_value", &Cvode_Wrapper_t::set_intial_value)
      .def("set_tolerances", &Cvode_Wrapper_t::set_tolerances)
      .def("set_step_limits", &Cvode_Wrapper_t::set_step_limits)
      .def("set_maximum_order", &Cvode_Wrapper_t::set_maximum_order)
      .def("set_max_nsteps", &Cvode_Wrapper_t::set_max_nsteps);

    m.def("create_sundials_integrator", &create_sundials_integrator,"");

    py::class_<RhsFunctor>(m, "OdeRhsFunctor")
        .def("__call__", &RhsFunctor::operator());

    m.def("get_ode_rhs_functor", &get_ode_rhs_functor, "Get ode_rhs functor to allow caching of parameters");
}
