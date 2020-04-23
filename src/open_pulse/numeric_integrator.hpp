/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _NUMERIC_INTEGRATOR_HPP
#define _NUMERIC_INTEGRATOR_HPP

#include <cmath>
#include <vector>
#include <complex>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "types.hpp"

namespace py = pybind11;

py::array_t<complex_t> td_ode_rhs(double t,
                                  py::array_t<complex_t> vec,
                                  py::object global_data,
                                  py::object exp,
                                  py::object system,
                                  py::object channels,
                                  py::object reg);

#endif // _NUMERIC_INTEGRATOR_HPP