/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_controller_binding_hpp_
#define _aer_controller_binding_hpp_

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/pybind11.h>
DISABLE_WARNING_POP
#if defined(_MSC_VER)
    #undef snprintf
#endif

#include <vector>

#include "framework/matrix.hpp"
#include "framework/python_parser.hpp"
#include "framework/pybind_casts.hpp"
#include "framework/types.hpp"
#include "framework/results/pybind_result.hpp"

#include "controllers/aer_controller.hpp"

#include "controllers/controller_execute.hpp"

namespace py = pybind11;
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

    py::object execute(std::vector<Circuit> &circuits, Noise::NoiseModel &noise_model, json_t& config) const {
        return AerToPy::to_python(controller_execute<T>(circuits, noise_model, config));
    }
};

template<typename MODULE>
void bind_aer_controller(MODULE m) {
    py::class_<ControllerExecutor<Controller> > aer_ctrl (m, "aer_controller_execute");
    aer_ctrl.def(py::init<>());
    aer_ctrl.def("__call__", &ControllerExecutor<Controller>::operator());
    aer_ctrl.def("__reduce__", [aer_ctrl](const ControllerExecutor<Controller> &self) {
        return py::make_tuple(aer_ctrl, py::tuple());
    });
    aer_ctrl.def("execute", [aer_ctrl](ControllerExecutor<Controller> &self,
                                       std::vector<Circuit> &circuits,
                                       py::object noise_model,
                                       py::object config) {

        Noise::NoiseModel noise_model_native;
        if (noise_model)
          noise_model_native.load_from_json(noise_model);

        json_t config_json;
        if (config)
          config_json = config;

        return self.execute(circuits, noise_model_native, config_json);
    });
}
#endif
