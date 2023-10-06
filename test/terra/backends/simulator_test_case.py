# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator test case class
"""

from qiskit_aer.backends.backend_utils import available_devices
import ddt
import itertools as it
from qiskit_aer import AerSimulator
from test.terra.common import QiskitAerTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit_aer.backends.backend_utils import cpp_execute_qobj
from qiskit_aer.backends.controller_wrappers import aer_controller_execute


class SimulatorTestCase(QiskitAerTestCase):
    """Simulator test class"""

    BACKEND = AerSimulator
    OPTIONS = {"seed_simulator": 9000}

    def backend(self, **options):
        """Return AerSimulator backend using current class options"""
        sim_options = self.OPTIONS.copy()
        for key, val in options.items():
            if "device" == key and "cuStateVec" in val:
                sim_options["device"] = "GPU"
                sim_options["cuStateVec_enable"] = True
            elif "device" == key and "batch" in val:
                sim_options["device"] = "GPU"
                sim_options["batched_shots_gpu"] = True
            else:
                sim_options[key] = val
            # enable shot_branching is method is tensor_network
            if "method" == key and "tensor_network" in val:
                sim_options["shot_branching_enable"] = True
                sim_options["shot_branching_sampling_enable"] = True
        return self.BACKEND(**sim_options)


def supported_methods(methods, *other_args, product=True):
    """ddt decorator for iterating over supported methods and args"""
    method_args = _method_device(methods)
    if other_args:
        data_args = []
        if product:
            items = list(it.product(*other_args))
        else:
            items = list(zip(*other_args))
        for method, device in method_args:
            for args in items:
                data_args.append((method, device, *args))
    else:
        data_args = method_args

    def decorator(func):
        return ddt.data(*data_args)(ddt.unpack(func))

    return decorator


def supported_devices(func):
    """ddt decorator for iterative over supported devices on current system."""
    devices = AerSimulator().available_devices()
    return ddt.data(*devices)(func)


def _method_device(methods):
    """Return list of (methods, device) supported on current system"""
    available_methods = AerSimulator().available_methods()
    if not methods:
        methods = available_methods
    available_devices = AerSimulator().available_devices()
    # add special test device for cuStateVec if available
    cuStateVec = check_cuStateVec(available_devices)

    gpu_methods = ["statevector", "density_matrix", "unitary", "tensor_network"]
    batchable_methods = ["statevector", "density_matrix"]
    data_args = []
    for method in methods:
        if method in available_methods:
            if method in gpu_methods:
                if "tensor_network" == method:
                    for device in available_devices:
                        if device == "GPU":
                            data_args.append((method, "GPU"))
                else:
                    for device in available_devices:
                        data_args.append((method, device))
                        if device == "GPU":
                            if method in batchable_methods:
                                # add batched optimization test for GPU
                                data_args.append((method, "GPU_batch"))
                    # add test cases for cuStateVec if available using special device = 'GPU_cuStateVec'
                    #'GPU_cuStateVec' is used only inside tests not available in Aer
                    # and this is converted to "device='GPU'" and option "cuStateVec_enalbe = True" is added
                    if cuStateVec and "tensor_network" != method:
                        data_args.append((method, "GPU_cuStateVec"))
            else:
                data_args.append((method, "CPU"))
    return data_args


def check_cuStateVec(devices):
    """Return if the system supports cuStateVec or not"""
    if "GPU" in devices:
        dummy_circ = QuantumCircuit(1)
        dummy_circ.id(0)
        qobj = assemble(
            dummy_circ,
            optimization_level=0,
            shots=1,
            method="statevector",
            device="GPU",
            cuStateVec_enable=True,
        )
        # run dummy circuit to check if Aer is built with cuStateVec
        result = cpp_execute_qobj(aer_controller_execute(), qobj)
        return result.get("success", False)
    else:
        return False
