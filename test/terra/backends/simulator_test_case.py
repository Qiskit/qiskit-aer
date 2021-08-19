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

from qiskit.providers.aer.backends.backend_utils import available_devices
import ddt
import itertools as it
from qiskit.providers.aer import AerSimulator
from test.terra.common import QiskitAerTestCase


class SimulatorTestCase(QiskitAerTestCase):
    """Simulator test class"""

    BACKEND = AerSimulator
    OPTIONS = {"seed_simulator": 9000}

    def backend(self, **options):
        """Return AerSimulator backend using current class options"""
        sim_options = self.OPTIONS.copy()
        for key, val in options.items():
            sim_options[key] = val
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
    if not methods:
        methods = AerSimulator().available_methods()
    available_devices = AerSimulator().available_devices()
    gpu_methods = ['statevector', 'density_matrix', 'unitary']
    data_args = []
    for method in methods:
        if method in gpu_methods:
            for device in available_devices:
                data_args.append((method, device))
        else:
            data_args.append((method, 'CPU'))
    return data_args


