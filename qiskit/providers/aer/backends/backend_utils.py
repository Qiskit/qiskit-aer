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

# pylint: disable=invalid-name
"""
Qiskit Aer simulator backend utils
"""
import os
from math import log2
from qiskit.util import local_hardware_info
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import assemble

# Available system memory
SYSTEM_MEMORY_GB = local_hardware_info()['memory']

# Max number of qubits for complex double statevector
# given available system memory
MAX_QUBITS_STATEVECTOR = int(log2(SYSTEM_MEMORY_GB * (1024**3) / 16))

# Location where we put external libraries that will be
# loaded at runtime by the simulator extension
LIBRARY_DIR = os.path.dirname(__file__)


def cpp_execute(controller, qobj):
    """Execute qobj on C++ controller wrapper"""

    # Location where we put external libraries that will be
    # loaded at runtime by the simulator extension
    qobj.config.library_dir = LIBRARY_DIR

    return controller(qobj)


def available_methods(controller, methods):
    """Check available simulation methods by running a dummy circuit."""
    # Test methods are available using the controller
    dummy_circ = QuantumCircuit(1)
    dummy_circ.i(0)

    valid_methods = []
    for method in methods:
        qobj = assemble(dummy_circ,
                        optimization_level=0,
                        shots=1,
                        method=method)
        result = cpp_execute(controller, qobj)
        if result.get('success', False):
            valid_methods.append(method)
    return valid_methods


def available_devices(controller, devices):
    """Check available simulation devices by running a dummy circuit."""
    # Test methods are available using the controller
    dummy_circ = QuantumCircuit(1)
    dummy_circ.i(0)

    valid_devices = []
    for device in devices:
        qobj = assemble(dummy_circ,
                        optimization_level=0,
                        shots=1,
                        method="statevector",
                        device=device)
        result = cpp_execute(controller, qobj)
        if result.get('success', False):
            valid_devices.append(device)
    return valid_devices
