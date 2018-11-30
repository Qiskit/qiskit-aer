# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qiskit Aer qasm simulator backend.
"""

from math import log2
from qiskit._util import local_hardware_info
from qiskit.backends.models import BackendConfiguration

from ..version import VERSION
from .aerbackend import AerBackend
from qasm_controller_wrapper import qasm_controller_execute


class QasmSimulator(AerBackend):
    """Aer quantum circuit simulator"""

    DEFAULT_CONFIGURATION = {
        'backend_name': 'qasm_simulator',
        'backend_version': VERSION,
        'n_qubits': int(log2(local_hardware_info()['memory'] * (1024 ** 3) / 16)),
        'url': 'TODO',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 100000,
        'description': 'A C++ simulator for QASM experiments with noise',
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z',
                        'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap', 'snapshot'],
        'gates': [{'name': 'TODO', 'parameters': [], 'qasm_def': 'TODO'}]
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(qasm_controller_execute,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def _validate(self, qobj):
        # TODO
        return
