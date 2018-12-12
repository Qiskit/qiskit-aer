# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Qiskit Aer statevector simulator backend.
"""

import logging
from math import log2
from qiskit._util import local_hardware_info
from qiskit.backends.models import BackendConfiguration

from ..version import __version__
from .aerbackend import AerBackend
from statevector_controller_wrapper import statevector_controller_execute

# Logger
logger = logging.getLogger(__name__)


class StatevectorSimulator(AerBackend):
    """Aer statevector simulator"""

    DEFAULT_CONFIGURATION = {
        'backend_name': 'statevector_simulator',
        'backend_version': __version__,
        'n_qubits': int(log2(local_hardware_info()['memory'] * (1024 ** 3) / 16)),
        'url': 'TODO',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 1,
        'description': 'A C++ statevector simulator for QASM experiments',
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z',
                        'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap', 'snapshot'],
        'gates': [{'name': 'TODO', 'parameters': [], 'qasm_def': 'TODO'}]
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(statevector_controller_execute,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def run(self, qobj, backend_options=None):
        """Run a qobj on the backend."""
        return super().run(qobj, backend_options=backend_options)

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        This forces the simulation to execute with shots=1.
        """
        if qobj.config.shots != 1:
            logger.info("Statevector simulator only supports 1 shot. "
                        "Setting shots=1.")
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            # Set shots to 1
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info("statevector simulator only supports 1 shot. "
                            "Setting shots=1 for circuit %s.", experiment.header.name)
                experiment.config.shots = 1
