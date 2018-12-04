# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Qiskit Aer Unitary Simulator Backend.
"""

import logging

from .aerbackend import AerBackend
from .aersimulatorerror import AerSimulatorError
from unitary_controller_wrapper import UnitaryControllerWrapper

# Logger
logger = logging.getLogger(__name__)


class UnitarySimulator(AerBackend):
    """Unitary circuit simulator."""

    DEFAULT_CONFIGURATION = {
        'name': 'unitary_simulator',
        'url': 'NA',
        'simulator': True,
        'local': True,
        'description': 'A C++ unitary simulator for qobj files',
        'coupling_map': 'all-to-all',
        "basis_gates": 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,ccx,swap'
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy(),
                         UnitaryControllerWrapper(), provider=provider)

    def run(self, qobj):
        """Run a qobj on the backend."""
        return super().run(qobj)

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        1. No shots
        2. No measurements or reset
        """
        if qobj.config.shots != 1:
            logger.warning("UnitarySimulator only supports 1 shot. "
                           "Setting shots=1.")
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            # Check for measure or reset operations
            for pos, instr in reversed(list(enumerate(experiment.instructions))):
                if instr.name == "measure":
                    raise AerSimulatorError("UnitarySimulator: circuit contains measure.")
                if instr.name == "reset":
                    raise AerSimulatorError("UnitarySimulator: circuit contains reset.")
            # Set shots to 1
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.warning("UnitarySimulator only supports 1 shot. "
                               "Setting shots=1 for circuit %s.", experiment.header.name)
                experiment.config.shots = 1
            # Set memory slots to 0
            if getattr(experiment.config, 'memory_slots', 0) != 0:
                experiment.config.memory_slots = 0
