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

from .aerbackend import AerBackend
from .aersimulatorerror import AerSimulatorError
from statevector_controller_wrapper import StatevectorControllerWrapper

# Logger
logger = logging.getLogger(__name__)


class StatevectorSimulator(AerBackend):
    """Aer statevector simulator"""

    DEFAULT_CONFIGURATION = {
        'name': 'statevector_simulator',
        'url': 'NA',
        'simulator': True,
        'local': True,
        'description': 'A C++ statevector simulator for qobj files',
        'coupling_map': 'all-to-all',
        "basis_gates": 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz,ccx,swap'
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy(),
                         StatevectorControllerWrapper(), provider=provider)

    def run(self, qobj):
        """Run a qobj on the backend."""
        return super().run(qobj)

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        This forces the simulation to execute with shots=1. 
        """
        if qobj.config.shots != 1:
            logger.warning("Statevector simulator only supports 1 shot. "
                           "Setting shots=1.")
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            # Set shots to 1
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.warning("statevector simulator only supports 1 shot. "
                               "Setting shots=1 for circuit %s.", experiment.header.name)
                experiment.config.shots = 1

