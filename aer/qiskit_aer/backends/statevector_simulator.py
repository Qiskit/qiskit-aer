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

from qiskit.qobj import QobjInstruction
from .qasm_simulator import QasmSimulator
from .aersimulatorerror import AerSimulatorError

# Logger
logger = logging.getLogger(__name__)


class StatevectorSimulator(QasmSimulator):
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
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy(), provider=provider)

    def _run_job(self, job_id, qobj):
        # Add final snapshots to circuits
        final_state_key = '__AER_FINAL_STATE__'
        for experiment in qobj.experiments:
            experiment.instructions.append(
                QobjInstruction(name='snapshot', type='state', label=final_state_key)
            )
        # Get result from parent class _run_job method
        result = super()._run_job(job_id, qobj)
        # Remove added snapshot from qobj
        for experiment in qobj.experiments:
            del experiment.instructions[-1]
        # Extract final state snapshot and move to 'statevector' data field
        for experiment_result in result.results.values():
            snapshots = experiment_result.snapshots

            # Pop off final snapshot added above
            if 'state' in snapshots:
                final_state = snapshots['state'].pop(final_state_key, None)
                final_state = final_state[0]
            # Add final state to results data
            experiment_result.data['statevector'] = final_state
            # Remove snapshot dict if empty
            if 'state' in snapshots:
                if snapshots['state'] == {}:
                    snapshots.pop('state', None)
            if snapshots == {}:
                experiment_result.data.pop('snapshots', None)
        return result

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        1. No shots
        2. No measurements in the middle
        """
        if qobj.config.shots != 1:
            logger.warning("statevector simulator only supports 1 shot. "
                           "Setting shots=1.")
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            # Set shots to 1
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.warning("statevector simulator only supports 1 shot. "
                               "Setting shots=1 for circuit %s.", experiment.header.name)
                experiment.config.shots = 1
            # Set memory slots to 0
            if getattr(experiment.config, 'memory_slots', 0) != 0:
                logger.warning("statevector simulator does not use classical registers. "
                               "Setting memory_slots=0 for circuit %s.", experiment.header.name)
                experiment.config.memory_slots = 0
            for op in experiment.instructions:
                if op.name in ['measure', 'reset']:
                    raise AerSimulatorError(
                        "In circuit {}: statevector simulator does not support "
                        "measure or reset.".format(experiment.header.name))
