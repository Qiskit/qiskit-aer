"""
Cython quantum circuit simulator.
"""

import sys
import os
import json
import logging
import warnings
import numpy as np

# Import QISKit classes
from qiskit._result import Result
from qiskit.backends import BaseBackend
from qiskit.backends.local.localjob import LocalJob

# Import Simulator tools
from json_encoder import SimulatorJSONEncoder
from qv_wrapper import QasmSimulatorCppWrapper

# Logger
logger = logging.getLogger(__name__)


class QasmSimulator(BaseBackend):
    """Cython quantum circuit simulator"""

    DEFAULT_CONFIGURATION = {
        'name': 'local_qasm_simulator_aer',
        'url': 'NA',
        'simulator': True,
        'local': True,
        'description': 'A C++ statevector QASM simulator for qobj files',
        'coupling_map': 'all-to-all',
        "basis_gates": 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz'
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())
        self.simulator = QasmSimulatorCppWrapper()

    def run(self, qobj):
        """Run a QOBJ on the the backend."""
        return LocalJob(self._run_job, qobj)

    def _run_job(self, qobj):
        self._validate(qobj)
        qobj_str = json.dumps(qobj, cls=SimulatorJSONEncoder)
        result = json.loads(self.simulator.execute(qobj_str))
        Result(result)

    def _validate(self, qobj):
        for circ in qobj['circuits']:
            if 'measure' not in [op['name'] for
                                 op in circ['compiled_circuit']['operations']]:
                logger.warning("no measurements in circuit '%s', "
                               "classical register will remain all zeros.", circ['name'])
        return
