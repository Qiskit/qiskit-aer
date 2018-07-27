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
from qv_wrapper import StateVectorSimulatorCppWrapper

# Logger
logger = logging.getLogger(__name__)


class StatevectorSimulator(BaseBackend):
    """Cython quantum circuit simulator"""

    DEFAULT_CONFIGURATION = {
        'name': 'local_statevector_simulator_aer',
        'url': 'NA',
        'simulator': True,
        'local': True,
        'description': 'A C++ statevector simualtor for qobj files',
        'coupling_map': 'all-to-all',
        "basis_gates": 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz,'
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())
        self.simulator = StateVectorSimulatorCppWrapper()

    def run(self, qobj):
        """Run a QOBJ on the the backend."""
        return LocalJob(self._run_job, qobj)

    def _run_job(self, qobj):
        self._validate(qobj)
        qobj_str = json.dumps(qobj, cls=SimulatorJSONEncoder)
        result = json.loads(self.simulator.execute(qobj_str),
                            cls=StatevectorJSONDecoder)
        return Result(result)

    def _validate(self, qobj):
        return


class StatevectorJSONDecoder(json.JSONDecoder):
    """
    JSON decoder for the output from C++ qasm_simulator.

    This converts complex vectors and matrices into numpy arrays
    for the following keys.
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    # pylint: disable=method-hidden
    def object_hook(self, obj):
        """Special decoding rules for simulator output."""

        for key in ['statevector']:
            # JSON is a list of complex vectors
            if key in obj:
                for j in range(len(obj[key])):
                    if isinstance(obj[key][j], list):
                        tmp = np.array(obj[key][j])
                        obj[key][j] = tmp[::, 0] + 1j * tmp[::, 1]
        return obj
