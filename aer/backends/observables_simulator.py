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
from qv_wrapper import ObservablesSimulatorWrapper

# Logger
logger = logging.getLogger(__name__)


class ObservablesSimulator(BaseBackend):
    """Cython quantum circuit simulator"""

    DEFAULT_CONFIGURATION = {
        'name': 'local_observables_simulator_aer',
        'url': 'NA',
        'simulator': True,
        'local': True,
        'description': 'A C++ statevector QASM simulator for qobj files',
        'coupling_map': 'all-to-all',
        "basis_gates": 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz'
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())
        self.simulator = ObservablesSimulatorWrapper()

    def run(self, qobj):
        """Run a QOBJ on the the backend."""
        return LocalJob(self._run_job, qobj)

    def _run_job(self, qobj):
        self._validate(qobj)
        qobj_str = json.dumps(qobj, cls=SimulatorJSONEncoder)
        result = json.loads(self.simulator.execute(qobj_str),
                            cls=ObservablesJSONDecoder)
        # TODO: get schema in line with result object
        return result  # Result(result)

    def _validate(self, qobj):
        for circ in qobj['circuits']:
            if 'measure' not in [op['name'] for
                                 op in circ['compiled_circuit']['operations']]:
                logger.warning("no measurements in circuit '%s', "
                               "classical register will remain all zeros.", circ['name'])
        return


class ObservablesJSONDecoder(json.JSONDecoder):
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

        if "observables" in obj:
            for i, item in enumerate(obj["observables"]):
                for j, val in enumerate(item["value"]):
                    obj["observables"][i]["value"][j] = val[0] + 1j * val[0]
        return obj
