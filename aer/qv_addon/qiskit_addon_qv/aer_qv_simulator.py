"""
Cython quantum circuit simulator.
"""

import sys
import os
import json
import logging
import warnings
import datetime
import uuid
import numpy as np

# Import qiskit classes
import qiskit
from qiskit.backends import BaseBackend
from qiskit.backends.local.localjob import LocalJob
from qiskit.qobj import qobj_to_dict
from qiskit.result._result import Result
from qiskit.result._utils import copy_qasm_from_qobj_into_result

# Import Simulator tools
from aer_qv_wrapper import AerQvSimulatorWrapper

# Logger
logger = logging.getLogger(__name__)


class AerQvSimulator(BaseBackend):
    """Cython quantum circuit simulator"""

    DEFAULT_CONFIGURATION = {
        'name': 'local_qv_simulator',
        'url': 'NA',
        'simulator': True,
        'local': True,
        'description': 'A C++ statevector simulator for qobj files',
        'coupling_map': 'all-to-all',
        "basis_gates": 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz',
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())
        self.simulator = AerQvSimulatorWrapper()

    def run(self, qobj):
        """Run qobj asynchronously.

        Args:
            qobj (dict): job description

        Returns:
            LocalJob: derived from BaseJob
        """
        local_job = LocalJob(self._run_job, qobj)
        local_job.submit()
        return local_job

    def _run_job(self, qobj):
        self._validate(qobj)
        qobj_str = json.dumps(qobj_to_dict(qobj), cls=SimulatorJSONEncoder)
        output = json.loads(self.simulator.execute(qobj_str))
        # Add result metadata
        output["date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        output["job_id"] = str(uuid.uuid4())
        output["backend_name"] = self.DEFAULT_CONFIGURATION['name']
        output["backend_version"] = "0.0.1"  # TODO: get this from somewhere else
        # Parse result dict into Result class
        exp_results = output.get("results", {})
        experiment_names = [data.get("header", {}).get("name", None)
                            for data in exp_results]
        qobj_result = qiskit.qobj.Result(**output)
        qobj_result.results = [qiskit.qobj.ExperimentResult(**res) for res in exp_results]
        return Result(qobj_result, experiment_names=experiment_names)

    def load_noise_model(self, noise_model):
        self.simulator.load_noise_model(json.dumps(noise_model, cls=SimulatorJSONEncoder))

    def clear_noise_model(self):
        self.simulator.clear_noise_model()

    def load_config(self, engine=None, state=None):
        if engine is not None:
            self.simulator.load_engine_config(json.dumps(engine, cls=SimulatorJSONEncoder))
        if state is not None:
            self.simulator.load_state_config(json.dumps(state, cls=SimulatorJSONEncoder))

    def set_max_threads_shot(self, threads):
        """
        Set the maximum threads used for parallel shot execution.

        Args:
            threads (int): the thread limit, set to -1 to use maximum available

        Note that using parallel shot evaluation disables parallel circuit
        evaluation.
        """

        self.simulator.set_max_threads_shot(int(threads))

    def set_max_threads_circuit(self, threads):
        """
        Set the maximum threads used for parallel circuit execution.

        Args:
            threads (int): the thread limit, set to -1 to use maximum available

        Note that using parallel circuit evaluation disables parallel shot
        evaluation.
        """
        self.simulator.set_max_threads_circuit(int(threads))

    def set_max_threads_state(self, threads):
        """
        Set the maximum threads used for state update parallel  routines.

        Args:
            threads (int): the thread limit, set to -1 to use maximum available.

        Note that using parallel circuit or shot execution takes precidence over
        parallel state evaluation.
        """
        self.simulator.set_max_threads_state(int(threads))

    def _validate(self, qobj):
        # TODO
        return


class SimulatorJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:
        complex numbers z as lists [z.real, z.imag]
        ndarrays as nested lists.
    """

    # pylint: disable=method-hidden,arguments-differ
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        return json.JSONEncoder.default(self, obj)
