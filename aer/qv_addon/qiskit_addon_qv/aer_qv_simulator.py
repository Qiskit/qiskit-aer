"""
Cython quantum circuit simulator.
"""

import json
import logging
import datetime
import uuid
import numpy as np

# Import qiskit classes
import qiskit
from qiskit.backends import BaseBackend
from qiskit.backends.aer.aerjob import AerJob
from qiskit.qobj import qobj_to_dict
from qiskit.result._result import Result

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
        "basis_gates": 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz,ccx,swap'
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy(),
                         provider=provider)
        self.simulator = AerQvSimulatorWrapper()

    def run(self, qobj):
        """Run a qobj on the backend."""
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj)
        aer_job.submit()
        return aer_job

    def _run_job(self, job_id, qobj):
        self._validate(qobj)
        qobj_str = json.dumps(qobj_to_dict(qobj), cls=QvSimulatorJSONEncoder)
        output = json.loads(self.simulator.execute(qobj_str),
                            cls=QvSimulatorJSONDecoder)
        # Add result metadata
        output["job_id"] = job_id
        output["date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        output["backend_name"] = self.DEFAULT_CONFIGURATION['name']
        output["backend_version"] = "0.0.1"  # TODO: get this from somewhere else
        # Parse result dict into Result class
        exp_results = output.get("results", {})
        experiment_names = [data.get("header", {}).get("name", None)
                            for data in exp_results]
        qobj_result = qiskit.qobj.Result(**output)
        qobj_result.results = [qiskit.qobj.ExperimentResult(**res) for res in exp_results]
        return Result(qobj_result, experiment_names=experiment_names)

    def set_noise_model(self, noise_model):
        self.simulator.set_noise_model(json.dumps(noise_model, cls=QvSimulatorJSONEncoder))

    def clear_noise_model(self):
        self.simulator.clear_noise_model()

    def set_config(self, config):
        self.simulator.set_engine_config(json.dumps(config, cls=QvSimulatorJSONEncoder))
        self.simulator.set_state_config(json.dumps(config, cls=QvSimulatorJSONEncoder))

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


class QvSimulatorJSONEncoder(json.JSONEncoder):
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


class QvSimulatorJSONDecoder(json.JSONDecoder):
    """
    JSON decoder for the output with complex vector snapshots.

    This converts complex vectors and matrices into numpy arrays
    for the following keys.
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def decode_complex(self, obj):
        if isinstance(obj, list) and len(obj) == 2:
            if obj[1] == 0.:
                obj = obj[0]
            else:
                obj = obj[0] + 1j * obj[1]
        return obj

    def decode_complex_vector(self, obj):
        if isinstance(obj, list):
            obj = np.array([self.decode_complex(z) for z in obj], dtype=complex)
        return obj

    # pylint: disable=method-hidden
    def object_hook(self, obj):
        """Special decoding rules for simulator output."""

        # Decode snapshots
        if 'snapshots' in obj:
            # Decode state
            if 'state' in obj['snapshots']:
                for key in obj['snapshots']['state']:
                    tmp = [self.decode_complex_vector(vec)
                           for vec in obj['snapshots']['state'][key]]
                    obj['snapshots']['state'][key] = tmp
            # Decode observables
            if 'observables' in obj['snapshots']:
                for key in obj['snapshots']['observables']:
                    for j, val in enumerate(obj['snapshots']['observables'][key]):
                        obj['snapshots']['observables'][key][j]['value'] = self.decode_complex(val['value'])
        return obj
