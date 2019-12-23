# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=arguments-differ, missing-return-type-doc

"""
Qiskit Aer OpenPulse simulator backend.
"""

import uuid
import time
import datetime
import logging
from numpy import inf
from qiskit.result import Result
from qiskit.providers.models import BackendConfiguration, PulseDefaults
from .aerbackend import AerBackend
from ..aerjob import AerJob
from ..aererror import AerError
from ..version import __version__
from ..openpulse.qobj.digest import digest_pulse_obj
from ..openpulse.solver.opsolve import opsolve

logger = logging.getLogger(__name__)


class PulseSimulator(AerBackend):
    """Aer OpenPulse simulator
    """
    DEFAULT_CONFIGURATION = {
        'backend_name': 'pulse_simulator',
        'backend_version': __version__,
        'n_qubits': 20,
        'coupling_map': None,
        'url': 'https://github.com/Qiskit/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': True,
        'memory': False,
        'max_shots': 50000,
        'description': 'A pulse-based Hamiltonian simulator',
        'gates': [],
        'basis_gates': []
    }

    def __init__(self, configuration=None, provider=None):

        # purpose of defaults is to pass assemble checks
        self._defaults = PulseDefaults(qubit_freq_est=[inf],
                                       meas_freq_est=[inf],
                                       buffer=0,
                                       cmd_def=[],
                                       pulse_library=[])
        super().__init__(self,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def run(self, qobj,
            system_model,
            backend_options=None,
            validate=False):
        """Run a qobj on the backend."""
        # Submit job
        print('you are here')
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj, system_model,
                         backend_options, validate)
        aer_job.submit()
        return aer_job

    def _run_job(self, job_id, qobj,
                 system_model,
                 backend_options,
                 validate):
        """Run a qobj job"""
        start = time.time()
        if validate:
            self._validate(qobj, backend_options, noise_model=None)
        # Send to solver
        openpulse_system = digest_pulse_obj(qobj, system_model, backend_options)
        results = opsolve(openpulse_system)
        end = time.time()
        return self._format_results(job_id, results, end - start, qobj.qobj_id)

    def _format_results(self, job_id, results, time_taken, qobj_id):
        """Construct Result object from simulator output."""
        # Add result metadata
        output = {}
        output['qobj_id'] = qobj_id
        output['results'] = results
        output['success'] = True
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version
        output["time_taken"] = time_taken
        return Result.from_dict(output)

    def _validate(self, qobj, backend_options, noise_model):
        """Validate the pulse object. Make sure a
        config has been attached in the proper location"""

        super()._validate(qobj, backend_options, noise_model)

    def defaults(self):
        """Return defaults.

        Returns:
            PulseDefaults: object for passing assemble.
        """
        return self._defaults
