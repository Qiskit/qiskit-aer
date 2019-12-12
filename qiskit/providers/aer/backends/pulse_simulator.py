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
# pylint: disable=arguments-differ

"""
Qiskit Aer OpenPulse simulator backend.
"""

import uuid
import time
import datetime
import logging
from qiskit.result import Result
from qiskit.providers.models import BackendConfiguration
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
        super().__init__(self,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def run(self, qobj,
            backend_options=None,
            noise_model=None,
            validate=False):
        """Run a qobj on the backend."""
        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj,
                         backend_options, noise_model, validate)
        aer_job.submit()
        return aer_job

    def _run_job(self, job_id, qobj,
                 backend_options,
                 noise_model,
                 validate):
        """Run a qobj job"""
        start = time.time()
        if validate:
            self._validate(qobj, backend_options, noise_model)
        # Send to solver
        qobj_dict = self._format_qobj_dict(qobj, backend_options,
                                           noise_model)
        openpulse_system = digest_pulse_obj(qobj_dict)
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

    def _format_qobj_dict(self, qobj, backend_options, noise_model):
        """Add additional fields to qobj dictionary"""
        # Convert qobj to dict and add additional fields
        qobj_dict = qobj.to_dict()
        if 'backend_options' not in qobj_dict['config']:
            qobj_dict['config']['backend_options'] = {}

        # Temp backwards compatibility
        if 'sim_config' in qobj_dict['config']:
            for key, val in qobj_dict['config']['sim_config'].itmes():
                qobj_dict['config']['backend_options'][key] = val
            qobj_dict['config'].pop('sim_config')

        # Add additional backend options
        if backend_options is not None:
            for key, val in backend_options.items():
                qobj_dict['config']['backend_options'][key] = val
        # Add noise model
        if noise_model is not None:
            qobj_dict['config']['backend_options']['noise_model'] = noise_model
        return qobj_dict

    def get_dressed_energies(self, qobj,
                             backend_options=None,
                             noise_model=None):
        """Digest the pulse qobj and return the eigenenergies
        of the Hamiltonian"""
        qobj_dict = self._format_qobj_dict(qobj, backend_options,
                                           noise_model)
        openpulse_system = digest_pulse_obj(qobj_dict)
        return openpulse_system.evals, openpulse_system.estates

    def _validate(self, qobj, backend_options, noise_model):
        """Validate the pulse object. Make sure a
        config has been attached in the proper location"""

        # Check to make sure a sim_config has been added
        if not hasattr(qobj.config, 'sim_config'):
            raise AerError('The pulse simulator qobj must have a sim_config '
                           'entry to configure the simulator')

        super()._validate(qobj, backend_options, noise_model)
