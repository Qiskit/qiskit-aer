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

"""
Qiskit Aer qasm simulator backend.
"""

import json
import logging
import datetime
import os
import time
import uuid
from numpy import ndarray

from qiskit.providers import BaseBackend
from qiskit.providers.models import BackendStatus
from qiskit.qobj import QasmQobjConfig, validate_qobj_against_schema
from qiskit.result import Result
from qiskit.util import local_hardware_info

from ..aerjob import AerJob
from ..aererror import AerError

# Logger
logger = logging.getLogger(__name__)

# Location where we put external libraries that will be loaded at runtime
# by the simulator extension
LIBRARY_DIR = os.path.dirname(__file__)


class AerJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:
        complex numbers z as lists [z.real, z.imag]
        ndarrays as nested lists.
    """

    # pylint: disable=method-hidden,arguments-differ
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


class AerBackend(BaseBackend):
    """Qiskit Aer Backend class."""

    def __init__(self, controller, configuration, provider=None):
        """Aer class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            controller (function): Aer cython controller to be executed
            configuration (BackendConfiguration): backend configuration
            provider (BaseProvider): provider responsible for this backend

        Raises:
            FileNotFoundError if backend executable is not available.
            AerError: if there is no name in the configuration
        """
        super().__init__(configuration, provider=provider)
        self._controller = controller

    # pylint: disable=arguments-differ
    def run(self, qobj, backend_options=None, noise_model=None, validate=True):
        """Run a qobj on the backend."""
        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj,
                         backend_options, noise_model, validate)
        aer_job.submit()
        return aer_job

    def status(self):
        """Return backend status.

        Returns:
            BackendStatus: the status of the backend.
        """
        return BackendStatus(backend_name=self.name(),
                             backend_version=self.configuration().backend_version,
                             operational=True,
                             pending_jobs=0,
                             status_msg='')

    def _run_job(self, job_id, qobj, backend_options, noise_model, validate):
        """Run a qobj job"""
        start = time.time()
        if validate:
            validate_qobj_against_schema(qobj)
            self._validate(qobj, backend_options, noise_model)
        qobj_str = self._format_qobj_str(qobj, backend_options, noise_model)
        output = json.loads(self._controller(qobj_str).decode('UTF-8'))
        self._validate_controller_output(output)
        end = time.time()
        return self._format_results(job_id, output, end - start)

    def _format_qobj_str(self, qobj, backend_options, noise_model):
        """Format qobj string for qiskit aer controller"""
        # Save original qobj config so we can revert our modification
        # after execution
        original_config = qobj.config
        # Convert to dictionary and add new parameters
        # from noise model and backend options
        config = original_config.to_dict()
        if backend_options is not None:
            for key, val in backend_options.items():
                config[key] = val
        if "max_memory_mb" not in config:
            max_memory_mb = int(local_hardware_info()['memory'] * 1024 / 2)
            config['max_memory_mb'] = max_memory_mb
        # Add noise model
        if noise_model is not None:
            config["noise_model"] = noise_model

        # Add runtime config
        config['library_dir'] = LIBRARY_DIR
        qobj.config = QasmQobjConfig.from_dict(config)
        # Get the JSON serialized string
        output = json.dumps(qobj, cls=AerJSONEncoder).encode('UTF-8')
        # Revert original qobj
        qobj.config = original_config
        # Return output
        return output

    def _format_results(self, job_id, output, time_taken):
        """Construct Result object from simulator output."""
        # Add result metadata
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version
        output["time_taken"] = time_taken
        return Result.from_dict(output)

    def _validate_controller_output(self, output):
        """Validate output from the controller wrapper."""
        if not isinstance(output, dict):
            logger.error("%s: simulation failed.", self.name())
            if output:
                logger.error('Output: %s', output)
            raise AerError("simulation terminated without returning valid output.")

    def _validate(self, qobj, backend_options, noise_model):
        """Validate the qobj, backend_options, noise_model for the backend"""
        pass

    def __repr__(self):
        """Official string representation of an AerBackend."""
        display = "{}('{}')".format(self.__class__.__name__, self.name())
        provider = self.provider()
        if provider is not None:
            display = display + " from {}()".format(provider)
        return "<" + display + ">"
