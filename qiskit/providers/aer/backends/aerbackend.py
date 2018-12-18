# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qiskit Aer qasm simulator backend.
"""

import json
import logging
import datetime
import time
import uuid
from numpy import ndarray

from qiskit.providers import BaseBackend
from qiskit.providers.models import BackendStatus
from qiskit.qobj import QobjConfig
from qiskit.result import Result

from ..aerjob import AerJob
from ..aererror import AerError

# Logger
logger = logging.getLogger(__name__)


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
        if hasattr(obj, "as_dict"):
            return obj.as_dict()
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
            QiskitError: if there is no name in the configuration
        """
        super().__init__(configuration, provider=provider)
        self._controller = controller

    def run(self, qobj, backend_options=None, noise_model=None):
        """Run a qobj on the backend."""
        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj, backend_options, noise_model)
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

    def _run_job(self, job_id, qobj, backend_options, noise_model):
        """Run a qobj job"""
        start = time.time()
        self._validate(qobj)
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
        config = original_config.as_dict()
        if backend_options is not None:
            for key, val in backend_options.items():
                config[key] = val
        # Add noise model
        if noise_model is not None:
            config["noise_model"] = noise_model
        qobj.config = QobjConfig.from_dict(config)
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
        # Check results
        # TODO: Once https://github.com/Qiskit/qiskit-terra/issues/1023
        #       is merged this should be updated to deal with errors using
        #       the Result object methods
        if not output.get("success", False):
            logger.error("AerBackend: simulation failed")
            # Check for error message in the failed circuit
            for res in output.get('results'):
                if not res.get('success', False):
                    raise AerError(res.get("status", None))
            # If no error was found check for error message at qobj level
            raise AerError(output.get("status", None))

    def _validate(self, qobj):
        # TODO
        return

    def __repr__(self):
        """Official string representation of an AerBackend."""
        display = "{}('{}')".format(self.__class__.__name__, self.name())
        provider = self.provider()
        if provider is not None:
            display = display + " from {}()".format(provider)
        return "<" + display + ">"
