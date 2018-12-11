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
import uuid
from numpy import ndarray

from qiskit.backends import BaseBackend
from qiskit.result import Result

from .aerjob import AerJob
from .aersimulatorerror import AerSimulatorError

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

    def __init__(self, controller_wrapper, configuration, provider=None):
        """Aer class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            controller_wrapper (class): Aer Controller cython wrapper class
            configuration (BackendConfiguration): backend configuration
            provider (BaseProvider): provider responsible for this backend

        Raises:
            FileNotFoundError if backend executable is not available.
            QISKitError: if there is no name in the configuration
        """
        super().__init__(configuration, provider=provider)
        self._controller = controller_wrapper

    def run(self, qobj, backend_options=None, noise_model=None):
        """Run a qobj on the backend."""
        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj, backend_options, noise_model)
        aer_job.submit()
        return aer_job

    def properties(self):
        """Return backend properties."""
        return None

    def _run_job(self, job_id, qobj, backend_options, noise_model):
        """Run a qobj job"""
        self._validate(qobj)
        options_str = self._format_options(qobj, backend_options)
        qobj_str = json.dumps(qobj.as_dict(), cls=AerJSONEncoder)
        noise_str = json.dumps(noise_model, cls=AerJSONEncoder)
        output = json.loads(self._controller.execute(qobj_str, options_str, noise_str))
        self._validate_controller_output(output)
        return self._format_results(job_id, output)

    def _format_options(self, qobj, backend_options):
        """Format options string for qiskit aer controller"""
        # Note: This is a temp workaround until PR #127 is merged
        # Add qobj config to backend_options
        if backend_options is None:
            backend_options = {}
        for key, val in qobj.config.as_dict().items():
            backend_options[key] = val
        # Get the JSON serialized string
        return json.dumps(backend_options, cls=AerJSONEncoder)

    def _format_results(self, job_id, output):
        """Construct Result object from simulator output."""
        # Add result metadata
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.DEFAULT_CONFIGURATION['backend_name']
        output["backend_version"] = self.DEFAULT_CONFIGURATION['backend_version']
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
                    raise AerSimulatorError(res.get("status", None))
            # If no error was found check for error message at qobj level
            raise AerSimulatorError(output.get("status", None))

    def set_max_threads_shot(self, threads):
        """
        Set the maximum threads used for parallel shot execution.

        Args:
            threads (int): the thread limit, set to -1 to use maximum available

        Note that using parallel shot evaluation disables parallel circuit
        evaluation.
        """
        self._controller.set_max_threads_shot(int(threads))

    def set_max_threads_circuit(self, threads):
        """
        Set the maximum threads used for parallel circuit execution.

        Args:
            threads (int): the thread limit, set to -1 to use maximum available

        Note that using parallel circuit evaluation disables parallel shot
        evaluation.
        """
        self._controller.set_max_threads_circuit(int(threads))

    def set_max_threads_state(self, threads):
        """
        Set the maximum threads used for state update parallel  routines.

        Args:
            threads (int): the thread limit, set to -1 to use maximum available.

        Note that using parallel circuit or shot execution takes precidence over
        parallel state evaluation.
        """
        self._controller.set_max_threads_state(int(threads))

    def _validate(self, qobj):
        # TODO
        return

    def __repr__(self):
        """Official string representation of an AerBackend."""
        display = "{}('{}')".format(self.__class__.__name__, self.name())
        if self.provider is not None:
            display = display + " from {}()".format(self._provider)
        return "<" + display + ">"
