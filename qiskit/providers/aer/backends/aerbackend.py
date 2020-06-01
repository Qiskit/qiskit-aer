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
from qiskit.qobj import validate_qobj_against_schema
from qiskit.result import Result
from qiskit.util import local_hardware_info

from ..aerjob import AerJob

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
            controller (function): Aer controller to be executed
            configuration (BackendConfiguration): backend configuration
            provider (BaseProvider): provider responsible for this backend

        Raises:
            FileNotFoundError if backend executable is not available.
            AerError: if there is no name in the configuration
        """
        super().__init__(configuration, provider=provider)
        self._controller = controller

    # pylint: disable=arguments-differ
    def run(self, qobj, backend_options=None, noise_model=None, validate=False):
        """Run a qobj on the backend.

        Args:
            qobj (QasmQobj): The Qobj to be executed.
            backend_options (dict or None): dictionary of backend options
                                            for the execution (default: None).
            noise_model (NoiseModel or None): noise model to use for
                                              simulation (default: None).
            validate (bool): validate the Qobj before running (default: True).

        Returns:
            AerJob: The simulation job.

        Additional Information:
            * The entries in the ``backend_options`` will be combined with
              the ``Qobj.config`` dictionary with the values of entries in
              ``backend_options`` taking precedence.

            * If present the ``noise_model`` will override any noise model
              specified in the ``backend_options`` or ``Qobj.config``.
        """
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
        output = self._controller(self._format_qobj(qobj, backend_options, noise_model))
        end = time.time()
        return Result.from_dict(self._format_results(job_id, output, end - start))

    def _format_qobj(self, qobj, backend_options, noise_model):
        """Format qobj string for qiskit aer controller"""
        # Convert qobj to dict so as to avoid editing original
        output = qobj.to_dict()
        # Add new parameters to config from backend options
        config = output["config"]
        if backend_options is not None:
            for key, val in backend_options.items():
                config[key] = val if not hasattr(val, 'to_dict') else val.to_dict()
        # Add noise model to config
        if noise_model is not None:
            config["noise_model"] = noise_model

        # Add runtime config
        if 'library_dir' not in config:
            config['library_dir'] = LIBRARY_DIR
        if "max_memory_mb" not in config:
            max_memory_mb = int(local_hardware_info()['memory'] * 1024 / 2)
            config['max_memory_mb'] = max_memory_mb

        self._validate_config(config)
        # Return output
        return output

    def _validate_config(self, config):
        # sanity checks on config- should be removed upon fixing of assemble w.r.t. backend_options
        if 'backend_options' in config:
            if isinstance(config['backend_options'], dict):
                for key, val in config['backend_options'].items():
                    if hasattr(val, 'to_dict'):
                        config['backend_options'][key] = val.to_dict()
            elif not isinstance(config['backend_options'], list):
                raise ValueError("config[backend_options] must be a dict or list!")
        # Double-check noise_model is a dict type
        if 'noise_model' in config and not isinstance(config["noise_model"], dict):
            if hasattr(config["noise_model"], 'to_dict'):
                config["noise_model"] = config["noise_model"].to_dict()
            else:
                raise ValueError("noise_model must be a dict : " + str(type(config["noise_model"])))

    def _format_results(self, job_id, output, time_taken):
        """Construct Result object from simulator output."""
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version
        output["time_taken"] = time_taken
        return output

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
