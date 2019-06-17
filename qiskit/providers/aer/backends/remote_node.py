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

# pylint: disable=invalid-name, bad-continuation

"""This module implements the remote node used for AerBackend objects."""

import logging
import json

from marshmallow import ValidationError
from ..aererror import AerError
from ..version import __version__
from ..api import HttpConnector
from ..aererror import AerError
from .aerbackend import AerBackend
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit.result import Result


logger = logging.getLogger(__name__)

class RemoteNode():
    """
    Remote Node Class
    """

    def __init__(self, url=None, method=None):
        """
        Args:
            url (string) : API Url
            method (string) : Portocol
        """
        self._method = method
        self._url = url
        self._status = None


        if method is "http":
            self._api = HttpConnector(self._url)

        _raw_config = self._api.available_backends()
        raw_config = _raw_config[0]

        try:
            config = BackendConfiguration.from_dict(raw_config)
        except ValidationError as ex:
            logger.warning(
                'Remote backend "%s" could not be instantiated due to an '
                'invalid config: %s',
                raw_config.get('backend_name',
                raw_config.get('name', 'unknown')),
                ex)

        self._backend_name = config.backend_name
        self._config = config
        self._api.config = config

    def get_status_job(self, job_id):
        """
        Get job status from the node

        Args:
            job_id (string) : Job ID
        Returns
            job_status (string) : Dict data of job status
        """
        try:
            job_status = self._api.get_status_job(job_id)
        except Exception as err:
            raise AerError('Raise Error from Remote Simulator ', str(err))

        return job_status

    def get_job(self, job_id):
        """
        Get Result from the node.

        Args:
            job_id (string) : Job ID
        Returns:
            result (string) : dict data of job result
        Raises:
            AerError : Not receive the result from the node
        """
        try:
            result = self._api.get_job(job_id)
        except Exception as err:
            raise AerError('Raise Error from Remote Simulator ', str(err))

        return result

    def execute_job(self, qobj):
        """
        Submit job to the node.

        Args:
            qobj (Qobj) : Submittion qobj
        Returns:
            subimit_info (dict) : submission info
        Raises:
            AerError : Can not submit qobj to remote node
        """
        try:
            submit_info = self._api.run_job(qobj, 'qasm_simulator')

        # pylint: disable=broad-except
        except Exception as err:
            # Undefined error during submission:
            # Capture and keep it for raising it when calling status().
            raise AerError('Raise Error from Remote Simulator ', str(err))

        # Error in the job after submission:
        # Transition to the `ERROR` final state.
        if 'error' in submit_info:
            self._status = JobStatus.ERROR
            self._api_error_msg = str(submit_info['error'])
            return submit_info

        # Submission success.
        return submit_info