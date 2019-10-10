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

from marshmallow import ValidationError
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.jobstatus import JobStatus
from ..aererror import AerError
from ..version import __version__
from ..api import HttpConnector
from ..api import SshConnector


logger = logging.getLogger(__name__)


class RemoteNode():
    """
    Remote Node Class
    """

    def __init__(self, host=None, method=None, connect_config=None):
        """
        Args:
            host (string) : Host address
            method (string) : Portocol
            connect_config(dict) : Connect configuration
        """
        self._method = method
        self._host = host
        self._status = None
        self._connect_config = None

        if method == "http":
            self._api = HttpConnector(self._host)

        if method == "ssh":
            self._api = SshConnector(self._host, connect_config)

        _raw_config = self._api.available_backends()
        raw_config = _raw_config[0]

        try:
            config = BackendConfiguration.from_dict(raw_config)
        except ValidationError as ex:
            logger.warning('Remote backend "%s" could not be instantiated due to an '
                           'invalid config: %s',
                           raw_config.get('backend_name',
                                          raw_config.get('name', 'unknown')), ex)

        self._backend_name = config.backend_name
        self._config = config
        self._api.config = config

    def get_status_job(self, job_id):
        """
        Get job status from the node.

        Args:
            job_id (string) : Job ID

        Returns:
            dict: job status

        Raises:
            AerError : No receive job status from the node
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
            dict: job result
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
            dict: submission info
        Raises:
            AerError : Can not submit qobj to remote node
        """
        try:
            submit_info = self._api.run_job(qobj, 'remote_qasm_simulator')

        # pylint: disable=broad-except
        except Exception as err:
            # Undefined error during submission:
            # Capture and keep it for raising it when calling status().
            raise AerError('Raise Error from Remote Simulator ', str(err))

        # Error in the job after submission:
        # Transition to the `ERROR` final state.
        if 'error' in submit_info:
            self._status = JobStatus.ERROR
            return submit_info

        # Submission success.
        return submit_info
