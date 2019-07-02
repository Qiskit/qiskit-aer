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

"""This module implements the remote job class used for AerBackend objects."""

from concurrent import futures
import time
import logging
import pprint
import datetime

from qiskit.providers import BaseJob, JobError, JobTimeoutError
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit.result import Result
from qiskit.qobj import QasmQobjConfig, validate_qobj_against_schema

from .api import ApiError

logger = logging.getLogger(__name__)


class AerNodeStatus():
    """
    AerNodeStatus class

    This class stores the each node status for the job.
    """
    def __init__(self, job_id, status, node):
        """
        Args:
            job_id (string) : Job id for the node
            status (string) :: Job status for the node
            node (remote_node) : node class
        """
        self._job_id = job_id
        self._status = status
        self._remote_node = node


class AerRemoteJob(BaseJob):
    """
    AerRemoteJob class

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """
    _executor = futures.ThreadPoolExecutor(max_workers=5)

    def __init__(self, backend, simulator, job_id, qobj=None, noise_model=None, run_config=None):
        """
        Args:
            backend(AerBackend) : Backend for this job
            simulator(RemoteSimulator) : Simulator of this job
            job_id (string) : Job ID
            qobj (Qobj) : Submission qobj
            noise_model (Qobj) : Noise Model
            run_config (dict): run configuration
        """
        super().__init__(backend, job_id)
        self._simulator = simulator
        self._job_id = job_id
        self._job_data = None
        self._backend = backend
        self._status = JobStatus.INITIALIZING
        self._future = None
        self._node_status = []
        self._noise = False
        self._future_captured_exception = None
        self._gpu_enable = False

        if run_config is not None and "GPU" in run_config:
            self._gpu_enable = run_config["GPU"]

        if qobj is not None:
            config = qobj.config.to_dict()
            if noise_model:
                config["noise_model"] = noise_model.to_dict()
                del config["run_config"]
                qobj.config = QasmQobjConfig.from_dict(config)
                self._noise = True

            if self._gpu_enable:
                # with noise, copy qobj and submit to the nodes
                config["GPU"] = True
                del config["run_config"]
                qobj.config = QasmQobjConfig.from_dict(config)

            validate_qobj_against_schema(qobj)
            self._qobj_payload = qobj.to_dict()
        else:
            self._qobj_payload = {}

        self._creation_date = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

    def result(self):
        """Return the result from the job.
        Returns:
            qiskit.Result: Result object
        """

        job_response = self._wait_for_result(timeout=None, wait=5)
        return Result.from_dict(job_response['qObjectResult'])

    def status(self):
        """Query the API to update the status.

        Returns:
            qiskit.providers.JobStatus: The status of the job, once updated.

        Raises:
            JobError: if there was an exception in the future being executed
                          or the server sent an unknown answer.
        """
        self._wait_for_submission()

        if self._job_id is None or self._status in JOB_FINAL_STATES:
            return self._status

        try:
            api_job = self._simulator.get_status_job(self._job_id, self._node_status)
            if 'status' not in api_job:
                raise JobError('get_status_job didn\'t return status: %s' %
                               pprint.pformat(api_job))
        # pylint: disable=broad-except
        except Exception as err:
            raise JobError(str(err))

        if api_job['status'] == 'RUNNING':
            self._status = JobStatus.RUNNING

        elif api_job['status'] == 'COMPLETED':
            self._status = JobStatus.DONE

        elif 'ERROR' in api_job['status']:
            # Error status are of the form "ERROR_*_JOB"
            self._status = JobStatus.ERROR
        else:
            raise JobError('Unrecognized answer from server: \n{}'
                           .format(pprint.pformat(api_job)))

        return self._status

    def submit(self):
        """Submit job to On-premise simulator.

        Raises:
            JobError: If we have already submitted the job.
        """
        # TODO: Validation against the schema should be done here and not
        # during initialization. Once done, we should document that the method
        # can raise QobjValidationError.

        if self._future is not None:
            raise JobError("We have already submitted the job!")
        self._future = self._executor.submit(self._submit_callback)

    def cancel(self):
        return self._future.cancel()

    def _wait_for_result(self, timeout=None, wait=5):
        """
        Wait for job result until status is COMPLETED
        """

        try:
            job_response = self._wait_for_job(timeout=timeout, wait=wait)
            if not self._qobj_payload:
                self._qobj_payload = job_response.get('qObject', {})
        except ApiError as api_err:
            raise JobError(str(api_err))

        status = self.status()
        if status is not JobStatus.DONE:
            raise JobError('Invalid job state. The job should be DONE but '
                           'it is {}'.format(str(status)))

        return job_response

    def _submit_callback(self):
        """Submit qobj job to On-premise simulator.

        Returns:
            dict: A dictionary with the response of the submitted job
        """
        try:
            submit_info_list = self._simulator.run_job(self._qobj_payload, self._noise)
        # pylint: disable=broad-except
        except Exception as err:
            # Undefined error during submission:
            # Capture and keep it for raising it when calling status().
            self._future_captured_exception = err
            return None

        for submit_info in submit_info_list:

            # Error in the job after submission:
            # Transition to the `ERROR` final state.
            if 'error' in submit_info:
                self._status = JobStatus.ERROR
                return submit_info

            # Submission success.
            node_job_id = submit_info["info"].get('id')
            job_status = JobStatus.QUEUED

            state = AerNodeStatus(node_job_id, job_status, submit_info["node"])
            self._node_status.append(state)

        self._status = JobStatus.QUEUED
        return submit_info_list

    def _wait_for_job(self, timeout=60, wait=5):
        """Wait until all online ran circuits of a qobj are 'COMPLETED'.

        Args:
            timeout (float or None): seconds to wait for job. If None, wait
                indefinitely.
            wait (float): seconds between queries

        Returns:
            dict: A dict with the contents of the API request.

        Raises:
            JobTimeoutError: if the job does not return results before a specified timeout.
            JobError: if something wrong happened in some of the server API calls
        """
        self._wait_for_submission()

        start_time = time.time()
        while self.status() not in JOB_FINAL_STATES:
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise JobTimeoutError(
                    'Timeout while waiting for the job: {}'.format(self._job_id)
                )

            logger.info('status = %s (%d seconds)', self._status, elapsed_time)
            time.sleep(wait)

        return self._simulator.get_job(self._job_id, self._node_status)

    def _wait_for_submission(self, timeout=60):
        """Waits for the request to return a job ID"""

        if not self._node_status:
            if self._future is None:
                raise JobError("You have to submit before asking for status or results!")
            try:
                submit_info_list = self._future.result(timeout=timeout)
            except TimeoutError as ex:
                raise JobTimeoutError(
                    "Timeout waiting for the job being submitted: {}".format(ex)
                )
            for submit_info in submit_info_list:
                if 'error' in submit_info["info"]:
                    self._status = JobStatus.ERROR
                    raise JobError(str(submit_info['error']))
