# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from concurrent import futures
import time
import logging
import pprint
import datetime

from qiskit.qobj import Qobj
from qiskit.providers import BaseJob, JobError, JobTimeoutError
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit.result import Result
from qiskit.qobj import validate_qobj_against_schema

from .api import ApiError


logger = logging.getLogger(__name__)

API_FINAL_STATES = (
    'COMPLETED',
    'CANCELLED',
    'ERROR_CREATING_JOB',
    'ERROR_VALIDATING_JOB',
    'ERROR_RUNNING_JOB'
)


class AerCloudJob(BaseJob):
    _executor = futures.ThreadPoolExecutor()

    def __init__(self, backend, job_id, api, is_device, qobj=None,
                 creation_date=None, api_status=None):
        super().__init__(backend, job_id)
        self._job_data = None

        if qobj is not None:
            validate_qobj_against_schema(qobj)
            self._qobj_payload = qobj.as_dict()
        else:
            self._qobj_payload = {}

        self._future_captured_exception = None
        self._api = api
        self._backend = backend
        self._status = JobStatus.INITIALIZING
        # In case of not providing a `qobj`, it is assumed the job already
        # exists in the API (with `job_id`).
        if qobj is None:
            # Some API calls (`get_status_jobs`, `get_status_job`) provide
            # enough information to recreate the `Job`. If that is the case, try
            # to make use of that information during instantiation, as
            # `self.status()` involves an extra call to the API.
            if api_status == 'COMPLETED':
                self._status = JobStatus.DONE
            else:
                self.status()
        self._queue_position = None
        self._is_device = is_device

        def current_utc_time():
            """Gets the current time in UTC format"""
            datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

        self._creation_date = creation_date or current_utc_time()
        self._future = None
        self._api_error_msg = None

    # pylint: disable=arguments-differ
    def result(self, timeout=None, wait=5):
        """Return the result from the job.

        Args:
           timeout (int): number of seconds to wait for job
           wait (int): time between queries to On-premise server

        Returns:
            qiskit.Result: Result object

        Raises:
            JobError: exception raised during job initialization
        """
        job_response = self._wait_for_result(timeout=timeout, wait=wait)
        #print(job_response)
        return self._result_from_job_response(job_response)

    def _wait_for_result(self, timeout=None, wait=5):
        self._wait_for_submission(timeout)

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

    def _result_from_job_response(self, job_response):
        return Result.from_dict(job_response['qObjectResult'])

    def status(self):
        """Query the API to update the status.

        Returns:
            qiskit.providers.JobStatus: The status of the job, once updated.

        Raises:
            JobError: if there was an exception in the future being executed
                          or the server sent an unknown answer.
        """
        # Implies self._job_id is None
        if self._future_captured_exception is not None:
            raise JobError(str(self._future_captured_exception))

        if self._job_id is None or self._status in JOB_FINAL_STATES:
            return self._status

        try:
            # TODO: See result values
            api_job = self._api.get_status_job(self._job_id)
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
            # TODO: This seems to be an inconsistency in the API package.
            self._api_error_msg = api_job.get('error') or api_job.get('Error')

        else:
            raise JobError('Unrecognized answer from server: \n{}'
                           .format(pprint.pformat(api_job)))

        return self._status

    def error_message(self):
        """Return the error message returned from the API server response."""
        return self._api_error_msg

    def creation_date(self):
        """
        Return creation date.
        """
        return self._creation_date

    def job_id(self):
        """Return backend determined id.

        If the Id is not set because the job is already initializing, this call
        will block until we have an Id.
        """
        self._wait_for_submission()
        return self._job_id

    def submit(self):
        """Submit job to On-premise simulator.

        Raises:
            JobError: If we have already submitted the job.
        """
        # TODO: Validation against the schema should be done here and not
        # during initialization. Once done, we should document that the method
        # can raise QobjValidationError.
        if self._future is not None or self._job_id is not None:
            raise JobError("We have already submitted the job!")
        self._future = self._executor.submit(self._submit_callback)

    def cancel(self): 
        raise JobError("Not Support")

    def _submit_callback(self):
        """Submit qobj job to On-premise simulator.

        Returns:
            dict: A dictionary with the response of the submitted job
        """
        backend_name = self.backend().name()

        try:
            submit_info = self._api.run_job(self._qobj_payload, backend=backend_name)
        # pylint: disable=broad-except
        except Exception as err:
            # Undefined error during submission:
            # Capture and keep it for raising it when calling status().
            self._future_captured_exception = err
            return None

        # Error in the job after submission:
        # Transition to the `ERROR` final state.
        if 'error' in submit_info:
            self._status = JobStatus.ERROR
            self._api_error_msg = str(submit_info['error'])
            return submit_info

        # Submission success.
        self._creation_date = submit_info.get('creationDate')
        self._status = JobStatus.QUEUED
        self._job_id = submit_info.get('id')
        return submit_info

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
        start_time = time.time()
        while self.status() not in JOB_FINAL_STATES:
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise JobTimeoutError(
                    'Timeout while waiting for the job: {}'.format(self._job_id)
                )

            logger.info('status = %s (%d seconds)', self._status, elapsed_time)
            time.sleep(wait)

        return self._api.get_job(self._job_id)

    def _wait_for_submission(self, timeout=60):
        """Waits for the request to return a job ID"""
        if self._job_id is None:
            if self._future is None:
                raise JobError("You have to submit before asking for status or results!")
            try:
                submit_info = self._future.result(timeout=timeout)
                if self._future_captured_exception is not None:
                    # pylint can't see if catch of None type
                    # pylint: disable=raising-bad-type
                    raise self._future_captured_exception
            except TimeoutError as ex:
                raise JobTimeoutError(
                    "Timeout waiting for the job being submitted: {}".format(ex)
                )
            if 'error' in submit_info:
                self._status = JobStatus.ERROR
                self._api_error_msg = str(submit_info['error'])
                raise JobError(str(submit_info['error']))

    def qobj(self):
        """Return the Qobj submitted for this job.

        Note that this method might involve querying the API for results if the
        Job has been created in a previous Qiskit session.

        Returns:
            Qobj: the Qobj submitted for this job.
        """
        if not self._qobj_payload:
            # Populate self._qobj_payload by retrieving the results.
            self._wait_for_result()

        return Qobj(**self._qobj_payload)
