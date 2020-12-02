# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Job managed by the Job Manager."""

import warnings
import logging
from typing import Optional, Any
from concurrent import futures

from qiskit.qobj import QasmQobj
from qiskit.result import Result
from qiskit.providers import JobStatus, JobError
from ..aerbackend import AerBackend
from .utils import requires_submit

logger = logging.getLogger(__name__)


class CJob:
    """Job managed by the ClusterBackend.

    Attributes:
       _executor (futures.Executor): default executor to handle asynchronous jobs
    """

    _executor = futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self, backend: AerBackend, qobj: QasmQobj, *run_args: Any, **run_kwargs: Any):
        """ManagedJob constructor.

        Args:
            backend: Backend to execute the experiments on.
            qobj: Assembled qobj object.
            *run_args: Positional args passed through to backend.run
            **run_kwargs: Keyword args passed through to backend.run
        """
        self._qobj = qobj
        self._id = self._qobj.qobj_id
        self._backend = backend
        self._run_args = run_args
        self._run_kwargs = run_kwargs
        self._future = None
        self._result = None
        # Properties that may be populated by the future.
        self.submit_error = None

    def submit(self, executor: Optional[futures.Executor] = None) -> None:
        """Submit the job.

        Args:
            executor: The executor to be used to submit the job.
        """

        # Submit the job in its own future.
        logger.debug("Submitting job %s in future", self._id)
        #self._future = executor.submit(self._backend._run_job, self._id, self._qobj,
        #                               *self._run_args, **self._run_kwargs)
        if executor:
            self._future = executor.submit(self._backend._run_job, self._id, self._qobj,
                                           *self._run_args, **self._run_kwargs)
        else:
            self._future = self._executor.submit(self._backend._run_job, self._id, self._qobj,
                                                 *self._run_args, **self._run_kwargs)
        logger.debug("Job %s future obtained", self._id)

    @property
    @requires_submit
    def future(self) -> futures.Future:
        """Return this job's associated future.

        Returns:
            Future: A future-like object
        """
        return self._future

    @requires_submit
    def result(self, timeout: Optional[float] = None, raises: Optional[bool] = False) -> Result:
        """Return the result of the job.

        Args:
           timeout: Number of seconds to wait for job.
           raises: if True and the job raised an exception
                   this will raise the same exception
        Returns:
            Job result or ``None`` if result could not be retrieved.

        Raises:
            JobError: If the job does not return results before a
                specified timeout.
        """
        if self._result:
            return self._result
        result = None
        if self._future is not None:
            try:
                result = self._future.result(timeout=timeout)
            except JobError as err:
                warnings.warn(
                    "Unable to retrieve job result for job {}".format(self._id))
                if raises:
                    raise err
        self._result = result
        return result

    @requires_submit
    def status(self) -> JobStatus:
        """Query the future for it's status

        Returns:
            Current job status, or ``None`` if an error occurred.
        """
        _status = None
        if self._future.running():
            _status = JobStatus.RUNNING
        elif self._future.cancelled():
            _status = JobStatus.CANCELLED
        elif self._future.done():
            _status = JobStatus.DONE if self._future.exception() is None else JobStatus.ERROR
        else:
            # Note: There is an undocumented Future state: PENDING, that seems to show up when
            # the job is enqueued, waiting for someone to pick it up. We need to deal with this
            # state but there's no public API for it, so we are assuming that if the job is not
            # in any of the previous states, is PENDING, ergo INITIALIZING for us.
            _status = JobStatus.INITIALIZING
        return _status

    @requires_submit
    def cancel(self) -> bool:
        """Attempt to cancel the Job.

        Returns:
            False if the call cannot be cancelled, True otherwise"""
        return self._future.cancel()

    def qobj(self) -> QasmQobj:
        """Return the Qobj submitted for this job.

        Returns:
            Qobj: the Qobj submitted for this job.
        """
        return self._qobj

    def name(self) -> str:
        """Return this job's name.

        Returns:
            Name: str name of this job.
        """
        return self._id
