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

"""This module implements the job class used for AerBackend objects."""

import logging
import warnings

from qiskit.providers import JobV1 as Job
from qiskit.providers import JobStatus, JobError
from .utils import DEFAULT_EXECUTOR, requires_submit

LOGGER = logging.getLogger(__name__)


class AerJob(Job):
    """AerJob class for Qiskit Aer Simulators."""

    def __init__(
        self,
        backend,
        job_id,
        fn,
        qobj=None,
        circuits=None,
        parameter_binds=None,
        run_options=None,
        executor=None,
    ):
        """Initializes the asynchronous job.

        Args:
            backend(AerBackend): the backend used to run the job.
            job_id(str): a unique id in the context of the backend used to run the job.
            fn(function): a callable function to execute qobj on backend.
                This should usually be a bound :meth:`AerBackend._run()` method,
                with the signature `(qobj: QasmQobj, job_id: str) -> Result`.
            qobj(QasmQobj): qobj to execute
            circuits(list of QuantumCircuit): circuits to execute.
                If `qobj` is set, this argument is ignored.
            parameter_binds(list): parameters for circuits.
                If `qobj` is set, this argument is ignored.
            run_options(dict): run_options to execute.
                If `qobj` is set, this argument is ignored.
            executor(ThreadPoolExecutor or dask.distributed.client):
                The executor to be used to submit the job.

        Raises:
            JobError: if no qobj and no circuits.
        """
        super().__init__(backend, job_id)
        self._fn = fn
        if qobj:
            self._qobj = qobj
            self._circuits = None
            self._parameter_binds = None
            self._run_options = None
        elif circuits:
            self._qobj = None
            self._circuits = circuits
            self._parameter_binds = parameter_binds
            self._run_options = run_options
        else:
            raise JobError("AerJob needs a qobj or circuits")
        self._executor = executor or DEFAULT_EXECUTOR
        self._future = None

    def submit(self):
        """Submit the job to the backend for execution.

        Raises:
            QobjValidationError: if the JSON serialization of the Qobj passed
            during construction does not validate against the Qobj schema.
            JobError: if trying to re-submit the job.
        """
        if self._future is not None:
            raise JobError("Aer job has already been submitted.")
        if self._qobj:
            self._future = self._executor.submit(self._fn, self._qobj, self._job_id)
        else:
            self._future = self._executor.submit(
                self._fn, self._circuits, self._parameter_binds, self._run_options, self._job_id
            )

    @requires_submit
    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        """Get job result. The behavior is the same as the underlying
        concurrent Future objects,

        https://docs.python.org/3/library/concurrent.futures.html#future-objects

        Args:
            timeout (float): number of seconds to wait for results.

        Returns:
            qiskit.Result: Result object

        Raises:
            concurrent.futures.TimeoutError: if timeout occurred.
            concurrent.futures.CancelledError: if job cancelled before completed.
        """
        return self._future.result(timeout=timeout)

    @requires_submit
    def cancel(self):
        """Attempt to cancel the job."""
        return self._future.cancel()

    @requires_submit
    def status(self):
        """Gets the status of the job by querying the Python's future

        Returns:
            JobStatus: The current JobStatus

        Raises:
            JobError: If the future is in unexpected state
            concurrent.futures.TimeoutError: if timeout occurred.
        """
        # The order is important here
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

    def backend(self):
        """Return the instance of the backend used for this job."""
        return self._backend

    def qobj(self):
        """Return the Qobj submitted for this job.

        Returns:
            Qobj: the Qobj submitted for this job.
        """
        warnings.warn(
            "`AerJob.qobj() is deprecated as of qiskit-aer 0.12.0`. "
            "Using a qobj for `backend.run()` is deprecated as of qiskit-aer 0.9.0"
            " and will be removed no sooner than 3 months from that release"
            " date. Once it is removed, this `qobj()` returns always `None`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._qobj

    def circuits(self):
        """Return the list of QuantumCircuit submitted for this job.

        Returns:
            list of QuantumCircuit: the list of QuantumCircuit submitted for this job.
        """
        return self._circuits

    def executor(self):
        """Return the executor for this job"""
        return self._executor
