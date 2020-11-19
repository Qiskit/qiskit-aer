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

"""A set of jobs being managed by the :class:`AerClusterManager`."""

from typing import List, Optional, Union, Any, Tuple
from concurrent import futures
import time
import logging
import uuid

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import Schedule
from qiskit.qobj import QasmQobj
from qiskit.providers.jobstatus import JobStatus

from ..aerbackend import AerBackend
from .clusterjob import CJob
from .clusterresults import CResults
from .exceptions import AerClusterTimeoutError, AerClusterJobNotFound
from .utils import requires_submit

logger = logging.getLogger(__name__)


class JobSet:
    """A set of cluster jobs.

    An instance of this class is returned when you submit experiments using
    :meth:`AerClusterManager.run()`.
    It provides methods that allow you to interact
    with the jobs as a single entity. For example, you can retrieve the results
    for all of the jobs using :meth:`results()` and cancel all jobs using
    :meth:`cancel()`.
    """

    def __init__(self, experiments: List[QasmQobj], name: Optional[str] = None):
        """JobSet constructor.

        Args:
            experiments: List[QasmQobjs] to execute.
            name: Name for this set of jobs.
        """
        self._name = name or str(uuid.uuid4())
        self._experiments = experiments

        # Used for caching
        self._future = None
        self._futures = []
        self._results = None
        self._error_msg = None

    def run(self,
            executor: futures.Executor,
            backend: AerBackend,
            *run_args: Any,
            **run_kwargs: Any) -> None:
        """Execute this set of jobs on an executor.

        Args:
            executor: The executor used to submit jobs asynchronously.
            backend: The backend to run the jobs on.
            *run_args: Positional arguments passed through to backend.run.
            **run_kwargs: Keyword arguments passed through to backend.run.

        Raises:
            RuntimeError: If the jobs were already submitted.
        """
        if self._futures:
            raise RuntimeError(
                'The jobs for this managed job set have already been submitted.')

        self._future = True
        total_jobs = len(self._experiments)
        for i, exp in enumerate(self._experiments):
            cjob = CJob(backend, exp, *run_args, **run_kwargs)
            cjob.submit(executor=executor)
            logger.debug("Job %s submitted", i + 1)
            self._futures.append(cjob)

    @requires_submit
    def statuses(self) -> List[Union[JobStatus]]:
        """Return the status of each job in this set.

        Returns:
            A list of job statuses.
        """
        return [cjob.status() for cjob in self._futures]

    @requires_submit
    def results(
            self,
            timeout: Optional[float] = None,
            raises: Optional[bool] = False
    ) -> CResults:
        """Return the results of the jobs.

        This call will block until all job results become available or
        the timeout is reached. Analogous to dask.client.gather()

        Args:
           timeout: Number of seconds to wait for job results.
           raises: whether or not to re-raise exceptions thrown by jobs

        Returns:
            A :class:`CResults`
            instance that can be used to retrieve results
            for individual experiments.

        Raises:
            AerClusterTimeoutError: if unable to retrieve all job results before the
                specified timeout.
        """
        if self._results is not None:
            return self._results

        success = True
        start_time = time.time()
        original_timeout = timeout

        # We'd like to use futures.as_completed or futures.wait
        #   however this excludes the use of dask as executor
        #   because dask's futures are not ~exactly~ the same.
        for cjob in self._futures:
            try:
                result = cjob.result(timeout=timeout, raises=raises)
                if result is None or not result.success:
                    logger.warning('ClusterJob %s failed.', cjob.name())
                    success = False
            except AerClusterTimeoutError as ex:
                raise AerClusterTimeoutError(
                    'Timeout while waiting for the results of experiment {}'.format(
                        cjob.name())) from ex
            if timeout:
                timeout = original_timeout - (time.time() - start_time)
                if timeout <= 0:
                    raise AerClusterTimeoutError(
                        "Timeout while waiting for JobSet results")

        self._results = CResults(self, success)

        return self._results

    @requires_submit
    def cancel(self) -> None:
        """Cancel all jobs in this job set."""
        for cjob in self._futures:
            cjob.cancel()

    @requires_submit
    def job(self, experiment: Union[str, QuantumCircuit, Schedule]) -> Tuple[CJob, int]:
        """Retrieve the job used to submit the specified experiment and its index.

        Args:
            experiment: Retrieve the job used to submit this experiment. Several
                types are accepted for convenience:

                    * str: The name of the experiment.
                    * QuantumCircuit: The name of the circuit instance will be used.
                    * Schedule: The name of the schedule instance will be used.

        Returns:
            A tuple of the job used to submit the experiment and the experiment index.

        Raises:
            AerClusterJobNotFound: If the job for the experiment could not
                be found.
        """
        if isinstance(experiment, (QuantumCircuit, Schedule)):
            experiment = experiment.name
        for job in self.jobs():
            for i, exp in enumerate(job.qobj().experiments):
                if hasattr(exp.header, 'name') and exp.header.name == experiment:
                    return job, i

        raise AerClusterJobNotFound(
            'Unable to find the job for experiment {}.'.format(experiment))

    @requires_submit
    def jobs(self) -> List[Union[CJob, None]]:
        """Return jobs in this job set.

        Returns:
            A list of :class:`~qiskit.providers.aer.cluster.CJob`
            instances that represents the submitted jobs.
            An entry in the list is ``None`` if the job failed to be submitted.
        """
        return self._futures

    def name(self) -> str:
        """Return the name of this job set.

        Returns:
            Name of this job set.
        """
        return self._name

    def managed_jobs(self) -> List[CJob]:
        """Return the managed jobs in this set.

        Returns:
            A list of managed jobs.
        """
        return self._futures
