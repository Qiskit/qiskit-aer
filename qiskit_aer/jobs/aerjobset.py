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

# pylint: disable=arguments-differ

"""A set of cluster jobs for Aer."""

from typing import List, Optional, Union, Tuple, Iterable
import time
import logging
import datetime
import uuid
from collections import Counter

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import Schedule
from qiskit.qobj import QasmQobj
from qiskit.providers import JobV1 as Job
from qiskit.providers import JobStatus, JobError
from qiskit.result import Result

from .utils import DEFAULT_EXECUTOR, requires_submit
from .aerjob import AerJob

logger = logging.getLogger(__name__)


class AerJobSet(Job):
    """A set of :class:`~AerJob` classes for Qiskit Aer simulators.

    An instance of this class is returned when you submit experiments with
    executor option.  It provides methods that allow you to interact
    with the jobs as a single entity. For example, you can retrieve the results
    for all of the jobs using :meth:`result()` and cancel all jobs using
    :meth:`cancel()`.
    """

    def __init__(self, backend, job_id, fn, experiments: List[QasmQobj], executor=None):
        """AerJobSet constructor.

        Args:
            backend(Aerbackend): Aerbackend.
            job_id(int): Job Id.
            fn(function): a callable function to execute qobj on backend.
                This should usually be a bound :meth:`AerBackend._run()` method,
                with the signature `(qobj: QasmQobj, job_id: str) -> Result`.
            experiments(List[QasmQobj]): List[QasmQobjs] to execute.
            executor(ThreadPoolExecutor or dask.distributed.client):
                The executor to be used to submit the job.
        """
        super().__init__(backend, job_id)
        self._experiments = experiments

        # Used for caching
        self._future = None
        self._futures = []
        self._results = None
        self._fn = fn
        self._executor = executor or DEFAULT_EXECUTOR
        self._start_time = None
        self._end_time = None
        self._combined_result = []

    def submit(self):
        """Execute this set of jobs on an executor.

        Raises:
            RuntimeError: If the jobs were already submitted.
        """
        if self._futures:
            raise RuntimeError("The jobs for this managed job set have already been submitted.")

        self._future = True
        worker_id = 0
        self._start_time = datetime.datetime.now()
        for experiments in self._experiments:
            _worker_id_list = []
            for exp in experiments:
                job_id = str(uuid.uuid4())
                logger.debug("Job %s submitted", worker_id)
                aer_job = AerJob(self._backend, job_id, self._fn, exp, self._executor)
                aer_job.submit()
                aer_job._future.add_done_callback(self._set_end_time)
                self._futures.append(aer_job)
                _worker_id_list.append(worker_id)
                worker_id = worker_id + 1
            self._combined_result.append(_worker_id_list)

    @requires_submit
    def status(self, worker: Union[None, int, Iterable[int]]) -> Union[JobStatus, List[JobStatus]]:
        """Return the status of each job in this set.

        Args
            worker: Worker id. When None, all workers' statuses are returned.

        Returns:
            A list of job statuses.
        """
        if isinstance(worker, int):
            aer_job = self._futures[worker]
            return aer_job.status()
        elif isinstance(worker, Iterable):
            job_list = []
            for worker_id in worker:
                aer_job = self._futures[worker_id]
                job_list.append(aer_job.status())
            return job_list
        else:
            return [aer.status() for aer in self._futures]

    @requires_submit
    def result(
        self,
        timeout: Optional[float] = None,
    ) -> Result:
        """Return the results of the jobs as a single Result object.

        This call will block until all job results become available or
        the timeout is reached.

        Args:
           timeout: Number of seconds to wait for job results.

        Returns:
            qiskit.Result: Result object

        Raises:
            JobError: if unable to retrieve all job results before the
                specified timeout.

        """
        res = self.worker_results(worker=None, timeout=timeout)
        return res

    @requires_submit
    def worker_results(
        self,
        worker: Union[None, int, Iterable[int]],
        timeout: Optional[float] = None,
    ) -> Union[Result, List[Result]]:
        """Return the result of the jobs specified with worker_id.

        When the worker is None, this call return all worker's result.

        Args:
           worker: Worker id to wait for job result.
           timeout: Number of seconds to wait for job results.

        Returns:
            qiskit.Result: Result object
            instance that can be used to retrieve results
            for individual experiments.

        Raises:
            JobError: if unable to retrieve all job results before the
                specified timeout.
        """

        # We'd like to use futures.as_completed or futures.wait
        #   however this excludes the use of dask as executor
        #   because dask's futures are not ~exactly~ the same.
        res = []

        if isinstance(worker, int):
            res = self._get_worker_result(worker, timeout)
        elif isinstance(worker, Iterable):
            _res = []
            for worker_id in worker:
                _res.append(self._get_worker_result(worker_id, timeout))
            res = self._combine_results(_res)
        else:
            for _worker_id_list in self._combined_result:
                _res = []
                for worker_id in _worker_id_list:
                    _res.append(self._get_worker_result(worker_id, timeout))
                res.append(self._combine_results(_res))

        res = self._accumulate_experiment_results(res)
        return self._combine_job_results(res)

    def _get_worker_result(self, worker: int, timeout: Optional[float] = None):
        """Return the result of the jobs specified with worker_id.

        this call return all worker's result specified worker and
        block until job result become available or the timeout is reached.
        Analogous to dask.client.gather()

        Args:
           worker: Worker id to wait for job result.
           timeout: Number of seconds to wait for job results.

        Returns:
            qiskit.Result: Result object
            instance that can be used to retrieve a result.

        Raises:
            JobError: if unable to retrieve all job results before the
                specified timeout.
        """
        start_time = time.time()
        original_timeout = timeout
        aer_job = self._futures[worker]

        try:
            result = aer_job.result(timeout=timeout)
            if result is None or not result.success:
                if result:
                    logger.warning("AerJobSet %s Error: %s", aer_job.name(), result.header)
                else:
                    logger.warning("AerJobSet %s did not return a result", aer_job.name())
        except JobError:
            raise JobError(
                "Timeout while waiting for the results of experiment {}".format(aer_job.name())
            )

        if timeout:
            timeout = original_timeout - (time.time() - start_time)
            if timeout <= 0:
                raise JobError("Timeout while waiting for JobSet results")
        return result

    def _combine_job_results(self, result_list: List[Result]):
        if len(result_list) == 1:
            return result_list[0]

        master_result = result_list[0]
        _merge_result_list = []

        for _result in result_list[1:]:
            for _master_result, _sub_result in zip(master_result.results, _result.results):
                _merge_result_list.append(self._merge_exp(_master_result, _sub_result))
        master_result.results = _merge_result_list
        return master_result

    def _accumulate_experiment_results(self, results: List[Result]):
        """Merge all experiments into a single in a`Result`

        this function merges the counts and the number of shots
        from each experiment in a `Result` for a noise simulation
        if `id` in metadata field is the same.

        Args:
            results: Result list whose experiments will be combined.

        Returns:
            list: Result list

        Raises:
            JobError: If results do not have count or memory data
        """
        results_list = []
        for each_result in results:
            _merge_results = []
            master_id = None
            master_result = None

            for _result in each_result.results:
                if not hasattr(_result.data, "counts") and not hasattr(_result.data, "memory"):
                    raise JobError("Results do not include counts or memory data")
                meta_data = getattr(_result.header, "metadata", None)
                if meta_data and "id" in meta_data:
                    _id = meta_data["id"]
                    if master_id == _id:
                        master_result = self._merge_exp(master_result, _result)
                    else:
                        master_id = _id
                        master_result = _result
                        _merge_results.append(master_result)
                else:
                    _merge_results.append(_result)
            each_result.results = _merge_results
            results_list.append(each_result)
        return results_list

    def _merge_exp(self, master: Result, sub: Result):
        master.shots = master.shots + sub.shots
        if hasattr(master.data, "counts"):
            master.data.counts = Counter(master.data.counts) + Counter(sub.data.counts)

        if hasattr(master.data, "memory"):
            master.data.memory = master.data.memory + sub.data.memory

        return master

    def _combine_results(self, results: List[Union[Result, None]] = None) -> Result:
        """Combine results from all jobs into a single `Result`.

        Note:
            Since the order of the results must match the order of the initial
            experiments, job results can only be combined if all jobs succeeded.

        Args:
            results: Result will be combined.
        Returns:
            A :class:`~qiskit.result.Result` object that contains results from
                all jobs.
        Raises:
            JobError: If results cannot be combined because some jobs failed.
        """
        if not results:
            raise JobError("Results cannot be combined - no results.")

        # find first non-null result and copy it's config
        _result = next((r for r in results if r is not None), None)

        if _result:
            combined_result = {
                "backend_name": _result.backend_name,
                "backend_version": _result.backend_version,
                "qobj_id": _result.qobj_id,
                "job_id": _result.job_id,
                "success": _result.success,
            }
            combined_result["results"] = []
            if hasattr(_result, "status"):
                combined_result["status"] = _result.status
            if hasattr(_result, "header"):
                combined_result["header"] = _result.header.to_dict()
            combined_result.update(_result._metadata)
        else:
            raise JobError("Results cannot be combined - no results.")

        for each_result in results:
            if each_result is not None:
                combined_result["results"].extend(x.to_dict() for x in each_result.results)

        if self._end_time is None:
            self._end_time = datetime.datetime.now()

        if self._start_time:
            _time_taken = self._end_time - self._start_time
            combined_result["time_taken"] = _time_taken.total_seconds()
        else:
            combined_result["time_taken"] = 0

        combined_result["date"] = datetime.datetime.isoformat(self._end_time)
        return Result.from_dict(combined_result)

    @requires_submit
    def cancel(self) -> None:
        """Cancel all jobs in this job set."""
        for aer_job in self._futures:
            aer_job.cancel()

    @requires_submit
    def job(self, experiment: Union[str, QuantumCircuit, Schedule]) -> Tuple[AerJob, int]:
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
            JobError: If the job for the experiment could not be found.
        """
        worker_index = self.worker(experiment)
        return self.worker_job(worker_index)

    @requires_submit
    def worker(self, experiment: Union[str, QuantumCircuit, Schedule]) -> Union[int, List[int]]:
        """Retrieve the index of job.

        Args:
            experiment: Retrieve the job used to submit this experiment. Several
                types are accepted for convenience:

                    * str: The name of the experiment.
                    * QuantumCircuit: The name of the circuit instance will be used.
                    * Schedule: The name of the schedule instance will be used.

        Returns:
            list or integer value of the job id

        Raises:
            JobError: If the job for the experiment could not be found.
        """

        if isinstance(experiment, (QuantumCircuit, Schedule)):
            experiment = experiment.name
        job_list = []
        for job in self._futures:
            for i, exp in enumerate(job.qobj().experiments):
                if hasattr(exp.header, "name") and exp.header.name == experiment:
                    job_list.append(i)

        if len(job_list) == 1:
            return job_list[0]
        elif len(job_list) > 1:
            return job_list

        raise JobError("Unable to find the job for experiment {}.".format(experiment))

    @requires_submit
    def worker_job(self, worker: Union[None, int, Iterable[int]]) -> Union[AerJob, List[AerJob]]:
        """Retrieve the job specified with job's id

        Args:
            worker: retrive job used to submit with this job id.

        Returns:
            A list of :class:`~qiskit_aer.AerJob`
            instances that represents the submitted jobs.

        Raises:
            JobError: If the job for the experiment could not be found.
        """
        aer_jobs = []
        if isinstance(worker, int):
            return self._futures[worker]
        elif isinstance(worker, Iterable):
            for worker_id in worker:
                aer_jobs.append(self._futures[worker_id])
            return aer_jobs
        else:
            return self._futures

    def _set_end_time(self, future):
        """Set job's end time to calculate "time_taken" value

        Args:
            future(concurrent.futures or dask.distributed.futures): callback future object
        """
        # pylint: disable=unused-argument
        self._end_time = datetime.datetime.now()

    def executor(self):
        """Return the executor for this job"""
        return self._executor
