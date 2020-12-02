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

import logging
from typing import Any, Union, List

from qiskit import assemble, QuantumCircuit
from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.pulse import Schedule
from qiskit.providers.aer import AerProvider
from .clusterjobset import JobSet
from .utils import split
from ..aerbackend import AerBackend

logger = logging.getLogger(__name__)


class ClusterBackend(AerBackend):
    """
    Backend which can (and must) be configured at initialization with another
    pre-configured simulator instance and an executor which will be used to 
    submit experiments.
    A simulator wrapped in a ClusterBackend overrides the usual run(..)
    method, in that it submits each circuit to the executor independently and
    returns a JobSet object rather than a (atomic) Job.
    A JobSet can be treated as a single Job - e.g. JobSet.result() functions
    the same - but they also add functionality for interacting with the
    constituent jobs.
    Upon initialization this backend inherits all of its wrapped backend's
    attributes, and obeys (to a point) the AerBackend interface allowing
    wrapped backends to be 'drop-in' replacements for ones currently in use.

    For example:

    .. code-block:: python

        backend_impl = Aer.get_backend('qasm_simulator')
        executor = futures.ThreadPoolExecutor(max_workers=1)
        backend = ClusterBackend(backend_impl, executor)
        circs = ...
        qobj = assemble(circs, backend=backend)
        result = backend.run(...).result()

    """

    def __init__(self,
                 backend: AerBackend,
                 executor: Any,
                 provider: AerProvider = None,
                 **assemble_config: Any):
        if not isinstance(backend, AerBackend):
            raise ValueError(
                "AerClusterManager only supports AerBackends. "
                "{} is not an AerBackend.".format(backend))

        super().__init__(configuration=backend.configuration(),
                         properties=backend.properties(),
                         available_methods=backend.available_methods(),
                         defaults=backend.defaults(),
                         backend_options=backend.options,
                         provider=provider)
        self._executor = executor
        self._backend_impl = backend
        self._assemble_config = assemble_config

    @property
    def executor(self):
        """Return the currently configured executor."""
        return self._executor

    @executor.setter
    def set_executor(self, executor: Any):
        """Set the executor to be used in future run() calls."""
        self._executor = executor

    @property
    def backend(self):
        """Return the wrapped backend."""
        return self._backend_impl

    @backend.setter
    def set_backend(self, backend: AerBackend):
        """(Attempt to) Set the wrapped backend."""
        raise NotImplementedError("Backend must be configured at initialization")

    def set_assemble_config(self, **assemble_config: Any) -> None:
        """Set the arguments passed to assemble on subsequent calls to run."""
        self._assemble_config = assemble_config
        return self

    # pylint: disable=arguments-differ
    def run(self,
            exp: Union[List[Union[QuantumCircuit, Schedule]], QasmQobj],
            *run_args: Any,
            **run_kwargs: Any) -> JobSet:
        """Execute an experiment on the configured backend.

        This backend splits up experiments and submits multiple individual jobs
        to the executor.
        A JobSet is returned and can be used to interact with the jobs, either
        as a whole (single) or individually.
        The wrapped backend's controller function is used and any relevant
        configuration/experiment-specific information is taken from the wrapped
        backend(_impl).

        Args:
            exp: Experiment to be executed
            *run_args: Positional arguments passed through to the backend.run
            **run_kwargs: Keyword arguments to be passed through to the backend.run

        Returns:
            A :class:`JobSet` instance representing the set of simulation jobs.

        Raises:
            ValueError: If the qobj/backend are incompatible
        """
        if isinstance(exp, (QasmQobj, PulseQobj)):
            experiments = split(exp)
        elif isinstance(exp, (QuantumCircuit, Schedule)):
            experiments = [assemble(exp, self.backend, **self._assemble_config)]
        elif (
                isinstance(exp, list) and all(isinstance(e, QuantumCircuit) for e in exp) or
                isinstance(exp, list) and all(isinstance(e, Schedule) for e in exp)
        ):
            experiments = [assemble(e, self.backend, **self._assemble_config) for e in exp]
        else:
            raise ValueError(
                "run() is not implemented for this type of experiment ({})".format(str(type(exp))))

        job_set = JobSet(experiments)
        job_set.run(self.executor, self.backend, *run_args, **run_kwargs)
        return job_set

    def _execute(self):
        """Empty implementation of non-essential (because we override run instead) abstract method."""
        pass
