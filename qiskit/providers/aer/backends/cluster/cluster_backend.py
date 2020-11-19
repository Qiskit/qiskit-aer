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

import copy
import logging
from typing import Any, Union, List
from concurrent import futures

from qiskit import assemble, QuantumCircuit
from qiskit.qobj import QasmQobj
from qiskit.pulse import Schedule
from .clusterjobset import JobSet
from .utils import methdispatch, split


from ..aerbackend import AerBackend
#from ..cluster.clusterjobset import JobSet

logger = logging.getLogger(__name__)


class ClusterBackend(AerBackend):
    """
    """

    def __init__(self,
                 backend,
                 executor,
                 provider=None,
                 **assemble_config):
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
        return self._executor

    @executor.setter
    def set_executor(self, executor):
        self._executor = executor

    @property
    def backend(self):
        return self._backend_impl

    @backend.setter
    def set_backend(self, backend):
        raise NotImplementedError("Cannot dynamically set the backend_impl")

    def set_assemble_config(self, **assemble_config: Any) -> None:
        """Set the arguments passed to assemble on subsequent calls to run."""
        self._assemble_config = assemble_config
        return self

    def run(self,
            exp: Union[List[Union[QuantumCircuit, Schedule]], QasmQobj],
            *run_args: Any,
            **run_kwargs: Any) -> JobSet:
        """Execute a qobj on a backend.

        Args:
            qobj: QasmQobj to be executed
            *run_args: Positional arguments passed through to the backend.run
            **run_kwargs: Keyword arguments to be passed through to the backend.run

        Returns:
            A :class:`JobSet` instance representing the set of simulation jobs.

        Raises:
            ValueError: If the qobj/backend are incompatible
        """
        if isinstance(exp, QasmQobj):
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

    def _execute(self, qobj):
        pass
