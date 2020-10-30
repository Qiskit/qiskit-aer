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

"""Manager for Qiskit Aer cluster-backed simulations."""
from typing import Any, Union, List
from concurrent import futures

from qiskit import assemble, QuantumCircuit
from qiskit.qobj import QasmQobj
from qiskit.pulse import Schedule
from ..backends.aerbackend import AerBackend
from .clusterjobset import JobSet
from .utils import methdispatch, split


class AerClusterManager:
    """Manager for Qiskit Aer cluster-backed simulations.

    The AerClusterManager is a higher level mechanism for running
    multiple circuits or pulse schedules. It connects to an existing
    cluster (or creates a local one) and submits circuits individually
    rather than building a single qobj to be submitted. When the jobs are
    finished, it collects and presents the results in a unified view.

    You can use the :meth:`run()` method to submit multiple experiments
    with the Manager::

        from qiskit import Aer, transpile
        from qiskit.providers.aer.cluster import AerClusterManager
        from qiskit.circuit.random import random_circuit

        backend = Aer.get_backend('qasm_simulator')

        # Build a thousand circuits.
        circs = []
        for _ in range(1000):
            circs.append(random_circuit(num_qubits=5, depth=4, measure=True))

        # Need to transpile the circuits first.
        circs = transpile(circs, backend=backend)

        # Use Job Manager to break the circuits into multiple jobs.
        job_manager = AerClusterManager(shots=4000)
        job_set_foo = job_manager.run(circs, backend=backend, name='foo')

    The :meth:`run()` method returns a :class:`JobSet` instance, which
    represents the set of jobs for the experiments. You can use the
    :class:`JobSet` methods, such as :meth:`statuses()<JobSet.statuses>`,
    :meth:`results()<JobSet.results>`, and
    :meth:`error_messages()<JobSet.error_messages>` to get a combined
    view of the jobs in the set.
    For example::

        results = job_set_foo.results()
        results.get_counts(5)  # Counts for experiment 5.

    """

    def __init__(self, executor: futures.Executor, **assemble_config: Any) -> None:
        """ClusterManager Constructor."""
        self._exec = executor
        self._assemble_config = assemble_config

        self._job_sets = []

    def set_assemble_config(self, **assemble_config: Any) -> None:
        """Set the arguments passed to assemble on subsequent calls to run."""
        self._assemble_config = assemble_config
        return self

    def run(self,
            backend: AerBackend,
            exp: Union[List[Union[QuantumCircuit, Schedule]], QasmQobj],
            *run_args: Any,
            **run_kwargs: Any) -> JobSet:
        """Execute a qobj on a backend.

        Args:
            backend: Backend to execute the experiments on.
            qobj: QasmQobj to be executed
            *run_args: Positional arguments passed through to the backend.run
            **run_kwargs: Keyword arguments to be passed through to the backend.run

        Returns:
            A :class:`JobSet` instance representing the set of simulation jobs.

        Raises:
            ValueError: If the qobj/backend are incompatible
        """
        if not isinstance(backend, AerBackend):
            raise ValueError(
                "AerClusterManager only supports AerBackends. "
                "{} is not an AerBackend.".format(backend))

        if isinstance(exp, QasmQobj):
            experiments = split(exp)
        elif isinstance(exp, (QuantumCircuit, Schedule)):
            experiments = [assemble(exp, backend, **self._assemble_config)]
        elif (
            isinstance(exp, list) and all(isinstance(e, QuantumCircuit) for e in exp) or
            isinstance(exp, list) and all(isinstance(e, Schedule) for e in exp)
        ):
            experiments = [assemble(e, backend, **self._assemble_config) for e in exp]
        else:
            raise ValueError(
                "run() is not implemented for this type of experiment ({})".format(str(type(exp))))

        job_set = JobSet(experiments)
        job_set.run(self._exec, backend, *run_args, **run_kwargs)
        self._job_sets.append(job_set)

        return job_set
