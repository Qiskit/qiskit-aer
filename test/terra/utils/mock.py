# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""
Utilities for mocking the IBMQ provider, including job responses and backends.

The module includes dummy provider, backends, and jobs. The purpose of
these classes is to trick backends for testing purposes:
testing local timeouts, arbitrary responses or behavior, etc.

The mock devices are mainly for testing the compiler.
"""

import uuid
import logging
from concurrent import futures
import time

from qiskit.result import Result
from qiskit.providers import BaseBackend, BaseJob
from qiskit.providers.models import BackendProperties, BackendConfiguration
from qiskit.providers.models.backendconfiguration import GateConfig
from qiskit.qobj import (QasmQobj, QobjExperimentHeader, QobjHeader,
                         QasmQobjInstruction, QasmQobjExperimentConfig,
                         QasmQobjExperiment, QasmQobjConfig)
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.baseprovider import BaseProvider
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.aer import AerError


logger = logging.getLogger(__name__)


class FakeProvider(BaseProvider):
    """Dummy provider just for testing purposes.

    Only filtering backends by name is implemented.
    """

    def get_backend(self, name=None, **kwargs):
        backend = self._backends[0]
        if name:
            filtered_backends = [backend for backend in self._backends
                                 if backend.name() == name]
            if not filtered_backends:
                raise QiskitBackendNotFoundError()
            else:
                backend = filtered_backends[0]
        return backend

    def backends(self, name=None, **kwargs):
        return self._backends

    def __init__(self):
        # TODO Add the rest of simulators that we want to mock
        self._backends = [FakeSuccessQasmSimulator(),
                          FakeFailureQasmSimulator()]
        super().__init__()


class FakeBackend(BaseBackend):
    """This is a dummy backend just for testing purposes."""

    def __init__(self, configuration, time_alive=10):
        """
        Args:
            configuration (BackendConfiguration): backend configuration
            time_alive (int): time to wait before returning result
        """
        super().__init__(configuration)
        self.time_alive = time_alive

    def properties(self):
        """Return backend properties"""
        properties = {
            'backend_name': self.name(),
            'backend_version': self.configuration().backend_version,
            'last_update_date': '2000-01-01 00:00:00Z',
            'qubits': [[{'name': 'TODO', 'date': '2000-01-01 00:00:00Z',
                         'unit': 'TODO', 'value': 0}]],
            'gates': [{'qubits': [0], 'gate': 'TODO',
                       'parameters':
                           [{'name': 'TODO', 'date': '2000-01-01 00:00:00Z',
                             'unit': 'TODO', 'value': 0}]}],
            'general': []
        }

        return BackendProperties.from_dict(properties)

    def run(self, qobj):
        job_id = str(uuid.uuid4())
        job = FakeJob(self, self.run_job, job_id, qobj)
        job.submit()
        return job

    # pylint: disable=unused-argument
    def run_job(self, job_id, qobj):
        """Main dummy run loop"""
        time.sleep(self.time_alive)

        return Result.from_dict({
            'job_id': job_id,
            'backend_name': self.name(),
            'backend_version': self.configuration().backend_version,
            'qobj_id': qobj.qobj_id,
            'results': [],
            'status': 'COMPLETED',
            'success': True
        })


class FakeSuccessQasmSimulator(FakeBackend):
    """A fake QASM simulator backend that always returns SUCCESS"""

    def __init__(self, time_alive=10):
        configuration = BackendConfiguration(
            backend_name='fake_success_qasm_simulator',
            backend_version='0.0.0',
            n_qubits=5,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z',
                         'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap',
                         'snapshot', 'unitary'],
            simulator=True,
            local=True,
            conditional=True,
            open_pulse=False,
            memory=True,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')]
        )

        super().__init__(configuration, time_alive=time_alive)


class FakeFailureQasmSimulator(FakeBackend):
    """A fake simulator backend."""

    def __init__(self, time_alive=10):
        configuration = BackendConfiguration(
            backend_name='fake_failure_qasm_simulator',
            backend_version='0.0.0',
            n_qubits=5,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z',
                         'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap',
                         'snapshot', 'unitary'],
            simulator=True,
            local=True,
            conditional=True,
            open_pulse=False,
            memory=True,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')]
        )

        super().__init__(configuration, time_alive=time_alive)


    # pylint: disable=unused-argument
    def run_job(self, job_id, qobj):
        """Main dummy run loop"""
        time.sleep(self.time_alive)

        raise AerError("Mocking a failure in the QASM Simulator")

class FakeJob(BaseJob):
    """Fake simulator job"""
    _executor = futures.ThreadPoolExecutor()

    def __init__(self, backend, fn, job_id, qobj):
        super().__init__(backend, job_id)
        self._backend = backend
        self._job_id = job_id
        self._qobj = qobj
        self._future = None
        self._future_callback = fn

    def submit(self):
        self._future = self._executor.submit(
            self._future_callback, self._job_id, self._qobj
        )

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        return self._future.result(timeout=timeout)

    def cancel(self):
        return self._future.cancel()

    def status(self):
        if self._running:
            _status = JobStatus.RUNNING
        elif not self._done:
            _status = JobStatus.QUEUED
        elif self._cancelled:
            _status = JobStatus.CANCELLED
        elif self._done:
            _status = JobStatus.DONE
        elif self._error:
            _status = JobStatus.ERROR
        else:
            raise Exception('Unexpected state of {0}'.format(
                self.__class__.__name__))
        _status_msg = None
        return {'status': _status,
                'status_msg': _status_msg}

    def job_id(self):
        return self._job_id

    def backend(self):
        return self._backend

    @property
    def _cancelled(self):
        return self._future.cancelled()

    @property
    def _done(self):
        return self._future.done()

    @property
    def _running(self):
        return self._future.running()

    @property
    def _error(self):
        return self._future.exception(timeout=0)


def new_fake_qobj():
    """Create fake `Qobj` and backend instances."""
    backend = FakeQasmSimulator()
    return QasmQobj(
        qobj_id='test-id',
        config=QasmQobjConfig(shots=1024, memory_slots=1, max_credits=100),
        header=QobjHeader(backend_name=backend.name()),
        experiments=[QasmQobjExperiment(
            instructions=[
                QasmQobjInstruction(name='barrier', qubits=[1])
            ],
            header=QobjExperimentHeader(),
            config=QasmQobjExperimentConfig(seed=123456)
        )]
    )
