# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimualtor options tests
"""
import logging
from math import ceil
import concurrent.futures

from ddt import ddt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.providers.aer.jobs import AerJob, AerJobSet
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)


DASK = False

try:
    from dask.distributed import LocalCluster, Client
    DASK = True
except ImportError:
    DASK = False


def run_random_circuits(backend, shots=None, **run_options):
    """Test random circuits on different executor fictures"""
    job_size = 10
    circuits = [random_circuit(num_qubits=2, depth=2, seed=i)
                for i in range(job_size)]
    # Sample references counts
    targets = []
    for circ in circuits:
        state = Statevector(circ)
        state.seed = 101
        targets.append(state.sample_counts(shots=shots))
    
    # Add measurements for simulation
    for circ in circuits:
        circ.measure_all()

    circuits = transpile(circuits, backend)
    job = backend.run(circuits, shots=shots, **run_options)
    result = job.result()
    return result, circuits, targets


class CBFixture(SimulatorTestCase):
    """Extension tests for Aerbackend with cluster backend"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        """Override me with an executor init."""
        cls._test_executor = None

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls._test_executor:
            cls._test_executor.shutdown()

    def backend(self, **options):
        """Return AerSimulator backend using current class options"""
        return super().backend(executor=self._test_executor, **options)     


@ddt
class TestDaskExecutor(CBFixture):
    """Tests of Dask executor"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if DASK:
            cls._test_executor = Client(address=LocalCluster(n_workers=1, processes=True))

    def setUp(self):
        super().setUp()
        if not DASK:
            self.skipTest('Dask not installed, skipping ClusterBackend-dask tests')

    @supported_methods(['statevector'], [None, 1, 2, 3])
    def test_random_circuits_job(self, method, device, max_job_size):
        """Test random circuits with custom executor."""
        shots = 4000
        backend = self.backend(
            method=method, device=device, max_job_size=max_job_size)
        result, circuits, targets = run_random_circuits(backend, shots=shots)    
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0.05 * shots)

@ddt
class TestThreadPoolExecutor(CBFixture):
    """Tests of ThreadPool executor"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._test_executor = None
        cls._test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    @supported_methods(['statevector'], [None, 1, 2, 3])
    def test_random_circuits_job(self, method, device, max_job_size):
        """Test random circuits with custom executor."""
        shots = 4000
        backend = self.backend(
            method=method, device=device, max_job_size=max_job_size)
        result, circuits, targets = run_random_circuits(backend, shots=shots)    
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0.05 * shots)   
