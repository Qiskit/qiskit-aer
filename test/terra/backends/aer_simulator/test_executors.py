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
AerSimulator options tests
"""

import json
import concurrent.futures
import multiprocessing
import os
import pickle
import tempfile
import unittest

from ddt import ddt
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import Statevector
from qiskit_aer.jobs import utils
from qiskit_aer.noise.noise_model import AerJSONEncoder
from test.terra.reference import ref_kraus_noise
from qiskit_aer.backends.backend_utils import BASIS_GATES
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

# skip fractional gates in multiprocessing transpilation tests until
# https://github.com/Qiskit/qiskit/issues/14002 is resolved
FRACTIONAL_GATES = sorted(["rxx", "rzz", "ryy", "rzx", "cphase", "crx", "cry", "crz"])
STATEVEC_BASIS_GATES = [gate for gate in BASIS_GATES["statevector"] if gate not in FRACTIONAL_GATES]


def run_random_circuits(backend, shots=None, **run_options):
    """Test random circuits on different executor fictures"""
    job_size = 10
    circuits = [random_circuit(num_qubits=2, depth=2, seed=i) for i in range(job_size)]
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


def _run_in_child(backend, circuit, parent_executor, queue):
    """Run a circuit in a forked child and report whether the executor was reset"""
    replaced = utils.DEFAULT_EXECUTOR is not parent_executor
    try:
        backend.run(circuit).result(timeout=30)
        queue.put((replaced, True))
    except Exception:  # pylint: disable=broad-except
        queue.put((replaced, False))


class TestResultSerialization(SimulatorTestCase):
    """Test seriallization of AerJob"""

    def test_aer_job_json_dump(self):
        circuit = QuantumVolume(4, seed=111)
        circuit.measure_all()
        backend = self.backend(method="statevector")
        result = backend.run(transpile(circuit, backend)).result()
        data = json.dumps(result, cls=AerJSONEncoder)
        result_copy = json.loads(data)
        self.compare_counts(result, [circuit], [result_copy["results"][0]["data"]["counts"]])

    def test_aer_job_picklable(self):
        circuit = QuantumVolume(4, seed=111)
        circuit.measure_all()
        backend = self.backend(method="statevector")
        result = backend.run(transpile(circuit, backend)).result()

        with tempfile.TemporaryFile() as f:
            pickle.dump(result, f)
            f.seek(0)
            result_copy = pickle.load(f)

        self.assertEqual(result.get_counts(), result_copy.get_counts())

    @unittest.skipUnless(hasattr(os, "fork"), "requires os.fork")
    def test_default_executor_fork_safe(self):
        """Test that a forked child gets a working DEFAULT_EXECUTOR"""
        circuit = QuantumVolume(4, seed=111)
        circuit.measure_all()
        backend = self.backend(method="statevector")
        circuit = transpile(circuit, backend)
        # Warm up the executor in the parent so that its worker thread exists before the fork
        backend.run(circuit).result()
        parent_executor = utils.DEFAULT_EXECUTOR

        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        proc = ctx.Process(target=_run_in_child, args=(backend, circuit, parent_executor, queue))
        proc.start()
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            self.fail("Forked child did not finish; DEFAULT_EXECUTOR is not fork-safe")
        self.assertEqual(proc.exitcode, 0)
        child_executor_replaced, success = queue.get(timeout=5)
        self.assertTrue(success)
        self.assertTrue(child_executor_replaced)
        self.assertIs(utils.DEFAULT_EXECUTOR, parent_executor)


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
class TestThreadPoolExecutor(CBFixture):
    """Tests of ThreadPool executor"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._test_executor = None
        cls._test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    @supported_methods(["statevector"], [None, 1, 2, 3])
    def test_random_circuits_job(self, method, device, max_job_size):
        """Test random circuits with custom executor."""
        shots = 4000
        backend = self.backend(
            method=method,
            device=device,
            max_job_size=max_job_size,
            basis_gates=STATEVEC_BASIS_GATES,
        )
        result, circuits, targets = run_random_circuits(backend, shots=shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0.05 * shots)

    @supported_methods(["statevector"], [None, 1, 1, 1], [None, 100, 500, 1000])
    def test_noise_circuits_job(self, method, device, max_job_size, max_shot_size):
        """Test random circuits with custom executor."""
        shots = 4000
        backend = self.backend(
            method=method, device=device, max_job_size=max_job_size, max_shot_size=max_shot_size
        )

        circuits = ref_kraus_noise.kraus_gate_error_circuits()
        noise_models = ref_kraus_noise.kraus_gate_error_noise_models()
        targets = ref_kraus_noise.kraus_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(["statevector"], [None, 1, 2, 3])
    def test_result_time_val(self, method, device, max_job_size):
        """Test random circuits with custom executor."""
        shots = 4000
        backend = self.backend(
            method=method,
            device=device,
            max_job_size=max_job_size,
            basis_gates=STATEVEC_BASIS_GATES,
        )
        result, _, _ = run_random_circuits(backend, shots=shots)
        self.assertSuccess(result)
        self.assertGreaterEqual(result.time_taken, 0)
