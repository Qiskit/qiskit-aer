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
AerSimulator Integration Tests
"""

import multiprocessing
import psutil
from ddt import ddt, data

from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import pauli_error
from test.terra.decorators import requires_omp, requires_multiprocessing
from test.terra.backends.simulator_test_case import SimulatorTestCase


@ddt
class TestThreadManagement(SimulatorTestCase):
    """AerSimulator thread tests."""

    def backend_options_parallel(
        self, total_threads=None, state_threads=None, shot_threads=None, exp_threads=None
    ):
        """Backend options with thread manangement."""
        opts = {}
        if total_threads is not None:
            opts["max_parallel_threads"] = total_threads
        else:
            opts["max_parallel_threads"] = 0
        if shot_threads is not None:
            opts["max_parallel_shots"] = shot_threads
        if state_threads is not None:
            opts["max_parallel_state_update"] = state_threads
        if exp_threads is not None:
            opts["max_parallel_experiments"] = exp_threads
        return opts

    def dummy_noise_model(self):
        """Return dummy noise model for dummy circuit"""
        noise_model = NoiseModel()
        error = pauli_error([("X", 0.25), ("I", 0.75)])
        noise_model.add_all_qubit_quantum_error(error, "x")
        return noise_model

    def dummy_circuit(self, num_qubits):
        """Dummy circuit for testing thread settings"""
        circ = QuantumCircuit(num_qubits, num_qubits)
        circ.x(range(num_qubits))
        circ.measure(range(num_qubits), range(num_qubits))
        return circ

    def measure_in_middle_circuit(self, num_qubits):
        """Dummy circuit for testing thread settings"""
        circ = QuantumCircuit(num_qubits, num_qubits)
        circ.measure(range(num_qubits), range(num_qubits))
        circ.x(range(num_qubits))
        circ.measure(range(num_qubits), range(num_qubits))
        return circ

    def threads_used(self, result):
        """Return a list of threads used for each execution"""
        exp_threads = getattr(result, "metadata", {}).get("parallel_experiments", 1)
        threads = []
        for exp_result in getattr(result, "results", []):
            exp_meta = getattr(exp_result, "metadata", {})
            shot_threads = exp_meta.get("parallel_shots", 1)
            state_threads = exp_meta.get("parallel_state_update", 1)
            threads.append(
                {
                    "experiments": exp_threads,
                    "shots": shot_threads,
                    "state_update": state_threads,
                    "total": exp_threads * shot_threads * state_threads,
                }
            )
        return threads

    def test_max_memory_settings(self):
        """test max memory configuration"""
        backend = self.backend(**self.backend_options_parallel())
        circuit = transpile(QuantumVolume(4, 1, seed=0), backend)
        circuit.measure_all()
        system_memory = int(psutil.virtual_memory().total / 1024 / 1024)

        result = backend.run(circuit, shots=100).result()
        max_mem_result = result.metadata.get("max_memory_mb")
        self.assertGreaterEqual(
            max_mem_result, int(system_memory / 2), msg="Default 'max_memory_mb' is too small."
        )
        self.assertLessEqual(
            max_mem_result, system_memory, msg="Default 'max_memory_mb' is too large."
        )

    def test_custom_memory_settings(self):
        """test max memory configuration"""
        max_mem_target = 128
        backend = self.backend(max_memory_mb=max_mem_target, **self.backend_options_parallel())

        circuit = transpile(QuantumVolume(4, 1, seed=0), backend)
        circuit.measure_all()
        result = backend.run(circuit, shots=100).result()
        max_mem_result = result.metadata.get("max_memory_mb")
        self.assertEqual(
            max_mem_result, max_mem_target, msg="Custom 'max_memory_mb' is not being set correctly."
        )

    def available_threads(self):
        """ "Return the threads reported by the simulator"""
        backend = self.backend(**self.backend_options_parallel())
        result = backend.run(self.dummy_circuit(1), shots=1).result()
        return self.threads_used(result)[0]["total"]

    @requires_omp
    @requires_multiprocessing
    def test_parallel_defaults_single_ideal(self):
        """Test parallel thread assignment defaults"""

        backend = self.backend(**self.backend_options_parallel())
        max_threads = self.available_threads()

        # Test single circuit, no noise
        # Parallel experiments and shots should always be 1
        result = backend.run(self.dummy_circuit(1), shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": 1,
                "state_update": max_threads,
                "total": max_threads,
            }
            self.assertEqual(threads, target, msg="single circuit ideal case")

    @requires_omp
    @requires_multiprocessing
    def test_parallel_defaults_single_noise(self):
        """Test parallel thread assignment defaults"""
        backend = self.backend(
            method="statevector",
            noise_model=self.dummy_noise_model(),
            **self.backend_options_parallel(),
        )
        max_threads = self.available_threads()

        # Test single circuit, with noise
        # Parallel experiments should always be 1
        # parallel shots should be greater than 1
        result = backend.run(self.dummy_circuit(1), shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": max_threads,
                "state_update": 1,
                "total": max_threads,
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_defaults_single_meas(self):
        """Test parallel thread assignment defaults"""
        backend = self.backend(**self.backend_options_parallel())
        max_threads = self.available_threads()

        # Test single circuit, with measure in middle, no noise
        # Parallel experiments should always be 1
        # parallel shots should be greater than 1
        result = backend.run(self.measure_in_middle_circuit(1), shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": max_threads,
                "state_update": 1,
                "total": max_threads,
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_defaults_multi_ideal(self):
        """Test parallel thread assignment defaults"""
        backend = self.backend(**self.backend_options_parallel())
        max_threads = self.available_threads()

        # Test multiple circuit, no noise
        # Parallel experiments always be 1
        # parallel shots should always be 1
        result = backend.run(max_threads * [self.dummy_circuit(1)], shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": 1,
                "state_update": max_threads,
                "total": max_threads,
            }
            self.assertEqual(threads, target, msg="multiple circuits ideal case")

    @requires_omp
    @requires_multiprocessing
    def test_parallel_defaults_multi_noise(self):
        """Test parallel thread assignment defaults"""
        backend = self.backend(
            method="statevector",
            noise_model=self.dummy_noise_model(),
            **self.backend_options_parallel(),
        )
        max_threads = self.available_threads()

        # Test multiple circuits, with noise
        # Parallel experiments should always be 1
        # parallel shots should be greater than 1
        result = backend.run(
            max_threads * [self.dummy_circuit(1)],
            shots=10 * max_threads,
            noise_model=self.dummy_noise_model(),
        ).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": max_threads,
                "state_update": 1,
                "total": max_threads,
            }
            self.assertEqual(threads, target, msg="multiple circuits noise case")

    @requires_omp
    @requires_multiprocessing
    def test_parallel_defaults_multi_meas(self):
        """Test parallel thread assignment defaults"""

        backend = self.backend(
            noise_model=self.dummy_noise_model(), **self.backend_options_parallel()
        )
        max_threads = self.available_threads()

        # Test multiple circuit, with measure in middle, no noise
        # Parallel experiments should always be 1
        # parallel shots should be greater than 1
        result = backend.run(
            max_threads * [self.measure_in_middle_circuit(1)], shots=10 * max_threads
        ).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": max_threads,
                "state_update": 1,
                "total": max_threads,
            }
            self.assertEqual(threads, target, msg="multiple circuits no meas opt")

    @requires_omp
    @requires_multiprocessing
    @data(0, 1, 2, 4)
    def test_parallel_thread_assignment(self, custom_max_threads):
        """Test parallel thread assignment priority"""

        # If we set all values to max test output
        # We intentionally set the max shot and experiment threads to
        # twice the max threads to check they are limited correctly
        parallel_opts = self.backend_options_parallel(
            total_threads=custom_max_threads,
            shot_threads=2 * custom_max_threads,
            exp_threads=2 * custom_max_threads,
        )

        # Calculate actual max threads from custom max and CPU number
        max_threads = self.available_threads()
        if custom_max_threads > 0:
            max_threads = min(max_threads, custom_max_threads)
        shots = 10 * max_threads

        with self.subTest(msg="single circuit, no noise"):
            # Test single circuit, no noise
            # Parallel experiments and shots should always be 1
            backend = self.backend(**parallel_opts)
            circuits = self.dummy_circuit(1)
            result = backend.run(circuits, shots=shots).result()
            for threads in self.threads_used(result):
                target = {
                    "experiments": 1,
                    "shots": 1,
                    "state_update": max_threads,
                    "total": max_threads,
                }
                self.assertEqual(threads, target, msg="single, no noise")

        with self.subTest(msg="single circuit, noise"):
            # Test single circuit, with noise
            # Parallel experiments should always be 1
            # parallel shots should be greater than 1
            backend = self.backend(
                method="statevector", noise_model=self.dummy_noise_model(), **parallel_opts
            )
            circuits = self.dummy_circuit(1)
            result = backend.run(circuits, shots=shots).result()
            for threads in self.threads_used(result):
                target = {
                    "experiments": 1,
                    "shots": max_threads,
                    "state_update": 1,
                    "total": max_threads,
                }
                self.assertEqual(threads, target, msg="single, noise")

        with self.subTest(msg="single circuit, middle meas"):
            # Test single circuit, with measure in middle, no noise
            # Parallel experiments should always be 1
            # parallel shots should be greater than 1
            backend = self.backend(**parallel_opts)
            circuits = self.measure_in_middle_circuit(1)
            result = backend.run(circuits, shots=shots).result()
            for threads in self.threads_used(result):
                target = {
                    "experiments": 1,
                    "shots": max_threads,
                    "state_update": 1,
                    "total": max_threads,
                }
                self.assertEqual(threads, target, msg="single, meas")

        with self.subTest(msg="multiple circuit, no noise"):
            # Test multiple circuit, no noise
            # Parallel experiments always be greater than 1
            # parallel shots should always be 1
            backend = self.backend(**parallel_opts)
            circuits = max_threads * [self.dummy_circuit(1)]
            result = backend.run(circuits, shots=shots).result()
            for threads in self.threads_used(result):
                target = {
                    "experiments": max_threads,
                    "shots": 1,
                    "state_update": 1,
                    "total": max_threads,
                }
                self.assertEqual(threads, target, msg="multiple, no noise")

        with self.subTest(msg="multiple circuit, noise"):
            # Test multiple circuits, with noise
            # Parallel experiments always be greater than 1
            # parallel shots should always be 1
            backend = self.backend(noise_model=self.dummy_noise_model(), **parallel_opts)
            circuits = max_threads * [self.dummy_circuit(1)]
            result = backend.run(circuits, shots=shots).result()
            for threads in self.threads_used(result):
                target = {
                    "experiments": max_threads,
                    "shots": 1,
                    "state_update": 1,
                    "total": max_threads,
                }
                self.assertEqual(threads, target, msg="multiple, noise")

        with self.subTest(msg="multiple circuit, middle meas"):
            # Test multiple circuit, with measure in middle, no noise
            # Parallel experiments always be greater than 1
            # parallel shots should always be 1
            backend = self.backend(**parallel_opts)
            circuits = max_threads * [self.measure_in_middle_circuit(1)]
            result = backend.run(circuits, shots=shots).result()
            for threads in self.threads_used(result):
                target = {
                    "experiments": max_threads,
                    "shots": 1,
                    "state_update": 1,
                    "total": max_threads,
                }
                self.assertEqual(threads, target, msg="multiple, meas")

    @requires_omp
    @requires_multiprocessing
    def test_parallel_experiment_thread_single(self):
        """Test parallel experiment thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(**self.backend_options_parallel(exp_threads=max_threads))

        # Test single circuit
        # Parallel experiments and shots should always be 1
        result = backend.run(self.dummy_circuit(1), shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": 1,
                "state_update": max_threads,
                "total": max_threads,
            }
        self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_experiment_thread_multiple(self):
        """Test parallel experiment thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(**self.backend_options_parallel(exp_threads=max_threads))

        # Test multiple circuit, no noise
        # Parallel experiments should take priority
        result = backend.run(max_threads * [self.dummy_circuit(1)], shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": max_threads,
                "shots": 1,
                "state_update": 1,
                "total": max_threads,
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_experiment_thread_single_noise(self):
        """Test parallel experiment thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(**self.backend_options_parallel(exp_threads=max_threads))

        # Test multiple circuits, with noise
        # Parallel experiments should take priority
        result = backend.run(
            max_threads * [self.dummy_circuit(1)],
            shots=10 * max_threads,
            noise_model=self.dummy_noise_model(),
        ).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": max_threads,
                "shots": 1,
                "state_update": 1,
                "total": max_threads,
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_experiment_thread_multiple_midmeas(self):
        """Test parallel experiment thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(**self.backend_options_parallel(exp_threads=max_threads))

        # Test multiple circuit, with measure in middle, no noise
        # Parallel experiments should take priority
        result = backend.run(
            max_threads * [self.measure_in_middle_circuit(1)], shots=10 * max_threads
        ).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": max_threads,
                "shots": 1,
                "state_update": 1,
                "total": max_threads,
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_experiment_thread_mem_limit(self):
        """Test parallel experiment thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(
            method="statevector",
            max_memory_mb=1,
            **self.backend_options_parallel(exp_threads=max_threads),
        )

        # Test multiple circuits, with memory limitation
        # NOTE: this assumes execution on statevector simulator
        # which required approx 2 MB for 16 qubit circuit.
        circuit = transpile(QuantumVolume(16, 1, seed=0), backend)
        circuit.measure_all()
        result = backend.run(2 * [circuit], shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": 1,
                "state_update": max_threads,
                "total": max_threads,
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_shot_thread_single_ideal(self):
        """Test parallel shot thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(**self.backend_options_parallel(shot_threads=max_threads))

        # Test single circuit
        # Parallel experiments and shots should always be 1
        result = backend.run(max_threads * [self.dummy_circuit(1)], shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": 1,
                "state_update": max_threads,
                "total": max_threads,
            }
            self.assertEqual(threads, target, msg="single circuit ideal case failed")

    @requires_omp
    @requires_multiprocessing
    def test_parallel_shot_thread_multi_ideal(self):
        """Test parallel shot thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(**self.backend_options_parallel(shot_threads=max_threads))

        # Test multiple circuit, no noise
        # Parallel experiments and shots should always be 1
        result = backend.run(max_threads * [self.dummy_circuit(1)], shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": 1,
                "state_update": max_threads,
                "total": max_threads,
            }
            self.assertEqual(threads, target, msg="multi circuit ideal case failed")

    @requires_omp
    @requires_multiprocessing
    def test_parallel_shot_thread_multi_noise(self):
        """Test parallel shot thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(
            method="statevector",
            noise_model=self.dummy_noise_model(),
            **self.backend_options_parallel(shot_threads=max_threads),
        )

        # Test multiple circuits, with noise
        # Parallel shots should take priority
        result = backend.run(max_threads * [self.dummy_circuit(1)], shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": max_threads,
                "state_update": 1,
                "total": max_threads,
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_shot_thread_multi_meas(self):
        """Test parallel shot thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(
            noise_model=self.dummy_noise_model(),
            **self.backend_options_parallel(shot_threads=max_threads),
        )

        # Test multiple circuit, with measure in middle, no noise
        # Parallel shots should take priority
        result = backend.run(
            max_threads * [self.measure_in_middle_circuit(1)], shots=10 * max_threads
        ).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": max_threads,
                "state_update": 1,
                "total": max_threads,
            }
            self.assertEqual(threads, target, msg="multi circuit noise case failed")

    @requires_omp
    @requires_multiprocessing
    def test_parallel_shot_thread_mem_limit(self):
        """Test parallel shot thread assignment"""

        max_threads = self.available_threads()
        backend = self.backend(
            max_memory_mb=1, **self.backend_options_parallel(shot_threads=max_threads)
        )

        # Test multiple circuits, with memory limitation
        # NOTE: this assumes execution on statevector simulator
        # which required approx 2 MB for 16 qubit circuit.
        circuit = transpile(QuantumVolume(16, 1, seed=0), backend)
        circuit.measure_all()
        result = backend.run(2 * [circuit], shots=10 * max_threads).result()
        for threads in self.threads_used(result):
            target = {
                "experiments": 1,
                "shots": 1,
                "state_update": max_threads,
                "total": max_threads,
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def _test_qasm_explicit_parallelization(self):
        """test disabling parallel shots because max_parallel_shots is 1"""
        backend = self.backend(
            noise_model=self.dummy_noise_model(),
            **self.backend_options_parallel(shot_threads=1, exp_threads=1),
        )
        # Test circuit
        shots = multiprocessing.cpu_count()
        circuit = QuantumVolume(16, 1, seed=0)
        circuit.measure_all()

        run_opts = {"_parallel_experiments": 2, "_parallel_shots": 3, "_parallel_state_update": 4}

        result = self.backend(circuit, shots=shots, **run_opts).result()
        if result.metadata["omp_enabled"]:
            self.assertEqual(
                result.metadata["parallel_experiments"], 2, msg="parallel_experiments should be 2"
            )
            self.assertEqual(
                result.to_dict()["results"][0]["metadata"]["parallel_shots"],
                3,
                msg="parallel_shots must be 3",
            )
            self.assertEqual(
                result.to_dict()["results"][0]["metadata"]["parallel_state_update"],
                4,
                msg="parallel_state_update should be 4",
            )
