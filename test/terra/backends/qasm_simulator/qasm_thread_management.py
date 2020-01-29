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
QasmSimulator Integration Tests
"""

import multiprocessing
import psutil

from test.benchmark.tools import quantum_volume_circuit
from qiskit import execute, QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error
from test.terra.decorators import requires_omp, requires_multiprocessing

# pylint: disable=no-member
class QasmThreadManagementTests:
    """QasmSimulator thread tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    def dummy_noise_model(self):
        """Return dummy noise model for dummy circuit"""
        noise_model = NoiseModel()
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model.add_all_qubit_quantum_error(error, 'x')
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
        exp_threads = getattr(result, 'metadata', {}).get('parallel_experiments', 1)
        threads = []
        for exp_result in getattr(result, 'results', []):
            exp_meta = getattr(exp_result, 'metadata', {})
            shot_threads = exp_meta.get('parallel_shots', 1)
            state_threads = exp_meta.get('parallel_state_update', 1)
            threads.append({
                'experiments': exp_threads,
                'shots': shot_threads,
                'state_update': state_threads,
                'total': exp_threads * shot_threads * state_threads
            })
        return threads

    def test_max_memory_settings(self):
        """test max memory configuration"""

        # 4-qubit quantum volume test circuit
        shots = 100
        circuit = quantum_volume_circuit(4, 1, measure=True, seed=0)
        system_memory = int(psutil.virtual_memory().total / 1024 / 1024)

        # Test defaults
        opts = self.BACKEND_OPTS.copy()
        result = execute(circuit, self.SIMULATOR, shots=shots,
                         backend_options=opts).result()
        max_mem_result = result.metadata.get('max_memory_mb')
        self.assertGreaterEqual(max_mem_result, int(system_memory / 2),
                                msg="Default 'max_memory_mb' is too small.")
        self.assertLessEqual(max_mem_result, system_memory,
                             msg="Default 'max_memory_mb' is too large.")

        # Test custom value
        max_mem_target = 128
        opts['max_memory_mb'] = max_mem_target
        result = execute(circuit, self.SIMULATOR, shots=shots,
                         backend_options=opts).result()
        max_mem_result = result.metadata.get('max_memory_mb')
        self.assertEqual(max_mem_result, max_mem_target,
                         msg="Custom 'max_memory_mb' is not being set correctly.")

    @requires_omp
    @requires_multiprocessing
    def test_parallel_thread_defaults(self):
        """Test parallel thread assignment defaults"""

        opts = self.BACKEND_OPTS
        max_threads = multiprocessing.cpu_count()

        # Test single circuit, no noise
        # Parallel experiments and shots should always be 1
        result = execute(self.dummy_circuit(1),
                         self.SIMULATOR,
                         shots=10*max_threads,
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': 1,
                'state_update': max_threads,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test single circuit, with noise
        # Parallel experiments should always be 1
        # parallel shots should be greater than 1
        result = execute(self.dummy_circuit(1),
                         self.SIMULATOR,
                         shots=10*max_threads,
                         noise_model=self.dummy_noise_model(),
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': max_threads,
                'state_update': 1,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test single circuit, with measure in middle, no noise
        # Parallel experiments should always be 1
        # parallel shots should be greater than 1
        result = execute(self.measure_in_middle_circuit(1),
                         self.SIMULATOR,
                         shots=10*max_threads,
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': max_threads,
                'state_update': 1,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuit, no noise
        # Parallel experiments always be 1
        # parallel shots should always be 1
        result = execute(max_threads*[self.dummy_circuit(1)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': 1,
                'state_update': max_threads,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuits, with noise
        # Parallel experiments should always be 1
        # parallel shots should be greater than 1
        result = execute(max_threads*[self.dummy_circuit(1)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         noise_model=self.dummy_noise_model(),
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': max_threads,
                'state_update': 1,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuit, with measure in middle, no noise
        # Parallel experiments should always be 1
        # parallel shots should be greater than 1
        result = execute(max_threads*[self.measure_in_middle_circuit(1)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         noise_model=self.dummy_noise_model(),
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': max_threads,
                'state_update': 1,
                'total': max_threads
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_thread_assignment_priority(self):
        """Test parallel thread assignment priority"""

        # If we set all values to max test output
        # We intentionally set the max shot and experiment threads to
        # twice the max threads to check they are limited correctly
        for custom_max_threads in [0, 1, 2, 4]:
            opts = self.BACKEND_OPTS.copy()
            opts['max_parallel_threads'] = custom_max_threads
            opts['max_parallel_experiments'] = 2 * custom_max_threads
            opts['max_parallel_shots'] = 2 * custom_max_threads

            # Calculate actual max threads from custom max and CPU number
            max_threads = multiprocessing.cpu_count()
            if custom_max_threads > 0:
                max_threads = min(max_threads, custom_max_threads)

            # Test single circuit, no noise
            # Parallel experiments and shots should always be 1
            result = execute(self.dummy_circuit(1),
                             self.SIMULATOR,
                             shots=10*max_threads,
                             backend_options=opts).result()
            for threads in self.threads_used(result):
                target = {
                    'experiments': 1,
                    'shots': 1,
                    'state_update': max_threads,
                    'total': max_threads
                }
                self.assertEqual(threads, target)

            # Test single circuit, with noise
            # Parallel experiments should always be 1
            # parallel shots should be greater than 1
            result = execute(self.dummy_circuit(1),
                             self.SIMULATOR,
                             shots=10*max_threads,
                             noise_model=self.dummy_noise_model(),
                             backend_options=opts).result()
            for threads in self.threads_used(result):
                target = {
                    'experiments': 1,
                    'shots': max_threads,
                    'state_update': 1,
                    'total': max_threads
                }
                self.assertEqual(threads, target)

            # Test single circuit, with measure in middle, no noise
            # Parallel experiments should always be 1
            # parallel shots should be greater than 1
            result = execute(self.measure_in_middle_circuit(1),
                             self.SIMULATOR,
                             shots=10*max_threads,
                             backend_options=opts).result()
            for threads in self.threads_used(result):
                target = {
                    'experiments': 1,
                    'shots': max_threads,
                    'state_update': 1,
                    'total': max_threads
                }
                self.assertEqual(threads, target)

            # Test multiple circuit, no noise
            # Parallel experiments always be greater than 1
            # parallel shots should always be 1
            result = execute(max_threads*[self.dummy_circuit(1)],
                             self.SIMULATOR,
                             shots=10*max_threads,
                             backend_options=opts).result()
            for threads in self.threads_used(result):
                target = {
                    'experiments': max_threads,
                    'shots': 1,
                    'state_update': 1,
                    'total': max_threads
                }
                self.assertEqual(threads, target)

            # Test multiple circuits, with noise
            # Parallel experiments always be greater than 1
            # parallel shots should always be 1
            result = execute(max_threads*[self.dummy_circuit(1)],
                             self.SIMULATOR,
                             shots=10*max_threads,
                             noise_model=self.dummy_noise_model(),
                             backend_options=opts).result()
            for threads in self.threads_used(result):
                target = {
                    'experiments': max_threads,
                    'shots': 1,
                    'state_update': 1,
                    'total': max_threads
                }
                self.assertEqual(threads, target)

            # Test multiple circuit, with measure in middle, no noise
            # Parallel experiments always be greater than 1
            # parallel shots should always be 1
            result = execute(max_threads*[self.measure_in_middle_circuit(1)],
                             self.SIMULATOR,
                             shots=10*max_threads,
                             noise_model=self.dummy_noise_model(),
                             backend_options=opts).result()
            for threads in self.threads_used(result):
                target = {
                    'experiments': max_threads,
                    'shots': 1,
                    'state_update': 1,
                    'total': max_threads
                }
                self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_experiment_thread_assignment(self):
        """Test parallel experiment thread assignment"""

        max_threads = multiprocessing.cpu_count()
        opts = self.BACKEND_OPTS.copy()
        opts['max_parallel_experiments'] = max_threads

        # Test single circuit
        # Parallel experiments and shots should always be 1
        result = execute(self.dummy_circuit(1),
                         self.SIMULATOR,
                         shots=10*max_threads,
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': 1,
                'state_update': max_threads,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuit, no noise
        # Parallel experiments should take priority
        result = execute(max_threads*[self.dummy_circuit(1)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': max_threads,
                'shots': 1,
                'state_update': 1,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuits, with noise
        # Parallel experiments should take priority
        result = execute(max_threads*[self.dummy_circuit(1)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         noise_model=self.dummy_noise_model(),
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': max_threads,
                'shots': 1,
                'state_update': 1,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuit, with measure in middle, no noise
        # Parallel experiments should take priority
        result = execute(max_threads*[self.measure_in_middle_circuit(1)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         noise_model=self.dummy_noise_model(),
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': max_threads,
                'shots': 1,
                'state_update': 1,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuits, with memory limitation
        # NOTE: this assumes execution on statevector simulator
        # which required approx 2 MB for 16 qubit circuit.
        opts['max_memory_mb'] = 1
        result = execute(2 * [quantum_volume_circuit(16, 1, measure=True, seed=0)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': 1,
                'state_update': max_threads,
                'total': max_threads
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def test_parallel_shot_thread_assignment(self):
        """Test parallel shot thread assignment"""

        max_threads = multiprocessing.cpu_count()
        opts = self.BACKEND_OPTS.copy()
        opts['max_parallel_shots'] = max_threads

        # Test single circuit
        # Parallel experiments and shots should always be 1
        result = execute(self.dummy_circuit(1),
                         self.SIMULATOR,
                         shots=10*max_threads,
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': 1,
                'state_update': max_threads,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuit, no noise
        # Parallel experiments and shots should always be 1
        result = execute(max_threads*[self.dummy_circuit(1)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': 1,
                'state_update': max_threads,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuits, with noise
        # Parallel shots should take priority
        result = execute(max_threads*[self.dummy_circuit(1)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         noise_model=self.dummy_noise_model(),
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': max_threads,
                'state_update': 1,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuit, with measure in middle, no noise
        # Parallel shots should take priority
        result = execute(max_threads*[self.measure_in_middle_circuit(1)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         noise_model=self.dummy_noise_model(),
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': max_threads,
                'state_update': 1,
                'total': max_threads
            }
            self.assertEqual(threads, target)

        # Test multiple circuits, with memory limitation
        # NOTE: this assumes execution on statevector simulator
        # which required approx 2 MB for 16 qubit circuit.
        opts['max_memory_mb'] = 1
        result = execute(2 * [quantum_volume_circuit(16, 1, measure=True, seed=0)],
                         self.SIMULATOR,
                         shots=10*max_threads,
                         backend_options=opts).result()
        for threads in self.threads_used(result):
            target = {
                'experiments': 1,
                'shots': 1,
                'state_update': max_threads,
                'total': max_threads
            }
            self.assertEqual(threads, target)

    @requires_omp
    @requires_multiprocessing
    def _test_qasm_explicit_parallelization(self):
        """test disabling parallel shots because max_parallel_shots is 1"""
        # Test circuit
        shots = multiprocessing.cpu_count()
        circuit = quantum_volume_circuit(16, 1, measure=True, seed=0)

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_shots'] = 1
        backend_opts['max_parallel_experiments'] = 1
        backend_opts['noise_model'] = self.dummy_noise_model()
        backend_opts['_parallel_experiments'] = 2
        backend_opts['_parallel_shots'] = 3
        backend_opts['_parallel_state_update'] = 4

        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                2,
                msg="parallel_experiments should be 2")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                3,
                msg="parallel_shots must be 3")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'],
                4,
                msg="parallel_state_update should be 4")
