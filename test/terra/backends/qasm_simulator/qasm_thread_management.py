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
from qiskit import execute
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error


class QasmThreadManagementTests:
    """QasmSimulator thread tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    def dummy_noise_model(self):
        """Return test reset noise model"""
        noise_model = NoiseModel()
        error = pauli_error([('X', 0.25), ('I', 0.75)])
        noise_model.add_all_qubit_quantum_error(error, 'reset')
        return noise_model

    def test_qasm_default_parallelization(self):
        """test default parallelization"""
        # Test circuit
        shots = 100
        circuit = quantum_volume_circuit(4, 1, measure=True, seed=0)
        backend_opts = self.BACKEND_OPTS.copy()
        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                1,
                msg="parallel_experiments should be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                1,
                msg="parallel_shots must be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'],
                multiprocessing.cpu_count(),
                msg="parallel_state_update should be " + str(
                    multiprocessing.cpu_count()))

    def test_qasm_max_memory_default(self):
        """test default max memory"""
        # Test circuit
        shots = 100
        circuit = quantum_volume_circuit(4, 1, measure=True, seed=0)
        backend_opts = self.BACKEND_OPTS.copy()
        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        system_memory = int(psutil.virtual_memory().total / 1024 / 1024)
        self.assertGreaterEqual(
            result.metadata['max_memory_mb'],
            int(system_memory / 2),
            msg="statevector_memory is too small.")
        self.assertLessEqual(
            result.metadata['max_memory_mb'],
            system_memory,
            msg="statevector_memory is too big.")

    def test_qasm_max_memory_specified(self):
        """test default max memory configuration"""
        # Test circuit
        shots = 100
        circuit = quantum_volume_circuit(4, 1, measure=True, seed=0)

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_memory_mb'] = 128

        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        self.assertEqual(
            result.metadata['max_memory_mb'],
            128,
            msg="max_memory_mb is not configured correctly.")

    def test_qasm_auto_enable_parallel_experiments(self):
        """test default max memory configuration"""
        # Test circuit
        shots = 100
        experiments = multiprocessing.cpu_count()
        circuits = []
        for _ in range(experiments):
            circuits.append(quantum_volume_circuit(4, 1, measure=True, seed=0))

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_experiments'] = experiments
        backend_opts['max_memory_mb'] = 1024

        result = execute(
            circuits, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                multiprocessing.cpu_count(),
                msg="parallel_experiments should be multiprocessing.cpu_count()"
            )
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                1,
                msg="parallel_shots must be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'],
                1,
                msg="parallel_state_update should be 1")

    def test_qasm_auto_disable_parallel_experiments_with_memory_shortage(self):
        """test auto-disabling max_parallel_experiments because memory is short"""
        # Test circuit
        shots = 100
        experiments = multiprocessing.cpu_count()
        circuits = []
        for _ in range(experiments):
            circuits.append(
                quantum_volume_circuit(16, 1, measure=True,
                                       seed=0))  # 2 MB for each

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_experiments'] = experiments
        backend_opts['max_memory_mb'] = 1

        result = execute(
            circuits, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                1,
                msg="parallel_experiments should be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                1,
                msg="parallel_shots must be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'],
                multiprocessing.cpu_count(),
                msg="parallel_state_update should be " + str(
                    multiprocessing.cpu_count()))

    def test_qasm_auto_short_parallel_experiments(self):
        """test auto-disabling max_parallel_experiments because a number of circuits is few"""
        # Test circuit
        shots = 1
        experiments = multiprocessing.cpu_count()
        if experiments == 1:
            return
        circuits = []
        for _ in range(experiments - 1):
            circuits.append(
                quantum_volume_circuit(4, 1, measure=True,
                                       seed=0))  # 2 MB for each

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_experiments'] = experiments
        backend_opts['max_memory_mb'] = 1024

        result = execute(
            circuits, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                multiprocessing.cpu_count() - 1,
                msg="parallel_experiments should be {}".format(
                    multiprocessing.cpu_count() - 1))
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                1,
                msg="parallel_shots must be 1")

    def test_qasm_auto_enable_shot_parallelization(self):
        """test explicit shot parallelization"""
        # Test circuit
        shots = multiprocessing.cpu_count()
        circuit = quantum_volume_circuit(4, 1, measure=True, seed=0)

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_shots'] = shots
        backend_opts['noise_model'] = self.dummy_noise_model()

        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                1,
                msg="parallel_experiments should be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                multiprocessing.cpu_count(),
                msg="parallel_shots must be " + str(
                    multiprocessing.cpu_count()))
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'],
                1,
                msg="parallel_state_update should be 1")

    def test_qasm_auto_disable_shot_parallelization_with_sampling(self):
        """test auto-disabling max_parallel_shots because sampling is enabled"""
        # Test circuit
        shots = multiprocessing.cpu_count()
        circuit = quantum_volume_circuit(4, 1, measure=True, seed=0)

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_shots'] = shots

        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                1,
                msg="parallel_experiments should be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                1,
                msg="parallel_shots must be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'],
                multiprocessing.cpu_count(),
                msg="parallel_state_update should be " + str(
                    multiprocessing.cpu_count()))

    def test_qasm_auto_short_shot_parallelization(self):
        """test auto-disabling max_parallel_shots because a number of shots is few"""
        # Test circuit
        max_threads = 4
        shots = 2
        circuit = quantum_volume_circuit(4, 1, measure=True, seed=0)

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_threads'] = max_threads
        backend_opts['max_parallel_shots'] = max_threads
        backend_opts['noise_model'] = self.dummy_noise_model()

        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                1,
                msg="parallel_experiments should be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                shots, msg="parallel_shots must be " + str(shots))
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'], max_threads/shots,
                msg="parallel_state_update should be " + str(max_threads/shots))

    def test_qasm_auto_disable_shot_parallelization_with_memory_shortage(self):
        """test auto-disabling max_parallel_shots because memory is short"""
        # Test circuit
        max_threads = 2
        shots = 2
        circuit = quantum_volume_circuit(17, 1, measure=True, seed=0)

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_threads'] = max_threads
        backend_opts['max_parallel_shots'] = shots
        backend_opts['noise_model'] = self.dummy_noise_model()
        backend_opts['max_memory_mb'] = 2

        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                1,
                msg="parallel_experiments should be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                1, msg="parallel_shots must be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'], max_threads,
                msg="parallel_state_update should be " + str(max_threads))

    def test_qasm_auto_enable_shot_parallelization_with_single_precision(self):
        """test auto-enabling max_parallel_shots with single-precision"""
        # Test circuit
        max_threads = 2
        shots = 2
        circuit = quantum_volume_circuit(17, 1, measure=True, seed=0)

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_threads'] = max_threads
        backend_opts['max_parallel_shots'] = shots
        backend_opts['noise_model'] = self.dummy_noise_model()
        backend_opts['max_memory_mb'] = 2
        backend_opts['precision'] = "single"

        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                1,
                msg="parallel_experiments should be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                shots, msg="parallel_shots must be " + str(shots))
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'], 1,
                msg="parallel_state_update should be 1")
            
    def test_qasm_auto_disable_shot_parallelization_with_max_parallel_shots(self):
        """test disabling parallel shots because max_parallel_shots is 1"""
        # Test circuit
        shots = multiprocessing.cpu_count()
        circuit = quantum_volume_circuit(16, 1, measure=True, seed=0)

        backend_opts = self.BACKEND_OPTS.copy()
        backend_opts['max_parallel_shots'] = 1
        backend_opts['noise_model'] = self.dummy_noise_model()

        result = execute(
            circuit, self.SIMULATOR, shots=shots,
            backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(
                result.metadata['parallel_experiments'],
                1,
                msg="parallel_experiments should be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']['parallel_shots'],
                1,
                msg="parallel_shots must be 1")
            self.assertEqual(
                result.to_dict()['results'][0]['metadata']
                ['parallel_state_update'],
                multiprocessing.cpu_count(),
                msg="parallel_state_update should be " + str(
                    multiprocessing.cpu_count()))


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
