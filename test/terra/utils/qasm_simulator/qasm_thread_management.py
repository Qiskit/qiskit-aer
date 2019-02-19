# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

import multiprocessing
import psutil
from test.terra.utils import common
from test.terra.utils import ref_qvolume
from qiskit import compile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error

class QasmThreadManagementTests(common.QiskitAerTestCase):
    """QasmSimulator thread tests."""

    SIMULATOR = QasmSimulator()

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
        circuit = ref_qvolume.quantum_volume(4, depth=1, final_measure=True)
        qobj = compile(circuit, self.SIMULATOR, shots=shots)
        backend_opts = {}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], 1, msg="default parallel_experiments should be 1")
            self.assertEqual(result.metadata['parallel_shots'], 1, msg="default parallel_shots should be 1")
            self.assertEqual(result.metadata['parallel_state_update'], multiprocessing.cpu_count(), msg="default parallel_state_update should be same with multiprocessing.cpu_count()")

    def test_qasm_max_memory_default(self):
        """test default max memory"""
        # Test circuit
        shots = 100
        experiments = multiprocessing.cpu_count()
        circuit = ref_qvolume.quantum_volume(4, depth=1, final_measure=True)
        qobj = compile(circuit, self.SIMULATOR, shots=shots)
        backend_opts = {}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        system_memory = psutil.virtual_memory().total / 1024 / 1024
        self.assertGreaterEqual(result.metadata['max_statevector_memory_mb'], system_memory / 2, msg="statevector_memory is too small.")
        self.assertLessEqual(result.metadata['max_statevector_memory_mb'], system_memory, msg="statevector_memory is too big.")

    def test_qasm_max_memory_specified(self):
        """test default max memory configuration"""
        # Test circuit
        shots = 100
        experiments = multiprocessing.cpu_count()
        circuit = ref_qvolume.quantum_volume(4, depth=1, final_measure=True)
        qobj = compile(circuit, self.SIMULATOR, shots=shots)
        backend_opts = {'max_statevector_memory_mb': 128}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        self.assertEqual(result.metadata['max_statevector_memory_mb'], 128, msg="statevector_memory is not configured correctly.")

    def test_qasm_auto_enable_parallel_experiments(self):
        """test default max memory configuration"""
        # Test circuit
        shots = 100
        experiments = multiprocessing.cpu_count()
        circuits = []
        for i in range(experiments):
            circuits.append(ref_qvolume.quantum_volume(4, depth=1, final_measure=True))
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        backend_opts = {'max_parallel_experiments': experiments, 'max_statevector_memory_mb': 1024}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], multiprocessing.cpu_count(), msg="parallel_experiments should be multiprocessing.cpu_count()")
            self.assertEqual(result.metadata['parallel_shots'], 1, msg="parallel_shots must be 1")
            self.assertEqual(result.metadata['parallel_state_update'], 1, msg="parallel_state_update should be 1")

    def test_qasm_auto_disable_parallel_experiments_with_memory_shortage(self):
        """test auto-disabling max_parallel_experiments because memory is short"""
        # Test circuit
        shots = 100
        experiments = multiprocessing.cpu_count()
        circuits = []
        for i in range(experiments):
            circuits.append(ref_qvolume.quantum_volume(17, depth=1, final_measure=True)) # 2 MB for each
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        backend_opts = {'max_parallel_experiments': experiments, 'max_statevector_memory_mb': 1}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], 1, msg="parallel_experiments should be 1")
            self.assertEqual(result.metadata['parallel_shots'], 1, msg="parallel_shots must be 1")
            self.assertEqual(result.metadata['parallel_state_update'], multiprocessing.cpu_count(), msg="parallel_state_update should be 1")

    def test_qasm_auto_disable_parallel_experiments_with_circuits_shortage(self):
        """test auto-disabling max_parallel_experiments because a number of circuits is few"""
        # Test circuit
        shots = 1
        experiments = multiprocessing.cpu_count()
        circuits = []
        for i in range(experiments - 1):
            circuits.append(ref_qvolume.quantum_volume(4, depth=1, final_measure=True)) # 2 MB for each
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        backend_opts = {'max_parallel_experiments': experiments, 'max_statevector_memory_mb': 1024}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], 1, msg="parallel_experiments should be 1")
            self.assertEqual(result.metadata['parallel_shots'], 1, msg="parallel_shots must be 1")
            self.assertEqual(result.metadata['parallel_state_update'], multiprocessing.cpu_count(), msg="parallel_state_update should be 1")

    def test_qasm_auto_enable_shot_parallelization(self):
        """test explicit shot parallelization"""
        # Test circuit
        shots = multiprocessing.cpu_count()
        circuit = ref_qvolume.quantum_volume(4, depth=1, final_measure=True)
        qobj = compile(circuit, self.SIMULATOR, shots=shots)
        backend_opts = {'max_parallel_shots': shots, 'noise_model': self.dummy_noise_model()}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], 1, msg="default parallel_experiments should be 1")
            self.assertEqual(result.metadata['parallel_shots'], multiprocessing.cpu_count(), msg="parallel_shots must be multiprocessing.cpu_count()")
            self.assertEqual(result.metadata['parallel_state_update'], 1, msg="parallel_state_update should be 1")

    def test_qasm_auto_disable_shot_parallelization_with_sampling(self):
        """test auto-disabling max_parallel_shots because sampling is enabled"""
        # Test circuit
        shots = multiprocessing.cpu_count()
        circuit = ref_qvolume.quantum_volume(4, depth=1, final_measure=True)
        qobj = compile(circuit, self.SIMULATOR, shots=shots)
        backend_opts = {'max_parallel_shots': shots}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], 1, msg="parallel_experiments should be 1")
            self.assertEqual(result.metadata['parallel_shots'], 1, msg="parallel_shots must be 1")
            self.assertEqual(result.metadata['parallel_state_update'], multiprocessing.cpu_count(), msg="parallel_state_update should be 1")

    def test_qasm_auto_disable_shot_parallelization_with_shots_shortage(self):
        """test auto-disabling max_parallel_shots because a number of shots is few"""
        # Test circuit
        shots = multiprocessing.cpu_count() - 1
        circuit = ref_qvolume.quantum_volume(4, depth=1, final_measure=True)
        qobj = compile(circuit, self.SIMULATOR, shots=shots)
        backend_opts = {'max_parallel_shots': shots, 'noise_model': self.dummy_noise_model()}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], 1, msg="parallel_experiments should be 1")
            self.assertEqual(result.metadata['parallel_shots'], 1, msg="parallel_shots must be 1")
            self.assertEqual(result.metadata['parallel_state_update'], multiprocessing.cpu_count(), msg="parallel_state_update should be 1")
            
    def test_qasm_auto_disable_shot_parallelization_with_memory_shortage(self):
        """test auto-disabling max_parallel_shots because memory is short"""
        # Test circuit
        shots = multiprocessing.cpu_count()
        circuit = ref_qvolume.quantum_volume(17, depth=1, final_measure=True)
        qobj = compile(circuit, self.SIMULATOR, shots=shots)
        backend_opts = {'max_parallel_shots': shots, 'noise_model': self.dummy_noise_model(), 'max_statevector_memory_mb': 1}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], 1, msg="parallel_experiments should be 1")
            self.assertEqual(result.metadata['parallel_shots'], 1, msg="parallel_shots must be 1")
            self.assertEqual(result.metadata['parallel_state_update'], multiprocessing.cpu_count(), msg="parallel_state_update should be 1")
            