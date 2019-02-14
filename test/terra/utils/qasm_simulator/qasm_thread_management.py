# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

import multiprocessing
from test.terra.utils import common
from test.terra.utils import ref_qvolume
from qiskit import compile
from qiskit.providers.aer import QasmSimulator


class QasmThreadManagementTests(common.QiskitAerTestCase):
    """QasmSimulator thread tests."""

    SIMULATOR = QasmSimulator()

    def test_qasm_default_parallelization(self):
        """Test statevector method is used for Clifford circuit"""
        # Test circuit
        shots = 100
        circuit = ref_qvolume.quantum_volume(4, final_measure=True)
        qobj = compile(circuit, self.SIMULATOR, shots=shots)
        backend_opts = {}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], 1, msg="default parallel_experiments should be 1")
            self.assertEqual(result.metadata['parallel_shots'], 1, msg="default parallel_shots should be 1")
            self.assertEqual(result.metadata['parallel_state_update'], multiprocessing.cpu_count(), msg="default parallel_state_update should be same with multiprocessing.cpu_count()")

    def test_qasm_shot_parallelization(self):
        """Test statevector method is used for Clifford circuit"""
        # Test circuit
        shots = multiprocessing.cpu_count()
        circuit = ref_qvolume.quantum_volume(4, final_measure=True)
        qobj = compile(circuit, self.SIMULATOR, shots=shots)
        backend_opts = {'max_parallel_shots': shots}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], 1, msg="default parallel_experiments should be 1")
            self.assertEqual(result.metadata['parallel_shots'], multiprocessing.cpu_count(), msg="parallel_shots must be multiprocessing.cpu_count()")
            self.assertEqual(result.metadata['parallel_state_update'], 1, msg="parallel_state_update should be 1")

    def test_qasm_experiments_parallelization(self):
        """Test statevector method is used for Clifford circuit"""
        # Test circuit
        shots = 100
        experiments = multiprocessing.cpu_count()
        circuits = []
        for i in range(experiments):
            circuits.append(ref_qvolume.quantum_volume(4, final_measure=True))
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        backend_opts = {'max_parallel_experiments': experiments}
        result = self.SIMULATOR.run(qobj, backend_options=backend_opts).result()
        if result.metadata['omp_enabled']:
            self.assertEqual(result.metadata['parallel_experiments'], multiprocessing.cpu_count(), msg="parallel_experiments should be multiprocessing.cpu_count()")
            self.assertEqual(result.metadata['parallel_shots'], 1, msg="parallel_shots must be 1")
            self.assertEqual(result.metadata['parallel_state_update'], 1, msg="parallel_state_update should be 1")