# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Profiler Tests
"""

import sys
import unittest

import numpy as np

from test.terra import common

from qiskit import execute, transpile
from qiskit.compiler import assemble
from qiskit.circuit.library import QuantumVolume
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.profile import optimize_backend_options, clear_optimized_backend_options

class TestProfileQasmSimulator(common.QiskitAerTestCase):
    
    
    def test_profile(self):
        profile = optimize_backend_options()
        
        simulator = QasmSimulator()

        if 'statevector_parallel_threshold' in profile:
            circuit = QuantumVolume(profile['statevector_parallel_threshold'], 1)
            job = execute(circuit, simulator, method='statevector')
            self.assertTrue(hasattr(job.qobj().config, 'statevector_parallel_threshold'))

        if 'fusion_threshold' in profile:
            circuit = QuantumVolume(profile['fusion_threshold'], 1)
            job = execute(circuit, simulator, method='statevector')
            self.assertTrue(hasattr(job.qobj().config, 'fusion_threshold'))
            self.assertEqual(job.result().results[0].metadata.get('fusion', {})['threshold'],
                             job.qobj().config.fusion_threshold)

        if 'fusion_cost.1' in profile:
            circuit = QuantumVolume(profile['fusion_threshold'], 1)
            job = execute(circuit, simulator, method='statevector')
            self.assertTrue(hasattr(job.qobj().config, 'fusion_cost.1'))
        
        clear_optimized_backend_options()
        circuit = QuantumVolume(10, 1)
        job = execute(circuit, simulator, method='statevector')
        self.assertFalse(hasattr(job.qobj().config, 'statevector_parallel_threshold'))
        self.assertFalse(hasattr(job.qobj().config, 'fusion_threshold'))
        self.assertFalse(hasattr(job.qobj().config, 'fusion_cost.1'))

    def test_profile_with_custom_circuit(self):
        
        circuit = transpile(QuantumVolume(20, 5), basis_gates = ['id', 'u', 'cx'])
        
        profile = optimize_backend_options(min_qubits=10, max_qubits=25, ntrials=5, circuit=circuit)
        
        simulator = QasmSimulator()

        if 'statevector_parallel_threshold' in profile:
            circuit = QuantumVolume(profile['statevector_parallel_threshold'], 1)
            job = execute(circuit, simulator, method='statevector')
            self.assertTrue(hasattr(job.qobj().config, 'statevector_parallel_threshold'))

        if 'fusion_threshold' in profile:
            circuit = QuantumVolume(profile['fusion_threshold'], 1)
            job = execute(circuit, simulator, method='statevector')
            self.assertTrue(hasattr(job.qobj().config, 'fusion_threshold'))
            self.assertEqual(job.result().results[0].metadata.get('fusion', {})['threshold'],
                             job.qobj().config.fusion_threshold)

        if 'fusion_cost.1' in profile:
            circuit = QuantumVolume(profile['fusion_threshold'], 1)
            job = execute(circuit, simulator, method='statevector')
            self.assertTrue(hasattr(job.qobj().config, 'fusion_cost.1'))

        clear_optimized_backend_options()
        circuit = QuantumVolume(10, 1)
        job = execute(circuit, simulator, method='statevector')
        self.assertFalse(hasattr(job.qobj().config, 'statevector_parallel_threshold'))
        self.assertFalse(hasattr(job.qobj().config, 'fusion_threshold'))
        self.assertFalse(hasattr(job.qobj().config, 'fusion_cost.1'))
