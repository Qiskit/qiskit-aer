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
from ddt import ddt
from qiskit import QuantumCircuit
from test.terra.reference import ref_initialize

import numpy as np
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)

SUPPORTED_METHODS = ['automatic', 'statevector', 'matrix_product_state']


@ddt
class TestInitialize(SimulatorTestCase):
    """AerSimulator initialize tests."""

    # ---------------------------------------------------------------------
    # Test initialize instr make it through the wrapper
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_initialize_wrapper_1(self, method, device):
        """Test AerSimulator initialize"""
        backend = self.backend(method=method, device=device)
        shots = 100
        lst = [0, 1]
        init_states = [
            np.array(lst),
            np.array(lst, dtype=float),
            np.array(lst, dtype=np.float32),
            np.array(lst, dtype=complex),
            np.array(lst, dtype=np.complex64) ]
        circuits = []
        [ circuits.extend(ref_initialize.initialize_circuits_w_1(init_state)) for init_state in init_states ]
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
 
    # ---------------------------------------------------------------------
    # Test initialize instr make it through the wrapper
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_initialize_wrapper_2(self, method, device):
        """Test AerSimulator initialize"""
        backend = self.backend(method=method, device=device)
        shots = 100
        lst = [0, 1, 0, 0]
        init_states = [
            np.array(lst),
            np.array(lst, dtype=float),
            np.array(lst, dtype=np.float32),
            np.array(lst, dtype=complex),
            np.array(lst, dtype=np.complex64) ]
        circuits = []
        [ circuits.extend(ref_initialize.initialize_circuits_w_2(init_state)) for init_state in init_states ]
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)

    # ---------------------------------------------------------------------
    # Test initialize
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_initialize_1(self, method, device):
        """Test AerSimulator initialize"""
        backend = self.backend(method=method, device=device)
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 1000
        circuits = ref_initialize.initialize_circuits_1(final_measure=True)
        targets = ref_initialize.initialize_counts_1(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_initialize_2(self, method, device):
        """Test AerSimulator initialize"""
        backend = self.backend(method=method, device=device)
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 1000
        circuits = ref_initialize.initialize_circuits_2(final_measure=True)
        targets = ref_initialize.initialize_counts_2(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_initialize_sampling_opt(self, method, device):
        """Test sampling optimization"""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_initialize.initialize_sampling_optimization()
        targets = ref_initialize.initialize_counts_sampling_optimization(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_initialize_entangled_qubits(self, method, device):
        """Test initialize entangled qubits"""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_initialize.initialize_entangled_qubits()
        targets = ref_initialize.initialize_counts_entangled_qubits(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_initialize_sampling_opt_enabled(self, method, device):
        """Test sampling optimization"""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuit = QuantumCircuit(2)
        circuit.initialize([0, 1], [1])
        circuit.h([0, 1])
        circuit.initialize([0, 0, 1, 0], [0, 1])
        circuit.measure_all()
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        sampling = result.results[0].metadata.get('measure_sampling', None)
        self.assertTrue(sampling)

    @supported_methods(SUPPORTED_METHODS)
    def test_initialize_sampling_opt_disabled(self, method, device):
        """Test sampling optimization"""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        circuit.initialize([0, 1], [1])
        circuit.measure_all()
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        sampling = result.results[0].metadata.get('measure_sampling', None)
        self.assertFalse(sampling)
