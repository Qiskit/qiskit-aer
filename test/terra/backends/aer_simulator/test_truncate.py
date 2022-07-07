# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
AerSimulator Integration Tests
"""
from ddt import ddt
from qiskit import transpile, QuantumCircuit, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeQuito
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)

ALL_METHODS = [
    'automatic', 'stabilizer', 'statevector', 'density_matrix',
    'matrix_product_state', 'extended_stabilizer'
]

@ddt
class TestTruncateQubits(SimulatorTestCase):
    """AerSimulator Qubits Truncate tests."""

    def create_circuit_for_truncate(self):
        circuit = QuantumCircuit(4, 4)
        circuit.u(0.1,0.1,0.1, 1)
        circuit.barrier(range(4))
        circuit.x(2)
        circuit.barrier(range(4))
        circuit.x(1)
        circuit.barrier(range(4))
        circuit.x(3)
        circuit.barrier(range(4))
        circuit.u(0.1,0.1,0.1, 0)
        circuit.barrier(range(4))
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        return circuit

    def device_backend(self):
        return FakeQuito()

    def test_truncate_ideal_sparse_circuit(self):
        """Test qubit truncation for large circuit with unused qubits."""
        backend = self.backend()

        # Circuit that uses just 2-qubits
        circuit = QuantumCircuit(50, 2)
        circuit.x(10)
        circuit.x(20)
        circuit.measure(10, 0)
        circuit.measure(20, 1)

        result = backend.run(circuit, shots=1).result()
        metadata = result.results[0].metadata
        self.assertEqual(metadata["num_qubits"], 2, msg="wrong number of truncated qubits.")
        self.assertEqual(metadata["active_input_qubits"], [10, 20], msg="incorrect truncated qubits.")

    def test_truncate_default(self):
        """Test truncation with noise model option"""
        coupling_map = [# 10-qubit device
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
            [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]
        ]
        noise_model = NoiseModel.from_backend(self.device_backend())
        backend = self.backend(noise_model=noise_model)
        circuit = transpile(
            self.create_circuit_for_truncate(),
            backend, coupling_map=coupling_map)

        result = backend.run(circuit, shots=1).result()        
        metadata = result.results[0].metadata
        self.assertEqual(metadata["num_qubits"], 2)
        self.assertEqual(metadata["active_input_qubits"], [0, 1])                  

    def test_truncate_non_measured_qubits(self):
        """Test truncation of non-measured uncoupled qubits."""
        noise_model = NoiseModel.from_backend(self.device_backend())
        backend = self.backend(noise_model=noise_model)
        circuit = transpile(
            self.create_circuit_for_truncate(),
            backend)

        result = backend.run(circuit, shots=1).result()
        metadata = result.results[0].metadata
        self.assertEqual(metadata["num_qubits"], 2)
        self.assertEqual(metadata["active_input_qubits"], [0, 1])

    def test_truncate_disable_noise(self):
        """Test explicitly disabling truncation with noise model option"""
        coupling_map = [  # 10-qubit device
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
            [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]
        ]
        noise_model = NoiseModel.from_backend(self.device_backend())
        backend = self.backend(noise_model=noise_model, enable_truncation=False)
        circuit = transpile(
            self.create_circuit_for_truncate(),
            backend, coupling_map=coupling_map)

        result = backend.run(circuit, shots=100).result()
        metadata = result.results[0].metadata
        self.assertEqual(metadata["num_qubits"], 10)
        self.assertEqual(metadata["active_input_qubits"], list(range(4)))      

    def test_truncate_connected_qubits(self):
        """Test truncation isn't applied to coupled qubits."""
        backend = self.backend()
        circuit = QuantumCircuit(20, 1)
        circuit.h(5)
        circuit.cx(5, 6)
        circuit.cx(6, 2),
        circuit.cx(2, 3)
        circuit.measure(3, 0)
        result = backend.run(circuit, shots=1).result()
        metadata = result.results[0].metadata
        self.assertEqual(metadata["num_qubits"], 4)
        self.assertEqual(metadata["active_input_qubits"], [2, 3, 5, 6])

    def test_delay_measure(self):
        """Test truncation delays measure for measure sampling"""
        backend = self.backend()
        circuit = QuantumCircuit(2, 2)
        circuit.x(0)
        circuit.measure(0, 0)
        circuit.barrier([0, 1])
        circuit.x(1)
        circuit.measure(1, 1)
        shots = 100
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertIn('measure_sampling', metadata)
        self.assertTrue(metadata['measure_sampling'])
