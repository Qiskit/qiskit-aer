# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
AerSimulator Integration Tests
"""
from ddt import ddt
import json
from qiskit import transpile, QuantumRegister, ClassicalRegister, QuantumCircuit, Aer
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import ReadoutError, depolarizing_error
from qiskit.providers.models import BackendProperties
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


    def device_properties(self):
        properties = {"general": [],
                      "last_update_date": "2019-04-22T03:26:08+00:00", 
                      "gates": [
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [0]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [0]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [0]}, 
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [1]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [1]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [1]}, 
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [2]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [2]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [2]}, 
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [3]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [3]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [3]}, 
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [4]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [4]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [4]}, 
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [5]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [5]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [5]}, 
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [6]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [6]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [6]}, 
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [7]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [7]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [7]}, 
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [8]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [8]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [8]}, 
                          {"gate": "u1", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [9]}, 
                          {"gate": "u2", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [9]}, 
                          {"gate": "u3", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-23T01:45:04+00:00", "unit": ""}], "qubits": [9]}, 
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:26:00+00:00", "unit": ""}], "qubits": [0, 1], "name": "CX0_1"}, 
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:29:15+00:00", "unit": ""}], "qubits": [1, 2], "name": "CX1_2"}, 
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:32:48+00:00", "unit": ""}], "qubits": [2, 3], "name": "CX2_3"},
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:26:00+00:00", "unit": ""}], "qubits": [3, 4], "name": "CX3_4"}, 
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:29:15+00:00", "unit": ""}], "qubits": [4, 5], "name": "CX4_5"}, 
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:32:48+00:00", "unit": ""}], "qubits": [5, 6], "name": "CX5_6"},
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:26:00+00:00", "unit": ""}], "qubits": [6, 7], "name": "CX6_7"}, 
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:29:15+00:00", "unit": ""}], "qubits": [7, 8], "name": "CX7_8"}, 
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:32:48+00:00", "unit": ""}], "qubits": [8, 9], "name": "CX8_9"},
                          {"gate": "cx", "parameters": [{"name": "gate_error", "value": 0.001, "date": "2019-04-22T02:26:00+00:00", "unit": ""}], "qubits": [9, 0], "name": "CX9_0"}], 
                      "qubits": [
                          [
                              {"name": "T1", "value": 23.809868955712616, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 43.41142418044261, "date": "2019-04-22T01:33:33+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 5.032871440179164, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.03489999999999993, "date": "2019-04-22T01:29:47+00:00", "unit": ""}], 
                          [
                              {"name": "T1", "value": 68.14048367144501, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 56.95903203933663, "date": "2019-04-22T01:34:36+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 4.896209948700639, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.19589999999999996, "date": "2019-04-22T01:29:47+00:00", "unit": ""}], 
                          [
                              {"name": "T1", "value": 83.26776276928099, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 23.49615795695734, "date": "2019-04-22T01:31:32+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 5.100093544085939, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.09050000000000002, "date": "2019-04-22T01:29:47+00:00", "unit": ""}], 
                          [
                              {"name": "T1", "value": 57.397746445609975, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 98.47976889309517, "date": "2019-04-22T01:32:32+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 5.238526396839902, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.24350000000000005, "date": "2019-04-20T15:31:39+00:00", "unit": ""}], 
                          [
                              {"name": "T1", "value": 23.809868955712616, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 43.41142418044261, "date": "2019-04-22T01:33:33+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 5.032871440179164, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.03489999999999993, "date": "2019-04-22T01:29:47+00:00", "unit": ""}], 
                          [
                              {"name": "T1", "value": 68.14048367144501, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 56.95903203933663, "date": "2019-04-22T01:34:36+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 4.896209948700639, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.19589999999999996, "date": "2019-04-22T01:29:47+00:00", "unit": ""}], 
                          [
                              {"name": "T1", "value": 83.26776276928099, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 23.49615795695734, "date": "2019-04-22T01:31:32+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 5.100093544085939, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.09050000000000002, "date": "2019-04-22T01:29:47+00:00", "unit": ""}], 
                          [
                              {"name": "T1", "value": 57.397746445609975, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 98.47976889309517, "date": "2019-04-22T01:32:32+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 5.238526396839902, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.24350000000000005, "date": "2019-04-20T15:31:39+00:00", "unit": ""}], 
                          [
                              {"name": "T1", "value": 23.809868955712616, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 43.41142418044261, "date": "2019-04-22T01:33:33+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 5.032871440179164, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.03489999999999993, "date": "2019-04-22T01:29:47+00:00", "unit": ""}], 
                          [
                              {"name": "T1", "value": 68.14048367144501, "date": "2019-04-22T01:30:15+00:00", "unit": "\u00b5s"}, 
                              {"name": "T2", "value": 56.95903203933663, "date": "2019-04-22T01:34:36+00:00", "unit": "\u00b5s"}, 
                              {"name": "frequency", "value": 4.896209948700639, "date": "2019-04-22T03:26:08+00:00", "unit": "GHz"}, 
                              {"name": "readout_error", "value": 0.19589999999999996, "date": "2019-04-22T01:29:47+00:00", "unit": ""}], 
                          ], 
                      "backend_name": "mock_4q", 
                      "backend_version": "1.0.0"}
        return BackendProperties.from_dict(properties)

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
        metadata = result.results[0].metadata
        self.assertEqual(metadata["num_qubits"], 2, msg="wrong number of truncated qubits.")
        self.assertEqual(metadata["active_input_qubits"], [10, 20], msg="incorrect truncated qubits.")

    def test_truncate_default(self):
        """Test truncation with noise model option"""
        coupling_map = [# 10-qubit device
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
            [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]
        ]
        noise_model = NoiseModel.from_backend(self.device_properties())
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
        noise_model = NoiseModel.from_backend(self.device_properties())
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
        noise_model = NoiseModel.from_backend(self.device_properties())
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
