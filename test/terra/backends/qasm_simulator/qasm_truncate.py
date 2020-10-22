# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""
import json
from qiskit import execute, QuantumRegister, ClassicalRegister, QuantumCircuit, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import ReadoutError, depolarizing_error
from qiskit.providers.models import BackendProperties


class QasmQubitsTruncateTests:
    """QasmSimulator Qubits Truncate tests."""

    SIMULATOR = QasmSimulator()

    def create_circuit_for_truncate(self):
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circuit = QuantumCircuit(qr, cr)
        circuit.u3(0.1,0.1,0.1,qr[1])
        circuit.barrier(qr)
        circuit.x(qr[2])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.x(qr[3])
        circuit.barrier(qr)
        circuit.u3(0.1,0.1,0.1,qr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
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
        
        # Circuit that uses just 2-qubits
        circuit = QuantumCircuit(50, 2)
        circuit.x(10)
        circuit.x(20)
        circuit.measure(10, 0)
        circuit.measure(20, 1)


        qasm_sim = Aer.get_backend('qasm_simulator')
        backend_options = self.BACKEND_OPTS.copy()
        backend_options["truncate_verbose"] = True
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result = execute(circuit, 
                         qasm_sim, 
                         shots=100,
                         **backend_options).result()
        metadata = result.results[0].metadata
        self.assertTrue('truncate_qubits' in metadata, msg="truncate_qubits must work.")
        active_qubits = sorted(metadata['truncate_qubits'].get('active_qubits', []))
        mapping = sorted(metadata['truncate_qubits'].get('mapping', []))
        self.assertEqual(active_qubits, [10, 20])
        self.assertIn(mapping, [[[10, 0], [20, 1]], [[10, 1], [20, 0]]])

    def test_truncate_nonlocal_noise(self):
        """Test qubit truncation with non-local noise."""
        
        # Circuit that uses just 2-qubits
        circuit = QuantumCircuit(10, 1)
        circuit.x(5)
        circuit.measure(5, 0)

        # Add non-local 2-qubit depolarizing error
        # that acts on qubits [4, 6] when X applied to qubit 5
        noise_model = NoiseModel()
        error = depolarizing_error(0.1, 2)
        noise_model.add_nonlocal_quantum_error(error, ['x'], [5], [4, 6])

        qasm_sim = Aer.get_backend('qasm_simulator')
        backend_options = self.BACKEND_OPTS.copy()
        backend_options["truncate_verbose"] = True
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result = execute(circuit, 
                         qasm_sim, 
                         shots=100,
                         noise_model=noise_model,
                         **backend_options).result()
        metadata = result.results[0].metadata
        self.assertTrue('truncate_qubits' in metadata, msg="truncate_qubits must work.")
        active_qubits = sorted(metadata['truncate_qubits'].get('active_qubits', []))
        mapping = metadata['truncate_qubits'].get('mapping', [])
        active_remapped = sorted([i[1] for i in mapping if i[0] in active_qubits])
        self.assertEqual(active_qubits, [4, 5, 6])
        self.assertEqual(active_remapped, [0, 1, 2])

    def test_truncate(self):
        """Test truncation with noise model option"""
        circuit = self.create_circuit_for_truncate()
        
        qasm_sim = Aer.get_backend('qasm_simulator')
        backend_options = self.BACKEND_OPTS.copy()
        backend_options["truncate_verbose"] = True
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result = execute(circuit, 
                            qasm_sim, 
                            noise_model=NoiseModel.from_backend(self.device_properties()), 
                            shots=100,
                            coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]], # 10-qubit device
                            **backend_options).result()
                            
        self.assertTrue('truncate_qubits' in result.to_dict()['results'][0]['metadata'], msg="truncate_qubits must work.")

    def test_no_truncate(self):
        """Test truncation with noise model option"""
        circuit = self.create_circuit_for_truncate()
        
        qasm_sim = Aer.get_backend('qasm_simulator')
        backend_options = self.BACKEND_OPTS.copy()
        backend_options["truncate_verbose"] = True
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result = execute(circuit, 
                            qasm_sim, 
                            noise_model=NoiseModel.from_backend(self.device_properties()), 
                            shots=100,
                            coupling_map=[[1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]], # 4-qubit device
                            **backend_options).result()
                            
        self.assertFalse('truncate_qubits' in result.to_dict()['results'][0]['metadata'], msg="truncate_qubits must work.")

    
    def test_truncate_disable(self):
        """Test explicitly disabling truncation with noise model option"""
        circuit = self.create_circuit_for_truncate()
        
        qasm_sim = Aer.get_backend('qasm_simulator')
        backend_options = self.BACKEND_OPTS.copy()
        backend_options["truncate_verbose"] = True
        backend_options["truncate_enable"] = False
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result = execute(circuit, 
                            qasm_sim, 
                            noise_model=NoiseModel.from_backend(self.device_properties()), 
                            shots=100,
                            coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]], # 10-qubit device
                            **backend_options).result()
                            
        self.assertFalse('truncate_qubits' in result.to_dict()['results'][0]['metadata'], msg="truncate_qubits must not work.")
     