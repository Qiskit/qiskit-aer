# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
QasmSimulator Integration Tests
"""
from test.terra.utils.mock import FakeFailureQasmSimulator, FakeSuccessQasmSimulator
from qiskit.transpiler import transpile
from qiskit.compiler import assemble_circuits
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import AerError


class QasmBasicsTests:
    """QasmSimulator basic tests."""

    def test_simulation_succeed(self):
        """Test the we properly manage simulation failures."""
        mocked_backend = FakeSuccessQasmSimulator(time_alive=0)
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        succeed_circuit = QuantumCircuit(qr, cr)
        quantum_circuit = transpile(succeed_circuit, mocked_backend)
        qobj = assemble_circuits(quantum_circuit)
        result = mocked_backend.run(qobj).result()
        self.is_completed(result)


    def test_simulation_failed(self):
        """Test the we properly manage simulation failures."""
        mocked_backend = FakeFailureQasmSimulator(time_alive=0)
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        failed_circuit = QuantumCircuit(qr, cr)
        quantum_circuit = transpile(failed_circuit, mocked_backend)
        qobj = assemble_circuits(quantum_circuit)
        job = mocked_backend.run(qobj)
        self.assertRaises(AerError, job.result)
