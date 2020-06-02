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
from test.terra.utils.mock import FakeFailureQasmSimulator, FakeSuccessQasmSimulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import transpile, assemble
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
        qobj = assemble(quantum_circuit)
        result = mocked_backend.run(qobj).result()
        self.assertSuccess(result)

    def test_simulation_failed(self):
        """Test the we properly manage simulation failures."""
        mocked_backend = FakeFailureQasmSimulator(time_alive=0)
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        failed_circuit = QuantumCircuit(qr, cr)
        quantum_circuit = transpile(failed_circuit, mocked_backend)
        qobj = assemble(quantum_circuit)
        job = mocked_backend.run(qobj)
        self.assertRaises(AerError, job.result)
