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
from math import sqrt
from copy import deepcopy
from ddt import ddt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, assemble
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import HGate
from test.terra.reference import ref_algorithms

from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


@ddt
class TestVariousCircuit(SimulatorTestCase):
    """AerSimulator tests to simulate various types of circuits"""

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_quantum_register_circuit(self, method, device):
        """Test circuits with quantum registers."""

        qubits = QuantumRegister(3)
        clbits = ClassicalRegister(3)

        circuit = QuantumCircuit(qubits, clbits)
        circuit.h(qubits[0])
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[0], qubits[2])

        for q, c in zip(qubits, clbits):
            circuit.measure(q, c)

        backend = self.backend(method=method, device=device, seed_simulator=1111)

        shots = 1000
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [{"0x0": 500, "0x7": 500}], delta=0.05 * shots)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_qubits_circuit(self, method, device):
        """Test circuits with quantum registers."""

        qubits = QuantumRegister(3)
        clbits = ClassicalRegister(3)

        circuit = QuantumCircuit()
        circuit.add_bits(qubits)
        circuit.add_bits(clbits)
        circuit.h(qubits[0])
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[0], qubits[2])

        for q, c in zip(qubits, clbits):
            circuit.measure(q, c)

        backend = self.backend(method=method, device=device, seed_simulator=1111)

        shots = 1000
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [{"0x0": 500, "0x7": 500}], delta=0.05 * shots)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_qubits_quantum_register_circuit(self, method, device):
        """Test circuits with quantum registers."""

        qubits0 = QuantumRegister(2)
        clbits1 = ClassicalRegister(2)
        qubits1 = QuantumRegister(1)
        clbits2 = ClassicalRegister(1)

        circuit = QuantumCircuit(qubits0, clbits1)
        circuit.add_bits(qubits1)
        circuit.add_bits(clbits2)
        circuit.h(qubits0[0])
        circuit.cx(qubits0[0], qubits0[1])
        circuit.cx(qubits0[0], qubits1[0])

        for qubits, clbits in zip([qubits0, qubits1], [clbits1, clbits2]):
            for q, c in zip(qubits, clbits):
                circuit.measure(q, c)

        backend = self.backend(method=method, device=device, seed_simulator=1111)

        shots = 1000
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [{"0x0": 500, "0x7": 500}], delta=0.05 * shots)

        qubits0 = QuantumRegister(1)
        clbits1 = ClassicalRegister(1)
        qubits1 = QuantumRegister(1)
        clbits2 = ClassicalRegister(1)
        qubits2 = QuantumRegister(1)
        clbits3 = ClassicalRegister(1)

        circuit = QuantumCircuit(qubits0, clbits1)
        circuit.add_bits(qubits1)
        circuit.add_bits(clbits2)
        circuit.add_register(qubits2)
        circuit.add_register(clbits3)
        circuit.h(qubits0[0])
        circuit.cx(qubits0[0], qubits1[0])
        circuit.cx(qubits1[0], qubits2[0])

        for qubits, clbits in zip([qubits0, qubits1, qubits2], [clbits1, clbits2, clbits3]):
            for q, c in zip(qubits, clbits):
                circuit.measure(q, c)

        backend = self.backend(method=method, device=device, seed_simulator=1111)

        shots = 1000
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [{"0x0": 500, "0x7": 500}], delta=0.05 * shots)

    def test_partial_result_a_single_invalid_circuit(self):
        """Test a partial result is returned with a job with a valid and invalid circuit."""

        circuits = []
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        qc_2 = QuantumCircuit(50)
        qc_2.h(range(50))
        qc_2.measure_all()
        circuits.append(qc_2)
        circuits.append(qc)
        backend = self.backend()
        shots = 100
        result = backend.run(circuits, shots=shots, method="statevector").result()
        self.assertEqual(result.status, "PARTIAL COMPLETED")
        self.assertTrue(hasattr(result.results[1].data, "counts"))
        self.assertFalse(hasattr(result.results[0].data, "counts"))

    def test_metadata_protected(self):
        """Test metadata is consitently viewed from users"""

        qc = QuantumCircuit(2)
        qc.metadata = {"foo": "bar", "object": object}

        circuits = [qc.copy() for _ in range(5)]

        backend = self.backend()
        job = backend.run(circuits)

        for circuit in circuits:
            self.assertTrue("foo" in circuit.metadata)
            self.assertEqual(circuit.metadata["foo"], "bar")
            self.assertEqual(circuit.metadata["object"], object)

        deepcopy(job.result())

    def test_run_qobj(self):
        """Test qobj run"""

        qubits = QuantumRegister(3)
        clbits = ClassicalRegister(3)

        circuit = QuantumCircuit(qubits, clbits)
        circuit.h(qubits[0])
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[0], qubits[2])

        for q, c in zip(qubits, clbits):
            circuit.measure(q, c)

        backend = self.backend()

        shots = 1000
        with self.assertWarns(DeprecationWarning):
            result = backend.run(assemble(circuit), shots=shots).result()

        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [{"0x0": 500, "0x7": 500}], delta=0.05 * shots)

    def test_numpy_integer_shots(self):
        """Test implicit cast of shot option from np.int_ to int."""

        backend = self.backend()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        shots = 333

        for np_type in {
            np.int_,
            np.uint,
            np.short,
            np.ushort,
            np.intc,
            np.uintc,
            np.longlong,
            np.ulonglong,
        }:
            result = backend.run(qc, shots=np_type(shots), method="statevector").result()
            self.assertSuccess(result)
            self.assertEqual(sum([result.get_counts()[key] for key in result.get_counts()]), shots)

    def test_floating_shots(self):
        """Test implicit cast of shot option from float to int."""

        backend = self.backend()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        for shots in {1e4, "300"}:
            with self.assertWarns(DeprecationWarning):
                result = backend.run(qc, shots=shots, method="statevector").result()
            shots = int(shots)
            self.assertSuccess(result)
            self.assertEqual(sum([result.get_counts()[key] for key in result.get_counts()]), shots)

    def test_invalid_parameters(self):
        """Test gates with invalid parameter length."""

        backend = self.backend()

        class Custom(Gate):
            def __init__(self, label=None):
                super().__init__("p", 1, [], label=label)

            def _define(self):
                q = QuantumRegister(1, "q")
                qc = QuantumCircuit(q, name=self.name)
                qc._append(HGate(), [q[0]], [])
                self.definition = qc

        qc = QuantumCircuit(1)
        qc.append(Custom(), [0])
        qc.measure_all()

        try:
            backend.run(qc).result()
            self.fail("do not reach here")
        except Exception as e:
            self.assertTrue('"params" is incorrect length' in repr(e))
