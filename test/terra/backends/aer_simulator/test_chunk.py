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
AerSimulator Integration Tests
"""
# pylint: disable=no-member
import copy

from ddt import ddt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import QuantumVolume, QFT
from qiskit.compiler import transpile
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


@ddt
class TestChunkSimulators(SimulatorTestCase):
    """AerSimulator Multi-chunk tests."""

    OPTIONS = {"seed_simulator": 271828, "max_parallel_threads": 1}

    @supported_methods(["statevector", "density_matrix"])
    def test_chunk_QuantumVolume(self, method, device):
        """Test multi-chunk with quantum volume"""
        opts = {"blocking_enable": True, "blocking_qubits": 2}
        opts["basis_gates"] = ["h", "cx", "u3"]

        backend = self.backend(method=method, device=device, **opts)
        backend_no_chunk = self.backend(method=method, device=device)

        shots = 100
        num_qubits = 4
        depth = 10
        circuit = transpile(
            QuantumVolume(num_qubits, depth, seed=0), backend=backend, optimization_level=0
        )
        circuit.measure_all()

        result = backend.run(circuit, shots=shots, memory=True).result()
        counts = result.get_counts(circuit)
        result_no_chunk = backend_no_chunk.run(circuit, shots=shots, memory=True).result()
        counts_no_chunk = result_no_chunk.get_counts(circuit)

        self.assertEqual(counts_no_chunk, counts)

    @supported_methods(["statevector", "density_matrix"])
    def test_chunk_QuantumVolumeWithFusion(self, method, device):
        """Test multi-chunk with fused quantum volume"""
        opts_no_chunk = {
            "fusion_enable": True,
            "fusion_threshold": 5,
            "fusion_max_qubit": 4,
        }
        opts_chunk = copy.copy(opts_no_chunk)
        opts_chunk["blocking_enable"] = True
        opts_chunk["blocking_qubits"] = 5
        opts_chunk["basis_gates"] = ["h", "cx", "u3"]

        backend = self.backend(method=method, device=device, **opts_chunk)
        backend_no_chunk = self.backend(method=method, device=device, **opts_no_chunk)

        shots = 100
        num_qubits = 8
        depth = 10
        circuit = transpile(
            QuantumVolume(num_qubits, depth, seed=0), backend=backend, optimization_level=0
        )
        circuit.measure_all()

        result = backend.run(circuit, shots=shots, memory=True).result()
        counts = result.get_counts(circuit)
        result_no_chunk = backend_no_chunk.run(circuit, shots=shots, memory=True).result()
        counts_no_chunk = result_no_chunk.get_counts(circuit)

        self.assertEqual(counts_no_chunk, counts)

    @supported_methods(["statevector", "density_matrix"])
    def test_chunk_QFT(self, method, device):
        """Test multi-chunk with QFT"""
        opts_no_chunk = {
            "fusion_enable": False,
            "fusion_threshold": 10,
        }
        opts_chunk = copy.copy(opts_no_chunk)
        opts_chunk["blocking_enable"] = True
        opts_chunk["blocking_qubits"] = 2

        backend = self.backend(method=method, device=device, **opts_chunk)
        backend_no_chunk = self.backend(method=method, device=device, **opts_no_chunk)

        shots = 100
        num_qubits = 3
        circuit = transpile(QFT(num_qubits), backend=backend, optimization_level=0)
        circuit.measure_all()

        result = backend.run(circuit, shots=shots, memory=True).result()
        counts = result.get_counts(circuit)
        result_no_chunk = backend_no_chunk.run(circuit, shots=shots, memory=True).result()
        counts_no_chunk = result_no_chunk.get_counts(circuit)

        self.assertEqual(counts_no_chunk, counts)

    @supported_methods(["statevector", "density_matrix"])
    def test_chunk_QFTWithFusion(self, method, device):
        """Test multi-chunk with fused QFT (testing multi-chunk diagonal matrix)"""
        opts_no_chunk = {
            "fusion_enable": True,
            "fusion_threshold": 5,
        }
        opts_chunk = copy.copy(opts_no_chunk)
        opts_chunk["blocking_enable"] = True
        opts_chunk["blocking_qubits"] = 4

        backend = self.backend(method=method, device=device, **opts_chunk)
        backend_no_chunk = self.backend(method=method, device=device, **opts_no_chunk)

        shots = 100
        num_qubits = 8
        circuit = transpile(QFT(num_qubits), backend=backend, optimization_level=0)
        circuit.measure_all()

        result = backend.run(circuit, shots=shots, memory=True).result()
        counts = result.get_counts(circuit)
        result_no_chunk = backend_no_chunk.run(circuit, shots=shots, memory=True).result()
        counts_no_chunk = result_no_chunk.get_counts(circuit)

        self.assertEqual(counts_no_chunk, counts)

    @supported_methods(["statevector", "density_matrix"])
    def test_chunk_pauli(self, method, device):
        """Test multi-chunk pauli gate"""
        opts_no_chunk = {"fusion_enable": False}
        opts_chunk = copy.copy(opts_no_chunk)
        opts_chunk["blocking_enable"] = True
        opts_chunk["blocking_qubits"] = 3

        backend = self.backend(method=method, device=device, **opts_chunk)
        backend_no_chunk = self.backend(method=method, device=device, **opts_no_chunk)

        shots = 100

        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        regs = (qr, cr)
        circuit = QuantumCircuit(*regs)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.h(qr[2])
        circuit.h(qr[3])
        circuit.h(qr[4])
        circuit.pauli("YXZYX", qr)
        circuit.measure_all()

        result = backend.run(circuit, shots=shots, memory=True).result()
        counts = result.get_counts(circuit)
        result_no_chunk = backend_no_chunk.run(circuit, shots=shots, memory=True).result()
        counts_no_chunk = result_no_chunk.get_counts(circuit)

        self.assertEqual(counts_no_chunk, counts)
