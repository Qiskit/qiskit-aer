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
QasmSimulator Integration Tests
"""
# pylint: disable=no-member
import copy

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import QuantumVolume, QFT
from qiskit.compiler import assemble, transpile
from qiskit.providers.aer import QasmSimulator

class QasmChunkTests:
    """QasmSimulator Multi-chunk tests."""

    SIMULATOR = QasmSimulator()
    SUPPORTED_QASM_METHODS = [
        'statevector', 'statevector_gpu', 'statevector_thrust',
        'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust'
    ]

    def test_chunk_QuantumVolume(self):
        """Test multi-chunk with quantum volume"""
        shots = 100
        num_qubits = 4
        depth = 10
        backend_options = self.BACKEND_OPTS.copy()
        backend_options_no_chunk = self.BACKEND_OPTS.copy()
        backend_options_no_chunk.pop("blocking_enable")
        backend_options_no_chunk.pop("blocking_qubits")

        circuit = transpile(QuantumVolume(num_qubits, depth, seed=0),
                            backend=self.SIMULATOR,
                            optimization_level=0)
        circuit.measure_all()
        qobj = assemble(circuit, shots=shots, memory=True)
        result = self.SIMULATOR.run(qobj, **backend_options_no_chunk).result()
        counts_no_chunk = result.get_counts(circuit)
        result = self.SIMULATOR.run(qobj, **backend_options).result()
        counts = result.get_counts(circuit)

        self.assertEqual(counts_no_chunk,counts)

    def test_chunk_QuantumVolumeWithFusion(self):
        """Test multi-chunk with fused quantum volume"""
        shots = 100
        num_qubits = 8
        depth = 10
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['fusion_threshold'] = 5
        backend_options["blocking_qubits"] = 4
        backend_options_no_chunk = self.BACKEND_OPTS.copy()
        backend_options_no_chunk.pop("blocking_enable")
        backend_options_no_chunk.pop("blocking_qubits")
        backend_options_no_chunk['fusion_enable'] = True
        backend_options_no_chunk['fusion_threshold'] = 5

        circuit = transpile(QuantumVolume(num_qubits, depth, seed=0),
                            backend=self.SIMULATOR,
                            optimization_level=0)
        circuit.measure_all()
        qobj = assemble(circuit, shots=shots, memory=True)
        result = self.SIMULATOR.run(qobj, **backend_options_no_chunk).result()
        counts_no_chunk = result.get_counts(circuit)
        result = self.SIMULATOR.run(qobj, **backend_options).result()
        counts = result.get_counts(circuit)

        self.assertEqual(counts_no_chunk,counts)

    def test_chunk_QFTWithFusion(self):
        """Test multi-chunk with fused QFT (testing multi-chunk diagonal matrix)"""
        shots = 100
        num_qubits = 8
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['fusion_enable'] = True
        backend_options['fusion_threshold'] = 5
        backend_options["blocking_qubits"] = 4
        backend_options_no_chunk = self.BACKEND_OPTS.copy()
        backend_options_no_chunk.pop("blocking_enable")
        backend_options_no_chunk.pop("blocking_qubits")
        backend_options_no_chunk['fusion_enable'] = True
        backend_options_no_chunk['fusion_threshold'] = 5

        circuit = transpile(QFT(num_qubits),
                            backend=self.SIMULATOR,
                            optimization_level=0)
        circuit.measure_all()
        qobj = assemble(circuit, shots=shots, memory=True)
        result = self.SIMULATOR.run(qobj, **backend_options_no_chunk).result()
        counts_no_chunk = result.get_counts(circuit)
        result = self.SIMULATOR.run(qobj, **backend_options).result()
        counts = result.get_counts(circuit)

        self.assertEqual(counts_no_chunk,counts)

    def test_chunk_pauli(self):
        """Test multi-chunk pauli gate"""
        shots = 100
        backend_options = self.BACKEND_OPTS.copy()
        backend_options["blocking_qubits"] = 3
        backend_options['fusion_enable'] = False
        backend_options_no_chunk = self.BACKEND_OPTS.copy()
        backend_options_no_chunk.pop("blocking_enable")
        backend_options_no_chunk.pop("blocking_qubits")

        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        regs = (qr, cr)
        circuit = QuantumCircuit(*regs)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.h(qr[2])
        circuit.h(qr[3])
        circuit.h(qr[4])
        circuit.pauli('YXZYX',qr)
        circuit.measure_all()

        qobj = assemble(circuit, shots=shots, memory=True)
        result = self.SIMULATOR.run(qobj, **backend_options_no_chunk).result()
        counts_no_chunk = result.get_counts(circuit)
        result = self.SIMULATOR.run(qobj, **backend_options).result()
        counts = result.get_counts(circuit)

        self.assertEqual(counts_no_chunk,counts)

