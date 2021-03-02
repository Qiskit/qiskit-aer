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

class QasmMPITests:
    """QasmSimulator MPI tests."""

    ref_counts_qv10 = {'1011001001': 1, '1110101010': 1, '0101111001': 1, '0000110110': 1, '0110100101': 1, '0000000011': 1, '1110001001': 1, '0101101010': 1, '1100011001': 1, '1111001101': 1, '0011001000': 1, '1011011100': 1, '0010010101': 2, '1010001110': 1, '1001000110': 1, '0111010111': 1, '0001001101': 1, '0101111010': 1, '1110010001': 1, '1001100011': 1, '1011111111': 1, '1100011011': 1, '1001111000': 2, '1110110001': 1, '0100010111': 1, '1101100101': 2, '1101111101': 1, '1111100011': 1, '1000101101': 1, '1110100000': 2, '1001110110': 1, '1101110010': 1, '1001100110': 1, '1100001100': 2, '1001000000': 1, '1111010111': 1, '0110101101': 1, '0100110100': 1, '1110011100': 1, '0000101010': 1, '0111101001': 2, '1101000101': 1, '0101011000': 1, '1110100110': 1, '0111011110': 1, '0010010110': 1, '0101100011': 1, '1110000110': 1, '1110000101': 1, '0011111011': 1, '0001110011': 1, '0100101010': 1, '0100101011': 1, '1111011001': 1, '0001010100': 1, '1110100010': 1, '1001000100': 1, '0110010001': 1, '0101010011': 1, '0010000100': 1, '0000010000': 1, '1001011111': 1, '0001111111': 1, '0001110100': 1, '1100000011': 1, '0011101010': 1, '1101111000': 1, '1011100001': 1, '0101000110': 1, '0000001111': 1, '0100010010': 1, '0011100110': 1, '0100001011': 1, '0101101100': 1, '0001100101': 1, '1010111010': 1, '0000111111': 1, '0001001001': 1, '0011010011': 1, '0010101001': 1, '1110111010': 1, '1101001001': 1, '1100010011': 1, '1110001000': 1, '1000000110': 1, '1010111001': 1, '1111011110': 1, '1111111001': 1, '0001100110': 1, '0010001110': 1, '1011010101': 1, '0011101000': 1, '1110011110': 1, '1101111011': 1}
    ref_counts_qft10 = {'1110011010': 1, '0101111010': 1, '1001000110': 1, '0110011101': 1, '0101100001': 1, '0110010100': 1, '1101110011': 1, '0101010000': 1, '0000000010': 1, '0011111111': 1, '1100101011': 1, '0011100000': 1, '1101000110': 1, '1110010000': 1, '1111011011': 1, '1110001010': 1, '0111100011': 1, '1011000100': 1, '1001100000': 1, '1000100001': 1, '0011000011': 1, '0011111000': 1, '1001110110': 1, '1100101001': 1, '1101100110': 1, '0000101001': 1, '0010011011': 2, '0000001101': 1, '0011010000': 1, '1111111000': 1, '0011100001': 1, '1101001011': 1, '1110001100': 1, '1111010001': 1, '1001100011': 2, '1000101100': 1, '1010110100': 1, '0000110000': 1, '0001100001': 2, '0101010100': 1, '1011101011': 1, '1001010000': 1, '1011100110': 1, '0111001101': 1, '0010000100': 1, '0010011000': 1, '1011011001': 1, '1011100001': 1, '0011100011': 1, '1100001100': 1, '1011011111': 1, '1101111010': 1, '0100000100': 1, '1100111111': 1, '1101101110': 1, '1000001010': 1, '0101001001': 1, '0001101100': 1, '1111000101': 1, '0100100110': 1, '1001001001': 1, '0000111001': 1, '0000001100': 1, '1110100011': 1, '1111001111': 1, '0001010001': 1, '0011101111': 1, '1110000100': 1, '0001000110': 1, '1110000110': 2, '0101100100': 1, '0010101000': 1, '0111000001': 1, '1111010110': 1, '0001101111': 1, '1000101011': 1, '1100001110': 1, '0100010100': 1, '0100111010': 1, '1110101110': 1, '0010001101': 1, '1011011000': 1, '1010111011': 1, '0111010101': 2, '0100011110': 1, '0100010011': 1, '1101010111': 1, '1010100111': 1, '0100111110': 1, '1010100101': 1, '0001001001': 1, '1101101001': 1, '1011101010': 1, '1010111110': 1, '0001111101': 1}

    SIMULATOR = QasmSimulator()
    SUPPORTED_QASM_METHODS = [
        'statevector', 'statevector_gpu', 'statevector_thrust'
    ]
    MPI_OPTIONS = {
        "blocking_enable": True,
        "blocking_qubits": 6,
        "blocking_ignore_diagonal": True,
        "max_parallel_threads": 1
    }

    def test_MPI_QuantumVolume(self):
        """Test MPI with quantum volume"""
        shots = 100
        num_qubits = 10
        depth = 10
        backend_options = self.BACKEND_OPTS.copy()
        for opt, val in self.MPI_OPTIONS.items():
            backend_options[opt] = val

        circuit = transpile(QuantumVolume(num_qubits, depth, seed=0),
                            backend=self.SIMULATOR,
                            optimization_level=0)
        circuit.measure_all()
        qobj = assemble(circuit, shots=shots, memory=True)
        result = self.SIMULATOR.run(qobj, **backend_options).result()

        counts = result.get_counts(circuit)
        # Comparing counts with pre-computed counts
        self.assertEqual(counts, self.ref_counts_qv10)

    def test_MPI_QFT(self):
        """Test MPI with QFT"""
        shots = 100
        num_qubits = 10
        backend_options = self.BACKEND_OPTS.copy()
        for opt, val in self.MPI_OPTIONS.items():
            backend_options[opt] = val

        circuit = transpile(QFT(num_qubits),
                            backend=self.SIMULATOR,
                            optimization_level=0)
        circuit.measure_all()
        qobj = assemble(circuit, shots=shots, memory=True)
        result = self.SIMULATOR.run(qobj, **backend_options).result()

        counts = result.get_counts(circuit)
        #comparing counts with pre-computed counts
        self.assertEqual(counts, self.ref_counts_qft10)
