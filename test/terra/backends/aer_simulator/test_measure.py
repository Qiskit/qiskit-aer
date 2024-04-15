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
from test.terra.reference import ref_measure
from qiskit import QuantumCircuit
from qiskit import transpile
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import ReadoutError, depolarizing_error
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info.random import random_unitary
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods
import numpy as np

import os

SUPPORTED_METHODS = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "extended_stabilizer",
    "tensor_network",
]


@ddt
class TestMeasure(SimulatorTestCase):
    """AerSimulator measure tests."""

    OPTIONS = {"seed_simulator": 41411}

    # ---------------------------------------------------------------------
    # Test measure
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_measure_deterministic_with_sampling(self, method, device):
        """Test AerSimulator measure with deterministic counts with sampling"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        target_counts = ref_measure.measure_counts_deterministic(shots)
        target_memory = ref_measure.measure_memory_deterministic(shots)
        result = backend.run(circuits, memory=True, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_deterministic_without_sampling(self, method, device):
        """Test AerSimulator measure with deterministic counts without sampling"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=False)
        target_counts = ref_measure.measure_counts_deterministic(shots)
        target_memory = ref_measure.measure_memory_deterministic(shots)
        result = backend.run(circuits, memory=True, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_nondeterministic_with_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts with sampling"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        circuits = ref_measure.measure_circuits_nondeterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        # Test sampling was enabled
        for res in result.results:
            self.assertIn("measure_sampling", res.metadata)
            self.assertEqual(res.metadata["measure_sampling"], True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_nondeterministic_without_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts without sampling"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        delta = 0.05
        circuits = ref_measure.measure_circuits_nondeterministic(allow_sampling=False)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=delta * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_sampling_with_readouterror(self, method, device):
        """Test AerSimulator measure with deterministic counts with sampling and readout-error"""
        readout_error = [0.01, 0.1]
        noise_model = NoiseModel()
        readout = [
            [1.0 - readout_error[0], readout_error[0]],
            [readout_error[1], 1.0 - readout_error[1]],
        ]
        noise_model.add_all_qubit_readout_error(ReadoutError(readout))

        backend = self.backend(method=method, device=device, noise_model=noise_model)
        shots = 1000
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_sampling_with_quantum_noise(self, method, device):
        """Test AerSimulator measure with deterministic counts with sampling and readout-error"""
        readout_error = [0.01, 0.1]
        noise_model = NoiseModel()
        depolarizing = {"u3": (1, 0.001), "cx": (2, 0.02)}
        readout = [
            [1.0 - readout_error[0], readout_error[0]],
            [readout_error[1], 1.0 - readout_error[1]],
        ]
        noise_model.add_all_qubit_readout_error(ReadoutError(readout))
        for gate, (num_qubits, gate_error) in depolarizing.items():
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(gate_error, num_qubits), gate
            )

        backend = self.backend(method=method, device=device, noise_model=noise_model)
        shots = 1000
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        method_used = result.results[0].metadata.get("method")
        sampling = method_used == "density_matrix" or method_used == "tensor_network"
        self.compare_result_metadata(result, circuits, "measure_sampling", sampling)

    # ---------------------------------------------------------------------
    # Test multi-qubit measure qobj instruction
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_measure_deterministic_multi_qubit_with_sampling(self, method, device):
        """Test AerSimulator multi-qubit measure with deterministic counts with sampling"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_measure.multiqubit_measure_circuits_deterministic(allow_sampling=True)
        target_counts = ref_measure.multiqubit_measure_counts_deterministic(shots)
        target_memory = ref_measure.multiqubit_measure_memory_deterministic(shots)
        result = backend.run(circuits, memory=True, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_deterministic_multi_qubit_without_sampling(self, method, device):
        """Test AerSimulator multi-qubit measure with deterministic counts without sampling"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_measure.multiqubit_measure_circuits_deterministic(allow_sampling=False)
        target_counts = ref_measure.multiqubit_measure_counts_deterministic(shots)
        target_memory = ref_measure.multiqubit_measure_memory_deterministic(shots)
        result = backend.run(circuits, memory=True, shots=shots).result()
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_nondeterministic_multi_qubit_with_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(allow_sampling=True)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    @supported_methods(SUPPORTED_METHODS)
    def test_measure_nondeterministic_multi_qubit_without_sampling(self, method, device):
        """Test AerSimulator measure with non-deterministic counts"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        delta = 0.05
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(allow_sampling=False)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=delta * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    # ---------------------------------------------------------------------
    # Test stabilizer measure
    # ---------------------------------------------------------------------
    @supported_methods(["stabilizer"])
    def test_measure_stablizer_64bit(self, method, device):
        backend = self.backend(method=method, device=device)
        shots = 10000
        delta = 0.05
        circ = QuantumCircuit(65, 32)

        circ.reset(0)
        for i in range(0, 30, 6):
            circ.h(i)
            circ.h(i + 4)
        circ.h(30)
        circ.h(31)

        for i in range(1, 32, 2):
            circ.cx(i + 32, i)
        for i in range(0, 30, 6):
            circ.cx(i, i + 32)
            circ.cx(i + 4, i + 36)
        circ.cx(30, 62)

        for i in range(1, 30, 2):
            circ.cx(i + 35, i)
        for i in range(4, 32, 4):
            circ.cx(i, i + 29)

        for i in range(0, 30, 2):
            circ.cx(i + 35, i)
        for i in range(1, 30, 6):
            circ.cx(i, i + 33)
            circ.cx(i + 2, i + 35)
        circ.cx(31, 64)

        for i in range(0, 32):
            circ.measure(i, i)
        result = backend.run(circ, shots=shots).result()
        counts = result.get_counts()
        self.assertSuccess(result)

        n_anc = 32
        totals = np.zeros(n_anc, dtype=int)
        for outcomes, num_counts in counts.items():
            new_totals = num_counts * np.array([int(bit) for bit in outcomes][::-1])
            assert len(new_totals) == n_anc
            totals += new_totals
        output = {}
        for i in range(0, 32):
            output[hex(i)] = totals[i]

        targets = {}
        for i in range(0, 30, 3):
            targets[hex(i)] = shots / 2
            targets[hex(i + 1)] = shots / 2
            targets[hex(i + 2)] = 0
        targets[hex(30)] = shots / 2
        targets[hex(31)] = shots / 2

        self.assertDictAlmostEqual(output, targets, delta=delta * shots)

    @supported_methods(["stabilizer"], [65, 127, 433])
    def test_measure_sampling_large_stabilizer(self, method, device, num_qubits):
        """Test sampling measure for large stabilizer circuit"""
        paulis = qi.PauliList([num_qubits * i for i in ["X", "Y", "Z"]])
        qc = QuantumCircuit(num_qubits)
        for pauli in paulis:
            for i, p in enumerate(pauli):
                if p == qi.Pauli("Y"):
                    qc.sdg(i)
                    qc.h(i)
                elif p == qi.Pauli("X"):
                    qc.h(i)
        qc.measure_all()
        backend = self.backend(method=method)
        result = backend.run(
            transpile(qc, backend, optimization_level=0), shots=10, seed_simulator=12345
        ).result()
        counts = result.get_counts()

        ref_counts = {}
        if num_qubits == 65:
            ref_counts.update(
                {
                    "00000010101011110100110101101101101100010011000110011100101001010": 1,
                    "00110001000010110100000001111110011100110111000000110000010011100": 1,
                    "10100010101000011000101011111100101010011101000011010111111101011": 1,
                    "10000011001110111110110001111001100001010111100000001000110011110": 1,
                    "00011100011000110011001001000011110000010101000111000101111110110": 1,
                    "00000100000101110111010011011110001101100111111100001001100101011": 1,
                    "10101011100110001001001100000101000001000010100101001000100010111": 1,
                    "10000110100111001010101010011100010011110001011001110010100010100": 1,
                    "00010011011100111010110000100101111101001011110001100110101101000": 1,
                    "10100100011111100000100111011100101110001100100000111111111111011": 1,
                }
            )
        elif num_qubits == 127:
            ref_counts.update(
                {
                    "0001010101111010011010110110110110001001100011001110010100101000110001000010110100000001111110011100110111000000110000010011100": 1,
                    "0010111011101001101111000110110011111110000100110010101100011100011000110011001001000011110000010101000111000101111110110100000": 1,
                    "0000010011101110010111000110010000011111111111101100010011011100111010110000100101111101001011110001100110101101000101010111001": 1,
                    "1100100000010000100001000010101101101100010110010100001001101101101111110011111001000111011011010010110111100000101001000111111": 1,
                    "1000100100110000010100000100001010010100100010001011110000110100111001010101010011100010011110001011001110010100010100000001000": 1,
                    "0000101010110000000011000000100011011100100101100001111101100100011011111111000001011100110111110100101001110000010110101010101": 1,
                    "0001010110001001011010110001101100100001101100011010011110011101111110001110101110111000000011001000100011001111100000111110100": 1,
                    "1011101111011100010101110011011111010110011011001001011000111101101100111001000010110010011001100010010111101011110110011011001": 1,
                    "1100111011111011000111100110000101011110000000100011001111010100010101000011000101011111100101010011101000011010111111101011000": 1,
                    "0000000000000001101000100001111100000010101001010011001011001101101000110111100110111101111110111100111101000011110110011000110": 1,
                }
            )
        elif num_qubits == 433:
            ref_counts.update(
                {
                    "1010011100010011110001011001110010100010100000001000001011101110100110111100011011001111111000010011001010110001110001100011001100100100001111000001010100011100010111111011010000011001110111110110001111001100001010111100000001000110011110101000101010000110001010111111001010100111010000110101111111010110000001010101111010011010110110110110001001100011001110010100101000110001000010110100000001111110011100110111000000110000010011100": 1,
                    "0000010001101110010010110000111110110010001101111111100000101110011011111010010100111000001011010101010111001000000100001000010000101011011011000101100101000010011011011011111100111110010001110110110100101101111000001010010001111110000010011101110010111000110010000011111111111101100010011011100111010110000100101111101001011110001100110101101000101010111001100010010011000001010000010000101001010010001000101111000011010011100101010": 1,
                    "1001010110100111010001000101010000000000111101001100100101011010110111111101010000111101011001001100001110101111101011010011101000111110011100010100111000010000100001100000010110100001111101010000100000100110011101100010000000011110010000100100000111001111010101011110010000011001000101001110100011010100110101111000010001000110001110010011111010101011011000101010110001001110111100001100010101001011000101101110010100001000111010011": 1,
                    "0011000011110000001010100100010101001111000000100101011000101000010010110100111100101000100100101111101010011110110110011010000010110111110111000000110111111101101000100100101101000001100100010011101011010011111001010100111110000111100111110110010110100101000110110111010001010000001011000100010000000000100101111010101110001001100110011010110000111111101101010011000011011010111110111000101110001001001101110001010111110010001000111": 1,
                    "0110010101110010011110110111100111111100011111010010001000010101011111101001010011110000011001100111010111000111011010000011011011100101111010111101000010010011111011000011000000111000010101001101101110100100001100010011000001110011011011111000110100100110010100011000110111101010111000101101100101111010101110010010000000111101110000101010101100011101110110000001101100111010110110011000111010101111100001111010000010111000010011000": 1,
                    "0100111000011011011111101100100000000000000011010001000011111000000101010010100110010110011011010001101111001101111011111101111001111010000111101100110001101011101111011100010101110011011111010110011011001001011000111101101100111001000010110010011001100010010111101011110110011011001000101011000100101101011000110110010000110110001101001111001110111111000111010111011100000001100100010001100111110000011111010000001010101100000000110": 1,
                    "1101001011100010000010110000111110110011011111000110111111101110010101101001101110011110011101101000101101110010101010111010101100001110111011101001101110010001101111000101110001000110110100010101010101110101100010001010110011010110011100011010010010100100010100111101101101010100101011111010111010110001011100000010000000101000000000010011000100100000100101100011011100110000101010011110101000101110101010101010101101110000010011011": 1,
                    "0111111110110101111001101110001111101111000110110010110000101111100101010100100010001010110111100110010111000010000101001110011011111010011111100000001010101101010000110110001000001000111101110101101100100100010100010011101001000010001101111101000010000101000010001001000101011011011011101010101111100110000111000100011010110000100011000000000010101011111110011111010010110101100100111001000000100011100000000111101111001101100011101": 1,
                    "1111000111000110110000010111101110110010110001110110010001110010011110110111111011010110111110011010010011010110010010010011001001001100000000100110010101110100011011100001010010011100001011000000010000010010110101001000010100111011000001011000101111110100011000100011010001101000111111110011110010100010011011100100100011010100110111010101100000101110110111100111101001111000101100110101011011010000111101111001011110101011011000011": 1,
                    "0100100100011111111001111110100110001100101000011110011010001110001101100111111010010011110010000110011001010001001110110000111011000010000000011111011100001001010001110110111001011100001110101000101101011011010110110011110101111010111101011010000110101010111100110100101101011110100000100000110011100010011001111000010101010000011111001010010100001001101100001010101000101001100000001001110011010011000001011101011110010000110000111": 1,
                }
            )

        for count in counts:
            self.assertIn(count, ref_counts)

    # ---------------------------------------------------------------------
    # Test MPS algorithms for measure
    # ---------------------------------------------------------------------
    def test_mps_measure_alg_qv(self):
        """Test MPS measure algorithms with quantum volume"""
        backend = self.backend(method="matrix_product_state")
        shots = 1000
        n = 5
        depth = 2
        circuit = QuantumVolume(n, depth, seed=9)
        circuit.measure_all()
        circuit = transpile(circuit, backend)

        result1 = backend.run(
            circuit, shots=shots, mps_sample_measure_algorithm="mps_apply_measure"
        ).result()
        self.assertTrue(getattr(result1, "success", "True"))

        result2 = backend.run(
            circuit, shots=shots, mps_sample_measure_algorithm="mps_probabilities"
        ).result()
        self.assertTrue(getattr(result2, "success", "True"))

        self.assertDictAlmostEqual(
            result1.get_counts(circuit), result2.get_counts(circuit), delta=0.1 * shots
        )

    def test_mps_measure_subset_alg_qv(self):
        """Test MPS measure algorithms with quantum volume"""
        backend = self.backend(method="matrix_product_state")
        shots = 1000
        n = 5
        circuits = []
        for i in range(2):
            circuit = QuantumCircuit(n, n)
            circuit.unitary(random_unitary(4), [0, 1])
            circuit.unitary(random_unitary(4), [1, 2])
            circuit.unitary(random_unitary(4), [2, 3])
            circuit.unitary(random_unitary(4), [3, 4])
            circuits.append(circuit)
        circuits[0].measure([0, 2, 4], [0, 2, 4])
        circuits[1].measure([4, 1], [4, 1])
        circuits = transpile(circuits, backend)

        for circuit in circuits:
            result1 = backend.run(
                circuit, shots=shots, mps_sample_measure_algorithm="mps_apply_measure"
            ).result()
            self.assertTrue(getattr(result1, "success", "True"))

            result2 = backend.run(
                circuit, shots=shots, mps_sample_measure_algorithm="mps_probabilities"
            ).result()
            self.assertTrue(getattr(result2, "success", "True"))

            self.assertDictAlmostEqual(
                result1.get_counts(circuit), result2.get_counts(circuit), delta=0.1 * shots
            )

            # Test also parallel version
            os.environ["PRL_PROB_MEAS"] = "1"
            result2_prl = backend.run(
                circuit, shots=shots, mps_sample_measure_algorithm="mps_probabilities"
            ).result()
            self.assertTrue(getattr(result2_prl, "success", "True"))
            del os.environ["PRL_PROB_MEAS"]  # Python 3.8 in Windows
            # os.unsetenv("PRL_PROB_MEAS")  # SInce Python 3.9

            self.assertDictAlmostEqual(
                result1.get_counts(circuit), result2_prl.get_counts(circuit), delta=0.1 * shots
            )
            self.assertDictAlmostEqual(
                result2.get_counts(circuit), result2_prl.get_counts(circuit), delta=0.1 * shots
            )

    def test_mps_measure_with_limited_bond_dimension(self):
        """Test MPS measure with limited bond dimension,
        where the qubits are not in sorted order
        """
        backend_statevector = self.backend(method="statevector")
        shots = 1000
        n = 4
        for bd in [2, 4]:
            backend_mps = self.backend(
                method="matrix_product_state", matrix_product_state_max_bond_dimension=bd
            )
            for measured_qubits in [
                [0, 1, 2, 3],
                [3, 2, 1, 0],
                [2, 0, 1, 3],
                [0, 1, 2],
                [2, 1, 3],
                [1, 3, 0],
                [0, 2, 3],
            ]:
                circuit = QuantumCircuit(n, n)
                circuit.h(3)
                circuit.h(1)
                circuit.cx(1, 2)
                circuit.cx(3, 0)
                circuit.measure(measured_qubits, measured_qubits)
                res_mps = backend_mps.run(circuit, shots=shots).result().get_counts()
                self.assertTrue(getattr(res_mps, "success", "True"))
                res_sv = backend_statevector.run(circuit, shots=shots).result().get_counts()
                self.assertDictAlmostEqual(res_mps, res_sv, delta=0.1 * shots)
