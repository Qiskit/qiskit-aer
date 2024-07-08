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
    def test_measure_sampling_large_ghz_stabilizer(self, method, device, num_qubits):
        """Test sampling measure for large stabilizer circuit"""
        shots = 1000
        delta = 0.05
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        for q in range(1, num_qubits):
            qc.cx(q - 1, q)
        qc.measure_all()
        backend = self.backend(method=method)
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts()
        targets = {}
        targets["0" * num_qubits] = shots / 2
        targets["1" * num_qubits] = shots / 2
        self.assertDictAlmostEqual(counts, targets, delta=delta * shots)

    @supported_methods(["stabilizer"])
    def test_measure_stablizer_deterministic(self, method, device):
        """Test stabilizer measure for deterministic case"""
        backend = self.backend(method=method, device=device)
        shots = 10000
        qc = QuantumCircuit(5, 1)
        qc.h([2, 4])
        qc.cx(2, 0)
        qc.s(0)
        qc.cx(4, 2)
        qc.h(0)
        qc.cx(2, 3)
        qc.s(4)
        qc.cx(1, 0)
        qc.h([3, 4])
        qc.cx(3, 2)
        qc.h(3)
        qc.cx(0, 3)
        qc.cx(3, 1)
        qc.s(0)
        qc.s(1)
        qc.h(0)
        qc.s(0)
        qc.cx(4, 0)
        qc.cx(0, 1)
        qc.measure(1, 0)
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts()
        self.assertSuccess(result)

        self.assertDictEqual(counts, {"1": shots})

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
