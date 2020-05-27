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

from test.terra.reference import ref_measure
from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import ReadoutError, depolarizing_error


class QasmMeasureTests:
    """QasmSimulator measure tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test measure
    # ---------------------------------------------------------------------
    def test_measure_deterministic_with_sampling(self):
        """Test QasmSimulator measure with deterministic counts with sampling"""
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(
            allow_sampling=True)
        target_counts = ref_measure.measure_counts_deterministic(shots)
        target_memory = ref_measure.measure_memory_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots, memory=True)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    def test_measure_deterministic_without_sampling(self):
        """Test QasmSimulator measure with deterministic counts without sampling"""
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(
            allow_sampling=False)
        target_counts = ref_measure.measure_counts_deterministic(shots)
        target_memory = ref_measure.measure_memory_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots, memory=True)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    def test_measure_nondeterministic_with_sampling(self):
        """Test QasmSimulator measure with non-deterministic counts with sampling"""
        shots = 4000
        circuits = ref_measure.measure_circuits_nondeterministic(
            allow_sampling=True)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        # Test sampling was enabled
        for res in result.results:
            self.assertIn("measure_sampling", res.metadata)
            self.assertEqual(res.metadata["measure_sampling"], True)

    def test_measure_nondeterministic_without_sampling(self):
        """Test QasmSimulator measure with nin-deterministic counts without sampling"""
        shots = 4000
        circuits = ref_measure.measure_circuits_nondeterministic(
            allow_sampling=False)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    def test_measure_sampling_with_readouterror(self):
        """Test QasmSimulator measure with deterministic counts with sampling and readout-error"""
        readout_error = [0.01, 0.1]
        noise_model = NoiseModel()
        readout = [[1.0 - readout_error[0], readout_error[0]],
                   [readout_error[1], 1.0 - readout_error[1]]]
        noise_model.add_all_qubit_readout_error(ReadoutError(readout))

        shots = 1000
        circuits = ref_measure.measure_circuits_deterministic(
            allow_sampling=True)
        targets = ref_measure.measure_counts_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, noise_model=noise_model,
            backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    def test_measure_sampling_with_quantum_noise(self):
        """Test QasmSimulator measure with deterministic counts with sampling and readout-error"""
        readout_error = [0.01, 0.1]
        noise_model = NoiseModel()
        depolarizing = {'u3': (1, 0.001), 'cx': (2, 0.02)}
        readout = [[1.0 - readout_error[0], readout_error[0]],
                   [readout_error[1], 1.0 - readout_error[1]]]
        noise_model.add_all_qubit_readout_error(ReadoutError(readout))
        for gate, (num_qubits, gate_error) in depolarizing.items():
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(gate_error, num_qubits), gate)

        shots = 1000
        circuits = ref_measure.measure_circuits_deterministic(
            allow_sampling=True)
        targets = ref_measure.measure_counts_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, noise_model=noise_model,
            backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        sampling = (self.BACKEND_OPTS.get("method", "automatic").startswith("density_matrix"))
        self.compare_result_metadata(result, circuits, "measure_sampling", sampling)


class QasmMultiQubitMeasureTests:
    """QasmSimulator measure tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test multi-qubit measure qobj instruction
    # ---------------------------------------------------------------------
    def test_measure_deterministic_multi_qubit_with_sampling(self):
        """Test QasmSimulator multi-qubit measure with deterministic counts with sampling"""
        shots = 100
        circuits = ref_measure.multiqubit_measure_circuits_deterministic(
            allow_sampling=True)
        target_counts = ref_measure.multiqubit_measure_counts_deterministic(shots)
        target_memory = ref_measure.multiqubit_measure_memory_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots, memory=True)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    def test_measure_deterministic_multi_qubit_without_sampling(self):
        """Test QasmSimulator multi-qubit measure with deterministic counts without sampling"""
        shots = 100
        circuits = ref_measure.multiqubit_measure_circuits_deterministic(
            allow_sampling=False)
        target_counts = ref_measure.multiqubit_measure_counts_deterministic(shots)
        target_memory = ref_measure.multiqubit_measure_memory_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots, memory=True)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.compare_counts(result, circuits, target_counts, delta=0)
        self.compare_memory(result, circuits, target_memory)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)

    def test_measure_nondeterministic_multi_qubit_with_sampling(self):
        """Test QasmSimulator measure with non-deterministic counts"""
        shots = 4000
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(
            allow_sampling=True)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", True)

    def test_measure_nondeterministic_multi_qubit_without_sampling(self):
        """Test QasmSimulator measure with non-deterministic counts"""
        shots = 4000
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(
            allow_sampling=False)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        self.compare_result_metadata(result, circuits, "measure_sampling", False)
