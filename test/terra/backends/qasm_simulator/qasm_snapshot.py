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
QasmSimulator Integration Tests for Snapshot instructions
"""

import logging
import numpy as np

from qiskit.compiler import assemble
from qiskit.quantum_info.operators import Pauli
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import AerError

from test.terra.reference.ref_snapshot_state import (
    snapshot_state_circuits_deterministic, snapshot_state_counts_deterministic,
    snapshot_state_pre_measure_statevector_deterministic,
    snapshot_state_post_measure_statevector_deterministic,
    snapshot_state_circuits_nondeterministic,
    snapshot_state_counts_nondeterministic,
    snapshot_state_pre_measure_statevector_nondeterministic,
    snapshot_state_post_measure_statevector_nondeterministic)
from test.terra.reference.ref_snapshot_probabilities import (
    snapshot_probabilities_circuits,
    snapshot_probabilities_counts,
    snapshot_probabilities_labels_qubits,
    snapshot_probabilities_post_meas_probs,
    snapshot_probabilities_pre_meas_probs)
from test.terra.reference.ref_snapshot_expval import (
    snapshot_expval_circuits,
    snapshot_expval_counts,
    snapshot_expval_labels,
    snapshot_expval_post_meas_values,
    snapshot_expval_pre_meas_values)


class QasmSnapshotStatevectorTests:
    """QasmSimulator snapshot statevector tests."""

    SIMULATOR = QasmSimulator()
    SUPPORTED_QASM_METHODS = [
        'automatic', 'statevector', 'matrix_product_state'
    ]
    BACKEND_OPTS = {}

    @staticmethod
    def statevector_snapshots(data, label):
        """Format snapshots as list of Numpy arrays"""
        snaps = data.get("snapshots", {}).get("statevector", {}).get(label, [])
        statevecs = []
        for snap in snaps:
            svec = np.array(snap)
            statevecs.append(svec[:, 0] + 1j * svec[:, 1])
        return statevecs

    def test_snapshot_statevector_pre_measure_det(self):
        """Test snapshot statevector before deterministic final measurement"""
        shots = 10
        label = "snap"
        counts_targets = snapshot_state_counts_deterministic(shots)
        statevec_targets = snapshot_state_pre_measure_statevector_deterministic(
        )
        circuits = snapshot_state_circuits_deterministic(label,
                                                         'statevector',
                                                         post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj,
                                 backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStatevectorTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.statevector_snapshots(data, label)
                self.assertTrue(len(snaps), 1)
                target = statevec_targets[j]
                value = snaps[0]
                self.assertTrue(np.allclose(value, target))

    def test_snapshot_statevector_pre_measure_nondet(self):
        """Test snapshot statevector before non-deterministic final measurement"""
        shots = 100
        label = "snap"
        counts_targets = snapshot_state_counts_nondeterministic(shots)
        statevec_targets = snapshot_state_pre_measure_statevector_nondeterministic(
        )
        circuits = snapshot_state_circuits_nondeterministic(label,
                                                            'statevector',
                                                            post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj,
                                 backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStatevectorTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result,
                                circuits,
                                counts_targets,
                                delta=0.2 * shots)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.statevector_snapshots(data, label)
                self.assertTrue(len(snaps), 1)
                target = statevec_targets[j]
                value = snaps[0]
                self.assertTrue(np.allclose(value, target))

    def test_snapshot_statevector_post_measure_det(self):
        """Test snapshot statevector after deterministic final measurement"""
        shots = 10
        label = "snap"
        counts_targets = snapshot_state_counts_deterministic(shots)
        statevec_targets = snapshot_state_post_measure_statevector_deterministic(
        )
        circuits = snapshot_state_circuits_deterministic(label,
                                                         'statevector',
                                                         post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, memory=True, shots=shots)
        job = self.SIMULATOR.run(qobj,
                                 backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStatevectorTests.SUPPORTED_QASM_METHODS:
            logging.getLogger().setLevel(logging.CRITICAL)
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0)
            # Check snapshots
            for i, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.statevector_snapshots(data, label)
                for j, mem in enumerate(data['memory']):
                    target = statevec_targets[i].get(mem)
                    self.assertTrue(np.allclose(snaps[j], target))

    def test_snapshot_statevector_post_measure_nondet(self):
        """Test snapshot statevector after non-deterministic final measurement"""
        shots = 100
        label = "snap"
        counts_targets = snapshot_state_counts_nondeterministic(shots)
        statevec_targets = snapshot_state_post_measure_statevector_nondeterministic(
        )
        circuits = snapshot_state_circuits_nondeterministic(label,
                                                            'statevector',
                                                            post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, memory=True, shots=shots)
        job = self.SIMULATOR.run(qobj,
                                 backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStatevectorTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result,
                                circuits,
                                counts_targets,
                                delta=0.2 * shots)
            # Check snapshots
            for i, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.statevector_snapshots(data, label)
                for j, mem in enumerate(data['memory']):
                    target = statevec_targets[i].get(mem)
                    self.assertTrue(np.allclose(snaps[j], target))


class QasmSnapshotStabilizerTests:
    """QasmSimulator method snapshot stabilizer tests."""

    SIMULATOR = QasmSimulator()
    SUPPORTED_QASM_METHODS = ['automatic', 'stabilizer']
    BACKEND_OPTS = {}

    @staticmethod
    def stabilizer_snapshots(data, label):
        """Get stabilizer snapshots"""
        return data.get("snapshots", {}).get("stabilizer", {}).get(label, [])

    @staticmethod
    def stabilizes_statevector(stabilizer, statevector):
        """Return True if two stabilizer states are equal."""
        # Get stabilizer and destabilizers and convert to sets
        for stab in stabilizer:
            if stab[0] == '-':
                pauli_mat = -1 * Pauli.from_label(stab[1:]).to_matrix()
            else:
                pauli_mat = Pauli.from_label(stab).to_matrix()
            val = statevector.conj().dot(pauli_mat.dot(statevector))
            if not np.isclose(val, 1):
                return False
        return True

    def test_snapshot_stabilizer_pre_measure_det(self):
        """Test snapshot stabilizer before deterministic final measurement"""
        shots = 10
        label = "snap"
        counts_targets = snapshot_state_counts_deterministic(shots)
        statevec_targets = snapshot_state_pre_measure_statevector_deterministic(
        )
        circuits = snapshot_state_circuits_deterministic(label,
                                                         'stabilizer',
                                                         post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj,
                                 backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStabilizerTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.stabilizer_snapshots(data, label)
                self.assertEqual(len(snaps), 1)
                statevec = statevec_targets[j]
                stabilizer = snaps[0]
                self.assertTrue(
                    self.stabilizes_statevector(stabilizer, statevec))

    def test_snapshot_stabilizer_pre_measure_nondet(self):
        """Test snapshot stabilizer before non-deterministic final measurement"""
        shots = 100
        label = "snap"
        counts_targets = snapshot_state_counts_nondeterministic(shots)
        statevec_targets = snapshot_state_pre_measure_statevector_nondeterministic(
        )
        circuits = snapshot_state_circuits_nondeterministic(label,
                                                            'stabilizer',
                                                            post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj,
                                 backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStabilizerTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result,
                                circuits,
                                counts_targets,
                                delta=0.2 * shots)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.stabilizer_snapshots(data, label)
                self.assertEqual(len(snaps), 1)
                statevec = statevec_targets[j]
                stabilizer = snaps[0]
                self.assertTrue(
                    self.stabilizes_statevector(stabilizer, statevec))

    def test_snapshot_stabilizer_post_measure_det(self):
        """Test snapshot stabilizer after deterministic final measurement"""
        shots = 10
        label = "snap"
        counts_targets = snapshot_state_counts_deterministic(shots)
        statevec_targets = snapshot_state_post_measure_statevector_deterministic(
        )
        circuits = snapshot_state_circuits_deterministic(label,
                                                         'stabilizer',
                                                         post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, memory=True, shots=shots)
        job = self.SIMULATOR.run(qobj,
                                 backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStabilizerTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0)
            # Check snapshots
            for i, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.stabilizer_snapshots(data, label)
                for j, mem in enumerate(data['memory']):
                    statevec = statevec_targets[i].get(mem)
                    stabilizer = snaps[j]
                    self.assertTrue(
                        self.stabilizes_statevector(stabilizer, statevec))

    def test_snapshot_stabilizer_post_measure_nondet(self):
        """Test snapshot stabilizer after non-deterministic final measurement"""
        shots = 100
        label = "snap"
        counts_targets = snapshot_state_counts_nondeterministic(shots)
        statevec_targets = snapshot_state_post_measure_statevector_nondeterministic(
        )
        circuits = snapshot_state_circuits_nondeterministic(label,
                                                            'stabilizer',
                                                            post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, memory=True, shots=shots)
        job = self.SIMULATOR.run(qobj,
                                 backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStabilizerTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result,
                                circuits,
                                counts_targets,
                                delta=0.2 * shots)
            # Check snapshots
            for i, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.stabilizer_snapshots(data, label)
                for j, mem in enumerate(data['memory']):
                    statevec = statevec_targets[i].get(mem)
                    stabilizer = snaps[j]
                    self.assertTrue(
                        self.stabilizes_statevector(stabilizer, statevec))


class QasmSnapshotDensityMatrixTests:
    """QasmSimulator snapshot density matrix tests."""

    SIMULATOR = QasmSimulator()
    SUPPORTED_QASM_METHODS = [
        'automatic', 'density_matrix'
    ]
    BACKEND_OPTS = {}

    @staticmethod
    def density_snapshots(data, label):
        """Format snapshots as list of Numpy arrays"""
        # Check snapshot entry exists in data
        snaps = data.get("snapshots", {}).get("density_matrix", {}).get(label, [])
        # Convert nested lists to numpy arrays
        output = {}
        for snap_dict in snaps:
            memory = snap_dict['memory']
            mat = np.array(snap_dict['value'])
            output[memory] = mat[:, :, 0] + 1j * mat[:, :, 1]
        return output

    def test_snapshot_density_matrix_pre_measure_det(self):
        """Test snapshot density matrix before deterministic final measurement"""
        shots = 10
        label = "snap"
        counts_targets = snapshot_state_counts_deterministic(shots)
        statevec_targets = snapshot_state_pre_measure_statevector_deterministic(
        )
        circuits = snapshot_state_circuits_deterministic(label,
                                                         'density_matrix',
                                                         post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotDensityMatrixTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.density_snapshots(data, label)
                self.assertTrue(len(snaps), 1)
                target = np.outer(statevec_targets[j], statevec_targets[j].conj())
                # Pre-measurement all memory bits should be 0
                value = snaps.get('0x0')
                self.assertTrue(np.allclose(value, target))

    def test_snapshot_density_matrix_pre_measure_nondet(self):
        """Test snapshot density matrix before non-deterministic final measurement"""
        shots = 100
        label = "snap"
        counts_targets = snapshot_state_counts_nondeterministic(shots)
        statevec_targets = snapshot_state_pre_measure_statevector_nondeterministic(
        )
        circuits = snapshot_state_circuits_nondeterministic(label,
                                                            'density_matrix',
                                                            post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotDensityMatrixTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result,
                                circuits,
                                counts_targets,
                                delta=0.2 * shots)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.density_snapshots(data, label)
                self.assertTrue(len(snaps), 1)
                target = np.outer(statevec_targets[j], statevec_targets[j].conj())
                value = snaps.get('0x0')
                self.assertTrue(np.allclose(value, target))

    def test_snapshot_density_matrix_post_measure_det(self):
        """Test snapshot density matrix after deterministic final measurement"""
        shots = 10
        label = "snap"
        counts_targets = snapshot_state_counts_deterministic(shots)
        statevec_targets = snapshot_state_post_measure_statevector_deterministic(
        )
        circuits = snapshot_state_circuits_deterministic(label,
                                                         'density_matrix',
                                                         post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, memory=True, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotDensityMatrixTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0)
            # Check snapshots
            for i, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.density_snapshots(data, label)
                for j, mem in enumerate(data['memory']):
                    target = statevec_targets[i].get(mem)
                    target = np.outer(target, target.conj())
                    value = snaps.get(mem)
                    self.assertTrue(np.allclose(value, target))

    def test_snapshot_density_matrix_post_measure_nondet(self):
        """Test snapshot density matrix after non-deterministic final measurement"""
        shots = 100
        label = "snap"
        counts_targets = snapshot_state_counts_nondeterministic(shots)
        statevec_targets = snapshot_state_post_measure_statevector_nondeterministic(
        )
        circuits = snapshot_state_circuits_nondeterministic(label,
                                                            'density_matrix',
                                                            post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, memory=True, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotDensityMatrixTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result,
                                circuits,
                                counts_targets,
                                delta=0.2 * shots)
            # Check snapshots
            for i, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.density_snapshots(data, label)
                for j, mem in enumerate(data['memory']):
                    target = statevec_targets[i].get(mem)
                    target = np.outer(target, target.conj())
                    value = snaps.get(mem)
                    self.assertTrue(np.allclose(value, target))


class QasmSnapshotProbabilitiesTests:
    """QasmSimulator snapshot probabilities tests."""

    SIMULATOR = QasmSimulator()
    SUPPORTED_QASM_METHODS = [
        'automatic', 'statevector', 'stabilizer', 'density_matrix', 'matrix_product_state'
    ]
    BACKEND_OPTS = {}

    @staticmethod
    def probability_snapshots(data, labels):
        """Format snapshots as nested dicts"""
        # Check snapshot entry exists in data
        output = {}
        for label in labels:
            snaps = data.get("snapshots", {}).get("probabilities", {}).get(label, [])
            output[label] = {snap_dict['memory']: snap_dict['value']
                             for snap_dict in snaps}
        return output

    def test_snapshot_probabilities_pre_measure(self):
        """Test snapshot probabilities before final measurement"""
        shots = 1000
        labels = list(snapshot_probabilities_labels_qubits().keys())
        counts_targets = snapshot_probabilities_counts(shots)
        prob_targets = snapshot_probabilities_pre_meas_probs()

        circuits = snapshot_probabilities_circuits(post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotProbabilitiesTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0.1 * shots)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                all_snapshots = self.probability_snapshots(data, labels)
                for label in labels:
                    snaps = all_snapshots.get(label, {})
                    self.assertTrue(len(snaps), 1)
                    for memory, value in snaps.items():
                        target = prob_targets[j].get(label, {}).get(memory, {})
                        self.assertDictAlmostEqual(value, target, delta=1e-7)

    def test_snapshot_probabilities_post_measure(self):
        """Test snapshot probabilities after final measurement"""
        shots = 1000
        labels = list(snapshot_probabilities_labels_qubits().keys())
        counts_targets = snapshot_probabilities_counts(shots)
        prob_targets = snapshot_probabilities_post_meas_probs()

        circuits = snapshot_probabilities_circuits(post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotProbabilitiesTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0.1 * shots)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                all_snapshots = self.probability_snapshots(data, labels)
                for label in labels:
                    snaps = all_snapshots.get(label, {})
                    for memory, value in snaps.items():
                        target = prob_targets[j].get(label, {}).get(memory, {})
                        self.assertDictAlmostEqual(value, target, delta=1e-7)


class QasmSnapshotExpValPauliTests:
    """QasmSimulator snapshot pauli expectation value tests."""

    SIMULATOR = QasmSimulator()
    SUPPORTED_QASM_METHODS = [
        'automatic', 'statevector', 'stabilizer', 'matrix_product_state'
    ]
    BACKEND_OPTS = {}

    @staticmethod
    def expval_snapshots(data, labels):
        """Format snapshots as nested dicts"""
        # Check snapshot entry exists in data
        output = {}
        for label in labels:
            snaps = data.get("snapshots", {}).get("expectation_value", {}).get(label, [])
            # Convert list into dict
            inner = {}
            for snap_dict in snaps:
                val = snap_dict['value']
                inner[snap_dict['memory']] = val[0] + 1j * val[1]
            output[label] = inner
        return output

    def test_snapshot_expval_pauli_pre_measure(self):
        """Test snapshot expectation value (pauli) before final measurement"""
        shots = 1000
        labels = snapshot_expval_labels()
        counts_targets = snapshot_expval_counts(shots)
        value_targets = snapshot_expval_pre_meas_values()

        circuits = snapshot_expval_circuits(pauli=True, post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotExpValPauliTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0.1 * shots)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                all_snapshots = self.expval_snapshots(data, labels)
                for label in labels:
                    snaps = all_snapshots.get(label, {})
                    self.assertTrue(len(snaps), 1)
                    for memory, value in snaps.items():
                        target = value_targets[j].get(label, {}).get(memory, {})
                        self.assertAlmostEqual(value, target, delta=1e-7)

    def test_snapshot_expval_pauli_post_measure(self):
        """Test snapshot expectation value (pauli) after final measurement"""
        shots = 1000
        labels = snapshot_expval_labels()
        counts_targets = snapshot_expval_counts(shots)
        value_targets = snapshot_expval_post_meas_values()

        circuits = snapshot_expval_circuits(pauli=True, post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotExpValPauliTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0.1 * shots)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                all_snapshots = self.expval_snapshots(data, labels)
                for label in labels:
                    snaps = all_snapshots.get(label, {})
                    self.assertTrue(len(snaps), 1)
                    for memory, value in snaps.items():
                        target = value_targets[j].get(label, {}).get(memory, {})
                        self.assertAlmostEqual(value, target, delta=1e-7)

class QasmSnapshotExpValMatrixTests:
    """QasmSimulator snapshot pauli expectation value tests."""

    SIMULATOR = QasmSimulator()
    SUPPORTED_QASM_METHODS = [
        'automatic', 'statevector', 'matrix_product_state'
    ]
    BACKEND_OPTS = {}

    @staticmethod
    def expval_snapshots(data, labels):
        """Format snapshots as nested dicts"""
        # Check snapshot entry exists in data
        output = {}
        for label in labels:
            snaps = data.get("snapshots", {}).get("expectation_value", {}).get(label, [])
            # Convert list into dict
            inner = {}
            for snap_dict in snaps:
                inner[snap_dict['memory']] = snap_dict['value']
            output[label] = inner
        return output

    def test_snapshot_expval_matrix_pre_measure(self):
        """Test snapshot expectation value (matrix) before final measurement"""
        shots = 1000
        labels = snapshot_expval_labels()
        counts_targets = snapshot_expval_counts(shots)
        value_targets = snapshot_expval_pre_meas_values()

        circuits = snapshot_expval_circuits(pauli=False, post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotExpValMatrixTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0.1 * shots)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                all_snapshots = self.expval_snapshots(data, labels)
                for label in labels:
                    snaps = all_snapshots.get(label, {})
                    self.assertTrue(len(snaps), 1)
                    for memory, value in snaps.items():
                        target = value_targets[j].get(label, {}).get(memory, {})
                        self.assertAlmostEqual(value, target, delta=1e-7)

    def test_snapshot_expval_matrix_post_measure(self):
        """Test snapshot expectation value (matrix) after final measurement"""
        shots = 1000
        labels = snapshot_expval_labels()
        counts_targets = snapshot_expval_counts(shots)
        value_targets = snapshot_expval_post_meas_values()

        circuits = snapshot_expval_circuits(pauli=False, post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        success = getattr(result, 'success', False)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotExpValMatrixTests.SUPPORTED_QASM_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            self.compare_counts(result, circuits, counts_targets, delta=0.1 * shots)
            # Check snapshots
            for j, circuit in enumerate(circuits):
                data = result.data(circuit)
                all_snapshots = self.expval_snapshots(data, labels)
                for label in labels:
                    snaps = all_snapshots.get(label, {})
                    self.assertTrue(len(snaps), 1)
                    for memory, value in snaps.items():
                        target = value_targets[j].get(label, {}).get(memory, {})
                        self.assertAlmostEqual(value, target, delta=1e-7)
