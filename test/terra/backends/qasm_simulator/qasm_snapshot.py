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


class QasmSnapshotStatevectorTests:
    """QasmSimulator snapshot statevector tests."""

    SIMULATOR = QasmSimulator()
    SUPPORTED_QASM_METHODS = [
        'automatic', 'statevector', 'matrix_product_state'
    ]
    BACKEND_OPTS = {}

    def statevector_snapshots(self, data, label):
        """Format snapshots as list of Numpy arrays"""
        # Check snapshot entry exists in data
        self.assertIn("snapshots", data)
        self.assertIn("statevector", data["snapshots"])
        self.assertIn(label, data["snapshots"]["statevector"])
        # Format output as list of numpy array
        snaps = data["snapshots"]["statevector"][label]
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
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStatevectorTests.SUPPORTED_QASM_METHODS:
            self.assertRaises(AerError, job.result)
        else:
            result = job.result()
            self.is_completed(result)
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
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStatevectorTests.SUPPORTED_QASM_METHODS:
            self.assertRaises(AerError, job.result)
        else:
            result = job.result()
            self.is_completed(result)
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
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStatevectorTests.SUPPORTED_QASM_METHODS:
            self.assertRaises(AerError, job.result)
        else:
            result = job.result()
            self.is_completed(result)
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
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStatevectorTests.SUPPORTED_QASM_METHODS:
            self.assertRaises(AerError, job.result)
        else:
            result = job.result()
            self.is_completed(result)
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

    def stabilizer_snapshots(self, data, label):
        """Format snapshots as list of Numpy arrays"""
        # Check snapshot entry exists in data
        self.assertIn("snapshots", data)
        self.assertIn("stabilizer", data["snapshots"])
        self.assertIn(label, data["snapshots"]["stabilizer"])
        # Format output as list of numpy array
        return data["snapshots"]["stabilizer"][label]

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
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStabilizerTests.SUPPORTED_QASM_METHODS:
            self.assertRaises(AerError, job.result)
        else:
            result = job.result()
            self.is_completed(result)
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
        shots = 1000
        label = "snap"
        counts_targets = snapshot_state_counts_nondeterministic(shots)
        statevec_targets = snapshot_state_pre_measure_statevector_nondeterministic(
        )
        circuits = snapshot_state_circuits_nondeterministic(label,
                                                            'stabilizer',
                                                            post_measure=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStabilizerTests.SUPPORTED_QASM_METHODS:
            self.assertRaises(AerError, job.result)
        else:
            result = job.result()
            self.is_completed(result)
            self.compare_counts(result,
                                circuits,
                                counts_targets,
                                delta=0.1 * shots)
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
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStabilizerTests.SUPPORTED_QASM_METHODS:
            self.assertRaises(AerError, job.result)
        else:
            result = job.result()
            self.is_completed(result)
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
        shots = 1000
        label = "snap"
        counts_targets = snapshot_state_counts_nondeterministic(shots)
        statevec_targets = snapshot_state_post_measure_statevector_nondeterministic(
        )
        circuits = snapshot_state_circuits_nondeterministic(label,
                                                            'stabilizer',
                                                            post_measure=True)

        qobj = assemble(circuits, self.SIMULATOR, memory=True, shots=shots)
        job = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS)
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method not in QasmSnapshotStabilizerTests.SUPPORTED_QASM_METHODS:
            self.assertRaises(AerError, job.result)
        else:
            result = job.result()
            self.is_completed(result)
            self.compare_counts(result,
                                circuits,
                                counts_targets,
                                delta=0.1 * shots)
            # Check snapshots
            for i, circuit in enumerate(circuits):
                data = result.data(circuit)
                snaps = self.stabilizer_snapshots(data, label)
                for j, mem in enumerate(data['memory']):
                    statevec = statevec_targets[i].get(mem)
                    stabilizer = snaps[j]
                    self.assertTrue(
                        self.stabilizes_statevector(stabilizer, statevec))
