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


import unittest

from qiskit import QuantumCircuit, assemble
from qiskit.extensions.exceptions import ExtensionError
from qiskit.providers.aer.extensions.snapshot_probabilities import SnapshotProbabilities

from ..common import QiskitAerTestCase


class TestSnapshotProbabilitiesExtension(QiskitAerTestCase):
    """SnapshotProbabilities extension tests"""

    @staticmethod
    def snapshot_circuit_instr(circ_qubits, label, qubits, variance=False):
        """Return QobjInstruction for circuit monkey patch method."""
        circuit = QuantumCircuit(circ_qubits)
        circuit.snapshot_probabilities(label, qubits, variance)
        qobj = assemble(circuit)
        instr = qobj.experiments[0].instructions[0]
        return instr

    def test_snapshot_label_raises(self):
        """Test snapshot label must be str"""
        self.assertRaises(ExtensionError, lambda: SnapshotProbabilities(1, 1))

    def test_snapshot_name(self):
        """Test snapshot instruction has correct name"""
        instrs = [
            SnapshotProbabilities('snap', 1, False).assemble(),
            SnapshotProbabilities('snap', 1, True).assemble(),
            self.snapshot_circuit_instr(1, 'snap', [0], False),
            self.snapshot_circuit_instr(1, 'snap', [0], True)
        ]
        for instr in instrs:
            self.assertTrue(hasattr(instr, 'name'))
            self.assertEqual(instr.name, 'snapshot')

    def test_snapshot_type(self):
        """Test snapshot instruction has correct type."""
        # without variance
        instrs = [
            SnapshotProbabilities('snap', 1, False).assemble(),
            self.snapshot_circuit_instr(1, 'snap', [0], False)
        ]
        for instr in instrs:
            self.assertTrue(hasattr(instr, 'snapshot_type'))
            self.assertEqual(instr.snapshot_type, 'probabilities')
        # with variance
        instrs = [
            SnapshotProbabilities('snap', 1, True).assemble(),
            self.snapshot_circuit_instr(1, 'snap', [0], True)
        ]
        for instr in instrs:
            self.assertTrue(hasattr(instr, 'snapshot_type'))
            self.assertEqual(instr.snapshot_type, 'probabilities_with_variance')

    def test_snapshot_label(self):
        """Test snapshot instruction has correct label"""
        for label in ['snap0', 'snap1']:
            instrs = [
                SnapshotProbabilities(label, 1, False).assemble(),
                SnapshotProbabilities(label, 1, True).assemble(),
                self.snapshot_circuit_instr(1, label, [0], False),
                self.snapshot_circuit_instr(1, label, [0], True)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'label'))
                self.assertEqual(instr.label, label)

    def test_snapshot_all_qubits(self):
        """Test snapshot instruction has correct qubits."""
        for j in range(1, 5):
            instrs = [
                SnapshotProbabilities('snap', j, False).assemble(),
                SnapshotProbabilities('snap', j, True).assemble(),
                self.snapshot_circuit_instr(j, 'snap', range(j), True),
                self.snapshot_circuit_instr(j, 'snap', range(j), False)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'qubits'))
                self.assertEqual(instr.qubits, list(range(j)))

    def test_snapshot_specific_qubits(self):
        """Test snapshot instruction has correct qubits."""
        for qubits in [[0], [0, 2], [1, 3, 0]]:
            instrs = [
                self.snapshot_circuit_instr(5, 'snap', qubits, False),
                self.snapshot_circuit_instr(5, 'snap', qubits, True)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'qubits'))
                self.assertEqual(instr.qubits, qubits)


if __name__ == '__main__':
    unittest.main()
