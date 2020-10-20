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
from qiskit.providers.aer.extensions.snapshot_density_matrix import SnapshotDensityMatrix

from ..common import QiskitAerTestCase


class TestSnapshotDensityMatrixExtension(QiskitAerTestCase):
    """SnapshotDensityMatrix extension tests"""

    @staticmethod
    def snapshot_circuit_instr(circ_qubits, label, qubits=None):
        """Return QobjInstruction for circuit monkey patch method."""
        circuit = QuantumCircuit(circ_qubits)
        circuit.snapshot_density_matrix(label, qubits)
        qobj = assemble(circuit)
        instr = qobj.experiments[0].instructions[0]
        return instr

    def test_snapshot_label_raises(self):
        """Test snapshot label must be str"""
        self.assertRaises(ExtensionError, lambda: SnapshotDensityMatrix(1, 1))

    def test_snapshot_name(self):
        """Test snapshot instruction has correct name"""
        instrs = [
            SnapshotDensityMatrix('snap', 1).assemble(),
            self.snapshot_circuit_instr(1, 'snap')
        ]
        for instr in instrs:
            self.assertTrue(hasattr(instr, 'name'))
            self.assertEqual(instr.name, 'snapshot')

    def test_snapshot_type(self):
        """Test snapshot instruction has correct type."""
        instrs = [
            SnapshotDensityMatrix('snap', 1).assemble(),
            self.snapshot_circuit_instr(1, 'snap')
        ]
        for instr in instrs:
            self.assertTrue(hasattr(instr, 'snapshot_type'))
            self.assertEqual(instr.snapshot_type, 'density_matrix')

    def test_snapshot_label(self):
        """Test snapshot instruction has correct label"""
        for label in ['snap0', 'snap1', 'snap2']:
            instrs = [
                SnapshotDensityMatrix(label, 1).assemble(),
                self.snapshot_circuit_instr(1, label)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'label'))
                self.assertEqual(instr.label, label)

    def test_snapshot_all_qubits(self):
        """Test snapshot instruction has correct qubits."""
        for j in range(1, 5):
            instrs = [
                SnapshotDensityMatrix('snap', j).assemble(),
                self.snapshot_circuit_instr(j, 'snap')
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'qubits'))
                self.assertEqual(instr.qubits, list(range(j)))

    def test_snapshot_specific_qubits(self):
        """Test snapshot instruction has correct qubits."""
        for qubits in [[0], [0, 2], [1, 3, 0]]:
            instr = self.snapshot_circuit_instr(5, 'snap', qubits)
            self.assertTrue(hasattr(instr, 'qubits'))
            self.assertEqual(instr.qubits, qubits)


if __name__ == '__main__':
    unittest.main()
