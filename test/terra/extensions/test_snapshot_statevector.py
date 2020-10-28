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
from qiskit.providers.aer.extensions.snapshot_statevector import SnapshotStatevector

from ..common import QiskitAerTestCase


class TestSnapshotStatevectorExtension(QiskitAerTestCase):
    """SnapshotStatevector extension tests"""

    @staticmethod
    def snapshot_circuit_instr(circ_qubits, label):
        """Return QobjInstruction for circuit monkey patch method."""
        circuit = QuantumCircuit(circ_qubits)
        circuit.snapshot_statevector(label)
        qobj = assemble(circuit)
        instr = qobj.experiments[0].instructions[0]
        return instr

    def test_snapshot_label_raises(self):
        """Test snapshot label must be str"""
        self.assertRaises(ExtensionError, lambda: SnapshotStatevector(1, 1))

    def test_snapshot_name(self):
        """Test snapshot instruction has correct name"""
        instrs = [
            SnapshotStatevector('snap', 1).assemble(),
            self.snapshot_circuit_instr(1, 'snap')
        ]
        for instr in instrs:
            self.assertTrue(hasattr(instr, 'name'))
            self.assertEqual(instr.name, 'snapshot')

    def test_snapshot_type(self):
        """Test snapshot instruction has correct type."""
        instrs = [
            SnapshotStatevector('snap', 1).assemble(),
            self.snapshot_circuit_instr(1, 'snap')
        ]
        for instr in instrs:
            self.assertTrue(hasattr(instr, 'snapshot_type'))
            self.assertEqual(instr.snapshot_type, 'statevector')

    def test_snapshot_label(self):
        """Test snapshot instruction has correct label"""
        for label in ['snap0', 'snap1', 'snap2']:
            instrs = [
                SnapshotStatevector(label, 1).assemble(),
                self.snapshot_circuit_instr(1, label)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'label'))
                self.assertEqual(instr.label, label)

    def test_snapshot_qubits(self):
        """Test snapshot instruction has correct qubits."""
        for j in range(1, 5):
            instrs = [
                SnapshotStatevector('snap', j).assemble(),
                self.snapshot_circuit_instr(j, 'snap')
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'qubits'))
                self.assertEqual(instr.qubits, list(range(j)))


if __name__ == '__main__':
    unittest.main()
