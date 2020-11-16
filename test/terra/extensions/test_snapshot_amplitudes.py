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
from qiskit.providers.aer.extensions.snapshot_amplitudes import SnapshotAmplitudes

from ..common import QiskitAerTestCase


class TestSnapshotAmplitudesExtension(QiskitAerTestCase):
    """SnapshotAmplitudes extension tests"""

    @staticmethod
    def snapshot_circuit_instr(circ_qubits, label, params, qubits):
        """Return QobjInstruction for circuit monkey patch method."""
        circuit = QuantumCircuit(circ_qubits)
        circuit.snapshot_amplitudes(label, params, list(range(qubits)))
        qobj = assemble(circuit)
        instr = qobj.experiments[0].instructions[0]
        return instr

    def test_snapshot_label_raises(self):
        """Test snapshot label must be str"""
        self.assertRaises(ExtensionError, lambda: SnapshotAmplitudes(1, 1))

    def test_snapshot_name(self):
        """Test snapshot instruction has correct name"""
        qubits = 2
        instrs = [
            SnapshotAmplitudes('snap', [0], 1).assemble(),
            self.snapshot_circuit_instr(qubits, 'snap', [0], qubits)
        ]
        for instr in instrs:
            self.assertTrue(hasattr(instr, 'name'))
            self.assertEqual(instr.name, 'snapshot')

    def test_snapshot_type(self):
        """Test snapshot instruction has correct type."""
        qubits = 2
        instrs = [
            SnapshotAmplitudes('snap', [0], 1).assemble(),
            self.snapshot_circuit_instr(qubits, 'snap', [0], qubits)
        ]
        for instr in instrs:
            self.assertTrue(hasattr(instr, 'snapshot_type'))
            self.assertEqual(instr.snapshot_type, 'amplitudes')


    def test_snapshot_label(self):
        """Test snapshot instruction has correct label"""
        for label in ['snap0', 'snap1']:
            instrs = [
                SnapshotAmplitudes(label,[0], 2).assemble(),
                self.snapshot_circuit_instr(2, label, [0], 2)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'label'))
                self.assertEqual(instr.label, label)

    def test_snapshot_all_qubits(self):
        """Test snapshot instruction has correct qubits."""
        for j in range(1, 5):
            instrs = [
                SnapshotAmplitudes('snap', list(range(j)), j).assemble(),
                self.snapshot_circuit_instr(j, 'snap', range(j), j)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'qubits'))
                self.assertEqual(instr.qubits, list(range(j)))


if __name__ == '__main__':
    unittest.main()
