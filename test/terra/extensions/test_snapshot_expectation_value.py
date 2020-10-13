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


import numpy

from qiskit import QuantumCircuit, assemble
from qiskit.extensions.exceptions import ExtensionError
from qiskit.providers.aer.extensions.snapshot_expectation_value import SnapshotExpectationValue
from qiskit.quantum_info.operators import Pauli, Operator

from ..common import QiskitAerTestCase


class TestSnapshotExpectationValueExtension(QiskitAerTestCase):
    """SnapshotExpectationValue extension tests"""

    @staticmethod
    def snapshot_circuit_instr(circ_qubits, label, op, qubits, single_shot=False, variance=False):
        """Return QobjInstruction for circuit monkey patch method."""
        circuit = QuantumCircuit(circ_qubits)
        circuit.snapshot_expectation_value(label, op, qubits,
                                           single_shot=single_shot,
                                           variance=variance)
        qobj = assemble(circuit)
        instr = qobj.experiments[0].instructions[0]
        return instr

    def test_snapshot_label_raises(self):
        """Test snapshot label must be str"""
        self.assertRaises(ExtensionError, lambda: SnapshotExpectationValue(1, [[1, 'X']]))

    def test_snapshot_name(self):
        """Test snapshot instruction has correct name"""
        for op in [Pauli.from_label('X'), Operator([[0, 1], [1, 0]])]:
            instrs = [
                SnapshotExpectationValue('snap', op).assemble(),
                self.snapshot_circuit_instr(1, 'snap', op, [0])
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'name'))
                self.assertEqual(instr.name, 'snapshot')

    def test_snapshot_label(self):
        """Test snapshot instruction has correct label"""
        for op in [Pauli.from_label('X'), Operator([[0, 1], [1, 0]])]:
            for label in ['snap0', 'snap1']:
                instrs = [
                    SnapshotExpectationValue(label, op).assemble(),
                    self.snapshot_circuit_instr(1, label, op, [0])
                ]
                for instr in instrs:
                    self.assertTrue(hasattr(instr, 'label'))
                    self.assertEqual(instr.label, label)

    def test_snapshot_pauli_type(self):
        """Test snapshot instruction has correct type."""
        pauli_ops = [
            [[1, 'I'], [0.5, 'X'], [0.25, 'Y'], [-3, 'Z']],
            [[1j, 'I'], [0.5j, 'X'], [0.25j, 'Y'], [-3j, 'Z']],
            [[0.5j, Pauli.from_label('X')], [-0.5j, Pauli.from_label('Z')]]
        ]
        for op in pauli_ops:
            # standard
            instrs = [
                SnapshotExpectationValue('snap', op,
                                         single_shot=False,
                                         variance=False).assemble(),
                self.snapshot_circuit_instr(1, 'snap', op, [0],
                                            single_shot=False,
                                            variance=False)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'snapshot_type'))
                self.assertEqual(instr.snapshot_type, 'expectation_value_pauli')
            # Single shot
            instrs = [
                SnapshotExpectationValue('snap', op,
                                         single_shot=True,
                                         variance=False).assemble(),
                self.snapshot_circuit_instr(1, 'snap', op, [0],
                                            single_shot=True,
                                            variance=False)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'snapshot_type'))
                self.assertEqual(instr.snapshot_type, 'expectation_value_pauli_single_shot')
            # Variance
            instrs = [
                SnapshotExpectationValue('snap', op,
                                         single_shot=False,
                                         variance=True).assemble(),
                self.snapshot_circuit_instr(1, 'snap', op, [0],
                                            single_shot=False,
                                            variance=True)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'snapshot_type'))
                self.assertEqual(instr.snapshot_type, 'expectation_value_pauli_with_variance')

    def test_snapshot_matrix_type(self):
        """Test snapshot instruction has correct type."""
        matrix_ops = [
            numpy.eye(2),
            numpy.array([[0, 1j], [-1j, 0]]),
            Operator(Pauli.from_label('Z'))
        ]
        for op in matrix_ops:
            # standard
            instrs = [
                SnapshotExpectationValue('snap', op,
                                         single_shot=False,
                                         variance=False).assemble(),
                self.snapshot_circuit_instr(1, 'snap', op, [0],
                                            single_shot=False,
                                            variance=False)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'snapshot_type'))
                self.assertEqual(instr.snapshot_type, 'expectation_value_matrix')
            # Single shot
            instrs = [
                SnapshotExpectationValue('snap', op,
                                         single_shot=True,
                                         variance=False).assemble(),
                self.snapshot_circuit_instr(1, 'snap', op, [0],
                                            single_shot=True,
                                            variance=False)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'snapshot_type'))
                self.assertEqual(instr.snapshot_type, 'expectation_value_matrix_single_shot')
            # Variance
            instrs = [
                SnapshotExpectationValue('snap', op,
                                         single_shot=False,
                                         variance=True).assemble(),
                self.snapshot_circuit_instr(1, 'snap', op, [0],
                                            single_shot=False,
                                            variance=True)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'snapshot_type'))
                self.assertEqual(instr.snapshot_type, 'expectation_value_matrix_with_variance')

    def test_snapshot_specific_qubits(self):
        """Test snapshot instruction has correct qubits."""
        for qubits in [[0], [0, 2], [1, 3, 0]]:
            pauli = Pauli.from_label(len(qubits) * 'X')
            instrs = [
                self.snapshot_circuit_instr(5, 'snap', pauli, qubits),
                self.snapshot_circuit_instr(5, 'snap', Operator(pauli), qubits)
            ]
            for instr in instrs:
                self.assertTrue(hasattr(instr, 'qubits'))
                self.assertEqual(instr.qubits, qubits)


if __name__ == '__main__':
    unittest.main()
