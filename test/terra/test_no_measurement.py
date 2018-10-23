# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the final statevector in circuits whose simulation is deterministic,
    i.e., contain no measurement or noise"""

import test.terra.common as common
import test.terra.qobj_hacks as qobj_hacks
import unittest
import numpy as np

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit, compile)
from qiskit_aer.backends import QasmSimulator


class NoMeasurementTest(common.QiskitAerTestCase):
    """Test the final statevector in circuits whose simulation is deterministic,
    i.e., contain no measurement or noise"""

    def setUp(self):
        # ***
        self.backend = QasmSimulator()

    def insert_state_snapshots_before_barrier(self, qobj):
        """Insert state snapshots before each full barrier in a qobj.

        The snapshot labels are integer strings "0", "1", up to the number of barriers.
        """
        for exp_index in range(len(qobj.experiments)):
            num_qubits = qobj.experiments[exp_index].config.n_qubits
            barrier = qobj_hacks.qobj_barrier(num_qubits)
            positions = qobj_hacks.get_item_positions(qobj, exp_index, barrier)
            for label, pos in reversed(list(enumerate(positions))):
                item = qobj_hacks.qobj_snapshot_state(label)
                qobj_hacks.qobj_insert_item(qobj, exp_index, item, pos)

    def insert_prob_snapshots_before_barrier(self, qobj):
        """Insert probability snapshots before each barrier in a qobj.

        The snapshot labels are integer strings "0", "1", up to the number of barriers.
        """
        for exp_index in range(len(qobj.experiments)):
            num_qubits = qobj.experiments[exp_index].config.n_qubits
            barrier = qobj_hacks.qobj_barrier(num_qubits)
            positions = qobj_hacks.get_item_positions(qobj, exp_index, barrier)
            for label, pos in reversed(list(enumerate(positions))):
                item = qobj_hacks.qobj_snapshot_probs(label, range(num_qubits))
                qobj_hacks.qobj_insert_item(qobj, exp_index, item, pos)

    def state_fid(self, vec0, vec1):
        """State fidelity of 2 vectors.

        Don't want to depend on terra qi module which may move in future.
        """
        return abs(np.dot(np.conj(vec0), vec1)) ** 2

    def test_qv_snapshot(self):
        """ Test QV snapshot instruction """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.barrier(qr)  # snapshot "0" inserted here
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[1])
        circuit.barrier(qr)  # snapshot "1" inserted here

        # Add snapshots to Qobj
        qobj = compile(circuit, self.backend, shots=1)
        self.insert_state_snapshots_before_barrier(qobj)
        result = self.backend.run(qobj).result()
        self.assertEqual(result.get_status(), 'COMPLETED')
        snapshots = result.get_snapshots(circuit)

        # Bell state snapshot
        state0 = snapshots["state"]["0"][0]
        target0 = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        self.assertAlmostEqual(self.state_fid(target0, state0), 1.0)

        # 0-state snapshot
        state1 = snapshots["state"]["1"][0]
        target1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        self.assertAlmostEqual(self.state_fid(target1, state1), 1.0)


if __name__ == '__main__':
    unittest.main()
