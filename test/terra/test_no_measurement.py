# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the final statevector in circuits whose simulation is deterministic,
    i.e., contain no measurement or noise"""

import test.terra.common as common
from qiskit_aer.utils import qobj_utils
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

    def state_fid(self, vec0, vec1):
        """State fidelity of 2 vectors.

        Don't want to depend on terra qi module which may move in future.
        """
        return abs(np.dot(np.conj(vec0), vec1)) ** 2

    def test_qasm_simulator_snapshot(self):
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
        snapshot = qobj_utils.qobj_snapshot_item("statevector", "")
        qobj_utils.qobj_insert_snapshots_after_barriers(qobj, snapshot)
        result = self.backend.run(qobj).result()
        self.assertEqual(result.get_status(), 'COMPLETED')
        snapshots = result.get_snapshots(circuit)

        # Bell state snapshot
        state0 = snapshots["statevector"]["0"][0]
        target0 = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        self.assertAlmostEqual(self.state_fid(target0, state0), 1.0)

        # 0-state snapshot
        state1 = snapshots["statevector"]["1"][0]
        target1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        self.assertAlmostEqual(self.state_fid(target1, state1), 1.0)


if __name__ == '__main__':
    unittest.main()
