# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import test.terra.common as common
import unittest

import qiskit.extensions.simulator
from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit, execute)
from qiskit_addon_qv import AerQvSimulator
from qiskit.tools.qi.qi import state_fidelity


class NoMeasurementTest(common.QiskitAerTestCase):
    """Test the final statevector in circuits whose simulation is deterministic,
    i.e., contain no measurement or noise"""

    def setUp(self):
        # ***
        self.backend_qv = AerQvSimulator()
        # !!!  Replace with register(provider_class=AerQvProvider)

    def test_qv_snapshot(self):
        """ Test QV snapshot instruction """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.snapshot(3)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[1])

        result = execute(circuit, self.backend_qv).result()
        snapshot = result.get_snapshot(slot='3')
        target = [0.70710678 + 0.j, 0. + 0.j, 0. + 0.j, 0.70710678 + 0.j]

        expected_fidelity = 0.99
        fidelity = state_fidelity(snapshot, target)
        self.assertGreater(
            fidelity, expected_fidelity,
            "snapshot has low fidelity{0:.2g}.".format(fidelity))


if __name__ == '__main__':
    unittest.main()
