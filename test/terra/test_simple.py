# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import test.terra.common as common
import unittest

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit, execute)
from qiskit_aer.backends import QasmSimulator


class TestSimple(common.QiskitAerTestCase):
    """Simple integrations tests"""
    def setUp(self):
        self.backend = QasmSimulator()

    def test_simple_cricuit_execution(self):
        """Test the result of executing a simple H"""
        q_reg = QuantumRegister(1)
        c_reg = ClassicalRegister(1)
        q_circuit = QuantumCircuit(q_reg, c_reg)

        q_circuit.h(q_reg[0])
        q_circuit.measure(q_reg[0], c_reg[0])

        shots = 1000
        result = execute(q_circuit, self.backend, shots=shots).result()
        counts = result.get_counts(q_circuit)
        target = {
            '0x0': shots / 2,
            '0x1': shots / 2
        }
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)


if __name__ == '__main__':
    unittest.main()
