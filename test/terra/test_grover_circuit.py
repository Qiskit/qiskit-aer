# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import test.terra.common as common
import unittest

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit,
                    execute)
from qiskit_addon_qv import AerQvSimulator


class TestGroverCircuit(common.QiskitAerTestCase):
    """Testing a circuit originated in the Grover algorithm"""
    def setUp(self):
        self.qv_backend = AerQvSimulator()


    def test_grover_circuit(self):
        """Testing a circuit originated in the Grover algorithm"""
        
        qreg = QuantumRegister(6)
        creg = ClassicalRegister(2)
        circuit = QuantumCircuit(qreg, creg)

        circuit.h(qreg[0])
        circuit.h(qreg[1])
        circuit.x(qreg[2])
        circuit.x(qreg[3])
        circuit.x(qreg[0])
        circuit.cx(qreg[0], qreg[2])
        circuit.x(qreg[0])
        circuit.cx(qreg[1], qreg[3])
        circuit.ccx(qreg[2], qreg[3], qreg[4])
        circuit.cx(qreg[1], qreg[3])
        circuit.x(qreg[0])
        circuit.cx(qreg[0], qreg[2])
        circuit.x(qreg[0])
        circuit.x(qreg[1])
        circuit.x(qreg[4])
        circuit.h(qreg[4])
        circuit.ccx(qreg[0], qreg[1], qreg[4])
        circuit.h(qreg[4])
        circuit.x(qreg[0])
        circuit.x(qreg[1])
        circuit.x(qreg[4])
        circuit.h(qreg[0])
        circuit.h(qreg[1])
        circuit.h(qreg[4])
        circuit.measure(qreg[0], creg[0])
        circuit.measure(qreg[1], creg[1])

        shots = 100000
        result = execute(circuit, self.qv_backend, shots=shots).result()
        counts = result.get_counts(circuit)
        
        target = {
            '0x0': shots*625/1000,
            '0x1': shots*125/1000,
            '0x2': shots*125/1000,
            '0x3': shots*125/1000
        }
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)


if __name__ == '__main__':
    unittest.main()
