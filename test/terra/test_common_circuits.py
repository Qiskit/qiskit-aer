# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import test.terra.common as common
import unittest

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit,
                    execute)
from qiskit_aer.backends import QasmSimulator


class TestGroverCircuit(common.QiskitAerTestCase):
    """Testing circuits originated in the famous algorithms"""
    def setUp(self):
        self.backend = QasmSimulator()

    def run_circuit(self, circuit, target_distribution, shots, threshold_factor=0.05):
        result = execute(circuit, self.backend, shots=shots).result()
        counts = result.get_counts(circuit)
        target = {base_element: amplitude * shots
                  for base_element, amplitude in target_distribution.items()}
        self.assertDictAlmostEqual(counts, target, threshold_factor * shots)

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

        target_distribution = {
            '0x0': 0.625,
            '0x1': 0.125,
            '0x2': 0.125,
            '0x3': 0.125
        }

        self.run_circuit(circuit, target_distribution, shots=1000)

    def test_teleport_circuit(self):
        """Testing a circuit originated in the teleportation algorithm"""

        qreg = QuantumRegister(3)
        c0 = ClassicalRegister(1)
        c1 = ClassicalRegister(1)
        c2 = ClassicalRegister(1)
        circuit = QuantumCircuit(qreg, c0, c1, c2)

        circuit.h(qreg[1])
        circuit.cx(qreg[1], qreg[2])
        circuit.barrier(qreg)
        circuit.cx(qreg[0], qreg[1])
        circuit.h(qreg[0])
        circuit.measure(qreg[0], c0[0])
        circuit.measure(qreg[1], c1[0])
        circuit.z(qreg[2]).c_if(c0, 1)
        circuit.x(qreg[2]).c_if(c1, 1)
        circuit.measure(qreg[2], c2[0])

        target_distribution = {
            '0x0': 0,
            '0x1': 0,
            '0x2': 0.25,
            '0x3': 0.25,
            '0x4': 0.25,
            '0x5': 0.25,
            '0x6': 0,
            '0x7': 0
        }

        self.run_circuit(circuit, target_distribution, shots=1000)


if __name__ == '__main__':
    unittest.main()
