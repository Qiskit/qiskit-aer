# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import test.terra.common as common
import unittest

from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit,
                    compile, execute)
from qiskit_addon_qv import AerQvSimulator


class TestSimple(common.QiskitAerTestCase):
    """Simple integrations tests"""
    def setUp(self):
        self.qv_backend = AerQvSimulator()


    def test_simple_cricuit_execution(self):
        """Test the result of executing a simple H"""
        q_reg = QuantumRegister(1)
        c_reg = ClassicalRegister(1)
        q_circuit = QuantumCircuit(q_reg, c_reg)

        q_circuit.h(q_reg[0])
        q_circuit.measure(q_reg[0], c_reg[0])

        qobj = compile(q_circuit, backend=self.qv_backend, seed=73846087)
        job = self.qv_backend.run(qobj)
        result = job.result()
        counts = result.get_counts(q_circuit)
        target = {
            '0x0': qobj.config.shots/2,
            '0x1': qobj.config.shots/2
        }
        threshold = 0.04 * qobj.config.shots
        self.assertDictAlmostEqual(counts, target, threshold)


if __name__ == '__main__':
    unittest.main()
