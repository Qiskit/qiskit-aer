# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Integration Tests for Parameterized Qobj execution, testing qasm_simulator,
statevector_simulator, and expectation value snapshots.
"""

import unittest
from math import pi
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.states import Statevector
from qiskit.circuit.library import QuantumVolume
from qiskit.providers.aer import AerSimulator

from test.terra import common
from qiskit.providers.aer.aererror import AerError

class TestAerStatevector(common.QiskitAerTestCase):
    """AerState tests"""

    def test_generate_aer_statevector(self):
        """Test generation of Aer's StateVector"""
        from qiskit.providers.aer.quantum_info.states import AerStatevector
        circ = QuantumCircuit(5)
        state = AerStatevector(circ)

    def test_qv(self):
        """Test generation of Aer's StateVector with QV """
        from qiskit.providers.aer.quantum_info.states import AerStatevector
        circ = QuantumVolume(5, seed=1111)
        state = AerStatevector(circ)
        expected = Statevector(circ)

        for e, s in zip(expected, state):
            self.assertAlmostEqual(e, s)

    def test_method_and_device_properties(self):
        """Test method and device properties"""
        from qiskit.providers.aer.quantum_info.states import AerStatevector
        circ = QuantumVolume(5, seed=1111)
        state1 = AerStatevector(circ)

        self.assertEqual('statevector', state1.method)
        self.assertEqual('CPU', state1.device)

        state2 = AerStatevector(circ, method='matrix_product_state')
        self.assertEqual('matrix_product_state', state2.method)
        self.assertEqual('CPU', state2.device)

        for pa1, pa2 in zip(state1, state2):
            self.assertAlmostEqual(pa1, pa2)

    def test_evolve(self):
        """Test method and device properties"""
        from qiskit.providers.aer.quantum_info.states import AerStatevector
        circ1 = QuantumVolume(5, seed=1111)
        circ2 = circ1.compose(circ1)

        state1 = AerStatevector(circ1).evolve(circ1)
        state2 = AerStatevector(circ2)

        for pa1, pa2 in zip(state1, state2):
            self.assertAlmostEqual(pa1, pa2)


if __name__ == '__main__':
    unittest.main()
