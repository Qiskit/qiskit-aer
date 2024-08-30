# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests for assemble with Aer simulators
"""

from test.terra.common import QiskitAerTestCase

from ddt import data, ddt
from qiskit import QuantumCircuit, assemble

from qiskit_aer.backends import AerSimulator, QasmSimulator, StatevectorSimulator, UnitarySimulator


@ddt
class TestAssemble(QiskitAerTestCase):
    """Tests for assemble with Aer simulators."""

    @data("aer", "statevector", "qasm", "unitary")
    def test_assemble(self, simulator: str):
        """Test for assemble"""
        if simulator == "aer":
            backend = AerSimulator()
        elif simulator == "statevector":
            backend = StatevectorSimulator()
        elif simulator == "qasm":
            backend = QasmSimulator()
        elif simulator == "unitary":
            backend = UnitarySimulator()

        qc = QuantumCircuit(1, 1)
        qobj = assemble(qc, backend=backend)
        self.assertIsNotNone(qobj)
