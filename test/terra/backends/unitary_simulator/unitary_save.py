# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
QasmSimulator Integration Tests for SaveStatevector instruction
"""

import qiskit.quantum_info as qi
from qiskit import transpile, assemble
from qiskit.circuit.library import QuantumVolume
from qiskit.providers.aer import UnitarySimulator


class UnitarySaveUnitaryTests:
    """QasmSimulator SaveStatevector instruction tests."""

    SIMULATOR = UnitarySimulator()
    BACKEND_OPTS = {}

    def test_save_unitary(self):
        """Test save unitary for instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'unitary', 'unitary_gpu', 'unitary_thrust'
        ]

        # Stabilizer test circuit
        circ = transpile(QuantumVolume(3), self.SIMULATOR)

        # Target unitary
        target = qi.Operator(circ)

        # Add save to circuit
        label = 'state'
        circ.save_unitary(label)

        # Run
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = qi.Operator(result.data(0)[label])
            self.assertEqual(value, target)
