# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
QasmSimulator Integration Tests
"""

from qiskit import execute
from qiskit.providers.aer import QasmSimulator

from test.terra.reference import ref_unitary_gate


class QasmUnitaryGateTests:
    """QasmSimulator additional tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------

    def test_unitary_gate(self):
        """Test simulation with unitary gate circuit instructions."""
        shots = 100
        circuits = ref_unitary_gate.unitary_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_unitary_gate.unitary_gate_counts_deterministic(
            shots)
        result = execute(circuits, self.SIMULATOR, shots=shots).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)
