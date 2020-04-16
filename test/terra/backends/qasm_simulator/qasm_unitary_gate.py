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


from test.terra.reference import ref_unitary_gate, ref_diagonal_gate

from qiskit import execute
from qiskit.providers.aer import QasmSimulator


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
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_random_unitary_gate(self):
        """Test simulation with random unitary gate circuit instructions."""
        shots = 2000
        circuits = ref_unitary_gate.unitary_random_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_unitary_gate.unitary_random_gate_counts_nondeterministic()
        result = execute(circuits, self.SIMULATOR, shots=shots).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmDiagonalGateTests:
    """QasmSimulator additional tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------

    def test_diagonal_gate(self):
        """Test simulation with unitary gate circuit instructions."""
        shots = 100
        circuits = ref_diagonal_gate.diagonal_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_diagonal_gate.diagonal_gate_counts_deterministic(
            shots)
        result = execute(circuits, self.SIMULATOR, shots=shots).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)
