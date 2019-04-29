# This code is part of Qiskit.
#
# (C) Copyright IBM Corp. 2017 and later.
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

from test.terra.reference import ref_unitary_gate
from qiskit.providers.aer import QasmSimulator


class QasmExtraTests:
    """QasmSimulator additional tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------
    def test_unitary_gate_real(self):
        """Test unitary qobj instruction with real matrices."""
        shots = 100
        qobj = ref_unitary_gate.unitary_gate_circuits_real_deterministic(
            final_measure=True)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_unitary_gate.unitary_gate_counts_real_deterministic(
            shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_unitary_gate_complex(self):
        """Test unitary qobj instruction with complex matrices."""
        shots = 100
        qobj = ref_unitary_gate.unitary_gate_circuits_complex_deterministic(
            final_measure=True)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_unitary_gate.unitary_gate_counts_complex_deterministic(
            shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)
