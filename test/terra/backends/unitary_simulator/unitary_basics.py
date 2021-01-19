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
UnitarySimulator Integration Tests
"""

from numpy import exp, pi

from test.terra.reference import ref_1q_clifford
from test.terra.reference import ref_unitary_gate
from test.terra.reference import ref_diagonal_gate

from qiskit import execute, assemble, transpile
from qiskit.providers.aer import UnitarySimulator


class UnitarySimulatorTests:
    """UnitarySimulator tests."""

    SIMULATOR = UnitarySimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------
    def test_unitary_gate(self):
        """Test simulation with unitary gate circuit instructions."""
        circuits = ref_unitary_gate.unitary_gate_circuits_deterministic(
            final_measure=False)
        targets = ref_unitary_gate.unitary_gate_unitary_deterministic()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_unitary(result, circuits, targets)

    def test_diagonal_gate(self):
        """Test simulation with diagonal gate circuit instructions."""
        circuits = ref_diagonal_gate.diagonal_gate_circuits_deterministic(
            final_measure=False)
        targets = ref_diagonal_gate.diagonal_gate_unitary_deterministic()
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=1,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_unitary(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test global phase
    # ---------------------------------------------------------------------

    def test_qobj_global_phase(self):
        """Test qobj global phase."""

        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(
            final_measure=False)
        targets = ref_1q_clifford.h_gate_unitary_nondeterministic()

        qobj = assemble(transpile(circuits, self.SIMULATOR),
                        shots=1, **self.BACKEND_OPTS)
        # Set global phases
        for i, _ in enumerate(circuits):
            global_phase = (-1) ** i * (pi / 4)
            qobj.experiments[i].header.global_phase = global_phase
            targets[i] = exp(1j * global_phase) * targets[i]
        result = self.SIMULATOR.run(qobj).result()
        self.assertSuccess(result)
        self.compare_unitary(result, circuits, targets, ignore_phase=False)
