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

from test.terra.reference import ref_1q_clifford
from test.terra.reference import ref_2q_clifford
from qiskit import execute
from qiskit.providers.aer import QasmSimulator


class QasmCliffordTests:
    """QasmSimulator Clifford gate tests in default basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------
    def test_h_gate_deterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_h_gate_nondeterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test x-gate
    # ---------------------------------------------------------------------
    def test_x_gate_deterministic_default_basis_gates(self):
        """Test x-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.x_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test z-gate
    # ---------------------------------------------------------------------
    def test_z_gate_deterministic_default_basis_gates(self):
        """Test z-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test y-gate
    # ---------------------------------------------------------------------
    def test_y_gate_deterministic_default_basis_gates(self):
        """Test y-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test s-gate
    # ---------------------------------------------------------------------
    def test_s_gate_deterministic_default_basis_gates(self):
        """Test s-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_s_gate_nondeterministic_default_basis_gates(self):
        """Test s-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test sdg-gate
    # ---------------------------------------------------------------------
    def test_sdg_gate_deterministic_default_basis_gates(self):
        """Test sdg-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_sdg_gate_nondeterministic_default_basis_gates(self):
        shots = 4000
        """Test sdg-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cx-gate
    # ---------------------------------------------------------------------
    def test_cx_gate_deterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cx_gate_nondeterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cz-gate
    # ---------------------------------------------------------------------
    def test_cz_gate_deterministic_default_basis_gates(self):
        """Test cz-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cz_gate_nondeterministic_default_basis_gates(self):
        """Test cz-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test swap-gate
    # ---------------------------------------------------------------------
    def test_swap_gate_deterministic_default_basis_gates(self):
        """Test swap-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_swap_gate_nondeterministic_default_basis_gates(self):
        """Test swap-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmCliffordTestsWaltzBasis:
    """QasmSimulator Clifford gate tests in Waltz u1,u2,u3,cx basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------
    def test_h_gate_deterministic_waltz_basis_gates(self):
        """Test h-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_h_gate_nondeterministic_waltz_basis_gates(self):
        """Test h-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test x-gate
    # ---------------------------------------------------------------------
    def test_x_gate_deterministic_waltz_basis_gates(self):
        """Test x-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.x_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test z-gate
    # ---------------------------------------------------------------------

    def test_z_gate_deterministic_waltz_basis_gates(self):
        """Test z-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_z_gate_deterministic_minimal_basis_gates(self):
        """Test z-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test y-gate
    # ---------------------------------------------------------------------
    def test_y_gate_deterministic_default_basis_gates(self):
        """Test y-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_y_gate_deterministic_waltz_basis_gates(self):
        shots = 100
        """Test y-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test s-gate
    # ---------------------------------------------------------------------
    def test_s_gate_deterministic_waltz_basis_gates(self):
        """Test s-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_s_gate_nondeterministic_waltz_basis_gates(self):
        """Test s-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test sdg-gate
    # ---------------------------------------------------------------------
    def test_sdg_gate_deterministic_waltz_basis_gates(self):
        """Test sdg-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_sdg_gate_nondeterministic_waltz_basis_gates(self):
        """Test sdg-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cx-gate
    # ---------------------------------------------------------------------
    def test_cx_gate_deterministic_waltz_basis_gates(self):
        shots = 100
        """Test cx-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cx_gate_nondeterministic_waltz_basis_gates(self):
        """Test cx-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cz-gate
    # ---------------------------------------------------------------------
    def test_cz_gate_deterministic_waltz_basis_gates(self):
        """Test cz-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cz_gate_nondeterministic_waltz_basis_gates(self):
        """Test cz-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test swap-gate
    # ---------------------------------------------------------------------
    def test_swap_gate_deterministic_waltz_basis_gates(self):
        """Test swap-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_swap_gate_nondeterministic_waltz_basis_gates(self):
        """Test swap-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmCliffordTestsMinimalBasis:
    """QasmSimulator Clifford gate tests in minimam U,CX basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------

    def test_h_gate_deterministic_minimal_basis_gates(self):
        """Test h-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_h_gate_nondeterministic_minimal_basis_gates(self):
        """Test h-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test x-gate
    # ---------------------------------------------------------------------
    def test_x_gate_deterministic_minimal_basis_gates(self):
        """Test x-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.x_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test z-gate
    # ---------------------------------------------------------------------

    def test_z_gate_deterministic_minimal_basis_gates(self):
        """Test z-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test y-gate
    # ---------------------------------------------------------------------

    def test_y_gate_deterministic_minimal_basis_gates(self):
        """Test y-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test s-gate
    # ---------------------------------------------------------------------

    def test_s_gate_deterministic_minimal_basis_gates(self):
        """Test s-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_s_gate_nondeterministic_minimal_basis_gates(self):
        """Test s-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test sdg-gate
    # ---------------------------------------------------------------------
    def test_sdg_gate_deterministic_minimal_basis_gates(self):
        """Test sdg-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_sdg_gate_nondeterministic_minimal_basis_gates(self):
        """Test sdg-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cx-gate
    # ---------------------------------------------------------------------
    def test_cx_gate_deterministic_minimal_basis_gates(self):
        """Test cx-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cx_gate_nondeterministic_minimal_basis_gates(self):
        """Test cx-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cz-gate
    # ---------------------------------------------------------------------
    def test_cz_gate_deterministic_minimal_basis_gates(self):
        """Test cz-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cz_gate_nondeterministic_minimal_basis_gates(self):
        """Test cz-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test swap-gate
    # ---------------------------------------------------------------------
    def test_swap_gate_deterministic_minimal_basis_gates(self):
        """Test swap-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_deterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_swap_gate_nondeterministic_minimal_basis_gates(self):
        """Test swap-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test multiplexer-cx-gate
    # ---------------------------------------------------------------------
    def test_multiplexer_cx_gate_deterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_2q_clifford.multiplexer_cx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.multiplexer_cx_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_multiplexer_cx_gate_nondeterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_2q_clifford.multiplexer_cx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.multiplexer_cx_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
