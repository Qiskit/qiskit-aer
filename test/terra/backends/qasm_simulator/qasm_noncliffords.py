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

from test.terra.reference import ref_non_clifford
from qiskit import execute
from qiskit.providers.aer import QasmSimulator


class QasmNonCliffordTests:
    """QasmSimulator non-Clifford gate tests in default basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    def test_t_gate_deterministic_default_basis_gates(self):
        """Test t-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_non_clifford.t_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.t_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_t_gate_nondeterministic_default_basis_gates(self):
        """Test t-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.t_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    def test_tdg_gate_deterministic_default_basis_gates(self):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_tdg_gate_nondeterministic_default_basis_gates(self):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    def test_ccx_gate_deterministic_default_basis_gates(self):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_ccx_gate_nondeterministic_default_basis_gates(self):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cswap-gate (Fredkin)
    # ---------------------------------------------------------------------
    def test_cswap_gate_deterministic_default_basis_gates(self):
        shots = 100
        circuits = ref_non_clifford.cswap_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.cswap_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cswap_gate_nondeterministic_default_basis_gates(self):
        shots = 4000
        circuits = ref_non_clifford.cswap_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.cswap_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cu1 gate
    # ---------------------------------------------------------------------
    def test_cu1_gate_nondeterministic_default_basis_gates(self):
        """Test cu1-gate gate circuits compiling to default basis."""
        shots = 4000
        circuits = ref_non_clifford.cu1_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.cu1_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cu3 gate
    # ---------------------------------------------------------------------
    def test_cu3_gate_deterministic_default_basis_gates(self):
        """Test cu3-gate gate circuits compiling to default basis."""
        shots = 100
        circuits = ref_non_clifford.cu3_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.cu3_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)


class QasmNonCliffordTestsWaltzBasis:
    """QasmSimulator non-Clifford gate tests in minimal u1,u2,u3,cx basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    def test_t_gate_deterministic_waltz_basis_gates(self):
        """Test t-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_non_clifford.t_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.t_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_t_gate_nondeterministic_waltz_basis_gates(self):
        """Test t-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.t_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    def test_tdg_gate_deterministic_waltz_basis_gates(self):
        """Test tdg-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_tdg_gate_nondeterministic_waltz_basis_gates(self):
        """Test tdg-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    def test_ccx_gate_deterministic_waltz_basis_gates(self):
        """Test ccx-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_ccx_gate_nondeterministic_waltz_basis_gates(self):
        """Test ccx-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cswap-gate (Fredkin)
    # ---------------------------------------------------------------------
    def test_cswap_gate_deterministic_waltz_basis_gates(self):
        """Test cswap-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_non_clifford.cswap_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.cswap_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cswap_gate_nondeterministic_waltz_basis_gates(self):
        """Test cswap-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.cswap_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.cswap_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cu1 gate
    # ---------------------------------------------------------------------
    def test_cu1_gate_nondeterministic_waltz_basis_gates(self):
        """Test cu1-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.cu1_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.cu1_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cu3 gate
    # ---------------------------------------------------------------------
    def test_cu3_gate_deterministic_default_basis_gates(self):
        """Test cu3-gate gate circuits compiling to u1,u2,u3,cx."""
        shots = 100
        circuits = ref_non_clifford.cu3_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.cu3_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)


class QasmNonCliffordTestsMinimalBasis:
    """QasmSimulator non-Clifford gate tests in minimal U,CX basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    def test_t_gate_deterministic_minimal_basis_gates(self):
        """Test t-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_non_clifford.t_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.t_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_t_gate_nondeterministic_minimal_basis_gates(self):
        """Test t-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.t_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    def test_tdg_gate_deterministic_minimal_basis_gates(self):
        """Test tdg-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_tdg_gate_nondeterministic_minimal_basis_gates(self):
        """Test tdg-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    def test_ccx_gate_deterministic_minimal_basis_gates(self):
        """Test ccx-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_ccx_gate_nondeterministic_minimal_basis_gates(self):
        """Test ccx-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    # ---------------------------------------------------------------------
    # Test cu1 gate
    # ---------------------------------------------------------------------
    def test_cu1_gate_nondeterministic_minimal_basis_gates(self):
        """Test cu1-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.cu1_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.cu1_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    # ---------------------------------------------------------------------
    # Test multiplexer-gate
    # ---------------------------------------------------------------------
    def test_multiplexer_cxx_gate_deterministic_default_basis_gates(self):
        """Test multiplexer-gate gate circuits """
        shots = 100
        circuits = ref_non_clifford.multiplexer_ccx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.multiplexer_ccx_gate_counts_deterministic(
            shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_multiplexer_cxx_gate_nondeterministic_default_basis_gates(self):
        """Test ccx-gate gate circuits """
        shots = 4000
        circuits = ref_non_clifford.multiplexer_ccx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.multiplexer_ccx_gate_counts_nondeterministic(
            shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cswap-gate (Fredkin)
    # ---------------------------------------------------------------------
    def test_cswap_gate_deterministic_minimal_basis_gates(self):
        """Test cswap-gate gate circuits compiling to u3,cx"""
        shots = 100
        circuits = ref_non_clifford.cswap_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.cswap_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cswap_gate_nondeterministic_minimal_basis_gates(self):
        """Test cswap-gate gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_non_clifford.cswap_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_non_clifford.cswap_gate_counts_nondeterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    # ---------------------------------------------------------------------
    # Test cu3 gate
    # ---------------------------------------------------------------------
    def test_cu3_gate_deterministic_default_basis_gates(self):
        """Test cu3-gate gate circuits compiling to u3, cx."""
        shots = 100
        circuits = ref_non_clifford.cu3_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_non_clifford.cu3_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)
