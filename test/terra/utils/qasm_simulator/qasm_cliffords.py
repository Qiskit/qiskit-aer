# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

from test.terra.utils import common
from test.terra.utils import ref_1q_clifford
from test.terra.utils import ref_2q_clifford
from qiskit import compile
from qiskit.providers.aer import QasmSimulator


class QasmCliffordTests(common.QiskitAerTestCase):
    """QasmSimulator Clifford gate tests in default basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------
    def test_h_gate_deterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_h_gate_nondeterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test x-gate
    # ---------------------------------------------------------------------
    def test_x_gate_deterministic_default_basis_gates(self):
        """Test x-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.x_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test z-gate
    # ---------------------------------------------------------------------
    def test_z_gate_deterministic_default_basis_gates(self):
        """Test z-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test y-gate
    # ---------------------------------------------------------------------
    def test_y_gate_deterministic_default_basis_gates(self):
        """Test y-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test s-gate
    # ---------------------------------------------------------------------
    def test_s_gate_deterministic_default_basis_gates(self):
        """Test s-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_s_gate_nondeterministic_default_basis_gates(self):
        """Test s-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test sdg-gate
    # ---------------------------------------------------------------------
    def test_sdg_gate_deterministic_default_basis_gates(self):
        """Test sdg-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_sdg_gate_nondeterministic_default_basis_gates(self):
        shots = 2000
        """Test sdg-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cx-gate
    # ---------------------------------------------------------------------
    def test_cx_gate_deterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cx_gate_nondeterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cz-gate
    # ---------------------------------------------------------------------
    def test_cz_gate_deterministic_default_basis_gates(self):
        """Test cz-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cz_gate_nondeterministic_default_basis_gates(self):
        """Test cz-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test swap-gate
    # ---------------------------------------------------------------------
    def test_swap_gate_deterministic_default_basis_gates(self):
        """Test swap-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_swap_gate_nondeterministic_default_basis_gates(self):
        """Test swap-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmCliffordTestsWaltzBasis(common.QiskitAerTestCase):
    """QasmSimulator Clifford gate tests in Waltz u1,u2,u3,cx basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------
    def test_h_gate_deterministic_waltz_basis_gates(self):
        """Test h-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_h_gate_nondeterministic_waltz_basis_gates(self):
        """Test h-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test x-gate
    # ---------------------------------------------------------------------
    def test_x_gate_deterministic_waltz_basis_gates(self):
        """Test x-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.x_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test z-gate
    # ---------------------------------------------------------------------

    def test_z_gate_deterministic_waltz_basis_gates(self):
        """Test z-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_z_gate_deterministic_minimal_basis_gates(self):
        """Test z-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test y-gate
    # ---------------------------------------------------------------------
    def test_y_gate_deterministic_default_basis_gates(self):
        """Test y-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_y_gate_deterministic_waltz_basis_gates(self):
        shots = 100
        """Test y-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test s-gate
    # ---------------------------------------------------------------------
    def test_s_gate_deterministic_waltz_basis_gates(self):
        """Test s-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_s_gate_nondeterministic_waltz_basis_gates(self):
        """Test s-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test sdg-gate
    # ---------------------------------------------------------------------
    def test_sdg_gate_deterministic_waltz_basis_gates(self):
        """Test sdg-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_sdg_gate_nondeterministic_waltz_basis_gates(self):
        """Test sdg-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cx-gate
    # ---------------------------------------------------------------------
    def test_cx_gate_deterministic_waltz_basis_gates(self):
        shots = 100
        """Test cx-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cx_gate_nondeterministic_waltz_basis_gates(self):
        """Test cx-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cz-gate
    # ---------------------------------------------------------------------
    def test_cz_gate_deterministic_waltz_basis_gates(self):
        """Test cz-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cz_gate_nondeterministic_waltz_basis_gates(self):
        """Test cz-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test swap-gate
    # ---------------------------------------------------------------------
    def test_swap_gate_deterministic_waltz_basis_gates(self):
        """Test swap-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_swap_gate_nondeterministic_waltz_basis_gates(self):
        """Test swap-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmCliffordTestsMinimalBasis(common.QiskitAerTestCase):
    """QasmSimulator Clifford gate tests in minimam U,CX basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------

    def test_h_gate_deterministic_minimal_basis_gates(self):
        """Test h-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_h_gate_nondeterministic_minimal_basis_gates(self):
        """Test h-gate gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test x-gate
    # ---------------------------------------------------------------------
    def test_x_gate_deterministic_minimal_basis_gates(self):
        """Test x-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.x_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test z-gate
    # ---------------------------------------------------------------------

    def test_z_gate_deterministic_minimal_basis_gates(self):
        """Test z-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test y-gate
    # ---------------------------------------------------------------------

    def test_y_gate_deterministic_minimal_basis_gates(self):
        """Test y-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test s-gate
    # ---------------------------------------------------------------------

    def test_s_gate_deterministic_minimal_basis_gates(self):
        """Test s-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_s_gate_nondeterministic_minimal_basis_gates(self):
        """Test s-gate gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test sdg-gate
    # ---------------------------------------------------------------------
    def test_sdg_gate_deterministic_minimal_basis_gates(self):
        """Test sdg-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_sdg_gate_nondeterministic_minimal_basis_gates(self):
        """Test sdg-gate gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cx-gate
    # ---------------------------------------------------------------------
    def test_cx_gate_deterministic_minimal_basis_gates(self):
        """Test cx-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cx_gate_nondeterministic_minimal_basis_gates(self):
        """Test cx-gate gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cz-gate
    # ---------------------------------------------------------------------
    def test_cz_gate_deterministic_minimal_basis_gates(self):
        """Test cz-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cz_gate_nondeterministic_minimal_basis_gates(self):
        """Test cz-gate gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test swap-gate
    # ---------------------------------------------------------------------
    def test_swap_gate_deterministic_minimal_basis_gates(self):
        """Test swap-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_swap_gate_nondeterministic_minimal_basis_gates(self):
        """Test swap-gate gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
