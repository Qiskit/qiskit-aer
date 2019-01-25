# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

from test.terra.utils import common
from test.terra.utils import ref_non_clifford
from qiskit import compile
from qiskit.providers.aer import QasmSimulator


class QasmNonCliffordTests(common.QiskitAerTestCase):
    """QasmSimulator non-Clifford gate tests in default basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    def test_t_gate_deterministic_default_basis_gates(self):
        """Test t-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_non_clifford.t_gate_circuits_deterministic(final_measure=True)
        targets = ref_non_clifford.t_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_t_gate_nondeterministic_default_basis_gates(self):
        """Test t-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_non_clifford.t_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    def test_tdg_gate_deterministic_default_basis_gates(self):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_tdg_gate_nondeterministic_default_basis_gates(self):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    def test_ccx_gate_deterministic_default_basis_gates(self):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_ccx_gate_nondeterministic_default_basis_gates(self):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmNonCliffordTestsWaltzBasis(common.QiskitAerTestCase):
    """QasmSimulator non-Clifford gate tests in minimal u1,u2,u3,cx basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    def test_t_gate_deterministic_waltz_basis_gates(self):
        """Test t-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_non_clifford.t_gate_circuits_deterministic(final_measure=True)
        targets = ref_non_clifford.t_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_t_gate_nondeterministic_waltz_basis_gates(self):
        """Test t-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_non_clifford.t_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    def test_tdg_gate_deterministic_waltz_basis_gates(self):
        """Test tdg-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_tdg_gate_nondeterministic_waltz_basis_gates(self):
        """Test tdg-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    def test_ccx_gate_deterministic_waltz_basis_gates(self):
        """Test ccx-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_ccx_gate_nondeterministic_waltz_basis_gates(self):
        """Test ccx-gate gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmNonCliffordTestsMinimalBasis(common.QiskitAerTestCase):
    """QasmSimulator non-Clifford gate tests in minimal U,CX basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    def test_t_gate_deterministic_minimal_basis_gates(self):
        """Test t-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_non_clifford.t_gate_circuits_deterministic(final_measure=True)
        targets = ref_non_clifford.t_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_t_gate_nondeterministic_minimal_basis_gates(self):
        """Test t-gate gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_non_clifford.t_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    def test_tdg_gate_deterministic_minimal_basis_gates(self):
        """Test tdg-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_tdg_gate_nondeterministic_minimal_basis_gates(self):
        """Test tdg-gate gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_non_clifford.tdg_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    def test_ccx_gate_deterministic_minimal_basis_gates(self):
        """Test ccx-gate gate circuits compiling to U,CX"""
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_deterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_ccx_gate_nondeterministic_minimal_basis_gates(self):
        """Test ccx-gate gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_non_clifford.ccx_gate_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
