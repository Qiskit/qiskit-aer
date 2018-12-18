# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
StatevectorSimulator Integration Tests
"""

import unittest
from test.terra.utils import common
from test.terra.utils import ref_measure
from test.terra.utils import ref_reset
from test.terra.utils import ref_conditionals
from test.terra.utils import ref_1q_clifford
from test.terra.utils import ref_2q_clifford
from test.terra.utils import ref_non_clifford
from test.terra.utils import ref_unitary_gate

from qiskit import execute
from qiskit.providers.aer import StatevectorSimulator


class TestStatevectorSimulator(common.QiskitAerTestCase):
    """StatevectorSimulator tests."""

    # ---------------------------------------------------------------------
    # Test reset
    # ---------------------------------------------------------------------
    def test_reset_deterministic(self):
        """Test StatevectorSimulator reset with for circuits with deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        circuits = ref_reset.reset_circuits_deterministic(final_measure=False)
        targets = ref_reset.reset_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_reset_nondeterministic(self):
        """Test StatevectorSimulator reset with for circuits with non-deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        circuits = ref_reset.reset_circuits_nondeterministic(final_measure=False)
        targets = ref_reset.reset_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test measure
    # ---------------------------------------------------------------------
    def test_measure(self):
        """Test StatevectorSimulator measure with deterministic counts"""
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        targets = ref_measure.measure_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_measure_multi_qubit(self):
        """Test StatevectorSimulator multi-qubit measure with deterministic counts"""
        qobj = ref_measure.measure_circuits_qobj_deterministic(allow_sampling=True)
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_measure.measure_statevector_qobj_deterministic()
        job = StatevectorSimulator().run(qobj)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    def test_conditional_1bit(self):
        """Test conditional operations on 1-bit conditional register."""
        circuits = ref_conditionals.conditional_circuits_1bit(final_measure=False)
        targets = ref_conditionals.conditional_statevector_1bit()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_conditional_2bit(self):
        """Test conditional operations on 2-bit conditional register."""
        circuits = ref_conditionals.conditional_circuits_2bit(final_measure=False)
        targets = ref_conditionals.conditional_statevector_2bit()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------
    def test_h_gate_deterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.h_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_h_gate_deterministic_waltz_basis_gates(self):
        """Test h-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.h_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_h_gate_deterministic_minimal_basis_gates(self):
        """Test h-gate gate circuits compiling to U,CX"""
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.h_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_h_gate_nondeterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.h_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_h_gate_nondeterministic_waltz_basis_gates(self):
        """Test h-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.h_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_h_gate_nondeterministic_minimal_basis_gates(self):
        """Test h-gate gate circuits compiling to U,CX"""
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.h_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test x-gate
    # ---------------------------------------------------------------------
    def test_x_gate_deterministic_default_basis_gates(self):
        """Test x-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.x_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_x_gate_deterministic_waltz_basis_gates(self):
        """Test x-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.x_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_x_gate_deterministic_minimal_basis_gates(self):
        """Test x-gate gate circuits compiling to U,CX"""
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.x_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test z-gate
    # ---------------------------------------------------------------------
    def test_z_gate_deterministic_default_basis_gates(self):
        """Test z-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.z_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_z_gate_deterministic_waltz_basis_gates(self):
        """Test z-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.z_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_z_gate_deterministic_minimal_basis_gates(self):
        """Test z-gate gate circuits compiling to U,CX"""
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.z_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test y-gate
    # ---------------------------------------------------------------------
    def test_y_gate_deterministic_default_basis_gates(self):
        """Test y-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.y_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_y_gate_deterministic_waltz_basis_gates(self):
        """Test y-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.y_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_y_gate_deterministic_minimal_basis_gates(self):
        """Test y-gate gate circuits compiling to U,CX
        DISABLED until transpiler bug is fixed.
        """
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.y_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test s-gate
    # ---------------------------------------------------------------------
    def test_s_gate_deterministic_default_basis_gates(self):
        """Test s-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.s_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_s_gate_deterministic_waltz_basis_gates(self):
        """Test s-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.s_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_s_gate_deterministic_minimal_basis_gates(self):
        """Test s-gate gate circuits compiling to U,CX"""
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.s_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_s_gate_nondeterministic_default_basis_gates(self):
        """Test s-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.s_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_s_gate_nondeterministic_waltz_basis_gates(self):
        """Test s-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.s_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_s_gate_nondeterministic_minimal_basis_gates(self):
        """Test s-gate gate circuits compiling to U,CX"""
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.s_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test sdg-gate
    # ---------------------------------------------------------------------
    def test_sdg_gate_deterministic_default_basis_gates(self):
        """Test sdg-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.sdg_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_sdg_gate_deterministic_waltz_basis_gates(self):
        """Test sdg-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.sdg_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_sdg_gate_deterministic_minimal_basis_gates(self):
        """Test sdg-gate gate circuits compiling to U,CX"""
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(final_measure=False)
        targets = ref_1q_clifford.sdg_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_sdg_gate_nondeterministic_default_basis_gates(self):
        """Test sdg-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.sdg_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_sdg_gate_nondeterministic_waltz_basis_gates(self):
        """Test sdg-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.sdg_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_sdg_gate_nondeterministic_minimal_basis_gates(self):
        """Test sdg-gate gate circuits compiling to U,CX"""
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_1q_clifford.sdg_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test cx-gate
    # ---------------------------------------------------------------------
    def test_cx_gate_deterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(final_measure=False)
        targets = ref_2q_clifford.cx_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cx_gate_deterministic_waltz_basis_gates(self):
        """Test cx-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(final_measure=False)
        targets = ref_2q_clifford.cx_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cx_gate_deterministic_minimal_basis_gates(self):
        """Test cx-gate gate circuits compiling to U,CX"""
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(final_measure=False)
        targets = ref_2q_clifford.cx_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cx_gate_nondeterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_2q_clifford.cx_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cx_gate_nondeterministic_waltz_basis_gates(self):
        """Test cx-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_2q_clifford.cx_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cx_gate_nondeterministic_minimal_basis_gates(self):
        """Test cx-gate gate circuits compiling to U,CX"""
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_2q_clifford.cx_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test cz-gate
    # ---------------------------------------------------------------------
    def test_cz_gate_deterministic_default_basis_gates(self):
        """Test cz-gate circuits compiling to backend default basis_gates."""
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=False)
        targets = ref_2q_clifford.cz_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cz_gate_deterministic_waltz_basis_gates(self):
        """Test cz-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=False)
        targets = ref_2q_clifford.cz_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cz_gate_deterministic_minimal_basis_gates(self):
        """Test cz-gate gate circuits compiling to U,CX"""
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=False)
        targets = ref_2q_clifford.cz_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cz_gate_nondeterministic_default_basis_gates(self):
        """Test cz-gate circuits compiling to backend default basis_gates."""
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_2q_clifford.cz_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cz_gate_nondeterministic_waltz_basis_gates(self):
        """Test cz-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_2q_clifford.cz_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_cz_gate_nondeterministic_minimal_basis_gates(self):
        """Test cz-gate gate circuits compiling to U,CX"""
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_2q_clifford.cz_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test swap-gate
    # ---------------------------------------------------------------------
    def test_swap_gate_deterministic_default_basis_gates(self):
        """Test swap-gate circuits compiling to backend default basis_gates."""
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(final_measure=False)
        targets = ref_2q_clifford.swap_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_swap_gate_deterministic_waltz_basis_gates(self):
        """Test swap-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(final_measure=False)
        targets = ref_2q_clifford.swap_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_swap_gate_deterministic_minimal_basis_gates(self):
        """Test swap-gate gate circuits compiling to U,CX"""
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(final_measure=False)
        targets = ref_2q_clifford.swap_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_swap_gate_nondeterministic_default_basis_gates(self):
        """Test swap-gate circuits compiling to backend default basis_gates."""
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_2q_clifford.swap_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_swap_gate_nondeterministic_waltz_basis_gates(self):
        """Test swap-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_2q_clifford.swap_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_swap_gate_nondeterministic_minimal_basis_gates(self):
        """Test swap-gate gate circuits compiling to U,CX"""
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_2q_clifford.swap_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    def test_t_gate_deterministic_default_basis_gates(self):
        """Test t-gate circuits compiling to backend default basis_gates."""
        circuits = ref_non_clifford.t_gate_circuits_deterministic(final_measure=False)
        targets = ref_non_clifford.t_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_t_gate_deterministic_waltz_basis_gates(self):
        """Test t-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_non_clifford.t_gate_circuits_deterministic(final_measure=False)
        targets = ref_non_clifford.t_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_t_gate_deterministic_minimal_basis_gates(self):
        """Test t-gate gate circuits compiling to U,CX"""
        circuits = ref_non_clifford.t_gate_circuits_deterministic(final_measure=False)
        targets = ref_non_clifford.t_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_t_gate_nondeterministic_default_basis_gates(self):
        """Test t-gate circuits compiling to backend default basis_gates."""
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_non_clifford.t_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_t_gate_nondeterministic_waltz_basis_gates(self):
        """Test t-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_non_clifford.t_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_t_gate_nondeterministic_minimal_basis_gates(self):
        """Test t-gate gate circuits compiling to U,CX"""
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_non_clifford.t_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    def test_tdg_gate_deterministic_default_basis_gates(self):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(final_measure=False)
        targets = ref_non_clifford.tdg_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_tdg_gate_deterministic_waltz_basis_gates(self):
        """Test tdg-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(final_measure=False)
        targets = ref_non_clifford.tdg_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_tdg_gate_deterministic_minimal_basis_gates(self):
        """Test tdg-gate gate circuits compiling to U,CX"""
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(final_measure=False)
        targets = ref_non_clifford.tdg_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_tdg_gate_nondeterministic_default_basis_gates(self):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_non_clifford.tdg_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_tdg_gate_nondeterministic_waltz_basis_gates(self):
        """Test tdg-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_non_clifford.tdg_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_tdg_gate_nondeterministic_minimal_basis_gates(self):
        """Test tdg-gate gate circuits compiling to U,CX"""
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_non_clifford.tdg_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    def test_ccx_gate_deterministic_default_basis_gates(self):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=False)
        targets = ref_non_clifford.ccx_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_ccx_gate_deterministic_waltz_basis_gates(self):
        """Test ccx-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=False)
        targets = ref_non_clifford.ccx_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_ccx_gate_deterministic_minimal_basis_gates(self):
        """Test ccx-gate gate circuits compiling to U,CX"""
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=False)
        targets = ref_non_clifford.ccx_gate_statevector_deterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_ccx_gate_nondeterministic_default_basis_gates(self):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_non_clifford.ccx_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_ccx_gate_nondeterministic_waltz_basis_gates(self):
        """Test ccx-gate gate circuits compiling to u1,u2,u3,cx"""
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_non_clifford.ccx_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='u1,u2,u3,cx')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_ccx_gate_nondeterministic_minimal_basis_gates(self):
        """Test ccx-gate gate circuits compiling to U,CX"""
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(final_measure=False)
        targets = ref_non_clifford.ccx_gate_statevector_nondeterministic()
        job = execute(circuits, StatevectorSimulator(), shots=1, basis_gates='U,CX')
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------
    def test_unitary_gate_real(self):
        """Test unitary qobj instruction with real matrices."""
        qobj = ref_unitary_gate.unitary_gate_circuits_real_deterministic(final_measure=False)
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_unitary_gate.unitary_gate_statevector_real_deterministic()
        job = StatevectorSimulator().run(qobj)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)

    def test_unitary_gate_complex(self):
        """Test unitary qobj instruction with complex matrices."""
        qobj = ref_unitary_gate.unitary_gate_circuits_complex_deterministic(final_measure=False)
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_unitary_gate.unitary_gate_statevector_complex_deterministic()
        job = StatevectorSimulator().run(qobj)
        result = job.result()
        self.is_completed(result)
        self.compare_statevector(result, circuits, targets)


if __name__ == '__main__':
    unittest.main()
