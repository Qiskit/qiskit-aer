# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
StabilizerSimulator Integration Tests
"""

import unittest
from test.terra.utils import common
from test.terra.utils import ref_measure
from test.terra.utils import ref_reset
from test.terra.utils import ref_conditionals
from test.terra.utils import ref_1q_clifford
from test.terra.utils import ref_2q_clifford
from test.terra.utils import ref_non_clifford
from test.terra.utils import ref_algorithms
from test.terra.utils import ref_unitary_gate

from qiskit import execute
from qiskit.providers.aer import StabilizerSimulator


class TestStabilizerSimulator(common.QiskitAerTestCase):
    """StabilizerSimulator tests."""

    # ---------------------------------------------------------------------
    # Test reset
    # ---------------------------------------------------------------------
    def test_reset_deterministic(self):
        """Test StabilizerSimulator reset with for circuits with deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 100
        circuits = ref_reset.reset_circuits_deterministic(final_measure=True)
        targets = ref_reset.reset_counts_deterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_reset_nondeterministic(self):
        """Test StabilizerSimulator reset with for circuits with non-deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 2000
        circuits = ref_reset.reset_circuits_nondeterministic(final_measure=True)
        targets = ref_reset.reset_counts_nondeterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test measure
    # ---------------------------------------------------------------------
    def test_measure_deterministic_with_sampling(self):
        """Test StabilizerSimulator measure with deterministic counts with sampling"""
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_deterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_deterministic_without_sampling(self):
        """Test StabilizerSimulator measure with deterministic counts without sampling"""
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(allow_sampling=False)
        targets = ref_measure.measure_counts_deterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_nondeterministic_with_sampling(self):
        """Test StabilizerSimulator measure with non-deterministic counts with sampling"""
        shots = 2000
        circuits = ref_measure.measure_circuits_nondeterministic(allow_sampling=True)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_measure_nondeterministic_without_sampling(self):
        """Test StabilizerSimulator measure with nin-deterministic counts without sampling"""
        shots = 2000
        circuits = ref_measure.measure_circuits_nondeterministic(allow_sampling=False)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test multi-qubit measure qobj instruction
    # ---------------------------------------------------------------------
    def test_measure_deterministic_multi_qubit_with_sampling(self):
        """Test StabilizerSimulator multi-qubit measure with deterministic counts with sampling"""
        shots = 100
        qobj = ref_measure.measure_circuits_qobj_deterministic(allow_sampling=True)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_measure.measure_counts_qobj_deterministic(shots)
        job = StabilizerSimulator().run(qobj)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_deterministic_multi_qubit_without_sampling(self):
        """Test StabilizerSimulator multi-qubit measure with deterministic counts without sampling"""
        shots = 100
        qobj = ref_measure.measure_circuits_qobj_deterministic(allow_sampling=False)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_measure.measure_counts_qobj_deterministic(shots)
        job = StabilizerSimulator().run(qobj)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_nondeterministic_multi_qubit_with_sampling(self):
        """Test StabilizerSimulator reset with non-deterministic counts"""
        shots = 2000
        qobj = ref_measure.measure_circuits_qobj_nondeterministic(allow_sampling=True)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_measure.measure_counts_qobj_nondeterministic(shots)
        job = StabilizerSimulator().run(qobj)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_measure_nondeterministic_multi_qubit_without_sampling(self):
        """Test StabilizerSimulator reset with non-deterministic counts"""
        shots = 2000
        qobj = ref_measure.measure_circuits_qobj_nondeterministic(allow_sampling=False)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_measure.measure_counts_qobj_nondeterministic(shots)
        job = StabilizerSimulator().run(qobj)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    def test_conditional_1bit(self):
        """Test conditional operations on 1-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_1bit(final_measure=True)
        targets = ref_conditionals.conditional_counts_1bit(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_2bit(self):
        """Test conditional operations on 2-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_2bit(final_measure=True)
        targets = ref_conditionals.conditional_counts_2bit(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------
    def test_h_gate_deterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_h_gate_nondeterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
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
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
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
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
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
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
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
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)


    def test_s_gate_nondeterministic_default_basis_gates(self):
        """Test s-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
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
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_sdg_gate_nondeterministic_default_basis_gates(self):
        shots = 2000
        """Test sdg-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
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
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cx_gate_nondeterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
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
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cz_gate_nondeterministic_default_basis_gates(self):
        """Test cz-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
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
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_swap_gate_nondeterministic_default_basis_gates(self):
        """Test swap-gate circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test algorithms
    # ---------------------------------------------------------------------

    def test_teleport_default_basis_gates(self):
        """Test teleport circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_algorithms.teleport_circuit()
        targets = ref_algorithms.teleport_counts(shots)
        job = execute(circuits, StabilizerSimulator(), shots=shots)
        result = job.result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


if __name__ == '__main__':
    unittest.main()
