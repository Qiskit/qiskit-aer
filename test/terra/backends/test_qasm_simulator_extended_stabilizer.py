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
ExtendedStabilizer Integration Tests
"""

import unittest
import logging
from test.terra import common
from test.terra.reference import ref_measure
from test.terra.reference import ref_reset
from test.terra.reference import ref_conditionals
from test.terra.reference import ref_1q_clifford
from test.terra.reference import ref_2q_clifford
from test.terra.reference import ref_non_clifford
from test.terra.reference import ref_algorithms

from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator

logger = logging.getLogger(__name__)


class TestQasmExtendedStabilizerSimulator(common.QiskitAerTestCase):
    """QasmSimulator extended_stabilizer method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 1984,
        "method": "extended_stabilizer"
    }

    BACKEND_OPTS_SAMPLING = {
        "seed_simulator": 1984,
        "method": "extended_stabilizer",
        "extended_stabilizer_measure_sampling": True
    }

    # ---------------------------------------------------------------------
    # Test reset
    # ---------------------------------------------------------------------
    def test_reset_deterministic(self):
        """Test ExtendedStabilizer reset with for circuits with deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 100
        circuits = ref_reset.reset_circuits_deterministic(final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_reset.reset_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_reset_nondeterministic(self):
        """Test ExtendedStabilizer reset with for circuits with non-deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 4000
        circuits = ref_reset.reset_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_reset.reset_counts_nondeterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # # ---------------------------------------------------------------------
    # # Test measure
    # # ---------------------------------------------------------------------
    def test_measure_deterministic_with_sampling(self):
        """Test ExtendedStabilizer measure with deterministic counts with sampling"""
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(
            allow_sampling=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_measure.measure_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_deterministic_without_sampling(self):
        """Test ExtendedStabilizer measure with deterministic counts without sampling"""
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(
            allow_sampling=False)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_measure.measure_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_nondeterministic_with_sampling(self):
        """Test CHimulator measure with non-deterministic counts with sampling"""
        shots = 4000
        circuits = ref_measure.measure_circuits_nondeterministic(
            allow_sampling=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_measure_nondeterministic_without_sampling(self):
        """Test CHimulator measure with non-deterministic counts without sampling"""
        shots = 4000
        circuits = ref_measure.measure_circuits_nondeterministic(
            allow_sampling=False)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # # ---------------------------------------------------------------------
    # # Test multi-qubit measure qobj instruction
    # # ---------------------------------------------------------------------
    def test_measure_deterministic_multi_qubit_with_sampling(self):
        """Test ExtendedStabilizer multi-qubit measure with deterministic counts with sampling"""
        shots = 100
        circuits = ref_measure.multiqubit_measure_circuits_deterministic(
            allow_sampling=True)
        targets = ref_measure.multiqubit_measure_counts_deterministic(shots)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_deterministic_multi_qubit_without_sampling(self):
        """Test ExtendedStabilizer multi-qubit measure with deterministic counts without sampling"""
        shots = 100
        circuits = ref_measure.multiqubit_measure_circuits_deterministic(
            allow_sampling=False)
        targets = ref_measure.multiqubit_measure_counts_deterministic(shots)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_nondeterministic_multi_qubit_with_sampling(self):
        """Test CHimulator reset with non-deterministic counts"""
        shots = 4000
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(
            allow_sampling=True)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_measure_nondeterministic_multi_qubit_without_sampling(self):
        """Test CHimulator reset with non-deterministic counts"""
        shots = 4000
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(
            allow_sampling=False)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # # ---------------------------------------------------------------------
    # # Test conditional
    # # ---------------------------------------------------------------------
    def test_conditional_1bit(self):
        """Test conditional operations on 1-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_conditionals.conditional_counts_1bit(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_2bit(self):
        """Test conditional operations on 2-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_conditionals.conditional_counts_2bit(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------
    def test_h_gate_deterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_h_gate_nondeterministic_default_basis_gates(self):
        """Test h-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
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
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_1q_clifford.x_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
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
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
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
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
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
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_1q_clifford.s_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_s_gate_nondeterministic_default_basis_gates(self):
        """Test s-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
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
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_1q_clifford.sdg_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_sdg_gate_nondeterministic_default_basis_gates(self):
        shots = 4000
        """Test sdg-gate circuits compiling to backend default basis_gates."""
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
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
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_2q_clifford.cx_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cx_gate_nondeterministic_default_basis_gates(self):
        """Test cx-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
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
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_2q_clifford.cz_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cz_gate_nondeterministic_default_basis_gates(self):
        """Test cz-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
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
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_2q_clifford.swap_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_swap_gate_nondeterministic_default_basis_gates(self):
        """Test swap-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS_SAMPLING)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    def test_t_gate_deterministic_default_basis_gates(self):
        """Test t-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_non_clifford.t_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_non_clifford.t_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    def test_t_gate_nondeterministic_default_basis_gates(self):
        """Test t-gate circuits compiling to backend default basis_gates."""
        shots = 500
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_non_clifford.t_gate_counts_nondeterministic(shots)
        opts = self.BACKEND_OPTS.copy()
        opts["extended_stabilizer_mixing_time"] = 50
        job = QasmSimulator().run(qobj, backend_options=opts)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    def test_tdg_gate_deterministic_default_basis_gates(self):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_non_clifford.tdg_gate_counts_deterministic(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))

        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    def test_tdg_gate_nondeterministic_default_basis_gates(self):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        shots = 500
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_non_clifford.tdg_gate_counts_nondeterministic(shots)
        opts = self.BACKEND_OPTS.copy()
        opts["extended_stabilizer_mixing_time"] = 50
        job = QasmSimulator().run(qobj, backend_options=opts)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    def test_ccx_gate_deterministic_default_basis_gates(self):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_non_clifford.ccx_gate_counts_deterministic(shots)
        opts = self.BACKEND_OPTS.copy()
        opts["extended_stabilizer_mixing_time"] = 100
        job = QasmSimulator().run(qobj, backend_options=opts)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_ccx_gate_nondeterministic_default_basis_gates(self):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        shots = 500
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(
            final_measure=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_non_clifford.ccx_gate_counts_nondeterministic(shots)
        opts = self.BACKEND_OPTS.copy()
        opts["extended_stabilizer_mixing_time"] = 100
        job = QasmSimulator().run(qobj, backend_options=opts)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.10 * shots)

    # # ---------------------------------------------------------------------
    # # Test algorithms
    # # ---------------------------------------------------------------------
    def test_grovers_default_basis_gates(self):
        """Test grovers circuits compiling to backend default basis_gates."""
        shots = 500
        circuits = ref_algorithms.grovers_circuit(
            final_measure=True, allow_sampling=True)
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_algorithms.grovers_counts(shots)
        opts = self.BACKEND_OPTS.copy()
        opts["extended_stabilizer_mixing_time"] = 100
        job = QasmSimulator().run(qobj, backend_options=opts)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    def test_teleport_default_basis_gates(self):
        """Test teleport circuits compiling to backend default basis_gates."""
        shots = 1000
        circuits = ref_algorithms.teleport_circuit()
        qobj = assemble(circuits, QasmSimulator(), shots=shots)
        targets = ref_algorithms.teleport_counts(shots)
        job = QasmSimulator().run(qobj, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


if __name__ == '__main__':
    unittest.main()
