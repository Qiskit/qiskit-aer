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
    """QasmSimulator Clifford gate tests"""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------
    def test_h_gate_deterministic(self):
        """Test h-gate circuits"""
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_h_gate_nondeterministic(self):
        """Test h-gate circuits"""
        shots = 4000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test x-gate
    # ---------------------------------------------------------------------
    def test_x_gate_deterministic(self):
        """Test x-gate circuits"""
        shots = 100
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.x_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test z-gate
    # ---------------------------------------------------------------------
    def test_z_gate_deterministic(self):
        """Test z-gate circuits"""
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test y-gate
    # ---------------------------------------------------------------------
    def test_y_gate_deterministic(self):
        """Test y-gate circuits"""
        shots = 100
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test s-gate
    # ---------------------------------------------------------------------
    def test_s_gate_deterministic(self):
        """Test s-gate circuits"""
        shots = 100
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_s_gate_nondeterministic(self):
        """Test s-gate circuits"""
        shots = 4000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test sdg-gate
    # ---------------------------------------------------------------------
    def test_sdg_gate_deterministic(self):
        """Test sdg-gate circuits"""
        shots = 100
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_deterministic(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_sdg_gate_nondeterministic(self):
        shots = 4000
        """Test sdg-gate circuits"""
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cx-gate
    # ---------------------------------------------------------------------
    def test_cx_gate_deterministic(self):
        """Test cx-gate circuits"""
        shots = 100
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cx_gate_nondeterministic(self):
        """Test cx-gate circuits"""
        shots = 4000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cz-gate
    # ---------------------------------------------------------------------
    def test_cz_gate_deterministic(self):
        """Test cz-gate circuits"""
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_cz_gate_nondeterministic(self):
        """Test cz-gate circuits"""
        shots = 4000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test swap-gate
    # ---------------------------------------------------------------------
    def test_swap_gate_deterministic(self):
        """Test swap-gate circuits"""
        shots = 100
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_swap_gate_nondeterministic(self):
        """Test swap-gate circuits"""
        shots = 4000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test pauli gate
    # ---------------------------------------------------------------------
    def test_pauli_gate_deterministic(self):
        """Test pauli gate circuits"""
        if 'method' in self.BACKEND_OPTS:
            conf = self.SIMULATOR._method_configuration(self.BACKEND_OPTS['method'])
            basis_gates = conf.basis_gates
        else:
            basis_gates = None
        shots = 100
        circuits = ref_1q_clifford.pauli_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_1q_clifford.pauli_gate_counts_deterministic(shots)
        job = execute(circuits,
                      self.SIMULATOR,
                      shots=shots,
                      basis_gates=basis_gates,
                      **self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)
