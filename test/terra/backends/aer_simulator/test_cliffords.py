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
AerSimulator Integration Tests
"""
from ddt import ddt
from test.terra.reference import ref_1q_clifford
from test.terra.reference import ref_2q_clifford
from qiskit import transpile
from qiskit import QuantumCircuit
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

SUPPORTED_METHODS = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "extended_stabilizer",
    "tensor_network",
]


@ddt
class TestCliffords(SimulatorTestCase):
    """AerSimulator Clifford gate tests"""

    SEED = 12345

    # ---------------------------------------------------------------------
    # Test h-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_h_gate_deterministic(self, method, device):
        """Test h-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_1q_clifford.h_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_h_gate_nondeterministic(self, method, device):
        """Test h-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 4000
        circuits = ref_1q_clifford.h_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.h_gate_counts_nondeterministic(shots)
        job = backend.run(circuits, shots=shots)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test x-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_x_gate_deterministic(self, method, device):
        """Test x-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_1q_clifford.x_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.x_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test z-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_z_gate_deterministic(self, method, device):
        """Test z-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_1q_clifford.z_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.z_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test y-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_y_gate_deterministic(self, method, device):
        """Test y-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_1q_clifford.y_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.y_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test s-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_s_gate_deterministic(self, method, device):
        """Test s-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_1q_clifford.s_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_s_gate_nondeterministic(self, method, device):
        """Test s-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 4000
        circuits = ref_1q_clifford.s_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.s_gate_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test sdg-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_sdg_gate_deterministic(self, method, device):
        """Test sdg-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_1q_clifford.sdg_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_sdg_gate_nondeterministic(self, method, device):
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 4000
        """Test sdg-gate circuits"""
        circuits = ref_1q_clifford.sdg_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_1q_clifford.sdg_gate_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cx-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_cx_gate_deterministic(self, method, device):
        """Test cx-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_cx_gate_nondeterministic(self, method, device):
        """Test cx-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 4000
        circuits = ref_2q_clifford.cx_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cx_gate_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test cz-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_cz_gate_deterministic(self, method, device):
        """Test cz-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_cz_gate_nondeterministic(self, method, device):
        """Test cz-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 4000
        circuits = ref_2q_clifford.cz_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.cz_gate_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test swap-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_swap_gate_deterministic(self, method, device):
        """Test swap-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_2q_clifford.swap_gate_circuits_deterministic(final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_swap_gate_nondeterministic(self, method, device):
        """Test swap-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 4000
        circuits = ref_2q_clifford.swap_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.swap_gate_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test pauli gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_pauli_gate_deterministic(self, method, device):
        """Test pauli gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_1q_clifford.pauli_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.pauli_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test ecr gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_ecr_gate_nondeterministic(self, method, device):
        """Test ecr gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_2q_clifford.ecr_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_2q_clifford.ecr_gate_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test identity gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_id_gate_deterministic(self, method, device):
        """Test id gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_1q_clifford.id_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.id_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    # ---------------------------------------------------------------------
    # Test delay gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_delay_gate_deterministic(self, method, device):
        """Test delay gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 100
        circuits = ref_1q_clifford.delay_gate_circuits_deterministic(final_measure=True)
        targets = ref_1q_clifford.delay_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)
