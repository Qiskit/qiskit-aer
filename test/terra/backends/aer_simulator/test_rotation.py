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
from test.terra.reference import ref_rotation
from qiskit import transpile
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

SUPPORTED_METHODS = [
    "automatic",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "tensor_network",
]

SUPPORTED_METHODS_RZ = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "tensor_network",
    "extended_stabilizer",
]

SUPPORTED_METHODS_RX_CLIFFORD = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "tensor_network",
    "extended_stabilizer",
]

SUPPORTED_METHODS_RZZ_CLIFFORD = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "tensor_network",
    "extended_stabilizer",
]

SUPPORTED_METHODS_RY_CLIFFORD = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "tensor_network",
    "extended_stabilizer",
]

SUPPORTED_METHODS_RXX_CLIFFORD = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "tensor_network",
    "extended_stabilizer",
]

SUPPORTED_METHODS_RYY_CLIFFORD = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "tensor_network",
    "extended_stabilizer",
]

SUPPORTED_METHODS_RZX_CLIFFORD = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "tensor_network",
    "extended_stabilizer",
]


@ddt
class TestRotation(SimulatorTestCase):
    """AerSimulator Rotation gate tests"""

    SEED = 12345

    # ---------------------------------------------------------------------
    # Test rx-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_rx_gate_deterministic(self, method, device):
        """Test rx-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_rotation.rx_gate_circuits_deterministic(final_measure=True)
        targets = ref_rotation.rx_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS_RX_CLIFFORD)
    def test_rx_gate_clifford(self, method, device):
        """Test rx-gate at Clifford angles (k * pi/2) including stabilizer methods.

        Uses sign-sensitive circuits (RX -> Sdg -> H) that yield different
        deterministic outcomes for +pi/2 vs -pi/2, catching sign errors.
        """
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_rotation.rx_gate_clifford_circuits(final_measure=True)
        targets = ref_rotation.rx_gate_clifford_counts(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test rz-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS_RZ)
    def test_rz_gate_deterministic(self, method, device):
        """Test rz-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_rotation.rz_gate_circuits_deterministic(final_measure=True)
        targets = ref_rotation.rz_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test rzz-gate (Clifford angles)
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS_RZZ_CLIFFORD)
    def test_rzz_gate_clifford(self, method, device):
        """Test rzz-gate at Clifford angles (k * pi/2)"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_rotation.rzz_gate_clifford_circuits(final_measure=True)
        targets = ref_rotation.rzz_gate_clifford_counts(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test ry-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_ry_gate_deterministic(self, method, device):
        """Test ry-gate circuits"""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_rotation.ry_gate_circuits_deterministic(final_measure=True)
        targets = ref_rotation.ry_gate_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS_RY_CLIFFORD)
    def test_ry_gate_clifford(self, method, device):
        """Test ry-gate at Clifford angles (k * pi/2) including stabilizer methods.

        Uses sign-sensitive circuits (RY -> H) that yield different
        deterministic outcomes for +pi/2 vs 3*pi/2, catching sign errors.
        """
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_rotation.ry_gate_clifford_circuits(final_measure=True)
        targets = ref_rotation.ry_gate_clifford_counts(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test rxx-gate (Clifford angles)
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS_RXX_CLIFFORD)
    def test_rxx_gate_clifford(self, method, device):
        """Test rxx-gate at Clifford angles (k * pi/2) including stabilizer methods.

        Uses sign-sensitive circuits (RXX -> S0 -> CX -> H0) that yield different
        deterministic outcomes for +pi/2 vs 3*pi/2, catching sign errors.
        """
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_rotation.rxx_gate_clifford_circuits(final_measure=True)
        targets = ref_rotation.rxx_gate_clifford_counts(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test ryy-gate (Clifford angles)
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS_RYY_CLIFFORD)
    def test_ryy_gate_clifford(self, method, device):
        """Test ryy-gate at Clifford angles (k * pi/2) including stabilizer methods.

        Uses sign-sensitive circuits (RYY -> S0 -> CX -> H0) that yield different
        deterministic outcomes for +pi/2 vs 3*pi/2, catching sign errors.
        """
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_rotation.ryy_gate_clifford_circuits(final_measure=True)
        targets = ref_rotation.ryy_gate_clifford_counts(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test rzx-gate (Clifford angles)
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS_RZX_CLIFFORD)
    def test_rzx_gate_clifford(self, method, device):
        """Test rzx-gate at Clifford angles (k * pi/2) including stabilizer methods.

        Uses sign-sensitive circuits (RZX -> Sdg1 -> H1) that yield different
        deterministic outcomes for +pi/2 vs 3*pi/2, catching sign errors.
        """
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        shots = 1000
        circuits = ref_rotation.rzx_gate_clifford_circuits(final_measure=True)
        targets = ref_rotation.rzx_gate_clifford_counts(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
