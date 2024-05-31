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
