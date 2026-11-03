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

import numpy as np
from ddt import ddt, data
from test.terra.reference import ref_rotation
from qiskit import transpile
from qiskit.quantum_info import Statevector
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


CLIFFORD_ROTATION_GATES = ref_rotation.CLIFFORD_ROTATION_GATES


def _without_save_instructions(circuit):
    """Copy of ``circuit`` without save instructions."""
    reference = circuit.copy()
    reference.data = [
        inst for inst in reference.data if not inst.operation.name.startswith("save_")
    ]
    return reference


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

    # ---------------------------------------------------------------------
    # Global phase
    # ---------------------------------------------------------------------
    @data(*ref_rotation.CLIFFORD_ROTATION_GATES)
    def test_clifford_rotation_global_phase(self, gate):
        """Test that extended_stabilizer tracks the global phase of rotation gates."""
        backend = self.backend(method="extended_stabilizer", seed_simulator=self.SEED)
        circuits = ref_rotation.clifford_rotation_statevector_circuits(gate)
        result = backend.run(circuits, shots=1).result()
        self.assertSuccess(result)
        for circuit in circuits:
            actual = np.asarray(result.data(circuit)["statevector"])
            target = np.asarray(Statevector(_without_save_instructions(circuit)))
            self.assertTrue(
                np.allclose(actual, target, atol=1e-7),
                msg=f"{circuit.name}: {actual} != {target}",
            )

    @data(*ref_rotation.CLIFFORD_ROTATION_GATES)
    def test_clifford_rotation_global_phase_many_terms(self, gate):
        """Test the rotation global phase when the decomposition has chi > 1 terms."""
        backend = self.backend(method="extended_stabilizer", seed_simulator=self.SEED)
        for num_t_gates in (1, 2, 3):
            circuits = ref_rotation.clifford_rotation_statevector_circuits(
                gate, num_t_gates=num_t_gates
            )
            result = backend.run(circuits, shots=1).result()
            self.assertSuccess(result)
            for circuit in circuits:
                actual = np.asarray(result.data(circuit)["statevector"])
                target = np.asarray(Statevector(_without_save_instructions(circuit)))
                overlap = np.vdot(target, actual)
                # A wrong global phase rotates the overlap off the positive real
                # axis; the smallest error this can produce is pi/4.
                self.assertAlmostEqual(
                    np.angle(overlap),
                    0.0,
                    places=6,
                    msg=f"{circuit.name}: overlap {overlap}",
                )

    # ---------------------------------------------------------------------
    # Non-Clifford angles
    # ---------------------------------------------------------------------
    @supported_methods(["stabilizer", "extended_stabilizer"], CLIFFORD_ROTATION_GATES)
    def test_rotation_non_clifford_angle_rejected(self, method, device, gate):
        """Test the stabilizer methods reject rotation gates at non-Clifford angles."""
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        for theta in ref_rotation.NON_CLIFFORD_ANGLES:
            circuit = ref_rotation.rotation_circuit(gate, theta)
            result = backend.run(circuit, shots=100).result()
            self.assertNotSuccess(result)

    @supported_methods(["stabilizer", "extended_stabilizer"], CLIFFORD_ROTATION_GATES)
    def test_rotation_clifford_angle_counts(self, method, device, gate):
        """Test counts at Clifford angles, including negative ones, against statevector."""
        shots = 2000
        circuits = ref_rotation.clifford_rotation_basis_circuits(gate)
        backend = self.backend(method=method, device=device, seed_simulator=self.SEED)
        reference = self.backend(method="statevector", seed_simulator=self.SEED)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        target = reference.run(circuits, shots=shots).result()
        self.assertSuccess(target)
        self.compare_counts(
            result,
            circuits,
            [target.get_counts(c) for c in circuits],
            hex_counts=False,
            delta=0.06 * shots,
        )

    @data(*CLIFFORD_ROTATION_GATES)
    def test_rotation_angle_tolerance(self, gate):
        """Test the tolerance used to recognise an angle as a multiple of pi/2.

        The angle is snapped to the nearest multiple of pi/2, so the tolerance
        has to be tight enough that the simulated gate is still the intended
        one, but loose enough that an angle which has accumulated some roundoff
        on its way to the simulator is still recognised.
        """
        backend = self.backend(method="stabilizer", seed_simulator=self.SEED)
        for relative_error, accepted in ref_rotation.ANGLE_TOLERANCE_CASES:
            theta = np.pi / 2 * (1 + relative_error)
            result = backend.run(ref_rotation.rotation_circuit(gate, theta), shots=100).result()
            self.assertEqual(
                result.success,
                accepted,
                msg=f"{gate} at pi/2*(1+{relative_error:g}): "
                f"expected {'accept' if accepted else 'reject'}",
            )

    # ---------------------------------------------------------------------
    # Automatic method selection
    # ---------------------------------------------------------------------
    @data(*CLIFFORD_ROTATION_GATES)
    def test_rotation_automatic_method_selection(self, gate):
        """Test automatic selects stabilizer at Clifford angles and falls back otherwise."""
        backend = self.backend(method="automatic", seed_simulator=self.SEED)
        for multiple in ref_rotation.CLIFFORD_MULTIPLES:
            circuit = ref_rotation.rotation_circuit(gate, multiple * np.pi / 2)
            result = backend.run(circuit, shots=100).result()
            self.assertSuccess(result)
            self.assertEqual(
                result.results[0].metadata.get("method"),
                "stabilizer",
                msg=f"{gate}({multiple}*pi/2) did not select the stabilizer method",
            )

        # A non-Clifford it falls back to statevector
        for theta in ref_rotation.NON_CLIFFORD_ANGLES:
            circuit = ref_rotation.rotation_circuit(gate, theta)
            result = backend.run(circuit, shots=100).result()
            self.assertSuccess(result)
            self.assertNotEqual(
                result.results[0].metadata.get("method"),
                "stabilizer",
                msg=f"{gate}({theta}) should not select the stabilizer method",
            )
