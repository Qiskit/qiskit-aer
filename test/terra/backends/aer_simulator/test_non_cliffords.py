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
from test.terra.reference import ref_non_clifford
from qiskit import transpile
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

SUPPORTED_METHODS = [
    "automatic",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "extended_stabilizer",
    "tensor_network",
]


@ddt
class TestNonCliffords(SimulatorTestCase):
    """AerSimulator T and CCX gate tests"""

    # ---------------------------------------------------------------------
    # Test t-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_t_gate_deterministic_default_basis_gates(self, method, device):
        """Test t-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_non_clifford.t_gate_circuits_deterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.t_gate_counts_deterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_t_gate_nondeterministic_default_basis_gates(self, method, device):
        """Test t-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(
            method=method, device=device, extended_stabilizer_metropolis_mixing_time=50
        )
        shots = 500
        circuits = ref_non_clifford.t_gate_circuits_nondeterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.t_gate_counts_nondeterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    # ---------------------------------------------------------------------
    # Test tdg-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_tdg_gate_deterministic_default_basis_gates(self, method, device):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_non_clifford.tdg_gate_circuits_deterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.tdg_gate_counts_deterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_tdg_gate_nondeterministic_default_basis_gates(self, method, device):
        """Test tdg-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(
            method=method, device=device, extended_stabilizer_metropolis_mixing_time=50
        )
        shots = 500
        circuits = ref_non_clifford.tdg_gate_circuits_nondeterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.tdg_gate_counts_nondeterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.1 * shots)

    # ---------------------------------------------------------------------
    # Test ccx-gate
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_ccx_gate_deterministic_default_basis_gates(self, method, device):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(
            method=method, device=device, extended_stabilizer_metropolis_mixing_time=100
        )
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.ccx_gate_counts_deterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(SUPPORTED_METHODS)
    def test_ccx_gate_nondeterministic_default_basis_gates(self, method, device):
        """Test ccx-gate circuits compiling to backend default basis_gates."""
        backend = self.backend(
            method=method, device=device, extended_stabilizer_metropolis_mixing_time=100
        )
        shots = 500
        circuits = ref_non_clifford.ccx_gate_circuits_nondeterministic(final_measure=True)
        circuits = transpile(circuits, backend, optimization_level=0)
        result = backend.run(circuits, shots=shots).result()
        targets = ref_non_clifford.ccx_gate_counts_nondeterministic(shots)
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.10 * shots)
