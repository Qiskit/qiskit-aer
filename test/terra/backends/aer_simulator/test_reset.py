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
from test.terra.reference import ref_reset
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

ALL_METHODS = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "extended_stabilizer",
    "tensor_network",
]


@ddt
class TestReset(SimulatorTestCase):
    """AerSimulator reset tests."""

    # ---------------------------------------------------------------------
    # Test reset
    # ---------------------------------------------------------------------
    @supported_methods(ALL_METHODS)
    def test_reset_deterministic(self, method, device):
        """Test AerSimulator reset with for circuits with deterministic counts"""
        backend = self.backend(method=method, device=device)
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 100
        circuits = ref_reset.reset_circuits_deterministic(final_measure=True)
        targets = ref_reset.reset_counts_deterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(ALL_METHODS)
    def test_reset_nondeterministic(self, method, device):
        """Test AerSimulator reset with for circuits with non-deterministic counts"""
        backend = self.backend(method=method, device=device)
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 4000
        circuits = ref_reset.reset_circuits_nondeterministic(final_measure=True)
        targets = ref_reset.reset_counts_nondeterministic(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(ALL_METHODS)
    def test_reset_sampling_opt(self, method, device):
        """Test sampling optimization"""
        backend = self.backend(method=method, device=device)
        shots = 4000
        circuits = ref_reset.reset_circuits_sampling_optimization()
        targets = ref_reset.reset_counts_sampling_optimization(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(ALL_METHODS)
    def test_repeated_resets(self, method, device):
        """Test repeated reset operations"""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_reset.reset_circuits_repeated()
        targets = ref_reset.reset_counts_repeated(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(ALL_METHODS)
    def test_reset_moving_qubits(self, method, device):
        """Test AerSimulator reset with for circuits where qubits have moved"""
        backend = self.backend(method=method, device=device)
        # count output circuits
        shots = 1000
        circuits = ref_reset.reset_circuits_with_entangled_and_moving_qubits(final_measure=True)
        targets = ref_reset.reset_counts_with_entangled_and_moving_qubits(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
