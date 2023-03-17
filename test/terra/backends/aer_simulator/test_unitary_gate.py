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
import itertools
from ddt import ddt

from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info import Statevector

from test.terra.reference import ref_unitary_gate, ref_diagonal_gate

from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


@ddt
class TestUnitaryGates(SimulatorTestCase):
    """AerSimulator unitary gate tests."""

    METHODS = [
        "automatic",
        "statevector",
        "density_matrix",
        "matrix_product_state",
        "tensor_network",
    ]

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------

    @supported_methods(METHODS)
    def test_unitary_gate(self, method, device):
        """Test simulation with unitary gate circuit instructions."""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_unitary_gate.unitary_gate_circuits_deterministic(final_measure=True)
        targets = ref_unitary_gate.unitary_gate_counts_deterministic(shots)
        circuits = transpile(circuits, backend)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(METHODS)
    def test_random_unitary_gate(self, method, device):
        """Test simulation with random unitary gate circuit instructions."""
        backend = self.backend(method=method, device=device)
        shots = 4000
        circuits = ref_unitary_gate.unitary_random_gate_circuits_nondeterministic(
            final_measure=True
        )
        targets = ref_unitary_gate.unitary_random_gate_counts_nondeterministic(shots)
        circuits = transpile(circuits, backend)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    @supported_methods(METHODS, list(itertools.permutations([0, 1, 2])))
    def test_random_unitary_gate_with_permutations(self, method, device, perm):
        """Test simulation with random unitary gate with permutations."""
        backend = self.backend(method=method, device=device)
        all_permutations = list(itertools.permutations([0, 1, 2]))
        unitary_matrix = random_unitary(8, seed=5)
        n = 3
        shots = 2000
        circuit = QuantumCircuit(n, n)
        circuit.unitary(unitary_matrix, perm)
        circuit.barrier(range(n))
        circuit.measure(range(n), range(n))
        circuits = transpile(circuit, backend)
        result = backend.run(circuits, shots=shots).result()

        state = Statevector.from_label(n * "0").evolve(unitary_matrix, perm)
        state.seed(11111)
        probs = state.probabilities_dict()
        hex_counts = {hex(int(key, 2)): val * shots for key, val in probs.items()}
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [hex_counts], delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------

    @supported_methods(METHODS)
    def test_diagonal_gate(self, method, device):
        """Test simulation with unitary gate circuit instructions."""
        backend = self.backend(method=method, device=device)
        shots = 100
        circuits = ref_diagonal_gate.diagonal_gate_circuits_deterministic(final_measure=True)
        targets = ref_diagonal_gate.diagonal_gate_counts_deterministic(shots)
        circuits = transpile(circuits, backend)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)
