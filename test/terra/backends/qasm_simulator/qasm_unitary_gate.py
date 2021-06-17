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


from test.terra.reference import ref_unitary_gate, ref_diagonal_gate

from qiskit import execute
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info import Statevector

import numpy as np
import itertools

class QasmUnitaryGateTests:
    """QasmSimulator unitary gate tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------

    def test_unitary_gate(self):
        """Test simulation with unitary gate circuit instructions."""
        shots = 100
        circuits = ref_unitary_gate.unitary_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_unitary_gate.unitary_gate_counts_deterministic(
            shots)
        result = execute(circuits, self.SIMULATOR, shots=shots,
                         **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_random_unitary_gate(self):
        """Test simulation with random unitary gate circuit instructions."""
        shots = 4000
        circuits = ref_unitary_gate.unitary_random_gate_circuits_nondeterministic(final_measure=True)
        targets = ref_unitary_gate.unitary_random_gate_counts_nondeterministic(shots)
        result = execute(circuits, self.SIMULATOR, shots=shots,
                         **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_random_unitary_gate_with_permutations(self):
        """Test simulation with random unitary gate with permutations."""
        all_permutations = list(itertools.permutations([0, 1, 2]))
        unitary_matrix = random_unitary(8, seed=5)
        n = 3
        shots = 2000
    
        for perm in all_permutations:
            circuit = QuantumCircuit(n, n)
            circuit.unitary(unitary_matrix, perm)
            circuit.barrier(range(n))
            circuit.measure(range(n), range(n))
            result = execute(circuit, self.SIMULATOR, shots=shots,
                         optimization_level=0, **self.BACKEND_OPTS).result()
            
            state = Statevector.from_label(n * '0').evolve(unitary_matrix, perm)
            counts = state.sample_counts(shots=shots)
            hex_counts = {hex(int(key, 2)): val for key, val in counts.items()}
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [hex_counts], delta=0.05 * shots)

class QasmDiagonalGateTests:
    """QasmSimulator diagonal gate tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------

    def test_diagonal_gate(self):
        """Test simulation with unitary gate circuit instructions."""
        shots = 100
        circuits = ref_diagonal_gate.diagonal_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_diagonal_gate.diagonal_gate_counts_deterministic(
            shots)
        result = execute(circuits, self.SIMULATOR, shots=shots,
                         **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)
