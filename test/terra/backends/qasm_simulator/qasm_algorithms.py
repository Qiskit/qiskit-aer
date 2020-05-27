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

from test.terra.reference import ref_algorithms
from qiskit import execute
from qiskit.providers.aer import QasmSimulator


class QasmAlgorithmTests:
    """QasmSimulator algorithm tests in the default basis"""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test algorithms
    # ---------------------------------------------------------------------
    def test_grovers_default_basis_gates(self):
        """Test grovers circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_algorithms.grovers_circuit(
            final_measure=True, allow_sampling=True)
        targets = ref_algorithms.grovers_counts(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_teleport_default_basis_gates(self):
        """Test teleport circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_algorithms.teleport_circuit()
        targets = ref_algorithms.teleport_counts(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots)
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmAlgorithmTestsWaltzBasis:
    """QasmSimulator algorithm tests in the Waltz u1,u2,u3,cx basis"""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test algorithms
    # ---------------------------------------------------------------------
    def test_grovers_waltz_basis_gates(self):
        """Test grovers gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_algorithms.grovers_circuit(
            final_measure=True, allow_sampling=True)
        targets = ref_algorithms.grovers_counts(shots)

        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_teleport_waltz_basis_gates(self):
        """Test teleport gate circuits compiling to u1,u2,u3,cx"""
        shots = 4000
        circuits = ref_algorithms.teleport_circuit()
        targets = ref_algorithms.teleport_counts(shots)
        job = execute(
            circuits,
            self.SIMULATOR,
            shots=shots,
            basis_gates=['u1', 'u2', 'u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmAlgorithmTestsMinimalBasis:
    """QasmSimulator algorithm tests in the minimal U,CX basis"""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test algorithms
    # ---------------------------------------------------------------------
    def test_grovers_minimal_basis_gates(self):
        """Test grovers circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_algorithms.grovers_circuit(
            final_measure=True, allow_sampling=True)
        targets = ref_algorithms.grovers_counts(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_teleport_minimal_basis_gates(self):
        """Test teleport gate circuits compiling to u3,cx"""
        shots = 4000
        circuits = ref_algorithms.teleport_circuit()
        targets = ref_algorithms.teleport_counts(shots)
        job = execute(
            circuits, self.SIMULATOR, shots=shots, basis_gates=['u3', 'cx'])
        result = job.result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
