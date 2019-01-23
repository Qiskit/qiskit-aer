# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

from test.terra.utils import common
from test.terra.utils import ref_algorithms
from qiskit import compile
from qiskit.providers.aer import QasmSimulator


class QasmAlgorithmTests(common.QiskitAerTestCase):
    """QasmSimulator algorithm tests in the default basis"""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test algorithms
    # ---------------------------------------------------------------------
    def test_grovers_default_basis_gates(self):
        """Test grovers circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_algorithms.grovers_circuit(final_measure=True,
                                                  allow_sampling=True)
        targets = ref_algorithms.grovers_counts(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_teleport_default_basis_gates(self):
        """Test teleport circuits compiling to backend default basis_gates."""
        shots = 2000
        circuits = ref_algorithms.teleport_circuit()
        targets = ref_algorithms.teleport_counts(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmAlgorithmTestsWaltzBasis(common.QiskitAerTestCase):
    """QasmSimulator algorithm tests in the Waltz u1,u2,u3,cx basis"""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test algorithms
    # ---------------------------------------------------------------------
    def test_grovers_waltz_basis_gates(self):
        """Test grovers gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_algorithms.grovers_circuit(final_measure=True,
                                                  allow_sampling=True)
        targets = ref_algorithms.grovers_counts(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_teleport_waltz_basis_gates(self):
        """Test teleport gate circuits compiling to u1,u2,u3,cx"""
        shots = 2000
        circuits = ref_algorithms.teleport_circuit()
        targets = ref_algorithms.teleport_counts(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='u1,u2,u3,cx')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)


class QasmAlgorithmTestsMinimalBasis(common.QiskitAerTestCase):
    """QasmSimulator algorithm tests in the minimal U,CX basis"""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test algorithms
    # ---------------------------------------------------------------------
    def test_grovers_minimal_basis_gates(self):
        """Test grovers circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_algorithms.grovers_circuit(final_measure=True,
                                                  allow_sampling=True)
        targets = ref_algorithms.grovers_counts(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_teleport_minimal_basis_gates(self):
        """Test teleport gate circuits compiling to U,CX"""
        shots = 2000
        circuits = ref_algorithms.teleport_circuit()
        targets = ref_algorithms.teleport_counts(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots, basis_gates='U,CX')
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
