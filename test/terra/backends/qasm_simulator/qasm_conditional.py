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

from test.terra.reference import ref_conditionals
from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator


class QasmConditionalGateTests:
    """QasmSimulator conditional tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    def test_conditional_gates_1bit(self):
        """Test conditional gate operations on 1-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type='gate')
        targets = ref_conditionals.conditional_counts_1bit(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_gates_2bit(self):
        """Test conditional gate operations on 2-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type='gate')
        targets = ref_conditionals.conditional_counts_2bit(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_gates_64bit(self):
        """Test conditional gate operations on 64-bit conditional register."""
        shots = 100
        # [value of conditional register, list of condtional values]
        cases = ref_conditionals.conditional_cases_64bit()
        circuits = ref_conditionals.conditional_circuits_nbit(64, cases,
                final_measure=True, conditional_type='gate')
        # not using hex counts because number of leading zeros in results
        # doesn't seem consistent
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots,
                hex_counts=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    def test_conditional_gates_132bit(self):
        """Test conditional gate operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        circuits = ref_conditionals.conditional_circuits_nbit(132, cases, final_measure=True,
                conditional_type='gate')
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots,
                hex_counts=False)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)


class QasmConditionalUnitaryTests:
    """QasmSimulator conditional tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    def test_conditional_unitary_1bit(self):
        """Test conditional unitary operations on 1-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type='unitary')
        targets = ref_conditionals.conditional_counts_1bit(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_unitary_2bit(self):
        """Test conditional unitary operations on 2-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type='unitary')
        targets = ref_conditionals.conditional_counts_2bit(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_unitary_64bit(self):
        """Test conditional unitary operations on 64-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_64bit()
        circuits = ref_conditionals.conditional_circuits_nbit(64, cases,
                final_measure=True, conditional_type='unitary')
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots,
                hex_counts=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    def test_conditional_unitary_132bit(self):
        """Test conditional unitary operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        circuits = ref_conditionals.conditional_circuits_nbit(132, cases, final_measure=True,
                conditional_type='unitary')
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots,
                hex_counts=False)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)


class QasmConditionalKrausTests:
    """QasmSimulator conditional tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    def test_conditional_unitary_1bit(self):
        """Test conditional kraus operations on 1-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type='kraus')
        targets = ref_conditionals.conditional_counts_1bit(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_kraus_2bit(self):
        """Test conditional kraus operations on 2-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type='kraus')
        targets = ref_conditionals.conditional_counts_2bit(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_kraus_64bit(self):
        """Test conditional kraus operations on 64-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_64bit()
        circuits = ref_conditionals.conditional_circuits_nbit(64, cases,
                final_measure=True, conditional_type='kraus')
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots,
                hex_counts=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    def test_conditional_kraus_132bit(self):
        """Test conditional kraus operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        circuits = ref_conditionals.conditional_circuits_nbit(132, cases, final_measure=True,
                conditional_type='kraus')
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots,
                hex_counts=False)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)


class QasmConditionalSuperOpTests:
    """QasmSimulator conditional tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    def test_conditional_superop_1bit(self):
        """Test conditional superop operations on 1-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type='superop')
        targets = ref_conditionals.conditional_counts_1bit(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_superop_2bit(self):
        """Test conditional superop operations on 2-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type='superop')
        targets = ref_conditionals.conditional_counts_2bit(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_superop_64bit(self):
        """Test conditional superop operations on 64-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_64bit()
        circuits = ref_conditionals.conditional_circuits_nbit(64, cases,
                final_measure=True, conditional_type='superop')
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots,
                hex_counts=False)

        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    def test_conditional_superop_132bit(self):
        """Test conditional superop operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        circuits = ref_conditionals.conditional_circuits_nbit(132, cases, final_measure=True,
                conditional_type='superop')
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots,
                hex_counts=False)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **self.BACKEND_OPTS).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)
