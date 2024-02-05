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
from test.terra.reference import ref_conditionals
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate


@ddt
class TestConditionalGates(SimulatorTestCase):
    """AerSimulator conditional tests."""

    SUPPORTED_METHODS = [
        "automatic",
        "stabilizer",
        "statevector",
        "density_matrix",
        "matrix_product_state",
        "extended_stabilizer",
    ]

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_gates_1bit(self, method, device):
        """Test conditional gate operations on 1-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type="gate"
        )
        targets = ref_conditionals.conditional_counts_1bit(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_gates_2bit(self, method, device):
        """Test conditional gate operations on 2-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type="gate"
        )
        targets = ref_conditionals.conditional_counts_2bit(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_gates_64bit(self, method, device):
        """Test conditional gate operations on 64-bit conditional register."""
        shots = 100
        # [value of conditional register, list of condtional values]
        cases = ref_conditionals.conditional_cases_64bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            64, cases, final_measure=True, conditional_type="gate"
        )
        # not using hex counts because number of leading zeros in results
        # doesn't seem consistent
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots, hex_counts=False)

        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_gates_132bit(self, method, device):
        """Test conditional gate operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            132, cases, final_measure=True, conditional_type="gate"
        )
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots, hex_counts=False)
        circuits = circuits[0:1]
        targets = targets[0:1]
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)


@ddt
class TestConditionalUnitary(SimulatorTestCase):
    """AerSimulator conditional unitary tests."""

    SUPPORTED_METHODS = [
        "automatic",
        "statevector",
        "density_matrix",
        "matrix_product_state",
    ]

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_unitary_1bit(self, method, device):
        """Test conditional unitary operations on 1-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type="unitary"
        )
        targets = ref_conditionals.conditional_counts_1bit(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_unitary_2bit(self, method, device):
        """Test conditional unitary operations on 2-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type="unitary"
        )
        targets = ref_conditionals.conditional_counts_2bit(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_unitary_64bit(self, method, device):
        """Test conditional unitary operations on 64-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_64bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            64, cases, final_measure=True, conditional_type="unitary"
        )
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots, hex_counts=False)

        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_unitary_132bit(self, method, device):
        """Test conditional unitary operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            132, cases, final_measure=True, conditional_type="unitary"
        )
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots, hex_counts=False)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)


@ddt
class TestConditionalKraus(SimulatorTestCase):
    """AerSimulator conditional kraus tests."""

    SUPPORTED_METHODS = [
        "automatic",
        "statevector",
        "density_matrix",
        "matrix_product_state",
    ]

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_unitary_1bit(self, method, device):
        """Test conditional kraus operations on 1-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type="kraus"
        )
        targets = ref_conditionals.conditional_counts_1bit(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_kraus_2bit(self, method, device):
        """Test conditional kraus operations on 2-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type="kraus"
        )
        targets = ref_conditionals.conditional_counts_2bit(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_kraus_64bit(self, method, device):
        """Test conditional kraus operations on 64-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_64bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            64, cases, final_measure=True, conditional_type="kraus"
        )
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots, hex_counts=False)

        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_kraus_132bit(self, method, device):
        """Test conditional kraus operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            132, cases, final_measure=True, conditional_type="kraus"
        )
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots, hex_counts=False)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)


@ddt
class TestConditionalSuperOp(SimulatorTestCase):
    """AerSimulator conditional superop tests."""

    SUPPORTED_METHODS = ["automatic", "density_matrix"]

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_superop_1bit(self, method, device):
        """Test conditional superop operations on 1-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type="superop"
        )
        targets = ref_conditionals.conditional_counts_1bit(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_superop_2bit(self, method, device):
        """Test conditional superop operations on 2-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_2bit(
            final_measure=True, conditional_type="superop"
        )
        targets = ref_conditionals.conditional_counts_2bit(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_superop_64bit(self, method, device):
        """Test conditional superop operations on 64-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_64bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            64, cases, final_measure=True, conditional_type="superop"
        )
        targets = ref_conditionals.condtional_counts_nbit(64, cases, shots, hex_counts=False)

        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)

    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_superop_132bit(self, method, device):
        """Test conditional superop operations on 132-bit conditional register."""
        shots = 100
        cases = ref_conditionals.conditional_cases_132bit()
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)
        circuits = ref_conditionals.conditional_circuits_nbit(
            132, cases, final_measure=True, conditional_type="superop"
        )
        targets = ref_conditionals.condtional_counts_nbit(132, cases, shots, hex_counts=False)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, hex_counts=False, delta=0)


@ddt
class TestConditionalReset(SimulatorTestCase):
    """AerSimulator conditional reset tests."""

    SUPPORTED_METHODS = [
        "automatic",
        "statevector",
        "density_matrix",
        "matrix_product_state",
        "tensor_network",
    ]

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_conditional_reset_1bit(self, method, device):
        """Test conditional reset on 1-bit conditional register."""
        shots = 100
        backend = self.backend(method=method, device=device)
        backend.set_options(max_parallel_experiments=0)

        circuits = ref_conditionals.conditional_circuits_1bit(
            final_measure=True, conditional_type="reset"
        )
        targets = ref_conditionals.conditional_counts_1bit_with_reset(shots)
        result = backend.run(circuits, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)


@ddt
class TestConditionalDiagonal(SimulatorTestCase):
    """AerSimulator conditional diagonal tests."""

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    def test_conditional_diagonal(self):
        """Test conditional diagonal with statevector."""
        shots = 100
        backend = self.backend(method="statevector", device="CPU")
        backend.set_options(max_parallel_experiments=0)

        circuit = QuantumCircuit(4, 4)
        for i in range(1, 4):
            circuit.h(i)
        circuit.save_statevector(label="base")

        circuit0 = QuantumCircuit(4, 4)
        for i in range(1, 4):
            circuit0.h(i)
        circuit0.append(DiagonalGate([-1, -1]), [1]).c_if(circuit0.clbits[0], 0)
        circuit0.save_statevector(label="diff")

        circuit1 = QuantumCircuit(4, 4)
        for i in range(1, 4):
            circuit1.h(i)
        circuit1.append(DiagonalGate([-1, -1]), [1]).c_if(circuit1.clbits[0], 1)
        circuit1.save_statevector(label="equal")

        result = backend.run([circuit, circuit0, circuit1], shots=1).result()
        self.assertSuccess(result)

        self.assertNotEqual(result.data(circuit)["base"], result.data(circuit0)["diff"])
        self.assertEqual(result.data(circuit)["base"], result.data(circuit1)["equal"])
