# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests for set state instructions
"""

from ddt import ddt
import numpy as np
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


@ddt
class TestSetState(SimulatorTestCase):
    """Test for set state instructions"""

    @supported_methods(["automatic", "stabilizer"], [1, 2, 3])
    def test_set_stabilizer_stabilizer_state(self, method, device, num_qubits):
        """Test SetStabilizer instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = "state"

        target = qi.StabilizerState(qi.random_clifford(num_qubits, seed=seed))

        circ = QuantumCircuit(num_qubits)
        circ.set_stabilizer(target)
        circ.save_stabilizer(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertEqual(value, target)

    @supported_methods(["automatic", "stabilizer"], [1, 2, 3])
    def test_set_stabilizer_clifford(self, method, device, num_qubits):
        """Test SetStabilizer instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = "state"

        target = qi.random_clifford(num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_stabilizer(target)
        circ.save_clifford(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertEqual(value, target)

    @supported_methods(["automatic", "statevector", "tensor_network"], [1, 2, 3])
    def test_set_statevector(self, method, device, num_qubits):
        """Test SetStatevector for instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = "state"

        target = qi.random_statevector(2**num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_statevector(target)
        circ.save_statevector(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertEqual(value, target)

    @supported_methods(["automatic", "density_matrix", "tensor_network"], [1, 2, 3])
    def test_set_density_matrix(self, method, device, num_qubits):
        """Test SetDensityMatrix instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = "state"

        target = qi.random_density_matrix(2**num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_density_matrix(target)
        circ.save_density_matrix(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertEqual(value, target)

    @supported_methods(["automatic", "unitary"], [1, 2, 3])
    def test_set_unitary(self, method, device, num_qubits):
        """Test SetUnitary instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = "state"

        target = qi.random_unitary(2**num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_unitary(target)
        circ.save_unitary(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertEqual(value, target)

    @supported_methods(["automatic", "superop"], [1, 2])
    def test_set_superop(self, method, device, num_qubits):
        """Test SetSuperOp instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = "state"

        target = qi.SuperOp(qi.random_quantum_channel(2**num_qubits, seed=seed))

        circ = QuantumCircuit(num_qubits)
        circ.set_superop(target)
        circ.save_superop(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = qi.SuperOp(simdata[label])
        self.assertEqual(value, target)


@ddt
class TestSetMPS(SimulatorTestCase):
    """Test for set_mps instruction"""

    @supported_methods(["automatic", "matrix_product_state"])
    def test_set_matrix_product_state(self, method, device):
        backend = self.backend(method=method, device=device)
        tests = []
        shots = 1000
        # circuit1 - |11>
        num_qubits = 2
        circ1 = QuantumCircuit(num_qubits)
        state1 = ([([[0]], [[1]]), ([[0]], [[1]])], [[1]])
        target1 = {"0x3": shots}
        tests.append((circ1, state1, target1))

        # circuit2 - |000>+|111>
        num_qubits = 3
        circ2 = QuantumCircuit(num_qubits)

        state2 = (
            [
                ([[1, 0]], [[0, 1]]),
                ([[np.sqrt(2), 0], [0, 0]], [[0, 0], [0, np.sqrt(2)]]),
                ([[1.0 - 0.0j], [0.0 - 0.0j]], [[0.0 - 0.0j], [1.0 - 0.0j]]),
            ],
            [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]],
        )
        target2 = {"0x0": shots / 2, "0x7": shots / 2}

        tests.append((circ2, state2, target2))
        for circ, state, target in tests:
            circ.set_matrix_product_state(state)
            circ.measure_all()

            # Run
            result = backend.run(
                transpile(circ, backend, optimization_level=0), shots=shots
            ).result()
            self.compare_counts(result, [circ], [target], delta=0.1 * shots)
