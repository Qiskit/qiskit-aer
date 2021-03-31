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
QasmSimulator Integration Tests for set state instructions
"""

from ddt import ddt
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile
from test.terra.backends.aer_simulator.aer_simulator_test_case import (
    AerSimulatorTestCase, supported_methods)


@ddt
class TestSetState(AerSimulatorTestCase):
    """Test for set state instructions"""
    @supported_methods(['automatic', 'stabilizer'], [1, 2, 3, 4])
    def test_set_stabilizer(self, method, device, num_qubits):
        """Test SetStabilizer instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = 'state'

        target = qi.random_clifford(num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_stabilizer(target)
        circ.save_stabilizer(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0),
                             shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = qi.Clifford.from_dict(simdata[label])
        self.assertEqual(value, target)

    @supported_methods(['automatic', 'statevector'], [1, 2, 3, 4, 5])
    def test_set_statevector(self, method, device, num_qubits):
        """Test SetStatevector for instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = 'state'

        target = qi.random_statevector(2**num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_statevector(target)
        circ.save_statevector(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0),
                             shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = qi.Statevector(simdata[label])
        self.assertEqual(value, target)

    @supported_methods(['automatic', 'density_matrix'], [1, 2, 3])
    def test_set_density_matrix(self, method, device, num_qubits):
        """Test SetDensityMatrix instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = 'state'

        target = qi.random_density_matrix(2**num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_density_matrix(target)
        circ.save_density_matrix(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0),
                             shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = qi.DensityMatrix(simdata[label])
        self.assertEqual(value, target)

    @supported_methods(['automatic', 'unitary'], [1, 2, 3])
    def test_set_unitary(self, method, device, num_qubits):
        """Test SetUnitary instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = 'state'

        target = qi.random_unitary(2**num_qubits, seed=seed)

        circ = QuantumCircuit(num_qubits)
        circ.set_unitary(target)
        circ.save_unitary(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0),
                             shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = qi.Operator(simdata[label])
        self.assertEqual(value, target)

    @supported_methods(['automatic', 'superop'], [1, 2])
    def test_set_superop(self, method, device, num_qubits):
        """Test SetSuperOp instruction"""
        backend = self.backend(method=method, device=device)

        seed = 100
        label = 'state'

        target = qi.SuperOp(qi.random_quantum_channel(2**num_qubits,
                                                      seed=seed))

        circ = QuantumCircuit(num_qubits)
        circ.set_superop(target)
        circ.save_superop(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0),
                             shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = qi.SuperOp(simdata[label])
        self.assertEqual(value, target)
