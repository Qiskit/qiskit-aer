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
Integration Tests for SaveStatevector instruction
"""

import sys
from ddt import ddt
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods
from qiskit.qasm3 import dumps, loads


@ddt
class TestSaveStatevector(SimulatorTestCase):
    """SaveStatevector instruction tests."""

    @supported_methods(
        [
            "automatic",
            "statevector",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_save_statevector(self, method, device):
        """Test save statevector instruction"""
        backend = self.backend(method=method, device=device)

        # Stabilizer test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.sdg(0)
        circ.cx(0, 1)
        circ.cx(0, 2)

        # Target statevector
        target = qi.Statevector(circ)

        # Add save to circuit
        label = "state"
        circ.save_statevector(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertEqual(value, target)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_save_statevector_conditional(self, method, device):
        """Test conditional save statevector instruction"""

        backend = self.backend(method=method, device=device)

        # Stabilizer test circuit
        # Add save to circuit
        label = "state"
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.sdg(0)
        circ.cx(0, 1)
        circ.measure_all()
        circ.save_statevector(label=label, conditional=True)

        # Target statevector
        target = {"0x0": qi.Statevector([1, 0, 0, 0]), "0x3": qi.Statevector([0, 0, 0, -1j])}

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        for key, vec in simdata[label].items():
            self.assertIn(key, target)
            self.assertEqual(vec, target[key])

    @supported_methods(
        [
            "automatic",
            "statevector",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_save_statevector_pershot(self, method, device):
        """Test pershot save statevector instruction"""
        backend = self.backend(method=method, device=device)

        # Stabilizer test circuit
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.reset(0)
        circ.h(0)
        circ.sdg(0)

        # Target statevector
        target = qi.Statevector(circ)

        # Add save
        label = "state"
        circ.save_statevector(label=label, pershot=True)

        # Run
        shots = 10
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=shots).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertEqual(len(value), shots)
        for vec in value:
            self.assertEqual(vec, target)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_save_statevector_pershot_conditional(self, method, device):
        """Test pershot conditional save statevector instruction"""

        backend = self.backend(method=method, device=device)

        # Stabilizer test circuit
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.reset(0)
        circ.h(0)
        circ.sdg(0)

        # Target statevector
        target = qi.Statevector(circ)

        # Add save
        label = "state"
        circ.save_statevector(label=label, pershot=True, conditional=True)
        circ.measure_all()

        # Run
        shots = 10
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=shots).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertIn("0x0", value)
        self.assertEqual(len(value["0x0"]), shots)
        for vec in value["0x0"]:
            self.assertEqual(vec, target)

    @supported_methods(["statevector"])
    def test_save_statevector_cache_blocking(self, method, device):
        """Test save statevector for instruction"""
        backend = self.backend(
            method=method, device=device, blocking_qubits=2, max_parallel_threads=1
        )

        # Stabilizer test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.sdg(0)
        circ.cx(0, 1)
        circ.cx(0, 2)

        # Target statevector
        target = qi.Statevector(circ)

        # Add save to circuit
        label = "state"
        circ.save_statevector(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertEqual(value, target)

    @supported_methods(
        [
            "automatic",
            "statevector",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_save_statevector_for_qasm3_circuit(self, method, device):
        """Test save statevector instruction"""
        # qiskit_qasm3_import, which is used in qiskit.qasm3 does not support 3.7
        if sys.version_info < (3, 8):
            return

        backend = self.backend(method=method, device=device)

        # Stabilizer test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.sdg(0)
        circ.cx(0, 1)
        circ.cx(0, 2)

        # Target statevector
        target = qi.Statevector(circ)

        circ = loads(dumps(circ))

        # Add save to circuit
        label = "state"
        circ.save_statevector(label=label)

        # Run
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        self.assertEqual(value, target)
