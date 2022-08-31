# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
LocalNoisePass class tests
"""

from qiskit_aer.noise import ReadoutError, LocalNoisePass

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import SXGate, HGate, CXGate
from qiskit.transpiler import TranspilerError
from test.terra.common import QiskitAerTestCase


class TestLocalNoisePass(QiskitAerTestCase):
    """Testing LocalNoisePass class"""

    def test_append_noise(self):
        qc = QuantumCircuit(2)
        qc.sx(0)
        qc.sx(1)

        def func(op, qubits):
            if tuple(qubits) == (1,):
                return HGate()
            return None

        noise_pass = LocalNoisePass(func=func, op_types=SXGate, method="append")
        actual = noise_pass(qc)

        expected = QuantumCircuit(2)
        expected.sx(0)  # do nothing for sx(0)
        expected.sx(1)
        expected.h(1)  # add H after sx(1)

        self.assertEqual(expected, actual)

    def test_prepend_noise(self):
        qc = QuantumCircuit(2)
        qc.sx(0)
        qc.sx(1)

        def func(op, qubits):
            # return H for qubit 1 and None for the other qubits
            if tuple(qubits) == (1,):
                return HGate()
            return None

        noise_pass = LocalNoisePass(func=func, op_types=SXGate, method="prepend")
        actual = noise_pass(qc)

        expected = QuantumCircuit(2)
        expected.sx(0)  # do nothing for sx(0)
        expected.h(1)  # add H before sx(1)
        expected.sx(1)

        self.assertEqual(expected, actual)

    def test_replace_noise(self):
        qc = QuantumCircuit(2)
        qc.sx(0)
        qc.sx(1)

        def func(op, qubits):
            if tuple(qubits) == (1,):
                return HGate()
            return None

        noise_pass = LocalNoisePass(func=func, op_types=SXGate, method="replace")
        actual = noise_pass(qc)

        expected = QuantumCircuit(2)
        # sx(0) is removed since func returns None for sx(0)
        expected.h(1)  # sx(1) is replaced with h(1)

        self.assertEqual(expected, actual)

    def test_append_readout_error(self):
        qc = QuantumCircuit(1)
        qc.h(0)

        def out_readout_error(op, qubits):
            return ReadoutError([[1, 0], [0, 1]])

        noise_pass = LocalNoisePass(func=out_readout_error, op_types=HGate)
        with self.assertRaises(TranspilerError):
            noise_pass(qc)

    def test_append_circuit(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        def composite_error(op, qubits):
            circ = QuantumCircuit(2)
            for q in qubits:
                circ.x(q)
            return circ

        noise_pass = LocalNoisePass(func=composite_error, op_types=CXGate)
        noise_qc = noise_pass(qc)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.x(0)
        expected.x(1)

        self.assertEqual(expected, noise_qc)
