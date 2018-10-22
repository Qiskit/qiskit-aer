# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import test.terra.common as common
import unittest
import numpy as np
from numpy.linalg import norm

from qiskit.backends import BaseBackend
from qiskit import QuantumRegister, QuantumCircuit, compile
from qiskit_aer.backends import UnitarySimulator


class TestUnitaryState(common.QiskitAerTestCase):
    """Integration tests for Unitary simulator backend."""
    def setUp(self, backend=None):
        if backend is None:
            self.backend = UnitarySimulator()
        elif isinstance(backend, BaseBackend):
            self.backend = backend

        # Single qubit unitaries
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self.S = np.array([[1, 0], [0, 1j]])
        self.T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

        # 2-qubit unitaries
        self.CX01 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        self.CX10 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        self.CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        self.SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        # 3-qubit unitary
        self.CCX012 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0]])
        self.CCX210 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 1, 0]])

    def execute_unitary(self, circuit, seed=None, basis_gates=None):
        """Execute a circuit and return unitary."""
        qobj = compile(circuit, backend=self.backend, shots=1, seed=seed,
                       basis_gates=basis_gates)
        job = self.backend.run(qobj)
        result = job.result()
        return result.get_unitary(circuit)

    def test_h_gate(self):
        """Test the result of H gate circuits."""
        qr = QuantumRegister(1)
        # H direct
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        target = self.H
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

    def test_x_gate(self):
        """Test the result of X gate circuits."""
        qr = QuantumRegister(1)
        # X
        circuit = QuantumCircuit(qr)
        circuit.x(qr)
        target = self.X
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # HXH=Z
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.x(qr)
        circuit.h(qr)
        target = self.Z
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

    def test_z_gate(self):
        """Test the result of Z gate circuits."""
        qr = QuantumRegister(1)

        # Z alone
        circuit = QuantumCircuit(qr)
        circuit.z(qr)
        target = self.Z
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # HZH = X
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.z(qr)
        circuit.h(qr)
        target = self.X
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

    def test_y_gate(self):
        """Test the result of Y gate circuits."""
        qr = QuantumRegister(1)
        # Y alone
        circuit = QuantumCircuit(qr)
        circuit.y(qr)
        target = self.Y
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # HYH = -Y
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.y(qr)
        circuit.h(qr)
        target = -1 * self.Y
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

    def test_s_gate(self):
        """Test the result of S gate circuits."""
        qr = QuantumRegister(1)

        # S alone
        circuit = QuantumCircuit(qr)
        circuit.s(qr)
        target = self.S
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # HSSH = HZH = X
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.s(qr)
        circuit.barrier(qr)
        circuit.s(qr)
        circuit.h(qr)
        target = self.X
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # HSH
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.s(qr)
        circuit.h(qr)
        target = np.dot(self.H, np.dot(self.S, self.H))
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

    def test_sdg_gate(self):
        """Test the result of S^dagger gate circuits."""
        qr = QuantumRegister(1)

        # Sdg alone
        circuit = QuantumCircuit(qr)
        circuit.sdg(qr)
        target = np.conj(self.S)
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # HSdgSdgH = HZH = X
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.sdg(qr)
        circuit.barrier(qr)
        circuit.sdg(qr)
        circuit.h(qr)
        target = self.X
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # HSdgH
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.sdg(qr)
        circuit.h(qr)
        target = np.dot(self.H, np.dot(np.conj(self.S), self.H))
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

    def test_cx_gate(self):
        """Test the result of CX gate circuits."""
        qr = QuantumRegister(2)

        # CX(0, 1)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        target = self.CX01
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)

        # CX(1, 0)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        target = self.CX10
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)

    def test_cz_gate(self):
        """Test the result of CZ gate circuits."""
        qr = QuantumRegister(2)

        # CZ(0, 1)
        circuit = QuantumCircuit(qr)
        circuit.cz(qr[0], qr[1])
        target = self.CZ
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # CZ(1, 0)
        circuit = QuantumCircuit(qr)
        circuit.cz(qr[1], qr[0])
        target = self.CZ
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

    def test_swap_gate(self):
        """Test the result of SWAP gate circuits."""
        qr = QuantumRegister(2)

        # SWAP(0, 1)
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[0], qr[1])
        target = self.SWAP
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # SWAP(1, 0)
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[1], qr[0])
        target = self.SWAP
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

    def test_toffoli_gate(self):
        """Test the result of Toffoli gate."""
        qr = QuantumRegister(3)

        # CCX(0, 1, 2)
        circuit = QuantumCircuit(qr)
        circuit.ccx(qr[0], qr[1], qr[2])
        target = self.CCX012
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)

        # CCX(2, 1, 0)
        circuit = QuantumCircuit(qr)
        circuit.ccx(qr[2], qr[1], qr[0])
        target = self.CCX210
        # Direct implementaiton
        output = self.execute_unitary(circuit)
        self.assertAlmostEqual(norm(output - target), 0.0)
        # Using u1,u2,u3,cx
        output = self.execute_unitary(circuit, basis_gates='u1,u2,u3,cx')
        self.assertAlmostEqual(norm(output - target), 0.0)


if __name__ == '__main__':
    unittest.main()
