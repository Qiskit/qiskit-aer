# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import test.terra.common as common
import unittest

from qiskit.backends import BaseBackend
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_addon_qv import AerQvSimulator


class TestCliffordCircuits(common.QiskitAerTestCase):
    """Integration tests for Clifford circuits."""
    def setUp(self, backend=None):
        if backend is None:
            self.qv_backend = AerQvSimulator()
        elif isinstance(backend, BaseBackend):
            self.qv_backend = backend

    def test_h_gate(self):
        """Test the result of H gate circuits."""
        shots = 1000
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)

        # H
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x1': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # HH=I
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.barrier(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

    def test_x_gate(self):
        """Test the result of X gate circuits."""
        shots = 1000
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)

        # X
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)
        circuit.measure(qr, cr)
        target = {'0x1': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # XX = I
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)
        circuit.barrier(qr)
        circuit.x(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # HXH=Z
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.x(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

    def test_z_gate(self):
        """Test the result of Z gate circuits."""
        shots = 1000
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)

        # Z alone
        circuit = QuantumCircuit(qr, cr)
        circuit.z(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # HZH = X
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.z(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x1': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # HZZH = I
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.z(qr)
        circuit.barrier(qr)
        circuit.z(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

    def test_y_gate(self):
        """Test the result of Y gate circuits."""
        shots = 1000
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)

        # Y alone
        circuit = QuantumCircuit(qr, cr)
        circuit.y(qr)
        circuit.measure(qr, cr)
        target = {'0x1': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # YY = I
        circuit = QuantumCircuit(qr, cr)
        circuit.y(qr)
        circuit.barrier(qr)
        circuit.y(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # HYH = -Y
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.y(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x1': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

    def test_s_gate(self):
        """Test the result of S gate circuits."""
        shots = 1000
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)

        # S alone
        circuit = QuantumCircuit(qr, cr)
        circuit.s(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # HSSH = HZH = X
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.s(qr)
        circuit.barrier(qr)
        circuit.s(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x1': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # SH
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.s(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x1': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # HSH
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.s(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x1': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

    def test_sdg_gate(self):
        """Test the result of S^dagger gate circuits."""
        shots = 1000
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)

        # Sdg alone
        circuit = QuantumCircuit(qr, cr)
        circuit.sdg(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # HSdgSdgH = HZH = X
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.sdg(qr)
        circuit.barrier(qr)
        circuit.sdg(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x1': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # SdgH
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.sdg(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x1': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # HSdgH
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.sdg(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x1': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # HSdgSH
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.s(qr)
        circuit.barrier(qr)
        circuit.sdg(qr)
        circuit.h(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

    def test_cx_gate(self):
        """Test the result of CX gate circuits."""
        shots = 1000
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)

        # 00 state
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.measure(qr, cr)
        target = {'0x0': shots}  # {"00": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 00 state
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[0])
        circuit.measure(qr, cr)
        target = {'0x0': shots}  # {"00": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 10 state
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.measure(qr, cr)
        target = {'0x2': shots}  # {"10": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 01 state
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.cx(qr[1], qr[0])
        circuit.measure(qr, cr)
        target = {'0x1': shots}  # {"01": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 11 state
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.measure(qr, cr)
        target = {'0x3': shots}  # {"11": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 11 state
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.measure(qr, cr)
        target = {'0x3': shots}  # {"11": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 01 state
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)
        circuit.cx(qr[0], qr[1])
        circuit.measure(qr, cr)
        target = {'0x1': shots}  # {"11": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 10 state
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr)
        circuit.cx(qr[1], qr[0])
        circuit.measure(qr, cr)
        target = {'0x2': shots}  # {"11": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # Bell state
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x3': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # Bell state
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x3': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

    def test_cz_gate(self):
        """Test the result of CZ gate circuits."""
        shots = 1000
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)

        # 00 state
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cz(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.measure(qr, cr)
        target = {'0x0': shots}  # {"00": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 00 state
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[1])
        circuit.cz(qr[1], qr[0])
        circuit.h(qr[1])
        circuit.measure(qr, cr)
        target = {'0x0': shots}  # {"00": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 11 state
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[1])
        circuit.h(qr[0])
        circuit.cz(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.measure(qr, cr)
        target = {'0x3': shots}  # {"11": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # 11 state
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.h(qr[1])
        circuit.cz(qr[0], qr[1])
        circuit.h(qr[1])
        circuit.measure(qr, cr)
        target = {'0x3': shots}  # {"11": shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

        # Bell state
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.cz(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x3': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # Bell state
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.cz(qr[0], qr[1])
        circuit.h(qr[1])
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x3': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

    def test_reset(self):
        """Test the result of reset in circuits."""
        shots = 1000
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)

        # Make a bell state than reset qubit 0
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.reset(qr[0])
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x2': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # Make a bell state than reset qubit 1
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.reset(qr[1])
        circuit.measure(qr, cr)
        target = {'0x0': shots / 2, '0x1': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # Make a bell state than reset both qubits
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.reset(qr)
        circuit.measure(qr, cr)
        target = {'0x0': shots}
        self.compare_circuit_counts(circuit, target, shots, threshold=0)

    def test_swap_gate(self):
        """Test the result of SWAP gate circuits."""
        shots = 1000
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        # Set initial state as |+01>
        # Permutation (0,1,2) -> (1,0,2)
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.x(qr[2])
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr, cr)
        # target = {'100': shots / 2, '110': shots / 2}
        target = {'0x4': shots / 2, '0x6': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # Permutation (0,1,2) -> (0,2,1)
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.x(qr[2])
        circuit.swap(qr[1], qr[2])
        circuit.measure(qr, cr)
        # target = {'010': shots / 2, '011': shots / 2}
        target = {'0x2': shots / 2, '0x3': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # Permutation (0,1,2) -> (2,1,0)
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.x(qr[2])
        circuit.swap(qr[0], qr[2])
        circuit.measure(qr, cr)
        # target = {'001': shots / 2, '101': shots / 2}
        target = {'0x1': shots / 2, '0x5': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # Permutation (0,1,2) -> (2,1,0)
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.x(qr[2])
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.swap(qr[0], qr[1])
        circuit.measure(qr, cr)
        # target = {'001': shots / 2, '101': shots / 2}
        target = {'0x1': shots / 2, '0x5': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)

        # Permutation (0,1,2) -> (2,0,1)
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.x(qr[2])
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[2], qr[0])
        circuit.measure(qr, cr)
        # target = {'001': shots / 2, '101': shots / 2}
        target = {'0x1': shots / 2, '0x3': shots / 2}
        self.compare_circuit_counts(circuit, target, shots)


if __name__ == '__main__':
    unittest.main()
