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
IdleScheduler class tests
"""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.providers.aer.noise.utils import schedule_idle_gates
import unittest

class TestIdleScheduler(unittest.TestCase):
    def test_circuits_equal(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.iden(qr[1])
        target_circuit.y(qr[2])

        result_circuit = schedule_idle_gates(circuit, op_times={'id': 2, 'x': 2})

        self.assertEqual(target_circuit, result_circuit)

    def test_small_circuit(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.iden(qr[1])
        target_circuit.y(qr[2])

        result_circuit = schedule_idle_gates(circuit)
        self.assertEqual(target_circuit, result_circuit)

    def test_small_circuit_double_x_time(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.iden(qr[1])
        target_circuit.iden(qr[1])
        target_circuit.y(qr[2])
        target_circuit.iden(qr[2])

        result_circuit = schedule_idle_gates(circuit, op_times={'x': 2})

        self.assertEqual(target_circuit, result_circuit)

    def test_small_circuit_double_id_time(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.y(qr[2])

        result_circuit = schedule_idle_gates(circuit, op_times={'id': 2})
        self.assertEqual(target_circuit, result_circuit)

    def test_small_circuit_double_id_and_x_time(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.iden(qr[1])
        target_circuit.y(qr[2])

        result_circuit = schedule_idle_gates(circuit, op_times={'id': 2, 'x':2})
        self.assertEqual(target_circuit, result_circuit)

    def test_small_circuit_double_id_and_x_time_double_length(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.x(qr[0])
        circuit.y(qr[2])
        circuit.y(qr[2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.x(qr[0])
        target_circuit.iden(qr[1])
        target_circuit.iden(qr[1])
        target_circuit.y(qr[2])
        target_circuit.y(qr[2])
        target_circuit.iden(qr[2])

        result_circuit = schedule_idle_gates(circuit, op_times={'id': 2, 'x': 2})

        self.assertEqual(target_circuit, result_circuit)

    def test_barrier_circuit(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[2])
        circuit.barrier()
        circuit.x(qr[1])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.iden(qr[1])
        target_circuit.y(qr[2])
        target_circuit.barrier()
        target_circuit.iden(qr[0])
        target_circuit.x(qr[1])
        target_circuit.iden(qr[2])

        result_circuit = schedule_idle_gates(circuit)
        self.assertEqual(target_circuit, result_circuit)

    def test_small_circuit_nondefault_time(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.iden(qr[1])
        target_circuit.y(qr[2])

        result_circuit = schedule_idle_gates(circuit, default_op_time=0.3153)
        self.assertEqual(target_circuit, result_circuit)

    def test_small_circuit_nondefault_time_and_different_x_y(self):
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.y(qr[2])

        target_circuit = QuantumCircuit(qr)
        target_circuit.x(qr[0])
        target_circuit.iden(qr[1])
        target_circuit.iden(qr[1])
        target_circuit.iden(qr[1])
        target_circuit.y(qr[2])

        result_circuit = schedule_idle_gates(circuit, default_op_time=0.3153, op_times={'x': 0.974, 'y': 0.734})
        self.assertEqual(target_circuit, result_circuit)

    def test_labels(self):
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])
        circuit.barrier()
        circuit.y(qr[0])
        circuit.barrier()
        circuit.h(qr[0])
        circuit.barrier()

        result_circuit = schedule_idle_gates(circuit)
        labels = [gate[0].label for gate in result_circuit if gate[0].name == 'id']
        target_labels = ['id_x', 'id_y', 'id_h']
        self.assertEqual(target_labels, labels)

        result_circuit = schedule_idle_gates(circuit, labels="uniform_id_label")
        labels = [gate[0].label for gate in result_circuit if gate[0].name == 'id']
        target_labels = ['uniform_id_label', 'uniform_id_label', 'uniform_id_label']
        self.assertEqual(target_labels, labels)

        result_circuit = schedule_idle_gates(circuit, labels={'x': 'id_label_for_x', 'y': 'y_id_label', 'h': '123'})
        labels = [gate[0].label for gate in result_circuit if gate[0].name == 'id']
        target_labels = ['id_label_for_x', 'y_id_label', '123']
        self.assertEqual(target_labels, labels)


