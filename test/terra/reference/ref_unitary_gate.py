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
Test circuits and reference outputs for measure instruction.
"""


import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


def unitary_gate_circuits_deterministic(final_measure=True):
    """Unitary gate test circuits with deterministic count output."""

    circuits = []

    # 2-qubit circuits
    qr = QuantumRegister(2, 'qr')
    if final_measure:
        cr = ClassicalRegister(2, 'cr')
        regs = (qr, cr)
    else:
        regs = (qr, )

    y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
    cx_mat = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                      dtype=complex)

    # CX01, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.unitary(cx_mat, [0, 1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.unitary(cx_mat, [1, 0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(Y^I), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.unitary(y_mat, [1])
    circuit.unitary(cx_mat, [0, 1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(I^Y), |01> state
    circuit = QuantumCircuit(*regs)
    circuit.unitary(y_mat, [0])
    circuit.unitary(cx_mat, [1, 0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX01.(I^Y), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.unitary(y_mat, [0])
    circuit.unitary(cx_mat, [0, 1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CX10.(Y^I), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.unitary(y_mat, [1])
    circuit.unitary(cx_mat, [1, 0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # 4-qubit circuits
    qr = QuantumRegister(4, 'qr')
    if final_measure:
        cr = ClassicalRegister(4, 'cr')
        regs = (qr, cr)
    else:
        regs = (qr, )
    x_mat = np.eye(2, dtype=complex)[::-1]
    x2_mat = np.eye(4, dtype=complex)[::-1]
    x3_mat = np.eye(8, dtype=complex)[::-1]
    x4_mat = np.eye(16, dtype=complex)[::-1]

    # prepare |1111> with 1-qubit unitary
    circuit = QuantumCircuit(*regs)
    circuit.unitary(x_mat, [0])
    circuit.unitary(x_mat, [1])
    circuit.unitary(x_mat, [2])
    circuit.unitary(x_mat, [3])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # prepare |1111> with 2-qubit unitary
    circuit = QuantumCircuit(*regs)
    circuit.unitary(x2_mat, [0, 1])
    circuit.unitary(x2_mat, [2, 3])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # prepare |1111> with 3-qubit unitary
    circuit = QuantumCircuit(*regs)
    circuit.unitary(x3_mat, [0, 1, 2])
    circuit.unitary(x_mat, [3])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # prepare |1111> with 4-qubit unitary
    circuit = QuantumCircuit(*regs)
    circuit.unitary(x4_mat, [0, 1, 2, 3])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def unitary_gate_counts_deterministic(shots, hex_counts=True):
    """Unitary gate circuits reference counts."""
    targets = []
    if hex_counts:
        # CX01, |00> state
        targets.append({'0x0': shots})
        # CX10, |00> state
        targets.append({'0x0': shots})
        # CX01.(Y^I), |10> state
        targets.append({'0x2': shots})
        # CX10.(I^Y), |01> state
        targets.append({'0x1': shots})
        # CX01.(I^Y), |11> state
        targets.append({'0x3': shots})
        # CX10.(Y^I), |11> state
        targets.append({'0x3': shots})
        # X^X^X^X, |1111> state
        targets.append({'0xf': shots})
        # X^X^X^X, |1111> state
        targets.append({'0xf': shots})
        # X^X^X^X, |1111> state
        targets.append({'0xf': shots})
        # X^X^X^X, |1111> state
        targets.append({'0xf': shots})
    else:
        # CX01, |00> state
        targets.append({'00': shots})
        # CX10, |00> state
        targets.append({'00': shots})
        # CX01.(Y^I), |10> state
        targets.append({'10': shots})
        # CX10.(I^Y), |01> state
        targets.append({'01': shots})
        # CX01.(I^Y), |11> state
        targets.append({'11': shots})
        # X^X^X^X, |1111> state
        targets.append({'1111': shots})
        # X^X^X^X, |1111> state
        targets.append({'1111': shots})
        # X^X^X^X, |1111> state
        targets.append({'1111': shots})
        # X^X^X^X, |1111> state
        targets.append({'1111': shots})
    return targets


def unitary_gate_statevector_deterministic():
    """Unitary gate test circuits with deterministic counts."""
    targets = []
    # CX01, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX10, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX01.(Y^I), |10> state
    targets.append(np.array([0, 0, 1j, 0]))
    # CX10.(I^Y), |01> state
    targets.append(np.array([0, 1j, 0, 0]))
    # CX01.(I^Y), |11> state
    targets.append(np.array([0, 0, 0, 1j]))
    # CX10.(Y^I), |11> state
    targets.append(np.array([0, 0, 0, 1j]))
    # X^X^X^X, |1111> state
    psi_1111 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=complex)
    targets.append(psi_1111)
    # X^X^X^X, |1111> state
    targets.append(psi_1111)
    # X^X^X^X, |1111> state
    targets.append(psi_1111)
    # X^X^X^X, |1111> state
    targets.append(psi_1111)
    return targets


def unitary_gate_unitary_deterministic():
    """Unitary gate circuits reference unitaries."""
    targets = []
    # CX01, |00> state
    targets.append(np.array([[1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0]]))
    # CX10, |00> state
    targets.append(np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]]))
    # CX01.(Y^I), |10> state
    targets.append(np.array([[0, 0, -1j, 0],
                             [0, 1j, 0, 0],
                             [1j, 0, 0, 0],
                             [0, 0, 0, -1j]]))
    # CX10.(I^Y), |01> state
    targets.append(np.array([[0, -1j, 0, 0],
                             [1j, 0, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 0, 0, -1j]]))
    # CX01.(I^Y), |11> state
    targets.append(np.array([[0, -1j, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 0, 0, -1j],
                             [1j, 0, 0, 0]]))
    # CX10.(Y^I), |11> state
    targets.append(np.array([[0, 0, -1j, 0],
                             [0, 0, 0, -1j],
                             [0, 1j, 0, 0],
                             [1j, 0, 0, 0]]))
    # X^X^X^X, |1111> state
    x4_mat = np.eye(16, dtype=complex)[::-1]
    targets.append(x4_mat)
    # X^X^X^X, |1111> state
    targets.append(x4_mat)
    # X^X^X^X, |1111> state
    targets.append(x4_mat)
    # X^X^X^X, |1111> state
    targets.append(x4_mat)
    return targets
