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
from qiskit.quantum_info.random import random_unitary


def unitary_gate_circuits_deterministic(final_measure=True):
    """Unitary gate test circuits with deterministic count output."""

    circuits = []

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

    return circuits


def unitary_gate_counts_deterministic(shots, hex_counts=True):
    """Unitary gate circuits reference counts."""
    targets = []
    if hex_counts:
        # CX01, |00> state
        targets.append({'0x0': shots})  # {"00": shots}
        # CX10, |00> state
        targets.append({'0x0': shots})  # {"00": shots}
        # CX01.(Y^I), |10> state
        targets.append({'0x2': shots})  # {"00": shots}
        # CX10.(I^Y), |01> state
        targets.append({'0x1': shots})  # {"00": shots}
        # CX01.(I^Y), |11> state
        targets.append({'0x3': shots})  # {"00": shots}
        # CX10.(Y^I), |11> state
        targets.append({'0x3': shots})  # {"00": shots}
    else:
        # CX01, |00> state
        targets.append({'00': shots})  # {"00": shots}
        # CX10, |00> state
        targets.append({'00': shots})  # {"00": shots}
        # CX01.(Y^I), |10> state
        targets.append({'10': shots})  # {"00": shots}
        # CX10.(I^Y), |01> state
        targets.append({'01': shots})  # {"00": shots}
        # CX01.(I^Y), |11> state
        targets.append({'11': shots})  # {"00": shots}
        # CX10.(Y^I), |11> state
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
    return targets

def unitary_random_gate_circuits_nondeterministic(final_measure=True):
    """Unitary gate test circuits with random unitary gate and nondeterministic count output."""

    circuits = []

    qr = QuantumRegister(2, 'qr')
    if final_measure:
        cr = ClassicalRegister(2, 'cr')
        regs = (qr, cr)
    else:
        regs = (qr, )

    # random 1
    circuit = QuantumCircuit(*regs)
    circuit.unitary(random_unitary(4, seed=1), [1, 0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # random 2
    circuit = QuantumCircuit(*regs)
    circuit.unitary(random_unitary(4, seed=2), [1, 0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits

def unitary_random_gate_counts_nondeterministic():
    """Unitary gate test circuits with nondeterministic counts."""
    targets = []
    # random unitary seed = 1
    targets.append({'0x0': 930, '0x1': 330, '0x2': 680, '0x3': 60})
    # random unitary seed = 2
    targets.append({'0x0': 41, '0x1': 343, '0x2': 1056, '0x3': 560})
    
    return targets
