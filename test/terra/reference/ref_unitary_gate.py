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
from qiskit.quantum_info import Statevector


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
    # random_unitary seed = nq
    circuits = []
    for n in range(1, 5):
        qr = QuantumRegister(n, 'qr')
        if final_measure:
            cr = ClassicalRegister(n, 'cr')
            regs = (qr, cr)
        else:
            regs = (qr, )

        circuit = QuantumCircuit(*regs)
        circuit.unitary(random_unitary(2 ** n, seed=n), list(range(n)))
        if final_measure:
            circuit.barrier(qr)
            circuit.measure(qr, cr)
        circuits.append(circuit)

    return circuits


def unitary_random_gate_counts_nondeterministic(shots):
    """Unitary gate test circuits with nondeterministic counts."""
    # random_unitary seed = nq
    targets = []
    for n in range(1, 5):
        unitary1 = random_unitary(2 ** n, seed=n)
        state = Statevector.from_label(n * '0').evolve(unitary1)
        state.seed(10)
        counts = state.sample_counts(shots=shots)
        hex_counts = {hex(int(key, 2)): val for key, val in counts.items()}
        targets.append(hex_counts)
    return targets
