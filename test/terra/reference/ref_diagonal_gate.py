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
Test circuits and reference outputs for diagonal instruction.
"""


import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Backwards compatibility for Terra <= 0.13
if not hasattr(QuantumCircuit, 'diagonal'):
    QuantumCircuit.diagonal = QuantumCircuit.diag_gate


def diagonal_gate_circuits_deterministic(final_measure=True):
    """Diagonal gate test circuits with deterministic count output."""

    circuits = []
    qr = QuantumRegister(2, 'qr')
    if final_measure:
        cr = ClassicalRegister(2, 'cr')
        regs = (qr, cr)
    else:
        regs = (qr, )

    # Swap |00> <--> |01> states
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.diagonal([1, -1], [0])
    circuit.h(0)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Swap |00> <--> |10> states
    circuit = QuantumCircuit(*regs)
    circuit.h(1)
    circuit.diagonal([1, -1], [1])
    circuit.h(1)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Swap |00> <--> |11> states
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.diagonal([1, -1, -1, 1], qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CS01.XX, 1j|11> state
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.diagonal([1, 1, 1, 1j], qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def diagonal_gate_counts_deterministic(shots, hex_counts=True):
    """Diagonal gate circuits reference counts."""
    targets = []
    if hex_counts:
        # Swap |00> <--> |01> states
        targets.append({'0x1': shots})
        # Swap |00> <--> |10> states
        targets.append({'0x2': shots})
        # Swap |00> <--> |11> states
        targets.append({'0x3': shots})
        # CS01.XX, 1j|11> state
        targets.append({'0x3': shots})
    else:
        # Swap |00> <--> |01> states
        targets.append({'01': shots})
        # Swap |00> <--> |10> states
        targets.append({'10': shots})
        # Swap |00> <--> |11> states
        targets.append({'11': shots})
        # CS01.XX, 1j|11> state
        targets.append({'11': shots})
    return targets


def diagonal_gate_statevector_deterministic():
    """Diagonal gate test circuits with deterministic counts."""
    targets = []
    # Swap |00> <--> |01> states
    targets.append(np.array([0, 1, 0, 0]))
    # Swap |00> <--> |10> states
    targets.append(np.array([0, 0, 1, 0]))
    # Swap |00> <--> |11> states
    targets.append(np.array([0, 0, 0, 1]))
    # CS01.XX, 1j|11> state
    targets.append(np.array([0, 0, 0, 1j]))
    return targets


def diagonal_gate_unitary_deterministic():
    """Diagonal gate circuits reference unitaries."""
    targets = []

    # Swap |00> <--> |01> states
    targets.append(np.array([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]]))
    # Swap |00> <--> |10> states
    targets.append(np.array([[0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [1, 0, 0, 0],
                             [0, 1, 0, 0]]))
    # Swap |00> <--> |11> states
    targets.append(np.array([[0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0]]))
    # CS01.XX, 1j|11> state
    targets.append(np.array([[0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [1j, 0, 0, 0]]))
    return targets
