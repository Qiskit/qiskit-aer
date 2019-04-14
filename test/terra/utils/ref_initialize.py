# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test circuits and reference outputs for initialize instruction.
"""

from numpy import array, sqrt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


# ==========================================================================
# Deterministic output
# ==========================================================================

def initialize_circuits_deterministic(final_measure=True):
    """Initialize test circuits with deterministic count output"""

    circuits = []
    qr = QuantumRegister(3)
    if final_measure:
        cr = ClassicalRegister(3)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # Start with |+++> state
    # Initialize qr[i] to |1> for i=0,1,2
    for qubit in range(3):
        circuit = QuantumCircuit(*regs)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.h(qr[2])
        circuit.initialize([0, 1], [qr[qubit]])

        if final_measure:
            circuit.barrier(qr)
            circuit.measure(qr, cr)
        circuits.append(circuit)

    # Start with |+++> state
    # Initialize qr[i] to |1> and qr[j] to |0>
    # For [i,j] = [0,1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]
    for qubit_i in range(3):
        for qubit_j in range(3):
            if (qubit_i != qubit_j):
                circuit = QuantumCircuit(*regs)
                circuit.h(qr[0])
                circuit.h(qr[1])
                circuit.h(qr[2])
                circuit.initialize([0, 1, 0, 0], [qr[qubit_i], qr[qubit_j]])

                if final_measure:
                    circuit.barrier(qr)
                    circuit.measure(qr, cr)
                circuits.append(circuit)

    # Start with |+++> state
    # Initialize qr[i] to |1>, qr[j] to |0> and qr[k] to |->
    # For [i,j,k] = [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
    for qubit_i in range(3):
        for qubit_j in range(3):
            for qubit_k in range(3):
                if ((qubit_i != qubit_j) & (qubit_i != qubit_k) & (qubit_k != qubit_j)):
                    circuit = QuantumCircuit(*regs)
                    circuit.h(qr[0])
                    circuit.h(qr[1])
                    circuit.h(qr[2])
                    circuit.initialize([0, 1, 0, 0, 0, -1, 0, 0] / sqrt(2), \
                                       [qr[qubit_i], qr[qubit_j], qr[qubit_k]])

                    if final_measure:
                        circuit.barrier(qr)
                        circuit.measure(qr, cr)
                    circuits.append(circuit)

    return circuits

def initialize_statevector_deterministic():
    """Initialize test circuits reference counts."""

    targets = []

    # Start with |+++> state
    # Initialize qr[i] to |1> for i=0,1,2
    targets.append(array([0. +0.j, 0.5+0.j, 0. +0.j, 0.5+0.j, 0. +0.j, 0.5+0.j, 0. +0.j, 0.5+0.j]))
    targets.append(array([0. +0.j, 0. +0.j, 0.5+0.j, 0.5+0.j, 0. +0.j, 0. +0.j, 0.5+0.j, 0.5+0.j]))
    targets.append(array([0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j]))

    # Start with |+++> state
    # Initialize qr[i] to |1> and qr[j] to |0>
    # For [i,j] = [0,1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]
    targets.append(array([0. + 0.j, 1.0 + 0.j, 0. + 0.j, 0. + 0.j, \
                          0. + 0.j, 1.0 + 0.j, 0. + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 1.0 + 0.j, 0. + 0.j, 1.0 + 0.j, \
                          0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 0. + 0.j, 1.0 + 0.j, 0. + 0.j, \
                          0. + 0.j, 0. + 0.j, 1.0 + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 0. + 0.j, 1.0 + 0.j, 1.0 + 0.j, \
                          0. + 0.j, 0. + 0.j, 0 + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 0. + 0.j, 0 + 0.j, 0. + 0.j, \
                          1.0 + 0.j, 0. + 0.j, 1.0 + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 0. + 0.j, 0 + 0.j, 0. + 0.j, \
                          1.0 + 0.j, 1.0 + 0.j, 0. + 0.j, 0. + 0.j] / sqrt(2)))

    # Start with |+++> state
    # Initialize qr[i] to |1>, qr[j] to |0> and qr[k] to |->
    # For [i,j,k] = [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
    targets.append(array([0. + 0.j, 1.0 + 0.j, 0. + 0.j, 0. + 0.j, \
                          0. + 0.j, -1.0 + 0.j, 0. + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 1.0 + 0.j, 0. + 0.j, -1.0 + 0.j,
                          0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 0. + 0.j, 1.0 + 0.j, 0. + 0.j, \
                          0. + 0.j, 0. + 0.j, -1.0 + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 0. + 0.j, 1.0 + 0.j, -1.0 + 0.j, \
                          0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, \
                          1.0 + 0.j, 0. + 0.j, -1.0 + 0.j, 0. + 0.j] / sqrt(2)))
    targets.append(array([0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, \
                          1.0 + 0.j, -1.0 + 0.j, 0. + 0.j, 0. + 0.j] / sqrt(2)))

    return targets

# ==========================================================================
# Non-Deterministic output
# ==========================================================================

def initialize_circuits_nondeterministic(final_measure=True):
    """Initialize test circuits with non-deterministic count output"""

    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # Start with a state (|00>+|11>)/sqrt(2)
    # Initialize qubit 0 to |+>
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr[0])
    # circuit.cx(qr[0], qr[1])
    circuit.initialize([1, 1]/sqrt(2), [qr[0]])

    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Start with a state (|00>+|11>)/sqrt(2)
    # Initialize qubit 0 to |->
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr[0])
    # circuit.cx(qr[0], qr[1])
    circuit.initialize([1, -1]/sqrt(2), [qr[0]])

    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits

def initialize_statevector_nondeterministic():
    """Initialize test circuits reference counts."""
    targets = []
    # Start with a state (|00>+|11>)/sqrt(2)
    # Initialize qubit 0 to |+>
    targets.append(array([1, 1, 0, 0]) / sqrt(2))
    # Initialize qubit 0 to |->
    targets.append(array([1, -1, 0, 0]) / sqrt(2))
    return targets