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
Test circuits and reference outputs for initialize instruction.
"""

from numpy import array, sqrt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def initialize_circuits_w_1(init_state, final_measure=True):
    """Initialize test circuits"""

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
        circuit.initialize(init_state, [qr[qubit]])

        if final_measure:
            circuit.barrier(qr)
            circuit.measure(qr, cr)
        circuits.append(circuit)

    return circuits

def initialize_circuits_w_2(init_state, final_measure=True):
    """Initialize test circuits"""

    circuits = []
    qr = QuantumRegister(3)
    if final_measure:
        cr = ClassicalRegister(3)
        regs = (qr, cr)
    else:
        regs = (qr, )
    
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
                circuit.initialize(init_state, [qr[qubit_i], qr[qubit_j]])

                if final_measure:
                    circuit.barrier(qr)
                    circuit.measure(qr, cr)
                circuits.append(circuit)

    return circuits


def initialize_circuits_1(final_measure=True):
    """Initialize test circuits"""

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
                if (qubit_i != qubit_j) & (qubit_i != qubit_k) & (qubit_k != qubit_j):
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

def initialize_counts_1(shots, hex_counts=True):
    """Initialize test circuits reference counts."""
    targets = []
    if hex_counts:
        # Initialize 0 to |1> from |+++>
        targets.append({'0x1': shots/4,
                        '0x3': shots/4,
                        '0x5': shots/4,
                        '0x7': shots/4})
        # Initialize 1 to |1> from |+++>
        targets.append({'0x2': shots/4,
                        '0x3': shots/4,
                        '0x6': shots/4,
                        '0x7': shots/4})
        # Initialize 2 to |1> from |+++>
        targets.append({'0x4': shots/4,
                        '0x5': shots/4,
                        '0x6': shots/4,
                        '0x7': shots/4})
        # Initialize 0,1 to |01> from |+++>
        targets.append({'0x1': shots/2,
                        '0x5': shots/2})
        # Initialize 0,2 to |01> from |+++>
        targets.append({'0x1': shots/2,
                        '0x3': shots/2})
        # Initialize 1,0 to |01> from |+++>
        targets.append({'0x2': shots/2,
                        '0x6': shots/2})
        # Initialize 1,2 to |01> from |+++>
        targets.append({'0x2': shots/2,
                        '0x3': shots/2})
        # Initialize 2,0 to |01> from |+++>
        targets.append({'0x4': shots/2,
                        '0x6': shots/2})
        # Initialize 2,1 to |01> from |+++>
        targets.append({'0x4': shots/2,
                        '0x5': shots/2})
        # Initialize 0,1,2 to |01-> from |+++>
        targets.append({'0x1': shots/2,
                        '0x5': shots/2})
        # Initialize 0,2,1 to |01-> from |+++>
        targets.append({'0x1': shots/2,
                        '0x3': shots/2})
        # Initialize 1,0,2 to |01-> from |+++>
        targets.append({'0x2': shots/2,
                        '0x6': shots/2})
        # Initialize 1,2,0 to |01-> from |+++>
        targets.append({'0x2': shots/2,
                        '0x3': shots/2})
        # Initialize 2,0,1 to |01-> from |+++>
        targets.append({'0x4': shots/2,
                        '0x6': shots/2})
        # Initialize 2,1,0 to |01-> from |+++>
        targets.append({'0x4': shots/2,
                        '0x5': shots/2})
    else:
        # Initialize 0 to |1> from |+++>
        targets.append({'001': shots/4,
                        '011': shots/4,
                        '101': shots/4,
                        '111': shots/4})
        # Initialize 1 to |1> from |+++>
        targets.append({'010': shots/4,
                        '011': shots/4,
                        '110': shots/4,
                        '111': shots/4})
        # Initialize 2 to |1> from |+++>
        targets.append({'100': shots/4,
                        '101': shots/4,
                        '110': shots/4,
                        '111': shots/4})
        # Initialize 0,1 to |01> from |+++>
        targets.append({'001': shots/2,
                        '101': shots/2})
        # Initialize 0,2 to |01> from |+++>
        targets.append({'001': shots/2,
                        '011': shots/2})
        # Initialize 1,0 to |01> from |+++>
        targets.append({'010': shots/2,
                        '110': shots/2})
        # Initialize 1,2 to |01> from |+++>
        targets.append({'010': shots/2,
                        '011': shots/2})
        # Initialize 2,0 to |01> from |+++>
        targets.append({'100': shots/2,
                        '110': shots/2})
        # Initialize 2,1 to |01> from |+++>
        targets.append({'100': shots/2,
                        '101': shots/2})
        # Initialize 0,1,2 to |01-> from |+++>
        targets.append({'001': shots/2,
                        '101': shots/2})
        # Initialize 0,2,1 to |01-> from |+++>
        targets.append({'001': shots/2,
                        '011': shots/2})
        # Initialize 1,0,2 to |01-> from |+++>
        targets.append({'010': shots/2,
                        '110': shots/2})
        # Initialize 1,2,0 to |01-> from |+++>
        targets.append({'010': shots/2,
                        '011': shots/2})
        # Initialize 2,0,1 to |01-> from |+++>
        targets.append({'100': shots/2,
                        '110': shots/2})
        # Initialize 2,1,0 to |01-> from |+++>
        targets.append({'100': shots/2,
                        '101': shots/2})

    return targets

def initialize_statevector_1():
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


def initialize_circuits_2(final_measure=True):
    """Initialize test circuits"""

    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # Initialize 0 to |1> from |++>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.initialize([0, 1], [qr[0]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Initialize 1 to |1> from |++>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.initialize([0, 1], [qr[1]])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def initialize_counts_2(shots, hex_counts=True):
    """Initialize test circuits reference counts."""
    targets = []
    if hex_counts:
        # Initialize 0 to |1> from |++>
        targets.append({'0x1': shots / 2, '0x3': shots / 2})
        # Initialize 1 to |1> from |++>
        targets.append({'0x2': shots / 2, '0x3': shots / 2})
    else:
        # Initialize 0 to |1> from |++>
        targets.append({'01': shots / 2, '11': shots / 2})
        # Initialize 1 to |1> from |++>
        targets.append({'10': shots / 2, '11': shots / 2})
    return targets


def initialize_statevector_2():
    """Initialize test circuits reference counts."""
    targets = []
    # Initialize 0 to |1> from |++>
    targets.append(array([0, 1, 0, 1]) / sqrt(2))
    # Initialize 1 to |1> from |++>
    targets.append(array([0, 0, 1, 1]) / sqrt(2))
    return targets


# ==========================================================================
# Sampling optimization
# ==========================================================================

def initialize_sampling_optimization():
    """Test sampling optimization"""
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    qc = QuantumCircuit(qr, cr)

    # The optimization should not be triggerred
    # because the initialize operation performs randomizations
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.initialize([1, 0], [qr[0]])
    qc.measure(qr, cr)

    return [qc]

def initialize_counts_sampling_optimization(shots, hex_counts=True):
    """Sampling optimization counts"""
    if hex_counts:
        return [{'0x0': shots/2, '0x2': shots/2}]
    else:
        return [{'0x00': shots/2, '0x10': shots/2}]
