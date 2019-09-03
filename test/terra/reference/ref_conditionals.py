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
Test circuits and reference outputs for conditional gates.
"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Instruction


def add_conditional_x(circuit, qreg, creg, val, conditional_type):
    """Add a conditional instruction to a circuit.

    Args:
        circuit (QuantumCircuit): circuit to add instruction to.
        qreg (QuantumRegister): qubit to apply conditional X to
        creg (ClassicalRegister): classical reg to condition on
        val (int): Classical reg value to condition on.
        conditional_type (string): instruction type to add conditional
                                   X as.

    Conditional type can be 'gate', 'unitary', 'kraus', 'superop'
    and will apply a conditional X-gate in that representation
    """
    # X-gate matrix
    x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    x_superop = Instruction('superop', 1, 0, [np.kron(x_mat, x_mat)])
    x_kraus = Instruction('kraus', 1, 0, [x_mat])

    if conditional_type == 'unitary':
        circuit.unitary(x_mat, [qreg]).c_if(creg, val)
    elif conditional_type == 'kraus':
        circuit.append(x_kraus, [qreg]).c_if(creg, val)
    elif conditional_type == 'superop':
        circuit.append(x_superop, [qreg]).c_if(creg, val)
    else:
        circuit.x(qreg).c_if(creg, val)

# ==========================================================================
# Conditionals on 1-bit register
# ==========================================================================


def conditional_circuits_1bit(final_measure=True, conditional_type='gate'):
    """Conditional gates on single bit classical register."""
    circuits = []
    qr = QuantumRegister(1)
    cond = ClassicalRegister(1, 'cond')
    if final_measure:
        cr = ClassicalRegister(1, 'meas')
        regs = (qr, cr, cond)
    else:
        regs = (qr, cond)

    # Conditional on 0 (cond = 0)
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 0, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 0 (cond = 1)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 0, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 1 (cond = 0)
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 1, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 1 (cond = 1)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 1, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def conditional_counts_1bit(shots, hex_counts=True):
    """Conditional circuits reference counts."""
    targets = []
    if hex_counts:
        # Conditional on 0 (cond = 0), result "0 1"
        targets.append({'0x1': shots})
        # Conditional on 0 (cond = 1), result "1 0"
        targets.append({'0x2': shots})
        # Conditional on 1 (cond = 0), # result "0 0"
        targets.append({'0x0': shots})
        # Conditional on 1 (cond = 1), # result "1 1"
        targets.append({'0x3': shots})
    else:
        # Conditional on 0 (cond = 0), result "0 1"
        targets.append({'0 1': shots})
        # Conditional on 0 (cond = 1), result "1 0"
        targets.append({'1 0': shots})
        # Conditional on 1 (cond = 0), # result "0 0"
        targets.append({'0 0': shots})
        # Conditional on 1 (cond = 1), # result "1 1"
        targets.append({'1 1': shots})
    return targets


def conditional_statevector_1bit():
    """Conditional circuits reference statevector."""
    targets = []
    # Conditional on 0 (cond = 0)
    targets.append(np.array([0, 1]))
    # Conditional on 0 (cond = 1)
    targets.append(np.array([1, 0]))
    # Conditional on 1 (cond = 0)
    targets.append(np.array([1, 0]))
    # Conditional on 1 (cond = 1)
    targets.append(np.array([0, 1]))
    return targets


# ==========================================================================
# Conditionals on 2-bit register
# ==========================================================================

def conditional_circuits_2bit(final_measure=True, conditional_type='gate'):
    """Conditional test circuits on 2-bit classical register."""
    circuits = []
    qr = QuantumRegister(1)
    cond = ClassicalRegister(2, 'cond')
    if final_measure:
        cr = ClassicalRegister(1, 'meas')
        regs = (qr, cr, cond)
    else:
        regs = (qr, cond)

    # Conditional on 00 (cr = 00)
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 0, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 00 (cr = 01)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 0, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 00 (cr = 10)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[1])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 0, conditional_type)
    circuits.append(circuit)

    # Conditional on 00 (cr = 11)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.measure(qr[0], cond[1])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 0, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 01 (cr = 00)
    circuit = QuantumCircuit(*regs)
    add_conditional_x(circuit, qr[0], cond, 1, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 01 (cr = 01)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 1, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 01 (cr = 10)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[1])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 1, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 01 (cr = 11)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.measure(qr[0], cond[1])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 1, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 10 (cr = 00)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr).c_if(cond, 2)
    circuits.append(circuit)
    # Conditional on 10 (cr = 01)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 2, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 10 (cr = 10)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[1])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 2, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 10 (cr = 11)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.measure(qr[0], cond[1])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 2, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 11 (cr = 00)
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 3, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 11 (cr = 01)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 3, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 11 (cr = 10)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[1])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 3, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 11 (cr = 11)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.measure(qr[0], cond[1])
    circuit.x(qr)
    circuit.barrier(qr)
    add_conditional_x(circuit, qr[0], cond, 3, conditional_type)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def conditional_counts_2bit(shots, hex_counts=True):
    """2-bit conditional circuits reference counts."""
    targets = []
    if hex_counts:
        # Conditional on 00 (cr = 00), result "00 1"
        targets.append({'0x1': shots})
        # Conditional on 00 (cr = 01), result "01 0"
        targets.append({'0x2': shots})
        # Conditional on 00 (cr = 10), result "10 0"
        targets.append({'0x4': shots})
        # Conditional on 00 (cr = 11), result "11 0"
        targets.append({'0x6': shots})
        # Conditional on 01 (cr = 00), result "00 0"
        targets.append({'0x0': shots})
        # Conditional on 01 (cr = 01), result "01 1"
        targets.append({'0x3': shots})
        # Conditional on 01 (cr = 10), result "10 0"
        targets.append({'0x4': shots})
        # Conditional on 01 (cr = 11), result "11 0"
        targets.append({'0x6': shots})
        # Conditional on 10 (cr = 00), result "00 0"
        targets.append({'0x0': shots})
        # Conditional on 10 (cr = 01), result "01 0"
        targets.append({'0x2': shots})
        # Conditional on 10 (cr = 10), result "10 1"
        targets.append({'0x5': shots})
        # Conditional on 10 (cr = 11), result "11 0"
        targets.append({'0x6': shots})
        # Conditional on 11 (cr = 00), result "00 0"
        targets.append({'0x0': shots})
        # Conditional on 11 (cr = 01), result "01 0"
        targets.append({'0x2': shots})
        # Conditional on 11 (cr = 10), result "10 0"
        targets.append({'0x4': shots})
        # Conditional on 11 (cr = 11), result "11 1"
        targets.append({'0x7': shots})
    else:
        # Conditional on 00 (cr = 00), result "00 1"
        targets.append({'00 1': shots})
        # Conditional on 00 (cr = 01), result "01 0"
        targets.append({'01 0': shots})
        # Conditional on 00 (cr = 10), result "10 0"
        targets.append({'10 0': shots})
        # Conditional on 00 (cr = 11), result "11 0"
        targets.append({'11 0': shots})
        # Conditional on 01 (cr = 00), result "00 0"
        targets.append({'00 0': shots})
        # Conditional on 01 (cr = 01), result "01 1"
        targets.append({'01 1': shots})
        # Conditional on 01 (cr = 10), result "10 0"
        targets.append({'10 0': shots})
        # Conditional on 01 (cr = 11), result "11 0"
        targets.append({'11 0': shots})
        # Conditional on 10 (cr = 00), result "00 0"
        targets.append({'00 0': shots})
        # Conditional on 10 (cr = 01), result "01 0"
        targets.append({'01 0': shots})
        # Conditional on 10 (cr = 10), result "10 1"
        targets.append({'10 0': shots})
        # Conditional on 10 (cr = 11), result "11 0"
        targets.append({'11 0': shots})
        # Conditional on 11 (cr = 00), result "00 0"
        targets.append({'00 0': shots})
        # Conditional on 11 (cr = 01), result "01 0"
        targets.append({'01 0': shots})
        # Conditional on 11 (cr = 10), result "10 0"
        targets.append({'10 0': shots})
        # Conditional on 11 (cr = 11), result "11 1"
        targets.append({'11 1': shots})
    return targets


def conditional_statevector_2bit():
    """2-bit conditional circuits reference statevector."""
    state_0 = np.array([1, 0])
    state_1 = np.array([0, 1])
    targets = []
    # Conditional on 00 (cr = 00)
    targets.append(state_1)
    # Conditional on 00 (cr = 01)
    targets.append(state_0)
    # Conditional on 00 (cr = 10)
    targets.append(state_0)
    # Conditional on 00 (cr = 11)
    targets.append(state_0)
    # Conditional on 01 (cr = 00)
    targets.append(state_0)
    # Conditional on 01 (cr = 01)
    targets.append(state_1)
    # Conditional on 01 (cr = 10)
    targets.append(state_0)
    # Conditional on 01 (cr = 11)
    targets.append(state_0)
    # Conditional on 10 (cr = 00)
    targets.append(state_0)
    # Conditional on 10 (cr = 01)
    targets.append(state_0)
    # Conditional on 10 (cr = 10)
    targets.append(state_1)
    # Conditional on 10 (cr = 11)
    targets.append(state_0)
    # Conditional on 11 (cr = 00)
    targets.append(state_0)
    # Conditional on 11 (cr = 01)
    targets.append(state_0)
    # Conditional on 11 (cr = 10)
    targets.append(state_0)
    # Conditional on 11 (cr = 11)
    targets.append(state_1)
    return targets
