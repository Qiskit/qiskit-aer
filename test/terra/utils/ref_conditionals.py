# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test circuits and reference outputs for conditional gates.
"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


# ==========================================================================
# Conditionals on 1-bit register
# ==========================================================================

def conditional_circuits_1bit(final_measure=True):
    """Conditional test circuits on single bit classical register."""
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
    circuit.x(qr).c_if(cond, 0)
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
    circuit.x(qr).c_if(cond, 0)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 1 (cond = 0)
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    circuit.x(qr).c_if(cond, 1)
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
    circuit.x(qr).c_if(cond, 1)
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

def conditional_circuits_2bit(final_measure=True):
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
    circuit.x(qr).c_if(cond, 0)
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
    circuit.x(qr).c_if(cond, 0)
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
    circuit.x(qr).c_if(cond, 0)
    circuits.append(circuit)

    # Conditional on 00 (cr = 11)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.measure(qr[0], cond[0])
    circuit.measure(qr[0], cond[1])
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.x(qr).c_if(cond, 0)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 01 (cr = 00)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr).c_if(cond, 1)
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
    circuit.x(qr).c_if(cond, 1)
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
    circuit.x(qr).c_if(cond, 1)
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
    circuit.x(qr).c_if(cond, 1)
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
    circuit.x(qr).c_if(cond, 2)
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
    circuit.x(qr).c_if(cond, 2)
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
    circuit.x(qr).c_if(cond, 2)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Conditional on 11 (cr = 00)
    circuit = QuantumCircuit(*regs)
    circuit.x(qr).c_if(cond, 3)
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
    circuit.x(qr).c_if(cond, 3)
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
    circuit.x(qr).c_if(cond, 3)
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
    circuit.x(qr).c_if(cond, 3)
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


def conditional_statevector_2bit(final_measure=True):
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
