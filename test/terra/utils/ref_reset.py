# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test circuits and reference outputs for reset instruction.
"""

from numpy import array, sqrt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


# ==========================================================================
# Deterministic output
# ==========================================================================

def reset_circuits_deterministic(final_measure=True):
    """Reset test circuits with deterministic count output"""

    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # Reset 0 from |11>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.reset(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 1 from |11>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.reset(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 0,1 from |11>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.reset(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 0,1 from |++>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.reset(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def reset_counts_deterministic(shots, hex_counts=True):
    """Reset test circuits reference counts."""
    targets = []
    if hex_counts:
        # Reset 0 from |11>
        targets.append({'0x2': shots})
        # Reset 1 from |11>
        targets.append({'0x1': shots})
        # Reset 0,1 from |11>
        targets.append({'0x0': shots})
        # Reset 0,1 from |++>
        targets.append({'0x0': shots})
    else:
        # Reset 0 from |11>
        targets.append({'10': shots})
        # Reset 1 from |11>
        targets.append({'01': shots})
        # Reset 0,1 from |11>
        targets.append({'00': shots})
        # Reset 0,1 from |++>
        targets.append({'00': shots})
    return targets


def reset_statevector_deterministic():
    """Reset test circuits reference counts."""
    targets = []
    # Reset 0 from |11>
    targets.append(array([0, 0, 1, 0]))
    # Reset 1 from |11>
    targets.append(array([0, 1, 0, 0]))
    # Reset 0,1 from |11>
    targets.append(array([1, 0, 0, 0]))
    # Reset 0,1 from |++>
    targets.append(array([1, 0, 0, 0]))
    return targets


# ==========================================================================
# Non-Deterministic output
# ==========================================================================

def reset_circuits_nondeterministic(final_measure=True):
    """Reset test circuits with deterministic count output"""

    circuits = []
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # Reset 0 from |++>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.reset(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Reset 1 from |++>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.reset(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def reset_counts_nondeterministic(shots, hex_counts=True):
    """Reset test circuits reference counts."""
    targets = []
    if hex_counts:
        # Reset 0 from |++>
        targets.append({'0x0': shots / 2, '0x2': shots / 2})
        # Reset 1 from |++>
        targets.append({'0x0': shots / 2, '0x1': shots / 2})
    else:
        # Reset 0 from |++>
        targets.append({'00': shots / 2, '10': shots / 2})
        # Reset 1 from |++>
        targets.append({'00': shots / 2, '01': shots / 2})
    return targets


def reset_statevector_nondeterministic():
    """Reset test circuits reference counts."""
    targets = []
    # Reset 0 from |++>
    targets.append(array([1, 0, 1, 0]) / sqrt(2))
    # Reset 1 from |++>
    targets.append(array([1, 1, 0, 0]) / sqrt(2))
    return targets
