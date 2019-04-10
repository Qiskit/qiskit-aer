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
    # Initialize qr[0] to |1>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.h(qr[2])
    circuit.initialize([0, 1], [qr[0]])

    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits

def initialize_statevector_deterministic():
    """Initialize test circuits reference counts."""
    targets = []
    # Start with |+++> state
    # Initialize qr[0] to |1>
    targets.append(array([0. +0.j, 0.5+0.j, 0. +0.j, 0.5+0.j, 0. +0.j, 0.5+0.j, 0. +0.j, 0.5+0.j]))

    return targets
