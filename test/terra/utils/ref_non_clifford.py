# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test circuits and reference outputs for non-Clifford gate instructions.
"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


# ==========================================================================
# T-gate
# ==========================================================================

def t_gate_circuits_deterministic(final_measure=True):
    """T-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # T
    circuit = QuantumCircuit(*regs)
    circuit.t(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # T.T = S
    circuit = QuantumCircuit(*regs)
    circuit.t(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # T.X
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # H.T.T.T.T.H = H.Z.H = X
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def t_gate_counts_deterministic(shots, hex_counts=True):
    """T-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # T
        targets.append({'0x0': shots})
        # T.T = S
        targets.append({'0x0': shots})
        # T.X
        targets.append({'0x1': shots})
        # H.T.T.T.TH = H.Z.H = X
        targets.append({'0x1': shots})
    else:
        # T
        targets.append({'0': shots})
        # T.T = S
        targets.append({'0': shots})
        # T.X
        targets.append({'1': shots})
        # H.T.T.T.TH = H.Z.H = X
        targets.append({'1': shots})
    return targets


def t_gate_statevector_deterministic():
    """T-gate circuits reference statevectors."""
    targets = []
    # T
    targets.append(np.array([1, 0]))
    # T.T = S
    targets.append(np.array([1, 0]))
    # T.X
    targets.append(np.array([0, 1 + 1j]) / np.sqrt(2))
    # H.T.T.T.T.H = H.Z.H = X
    targets.append(np.array([0, 1]))
    return targets


def t_gate_unitary_deterministic():
    """T-gate circuits reference unitaries."""
    targets = []
    # T
    targets.append(np.diag([1, (1 + 1j) / np.sqrt(2)]))
    # T.T = S
    targets.append(np.diag([1, 1j]))
    # T.X
    targets.append(np.array([[0, 1], [(1 + 1j) / np.sqrt(2), 0]]))
    # H.T.T.T.TH = H.Z.H = X
    targets.append(np.array([[0, 1], [1, 0]]))
    return targets


def t_gate_circuits_nondeterministic(final_measure=True):
    """T-gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # T.H
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # X.T.H
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    circuit.barrier(qr)
    circuit.x(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # H.T.T.H = H.S.H
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def t_gate_counts_nondeterministic(shots, hex_counts=True):
    """T-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # T.H
        targets.append({'0x0': shots / 2, '0x1': shots / 2})
        # X.T.H
        targets.append({'0x0': shots / 2, '0x1': shots / 2})
        # H.T.T.H = H.S.H
        targets.append({'0x0': shots / 2, '0x1': shots / 2})
    else:
        # T.H
        targets.append({'0': shots / 2, '1': shots / 2})
        # X.T.H
        targets.append({'0': shots / 2, '1': shots / 2})
        # H.T.T.H = H.S.H
        targets.append({'0': shots / 2, '1': shots / 2})
    return targets


def t_gate_statevector_nondeterministic():
    """T-gate circuits reference statevectors."""
    targets = []
    # T.H
    targets.append(np.array([1 / np.sqrt(2), 0.5 + 0.5j]))
    # X.T.H
    targets.append(np.array([0.5 + 0.5j, 1 / np.sqrt(2)]))
    # H.T.T.H = H.S.H
    targets.append(np.array([1 + 1j, 1 - 1j]) / 2)
    return targets


def t_gate_unitary_nondeterministic():
    """T-gate circuits reference unitaries."""
    targets = []
    # T.H
    targets.append(np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                             [0.5 + 0.5j, -0.5 - 0.5j]]))
    # X.T.H
    targets.append(np.array([[0.5 + 0.5j, -0.5 - 0.5j],
                             [1 / np.sqrt(2), 1 / np.sqrt(2)]]))
    # H.T.T.H = H.S.H
    targets.append(np.array([[1 + 1j, 1 - 1j],
                             [1 - 1j, 1 + 1j]]) / 2)
    return targets


# ==========================================================================
# T^dagger-gate
# ==========================================================================

def tdg_gate_circuits_deterministic(final_measure=True):
    """Tdg-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # Tdg
    circuit = QuantumCircuit(*regs)
    circuit.tdg(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # H.Tdg.T.H = I
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.t(qr)
    circuit.barrier(qr)
    circuit.tdg(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # H.Tdg.Tdg.Tdg.Tdg.H = H.Z.H = X
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.tdg(qr)
    circuit.barrier(qr)
    circuit.tdg(qr)
    circuit.barrier(qr)
    circuit.tdg(qr)
    circuit.barrier(qr)
    circuit.tdg(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)
    return circuits


def tdg_gate_counts_deterministic(shots, hex_counts=True):
    """Sdg-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # Tdg
        targets.append({'0x0': shots})
        # H.Tdg.T.H = I
        targets.append({'0x0': shots})
        # H.Tdg.Tdg.Tdg.Tdg.H = H.Z.H = X
        targets.append({'0x1': shots})
    else:
        # Tdg
        targets.append({'0': shots})
        # H.Tdg.T.H = I
        targets.append({'0': shots})
        # H.Tdg.Tdg.Tdg.Tdg.H = H.Z.H = X
        targets.append({'1': shots})
    return targets


def tdg_gate_statevector_deterministic():
    """Sdg-gate circuits reference statevectors."""
    targets = []
    # Tdg
    targets.append(np.array([1, 0]))
    # H.Tdg.T.H = I
    targets.append(np.array([1, 0]))
    # H.Tdg.Tdg.Tdg.Tdg.H = H.Z.H = X
    targets.append(np.array([0, 1]))
    return targets


def tdg_gate_unitary_deterministic():
    """Tdg-gate circuits reference unitaries."""
    targets = []
    # Tdg
    targets.append(np.diag([1, (1 - 1j) / np.sqrt(2)]))
    # H.Tdg.T.H = I
    targets.append(np.eye(2))
    # H.Tdg.Tdg.Tdg.Tdg.H = H.Z.H = X
    targets.append(np.array([[0, 1], [1, 0]]))
    return targets


def tdg_gate_circuits_nondeterministic(final_measure=True):
    """Tdg-gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(1)
    if final_measure:
        cr = ClassicalRegister(1)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # Tdg.H
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.tdg(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # X.Tdg.H
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.tdg(qr)
    circuit.barrier(qr)
    circuit.x(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # H.Tdg.Tdg.H = H.Sdg.H
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.tdg(qr)
    circuit.barrier(qr)
    circuit.tdg(qr)
    circuit.barrier(qr)
    circuit.h(qr)
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def tdg_gate_counts_nondeterministic(shots, hex_counts=True):
    """Tdg-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # Tdg.H
        targets.append({'0x0': shots / 2, '0x1': shots / 2})
        # X.Tdg.H
        targets.append({'0x0': shots / 2, '0x1': shots / 2})
        # H.Tdg.Tdg.H = H.Sdg.H
        targets.append({'0x0': shots / 2, '0x1': shots / 2})
    else:
        # Tdg.H
        targets.append({'0': shots / 2, '1': shots / 2})
        # X.Tdg.H
        targets.append({'0': shots / 2, '1': shots / 2})
        # H.Tdg.Tdg.H = H.Sdg.H
        targets.append({'0': shots / 2, '1': shots / 2})
    return targets


def tdg_gate_statevector_nondeterministic():
    """Tdg-gate circuits reference statevectors."""
    targets = []
    # Tdg.H
    targets.append(np.array([1 / np.sqrt(2), 0.5 - 0.5j]))
    # X.Tdg.H
    targets.append(np.array([0.5 - 0.5j, 1 / np.sqrt(2)]))
    # H.Tdg.Tdg.H = H.Sdg.H
    targets.append(np.array([1 - 1j, 1 + 1j]) / 2)
    return targets


def tdg_gate_unitary_nondeterministic():
    """Tdg-gate circuits reference unitaries."""
    targets = []
    # Tdg.H
    targets.append(np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                             [0.5 - 0.5j, -0.5 + 0.5j]]))
    # X.Tdg.H
    targets.append(np.array([[0.5 - 0.5j, -0.5 + 0.5j],
                             [1 / np.sqrt(2), 1 / np.sqrt(2)]]))
    # H.Tdg.Tdg.H = H.Sdg.H
    targets.append(np.array([[1 - 1j, 1 + 1j],
                             [1 + 1j, 1 - 1j]]) / 2)
    return targets


# ==========================================================================
# CCX-gate
# ==========================================================================

def ccx_gate_circuits_deterministic(final_measure=True):
    """CCX-gate test circuits with deterministic counts."""
    circuits = []
    qr = QuantumRegister(3)
    if final_measure:
        cr = ClassicalRegister(3)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # CCX(0,1,2)
    circuit = QuantumCircuit(*regs)
    circuit.ccx(qr[0], qr[1], qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (I^X^X).CCX(0,1,2).(I^X^X) -> |100>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.ccx(qr[0], qr[1], qr[2])
    circuit.barrier(qr)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (X^I^X).CCX(0,1,2).(X^I^X) -> |000>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.ccx(qr[0], qr[1], qr[2])
    circuit.barrier(qr)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (X^X^I).CCX(0,1,2).(X^X^I) -> |000>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.ccx(qr[0], qr[1], qr[2])
    circuit.barrier(qr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.x(qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # CCX(2,1,0)
    circuit = QuantumCircuit(*regs)
    circuit.ccx(qr[2], qr[1], qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (I^X^X).CCX(2,1,0).(I^X^X) -> |000>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.ccx(qr[2], qr[1], qr[0])
    circuit.barrier(qr)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (X^I^X).CCX(2,1,0).(X^I^X) -> |000>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.ccx(qr[2], qr[1], qr[0])
    circuit.barrier(qr)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (X^X^I).CCX(2,1,0).(X^X^I) -> |001>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.ccx(qr[2], qr[1], qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.x(qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def ccx_gate_counts_deterministic(shots, hex_counts=True):
    """CCX-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # CCX(0,1,2)
        targets.append({'0x0': shots})
        # (I^X^X).CCX(0,1,2).(I^X^X) -> |100>
        targets.append({'0x4': shots})
        # (X^I^X).CCX(0,1,2).(X^I^X) -> |000>
        targets.append({'0x0': shots})
        # (X^X^I).CCX(0,1,2).(X^X^I) -> |000>
        targets.append({'0x0': shots})
        # CCX(2,1,0)
        targets.append({'0x0': shots})
        # (I^X^X).CCX(2,1,0).(I^X^X) -> |000>
        targets.append({'0x0': shots})
        # (X^I^X).CCX(2,1,0).(X^I^X) -> |000>
        targets.append({'0x0': shots})
        # (X^X^I).CCX(2,1,0).(X^X^I) -> |001>
        targets.append({'0x1': shots})
    else:
        # CCX(0,1,2)
        targets.append({'000': shots})
        # (I^X^X).CCX(0,1,2).(I^X^X) -> |100>
        targets.append({'100': shots})
        # (X^I^X).CCX(0,1,2).(X^I^X) -> |000>
        targets.append({'000': shots})
        # (X^X^I).CCX(0,1,2).(X^X^I) -> |000>
        targets.append({'000': shots})
        # CCX(2,1,0)
        targets.append({'000': shots})
        # (I^X^X).CCX(2,1,0).(I^X^X) -> |000>
        targets.append({'000': shots})
        # (X^I^X).CCX(2,1,0).(X^I^X) -> |000>
        targets.append({'000': shots})
        # (X^X^I).CCX(2,1,0).(X^X^I) -> |001>
        targets.append({'001': shots})
    return targets


def ccx_gate_statevector_deterministic():
    """CCX-gate circuits reference statevectors."""
    targets = []
    zero_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    # CCX(0,1,2)
    targets.append(zero_state)
    # (I^X^X).CCX(0,1,2).(I^X^X) -> |100>
    targets.append(np.array([0, 0, 0, 0, 1, 0, 0, 0]))
    # (X^I^X).CCX(0,1,2).(X^I^X) -> |000>
    targets.append(zero_state)
    # (X^X^I).CCX(0,1,2).(X^X^I) -> |000>
    targets.append(zero_state)
    # CCX(2,1,0)
    targets.append(zero_state)
    # (I^X^X).CCX(2,1,0).(I^X^X) -> |000>
    targets.append(zero_state)
    # (X^I^X).CCX(2,1,0).(X^I^X) -> |000>
    targets.append(zero_state)
    # (X^X^I).CCX(2,1,0).(X^X^I) -> |001>
    targets.append(np.array([0, 1, 0, 0, 0, 0, 0, 0]))
    return targets


def ccx_gate_unitary_deterministic():
    """CCX-gate circuits reference unitaries."""
    targets = []

    # CCX(0,1,2)
    targets.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0]]))
    # (I^X^X).CCX(0,1,2).(I^X^X) -> |100>
    targets.append(np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]]))
    # (X^I^X).CCX(0,1,2).(X^I^X) -> |000>
    targets.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]]))
    # (X^X^I).CCX(0,1,2).(X^X^I) -> |000>
    targets.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]]))
    # CCX(2,1,0)
    targets.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 0]]))
    # (I^X^X).CCX(2,1,0).(I^X^X) -> |000>
    targets.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]]))
    # (X^I^X).CCX(2,1,0).(X^I^X) -> |000>
    targets.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]]))
    # (X^X^I).CCX(2,1,0).(X^X^I) -> |001>
    targets.append(np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]]))
    return targets


def ccx_gate_circuits_nondeterministic(final_measure=True):
    """CCX-gate test circuits with non-deterministic counts."""
    circuits = []
    qr = QuantumRegister(3)
    if final_measure:
        cr = ClassicalRegister(3)
        regs = (qr, cr)
    else:
        regs = (qr, )

    # (I^X^I).CCX(0,1,2).(I^X^H) -> |000> + |101>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.ccx(qr[0], qr[1], qr[2])
    circuit.barrier(qr)
    circuit.x(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (I^I^X).CCX(0,1,2).(I^H^X) -> |000> + |110>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[1])
    circuit.barrier(qr)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.ccx(qr[0], qr[1], qr[2])
    circuit.barrier(qr)
    circuit.x(qr[0])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (I^X^I).CCX(2,1,0).(H^X^I) -> |000> + |101>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[2])
    circuit.barrier(qr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.ccx(qr[2], qr[1], qr[0])
    circuit.barrier(qr)
    circuit.x(qr[1])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # (X^I^I).CCX(2,1,0).(X^H^I) -> |000> + |011>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[1])
    circuit.barrier(qr)
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.ccx(qr[2], qr[1], qr[0])
    circuit.barrier(qr)
    circuit.x(qr[2])
    if final_measure:
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits


def ccx_gate_counts_nondeterministic(shots, hex_counts=True):
    """CCX-gate circuits reference counts."""
    targets = []
    if hex_counts:
        # (I^X^I).CCX(0,1,2).(I^X^H) -> |000> + |101>
        targets.append({'0x0': shots / 2, '0x5': shots / 2})
        # (I^I^X).CCX(0,1,2).(I^H^X) -> |000> + |110>
        targets.append({'0x0': shots / 2, '0x6': shots / 2})
        # (I^X^I).CCX(2,1,0).(H^X^I) -> |000> + |101>
        targets.append({'0x0': shots / 2, '0x5': shots / 2})
        # (X^I^I).CCX(2,1,0).(X^H^I) -> |000> + |011>
        targets.append({'0x0': shots / 2, '0x3': shots / 2})
    else:
        # (I^X^I).CCX(0,1,2).(I^X^H) -> |000> + |101>
        targets.append({'000': shots / 2, '101': shots / 2})
        # (I^I^X).CCX(0,1,2).(I^H^X) -> |000> + |110>
        targets.append({'000': shots / 2, '110': shots / 2})
        # (I^X^I).CCX(2,1,0).(H^X^I) -> |000> + |101>
        targets.append({'000': shots / 2, '101': shots / 2})
        # (X^I^I).CCX(2,1,0).(X^H^I) -> |000> + |011>
        targets.append({'000': shots / 2, '011': shots / 2})
    return targets


def ccx_gate_statevector_nondeterministic():
    """CCX-gate circuits reference statevectors."""
    targets = []
    # (I^X^I).CCX(0,1,2).(I^X^H) -> |000> + |101>
    targets.append(np.array([1, 0, 0, 0, 0, 1, 0, 0]) / np.sqrt(2))
    # (I^I^X).CCX(0,1,2).(I^H^X) -> |000> + |110>
    targets.append(np.array([1, 0, 0, 0, 0, 0, 1, 0]) / np.sqrt(2))
    # (I^X^I).CCX(2,1,0).(H^X^I) -> |000> + |101>
    targets.append(np.array([1, 0, 0, 0, 0, 1, 0, 0]) / np.sqrt(2))
    # (X^I^I).CCX(2,1,0).(X^H^I) -> |000> + |011>
    targets.append(np.array([1, 0, 0, 1, 0, 0, 0, 0]) / np.sqrt(2))
    return targets


def ccx_gate_unitary_nondeterministic():
    """CCX-gate circuits reference unitaries."""
    targets = []
    # (I^X^I).CCX(0,1,2).(I^X^H) -> |000> + |101>
    targets.append(np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, -1, 0, 0],
                             [0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 1, -1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0],
                             [1, -1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0, 0, 1, -1]]) / np.sqrt(2))
    # (I^I^X).CCX(0,1,2).(I^H^X) -> |000> + |110>
    targets.append(np.array([[1, 0, 1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, -1, 0],
                             [0, 1, 0, -1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1, 0, 1],
                             [1, 0, -1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, -1]]) / np.sqrt(2))
    # (I^X^I).CCX(2,1,0).(H^X^I) -> |000> + |101>
    targets.append(np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, -1, 0, 0],
                             [1, 0, 0, 0, -1, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, -1, 0],
                             [0, 0, 0, 1, 0, 0, 0, -1]]) / np.sqrt(2))
    # (X^I^I).CCX(2,1,0).(X^H^I) -> |000> + |011>
    targets.append(np.array([[1, 0, 1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 1, 0, 0, 0, 0],
                             [0, 1, 0, -1, 0, 0, 0, 0],
                             [1, 0, -1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 0, 1, 0, -1, 0],
                             [0, 0, 0, 0, 0, 1, 0, -1]]) / np.sqrt(2))
    return targets
