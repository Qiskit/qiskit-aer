# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test circuits and reference outputs for measure instruction.
"""


import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, compile

from qiskit.providers.aer.backends import QasmSimulator
from qiskit.providers.aer.utils.qobj_utils import unitary_instr
from qiskit.providers.aer.utils.qobj_utils import append_instr
from qiskit.providers.aer.utils.qobj_utils import measure_instr




# ==========================================================================
# Multi-qubit measure
# ==========================================================================

def _dummy_qobj():
    """Return a dummy qobj to insert experiments into"""
    qr = QuantumRegister(1)
    circuit = QuantumCircuit(qr)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    # remove experiment
    qobj.experiments = []
    return qobj


def unitary_gate_circuits_real_deterministic(final_measure=True):
    """Unitary gate test circuits with deterministic count output."""

    final_qobj = _dummy_qobj()
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr, )
    x_mat = np.array([[0, 1], [1, 0]])
    cx_mat = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

    # CX01, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(cx_mat, [0, 1]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX10, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(cx_mat, [1, 0]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX01.(X^I), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(x_mat, [1]))
    append_instr(qobj, 0, unitary_instr(cx_mat, [0, 1]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX10.(I^X), |01> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(x_mat, [0]))
    append_instr(qobj, 0, unitary_instr(cx_mat, [1, 0]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX01.(I^X), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(x_mat, [0]))
    append_instr(qobj, 0, unitary_instr(cx_mat, [0, 1]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX10.(X^I), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(x_mat, [1]))
    append_instr(qobj, 0, unitary_instr(cx_mat, [1, 0]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    return final_qobj


def unitary_gate_counts_real_deterministic(shots, hex_counts=True):
    """Unitary gate circuits reference counts."""
    targets = []
    if hex_counts:
        # CX01, |00> state
        targets.append({'0x0': shots})  # {"00": shots}
        # CX10, |00> state
        targets.append({'0x0': shots})  # {"00": shots}
        # CX01.(X^I), |10> state
        targets.append({'0x2': shots})  # {"00": shots}
        # CX10.(I^X), |01> state
        targets.append({'0x1': shots})  # {"00": shots}
        # CX01.(I^X), |11> state
        targets.append({'0x3': shots})  # {"00": shots}
        # CX10.(X^I), |11> state
        targets.append({'0x3': shots})  # {"00": shots}
    else:
        # CX01, |00> state
        targets.append({'00': shots})  # {"00": shots}
        # CX10, |00> state
        targets.append({'00': shots})  # {"00": shots}
        # CX01.(X^I), |10> state
        targets.append({'10': shots})  # {"00": shots}
        # CX10.(I^X), |01> state
        targets.append({'01': shots})  # {"00": shots}
        # CX01.(I^X), |11> state
        targets.append({'11': shots})  # {"00": shots}
        # CX10.(X^I), |11> state
    return targets


def unitary_gate_statevector_real_deterministic():
    """Unitary gate test circuits with deterministic counts."""
    targets = []
    # CX01, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX10, |00> state
    targets.append(np.array([1, 0, 0, 0]))
    # CX01.(X^I), |10> state
    targets.append(np.array([0, 0, 1, 0]))
    # CX10.(I^X), |01> state
    targets.append(np.array([0, 1, 0, 0]))
    # CX01.(I^X), |11> state
    targets.append(np.array([0, 0, 0, 1]))
    # CX10.(X^I), |11> state
    targets.append(np.array([0, 0, 0, 1]))
    return targets


def unitary_gate_unitary_real_deterministic():
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
    # CX01.(X^I), |10> state
    targets.append(np.array([[0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1]]))
    # CX10.(I^X), |01> state
    targets.append(np.array([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]))
    # CX01.(I^X), |11> state
    targets.append(np.array([[0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [1, 0, 0, 0]]))
    # CX10.(X^I), |11> state
    targets.append(np.array([[0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0]]))
    return targets


def unitary_gate_circuits_complex_deterministic(final_measure=True):
    """Unitary gate test circuits with deterministic count output."""

    final_qobj = _dummy_qobj()
    qr = QuantumRegister(2)
    if final_measure:
        cr = ClassicalRegister(2)
        regs = (qr, cr)
    else:
        regs = (qr, )
    y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
    cx_mat = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                      dtype=complex)

    # CX01, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(cx_mat, [0, 1]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX10, |00> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(cx_mat, [1, 0]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX01.(Y^I), |10> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(y_mat, [1]))
    append_instr(qobj, 0, unitary_instr(cx_mat, [0, 1]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX10.(I^Y), |01> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(y_mat, [0]))
    append_instr(qobj, 0, unitary_instr(cx_mat, [1, 0]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX01.(I^Y), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(y_mat, [0]))
    append_instr(qobj, 0, unitary_instr(cx_mat, [0, 1]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    # CX10.(Y^I), |11> state
    circuit = QuantumCircuit(*regs)
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    append_instr(qobj, 0, unitary_instr(y_mat, [1]))
    append_instr(qobj, 0, unitary_instr(cx_mat, [1, 0]))
    if final_measure:
        append_instr(qobj, 0, measure_instr([0], [0]))
        append_instr(qobj, 0, measure_instr([1], [1]))
    final_qobj.experiments.append(qobj.experiments[0])

    return final_qobj


def unitary_gate_counts_complex_deterministic(shots, hex_counts=True):
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


def unitary_gate_statevector_complex_deterministic():
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


def unitary_gate_unitary_complex_deterministic():
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
