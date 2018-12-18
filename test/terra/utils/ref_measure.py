# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test circuits and reference outputs for measure instruction.
"""

from numpy import array
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

# The following is temporarily needed for qobj compiling
# to test multi-qubit measurements which must be implemeted
# direclty by qobj instructions until terra compiler supports them
from qiskit import compile
from qiskit.providers.aer.backends import QasmSimulator
from qiskit.providers.aer.utils.qobj_utils import insert_instr
from qiskit.providers.aer.utils.qobj_utils import measure_instr
from qiskit.providers.aer.utils.qobj_utils import iden_instr


# ==========================================================================
# Deterministic output
# ==========================================================================

def measure_circuits_deterministic(allow_sampling=True):
    """Measure test circuits with deterministic count output."""

    circuits = []
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)

    # Measure |00> state
    circuit = QuantumCircuit(qr, cr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.iden(qr)
    circuits.append(circuit)

    # Measure |01> state
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.iden(qr)
    circuits.append(circuit)

    # Measure |10> state
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.iden(qr)
    circuits.append(circuit)

    # Measure |11> state
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.iden(qr)
    circuits.append(circuit)

    return circuits


def measure_counts_deterministic(shots, hex_counts=True):
    """Measure test circuits reference counts."""

    targets = []
    if hex_counts:
        # Measure |00> state
        targets.append({'0x0': shots})
        # Measure |01> state
        targets.append({'0x1': shots})
        # Measure |10> state
        targets.append({'0x2': shots})
        # Measure |11> state
        targets.append({'0x3': shots})
    else:
        # Measure |00> state
        targets.append({'00': shots})
        # Measure |01> state
        targets.append({'01': shots})
        # Measure |10> state
        targets.append({'10': shots})
        # Measure |11> state
        targets.append({'11': shots})
    return targets


def measure_statevector_deterministic():
    """Measure test circuits reference counts."""

    targets = []
    # Measure |00> state
    targets.append(array([1, 0, 0, 0]))
    # Measure |01> state
    targets.append(array([0, 1, 0, 0]))
    # Measure |10> state
    targets.append(array([0, 0, 1, 0]))
    # Measure |11> state
    targets.append(array([0, 0, 0, 1]))
    return targets


# ==========================================================================
# Non-Deterministic output
# ==========================================================================

def measure_circuits_nondeterministic(allow_sampling=True):
    """"Measure test circuits with non-deterministic count output."""

    circuits = []
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)

    # Measure |++> state (sampled)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.iden(qr)
    circuits.append(circuit)

    return circuits


def measure_counts_nondeterministic(shots, hex_counts=True):
    """Measure test circuits reference counts."""

    targets = []
    if hex_counts:
        # Measure |++> state
        targets.append({'0x0': shots / 4, '0x1': shots / 4,
                        '0x2': shots / 4, '0x3': shots / 4})
    else:
        # Measure |++> state
        targets.append({'00': shots / 4, '01': shots / 4,
                        '10': shots / 4, '11': shots / 4})
    return targets


# ==========================================================================
# Multi-qubit qobj item deterministic output
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


def measure_circuits_qobj_deterministic(allow_sampling=True):
    """Measure test circuits with deterministic count output."""

    # Dummy qobj
    final_qobj = _dummy_qobj()

    # 2-qubit measure |10>
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    insert_instr(qobj, 0, measure_instr([0, 1], [0, 1]), -1)
    if not allow_sampling:
        insert_instr(qobj, 0, iden_instr(0), -1)
    final_qobj.experiments.append(qobj.experiments[0])

    # 3-qubit measure |101>
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[0])
    circuit.x(qr[2])
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    insert_instr(qobj, 0, measure_instr([0, 1, 2], [0, 1, 2]), -1)
    if not allow_sampling:
        insert_instr(qobj, 0, iden_instr(0), -1)
    final_qobj.experiments.append(qobj.experiments[0])

    # 4-qubit measure |1010>
    qr = QuantumRegister(4)
    cr = ClassicalRegister(4)
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[1])
    circuit.x(qr[3])
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    insert_instr(qobj, 0, measure_instr([0, 1, 2, 3], [0, 1, 2, 3]), -1)
    if not allow_sampling:
        insert_instr(qobj, 0, iden_instr(0), -1)
    final_qobj.experiments.append(qobj.experiments[0])

    return final_qobj


def measure_counts_qobj_deterministic(shots, hex_counts=True):
    """Measure test circuits reference counts."""

    targets = []
    if hex_counts:
        # 2-qubit measure |10>
        targets.append({'0x2': shots})
        # 3-qubit measure |101>
        targets.append({'0x5': shots})
        # 4-qubit measure |1010>
        targets.append({'0xa': shots})
    else:
        # 2-qubit measure |10>
        targets.append({'10': shots})
        # 3-qubit measure |101>
        targets.append({'101': shots})
        # 4-qubit measure |1010>
        targets.append({'1010': shots})
    return targets


def measure_statevector_qobj_deterministic():
    """Measure test circuits reference counts."""

    targets = []
    # 2-qubit measure |10>
    targets.append(array([0, 0, 1, 0]))
    # 3-qubit measure |101>
    targets.append(array([0, 0, 0, 0, 0, 1, 0, 0]))
    # 4-qubit measure |1010>
    targets.append(array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
    return targets


def measure_circuits_qobj_nondeterministic(allow_sampling=True):
    """Measure test circuits with deterministic count output."""

    # Dummy qobj
    final_qobj = _dummy_qobj()

    # 2-qubit measure |++>
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    insert_instr(qobj, 0, measure_instr([0, 1], [0, 1]), -1)
    if not allow_sampling:
        insert_instr(qobj, 0, iden_instr(0), -1)
    final_qobj.experiments.append(qobj.experiments[0])

    # 3-qubit measure |++0>
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.barrier(qr)
    qobj = compile(circuit, QasmSimulator(), shots=1)
    insert_instr(qobj, 0, measure_instr([0, 1, 2], [0, 1, 2]), -1)
    if not allow_sampling:
        insert_instr(qobj, 0, iden_instr(0), -1)
    final_qobj.experiments.append(qobj.experiments[0])

    return final_qobj


def measure_counts_qobj_nondeterministic(shots, hex_counts=True):
    """Measure test circuits reference counts."""

    targets = []
    if hex_counts:
        # 2-qubit measure |++>
        targets.append({'0x0': shots / 4, '0x1': shots / 4,
                        '0x2': shots / 4, '0x3': shots / 4})    
        # 3-qubit measure |0++>
        targets.append({'0x0': shots / 4, '0x1': shots / 4,
                        '0x2': shots / 4, '0x3': shots / 4})
    else:
        # 2-qubit measure |++>
        targets.append({'00': shots / 4, '01': shots / 4,
                        '10': shots / 4, '11': shots / 4})
        # 3-qubit measure |0++>
        targets.append({'000': shots / 4, '001': shots / 4,
                        '010': shots / 4, '011': shots / 4})
    return targets
