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
Test circuits and reference outputs for snapshot state instructions.
"""

from numpy import array, sqrt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.extensions.snapshot import Snapshot
from qiskit.providers.aer.extensions.snapshot_statevector import *


def snapshot_state_circuits_deterministic(snapshot_label='snap',
                                          snapshot_type='statevector',
                                          post_measure=False):
    """Snapshot Statevector test circuits"""

    circuits = []
    num_qubits = 3
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    regs = (qr, cr)

    # State snapshot instruction acting on all qubits
    snapshot = Snapshot(snapshot_label, snapshot_type, num_qubits)

    # Snapshot |000>
    circuit = QuantumCircuit(*regs)
    if not post_measure:
        circuit.append(snapshot, qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        circuit.append(snapshot, qr)
    circuits.append(circuit)

    # Snapshot |111>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    if not post_measure:
        circuit.append(snapshot, qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        circuit.append(snapshot, qr)
    circuits.append(circuit)

    return circuits


def snapshot_state_counts_deterministic(shots):
    """Snapshot Statevector test circuits reference counts."""
    targets = []
    # Snapshot |000>
    targets.append({'0x0': shots})
    # Snapshot |111>
    targets.append({'0x7': shots})
    return targets


def snapshot_state_pre_measure_statevector_deterministic():
    """Snapshot Statevector test circuits reference final statevector"""
    targets = []
    # Snapshot |000>
    targets.append(array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex))
    # Snapshot |111>
    targets.append(array([0, 0, 0, 0, 0, 0, 0, 1], dtype=complex))
    return targets


def snapshot_state_post_measure_statevector_deterministic():
    """Snapshot Statevector test circuits reference final statevector"""

    targets = []
    # Snapshot |000>
    targets.append({'0x0': array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)})
    # Snapshot |111>
    targets.append({'0x7': array([0, 0, 0, 0, 0, 0, 0, 1], dtype=complex)})
    return targets


def snapshot_state_circuits_nondeterministic(snapshot_label='snap',
                                             snapshot_type='statevector',
                                             post_measure=False):
    """Snapshot Statevector test circuits"""

    circuits = []
    num_qubits = 3
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    regs = (qr, cr)

    # State snapshot instruction acting on all qubits
    snapshot = Snapshot(snapshot_label, snapshot_type, num_qubits)

    # Snapshot |000> + i|111>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr[0])
    circuit.s(qr[0])
    circuit.cx(qr[0], qr[1])
    circuit.cx(qr[0], qr[2])
    if not post_measure:
        circuit.append(snapshot, qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        circuit.append(snapshot, qr)
    circuits.append(circuit)

    # Snapshot |+++>
    circuit = QuantumCircuit(*regs)
    circuit.h(qr)
    if not post_measure:
        circuit.append(snapshot, qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        circuit.append(snapshot, qr)
    circuits.append(circuit)

    return circuits


def snapshot_state_counts_nondeterministic(shots):
    """Snapshot Statevector test circuits reference counts."""
    targets = []
    # Snapshot |000> + i|111>
    targets.append({'0x0': shots/2,
                    '0x7': shots/2})
    # Snapshot |+++>
    targets.append({'0x0': shots/8,
                    '0x1': shots/8,
                    '0x2': shots/8,
                    '0x3': shots/8,
                    '0x4': shots/8,
                    '0x5': shots/8,
                    '0x6': shots/8,
                    '0x7': shots/8})
    return targets


def snapshot_state_pre_measure_statevector_nondeterministic():
    """Snapshot Statevector test circuits reference final statevector"""
    targets = []
    # Snapshot |000> + i|111>
    targets.append(array([1, 0, 0, 0, 0, 0, 0, 1j], dtype=complex) / sqrt(2))
    # Snapshot |+++>
    targets.append(array([1, 1, 1, 1, 1, 1, 1, 1], dtype=complex) / sqrt(8))
    return targets


def snapshot_state_post_measure_statevector_nondeterministic():
    """Snapshot Statevector test circuits reference final statevector"""

    targets = []
    # Snapshot |000> + i|111>
    targets.append({'0x0': array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex),
                    '0x7': array([0, 0, 0, 0, 0, 0, 0, 1j], dtype=complex)})
    # Snapshot |+++>
    targets.append({'0x0': array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex),
                    '0x1': array([0, 1, 0, 0, 0, 0, 0, 0], dtype=complex),
                    '0x2': array([0, 0, 1, 0, 0, 0, 0, 0], dtype=complex),
                    '0x3': array([0, 0, 0, 1, 0, 0, 0, 0], dtype=complex),
                    '0x4': array([0, 0, 0, 0, 1, 0, 0, 0], dtype=complex),
                    '0x5': array([0, 0, 0, 0, 0, 1, 0, 0], dtype=complex),
                    '0x6': array([0, 0, 0, 0, 0, 0, 1, 0], dtype=complex),
                    '0x7': array([0, 0, 0, 0, 0, 0, 0, 1], dtype=complex)})
    return targets

