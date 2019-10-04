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

from numpy import array
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.extensions.snapshot_probabilities import *


def snapshot_probabilities_labels_qubits():
    """Dictionary of labels and qubits for 3-qubit probability snapshots"""
    return {
        "[0]": [0],
        "[1]": [1],
        "[2]": [2],
        "[0, 1]": [0, 1],
        "[1, 0]": [1, 0],
        "[0, 2]": [0, 2],
        "[2, 0]": [2, 0],
        "[1, 2]": [1, 2],
        "[2, 1]": [2, 1],
        "[0, 1, 2]": [0, 1, 2],
        "[1, 2, 0]": [1, 2, 0],
        "[2, 0, 1]": [2, 0, 1]
    }


def snapshot_probabilities_circuits(post_measure=False):
    """Snapshot Probabilities test circuits with deterministic counts"""

    circuits = []
    num_qubits = 3
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    regs = (qr, cr)

    # State |01+>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.x(1)
    if not post_measure:
        for label, qubits in snapshot_probabilities_labels_qubits().items():
            circuit.snapshot_probabilities(label, qubits)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuit.barrier(qr)
    if post_measure:
        for label, qubits in snapshot_probabilities_labels_qubits().items():
            circuit.snapshot_probabilities(label, qubits)
    circuits.append(circuit)

    # State |010> -i|101>
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.sdg(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.x(1)
    if not post_measure:
        for label, qubits in snapshot_probabilities_labels_qubits().items():
            circuit.snapshot_probabilities(label, qubits)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuit.barrier(qr)
    if post_measure:
        for label, qubits in snapshot_probabilities_labels_qubits().items():
            circuit.snapshot_probabilities(label, qubits)
    circuits.append(circuit)

    return circuits


def snapshot_probabilities_counts(shots):
    """Snapshot Probabilities test circuits reference counts."""
    targets = []
    # State |01+>
    targets.append({'0x2': shots / 2, '0x3': shots / 2})

    # State |010> -i|101>
    targets.append({'0x2': shots / 2, '0x5': shots / 2})
    return targets


def snapshot_probabilities_pre_meas_probs():
    """Snapshot Probabilities test circuits reference final probs"""
    targets = []

    # State |01+>
    probs = {
        "[0]": {'0x0': {'0x0': 0.5, '0x1': 0.5}},
        "[1]": {'0x0': {'0x1': 1.0}},
        "[2]": {'0x0': {'0x0': 1.0}},
        "[0, 1]": {'0x0': {'0x2': 0.5, '0x3': 0.5}},
        "[1, 0]": {'0x0': {'0x1': 0.5, '0x3': 0.5}},
        "[0, 2]": {'0x0': {'0x0': 0.5, '0x1': 0.5}},
        "[2, 0]": {'0x0': {'0x0': 0.5, '0x2': 0.5}},
        "[1, 2]": {'0x0': {'0x1': 1.0}},
        "[2, 1]": {'0x0': {'0x2': 1.0}},
        "[0, 1, 2]": {'0x0': {'0x2': 0.5, '0x3': 0.5}},
        "[1, 2, 0]": {'0x0': {'0x1': 0.5, '0x5': 0.5}},
        "[2, 0, 1]": {'0x0': {'0x4': 0.5, '0x6': 0.5}},
    }
    targets.append(probs)

    # State |010> -i|101>
    probs = {
        "[0]": {'0x0': {'0x0': 0.5, '0x1': 0.5}},
        "[1]": {'0x0': {'0x0': 0.5, '0x1': 0.5}},
        "[2]": {'0x0': {'0x0': 0.5, '0x1': 0.5}},
        "[0, 1]": {'0x0': {'0x1': 0.5, '0x2': 0.5}},
        "[1, 0]": {'0x0': {'0x1': 0.5, '0x2': 0.5}},
        "[0, 2]": {'0x0': {'0x0': 0.5, '0x3': 0.5}},
        "[2, 0]": {'0x0': {'0x0': 0.5, '0x3': 0.5}},
        "[1, 2]": {'0x0': {'0x1': 0.5, '0x2': 0.5}},
        "[2, 1]": {'0x0': {'0x1': 0.5, '0x2': 0.5}},
        "[0, 1, 2]": {'0x0': {'0x2': 0.5, '0x5': 0.5}},
        "[1, 2, 0]": {'0x0': {'0x1': 0.5, '0x6': 0.5}},
        "[2, 0, 1]": {'0x0': {'0x3': 0.5, '0x4': 0.5}},
    }
    targets.append(probs)
    return targets


def snapshot_probabilities_post_meas_probs():
    """Snapshot Probabilities test circuits reference final statevector"""
    targets = []

    # State |01+>
    probs = {
        "[0]": {'0x2': {'0x0': 1.0}, '0x3': {'0x1': 1.0}},
        "[1]": {'0x2': {'0x1': 1.0}, '0x3': {'0x1': 1.0}},
        "[2]": {'0x2': {'0x0': 1.0}, '0x3': {'0x0': 1.0}},
        "[0, 1]": {'0x2': {'0x2': 1.0}, '0x3': {'0x3': 1.0}},
        "[1, 0]": {'0x2': {'0x1': 1.0}, '0x3': {'0x3': 1.0}},
        "[0, 2]": {'0x2': {'0x0': 1.0}, '0x3': {'0x1': 1.0}},
        "[2, 0]": {'0x2': {'0x0': 1.0}, '0x3': {'0x2': 1.0}},
        "[1, 2]": {'0x2': {'0x1': 1.0}, '0x3': {'0x1': 1.0}},
        "[2, 1]": {'0x2': {'0x2': 1.0}, '0x3': {'0x2': 1.0}},
        "[0, 1, 2]": {'0x2': {'0x2': 1.0}, '0x3': {'0x3': 1.0}},
        "[1, 2, 0]": {'0x2': {'0x1': 1.0}, '0x3': {'0x5': 1.0}},
        "[2, 0, 1]": {'0x2': {'0x4': 1.0}, '0x3': {'0x6': 1.0}},
    }
    targets.append(probs)

    # State |010> -i|101>
    probs = {
        "[0]": {'0x2': {'0x0': 1.0}, '0x5': {'0x1': 1.0}},
        "[1]": {'0x2': {'0x1': 1.0}, '0x5': {'0x0': 1.0}},
        "[2]": {'0x2': {'0x0': 1.0}, '0x5': {'0x1': 1.0}},
        "[0, 1]": {'0x2': {'0x2': 1.0}, '0x5': {'0x1': 1.0}},
        "[1, 0]": {'0x2': {'0x1': 1.0}, '0x5': {'0x2': 1.0}},
        "[0, 2]": {'0x2': {'0x0': 1.0}, '0x5': {'0x3': 1.0}},
        "[2, 0]": {'0x2': {'0x0': 1.0}, '0x5': {'0x3': 1.0}},
        "[1, 2]": {'0x2': {'0x1': 1.0}, '0x5': {'0x2': 1.0}},
        "[2, 1]": {'0x2': {'0x2': 1.0}, '0x5': {'0x1': 1.0}},
        "[0, 1, 2]": {'0x2': {'0x2': 1.0}, '0x5': {'0x5': 1.0}},
        "[1, 2, 0]": {'0x2': {'0x1': 1.0}, '0x5': {'0x6': 1.0}},
        "[2, 0, 1]": {'0x2': {'0x4': 1.0}, '0x5': {'0x3': 1.0}},
    }
    targets.append(probs)
    return targets
