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
Test circuits and reference outputs for snapshot amplitude instructions.
"""

from numpy import array, sqrt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.extensions.snapshot import Snapshot
from qiskit.providers.aer.extensions.snapshot_amplitudes import *
from qiskit.providers.aer.extensions.snapshot_statevector import *

def snapshot_amplitudes_labels_params():
    """Dictionary of labels and params for 3-qubit amplitude snapshots"""
    return {
        "[0]": [0],
        "[7]": [7],
        "[0,1]": [0,1],
        "[7,3,5,1]": [7,3,5,1],
        "[4,1,5]": [4,1,5],
        "[6,2]": [6,2],
        "all": [0,1,2,3,4,5,6,7],
        "[0x7]": [0x7],
        "[0x2, 0x4]": [0x2, 0x4]
    }

# Verify the snapshot_amplitudes by comparing with the corresponding amplitudes
# in snapshot_statevector
def snapshot_amplitudes_circuits(post_measure=False):
    """Snapshot Amplitudes test circuits"""

    circuits = []
    num_qubits = 3
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    regs = (qr, cr)

    # Amplitudes snapshot instruction acting on all qubits

    # Snapshot |000>
    circuit = QuantumCircuit(*regs)
    if not post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params)
            circuit.snapshot_statevector(label)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params)
            circuit.snapshot_statevector(label)

    circuits.append(circuit)

    # Snapshot |111>
    circuit = QuantumCircuit(*regs)
    circuit.x(qr)
    if not post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params)
            circuit.snapshot_statevector(label)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params)
            circuit.snapshot_statevector(label)

    circuits.append(circuit)

    # Snapshot 0.25*(|001>+|011>+|100>+|101>)
    circuit = QuantumCircuit(*regs)
    circuit.h(0)
    circuit.h(2)
    if not post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params)
            circuit.snapshot_statevector(label)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if post_measure:
        for label, params in snapshot_amplitudes_labels_params().items():
            circuit.snapshot_amplitudes(label, params)
            circuit.snapshot_statevector(label)

    circuits.append(circuit)

    return circuits


def snapshot_amplitudes_counts(shots):
    """Snapshot Amplitudes test circuits reference counts."""
    targets = []
    # Snapshot |000>
    targets.append({'0x0': shots})
    # Snapshot |111>
    targets.append({'0x7': shots})
    # Snapshot 0.25*(|001>+|011>+|100>+|101>)
    targets.append({'0x0': shots/4, '0x1': shots/4, '0x4': shots/4, '0x5': shots/4,})
    return targets


