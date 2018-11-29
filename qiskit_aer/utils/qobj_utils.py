# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Temporary hacks for qobj until Terra supports Aer instructions (likely 0.8)

THESE SHOULD ONLY BE USED UNTIL A PROPER QUANTUM CIRCUIT INTERFACE
IS ADDED TO QISKIT TERRA. THEY WILL NOT BE SUPPORTED AFTER THAT.
"""

import copy
import numpy as np
from qiskit.qobj import QobjItem


def qobj_insert_item(qobj, exp_index, item, pos):
    """Insert a QobjItem into a Qobj experiment.

    Args:
        qobj (Qobj): a Qobj object
        exp_index (int): The index of the experiment in the qobj
        item (QobjItem): The Qobj item to insert
        pos (int): the position to insert the item.
    """
    qobj.experiments[exp_index].instructions.insert(pos, item)
    return qobj


def qobj_get_item_positions(qobj, exp_index, name):
    """Return all locations of QobjItem in a Qobj experiment.

    Args:
        qobj (Qobj): a Qobj object
        exp_index (int): The index of the experiment in the qobj
        name (str): QobjItem name to find

    Returns:
        list[int]: A list of positions where the QobjItem is located.
    """
    # Check only the name string of the item
    return [i for i, val in enumerate(qobj.experiments[exp_index].instructions)
            if val.name == name]


def qobj_get_specific_item_positions(qobj, exp_index, item):
    """Return all locations of QobjItem in a Qobj experiment.

    Args:
        qobj (Qobj): a Qobj object
        exp_index (int): The index of the experiment in the qobj
        item (QobjItem): The item to find

    Returns:
        list[int]: A list of positions where the QobjItem is located.
    """
    return [i for i, val in enumerate(qobj.experiments[exp_index].instructions)
            if val == item]


def qobj_insert_snapshots_after_barriers(qobj, snapshot):
    """Insert a snapshot instruction after each barrier in qobj.

    The label of the input snapshot will be appended with "i" where
    "i" ranges from 0 to the 1 - number of barriers.

    Args:
        snapshot (QobjItem): a snapshot instruction.

    Additional Information:
    """
    if snapshot.name != "snapshot":
        raise ValueError("Invalid snapshot instruction")
    label = snapshot.label
    for exp_index in range(len(qobj.experiments)):
        positions = qobj_get_item_positions(qobj, exp_index, "barrier")
        for i, pos in reversed(list(enumerate(positions))):
            item = copy.copy(snapshot)
            item.label = label + "{}".format(i)
            qobj_insert_item(qobj, exp_index, item, pos)
    return qobj


def qobj_unitary_item(mat, qubits, label=None):
    """Create a unitary gate qobj item.

    Args:
        mat (matrix_like): an n-qubit unitary matrix
        qubits (list[int]): qubits to apply the matrix to.
        label (str): optional string label for the untiary matrix

    Returns:
        QobjItem: The qobj item for the unitary instruction.

    Raises:
        ValueError: if the input matrix is not unitary

    Additional Information:

        Qubit Ordering:
            The n-qubit matrix is ordered in little-endian with respect to
            the qubits in the label string. For example. If M is a tensor
            product of single qubit matrices `M = kron(M_(n-1), ..., M_1, M_0)`
            then `M_0` is applied to `qubits[0]`, `M_1` to `qubits[1]` etc.

        Label string:
            The string label is used for identifying the matrix in a noise
            model so that noise may be applied to the implementation of
            this matrix.
    """
    array = np.array(mat, dtype=complex)
    dim = 2 ** len(qubits)
    if array.shape not in [(dim, dim), (1, dim)]:
        raise ValueError("Invalid")
    instruction = {"name": "unitary", "qubits": list(qubits),
                   "params": np.array(mat, dtype=complex)}
    if label is not None:
        instruction["label"] = str(label)
    return QobjItem(**instruction)


def qobj_snapshot_item(snapshot_type, label, qubits=None, params=None):
    """Create a snapshot qobj item.

    Args:
        snapshot_type (str): the snapshot type identifier
        label (str): the snapshot label string
        qubits (list[int]): qubits snapshot applies to (optional)
        params (custom): optional parameters for special snapshot types.
                         See additional information.

    Returns:
        QobjItem: The qobj item for the snapshot instruction.


    Additional Information:
        Snapshot types:
            "statevector" -- returns the current statevector for each shot
            "memory" -- returns the current memory hex-string for each shot
            "register" -- returns the current register hex-string for each shot
            "probabilities" -- returns the measurement outcome probabilities
                               averaged over all shots, but conditioned on the
                               current memory value.
                               This requires the qubits field to be set.
            "expval_pauli" -- returns the expectation value of an operator
                              averaged over all shots, but conditioned on the
                              current memory value.
                              This requires the qubits field to be set and
                              the params field to be set.
            "expval_matrix" -- same as expval_pauli but with different params

        Pauli expectation value params:
            These are a list of terms [complex_coeff, pauli_str]
            where string is in little endian: pauli_str CBA applies Pauli
            A to qubits[0], B to qubits[1] and C to qubits[2].
            Example for op 0.5 XX + 0.7 IZ we have [[0.5, 'XX'], [0.7, 'IZ']]

        Matrix expectation value params:
            TODO
    """
    snap = {"name": "snapshot", "type": snapshot_type, "label": str(label)}
    if qubits is not None:
        snap["qubits"] = list(qubits)
    if params is not None:
        snap["params"] = params
    # Check if single-matrix expectation value
    if snapshot_type in ["expval", "expval_matrix"] and \
       isinstance(params, np.ndarray):
        snap["name"] = "expval_matrix"
        snap["params"] = [[1.0, qubits, params]]
    # TODO: implicit conversion for Pauli expval params
    return QobjItem(**snap)


def qobj_measure_item(qubits, memory, registers=None):
    """Create a multi-qubit measure instruction"""
    if len(qubits) != len(memory):
        raise ValueError("Number of qubits does not match number of memory")
    if registers is None:
        return QobjItem(name='measure', qubits=qubits, memory=memory)
    # Case where we also measure to registers
    if len(qubits) != len(registers):
        raise ValueError("Number of qubits does not match number of registers")
    return QobjItem(name='measure', qubits=qubits, memory=memory,
                    register=registers)


def qobj_reset_item(qubits):
    """Create a multi-qubit reset instruction"""
    return QobjItem(name='reset', qubits=qubits)


def qobj_barrier_item(num_qubits):
    """Create a barrier QobjItem."""
    return QobjItem(name='barrier', qubits=list(range(num_qubits)))


def qobj_iden_item(qubit):
    """Create a barrier QobjItem."""
    return QobjItem(name='id', qubits=[qubit])
