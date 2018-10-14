# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Temporary hacks for qobj until Terra supports Aer instructions.
"""

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


def get_item_positions(qobj, exp_index, item):
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


def qobj_barrier(num_qubits):
    """Create a barrier QobjItem."""
    return QobjItem(name='barrier', qubits=list(range(num_qubits)))


def qobj_unitary(mat, qubits, label=None):
    """Create a unitary gate qobj item."""
    instruction = {"name": "mat", "qubits": list(qubits),
                   "params": np.array(mat, dtype=complex)}
    if label is not None:
        instruction["label"] = str(label)
    return QobjItem(**instruction)


def qobj_snapshot_state(label):
    """Create a state snapshot QobjItem."""
    return QobjItem(**{"name": "snapshot", "type": "state", "label": str(label)})


def qobj_snapshot_probs(label, qubits):
    """Create a probabilities snapshot QobjItem."""
    return QobjItem(**{"name": "snapshot", "type": "probabilities",
                       "qubits": list(qubits), "label": str(label)})


def qobj_snapshot_pauli(label, params):
    """Create a Pauli expectation value snapshot QobjItem."""
    return QobjItem(**{"name": "snapshot", "type": "pauli_observable",
                       "params": list(params), "label": str(label)})


def qobj_snapshot_matrix(label, params):
    """Create a matrix expectation value snapshot QobjItem."""
    return QobjItem(**{"name": "snapshot", "type": "matrix_observable",
                       "params": list(params), "label": str(label)})
