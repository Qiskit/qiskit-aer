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
Temporary hacks for qobj until Terra supports Aer instructions (likely 0.8)

THESE SHOULD ONLY BE USED UNTIL A PROPER QUANTUM CIRCUIT INTERFACE
IS ADDED TO QISKIT TERRA. THEY WILL NOT BE SUPPORTED AFTER THAT.
"""

import copy
import warnings
import numpy as np
from qiskit.qobj import QasmQobjInstruction


def append_instr(qobj, exp_index, instruction):
    """Append a QasmQobjInstruction to a QobjExperiment.

    Args:
        qobj (Qobj): a Qobj object.
        exp_index (int): The index of the experiment in the qobj.
        instruction (QasmQobjInstruction): instruction to insert.

    Returns:
        qobj(Qobj): The Qobj object
    """
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    qobj.experiments[exp_index].instructions.append(instruction)
    return qobj


def insert_instr(qobj, exp_index, item, pos):
    """Insert a QasmQobjInstruction into a QobjExperiment.

    Args:
        qobj (Qobj): a Qobj object
        exp_index (int): The index of the experiment in the qobj.
        item (QasmQobjInstruction): instruction to insert.
        pos (int): the position to insert the item.

    Returns:
        qobj(Qobj): The Qobj object
    """
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    qobj.experiments[exp_index].instructions.insert(pos, item)
    return qobj


def get_instr_pos(qobj, exp_index, name):
    """Return all locations of QasmQobjInstruction in a Qobj experiment.

    The return list is sorted in reverse order so iterating over it
    to insert new items will work as expected.

    Args:
        qobj (Qobj): a Qobj object
        exp_index (int): The index of the experiment in the qobj
        name (str): QasmQobjInstruction name to find

    Returns:
        list[int]: A list of positions where the QasmQobjInstruction is located.
    """
    warnings.warn(
        'This funnction is deprecated and will be removed in a future release.',
        DeprecationWarning)
    # Check only the name string of the item
    positions = [
        i for i, val in enumerate(qobj.experiments[exp_index].instructions)
        if val.name == name
    ]
    return positions


def unitary_instr(mat, qubits, label=None):
    """Create a unitary gate QasmQobjInstruction.

    Args:
        mat (matrix_like): an n-qubit unitary matrix
        qubits (list[int]): qubits to apply the matrix to.
        label (str): optional string label for the unitary matrix

    Returns:
        QasmQobjInstruction: The qobj item for the unitary instruction.

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
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    array = np.array(mat, dtype=complex)
    dim = 2**len(qubits)
    if array.shape not in [(dim, dim), (1, dim)]:
        raise ValueError("Invalid")
    instruction = {
        "name": "unitary",
        "qubits": list(qubits),
        "params": [np.array(mat, dtype=complex)]
    }
    if label is not None:
        instruction["label"] = str(label)
    return QasmQobjInstruction(**instruction)


def measure_instr(qubits, memory, registers=None):
    """Create a multi-qubit measure instruction"""
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    if len(qubits) != len(memory):
        raise ValueError("Number of qubits does not match number of memory")
    if registers is None:
        return QasmQobjInstruction(name='measure',
                                   qubits=qubits,
                                   memory=memory)
    # Case where we also measure to registers
    if len(qubits) != len(registers):
        raise ValueError("Number of qubits does not match number of registers")
    return QasmQobjInstruction(name='measure',
                               qubits=qubits,
                               memory=memory,
                               register=registers)


def reset_instr(qubits):
    """Create a multi-qubit reset instruction"""
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    return QasmQobjInstruction(name='reset', qubits=qubits)


def barrier_instr(num_qubits):
    """Create a barrier QasmQobjInstruction."""
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    return QasmQobjInstruction(name='barrier', qubits=list(range(num_qubits)))


def iden_instr(qubit):
    """Create a barrier QasmQobjInstruction."""
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    return QasmQobjInstruction(name='id', qubits=[qubit])


def snapshot_instr(snapshot_type, label, qubits=None, params=None):
    """Create a snapshot qobj item.

    Args:
        snapshot_type (str): the snapshot type identifier
        label (str): the snapshot label string
        qubits (list[int]): qubits snapshot applies to (optional)
        params (custom): optional parameters for special snapshot types.
                         See additional information.

    Returns:
        QasmQobjInstruction: The qobj item for the snapshot instruction.


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
    warnings.warn(
        'This function is deprecated and will be removed in a future release.'
        ' Use the snapshot circuit instructions in'
        ' `qiskit.provider.aer.extensions` instead.', DeprecationWarning)
    snap = {
        "name": "snapshot",
        "snapshot_type": snapshot_type,
        "label": str(label)
    }
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
    return QasmQobjInstruction(**snap)


def insert_snapshots_after_barriers(qobj, snapshot):
    """Insert a snapshot instruction after each barrier in qobj.

    The label of the input snapshot will be appended with "i" where
    "i" ranges from 0 to the 1 - number of barriers.

    Args:
        qobj (Qobj): a qobj to insert snapshots into
        snapshot (QasmQobjInstruction): a snapshot instruction.

    Returns:
        qobj(Qobj): The Qobj object

    Raises:
        ValueError: if the name of the instruction is not an snapshot

    Additional Information:
    """
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    if snapshot.name != "snapshot":
        raise ValueError("Invalid snapshot instruction")
    label = snapshot.label
    for exp_index in range(len(qobj.experiments)):
        positions = get_instr_pos(qobj, exp_index, "barrier")
        for i, pos in reversed(list(enumerate(positions))):
            item = copy.copy(snapshot)
            item.label = label + "{}".format(i)
            insert_instr(qobj, exp_index, item, pos)
    return qobj
