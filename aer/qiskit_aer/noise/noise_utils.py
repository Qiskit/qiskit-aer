# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Helper functions for noise model creation.
"""

import numpy as np

from .aernoiseerror import AerNoiseError


def make_unitary_instruction(mat, qubits, threshold=1e-10):
    """Return a qobj instruction for a unitary matrix gate.

    Args:
        mat (matrix): A square or diagonal unitary matrix.
        qubits (list[int]): The qubits the matrix is applied to.
        threshold (double): The threshold parameter for testing if the
                            input matrix is unitary (default: 1e-10).
    Returns:
        dict: The qobj instruction object.

    Raises:
        AerNoiseError: if the input is not a unitary matrix.
    """
    if is_unitary(mat, threshold) is False:
        raise AerNoiseError("Input matrix is not unitary.")
    elif isinstance(qubits, int):
        qubits = [qubits]
    return {"name": "mat", "qubits": qubits, "params": mat}


def make_kraus_instruction(mats, qubits, threshold=1e-10):
    """Return a qobj instruction for a Kraus error.

    Args:
        mats (list[matrix]): A list of square or diagonal Kraus matrices.
        qubits (list[int]): The qubits the matrix is applied to.
        threshold (double): The threshold parameter for testing if the
                            Kraus matrices are unitary (default: 1e-10).
    Returns:
        dict: The qobj instruction object.

    Raises:
        AerNoiseError: if the input is not a CPTP Kraus channel.
    """
    if is_cptp(mats, threshold) is False:
        raise AerNoiseError("Input Kraus ops are not a CPTP channel.")
    elif isinstance(qubits, int):
        qubits = [qubits]
    return {"name": "kraus", "qubits": qubits, "params": mats}


def qubits_from_mat(mat):
    """Return the number of qubits for a multi-qubit matrix."""
    arr = np.array(mat)
    shape = arr.shape
    num_qubits = int(np.log2(shape[1]))
    if shape[1] != 2 ** num_qubits:
        raise AerNoiseError("Input Kraus channel is not a multi-qubit channel.")
    return num_qubits


def is_cptp(kraus_ops, threshold=1e-10):
    """Test if a list of Kraus matrices is a CPTP map."""
    accum = 0
    for op in kraus_ops:
        if is_diagonal(op):
            op = np.diag(op[0])
        accum += op.T.conj().dot(op)
    return (np.linalg.norm(accum - np.eye(len(accum))) <= threshold)


def is_square(op):
    """Test if an array is a square matrix."""
    mat = np.array(op)
    shape = mat.shape
    return len(shape) == 2 and shape[0] == shape[1]


def is_diagonal(op):
    """Test if an array is a diagonal matrix."""
    mat = np.array(op)
    shape = mat.shape
    return len(shape) == 2 and shape[0] == 1


def is_identity(op, threshold=1e-10):
    """Test if an array is an identity matrix."""
    mat = np.array(op)
    if is_diagonal(mat):
        # Check if diagonal identity: [[1, 1, ...]]
        diag = mat[0]
        iden = np.ones(len(diag))
        return (np.linalg.norm(diag - iden) <= threshold)
    if is_square(mat) is False:
        return False
    # Check if square identity
    iden = np.eye(len(mat))
    return (np.linalg.norm(mat - iden) <= threshold)


def is_unitary(op, threshold=1e-10):
    """Test if an array is a unitary matrix."""
    mat = np.array(op)
    if is_diagonal(mat):
        return is_identity(np.conj(mat) * mat, threshold)
    else:
        return is_identity(np.conj(mat.T).dot(mat), threshold)


def kraus2instructions(kraus_ops, threshold=1e-10):
    """
    Convert a list of Kraus matrices into qobj circuits.

    If any Kraus operators are a unitary matrix they will be converted
    into unitary qobj instructions. Identity unitary matrices will also be
    converted into identity qobj instructions.

    Args:
        kraus_ops (list[matrix]): A list of Kraus matrices for a CPTP map.
        threshold (double): The threshold for testing if Kraus matrices are
                            unitary or identity matrices (default: 1e-10).

    Returns:
        A list of pairs (p, circuit) where `circuit` is a list of qobj
        instructions, and `p` is the probability of that circuit for the
        given error.

    Raises:
        AerNoiseError: If the input Kraus channel is not CPTP.
    """
    # Check CPTP
    if is_cptp(kraus_ops, threshold) is False:
        raise AerNoiseError("Input Kraus channel is not CPTP.")

    # Get number of qubits
    num_qubits = int(np.log2(len(kraus_ops[0])))
    if len(kraus_ops[0]) != 2 ** num_qubits:
        raise AerNoiseError("Input Kraus channel is not a multi-qubit channel.")

    # Check if each matrix is a:
    # 1. scaled identity matrix
    # 2. scaled non-identity unitary matrix
    # 3. a non-unitary Kraus operator

    # Probabilities
    prob_identity = 0.
    prob_unitary = 0.  # total probability of all unitary ops
    probabilities = []  # initialize with probability of Identity

    # Matrices
    unitaries = []  # non-identity unitaries
    non_unitaries = []  # non-unitary Kraus matrices

    for op in kraus_ops:

        # Get the value of the first non-zero diagonal element
        # of op.H * op for rescaling
        prob = np.real(max(np.diag(np.conj(np.transpose(op)).dot(op))))
        if prob > 0:
            # Rescale the operator by square root of prob
            rescaled_op = np.array(op) / np.sqrt(prob)

            # Check if identity operator
            if is_identity(rescaled_op, threshold):
                prob_identity += prob
                prob_unitary += prob

            # Check if unitary
            elif is_unitary(rescaled_op, threshold):
                probabilities.append(prob)
                prob_unitary += prob
                unitaries.append(rescaled_op)

            # Non-unitary op
            else:
                non_unitaries.append(op)

    # Build qobj instructions
    instructions = []
    qubits = list(range(num_qubits))

    # Add unitary instructions
    for unitary in unitaries:
        instructions.append([make_unitary_instruction(unitary, qubits, threshold)])

    # Add identity instruction
    if prob_identity > threshold:
        probabilities.append(prob_identity)
        instructions.append([{"name": "id", "qubits": [0]}])

    # Add Kraus
    prob_kraus = 1 - prob_unitary
    if abs(prob_kraus) > threshold:
        # Rescale kraus operators by probabilities
        non_unitaries = [np.array(op) / np.sqrt(prob_kraus) for op in non_unitaries]
        instructions.append([make_kraus_instruction(non_unitaries, qubits, threshold)])
        probabilities.append(prob_kraus)

    return zip(probabilities, instructions)


def qubits_distinct(qubits0, qubits1):
    """Return true if two lists of qubits are distinct."""
    joint = qubits0 + qubits1
    return len(set(joint)) == len(joint)


def kraus_dot(kraus0, kraus1):
    qubits0 = kraus0['qubits']
    qubits1 = kraus1['qubits']
    if qubits0 != qubits1:
        raise AerNoiseError("Kraus instructions are on different qubits")
    params = [np.dot(b, a) for a in kraus0['params']
              for b in kraus1['params']]
    return {'name': 'kraus', 'qubits': qubits0, 'params': params}


def mat_dot(mat0, mat1):
    qubits0 = mat0['qubits']
    qubits1 = mat1['qubits']
    if qubits0 != qubits1:
        raise AerNoiseError("Unitary instructions are on different qubits")
    params = np.dot(mat1['params'], mat0['params'])
    return {'name': 'kraus', 'qubits': qubits0, 'params': params}


def kraus_kron(kraus0, kraus1, shift_qubits=0):
    """
    This assumes the Kraus operators are each defined on qubits
    range(n) and shifts qubits of second Kraus operator accordingly.
    """
    qubits0 = kraus0['qubits']
    qubits1 = [q + shift_qubits for q in kraus1['qubits']]
    if not qubits_distinct(qubits0, qubits1):
        raise AerNoiseError("Kraus operators not on distinct qubits")
    params = [np.kron(b, a) for a in kraus0['params']
              for b in kraus1['params']]
    return {'name': 'kraus', 'qubits': qubits0 + qubits1, 'params': params}


def mat_kron(mat0, mat1, shift_qubits=0):
    """
    This assumes the Unitary matrix operators are each defined on qubits
    range(n) and shifts qubits of second Kraus operator accordingly.
    """
    qubits0 = mat0['qubits']
    qubits1 = [q + shift_qubits for q in mat1['qubits']]
    if not qubits_distinct(qubits0, qubits1):
        raise AerNoiseError("Unitary matrix operators not on distinct qubits")
    params = np.kron(mat1['params'], mat0['params'])
    return {'name': 'kraus', 'qubits': qubits0 + qubits1, 'params': params}
