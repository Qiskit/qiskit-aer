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


def standard_gates_instructions(instructions, threshold=1e-10):
    """Convert a list with unitary matrix instructions into standard gates.

    Args:
        instructions (list): A list of qobj instructions.
        threshold (double): The threshold parameter for comparing unitary to
                            standard gate matrices.

    Returns:
        list: a list of qobj instructions equivalent to in input instruction.
    """
    output_instructions = []
    for instruction in instructions:
        output_instructions += standard_gate_instruction(instruction,
                                                         threshold=threshold)
    return output_instructions


def standard_gate_instruction(instruction, ignore_phase=True, threshold=1e-10):
    """Convert a unitary matrix instruction into a standard gate instruction.

    Args:
        instruction (dict): A qobj instruction.
        ignore_phase (bool): Ignore global phase on unitary matrix in
                             comparison to canonical unitary.
        threshold (double): The threshold parameter for comparing unitary to
                            standard gate matrices.

    Returns:
        list: a list of qobj instructions equivalent to in input instruction.
    """
    if instruction.get("name", None) not in ["mat", "unitary"]:
        return [instruction]

    qubits = instruction["qubits"]
    mat_dagger = np.conj(instruction["params"])

    def compare_mat(target):
        tr = np.trace(np.dot(mat_dagger, target)) / len(target)
        if ignore_phase:
            return np.abs(np.abs(tr) - 1) < threshold
        else:
            return np.abs(tr - 1) < threshold

    # Check single qubit gates
    if len(qubits) == 1:
        # Check clifford gates
        for j in range(24):
            if compare_mat(single_qubit_clifford_matrix(j)):
                return single_qubit_clifford_instructions(j, qubit=qubits[0])
        # Check t gates
        for name in ["t", "tdg"]:
            if compare_mat(standard_gate_unitary(name)):
                return [{"name": name, "qubits": qubits}]
        # TODO: u1,u2,u3 decomposition
    # Check two qubit gates
    if len(qubits) == 2:
        for name in ["cx", "cz", "swap"]:
            if compare_mat(standard_gate_unitary(name)):
                return [{"name": name, "qubits": qubits}]
        # Check reversed CX
        if compare_mat(standard_gate_unitary("cx_10")):
                return [{"name": "cx", "qubits": [qubits[1], qubits[0]]}]
        # Check 2-qubit Pauli's
        paulis = ["id", "x", "y", "z"]
        for q0 in paulis:
            for q1 in paulis:
                pmat = np.kron(standard_gate_unitary(q1), standard_gate_unitary(q0))
                if compare_mat(pmat):
                    if q0 is "id":
                        return [{"name": q1, "qubits": [qubits[1]]}]
                    elif q1 is "id":
                        return [{"name": q0, "qubits": [qubits[0]]}]
                    else:
                        return [{"name": q0, "qubits": [qubits[0]]},
                                {"name": q1, "qubits": [qubits[1]]}]
    # Check three qubit toffoli
    if len(qubits) == 3:
        if compare_mat(standard_gate_unitary("ccx_012")):
            return [{"name": "ccx", "qubits": qubits}]
        if compare_mat(standard_gate_unitary("ccx_021")):
            return [{"name": "ccx", "qubits": [qubits[0], qubits[2], qubits[1]]}]
        if compare_mat(standard_gate_unitary("ccx_120")):
            return [{"name": "ccx", "qubits": [qubits[1], qubits[2], qubits[0]]}]

    # Else return input in
    return [instruction]


def single_qubit_clifford_gates(j):
    """Return a QASM gate names for a single qubit clifford.

    The labels are returned in a basis set consisting of
    ('id', 's', 'sdg', 'z', 'h', x', 'y') gates decomposed to
    use the minimum number of X-90 pulses in a (u1, u2, u3)
    decomposition.

    Args:
        j (int): Clifford index 0, ..., 23.

    Returns:
        tuple(str): The tuple of basis gates."""

    if not isinstance(j, int) or j < 0 or j > 23:
        raise AerNoiseError("Index {} must be in the range [0, ..., 23]".format(j))

    labels = [
        ('id',), ('s',), ('sdg',), ('z',),
        # u2 gates
        ('h',), ('h', 'z'), ('z', 'h'), ('h', 's'), ('s', 'h'), ('h', 'sdg'), ('sdg', 'h'),
        ('s', 'h', 's'), ('sdg', 'h', 's'), ('z', 'h', 's'),
        ('s', 'h', 'sdg'), ('sdg', 'h', 'sdg'), ('z', 'h', 'sdg'),
        ('s', 'h', 'z'), ('sdg', 'h', 'z'), ('z', 'h', 'z'),
        # u3 gates
        ('x',), ('y',), ('s', 'x'), ('sdg', 'x')
    ]
    return labels[j]


def single_qubit_clifford_matrix(j):
    """Return Numpy array for a single qubit clifford.

    Args:
        j (int): Clifford index 0, ..., 23.

    Returns:
        np.array: The matrix for the indexed clifford."""

    if not isinstance(j, int) or j < 0 or j > 23:
        raise AerNoiseError("Index {} must be in the range [0, ..., 23]".format(j))

    basis_dict = {
        'id': np.eye(2),
        'x': np.array([[0, 1], [1, 0]]),
        'y': np.array([[0, -1j], [1j, 0]]),
        'z': np.array([[1, 0], [0, -1]]),
        'h': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        's': np.array([[1, 0], [0, 1j]]),
        'sdg': np.array([[1, 0], [0, -1j]])
    }
    mat = np.eye(2)
    for gate in single_qubit_clifford_gates(j):
        mat = np.dot(basis_dict[gate], mat)
    return mat


def single_qubit_clifford_instructions(j, qubit=0):
    """Return a list of qobj instructions for a single qubit cliffords.

    The instructions are returned in a basis set consisting of
    ('id', 's', 'sdg', 'z', 'h', x', 'y') gates decomposed to
    use the minimum number of X-90 pulses in a (u1, u2, u3)
    decomposition.

    Args:
        j (int): Clifford index 0, ..., 23.

    Returns:
        list(dict): The list of instructions."""

    if not isinstance(j, int) or j < 0 or j > 23:
        raise AerNoiseError("Index {} must be in the range [0, ..., 23]".format(j))
    if not isinstance(qubit, int) or qubit < 0:
        raise AerNoiseError("qubit position must be positive integer.")

    instructions = []
    for gate in single_qubit_clifford_gates(j):
        instructions.append({"name": gate, "qubits": [qubit]})
    return instructions


def standard_gate_unitary(name):
    """Return the unitary matrix for a standard gate."""
    if name in ["id", "I"]:
        return np.eye(2, dtype=complex)
    if name in ["x", "X"]:
        return np.array([[0, 1],
                         [1, 0]], dtype=complex)
    if name in ["y", "Y"]:
        return np.array([[0, -1j],
                         [1j, 0]], dtype=complex)
    if name in ["z", "Z"]:
        return np.array([[1, 0],
                         [0, -1]], dtype=complex)
    if name in ["h", "H"]:
        return np.array([[1, 1],
                         [1, -1]], dtype=complex) / np.sqrt(2)
    if name in ["s", "S"]:
        return np.array([[1, 0],
                         [0, 1j]], dtype=complex)
    if name in ["sdg", "Sdg"]:
        return np.array([[1, 0],
                         [0, -1j]], dtype=complex)
    if name in ["t", "T"]:
        return np.array([[1, 0],
                         [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    if name in ["tdg", "Tdg"]:
        return np.array([[1, 0],
                         [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    if name in ["cx", "CX", "cx_01"]:
        return np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0]], dtype=complex)
    if name is "cx_10":
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
    if name in ["cz", "CZ"]:
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, -1]], dtype=complex)
    if name in ["swap", "SWAP"]:
        return np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
    if name in ["ccx", "CCX", "ccx_012", "ccx_102"]:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0]], dtype=complex)
    if name in ["ccx_021", "ccx_201"]:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0]], dtype=complex)
    if name in ["ccx_120", "ccx_210"]:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex)
    return None


def make_unitary_instruction(mat, qubits, standard_gates=True, threshold=1e-10):
    """Return a qobj instruction for a unitary matrix gate.

    Args:
        mat (matrix): A square or diagonal unitary matrix.
        qubits (list[int]): The qubits the matrix is applied to.
        standard_gates (bool): Check if the matrix instruction is a
                               standard instruction.
        threshold (double): The threshold parameter for testing if the
                            input matrix is unitary (default: 1e-10).
    Returns:
        dict: The qobj instruction object.

    Raises:
        AerNoiseError: if the input is not a unitary matrix.
    """
    if is_unitary(mat, threshold=threshold) is False:
        raise AerNoiseError("Input matrix is not unitary.")
    elif isinstance(qubits, int):
        qubits = [qubits]
    instruction = {"name": "unitary", "qubits": qubits, "params": mat}
    if standard_gates is True:
        return standard_gate_instruction(instruction, threshold=threshold)
    else:
        return [instruction]


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
    if is_cptp(mats, threshold=threshold) is False:
        raise AerNoiseError("Input Kraus ops are not a CPTP channel.")
    elif isinstance(qubits, int):
        qubits = [qubits]
    return [{"name": "kraus", "qubits": qubits, "params": mats}]


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
        return is_identity(np.conj(mat) * mat, threshold=threshold)
    else:
        return is_identity(np.conj(mat.T).dot(mat), threshold=threshold)


def kraus2instructions(kraus_ops, standard_gates=True, threshold=1e-10):
    """
    Convert a list of Kraus matrices into qobj circuits.

    If any Kraus operators are a unitary matrix they will be converted
    into unitary qobj instructions. Identity unitary matrices will also be
    converted into identity qobj instructions.

    Args:
        kraus_ops (list[matrix]): A list of Kraus matrices for a CPTP map.
        standard_gates (bool): Check if the matrix instruction is a
                               standard instruction.
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
    if is_cptp(kraus_ops, threshold=threshold) is False:
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
            if is_identity(rescaled_op, threshold=threshold):
                prob_identity += prob
                prob_unitary += prob

            # Check if unitary
            elif is_unitary(rescaled_op, threshold=threshold):
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
        instructions.append(make_unitary_instruction(unitary, qubits,
                                                     standard_gates=standard_gates,
                                                     threshold=threshold))

    # Add identity instruction
    if prob_identity > threshold:
        probabilities.append(prob_identity)
        instructions.append([{"name": "id", "qubits": [0]}])

    # Add Kraus
    prob_kraus = 1 - prob_unitary
    if abs(prob_kraus) > threshold:
        # Rescale kraus operators by probabilities
        non_unitaries = [np.array(op) / np.sqrt(prob_kraus) for op in non_unitaries]
        instructions.append(make_kraus_instruction(non_unitaries, qubits,
                                                   threshold=threshold))
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
