# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Helper functions for noise model creation.
"""

import numpy as np
from ..noiseerror import NoiseError


def standard_gates_instructions(instructions):
    """Convert a list with unitary matrix instructions into standard gates.

    Args:
        instructions (list): A list of qobj instructions.

    Returns:
        list: a list of qobj instructions equivalent to in input instruction.
    """
    output_instructions = []
    for instruction in instructions:
        output_instructions += standard_gate_instruction(instruction)
    return output_instructions


def standard_gate_instruction(instruction, ignore_phase=True):
    """Convert a unitary matrix instruction into a standard gate instruction.

    Args:
        instruction (dict): A qobj instruction.
        ignore_phase (bool): Ignore global phase on unitary matrix in
                             comparison to canonical unitary.

    Returns:
        list: a list of qobj instructions equivalent to in input instruction.
    """
    if instruction.get("name", None) not in ["mat", "unitary"]:
        return [instruction]

    qubits = instruction["qubits"]
    mat_dagger = np.conj(instruction["params"])

    def compare_mat(target):
        precision = 7
        try:
            tr = np.trace(np.dot(mat_dagger, target)) / len(target)
        except:
            return False
        if ignore_phase:
            delta = round(np.conj(tr) * tr - 1, precision)
        else:
            delta = round(tr - 1, precision)
        return delta == 0

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
        raise NoiseError("Index {} must be in the range [0, ..., 23]".format(j))

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
        raise NoiseError("Index {} must be in the range [0, ..., 23]".format(j))

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
        raise NoiseError("Index {} must be in the range [0, ..., 23]".format(j))
    if not isinstance(qubit, int) or qubit < 0:
        raise NoiseError("qubit position must be positive integer.")

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


def make_unitary_instruction(mat, qubits, standard_gates=True):
    """Return a qobj instruction for a unitary matrix gate.

    Args:
        mat (matrix): A square or diagonal unitary matrix.
        qubits (list[int]): The qubits the matrix is applied to.
        standard_gates (bool): Check if the matrix instruction is a
                               standard instruction.

    Returns:
        dict: The qobj instruction object.

    Raises:
        NoiseError: if the input is not a unitary matrix.
    """
    if not is_unitary_matrix(mat):
        raise NoiseError("Input matrix is not unitary.")
    elif isinstance(qubits, int):
        qubits = [qubits]
    instruction = {"name": "unitary",
                   "qubits": qubits,
                   "params": mat}
    if standard_gates:
        return standard_gate_instruction(instruction)
    else:
        return [instruction]


def make_kraus_instruction(mats, qubits):
    """Return a qobj instruction for a Kraus error.

    Args:
        mats (list[matrix]): A list of square or diagonal Kraus matrices.
        qubits (list[int]): The qubits the matrix is applied to.
    Returns:
        dict: The qobj instruction object.

    Raises:
        NoiseError: if the input is not a CPTP Kraus channel.
    """
    if not is_cptp_kraus(mats):
        raise NoiseError("Input Kraus matrices are not a CPTP channel.")
    elif isinstance(qubits, int):
        qubits = [qubits]
    return [{"name": "kraus", "qubits": qubits, "params": mats}]


def qubits_from_mat(mat):
    """Return the number of qubits for a multi-qubit matrix."""
    arr = np.array(mat)
    shape = arr.shape
    num_qubits = int(np.log2(shape[1]))
    if shape[1] != 2 ** num_qubits:
        raise NoiseError("Input Kraus channel is not a multi-qubit channel.")
    return num_qubits


def is_square_matrix(op):
    """Test if an array is a square matrix."""
    mat = np.array(op)
    shape = mat.shape
    return len(shape) == 2 and shape[0] == shape[1]


def is_matrix_diagonal(op):
    """Test if row-vector representation of diagonal matrix."""
    mat = np.array(op)
    shape = mat.shape
    return len(shape) == 2 and shape[0] == 1


def is_cptp_kraus(kraus_ops, precision=7):
    """Test if a list of Kraus matrices is a CPTP map."""
    accum = 0j
    for op in kraus_ops:
        if is_matrix_diagonal(op):
            op = np.diag(op[0])
        accum += np.dot(np.transpose(np.conj(op)), op)
    return is_identity_matrix(accum, ignore_phase=False,
                              precision=precision)


def is_identity_matrix(op, ignore_phase=False, precision=7):
    """Test if an array is an identity matrix."""
    mat = np.array(op)
    if is_matrix_diagonal(mat):
        mat = np.diag(mat[0])  # convert to square
    if not is_square_matrix(mat):
        return False
    if ignore_phase:
        mat = np.conj(mat[0, 0]) * mat  # entrywise multiplication with conjugate
    # Check if square identity
    iden = np.eye(len(mat))
    delta = round(np.linalg.norm(mat - iden), precision)
    return delta == 0


def is_unitary_matrix(op, precision=7):
    """Test if an array is a unitary matrix."""
    mat = np.array(op)
    # Compute A^dagger.A and see if it is identity matrix
    if is_matrix_diagonal(mat):
        mat = np.conj(mat) * mat
    else:
        mat = np.conj(mat.T).dot(mat)
    return is_identity_matrix(mat, ignore_phase=False, precision=precision)


def kraus2choi(kraus_ops):
    """Convert Kraus matrices to a Choi matrix"""
    # Compute eigensystem of Choi matrix
    choi = 0j
    for op in kraus_ops:
        vec = np.ravel(op, order='F')
        choi += np.outer(vec, np.conj(vec))
    return choi


def choi2kraus(choi, threshold=1e-10):
    """Convert a Choi matrix to canonical Kraus matrices"""
    # Check threshold
    if threshold < 0:
        raise NoiseError("Threshold value cannot be negative")
    if threshold > 1e-3:
        raise NoiseError("Threshold value is too large. It should be close to zero.")
    # Compute eigensystem of Choi matrix
    w, v = np.linalg.eig(choi)
    kraus = []
    for val, vec in zip(w, v.T):
        if val > threshold:
            kraus.append(np.sqrt(val) * vec.reshape((2, 2), order='F'))
        if val < -threshold:
            raise NoiseError("Input Choi-matrix is not CP " +
                                " (eigenvalue {} < 0)".format(val))
    return kraus


def canonical_kraus_matrices(kraus_ops, threshold=1e-10):
    """Convert a list of Kraus ops into the canonical representation.

    In the canonical representation the vecorized Kraus operators are
    the eigenvectors of the Choi-matrix for the input channel.

    Args:
        kraus_ops (list[matrix]): A list of Kraus matrices for a CPTP map.
        threshold (double): Threshold for checking if eigenvalues are zero
                            (Default: 1e-10)
    """
    return choi2kraus(kraus2choi(kraus_ops), threshold=threshold)


def kraus2instructions(kraus_ops, standard_gates, threshold):
    """
    Convert a list of Kraus matrices into qobj circuits.

    If any Kraus operators are a unitary matrix they will be converted
    into unitary qobj instructions. Identity unitary matrices will also be
    converted into identity qobj instructions.

    Args:
        kraus_ops (list[matrix]): A list of Kraus matrices for a CPTP map.
        standard_gates (bool): Check if the matrix instruction is a
                               standard instruction (default: True).
        threshold (double): Threshold for testing if probabilities are zero.


    Returns:
        A list of pairs (p, circuit) where `circuit` is a list of qobj
        instructions, and `p` is the probability of that circuit for the
        given error.

    Raises:
        NoiseError: If the input Kraus channel is not CPTP.
    """
    # Check threshold
    if threshold < 0:
        raise NoiseError("Threshold cannot be negative")
    if threshold > 1e-3:
        raise NoiseError("Threhsold value is too large. It should be close to zero.")

    # Check CPTP
    if not is_cptp_kraus(kraus_ops):
        raise NoiseError("Input Kraus channel is not CPTP.")

    # Get number of qubits
    num_qubits = int(np.log2(len(kraus_ops[0])))
    if len(kraus_ops[0]) != 2 ** num_qubits:
        raise NoiseError("Input Kraus channel is not a multi-qubit channel.")

    # Check if each matrix is a:
    # 1. scaled identity matrix
    # 2. scaled non-identity unitary matrix
    # 3. a non-unitary Kraus operator

    # Probabilities
    prob_identity = 0
    prob_unitary = 0    # total probability of all unitary ops (including id)
    prob_kraus = 0      # total probability of non-unitary ops
    probabilities = []  # initialize with probability of Identity

    # Matrices
    unitaries = []  # non-identity unitaries
    non_unitaries = []  # non-unitary Kraus matrices

    for op in kraus_ops:
        # Get the value of the maximum diagonal element
        # of op.H * op for rescaling
        prob = abs(max(np.diag(np.conj(np.transpose(op)).dot(op))))
        if prob > threshold:
            if abs(prob - 1) > threshold:
                # Rescale the operator by square root of prob
                rescaled_op = np.array(op) / np.sqrt(prob)
            else:
                rescaled_op = op
            # Check if identity operator
            if is_identity_matrix(rescaled_op, ignore_phase=True):
                prob_identity += prob
                prob_unitary += prob
            # Check if unitary
            elif is_unitary_matrix(rescaled_op):
                probabilities.append(prob)
                prob_unitary += prob
                unitaries.append(rescaled_op)
            # Non-unitary op
            else:
                non_unitaries.append(op)

    # Check probabilities
    prob_kraus = 1 - prob_unitary
    if prob_unitary - 1 > threshold:
        raise NoiseError("Invalid kraus matrices: unitary probability" +
                            " {} > 1".format(prob_unitary))
    if prob_unitary < -threshold:
        raise NoiseError("Invalid kraus matrices: unitary probability" +
                            " {} < 1".format(prob_unitary))
    if prob_identity - 1 > threshold:
        raise NoiseError("Invalid kraus matrices: identity probability" +
                            " {} > 1".format(prob_identity))
    if prob_identity < -threshold:
        raise NoiseError("Invalid kraus matrices: identity probability" +
                            " {} < 1".format(prob_identity))
    if prob_kraus - 1 > threshold:
        raise NoiseError("Invalid kraus matrices: non-unitary probability" +
                            " {} > 1".format(prob_kraus))
    if prob_kraus < -threshold:
        raise NoiseError("Invalid kraus matrices: non-unitary probability" +
                            " {} < 1".format(prob_kraus))

    # Build qobj instructions
    instructions = []
    qubits = list(range(num_qubits))

    # Add unitary instructions
    for unitary in unitaries:
        instructions.append(make_unitary_instruction(unitary, qubits,
                                                     standard_gates=standard_gates))

    # Add identity instruction
    if prob_identity > threshold:
        if abs(prob_identity - 1) < threshold:
            probabilities.append(1)
        else:
            probabilities.append(prob_identity)
        instructions.append([{"name": "id", "qubits": [0]}])

    # Add Kraus
    if prob_kraus < threshold:
        # No Kraus operators
        return zip(instructions, probabilities)
    if prob_kraus < 1:
        # Rescale kraus operators by probabilities
        non_unitaries = [np.array(op) / np.sqrt(prob_kraus) for op in non_unitaries]
    instructions.append(make_kraus_instruction(non_unitaries, qubits))
    probabilities.append(prob_kraus)
    return zip(instructions, probabilities)
