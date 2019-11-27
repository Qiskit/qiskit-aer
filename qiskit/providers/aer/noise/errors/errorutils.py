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
Helper functions for noise model creation.
"""

import numpy as np

from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT

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


# pylint: disable=too-many-return-statements
def standard_gate_instruction(instruction, ignore_phase=True):
    """Convert a unitary matrix instruction into a standard gate instruction.

    Args:
        instruction (dict): A qobj instruction.
        ignore_phase (bool): Ignore global phase on unitary matrix in
                             comparison to canonical unitary.

    Returns:
        list: a list of qobj instructions equivalent to in input instruction.
    """

    name = instruction.get("name", None)
    if name not in ["mat", "unitary", "kraus"]:
        return [instruction]
    qubits = instruction["qubits"]
    params = instruction["params"]

    # Check for single-qubit reset Kraus
    if name == "kraus":
        if len(qubits) == 1:
            superop = SuperOp(Kraus(params))
            # Check if reset to |0>
            reset0 = reset_superop(1)
            if superop == reset0:
                return [{"name": "reset", "qubits": qubits}]
            # Check if reset to |1>
            reset1 = reset0.compose(Operator(standard_gate_unitary('x')))
            if superop == reset1:
                return [{"name": "reset", "qubits": qubits}, {"name": "x", "qubits": qubits}]
        # otherwise just return the kraus instruction
        return [instruction]

    # Check single qubit gates
    mat_dagger = np.conj(params[0])
    if len(qubits) == 1:
        # Check clifford gates
        for j in range(24):
            if matrix_equal(
                    mat_dagger,
                    single_qubit_clifford_matrix(j),
                    ignore_phase=ignore_phase):
                return single_qubit_clifford_instructions(j, qubit=qubits[0])
        # Check t gates
        for name in ["t", "tdg"]:
            if matrix_equal(
                    mat_dagger,
                    standard_gate_unitary(name),
                    ignore_phase=ignore_phase):
                return [{"name": name, "qubits": qubits}]
        # TODO: u1,u2,u3 decomposition
    # Check two qubit gates
    if len(qubits) == 2:
        for name in ["cx", "cz", "swap"]:
            if matrix_equal(
                    mat_dagger,
                    standard_gate_unitary(name),
                    ignore_phase=ignore_phase):
                return [{"name": name, "qubits": qubits}]
        # Check reversed CX
        if matrix_equal(
                mat_dagger,
                standard_gate_unitary("cx_10"),
                ignore_phase=ignore_phase):
            return [{"name": "cx", "qubits": [qubits[1], qubits[0]]}]
        # Check 2-qubit Pauli's
        paulis = ["id", "x", "y", "z"]
        for pauli0 in paulis:
            for pauli1 in paulis:
                pmat = np.kron(
                    standard_gate_unitary(pauli1),
                    standard_gate_unitary(pauli0))
                if matrix_equal(mat_dagger, pmat, ignore_phase=ignore_phase):
                    if pauli0 == "id":
                        return [{"name": pauli1, "qubits": [qubits[1]]}]
                    elif pauli1 == "id":
                        return [{"name": pauli0, "qubits": [qubits[0]]}]
                    else:
                        return [{
                            "name": pauli0,
                            "qubits": [qubits[0]]
                        }, {
                            "name": pauli1,
                            "qubits": [qubits[1]]
                        }]
    # Check three qubit toffoli
    if len(qubits) == 3:
        if matrix_equal(
                mat_dagger,
                standard_gate_unitary("ccx_012"),
                ignore_phase=ignore_phase):
            return [{"name": "ccx", "qubits": qubits}]
        if matrix_equal(
                mat_dagger,
                standard_gate_unitary("ccx_021"),
                ignore_phase=ignore_phase):
            return [{
                "name": "ccx",
                "qubits": [qubits[0], qubits[2], qubits[1]]
            }]
        if matrix_equal(
                mat_dagger,
                standard_gate_unitary("ccx_120"),
                ignore_phase=ignore_phase):
            return [{
                "name": "ccx",
                "qubits": [qubits[1], qubits[2], qubits[0]]
            }]

    # Else return input in
    return [instruction]


def single_qubit_clifford_gates(j):
    """Return a QASM gate names for a single qubit Clifford.

    The labels are returned in a basis set consisting of
    ('id', 's', 'sdg', 'z', 'h', x', 'y') gates decomposed to
    use the minimum number of X-90 pulses in a (u1, u2, u3)
    decomposition.

    Args:
        j (int): Clifford index 0, ..., 23.

    Returns:
        tuple(str): The tuple of basis gates.

    Raises:
        NoiseError: If index is out of range [0, 23].
    """

    if not isinstance(j, int) or j < 0 or j > 23:
        raise NoiseError(
            "Index {} must be in the range [0, ..., 23]".format(j))

    labels = [
        ('id', ),
        ('s', ),
        ('sdg', ),
        ('z', ),
        # u2 gates
        (
            'h', ),
        ('h', 'z'),
        ('z', 'h'),
        ('h', 's'),
        ('s', 'h'),
        ('h', 'sdg'),
        ('sdg', 'h'),
        ('s', 'h', 's'),
        ('sdg', 'h', 's'),
        ('z', 'h', 's'),
        ('s', 'h', 'sdg'),
        ('sdg', 'h', 'sdg'),
        ('z', 'h', 'sdg'),
        ('s', 'h', 'z'),
        ('sdg', 'h', 'z'),
        ('z', 'h', 'z'),
        # u3 gates
        (
            'x', ),
        ('y', ),
        ('s', 'x'),
        ('sdg', 'x')
    ]
    return labels[j]


def single_qubit_clifford_matrix(j):
    """Return Numpy array for a single qubit Clifford.

    Args:
        j (int): Clifford index 0, ..., 23.

    Returns:
        np.array: The matrix for the indexed clifford.

    Raises:
        NoiseError: If index is out of range [0, 23].
    """

    if not isinstance(j, int) or j < 0 or j > 23:
        raise NoiseError(
            "Index {} must be in the range [0, ..., 23]".format(j))

    basis_dict = {
        'id': np.eye(2),
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
        'h': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        's': np.array([[1, 0], [0, 1j]], dtype=complex),
        'sdg': np.array([[1, 0], [0, -1j]], dtype=complex)
    }
    mat = np.eye(2)
    for gate in single_qubit_clifford_gates(j):
        mat = np.dot(basis_dict[gate], mat)
    return mat


# pylint: disable=invalid-name
def single_qubit_clifford_instructions(index, qubit=0):
    """Return a list of qobj instructions for a single qubit Cliffords.

    The instructions are returned in a basis set consisting of
    ('id', 's', 'sdg', 'z', 'h', x', 'y') gates decomposed to
    use the minimum number of X-90 pulses in a (u1, u2, u3)
    decomposition.

    Args:
        index (int): Clifford index 0, ..., 23.
        qubit (int): the qubit to apply the Clifford to.

    Returns:
        list(dict): The list of instructions.

    Raises:
        NoiseError: If index is out of range [0, 23] or qubit invalid.
    """

    if not isinstance(index, int) or index < 0 or index > 23:
        raise NoiseError(
            "Index {} must be in the range [0, ..., 23]".format(index))
    if not isinstance(qubit, int) or qubit < 0:
        raise NoiseError("qubit position must be positive integer.")

    instructions = []
    for gate in single_qubit_clifford_gates(index):
        instructions.append({"name": gate, "qubits": [qubit]})
    return instructions


def standard_gate_unitary(name):
    """Return the unitary matrix for a standard gate."""

    unitary_matrices = {
        ("id", "I"):
            np.eye(2, dtype=complex),
        ("x", "X"):
            np.array([[0, 1], [1, 0]], dtype=complex),
        ("y", "Y"):
            np.array([[0, -1j], [1j, 0]], dtype=complex),
        ("z", "Z"):
            np.array([[1, 0], [0, -1]], dtype=complex),
        ("h", "H"):
            np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        ("s", "S"):
            np.array([[1, 0], [0, 1j]], dtype=complex),
        ("sdg", "Sdg"):
            np.array([[1, 0], [0, -1j]], dtype=complex),
        ("t", "T"):
            np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
        ("tdg", "Tdg"):
            np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
        ("cx", "CX", "cx_01"):
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex),
        ("cx_10",):
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex),
        ("cz", "CZ"):
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex),
        ("swap", "SWAP"):
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
        ("ccx", "CCX", "ccx_012", "ccx_102"):
            np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
                     dtype=complex),
        ("ccx_021", "ccx_201"):
            np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0]],
                     dtype=complex),
        ("ccx_120", "ccx_210"):
            np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]],
                     dtype=complex)
    }

    return next((value for key, value in unitary_matrices.items() if name in key), None)


def reset_superop(num_qubits):
    """Return a N-qubit reset SuperOp."""
    reset = SuperOp(
        np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    if num_qubits == 1:
        return reset
    reset_n = reset
    for _ in range(num_qubits - 1):
        reset_n.tensor(reset)
    return reset_n


def standard_instruction_operator(instr):
    """Return the Operator for a standard gate instruction."""
    # Convert to dict (for QobjInstruction types)
    if hasattr(instr, 'as_dict'):
        instr = instr.as_dict()
    # Get name and parameters
    name = instr.get('name', "")
    params = instr.get('params', [])
    # Check if standard unitary gate name
    mat = standard_gate_unitary(name)
    if isinstance(mat, np.ndarray):
        return Operator(mat)

    # Check if standard parameterized waltz gates
    if name == 'u1':
        lam = params[0]
        mat = np.diag([1, np.exp(1j * lam)])
        return Operator(mat)
    if name == 'u2':
        phi = params[0]
        lam = params[1]
        mat = np.array([[1, -np.exp(1j * lam)],
                        [np.exp(1j * phi),
                         np.exp(1j * (phi + lam))]]) / np.sqrt(2)
        return Operator(mat)
    if name == 'u3':
        theta = params[0]
        phi = params[1]
        lam = params[2]
        mat = np.array(
            [[np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
             [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]])
        return Operator(mat)

    # Check if unitary instruction
    if name == 'unitary':
        return Operator(params[0])

    # Otherwise return None if we cannot convert instruction
    return None


def standard_instruction_channel(instr):
    """Return the SuperOp channel for a standard instruction."""
    # Check if standard operator
    oper = standard_instruction_operator(instr)
    if oper is not None:
        return SuperOp(oper)

    # Convert to dict (for QobjInstruction types)
    if hasattr(instr, 'as_dict'):
        instr = instr.as_dict()
    # Get name and parameters
    name = instr.get('name', "")

    # Check if reset instruction
    if name == 'reset':
        # params should be the number of qubits being reset
        num_qubits = len(instr['qubits'])
        return reset_superop(num_qubits)
    # Check if Kraus instruction
    if name == 'kraus':
        params = instr['params']
        return SuperOp(Kraus(params))
    return None


def circuit2superop(circuit, min_qubits=1):
    """Return the SuperOp for a standard instruction."""
    # Get number of qubits
    max_qubits = 1
    for instr in circuit:
        qubits = []
        if hasattr(instr, 'qubits'):
            qubits = instr.qubits
        elif isinstance(instr, dict):
            qubits = instr.get('qubits', [])
        max_qubits = max(max_qubits, 1 + max(qubits))

    num_qubits = max(max_qubits, min_qubits)

    # Initialize N-qubit identity superoperator
    superop = SuperOp(np.eye(4**num_qubits))
    # compose each circuit element with the superoperator
    for instr in circuit:
        instr_op = standard_instruction_channel(instr)
        if instr_op is None:
            raise NoiseError('Cannot convert instruction {} to SuperOp'.format(instr))
        if hasattr(instr, 'qubits'):
            qubits = instr.qubits
        else:
            qubits = instr['qubits']
        superop = superop.compose(instr_op, qubits)
    return superop


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

    if isinstance(qubits, int):
        qubits = [qubits]

    instruction = {"name": "unitary", "qubits": qubits, "params": [mat]}
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
    kraus = Kraus(mats)
    if not kraus.is_cptp() or kraus._input_dim != kraus._output_dim:
        raise NoiseError("Input Kraus matrices are not a CPTP channel.")
    if isinstance(qubits, int):
        qubits = [qubits]
    return [{"name": "kraus", "qubits": qubits, "params": kraus.data}]


def qubits_from_mat(mat):
    """Return the number of qubits for a multi-qubit matrix."""
    arr = np.array(mat)
    shape = arr.shape
    num_qubits = int(np.log2(shape[1]))
    if shape[1] != 2**num_qubits:
        raise NoiseError("Input Kraus channel is not a multi-qubit channel.")
    return num_qubits


def is_matrix_diagonal(mat):
    """Test if row-vector representation of diagonal matrix."""
    mat = np.array(mat)
    shape = mat.shape
    return len(shape) == 2 and shape[0] == 1


def kraus2instructions(kraus_ops, standard_gates, atol=ATOL_DEFAULT):
    """
    Convert a list of Kraus matrices into qobj circuits.

    If any Kraus operators are a unitary matrix they will be converted
    into unitary qobj instructions. Identity unitary matrices will also be
    converted into identity qobj instructions.

    Args:
        kraus_ops (list[matrix]): A list of Kraus matrices for a CPTP map.
        standard_gates (bool): Check if the matrix instruction is a
                               standard instruction (default: True).
        atol (double): Threshold for testing if probabilities are zero.


    Returns:
        list: A list of pairs (p, circuit) where `circuit` is a list of qobj
        instructions, and `p` is the probability of that circuit for the
        given error.

    Raises:
        NoiseError: If the input Kraus channel is not CPTP.
    """
    # Check threshold
    if atol < 0:
        raise NoiseError("atol cannot be negative")
    if atol > 1e-5:
        raise NoiseError(
            "atol value is too large. It should be close to zero.")

    # Check CPTP
    if not Kraus(kraus_ops).is_cptp(atol=atol):
        raise NoiseError("Input Kraus channel is not CPTP.")

    # Get number of qubits
    num_qubits = int(np.log2(len(kraus_ops[0])))
    if len(kraus_ops[0]) != 2**num_qubits:
        raise NoiseError("Input Kraus channel is not a multi-qubit channel.")

    # Check if each matrix is a:
    # 1. scaled identity matrix
    # 2. scaled non-identity unitary matrix
    # 3. a non-unitary Kraus operator

    # Probabilities
    prob_identity = 0
    prob_unitary = 0  # total probability of all unitary ops (including id)
    prob_kraus = 0  # total probability of non-unitary ops
    probabilities = []  # initialize with probability of Identity

    # Matrices
    unitaries = []  # non-identity unitaries
    non_unitaries = []  # non-unitary Kraus matrices

    for mat in kraus_ops:
        # Get the value of the maximum diagonal element
        # of op.H * op for rescaling
        # pylint: disable=no-member
        prob = abs(max(np.diag(np.conj(np.transpose(mat)).dot(mat))))
        if prob > 0.0:
            if abs(prob - 1) > 0.0:
                # Rescale the operator by square root of prob
                rescaled_mat = np.array(mat) / np.sqrt(prob)
            else:
                rescaled_mat = mat
            # Check if identity operator
            if is_identity_matrix(rescaled_mat, ignore_phase=True):
                prob_identity += prob
                prob_unitary += prob
            # Check if unitary
            elif is_unitary_matrix(rescaled_mat):
                probabilities.append(prob)
                prob_unitary += prob
                unitaries.append(rescaled_mat)
            # Non-unitary op
            else:
                non_unitaries.append(mat)

    # Check probabilities
    prob_kraus = 1 - prob_unitary
    if prob_unitary - 1 > atol:
        raise NoiseError("Invalid kraus matrices: unitary probability "
                         "{} > 1".format(prob_unitary))
    if prob_unitary < -atol:
        raise NoiseError("Invalid kraus matrices: unitary probability "
                         "{} < 1".format(prob_unitary))
    if prob_identity - 1 > atol:
        raise NoiseError("Invalid kraus matrices: identity probability "
                         "{} > 1".format(prob_identity))
    if prob_identity < -atol:
        raise NoiseError("Invalid kraus matrices: identity probability "
                         "{} < 1".format(prob_identity))
    if prob_kraus - 1 > atol:
        raise NoiseError("Invalid kraus matrices: non-unitary probability "
                         "{} > 1".format(prob_kraus))
    if prob_kraus < -atol:
        raise NoiseError("Invalid kraus matrices: non-unitary probability "
                         "{} < 1".format(prob_kraus))

    # Build qobj instructions
    instructions = []
    qubits = list(range(num_qubits))

    # Add unitary instructions
    for unitary in unitaries:
        instructions.append(
            make_unitary_instruction(
                unitary, qubits, standard_gates=standard_gates))

    # Add identity instruction
    if prob_identity > atol:
        if abs(prob_identity - 1) < atol:
            probabilities.append(1)
        else:
            probabilities.append(prob_identity)
        instructions.append([{"name": "id", "qubits": [0]}])

    # Add Kraus
    if prob_kraus < atol:
        # No Kraus operators
        return zip(instructions, probabilities)
    if prob_kraus < 1:
        # Rescale kraus operators by probabilities
        non_unitaries = [
            np.array(op) / np.sqrt(prob_kraus) for op in non_unitaries
        ]
    instructions.append(make_kraus_instruction(non_unitaries, qubits))
    probabilities.append(prob_kraus)
    # Normalize probabilities to account for any rounding errors
    probabilities = list(np.array(probabilities) / np.sum(probabilities))
    return zip(instructions, probabilities)
