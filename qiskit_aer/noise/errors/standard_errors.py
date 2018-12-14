# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Standard quantum computing error channels for Qiskit Aer.
"""

import numpy as np
from itertools import product

from qiskit.quantum_info.operators.pauli import Pauli
from ..noiseerror import NoiseError
from .errorutils import make_unitary_instruction
from .errorutils import qubits_from_mat
from .errorutils import canonical_kraus_matrices
from .errorutils import choi2kraus
from .errorutils import standard_gate_unitary
from .errorutils import is_unitary_matrix
from .errorutils import is_identity_matrix
from .quantum_error import QuantumError

def kraus_error(noise_ops, standard_gates=True, canonical_kraus=False):
    """Kraus error channel.

    Args:
        noise_ops (list[matrix]): Kraus matrices.
        standard_gates (bool): Check if input matrices are standard gates.
        canonical_kraus (bool): Convert input Kraus matrices into the
                                canoical Kraus representation (default: False)

    Returns:
        QuantumError: The quantum error object.

    Raises:
        NoiseError: if error parameters are invalid.
    """
    if not isinstance(noise_ops, (list, tuple)):
        raise NoiseError("Invalid Kraus error input.")
    if len(noise_ops) == 0:
        raise NoiseError("Kraus error noise_ops must not be empty.")
    kraus_ops = [np.array(a, dtype=complex) for a in noise_ops]
    if canonical_kraus:
        kraus_ops = canonical_kraus_matrices(kraus_ops)
    return QuantumError(kraus_ops, standard_gates=standard_gates)


def mixed_unitary_error(noise_ops, standard_gates=True):
    """
    Mixed unitary quantum error channel.

    The input should be a list of pairs (U[j], p[j]), where `U[j]` is a
    unitary matrix and `p[j]` is a probability. All probabilities must
    sum to 1 for the input ops to be valid.

    Args:
        noise_ops (list[pair[matrix, double]]): unitary error matricies.
        standard_gates (bool): Check if input matrices are standard gates.

    Returns:
        QuantumError: The quantum error object.

    Raises:
        NoiseError: if error parameters are invalid.
    """

    # Error checking
    if not isinstance(noise_ops, (list, tuple, zip)):
        raise NoiseError("Input noise ops is not a list.")

    # Convert to numpy arrays
    noise_ops = [(np.array(op, dtype=complex), p) for op, p in noise_ops]
    if len(noise_ops) == 0:
        raise NoiseError("Input noise list is empty.")

    # Check for identity unitaries
    prob_identity = 0.
    instructions = []
    instructions_probs = []
    num_qubits = qubits_from_mat(noise_ops[0][0])
    qubits = list(range(num_qubits))
    for unitary, prob in noise_ops:
        # Check unitary
        if qubits_from_mat(unitary) != num_qubits:
            raise NoiseError("Input matrices different size.")
        if not is_unitary_matrix(unitary):
            raise NoiseError("Input matrix is not unitary.")
        if is_identity_matrix(unitary):
            prob_identity += prob
        else:
            instr = make_unitary_instruction(unitary, qubits,
                                             standard_gates=standard_gates)
            instructions.append(instr)
            instructions_probs.append(prob)
    if prob_identity > 0:
        instructions.append([{"name": "id", "qubits": [0]}])
        instructions_probs.append(prob_identity)
    return QuantumError(zip(instructions, instructions_probs))


def coherent_unitary_error(unitary):
    """
    Coherent unitary quantum error channel.

    Args:
        unitary (matrix like): unitary error matrix.

    Returns:
        QuantumError: The quantum error object.
    """
    return mixed_unitary_error([(unitary, 1)])


def pauli_error(noise_ops, standard_gates=False):
    """
    Pauli quantum error channel.

    The input should be a list of pairs (P[j], p[j]), where `P[j]` is a
    `Pauli` object and `p[j]` is a probability. All probabilities must
    sum to 1 for the input ops to be valid.

    Args:
        noise_ops (list[pair[Pauli, double]]): Pauli error terms.
        standard_gates (bool): if True return the operators as standard qobj
                               Pauli gate instructions. If false return as
                               unitary matrix qobj instructions.
                               (Default: False)

    Returns:
        QuantumError: The quantum error object.

    Raises:
        NoiseError: If depolarizing probability is less than 0 or greater than 1.
    """

    # Error checking
    if not isinstance(noise_ops, (list, tuple, zip)):
        raise NoiseError("Input noise ops is not a list.")
    noise_ops = list(noise_ops)
    if len(noise_ops) == 0:
        raise NoiseError("Input noise list is empty.")
    num_qubits = None
    for op in noise_ops:
        pauli = op[0]
        if isinstance(pauli, Pauli):
            pauli_str = pauli.to_label()
        elif isinstance(pauli, str):
            pauli_str = pauli
        else:
            raise NoiseError("Invalid Pauli input operator: {}".format(pauli))
        if num_qubits is None:
            num_qubits = len(pauli_str)
        elif num_qubits != len(pauli_str):
            raise NoiseError("Pauli's are not all of the same length.")

    # Compute Paulis as single matrix
    if standard_gates is False:
        return _pauli_error_unitary(noise_ops, num_qubits)
    # Compute as qobj Pauli gate instructions
    else:
        return _pauli_error_standard(noise_ops, num_qubits)


def _pauli_error_unitary(noise_ops, num_qubits):
    """Return Pauli error as unitary qobj instructions."""
    def single_pauli(s):
        if s == 'I':
            return standard_gate_unitary('id')
        if s == 'X':
            return standard_gate_unitary('x')
        if s == 'Y':
            return standard_gate_unitary('y')
        if s == 'Z':
            return standard_gate_unitary('z')

    prob_identity = 0.0
    pauli_circs = []
    pauli_probs = []
    for pauli, prob in noise_ops:
        if prob > 0:
            # Pauli strings go from qubit-0 on left to qubit-N on right
            # but pauli ops are tensor product of qubit-N on left to qubit-0 on right
            # We also drop identity operators to reduce dimension of matrix multiplication
            mat = 1
            qubits = []
            if isinstance(pauli, Pauli):
                pauli_str = pauli.to_label()
            else:
                pauli_str = pauli
            for qubit, s in enumerate(reversed(pauli_str)):
                if s in ['X', 'Y', 'Z']:
                    mat = np.kron(single_pauli(s), mat)
                    qubits.append(qubit)
                elif s != 'I':
                    raise NoiseError("Invalid Pauli string.")
            if mat is 1:
                prob_identity += prob
            else:
                circ = make_unitary_instruction(mat, qubits,
                                                standard_gates=False)
                pauli_circs.append(circ)
                pauli_probs.append(prob)
    if prob_identity > 0:
        pauli_probs.append(prob_identity)
        pauli_circs.append([{"name": "id", "qubits": [0]}])

    error = QuantumError(zip(pauli_circs, pauli_probs),
                         number_of_qubits=num_qubits)
    return error


def _pauli_error_standard(noise_ops, num_qubits):
    """Return Pauli error as standard Pauli gate qobj instructions."""

    def single_pauli(s):
        if s == 'I':
            return {'name': 'id'}
        if s == 'X':
            return {'name': 'x'}
        if s == 'Y':
            return {'name': 'y'}
        if s == 'Z':
            return {'name': 'z'}
    prob_identity = 0.0
    pauli_circuits = []
    pauli_probs = []
    for pauli, prob in noise_ops:
        if prob > 0:
            # Pauli strings go from qubit-0 on left to qubit-N on right
            # but pauli ops are tensor product of qubit-N on left to qubit-0 on right
            # We also drop identity operators to reduce dimension of matrix multiplication
            circuit = []
            if isinstance(pauli, Pauli):
                pauli_str = pauli.to_label()
            else:
                pauli_str = pauli
            for qubit, s in enumerate(reversed(pauli_str)):
                if s in ['X', 'Y', 'Z']:
                    instruction = single_pauli(s)
                    instruction["qubits"] = [qubit]
                    circuit.append(instruction)
                elif s != 'I':
                    raise NoiseError("Invalid Pauli string.")
            if circuit == []:
                prob_identity += prob
            else:
                pauli_circuits.append(circuit)
                pauli_probs.append(prob)
    if prob_identity > 0:
        pauli_circuits.append([{"name": "id", "qubits": [0]}])
        pauli_probs.append(prob_identity)

    error = QuantumError(zip(pauli_circuits, pauli_probs),
                         number_of_qubits=num_qubits)
    return error


def depolarizing_error(prob, num_qubits, standard_gates=False):
    """
    Depolarizing quantum error channel.

    Args:
        prob (double): completely depolarizing channel error probability.
        num_qubits (int): the number of qubits for the error channel.
        standard_gates (bool): if True return the operators as standard qobj
                               Pauli gate instructions. If false return as
                               unitary matrix qobj instructions.
                               (Default: False)

    Returns:
        QuantumError: The quantum error object.
    """

    if prob < 0 or prob > 1:
        raise NoiseError("Depolarizing probability must be in between 0 and 1.")

    # Rescale completely depolarizing channel error probs
    # with the identity component removed
    num_terms = 4 ** num_qubits  # terms in completely depolarizing channel
    prob_error = prob / num_terms
    prob_iden = 1 - (num_terms - 1) * prob_error  # subtract off non-identity terms
    probs = [prob_iden] + (num_terms - 1) * [prob_error]
    # Generate pauli strings. The order doesn't matter as long
    # as the all identity string is first.
    paulis = ["".join(tup) for tup in product(['I', 'X', 'Y', 'Z'], repeat=num_qubits)]
    return pauli_error(zip(paulis, probs), standard_gates=standard_gates)


def thermal_relaxation_error(t1, t2, time, excited_state_population=0):
    """
    Single-qubit thermal relaxation quantum error channel.

    Args:
        t1 (double > 0): the T_1 relaxation time constant.
        t2 (double > 0): the T_2 relaxation time constant.
        gate_time (double >= 0): the time period for relaxation error.
        excited_state_population (double): the population of |1> state at
                                           equilibrium (default: 0).

    Returns:
        QuantumError: a quantum error object for a noise model.

    Additional information:
        For parameters to be valid T_2 <= 2 * T_1.
        If T_2 <= T_1 the error can be expressed as a mixed reset and unitary
        error channel.
        If T_1 < T_2 <= 2 * T_1 the error must be expressed as a general
        non-unitary Kraus error channel.
    """
    if excited_state_population < 0:
        raise NoiseError("Invalid excited state population " +
                            "({} < 0).".format(excited_state_population))
    if excited_state_population > 1:
        raise NoiseError("Invalid excited state population " +
                            "({} > 1).".format(excited_state_population))
    if time < 0:
        raise NoiseError("Invalid gate_time ({} < 0)".format(time))
    if t1 <= 0:
        raise NoiseError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
    if t2 <= 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
    if t2 - 2 * t1 > 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")

    # T1 relaxation rate
    if t1 == np.inf:
        rate1 = 0
        p_reset = 0
    else:
        rate1 = 1 / t1
        p_reset = 1 - np.exp(-time * rate1)
    # T2 dephasing rate
    if t2 == np.inf:
        rate2 = 0
        exp_t2 = 1
    else:
        rate2 = 1 / t2
        exp_t2 = np.exp(-time * rate2)
    # Qubit state equilibrium probabilities
    p0 = 1 - excited_state_population
    p1 = excited_state_population

    if t2 > t1:
        # If T_2 > T_1 we must express this as a Kraus channel
        # We start with the Choi-matrix representation:
        choi = np.array([[1 - p1 * p_reset, 0, 0, exp_t2],
                         [0, p1 * p_reset, 0, 0],
                         [0, 0, p0 * p_reset, 0],
                         [exp_t2, 0, 0, 1 - p0 * p_reset]])
        # Find canonical Kraus operators by eigendecomposition of Choi-matrix
        kraus = choi2kraus(choi)
        return QuantumError(kraus)
    else:
        # If T_2 < T_1 we can express this channel as a probabilistic
        # mixture of reset operations and unitary errors:
        circuits = [
            [{'name': 'id', 'qubits': [0]}],
            [{'name': 'z', 'qubits': [0]}],
            [{'name': 'reset', 'qubits': [0]}],
            [{'name': 'reset', 'qubits': [0]}, {'name': 'x', 'qubits': [0]}]]
        # Probability
        p_reset0 = p_reset * p0
        p_reset1 = p_reset * p1
        p_z = (1 - p_reset) * (1 - np.exp(-time * (rate2 - rate1))) / 2
        p_identity = 1 - p_z - p_reset0 - p_reset1
        probabilities = [p_identity, p_z, p_reset0, p_reset1]
        return QuantumError(zip(circuits, probabilities))


def phase_amplitude_damping_error(param_amp, param_phase,
                                  excited_state_population=0,
                                  canonical_kraus=True):
    """
    Single-qubit combined phase and amplitude damping quantum error channel.

    Args:
        param_amp (double): the amplitude damping error parameter.
        param_phase (double): the phase damping error parameter.
        excited_state_population (double): the population of |1> state at
                                           equilibrium (default: 0).
        canonical_kraus (bool): Convert input Kraus matrices into the
                                canoical Kraus representation (default: True)

    Returns:
        QuantumError: a quantum error object for a noise model.

    Additional information:
        The single-qubit combined phase and amplitude damping channel is
        described by the following Kraus matrices:
            A0 = sqrt(1 - p1) * [[1, 0],
                                 [0, sqrt(1 - a - b)]]
            A1 = sqrt(1 - p1) * [[0, sqrt(a)],
                                 [0, 0]]
            A2 = sqrt(1 - p1) * [[0, 0],
                                  [0, sqrt(b)]]
            B0 = sqrt(p1) * [[sqrt(1 - a - b), 0],
                             [0, 1]]
            B1 = sqrt(p1) * [[0, 0],
                             [sqrt(a), 0]]
            B2 = sqrt(p1) * [[sqrt(b), 0],
                             [0, 0]]
            where a = param_amp, b = param_phase, p1 = excited_state_population
        The equilibrium state after infinitely many applications of the
        channel is:
            rho = [[1 - p1, 0]],
                   [0, p1]]
    """

    if param_amp < 0:
        raise NoiseError("Invalid amplitude damping to |0> parameter " +
                            "({} < 0)".format(param_amp))
    if param_phase < 0:
        raise NoiseError("Invalid phase damping parameter " +
                            "({} < 0)".format(param_phase))
    if param_phase + param_amp > 1:
        raise NoiseError("Invalid amplitude and phase damping parameters " +
                            "({} + {} > 1)".format(param_phase, param_amp))
    if excited_state_population < 0:
        raise NoiseError("Invalid excited state population " +
                            "({} < 0).".format(excited_state_population))
    if excited_state_population > 1:
        raise NoiseError("Invalid excited state population " +
                            "({} > 1).".format(excited_state_population))
    c0 = np.sqrt(1 - excited_state_population)
    c1 = np.sqrt(excited_state_population)
    param = 1 - param_amp - param_phase
    # Damping ops to 0 state
    A0 = c0 * np.array([[1, 0], [0, np.sqrt(param)]], dtype=complex)
    A1 = c0 * np.array([[0, np.sqrt(param_amp)], [0, 0]], dtype=complex)
    A2 = c0 * np.array([[0, 0], [0, np.sqrt(param_phase)]], dtype=complex)
    # Damping ops to 1 state
    B0 = c1 * np.array([[np.sqrt(param), 0], [0, 1]], dtype=complex)
    B1 = c1 * np.array([[0, 0], [np.sqrt(param_amp), 0]], dtype=complex)
    B2 = c1 * np.array([[np.sqrt(param_phase), 0], [0, 0]], dtype=complex)
    # Select non-zero ops
    noise_ops = [a for a in [A0, A1, A2, B0, B1, B2]
                 if np.linalg.norm(a) > 1e-10]
    return kraus_error(noise_ops, canonical_kraus=canonical_kraus)


def amplitude_damping_error(param_amp, excited_state_population=0,
                            canonical_kraus=True):
    """
    Single-qubit generalized amplitude damping quantum error channel.

    Args:
        param_amp (double): the amplitude damping parameter.
        excited_state_population (double): the population of |1> state at
                                           equilibrium (default: 0).
        canonical_kraus (bool): Convert input Kraus matrices into the
                                canoical Kraus representation (default: True)

    Returns:
        QuantumError: a quantum error object for a noise model.

    Additional information:
        The single-qubit amplitude damping channel is
        described by the following Kraus matrices:
            A0 = sqrt(1 - p1) * [[1, 0],
                                 [0, sqrt(1 - a)]]
            A1 = sqrt(1 - p1) * [[0, sqrt(a)],
                                 [0, 0]]
            B0 = sqrt(p1) * [[sqrt(1 - a), 0],
                             [0, 1]]
            B1 = sqrt(p1) * [[0, 0],
                             [sqrt(a), 0]]
            where a = param_amp, p1 = excited_state_population
        The equilibrium state after infinitely many applications of the
        channel is:
            rho = [[1 - p1, 0]],
                   [0, p1]]
    """
    return phase_amplitude_damping_error(param_amp, 0,
                                         excited_state_population=excited_state_population,
                                         canonical_kraus=canonical_kraus)


def phase_damping_error(param_phase, canonical_kraus=True):
    """
    Single-qubit combined phase and amplitude damping quantum error channel.

    Args:
        param_phase (double): the phase damping parameter.
        canonical_kraus (bool): Convert input Kraus matrices into the
                                canoical Kraus representation (default: True)

    Returns:
        QuantumError: a quantum error object for a noise model.

    Additional information:
        The single-qubit combined phase and amplitude damping channel is
        described by the following Kraus matrices:
            A0 = [[1, 0],
                  [0, sqrt(1 - b)]]
            A2 = [[0, 0],
                  [0, sqrt(b)]]
            where a = param_amp, b = param_phase, p1 = excited_state_population
        The equilibrium state after infinitely many applications of the
        channel is:
            rho = [[rho[0, 0], 0]],
                   [0, rho[1, 1]]]
    """

    return phase_amplitude_damping_error(0, param_phase,
                                         excited_state_population=0,
                                         canonical_kraus=canonical_kraus)
