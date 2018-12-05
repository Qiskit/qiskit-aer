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

from ..aernoiseerror import AerNoiseError
from ..quantum_error import QuantumError
from ..noise_utils import (make_unitary_instruction, qubits_from_mat,
                           is_identity, standard_gate_unitary)

# Temporary try block until qiskit 0.7 is released to stable
try:
    from qiskit.quantum_info.operators.pauli import Pauli
except:
    Pauli = None.__class__


def kraus_error(noise_ops, standard_gates=True, threshold=1e-10):
    """Kraus error channel.

    Args:
        noise_ops (list[matrix]): Kraus matrices.
        standard_gates (bool): Check if input matrices are standard gates.
        threshold (double): The threshold parameter for testing if
                            Kraus operators are unitary (default: 1e-10).

    Returns:
        QuantumError: The quantum error object.

    Raises:
        AerNoiseError: if error parameters are invalid.
    """
    if not isinstance(noise_ops, (list, tuple)):
        raise AerNoiseError("Invalid Kraus error input.")
    if len(noise_ops) == 0:
        raise AerNoiseError("Kraus error noise_ops must not be empty.")
    return QuantumError([np.array(a, dtype=complex) for a in noise_ops],
                        standard_gates=standard_gates,
                        threshold=1e-10)


def mixed_unitary_error(noise_ops, number_of_qubits=None, threshold=1e-10):
    """
    Mixed unitary quantum error channel.

    The input should be a list of pairs (U[j], p[j]), where `U[j]` is a
    unitary matrix and `p[j]` is a probability. All probabilities must
    sum to 1 for the input ops to be valid.

    Args:
        noise_ops (list[pair[matrix, double]]): unitary error matricies.
        threshold (double): threshold for checking if unitary is identity.

    Returns:
        QuantumError: The quantum error object.

    Raises:
        AerNoiseError: if error parameters are invalid.
    """

    # Error checking
    if not isinstance(noise_ops, (list, tuple, zip)):
        raise AerNoiseError("Input noise ops is not a list.")
    noise_ops = list(noise_ops)
    if len(noise_ops) == 0:
        raise AerNoiseError("Input noise list is empty.")

    # Check for identity unitaries
    prob_identity = 0.
    instructions = []
    instructions_probs = []
    qubits = list(range(qubits_from_mat(noise_ops[0][0])))
    for unitary, prob in noise_ops:
        if is_identity(unitary, threshold):
            prob_identity += prob
        else:
            instructions.append(make_unitary_instruction(unitary, qubits, threshold))
            instructions_probs.append(prob)
    if prob_identity > threshold:
        instructions.append([{"name": "id", "qubits": [0]}])
        instructions_probs.append(prob_identity)

    return QuantumError(zip(instructions, instructions_probs),
                        number_of_qubits=number_of_qubits)


def coherent_unitary_error(unitary, threshold=1e-10):
    """
    Coherent unitary quantum error channel.

    Args:
        unitary (matrix like): unitary error matrix.
        threshold (double): threshold for checking if unitary is identity.

    Returns:
        QuantumError: The quantum error object.
    """
    return mixed_unitary_error([(unitary, 1)], threshold)


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
                               unitary matrix qobj instructions. (Default: False)

    Returns:
        QuantumError: The quantum error object.

    Raises:
        AerNoiseError: If depolarizing probability is less than 0 or greater than 1.
    """

    # Error checking
    if not isinstance(noise_ops, (list, tuple, zip)):
        raise AerNoiseError("Input noise ops is not a list.")
    noise_ops = list(noise_ops)
    if len(noise_ops) == 0:
        raise AerNoiseError("Input noise list is empty.")
    num_qubits = None
    for op in noise_ops:
        pauli = op[0]
        if isinstance(pauli, Pauli):
            pauli_str = pauli.to_label()
        elif isinstance(pauli, str):
            pauli_str = pauli
        else:
            raise AerNoiseError("Invalid Pauli input operator: {}".format(pauli))
        if num_qubits is None:
            num_qubits = len(pauli_str)
        elif num_qubits != len(pauli_str):
            raise AerNoiseError("Pauli's are not all of the same length.")

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
    pauli_mats = []
    pauli_qubits = []
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
                    raise AerNoiseError("Invalid Pauli string.")
            if mat is 1:
                prob_identity += prob
            else:
                pauli_mats.append(mat)
                pauli_probs.append(prob)
                pauli_qubits.append(qubits)
    if prob_identity > 0:
        pauli_probs.append(prob_identity)
        pauli_mats.append(np.eye(2))
        pauli_qubits.append([0])

    error = mixed_unitary_error(zip(pauli_mats, pauli_probs),
                                number_of_qubits=num_qubits)
    # Update Pauli qubits
    new_circuits = []
    for circ, qubits in zip(error._noise_circuits, pauli_qubits):
        circ[0]["qubits"] = qubits
        new_circuits.append(circ)
    error._noise_circuits = new_circuits
    error._number_of_qubits = num_qubits

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
                    raise AerNoiseError("Invalid Pauli string.")
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
                               unitary matrix qobj instructions. (Default: False)

    Returns:
        QuantumError: The quantum error object.
    """

    if prob < 0 or prob > 1:
        raise AerNoiseError("Depolarizing probability must be in between 0 and 1.")

    # Rescale completely depolarizing channel error probs
    # with the identity component removed
    probs = ((4 ** num_qubits) - 1) * [prob / 4 ** num_qubits]
    # Add identity probability of identity component
    probs = [1.0 - np.sum(probs)] + probs

    # Generate pauli strings. The order doesn't matter as long
    # as the all identity string is first.
    paulis = ["".join(tup) for tup in product(['I', 'X', 'Y', 'Z'], repeat=num_qubits)]
    return pauli_error(zip(paulis, probs), standard_gates=standard_gates)


def thermal_relaxation_error(t1, t2, time, excited_state_population=0):
    """
    Single-qubit thermal relaxation quantum error channel.
    TODO: check kraus form

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
        raise AerNoiseError("Invalid excited state population " +
                            "({} < 0).".format(excited_state_population))
    if excited_state_population > 1:
        raise AerNoiseError("Invalid excited state population " +
                            "({} > 1).".format(excited_state_population))
    if time < 0:
        raise AerNoiseError("Invalid gate_time ({} < 0)".format(time))
    if t1 <= 0:
        raise AerNoiseError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
    if t2 <= 0:
        raise AerNoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
    if t2 > 2 * t1:
        raise AerNoiseError("Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")

    # T1 relaxation rate
    if t1 == np.inf:
        rate1 = 0
    else:
        rate1 = 1 / t1
    # T2 dephasing rate
    if t2 == np.inf:
        rate2 = - rate1
    else:
        rate2 = 2 / t2 - rate1
    # Relaxation probabilities
    pr = 1 - np.exp(-rate1 * time)
    p0 = 1 - excited_state_population
    p1 = excited_state_population

    if t2 > t1:
        # If T_2 > T_1 we must express this as a Kraus channel
        # We start with the Choi-matrix representation:
        p2 = np.exp(-0.5 * (rate1 + rate2) * time)
        choi = np.array([[1 - p1 * pr, 0, 0, p2],
                         [0, p1 * pr, 0, 0],
                         [0, 0, p0 * pr, 0],
                         [p2, 0, 0, 1 - p0 * pr]])
        # Find canonical Kraus operators by eigendecomposition of Choi-matrix
        w, v = np.linalg.eigh(choi)
        kraus = [np.sqrt(val) * vec.reshape((2, 2), order='F')
                 for val, vec in zip(w, v.T) if val > 0]
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
        p_reset0 = pr * p0
        p_reset1 = pr * p1
        p_z = 0.5 * (1 - pr) * (1 - np.exp(-0.5 * (rate2 - rate1) * time))
        p_identity = 1 - p_z - p_reset0 - p_reset1
        probabilities = [p_identity, p_z, p_reset0, p_reset1]
        return QuantumError(zip(circuits, probabilities))


def phase_amplitude_damping_error(param_amp, param_phase,
                                  excited_state_population=0):
    """
    Single-qubit combined phase and amplitude damping quantum error channel.

    Args:
        param_amp (double): the amplitude damping error parameter.
        param_phase (double): the phase damping error parameter.
        excited_state_population (double): the population of |1> state at
                                           equilibrium (default: 0).

    Returns:
        QuantumError: a quantum error object for a noise model.

    Additional information:
        The single-qubit amplitude damping channel is described by the
        following Kraus matrices:
            A0 = sqrt(1 - p1) * [[1, 0],
                                 [0, sqrt(1 - a - b)]]
            A1 = sqrt(1 - p1) * [[0, sqrt(a)],
                                 [0, 0]]
            A2 = sqrt(p1) * [[sqrt(1 - a - b), 0],
                             [0, 0]]
            A3 = sqrt(p1) * [[0, 0],
                             [sqrt(a), 0]]
            A4 = [[0, 0],
                  [0, sqrt(b)]]
        The equilibrium state after infinitely many applications of the
        channel is:
            rho = [[1 - p1]], [0, p1]]
    """

    if param_amp < 0:
        raise AerNoiseError("Invalid amplitude damping to |0> parameter " +
                            "({} < 0)".format(param_amp))
    if param_phase < 0:
        raise AerNoiseError("Invalid phase damping parameter " +
                            "({} < 0)".format(param_phase))
    if param_phase + param_amp > 1:
        raise AerNoiseError("Invalid amplitude and phase damping parameters " +
                            "({} + {} > 1)".format(param_phase, param_amp))
    if excited_state_population < 0:
        raise AerNoiseError("Invalid excited state population " +
                            "({} < 0).".format(excited_state_population))
    if excited_state_population > 1:
        raise AerNoiseError("Invalid excited state population " +
                            "({} > 1).".format(excited_state_population))
    c0 = np.sqrt(1 - excited_state_population)
    c1 = np.sqrt(excited_state_population)
    par = np.sqrt(1 - param_amp - param_phase)
    A0 = c0 * np.array([[1, 0], [0, par]], dtype=complex)
    A1 = c0 * np.array([[0, np.sqrt(param_amp)], [0, 0]], dtype=complex)
    A2 = c1 * np.array([[par, 0], [0, 0]], dtype=complex)
    A3 = c1 * np.array([[0, 0], [np.sqrt(param_amp), 0]], dtype=complex)
    A4 = np.array([[0, 0], [0, np.sqrt(param_phase)]], dtype=complex)
    return QuantumError((A0, A1, A2, A3, A4))


def amplitude_damping_error(param_amp, excited_state_population=0):
    """
    Single-qubit generalized amplitude damping quantum error channel.

    Args:
        param_amp (double): the amplitude damping parameter.
        excited_state_population (double): the population of |1> state at
                                           equilibrium (default: 0).

    Returns:
        QuantumError: a quantum error object for a noise model.

    Additional information:
        The single-qubit amplitude damping channel is described by the
        following Kraus matrices:
            A0 = sqrt(1 - p1) * [[1, 0], [0, sqrt(1 - a)]]
            A1 = sqrt(1 - p1) * [[0, sqrt(a)], [0, 0]]
            A2 = sqrt(p1) * [[sqrt(1 - a), 0], [0, 0]]
            A3 = sqrt(p1) * [[0, 0], [sqrt(a), 0]]
        The equilibrium state after infinitely many applications of the
        channel is:
            rho = [[1 - p1]], [0, p1]]
    """
    return phase_amplitude_damping_error(param_amp, 0,
                                         excited_state_population)


def phase_damping_error(param_phase):
    """
    Single-qubit combined phase and amplitude damping quantum error channel.

    Args:
        param_phase (double): the phase damping parameter.

    Returns:
        QuantumError: a quantum error object for a noise model.

    Additional information:
        The single-qubit combined phase and amplitude damping channel
        with phase damping parameter 'a', and amplitude damping param 'b'
        is described by the following Kraus matrices:
            A0 = [[1, 0], [0, sqrt(1 - a - b)]]
            A1 = [[0, 0], [0, sqrt(a)]]
            A2 = [[0, sqrt(b)], [0, 0]]
    """

    return phase_amplitude_damping_error(0, param_phase, 0)
