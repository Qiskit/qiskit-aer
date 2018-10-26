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


def mixed_unitary_error(unitaries, probabilities, threshold=1e-10):
    """
    Mixed unitary quantum error channel.

    Args:
        unitaries (list[matrix like]): unitary error matricies.
        probabilities (list[float]): the error probabilities.
        threshold (double): threshold for checking if unitary is identity.

    Returns:
        QuantumError: The quantum error object.

    Raises:
        AerNoiseError: if error parameters are invalid.
    """

    # Error checking
    if len(unitaries) != len(probabilities):
        raise AerNoiseError("Length of unitaries does not match length of probabilities")

    # Check for identity unitaries
    prob_identity = 0.
    instructions = []
    instructions_probs = []
    qubits = list(range(qubits_from_mat(unitaries[0])))
    for unitary, prob in zip(unitaries, probabilities):
        if is_identity(unitary, threshold):
            prob_identity += prob
        else:
            instructions.append([make_unitary_instruction(unitary, qubits, threshold)])
            instructions_probs.append(prob)
    if prob_identity > threshold:
        instructions.append([{"name": "id", "qubits": [0]}])
        instructions_probs.append(prob_identity)

    return QuantumError(zip(instructions_probs, instructions))


def coherent_unitary_error(unitary, threshold=1e-10):
    """
    Coherent unitary quantum error channel.

    Args:
        unitary (matrix like): unitary error matrix.
        threshold (double): threshold for checking if unitary is identity.

    Returns:
        QuantumError: The quantum error object.
    """
    return mixed_unitary_error([unitary], [1], threshold)


def pauli_error(pauli_dict, threshold=1e-10, as_matrix=False):
    """
    Pauli quantum error channel.

    Args:
        pauli_dict (dict[str, double]): Pauli error specification.
        threshold (double): threshold for checking if unitary is identity.
        as_matrix (bool): return the operators as unitary matrix gates,
                          if False return as Pauli circuit operations.

    Returns:
        QuantumError: The quantum error object.

    Raises:
        AerNoiseError: If depolarizing probability is less than 0 or greater than 1.
    """

    # Check Pauli strings
    keys = list(pauli_dict.keys())
    num_qubits = len(keys[0])
    for key in keys[1:]:
        if len(key) != num_qubits:
            raise AerNoiseError("Pauli strings are not all of the same length.")

    # Compute Paulis as single matrix
    if as_matrix is True:
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
        for pauli_str, prob in pauli_dict.items():
            if prob > 0:
                # Pauli strings go from qubit-0 on left to qubit-N on right
                # but pauli ops are tensor product of qubit-N on left to qubit-0 on right
                # We also drop identity operators to reduce dimension of matrix multiplication
                mat = np.eye(1)
                qubits = []
                for qubit, s in enumerate(pauli_str):
                    if s in ['X', 'Y', 'Z']:
                        mat = np.kron(single_pauli(s), mat)
                        qubits.append(qubit)
                    elif s != 'I':
                        raise AerNoiseError("Invalid Pauli string.")
                if list(mat) == list(np.eye(1)):
                    prob_identity += prob
                else:
                    pauli_mats.append(mat)
                    pauli_probs.append(prob)
                    pauli_qubits.append(qubits)
        if prob_identity > 0:
            pauli_probs.append(prob_identity)
            pauli_mats.append(np.eye(2))
            pauli_qubits.append([0])

        error = mixed_unitary_error(pauli_mats, pauli_probs)
        # Update Pauli qubits
        new_circuits = []
        for circ, qubits in zip(error._noise_circuits, pauli_qubits):
            circ[0]["qubits"] = qubits
            new_circuits.append(circ)
        error._noise_circuits = new_circuits
        error._number_of_qubits = num_qubits

        return error

    # Compute as qobj Pauli gate instructions
    else:
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
        for pauli_str, prob in pauli_dict.items():
            if prob > 0:
                # Pauli strings go from qubit-0 on left to qubit-N on right
                # but pauli ops are tensor product of qubit-N on left to qubit-0 on right
                # We also drop identity operators to reduce dimension of matrix multiplication
                circuit = []
                for qubit, s in enumerate(pauli_str):
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

        error = QuantumError(zip(pauli_probs, pauli_circuits), number_of_qubits=num_qubits)
        return error


def depolarizing_error(prob, num_qubits, threshold=1e-10, as_matrix=False):
    """
    Depolarizing quantum error channel.

    Args:
        prob (double): completely depolarizing channel error probability.
        num_qubits (int): the number of qubits for the error channel.
        threshold (double): threshold for checking if unitary is identity.
        as_matrix (bool): return the operators as unitary matrix gates,
                          if False return as Pauli circuit operations.

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
    return pauli_error(dict(zip(paulis, probs)), as_matrix=as_matrix)


def thermal_relaxation_error(t1, t2, time, polarization=0):
    """
    Single-qubit thermal relaxation quantum error channel.

    Args:
        t1 (double > 0): the T_1 relaxation time constant.
        t2 (double > 0): the T_2 relaxation time constant.
        gate_time (double >= 0): the time period for relaxation error.
        polarization (double): the population of |1> state at equilibrium
                               (default: 0).

    Returns:
        QuantumError: a quantum error object for a noise model.

    Additional information:
        For parameters to be valid T_2 <= 2 * T_1.
        If T_2 <= T_1 the error can be expressed as a mixed reset and unitary
        error channel.
        If T_1 < T_2 <= 2 * T_1 the error must be expressed as a general
        non-unitary Kraus error channel.
    """

    if polarization < 0:
        raise AerNoiseError("Invalid amplitude damping polarization parameter " +
                            "({} < 0).".format(polarization))
    if polarization > 1:
        raise AerNoiseError("Invalid amplitude damping polarization parameter " +
                            "({} > 1).".format(polarization))
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
        rate1 = 2 * np.pi / t1
    # T2 dephasing rate
    if t2 == np.inf:
        rate2 = - rate1
    else:
        rate2 = 4 * np.pi / t2 - rate1
    # Relaxation probabilities
    pr = 1 - np.exp(-rate1 * time)
    p0 = 1 - polarization
    p1 = polarization

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
        return QuantumError(zip(probabilities, circuits))


def phase_amplitude_damping_error(param_amp, param_phase, polarization=0):
    """
    Single-qubit combined phase and amplitude damping quantum error channel.

    Args:
        param_amp (double): the amplitude damping error parameter.
        param_phase (double): the phase damping error parameter.
        polarization (double): the population of |1> state at equilibrium
                               (default: 0).

    Returns:
        QuantumError: a quantum error object for a noise model.

    Additional information:
        The single-qubit amplitude damping channel is described by the
        following Kraus matrices:
            A0 = sqrt(1 - p1) * [[1, 0], [0, sqrt(1 - a - b)]]
            A1 = sqrt(1 - p1) * [[0, sqrt(a)], [0, 0]]
            A2 = sqrt(p1) * [[sqrt(1 - a - b), 0], [0, 0]]
            A3 = sqrt(p1) * [[0, 0], [sqrt(a), 0]]
            A4 = [[0, 0], [0, sqrt(b)]]
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
    if polarization < 0:
        raise AerNoiseError("Invalid amplitude damping polarization parameter " +
                            "({} < 0)".format(polarization))
    if polarization > 1:
        raise AerNoiseError("Invalid amplitude damping polarization parameter " +
                            "({} > 1)".format(polarization))
    c0 = np.sqrt(1 - polarization)
    c1 = np.sqrt(polarization)
    par = np.sqrt(1 - param_amp - param_phase)
    A0 = c0 * np.array([[1, 0], [0, par]], dtype=complex)
    A1 = c0 * np.array([[0, np.sqrt(param_amp)], [0, 0]], dtype=complex)
    A2 = c1 * np.array([[par, 0], [0, 0]], dtype=complex)
    A3 = c1 * np.array([[0, 0], [np.sqrt(param_amp), 0]], dtype=complex)
    A4 = np.array([[0, 0], [0, np.sqrt(param_phase)]], dtype=complex)
    return QuantumError((A0, A1, A2, A3, A4))


def amplitude_damping_error(param_amp, polarization=0):
    """
    Single-qubit generalized amplitude damping quantum error channel.

    Args:
        param_amp (double): the amplitude damping parameter.
        polarization (double): the population of |1> state at equilibrium
                               (default: 0).

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
    return phase_amplitude_damping_error(param_amp, 0, polarization)


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
