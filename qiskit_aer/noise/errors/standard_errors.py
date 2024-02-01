# This code is part of Qiskit.
#
# (C) Copyright IBM 2018-2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Standard quantum computing error channels for Aer.
"""

import itertools as it

import numpy as np

from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import IGate, XGate, ZGate
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Pauli
from qiskit.quantum_info.operators.channel import Choi, Kraus
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from .quantum_error import QuantumError
from ..noiseerror import NoiseError


def kraus_error(noise_ops, canonical_kraus=False):
    """
    Return a Kraus quantum error channel.

    Args:
        noise_ops (list[matrix]): Kraus matrices.
        canonical_kraus (bool): Convert input Kraus matrices into the
                                canonical Kraus representation (default: False)

    Returns:
        QuantumError: The quantum error object.

    Raises:
        NoiseError: if error parameters are invalid.
    """
    if not isinstance(noise_ops, (list, tuple)):
        raise NoiseError("Invalid Kraus error input.")
    if not noise_ops:
        raise NoiseError("Kraus error noise_ops must not be empty.")

    kraus = Kraus(noise_ops)
    if canonical_kraus:
        # Convert to Choi and back to get canonical Kraus
        kraus = Kraus(Choi(kraus))
    return QuantumError(kraus)


def mixed_unitary_error(noise_ops):
    """
    Return a mixed unitary quantum error channel.

    The input should be a list of pairs ``(U[j], p[j])``, where
    ``U[j]`` is a unitary matrix and ``p[j]`` is a probability. All
    probabilities must sum to 1 for the input ops to be valid.

    Args:
        noise_ops (list[pair[matrix, double]]): unitary error matrices.

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
    if not noise_ops:
        raise NoiseError("Input noise list is empty.")

    # Check for identity unitaries
    prob_identity = 0.0
    instructions = []
    instructions_probs = []
    num_qubits = int(np.log2(noise_ops[0][0].shape[0]))
    if noise_ops[0][0].shape != (2**num_qubits, 2**num_qubits):
        raise NoiseError("A unitary matrix in input noise_ops is not a multi-qubit matrix.")
    for unitary, prob in noise_ops:
        # Check unitary
        if unitary.shape != noise_ops[0][0].shape:
            raise NoiseError("Input matrices different size.")
        if not is_unitary_matrix(unitary):
            raise NoiseError("Input matrix is not unitary.")
        if is_identity_matrix(unitary):
            prob_identity += prob
        else:
            instr = UnitaryGate(unitary)
            instructions.append(instr)
            instructions_probs.append(prob)
    if prob_identity > 0:
        instructions.append(IGate())
        instructions_probs.append(prob_identity)
    return QuantumError(zip(instructions, instructions_probs))


def coherent_unitary_error(unitary):
    """
    Return a coherent unitary quantum error channel.

    Args:
        unitary (matrix like): unitary error matrix.

    Returns:
        QuantumError: The quantum error object.
    """
    return mixed_unitary_error([(unitary, 1)])


def pauli_error(noise_ops):
    """
    Return a mixed Pauli quantum error channel.

    The input should be a list of pairs ``(P[j], p[j])``, where
    ``P[j]`` is a ``Pauli`` object or string label, and ``p[j]`` is a
    probability. All probabilities must sum to 1 for the input ops to
    be valid.

    Args:
        noise_ops (list[pair[Pauli, double]]): Pauli error terms.

    Returns:
        QuantumError: The quantum error object.

    Raises:
        NoiseError: If depolarizing probability is less than 0 or greater than 1.
    """
    # Error checking
    if not isinstance(noise_ops, (list, tuple, zip)):
        raise NoiseError("Input noise ops is not a list.")
    noise_ops = list(noise_ops)
    if not noise_ops:
        raise NoiseError("Input noise list is empty.")

    ops, probs = zip(*noise_ops)  # unzip

    def to_pauli(op):
        if isinstance(op, Pauli):
            return op
        elif isinstance(op, str):
            try:
                return Pauli(op)
            except QiskitError:
                pass
        raise NoiseError("Invalid Pauli input operator: {}".format(op))

    paulis = [to_pauli(op) for op in ops]

    num_qubits = paulis[0].num_qubits
    for pauli in paulis:
        if num_qubits != pauli.num_qubits:
            raise NoiseError("Pauli's are not all of the same length.")

    return QuantumError(zip(paulis, probs))


def depolarizing_error(param, num_qubits):
    r"""
    Return a depolarizing quantum error channel.

    The depolarizing channel is defined as:

    .. math::

        E(ρ) = (1 - λ) ρ + λ \text{Tr}[ρ] \frac{I}{2^n}

    with :math:`0 \le λ \le 4^n / (4^n - 1)`

    where :math:`λ` is the depolarizing error param and :math:`n` is the
    number of qubits.

    * If :math:`λ = 0` this is the identity channel :math:`E(ρ) = ρ`
    * If :math:`λ = 1` this is a completely depolarizing channel
      :math:`E(ρ) = I / 2^n`
    * If :math:`λ = 4^n / (4^n - 1)` this is a uniform Pauli
      error channel: :math:`E(ρ) = \sum_j P_j ρ P_j / (4^n - 1)` for
      all :math:`P_j != I`.

    Args:
        param (double): depolarizing error parameter.
        num_qubits (int): the number of qubits for the error channel.

    Returns:
        QuantumError: The quantum error object.

    Raises:
        NoiseError: If noise parameters are invalid.
    """
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise NoiseError("num_qubits must be a positive integer.")
    # Check that the depolarizing parameter gives a valid CPTP
    num_terms = 4**num_qubits
    max_param = num_terms / (num_terms - 1)
    if param < 0 or param > max_param:
        raise NoiseError("Depolarizing parameter must be in between 0 " "and {}.".format(max_param))

    # Rescale completely depolarizing channel error probs
    # with the identity component removed
    prob_iden = 1 - param / max_param
    prob_pauli = param / num_terms
    probs = [prob_iden] + (num_terms - 1) * [prob_pauli]

    # Generate pauli strings. The order doesn't matter as long
    # as the all identity string is first.
    paulis = [Pauli("".join(tup)) for tup in it.product(["I", "X", "Y", "Z"], repeat=num_qubits)]
    return QuantumError(zip(paulis, probs))


def reset_error(prob0, prob1=0):
    r"""
    Return a single qubit reset quantum error channel.

    The error channel returned is given by the map

    .. math::

        E(ρ) = (1 - p_0 - p_1) ρ + \text{Tr}[ρ] \left(
                p_0 |0 \rangle\langle 0|
                + p_1 |1 \rangle\langle 1| \right)

    where the probability of no reset is given by :math:`1 - p_0 - p_1`.

    Args:
        prob0 (double): reset probability to :math:`|0\rangle`.
        prob1 (double): reset probability to :math:`|1\rangle`.

    Returns:
        QuantumError: the quantum error object.

    Raises:
        NoiseError: If noise parameters are invalid.
    """
    if prob0 < 0 or prob1 < 0 or prob0 > 1 or prob1 > 1 or (prob0 + prob1) > 1:
        raise NoiseError("Invalid reset probabilities.")
    noise_ops = [
        ([(IGate(), [0])], 1 - prob0 - prob1),
        ([(Reset(), [0])], prob0),
        ([(Reset(), [0]), (XGate(), [0])], prob1),
    ]
    return QuantumError(noise_ops)


# pylint: disable=invalid-name
def thermal_relaxation_error(t1, t2, time, excited_state_population=0):
    r"""
    Return a single-qubit thermal relaxation quantum error channel.

    Args:
        t1 (double): the :math:`T_1` relaxation time constant.
        t2 (double): the :math:`T_2` relaxation time constant.
        time (double): the gate time for relaxation error.
        excited_state_population (double): the population of :math:`|1\rangle`
                                           state at equilibrium (default: 0).

    Returns:
        QuantumError: a quantum error object for a noise model.

    Raises:
        NoiseError: If noise parameters are invalid.

    Additional information:
        * For parameters to be valid :math:`T_1` and :math:`T_2` must
          satisfy :math:`T_2 \le 2 T_1`.

        * If :math:`T_2 \le T_1` the error can be expressed as a mixed
          reset and unitary error channel.

        * If :math:`T_1 < T_2 \le 2 T_1` the error must be expressed as a
          general non-unitary Kraus error channel.
    """
    if excited_state_population < 0:
        raise NoiseError(
            "Invalid excited state population " "({} < 0).".format(excited_state_population)
        )
    if excited_state_population > 1:
        raise NoiseError(
            "Invalid excited state population " "({} > 1).".format(excited_state_population)
        )
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
        chan = Choi(
            np.array(
                [
                    [1 - p1 * p_reset, 0, 0, exp_t2],
                    [0, p1 * p_reset, 0, 0],
                    [0, 0, p0 * p_reset, 0],
                    [exp_t2, 0, 0, 1 - p0 * p_reset],
                ]
            )
        )
        return QuantumError(Kraus(chan))
    else:
        # If T_2 < T_1 we can express this channel as a probabilistic
        # mixture of reset operations and unitary errors:
        circuits = [
            [(IGate(), [0])],
            [(ZGate(), [0])],
            [(Reset(), [0])],
            [(Reset(), [0]), (XGate(), [0])],
        ]
        # Probability
        p_reset0 = p_reset * p0
        p_reset1 = p_reset * p1
        p_z = (1 - p_reset) * (1 - np.exp(-time * (rate2 - rate1))) / 2
        p_identity = 1 - p_z - p_reset0 - p_reset1
        probabilities = [p_identity, p_z, p_reset0, p_reset1]
        return QuantumError(zip(circuits, probabilities))


def phase_amplitude_damping_error(
    param_amp, param_phase, excited_state_population=0, canonical_kraus=True
):
    r"""
    Return a single-qubit combined phase and amplitude damping quantum error channel.

    The single-qubit combined phase and amplitude damping channel is
    described by the following Kraus matrices:

    .. code-block:: python

        A0 = sqrt(1 - p1) * [[1, 0], [0, sqrt(1 - a - b)]]
        A1 = sqrt(1 - p1) * [[0, sqrt(a)], [0, 0]]
        A2 = sqrt(1 - p1) * [[0, 0], [0, sqrt(b)]]
        B0 = sqrt(p1) * [[sqrt(1 - a - b), 0], [0, 1]]
        B1 = sqrt(p1) * [[0, 0], [sqrt(a), 0]]
        B2 = sqrt(p1) * [[sqrt(b), 0], [0, 0]]

    where ``a = param_amp``, ``b = param_phase``, and
    ``p1 = excited_state_population``. The equilibrium state after infinitely
    many applications of the channel is:

    .. code-block:: python

        rho_eq = [[1 - p1, 0]], [0, p1]]

    Args:
        param_amp (double): the amplitude damping error parameter.
        param_phase (double): the phase damping error parameter.
        excited_state_population (double): the population of :math:`|1\rangle`
                                           state at equilibrium (default: 0).
        canonical_kraus (bool): Convert input Kraus matrices into the
                                canonical Kraus representation (default: True)

    Returns:
        QuantumError: a quantum error object for a noise model.

    Raises:
        NoiseError: If noise parameters are invalid.
    """

    if param_amp < 0:
        raise NoiseError("Invalid amplitude damping to |0> parameter " "({} < 0)".format(param_amp))
    if param_phase < 0:
        raise NoiseError("Invalid phase damping parameter " "({} < 0)".format(param_phase))
    if param_phase + param_amp > 1:
        raise NoiseError(
            "Invalid amplitude and phase damping parameters "
            "({} + {} > 1)".format(param_phase, param_amp)
        )
    if excited_state_population < 0:
        raise NoiseError(
            "Invalid excited state population " "({} < 0).".format(excited_state_population)
        )
    if excited_state_population > 1:
        raise NoiseError(
            "Invalid excited state population " "({} > 1).".format(excited_state_population)
        )
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
    noise_ops = [a for a in [A0, A1, A2, B0, B1, B2] if np.linalg.norm(a) > 1e-10]
    return kraus_error(noise_ops, canonical_kraus=canonical_kraus)


def amplitude_damping_error(param_amp, excited_state_population=0, canonical_kraus=True):
    r"""
    Return a single-qubit generalized amplitude damping quantum error channel.

    The single-qubit amplitude damping channel is described by the
    following Kraus matrices:

    .. code-block:: python

        A0 = sqrt(1 - p1) * [[1, 0], [0, sqrt(1 - a)]]
        A1 = sqrt(1 - p1) * [[0, sqrt(a)], [0, 0]]
        B0 = sqrt(p1) * [[sqrt(1 - a), 0], [0, 1]]
        B1 = sqrt(p1) * [[0, 0], [sqrt(a), 0]]

    where ``a = param_amp``, ``p1 = excited_state_population``.
    The equilibrium state after infinitely many applications of the
    channel is:

    .. code-block:: python

        rho_eq = [[1 - p1, 0]], [0, p1]]

    Args:
        param_amp (double): the amplitude damping parameter.
        excited_state_population (double): the population of :math:`|0\rangle`
                                           state at equilibrium (default: 0).
        canonical_kraus (bool): Convert input Kraus matrices into the
                                canonical Kraus representation (default: True)

    Returns:
        QuantumError: a quantum error object for a noise model.
    """
    return phase_amplitude_damping_error(
        param_amp,
        0,
        excited_state_population=excited_state_population,
        canonical_kraus=canonical_kraus,
    )


def phase_damping_error(param_phase, canonical_kraus=True):
    r"""
    Return a single-qubit generalized phase damping quantum error channel.

    The single-qubit phase damping channel is described by the
    following Kraus matrices:

    .. code-block:: python

        A0 = [[1, 0], [0, sqrt(1 - b)]]
        A2 = [[0, 0], [0, sqrt(b)]]

    where ``b = param_phase``.
    The equilibrium state after infinitely many applications of the
    channel is:

    .. code-block:: python

        rho_eq = [[rho_init[0, 0], 0]], [0, rho_init[1, 1]]]

    where ``rho_init`` is the input state ρ.

    Args:
        param_phase (double): the phase damping parameter.
        canonical_kraus (bool): Convert input Kraus matrices into the
                                canonical Kraus representation (default: True)

    Returns:
        QuantumError: a quantum error object for a noise model.
    """

    return phase_amplitude_damping_error(
        0, param_phase, excited_state_population=0, canonical_kraus=canonical_kraus
    )
