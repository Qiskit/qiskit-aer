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
Noise transformation module

The goal of this module is to transform one 1-qubit noise channel
(given by the QuantumError class) into another, built from specified
"building blocks" (given as Kraus matrices) such that the new channel is
as close as possible to the original one in the Hilber-Schmidt metric.

For a typical use case, consider a simulator for circuits built from the
Clifford group. Computations on such circuits can be simulated at
polynomial time and space, but not all noise channels can be used in such
a simulation. To enable noisy Clifford simulation one can transform the
given noise channel into the closest one, Hilbert-Schmidt wise, that can be
used in a Clifford simulator.
"""
# pylint: disable=import-outside-toplevel

import itertools
import logging
import warnings
from typing import Sequence

import numpy
import sympy

from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.quantum_info.operators.channel import Kraus, SuperOp
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from ..noise.errors import QuantumError
from ..noise.errors.errorutils import _single_qubit_clifford_gates
from ..noise.errors.errorutils import _single_qubit_clifford_matrix
from ..noise.noise_model import NoiseModel
from ..noise.noiseerror import NoiseError

logger = logging.getLogger(__name__)


def approximate_quantum_error(error, *,
                              operator_string=None,
                              operator_dict=None,
                              operator_list=None):
    """
    Return an approximate QuantumError bases on the Hilbert-Schmidt metric.

    Currently this is only implemented for 1-qubit QuantumErrors.

    Args:
        error (QuantumError or QuantumChannel): the error to be approximated.
            The number of qubits must be 1 or 2.
        operator_string (string): a name for a pre-made set of
            building blocks for the output channel (Default: None).
            Possible values are ``'pauli'``, ``'reset'``, ``'clifford'``.
        operator_dict (dict): a dictionary whose values are the
            building blocks for the output channel (Default: None).
            E.g. {"x": XGate(), "y": YGate()}, keys "x" and "y"
            are not used in transformation.
        operator_list (list): list of building block operators for the
            output channel (Default: None). E.g. [XGate(), YGate()]

    Returns:
        QuantumError: the approximate quantum error.

    Raises:
        NoiseError: if any invalid argument is specified or approximation failed.

    Additional Information:
        The operator input precedence is: ``list`` < ``dict`` < ``str``.
        If a string is given, dict is overwritten; if a dict is given, list is
        overwritten. Possible values for string are ``'pauli'``, ``'reset'``,
        ``'clifford'``. The ``'clifford'`` does not support 2-qubit errors.
    """
    if not isinstance(error, (QuantumError, QuantumChannel)):
        warnings.warn(
            'Support of types other than QuantumError or QuantumChannel for error'
            ' to be approximated has been deprecated as of qiskit-aer 0.10.0'
            ' and will be removed no earlier than 3 months from that release date.',
            DeprecationWarning, stacklevel=2)
        if isinstance(error, tuple) and isinstance(error[0], numpy.ndarray):
            error = list(error)
        if isinstance(error, list) or \
                (isinstance(error, tuple) and isinstance(error[0], list)):
            # first case for ordinary Kraus [A_i, B_i]
            # second case for generalized Kraus ([A_i], [B_i])
            error = Kraus(error)
        else:
            raise NoiseError("Invalid input error type: {}".format(error.__class__.__name__))

    if isinstance(error, QuantumError):
        error = Kraus(error)
    # assert(isinstance(error, Kraus))

    if error.num_qubits > 2:
        raise NoiseError("Only 1-qubit and 2-qubit noises can be converted, {}-qubit "
                         "noise found in model".format(error.num_qubits))

    if operator_string is not None:
        valid_operator_strings = _PRESET_OPERATOR_TABLE.keys()
        operator_string = operator_string.lower()
        if operator_string not in valid_operator_strings:
            raise NoiseError("{} is not a valid operator_string. "
                             "It must be one of {}".format(operator_string, valid_operator_strings))
        operator_list = _PRESET_OPERATOR_TABLE[operator_string][error.num_qubits]
        if not operator_list:
            raise NoiseError("Preset '{}' operators do not support the approximation of"
                             " errors with {} qubits".format(operator_string, error.num_qubits))
    if operator_dict is not None:
        _, operator_list = zip(*operator_dict.items())
    if operator_list is not None:
        if not isinstance(operator_list, Sequence):
            raise NoiseError("operator_list is not a sequence: {}".format(operator_list))
        if operator_list and isinstance(operator_list[0], Sequence) and isinstance(
                operator_list[0][0], numpy.ndarray):
            warnings.warn(
                'Accepting a sequence of Kraus matrices (list of numpy arrays)'
                ' as an operator_list has been deprecated as of qiskit-aer 0.10.0'
                ' and will be removed no earlier than 3 months from that release date.'
                ' Please use a sequence of Kraus objects instead.',
                DeprecationWarning, stacklevel=2)
            operator_list = [Kraus(op) for op in operator_list]

        try:
            operator_list = [op if isinstance(op, QuantumChannel) else QuantumError([(op, 1)])
                             for op in operator_list]
        except NoiseError:
            raise NoiseError("Invalid type found in operator list: {}".format(operator_list))

        probabilities = _transform_by_operator_list(operator_list, error)
        identity_prob = numpy.round(1 - sum(probabilities), 9)
        if identity_prob < 0 or identity_prob > 1:
            raise NoiseError("Channel probabilities sum to {}".format(1 - identity_prob))
        noise_ops = [((IGate(), [0]), identity_prob)]
        for (operator, probability) in zip(operator_list, probabilities):
            noise_ops.append((operator, probability))
        return QuantumError(noise_ops)

    raise NoiseError(
        "Quantum error approximation failed - no approximating operators detected"
    )


def approximate_noise_model(model, *,
                            operator_string=None,
                            operator_dict=None,
                            operator_list=None):
    """
    Return an approximate noise model.

    Args:
        model (NoiseModel): the noise model to be approximated.
            All noises in the model must be 1- or 2-qubit noises.
        operator_string (string): a name for a pre-made set of
            building blocks for the output channel (Default: None).
            Possible values are ``'pauli'``, ``'reset'``, ``'clifford'``.
        operator_dict (dict): a dictionary whose values are the
            building blocks for the output channel (Default: None).
            E.g. {"x": XGate(), "y": YGate()}, keys "x" and "y"
            are not used in transformation.
        operator_list (list): list of building block operators for the
            output channel (Default: None). E.g. [XGate(), YGate()]

    Returns:
        NoiseModel: the approximate noise model.

    Raises:
        NoiseError: if any invalid argument is specified or approximation failed.

    Additional Information:
        The operator input precedence is: ``list`` < ``dict`` < ``str``.
        If a string is given, dict is overwritten; if a dict is given, list is
        overwritten. Possible values for string are ``'pauli'``, ``'reset'``,
        ``'clifford'``. The ``'clifford'`` does not support 2-qubit errors.
    """
    def approximated(noise):
        return approximate_quantum_error(
            noise,
            operator_string=operator_string,
            operator_dict=operator_dict,
            operator_list=operator_list
        )
    # Copy from original noise model
    new_model = NoiseModel()
    new_model._basis_gates = model._basis_gates
    # Transformation
    for inst_name, noise in model._default_quantum_errors.items():
        new_model.add_all_qubit_quantum_error(approximated(noise), inst_name)
    for inst_name, noise_dic in model._local_quantum_errors.items():
        for qubits, noise in noise_dic.items():
            new_model.add_quantum_error(approximated(noise), inst_name, qubits)
    for inst_name, outer_dic in model._nonlocal_quantum_errors.items():
        for qubits, inner_dic in outer_dic.items():
            for noise_qubits, noise in inner_dic.items():
                new_model.add_nonlocal_quantum_error(
                    approximated(noise), inst_name, qubits, noise_qubits
                )
    # No transformation for readout errors
    if model._default_readout_error:
        new_model.add_all_qubit_readout_error(model._default_readout_error)
    for qubits, noise in model._local_readout_errors.items():
        new_model.add_readout_error(noise, qubits)
    return new_model


# pauli operators
_ID_Q0 = [(IGate(), [0])]
_ID_Q1 = [(IGate(), [1])]
_PAULIS_Q0 = [[(IGate(), [0])], [(XGate(), [0])], [(YGate(), [0])], [(ZGate(), [0])]]
_PAULIS_Q1 = [[(IGate(), [1])], [(XGate(), [1])], [(YGate(), [1])], [(ZGate(), [1])]]
_PAULIS_Q0Q1 = [p_q0 + p_q1 for p_q0 in _PAULIS_Q0 for p_q1 in _PAULIS_Q1]
# reset operators
_RESET_Q0_TO_0 = [(Reset(), [0])]
_RESET_Q0_TO_1 = [(Reset(), [0]), (XGate(), [0])]
_RESET_Q1_TO_0 = [(Reset(), [1])]
_RESET_Q1_TO_1 = [(Reset(), [1]), (XGate(), [1])]
# preset operator table
_PRESET_OPERATOR_TABLE = {
    "pauli": {
        1: _PAULIS_Q0[1:],
        2: _PAULIS_Q0Q1[1:],
    },
    "reset": {
        1: [
            _RESET_Q0_TO_0,
            _RESET_Q0_TO_1,
        ],
        2: [
            _RESET_Q0_TO_0 + _ID_Q1,
            _RESET_Q0_TO_1 + _ID_Q1,
            _RESET_Q1_TO_0,
            _RESET_Q1_TO_1,
            _RESET_Q0_TO_0 + _RESET_Q1_TO_0,
            _RESET_Q0_TO_0 + _RESET_Q1_TO_1,
            _RESET_Q0_TO_1 + _RESET_Q1_TO_0,
            _RESET_Q0_TO_1 + _RESET_Q1_TO_1,
        ],
    },
    "clifford": {
        1: [[(gate, [0]) for gate in _single_qubit_clifford_gates(j)] for j in range(1, 24)],
        2: []  # not available
    }
}


def _transform_by_operator_list(basis_ops, kraus):
    r"""
    Transform input Kraus channel.

    Allows approximating an input Kraus channel as in terms of
    a different set of Kraus operators (basis_ops).

    For example, setting :math:`[X, Y, Z]` allows approximating by a
    Pauli channel, and :math:`[(|0 \langle\rangle 0|,
    |0\langle\rangle 1|), |1\langle\rangle 0|, |1 \langle\rangle 1|)]`
    represents the relaxation channel

    In the case the input is a list :math:`[A_1, A_2, ..., A_n]` of
    transform matrices and :math:`[E_0, E_1, ..., E_m]` of noise Kraus
    operators, the output is a list :math:`[p_1, p_2, ..., p_n]` of
    probabilities such that:

    1. :math:`p_i \ge 0`
    2. :math:`p_1 + ... + p_n \le 1`
    3. :math:`[\sqrt(p_1) A_1, \sqrt(p_2) A_2, ..., \sqrt(p_n) A_n,
       \sqrt(1-(p_1 + ... + p_n))I]` is a list of Kraus operators that
       define the output channel (which is "close" to the input channel
       given by :math:`[E_0, ..., E_m]`.)

    This channel can be thought of as choosing the operator :math:`A_i`
    in probability :math:`p_i` and applying this operator to the
    quantum state.

    More generally, if the input is a list of tuples (not necessarily
    of the same size): :math:`[(A_1, B_1, ...), (A_2, B_2, ...),
    ..., (A_n, B_n, ...)]` then the output is still a list
    :math:`[p_1, p_2, ..., p_n]` and now the output channel is defined
    by the operators:
    :math:`[\sqrt(p_1)A1, \sqrt(p_1)B_1, ..., \sqrt(p_n)A_n,
    \sqrt(p_n)B_n, ..., \sqrt(1-(p_1 + ... + p_n))I]`

    Args:
        kraus (Kraus): Kraus channel to be transformed.
        basis_ops (list): a list of QuantumError objects representing Kraus operators
        that can construct the output channel.

    Returns:
        list: A list of amplitudes (probabilities) that define the output channel.
    """
    basis_ops_mats = [Kraus(op).data for op in basis_ops]

    # prepare channel operator list
    # convert to sympy matrices and verify that each singleton is in a tuple; add identity matrix
    full_basis_ops_mats = []
    for ops in basis_ops_mats:
        if not isinstance(ops, tuple) and not isinstance(ops, list):
            ops = [ops]
        full_basis_ops_mats.append([sympy.Matrix(op) for op in ops])
    n = full_basis_ops_mats[0][0].shape[0]  # grab the dimensions from the first element
    full_basis_ops_mats = [[sympy.eye(n)]] + full_basis_ops_mats

    channel_matrices, const_channel_matrix = _generate_channel_matrices(full_basis_ops_mats)

    # prepare data to construct honesty constraint
    def fidelity(channel):
        return sum([numpy.abs(numpy.trace(E)) ** 2 for E in channel])

    coefficients = [fidelity(ops) for ops in full_basis_ops_mats]
    fidelity_data = {
        'goal': fidelity(kraus.data),
        'coefficients': coefficients[1:]  # coefficients[0] corresponds to I
    }

    # pylint: disable=invalid-name
    P, q = _create_obj_func_coef(kraus, channel_matrices, const_channel_matrix)
    probabilities = _solve_quadratic_program(P, q, fidelity_data)
    return probabilities


# methods relevant to the transformation to quadratic programming instance
def _generate_channel_matrices(full_basis_ops_mats):
    r"""
    Generate symbolic channel matrices.

    Generates a list of 4x4 symbolic matrices describing the channel
    defined from the given operators. The identity matrix is assumed
    to be the first element in the list:

    .. code-block:: python

        [(I, ), (A1, B1, ...), (A2, B2, ...), ..., (An, Bn, ...)]

    E.g. for a Pauli channel, the matrices are:

    .. code-block:: python

        [(I,), (X,), (Y,), (Z,)]

    For relaxation they are:

    .. code-block:: python

        [(I, ), (|0><0|, |0><1|), |1><0|, |1><1|)]

    We consider this input to symbolically represent a channel in the
    following manner: define indeterminates :math:`x_0, x_1, ..., x_n`
    which are meant to represent probabilities such that
    :math:`x_i \ge 0` and :math:`x0 = 1-(x_1 + ... + x_n)`.

    Now consider the quantum channel defined via the Kraus operators
    :math:`{\sqrt(x_0)I, \sqrt(x_1) A_1, \sqrt(x1) B_1, ...,
    \sqrt(x_m)A_n, \sqrt(x_n) B_n, ...}`
    This is the channel C symbolically represented by the operators.

    Args:
        full_basis_ops_mats (list): A list of tuples of
            matrices which represent Kraus operators.

    Returns:
        list: A list of 4x4 complex matrices ``([D1, D2, ..., Dn], E)``
        such that the matrix :math:`x_1 D_1 + ... + x_n D_n + E`
        represents the operation of the channel C on the density
        operator. we find it easier to work with this representation
        of C when performing the combinatorial optimization.
    """

    from sympy import symbols as sp_symbols, sqrt
    symbols_string = " ".join([
        "x{}".format(i)
        for i in range(len(full_basis_ops_mats))
    ])
    symbols = sp_symbols(symbols_string, real=True, positive=True)
    exp = symbols[
        1]  # exp will contain the symbolic expression "x1 +...+ xn"
    for i in range(2, len(symbols)):
        exp = symbols[i] + exp
    # symbolic_operators_list is a list of lists; we flatten it the next line
    symbolic_operators_list = [[
        sqrt(symbols[i]) * op for op in ops
    ] for (i, ops) in enumerate(full_basis_ops_mats)]
    symbolic_operators = [
        op for ops in symbolic_operators_list for op in ops
    ]
    # channel_matrix_representation() performs the required linear
    # algebra to find the representing matrices.
    channel = _channel_matrix_representation(
        symbolic_operators).subs(symbols[0], 1 - exp)
    symbols = symbols[1:]

    # pylint: disable=invalid-name
    def get_matrix_from_channel(channel, symbol):
        """
        Extract the numeric parameter matrix.

        Args:
            channel (matrix): a 4x4 symbolic matrix.
            symbol (list): a symbol xi

        Returns:
            matrix: a 4x4 numeric matrix.

        Additional Information:
            Each entry of the 4x4 symbolic input channel matrix is assumed to
            be a polynomial of the form a1x1 + ... + anxn + c. The corresponding
            entry in the output numeric matrix is ai.
        """
        n = channel.rows
        M = numpy.zeros((n, n), dtype=numpy.complex_)
        for (i, j) in itertools.product(range(n), range(n)):
            M[i, j] = complex(
                sympy.Poly(channel[i, j], symbol).coeff_monomial(symbol))
        return M

    def get_const_matrix_from_channel(channel, symbols):
        """
        Extract the numeric constant matrix.

        Args:
            channel (matrix): a 4x4 symbolic matrix.
            symbols (list): The full list [x1, ..., xn] of symbols
                used in the matrix.

        Returns:
            matrix: a 4x4 numeric matrix.

        Additional Information:
            Each entry of the 4x4 symbolic input channel matrix is assumed to
            be a polynomial of the form a1x1 + ... + anxn + c. The corresponding
            entry in the output numeric matrix is c.
        """
        n = channel.rows
        M = numpy.zeros((n, n), dtype=numpy.complex_)
        for (i, j) in itertools.product(range(n), range(n)):
            M[i, j] = complex(
                sympy.Poly(channel[i, j], symbols).coeff_monomial(1))
        return M

    Ds = [get_matrix_from_channel(channel, symbol) for symbol in symbols]
    E = get_const_matrix_from_channel(channel, symbols)
    return Ds, E


def _channel_matrix_representation(operators):
    """
    We convert the operators to a matrix by applying the channel to
    the four basis elements of the 2x2 matrix space representing
    density operators; this is standard linear algebra

    Args:
        operators (list): The list of operators to transform into a Matrix

    Returns:
        sympy.Matrix: The matrx representation of the operators
    """
    shape = operators[0].shape
    standard_base = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            basis_element_ij = sympy.zeros(*shape)
            basis_element_ij[(i, j)] = 1
            standard_base.append(basis_element_ij)

    def compute_channel_operation(rho, operators):
        """
        Given a quantum state's density function rho, the effect of the
        channel on this state is:
        rho -> sum_{i=1}^n E_i * rho * E_i^dagger

        Args:
            rho (number): Density function
            operators (list): List of operators

        Returns:
            number: The result of applying the list of operators
        """
        return sum([E * rho * E.H for E in operators],
                   sympy.zeros(operators[0].rows))

    return (sympy.Matrix([
        list(compute_channel_operation(rho, operators))  # flatten
        for rho in standard_base
    ]))


def _create_obj_func_coef(kraus,
                          channel_matrices,
                          const_channel_matrix):
    """
    Transform by by quantum channels.

    This method creates objective function representing the
    Hilbert-Schmidt norm of the matrix (A-B) obtained
    as the difference of the input noise channel and the output
    channel we wish to determine.

    This function is represented by a matrix P and a vector q, such that
    f(x) = 1/2(x*P*x)+q*x
    where x is the vector we wish to minimize, where x represents
    probabilities for the noise operators that construct the output channel

    Args:
        kraus (Kraus): A kraus to be transformed
        channel_matrices (list): A list of 4x4 symbolic matrices
        const_channel_matrix (matrix): a 4x4 constant matrix

    Returns:
        list: a list of the optimal probabilities for the channel matrices,
        determined by the quadratic program solver
    """
    target_channel = SuperOp(kraus)
    target_channel_matrix = target_channel._data.T

    const_matrix = const_channel_matrix - target_channel_matrix

    # pylint: disable=invalid-name
    def _compute_P(As):
        """
        This method creates the matrix P in the
        f(x) = 1/2(x*P*x)+q*x
        representation of the objective function
        Args:
            As (list): list of symbolic matrices representing the channel matrices

        Returns:
            matrix: The matrix P for the description of the quadatic program
        """
        vs = [numpy.array(A).flatten() for A in As]
        n = len(vs)
        P = sympy.zeros(n, n)
        for (i, j) in itertools.product(range(n), range(n)):
            P[i, j] = 2 * numpy.real(numpy.dot(vs[i], numpy.conj(vs[j])))
        return P

    def _compute_q(As, C):
        """
        This method creates the vector q for the
        f(x) = 1/2(x*P*x)+q*x
        representation of the objective function
        Args:
            As (list): list of symbolic matrices repersenting the quadratic program
            C (matrix): matrix representing the the constant channel matrix

        Returns:
            list: The vector q for the description of the quadaric program
        """
        vs = [numpy.array(A).flatten() for A in As]
        vC = numpy.array(C).flatten()
        n = len(vs)
        q = sympy.zeros(1, n)
        for i in range(n):
            q[i] = 2 * numpy.real(numpy.dot(numpy.conj(vC), vs[i]))
        return q

    return _compute_P(channel_matrices), _compute_q(channel_matrices, const_matrix)


# pylint: disable=invalid-name
def _solve_quadratic_program(P, q, fidelity_data):
    """
    Solve the quadratic program optimization problem.

    This function solved the quadratic program to minimize the objective function
    f(x) = 1/2(x*P*x)+q*x
    subject to the additional constraints
    Gx <= h

    Where P, q are given and G,h are computed to ensure that x represents
    a probability vector and subject to honesty constraints if required
    Args:
        P (matrix): A matrix representing the P component of the objective function
        q (list): A vector representing the q component of the objective function
        fidelity_data (dict): Fidelity data used to define the honesty constraints

    Returns:
        list: The solution of the quadratic program (represents probabilities)

    Additional information:
        This method is the only place in the code where we rely on the cvxpy library
        should we consider another library, only this method needs to change.
    """
    try:
        import cvxpy
    except ImportError:
        logger.error("cvxpy module needs to be installed to use this feature.")

    P = numpy.array(P).astype(float)
    q = numpy.array(q).astype(float).T
    n = len(q)
    # G and h constrain:
    #   1) sum of probs is less then 1
    #   2) All probs bigger than 0
    #   3) Honesty (measured using fidelity, if given)
    G_data = [[1] * n] + [([-1 if i == k else 0 for i in range(n)])
                          for k in range(n)]
    h_data = [1] + [0] * n
    if fidelity_data is not None:
        G_data.append(fidelity_data['coefficients'])
        h_data.append(fidelity_data['goal'])
    G = numpy.array(G_data).astype(float)
    h = numpy.array(h_data).astype(float)
    x = cvxpy.Variable(n)
    prob = cvxpy.Problem(
        cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x, P) + q.T @ x),
        [G @ x <= h])
    prob.solve()
    return x.value


# ================== Deprecated interfaces ================== #
# TODO: remove after deprecation period
def pauli_operators():
    """[Deprecated] Return a list of Pauli operators for 1 and 2 qubits."""
    warnings.warn(
        '"pauli_operators" has been deprecated as of qiskit-aer 0.10.0'
        ' and will be removed no earlier than 3 months from that release date.',
        DeprecationWarning, stacklevel=2)

    pauli_1_qubit = {
        'X': [{'name': 'x', 'qubits': [0]}],
        'Y': [{'name': 'y', 'qubits': [0]}],
        'Z': [{'name': 'z', 'qubits': [0]}]
    }
    pauli_2_qubit = {
        'XI': [{'name': 'x', 'qubits': [1]}, {'name': 'id', 'qubits': [0]}],
        'YI': [{'name': 'y', 'qubits': [1]}, {'name': 'id', 'qubits': [0]}],
        'ZI': [{'name': 'z', 'qubits': [1]}, {'name': 'id', 'qubits': [0]}],
        'IX': [{'name': 'id', 'qubits': [1]}, {'name': 'x', 'qubits': [0]}],
        'IY': [{'name': 'id', 'qubits': [1]}, {'name': 'y', 'qubits': [0]}],
        'IZ': [{'name': 'id', 'qubits': [1]}, {'name': 'z', 'qubits': [0]}],
        'XX': [{'name': 'x', 'qubits': [1]}, {'name': 'x', 'qubits': [0]}],
        'YY': [{'name': 'y', 'qubits': [1]}, {'name': 'y', 'qubits': [0]}],
        'ZZ': [{'name': 'z', 'qubits': [1]}, {'name': 'z', 'qubits': [0]}],
        'XY': [{'name': 'x', 'qubits': [1]}, {'name': 'y', 'qubits': [0]}],
        'XZ': [{'name': 'x', 'qubits': [1]}, {'name': 'z', 'qubits': [0]}],
        'YX': [{'name': 'y', 'qubits': [1]}, {'name': 'x', 'qubits': [0]}],
        'YZ': [{'name': 'y', 'qubits': [1]}, {'name': 'z', 'qubits': [0]}],
        'ZX': [{'name': 'z', 'qubits': [1]}, {'name': 'x', 'qubits': [0]}],
        'ZY': [{'name': 'z', 'qubits': [1]}, {'name': 'y', 'qubits': [0]}],
    }
    return [pauli_1_qubit, pauli_2_qubit]


def reset_operators():
    """[Deprecated] Return a list of reset operators for 1 and 2 qubits."""
    warnings.warn(
        '"reset_operators" has been deprecated as of qiskit-aer 0.10.0'
        ' and will be removed no earlier than 3 months from that release date.',
        DeprecationWarning, stacklevel=2)

    reset_0_to_0 = [{'name': 'reset', 'qubits': [0]}]
    reset_0_to_1 = [{'name': 'reset', 'qubits': [0]}, {'name': 'x', 'qubits': [0]}]
    reset_1_to_0 = [{'name': 'reset', 'qubits': [1]}]
    reset_1_to_1 = [{'name': 'reset', 'qubits': [1]}, {'name': 'x', 'qubits': [1]}]
    id_0 = [{'name': 'id', 'qubits': [0]}]
    id_1 = [{'name': 'id', 'qubits': [1]}]

    reset_1_qubit = {
        'p': reset_0_to_0,
        'q': reset_0_to_1
    }

    reset_2_qubit = {
        'p0': reset_0_to_0 + id_1,
        'q0': reset_0_to_1 + id_1,
        'p1': reset_1_to_0 + id_0,
        'q1': reset_1_to_1 + id_0,
        'p0_p1': reset_0_to_0 + reset_1_to_0,
        'p0_q1': reset_0_to_0 + reset_1_to_1,
        'q0_p1': reset_0_to_1 + reset_1_to_0,
        'q0_q1': reset_0_to_1 + reset_1_to_1,
    }
    return [reset_1_qubit, reset_2_qubit]


class NoiseTransformer:
    """[Deprecated] Transforms one quantum channel to another based on a specified criteria."""

    def __init__(self):
        warnings.warn(
            '"NoiseTransformer" class has been deprecated as of qiskit-aer 0.10.0'
            ' and will be removed no earlier than 3 months from that release date.',
            DeprecationWarning, stacklevel=2)
        self.named_operators = {
            'pauli': pauli_operators(),
            'reset': reset_operators(),
            'clifford': [{j: _single_qubit_clifford_matrix(j) for j in range(1, 24)}]
        }
        self.fidelity_data = None
        self.use_honesty_constraint = True
        self.noise_kraus_operators = None
        self.transform_channel_operators = None

    def operator_matrix(self, operator):
        """Converts an operator representation to Kraus matrix representation

        Args:
            operator (operator): operator representation. Can be a noise
                circuit or a matrix or a list of matrices.

        Returns:
            Kraus: the operator, converted to Kraus representation.
        """
        if isinstance(operator, list) and isinstance(operator[0], dict):
            operator_error = QuantumError([(operator, 1)])
            kraus_rep = Kraus(operator_error.to_quantumchannel()).data
            return kraus_rep
        return operator

    def operator_circuit(self, operator):
        """Converts an operator representation to noise circuit.

        Args:
            operator (operator): operator representation. Can be a noise
                circuit or a matrix or a list of matrices.
        Returns:
            List: The operator, converted to noise circuit representation.
        """
        if isinstance(operator, numpy.ndarray):
            return [{'name': 'unitary', 'qubits': [0], 'params': [operator]}]

        if isinstance(operator, list) and isinstance(operator[0],
                                                     numpy.ndarray):
            if len(operator) == 1:
                return [{'name': 'unitary', 'qubits': [0], 'params': operator}]
            else:
                return [{'name': 'kraus', 'qubits': [0], 'params': operator}]

        return operator

    # transformation interface methods
    def transform_by_operator_list(self, transform_channel_operators,
                                   noise_kraus_operators):
        r"""
        Transform input Kraus operators.

        Allows approximating a set of input Kraus operators as in terms of
        a different set of Kraus matrices.

        For example, setting :math:`[X, Y, Z]` allows approximating by a
        Pauli channel, and :math:`[(|0 \langle\rangle 0|,
        |0\langle\rangle 1|), |1\langle\rangle 0|, |1 \langle\rangle 1|)]`
        represents the relaxation channel

        In the case the input is a list :math:`[A_1, A_2, ..., A_n]` of
        transform matrices and :math:`[E_0, E_1, ..., E_m]` of noise Kraus
        operators, the output is a list :math:`[p_1, p_2, ..., p_n]` of
        probabilities such that:

        1. :math:`p_i \ge 0`
        2. :math:`p_1 + ... + p_n \le 1`
        3. :math:`[\sqrt(p_1) A_1, \sqrt(p_2) A_2, ..., \sqrt(p_n) A_n,
           \sqrt(1-(p_1 + ... + p_n))I]` is a list of Kraus operators that
           define the output channel (which is "close" to the input channel
           given by :math:`[E_0, ..., E_m]`.)

        This channel can be thought of as choosing the operator :math:`A_i`
        in probability :math:`p_i` and applying this operator to the
        quantum state.

        More generally, if the input is a list of tuples (not necessarily
        of the same size): :math:`[(A_1, B_1, ...), (A_2, B_2, ...),
        ..., (A_n, B_n, ...)]` then the output is still a list
        :math:`[p_1, p_2, ..., p_n]` and now the output channel is defined
        by the operators:
        :math:`[\sqrt(p_1)A1, \sqrt(p_1)B_1, ..., \sqrt(p_n)A_n,
        \sqrt(p_n)B_n, ..., \sqrt(1-(p_1 + ... + p_n))I]`

        Args:
            noise_kraus_operators (List): a list of matrices (Kraus operators)
                for the input channel.
            transform_channel_operators (List): a list of matrices or tuples
                of matrices representing Kraus operators that can construct the output channel.

        Returns:
            List: A list of amplitudes that define the output channel.
        """
        self.noise_kraus_operators = noise_kraus_operators
        # pylint: disable=invalid-name
        self.transform_channel_operators = transform_channel_operators
        full_transform_channel_operators = self.prepare_channel_operator_list(
            self.transform_channel_operators)
        channel_matrices, const_channel_matrix = self.generate_channel_matrices(
            full_transform_channel_operators)
        self.prepare_honesty_constraint(full_transform_channel_operators)
        probabilities = self.transform_by_given_channel(
            channel_matrices, const_channel_matrix)
        return probabilities

    @staticmethod
    def prepare_channel_operator_list(ops_list):
        """
        Prepares a list of channel operators.

        Args:
            ops_list (List): The list of operators to prepare

        Returns:
            List: The channel operator list
        """
        # convert to sympy matrices and verify that each singleton is
        # in a tuple; also add identity matrix
        result = []
        for ops in ops_list:
            if not isinstance(ops, tuple) and not isinstance(ops, list):
                ops = [ops]
            result.append([sympy.Matrix(op) for op in ops])
        n = result[0][0].shape[0]  # grab the dimensions from the first element
        result = [[sympy.eye(n)]] + result
        return result

    # pylint: disable=invalid-name
    def prepare_honesty_constraint(self, transform_channel_operators_list):
        """
        Prepares the honesty constraint.

        Args:
            transform_channel_operators_list (list): A list of tuples of matrices which represent
            Kraus operators.
         """
        if not self.use_honesty_constraint:
            return

        goal = self.fidelity(self.noise_kraus_operators)
        coefficients = [
            self.fidelity(ops) for ops in transform_channel_operators_list
        ]
        self.fidelity_data = {
            'goal': goal,
            'coefficients':
                coefficients[1:]  # coefficients[0] corresponds to I
        }

    # methods relevant to the transformation to quadratic programming instance

    @staticmethod
    def fidelity(channel):
        """ Calculates channel fidelity """
        return sum([numpy.abs(numpy.trace(E)) ** 2 for E in channel])

    # pylint: disable=invalid-name
    def generate_channel_matrices(self, transform_channel_operators_list):
        r"""
        Generate symbolic channel matrices.

        Generates a list of 4x4 symbolic matrices describing the channel
        defined from the given operators. The identity matrix is assumed
        to be the first element in the list:

        .. code-block:: python

            [(I, ), (A1, B1, ...), (A2, B2, ...), ..., (An, Bn, ...)]

        E.g. for a Pauli channel, the matrices are:

        .. code-block:: python

            [(I,), (X,), (Y,), (Z,)]

        For relaxation they are:

        .. code-block:: python

            [(I, ), (|0><0|, |0><1|), |1><0|, |1><1|)]

        We consider this input to symbolically represent a channel in the
        following manner: define indeterminates :math:`x_0, x_1, ..., x_n`
        which are meant to represent probabilities such that
        :math:`x_i \ge 0` and :math:`x0 = 1-(x_1 + ... + x_n)`.

        Now consider the quantum channel defined via the Kraus operators
        :math:`{\sqrt(x_0)I, \sqrt(x_1) A_1, \sqrt(x1) B_1, ...,
        \sqrt(x_m)A_n, \sqrt(x_n) B_n, ...}`
        This is the channel C symbolically represented by the operators.

        Args:
            transform_channel_operators_list (list): A list of tuples of
                matrices which represent Kraus operators.

        Returns:
            list: A list of 4x4 complex matrices ``([D1, D2, ..., Dn], E)``
            such that the matrix :math:`x_1 D_1 + ... + x_n D_n + E`
            represents the operation of the channel C on the density
            operator. we find it easier to work with this representation
            of C when performing the combinatorial optimization.
        """

        from sympy import symbols as sp_symbols, sqrt
        symbols_string = " ".join([
            "x{}".format(i)
            for i in range(len(transform_channel_operators_list))
        ])
        symbols = sp_symbols(symbols_string, real=True, positive=True)
        exp = symbols[
            1]  # exp will contain the symbolic expression "x1 +...+ xn"
        for i in range(2, len(symbols)):
            exp = symbols[i] + exp
        # symbolic_operators_list is a list of lists; we flatten it the next line
        symbolic_operators_list = [[
            sqrt(symbols[i]) * op for op in ops
        ] for (i, ops) in enumerate(transform_channel_operators_list)]
        symbolic_operators = [
            op for ops in symbolic_operators_list for op in ops
        ]
        # channel_matrix_representation() performs the required linear
        # algebra to find the representing matrices.
        operators_channel = self.channel_matrix_representation(
            symbolic_operators).subs(symbols[0], 1 - exp)
        return self.generate_channel_quadratic_programming_matrices(
            operators_channel, symbols[1:])

    @staticmethod
    def compute_channel_operation(rho, operators):
        """
        Given a quantum state's density function rho, the effect of the
        channel on this state is:
        rho -> sum_{i=1}^n E_i * rho * E_i^dagger

        Args:
            rho (number): Density function
            operators (list): List of operators

        Returns:
            number: The result of applying the list of operators
        """
        from sympy import zeros
        return sum([E * rho * E.H for E in operators],
                   zeros(operators[0].rows))

    @staticmethod
    def flatten_matrix(m):
        """
        Args:
            m (Matrix): The matrix to flatten

        Returns:
            list: A row vector repesenting the flattened matrix
        """
        return list(m)

    def channel_matrix_representation(self, operators):
        """
        We convert the operators to a matrix by applying the channel to
        the four basis elements of the 2x2 matrix space representing
        density operators; this is standard linear algebra

        Args:
            operators (list): The list of operators to transform into a Matrix

        Returns:
            sympy.Matrix: The matrx representation of the operators
        """
        from sympy import Matrix, zeros
        shape = operators[0].shape
        standard_base = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                basis_element_ij = zeros(*shape)
                basis_element_ij[(i, j)] = 1
                standard_base.append(basis_element_ij)

        return (Matrix([
            self.flatten_matrix(
                self.compute_channel_operation(rho, operators))
            for rho in standard_base
        ]))

    def generate_channel_quadratic_programming_matrices(
            self, channel, symbols):
        """
        Generate matrices for quadratic program.

        Args:
             channel (Matrix): a 4x4 symbolic matrix
             symbols (list): the symbols x1, ..., xn which may occur in the matrix

        Returns:
            list: A list of 4x4 complex matrices ([D1, D2, ..., Dn], E) such that:
            channel == x1*D1 + ... + xn*Dn + E
        """
        return (
            [self.get_matrix_from_channel(channel, symbol) for symbol in symbols],
            self.get_const_matrix_from_channel(channel, symbols)
        )

    @staticmethod
    def get_matrix_from_channel(channel, symbol):
        """
        Extract the numeric parameter matrix.

        Args:
            channel (matrix): a 4x4 symbolic matrix.
            symbol (list): a symbol xi

        Returns:
            matrix: a 4x4 numeric matrix.

        Additional Information:
            Each entry of the 4x4 symbolic input channel matrix is assumed to
            be a polynomial of the form a1x1 + ... + anxn + c. The corresponding
            entry in the output numeric matrix is ai.
        """
        from sympy import Poly
        n = channel.rows
        M = numpy.zeros((n, n), dtype=numpy.complex_)
        for (i, j) in itertools.product(range(n), range(n)):
            M[i, j] = complex(
                Poly(channel[i, j], symbol).coeff_monomial(symbol))
        return M

    @staticmethod
    def get_const_matrix_from_channel(channel, symbols):
        """
        Extract the numeric constant matrix.

        Args:
            channel (matrix): a 4x4 symbolic matrix.
            symbols (list): The full list [x1, ..., xn] of symbols
                used in the matrix.

        Returns:
            matrix: a 4x4 numeric matrix.

        Additional Information:
            Each entry of the 4x4 symbolic input channel matrix is assumed to
            be a polynomial of the form a1x1 + ... + anxn + c. The corresponding
            entry in the output numeric matrix is c.
        """
        from sympy import Poly
        n = channel.rows
        M = numpy.zeros((n, n), dtype=numpy.complex_)
        for (i, j) in itertools.product(range(n), range(n)):
            M[i, j] = complex(
                Poly(channel[i, j], symbols).coeff_monomial(1))
        return M

    def transform_by_given_channel(self, channel_matrices,
                                   const_channel_matrix):
        """
        Transform by by quantum channels.

        This method creates objective function representing the
        Hilbert-Schmidt norm of the matrix (A-B) obtained
        as the difference of the input noise channel and the output
        channel we wish to determine.

        This function is represented by a matrix P and a vector q, such that
        f(x) = 1/2(x*P*x)+q*x
        where x is the vector we wish to minimize, where x represents
        probabilities for the noise operators that construct the output channel

        Args:
            channel_matrices (list): A list of 4x4 symbolic matrices
            const_channel_matrix (matrix): a 4x4 constant matrix

        Returns:
            list: a list of the optimal probabilities for the channel matrices,
            determined by the quadratic program solver
        """
        target_channel = SuperOp(Kraus(self.noise_kraus_operators))
        target_channel_matrix = target_channel._data.T

        const_matrix = const_channel_matrix - target_channel_matrix
        P = self.compute_P(channel_matrices)
        q = self.compute_q(channel_matrices, const_matrix)
        return self.solve_quadratic_program(P, q)

    def compute_P(self, As):
        """
        This method creates the matrix P in the
        f(x) = 1/2(x*P*x)+q*x
        representation of the objective function
        Args:
            As (list): list of symbolic matrices repersenting the channel matrices

        Returns:
            matrix: The matrix P for the description of the quadaric program
        """
        from sympy import zeros
        vs = [numpy.array(A).flatten() for A in As]
        n = len(vs)
        P = zeros(n, n)
        for (i, j) in itertools.product(range(n), range(n)):
            P[i, j] = 2 * numpy.real(numpy.dot(vs[i], numpy.conj(vs[j])))
        return P

    def compute_q(self, As, C):
        """
        This method creates the vector q for the
        f(x) = 1/2(x*P*x)+q*x
        representation of the objective function
        Args:
            As (list): list of symbolic matrices repersenting the quadratic program
            C (matrix): matrix representing the the constant channel matrix

        Returns:
            list: The vector q for the description of the quadaric program
        """
        from sympy import zeros
        vs = [numpy.array(A).flatten() for A in As]
        vC = numpy.array(C).flatten()
        n = len(vs)
        q = zeros(1, n)
        for i in range(n):
            q[i] = 2 * numpy.real(numpy.dot(numpy.conj(vC), vs[i]))
        return q

    def solve_quadratic_program(self, P, q):
        """
        Solve the quadratic program optimization problem.

        This function solved the quadratic program to minimize the objective function
        f(x) = 1/2(x*P*x)+q*x
        subject to the additional constraints
        Gx <= h

        Where P, q are given and G,h are computed to ensure that x represents
        a probability vector and subject to honesty constraints if required
        Args:
            P (matrix): A matrix representing the P component of the objective function
            q (list): A vector representing the q component of the objective function

        Returns:
            list: The solution of the quadratic program (represents probabilities)

        Additional information:
            This method is the only place in the code where we rely on the cvxpy library
            should we consider another library, only this method needs to change.
        """
        try:
            import cvxpy
        except ImportError:
            logger.error("cvxpy module needs to be installed to use this feature.")

        P = numpy.array(P).astype(float)
        q = numpy.array(q).astype(float).T
        n = len(q)
        # G and h constrain:
        #   1) sum of probs is less then 1
        #   2) All probs bigger than 0
        #   3) Honesty (measured using fidelity, if given)
        G_data = [[1] * n] + [([-1 if i == k else 0 for i in range(n)])
                              for k in range(n)]
        h_data = [1] + [0] * n
        if self.fidelity_data is not None:
            G_data.append(self.fidelity_data['coefficients'])
            h_data.append(self.fidelity_data['goal'])
        G = numpy.array(G_data).astype(float)
        h = numpy.array(h_data).astype(float)
        x = cvxpy.Variable(n)
        prob = cvxpy.Problem(
            cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x, P) + q.T @ x),
            [G @ x <= h])
        prob.solve()
        return x.value
