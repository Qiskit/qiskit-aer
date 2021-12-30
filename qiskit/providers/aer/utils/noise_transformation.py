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

import copy
import itertools
import logging
import warnings
from typing import Sequence, List, Union, Callable

import numpy as np

from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.compiler import transpile
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.quantum_info.operators.channel import Kraus, SuperOp, Chi
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.transpiler.exceptions import TranspilerError
from ..noise.errors import QuantumError
from ..noise.errors.errorutils import _CLIFFORD_GATES
from ..noise.errors.errorutils import single_qubit_clifford_matrix
from ..noise.noise_model import NoiseModel
from ..noise.noiseerror import NoiseError

logger = logging.getLogger(__name__)


def transform_noise_model(noise_model: NoiseModel, func: Callable) -> NoiseModel:
    """Return a new noise model by applyign a function to all quantum errors.

    This returns a new noise model containing the resulting errors from
    applying the supplied function to all QuantumErrors in the noise model.
    This function should have singature `func(error: QuantumError) -> QuantumError`
    where the returned quantum error is defined on the same number of qubits
    as the original error.

    Args:
        noise_model: the noise model to be transformed.
        func: function for transforming QuantumErrors.
    Returns:
        The transpiled noise model.

    Raises:
        NoiseError: if the transformation failed.
    """
    # Make a deep copy of the noise model so we can update its
    # internal dicts without affecting the original model
    new_noise = copy.deepcopy(noise_model)

    for key, error in new_noise._default_quantum_errors.items():
        new_noise._default_quantum_errors[key] = func(error)

    for inst_name, noise_dic in new_noise._local_quantum_errors.items():
        for qubits, error in noise_dic.items():
            new_noise._local_quantum_errors[inst_name][qubits] = func(error)

    for inst_name, outer_dic in new_noise._nonlocal_quantum_errors.items():
        for qubits, inner_dic in outer_dic.items():
            for noise_qubits, error in inner_dic.items():
                new_noise._nonlocal_quantum_errors[inst_name][qubits][noise_qubits] = func(error)

    return new_noise


def transpile_quantum_error(error: QuantumError, **transpile_kwargs) -> QuantumError:
    """Return a new quantum error containin transpiled circuits.

    This returns a new QuantumError containing the circuits resulting from
    transpiling all error circuits using :func:`qiskit.transpile` with the
    passed keyword agruments.

    Args:
        error: the quantum error to be transformed.
        transpile_kwargs: kwargs for passing to qiskit transpile function.

    Returns:
        The transformed quantum error.

    Raises:
        NoiseError: if the transformation failed.
    """
    try:
        transpiled_circs = transpile(error.circuits, **transpile_kwargs)
    except TranspilerError as err:
        raise NoiseError(
            f"Failed to transpile circuits in {error} with kwargs {transpile_kwargs}"
        ) from err
    return QuantumError(zip(transpiled_circs, error.probabilities))


def transpile_noise_model(noise_model: NoiseModel, **transpile_kwargs) -> NoiseModel:
    """Return a new noise model with transpiled QuantumErrors.

    This returns a new noise model containing the resulting errors from
    transpiling all QuantumErrors in the noise model
    using :func:`transpile_quantum_error` function with the passed
    keyword agruments.

    Args:
        noise_model: the noise model to be transformed.
        transpile_kwargs: kwargs for passing to qiskit transpile function.

    Returns:
        The transpiled noise model.

    Raises:
        NoiseError: if the transformation failed.
    """
    def func(error):
        return transpile_quantum_error(error, **transpile_kwargs)

    return transform_noise_model(noise_model, func)


def approximate_quantum_error(error, *,
                              operator_string=None,
                              operator_dict=None,
                              operator_list=None):
    r"""
    Return a ``QuantumError`` object that approximates an error
    as a mixture of specified operators (channels).

    The approximation is done by minimizing the Hilbert-Schmidt distance
    between the process matrix of the target error channel (:math:`T`) and
    the process matrix of the output channel (:math:`S = \sum_i{p_i S_i}`),
    i.e. :math:`Tr[(T-S)^\dagger (T-S)]`,
    where
    :math:`[p_1, p_2, ..., p_n]` denote probabilities and
    :math:`[S_1, S_2, ..., S_n]` denote basis operators (channels).

    See `arXiv:1207.0046 <http://arxiv.org/abs/1207.0046>`_ for the details.

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
        MissingOptionalLibraryError: if cvxpy is not installed.

    Note:
        The operator input precedence is: ``list`` < ``dict`` < ``string``.
        If a ``string`` is given, ``dict`` is overwritten;
        if a ``dict`` is given, ``list`` is overwritten.
        The ``string`` supports only 1- or 2-qubit errors and
        its possible values are ``'pauli'``, ``'reset'``, ``'clifford'``.
        The ``'clifford'`` does not support 2-qubit errors.
    """
    if not isinstance(error, (QuantumError, QuantumChannel)):
        warnings.warn(
            'Support of types other than QuantumError or QuantumChannel for error'
            ' to be approximated has been deprecated as of qiskit-aer 0.10.0'
            ' and will be removed no earlier than 3 months from that release date.',
            DeprecationWarning, stacklevel=2)
        if isinstance(error, tuple) and isinstance(error[0], np.ndarray):
            error = list(error)
        if isinstance(error, list) or \
                (isinstance(error, tuple) and isinstance(error[0], list)):
            # first case for ordinary Kraus [A_i, B_i]
            # second case for generalized Kraus ([A_i], [B_i])
            error = Kraus(error)
        else:
            raise NoiseError(f"Invalid input error type: {error.__class__.__name__}")

    if error.num_qubits > 2:
        raise NoiseError("Only 1-qubit and 2-qubit noises can be converted, "
                         f"{error.num_qubits}-qubit noise found in model")

    if operator_string is not None:
        valid_operator_strings = _PRESET_OPERATOR_TABLE.keys()
        operator_string = operator_string.lower()
        if operator_string not in valid_operator_strings:
            raise NoiseError(f"{operator_string} is not a valid operator_string. "
                             f"It must be one of {valid_operator_strings}")
        try:
            operator_list = _PRESET_OPERATOR_TABLE[operator_string][error.num_qubits]
        except KeyError:
            raise NoiseError(f"Preset '{operator_string}' operators do not support the "
                             f"approximation of errors with {error.num_qubits} qubits")
    if operator_dict is not None:
        _, operator_list = zip(*operator_dict.items())
    if operator_list is not None:
        if not isinstance(operator_list, Sequence):
            raise NoiseError("operator_list is not a sequence: {}".format(operator_list))
        if operator_list and isinstance(operator_list[0], Sequence) and isinstance(
                operator_list[0][0], np.ndarray):
            warnings.warn(
                'Accepting a sequence of Kraus matrices (list of numpy arrays)'
                ' as an operator_list has been deprecated as of qiskit-aer 0.10.0'
                ' and will be removed no earlier than 3 months from that release date.'
                ' Please use a sequence of Kraus objects instead.',
                DeprecationWarning, stacklevel=2)
            operator_list = [Kraus(op) for op in operator_list]

        try:
            channel_list = [op if isinstance(op, QuantumChannel) else QuantumError([(op, 1)])
                            for op in operator_list]  # preserve operator_list
        except NoiseError:
            raise NoiseError(f"Invalid type found in operator list: {operator_list}")

        probabilities = _transform_by_operator_list(channel_list, error)[1:]
        identity_prob = np.round(1 - sum(probabilities), 9)
        if identity_prob < 0 or identity_prob > 1:
            raise NoiseError(f"Channel probabilities sum to {1 - identity_prob}")
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
    Replace all noises in a noise model with ones approximated
    by a mixture of operators (channels).

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
        MissingOptionalLibraryError: if cvxpy is not installed.

    Note:
        The operator input precedence is: ``list`` < ``dict`` < ``string``.
        If a ``string`` is given, ``dict`` is overwritten;
        if a ``dict`` is given, ``list`` is overwritten.
        The ``string`` supports only 1- or 2-qubit errors and
        its possible values are ``'pauli'``, ``'reset'``, ``'clifford'``.
        The ``'clifford'`` does not support 2-qubit errors.
    """
    def approximate(noise):
        return approximate_quantum_error(
            noise,
            operator_string=operator_string,
            operator_dict=operator_dict,
            operator_list=operator_list
        )
    return transform_noise_model(model, approximate)


# pauli operators
_PAULIS_Q0 = [[(IGate(), [0])], [(XGate(), [0])], [(YGate(), [0])], [(ZGate(), [0])]]
_PAULIS_Q1 = [[(IGate(), [1])], [(XGate(), [1])], [(YGate(), [1])], [(ZGate(), [1])]]
_PAULIS_Q0Q1 = [op_q0 + op_q1 for op_q0 in _PAULIS_Q0 for op_q1 in _PAULIS_Q1]
# reset operators
_RESET_Q0_TO_0 = [(Reset(), [0])]
_RESET_Q0_TO_1 = [(Reset(), [0]), (XGate(), [0])]
_RESET_Q0 = [[(IGate(), [0])], _RESET_Q0_TO_0, _RESET_Q0_TO_1]
_RESET_Q1_TO_0 = [(Reset(), [1])]
_RESET_Q1_TO_1 = [(Reset(), [1]), (XGate(), [1])]
_RESET_Q1 = [[(IGate(), [1])], _RESET_Q1_TO_0, _RESET_Q1_TO_1]
_RESET_Q0Q1 = [op_q0 + op_q1 for op_q0 in _RESET_Q0 for op_q1 in _RESET_Q1]
# preset operator table
_PRESET_OPERATOR_TABLE = {
    "pauli": {
        1: _PAULIS_Q0[1:],
        2: _PAULIS_Q0Q1[1:],
    },
    "reset": {
        1: _RESET_Q0[1:],
        2: _RESET_Q0Q1[1:],
    },
    "clifford": {
        1: [[(gate, [0]) for gate in _CLIFFORD_GATES[j]] for j in range(1, 24)],
    }
}


def _transform_by_operator_list(basis_ops: Sequence[Union[QuantumChannel, QuantumError]],
                                target: Union[QuantumChannel, QuantumError]) -> List[float]:
    r"""
    Transform (or approximate) the target quantum channel into a mixture of
    basis operators (channels) and return the mixing probabilities.

    The approximate channel is found by minimizing the Hilbert-Schmidt
    distance between the process matrix of the target channel (:math:`T`) and
    the process matrix of the output channel (:math:`S = \sum_i{p_i S_i}`),
    i.e. :math:`Tr[(T-S)^\dagger (T-S)]`,
    where
    :math:`[p_1, p_2, ..., p_n]` denote probabilities and
    :math:`[S_1, S_2, ..., S_n]` denote basis operators (channels).

    Such an optimization can represented as a quadratic program:
    Minimize :math:`p^T A p + b^T p`,
    subject to :math:`p \ge 0`, `\sum_i{p_i} = 1`, `\sum_i{p_i} = 1`.
    The last constraint is for making the approximation conservative by
    forcing the output channel have more error than the target channel
    in the sense that a "fidelity" against identity channel should be worse.

    See `arXiv:1207.0046 <http://arxiv.org/abs/1207.0046>`_ for the details.

    Args:
        target: Quantum channel to be transformed.
        basis_ops: a list of channel objects constructing the output channel.

    Returns:
        list: A list of amplitudes (probabilities) of basis that define the output channel.

    Raises:
        MissingOptionalLibraryError: if cvxpy is not installed.
    """
    # pylint: disable=invalid-name
    N = 2 ** basis_ops[0].num_qubits
    # add identity channel in front
    basis_ops = [Kraus(np.eye(N))] + basis_ops

    # create objective function coefficients
    basis_ops_mats = [Chi(op).data for op in basis_ops]
    T = Chi(target).data
    n = len(basis_ops_mats)
    A = np.zeros((n, n))
    for i, S_i in enumerate(basis_ops_mats):
        for j, S_j in enumerate(basis_ops_mats):
            # A[i][j] = 1/2 * {Tr(S_i^† S_j) - Tr(S_j^† S_i)} = Re[Tr(S_i^† S_j)]
            if i < j:
                A[i][j] = _hs_inner_prod_real(S_i, S_j)
            elif i > j:
                A[i][j] = A[j][i]
            else:  # i==j
                A[i][i] = _hs_norm(S_i)
    b = -2 * np.array([_hs_inner_prod_real(T, S_i) for S_i in basis_ops_mats])
    # c = _hs_norm(T)

    # create honesty constraint coefficients
    def fidelity(channel):  # fidelity w.r.t. identity omitting the N^-2 factor
        return float(np.abs(np.trace(SuperOp(channel))))

    source_fids = np.array([fidelity(op) for op in basis_ops])
    target_fid = fidelity(target)

    try:
        import cvxpy
    except ImportError as err:
        logger.error("cvxpy module needs to be installed to use this feature.")
        raise MissingOptionalLibraryError(
            libname="cvxpy",
            name="Transformation/Approximation of noise",
            pip_install="pip install cvxpy",
            msg="CVXPY is required to solve an optimization problem of"
                " approximating a noise channel."
        ) from err
    # create quadratic program
    x = cvxpy.Variable(n)
    prob = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.quad_form(x, A) + b.T @ x),
        constraints=[cvxpy.sum(x) <= 1, x >= 0, source_fids.T @ x <= target_fid]
    )
    # solve quadratic program
    prob.solve()
    probabilities = x.value
    return probabilities


def _hs_norm(A):  # pylint: disable=invalid-name
    # Tr(A^† A)
    return np.trace(np.conj(A).T @ A).real


def _hs_inner_prod_real(A, B):  # pylint: disable=invalid-name
    # Re[Tr(A^† B)]
    return np.trace(np.conj(A.T) @ B).real


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
            'clifford': [{j: single_qubit_clifford_matrix(j) for j in range(1, 24)}]
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
        if isinstance(operator, np.ndarray):
            return [{'name': 'unitary', 'qubits': [0], 'params': [operator]}]

        if isinstance(operator, list) and isinstance(operator[0],
                                                     np.ndarray):
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
        from sympy import Matrix, eye
        # convert to sympy matrices and verify that each singleton is
        # in a tuple; also add identity matrix
        result = []
        for ops in ops_list:
            if not isinstance(ops, tuple) and not isinstance(ops, list):
                ops = [ops]
            result.append([Matrix(op) for op in ops])
        n = result[0][0].shape[0]  # grab the dimensions from the first element
        result = [[eye(n)]] + result
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
        return sum([np.abs(np.trace(E)) ** 2 for E in channel])

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
        M = np.zeros((n, n), dtype=np.complex_)
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
        M = np.zeros((n, n), dtype=np.complex_)
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
        vs = [np.array(A).flatten() for A in As]
        n = len(vs)
        P = zeros(n, n)
        for (i, j) in itertools.product(range(n), range(n)):
            P[i, j] = 2 * np.real(np.dot(vs[i], np.conj(vs[j])))
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
        vs = [np.array(A).flatten() for A in As]
        vC = np.array(C).flatten()
        n = len(vs)
        q = zeros(1, n)
        for i in range(n):
            q[i] = 2 * np.real(np.dot(np.conj(vC), vs[i]))
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

        P = np.array(P).astype(float)
        q = np.array(q).astype(float).T
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
        G = np.array(G_data).astype(float)
        h = np.array(h_data).astype(float)
        x = cvxpy.Variable(n)
        prob = cvxpy.Problem(
            cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x, P) + q.T @ x),
            [G @ x <= h])
        prob.solve()
        return x.value
