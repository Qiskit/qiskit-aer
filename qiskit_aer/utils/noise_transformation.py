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
import logging
from typing import Sequence, List, Union, Callable

import numpy as np

from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import (
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
)
from qiskit.compiler import transpile
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.quantum_info.operators.channel import Kraus, SuperOp, Chi
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.transpiler.exceptions import TranspilerError
from ..noise.errors import QuantumError
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


def approximate_quantum_error(
    error, *, operator_string=None, operator_dict=None, operator_list=None
):
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
        raise NoiseError(f"Invalid input error type: {error.__class__.__name__}")

    if error.num_qubits > 2:
        raise NoiseError(
            "Only 1-qubit and 2-qubit noises can be converted, "
            f"{error.num_qubits}-qubit noise found in model"
        )

    if operator_string is not None:
        valid_operator_strings = _PRESET_OPERATOR_TABLE.keys()
        operator_string = operator_string.lower()
        if operator_string not in valid_operator_strings:
            raise NoiseError(
                f"{operator_string} is not a valid operator_string. "
                f"It must be one of {valid_operator_strings}"
            )
        try:
            operator_list = _PRESET_OPERATOR_TABLE[operator_string][error.num_qubits]
        except KeyError as err:
            raise NoiseError(
                f"Preset '{operator_string}' operators do not support the "
                f"approximation of errors with {error.num_qubits} qubits"
            ) from err
    if operator_dict is not None:
        _, operator_list = zip(*operator_dict.items())
    if operator_list is not None:
        if not isinstance(operator_list, Sequence):
            raise NoiseError(f"operator_list is not a sequence: {operator_list}")
        try:
            channel_list = [
                op if isinstance(op, QuantumChannel) else QuantumError([(op, 1)])
                for op in operator_list
            ]  # preserve operator_list
        except NoiseError as err:
            raise NoiseError(f"Invalid type found in operator list: {operator_list}") from err

        probabilities = _transform_by_operator_list(channel_list, error)[1:]
        identity_prob = np.round(1 - sum(probabilities), 9)
        if identity_prob < 0 or identity_prob > 1:
            raise NoiseError(f"Channel probabilities sum to {1 - identity_prob}")
        noise_ops = [((IGate(), [0]), identity_prob)]
        for operator, probability in zip(operator_list, probabilities):
            noise_ops.append((operator, probability))
        return QuantumError(noise_ops)

    raise NoiseError("Quantum error approximation failed - no approximating operators detected")


def approximate_noise_model(model, *, operator_string=None, operator_dict=None, operator_list=None):
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
            operator_list=operator_list,
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
# clifford operators
_CLIFFORD_GATES = [
    (IGate(),),
    (SGate(),),
    (SdgGate(),),
    (ZGate(),),
    # u2 gates
    (HGate(),),
    (HGate(), ZGate()),
    (ZGate(), HGate()),
    (HGate(), SGate()),
    (SGate(), HGate()),
    (HGate(), SdgGate()),
    (SdgGate(), HGate()),
    (SGate(), HGate(), SGate()),
    (SdgGate(), HGate(), SGate()),
    (ZGate(), HGate(), SGate()),
    (SGate(), HGate(), SdgGate()),
    (SdgGate(), HGate(), SdgGate()),
    (ZGate(), HGate(), SdgGate()),
    (SGate(), HGate(), ZGate()),
    (SdgGate(), HGate(), ZGate()),
    (ZGate(), HGate(), ZGate()),
    # u3 gates
    (XGate(),),
    (YGate(),),
    (SGate(), XGate()),
    (SdgGate(), XGate()),
]
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
    },
}


def _transform_by_operator_list(
    basis_ops: Sequence[Union[QuantumChannel, QuantumError]],
    target: Union[QuantumChannel, QuantumError],
) -> List[float]:
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
            " approximating a noise channel.",
        ) from err
    # create quadratic program
    x = cvxpy.Variable(n)
    prob = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.quad_form(x, A) + b.T @ x),
        constraints=[cvxpy.sum(x) <= 1, x >= 0, source_fids.T @ x <= target_fid],
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
