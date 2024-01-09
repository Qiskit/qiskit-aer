# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Simulator instruction to save exact operator expectation value.
"""

from numpy import allclose
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.circuit import QuantumCircuit
from .save_data import SaveAverageData


class SaveExpectationValue(SaveAverageData):
    """Save expectation value of an operator."""

    def __init__(
        self,
        operator,
        label="expectation_value",
        unnormalized=False,
        pershot=False,
        conditional=False,
    ):
        r"""Instruction to save the expectation value of a Hermitian operator.

        The expectation value of a Hermitian operator :math:`H` for a simulator
        in quantum state :math`\rho`is given by
        :math:`\langle H\rangle = \mbox{Tr}[H.\rho]`.

        Args:
            operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
            label (str): the key for retrieving saved data from results.
            unnormalized (bool): If True return save the unnormalized accumulated
                                 or conditional accumulated expectation value
                                 over all shot [Default: False].
            pershot (bool): if True save a list of expectation values for each shot
                            of the simulation rather than the average over
                            all shots [Default: False].
            conditional (bool): if True save the average or pershot data
                                conditional on the current classical register
                                values [Default: False].

        Raises:
            ValueError: if the input operator is not Hermitian.
            TypeError: if the input operator is of invalid type.

        .. note::

            This instruction can be directly appended to a circuit using the
            :func:`save_expectation_value` circuit method.
        """
        # Convert O to SparsePauliOp representation
        if isinstance(operator, Pauli):
            operator = SparsePauliOp(operator)
        elif not isinstance(operator, SparsePauliOp):
            operator = SparsePauliOp.from_operator(Operator(operator))
        if not allclose(operator.coeffs.imag, 0):
            raise ValueError("Input operator is not Hermitian.")
        params = _expval_params(operator, variance=False)
        super().__init__(
            "save_expval",
            operator.num_qubits,
            label,
            unnormalized=unnormalized,
            pershot=pershot,
            conditional=conditional,
            params=params,
        )


class SaveExpectationValueVariance(SaveAverageData):
    """Save expectation value and variance of an operator."""

    def __init__(
        self,
        operator,
        label="expectation_value_variance",
        unnormalized=False,
        pershot=False,
        conditional=False,
    ):
        r"""Instruction to save the expectation value and variance of a Hermitian operator.

        The expectation value of a Hermitian operator :math:`H` for a
        simulator in quantum state :math`\rho`is given by
        :math:`\langle H\rangle = \mbox{Tr}[H.\rho]`. The variance is given by
        :math:`\sigma^2 = \langle H^2 \rangle - \langle H \rangle>^2`.

        Args:
            operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
            label (str): the key for retrieving saved data from results.
            unnormalized (bool): If True return save the unnormalized accumulated
                                 or conditional accumulated expectation value
                                 over all shot [Default: False].
            pershot (bool): if True save a list of expectation values for each shot
                            of the simulation rather than the average over
                            all shots [Default: False].
            conditional (bool): if True save the average or pershot data
                                conditional on the current classical register
                                values [Default: False].

        Raises:
            ValueError: if the input operator is not Hermitian.
            TypeError: if the input operator is of invalid type.

        .. note::

            This instruction can be directly appended to a circuit using
            the :func:`save_expectation_value` circuit method.
        """
        # Convert O to SparsePauliOp representation
        if isinstance(operator, Pauli):
            operator = SparsePauliOp(operator)
        elif not isinstance(operator, SparsePauliOp):
            operator = SparsePauliOp.from_operator(Operator(operator))
        if not allclose(operator.coeffs.imag, 0):
            raise ValueError("Input operator is not Hermitian.")
        params = _expval_params(operator, variance=True)
        super().__init__(
            "save_expval_var",
            operator.num_qubits,
            label,
            unnormalized=unnormalized,
            pershot=pershot,
            conditional=conditional,
            params=params,
        )


def _expval_params(operator, variance=False):
    # Convert O to SparsePauliOp representation
    if isinstance(operator, Pauli):
        operator = SparsePauliOp(operator)
    elif not isinstance(operator, SparsePauliOp):
        operator = SparsePauliOp.from_operator(Operator(operator))
    if not isinstance(operator, SparsePauliOp):
        raise TypeError("Invalid input operator")

    params = {}

    # Add Pauli basis components of O
    for pauli, coeff in operator.label_iter():
        if pauli in params:
            coeff1 = params[pauli][0]
            params[pauli] = (coeff1 + coeff.real, 0)
        else:
            params[pauli] = (coeff.real, 0)

    # Add Pauli basis components of O^2
    if variance:
        for pauli, coeff in operator.dot(operator).label_iter():
            if pauli in params:
                coeff1, coeff2 = params[pauli]
                params[pauli] = (coeff1, coeff2 + coeff.real)
            else:
                params[pauli] = (0, coeff.real)

    # Convert to list
    return list(params.items())


def save_expectation_value(
    self,
    operator,
    qubits,
    label="expectation_value",
    unnormalized=False,
    pershot=False,
    conditional=False,
):
    r"""Save the expectation value of a Hermitian operator.

    Args:
        operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
        qubits (list): circuit qubits to apply instruction.
        label (str): the key for retrieving saved data from results.
        unnormalized (bool): If True return save the unnormalized accumulated
                             or conditional accumulated expectation value
                             over all shot [Default: False].
        pershot (bool): if True save a list of expectation values for each
                        shot of the simulation rather than the average over
                        all shots [Default: False].
        conditional (bool): if True save the average or pershot data
                            conditional on the current classical register
                            values [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ValueError: if the input operator is not Hermitian.
        TypeError: if the input operator is of invalid type.

    .. note::

        This method appends a :class:`SaveExpectationValue` instruction to
        the quantum circuit.
    """
    instr = SaveExpectationValue(
        operator, label=label, unnormalized=unnormalized, pershot=pershot, conditional=conditional
    )
    return self.append(instr, qubits)


def save_expectation_value_variance(
    self,
    operator,
    qubits,
    label="expectation_value_variance",
    unnormalized=False,
    pershot=False,
    conditional=False,
):
    r"""Save the expectation value of a Hermitian operator.

    Args:
        operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
        qubits (list): circuit qubits to apply instruction.
        label (str): the key for retrieving saved data from results.
        unnormalized (bool): If True return save the unnormalized accumulated
                             or conditional accumulated expectation value
                             and variance over all shot [Default: False].
        pershot (bool): if True save a list of expectation values and
                        variances for each shot of the simulation rather than
                        the average over all shots [Default: False].
        conditional (bool): if True save the data conditional on the current
                            classical register values [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ValueError: if the input operator is not Hermitian.
        TypeError: if the input operator is of invalid type.

    .. note::

        This method appends a :class:`SaveExpectationValueVariance`
        instruction to the quantum circuit.
    """
    instr = SaveExpectationValueVariance(
        operator, label=label, unnormalized=unnormalized, pershot=pershot, conditional=conditional
    )
    return self.append(instr, qubits)


QuantumCircuit.save_expectation_value = save_expectation_value
QuantumCircuit.save_expectation_value_variance = save_expectation_value_variance
