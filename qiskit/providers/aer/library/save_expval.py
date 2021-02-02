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
from qiskit.extensions.exceptions import ExtensionError
from .save_data import SaveAverageData


class SaveExpval(SaveAverageData):
    """Save expectation value of an operator."""
    def __init__(self,
                 key,
                 operator,
                 unnormalized=False,
                 conditional=False,
                 pershot=False):
        r"""Instruction to save the expectation value of a Hermitian operator.

        The expectation value of a Hermitian operator :math:`H` for a simulator
        in quantum state :math`\rho`is given by
        :math:`\langle H\rangle = \mbox{Tr}[H.\rho]`.

        Args:
            key (str): the key for retrieving saved data from results.
            operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
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
            ExtensionError: if the input operator is invalid or not Hermitian.
        """
        # Convert O to SparsePauliOp representation
        if isinstance(operator, Pauli):
            operator = SparsePauliOp(operator)
        elif not isinstance(operator, SparsePauliOp):
            operator = SparsePauliOp.from_operator(Operator(operator))
        if not isinstance(operator, SparsePauliOp):
            raise ExtensionError("Invalid input operator")
        if not allclose(operator.coeffs.imag, 0):
            raise ExtensionError("Input operator is not Hermitian.")
        params = _expval_params(operator, variance=False)
        super().__init__('save_expval',
                         key,
                         operator.num_qubits,
                         conditional=conditional,
                         pershot=pershot,
                         unnormalized=unnormalized,
                         params=params)


class SaveExpvalVar(SaveAverageData):
    """Save expectation value of an operator."""
    def __init__(self,
                 key,
                 operator,
                 unnormalized=False,
                 conditional=False,
                 pershot=False):
        r"""Instruction to save the expectation value and variance of a Hermitian operator.

        The expectation value of a Hermitian operator :math:`H` for a
        simulator in quantum state :math`\rho`is given by
        :math:`\langle H\rangle = \mbox{Tr}[H.\rho]`. The variance is given by
        :math:`\sigma^2 = \langle H^2 \rangle - \langle H \rangle>^2`.

        Args:
            key (str): the key for retrieving saved data from results.
            operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
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
            ExtensionError: if the input operator is invalid or not Hermitian.
        """
        # Convert O to SparsePauliOp representation
        if isinstance(operator, Pauli):
            operator = SparsePauliOp(operator)
        elif not isinstance(operator, SparsePauliOp):
            operator = SparsePauliOp.from_operator(Operator(operator))
        if not isinstance(operator, SparsePauliOp):
            raise ExtensionError("Invalid input operator")
        if not allclose(operator.coeffs.imag, 0):
            raise ExtensionError("Input operator is not Hermitian.")
        params = _expval_params(operator, variance=True)
        super().__init__('save_expval_var',
                         key,
                         operator.num_qubits,
                         conditional=conditional,
                         pershot=pershot,
                         unnormalized=unnormalized,
                         params=params)


def _expval_params(operator, variance=False):

    # Convert O to SparsePauliOp representation
    if isinstance(operator, Pauli):
        operator = SparsePauliOp(operator)
    elif not isinstance(operator, SparsePauliOp):
        operator = SparsePauliOp.from_operator(Operator(operator))
    if not isinstance(operator, SparsePauliOp):
        raise ExtensionError("Invalid input operator")

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


def save_expval(self,
                key,
                operator,
                qubits,
                unnormalized=False,
                conditional=False,
                pershot=False):
    r"""Save the expectation value of a Hermitian operator.

    Args:
        key (str): the key for retrieving saved data from results.
        operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
        qubits (list): circuit qubits to apply instruction.
        unnormalized (bool): If True return save the unnormalized accumulated
                             or conditional accumulated expectation value
                             over all shot [Default: False].
        pershot (bool): if True save a list of expectation values for each shot
                        of the simulation rather than the average over
                        all shots [Default: False].
        conditional (bool): if True save the average or pershot data
                            conditional on the current classical register
                            values [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if the input operator is invalid or not Hermitian.
    """
    instr = SaveExpval(key,
                       operator,
                       unnormalized=unnormalized,
                       conditional=conditional,
                       pershot=pershot)
    return self.append(instr, qubits)


def save_expval_var(self,
                    key,
                    operator,
                    qubits,
                    unnormalized=False,
                    conditional=False,
                    pershot=False):
    r"""Save the expectation value of a Hermitian operator.

    Args:
        key (str): the key for retrieving saved data from results.
        operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
        qubits (list): circuit qubits to apply instruction.
        unnormalized (bool): If True return save the unnormalized accumulated
                             or conditional accumulated expectation value
                             and variance over all shot [Default: False].
        pershot (bool): if True save a list of expectation values and variances
                        for each shot of the simulation rather than the
                        average over all shots [Default: False].
        conditional (bool): if True save the data conditional on the current
                            classical register values [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if the input operator is invalid or not Hermitian.
    """
    instr = SaveExpvalVar(key,
                          operator,
                          unnormalized=unnormalized,
                          conditional=conditional,
                          pershot=pershot)
    return self.append(instr, qubits)


QuantumCircuit.save_expval = save_expval
QuantumCircuit.save_expval_var = save_expval_var
