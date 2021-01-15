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
                 variance=False,
                 unnormalized=False,
                 conditional=False,
                 pershot=False):
        r"""Create new a instruction to save the expectation value of a Hermitian operator.

        Args:
            key (str): the key for retrieving saved data from results.
            operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
            variance (bool): if True save both the expectation value
                             :math:`\langle O \rangle>`, and the variance
                             :math:`\sigma^2 = \langle O \rangle^2 - \langle O \rangle>^2`
                             [Default: False].
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

        .. note ::

            In cetain cases the list returned by ``pershot=True`` may only
            contain a single value, rather than the number of shots. This
            happens when a run circuit supports measurement sampling because
            it is either

            1. An ideal simulation with all measurements at the end.

            2. A noisy simulation using the density matrix method with all
            measurements at the end.

            In both these cases only a single shot is actually simulated and
            measurement samples for all shots are calculated from the final
            state.
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
        params = _expval_params(operator, variance=variance)
        self._variance = variance
        super().__init__('save_expval',
                         key,
                         operator.num_qubits,
                         conditional=conditional,
                         pershot=pershot,
                         unnormalized=unnormalized,
                         params=params)

    def assemble(self):
        """Return the QasmQobjInstruction for the intructions."""
        instr = super().assemble()
        if self._variance:
            instr.name += '_var'
        return instr


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
                variance=False,
                unnormalized=False,
                conditional=False,
                pershot=False):
    r"""Save the expectation value of a Hermitian operator.

    Args:
        key (str): the key for retrieving saved data from results.
        operator (Pauli or SparsePauliOp or Operator): a Hermitian operator.
        qubits (list): circuit qubits to apply instruction.
        variance (bool): if True save both the expectation value
                         :math:`\langle O \rangle>`, and the variance
                         :math:`\sigma^2 = \langle O \rangle^2 - \langle O \rangle>^2`
                         [Default: False].
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

    .. note ::

        In cetain cases the list returned by ``pershot=True`` may only
        contain a single value, rather than the number of shots. This
        happens when a run circuit supports measurement sampling because
        it is either

        1. An ideal simulation with all measurements at the end.

        2. A noisy simulation using the density matrix method with all
           measurements at the end.

        In both these cases only a single shot is actually simulated and
        measurement samples for all shots are calculated from the final
        state.
    """
    instr = SaveExpval(key,
                       operator,
                       variance=variance,
                       unnormalized=unnormalized,
                       conditional=conditional,
                       pershot=pershot)
    return self.append(instr, qubits)


QuantumCircuit.save_expval = save_expval
