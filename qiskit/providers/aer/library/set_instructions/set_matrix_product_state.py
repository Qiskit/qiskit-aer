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
Instruction to set the state simulator state to a matrix.
"""

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.extensions.exceptions import ExtensionError
from qiskit.quantum_info import Statevector
from ..default_qubits import default_qubits


class SetMatrixProductState(Instruction):
    """Set the matrix product state of the simulator"""

    _directive = True

    def __init__(self, mps_state):
        """Create new instruction to set the matrix product state of the simulator.

        Args:
            mps_state (mps_container_t): a matrix product state.

        .. note::

            This set instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        super().__init__('set_matrix_product_state', len(mps_state[0]), 0, [mps_state])


def set_matrix_product_state(self, mps_state):
    """Set the matrix product state of the simulator.

    Args:
        state (matrix_product_state): A matrix product state.

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: If the structure of the mps_state is incorrect

    .. note:

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    if not isinstance(mps_state, tuple) or len(mps_state) != 2:
        raise ExtensionError(
            "The input matrix product state is not valid.  Should be a list of 2 elements")
    if not isinstance(mps_state[0], list) or not isinstance(mps_state[1], list):
        raise ExtensionError(
            "The first element of the input matrix product state is not valid. Should be a list.")
    if len(mps_state[0]) != len(mps_state[1]) + 1:
        raise ExtensionError(
            "The input matrix product state is not valid. "
            "Length of q_reg vector should be 1 more than length of lambda_reg")
    for elem in mps_state[0]:
        if not isinstance(elem, tuple) or len(elem) != 2:
            raise ExtensionError(
                "The input matrix product state is not valid."
                "The first element should be a list of length 2")
    return self.append(SetMatrixProductState(mps_state), qubits)


QuantumCircuit.set_matrix_product_state = set_matrix_product_state
