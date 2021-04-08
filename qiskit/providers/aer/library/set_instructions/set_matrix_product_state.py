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

        Raises:
            ExtensionError: if the input is not a valid state.

        .. note::

            This set instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        #if not isinstance(state, matrix_product_state):
        #    state = matrix_product_state(state)
        #if not state.num_qubits or not state.is_valid():
         #   raise ExtensionError("The input matrix product state is not valid")
        print("num qubits is " + str(len(mps_state[0])))
        print(mps_state)
        print("aaa")
        super().__init__('set_matrix_product_state', len(mps_state[0]), 0, [mps_state])


def set_matrix_product_state(self, mps_state):
    """Set the matrix product state of the simulator.

    Args:
        state (matrix_product_state): A matrix product state.

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: If the state is the incorrect size for the
                        current circuit.

    .. note:

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    print("qubits = " + str(qubits))
    #if not isinstance(state, mps_container_t):
    #    state = matrix_product_state(state)
    #if not state.num_qubits or state.num_qubits != len(qubits):
 #   if state.first.size != len(qubits):
#        raise ExtensionError(
 #           "The size of the matrix product for the set_matrix_product_state"
#            " instruction must be equal to the number of qubits"
#            f" in the circuit (state.num_qubits ({state.num_qubits})"
#            f" != QuantumCircuit.num_qubits ({self.num_qubits})).")
    print("3")
    return self.append(SetMatrixProductState(mps_state), qubits)


QuantumCircuit.set_matrix_product_state = set_matrix_product_state
