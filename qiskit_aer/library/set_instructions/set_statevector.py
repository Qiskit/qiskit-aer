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
from qiskit.quantum_info import Statevector
from ..default_qubits import default_qubits


class SetStatevector(Instruction):
    """Set the statevector state of the simulator"""

    _directive = True

    def __init__(self, state):
        """Create new instruction to set the statevector state of the simulator.

        Args:
            state (Statevector): a statevector.

        Raises:
            ValueError: if the input is not a valid state.

        .. note::

            This set instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        if not isinstance(state, Statevector):
            state = Statevector(state)
        if not state.num_qubits or not state.is_valid():
            raise ValueError("The input statevector is not valid")
        super().__init__("set_statevector", state.num_qubits, 0, [state.data])


def set_statevector(self, state):
    """Set the statevector state of the simulator.

    Args:
        state (Statevector): A state matrix.

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ValueError: If the state is the incorrect size for the
            current circuit.

    .. note:

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    if not isinstance(state, Statevector):
        state = Statevector(state)
    if not state.num_qubits or state.num_qubits != len(qubits):
        raise ValueError(
            "The size of the statevector for the set_statevector"
            " instruction must be equal to the number of qubits"
            f" in the circuit (state.num_qubits ({state.num_qubits})"
            f" != QuantumCircuit.num_qubits ({self.num_qubits}))."
        )
    return self.append(SetStatevector(state), qubits)


QuantumCircuit.set_statevector = set_statevector
