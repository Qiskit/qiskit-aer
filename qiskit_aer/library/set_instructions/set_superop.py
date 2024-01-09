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
Instruction to set the state simulator state to a superop matrix.
"""

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info import SuperOp
from ..default_qubits import default_qubits


class SetSuperOp(Instruction):
    """Set superop state of the simulator"""

    _directive = True

    def __init__(self, state):
        """Create new instruction to set the superop simulator state.

        Args:
            state (QuantumChannel): A CPTP quantum channel.

        Raises:
            ValueError: if the input QuantumChannel is not CPTP.

        .. note::

            This set instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        if not isinstance(state, SuperOp):
            state = SuperOp(state)
        if not state.num_qubits or not state.is_cptp():
            raise ValueError("The input quantum channel is not CPTP")
        super().__init__("set_superop", state.num_qubits, 0, [state.data])


def set_superop(self, state):
    """Set the superop state of the simulator.

    Args:
        state (QuantumChannel): A CPTP quantum channel.

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ValueError: If the state is the incorrect size for the current circuit.
        ValueError: if the input QuantumChannel is not CPTP.

    .. note:

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    if not isinstance(state, SuperOp):
        state = SuperOp(state)
    if not state.num_qubits or state.num_qubits != len(qubits):
        raise ValueError(
            "The size of the quantum channel for the set_superop"
            " instruction must be equal to the number of qubits"
            f" in the circuit (state.num_qubits ({state.num_qubits})"
            f" != QuantumCircuit.num_qubits ({self.num_qubits}))."
        )
    return self.append(SetSuperOp(state), qubits)


QuantumCircuit.set_superop = set_superop
