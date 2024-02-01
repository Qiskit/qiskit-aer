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
Instruction to set the simulator state to a stabilizer state.
"""

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info import StabilizerState, Clifford
from ..default_qubits import default_qubits


class SetStabilizer(Instruction):
    """Set the Clifford stabilizer state of the simulator"""

    _directive = True

    def __init__(self, state):
        """Create new instruction to set the Clifford stabilizer state of the simulator.

        Args:
            state (StabilizerState or Clifford): A clifford operator.

        .. note::

            This set instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        if isinstance(state, StabilizerState):
            state = state.clifford
        elif not isinstance(state, Clifford):
            state = Clifford(state)
        super().__init__("set_stabilizer", state.num_qubits, 0, [state.to_dict()])


def set_stabilizer(self, state):
    """Set the Clifford stabilizer state of the simulator.

    Args:
        state (Clifford): A clifford operator.

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ValueError: If the state is the incorrect size for the
            current circuit.

    .. note:

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    if isinstance(state, StabilizerState):
        state = state.clifford
    if not isinstance(state, Clifford):
        state = Clifford(state)
    if state.num_qubits != len(qubits):
        raise ValueError(
            "The size of the Clifford for the set_stabilizer"
            " instruction must be equal to the number of qubits"
            f" in the circuit (state.num_qubits ({state.num_qubits})"
            f" != QuantumCircuit.num_qubits ({self.num_qubits}))."
        )
    return self.append(SetStabilizer(state), qubits)


QuantumCircuit.set_stabilizer = set_stabilizer
