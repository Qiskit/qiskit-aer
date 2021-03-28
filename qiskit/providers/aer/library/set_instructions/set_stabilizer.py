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
from qiskit.quantum_info import Clifford
from ..default_qubits import default_qubits


class SetStabilizer(Instruction):
    """Set the Clifford stabilizer state of the simulator"""

    _directive = True

    def __init__(self, state):
        """Create new instruction to set the Clifford stabilizer state of the simulator.

        Args:
            state (Clifford): A clifford operator.

        .. note::

            This set instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        if not isinstance(state, Clifford):
            state = Clifford(state)
        super().__init__('set_stabilizer', state.num_qubits, 0, [state.to_dict()])


def set_stabilizer(self, state):
    """Set the Clifford stabilizer state of the simulator.

    Args:
        state (Clifford): A clifford operator.

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: If the state is the incorrect size for the
                        current circuit.

    .. note:

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    if not isinstance(state, Clifford):
        state = Clifford(state)
    if state.num_qubits != len(qubits):
        raise ExtensionError(
            "The size of the Clifford for the set_stabilizer"
            " instruction must be equal to the number of qubits"
            f" in the circuit (state.num_qubits ({state.num_qubits})"
            f" != QuantumCircuit.num_qubits ({self.num_qubits})).")
    return self.append(SetStabilizer(state), qubits)


QuantumCircuit.set_stabilizer = set_stabilizer
