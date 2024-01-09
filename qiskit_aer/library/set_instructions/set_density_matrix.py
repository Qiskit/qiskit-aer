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
Instruction to set the density matrix simulator state to a matrix.
"""

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info import DensityMatrix
from ..default_qubits import default_qubits


class SetDensityMatrix(Instruction):
    """Set density matrix state of the simulator"""

    _directive = True

    def __init__(self, state):
        """Create new instruction to set the density matrix state of the simulator.

        Args:
            state (DensityMatrix): a density matrix.

        Raises:
            ValueError: if the input density matrix is not valid.

        .. note::

            This set instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        if not isinstance(state, DensityMatrix):
            state = DensityMatrix(state)
        if not state.num_qubits or not state.is_valid():
            raise ValueError("The input state is not valid")
        super().__init__("set_density_matrix", state.num_qubits, 0, [state.data])


def set_density_matrix(self, state):
    """Set the density matrix state of the simulator.

    Args:
        state (DensityMatrix): a density matrix.

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ValueError: If the density matrix is the incorrect size for the
            current circuit.

    .. note:

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    if not isinstance(state, DensityMatrix):
        state = DensityMatrix(state)
    if not state.num_qubits or state.num_qubits != len(qubits):
        raise ValueError(
            "The size of the density matrix for the set state"
            " instruction must be equal to the number of qubits"
            f" in the circuit (state.num_qubits ({state.num_qubits})"
            f" != QuantumCircuit.num_qubits ({self.num_qubits}))."
        )
    return self.append(SetDensityMatrix(state), qubits)


QuantumCircuit.set_density_matrix = set_density_matrix
