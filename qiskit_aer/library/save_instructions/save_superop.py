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
Simulator instruction to save a SuperOp matrix.
"""

from qiskit.circuit import QuantumCircuit
from .save_data import SaveSingleData
from ..default_qubits import default_qubits


class SaveSuperOp(SaveSingleData):
    """Save a SuperOp matrix."""

    def __init__(self, num_qubits, label="superop", pershot=False):
        """Create new instruction to save the superop simulator state.

        Args:
            num_qubits (int): the number of qubits for the save instruction.
            label (str): the key for retrieving saved data from results.
            pershot (bool): if True save a list of SuperOp matrices for each shot
                            of the simulation [Default: False].

        .. note::

            This save instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        super().__init__("save_superop", num_qubits, label, pershot=pershot)


def save_superop(self, label="superop", pershot=False):
    """Save the current state of the superop simulator.

    Args:
        label (str): the key for retrieving saved data from results.
        pershot (bool): if True save a list of SuperOp matrices for each shot
                        of the  simulation [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    .. note::

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    instr = SaveSuperOp(len(qubits), label=label, pershot=pershot)
    return self.append(instr, qubits)


QuantumCircuit.save_superop = save_superop
