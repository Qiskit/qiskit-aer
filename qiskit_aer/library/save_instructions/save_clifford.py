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
Simulator instruction to save Clifford state.
"""

from qiskit.circuit import QuantumCircuit
from .save_data import SaveSingleData
from ..default_qubits import default_qubits


class SaveClifford(SaveSingleData):
    """Save Clifford instruction"""

    def __init__(self, num_qubits, label="clifford", pershot=False):
        """Create new instruction to save the stabilizer simulator state as a Clifford.

        Args:
            num_qubits (int): the number of qubits of the
            label (str): the key for retrieving saved data from results.
            pershot (bool): if True save a list of Cliffords for each
                            shot of the simulation rather than a single
                            statevector [Default: False].

        .. note::

            This save instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        super().__init__("save_clifford", num_qubits, label, pershot=pershot)


def save_clifford(self, label="clifford", pershot=False):
    """Save the current stabilizer simulator quantum state as a Clifford.

    Args:
        label (str): the key for retrieving saved data from results.
        pershot (bool): if True save a list of Cliffords for each
                        shot of the simulation [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    .. note::

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    instr = SaveClifford(len(qubits), label=label, pershot=pershot)
    return self.append(instr, qubits)


QuantumCircuit.save_clifford = save_clifford
