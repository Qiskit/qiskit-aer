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
Simulator instruction to save a density matrix.
"""

from qiskit.circuit import QuantumCircuit
from .save_data import SaveAverageData, default_qubits


class SaveDensityMatrix(SaveAverageData):
    """Save a reduced density matrix."""
    def __init__(self,
                 key,
                 num_qubits,
                 unnormalized=False,
                 pershot=False,
                 conditional=False):
        """Create new instruction to save the simulator reduced density matrix.

        Args:
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits for the save instruction.
            unnormalized (bool): If True return save the unnormalized accumulated
                                 or conditional accumulated density matrix over
                                 all shots [Default: False].
            pershot (bool): if True save a list of density matrices for each shot
                            of the  simulation rather than the average over
                            all shots [Default: False].
            conditional (bool): if True save the average or pershot data
                                conditional on the current classical register
                                values [Default: False].
        """
        super().__init__("save_density_matrix",
                         key,
                         num_qubits,
                         unnormalized=unnormalized,
                         pershot=pershot,
                         conditional=conditional)


def save_density_matrix(self,
                        key,
                        qubits=None,
                        unnormalized=False,
                        pershot=False,
                        conditional=False):
    """Save the current simulator quantum state as a density matrix.

    Args:
        key (str): the key for retrieving saved data from results.
        qubits (list or None): the qubits to save reduced density matrix on.
                               If None the full density matrix of qubits will
                               be saved [Default: None].
        unnormalized (bool): If True return save the unnormalized accumulated
                             or conditional accumulated density matrix over
                             all shots [Default: False].
        pershot (bool): if True save a list of density matrices for each shot
                        of the  simulation rather than the average over
                        all shots [Default: False].
        conditional (bool): if True save the average or pershot data
                            conditional on the current classical register
                            values [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.
    """
    qubits = default_qubits(self, qubits=qubits)
    instr = SaveDensityMatrix(key,
                              len(qubits),
                              unnormalized=unnormalized,
                              pershot=pershot,
                              conditional=conditional)
    return self.append(instr, qubits)


QuantumCircuit.save_density_matrix = save_density_matrix
