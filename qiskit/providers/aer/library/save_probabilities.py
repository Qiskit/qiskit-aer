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
Simulator instruction to save measurement outcome probabilites.
"""

from qiskit.circuit import QuantumCircuit
from .save_data import SaveAverageData, default_qubits


class SaveProbabilities(SaveAverageData):
    """Save measurement outcome probabilities vector."""
    def __init__(self,
                 key,
                 num_qubits,
                 unnormalized=False,
                 pershot=False,
                 conditional=False):
        """Instruction to save measurement probabilities vector.

        Args:
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits for the snapshot type.
            unnormalized (bool): If True return save the unnormalized accumulated
                                 probabilities over all shots [Default: False].
            pershot (bool): if True save a list of probabilities for each shot
                            of the simulation rather than the average over
                            all shots [Default: False].
            conditional (bool): if True save the probabilities data conditional
                                on the current classical register values
                                [Default: False].
        """
        super().__init__("save_probabilities",
                         key,
                         num_qubits,
                         conditional=conditional,
                         pershot=pershot,
                         unnormalized=unnormalized)


class SaveProbabilitiesDict(SaveAverageData):
    """Save measurement outcome probabilities dict."""
    def __init__(self,
                 key,
                 num_qubits,
                 unnormalized=False,
                 pershot=False,
                 conditional=False):
        """Instruction to save measurement probabilities dict.

        Args:
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits for the snapshot type.
            unnormalized (bool): If True return save the unnormalized accumulated
                                 probabilities over all shots [Default: False].
            pershot (bool): if True save a list of probabilities for each shot
                            of the simulation rather than the average over
                            all shots [Default: False].
            conditional (bool): if True save the probabilities data conditional
                                on the current classical register values
                                [Default: False].
        """
        super().__init__("save_probabilities_dict",
                         key,
                         num_qubits,
                         unnormalized=unnormalized,
                         pershot=pershot,
                         conditional=conditional)


def save_probabilities(self,
                       key,
                       qubits=None,
                       unnormalized=False,
                       pershot=False,
                       conditional=False):
    """Save measurement outcome probabilities vector.

    Args:
        key (str): the key for retrieving saved data from results.
        qubits (list or None): the qubits to apply snapshot to. If None all
                               qubits will be snapshot [Default: None].
        unnormalized (bool): If True return save the unnormalized accumulated
                             probabilities over all shots [Default: False].
        pershot (bool): if True save a list of probabilities for each shot
                        of the simulation rather than the average over
                        all shots [Default: False].
        conditional (bool): if True save the probabilities data conditional
                            on the current classical register values
                            [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.
    """
    qubits = default_qubits(self, qubits=qubits)
    instr = SaveProbabilities(key,
                              len(qubits),
                              unnormalized=unnormalized,
                              pershot=pershot,
                              conditional=conditional)
    return self.append(instr, qubits)


def save_probabilities_dict(self,
                            key,
                            qubits=None,
                            unnormalized=False,
                            pershot=False,
                            conditional=False):
    """Save measurement outcome probabilities vector.

    Args:
        key (str): the key for retrieving saved data from results.
        qubits (list or None): the qubits to apply snapshot to. If None all
                               qubits will be snapshot [Default: None].
        unnormalized (bool): If True return save the unnormalized accumulated
                             probabilities over all shots [Default: False].
        pershot (bool): if True save a list of probabilities for each shot
                        of the simulation rather than the average over
                        all shots [Default: False].
        conditional (bool): if True save the probabilities data conditional
                            on the current classical register values
                            [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.
    """
    qubits = default_qubits(self, qubits=qubits)
    instr = SaveProbabilitiesDict(key,
                                  len(qubits),
                                  unnormalized=unnormalized,
                                  pershot=pershot,
                                  conditional=conditional)
    return self.append(instr, qubits)


QuantumCircuit.save_probabilities = save_probabilities
QuantumCircuit.save_probabilities_dict = save_probabilities_dict
