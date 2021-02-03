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
Simulator instruction to save statevector.
"""

from qiskit.circuit import QuantumCircuit
from .save_data import SaveSingleData, default_qubits


class SaveStatevector(SaveSingleData):
    """Save statevector"""
    def __init__(self, key, num_qubits, pershot=False, conditional=False):
        """Create new instruction to save the simualtor statevector.

        Args:
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits of the
            pershot (bool): if True save a list of statevectors for each
                            shot of the simulation rather than a single
                            statevector [Default: False].
            conditional (bool): if True save data conditional on the current
                                classical register values [Default: False].

        .. note::

            This save instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        super().__init__('save_statevector',
                         key,
                         num_qubits,
                         pershot=pershot,
                         conditional=conditional)


class SaveStatevectorDict(SaveSingleData):
    """Save statevector as ket-form dictionary."""
    def __init__(self, key, num_qubits, pershot=False, conditional=False):
        """Create new instruction to save the simualtor statevector as a dict.

        Args:
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits of the
            pershot (bool): if True save a list of statevectors for each
                            shot of the simulation rather than a single
                            statevector [Default: False].
            conditional (bool): if True save data conditional on the current
                                classical register values [Default: False].

        .. note::

            This save instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        super().__init__('save_statevector_dict',
                         key,
                         num_qubits,
                         pershot=pershot,
                         conditional=conditional)


def save_statevector(self, key, pershot=False, conditional=False):
    """Save the current simulator quantum state as a statevector.

    Args:
        key (str): the key for retrieving saved data from results.
        pershot (bool): if True save a list of statevectors for each
                        shot of the simulation [Default: False].
        conditional (bool): if True save pershot data conditional on the
                            current classical register values
                            [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    .. note:

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    instr = SaveStatevector(key,
                            len(qubits),
                            pershot=pershot,
                            conditional=conditional)
    return self.append(instr, qubits)


def save_statevector_dict(self, key, pershot=False, conditional=False):
    """Save the current simulator quantum state as a statevector as a dict.

    Args:
        key (str): the key for retrieving saved data from results.
        pershot (bool): if True save a list of statevectors for each
                        shot of the simulation [Default: False].
        conditional (bool): if True save pershot data conditional on the
                            current classical register values
                            [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    .. note:

        This instruction is always defined across all qubits in a circuit.
    """
    qubits = default_qubits(self)
    instr = SaveStatevectorDict(key,
                                len(qubits),
                                pershot=pershot,
                                conditional=conditional)
    return self.append(instr, qubits)


QuantumCircuit.save_statevector = save_statevector
QuantumCircuit.save_statevector_dict = save_statevector_dict
