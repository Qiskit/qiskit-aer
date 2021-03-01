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
Simulator instruction to save matrix product state.
"""

from qiskit.circuit import QuantumCircuit
from .save_data import SaveSingleData, default_qubits


class SaveMatrixProductState(SaveSingleData):
    """Save matrix product state instruction"""
    def __init__(self, key, num_qubits, pershot=False, conditional=False):
        """Create new instruction to save the matrix product state.

        Args:
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits
            pershot (bool): if True save the mps for each
                            shot of the simulation [Default: False].
            conditional (bool): if True save data conditional on the current
                                classical register values [Default: False].

        .. note::

            This save instruction must always be performed on the full width of
            qubits in a circuit, otherwise an exception will be raised during
            simulation.
        """
        super().__init__('save_matrix_product_state',
                         key,
                         num_qubits,
                         pershot=pershot,
                         conditional=conditional)


def save_matrix_product_state(self, key, pershot=False, conditional=False):
    """Save the current simulator quantum state as a matrix product state.

    Args:
        key (str): the key for retrieving saved data from results.
        pershot (bool): if True save the mps for each
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
    instr = SaveMatrixProductState(key,
                                   len(qubits),
                                   pershot=pershot,
                                   conditional=conditional)
    return self.append(instr, qubits)


QuantumCircuit.save_matrix_product_state = save_matrix_product_state
