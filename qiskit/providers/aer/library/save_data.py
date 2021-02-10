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
Simulator instruction to save custom internal data to results.
"""

import copy

from qiskit.circuit import QuantumCircuit, QuantumRegister
# TEMP For compatiblity until Terra PR #5701 is merged
try:
    from qiskit.circuit import Directive
except ImportError:
    from qiskit.circuit import Instruction as Directive
from qiskit.extensions.exceptions import ExtensionError


class SaveData(Directive):
    """Pragma Instruction to save simulator data."""

    _allowed_subtypes = set([
        'single', 'list', 'c_list', 'average', 'c_average', 'accum', 'c_accum'
    ])

    def __init__(self, name, key, num_qubits, subtype='single', params=None):
        """Create new save data instruction.

        Args:
            name (str): the name of hte save instruction.
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits for the snapshot type.
            subtype (str): the data subtype for the instruction [Default: 'single'].
            params (list or None): Optional, the parameters for instruction
                                   [Default: None].

        Raises:
            ExtensionError: if the subtype string is invalid.

        Additional Information:
            The supported subtypes are 'single', 'list', 'c_list', 'average',
            'c_average', 'accum', 'c_accum'.
        """
        if params is None:
            params = {}

        if subtype not in self._allowed_subtypes:
            raise ExtensionError(
                "Invalid data subtype for SaveData instruction.")

        if not isinstance(key, str):
            raise ExtensionError("Invalid key for save data instruction, key must be a string.")

        self._key = key
        self._subtype = subtype
        super().__init__(name, num_qubits, 0, params)

    def assemble(self):
        """Return the QasmQobjInstruction for the intructions."""
        instr = super().assemble()
        # Use same fields as Snapshot instruction
        # so we dont need to modify QasmQobjInstruction
        instr.snapshot_type = self._subtype
        instr.label = self._key
        return instr

    def inverse(self):
        """Special case. Return self."""
        return copy.copy(self)


class SaveAverageData(SaveData):
    """Save averageble data"""
    def __init__(self,
                 name,
                 key,
                 num_qubits,
                 pershot=False,
                 conditional=False,
                 unnormalized=False,
                 params=None):
        """Create new save data instruction.

        Args:
            name (str): the name of hte save instruction.
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits for the snapshot type.
            pershot (bool): if True save a list of data for each shot of the
                            simulation rather than the average over  all shots
                            [Default: False].
            conditional (bool): if True save the average or pershot data
                                conditional on the current classical register
                                values [Default: False].
            unnormalized (bool): If True return save the unnormalized accumulated
                                 or conditional accumulated data over all shot.
                                 [Default: False].
            params (list or None): Optional, the parameters for instruction
                                   [Default: None].
        """
        if pershot:
            subtype = 'list'
        elif unnormalized:
            subtype = 'accum'
        else:
            subtype = 'average'
        if conditional:
            subtype = 'c_' + subtype
        super().__init__(name, key, num_qubits, subtype=subtype, params=params)


class SaveSingleData(SaveData):
    """Save non-averagable single data type."""

    def __init__(self,
                 name,
                 key,
                 num_qubits,
                 pershot=False,
                 conditional=False,
                 params=None):
        """Create new save data instruction.

        Args:
            name (str): the name of the save instruction.
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits for the snapshot type.
            pershot (bool): if True save a list of data for each shot of the
                            simulation [Default: False].
            conditional (bool): if True save pershot data conditional on the
                                current classical register values
                                [Default: False].
            params (list or None): Optional, the parameters for instruction
                                   [Default: None].
        """
        subtype = 'single'
        if pershot:
            subtype = 'c_list' if conditional else 'list'
        super().__init__(name, key, num_qubits, subtype=subtype, params=params)


def default_qubits(circuit, qubits=None):
    """Helper method to return list of qubits.

    Args:
        circuit (QuantumCircuit): a quantum circuit.
        qubits (list or QuantumRegister): Optional, qubits argument,
            If None the returned list will be all qubits in the circuit.
            [Default: None]

    Raises:
            ExtensionError: if default qubits fails.

    Returns:
        list: qubits list.
    """
    # Convert label to string for backwards compatibility
    # If no qubits are specified we add all qubits so it acts as a barrier
    # This is needed for full register snapshots like statevector
    if isinstance(qubits, QuantumRegister):
        qubits = qubits[:]
    if not qubits:
        tuples = []
        if isinstance(circuit, QuantumCircuit):
            for register in circuit.qregs:
                tuples.append(register)
        if not tuples:
            raise ExtensionError('no qubits for snapshot')
        qubits = []
        for tuple_element in tuples:
            if isinstance(tuple_element, QuantumRegister):
                for j in range(tuple_element.size):
                    qubits.append(tuple_element[j])
            else:
                qubits.append(tuple_element)

    return qubits
