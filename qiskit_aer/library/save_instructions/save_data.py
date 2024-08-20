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

from qiskit.circuit import Instruction


class SaveData(Instruction):
    """Pragma Instruction to save simulator data."""

    _directive = True
    _allowed_subtypes = set(
        ["single", "c_single", "list", "c_list", "average", "c_average", "accum", "c_accum"]
    )

    def __init__(self, name, num_qubits, label, subtype="single", params=None):
        """Create new save data instruction.

        Args:
            name (str): the name of hte save instruction.
            num_qubits (int): the number of qubits for the snapshot type.
            label (str): the key for retrieving saved data from results.
            subtype (str): the data subtype for the instruction [Default: 'single'].
            params (list or None): Optional, the parameters for instruction
                                   [Default: None].

        Raises:
            TypeError: if the subtype string is invalid.

        Additional Information:
            The supported subtypes are 'single', 'list', 'c_list', 'average',
            'c_average', 'accum', 'c_accum'.
        """
        if subtype not in self._allowed_subtypes:
            raise TypeError("Invalid data subtype for SaveData instruction.")

        if not isinstance(label, str):
            raise TypeError(f"Invalid label for save data instruction, {label} must be a string.")

        if params is None:
            params = {}

        super().__init__(name, num_qubits, 0, params)

        self._label = label
        self._subtype = subtype

    def inverse(self, annotated=False):
        """Special case. Return self."""
        return copy.copy(self)


class SaveAverageData(SaveData):
    """Save averageble data"""

    def __init__(
        self,
        name,
        num_qubits,
        label,
        unnormalized=False,
        pershot=False,
        conditional=False,
        params=None,
    ):
        """Create new save data instruction.

        Args:
            name (str): the name of hte save instruction.
            num_qubits (int): the number of qubits for the snapshot type.
            label (str): the key for retrieving saved data from results.
            unnormalized (bool): If True return save the unnormalized accumulated
                                 or conditional accumulated data over all shot.
                                 [Default: False].
            pershot (bool): if True save a list of data for each shot of the
                            simulation rather than the average over  all shots
                            [Default: False].
            conditional (bool): if True save the average or pershot data
                                conditional on the current classical register
                                values [Default: False].
            params (list or None): Optional, the parameters for instruction
                                   [Default: None].
        """
        if pershot:
            subtype = "list"
        elif unnormalized:
            subtype = "accum"
        else:
            subtype = "average"
        if conditional:
            subtype = "c_" + subtype
        super().__init__(name, num_qubits, label, subtype=subtype, params=params)


class SaveSingleData(SaveData):
    """Save non-averagable single data type."""

    def __init__(self, name, num_qubits, label, pershot=False, conditional=False, params=None):
        """Create new save data instruction.

        Args:
            name (str): the name of the save instruction.
            num_qubits (int): the number of qubits for the snapshot type.
            label (str): the key for retrieving saved data from results.
            pershot (bool): if True save a list of data for each shot of the
                            simulation [Default: False].
            conditional (bool): if True save data conditional on the
                                current classical register values
                                [Default: False].
            params (list or None): Optional, the parameters for instruction
                                   [Default: None].
        """
        subtype = "list" if pershot else "single"
        if conditional:
            subtype = "c_" + subtype
        super().__init__(name, num_qubits, label, subtype=subtype, params=params)
