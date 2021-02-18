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
Simulator instruction to save statevector amplitudes and amplitudes squared.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.extensions.exceptions import ExtensionError
from .save_data import SaveSingleData, SaveAverageData, default_qubits


class SaveAmplitudes(SaveSingleData):
    """Save complex statevector amplitudes."""
    def __init__(self,
                 key,
                 num_qubits,
                 params,
                 pershot=False,
                 conditional=False):
        """Instruction to save complex statevector amplitudes.

        Args:
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits for the snapshot type.
            params (list): list of entries to vale.
            pershot (bool): if True save a list of amplitudes vectors for each
                            shot of the simulation rather than the a single
                            amplitude vector [Default: False].
            conditional (bool): if True save the amplitudes vector conditional
                                on the current classical register values
                                [Default: False].

        Raises:
            ExtensionError: if params is invalid for the specified number of qubits.
        """
        params = _format_amplitude_params(params, num_qubits)
        super().__init__("save_amplitudes",
                         key,
                         num_qubits,
                         pershot=pershot,
                         conditional=conditional,
                         params=params)


class SaveAmplitudesSquared(SaveAverageData):
    """Save squared statevector amplitudes (probabilities)."""
    def __init__(self,
                 key,
                 num_qubits,
                 params,
                 unnormalized=False,
                 pershot=False,
                 conditional=False):
        """Instruction to save squared statevector amplitudes (probabilities).

        Args:
            key (str): the key for retrieving saved data from results.
            num_qubits (int): the number of qubits for the snapshot type.
            params (list): list of entries to vale.
            unnormalized (bool): If True return save the unnormalized accumulated
                                 probabilities over all shots [Default: False].
            pershot (bool): if True save a list of probability vectors for each
                            shot of the simulation rather than the a single
                            amplitude vector [Default: False].
            conditional (bool): if True save the probability vector conditional
                                on the current classical register values
                                [Default: False].

        Raises:
            ExtensionError: if params is invalid for the specified number of qubits.
        """
        params = _format_amplitude_params(params, num_qubits)
        super().__init__("save_amplitudes_sq",
                         key,
                         num_qubits,
                         unnormalized=unnormalized,
                         pershot=pershot,
                         conditional=conditional,
                         params=params)


def save_amplitudes(self, key, params, pershot=False, conditional=False):
    """Save complex statevector amplitudes.

    Args:
        key (str): the key for retrieving saved data from results.
        params (List[int] or List[str]): the basis states to return amplitudes for.
        pershot (bool): if True save a list of amplitudes vectors for each
                        shot of the simulation rather than the a single
                        amplitude vector [Default: False].
        conditional (bool): if True save the amplitudes vector conditional
                            on the current classical register values
                            [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if params is invalid for the specified number of qubits.
    """
    qubits = default_qubits(self)
    instr = SaveAmplitudes(key, len(qubits), params,
                           pershot=pershot, conditional=conditional)
    return self.append(instr, qubits)


def save_amplitudes_squared(self, key, params,
                            unnormalized=False,
                            pershot=False,
                            conditional=False):
    """Save squared statevector amplitudes (probabilities).

    Args:
        key (str): the key for retrieving saved data from results.
        params (List[int] or List[str]): the basis states to return amplitudes for.
        unnormalized (bool): If True return save the unnormalized accumulated
                             probabilities over all shots [Default: False].
        pershot (bool): if True save a list of probability vectors for each
                        shot of the simulation rather than the a single
                        amplitude vector [Default: False].
        conditional (bool): if True save the probability vector conditional
                            on the current classical register values
                            [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if params is invalid for the specified number of qubits.
    """
    qubits = default_qubits(self)
    instr = SaveAmplitudesSquared(key, len(qubits), params,
                                  unnormalized=unnormalized,
                                  pershot=pershot,
                                  conditional=conditional)
    return self.append(instr, qubits)


def _format_amplitude_params(params, num_qubits=None):
    """Format amplitude params as a interger list."""
    if isinstance(params[0], str):
        if params[0].find('0x') == 0:
            params = [int(i, 16) for i in params]
        else:
            params = [int(i, 2) for i in params]
    if num_qubits and max(params) >= 2 ** num_qubits:
        raise ExtensionError(
            "Param values contain a state larger than the number of qubits")
    return params


QuantumCircuit.save_amplitudes = save_amplitudes
QuantumCircuit.save_amplitudes_squared = save_amplitudes_squared
