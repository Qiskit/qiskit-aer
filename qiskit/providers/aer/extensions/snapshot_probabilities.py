# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals

"""
Simulator command to snapshot internal simulator representation.
"""

from warnings import warn
from qiskit import QuantumCircuit
from .snapshot import Snapshot


class SnapshotProbabilities(Snapshot):
    """Snapshot instruction for all methods of Qasm simulator."""

    def __init__(self, label, num_qubits, variance=False):
        """Create a probability snapshot instruction.

        Args:
            label (str): the snapshot label.
            num_qubits (int): the number of qubits to snapshot.
            variance (bool): compute variance of probabilities [Default: False]

        Raises:
            ExtensionError: if snapshot is invalid.

        .. note::

            This instruction will be deprecated after the qiskit-aer 0.8 release.
            It has been superseded by the
            :class:`qiskit.providers.aer.library.SaveProbabilities` and
            :class:`qiskit.providers.aer.library.SaveProbabilitiesDict`
            instructions.
        """
        warn('The `SnapshotProbabilities` instruction will be deprecated in the'
             ' future. It has been superseded by the `SaveProbabilities` and'
             ' `SaveProbabilitiesDict` instructions.',
             PendingDeprecationWarning)
        if variance:
            warn('The snapshot `variance` kwarg has been deprecated and will be removed'
                 ' in qiskit-aer 0.8.', DeprecationWarning)
        snapshot_type = 'probabilities_with_variance' if variance else 'probabilities'
        super().__init__(label, snapshot_type=snapshot_type,
                         num_qubits=num_qubits)


def snapshot_probabilities(self, label, qubits, variance=False):
    """Take a probability snapshot of the simulator state.

    Args:
        label (str): a snapshot label to report the result
        qubits (list): the qubits to snapshot.
        variance (bool): compute variance of probabilities [Default: False]

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if snapshot is invalid.

    .. note::

        This method will be deprecated after the qiskit-aer 0.8 release.
        It has been superseded by the
        :func:`qiskit.providers.aer.library.save_probabilities` and
        :func:`qiskit.providers.aer.library.save_probabilities_dict`
        circuit methods.
    """
    warn('The `snapshot_probabilities` circuit method will be deprecated '
         ' in the future. It has been superseded by the `save_probabilities`'
         ' and `save_probabilities_dict` circuit methods.',
         PendingDeprecationWarning)
    snapshot_register = Snapshot.define_snapshot_register(self, qubits=qubits)

    return self.append(
        SnapshotProbabilities(label,
                              num_qubits=len(snapshot_register),
                              variance=variance),
        snapshot_register)


QuantumCircuit.snapshot_probabilities = snapshot_probabilities
