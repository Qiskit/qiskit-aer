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
# that they have been altered from the originals.

"""
Simulator command to snapshot internal simulator representation.
"""

from warnings import warn
from qiskit import QuantumCircuit
from .snapshot import Snapshot


class SnapshotDensityMatrix(Snapshot):
    """Snapshot instruction for density matrix method of Qasm simulator."""

    def __init__(self, label, num_qubits):
        """Create a density matrix state snapshot instruction.

        Args:
            label (str): the snapshot label.
            num_qubits (int): the number of qubits to snapshot.

        Raises:
            ExtensionError: if snapshot is invalid.

        .. note::

            This instruction will be deprecated after the qiskit-aer 0.8 release.
            It has been superseded by the
            :class:`qiskit.providers.aer.library.SaveDensityMatrix` instruction.
        """
        warn('`The `SnapshotDensityMatrix` instruction will be deprecated in the'
             'future. It has been superseded by the `SaveDensityMatrix`'
             ' instructions.', PendingDeprecationWarning)
        super().__init__(label,
                         snapshot_type='density_matrix',
                         num_qubits=num_qubits)


def snapshot_density_matrix(self, label, qubits=None):
    """Take a density matrix snapshot of simulator state.

    Args:
        label (str): a snapshot label to report the result
        qubits (list or None): the qubits to apply snapshot to. If None all
                               qubits will be snapshot [Default: None].

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if snapshot is invalid.

    .. note::

        This method will be deprecated after the qiskit-aer 0.8 release.
        It has been superseded by the
        :func:`qiskit.providers.aer.library.save_density_matrix`
        circuit method.
    """
    warn('`The `save_density_matrix` circuit method will be deprecated in the'
         ' future. It has been superseded by the `save_density_matrix`'
         ' circuit method.', PendingDeprecationWarning)
    snapshot_register = Snapshot.define_snapshot_register(self, qubits=qubits)

    return self.append(
        SnapshotDensityMatrix(label, num_qubits=len(snapshot_register)),
        snapshot_register)


QuantumCircuit.snapshot_density_matrix = snapshot_density_matrix
