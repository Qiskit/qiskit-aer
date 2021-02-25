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


class SnapshotStabilizer(Snapshot):
    """Snapshot instruction for stabilizer method of Qasm simulator."""

    def __init__(self, label, num_qubits=0):
        """Create a stabilizer state snapshot instruction.

        Args:
            label (str): the snapshot label.
            num_qubits (int): the instruction barrier size [Default: 0].

        Raises:
            ExtensionError: if snapshot is invalid.

        Additional Information:
            This snapshot is always performed on all qubits in a circuit.
            The number of qubits parameter specifies the size of the
            instruction as a barrier and should be set to the number of
            qubits in the circuit.

        .. note::

            This instruction will be deprecated after the qiskit-aer 0.8 release.
            It has been superseded by the
            :class:`qiskit.providers.aer.library.SaveStabilizer` instruction.
        """
        warn('The `SnapshotStabilizer` instruction will be deprecated in the'
             ' future. It has been superseded by the `save_stabilizer`'
             ' instructions.', PendingDeprecationWarning)
        super().__init__(label, snapshot_type='stabilizer', num_qubits=num_qubits)


def snapshot_stabilizer(self, label):
    """Take a stabilizer snapshot of the simulator state.

    Args:
        label (str): a snapshot label to report the result.

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if snapshot is invalid.

    Additional Information:
        This snapshot is always performed on all qubits in a circuit.
        The number of qubits parameter specifies the size of the
        instruction as a barrier and should be set to the number of
        qubits in the circuit.

    .. note::

        This method will be deprecated after the qiskit-aer 0.8 release.
        It has been superseded by the
        :func:`qiskit.providers.aer.library.save_stabilizer` circuit
        method.
    """
    warn('`The `save_stabilizer` circuit method will be deprecated in the'
         ' future. It has been superseded by the `save_stabilizer`'
         ' circuit method.', PendingDeprecationWarning)
    snapshot_register = Snapshot.define_snapshot_register(self)

    return self.append(
        SnapshotStabilizer(label, num_qubits=len(snapshot_register)),
        snapshot_register)


QuantumCircuit.snapshot_stabilizer = snapshot_stabilizer
