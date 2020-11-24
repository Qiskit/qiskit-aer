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

from qiskit import QuantumCircuit
from .snapshot import Snapshot


class SnapshotAmplitudes(Snapshot):
    """ Snapshot instruction for amplitudes snapshot type """

    def __init__(self, label: str, params: list, num_qubits=0):
        """Create an amplitudes snapshot instruction.

        Args:
            label (str): the snapshot label.
            params (List[int]): the basis values whose amplitudes to return
            num_qubits (int): the instruction barrier size [Default: 0].

        Raises:
            ExtensionError: if snapshot is invalid.

        Additional Information:
            This snapshot is always performed on all qubits in a circuit.
            The number of qubits parameter specifies the size of the
            instruction as a barrier and should be set to the number of
            qubits in the circuit.
        """
        super().__init__(label, snapshot_type='amplitudes', params=params, num_qubits=num_qubits)


def snapshot_amplitudes(self, label, params):
    """Take a snapshot of a subset of the amplitudes of the simulator state.

    Args:
        label (str): a snapshot label to report the result.
        params (List[int]): the basis values whose amplitudes to return

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if snapshot is invalid.

    Additional Information:
        This snapshot is always performed on all qubits in a circuit.
    """
    snapshot_register = Snapshot.define_snapshot_register(self)

    return self.append(
        SnapshotAmplitudes(label, params, num_qubits=len(snapshot_register)),
        snapshot_register)


QuantumCircuit.snapshot_amplitudes = snapshot_amplitudes
