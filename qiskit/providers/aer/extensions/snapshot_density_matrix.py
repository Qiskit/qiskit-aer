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
from qiskit.providers.aer.extensions import Snapshot


class SnapshotDensityMatrix(Snapshot):
    """Snapshot instruction for density matrix method of Qasm simulator."""

    def __init__(self,
                 label,
                 num_qubits=0,
                 num_clbits=0,
                 params=None):

        super().__init__(label, 'density_matrix', num_qubits, num_clbits, params)


def snapshot_density_matrix(self,
                            label,
                            qubits=None,
                            params=None):
    """Take a density matrix snapshot of the internal simulator representation.
    Works on all qubits, and prevents reordering (like barrier).
    Args:
        label (str): a snapshot label to report the result
        qubits (list or None): the qubits to apply snapshot to [Default: None].
        params (list or None): the parameters for snapshot_type [Default: None].
    Returns:
        QuantumCircuit: with attached command
    Raises:
        ExtensionError: malformed command
    """

    snapshot_register = Snapshot.define_snapshot_register(self, label, qubits)

    return self.append(
        SnapshotDensityMatrix(
            label,
            num_qubits=len(snapshot_register),
            params=params), snapshot_register)


QuantumCircuit.snapshot_density_matrix = snapshot_density_matrix
